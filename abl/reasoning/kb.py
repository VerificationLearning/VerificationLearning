from abc import ABC, abstractmethod
import bisect
import numpy as np
import copy
from collections import defaultdict
from itertools import product, combinations

from abl.utils.utils import flatten, reform_idx, hamming_dist, check_equal, to_hashable, hashable_to_list
import torch
from multiprocessing import Pool
import pickle
import os
from functools import lru_cache
import pyswip
import heapq
class Heap:
    def __init__(self):
        self._heap = []

    def push(self, item):
        heapq.heappush(self._heap, item)

    def pop(self):
        return heapq.heappop(self._heap)

    def is_empty(self):
        return len(self._heap) == 0

class KBBase(ABC):
    def __init__(self, pseudo_label_list, prebuild_GKB=False, GKB_len_list=None, max_err=0, use_cache=True, kb_file_path=None):
        """
        Initialize the KBBase instance.

        Args:
        pseudo_label_list (list): List of pseudo labels.
        prebuild_GKB (bool): Whether to prebuild the General Knowledge Base (GKB).
        GKB_len_list (list): List of lengths for the GKB.
        max_err (int): Maximum error threshold.
        use_cache (bool): Whether to use caching.
        kb_file_path (str, optional): Path to the file from which to load the pre-built knowledge base. If None, build a new knowledge base.
        """

        self.pseudo_label_list = pseudo_label_list
        self.prebuild_GKB = prebuild_GKB
        self.GKB_len_list = GKB_len_list
        self.max_err = max_err
        self.use_cache = use_cache
        self.base = {}

        if kb_file_path and os.path.exists(kb_file_path):
            self.load_kb(kb_file_path)
        elif prebuild_GKB:
            X, Y = self._get_GKB()
            for x, y in zip(X, Y):
                self.base.setdefault(len(x), defaultdict(list))[y].append(x)
            if kb_file_path:
                self.save_kb(kb_file_path)

    # For parallel version of _get_GKB
    def _get_XY_list(self, args):
        pre_x, post_x_it = args[0], args[1]
        XY_list = []
        for post_x in post_x_it:
            x = (pre_x,) + post_x
            y = self.logic_forward(x)
            if y not in [None, np.inf]:
                XY_list.append((x, y))
        return XY_list

    # Parallel _get_GKB
    def _get_GKB(self):
        X, Y = [], []
        for length in self.GKB_len_list:
            arg_list = []
            for pre_x in self.pseudo_label_list:
                post_x_it = product(self.pseudo_label_list, repeat=length - 1)
                arg_list.append((pre_x, post_x_it))
            with Pool(processes=len(arg_list)) as pool:
                ret_list = pool.map(self._get_XY_list, arg_list)
            for XY_list in ret_list:
                if len(XY_list) == 0:
                    continue
                part_X, part_Y = zip(*XY_list)
                X.extend(part_X)
                Y.extend(part_Y)
        if Y and isinstance(Y[0], (int, float)):
            X, Y = zip(*sorted(zip(X, Y), key=lambda pair: pair[1]))
        return X, Y

    def save_kb(self, file_path):
        """
        Save the knowledge base to a file.

        Args:
        file_path (str): The path to the file where the knowledge base will be saved.
        """
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, 'wb') as f:
            pickle.dump(self.base, f)

    def load_kb(self, file_path):
        """
        Load the knowledge base from a file.

        Args:
        file_path (str): The path to the file from which the knowledge base will be loaded.
        """
        if os.path.exists(file_path):
            with open(file_path, 'rb') as f:
                self.base = pickle.load(f)
        else:
            print(
                f"File {file_path} not found. Starting with an empty knowledge base.")
            self.base = {}

    @abstractmethod
    def logic_forward(self, pseudo_labels):
        pass

    def abduce_candidates(self, pred_res, y, max_revision_num, require_more_revision=0):
        if self.prebuild_GKB:
            return self._abduce_by_GKB(pred_res, y, max_revision_num, require_more_revision)
        else:
            if not self.use_cache:
                return self._abduce_by_search(pred_res, y, max_revision_num, require_more_revision)
            else:
                return self._abduce_by_search_cache(to_hashable(pred_res), to_hashable(y), max_revision_num, require_more_revision)

    def _find_candidate_GKB(self, pred_res, y):
        if self.max_err == 0:
            return self.base[len(pred_res)][y]
        else:
            potential_candidates = self.base[len(pred_res)]
            key_list = list(potential_candidates.keys())
            key_idx = bisect.bisect_left(key_list, y)

            all_candidates = []
            for idx in range(key_idx - 1, 0, -1):
                k = key_list[idx]
                if abs(k - y) <= self.max_err:
                    all_candidates.extend(potential_candidates[k])
                else:
                    break

            for idx in range(key_idx, len(key_list)):
                k = key_list[idx]
                if abs(k - y) <= self.max_err:
                    all_candidates.extend(potential_candidates[k])
                else:
                    break
            return all_candidates

    def _abduce_by_GKB(self, pred_res, y, max_revision_num, require_more_revision):
        if self.base == {} or len(pred_res) not in self.GKB_len_list:
            return []

        all_candidates = self._find_candidate_GKB(pred_res, y)
        if len(all_candidates) == 0:
            return []

        cost_list = hamming_dist(pred_res, all_candidates)
        min_revision_num = np.min(cost_list)
        revision_num = min(
            max_revision_num, min_revision_num + require_more_revision)
        idxs = np.where(cost_list <= revision_num)[0]
        candidates = [all_candidates[idx] for idx in idxs]
        return candidates

    def revise_by_idx(self, pred_res, y, revision_idx):
        candidates = []
        abduce_c = product(self.pseudo_label_list, repeat=len(revision_idx))
        for c in abduce_c:
            candidate = pred_res.copy()
            for i, idx in enumerate(revision_idx):
                candidate[idx] = c[i]
            if check_equal(self.logic_forward(candidate), y, self.max_err):
                candidates.append(candidate)
        return candidates

    def _revision(self, revision_num, pred_res, y):
        new_candidates = []
        revision_idx_list = combinations(range(len(pred_res)), revision_num)

        for revision_idx in revision_idx_list:
            candidates = self.revise_by_idx(pred_res, y, revision_idx)
            new_candidates.extend(candidates)
        return new_candidates

    def _abduce_by_search(self, pred_res, y, max_revision_num, require_more_revision):
        candidates = []
        for revision_num in range(len(pred_res) + 1):
            if revision_num == 0 and check_equal(self.logic_forward(pred_res), y, self.max_err):
                candidates.append(pred_res)
            elif revision_num > 0:
                candidates.extend(self._revision(revision_num, pred_res, y))
            if len(candidates) > 0:
                min_revision_num = revision_num
                break
            if revision_num >= max_revision_num:
                return []

        for revision_num in range(min_revision_num + 1, min_revision_num + require_more_revision + 1):
            if revision_num > max_revision_num:
                return candidates
            candidates.extend(self._revision(revision_num, pred_res, y))
        return candidates

    @lru_cache(maxsize=None)
    def _abduce_by_search_cache(self, pred_res, y, max_revision_num, require_more_revision):
        pred_res = hashable_to_list(pred_res)
        y = hashable_to_list(y)
        return self._abduce_by_search(pred_res, y, max_revision_num, require_more_revision)

    def _dict_len(self, dic):
        if not self.GKB_flag:
            return 0
        else:
            return sum(len(c) for c in dic.values())

    def __len__(self):
        if not self.GKB_flag:
            return 0
        else:
            return sum(self._dict_len(v) for v in self.base.values())


class prolog_KB(KBBase):
    def __init__(self, pseudo_label_list, pl_file):
        super().__init__(pseudo_label_list)
        self.prolog = pyswip.Prolog()
        self.prolog.consult(pl_file)

    def logic_forward(self, pseudo_labels):
        result = list(self.prolog.query(
            "logic_forward(%s, Res)." % pseudo_labels))[0]['Res']
        if result == 'true':
            return True
        elif result == 'false':
            return False
        return result

    def _revision_pred_res(self, pred_res, revision_idx):
        import re
        revision_pred_res = pred_res.copy()
        revision_pred_res = flatten(revision_pred_res)

        for idx in revision_idx:
            revision_pred_res[idx] = 'P' + str(idx)
        revision_pred_res = reform_idx(revision_pred_res, pred_res)

        regex = r"'P\d+'"
        return re.sub(regex, lambda x: x.group().replace("'", ""), str(revision_pred_res))

    def get_query_string(self, pred_res, y, revision_idx):
        query_string = "logic_forward("
        query_string += self._revision_pred_res(pred_res, revision_idx)
        key_is_none_flag = y is None or (type(y) == list and y[0] is None)
        query_string += ",%s)." % y if not key_is_none_flag else ")."
        return query_string

    def revise_by_idx(self, pred_res, y, revision_idx):
        candidates = []
        query_string = self.get_query_string(pred_res, y, revision_idx)
        save_pred_res = pred_res
        pred_res = flatten(pred_res)
        abduce_c = [list(z.values()) for z in self.prolog.query(query_string)]
        for c in abduce_c:
            candidate = pred_res.copy()
            for i, idx in enumerate(revision_idx):
                candidate[idx] = c[i]
            candidate = reform_idx(candidate, save_pred_res)
            candidates.append(candidate)
        return candidates

class UnsupKBBase(ABC):
    def __init__(self, pseudo_label_list, prebuild_GKB=False, GKB_len_list=None, max_err=0, use_cache=False, kb_file_path=None,num=1,max_times=10000,ind=False,num_digits=1):
        """
        Initialize the KBBase instance.

        Args:
        pseudo_label_list (list): List of pseudo labels.
        prebuild_GKB (bool): Whether to prebuild the General Knowledge Base (GKB).
        GKB_len_list (list): List of lengths for the GKB.
        max_err (int): Maximum error threshold.
        use_cache (bool): Whether to use caching.
        kb_file_path (str, optional): Path to the file from which to load the pre-built knowledge base. If None, build a new knowledge base.
        """
        self.num=num
        self.max_times=max_times
        self.pseudo_label_list = pseudo_label_list
        self.prebuild_GKB = prebuild_GKB
        self.GKB_len_list = GKB_len_list
        self.max_err = max_err
        self.use_cache = use_cache
        self.base = {}
        self.ind=ind
        self.num_digits=num_digits
        self.kb_file_path=kb_file_path
        self.count=0

    def prebuild_kb(self):
        if self.kb_file_path and os.path.exists(self.kb_file_path):
            self.load_kb(self.kb_file_path)
        elif self.prebuild_GKB:
            X, Y = self._get_GKB()
            for x, y in zip(X, Y):
                # print('len(x):',x)
                self.base.setdefault(len(x), defaultdict(list))[y].append(x)
            if self.kb_file_path:
                self.save_kb(self.kb_file_path)
    
    def sort_mask_ind(self, mask_probability):
        selected_dict = {}
        max_hard_label=[np.argmax(probability) for probability in mask_probability]
        max_score=1
        # prob_list=[]
        for i in range(len(max_hard_label)):
            max_score*=mask_probability[i][max_hard_label[i]]
        sorted_V_tuple=[]
        sorted_probs=[]
        sorted_indices=[]
        max_heap_cls = Heap()
        origin_root_state=[]
        mask=[0 for _ in range(len(max_hard_label))]
        for i in range(len(max_hard_label)):
            sorted_prob_with_indices=sorted(enumerate(mask_probability[i]),key=lambda x: (-x[1],-x[0]))
            sorted_prob=[x[1] for x in sorted_prob_with_indices]
            sorted_indice = [x[0] for x in sorted_prob_with_indices]
            sorted_probs.append(sorted_prob)
            sorted_indices.append(sorted_indice)
        # for i in range(len(max_hard_label)-1,-1,-1):
            if mask[i]+1 < len(mask_probability[i]):
                origin_root_state.append((i,0,sorted_probs[i][1]/sorted_probs[i][0]))
        origin_root_state=sorted(origin_root_state,key=lambda x: (-x[2],-x[1],-x[0]))
        origin_state=[copy.deepcopy(origin_root_state)]
        state=[copy.deepcopy(origin_root_state)]
        suc_prob=sorted_probs[state[0][0][0]][1]/sorted_probs[state[0][0][0]][0]*max_score
        suc_mask=copy.deepcopy(mask)
        suc_mask[state[0][0][0]]=suc_mask[state[0][0][0]]+1
        max_heap = Heap()
        max_heap.push((-suc_prob,1))
        labels=[]
        masks=[]
        probs=[]
        label=[sorted_indices[_][mask[_]] for _ in range(len(mask))]
        masks.append(mask)
        yield label,max_score,[mask_probability[i][label[i]] for i  in range(len(label))]
        labels.append(label)
        probs.append(max_score)
        selected_dict[tuple(mask)]=max_score
        selected_dict[tuple(suc_mask)]=suc_prob
        cur_budget=1
        sum_conflict=0
        max_conflict=0
        max_son_conflict=0
        max_father_conflict=0
        while not max_heap.is_empty():
            heaptop=max_heap.pop()
            # print('heaptop:',heaptop)
            suc_prob=-heaptop[0]
            father_idx=heaptop[1]
            origin_state_f=origin_state[father_idx-1]
            state_f=state[father_idx-1]
            mask_f=masks[father_idx-1]
            # print('mask_f:',mask_f)
            prob_f=probs[father_idx-1]
            suc_mask=copy.deepcopy(mask_f)
            # print('state_f:',state_f)
            i,j,prob=state_f.pop(0)
            # print(i,j,prob)
            suc_mask[i]=j+1
            # print('suc_mask:',suc_mask)
            origin_suc_state=[]
            for _ in range(len(max_hard_label)):
                if suc_mask[_]+1 < len(mask_probability[_]):
                    origin_suc_state.append((_,suc_mask[_],sorted_probs[_][suc_mask[_]+1]/sorted_probs[_][suc_mask[_]]))
            origin_suc_state=sorted(origin_suc_state,key=lambda x: (-x[2],-x[1],-x[0]))
            origin_state.append(origin_suc_state)
            suc_state=copy.deepcopy(origin_suc_state)
            suc_label=[sorted_indices[_][suc_mask[_]] for _ in range(len(suc_mask))]
            yield suc_label,suc_prob,[mask_probability[i][suc_label[i]] for i in range(len(suc_label))]
            labels.append(suc_label)
            masks.append(suc_mask)
            probs.append(suc_prob)
            # conflict=0
            while len(suc_state) != 0:
                suc_suc_mask=copy.deepcopy(suc_mask)
                _i,_j,_prob=suc_state[0]
                suc_suc_mask[_i]=_j+1
                suc_suc_prob=suc_prob*_prob
                if tuple(suc_suc_mask) in selected_dict:
                    suc_state.pop(0)
                    continue
                else:
                    max_heap.push((-suc_suc_prob,cur_budget+1))
                    selected_dict[tuple(suc_suc_mask)]=suc_suc_prob
                    # suc_state.pop(0)
                    break
            state.append(suc_state)
            while len(state_f) != 0:
                suc_f_mask=copy.deepcopy(mask_f)
                _i,_j,_prob=state_f[0]
                suc_f_mask[_i]=_j+1
                suc_f_prob=prob_f*_prob
                if tuple(suc_f_mask) in selected_dict:
                    state_f.pop(0)
                    continue
                else:
                    max_heap.push((-suc_f_prob,father_idx))
                    selected_dict[tuple(suc_f_mask)]=suc_f_prob
                    break
            state[father_idx-1]=state_f
            cur_budget+=1

    def sort_mask(self, mask_probability):
        selected_dict = {}
        max_hard_label=[np.argmax(probability) for probability in mask_probability]
        max_score=1
        # prob_list=[]
        for i in range(len(max_hard_label)):
            max_score*=mask_probability[i][max_hard_label[i]]
        sorted_V_tuple=[]
        sorted_probs=[]
        sorted_indices=[]
        max_heap_cls = Heap()
        origin_root_state=[]
        mask=[0 for _ in range(len(max_hard_label))]
        for i in range(len(max_hard_label)):
            sorted_prob_with_indices=sorted(enumerate(mask_probability[i]),key=lambda x: (-x[1],-x[0]))
            sorted_prob=[x[1] for x in sorted_prob_with_indices]
            sorted_indice = [x[0] for x in sorted_prob_with_indices]
            sorted_probs.append(sorted_prob)
            sorted_indices.append(sorted_indice)
        # for i in range(len(max_hard_label)-1,-1,-1):
            if mask[i]+1 < len(mask_probability[i]):
                origin_root_state.append((i,0,sorted_probs[i][1]/sorted_probs[i][0],True))
        origin_root_state=sorted(origin_root_state,key=lambda x: (x[3],-x[2],-x[1],-x[0]))
        origin_state=[copy.deepcopy(origin_root_state)]
        state=[copy.deepcopy(origin_root_state)]
        suc_prob=sorted_probs[state[0][0][0]][1]/sorted_probs[state[0][0][0]][0]*max_score
        suc_mask=copy.deepcopy(mask)
        suc_mask[state[0][0][0]]=suc_mask[state[0][0][0]]+1
        max_heap = Heap()
        max_heap.push((1,-suc_prob,1))
        labels=[]
        masks=[]
        probs=[]
        label=[sorted_indices[_][mask[_]] for _ in range(len(mask))]
        masks.append(mask)
        yield label,max_score,[mask_probability[i][label[i]] for i  in range(len(label))]
        labels.append(label)
        probs.append(max_score)
        selected_dict[tuple(mask)]=max_score
        selected_dict[tuple(suc_mask)]=suc_prob
        cur_budget=1
        sum_conflict=0
        max_conflict=0
        max_son_conflict=0
        max_father_conflict=0
        while not max_heap.is_empty():
            heaptop=max_heap.pop()
            # print('heaptop:',heaptop)
            suc_revision=heaptop[0]
            suc_prob=-heaptop[1]
            father_idx=heaptop[2]
            origin_state_f=origin_state[father_idx-1]
            state_f=state[father_idx-1]
            mask_f=masks[father_idx-1]
            # print('mask_f:',mask_f)
            prob_f=probs[father_idx-1]
            suc_mask=copy.deepcopy(mask_f)
            # print('state_f:',state_f)
            i,j,prob,change=state_f.pop(0)
            # print(i,j,prob)
            suc_mask[i]=j+1
            father_revision=suc_revision-1 if change else suc_revision
            # print('suc_mask:',suc_mask)
            origin_suc_state=[]
            for _ in range(len(max_hard_label)):
                if suc_mask[_]+1 < len(mask_probability[_]):
                    origin_suc_state.append((_,suc_mask[_],sorted_probs[_][suc_mask[_]+1]/sorted_probs[_][suc_mask[_]],suc_mask[_]==0))
            origin_suc_state=sorted(origin_suc_state,key=lambda x: (x[3],-x[2],-x[1],-x[0]))
            origin_state.append(origin_suc_state)
            suc_state=copy.deepcopy(origin_suc_state)
            suc_label=[sorted_indices[_][suc_mask[_]] for _ in range(len(suc_mask))]
            yield suc_label, suc_prob,[mask_probability[i][suc_label[i]] for i in range(len(suc_label))]
            labels.append(suc_label)
            masks.append(suc_mask)
            probs.append(suc_prob)
            while len(suc_state) != 0:
                suc_suc_mask=copy.deepcopy(suc_mask)
                _i,_j,_prob,change=suc_state[0]
                suc_suc_mask[_i]=_j+1
                suc_suc_prob=suc_prob*_prob
                if tuple(suc_suc_mask) in selected_dict:
                    suc_state.pop(0)
                    continue
                else:
                    if change is True:
                        suc_suc_revision=suc_revision+1
                    else:
                        suc_suc_revision=suc_revision
                    max_heap.push((suc_suc_revision,-suc_suc_prob,cur_budget+1))
                    selected_dict[tuple(suc_suc_mask)]=suc_suc_prob
                    # suc_state.pop(0)
                    break
            state.append(suc_state)
            while len(state_f) != 0:
                suc_f_mask=copy.deepcopy(mask_f)
                _i,_j,_prob,change=state_f[0]
                suc_f_mask[_i]=_j+1
                suc_f_prob=prob_f*_prob
                if tuple(suc_f_mask) in selected_dict:
                    state_f.pop(0)
                    continue
                else:
                    if change is True:
                        suc_revision=father_revision+1
                    else:
                        suc_revision=father_revision
                    max_heap.push((suc_revision,-suc_f_prob,father_idx))
                    selected_dict[tuple(suc_f_mask)]=suc_f_prob
                    break
            state[father_idx-1]=state_f
            cur_budget+=1

    def val(self,probability):
        # probability = np.clip(probability, 1e-9, 1)
        if self.ind:
            generator=self.sort_mask_ind(probability)
        else:
            generator=self.sort_mask(probability)
        condidates=[]
        condidates_prob=[]
        condidates_weight=[]
        n=0
        times=0
        if self.num==0 or self.max_times==0:
            return candidates, condidates_prob
        # print('self.max_times:',self.max_times)
        # print('probability:',probability)
        for label, prob, weight in generator:
            res=self.logic_forward(label)
            # print('label:',label)
            # print('res:',res)
            if res:
                condidates.append(label)
                condidates_prob.append(prob)
                condidates_weight.append(weight)
                n+=1
                if n==self.num:
                    break
            times+=1
            if times==self.max_times:
                break
        # origin_label,origin_prob,origin_weight=next(generator)
        # if n == 0:
        #     candidates.append(origin_label)
        #     condidates_prob.append(origin_prob)
        #     condidates_weight.append(origin_weight)
        # condidates=condidates*(self.num//n)+condidates[:self.num%n]
        # condidates_prob=condidates_prob*(self.num//n)+condidates[:self.num%n]
        # condidates_weight=condidates_weight*(self.num//n)+condidates_weight[:self.num%n]
        return condidates, condidates_prob, condidates_weight
    def val_chess(self,probability,pos):
        # probability = np.clip(probability, 1e-9, 1)
        if self.ind:
            generator=self.sort_mask_ind(probability)
        else:
            generator=self.sort_mask(probability)
        condidates=[]
        condidates_prob=[]
        condidates_weight=[]
        n=0
        times=0
        if self.num==0 or self.max_times==0:
            return candidates, condidates_prob
        # print('self.max_times:',self.max_times)
        # print('probability:',probability)
        for label, prob, weight in generator:
            res=self.logic_forward(label,pos)
            # print('label:',label)
            # print('res:',res)
            if res:
                condidates.append(label)
                condidates_prob.append(prob)
                condidates_weight.append(weight)
                n+=1
                if n==self.num:
                    break
            times+=1
            if times==self.max_times:
                break
        # origin_label,origin_prob,origin_weight=next(generator)
        # if n == 0:
        #     candidates.append(origin_label)
        #     condidates_prob.append(origin_prob)
        #     condidates_weight.append(origin_weight)
        # condidates=condidates*(self.num//n)+condidates[:self.num%n]
        # condidates_prob=condidates_prob*(self.num//n)+condidates[:self.num%n]
        # condidates_weight=condidates_weight*(self.num//n)+condidates_weight[:self.num%n]
        return condidates, condidates_prob, condidates_weight
    # For parallel version of _get_GKB
    # def _get_XY_list(self, args):
    #     pre_x, post_x_it = args[0], args[1]
    #     XY_list = []
    #     for post_x in post_x_it:
    #         x = (pre_x,) + post_x
    #         y = self.logic_forward(x)
    #         if y not in [None, np.inf]:
    #             XY_list.append((x, y))
    #     return XY_list

    # Parallel _get_GKB
    # def _get_GKB(self):
    #     X=[]
    #     Y=[]
    #     X_True=[]
    #     for length in self.GKB_len_list:
    #         X.extend(list(product(self.pseudo_label_list, repeat=length - 1)))
    #     for l in X:
    #         if self.logic_forward(l):
    #             Y.append(True)
    #             X_True.append(l)
    #         else:
    #             Y.append(False)
    #             # Y.append(True)
    #     return X_True#, Y

    def _get_GKB(self):
        X=[]
        Y=[]
        X_True=[]
        for length in self.GKB_len_list:
            X.extend(list(product(self.pseudo_label_list, repeat=length )))
        for l in X:
            if self.logic_forward(l):
                Y.append(True)
                X_True.append(l)
            else:
                Y.append(False)
                # Y.append(True)
        return X, Y


    def save_kb(self, file_path):
        """
        Save the knowledge base to a file.

        Args:
        file_path (str): The path to the file where the knowledge base will be saved.
        """
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, 'wb') as f:
            pickle.dump(self.base, f)

    def load_kb(self, file_path):
        """
        Load the knowledge base from a file.

        Args:
        file_path (str): The path to the file from which the knowledge base will be loaded.
        """
        if os.path.exists(file_path):
            with open(file_path, 'rb') as f:
                self.base = pickle.load(f)
        else:
            print(
                f"File {file_path} not found. Starting with an empty knowledge base.")
            self.base = {}

    @abstractmethod
    def logic_forward(self, pseudo_labels):
        pass

    def abduce_candidates(self, pred_res,  max_revision_num, require_more_revision=0):
        if self.prebuild_GKB:
            return self._abduce_by_GKB(pred_res,  max_revision_num, require_more_revision)
        else:
            if not self.use_cache:
                return self._abduce_by_search(pred_res,  max_revision_num, require_more_revision)
            else:
                # print('pred_res:',pred_res)
                return self._abduce_by_search_cache(to_hashable(pred_res), max_revision_num, require_more_revision)
    
    def abduce_candidates_chess(self, pred_res, pos, max_revision_num, require_more_revision=0):
        if self.prebuild_GKB:
            return self._abduce_by_GKB_chess(pred_res, pos, max_revision_num, require_more_revision)
        else:
            if not self.use_cache:
                return self._abduce_by_search_chess(pred_res, pos, max_revision_num, require_more_revision)
            else:
                # print('pred_res:',pred_res)
                return self._abduce_by_search_cache_chess(to_hashable(pred_res),to_hashable(pos), max_revision_num, require_more_revision)

    def _find_candidate_GKB(self, pred_res):
        return self.base[len(pred_res)][True]

    def _abduce_by_GKB(self, pred_res,  max_revision_num, require_more_revision):
        # if self.base == {} or len(pred_res) not in self.GKB_len_list:
        #     return []

        all_candidates = self._find_candidate_GKB(pred_res)
        if len(all_candidates) == 0:
            return []
        # if mask:
        #     cost_list = hamming_dist(pred_res, all_candidates)
        # else:
        cost_list = hamming_dist(pred_res, all_candidates)
        min_revision_num = np.min(cost_list)
        revision_num = min(
            max_revision_num, min_revision_num + require_more_revision)
        idxs = np.where(cost_list <= revision_num)[0]
        candidates = [all_candidates[idx] for idx in idxs]
        return candidates

    def _abduce_by_GKB_chess(self, pred_res, pos, max_revision_num, require_more_revision):
        # if self.base == {} or len(pred_res) not in self.GKB_len_list:
        #     return []

        all_candidates = self._find_candidate_GKB(pred_res)
        if len(all_candidates) == 0:
            return []
        # if mask:
        #     cost_list = hamming_dist(pred_res, all_candidates)
        # else:
        cost_list = hamming_dist(pred_res, all_candidates)
        min_revision_num = np.min(cost_list)
        revision_num = min(
            max_revision_num, min_revision_num + require_more_revision)
        idxs = np.where(cost_list <= revision_num)[0]
        candidates = [all_candidates[idx] for idx in idxs]
        return candidates

    def revise_by_idx(self, pred_res, revision_idx):
        candidates = []
        abduce_c = product(self.pseudo_label_list, repeat=len(revision_idx))
        for c in abduce_c:
            candidate = pred_res.copy()
            for i, idx in enumerate(revision_idx):
                candidate[idx] = c[i]
            if self.logic_forward(candidate):
                candidates.append(candidate)
        return candidates
    
    def revise_by_idx_chess(self, pred_res,pos, revision_idx):
        candidates = []
        abduce_c = product(self.pseudo_label_list, repeat=len(revision_idx))
        for c in abduce_c:
            candidate = pred_res.copy()
            for i, idx in enumerate(revision_idx):
                candidate[idx] = c[i]
            if self.logic_forward(candidate,pos):
                candidates.append(candidate)
        return candidates

    def _revision(self, revision_num, pred_res):
        new_candidates = []
        # if mask:
            # revision_idx_list = combinations(range(len(pred_res))[mask], revision_num)
        # else:
        revision_idx_list = combinations(range(len(pred_res)), revision_num)
        for revision_idx in revision_idx_list:
            candidates = self.revise_by_idx(pred_res, revision_idx)
            new_candidates.extend(candidates)
        return new_candidates
    def _revision_chess(self, revision_num, pred_res,pos):
        new_candidates = []
        # if mask:
            # revision_idx_list = combinations(range(len(pred_res))[mask], revision_num)
        # else:
        revision_idx_list = combinations(range(len(pred_res)), revision_num)
        for revision_idx in revision_idx_list:
            candidates = self.revise_by_idx_chess(pred_res, pos, revision_idx)
            new_candidates.extend(candidates)
        return new_candidates

    def _abduce_by_search(self, pred_res, max_revision_num, require_more_revision):
        candidates = []
        for revision_num in range(len(pred_res) + 1):
            if revision_num == 0 and self.logic_forward(pred_res):
                candidates.append(pred_res)
            elif revision_num > 0:
                candidates.extend(self._revision(revision_num, pred_res))
            if len(candidates) > 0:
                min_revision_num = revision_num
                break
            if revision_num >= max_revision_num:
                return []

        for revision_num in range(min_revision_num + 1, min_revision_num + require_more_revision + 1):
            if revision_num > max_revision_num:
                return candidates
            candidates.extend(self._revision(revision_num, pred_res))
        return candidates

    def _abduce_by_search_chess(self, pred_res, pos,max_revision_num, require_more_revision):
        candidates = []
        for revision_num in range(len(pred_res) + 1):
            # print('pred_res:',pred_res)
            # print('pos:',pos)
            if revision_num == 0 and self.logic_forward(pred_res,pos):
                candidates.append(pred_res)
            elif revision_num > 0:
                candidates.extend(self._revision_chess(revision_num, pred_res,pos))
            if len(candidates) > 0:
                min_revision_num = revision_num
                break
            if revision_num >= max_revision_num:
                return []

        for revision_num in range(min_revision_num + 1, min_revision_num + require_more_revision + 1):
            if revision_num > max_revision_num:
                return candidates
            candidates.extend(self._revision_chess(revision_num, pred_res,pos))
        return candidates

    @lru_cache(maxsize=None)
    def _abduce_by_search_cache(self, pred_res,  max_revision_num, require_more_revision):
        pred_res = hashable_to_list(pred_res)
        # y = hashable_to_list(y)
        return self._abduce_by_search(pred_res, max_revision_num, require_more_revision)

    @lru_cache(maxsize=None)
    def _abduce_by_search_cache_chess(self, pred_res, pos, max_revision_num, require_more_revision):
        pred_res = hashable_to_list(pred_res)
        # y = hashable_to_list(y)
        return self._abduce_by_search_chess(pred_res, pos,max_revision_num, require_more_revision)
    def _dict_len(self, dic):
        if not self.GKB_flag:
            return 0
        else:
            return sum(len(c) for c in dic.values())

    def __len__(self):
        if not self.GKB_flag:
            return 0
        else:
            return sum(self._dict_len(v) for v in self.base.values())

class val_KB(KBBase):
    def __init__(self, pseudo_label_list,num=1,max_times=10000):
        super().__init__(pseudo_label_list)
        self.num=num
        self.max_times=max_times

    def sort_mask(self, mask_probability):
        selected_dict = {}
        max_hard_label=[np.argmax(probability) for probability in mask_probability]
        max_score=1
        # prob_list=[]
        for i in range(len(max_hard_label)):
            max_score*=mask_probability[i][max_hard_label[i]]
        sorted_V_tuple=[]
        sorted_probs=[]
        sorted_indices=[]
        max_heap_cls = Heap()
        origin_root_state=[]
        mask=[0 for _ in range(len(max_hard_label))]
        for i in range(len(max_hard_label)):
            sorted_prob_with_indices=sorted(enumerate(mask_probability[i]),key=lambda x: (-x[1],-x[0]))
            sorted_prob=[x[1] for x in sorted_prob_with_indices]
            sorted_indice = [x[0] for x in sorted_prob_with_indices]
            sorted_probs.append(sorted_prob)
            sorted_indices.append(sorted_indice)
        # for i in range(len(max_hard_label)-1,-1,-1):
            if mask[i]+1 < len(mask_probability[i]):
                origin_root_state.append((i,0,sorted_probs[i][1]/sorted_probs[i][0]))
        origin_root_state=sorted(origin_root_state,key=lambda x: (-x[2],-x[1],-x[0]))
        origin_state=[copy.deepcopy(origin_root_state)]
        state=[copy.deepcopy(origin_root_state)]
        suc_prob=sorted_probs[state[0][0][0]][1]/sorted_probs[state[0][0][0]][0]*max_score
        suc_mask=copy.deepcopy(mask)
        suc_mask[state[0][0][0]]=suc_mask[state[0][0][0]]+1
        max_heap = Heap()
        max_heap.push((-suc_prob,1))
        labels=[]
        masks=[]
        probs=[]
        label=[sorted_indices[_][mask[_]] for _ in range(len(mask))]
        masks.append(mask)
        yield label,max_score,[mask_probability[i][label[i]] for i  in range(len(label))]
        labels.append(label)
        probs.append(max_score)
        selected_dict[tuple(mask)]=max_score
        selected_dict[tuple(suc_mask)]=suc_prob
        cur_budget=1
        sum_conflict=0
        max_conflict=0
        max_son_conflict=0
        max_father_conflict=0
        while not max_heap.is_empty():
            heaptop=max_heap.pop()
            # print('heaptop:',heaptop)
            suc_prob=-heaptop[0]
            father_idx=heaptop[1]
            origin_state_f=origin_state[father_idx-1]
            state_f=state[father_idx-1]
            mask_f=masks[father_idx-1]
            # print('mask_f:',mask_f)
            prob_f=probs[father_idx-1]
            suc_mask=copy.deepcopy(mask_f)
            # print('state_f:',state_f)
            i,j,prob=state_f.pop(0)
            # print(i,j,prob)
            suc_mask[i]=j+1
            # print('suc_mask:',suc_mask)
            origin_suc_state=[]
            for _ in range(len(max_hard_label)):
                if suc_mask[_]+1 < len(mask_probability[_]):
                    origin_suc_state.append((_,suc_mask[_],sorted_probs[_][suc_mask[_]+1]/sorted_probs[_][suc_mask[_]]))
            origin_suc_state=sorted(origin_suc_state,key=lambda x: (-x[2],-x[1],-x[0]))
            origin_state.append(origin_suc_state)
            suc_state=copy.deepcopy(origin_suc_state)
            suc_label=[sorted_indices[_][suc_mask[_]] for _ in range(len(suc_mask))]
            yield suc_label,suc_prob,[mask_probability[i][suc_label[i]] for i in range(len(suc_label))]
            labels.append(suc_label)
            masks.append(suc_mask)
            probs.append(suc_prob)
            # conflict=0
            while len(suc_state) != 0:
                suc_suc_mask=copy.deepcopy(suc_mask)
                _i,_j,_prob=suc_state[0]
                suc_suc_mask[_i]=_j+1
                suc_suc_prob=suc_prob*_prob
                if tuple(suc_suc_mask) in selected_dict:
                    suc_state.pop(0)
                    continue
                else:
                    max_heap.push((-suc_suc_prob,cur_budget+1))
                    selected_dict[tuple(suc_suc_mask)]=suc_suc_prob
                    # suc_state.pop(0)
                    break
            state.append(suc_state)
            while len(state_f) != 0:
                suc_f_mask=copy.deepcopy(mask_f)
                _i,_j,_prob=state_f[0]
                suc_f_mask[_i]=_j+1
                suc_f_prob=prob_f*_prob
                if tuple(suc_f_mask) in selected_dict:
                    state_f.pop(0)
                    continue
                else:
                    max_heap.push((-suc_f_prob,father_idx))
                    selected_dict[tuple(suc_f_mask)]=suc_f_prob
                    break
            state[father_idx-1]=state_f
            cur_budget+=1

    def val(self,probability):
        generator=self.sort_mask(probability)
        condidates=[]
        condidates_prob=[]
        condidates_weight=[]
        n=0
        times=0
        if self.num==0 or self.max_times==0:
            return candidates, condidates_prob
        for label, prob, weight in generator:
            res=self.logic_forward(label)
            # print('label:',label)
            # print('res:',res)
            if res:
                condidates.append(label)
                condidates_prob.append(prob)
                condidates_weight.append(weight)
                n+=1
                if n==self.num:
                    break
            times+=1
            if times==self.max_times:
                break
        origin_label,origin_prob,origin_weight=next(generator)
        if n == 0:
            candidates.append(origin_label)
            condidates_prob.append(origin_prob)
            condidates_weight.append(origin_weight)
        condidates=condidates*(self.num//n)+condidates[:self.num%n]
        condidates_prob=condidates_prob*(self.num//n)+condidates[:self.num%n]
        condidates_weight=condidates_weight*(self.num//n)+condidates_weight[:self.num%n]
        return condidates, condidates_prob, condidates_weight

class add_KB(UnsupKBBase):
    def __init__(
        self,
        pseudo_label_list=list(range(2)),
        num=1,
        max_times=10000,
        num_digits=1,
        num_classes=2
        # prebuild_GKB=False,
        # GKB_len_list=[1 * 2],
        # max_err=0,
        # use_cache=True,
    ):
        super().__init__(
            pseudo_label_list,
        )
        self.num=num
        self.num_classes=num_classes
        self.pseudo_label_list=list(range(num_classes))
        self.num_digits=num_digits

    def logic_forward(self, nums):
        nums1,nums2,nums3=nums[:self.num_digits],nums[self.num_digits:self.num_digits*2],nums[self.num_digits*2:]
        # print('nums1:',nums1)
        # print('nums2:',nums2)
        # print('nums3:',nums3)
        # if digits_to_number(nums1) + digits_to_number(nums2)==digits_to_number(nums3):
            
        #     if ((nums3[0] != 0) | (len(nums3)==1)) is not True:
        #         print(nums)
        # print('(digits_to_number(nums1) + digits_to_number(nums2)==digits_to_number(nums3)) & ((nums3[0] != 0) | (len(nums3)==1) ):',(digits_to_number(nums1) + digits_to_number(nums2)==digits_to_number(nums3)) & ((nums3[0] != 0) | (len(nums3)==1) ))
        return (digits_to_number(nums1,num_classes=self.num_classes) + digits_to_number(nums2,num_classes=self.num_classes)==digits_to_number(nums3,num_classes=self.num_classes)) & ((nums3[0] != 0) | (len(nums3)==1) )


# prob=[[0.1,0.7,0.2],[0.3,0.2,0.5],[0.81,0.07,0.12]]

# KB=add_KB(num_classes=3,num=27)

# abduced_result=KB.sort_mask(prob)
# for revision,label, prob, weight in abduced_result:
#     print(revision,label,prob,weight)

# [1,2,0]
class UnsupKBBase_sudoku(ABC):
    def __init__(self, pseudo_label_list, prebuild_GKB=False, GKB_len_list=None, max_err=0, use_cache=True, kb_file_path=None,num=1,max_times=10000,ind=False,num_digits=1):
        """
        Initialize the KBBase instance.

        Args:
        pseudo_label_list (list): List of pseudo labels.
        prebuild_GKB (bool): Whether to prebuild the General Knowledge Base (GKB).
        GKB_len_list (list): List of lengths for the GKB.
        max_err (int): Maximum error threshold.
        use_cache (bool): Whether to use caching.
        kb_file_path (str, optional): Path to the file from which to load the pre-built knowledge base. If None, build a new knowledge base.
        """
        self.num=num
        self.max_times=max_times
        self.pseudo_label_list = pseudo_label_list
        self.prebuild_GKB = prebuild_GKB
        self.GKB_len_list = GKB_len_list
        self.max_err = max_err
        self.use_cache = use_cache
        self.base = {}
        self.ind=ind
        self.num_digits=num_digits
        self.kb_file_path=kb_file_path
        self.count=0

    def prebuild_kb(self):
        if self.kb_file_path and os.path.exists(self.kb_file_path):
            self.load_kb(self.kb_file_path)
        elif self.prebuild_GKB:
            X, Y = self._get_GKB()
            for x, y in zip(X, Y):
                # print('len(x):',x)
                self.base.setdefault(len(x), defaultdict(list))[y].append(x)
            if self.kb_file_path:
                self.save_kb(self.kb_file_path)
    
    def sort_mask_ind(self, mask_probability):
        selected_dict = {}
        max_hard_label=[np.argmax(probability) for probability in mask_probability]
        max_score=1
        # prob_list=[]
        for i in range(len(max_hard_label)):
            max_score*=mask_probability[i][max_hard_label[i]]
        sorted_V_tuple=[]
        sorted_probs=[]
        sorted_indices=[]
        max_heap_cls = Heap()
        origin_root_state=[]
        mask=[0 for _ in range(len(max_hard_label))]
        for i in range(len(max_hard_label)):
            sorted_prob_with_indices=sorted(enumerate(mask_probability[i]),key=lambda x: (-x[1],-x[0]))
            sorted_prob=[x[1] for x in sorted_prob_with_indices]
            sorted_indice = [x[0] for x in sorted_prob_with_indices]
            sorted_probs.append(sorted_prob)
            sorted_indices.append(sorted_indice)
        # for i in range(len(max_hard_label)-1,-1,-1):
            if mask[i]+1 < len(mask_probability[i]):
                origin_root_state.append((i,0,sorted_probs[i][1]/sorted_probs[i][0]))
        origin_root_state=sorted(origin_root_state,key=lambda x: (-x[2],-x[1],-x[0]))
        origin_state=[copy.deepcopy(origin_root_state)]
        state=[copy.deepcopy(origin_root_state)]
        suc_prob=sorted_probs[state[0][0][0]][1]/sorted_probs[state[0][0][0]][0]*max_score
        suc_mask=copy.deepcopy(mask)
        suc_mask[state[0][0][0]]=suc_mask[state[0][0][0]]+1
        max_heap = Heap()
        max_heap.push((-suc_prob,1))
        labels=[]
        masks=[]
        probs=[]
        label=[sorted_indices[_][mask[_]] for _ in range(len(mask))]
        masks.append(mask)
        yield label,max_score,[mask_probability[i][label[i]] for i  in range(len(label))]
        labels.append(label)
        probs.append(max_score)
        selected_dict[tuple(mask)]=max_score
        selected_dict[tuple(suc_mask)]=suc_prob
        cur_budget=1
        sum_conflict=0
        max_conflict=0
        max_son_conflict=0
        max_father_conflict=0
        while not max_heap.is_empty():
            heaptop=max_heap.pop()
            # print('heaptop:',heaptop)
            suc_prob=-heaptop[0]
            father_idx=heaptop[1]
            origin_state_f=origin_state[father_idx-1]
            state_f=state[father_idx-1]
            mask_f=masks[father_idx-1]
            # print('mask_f:',mask_f)
            prob_f=probs[father_idx-1]
            suc_mask=copy.deepcopy(mask_f)
            # print('state_f:',state_f)
            i,j,prob=state_f.pop(0)
            # print(i,j,prob)
            suc_mask[i]=j+1
            # print('suc_mask:',suc_mask)
            origin_suc_state=[]
            for _ in range(len(max_hard_label)):
                if suc_mask[_]+1 < len(mask_probability[_]):
                    origin_suc_state.append((_,suc_mask[_],sorted_probs[_][suc_mask[_]+1]/sorted_probs[_][suc_mask[_]]))
            origin_suc_state=sorted(origin_suc_state,key=lambda x: (-x[2],-x[1],-x[0]))
            origin_state.append(origin_suc_state)
            suc_state=copy.deepcopy(origin_suc_state)
            suc_label=[sorted_indices[_][suc_mask[_]] for _ in range(len(suc_mask))]
            yield suc_label,suc_prob,[mask_probability[i][suc_label[i]] for i in range(len(suc_label))]
            labels.append(suc_label)
            masks.append(suc_mask)
            probs.append(suc_prob)
            # conflict=0
            while len(suc_state) != 0:
                suc_suc_mask=copy.deepcopy(suc_mask)
                _i,_j,_prob=suc_state[0]
                suc_suc_mask[_i]=_j+1
                suc_suc_prob=suc_prob*_prob
                if tuple(suc_suc_mask) in selected_dict:
                    suc_state.pop(0)
                    continue
                else:
                    max_heap.push((-suc_suc_prob,cur_budget+1))
                    selected_dict[tuple(suc_suc_mask)]=suc_suc_prob
                    # suc_state.pop(0)
                    break
            state.append(suc_state)
            while len(state_f) != 0:
                suc_f_mask=copy.deepcopy(mask_f)
                _i,_j,_prob=state_f[0]
                suc_f_mask[_i]=_j+1
                suc_f_prob=prob_f*_prob
                if tuple(suc_f_mask) in selected_dict:
                    state_f.pop(0)
                    continue
                else:
                    max_heap.push((-suc_f_prob,father_idx))
                    selected_dict[tuple(suc_f_mask)]=suc_f_prob
                    break
            state[father_idx-1]=state_f
            cur_budget+=1

    def sort_mask(self, mask_probability):
        selected_dict = {}
        max_hard_label=[np.argmax(probability) for probability in mask_probability]
        max_score=1
        # prob_list=[]
        for i in range(len(max_hard_label)):
            max_score*=mask_probability[i][max_hard_label[i]]
        sorted_V_tuple=[]
        sorted_probs=[]
        sorted_indices=[]
        max_heap_cls = Heap()
        origin_root_state=[]
        mask=[0 for _ in range(len(max_hard_label))]
        for i in range(len(max_hard_label)):
            sorted_prob_with_indices=sorted(enumerate(mask_probability[i]),key=lambda x: (-x[1],-x[0]))
            sorted_prob=[x[1] for x in sorted_prob_with_indices]
            sorted_indice = [x[0] for x in sorted_prob_with_indices]
            sorted_probs.append(sorted_prob)
            sorted_indices.append(sorted_indice)
        # for i in range(len(max_hard_label)-1,-1,-1):
            if mask[i]+1 < len(mask_probability[i]):
                origin_root_state.append((i,0,sorted_probs[i][1]/sorted_probs[i][0],True))
        origin_root_state=sorted(origin_root_state,key=lambda x: (x[3],-x[2],-x[1],-x[0]))
        origin_state=[copy.deepcopy(origin_root_state)]
        state=[copy.deepcopy(origin_root_state)]
        suc_prob=sorted_probs[state[0][0][0]][1]/sorted_probs[state[0][0][0]][0]*max_score
        suc_mask=copy.deepcopy(mask)
        suc_mask[state[0][0][0]]=suc_mask[state[0][0][0]]+1
        max_heap = Heap()
        max_heap.push((1,-suc_prob,1))
        labels=[]
        masks=[]
        probs=[]
        label=[sorted_indices[_][mask[_]] for _ in range(len(mask))]
        masks.append(mask)
        yield label,max_score,[mask_probability[i][label[i]] for i  in range(len(label))]
        labels.append(label)
        probs.append(max_score)
        selected_dict[tuple(mask)]=max_score
        selected_dict[tuple(suc_mask)]=suc_prob
        cur_budget=1
        sum_conflict=0
        max_conflict=0
        max_son_conflict=0
        max_father_conflict=0
        while not max_heap.is_empty():
            heaptop=max_heap.pop()
            # print('heaptop:',heaptop)
            suc_revision=heaptop[0]
            suc_prob=-heaptop[1]
            father_idx=heaptop[2]
            origin_state_f=origin_state[father_idx-1]
            state_f=state[father_idx-1]
            mask_f=masks[father_idx-1]
            # print('mask_f:',mask_f)
            prob_f=probs[father_idx-1]
            suc_mask=copy.deepcopy(mask_f)
            # print('state_f:',state_f)
            i,j,prob,change=state_f.pop(0)
            # print(i,j,prob)
            suc_mask[i]=j+1
            father_revision=suc_revision-1 if change else suc_revision
            # print('suc_mask:',suc_mask)
            origin_suc_state=[]
            for _ in range(len(max_hard_label)):
                if suc_mask[_]+1 < len(mask_probability[_]):
                    origin_suc_state.append((_,suc_mask[_],sorted_probs[_][suc_mask[_]+1]/sorted_probs[_][suc_mask[_]],suc_mask[_]==0))
            origin_suc_state=sorted(origin_suc_state,key=lambda x: (x[3],-x[2],-x[1],-x[0]))
            origin_state.append(origin_suc_state)
            suc_state=copy.deepcopy(origin_suc_state)
            suc_label=[sorted_indices[_][suc_mask[_]] for _ in range(len(suc_mask))]
            yield suc_label, suc_prob,[mask_probability[i][suc_label[i]] for i in range(len(suc_label))]
            labels.append(suc_label)
            masks.append(suc_mask)
            probs.append(suc_prob)
            while len(suc_state) != 0:
                suc_suc_mask=copy.deepcopy(suc_mask)
                _i,_j,_prob,change=suc_state[0]
                suc_suc_mask[_i]=_j+1
                suc_suc_prob=suc_prob*_prob
                if tuple(suc_suc_mask) in selected_dict:
                    suc_state.pop(0)
                    continue
                else:
                    if change is True:
                        suc_suc_revision=suc_revision+1
                    else:
                        suc_suc_revision=suc_revision
                    max_heap.push((suc_suc_revision,-suc_suc_prob,cur_budget+1))
                    selected_dict[tuple(suc_suc_mask)]=suc_suc_prob
                    # suc_state.pop(0)
                    break
            state.append(suc_state)
            while len(state_f) != 0:
                suc_f_mask=copy.deepcopy(mask_f)
                _i,_j,_prob,change=state_f[0]
                suc_f_mask[_i]=_j+1
                suc_f_prob=prob_f*_prob
                if tuple(suc_f_mask) in selected_dict:
                    state_f.pop(0)
                    continue
                else:
                    if change is True:
                        suc_revision=father_revision+1
                    else:
                        suc_revision=father_revision
                    max_heap.push((suc_revision,-suc_f_prob,father_idx))
                    selected_dict[tuple(suc_f_mask)]=suc_f_prob
                    break
            state[father_idx-1]=state_f
            cur_budget+=1

    def val(self,probability):
        # probability = np.clip(probability, 1e-9, 1)
        if self.ind:
            generator=self.sort_mask_ind(probability)
        else:
            generator=self.sort_mask(probability)
        condidates=[]
        condidates_prob=[]
        condidates_weight=[]
        n=0
        times=0
        if self.num==0 or self.max_times==0:
            return candidates, condidates_prob
        for label, prob, weight in generator:
            res=self.logic_forward(label)
            # print('label:',label)
            # print('res:',res)
            if res:
                condidates.append(label)
                condidates_prob.append(prob)
                condidates_weight.append(weight)
                n+=1
                if n==self.num:
                    break
            times+=1
            if times==self.max_times:
                break
        # origin_label,origin_prob,origin_weight=next(generator)
        # if n == 0:
        #     candidates.append(origin_label)
        #     condidates_prob.append(origin_prob)
        #     condidates_weight.append(origin_weight)
        # condidates=condidates*(self.num//n)+condidates[:self.num%n]
        # condidates_prob=condidates_prob*(self.num//n)+condidates[:self.num%n]
        # condidates_weight=condidates_weight*(self.num//n)+condidates_weight[:self.num%n]
        return condidates, condidates_prob, condidates_weight

    def val_sudoku(self,probability,inputs):
        # probability = np.clip(probability, 1e-9, 1)
        mask = (inputs == 9)
        # print('inputs:',inputs)
        # print('mask:',mask)
        prob= probability[mask]
        # print('prob:',prob)
        if self.ind:
            generator=self.sort_mask_ind(prob)
        else:
            generator=self.sort_mask(prob)
        condidates=[]
        condidates_prob=[]
        condidates_weight=[]
        n=0
        times=0
        if self.num==0 or self.max_times==0:
            return candidates, condidates_prob
        # print('self.max_times:',self.max_times)
        # print('prob:',prob.shape)
        # count=0
        for label, prob, weight in generator:
            # count+=1
            # print('inputs:',inputs)
            # print('mask:',mask)
            nums=copy.deepcopy(torch.Tensor(inputs))
            # res=False
            # if label==[6,1,8,1]:

            #     print('nums:',nums)
            # print('mask:',mask)
            # inputs=torch.Tensor(inputs)
            nums[mask]=torch.Tensor(label).float()
            nums=nums.long().tolist()
            res=self.logic_forward(nums)
            # if label==[6,1,8,1]:
            #     print('nums:',nums)
            #     print('res:',res)
            # print('label:',label)
            # print('res:',res)
            if res:
                condidates.append(nums)
                condidates_prob.append(prob)
                condidates_weight.append(weight)
                n+=1
                if n==self.num:
                    break
            times+=1
        # print('times:',times)
            # if times==self.max_times:
            #     break
        # origin_label,origin_prob,origin_weight=next(generator)
        # if n == 0:
        #     candidates.append(origin_label)
        #     condidates_prob.append(origin_prob)
        #     condidates_weight.append(origin_weight)
        # condidates=condidates*(self.num//n)+condidates[:self.num%n]
        # condidates_prob=condidates_prob*(self.num//n)+condidates[:self.num%n]
        # condidates_weight=condidates_weight*(self.num//n)+condidates_weight[:self.num%n]
        return condidates, condidates_prob, condidates_weight
    # For parallel version of _get_GKB
    # def _get_XY_list(self, args):
    #     pre_x, post_x_it = args[0], args[1]
    #     XY_list = []
    #     for post_x in post_x_it:
    #         x = (pre_x,) + post_x
    #         y = self.logic_forward(x)
    #         if y not in [None, np.inf]:
    #             XY_list.append((x, y))
    #     return XY_list

    # Parallel _get_GKB
    # def _get_GKB(self):
    #     X=[]
    #     Y=[]
    #     X_True=[]
    #     for length in self.GKB_len_list:
    #         X.extend(list(product(self.pseudo_label_list, repeat=length - 1)))
    #     for l in X:
    #         if self.logic_forward(l):
    #             Y.append(True)
    #             X_True.append(l)
    #         else:
    #             Y.append(False)
    #             # Y.append(True)
    #     return X_True#, Y

    def _get_GKB(self):
        X=[]
        Y=[]
        X_True=[]
        for length in self.GKB_len_list:
            X.extend(list(product(self.pseudo_label_list, repeat=length )))
        for l in X:
            if self.logic_forward(l):
                Y.append(True)
                X_True.append(l)
            else:
                Y.append(False)
                # Y.append(True)
        return X, Y


    def save_kb(self, file_path):
        """
        Save the knowledge base to a file.

        Args:
        file_path (str): The path to the file where the knowledge base will be saved.
        """
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, 'wb') as f:
            pickle.dump(self.base, f)

    def load_kb(self, file_path):
        """
        Load the knowledge base from a file.

        Args:
        file_path (str): The path to the file from which the knowledge base will be loaded.
        """
        if os.path.exists(file_path):
            with open(file_path, 'rb') as f:
                self.base = pickle.load(f)
        else:
            print(
                f"File {file_path} not found. Starting with an empty knowledge base.")
            self.base = {}

    @abstractmethod
    def logic_forward(self, pseudo_labels):
        pass

    def abduce_candidates(self, pred_res,  max_revision_num, require_more_revision=0):
        if self.prebuild_GKB:
            return self._abduce_by_GKB(pred_res,  max_revision_num, require_more_revision)
        else:
            if not self.use_cache:
                return self._abduce_by_search(pred_res,  max_revision_num, require_more_revision)
            else:
                return self._abduce_by_search_cache(to_hashable(pred_res), max_revision_num, require_more_revision)

    def abduce_candidates_sudoku(self, pred_res, inputs, max_revision_num, require_more_revision=0):
        mask=(inputs==9)
        # print('mask:',mask)
        # print('~mask:',~mask)
        # print('pred_res:',pred_res)
        # print('inputs:',inputs)
        pred_res=torch.tensor(pred_res)
        pred_res[~mask]=inputs[~mask].long()
        pred_res=pred_res.tolist()
        
        if self.prebuild_GKB:
            return self._abduce_by_GKB(pred_res, max_revision_num, require_more_revision,mask)
        else:
            if not self.use_cache:
                return self._abduce_by_search(pred_res,  max_revision_num, require_more_revision,mask)
            else:
                return self._abduce_by_search_cache(to_hashable(pred_res), max_revision_num, require_more_revision,mask)

    def _find_candidate_GKB(self, pred_res):
        return self.base[len(pred_res)][True]

    def _abduce_by_GKB(self, pred_res,  max_revision_num, require_more_revision,mask=None):
        # if self.base == {} or len(pred_res) not in self.GKB_len_list:
        #     return []

        all_candidates = self._find_candidate_GKB(pred_res)
        if len(all_candidates) == 0:
            return []

        cost_list = hamming_dist(pred_res, all_candidates[:,mask])
        min_revision_num = np.min(cost_list)
        revision_num = min(
            max_revision_num, min_revision_num + require_more_revision)
        idxs = np.where(cost_list <= revision_num)[0]
        candidates = [all_candidates[idx] for idx in idxs]
        return candidates

    def revise_by_idx(self, pred_res, revision_idx):
        candidates = []
        abduce_c = product(self.pseudo_label_list, repeat=len(revision_idx))
        for c in abduce_c:
            candidate = pred_res.copy()
            for i, idx in enumerate(revision_idx):
                candidate[idx] = c[i]
            if self.logic_forward(candidate):
                candidates.append(candidate)
        return candidates

    def _revision(self, revision_num, pred_res,mask):
        new_candidates = []
        if mask is not None:
            idx_list=torch.tensor(range(len(pred_res)))[mask].tolist()
            revision_idx_list = combinations(idx_list, revision_num)
        else:
            revision_idx_list = combinations(range(len(pred_res)), revision_num)

        for revision_idx in revision_idx_list:
            candidates = self.revise_by_idx(pred_res, revision_idx)
            new_candidates.extend(candidates)
        return new_candidates

    def _abduce_by_search(self, pred_res, max_revision_num, require_more_revision,mask):
        candidates = []
        for revision_num in range(len(pred_res) + 1):
            if revision_num == 0 and self.logic_forward(pred_res):
                candidates.append(pred_res)
            elif revision_num > 0:
                candidates.extend(self._revision(revision_num, pred_res,mask))
            if len(candidates) > 0:
                min_revision_num = revision_num
                break
            if revision_num >= max_revision_num:
                return []

        for revision_num in range(min_revision_num + 1, min_revision_num + require_more_revision + 1):
            if revision_num > max_revision_num:
                return candidates
            candidates.extend(self._revision(revision_num, pred_res,mask))
        return candidates

    @lru_cache(maxsize=None)
    def _abduce_by_search_cache(self, pred_res,  max_revision_num, require_more_revision,mask):
        pred_res = hashable_to_list(pred_res)
        # y = hashable_to_list(y)
        return self._abduce_by_search(pred_res, max_revision_num, require_more_revision,mask)

    def _dict_len(self, dic):
        if not self.GKB_flag:
            return 0
        else:
            return sum(len(c) for c in dic.values())

    def __len__(self):
        if not self.GKB_flag:
            return 0
        else:
            return sum(self._dict_len(v) for v in self.base.values())
