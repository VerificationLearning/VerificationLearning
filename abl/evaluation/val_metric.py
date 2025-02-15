from typing import Optional, Sequence, Callable
from .base_metric import BaseMetric
import copy
import heapq
import numpy as np
import wandb
from itertools import chain
from sklearn.metrics import confusion_matrix

class Heap:
    def __init__(self):
        self._heap = []

    def push(self, item):
        heapq.heappush(self._heap, item)

    def pop(self):
        return heapq.heappop(self._heap)

    def is_empty(self):
        return len(self._heap) == 0

class ValMetric(BaseMetric):
    def __init__(self, prefix: Optional[str] = None,num=1,max_times=10000000000) -> None:
        super().__init__(prefix)
        self.num=1
        self.max_times=max_times
        self.y_pred = []
        self.y_gt = []
        self.y_val = []
        self.pred_results = []
    def clear(self):
        self.y_pred = []
        self.y_gt = []
        self.y_val = []
        self.pred_results = []
    def process(self, data_samples: Sequence[dict]) -> None:
        pred_pseudo_label = data_samples["pred_pseudo_label"]
        pred_prob = data_samples["pred_prob"]
        gt_pseudo_label = data_samples["gt_pseudo_label"]
        self.logic_forward = data_samples["logic_forward"]

        if not len(pred_pseudo_label) == len(gt_pseudo_label):
            raise ValueError("lengthes of pred_pseudo_label and gt_pseudo_label should be equal")
        self.y_gt.extend(gt_pseudo_label)
        self.y_pred.extend(pred_pseudo_label)
        val_res=[self.val(prob) for prob in pred_prob]
        val_pseudo_label,val_prob=[res[0][0] for res in val_res],[res[1][0] for res in val_res]
        # print('val_pseudo_label:',val_pseudo_label)
        # print('val_prob:',val_prob)
        self.y_val.extend(val_pseudo_label)
        for val_z, z in zip(val_pseudo_label, gt_pseudo_label):
            correct_num = 0
            for pred_symbol, symbol in zip(val_z, z):
                if pred_symbol == symbol:
                    correct_num += 1
            self.results.append(correct_num / len(z))
        # self.y_val.extend(val_pseudo_label)
        for pred_z, z in zip(pred_pseudo_label, gt_pseudo_label):
            correct_num = 0
            for pred_symbol, symbol in zip(pred_z, z):
                if pred_symbol == symbol:
                    correct_num += 1
            self.pred_results.append(correct_num / len(z))
        
    def sort_mask(self, mask_probability):
        selected_dict = {}
        max_hard_label=[np.argmax(probability) for probability in mask_probability]
        max_score=1
        for i in range(len(max_hard_label)):
            max_score*=mask_probability[i][max_hard_label[i]]
        sorted_V_tuple=[]
        sorted_probs=[]
        sorted_indices=[]
        max_heap_cls = Heap()
        origin_root_state=[]
        mask=[0 for _ in range(len(max_hard_label))]
        for i in range(len(max_hard_label)):
            sorted_prob_with_indices=sorted(enumerate(mask_probability[i]),key=lambda x: x[1],reverse=True)
            sorted_prob=[x[1] for x in sorted_prob_with_indices]
            sorted_indice = [x[0] for x in sorted_prob_with_indices]
            sorted_probs.append(sorted_prob)
            sorted_indices.append(sorted_indice)
            if mask[i]+1 < len(mask_probability[i]):
                origin_root_state.append((i,0,sorted_probs[i][1]/sorted_probs[i][0]))
        origin_root_state=sorted(origin_root_state,key=lambda x: x[2],reverse=True)
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
        yield label,max_score
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
            origin_suc_state=sorted(origin_suc_state,key=lambda x: x[2],reverse=True)
            origin_state.append(origin_suc_state)
            suc_state=copy.deepcopy(origin_suc_state)
            suc_label=[sorted_indices[_][suc_mask[_]] for _ in range(len(suc_mask))]
            yield suc_label,suc_prob
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
        # return labels

    def val(self,probability):
        # print('probability:',probability)
        generator=self.sort_mask(probability)
        condidates=[]
        condidates_prob=[]
        n=0
        times=0
        if self.num==0 or self.max_times==0:
            return candidates, condidates_prob
        for label, prob in generator:
            if self.logic_forward(label):
                condidates.append(label)
                condidates_prob.append(prob)
                n+=1
                if n==self.num:
                    break
            times+=1
            if times==self.max_times:
                break
        return condidates, condidates_prob

    def compute_metrics(self, results: list) -> dict:
        metrics = dict()
        metrics["character_accuracy"] = sum(results) / len(results)
        metrics["pred_character_accuracy"] = sum(self.pred_results) / len(self.pred_results)
        print('character_accuracy:',metrics["character_accuracy"] )
        print('pred_character_accuracy:',metrics["pred_character_accuracy"] )
        # print('self.y_val',self.y_val)
        # print('self.y_pred',self.y_pred)
        # print('self.y_gt',self.y_gt)
        print('val_cm:',confusion_matrix(list(chain(*self.y_val)),list(chain(*self.y_gt))))
        print('pred_cm:',confusion_matrix(list(chain(*self.y_pred)),list(chain(*self.y_gt))))
        cm = wandb.plot.confusion_matrix(preds=list(chain(*self.y_pred)), y_true=list(chain(*self.y_gt)))
        metrics['Confusing Matrix'] = cm
        print('cm:',cm)
        self.clear()
        return metrics

class ValChessMetric(BaseMetric):
    def __init__(self, prefix: Optional[str] = None,num=1,max_times=100000) -> None:
        super().__init__(prefix)
        self.num=1
        self.max_times=max_times
        self.y_pred = []
        self.y_gt = []
        self.y_val = []
        self.pred_results = []

    def process(self, data_samples: Sequence[dict]) -> None:
        pred_pseudo_label = data_samples["pred_pseudo_label"]
        pred_prob = data_samples["pred_prob"]
        gt_pseudo_label = data_samples["gt_pseudo_label"]
        self.logic_forward = data_samples["logic_forward"]
        pos=data_samples["P"]

        if not len(pred_pseudo_label) == len(gt_pseudo_label):
            raise ValueError("lengthes of pred_pseudo_label and gt_pseudo_label should be equal")
        self.y_gt.extend(gt_pseudo_label)
        self.y_pred.extend(pred_pseudo_label)
        val_res=[self.val(_prob,_pos) for _prob,_pos in zip(pred_prob,pos)]
        val_pseudo_label,val_prob=[res[0][0] for res in val_res],[res[1][0] for res in val_res]
        # print('val_pseudo_label:',val_pseudo_label)
        # print('val_prob:',val_prob)
        self.y_val.extend(val_pseudo_label)
        for val_z, z in zip(val_pseudo_label, gt_pseudo_label):
            correct_num = 0
            for pred_symbol, symbol in zip(val_z, z):
                if pred_symbol == symbol:
                    correct_num += 1
            self.results.append(correct_num / len(z))
        # self.y_val.extend(val_pseudo_label)
        for pred_z, z in zip(pred_pseudo_label, gt_pseudo_label):
            correct_num = 0
            for pred_symbol, symbol in zip(pred_z, z):
                if pred_symbol == symbol:
                    correct_num += 1
            self.pred_results.append(correct_num / len(z))
        
    def sort_mask(self, mask_probability):
        selected_dict = {}
        max_hard_label=[np.argmax(probability) for probability in mask_probability]
        max_score=1
        for i in range(len(max_hard_label)):
            max_score*=mask_probability[i][max_hard_label[i]]
        sorted_V_tuple=[]
        sorted_probs=[]
        sorted_indices=[]
        max_heap_cls = Heap()
        origin_root_state=[]
        mask=[0 for _ in range(len(max_hard_label))]
        for i in range(len(max_hard_label)):
            sorted_prob_with_indices=sorted(enumerate(mask_probability[i]),key=lambda x: x[1],reverse=True)
            sorted_prob=[x[1] for x in sorted_prob_with_indices]
            sorted_indice = [x[0] for x in sorted_prob_with_indices]
            sorted_probs.append(sorted_prob)
            sorted_indices.append(sorted_indice)
            if mask[i]+1 < len(mask_probability[i]):
                origin_root_state.append((i,0,sorted_probs[i][1]/sorted_probs[i][0]))
        origin_root_state=sorted(origin_root_state,key=lambda x: x[2],reverse=True)
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
        yield label,max_score
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
            origin_suc_state=sorted(origin_suc_state,key=lambda x: x[2],reverse=True)
            origin_state.append(origin_suc_state)
            suc_state=copy.deepcopy(origin_suc_state)
            suc_label=[sorted_indices[_][suc_mask[_]] for _ in range(len(suc_mask))]
            yield suc_label,suc_prob
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
        # return labels

    def val(self,probability,pos):
        # print('probability:',probability)
        generator=self.sort_mask(probability)
        condidates=[]
        condidates_prob=[]
        n=0
        times=0
        if self.num==0 or self.max_times==0:
            return candidates, condidates_prob
        for label, prob in generator:
            if self.logic_forward(label,pos):
                condidates.append(label)
                condidates_prob.append(prob)
                n+=1
                if n==self.num:
                    break
            times+=1
            if times==self.max_times:
                break
        return condidates, condidates_prob

    def compute_metrics(self, results: list) -> dict:
        metrics = dict()
        metrics["character_accuracy"] = sum(results) / len(results)
        metrics["pred_character_accuracy"] = sum(self.pred_results) / len(self.pred_results)
        print('character_accuracy:',metrics["character_accuracy"] )
        print('pred_character_accuracy:',metrics["pred_character_accuracy"] )
        # print('self.y_val',self.y_val)
        # print('self.y_pred',self.y_pred)
        # print('self.y_gt',self.y_gt)
        print('val_cm:',confusion_matrix(list(chain(*self.y_val)),list(chain(*self.y_gt))))
        print('pred_cm:',confusion_matrix(list(chain(*self.y_pred)),list(chain(*self.y_gt))))
        cm = wandb.plot.confusion_matrix(preds=list(chain(*self.y_pred)), y_true=list(chain(*self.y_gt)))
        metrics['Confusing Matrix'] = cm
        print('cm:',cm)
        return metrics