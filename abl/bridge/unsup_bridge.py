from abl.evaluation import BaseMetric
from abl.learning import ABLModel
from abl.reasoning import ReasonerBase, WeaklySupervisedReasoner
from abl.learning import ABLModel, WeaklySupervisedABLModel
from abl.reasoning import ReasonerBase
from abl.evaluation import BaseMetric
from abl.bridge.base_bridge import BaseBridge
from typing import List, Union, Any, Tuple, Dict, Optional
from numpy import ndarray
import wandb
from torch.utils.data import DataLoader
import copy
from abl.dataset import BridgeDataset,BridgeDataset_ulb,BridgeDataset_ulb_chess
from abl.utils.logger import print_log
import numpy as np
import time
import heapq
from sklearn.metrics import confusion_matrix
import torch

class UnsupBridge(BaseBridge):
    def __init__(
        self,
        model: ABLModel,
        abducer: ReasonerBase,
        metric_list: List[BaseMetric],
        use_prob=False,
        use_weight=False,
        val=False,
        require_more_revision=0,
        sudoku=False
    ) -> None:
        super().__init__(model, abducer)
        self.metric_list = metric_list
        self.use_prob=use_prob
        self.use_weight=use_weight
        self.val=val
        self.sudoku=sudoku
        self.require_more_revision=require_more_revision

    def predict(self, X,prior=None) -> Tuple[List[List[Any]], ndarray]:
        # print('brige_predict:',prior)
        pred_res = self.model.predict(X,prior=prior)
        pred_idx, pred_prob = pred_res["label"], pred_res["prob"]
        return pred_idx, pred_prob

    # def abduce_pseudo_label(
    #     self,
    #     pred_prob: ndarray,
    #     pred_pseudo_label: List[List[Any]],
    #     require_more_revision=10
    # ) -> List[List[Any]]:
    #     return self.abducer.batch_abduce(pred_prob, pred_pseudo_label,inputs,require_more_revision=require_more_revision)
    # def val_pseudo_label(
    #     self,
    #     pred_prob: ndarray,
    #     # pred_pseudo_label: List[List[Any]]
    # ) -> List[List[Any]]:
    #     return self.abducer.batch_val(pred_prob)

    def abduce_pseudo_label(
        self,
        pred_prob: ndarray,
        pred_pseudo_label: List[List[Any]],
        inputs=None,
        require_more_revision=10
    ) -> List[List[Any]]:
        if self.sudoku:
            return self.abducer.batch_abduce_sudoku(pred_prob, pred_pseudo_label,inputs,require_more_revision=require_more_revision)
        else:
            return self.abducer.batch_abduce(pred_prob, pred_pseudo_label,require_more_revision=require_more_revision)
    def val_pseudo_label(
        self,
        pred_prob: ndarray,
        inputs=None
        # pred_pseudo_label: List[List[Any]]
    ) -> List[List[Any]]:
        if self.sudoku:
            return self.abducer.batch_val_sudoku(pred_prob,inputs=inputs)
        else:
            return self.abducer.batch_val(pred_prob)

    def abduce_pseudo_label_chess(
        self,
        pred_prob: ndarray,
        pred_pseudo_label: List[List[Any]],
        pos=None,
        # inputs=None,
        require_more_revision=10
    ) -> List[List[Any]]:
        return self.abducer.batch_abduce_chess(pred_prob, pred_pseudo_label,pos=pos,require_more_revision=require_more_revision)
    
    def val_pseudo_label_chess(
        self,
        pred_prob: ndarray,
        # inputs=None
        pos=None
        # pred_pseudo_label: List[List[Any]]
    ) -> List[List[Any]]:
        return self.abducer.batch_val_chess(pred_prob,pos=pos)

    def idx_to_pseudo_label(self, idx: List[List[Any]], mapping: Dict = None) -> List[List[Any]]:
        if mapping is None:
            mapping = self.abducer.mapping
        return [[mapping[_idx] for _idx in sub_list] for sub_list in idx]

    def pseudo_label_to_idx(self, pseudo_label: List[List[Any]], mapping: Dict = None) -> List[List[Any]]:
        if mapping is None:
            mapping = self.abducer.remapping

        def recursive_map(func, nested_list):
            if isinstance(nested_list, (list, tuple)):
                return [recursive_map(func, x) for x in nested_list]
            else:
                return func(nested_list)

        return recursive_map(lambda x: mapping[x], pseudo_label)

    def train(
        self,
        train_data: Tuple[List[List[Any]], Optional[List[List[Any]]], List[List[Any]]],
        epochs: int = 50,
        batch_size: Union[int, float] = -1,
        eval_interval: int = 1,
        test_data: Tuple[List[List[Any]], Optional[List[List[Any]]], List[List[Any]]] = None,
        prior=None
    ):
        dataset = BridgeDataset_ulb(*train_data)
        data_loader = DataLoader(
            dataset,
            batch_size=batch_size,
            collate_fn=lambda data_list: [list(data) for data in zip(*data_list)],
        )
        for epoch in range(epochs):
            for seg_idx, (X, Z) in enumerate(data_loader):
                # print('X:',X)
                # print('X[0]:',X[0])
                # print('Z[0]:',Z[0])
                # print('prior_train:',prior)
                if self.val:
                    pred_idx, pred_prob = self.predict(X,prior=prior)
                else:
                    pred_idx, pred_prob = self.predict(X)
                # print('pred_idx:',len(pred_idx))
                pred_pseudo_label = self.idx_to_pseudo_label(pred_idx)
                # print('pse:',pred_pseudo_label[0])
                # print('X:',len(X))
                # print('Z:',len(Z))
                # abduced_pseudo_label = self.abduce_pseudo_label(pred_prob, pred_pseudo_label,require_more_revision=self.require_more_revision)
                
                # print('abduced_pseudo_label:',abduced_pseudo_label)
                # print('X:',X)
                if self.val:
                    abduced_pseudo_label=self.val_pseudo_label(pred_prob,inputs=X)
                else:
                    abduced_pseudo_label = self.abduce_pseudo_label(pred_prob, pred_pseudo_label,inputs=X,require_more_revision=self.require_more_revision)
                # print('pred_idx:',len(pred_idx))
                # print('pred_pseudo_label:',len(pred_pseudo_label))
                # abduced_pseudo_label=torch.Tensor(abduced_pseudo_label).long()
                # for i in range(len(abduced_pseudo_label)):
                #     if abduced_pseudo_label[i]!=val_pseudo_label[i]:
                #         print('i:',i)
                #         # print('pred_prob[i]:',pred_prob[i])
                #         print('pred_idx[i]:',pred_idx[i])
                #         print('abduced_pseudo_label[i]:',abduced_pseudo_label[i])

                #         print('val_pseudo_label[i]:',val_pseudo_label[i])                
                # print('pred_idx:',pred_idx)
                # print('pred_prob:',pred_prob)
                # print('results:',abduced_pseudo_label)
                # print('ground:',)
                
                    # print('val_pseudo_label:',val_pseudo_label)
                # if self.val:

                    # abduced_pseudo_label=val_pseudo_label

                # print('abduced_pseudo_label:',abduced_pseudo_label)
                abduced_label = self.pseudo_label_to_idx(abduced_pseudo_label)
                # print('abd:',abduced_label)
                # abduced_label=torch.Tensor(abduced_label)
                # print('X:',X.shape)
                # abduced_label=[]
                # abduced_label=[]
                # print('abduced_label:',len(abduced_label))
                if self.sudoku:
                    loss = self.model.train_sudoku(X, abduced_label)
                else:
                    loss = self.model.train(X, abduced_label)
                # print('loss:',loss)
                # print('Z:',Z)
                # print('self.pseudo_label_to_idx(Z)',self.pseudo_label_to_idx(Z))
                abduce_acc = np.mean([i==j for i,j in zip(abduced_label, self.pseudo_label_to_idx(Z))])
                print_log(
                    f"Epoch(train) [{epoch + 1}] [{(seg_idx + 1):3}/{len(data_loader)}] model loss is {loss:.5f}",
                    logger="current",
                )
                wandb.log({"train loss": loss, "abduce_acc":abduce_acc})

                # for k in range(len(results[0][0])):
                #     abduced_pseudo_label, abduced_prob, abduced_weight = [res[0][k] for res in results], [res[1][k] for res in results], [res[2][k] for res in results]
                #     abduced_label = self.pseudo_label_to_idx(abduced_pseudo_label)
                #     if self.use_prob:
                #         abduced_weight = [[abduced_prob[i] for j in range(len(abduced_label[i]))] for i in range(len(abduced_label))]
                #     if self.use_weight or self.use_prob:
                #         loss = self.model.train(X, abduced_label,weight=abduced_weight)
                #     else:
                #         loss = self.model.train(X, abduced_label)
                #     abduce_acc = np.mean([i == j for i, j in zip(abduced_label, self.pseudo_label_to_idx(Z))])
                #     print_log(
                #         f"Epoch(train) [{epoch + 1}] [{(seg_idx + 1):3}/{len(data_loader)}] model loss is {loss:.5f}",
                #         logger="current",
                #     )
                #     print('abduce_acc:',abduce_acc)
                #     wandb.log({"train loss": loss, "abduce_acc":abduce_acc})
            if (epoch + 1) % eval_interval == 0 or epoch == epochs - 1:
                print_log(f"Evaluation start: Epoch(val) [{epoch}]", logger="current")
                if test_data:
                    # print('test')
                    self.test(test_data)
    def train_chess(
        self,
        train_data: Tuple[List[List[Any]], Optional[List[List[Any]]], List[List[Any]]],
        epochs: int = 50,
        batch_size: Union[int, float] = -1,
        eval_interval: int = 1,
        test_data: Tuple[List[List[Any]], Optional[List[List[Any]]], List[List[Any]]] = None,
        prior=None
    ):
        dataset = BridgeDataset_ulb_chess(*train_data)
        data_loader = DataLoader(
            dataset,
            batch_size=batch_size,
            collate_fn=lambda data_list: [list(data) for data in zip(*data_list)],
        )
        for epoch in range(epochs):
            for seg_idx, (X, Z,P) in enumerate(data_loader):
                # print('X:',X)
                # print('X[0]:',X[0])
                # print('Z[0]:',Z[0])
                # print('prior_train:',prior)
                if self.val:
                    pred_idx, pred_prob = self.predict(X,prior=prior)
                else:
                    pred_idx, pred_prob = self.predict(X)
                # print('pred_idx:',len(pred_idx))
                pred_pseudo_label = self.idx_to_pseudo_label(pred_idx)
                # print('X:',len(X))
                # print('Z:',len(Z))
                # abduced_pseudo_label = self.abduce_pseudo_label(pred_prob, pred_pseudo_label,require_more_revision=self.require_more_revision)
                
                # print('abduced_pseudo_label:',abduced_pseudo_label)
                # print('X:',X)
                if self.val:
                    abduced_pseudo_label=self.val_pseudo_label_chess(pred_prob,pos=P)
                else:
                    abduced_pseudo_label = self.abduce_pseudo_label_chess(pred_prob, pred_pseudo_label,pos=P,require_more_revision=self.require_more_revision)
                # print('pred_idx:',len(pred_idx))
                # print('pred_pseudo_label:',len(pred_pseudo_label))
                # abduced_pseudo_label=torch.Tensor(abduced_pseudo_label).long()
                # for i in range(len(abduced_pseudo_label)):
                #     if abduced_pseudo_label[i]!=val_pseudo_label[i]:
                #         print('i:',i)
                #         # print('pred_prob[i]:',pred_prob[i])
                #         print('pred_idx[i]:',pred_idx[i])
                #         print('abduced_pseudo_label[i]:',abduced_pseudo_label[i])

                #         print('val_pseudo_label[i]:',val_pseudo_label[i])                
                # print('pred_idx:',pred_idx)
                # print('pred_prob:',pred_prob)
                # print('results:',abduced_pseudo_label)
                # print('ground:',)
                
                    # print('val_pseudo_label:',val_pseudo_label)
                # if self.val:

                    # abduced_pseudo_label=val_pseudo_label

                # print('abduced_pseudo_label:',abduced_pseudo_label)
                abduced_label = self.pseudo_label_to_idx(abduced_pseudo_label)
                # abduced_label=torch.Tensor(abduced_label)
                # print('X:',X.shape)
                # abduced_label=[]
                # abduced_label=[]
                # print('abduced_label:',len(abduced_label))
                if self.sudoku:
                    loss = self.model.train_sudoku(X, abduced_label)
                else:
                    loss = self.model.train(X, abduced_label)
                # print('loss:',loss)
                # print('Z:',Z)
                # print('self.pseudo_label_to_idx(Z)',self.pseudo_label_to_idx(Z))
                abduce_acc = np.mean([i==j for i,j in zip(abduced_label, self.pseudo_label_to_idx(Z))])
                print_log(
                    f"Epoch(train) [{epoch + 1}] [{(seg_idx + 1):3}/{len(data_loader)}] model loss is {loss:.5f}",
                    logger="current",
                )
                wandb.log({"train loss": loss, "abduce_acc":abduce_acc})

                # for k in range(len(results[0][0])):
                #     abduced_pseudo_label, abduced_prob, abduced_weight = [res[0][k] for res in results], [res[1][k] for res in results], [res[2][k] for res in results]
                #     abduced_label = self.pseudo_label_to_idx(abduced_pseudo_label)
                #     if self.use_prob:
                #         abduced_weight = [[abduced_prob[i] for j in range(len(abduced_label[i]))] for i in range(len(abduced_label))]
                #     if self.use_weight or self.use_prob:
                #         loss = self.model.train(X, abduced_label,weight=abduced_weight)
                #     else:
                #         loss = self.model.train(X, abduced_label)
                #     abduce_acc = np.mean([i == j for i, j in zip(abduced_label, self.pseudo_label_to_idx(Z))])
                #     print_log(
                #         f"Epoch(train) [{epoch + 1}] [{(seg_idx + 1):3}/{len(data_loader)}] model loss is {loss:.5f}",
                #         logger="current",
                #     )
                #     print('abduce_acc:',abduce_acc)
                #     wandb.log({"train loss": loss, "abduce_acc":abduce_acc})
            if (epoch + 1) % eval_interval == 0 or epoch == epochs - 1:
                print_log(f"Evaluation start: Epoch(val) [{epoch}]", logger="current")
                if test_data:
                    self.test_chess(test_data)

    def _valid(self, data_loader, tag=""):
        # print('_valid')
        for (X, Z) in data_loader:
            # print('X:',X)
            # print('Z:',Z)
            pred_idx, pred_prob = self.predict(X)
            pred_pseudo_label = self.idx_to_pseudo_label(pred_idx)
            mask=(X==9)
            # print('X:',X)
            data_samples = dict(
                pred_idx=pred_idx,
                pred_prob=pred_prob,
                pred_pseudo_label=pred_pseudo_label,
                gt_pseudo_label=Z,
                logic_forward=self.abducer.kb.logic_forward,
                mask=mask if self.sudoku else None,
                X=X
                # P=P
            )
            for metric in self.metric_list:
                metric.process(data_samples)
        res = dict()
        for metric in self.metric_list:
            res.update(metric.evaluate())
        wandb.log({f"{k}/{tag}": v for k, v in res.items()})
        msg = "Evaluation ended, "
        try:
            for k, v in res.items():
                msg += k + f": {v:.3f} "
            print_log(msg, logger="current")
        except:
            pass

    def _valid_chess(self, data_loader, tag=""):
        for (X, Z,P) in data_loader:
            # print('X:',X)
            # print('Z:',Z)
            pred_idx, pred_prob = self.predict(X)
            pred_pseudo_label = self.idx_to_pseudo_label(pred_idx)
            mask=(X==9)
            # print('X:',X)
            data_samples = dict(
                pred_idx=pred_idx,
                pred_prob=pred_prob,
                pred_pseudo_label=pred_pseudo_label,
                gt_pseudo_label=Z,
                logic_forward=self.abducer.kb.logic_forward,
                mask=mask if self.sudoku else None,
                X=X,
                P=P
            )
            for metric in self.metric_list:
                metric.process(data_samples)

        res = dict()
        for metric in self.metric_list:
            res.update(metric.evaluate())
        wandb.log({f"{k}/{tag}": v for k, v in res.items()})
        msg = "Evaluation ended, "
        try:
            for k, v in res.items():
                msg += k + f": {v:.3f} "
            print_log(msg, logger="current")
        except:
            pass

    def valid(self, valid_data, batch_size=1000, tag="valid"):
        dataset = BridgeDataset_ulb(*valid_data)
        data_loader = DataLoader(
            dataset,
            batch_size=batch_size,
            collate_fn=lambda data_list: [list(data) for data in zip(*data_list)]
        )
        self._valid(data_loader, tag)

    def test(self, test_data, batch_size=1000):
        self.valid(test_data, batch_size, tag="test")

    def valid_chess(self, valid_data, batch_size=1000, tag="valid"):
        dataset = BridgeDataset_ulb_chess(*valid_data)
        data_loader = DataLoader(
            dataset,
            batch_size=batch_size,
            collate_fn=lambda data_list: [list(data) for data in zip(*data_list)]
        )
        self._valid_chess(data_loader, tag)

    def test_chess(self, test_data, batch_size=1000):
        self.valid_chess(test_data, batch_size, tag="test")
  
# a=[[0.1,0.05,0.05,0.12,0.08,0.03,0.06,0.11,0.33,0.07],[0.1,0.05,0.05,0.12,0.08,0.03,0.06,0.11,0.33,0.07],[0.1,0.05,0.05,0.12,0.08,0.03,0.06,0.11,0.33,0.07]]
# start=time.time()
# res=sort_mask_psp(a)
# print('slow:')
# cnt=0
# for r in res:
#     print(cnt,r)
#     cnt+=1
# end=time.time()
# print(end-start)

# start=time.time()
# res=sort_mask_psp_fast(a)
# print('fast:')
# cnt=0
# for r in res:
#     print(cnt,r)
#     cnt+=1
# end=time.time()
# print(end-start)
