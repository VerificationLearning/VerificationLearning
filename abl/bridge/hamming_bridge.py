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
from abl.dataset import BridgeDataset,BridgeDataset_ulb
from abl.utils.logger import print_log
import numpy as np
import time
import heapq
from sklearn.metrics import confusion_matrix

class UnsupBridge(BaseBridge):
    def __init__(
        self,
        model: ABLModel,
        abducer: ReasonerBase,
        metric_list: List[BaseMetric],
        use_prob=False,
        use_weight=False
    ) -> None:
        super().__init__(model, abducer)
        self.metric_list = metric_list
        self.use_prob=use_prob
        self.use_weight=use_weight

    def predict(self, X) -> Tuple[List[List[Any]], ndarray]:
        pred_res = self.model.predict(X)
        pred_idx, pred_prob = pred_res["label"], pred_res["prob"]
        return pred_idx, pred_prob

    def abduce_pseudo_label(
        self,
        pred_prob: ndarray,
    ) -> List[List[Any]]:
        return self.abducer.batch_abduce_candidates_set(pred_prob)

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
    ):
        dataset = BridgeDataset_ulb(*train_data)
        data_loader = DataLoader(
            dataset,
            batch_size=batch_size,
            collate_fn=lambda data_list: [list(data) for data in zip(*data_list)],
        )
        for epoch in range(epochs):
            for seg_idx, (X, Z) in enumerate(data_loader):
                pred_idx, pred_prob = self.predict(X)
                pred_pseudo_label = self.idx_to_pseudo_label(pred_idx)
                abduced_pseudo_label = self.abduce_pseudo_label(pred_prob, pred_pseudo_label)
                abduced_label = self.pseudo_label_to_idx(abduced_pseudo_label)
                loss = self.model.train(X, abduced_label)
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
                    self.test(test_data)

    def _valid(self, data_loader, tag=""):
        for X, Z in data_loader:
            pred_idx, pred_prob = self.predict(X)
            pred_pseudo_label = self.idx_to_pseudo_label(pred_idx)
            data_samples = dict(
                pred_idx=pred_idx,
                pred_prob=pred_prob,
                pred_pseudo_label=pred_pseudo_label,
                gt_pseudo_label=Z,
                logic_forward=self.abducer.kb.logic_forward,
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