from typing import Optional, Sequence, Callable
from .base_metric import BaseMetric
import numpy as np
import wandb
from itertools import chain 
import torch
from sklearn.metrics import confusion_matrix

class SudokuMetric(BaseMetric):
    def __init__(self, prefix: Optional[str] = None) -> None:
        super().__init__(prefix)
        self.y_pred = []
        self.y_gt = []
    def process(self, data_samples: Sequence[dict]) -> None:
        pred_pseudo_label = data_samples["pred_pseudo_label"]

        gt_pseudo_label = data_samples["gt_pseudo_label"]

        X=data_samples["X"]
        
        logic_forward = data_samples["logic_forward"]
        
        # mask=data_samples["mask"]
        # print('mask:',mask)
        # if mask is not None:
            # gt_pseudo_label=gt_pseudo_label[mask]
            # pred_pseudo_label=pred_pseudo_label[mask]
        if not len(pred_pseudo_label) == len(gt_pseudo_label):
            raise ValueError("lengthes of pred_pseudo_label and gt_pseudo_label should be equal")
        # print('gt_pseudo_label:',gt_pseudo_label)
        # print('pred_pseudo_label:',pred_pseudo_label)
        self.y_gt.extend(gt_pseudo_label)
        self.y_pred.extend(pred_pseudo_label)
        for pred_z, z,x in zip(pred_pseudo_label, gt_pseudo_label,X):
            correct_num = 0
            mask=(x==9)
            # print('mask:',mask)
            _pred_z,_z=torch.Tensor(pred_z)[mask],torch.Tensor(z)[mask]
            for pred_symbol, symbol in zip(pred_z, z):
                if pred_symbol == symbol:
                    correct_num += 1
            self.results.append(correct_num / len(z))
        
    
    def compute_metrics(self, results: list) -> dict:
        metrics = dict()
        metrics["character_accuracy"] = sum(results) / len(results)
        print('metrics["character_accuracy"]',metrics["character_accuracy"])
        cm = wandb.plot.confusion_matrix(preds=list(chain(*self.y_pred)), y_true=list(chain(*self.y_gt)))
        metrics['Confusing Matrix'] = cm
        # print('val_cm:',confusion_matrix(list(chain(*self.y_val)),list(chain(*self.y_gt))))
        print('pred_cm:',confusion_matrix(list(chain(*self.y_pred)),list(chain(*self.y_gt))))
        # print('metrics[Confusing Matrix]:',metrics['Confusing Matrix'])
        return metrics