import torch.nn as nn
import torch

from abl.reasoning import ReasonerBase, KBBase,val_KB,ValReasoner

from abl.learning import BasicNN, ABLModel, ValNN,ValModel
from abl.bridge import SimpleBridge, ValBridge
from abl.evaluation import SymbolMetric, ABLMetric, ValMetric
from abl.utils import ABLLogger
from collections import defaultdict
from models.nn import LeNet5, ResNet50
import argparse
import wandb
from datasets.addition_val import get_mnist_add, digits_to_number, get_cifar_add, get_kmnist_add, get_svhn_add


# def split_list(lst):
#     middle = len(lst) // 2
#     list1 = lst[:middle]
#     list2 = lst[middle:]
#     return list1, list2


class add_KB(val_KB):
    def __init__(
        self,
        pseudo_label_list=list(range(2)),
        nums=1,
        max_times=10000,
        num_digits=1,
        num_classes=2
        # prebuild_GKB=False,
        # GKB_len_list=[1 * 2],
        # max_err=0,
        # use_cache=True,
    ):
        super().__init__(
            pseudo_label_list,nums,max_times
        )
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
        # return (digits_to_number(nums1,num_classes=self.num_classes) + digits_to_number(nums2,num_classes=self.num_classes)==digits_to_number(nums3,num_classes=self.num_classes)) #& ((nums3[0] != 0) | (len(nums3)==1) )


def main(args):
    logger = ABLLogger.get_instance("abl")
    wandb.init(
        project="ws_abl", group=f"addition {args.dataset}-{args.digit_size} naive abl"
    )
    seed_everything(args.seed)
    kb = add_KB(num_classes=args.num_classes,num_digits=args.digit_size,max_times=int(pow(args.num_classes,args.digit_size*3+1)),nums=args.top_k)
    abducer = ValReasoner(kb)
    cls_map = defaultdict(lambda:LeNet5(num_classes=len(kb.pseudo_label_list)))
    cls_map.update({"MNIST": LeNet5(num_classes=len(kb.pseudo_label_list))})
    cls_map.update({"KMNIST": LeNet5(num_classes=len(kb.pseudo_label_list))})
    cls_map.update({"CIFAR": ResNet50(num_classes=len(kb.pseudo_label_list))})
    cls_map.update({"SVHN": ResNet50(num_classes=len(kb.pseudo_label_list))})
    cls = cls_map[args.dataset]
    device = torch.device("cuda:"+str(args.device) if torch.cuda.is_available() else "cpu")
    criterion = nn.CrossEntropyLoss(reduction='none')
    optimizer = torch.optim.Adam(cls.parameters(), lr=args.lr, betas=(0.9, 0.99))
    # base_model = ValNN(
    base_model = BasicNN(
        cls,
        criterion,
        optimizer,
        device,
        save_interval=1,
        save_dir=logger.save_dir,
        batch_size=256,
        num_epochs=1,
        T=args.T
    )
    model = ValModel(base_model)
    metric = [
        # SymbolMetric(prefix=f"{args.dataset}_add"),
        # ABLMetric(prefix=f"{args.dataset}_add"),
        ValMetric(prefix=f"{args.dataset}_add")
    ]
    dta_map = defaultdict(None)
    dta_map.update({"MNIST": get_mnist_add})
    dta_map.update({"KMNIST": get_kmnist_add})
    dta_map.update({"CIFAR": get_cifar_add})
    dta_map.update({"SVHN": get_svhn_add})
    get_data = dta_map[args.dataset]
    train_data = get_data(train=True, get_pseudo_label=True, n=args.digit_size,num_classes=len(kb.pseudo_label_list),sequence_num=args.train_sequence_num,seed=args.seed)
    test_data = get_data(train=False, get_pseudo_label=True, n=args.digit_size,num_classes=len(kb.pseudo_label_list),sequence_num=args.test_sequence_num,seed=args.seed)
    bridge = ValBridge(model, abducer, metric)#,use_prob=args.use_prob,use_weight=args.use_weight)
    bridge.train(train_data, epochs=args.epoches, batch_size=1024, test_data=test_data)
    bridge.test(test_data)


def get_args():
    parser = argparse.ArgumentParser(prog="Addition Experiment, Naive ABL")
    parser.add_argument("--dataset", type=str, default="MNIST")
    parser.add_argument("--epoches", type=int, default=10)
    parser.add_argument("--digit_size", type=int, default=1)
    parser.add_argument("--dist_func", type=str, choices=['hamming', 'confidence'], default='hamming')
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num_classes", type=int, default=2)
    parser.add_argument("--train_sequence_num", type=int, default=30000)
    parser.add_argument("--test_sequence_num", type=int, default=1000)
    parser.add_argument("--device", type=int, default=0)
    # parser.add_argument("--shuffle_times", type=int, default=1)
    parser.add_argument("--T", type=float, default=1)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--use_prob", type=bool, default=False)
    parser.add_argument("--use_weight", type=bool, default=False)
    parser.add_argument("--top_k", type=int, default=1)
    parser.add_argument("--algo", type=str, default='VAL')
    args = parser.parse_args()
    return args


def seed_everything(seed: int = 0):
    import random, os
    import numpy as np
    import torch

    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


if __name__ == "__main__":
    args = get_args()
    main(args)
