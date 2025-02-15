import torch.nn as nn
import torch
import time
from abl.reasoning import ReasonerBase,UnsupKBBase, KBBase,val_KB,ValReasoner,UnsupReasonerBase,WeaklyUnsupervisedReasoner

from abl.learning import BasicNN, ABLModel, ValNN,ValModel,WeaklySupervisedNN, WeaklySupervisedABLModel
from abl.bridge import SimpleBridge, ValBridge, UnsupBridge,WeaklyUnsupervisedBridge
from abl.evaluation import SymbolMetric, ABLMetric, ValMetric,ValChessMetric
from abl.utils import ABLLogger
from collections import defaultdict
from models.nn import LeNet5, ResNet50
import argparse
import wandb
from datasets.addition_val import get_mnist_sort, digits_to_number, get_cifar_sort, get_kmnist_sort, get_svhn_sort
import random, os
import numpy as np
from torch.utils.data import Dataset
import torch
import time

# def split_list(lst):
#     middle = len(lst) // 2
#     list1 = lst[:middle]
#     list2 = lst[middle:]
#     return list1, list2


class sort_KB(UnsupKBBase):
    def __init__(
        self,
        pseudo_label_list=list(range(2)),
        nums=1,
        max_times=10000,
        num_digits=1,
        num_classes=2,
        ind=False,
        prebuild_GKB=False,
        GKB_len_list=None
        # prebuild_GKB=False,
        # GKB_len_list=[1 * 2],
        # max_err=0,
        # use_cache=True,
    ):
        super().__init__(
            pseudo_label_list,
            ind=ind,
            prebuild_GKB=prebuild_GKB,
            GKB_len_list=GKB_len_list,
            num_digits=num_digits
        )
        self.num_classes=num_classes
        self.max_times=num_classes**(num_digits)
        self.pseudo_label_list=list(range(num_classes))
        self.num_digits=num_digits
        self.require_more_revision=num_digits
        self.count=0
        self.prebuild_GKB=prebuild_GKB
        self.prebuild_kb()

    def attack(self, type, x1, y1, x2, y2):
        if type == 0:
            return self.bishop_attack(x1, y1, x2, y2)
        elif type == 1:
            return self.king_attack(x1, y1, x2, y2)
        elif type == 2:
            return self.knight_attack(x1, y1, x2, y2)
        elif type == 3:
            return self.pawn_attack(x1, y1, x2, y2)
        elif type == 4:
            return self.queen_attack(x1, y1, x2, y2)
        elif type == 5:
            return self.rook_attack(x1, y1, x2, y2)
        return False
    
    def king_attack(self, x1, y1, x2, y2):
        # King moves one step in any direction
        return abs(x1 - x2) <= 1 and abs(y1 - y2) <= 1
    
    def queen_attack(self, x1, y1, x2, y2):
        # Queen moves straight or diagonal
        return self.straight_attack(x1, y1, x2, y2) or self.diagonal_attack(x1, y1, x2, y2)
    
    def rook_attack(self, x1, y1, x2, y2):
        # Rook moves straight
        return self.straight_attack(x1, y1, x2, y2)
    
    def bishop_attack(self, x1, y1, x2, y2):
        # Bishop moves diagonally
        return self.diagonal_attack(x1, y1, x2, y2)
    
    def knight_attack(self, x1, y1, x2, y2):
        # Knight moves in an "L" shape
        return (abs(x1 - x2) == 2 and abs(y1 - y2) == 1) or (abs(x1 - x2) == 1 and abs(y1 - y2) == 2)
    
    def pawn_attack(self, x1, y1, x2, y2):
        # Pawn attacks diagonally (assuming it's a white pawn)
        return abs(x1 - x2) == 1 and y2 - y1 == 1
    
    def straight_attack(self, x1, y1, x2, y2):
        # Moves straight: either same row or same column
        return x1 == x2 or y1 == y2
    
    def diagonal_attack(self, x1, y1, x2, y2):
        # Diagonal move: difference between x and y is the same
        return abs(x1 - x2) == abs(y1 - y2)

    # Check if a position is valid
    # def valid_coordinate(x, y):
    #     return 0 <= x < 8 and 0 <= y < 8

    # Check if a piece at (x1, y1) attacks another piece at (x2, y2)
    # def is_attack(piece, x1, y1, x2, y2):
    #     return piece.attack(x1, y1, x2, y2)
    def logic_forward(self, type, pos):
        # type, pos =nums
        self.count+=1
        l=len(type)
        for i in range(l):
            for j in range(i+1,l):
                if self.attack(type[i],pos[i][0],pos[i][1],pos[j][0],pos[j][1]):
                    return True
                # if self.attack(type[j],pos[j][0],pos[j][1],pos[i][0],pos[i][1]):
                #     return True
        return False

def main(args):
    logger = ABLLogger.get_instance("abl")
    wandb.init(
        project="ws_abl", group=f"addition {args.dataset}-{args.digit_size} naive abl"
    )
    seed_everything(args.seed)
    kb = sort_KB(num_classes=args.num_classes,num_digits=args.digit_size)
    abducer = WeaklyUnsupervisedReasoner(kb, dist_func=args.dist_func)
    cls_map = defaultdict(lambda:LeNet5(num_classes=len(kb.pseudo_label_list)))
    cls_map.update({"MNIST": LeNet5(num_classes=len(kb.pseudo_label_list))})
    cls_map.update({"KMNIST": LeNet5(num_classes=len(kb.pseudo_label_list))})
    cls_map.update({"CIFAR": ResNet50(num_classes=len(kb.pseudo_label_list))})
    cls_map.update({"SVHN": ResNet50(num_classes=len(kb.pseudo_label_list))})
    cls = cls_map[args.dataset]
    device = torch.device("cuda:"+str(args.device) if torch.cuda.is_available() else "cpu")
    criterion = nn.CrossEntropyLoss(reduction='none')
    optimizer = torch.optim.Adam(cls.parameters(), lr=args.lr, betas=(0.9, 0.99))
    base_model = WeaklySupervisedNN(
        cls,
        criterion,
        optimizer,
        device,
        save_interval=1,
        save_dir=logger.save_dir,
        batch_size=10,
        num_epochs=1,
        T=args.T
    )
    model = WeaklySupervisedABLModel(base_model,topK=args.top_k,num_classes=args.num_classes)
    metric = [
        # SymbolMetric(prefix=f"{args.dataset}_add"),
        # ABLMetric(prefix=f"{args.dataset}_add"),
        ValChessMetric(prefix=f"{args.dataset}_add")
    ]
    dta_map = defaultdict(None)
    dta_map.update({"MNIST": get_mnist_sort})
    dta_map.update({"KMNIST": get_kmnist_sort})
    dta_map.update({"CIFAR": get_cifar_sort})
    dta_map.update({"SVHN": get_svhn_sort})
    get_data = dta_map[args.dataset]
    train_data,prior = get_data(train=True, get_pseudo_label=True, n=args.digit_size,num_classes=len(kb.pseudo_label_list),sequence_num=args.train_sequence_num,seed=args.seed)
    print('prior:',prior)
    test_data,prior = get_data(train=False, get_pseudo_label=True, n=args.digit_size,num_classes=len(kb.pseudo_label_list),sequence_num=args.test_sequence_num,seed=args.seed)
    # train_data = get_data(train=True, get_pseudo_label=True, n=args.digit_size,num_classes=len(kb.pseudo_label_list),sequence_num=args.train_sequence_num,seed=args.seed)
    # test_data = get_data(train=False, get_pseudo_label=True, n=args.digit_size,num_classes=len(kb.pseudo_label_list),sequence_num=args.test_sequence_num,seed=args.seed)
    bridge = WeaklyUnsupervisedBridge(model, abducer, metric)
    start_time=time.time()
    bridge.train_chess(train_data, epochs=args.epoches, batch_size=10)
    end_time=time.time()
    used_time=end_time-start_time
    print('used_time:',used_time)
    print('count:',kb.count)
    bridge.test_chess(test_data)


def get_args():
    parser = argparse.ArgumentParser(prog="Addition Experiment, Naive ABL")
    parser.add_argument("--dataset", type=str, default="MNIST")
    parser.add_argument("--epoches", type=int, default=10)
    parser.add_argument("--digit_size", type=int, default=2)
    parser.add_argument("--dist_func", type=str, choices=['hamming', 'confidence'], default='hamming')
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num_classes", type=int, default=6)
    parser.add_argument("--train_sequence_num", type=int, default=10000)
    parser.add_argument("--test_sequence_num", type=int, default=1000)
    parser.add_argument("--device", type=int, default=0)
    # parser.add_argument("--shuffle_times", type=int, default=1)
    parser.add_argument("--T", type=float, default=1)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--use_prob", type=bool, default=False)
    parser.add_argument("--use_weight", type=bool, default=False)
    parser.add_argument("--top_k", type=int, default=20)
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
