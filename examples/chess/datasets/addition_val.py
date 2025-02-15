import itertools
import json
import random
from pathlib import Path
from typing import Callable, List, Iterable, Tuple, Any

import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset as TorchDataset
from collections import Counter, defaultdict
import numpy as np
from itertools import combinations
_DATA_ROOT = Path(__file__).parent
import copy


# 示例：生成所有长度为 4，最大值为 3 的单调非递减序列
n = 4
m = 3
# generate_sequences(n, m)

class ShuffleIterator:
    def __init__(self, iterable):
        self.iterable = iterable
        self.buffer = []
        self.index = 0

    def __iter__(self):
        return self

    def __next__(self):
        if self.index >= len(self.buffer):
            self.buffer = list(self.iterable)
            random.shuffle(self.buffer)
            self.index = 0

        if not self.buffer:
            self.buffer = list(self.iterable)
            random.shuffle(self.buffer)

        item = self.buffer[self.index]
        self.index += 1
        return item


def MNIST_datasets():
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
    )

    datasets = {
        "train": torchvision.datasets.MNIST(
            root=str(_DATA_ROOT), train=True, download=True, transform=transform
        ),
        "test": torchvision.datasets.MNIST(
            root=str(_DATA_ROOT), train=False, download=True, transform=transform
        ),
    }
    return datasets


def KMNIST_datasets():
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
    )

    datasets = {
        "train": torchvision.datasets.KMNIST(
            root=str(_DATA_ROOT), train=True, download=True, transform=transform
        ),
        "test": torchvision.datasets.KMNIST(
            root=str(_DATA_ROOT), train=False, download=True, transform=transform
        ),
    }
    return datasets


def CIFAR_datasets():
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
    )

    datasets = {
        "train": torchvision.datasets.cifar.CIFAR10(
            root=str(_DATA_ROOT), train=True, download=True, transform=transform
        ),
        "test": torchvision.datasets.cifar.CIFAR10(
            root=str(_DATA_ROOT), train=False, download=True, transform=transform
        ),
    }
    return datasets


def SVHN_datasets():
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
    )

    datasets = {
        "train": torchvision.datasets.SVHN(
            root=str(_DATA_ROOT), split='train', download=True, transform=transform
        ),
        "test": torchvision.datasets.SVHN(
            root=str(_DATA_ROOT), split='test', download=True, transform=transform
        ),
    }
    return datasets


def get_datasets(dataset_name: str = "MNIST"):
    if dataset_name == "MNIST":
        return MNIST_datasets()
    elif dataset_name == 'KMNIST':
        return KMNIST_datasets()
    elif dataset_name in ["CIFAR10", 'CIFAR']:
        return CIFAR_datasets()
    elif dataset_name == 'SVHN':
        return SVHN_datasets()

def digits_to_number_binary(digits: Iterable[int]) -> int:
    """
    Converts a list of binary digits (0 or 1) to its equivalent integer value.
    """
    number = 0
    for d in digits:
        number *= 2  # Shift left by 1 bit (multiply by base 2)
        number += d  # Add the current binary digit
    return number

def number_to_digits_binary(number: int, digit_size=None) -> List[int]:
    """
    Converts an integer to a list of binary digits (0 or 1).
    If digit_size is provided, pads the result with leading zeros to match the size.
    """
    if digit_size is not None:
        digits = []
        for i in range(digit_size):
            digits.append(number % 2)
            number //= 2
        return digits[::-1]  # Reverse to get the correct order

    if number == 0:
        return [0]

    digits = []
    while number > 0:
        digits.append(number % 2)
        number //= 2

    return digits[::-1]  # Reverse to get the correct order

def digits_to_number(digits,num_classes=2) -> int:
    number = 0
    for d in digits:
        number *= num_classes
        number += d
    return number

def number_to_digits(number: int, digit_size=None,num_classes=2) -> List[int]:
    digits=[]
    if digit_size is not None:
        for i in range(digit_size):
            digits.append(number%num_classes)
            number//=num_classes
        # print('digits:',digits[::-1])
        return digits[::-1]

    if number == 0:
        return [0]
    
    digits = []
    while number > 0:
        digits.append(number % num_classes)
        number //= num_classes
    
    # The digits are appended in reverse order, so reverse the list before returning
    return digits[::-1]

# def number_to_digits(number: int, digit_size=None) -> List[int]:
#     if number == 0:
#         return [0]
    
#     digits = []
#     while number > 0:
#         digits.append(number % 10)
#         number //= 10
    
#     # The digits are appended in reverse order, so reverse the list before returning
#     return digits[::-1]

def sort_data(n: int, dataset: str, seed=None, train: bool = True, num_classes=2,sequence_num=30000):
    """Returns a dataset for binary addition"""
    return DigitsOperator(
        dataset_name=dataset,
        function_name="addition" if n == 1 else "multi_addition",
        operator=sum,
        size=n,
        arity=2,
        seed=seed,
        train=train,
        num_classes=num_classes,
        sequence_num=sequence_num
    )


class DigitsOperator(TorchDataset):
    def __getitem__(self, index: int) -> Tuple[list, list, int]:
        l = self.data[index]
        inputs=[x[0] for x in l]
        # label = self._get_label(index)
        # l1 = [x[0] for x in l1]
        # l2 = [x[0] for x in l2]
        # res = [x[0] for x in res]
        # print('index:',index)
        # print('label:',self._get_symbol_label(index))
        return inputs

    def get_attacked_positions(self, type,x, y):
        if type == 0:
            return self.get_bishop_attacked_positions(x, y)
        elif type == 1:
            return self.get_king_attacked_positions(x, y)
        elif type == 2:
            return self.get_knight_attacked_positions(x, y)
        elif type == 3:
            return self.get_pawn_attacked_positions(x, y)
        elif type == 4:
            return self.get_queen_attacked_positions(x, y)
        elif type == 5:
            return self.get_rook_attacked_positions(x, y)
        return []

    def get_queen_attacked_positions(self, x, y):
        # attacked_positions = set()

        # Queen attacks horizontally and vertically (rook-like)
        attacked_positions=self.get_rook_attacked_positions(x, y)+self.get_bishop_attacked_positions(x, y)

        # Queen attacks diagonally (bishop-like)

        return attacked_positions

    def get_rook_attacked_positions(self, x, y):
        attacked_positions = []

        # Rook can attack along the row and column
        for i in range(8):
            if i != x and self.vis[i][y] is not True:
                attacked_positions.append((i, y))  # Same column
            if i != y and self.vis[x][i] is not True:
                attacked_positions.append((x, i))  # Same row

        return attacked_positions

    def get_bishop_attacked_positions(self, x, y):
        attacked_positions = []

        # Bishop attacks diagonally
        for i in range(1, 8):
            if 0 <= x + i < 8 and 0 <= y + i < 8 and self.vis[x + i][y+i] is not True:
                attacked_positions.append((x + i, y + i))
            if 0 <= x + i < 8 and 0 <= y - i < 8 and self.vis[x + i][y-i] is not True:
                attacked_positions.append((x + i, y - i))
            if 0 <= x - i < 8 and 0 <= y + i < 8 and self.vis[x - i][y+i] is not True:
                attacked_positions.append((x - i, y + i))
            if 0 <= x - i < 8 and 0 <= y - i < 8 and self.vis[x - i][y-i] is not True:
                attacked_positions.append((x - i, y - i))

        return attacked_positions

    def get_knight_attacked_positions(self, x, y):
        attacked_positions = []

        # Knight moves in "L" shape
        moves = [
            (2, 1), (2, -1), (-2, 1), (-2, -1),
            (1, 2), (1, -2), (-1, 2), (-1, -2)
        ]

        for dx, dy in moves:
            if 0 <= x + dx < 8 and 0 <= y + dy < 8 and self.vis[x + dx][y+dy] is not True:
                attacked_positions.append((x + dx, y + dy))

        return attacked_positions

    def get_king_attacked_positions(self, x, y):
        attacked_positions = []

        # King moves one square in any direction
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                if dx == 0 and dy == 0:
                    continue
                if 0 <= x + dx < 8 and 0 <= y + dy < 8 and self.vis[x + dx][y+dy] is not True:
                    attacked_positions.append((x + dx, y + dy))

        return attacked_positions

    def get_pawn_attacked_positions(self, x, y):
        attacked_positions = []

        # Pawn attacks diagonally (assumes white pawn, so moves upwards)
        if 0 <= x - 1 < 8 and 0 <= y - 1 < 8 and self.vis[x - 1][y-1] is not True:
            attacked_positions.append((x - 1, y - 1))  # Diagonal left
        if 0 <= x - 1 < 8 and 0 <= y + 1 < 8 and self.vis[x - 1][y+1] is not True:
            attacked_positions.append((x - 1, y + 1))  # Diagonal right

        return attacked_positions



    def generate_sequences(self,n):
        types=[]
        positions = []
        all_positions=[]
        for x in range(8):
            for y in range(8):
                all_positions.append((x, y))
        for _ in range(n):
            while True:
                self.vis=[[False for i in range(8)] for j in range(8)]
                rest_pos=copy.deepcopy(all_positions)
                rest_type=list(range(self.num_classes))
                cur_type=[]
                cur_pos=[]
                pre_type=None
                find=True
                for i in range(self.size):
                    type=random.choice(rest_type)
                    if i==0:
                        pos=random.choice(rest_pos)
                    else:
                        condidate_pos=self.get_attacked_positions(pre_type,pre_pos[0],pre_pos[1])
                        if len(condidate_pos) ==0:
                            find=False
                            break
                        else:
                            pos=random.choice(condidate_pos)
                    self.vis[pos[0]][pos[1]]=True
                    cur_type.append(type)
                    cur_pos.append(pos)
                    pre_type=type
                    pre_pos=pos
                    rest_pos.remove(pos)
                    # rest_type.remove(type)
                
                if find is True:
                    types.append(cur_type)
                    positions.append(cur_pos)
                    break
        return types,positions

    def balance_indices(self):
        balance_size = sorted(Counter(self.labels).items())[0][1]
        labels_dist = defaultdict(int)
        sampler_iter = ShuffleIterator(list(range(len(self.dataset))))
        balanced_indices = []
        while len(balanced_indices) < balance_size * self.num_classes:
            sample = next(sampler_iter)
            sampled_class = self.labels[sample]
            if labels_dist[sampled_class] >= balance_size:
                continue
            balanced_indices.append(sample)
            labels_dist[sampled_class] += 1
        return balanced_indices
    
    def split_dataset_by_category(self,dataset):
        """
        Splits a dataset into groups based on categories.
        
        :param dataset: A list of tuples where each tuple contains (data, category).
        :return: A dictionary where keys are categories and values are lists of data points.
        """
        categorized_data = defaultdict(list)
        for data, category in dataset:
            categorized_data[category].append(data)
        return dict(categorized_data)

    def indices(self):
        return list(range(len(self.dataset)))
    
    # def generate_sequences(self,n, m, start=0, current=[]):
    #     # if len(current) == n:
    #     #     # print(current)
    #     #     self.all_list.append(current)
    #     #     return
    #     # for num in range(start, m ):
    #     #     self.generate_sequences(n, m, num+1, current + [num])
    #     return list(combinations(range(m ), n))
    
    def __init__(
        self,
        dataset_name: str,
        function_name: str,
        operator: Callable[[List[int]], int],
        size=1,
        arity=2,
        seed=None,
        train: bool = True,
        sequence_num: int = 30000,
        num_classes=2
        # shuffle_times=1
    ):
        """Generic dataset for operator(img, img) style datasets.

        :param dataset_name: Dataset to use (train, val, test)
        :param function_name: Name of Problog function to query.
        :param operator: Operator to generate correct examples
        :param size: Size of numbers (number of digits)
        :param arity: Number of arguments for the operator
        :param seed: Seed for RNG
        """
        super(DigitsOperator, self).__init__()
        assert size >= 1
        assert arity >= 1
        self.dataset_name = dataset_name
        self.datasets = get_datasets(dataset_name)
        self.dataset = self.datasets["train" if train else "test"]
        self.function_name = function_name
        self.operator = operator
        self.size = size
        self.arity = arity
        self.seed = seed
        self.num_classes=num_classes
        self.sequence_num=sequence_num
        self.all_list=[]
        # self.shuffle_times=shuffle_times
        
        
        # rng = random.Random(seed)
        # rng.shuffle(mnist_indices)
        rng = random.Random(seed)
        self.all_list,self.all_pos= self.generate_sequences(self.sequence_num)
        # print('len(self.all_list):',len(self.all_list))
        # rng.shuffle(self.all_list)
        # sequences=[]
        # for i in range(sequence_num):
        #     sequences.append(self.all_list[i%len(self.all_list)])
        # sequences = random.sample(self.all_list, self.sequence_num)
        # Build list of examples (mnist indices)
        # rng.shuffle(sequences)
        self.categorized_data=[self.split_dataset_by_category(self.dataset)[c] for c in range(num_classes)]
        self.dataset=[]
        self.labels=[]
        for c in range(num_classes):
            self.dataset.extend([(self.categorized_data[c][i],c) for i in range(len(self.categorized_data[c])) ])
            self.labels.extend([c for i in range(len(self.categorized_data[c]))])
        # if seed is not None:
        # mnist_indices = self.indices()
        
        # for idx in range(self.shuffle_times):
        # rng.shuffle(mnist_indices)
        # dataset_iter = ShuffleIterator(mnist_indices)
        # Build list of examples (mnist indices)
        self.data = []
        self.counts=np.zeros(num_classes)
        try:
            for sequence, pos in zip(self.all_list,self.all_pos):
                inputs=[]
                for c,p in zip(sequence,pos):
                    self.counts[c]+=1
                    inputs.append((random.choice(self.categorized_data[c]),c,p))
                # inputs.append([(random.choice(self.categorized_data[c]),c) for c in digits_result])
                self.data.append(inputs)
            # while len(self.data) < sequence_num:
            #     inputs=[[self.dataset[next(dataset_iter)] for _ in range(self.size)]
            #             for _ in range(self.arity)]
            #     ground_truth=[digits_to_number((digit[1] for digit in digits), num_classes=self.num_classes) for digits in inputs]
            #     for digits in inputs:
            #         for digit in digits:
            #             self.counts[digit[1]]+=1
            #     # for
            #     expected_result = self.operator(ground_truth)
            #     # digits_result = number_to_digits(expected_result, num_classes=self.num_classes)
    
            #     digits_result = number_to_digits(expected_result, digit_size=self.size+1,num_classes=self.num_classes)
            #     for c in digits_result:
            #             self.counts[c]+=1
            #     inputs.append([(random.choice(self.categorized_data[c]),c) for c in digits_result])
            #     self.data.append(inputs)
        except StopIteration:
            pass
        self.prior=self.counts/np.sum(self.counts)
        print('self.prior:',self.prior)
        
        # dataset_iter = ShuffleIterator(mnist_indices)
        # try:
        #     while len(self.data) < sequence_num:
        #         inputs=[[self.dataset[next(dataset_iter)] for _ in range(self.size)]
        #                 for _ in range(self.arity)]
        #         ground_truth=[digits_to_number((digit[1] for digit in digits), num_classes=self.num_classes) for digits in inputs]
        #         # for
        #         expected_result = self.operator(ground_truth)
        #         digits_result = number_to_digits(expected_result, num_classes=self.num_classes)
        #         inputs.append([(random.choice(self.categorized_data[c]),c) for c in digits_result])
        #         self.data.append(inputs)
        # except StopIteration:
        #     pass

    def to_file_repr(self, i):
        """Old file represenation dump. Not a very clear format as multi-digit arguments are not separated"""
        return f"{tuple(itertools.chain(*self.data[i]))}"#\t{self._get_label(i)}"

    def to_json(self):
        """
        Convert to JSON, for easy comparisons with other systems.

        Format is [EXAMPLE, ...]
        EXAMPLE :- [ARGS, expected_result]
        ARGS :- [MULTI_DIGIT_NUMBER, ...]
        MULTI_DIGIT_NUMBER :- [mnist_img_id, ...]
        """
        data = [(self.data[i], self._get_label(i)) for i in range(len(self))]
        return json.dumps(data)

    # def _get_label(self, i: int):
    #     mnist_indices = self.data[i]
    #     # Figure out what the ground truth is, first map each parameter to the value:
    #     ground_truth = [
    #         digits_to_number(self.dataset[j][1] for j in i) for i in mnist_indices
    #     ]
    #     # Then compute the expected value:
    #     expected_result = self.operator(ground_truth)
    #     return expected_result
    
    def _get_pos(self, i: int):
        l=self.data[i]
        pos = [x[2] for x in l]
        return pos

    def _get_symbol_label(self, i: int):
        # print('self.data[i]:',self.data[i])
        l=self.data[i]
        ground_truth = [x[1] for x in l]
        # l2 = [x[1] for x in l2]
        # res = [x[1] for x in res]
        # mnist_indices = self.data[i]
        # ground_truth=[l1,l2,res]
        # Figure out what the ground truth is, first map each parameter to the value:
        # ground_truth = [self.dataset[j][1] for i in mnist_indices for j in i]
        # print('ground_truth:',ground_truth)
        return ground_truth

    def __len__(self):
        return len(self.data)


def get_mnist_sort(train=True, get_pseudo_label=False, n=1,num_classes=2,sequence_num=30000,seed=None):
    mnistDataset = sort_data(n, "MNIST", train=train,num_classes=num_classes,sequence_num=sequence_num,seed=seed)
    X,  Z,P = [], [],[]
    for idx in range(len(mnistDataset)):
        x= mnistDataset[idx]
        z = mnistDataset._get_symbol_label(idx)
        p = mnistDataset._get_pos(idx)
        X.append(x),  Z.append(z),P.append(p)
    # if get_pseudo_label:
    #     return (X, Z),mnistDataset.prior
    print('Z:',Z)
    return (X, Z,P),mnistDataset.prior


def get_kmnist_sort(train=True, get_pseudo_label=False, n=1,num_classes=2,sequence_num=30000,seed=None):
    mnistDataset = sort_data(n, "KMNIST", train=train,num_classes=num_classes,sequence_num=sequence_num,seed=seed)
    X,  Z,P = [], [],[]
    for idx in range(len(mnistDataset)):
        x= mnistDataset[idx]
        z = mnistDataset._get_symbol_label(idx)
        p = mnistDataset._get_pos(idx)
        X.append(x),  Z.append(z),P.append(p)
    # if get_pseudo_label:
    #     return (X, Z),mnistDataset.prior
    print('Z:',Z)
    return (X, Z,P),mnistDataset.prior


def get_cifar_sort(train=True, get_pseudo_label=False, n=1,num_classes=2,sequence_num=30000,seed=None):
    mnistDataset = sort_data(n, "CIFAR10", train=train,num_classes=num_classes,sequence_num=sequence_num,seed=seed)
    X,  Z,P = [], [],[]
    for idx in range(len(mnistDataset)):
        x= mnistDataset[idx]
        z = mnistDataset._get_symbol_label(idx)
        p = mnistDataset._get_pos(idx)
        # X.extend([x1 + x2 + x3]),  Z.extend([z1+z2+z3])
        X.append(x),  Z.append(z),P.append(p)
    # if get_pseudo_label:
    #     return (X, Z),mnistDataset.prior
    print('Z:',Z)
    return (X, Z,P),mnistDataset.prior  


def get_svhn_sort(train=True, get_pseudo_label=False, n=1,num_classes=2,sequence_num=30000,seed=None):
    mnistDataset = sort_data(n, "SVHN", train=train,num_classes=num_classes,sequence_num=sequence_num,seed=seed)
    X,  Z,P = [], [],[]
    for idx in range(len(mnistDataset)):
        x= mnistDataset[idx]
        z = mnistDataset._get_symbol_label(idx)
        p = mnistDataset._get_pos(idx)
        X.append(x),  Z.append(z),P.append(p)
    # if get_pseudo_label:
    #     return (X, Z),mnistDataset.prior
    print('Z:',Z)
    return (X, Z,P),mnistDataset.prior


if __name__ == "__main__":
    mnist_add = get_mnist_add()
