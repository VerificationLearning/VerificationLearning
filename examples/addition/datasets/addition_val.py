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
_DATA_ROOT = Path(__file__).parent


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

def addition(n: int, dataset: str, seed=None, train: bool = True, num_classes=2,sequence_num=30000):
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
        l1, l2, res = self.data[index]
        # label = self._get_label(index)
        l1 = [x[0] for x in l1]
        l2 = [x[0] for x in l2]
        res = [x[0] for x in res]
        # print('index:',index)
        # print('label:',self._get_symbol_label(index))
        return l1, l2, res

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
        # self.shuffle_times=shuffle_times
        
        
        # rng = random.Random(seed)
        # rng.shuffle(mnist_indices)
        
        # Build list of examples (mnist indices)
        self.categorized_data=[self.split_dataset_by_category(self.dataset)[c] for c in range(num_classes)]
        self.dataset=[]
        self.labels=[]
        for c in range(num_classes):
            self.dataset.extend([(self.categorized_data[c][i],c) for i in range(len(self.categorized_data[c])) ])
            self.labels.extend([c for i in range(len(self.categorized_data[c]))])
        # if seed is not None:
        mnist_indices = self.indices()
        rng = random.Random(seed)
        # for idx in range(self.shuffle_times):
        rng.shuffle(mnist_indices)
        dataset_iter = ShuffleIterator(mnist_indices)
        # Build list of examples (mnist indices)
        self.data = []
        self.counts=np.zeros(num_classes)
        try:
            while len(self.data) < sequence_num:
                inputs=[[self.dataset[next(dataset_iter)] for _ in range(self.size)]
                        for _ in range(self.arity)]
                ground_truth=[digits_to_number((digit[1] for digit in digits), num_classes=self.num_classes) for digits in inputs]
                for digits in inputs:
                    for digit in digits:
                        self.counts[digit[1]]+=1
                # for
                expected_result = self.operator(ground_truth)
                # digits_result = number_to_digits(expected_result, num_classes=self.num_classes)
    
                digits_result = number_to_digits(expected_result, digit_size=self.size+1,num_classes=self.num_classes)
                for c in digits_result:
                        self.counts[c]+=1
                inputs.append([(random.choice(self.categorized_data[c]),c) for c in digits_result])
                self.data.append(inputs)
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

    def _get_symbol_label(self, i: int):
        # print('self.data[i]:',self.data[i])
        l1,l2,res=self.data[i]
        l1 = [x[1] for x in l1]
        l2 = [x[1] for x in l2]
        res = [x[1] for x in res]
        # mnist_indices = self.data[i]
        ground_truth=[l1,l2,res]
        # Figure out what the ground truth is, first map each parameter to the value:
        # ground_truth = [self.dataset[j][1] for i in mnist_indices for j in i]
        # print('ground_truth:',ground_truth)
        return ground_truth

    def __len__(self):
        return len(self.data)


def get_mnist_add(train=True, get_pseudo_label=False, n=1,num_classes=2,sequence_num=30000,seed=None):
    mnistDataset = addition(n, "MNIST", train=train,num_classes=num_classes,sequence_num=sequence_num,seed=seed)
    X,  Z = [], []
    for idx in range(len(mnistDataset)):
        x1,x2,x3= mnistDataset[idx]
        z1,z2,z3 = mnistDataset._get_symbol_label(idx)
        X.extend([x1 + x2 + x3]),  Z.extend([z1+z2+z3])
    # if get_pseudo_label:
    #     return (X, Z),mnistDataset.prior
    print('Z:',Z)
    return (X, Z),mnistDataset.prior


def get_kmnist_add(train=True, get_pseudo_label=False, n=1,num_classes=2,sequence_num=30000,seed=None):
    mnistDataset = addition(n, "KMNIST", train=train,num_classes=num_classes,sequence_num=sequence_num,seed=seed)
    X,  Z = [], []
    for idx in range(len(mnistDataset)):
        x1,x2,x3= mnistDataset[idx]
        z1,z2,z3 = mnistDataset._get_symbol_label(idx)
        X.extend([x1 + x2 + x3]),  Z.extend([z1+z2+z3])
    # if get_pseudo_label:
    #     return (X, Z),mnistDataset.prior
    print('Z:',Z)
    return (X, Z),mnistDataset.prior


def get_cifar_add(train=True, get_pseudo_label=False, n=1,num_classes=2,sequence_num=30000,seed=None):
    mnistDataset = addition(n, "CIFAR10", train=train,num_classes=num_classes,sequence_num=sequence_num,seed=seed)
    X,  Z = [], []
    for idx in range(len(mnistDataset)):
        x1,x2,x3= mnistDataset[idx]
        z1,z2,z3 = mnistDataset._get_symbol_label(idx)
        X.extend([x1 + x2 + x3]),  Z.extend([z1+z2+z3])
    # if get_pseudo_label:
    #     return (X, Z),mnistDataset.prior
    print('Z:',Z)
    return (X, Z),mnistDataset.prior  


def get_svhn_add(train=True, get_pseudo_label=False, n=1,num_classes=2,sequence_num=30000,seed=None):
    mnistDataset = addition(n, "SVHN", train=train,num_classes=num_classes,sequence_num=sequence_num,seed=seed)
    X,  Z = [], []
    for idx in range(len(mnistDataset)):
        x1,x2,x3= mnistDataset[idx]
        z1,z2,z3 = mnistDataset._get_symbol_label(idx)
        X.extend([x1 + x2 + x3]),  Z.extend([z1+z2+z3])
    # if get_pseudo_label:
    #     return (X, Z),mnistDataset.prior
    print('Z:',Z)
    return (X, Z),mnistDataset.prior


if __name__ == "__main__":
    mnist_add = get_mnist_add()
