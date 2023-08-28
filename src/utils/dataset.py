import os
import csv
from random import sample
from torch import stack, Tensor
from numpy import array, ndarray
from abc import ABC, abstractmethod
from scipy.sparse import csr_matrix
from torch.utils.data import Dataset
from collections.abc import Sized, Callable, Generator
from typing import TypeVar, Protocol, runtime_checkable

from src.utils import tools


class _SplittableDataset(Dataset, Sized, ABC):
    __T = TypeVar('__T', bound='_SplittableDataset')

    @runtime_checkable
    class __Gettable(Protocol):
        def __getitem__(self, item) -> tuple:
            pass

    def _create_and_init_subset_with_data(self: __T,
                                          *index: int,
                                          **data: tuple[Callable[[Generator], __Gettable], __Gettable]) -> __T:
        """创建一个子集，其数据与当前数据集"""
        ds = object.__new__(self.__class__)
        for key, value in data.items():
            setattr(ds, key, value[0](value[1][i] for i in index))
        return ds

    @abstractmethod
    def get_subset(self: __T, *index: int) -> __T:
        """获取数据集中的某一子集"""

    def split(self: __T, ratio: float, shuffle: bool = False) -> tuple[__T, __T]:
        """将数据集分割为两个子集"""
        tools.TypeCheck(float)(ratio, extra_checks=[(lambda x: 0 < x < 1, ValueError('ratio ∈ (0, 1)'))])
        length = len(self)
        split_index = int(length * ratio)
        if shuffle:
            index = sample(range(length), length)
        else:
            index = range(length)
        return self.get_subset(*index[:split_index]), self.get_subset(*index[split_index:])


class IMDBDataset(_SplittableDataset):
    """IMDB数据集"""

    def __init__(self, dataset_pathname: str):
        """
        :param dataset_pathname: 数据集路径
        """
        self.__items: tuple[tuple[str, bool]] | None = None

        self.__dataset_pathname: str | None = None
        self.dataset_pathname = dataset_pathname

    def __len__(self):
        """获取数据集长度"""
        return len(self.__items)

    def __getitem__(self, item):
        """获取数据集中的某一项"""
        return self.__items[item]

    @property
    def dataset_pathname(self):
        """数据集路径"""
        return self.__dataset_pathname

    @dataset_pathname.setter
    def dataset_pathname(self, dataset_pathname: str):
        """数据集路径必须存在，且必须是csv文件"""
        if os.path.splitext(tools.check_file(dataset_pathname))[1] != '.csv':
            raise ValueError('数据集必须是csv文件')
        self.__dataset_pathname = dataset_pathname
        self._read()

    @property
    def items(self):
        """所有项"""
        return self.__items

    def _read(self):
        """读取数据"""
        with open(self.__dataset_pathname, encoding='utf-8-sig', errors='ignore') as file:
            csv_reader = csv.reader(file)
            next(csv_reader)  # 跳过标题行
            self.__items = [(row[0], row[1] == "positive") for row in csv_reader]
        return self.__items

    def get_subset(self, *index: int) -> 'IMDBDataset':
        """获取数据集中的某一子集"""
        subset = self._create_and_init_subset_with_data(*index, _IMDBDataset__items=(tuple, self.__items))
        subset.__dataset_pathname = self.__dataset_pathname
        return subset


class TfIdfDataset(_SplittableDataset):
    """TF-IDF数据集"""

    def __init__(self, values: csr_matrix, labels: ndarray):
        """
        :param values: TF-IDF值
        :param labels: 标签
        """
        tools.TypeCheck(csr_matrix)(values, extra_checks=[
            (lambda x: len(x.shape) == 2, ValueError('values的维度必须为2'))
        ])
        tools.check_ndarray(labels, extra_checks=[
            (lambda x: x.ndim == 1, ValueError('labels的维度必须为1')),
            (lambda x: x.dtype == bool, ValueError('labels的数据类型必须为bool')),
            (lambda x: x.shape[0] == values.shape[0], ValueError('values的列数和labels的个数必须相等'))
        ])
        self.__values = values
        self.__labels = labels
        self.__labels.setflags(write=False)

    def __len__(self):
        return self.__labels.shape[0]

    def __getitem__(self, item) -> tuple[ndarray, bool]:
        values = self.__values[item].toarray()[0]
        values.setflags(write=False)
        return values, self.__labels[item]

    @property
    def values(self):
        """TF-IDF值"""
        return self.__values

    @property
    def labels(self):
        """标签"""
        return self.__labels

    def get_subset(self, *index: int) -> 'TfIdfDataset':
        """获取数据集中的某一子集"""
        subset = self._create_and_init_subset_with_data(
            *index,
            _TfIdfDataset__values=(lambda x: x, self.__values),
            _TfIdfDataset__labels=(lambda x: array(tuple(x)), self.__labels)
        )
        subset.labels.setflags(write=False)
        return subset


class Word2VecDataset(_SplittableDataset):
    """Word2Vec数据集"""

    def __init__(self, values: Tensor, labels: ndarray):
        """
        :param values: 词向量
        :param labels: 标签
        """
        tools.TypeCheck(Tensor)(values)
        tools.check_ndarray(labels, extra_checks=[
            (lambda x: x.dtype == bool, ValueError('labels的数据类型必须为bool')),
            (lambda x: x.ndim == 1, ValueError('labels的维度必须为1')),
            # (lambda x: x.shape[0] == len(values), ValueError('values的长度和labels的个数必须相等'))
        ])
        self.__values = values
        self.__labels = labels
        self.__labels.setflags(write=False)

    def __len__(self):
        return self.__labels.shape[0]

    def __getitem__(self, item) -> tuple[ndarray, bool]:
        return self.__values[item], self.__labels[item]

    @property
    def values(self) -> Tensor:
        """词向量"""
        return self.__values

    @property
    def labels(self) -> ndarray:
        """标签"""
        return self.__labels

    def get_subset(self, *index: int) -> 'Word2VecDataset':
        """获取数据集中的某一子集"""
        subset = self._create_and_init_subset_with_data(
            *index, _Word2VecDataset__values=(lambda x: stack(tuple(x)), self.__values),
            _Word2VecDataset__labels=(lambda x: array(tuple(x)), self.__labels)
        )
        subset.labels.setflags(write=False)
        return subset
