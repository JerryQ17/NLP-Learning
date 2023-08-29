import os
import csv
from torch import Tensor
from random import sample
from numpy import ndarray
from abc import ABC, abstractmethod
from scipy.sparse import csr_matrix
from torch.utils.data import Dataset
from typing import TypeVar, Protocol, runtime_checkable
from collections.abc import Sized, Callable, Generator, Sequence

from src.utils import typecheck
from src.utils.tensor import pad_tensor_with_tensor


class _SplittableDataset(Dataset, Sized, ABC):
    __T = TypeVar('__T', bound='_SplittableDataset')

    @runtime_checkable
    class __Gettable(Protocol):
        def __getitem__(self, item) -> tuple:
            pass

    def __init__(self, is_root: bool = True, indexs: tuple[int] | None = None, parent: __T = None):
        self._is_root = is_root
        self._indexs: tuple[int] | None = indexs
        self._parent: _SplittableDataset | None = parent

    def _mapping_parent_data(self, data: __Gettable, index: int):
        return data[index if self._is_root else self._indexs[index]]

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
        typecheck.TypeCheck(float)(ratio, extra_checks=[(lambda x: 0 < x < 1, ValueError('ratio ∈ (0, 1)'))])
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
        super().__init__()
        self.__items: tuple[tuple[str, float]] | None = None

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
        if os.path.splitext(typecheck.check_file(dataset_pathname))[1] != '.csv':
            raise ValueError('数据集必须是csv文件')
        self.__dataset_pathname = dataset_pathname
        self._read()

    @property
    def items(self):
        """所有项"""
        return self.__items

    def _read(self) -> tuple[tuple[str, float]]:
        """读取数据"""
        with open(self.__dataset_pathname, encoding='utf-8-sig', errors='ignore') as file:
            csv_reader = csv.reader(file)
            next(csv_reader)  # 跳过标题行
            self.__items = tuple((row[0], float(row[1] == "positive")) for row in csv_reader)
        return self.__items

    def get_subset(self, *index: int) -> 'IMDBDataset':
        """获取数据集中的某一子集"""
        subset = self._create_and_init_subset_with_data(*index, _IMDBDataset__items=(tuple, self.__items))
        subset.__dataset_pathname = self.__dataset_pathname
        return subset


class TfIdfDataset(_SplittableDataset):
    """TF-IDF数据集"""

    def __init__(self, values: csr_matrix, labels: Tensor):
        """
        :param values: TF-IDF值
        :param labels: 标签
        """
        super().__init__()
        typecheck.TypeCheck(csr_matrix)(values,
                                        extra_checks=[(lambda x: x.ndim == 2, ValueError('values的维度必须为2'))])
        typecheck.TypeCheck(Tensor)(labels, extra_checks=[
            (lambda x: x.ndim == 1, ValueError('labels的维度必须为1')),
            (lambda x: len(x) == values.shape[0], ValueError('values的列数和labels的个数必须相等'))
        ])
        self.__values = values
        self.__labels = labels

    def __len__(self):
        return len(self.__labels)

    def __getitem__(self, item) -> tuple[ndarray, float]:
        values = self.__values[item].toarray()[0]
        values.setflags(write=False)
        return values, self.__labels[item]

    @property
    def values(self) -> csr_matrix:
        """TF-IDF值"""
        return self.__values

    @property
    def labels(self) -> Tensor:
        """标签"""
        return self.__labels

    def get_subset(self, *index: int) -> 'TfIdfDataset':
        """获取数据集中的某一子集"""
        subset = self._create_and_init_subset_with_data(
            *index,
            _TfIdfDataset__values=(lambda x: x, self.__values),
            _TfIdfDataset__labels=(lambda x: Tensor(tuple(x)), self.__labels)
        )
        return subset


class Word2VecDataset(_SplittableDataset):
    """
    Word2Vec数据集

    Args:
        values: 词向量，列表，长度为所有句子的数量，元素为二维张量，[句子长度, 词向量维度]
        labels: 标签，一维张量，[句子数]
    """

    def __init__(self, values: Sequence[Tensor], labels: Tensor, pad_tensor: Tensor):
        super().__init__()
        typecheck.TypeCheck(Tensor)(*values, extra_checks=[(lambda x: x.ndim == 2, ValueError('元素必须为二维张量'))])
        typecheck.TypeCheck(Tensor)(labels, extra_checks=[
            (lambda x: x.ndim == 1, ValueError('labels必须为一维张量')),
            (lambda x: len(x) == len(values), ValueError('values和labels的长度必须相等'))
        ])
        self.__values = values
        self.__labels = labels
        self.__max_len = max(len(value) for value in values)
        self.__pad_tensor = pad_tensor

    def __len__(self):
        return self.__labels.shape[0]

    def __getitem__(self, item) -> tuple[Tensor, Tensor]:
        value = self.__values[item]
        return pad_tensor_with_tensor(value, self.__pad_tensor, self.__max_len - len(value)), self.__labels[item]

    @property
    def values(self) -> Sequence[Tensor]:
        """词向量"""
        return self.__values

    @property
    def labels(self) -> Tensor:
        """标签"""
        return self.__labels

    def get_subset(self, *index: int) -> 'Word2VecDataset':
        """获取数据集中的某一子集"""
        # noinspection PyTypeChecker
        subset = self._create_and_init_subset_with_data(
            *index, _Word2VecDataset__values=(tuple, self.__values),
            _Word2VecDataset__labels=(lambda x: Tensor(tuple(x)), self.__labels)
        )
        subset.__max_len = max(len(value) for value in subset.__values)
        subset.__pad_tensor = self.__pad_tensor
        return subset
