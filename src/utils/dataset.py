import os
import csv
from torch import Tensor
from random import sample
from numpy import array, ndarray
from abc import ABC, abstractmethod
from scipy.sparse import csr_matrix
from torch.utils.data import Dataset
from typing import TypeVar, Protocol, runtime_checkable
from collections.abc import Sized, Callable, Generator, Sequence

from src.utils import typecheck


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

    def _mapping_index(self, index: int):
        """将索引映射到根集"""
        return index if self._is_root else self._parent._mapping_index(self._indexs[index])

    def _mapping_data(self, data: str):
        """将数据映射到根集"""
        return getattr(self, data) if self._is_root else self._parent._mapping_data(data)

    def _get(self, data: str, index: int):
        return self._mapping_data(data)[self._mapping_index(index)]

    def _create_and_init_subset_with_data(self: __T,
                                          *index: int,
                                          **data: tuple[Callable[[Generator], __Gettable], __Gettable]) -> __T:
        """创建一个子集，其数据与当前数据集"""
        ds = object.__new__(self.__class__)
        _SplittableDataset.__init__(ds, is_root=False, indexs=index, parent=self)
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
        self._items: tuple[tuple[str, float]] | None = None

        self.__dataset_pathname: str | None = None
        self.dataset_pathname = dataset_pathname

    def __len__(self):
        """获取数据集长度"""
        return len(self._items) if self._is_root else len(self._indexs)

    def __getitem__(self, item) -> tuple[str, float]:
        """获取数据集中的某一项"""
        return self._get('_items', item)

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
        self._read()  # 读取数据无异常再改变属性
        super().__init__()

    @property
    def items(self):
        """所有项"""
        return self._items if self._is_root else tuple(self._mapping_data('_items')[i] for i in self._indexs)

    def _read(self) -> tuple[tuple[str, float]]:
        """读取数据"""
        with open(self.__dataset_pathname, encoding='utf-8-sig', errors='ignore') as file:
            csv_reader = csv.reader(file)
            next(csv_reader)  # 跳过标题行
            self._items = tuple((row[0], float(row[1] == "positive")) for row in csv_reader)
        return self._items

    def get_subset(self, *index: int) -> 'IMDBDataset':
        """获取数据集中的某一子集"""
        subset = self._create_and_init_subset_with_data(*index)
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
        typecheck.TypeCheck(csr_matrix)(values, extra_checks=[
            (lambda x: x.ndim == 2, ValueError('values的维度必须为2'))
        ])
        typecheck.TypeCheck(Tensor)(labels, extra_checks=[
            (lambda x: x.ndim == 1, ValueError('labels的维度必须为1')),
            (lambda x: len(x) == values.shape[0], ValueError('values的列数和labels的个数必须相等'))
        ])
        self._values = values
        self._labels = labels

    def __len__(self):
        return len(self._labels) if self._is_root else len(self._indexs)

    def __getitem__(self, item) -> tuple[ndarray, float]:
        return self._get('_values', item).toarray()[0], self._get('_labels', item)

    @property
    def values(self) -> csr_matrix:
        """TF-IDF值"""
        return self._values if self._is_root else self._mapping_data('_values')[array(self._indexs)]

    @property
    def labels(self) -> Tensor:
        """标签"""
        return self._labels if self._is_root else self._mapping_data('_labels')[Tensor(self._indexs)]

    def get_subset(self, *index: int) -> 'TfIdfDataset':
        """获取数据集中的某一子集"""
        subset = self._create_and_init_subset_with_data(*index)
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
        typecheck.TypeCheck(Sequence)(values)
        typecheck.TypeCheck(Tensor)(labels, extra_checks=[
            (lambda x: x.ndim == 1, ValueError('labels必须为一维张量')),
            (lambda x: len(x) == len(values), ValueError('values和labels的长度必须相等'))
        ])
        self._values = values
        self._labels = labels
        self.__max_len = max(len(value) for value in values)
        self.__pad_tensor = pad_tensor

    def __len__(self):
        return len(self._labels) if self._is_root else len(self._indexs)

    def __getitem__(self, item) -> tuple[Tensor, Tensor]:
        return self._get('_values', item), self._get('_labels', item)

    @property
    def values(self) -> Sequence[Tensor]:
        """词向量"""
        return self._values if self._is_root else tuple(self._mapping_data('_values')[i] for i in self._indexs)

    @property
    def labels(self) -> Tensor:
        """标签"""
        return self._labels if self._is_root else self._mapping_data('_labels')[Tensor(self._indexs)]

    def get_subset(self, *index: int) -> 'Word2VecDataset':
        """获取数据集中的某一子集"""
        subset = self._create_and_init_subset_with_data(*index)
        subset.__max_len = self.__max_len
        subset.__pad_tensor = self.__pad_tensor
        return subset
