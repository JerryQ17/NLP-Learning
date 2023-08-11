import os
import csv
from numpy import ndarray
from collections.abc import Sequence
from scipy.sparse import csr_matrix
from torch.utils.data import Dataset

from src import tools


class IMDBDataset(Dataset):
    """IMDB数据集"""

    def __init__(self, dataset_pathname: str):
        """
        :param dataset_pathname: 数据集路径
        """
        self.__items: ndarray | None = None

        self.__dataset_pathname: str | None = None
        self.__dataset_title: str | None = None
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
        title, ext = os.path.splitext(tools.check_file(dataset_pathname))
        if ext != '.csv':
            raise ValueError('数据集必须是csv文件')
        self.__dataset_pathname = dataset_pathname
        self.__dataset_title = title
        self._read()

    @property
    def dataset_title(self):
        """数据集标题"""
        return self.__dataset_title

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


class TfIdfDataset(Dataset):
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
        return self.__values.getrow(item).toarray(), self.__labels[item]

    @property
    def values(self):
        """TF-IDF值"""
        return self.__values

    @property
    def labels(self):
        """标签"""
        return self.__labels


class Word2VecDataset(Dataset):
    """Word2Vec数据集"""

    def __init__(self, values: Sequence[ndarray], labels: ndarray):
        """
        :param values: 词向量
        :param labels: 标签
        """
        tools.TypeCheck(Sequence)(values)
        tools.check_ndarray(*values, extra_checks=[
            (lambda x: x.ndim == 1, ValueError('values的元素的维度必须为1')),
            (lambda x: not x.setflags(write=False), NotImplementedError('ndarray应该有setflags方法，请检查numpy版本'))
        ])
        tools.check_ndarray(labels, extra_checks=[
            (lambda x: x.dtype == bool, ValueError('labels的数据类型必须为bool')),
            (lambda x: x.ndim == 1, ValueError('labels的维度必须为1')),
            (lambda x: x.shape[0] == len(values), ValueError('values的长度和labels的个数必须相等'))
        ])
        self.__values = values
        self.__labels = labels
        self.__labels.setflags(write=False)

    def __len__(self):
        return self.__labels.shape[0]

    def __getitem__(self, item) -> tuple[ndarray, bool]:
        return self.__values[item], self.__labels[item]

    @property
    def values(self):
        """词向量"""
        return self.__values

    @property
    def labels(self):
        """标签"""
        return self.__labels
