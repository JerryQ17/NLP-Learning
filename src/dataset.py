import os
import csv
from numpy import ndarray
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
        tools.TypeCheck(csr_matrix)(values)
        tools.check_ndarray(labels)
        if labels.ndim != 1:
            raise ValueError('labels的维度必须为1')
        if len(values.shape) != 2:
            raise ValueError('values的维度必须为2')
        if values.shape[0] != labels.shape[0]:
            raise ValueError('values的列数和labels的个数必须相等')
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

    def __init__(self, values: list[ndarray], labels: ndarray):
        """
        :param values: 词向量
        :param labels: 标签
        """
        tools.TypeCheck(list)(values)
        tools.check_ndarray(labels)
        if labels.ndim != 1:
            raise ValueError('labels的维度必须为1')
        if len(values) != labels.shape[0]:
            raise ValueError('values的长度和labels的个数必须相等')
        for value in values:
            tools.check_ndarray(value)
            if value.ndim != 1:
                raise ValueError('values的元素的维度必须为1')
            value.setflags(write=False)
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
