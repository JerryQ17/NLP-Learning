import os
import csv
import numpy as np
from src import Review
from typing import Generator, Any
from scipy.sparse import csr_matrix
from torch.utils.data import Dataset


class IMDBDataset(Dataset):
    """IMDB数据集"""

    def __init__(self, dataset_pathname: str, save_memory: bool = False, get_item_by_tuple: bool = False):
        """
        :param dataset_pathname: 数据集路径
        :param save_memory: 是否节省内存，
        """
        self.__iterator: Generator[Review, Any, None] | None = None
        self.__save_memory: bool = save_memory is True
        self.__get_item_by_tuple: bool = get_item_by_tuple is True

        if self.__save_memory:
            self.__item: Review | None = None
        else:
            self.__index: int = 0
            self.__items: np.ndarray | None = None

        self.__dataset_pathname: str | None = None
        self.__dataset_title: str | None = None
        self.dataset_pathname = dataset_pathname

    def __len__(self):
        """获取数据集长度"""
        if self.__save_memory:
            with open(self.__dataset_pathname, encoding='utf-8-sig', errors='ignore') as file:
                csv_reader = csv.reader(file)
                next(csv_reader)  # 跳过标题行
                length = 0
                for _ in csv_reader:
                    length += 1
            return length
        if self.__items is None:
            return 0
        return len(self.__items)

    def __getitem__(self, item):
        """获取数据集中的某一项"""
        assert isinstance(item, int), 'item必须是一个整数'
        assert item >= 0, 'item必须大于等于0'
        if self.__save_memory:
            with open(self.__dataset_pathname, encoding='utf-8-sig', errors='ignore') as file:
                csv_reader = csv.reader(file)
                for _ in range(item + 2):
                    i = next(csv_reader)
                if self.__get_item_by_tuple:
                    return i[0], i[1] == 'positive'
                else:
                    return Review(review=i[0], sentiment=i[1] == 'positive')
        else:
            if self.__get_item_by_tuple:
                return self.__items[item].review, self.__items[item].sentiment
            else:
                return self.__items[item]

    def __iter__(self):
        """获取数据集迭代器"""
        if self.__save_memory:
            self._readline().close()
            self.__iterator = self._readline()
        else:
            self.__iterator = iter(self.__items)
        return self

    def __next__(self) -> Review:
        """获取数据集中的下一项"""
        return next(self.__iterator)

    @property
    def save_memory(self):
        """是否节省内存"""
        return self.__save_memory

    @property
    def get_item_by_tuple(self):
        """是否通过元组获取项"""
        return self.__get_item_by_tuple

    @get_item_by_tuple.setter
    def get_item_by_tuple(self, get_item_by_tuple: bool):
        """设置是否通过元组获取项"""
        self.__get_item_by_tuple = get_item_by_tuple is True

    @property
    def dataset_pathname(self):
        """数据集路径"""
        return self.__dataset_pathname

    @dataset_pathname.setter
    def dataset_pathname(self, dataset_pathname: str):
        """数据集路径必须存在，且必须是csv文件"""
        if not os.path.exists(dataset_pathname):
            raise FileNotFoundError(dataset_pathname)
        title = os.path.split(dataset_pathname)[1]
        if not title.endswith('.csv'):
            raise ValueError('数据集必须是csv文件')
        self.__dataset_pathname = dataset_pathname
        self.__dataset_title = '.'.join(title.split('.')[:-1])
        if self.__save_memory:
            self.__item = None
        else:
            self._read()

    @property
    def dataset_title(self):
        """数据集标题"""
        return self.__dataset_title

    @property
    def item(self):
        """当前项"""
        return self.__item

    @property
    def items(self):
        """所有项"""
        return self.__items

    def _readline(self):
        """读取一项数据"""
        with open(self.__dataset_pathname, encoding='utf-8-sig', errors='ignore') as file:
            csv_reader = csv.reader(file)
            next(csv_reader)  # 跳过标题行
            for row in csv_reader:
                self.__item = Review(review=row[0], sentiment=row[1] == 'positive')
                yield self.__item

    def _read(self):
        """读取数据"""
        if self.__save_memory:
            raise RuntimeError('save_memory=True时，不能调用_read()方法')
        self.__items = np.array(list(self._readline()), dtype=Review)
        self.__items.setflags(write=False)
        return self.__items


class TfIdfDataset(Dataset):
    """TF-IDF数据集"""

    def __init__(self, values: csr_matrix, labels: np.ndarray):
        """
        :param values: TF-IDF值
        :param labels: 标签
        """
        if not isinstance(values, csr_matrix):
            raise TypeError('values必须是一个csr_matrix对象')
        if not isinstance(labels, np.ndarray):
            raise TypeError('labels必须是一个ndarray对象')
        if labels.ndim != 1:
            raise ValueError('labels的维度必须为1')
        if len(values.shape) != 2:
            raise ValueError('values的维度必须为2')
        if values.shape[0] != labels.shape[0]:
            raise ValueError('values的列数和labels的个数必须相等')
        self.__values: csr_matrix | None = values
        self.__labels: np.ndarray | None = labels
        self.__labels.setflags(write=False)

    def __len__(self):
        return self.__labels.shape[0]

    def __getitem__(self, item) -> tuple[csr_matrix, bool]:
        return self.__values.getrow(item), self.__labels[item]

    @property
    def values(self):
        """TF-IDF值"""
        return self.__values

    @property
    def labels(self):
        """标签"""
        return self.__labels
