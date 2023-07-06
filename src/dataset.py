import os
import csv
import numpy as np
from sys import stderr
from models import Review
from multiprocessing import Pool
from typing import Generator, Any
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer


# [b'.............................*............*', b'optimization finished, #iter = 41137', b'nu = 0.298037',
#  b'obj = -13990.416638, rho = 0.031264', b'nSV = 20456, nBSV = 4944', b'Total nSV = 20456',
#  b'.............................*............*', b'optimization finished, #iter = 41061', b'nu = 0.296250',
#  b'obj = -13870.438031, rho = 0.027070', b'nSV = 20329, nBSV = 4937', b'Total nSV = 20329',
#  b'.............................*............*', b'optimization finished, #iter = 41187', b'nu = 0.298669',
#  b'obj = -14006.262722, rho = 0.022416', b'nSV = 20374, nBSV = 4996', b'Total nSV = 20374',
#  b'.............................*............*', b'optimization finished, #iter = 41190', b'nu = 0.297351',
#  b'obj = -13917.599077, rho = 0.033382', b'nSV = 20417, nBSV = 5020', b'Total nSV = 20417',
#  b'.............................*............*', b'optimization finished, #iter = 41374', b'nu = 0.298531',
#  b'obj = -13978.958947, rho = 0.030846', b'nSV = 20462, nBSV = 4959', b'Total nSV = 20462',
#  b'Cross Validation Accuracy = 90.112%']


class DataSet:
    """
    数据集类
    """

    def __init__(self, dataset_pathname: str, processes: int = os.cpu_count(), save_memory: bool = False):
        """
        :param dataset_pathname: 数据集路径
        :param processes: 进程数
        :param save_memory: 是否节省内存，
        """
        self.__iterator: Generator[Review, Any, None] | None = None

        self.__save_memory: bool = save_memory is True
        self.__processes: int | None = None
        self.processes = processes

        if self.__save_memory:
            self.__item: Review | None = None
        else:
            self.__index: int = 0
            self.__items: np.ndarray | None = None

        self.__dataset_pathname: str | None = None
        self.__dataset_title: str | None = None
        self.dataset_pathname = dataset_pathname

        self.__tfidf_matrix: csr_matrix | None = None
        self.__feature_names: np.ndarray | None = None

    def __len__(self):
        if self.__save_memory:
            return 1
        if self.__items is None:
            return 0
        return len(self.__items)

    def __getitem__(self, item):
        if self.__save_memory:
            raise IndexError('DataSet对象只有一个元素，不能使用索引，请使用迭代器')
        else:
            return self.__items[item]

    def __iter__(self):
        if self.__save_memory:
            self._readline().close()
            self.__iterator = self._readline()
        else:
            self.__iterator = iter(self.__items)
        return self

    def __next__(self) -> Review:
        return next(self.__iterator)

    @property
    def save_memory(self):
        return self.__save_memory

    @property
    def processes(self):
        return self.__processes

    @processes.setter
    def processes(self, processes: int):
        if self.__save_memory:
            if processes != 1:
                print('Warning:由于save_memory=True，processes将被强制设置为1', file=stderr)
            self.__processes = 1
            return
        if isinstance(processes, int) and processes > 0:
            self.__processes = processes
        else:
            try:
                self.processes = int(processes)  # 尝试转换为整数，递归调用setter
            except ValueError:
                print(f'Warning:进程数必须是一个正整数，得到了{processes}，将使用默认值1', file=stderr)
                self.__processes = 1

    @property
    def dataset_pathname(self):
        return self.__dataset_pathname

    @dataset_pathname.setter
    def dataset_pathname(self, dataset_pathname: str):
        if not os.path.exists(dataset_pathname):
            raise FileNotFoundError(dataset_pathname)
        self.__dataset_pathname = dataset_pathname
        self.__dataset_title = os.path.split(self.dataset_pathname)[1]
        if not self.__dataset_title.endswith('.csv'):
            raise ValueError('数据集必须是csv文件')
        if self.__save_memory:
            self.__item = None
        else:
            self._read()

    @property
    def dataset_title(self):
        return self.__dataset_title

    @property
    def item(self):
        return self.__item

    @property
    def items(self):
        return self.__items

    @property
    def tfidf_matrix(self):
        return self.__tfidf_matrix

    @property
    def feature_names(self):
        return self.__feature_names

    def _readline(self):
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

    def tfidf(self) -> csr_matrix:
        """计算tf-idf"""

        def reviews():
            for item in self:
                yield item.review

        vectorizer = TfidfVectorizer(stop_words='english')  # 实例化TfidfVectorizer对象，并设置定制化选项
        # 对文本数据进行tf-idf转换
        self.__tfidf_matrix: csr_matrix = vectorizer.fit_transform(reviews())
        self.__feature_names: np.ndarray = vectorizer.get_feature_names_out()  # 检索词汇表
        self.__feature_names.setflags(write=False)
        return self.__tfidf_matrix

    def word2vec(self):
        raise NotImplementedError

    def to_svm(self, save_path: str = None) -> str:
        """保存为svm格式"""
        if self.__tfidf_matrix is None:
            self.tfidf()
        if save_path is None:
            save_path = rf'..\svm\train\{self.__dataset_title}.txt'

        if self.__processes > 1:
            args = [(item.sentiment, row) for item, row in zip(self.__items, self.__tfidf_matrix)]
            with Pool(processes=self.__processes) as pool:
                results = pool.starmap(DataSet._generate_line, args)
            with open(save_path, 'w') as file:
                for line in results:
                    file.write(line)
        elif self.__save_memory:
            with open(save_path, 'w') as file:
                for item in self:
                    file.write(DataSet._generate_line(item.sentiment, self.__tfidf_matrix))
        else:
            with open(save_path, 'w') as file:
                for item, row in zip(self.__items, self.__tfidf_matrix):
                    file.write(DataSet._generate_line(item.sentiment, row))
        return save_path

    @staticmethod
    def _generate_line(sentiment: bool, row: csr_matrix) -> str:
        return f'{"+" if sentiment else "-"}1 {" ".join([f"{i}:{row[0, i]}" for i in sorted(row.nonzero()[1])])}\n'
