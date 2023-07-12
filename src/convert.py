import os
import time
import numpy as np
from sys import stderr
from .dataset import TfIdfDataset
from scipy.sparse import csr_matrix
from torch.utils.data import Dataset
from multiprocessing.pool import Pool
from sklearn.feature_extraction.text import TfidfVectorizer


class Converter(object):
    """数据转换器"""
    def __init__(self, dataset: Dataset, processes: int = os.cpu_count()):
        # 数据集
        self.__dataset: Dataset | None = None
        self.dataset = dataset
        self.__items: list | None = None
        # 进程数
        self.__processes: int | None = None
        self.processes = processes
        # tf-idf转换
        self.__tfidf_matrix: csr_matrix | None = None
        self.__feature_names: np.ndarray | None = None
        self.__tfidf_dataset: TfIdfDataset | None = None
        # 词向量转换

    @property
    def dataset(self):
        return self.__dataset

    @dataset.setter
    def dataset(self, dataset: Dataset):
        if not isinstance(dataset, Dataset):
            raise TypeError('dataset必须是一个torch.utils.data.Dataset对象')
        if not hasattr(dataset, '__len__'):
            raise AttributeError('dataset必须实现__len__方法')
        self.__dataset = dataset
        self.__items = None
        self.__tfidf_matrix = None
        self.__feature_names = None
        self.__tfidf_dataset = None

    @property
    def processes(self):
        return self.__processes

    @processes.setter
    def processes(self, processes: int):
        try:
            self.__processes = int(float(processes))
            if self.__processes <= 0:
                raise ValueError
        except ValueError:
            print(f'Warning:进程数必须是一个正整数，得到了{processes}，将使用默认值1', file=stderr)
            self.__processes = 1

    @property
    def tfidf_matrix(self):
        if self.__tfidf_matrix is None:
            self.tfidf()
        return self.__tfidf_matrix

    @property
    def feature_names(self):
        if self.__feature_names is None:
            self.tfidf()
        return self.__feature_names

    @property
    def items(self):
        """获取数据集中的所有数据项，列表"""
        if self.__items is None:
            # noinspection PyTypeChecker
            self.__items = [self.__dataset[_] for _ in range(len(self.__dataset))]
        return self.__items

    @property
    def items_generator(self):
        """获取数据集中的所有数据项，生成器"""
        # noinspection PyTypeChecker
        return (self.__dataset[_] for _ in range(len(self.__dataset)))

    @property
    def tfidf_dataset(self):
        """tf-idf数据集"""
        if self.__tfidf_dataset is None:
            return self.tfidf()
        return self.__tfidf_dataset

    def tfidf(self) -> TfIdfDataset:
        """计算tf-idf"""
        vectorizer = TfidfVectorizer(stop_words='english')  # 实例化TfidfVectorizer对象，并设置定制化选项
        # 对文本数据进行tf-idf转换
        self.__tfidf_matrix = vectorizer.fit_transform(item.review for item in self.items)
        self.__feature_names = vectorizer.get_feature_names_out()  # 检索词汇表
        self.__feature_names.setflags(write=False)
        self.__tfidf_dataset = TfIdfDataset(self.__tfidf_matrix, np.array([item.sentiment for item in self.items]))
        return self.__tfidf_dataset

    def word2vec(self):
        raise NotImplementedError

    def to_svm(self, save_path: str = None) -> str:
        """保存为svm格式"""
        if save_path is None:
            if hasattr(self.__dataset, 'dataset_title'):
                save_path = rf'..\svm\data\{self.__dataset.dataset_title}.txt'
            else:
                save_path = rf'..\svm\data\to_svm_output({time.time()}).txt'
        if self.__processes > 1:
            args = [(item.sentiment, row) for item, row in zip(self.items_generator, self.tfidf_matrix)]
            with Pool(processes=self.__processes) as pool:
                results = pool.starmap(self._generate_line, args)
            with open(save_path, 'w') as file:
                for line in results:
                    file.write(line)
        else:
            with open(save_path, 'w') as file:
                for item, row in zip(self.items_generator, self.__tfidf_matrix):
                    file.write(self._generate_line(item.sentiment, row))
        return os.path.abspath(save_path)

    @staticmethod
    def _generate_line(sentiment: bool, row: csr_matrix) -> str:
        return f'{"+" if sentiment else "-"}1 {" ".join([f"{i}:{row[0, i]}" for i in sorted(row.nonzero()[1])])}\n'
