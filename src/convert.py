import os
import time
import numpy as np
from sys import stderr
from typing import Callable
from scipy.sparse import csr_matrix
from torch.utils.data import Dataset
from multiprocessing.pool import Pool
from gensim.models.word2vec import Word2Vec
from sklearn.feature_extraction.text import TfidfVectorizer

from src import tools
from src.dataset import TfIdfDataset, Word2VecDataset


class Converter:
    """数据转换器"""

    def __init__(self, dataset: Dataset, processes: int = 1):
        # 数据集
        self.__dataset: Dataset | None = None
        self.dataset = dataset
        self.__reviews: tuple[str] | None = None
        self.__labels: tuple[bool] | None = None
        # 进程数
        self.__processes: int | None = None
        self.processes = processes
        # tf-idf转换
        self.__tfidf_matrix: csr_matrix | None = None
        self.__feature_names: np.ndarray | None = None
        self.__tfidf_dataset: TfIdfDataset | None = None
        # 词向量转换
        self.__word2vec_model: Word2Vec | None = None
        self.__word2vec_dataset: Word2VecDataset | None = None

    @property
    def dataset(self):
        return self.__dataset

    @dataset.setter
    def dataset(self, dataset: Dataset):
        self.__dataset = tools.check_dataset(dataset)
        reviews = []
        labels = []
        # noinspection PyTypeChecker
        for i in range(len(self.__dataset)):
            reviews.append(self.__dataset[i][0])
            labels.append(self.__dataset[i][1])
        self.__reviews = tuple(reviews)
        self.__labels = tuple(labels)
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
            if self.__processes <= 0 or self.__processes > os.cpu_count():
                raise ValueError
        except ValueError:
            print(f'Warning:进程数必须是一个正整数，得到了{processes}，将使用默认值1', file=stderr)
            self.__processes = 1

    @property
    def reviews(self):
        """获取数据集中的所有评论，元组"""
        return self.__reviews

    @property
    def labels(self):
        """获取数据集中的所有标签，元组"""
        return self.__labels

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

    @property
    def word2vec_model(self):
        """词向量模型"""
        if self.__word2vec_model is None:
            self.word2vec()
        return self.__word2vec_model

    @property
    def word2vec_dataset(self):
        """词向量数据集"""
        if self.__word2vec_dataset is None:
            return self.word2vec()
        return self.__word2vec_dataset

    def tfidf(self, **kwargs) -> TfIdfDataset:
        """计算tf-idf"""
        # 实例化TfidfVectorizer对象，并设置定制化选项
        vectorizer = TfidfVectorizer(**kwargs)
        # 对文本数据进行tf-idf转换
        self.__tfidf_matrix = vectorizer.fit_transform(self.__reviews)
        self.__feature_names = vectorizer.get_feature_names_out()  # 检索词汇表
        self.__feature_names.setflags(write=False)
        self.__tfidf_dataset = TfIdfDataset(self.__tfidf_matrix, np.array(self.__labels, dtype=np.bool))
        return self.__tfidf_dataset

    def word2vec(self, **kwargs):
        """计算词向量"""
        if "sentences" in kwargs:
            kwargs.update(sentences=[review.split() for review in self.__reviews])
        if "workers" not in kwargs:
            kwargs.update(workers=self.__processes)
        self.__word2vec_model = Word2Vec(**kwargs)
        sentence_vectors = []
        for sentence in kwargs["sentences"]:
            sentence_vector = np.zeros(self.__word2vec_model.vector_size, dtype=self.__word2vec_model.wv.vectors.dtype)
            word_count = 0
            for word in sentence:
                if word in self.__word2vec_model.wv.key_to_index:
                    sentence_vector += self.__word2vec_model.wv[word]
                    word_count += 1
            if word_count > 0:
                sentence_vector /= word_count
            sentence_vectors.append(sentence_vector)
        self.__word2vec_dataset = Word2VecDataset(sentence_vectors, np.array(self.__labels, dtype=np.bool))
        return self.__word2vec_dataset

    def __to_svm(self, save_path: str, generate_func: Callable, values: csr_matrix | list[np.ndarray]) -> str:
        """保存为svm格式"""
        tools.check_str(save_path)
        tools.check_callable(generate_func)
        if not os.path.exists(os.path.dirname(save_path)):
            os.makedirs(os.path.dirname(save_path))
        if self.__processes > 1:
            args = [(label, value) for label, value in zip(self.__labels, values)]
            with Pool(processes=self.__processes) as pool:
                results = pool.starmap(generate_func, args)
            with open(save_path, 'w') as file:
                for line in results:
                    file.write(line)
        else:
            with open(save_path, 'w') as file:
                for label, value in zip(self.__labels, values):
                    file.write(generate_func(label, value))
        return os.path.abspath(save_path)

    def tfidf_to_svm(self, save_path: str = None) -> str:
        """保存tf-idf为svm格式"""
        if save_path is None:
            if hasattr(self.__dataset, 'dataset_title'):
                save_path = rf'.\svm\data\{self.__dataset.dataset_title}.txt'
            else:
                save_path = rf'.\svm\data\tfidf_to_svm_output({time.time()}).txt'
        return self.__to_svm(save_path, self._tfidf_generate_line, self.tfidf_matrix)

    def word2vec_to_svm(self, save_path: str = None) -> str:
        if save_path is None:
            if hasattr(self.__dataset, 'dataset_title'):
                save_path = rf'.\svm\data\{self.__dataset.dataset_title}.txt'
            else:
                save_path = rf'.\svm\data\word2vec_to_svm_output({time.time()}).txt'
        return self.__to_svm(save_path, self._word2vec_generate_line, self.word2vec_dataset.values)

    @staticmethod
    def _tfidf_generate_line(sentiment: bool, row: csr_matrix) -> str:
        return f'{"+" if sentiment else "-"}1 {" ".join([f"{i}:{row[0, i]}" for i in sorted(row.nonzero()[1])])}\n'

    @staticmethod
    def _word2vec_generate_line(sentiment: bool, review: np.ndarray) -> str:
        return f'{"+" if sentiment else "-"}1 {" ".join([f"{i}:{review[i]}" for i in range(len(review))])}\n'
