import os
import nltk
import time
import logging
import numpy as np
from enum import Enum
from typing import Callable
from scipy.sparse import csr_matrix
from torch.utils.data import Dataset
from collections.abc import Generator
from multiprocessing.pool import Pool
from gensim.models.word2vec import Word2Vec
from torch import stack, Tensor, from_numpy
from sklearn.feature_extraction.text import TfidfVectorizer

from src.utils import typecheck
from src.utils.tensor import random_tensors_outside_existed_tensors
from src.utils.dataset import IMDBDataset, TfIdfDataset, Word2VecDataset


class UniqueWords(str, Enum):
    """词汇表中的特殊词，**不保证**这两个字符串不在词汇表中"""
    PADDING = "<PADDING>"
    UNKNOWN = "<UNKNOWN>"


class Converter:
    """数据转换器"""

    def __init__(self, dataset: IMDBDataset, processes: int = 1, logger: logging.Logger = logging.getLogger(__name__)):
        # 日志
        self.__logger: logging.Logger | None = None
        self.logger = logger
        # 数据集
        self.__dataset: IMDBDataset | None = None
        self.dataset = dataset
        self.__reviews_cut: tuple[tuple[str]] | None = None
        # 进程数
        self.__processes: int | None = None
        self.processes = processes
        # tf-idf转换
        self.__tfidf_matrix: csr_matrix | None = None
        self.__feature_names: np.ndarray | None = None
        self.__tfidf_dataset: TfIdfDataset | None = None
        # 词向量转换
        self.__word_tensors: dict[str, Tensor] = {}
        self.__word2vec_dataset: Word2VecDataset | None = None

    @property
    def logger(self):
        return self.__logger

    @logger.setter
    def logger(self, logger):
        self.__logger = typecheck.TypeCheck(logging.Logger)(logger, default=logging.getLogger(__name__))

    @property
    def dataset(self):
        return self.__dataset

    @dataset.setter
    def dataset(self, dataset: Dataset):
        self.__dataset = typecheck.TypeCheck(IMDBDataset)(dataset)
        self.__tfidf_matrix = None
        self.__feature_names = None
        self.__tfidf_dataset = None
        self.__word_tensors = {}
        self.__word2vec_dataset = None

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
            self.__logger.warning(f'进程数不合法，得到了{processes}，将使用默认值1')
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
    def reviews_generator(self) -> Generator[str, None, None]:
        """获取数据集中的所有评论，生成器"""
        return (item[0] for item in self.__dataset.items)

    @property
    def reviews_cut(self):
        """分词后的评论"""
        nltk.download('punkt')
        self.__reviews_cut = tuple(tuple(nltk.word_tokenize(review)) for review in self.reviews_generator)
        return self.__reviews_cut

    @property
    def labels_generator(self) -> Generator[float, None, None]:
        """获取数据集中的所有标签，生成器"""
        return (item[1] for item in self.__dataset.items)

    @property
    def tfidf_dataset(self):
        """tf-idf数据集"""
        if self.__tfidf_dataset is None:
            return self.tfidf()
        return self.__tfidf_dataset

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
        self.__tfidf_matrix = vectorizer.fit_transform(self.reviews_generator)
        self.__feature_names = vectorizer.get_feature_names_out()  # 检索词汇表
        self.__feature_names.setflags(write=False)
        self.__tfidf_dataset = TfIdfDataset(self.__tfidf_matrix, Tensor(tuple(self.labels_generator)))
        return self.__tfidf_dataset

    def word2vec(self, **kwargs):
        """计算词向量"""
        # 实例化Word2Vec对象，并设置定制化选项
        if "sentences" not in kwargs:
            kwargs.update(sentences=self.reviews_cut)
        if "workers" not in kwargs:
            kwargs.update(workers=self.__processes)
        model = Word2Vec(**kwargs)
        word_vectors = model.wv
        del model  # 释放内存
        # 将词向量转换为张量
        for word in word_vectors.key_to_index:
            self.__word_tensors[word] = from_numpy(word_vectors[word])
        del word_vectors  # 释放内存
        # 生成padding和unknown张量
        random_tensors = random_tensors_outside_existed_tensors(*self.__word_tensors.values(), num=2)
        pad_tensor: Tensor = random_tensors[0]
        unknown_tensor: Tensor = random_tensors[1]
        # 将padding和unknown张量加入词向量字典
        self.__word_tensors[UniqueWords.PADDING] = pad_tensor
        self.__word_tensors[UniqueWords.UNKNOWN] = unknown_tensor
        # 将句子转换为张量
        sentence_tensors = []
        for sentence in kwargs["sentences"]:
            sentence_tensor = []
            for word in sentence:
                if word in self.__word_tensors:
                    sentence_tensor.append(self.__word_tensors[word])
                else:
                    sentence_tensor.append(unknown_tensor)
            sentence_tensors.append(stack(sentence_tensor))
        # 生成数据集
        self.__word2vec_dataset = Word2VecDataset(sentence_tensors, Tensor(tuple(self.labels_generator)), pad_tensor)
        return self.__word2vec_dataset

    def __to_svm(self, save_path: str, generate_func: Callable, values: csr_matrix | Tensor) -> str:
        """保存为svm格式"""
        typecheck.check_str(save_path)
        typecheck.check_callable(generate_func)
        if self.__processes > 1:
            args = [(label, value) for label, value in zip(self.labels_generator, values)]
            with Pool(processes=self.__processes) as pool:
                results = pool.starmap(generate_func, args)
            with open(save_path, 'w') as file:
                for line in results:
                    file.write(line)
        else:
            with open(save_path, 'w') as file:
                for label, value in zip(self.labels_generator, values):
                    file.write(generate_func(label, value))
        return os.path.abspath(save_path)

    def tfidf_to_svm(self, save_path: str = None) -> str:
        """保存tf-idf为svm格式"""
        if save_path is None:
            save_path = rf'.\svm\data\tfidf_to_svm_output({time.time()}).txt'
        return self.__to_svm(save_path, self._tfidf_generate_line, self.tfidf_matrix)

    def word2vec_to_svm(self, save_path: str = None) -> str:
        raise NotImplementedError
        # if save_path is None:
        #     save_path = rf'.\svm\data\word2vec_to_svm_output({time.time()}).txt'
        # return self.__to_svm(save_path, self._word2vec_generate_line, self.word2vec_dataset.values)

    @staticmethod
    def _tfidf_generate_line(sentiment: bool, row: csr_matrix) -> str:
        return f'{"+" if sentiment else "-"}1 {" ".join([f"{i}:{row[0, i]}" for i in sorted(row.nonzero()[1])])}\n'

    @staticmethod
    def _word2vec_generate_line(sentiment: bool, review: Tensor) -> str:
        # return f'{"+" if sentiment else "-"}1 {" ".join([f"{i}:{review[i]}" for i in range(len(review))])}\n'
        ...
