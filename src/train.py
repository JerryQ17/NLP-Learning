import os
import torch
from src import SVM
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import Optimizer


class Trainer:
    def __init__(
            self, tfidf_dataset: Dataset = None, word2vec_dataset: Dataset = None,
            svm_train_path: str = None, svm_model_path: str = None,
            model: nn.Module = None, optimizer: Optimizer = None, criterion: nn.Module = None, device: str = None
    ):
        # 数据集
        # tfidf数据集
        self.__tfidf_dataset: Dataset | None = None
        self.tfidf_dataset = tfidf_dataset
        # 词向量数据集
        self.__word2vec_dataset: Dataset | None = None
        self.word2vec_dataset = word2vec_dataset

        # svm
        if svm_train_path and not os.path.exists(svm_train_path):
            raise FileNotFoundError(f'文件{svm_train_path}不存在')
        self.__svm_train_path: str = svm_train_path
        self.__svm_model_path: str = svm_model_path
        self.__svm: SVM = SVM(problem_path=self.__svm_train_path, model_path=self.__svm_model_path)

        # lstm
        self.__model: nn.Module | None = None
        self.model = model
        self.__optimizer = optimizer
        self.__criterion = criterion
        self.__device: torch.device | None = None
        self.device = device

    @property
    def tfidf_dataset(self):
        return self.__tfidf_dataset

    @tfidf_dataset.setter
    def tfidf_dataset(self, tfidf_dataset: Dataset):
        if tfidf_dataset is None:
            self.__tfidf_dataset = None
            return
        if not isinstance(tfidf_dataset, Dataset):
            raise TypeError('tfidf_dataset必须是一个torch.utils.data.Dataset对象')
        if not hasattr(tfidf_dataset, '__len__'):
            raise AttributeError('tfidf_dataset必须实现__len__方法')
        self.__tfidf_dataset = tfidf_dataset

    @property
    def word2vec_dataset(self):
        return self.__word2vec_dataset

    @word2vec_dataset.setter
    def word2vec_dataset(self, word2vec_dataset: Dataset):
        if word2vec_dataset is None:
            self.__word2vec_dataset = None
            return
        if not isinstance(word2vec_dataset, Dataset):
            raise TypeError('word2vec_dataset必须是一个torch.utils.data.Dataset对象')
        if not hasattr(word2vec_dataset, '__len__'):
            raise AttributeError('word2vec_dataset必须实现__len__方法')
        self.__word2vec_dataset = word2vec_dataset

    @property
    def svm_train_path(self):
        return self.__svm_train_path

    @svm_train_path.setter
    def svm_train_path(self, svm_train_path: str):
        if not os.path.exists(svm_train_path):
            raise FileNotFoundError(f'文件{svm_train_path}不存在')
        self.__svm_train_path = svm_train_path
        self.__svm = SVM(problem_path=svm_train_path, model_path=self.__svm_model_path)

    @property
    def svm_model_path(self):
        return self.__svm_model_path

    @svm_model_path.setter
    def svm_model_path(self, svm_model_path: str):
        self.__svm_model_path = svm_model_path
        self.__svm = SVM(problem_path=self.__svm_train_path, model_path=svm_model_path)

    @property
    def svm(self):
        return self.__svm

    @property
    def model(self):
        return self.__model

    @model.setter
    def model(self, model: nn.Module):
        if model is None:
            self.__model = None
            return
        if not isinstance(model, nn.Module):
            raise TypeError('model必须是一个torch.nn.Module对象')
        self.__model = model

    @property
    def optimizer(self):
        return self.__optimizer

    @optimizer.setter
    def optimizer(self, optimizer: Optimizer):
        if optimizer is None:
            self.__optimizer = None
            return
        if not isinstance(optimizer, Optimizer):
            raise TypeError('optimizer必须是一个torch.optim.Optimizer对象')
        self.__optimizer = optimizer

    @property
    def criterion(self):
        return self.__criterion

    @criterion.setter
    def criterion(self, criterion: nn.Module):
        if criterion is None:
            self.__criterion = None
            return
        if not isinstance(criterion, nn.Module):
            raise TypeError('criterion必须是一个torch.nn.Module对象')
        self.__criterion = criterion

    @property
    def device(self):
        return self.__device

    @device.setter
    def device(self, device: list[str, int] | str | int):
        if device is None:
            if torch.cuda.is_available():
                self.__device = torch.device('cuda')
            elif torch.backends.mps.is_available():
                self.__device = torch
            else:
                self.__device = torch.device('cpu')
        elif isinstance(device, list) and len(device) == 2 \
                and isinstance(device[0], str) and isinstance(device[1], int):
            self.__device = torch.device(*device)
        elif isinstance(device, (str, int)):
            self.__device = torch.device(device)
        else:
            raise TypeError(f'device必须是一个长度为2的列表或者是一个字符串或者是一个整数')

    def train(self, train_loader: DataLoader, num_epochs: int, enable_logging: bool = False):
        if self.__model is None:
            raise RuntimeError('请先设置model')
        if self.__optimizer is None:
            raise RuntimeError('请先设置optimizer')
        if self.__criterion is None:
            raise RuntimeError('请先设置criterion')
        if not hasattr(train_loader, '__iter__'):
            raise TypeError('train_loader必须可迭代')
        if not isinstance(num_epochs, int) and num_epochs < 1:
            raise ValueError('num_epochs必须是一个正整数')

        self.__model.train()

        for epoch in range(num_epochs):
            if enable_logging:
                print(f"Epoch {epoch + 1}/{num_epochs}")
            for i, (texts, labels) in enumerate(train_loader):
                texts = torch.Tensor(texts)
                labels = torch.Tensor(labels).type(torch.LongTensor)

                outputs = self.__model(texts)
                loss = self.__criterion(outputs, labels)

                self.__optimizer.zero_grad()
                loss.backward()
                self.__optimizer.step()
                if enable_logging and (i + 1) % 100 == 0:
                    print(f"Step {i + 1}/{len(train_loader)}, Loss: {loss.item():.4f}")
            if enable_logging:
                print('-' * 50)

    def evaluate(self, test_loader: DataLoader, enable_logging: bool = False) -> float:
        if self.__model is None:
            raise RuntimeError('请先设置model')
        if not hasattr(test_loader, '__iter__'):
            raise TypeError('test_loader必须可迭代')

        self.__model.eval()

        with torch.no_grad():
            correct = 0
            total = 0
            for texts, labels in test_loader:
                texts = torch.Tensor(texts)
                labels = torch.Tensor(labels).type(torch.LongTensor)

                outputs = self.__model(texts)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += torch.sum(predicted.eq(labels)).item()

            accuracy = 100 * correct / total
            if enable_logging:
                print(f"Accuracy: {accuracy:.4f}%")
            return accuracy

    def save(self, save_path: str = r'..\lstm\model\lstm.pth') -> str:
        if self.__model is None:
            raise RuntimeError('请先设置model')
        torch.save(self.__model.state_dict(), save_path)
        return os.path.abspath(save_path)

    def load(self, load_path: str):
        if self.__model is None:
            raise RuntimeError('请先设置model')
        self.__model.load_state_dict(torch.load(load_path))
