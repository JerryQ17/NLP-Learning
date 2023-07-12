import os
import torch
import signal
from src import SVM
from torch import nn
from types import FrameType
from torch.optim import Optimizer
from src.models import NNTrainingState
from torch.utils.data import Dataset, DataLoader


class Trainer(object):
    """数据训练器"""

    def __init__(
            self,
            autosave: bool = True, autosave_dir: str = r'..\autosave',
            tfidf_dataset: Dataset = None, word2vec_dataset: Dataset = None,
            svm_train_path: str = None, svm_model_path: str = None,
            model: nn.Module = None, optimizer: Optimizer = None, criterion: nn.Module = None, device: str = None
    ):
        # 自动保存
        self.__autosave: bool = autosave is True
        self.__autosave_dir: str = ''
        self.autosave_dir = autosave_dir
        self.__auto_save_signals: tuple[signal.Signals, ...] = (signal.SIGINT, signal.SIGTERM, signal.SIGABRT,)
        if self.__autosave:
            for auto_save_signal in self.__auto_save_signals:
                signal.signal(auto_save_signal, self._auto_save_handler)

        # 数据集
        # tfidf数据集
        self.__tfidf_dataset: Dataset | None = None
        self.tfidf_dataset = tfidf_dataset
        # 词向量数据集
        self.__word2vec_dataset: Dataset | None = None
        self.word2vec_dataset = word2vec_dataset

        # svm
        self.__svm_train_path: str | None = None
        self.svm_train_path = svm_train_path
        self.__svm_model_path: str | None = None
        self.svm_model_path = svm_model_path
        self.__svm: SVM = SVM()

        # nn
        self.__nn_training_state: NNTrainingState = NNTrainingState()
        self.__model: nn.Module | None = None
        self.model = model
        self.__optimizer = optimizer
        self.__criterion = criterion
        self.__device: torch.device | None = None
        self.device = device

    @property
    def autosave(self):
        return self.__autosave

    @autosave.setter
    def autosave(self, autosave: bool):
        self.__autosave = autosave is True

    @property
    def autosave_dir(self):
        return self.__autosave_dir

    @autosave_dir.setter
    def autosave_dir(self, autosave_dir: str):
        if not isinstance(autosave_dir, str):
            raise TypeError('autosave_dir必须是一个字符串')
        self.__autosave_dir = autosave_dir

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
        if svm_train_path is None:
            self.__svm_train_path = None
            return
        if not os.path.isfile(svm_train_path):
            raise FileNotFoundError(f'文件{svm_train_path}不存在')
        self.__svm_train_path = svm_train_path

    @property
    def svm_model_path(self):
        return self.__svm_model_path

    @svm_model_path.setter
    def svm_model_path(self, svm_model_path: str):
        if svm_model_path is None:
            self.__svm_model_path = None
            return
        if not os.path.isfile(svm_model_path):
            raise FileNotFoundError(f'文件{svm_model_path}不存在')
        self.__svm_model_path = svm_model_path
        self.__svm.load(svm_model_path)

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
        self.__model = model.to(self.__device)

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
    def device(self, device: list[str, int] | str | int | None):
        if device is None:
            if torch.cuda.is_available():
                self.__device = torch.device('cuda')
            else:
                self.__device = torch.device('cpu')
        elif isinstance(device, list) and len(device) == 2 \
                and isinstance(device[0], str) and isinstance(device[1], int):
            self.__device = torch.device(*device)
        elif isinstance(device, (str, int)):
            self.__device = torch.device(device)
        else:
            raise TypeError(f'device必须是一个长度为2的列表或者是一个字符串或者是一个整数')
        if self.__model is not None:
            self.__model = self.__model.to(self.__device)

    def train(
            self,
            train_loader: DataLoader, num_epochs: int, enable_logging: bool = False,
            from_record: bool = False, record_path: str = None
    ):
        if self.__model is None:
            raise RuntimeError('请先设置model')
        if self.__optimizer is None:
            raise RuntimeError('请先设置optimizer')
        if self.__criterion is None:
            raise RuntimeError('请先设置criterion')
        if not hasattr(train_loader, '__iter__'):
            raise AttributeError('train_loader必须可迭代')
        if not isinstance(num_epochs, int) and num_epochs < 1:
            raise ValueError('num_epochs必须是一个正整数')
        if from_record:
            if not isinstance(record_path, str):
                raise TypeError('record_path必须是一个文件路径')
            if not os.path.isfile(record_path):
                raise FileNotFoundError(f'文件{record_path}不存在')
            self.__nn_training_state = NNTrainingState(**torch.load(record_path))
            self.__model.load_state_dict(self.__nn_training_state.model_state_dict)
            self.__optimizer.load_state_dict(self.__nn_training_state.optimizer_state_dict)
            record_epoch = self.__nn_training_state.current_epoch
        else:
            record_epoch = 0

        self.__model.to(self.__device)
        self.__model.train()

        if self.autosave and not from_record:
            self.__nn_training_state.current_epoch = 0
            self.__nn_training_state.total_epochs = num_epochs

        try:
            for epoch in range(num_epochs - record_epoch):
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
                if self.autosave:
                    self.__nn_training_state.current_epoch = epoch
                    self.__nn_training_state.model_state_dict = self.__model.state_dict()
                    self.__nn_training_state.optimizer_state_dict = self.__optimizer.state_dict()
                if enable_logging:
                    print('-' * 50)
        except Exception as error:
            self._auto_save_handler()
            raise error

    def evaluate(self, test_loader: DataLoader, enable_logging: bool = False) -> float:
        if self.__model is None:
            raise RuntimeError('请先设置model')
        if not hasattr(test_loader, '__iter__'):
            raise AttributeError('test_loader必须可迭代')

        self.__model.to(self.__device)
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

    def predict(self, texts: torch.Tensor) -> torch.Tensor:
        if self.__model is None:
            raise RuntimeError('请先设置model')
        if not isinstance(texts, torch.Tensor):
            raise TypeError('texts必须是一个torch.Tensor对象')

        self.__model.to(self.__device)
        self.__model.eval()

        with torch.no_grad():
            outputs: torch.Tensor = self.__model(texts)
            _, predicted = torch.max(outputs.data, 1)
        return predicted

    def save(self, save_path: str = r'..\lstm\model\lstm.pth') -> str:
        if self.__model is None:
            raise RuntimeError('请先设置model')
        torch.save(self.__model.state_dict(), save_path)
        return os.path.abspath(save_path)

    def load(self, load_path: str):
        if self.__model is None:
            raise RuntimeError('请先设置model')
        self.__model.load_state_dict(torch.load(load_path))

    # noinspection PyUnusedLocal
    def _auto_save_handler(self, auto_save_signal: signal.Signals | int = None, frame: FrameType = None):
        print(f'接收到退出信号{auto_save_signal}，保存中...')
        if self.__nn_training_state != NNTrainingState():
            torch.save(self.__nn_training_state, self.autosave_dir + r'\nn.autosave.pth')
            print('nn.autosave.pth已保存')
        if self.svm.grid_results:
            torch.save([i.dict() for i in self.svm.grid_results], self.autosave_dir + r'\svm.autosave.pth')
            print('svm.autosave.pth已保存')
