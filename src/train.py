import os
import torch
import atexit
import signal
import logging
from torch import nn
from types import FrameType
from torch.optim import Optimizer
from torch.utils.data import Dataset, DataLoader

from src import tools, SVM
from src.models import NNTrainingState


class Trainer:
    """数据训练器"""
    __instances: list['Trainer'] = []
    __autosave_registered: bool = False
    __autosave_signals: set[signal.Signals, ...] = {signal.SIGINT, signal.SIGTERM, signal.SIGABRT}

    def __new__(cls, *args, **kwargs):
        if not cls.__autosave_registered:
            atexit.register(cls.__autosave_all)
            cls.__autosave_registered = True
        instance = super().__new__(cls)
        cls.__instances.append(instance)
        return instance

    def __del__(self):
        self.__class__.__instances.remove(self)

    @classmethod
    def autosave_signals(cls):
        return frozenset(cls.__autosave_signals)

    @classmethod
    def add_autosave_signals(cls, *autosave_signals: signal.Signals):
        autosave_signals = set(autosave_signals)
        tools.TypeCheck(signal.Signals)(*autosave_signals)
        for sig in autosave_signals - cls.__autosave_signals:
            signal.signal(sig, cls._auto_save_handler)
            cls.__autosave_signals.add(sig)

    @classmethod
    def remove_autosave_signals(cls, *autosave_signals: signal.Signals):
        autosave_signals = set(autosave_signals)
        tools.TypeCheck(signal.Signals)(*autosave_signals)
        for sig in autosave_signals & cls.__autosave_signals:
            signal.signal(sig, signal.SIG_DFL)
            cls.__autosave_signals.remove(sig)

    @classmethod
    def __autosave_all(cls):
        for instance in cls.__instances:
            if instance.__autosave:
                instance._auto_save_handler()

    # noinspection PyUnusedLocal
    def _auto_save_handler(self, error: signal.Signals | Exception | int = None, frame: FrameType = None):
        self.__logger.error(f'发生异常：{error}\n保存中...')
        if self.__nn_training_state != NNTrainingState():
            nn_savepath = fr'{self.autosave_dir}\nn{self.__instances.index(self)}.pth'
            torch.save(self.__nn_training_state, nn_savepath)
            self.__logger.info(f'神经网络训练状态已保存于{nn_savepath}')
        if len(self.svm.grid_results):
            svm_savepath = fr'{self.autosave_dir}\svm{self.__instances.index(self)}.pth'
            torch.save([i.dict() for i in self.svm.grid_results], svm_savepath)
            self.__logger.info(f'支持向量机训练状态已保存于{svm_savepath}')

    def __init__(
            self, logger: logging.Logger = logging.getLogger(__name__),
            autosave: bool = True, autosave_dir: str = r'.\autosave',
            tfidf_dataset: Dataset = None, word2vec_dataset: Dataset = None,
            svm_train_path: str = None, svm_model_path: str = None,
            model: nn.Module = None, optimizer: Optimizer = None, criterion: nn.Module = None,
            device: torch.device | list[str, int] | str | int | None = None
    ):
        # 日志记录
        self.__logger: logging.Logger | None = None
        self.logger = logger

        # 自动保存
        self.__autosave: bool = bool(autosave)
        self.__autosave_dir: str = ''
        self.autosave_dir = autosave_dir

        if self.__autosave:
            for auto_save_signal in self.__autosave_signals:
                signal.signal(auto_save_signal, self._auto_save_handler)

        # 数据集
        # tfidf数据集
        self.__tfidf_dataset: Dataset | None = None
        self.tfidf_dataset = tfidf_dataset
        # 词向量数据集
        self.__word2vec_dataset: Dataset | None = None
        self.word2vec_dataset = word2vec_dataset

        # SVM
        self.__svm_train_path: str | None = None
        self.svm_train_path = svm_train_path
        self.__svm_model_path: str | None = None
        self.svm_model_path = svm_model_path
        self.__svm: SVM = SVM(self.__svm_train_path, self.__svm_model_path)

        # Neural Network
        self.__nn_training_state: NNTrainingState = NNTrainingState()
        self.__model: nn.Module | None = None
        self.model = model
        self.__optimizer: Optimizer | None = None
        self.optimizer = optimizer
        self.__criterion: nn.Module | None = None
        self.criterion = criterion
        self.__device: torch.device | None = None
        self.device = device

    @property
    def logger(self):
        return self.__logger

    @logger.setter
    def logger(self, logger):
        self.__logger = tools.TypeCheck(logging.Logger)(logger, default=logging.getLogger(__name__))

    @property
    def autosave(self):
        return self.__autosave

    @autosave.setter
    def autosave(self, autosave: bool):
        self.__autosave = bool(autosave)

    @property
    def autosave_dir(self):
        return self.__autosave_dir

    @autosave_dir.setter
    def autosave_dir(self, autosave_dir: str):
        self.__autosave_dir = tools.check_dir(autosave_dir)

    @property
    def tfidf_dataset(self):
        return self.__tfidf_dataset

    @tfidf_dataset.setter
    def tfidf_dataset(self, tfidf_dataset: Dataset):
        self.__tfidf_dataset = tools.check_dataset(tfidf_dataset, include_none=True)

    @property
    def word2vec_dataset(self):
        return self.__word2vec_dataset

    @word2vec_dataset.setter
    def word2vec_dataset(self, word2vec_dataset: Dataset):
        self.__word2vec_dataset = tools.check_dataset(word2vec_dataset, include_none=True)

    @property
    def svm_train_path(self):
        return self.__svm_train_path

    @svm_train_path.setter
    def svm_train_path(self, svm_train_path: str):
        self.__svm_train_path = tools.check_file(svm_train_path, include_none=True)
        if self.__svm_train_path is not None:
            self.__svm.problem_path = svm_train_path

    @property
    def svm_model_path(self):
        return self.__svm_model_path

    @svm_model_path.setter
    def svm_model_path(self, svm_model_path: str):
        self.__svm_model_path = tools.check_file(svm_model_path, include_none=True)
        if self.__svm_model_path:
            self.__svm.load(model_path=svm_model_path)

    @property
    def svm(self):
        return self.__svm

    @property
    def model(self):
        return self.__model

    @model.setter
    def model(self, model: nn.Module):
        self.__model = tools.TypeCheck(nn.Module)(model, include_none=True)
        if hasattr(self, f'_{self.__class__.__name__}__device') and self.__model and self.__device:
            self.__model.to(self.__device)

    @property
    def optimizer(self):
        return self.__optimizer

    @optimizer.setter
    def optimizer(self, optimizer: Optimizer):
        self.__optimizer = tools.TypeCheck(Optimizer)(optimizer, include_none=True)

    @property
    def criterion(self):
        return self.__criterion

    @criterion.setter
    def criterion(self, criterion: nn.Module):
        self.__criterion = tools.TypeCheck(nn.Module)(criterion, include_none=True)

    @property
    def device(self):
        return self.__device

    @device.setter
    def device(self, device: torch.device | list[str, int] | str | int | None):
        if device is None:
            if torch.cuda.is_available():
                self.__device = torch.device('cuda')
            elif torch.backends.mps.is_available():
                self.__device = torch.device('mps')
            else:
                self.__device = torch.device('cpu')
        elif isinstance(device, torch.device):
            self.__device = device
        elif isinstance(device, list) and len(device) == 2 \
                and isinstance(device[0], str) and isinstance(device[1], int):
            self.__device = torch.device(*device)
        elif isinstance(device, (str, int)):
            self.__device = torch.device(device)
        else:
            raise TypeError(f'device的类型应为torch.device | list[str, int] | str | int | None')
        if self.__model:
            self.__model = self.__model.to(self.__device)
        if self.__criterion:
            self.__criterion = self.__criterion.to(self.__device)

    def __ready_to_train_nn(self):
        if self.__model is None:
            raise ValueError('请先设置model')
        if self.__optimizer is None:
            raise ValueError('请先设置optimizer')
        if self.__criterion is None:
            raise ValueError('请先设置criterion')

    def train(
            self,
            epochs: int = 1, *,
            tfidf_mode: bool = False,
            word2vec_mode: bool = False,
            **dataloader_kwargs
    ) -> 'Trainer':
        # 检查内部属性
        self.__ready_to_train_nn()
        # 检查参数
        if bool(tfidf_mode) == bool(word2vec_mode):
            raise ValueError('svm_mode和word2vec_mode不能同时为True或同时为False')
        if not isinstance(epochs, int) and epochs < 1:
            raise ValueError('epochs必须是一个正整数')
        if 'dataset' in dataloader_kwargs:
            raise ValueError('loader_kwargs不能包含dataset参数')
        # 选择数据集并转化为DataLoader
        if tfidf_mode:
            if self.__tfidf_dataset is None:
                raise ValueError('tfidf_dataset不能为None')
            loader = DataLoader(dataset=self.__tfidf_dataset, **dataloader_kwargs)
        else:
            if self.__word2vec_dataset is None:
                raise ValueError('word2vec_dataset不能为None')
            loader = DataLoader(dataset=self.__word2vec_dataset, **dataloader_kwargs)

        self.__model.to(self.__device)
        self.__model.train()

        if self.autosave:
            self.__nn_training_state.current_epoch = 0
            self.__nn_training_state.total_epoch = epochs

        try:
            for epoch in range(epochs):
                self.__logger.info(f"Epoch {epoch + 1}/{epochs}")
                for i, (texts, labels) in enumerate(loader):
                    texts = torch.Tensor(texts).to(self.__device).float()
                    labels = torch.Tensor(labels).type(torch.LongTensor).to(self.__device)

                    outputs = self.__model(texts)
                    loss = self.__criterion(outputs, labels)

                    self.__optimizer.zero_grad()
                    loss.backward()
                    self.__optimizer.step()
                    if (i + 1) % 100 == 0:
                        self.__logger.info(f"Step {i + 1}/{len(loader)}, Loss: {loss.item():.4f}")
                if self.autosave:
                    self.__nn_training_state.current_epoch = epoch
                    self.__nn_training_state.model_state_dict = self.__model.state_dict()
                    self.__nn_training_state.optimizer_state_dict = self.__optimizer.state_dict()
            return self
        except Exception as error:
            self.__logger.exception('训练时遇到错误', exc_info=error)
            self._auto_save_handler(error)
            raise RuntimeError(error) from error

    def train_from_state(self, path: str, *,
                         tfidf_mode: bool = False, word2vec_mode: bool = False,
                         **dataloader_kwargs) -> 'Trainer':
        tools.check_file(path)
        self.__ready_to_train_nn()
        self.__nn_training_state = NNTrainingState(**torch.load(path))
        self.__model.load_state_dict(self.__nn_training_state.model_state_dict)
        self.__optimizer.load_state_dict(self.__nn_training_state.optimizer_state_dict)
        epoch = self.__nn_training_state.total_epoch - self.__nn_training_state.current_epoch
        return self.train(epoch, tfidf_mode=tfidf_mode, word2vec_mode=word2vec_mode, **dataloader_kwargs)

    def evaluate(self, test_loader: DataLoader) -> float:
        if self.__model is None:
            raise RuntimeError('请先设置model')
        tools.TypeCheck(DataLoader)(test_loader)

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
            return 100 * correct / total

    def predict(self, texts: torch.Tensor) -> torch.Tensor:
        if self.__model is None:
            raise RuntimeError('请先设置model')
        tools.TypeCheck(torch.Tensor)(texts)

        self.__model.to(self.__device)
        self.__model.eval()

        with torch.no_grad():
            outputs: torch.Tensor = self.__model(texts)
            _, predicted = torch.max(outputs.data, 1)
        return predicted

    def save(self, save_path: str = r'.\lstm\model\lstm.pth') -> str:
        if self.__model is None:
            raise RuntimeError('请先设置model')
        torch.save(self.__model.state_dict(), tools.check_str(save_path))
        return os.path.abspath(save_path)

    def load(self, load_path: str):
        if self.__model is None:
            raise RuntimeError('请先设置model')
        self.__model.load_state_dict(torch.load(tools.check_str(load_path)))
        return self
