import torch
from torch import nn

from src.utils import typecheck


class TextClassifier(nn.Module):
    def __init__(
            self,
            input_size: int, hidden_size: int, num_layers: int,
            output_size: int = None, fc: nn.Module = None, dropout: float = 0,
    ):
        if output_size is None and fc is None:
            raise ValueError('output_size和fc不能同时为None')
        super().__init__()
        self.__input_size = typecheck.check_pint(input_size)
        self.__hidden_size = typecheck.check_pint(hidden_size)
        self.__num_layers = typecheck.check_pint(num_layers)
        self.__output_size = typecheck.check_pint(output_size, include_none=True)
        # LSTM层
        self.__lstm = nn.LSTM(
            input_size=self.__input_size, hidden_size=self.__hidden_size,
            num_layers=self.__num_layers, dropout=typecheck.check_nnfloat(dropout, auto_convert=True),
            batch_first=True
        )
        self.__hidden = None
        self.__cell = None
        # 全连接层
        self.__fc = nn.Linear(hidden_size, output_size) if fc is None else typecheck.TypeCheck(nn.Module)(fc)
        # sigmoid层
        self.__sigmoid = nn.Sigmoid()

    def init_hidden(self, input_tensor: torch.Tensor):
        if input_tensor.ndim == 2:
            init = torch.zeros(self.__num_layers, self.__hidden_size).to(input_tensor.device)
        else:
            init = torch.zeros(self.__num_layers, input_tensor.size(0), self.__hidden_size).to(input_tensor.device)
        self.__hidden = init
        self.__cell = init.clone()

    def detach_hidden(self):
        self.__hidden = self.__hidden.detach() if self.__hidden is not None else None
        self.__cell = self.__cell.detach() if self.__cell is not None else None

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        self.init_hidden(input_tensor)
        lstm_out, (self.__hidden, self.__cell) = self.__lstm(input_tensor, (self.__hidden, self.__cell))
        fc_out = self.__fc(lstm_out if lstm_out.ndim == 2 else lstm_out[:, -1, :])
        sigmoid_out = self.__sigmoid(fc_out)
        return sigmoid_out

    @property
    def input_size(self):
        return self.__input_size

    @property
    def hidden_size(self):
        return self.__hidden_size

    @property
    def num_layers(self):
        return self.__num_layers

    @property
    def output_size(self):
        return self.__output_size

    @property
    def lstm(self):
        return self.__lstm

    @property
    def fc(self):
        return self.__fc

    @property
    def sigmoid(self):
        return self.__sigmoid


class SelfAttention(nn.Module):
    def __init__(self, hidden_size: int):
        super().__init__()
        self.__hidden_size = typecheck.check_pint(hidden_size)
        self.__query = nn.Linear(self.__hidden_size, self.__hidden_size)
        self.__key = nn.Linear(self.__hidden_size, self.__hidden_size)
        self.__value = nn.Linear(self.__hidden_size, self.__hidden_size)

    def forward(self, input_tensor: torch.Tensor):
        q: torch.Tensor = self.__query(input_tensor)
        k: torch.Tensor = self.__key(input_tensor)
        v: torch.Tensor = self.__value(input_tensor)
        return ((q @ k.t()) / torch.sqrt(torch.tensor(input_tensor.size(1)))) @ v

    @property
    def hidden_size(self):
        return self.__hidden_size

    @property
    def query(self):
        return self.__query

    @property
    def key(self):
        return self.__key

    @property
    def value(self):
        return self.__value
