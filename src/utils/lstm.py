import torch
from torch import nn

from src.utils import tools


class TextClassifier(nn.Module):
    def __init__(
            self,
            input_size: int, hidden_size: int, num_layers: int,
            output_size: int = None, fc: nn.Module = None, dropout: float = 0,
    ):
        if output_size is None and fc is None:
            raise ValueError('output_size和fc不能同时为None')
        super().__init__()
        self.__input_size = tools.check_pint(input_size)
        self.__hidden_size = tools.check_pint(hidden_size)
        self.__num_layers = tools.check_pint(num_layers)
        self.__output_size = tools.check_pint(output_size, include_none=True)
        # LSTM层
        self.__lstm = nn.LSTM(
            input_size=self.__input_size, hidden_size=self.__hidden_size,
            num_layers=self.__num_layers, dropout=tools.check_nnfloat(dropout, auto_convert=True),
            batch_first=True
        )
        # 全连接层
        self.__fc = nn.Linear(hidden_size, output_size) if fc is None else tools.TypeCheck(nn.Module)(fc)
        # sigmoid层
        self.__sigmoid = nn.Sigmoid()

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        lstm_out: torch.Tensor = self.__lstm(input_tensor)[0]
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
        self.__hidden_size = tools.check_pint(hidden_size)
        self.__query = nn.Linear(self.__hidden_size, self.__hidden_size)
        self.__key = nn.Linear(self.__hidden_size, self.__hidden_size)
        self.__value = nn.Linear(self.__hidden_size, self.__hidden_size)

    def forward(self, x):
        q = self.__query(x)
        k = self.__key(x)
        v = self.__value(x)

        attn_scores = torch.matmul(q, k.transpose(-2, -1))
        attn_scores = torch.softmax(attn_scores, dim=-1)

        weighted_sum = torch.matmul(attn_scores, v)
        return weighted_sum

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
