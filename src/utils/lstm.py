import torch
from torch import nn

from src.utils import tools


class LSTMModel(nn.Module):
    def __init__(
            self,
            input_size: int, hidden_size: int, num_layers: int,
            output_size: int = None, fc: nn.Module = None, dropout: float = 0,
            device: torch.device = torch.device('cpu')
    ):
        if output_size is None and fc is None:
            raise ValueError('output_size和fc不能同时为None')
        super().__init__()
        self.__input_size = tools.check_pint(input_size)
        self.__hidden_size = tools.check_pint(hidden_size)
        self.__num_layers = tools.check_pint(num_layers)
        self.__output_size = tools.check_pint(output_size, include_none=True)
        self.__device: torch.device = tools.TypeCheck(torch.device)(device)
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
        # 将模型移动到指定设备
        self.to(self.__device)

    def forward(self, input_tensor: torch.Tensor):
        lstm_out, _ = self.__lstm(input_tensor)
        fc_out = self.__fc(lstm_out)
        sigmoid_out = self.__sigmoid(fc_out)
        return sigmoid_out

    @property
    def device(self):
        return self.__device

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
