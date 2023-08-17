import torch
from torch import nn

from src.utils import tools


class LSTMModel(nn.Module):
    def __init__(
            self,
            input_dim: int, hidden_dim: int, num_layers: int, output_dim: int,
            fc: nn.Module = None, dropout_rate: float = 0,
            device: torch.device = torch.device('cpu')
    ):
        super().__init__()
        self.__input_dim: int = tools.TypeCheck(int)(input_dim)
        self.__hidden_dim: int = tools.TypeCheck(int)(hidden_dim)
        self.__num_layers: int = tools.TypeCheck(int)(num_layers)
        self.__output_dim: int = tools.TypeCheck(int)(output_dim)
        self.__device: torch.device = tools.TypeCheck(torch.device)(device)
        # LSTM层
        self.__lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        # 全连接层
        self.__fc = nn.Linear(hidden_dim, output_dim) if fc is None else tools.TypeCheck(nn.Module)(fc)
        # dropout层
        self.__dropout_rate: float | None = None
        self.__dropout: nn.Dropout | None = None
        self.dropout_rate = dropout_rate
        # sigmoid层
        self.__sigmoid = nn.Sigmoid()
        # 将模型移动到指定设备
        self.to(self.__device)

    def forward(self, input_tensor: torch.Tensor):
        lstm_out, _ = self.__lstm(input_tensor.view(len(input_tensor), 1, -1))
        fc_out = self.__fc(lstm_out.view(len(input_tensor), -1))
        if self.__dropout is not None:
            fc_out = self.__dropout(fc_out)
        sigmoid_out = self.__sigmoid(fc_out)
        return sigmoid_out

    @property
    def device(self):
        return self.__device

    @property
    def input_dim(self):
        return self.__input_dim

    @property
    def hidden_dim(self):
        return self.__hidden_dim

    @property
    def num_layers(self):
        return self.__num_layers

    @property
    def output_dim(self):
        return self.__output_dim

    @property
    def lstm(self):
        return self.__lstm

    @property
    def dropout_rate(self):
        return self.__dropout_rate

    @dropout_rate.setter
    def dropout_rate(self, value: float):
        try:
            value = float(value)
        except ValueError:
            raise ValueError(f'droupout_rate应为float类型，而不是{type(value)}类型')
        if value < 0 or value > 1:
            raise ValueError(f'droupout_rate应在0到1之间，而不是{value}')
        self.__dropout_rate = value
        if value == 0:
            self.__dropout = None
        else:
            self.__dropout = nn.Dropout(value).to(self.__device)

    @property
    def dropout(self):
        return self.__dropout

    @property
    def fc(self):
        return self.__fc

    @property
    def sigmoid(self):
        return self.__sigmoid
