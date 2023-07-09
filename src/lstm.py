import torch
from torch import nn


class LSTMModel(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, num_layers: int, output_dim: int, device: torch.device):
        super().__init__()
        self.__device: torch.device = device
        self.__input_dim: int = input_dim
        self.__hidden_dim: int = hidden_dim
        self.__num_layers: int = num_layers
        self.__output_dim: int = output_dim

        # LSTM层
        self.__lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, device=self.__device)
        # 全连接层
        self.__fc = nn.Linear(hidden_dim, output_dim, device=self.__device)
        # softmax层
        self.__softmax = nn.Softmax(dim=1)

    def forward(self, input_tensor: torch.Tensor):
        lstm_out, _ = self.__lstm(input_tensor.view(len(input_tensor), 1, -1))
        fc_out = self.__fc(lstm_out.view(len(input_tensor), -1))
        softmax_out = self.__softmax(fc_out)
        return softmax_out

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
    def fc(self):
        return self.__fc

    @property
    def softmax(self):
        return self.__softmax
