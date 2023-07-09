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
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, device=self.__device)

        # 全连接层
        self.fc = nn.Linear(hidden_dim, output_dim, device=self.__device)

    def forward(self, x):
        lstm_out, _ = self.lstm(x.view(len(x), 1, -1))
        out = self.fc(lstm_out.view(len(x), -1))
        return out

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
