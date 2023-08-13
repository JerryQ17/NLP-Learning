import time
import torch
import logging
from torch import nn, optim

from src.utils import *


def tfidf_lstm():
    _logger = logging.getLogger(__name__)
    start = time.time()

    dataset = IMDBDataset(r'..\dataset\IMDB Dataset.csv')
    _logger.info(f'加载数据集耗时{time.time() - start}秒')

    converter = Converter(dataset, processes=15)
    _logger.info(f'初始化转换器耗时{time.time() - start}秒')

    model = LSTMModel(
        input_dim=converter.tfidf_matrix.shape[1],
        hidden_dim=128,
        output_dim=2,
        num_layers=1,
        fc=None,
        dropout_rate=0.5,
        device=torch.device('cuda')
    )
    trainer = Trainer(tfidf_dataset=dataset, device=torch.device('cuda'), model=model,
                      optimizer=optim.Adam(model.parameters(), lr=0.001), criterion=nn.CrossEntropyLoss())
    _logger.info(f'初始化训练器耗时{time.time() - start}秒')

    _logger.info(trainer.train(10, batch_size=64, shuffle=True, tfidf_mode=True, enable_logging=True).save())
    _logger.info(f'训练耗时{time.time() - start}秒')
