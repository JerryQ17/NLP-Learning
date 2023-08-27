import time
import torch
import logging
from torch import nn, optim
from torch.utils.data import DataLoader

from src.utils import *


def word2vec_lstm():
    _logger = logging.getLogger(__name__)
    start = time.time()

    src_dataset = IMDBDataset(r'.\dataset\IMDB Dataset.csv')
    train_dataset, eval_dataset = src_dataset.split(0.96, shuffle=True)
    _logger.info(f'加载数据集耗时{time.time() - start}秒')

    train_converter = Converter(train_dataset, processes=10)
    eval_converter = Converter(eval_dataset, processes=10)
    vector_size = 512
    train_converter.word2vec(vector_size=vector_size)
    eval_converter.word2vec(vector_size=vector_size)
    _logger.info(f'计算词向量耗时{time.time() - start}秒')

    model = LSTMModel(
        input_size=vector_size,
        hidden_size=256,
        num_layers=1,
        device=torch.device('cuda'),
        fc=nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 2)
        )
    )
    trainer = Trainer(device=torch.device('cuda'), model=model,
                      optimizer=optim.Adam(model.parameters(), lr=0.001),
                      criterion=nn.CrossEntropyLoss(), autosave=False)
    _logger.info(f'初始化训练器耗时{time.time() - start}秒')

    trainer.early_stopping(
        DataLoader(train_converter.word2vec_dataset,
                   batch_size=64, shuffle=True, num_workers=9, persistent_workers=True),
        DataLoader(eval_converter.word2vec_dataset,
                   batch_size=64, shuffle=True, num_workers=1, persistent_workers=True),
        draw=True, patience=8
    )
    _logger.info(f'训练耗时{time.time() - start}秒')

    savepath = trainer.save(r".\lstm\model\test_w2v_lstm.pth")
    _logger.info(f'保存模型耗时{time.time() - start}秒, 保存于{savepath}')

    return trainer
