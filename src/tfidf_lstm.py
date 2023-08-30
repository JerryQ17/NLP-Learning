import time
import torch
import logging
from torch import nn, optim

from src.utils import *


def tfidf_lstm():
    _logger = logging.getLogger(__name__)
    start = time.time()

    dataset = IMDBDataset(r'.\dataset\IMDB Dataset.csv')
    _logger.info(f'加载数据集耗时{time.time() - start}秒')

    converter = Converter(dataset, processes=10)
    _logger.info(f'初始化转换器耗时{time.time() - start}秒')

    model = TextClassifier(
        input_size=converter.tfidf_matrix.shape[1],
        hidden_size=256,
        output_size=2,
        num_layers=1,
        fc=nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 2)
        )
    )
    trainer = Trainer(tfidf_dataset=converter.tfidf_dataset, device=torch.device('cuda'), model=model,
                      optimizer=optim.Adam(model.parameters(), lr=0.001), criterion=nn.CrossEntropyLoss(),
                      autosave=False)
    _logger.info(f'初始化训练器耗时{time.time() - start}秒')

    trainer.train(1, batch_size=64, shuffle=True, tfidf_mode=True, draw=True)
    _logger.info(f'训练耗时{time.time() - start}秒')

    # savepath = trainer.save(r".\lstm\model\new_train_lstm.pth")
    # _logger.info(f'保存模型耗时{time.time() - start}秒, 保存于{savepath}')

    return trainer
