import time
import torch
import logging
from torch import nn, optim
from torch.utils.data import DataLoader

from src.utils import *


def ds_info(ds: Word2VecDataset):  # ddl要到了了，应该写在类里面的
    pos = 0
    neg = 0
    for data in ds:
        if data[1] == 0:
            neg += 1
        else:
            pos += 1
    return pos, neg


def word2vec_lstm():
    _logger = logging.getLogger(__name__)
    start = time.time()

    src_dataset = IMDBDataset(r'.\dataset\IMDB Dataset.csv')
    _logger.info(f'加载数据集耗时{time.time() - start}秒')

    converter = Converter(src_dataset, processes=10)
    _logger.info(f'初始化转换器耗时{time.time() - start}秒')

    middle_wv_ds, test_wv_ds = converter.word2vec_dataset.split(0.8, shuffle=True)
    train_wv_ds, eval_wv_ds = middle_wv_ds.split(7 / 8, shuffle=True)

    _logger.info(f'计算词向量耗时{time.time() - start}秒')
    _logger.info(f'训练集(positive, negative): {ds_info(train_wv_ds)}')
    _logger.info(f'验证集(positive, negative): {ds_info(eval_wv_ds)}')
    _logger.info(f'测试集(positive, negative): {ds_info(test_wv_ds)}')

    model_list = [
        TextClassifier(input_size=100, hidden_size=100, num_layers=1, fc=nn.Linear(100, 2)),
        TextClassifier(input_size=100, hidden_size=100, num_layers=1, fc=nn.Sequential(
            nn.Linear(100, 50),
            nn.ReLU(),
            nn.Linear(50, 2)
        )),
        TextClassifier(input_size=100, hidden_size=144, num_layers=1, fc=nn.Linear(144, 2)),
        TextClassifier(input_size=100, hidden_size=144, num_layers=1, fc=nn.Sequential(
            nn.Linear(144, 72),
            nn.ReLU(),
            nn.Linear(72, 2)
        )),
        TextClassifier(input_size=100, hidden_size=144, num_layers=1, fc=nn.Sequential(
            nn.Linear(144, 100),
            nn.ReLU(),
            nn.Linear(100, 2)
        )),
        TextClassifier(input_size=100, hidden_size=144, num_layers=2, fc=nn.Sequential(
            nn.Linear(144, 100),
            nn.ReLU(),
            nn.Linear(100, 64),
            nn.ReLU(),
            nn.Linear(64, 2)
        )),
        TextClassifier(input_size=100, hidden_size=144, num_layers=1, fc=nn.Sequential(
            SelfAttention(144),
            nn.Linear(144, 2)
        )),
        TextClassifier(input_size=100, hidden_size=144, num_layers=2, fc=nn.Sequential(
            SelfAttention(144),
            nn.Linear(144, 100),
            nn.ReLU(),
            nn.Linear(100, 64),
            nn.ReLU(),
            nn.Linear(64, 2)
        )),
    ]

    trainer = Trainer(device=torch.device('cuda'), criterion=nn.CrossEntropyLoss())
    train_dl = DataLoader(train_wv_ds, batch_size=64, shuffle=True, num_workers=2, persistent_workers=True)
    eval_dl = DataLoader(eval_wv_ds, batch_size=64, shuffle=True)
    test_dl = DataLoader(test_wv_ds, batch_size=64, shuffle=True, num_workers=1, persistent_workers=True)
    _logger.info(f'初始化训练器耗时{time.time() - start}秒')

    for i, model in enumerate(model_list):
        trainer.model = model
        trainer.optimizer = optim.Adam(model.parameters(), lr=0.001)

        trainer.early_stopping(train_dl, eval_dl, draw=True, savepath=rf'.\assets\w2v_model{i}.png')
        _logger.info(f'训练模型{i}耗时{time.time() - start}秒')

        savepath = trainer.save(fr".\lstm\model\w2v_model{i}.pth")
        _logger.info(f'保存模型{i}耗时{time.time() - start}秒, 保存于{savepath}')

        acc = trainer.evaluate(test_dl)
        _logger.info(f'评估模型{i}耗时{time.time() - start}秒, 准确率为{acc}%')

    return trainer
