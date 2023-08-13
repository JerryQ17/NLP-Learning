import time
import pickle
import logging

from src.utils import *


def tfidf_svm():
    _logger = logging.getLogger(__name__)
    start = time.time()

    dataset = IMDBDataset(r'..\dataset\IMDB Dataset.csv')
    _logger.info(f'加载数据集耗时{time.time() - start}秒')

    converter = Converter(dataset, processes=15)
    path = converter.tfidf_to_svm()
    _logger.info(f'转换为libsvm格式耗时{time.time() - start}秒')

    trainer = Trainer(tfidf_dataset=dataset, svm_train_path=path)
    _logger.info(f'初始化训练器耗时{time.time() - start}秒')

    results = trainer.svm.grid(enable_logging=True, detailed=True)
    _logger.info(f'网格搜索耗时{time.time() - start}秒')

    with open('results.pkl', 'wb') as f:
        pickle.dump([map(lambda x: x.dict(), results)], f)
    _logger.info(f'保存结果耗时{time.time() - start}秒')

    # 用results中准确率前三的参数训练模型
    for i, result in enumerate(results.sort(key=lambda x: x.accuracy, reverse=True)[:3]):
        trainer.svm.train(cost=result.c_min, gamma=result.g_min)
        _logger.info(f"训练模型{i}耗时{time.time() - start}秒")

        savepath = trainer.svm.save(fr'..\svm\data\svm_model{i}_c_{result.c_min}_g_{result.g_min}.model')
        _logger.info(f"保存模型{i}耗时{time.time() - start}秒, 保存于{savepath}")
