import time
import pickle
from src import *

if __name__ == "__main__":
    start = time.time()

    dataset = IMDBDataset(r'.\dataset\IMDB Dataset.csv')
    print(f'加载数据集耗时{time.time() - start}秒')

    converter = Converter(dataset, processes=8)
    path = converter.to_svm()
    print(f'转换为libsvm格式耗时{time.time() - start}秒')

    trainer = Trainer(tfidf_dataset=dataset, svm_train_path=path)
    print(f'初始化训练器耗时{time.time() - start}秒')

    results = trainer.svm.grid(problem_path=path, enable_logging=True, detailed=True)
    print(f'网格搜索耗时{time.time() - start}秒')

    with open('results.pkl', 'wb') as f:
        pickle.dump(results, f)
    print(f'保存结果耗时{time.time() - start}秒')

    # 用results中准确率前三的参数训练模型
    for i, result in enumerate(results.sort(key=lambda x: x.rate, reverse=True)[:3]):
        trainer.svm.train(problem_path=path, cost=result.c_min, gamma=result.g_min)
        print(f"训练模型{i}耗时{time.time() - start}秒, 保存于",
              trainer.svm.save(fr'.\svm\data\svm_model{i}_c_{result.c_min}_g_{result.g_min}.model'))
