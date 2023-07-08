import time
from src import *

if __name__ == "__main__":
    start = time.time()
    dataset = IMDBDataset(r'.\dataset\IMDB Dataset.csv')
    print(f'加载数据集耗时{time.time() - start}秒')
    converter = DataConverter(dataset, processes=10)
    converter.tfidf()
    print(f'计算TF-IDF耗时{time.time() - start}秒')
    path = converter.to_svm(save_path=r'.\svm\train\tfidf.txt')
    print(f'转换为libsvm格式耗时{time.time() - start}秒')
    results = SVM.grid(path, enable_logging=True, detailed=True, img_name=r'.\svm\train\tfidf_svm.png')
    print(f'筛选参数耗时{time.time() - start}秒')
