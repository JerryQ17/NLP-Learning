from src import *

if __name__ == "__main__":
    dataset = DataSet(r'.\dataset\IMDB Dataset.csv')
    dataset.tfidf()
    dataset.to_svm()
