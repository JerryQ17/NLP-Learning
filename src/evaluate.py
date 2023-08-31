import torch
from torch import nn, optim
from torch.utils.data import DataLoader

from src.utils import *

if __name__ == '__main__':
    dataset = IMDBDataset(r'..\dataset\IMDB Dataset.csv')
    converter = Converter(dataset, processes=10)

    # path = converter.tfidf_to_svm(r'..\svm\data\test.txt')
    # svm = SVM()
    # svm.load(model_path=r'..\svm\model\svm_model0_c_1.0_g_1.0.model')
    # svm_predict_result = svm.predict(path)

    # p_labels: a list of predicted labels
    # p_acc: a tuple including accuracy (for classification), mean-squared error, and squared correlation coefficient (for regression).
    # p_vals: a list of decision values or probability estimates (if '-b 1' is specified).
    # If k is the number of classes, for decision values, each element includes results of predicting k(k-1)/2 binary-class SVMs.
    # For probabilities, each element contains k values indicating the probability that the testing instance is in each class. Note that the order of classes here is the same as 'model.label' field in the model structure.
    model = TextClassifier(
        input_size=101895,
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
                      autosave=False, autosave_dir=r'..\autosave')
    trainer.load(r'..\lstm\model\new_train_lstm.pth')
    nn_predict_result = trainer.evaluate(DataLoader(converter.tfidf_dataset, batch_size=64, num_workers=5))
