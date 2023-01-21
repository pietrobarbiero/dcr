import time

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
from xgboost import XGBClassifier

from causality.counterfactuals import counterfact
from dcr.semantics import GodelTNorm
import pandas as pd
import torch
import os
from dcr.nn import ConceptReasoningLayer
from experiments.dcr.sequential.load_data import load_data
# torch.autograd.set_detect_anomaly(True)


def main():
    random_state = 42
    datasets = ['cub']
    train_epochs = [300]
    n_epochs = [500]
    temperature = 1
    n_concepts = [10, 50, 100, 150]

    learning_rate = 0.001
    logic = GodelTNorm()
    folds = [i+1 for i in range(5)]

    # we will save all results in this directory
    results_dir = f"results/ablation/"
    os.makedirs(results_dir, exist_ok=True)

    results = []
    cols = ['rules', 'accuracy', 'fold', 'model', 'dataset', 'n concepts', 'eta train', 'eta test']
    for dataset, train_epoch, epochs in zip(datasets, train_epochs, n_epochs):
        for n_concept in n_concepts:
            for fold in folds:
                # load data
                c_emb_train, c_scores_train, y_train, c_emb_test, c_scores_test, y_test, n_concepts_all = load_data(dataset, fold, train_epoch)
                c_scores_train = c_scores_train[:, :n_concept]
                c_emb_train = c_emb_train[:, :n_concept]
                c_scores_test = c_scores_test[:, :n_concept]
                c_emb_test = c_emb_test[:, :n_concept]

                emb_size = c_emb_train.shape[2]
                n_classes = y_train.shape[1]

                # train model
                model = ConceptReasoningLayer(emb_size, n_classes, logic, temperature)
                optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
                loss_form = torch.nn.CrossEntropyLoss()
                model.train()
                start = time.time()
                for epoch in range(epochs):
                    optimizer.zero_grad()
                    y_pred = model.forward(c_emb_train, c_scores_train)
                    loss = loss_form(y_pred, y_train)
                    loss.backward()
                    optimizer.step()
                end = time.time()
                eta_train = end - start

                # make predictions on test set and evaluate results
                start = time.time()
                y_pred, sign_attn, filter_attn = model(c_emb_test, c_scores_test, return_attn=True)
                end = time.time()
                eta_test = end - start

                test_auc = roc_auc_score(y_test.detach(), y_pred.detach())
                print(f'Test accuracy: {test_auc:.4f}')
                results.append(['', test_auc, fold, 'DCR (ours)', dataset, n_concept, eta_train, eta_test])
                pd.DataFrame(results, columns=cols).to_csv(os.path.join(results_dir, 'time.csv'))


if __name__ == '__main__':
    main()
