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
    temperatures = [0.1, 0.2, 0.5, 0.8, 1, 2, 10]

    learning_rate = 0.001
    logic = GodelTNorm()
    folds = [i+1 for i in range(5)]

    # we will save all results in this directory
    results_dir = f"results/ablation/"
    os.makedirs(results_dir, exist_ok=True)

    results = []
    cols = ['rules', 'accuracy', 'fold', 'model', 'dataset', 'temperature', 'relevant concepts']
    for dataset, train_epoch, epochs in zip(datasets, train_epochs, n_epochs):
        for temperature in temperatures:
            for fold in folds:
                # load data
                c_emb_train, c_scores_train, y_train, c_emb_test, c_scores_test, y_test, n_concepts_all = load_data(dataset, fold, train_epoch)
                c_scores_train = c_scores_train[:, :50]
                c_emb_train = c_emb_train[:, :50]
                c_scores_test = c_scores_test[:, :50]
                c_emb_test = c_emb_test[:, :50]

                emb_size = c_emb_train.shape[2]
                n_classes = y_train.shape[1]

                # train model
                model = ConceptReasoningLayer(emb_size, n_classes, logic, temperature)
                optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
                loss_form = torch.nn.CrossEntropyLoss()
                model.train()
                for epoch in range(epochs):
                    optimizer.zero_grad()
                    y_pred = model.forward(c_emb_train, c_scores_train)
                    loss = loss_form(y_pred, y_train)
                    loss.backward()
                    optimizer.step()

                    # monitor AUC
                    if epoch % 100 == 0:
                        train_auc = roc_auc_score(y_train.detach(), y_pred.detach())
                        print(f'Epoch {epoch}: loss {loss:.4f} train AUC: {train_auc:.4f}')

                # make predictions on test set and evaluate results
                y_pred, sign_attn, filter_attn = model(c_emb_test, c_scores_test, return_attn=True)

                # compute relevant concept ratio
                values = c_scores_test.unsqueeze(-1).repeat(1, 1, n_classes)
                sign_terms = logic.iff_pair(sign_attn, values)
                relevant_concepts = (logic.neg(filter_attn) < sign_terms).float()
                avg_relevant_concepts = relevant_concepts.mean()

                test_auc = roc_auc_score(y_test.detach(), y_pred.detach())
                print(f'Test accuracy: {test_auc:.4f}')
                results.append(['', test_auc, fold, 'DCR (ours)', dataset, temperature, avg_relevant_concepts.item()])
                pd.DataFrame(results, columns=cols).to_csv(os.path.join(results_dir, 'temperature.csv'))


if __name__ == '__main__':
    main()
