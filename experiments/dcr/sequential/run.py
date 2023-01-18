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
    datasets = ['xor', 'trig', 'vec', 'celeba']
    train_epochs = [500, 500, 500, 200]
    n_epochs = [3000, 3000, 3000, 3000]
    temperatures = [100, 100, 100, 100]

    learning_rate = 0.001
    logic = GodelTNorm()
    folds = [i+1 for i in range(5)]

    # we will save all results in this directory
    results_dir = f"results/dcr/"
    os.makedirs(results_dir, exist_ok=True)

    competitors = [
        GridSearchCV(DecisionTreeClassifier(random_state=random_state), cv=3, param_grid={'max_depth': [2, 4, 10, None], 'min_samples_split': [2, 4, 10], 'min_samples_leaf': [1, 2, 5, 10]}),
        GridSearchCV(LogisticRegression(random_state=random_state), cv=3, param_grid={'solver': ['lbfgs', 'saga'], 'penalty': ['l1', 'l2', 'elasticnet']}),
        GridSearchCV(XGBClassifier(random_state=random_state), cv=3, param_grid={'booster': ['gbtree', 'gblinear', 'dart']}),
    ]

    results = []
    local_explanations_df = pd.DataFrame()
    global_explanations_df = pd.DataFrame()
    counterfactuals_df = pd.DataFrame()
    cols = ['rules', 'accuracy', 'fold', 'model', 'dataset']
    for dataset, train_epoch, epochs, temperature in zip(datasets, train_epochs, n_epochs, temperatures):
        for fold in folds:
            # load data
            c_emb_train, c_scores_train, y_train, c_emb_test, c_scores_test, y_test = load_data(dataset, fold, train_epoch)
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
            y_pred = model(c_emb_test, c_scores_test)
            test_auc = roc_auc_score(y_test.detach(), y_pred.detach())
            print(f'Test accuracy: {test_auc:.4f}')
            results.append(['', test_auc, fold, 'DCR (ours)', dataset])
            pd.DataFrame(results, columns=cols).to_csv(os.path.join(results_dir, 'accuracy.csv'))

            local_explanations = model.explain(c_emb_test, c_scores_test, 'local')
            local_explanations = pd.DataFrame(local_explanations)
            local_explanations['fold'] = fold
            local_explanations['dataset'] = dataset
            local_explanations_df = pd.concat([local_explanations_df, local_explanations], axis=0)
            local_explanations_df.to_csv(os.path.join(results_dir, 'local_explanations.csv'))

            global_explanations = model.explain(c_emb_test, c_scores_test, 'global')
            global_explanations = pd.DataFrame(global_explanations)
            global_explanations['fold'] = fold
            global_explanations['dataset'] = dataset
            global_explanations_df = pd.concat([global_explanations_df, global_explanations], axis=0)
            global_explanations_df.to_csv(os.path.join(results_dir, 'global_explanations.csv'))

            counterfactuals = counterfact(model, c_emb_test, c_scores_test)
            counterfactuals = pd.DataFrame(counterfactuals)
            counterfactuals['fold'] = fold
            counterfactuals['dataset'] = dataset
            counterfactuals['model'] = 'DCR (ours)'
            counterfactuals_df = pd.concat([counterfactuals_df, counterfactuals], axis=0)
            counterfactuals_df.to_csv(os.path.join(results_dir, 'counterfactuals.csv'))

            if dataset == 'celeba':
                # here we simulate that concept 0 and concept 1 are not available at test time
                # by setting them to a default value of zero
                c_emb_empty = torch.zeros((c_emb_test.shape[0], c_emb_train.shape[1], c_emb_test.shape[2]))
                c_scores_empty = torch.zeros((c_scores_test.shape[0], c_scores_train.shape[1]))
                c_emb_empty[:, 1:] = c_emb_test
                c_scores_empty[:, 1:] = c_scores_test
                c_emb_test = c_emb_empty
                c_scores_test = c_scores_empty

            print('\nAnd now run competitors!\n')
            for classifier in competitors:
                classifier.fit(c_scores_train, y_train.argmax(dim=-1).detach())
                y_pred = classifier.predict(c_scores_test)
                test_accuracy = f1_score(y_test.argmax(dim=-1).detach(), y_pred, average='weighted')
                print(f'{classifier.best_estimator_.__class__.__name__}: Test accuracy: {test_accuracy:.4f}')

                results.append(['', test_accuracy, fold, classifier.best_estimator_.__class__.__name__, dataset])
                pd.DataFrame(results, columns=cols).to_csv(os.path.join(results_dir, 'accuracy.csv'))

            print('\nAnd now run competitors with embeddings!\n')
            for classifier in competitors:
                classifier.fit(c_emb_train.reshape(c_emb_train.shape[0], -1), y_train.argmax(dim=-1).detach())
                y_pred = classifier.predict(c_emb_test.reshape(c_emb_test.shape[0], -1))
                test_accuracy = f1_score(y_test.argmax(dim=-1).detach(), y_pred, average='weighted')
                print(f'{classifier.best_estimator_.__class__.__name__}: Test accuracy: {test_accuracy:.4f}')

                results.append(['', test_accuracy, fold, classifier.best_estimator_.__class__.__name__ + ' (Emb.)', dataset])
                pd.DataFrame(results, columns=cols).to_csv(os.path.join(results_dir, 'accuracy.csv'))


if __name__ == '__main__':
    main()
