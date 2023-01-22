from functools import partial

from matplotlib import pyplot as plt
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.tree import DecisionTreeClassifier
from torch.utils.data import TensorDataset

from causality.counterfactuals import counterfact
from dcr.semantics import GodelTNorm
import pandas as pd
import torch
import os
from dcr.nn import ConceptReasoningLayer
from experiments.dcr.sequential.counterfactual import counterfactual_functions, counterfactual_dcr
from experiments.dcr.sequential.load_data import load_data
# torch.autograd.set_detect_anomaly(True)

from lens.models import XReluNN
from lens.utils.base import ClassifierNotTrainedError
from lens.utils.metrics import F1Score, RocAUC
import seaborn as sns


# def main():
random_state = 42
datasets = ['xor', 'trig', 'vec', 'celeba']
train_epochs = [500, 500, 500, 200]
n_epochs = [3000, 3000, 3000, 3000]
temperatures = [100, 100, 100, 100]

learning_rate = 0.001
logic = GodelTNorm()
folds = [i+1 for i in range(5)]
loss_form = torch.nn.CrossEntropyLoss()

# we will save all results in this directory
results_dir = f"results/dcr/"
os.makedirs(results_dir, exist_ok=True)

competitors = [
    LogisticRegression(random_state=random_state),
    partial(XReluNN, loss=loss_form),
    # DecisionTreeClassifier(random_state=random_state),
    # GradientBoostingClassifier(random_state=random_state),
    # XGBClassifier(),
    # RandomForestClassifier(random_state=random_state)
]

results = []
local_explanations_df = pd.DataFrame()
global_explanations_df = pd.DataFrame()
counterfactuals_df = pd.DataFrame()
cols = ['rules', 'accuracy', 'fold', 'model', 'dataset']
for dataset, train_epoch, epochs, temperature in zip(datasets, train_epochs, n_epochs, temperatures):
    if dataset !="celeba":
        continue
    for fold in folds:
        # load data
        c_emb_train, c_scores_train, y_train, c_emb_test, c_scores_test, y_test = load_data(dataset, fold, train_epoch)
        emb_size = c_emb_train.shape[2]
        n_classes = y_train.shape[1]

        # train model
        model = ConceptReasoningLayer(emb_size, n_classes, logic, temperature)
        model_path = os.path.join(results_dir, f"{dataset}_{fold}_dcr.pt")
        if os.path.exists(model_path):
            model.load_state_dict(torch.load(model_path))
        else:
            optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
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

            torch.save(model.state_dict(), model_path)

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

        counterfactuals = counterfactual_dcr(model, c_emb_test, c_scores_test)
        counterfactuals = pd.DataFrame(counterfactuals)
        counterfactuals['fold'] = fold
        counterfactuals['dataset'] = dataset
        counterfactuals['model'] = 'DCR (ours)'
        counterfactuals_df = pd.concat([counterfactuals_df, counterfactuals], axis=0)
        counterfactuals_df.to_csv(os.path.join(results_dir, 'counterfactuals.csv'))

        sns.lineplot(counterfactuals_df, x="iteration", y="counterfactual_preds", hue="model")
        plt.show()

        # counterfactuals = counterfact(model, c_emb_test, c_scores_test)
        # counterfactuals = pd.DataFrame(counterfactuals)
        # counterfactuals['fold'] = fold
        # counterfactuals['dataset'] = dataset
        # counterfactuals['model'] = 'DCR (ours)'
        # counterfactuals_df = pd.concat([counterfactuals_df, counterfactuals], axis=0)
        # counterfactuals_df.to_csv(os.path.join(results_dir, 'counterfactuals.csv'))

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
            if isinstance(classifier, partial):
                n_concepts = c_scores_train.shape[1]
                train_set = TensorDataset(c_scores_train, y_train.argmax(dim=-1).detach())
                test_set = TensorDataset(c_scores_test, y_test.argmax(dim=-1).detach())

                model_path = os.path.join(results_dir, f"{dataset}_{fold}_relunn.pt")
                classifier = classifier(n_classes=n_classes, n_features=n_concepts, hidden_neurons=[emb_size,],
                                        name=model_path)
                try:
                    classifier.load(device=torch.device("cpu"))
                except ClassifierNotTrainedError:
                    classifier.fit(train_set, train_set, epochs=1000, l_r=learning_rate, metric=RocAUC(),
                                   save=True, early_stopping=True)
                test_accuracy = classifier.evaluate(dataset=test_set, metric=RocAUC())
            else:
                classifier.fit(c_scores_train, y_train.argmax(dim=-1).detach())
                if n_classes > 1:
                    y_pred = classifier.predict_proba(c_scores_test)
                else:
                    y_pred = classifier.predict(c_scores_test)
                # test_accuracy = f1_score(y_test.argmax(dim=-1).detach(), y_pred, average='weighted')
                test_accuracy = roc_auc_score(y_test.argmax(dim=-1).detach(), y_pred, multi_class="ovr")

            classifier_name = classifier.__class__.__name__
            print(f'{classifier_name}: Test accuracy: {test_accuracy:.4f}')

            counterfactuals = counterfactual_functions[classifier_name](classifier, c_scores_test)
            counterfactuals = pd.DataFrame(counterfactuals)
            counterfactuals['fold'] = fold
            counterfactuals['dataset'] = dataset
            counterfactuals['model'] = classifier_name
            counterfactuals_df = pd.concat([counterfactuals_df, counterfactuals], axis=0)
            counterfactuals_df.to_csv(os.path.join(results_dir, 'counterfactuals.csv'))

            results.append(['', test_accuracy, fold, classifier.__class__.__name__, dataset])
        pd.DataFrame(results, columns=cols).to_csv(os.path.join(results_dir, 'accuracy.csv'))

        sns.lineplot(counterfactuals_df, x="iteration", y="counterfactual_preds", hue="model")
        plt.show()

        print('\nAnd now run competitors with embeddings!\n')
        for classifier in competitors:
            if isinstance(classifier, partial):
                emb_size_flat = c_emb_train.shape[1]*c_emb_train.shape[2]
                train_set = TensorDataset(c_emb_train.reshape(c_emb_train.shape[0], -1),
                                          y_train.argmax(dim=-1).detach())
                test_set = TensorDataset(c_emb_test.reshape(c_emb_test.shape[0], -1),
                                         y_test.argmax(dim=-1).detach())

                model_path = os.path.join(results_dir, f"{dataset}_{fold}_relunn.pt")
                classifier = classifier(n_classes=n_classes, n_features=emb_size_flat,
                                        hidden_neurons=[emb_size_flat], name=model_path)
                try:
                    classifier.load(device=torch.device("cpu"))
                except ClassifierNotTrainedError:
                    classifier.fit(train_set, train_set, epochs=1000, l_r=learning_rate, metric=RocAUC(),
                                   save=True, early_stopping=True)
                test_accuracy = classifier.evaluate(dataset=test_set, metric=RocAUC())
            else:
                classifier.fit(c_emb_train.reshape(c_emb_train.shape[0], -1), y_train.argmax(dim=-1).detach())
                if n_classes > 1:
                    y_pred = classifier.predict_proba(c_emb_test.reshape(c_emb_test.shape[0], -1))
                else:
                    y_pred = classifier.predict(c_emb_test.reshape(c_emb_test.shape[0], -1))

                # test_accuracy = f1_score(y_test.argmax(dim=-1).detach(), y_pred, average='weighted')
                test_accuracy = roc_auc_score(y_test.argmax(dim=-1).detach(), y_pred, multi_class="ovr")
            print(f'{classifier.__class__.__name__}: Test accuracy: {test_accuracy:.4f}')

            results.append(['', test_accuracy, fold, classifier.__class__.__name__ + ' (Emb.)', dataset])
        pd.DataFrame(results, columns=cols).to_csv(os.path.join(results_dir, 'accuracy.csv'))

#
#
#
# if __name__ == '__main__':
#     main()
