import os
from functools import partial
from typing import List

import numpy as np
import pandas as pd
import seaborn as sns
import torch
from matplotlib import pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier, _tree
from torch.nn.functional import one_hot
from torch.utils.data import TensorDataset
from xgboost import XGBClassifier
from tqdm import tqdm

from dcr.nn import ConceptReasoningLayer
from dcr.semantics import GodelTNorm
from experiments.dcr.sequential.counterfactual import counterfactual_functions, counterfactual_dcr, counterfactual_lime
from experiments.dcr.sequential.load_data import load_data
from experiments.dcr.sequential.sensitivity import sensitivity_dcr, sensitivity_functions
from lens.models import XReluNN
from lens.utils.base import ClassifierNotTrainedError
from lens.utils.metrics import RocAUC
# torch.autograd.set_detect_anomaly(True)
import warnings
warnings.filterwarnings("ignore")



def load_celeba_data(dataset, fold):
    c_emb_train = torch.as_tensor(np.load(f'../sequential/results/{dataset}/train_c_embeddings_ConceptEmbeddingModel_{fold}.npy'))[:]
    c_emb_test = torch.as_tensor(np.load(f'../sequential/results/{dataset}/test_c_embeddings_ConceptEmbeddingModel_{fold}.npy'))[:]
    c_scores_train = torch.as_tensor(np.load(f'../sequential/results/{dataset}/train_c_pred_semantics_ConceptEmbeddingModel_{fold}.npy'))[:]
    c_scores_test = torch.as_tensor(np.load(f'../sequential/results/{dataset}/test_c_pred_semantics_ConceptEmbeddingModel_{fold}.npy'))[:]
    y_train = torch.as_tensor(np.load(f'../sequential/results/{dataset}/y_train.npy'))[:]
    y_test = torch.as_tensor(np.load(f'../sequential/results/{dataset}/y_test.npy'))[:]

    valid_classes_train = y_train.unique()
    valid_classes_test = y_test.unique()
    valid_classes = torch.as_tensor([c for c in valid_classes_train if c in valid_classes_test])

    y_train_one_hot = one_hot(y_train, num_classes=228)
    valid_rows_train = y_train_one_hot[:, valid_classes].max(dim=1)[0] > 0
    y_train_final = y_train_one_hot[valid_rows_train][:, valid_classes]
    c_emb_train = c_emb_train[valid_rows_train]
    c_scores_train = c_scores_train[valid_rows_train]

    y_test_one_hot = one_hot(y_test, num_classes=228)
    valid_rows_test = y_test_one_hot[:,valid_classes].max(dim=1)[0] > 0
    y_test_final = y_test_one_hot[valid_rows_test][:, valid_classes]
    c_emb_test = c_emb_test[valid_rows_test]
    c_scores_test = c_scores_test[valid_rows_test]

    n_concepts_all = c_scores_train.shape[1]

    return c_emb_train, c_scores_train, y_train_final, c_emb_test, \
           c_scores_test, y_test_final, n_concepts_all


def load_ba_shapes_data(dataset, fold, num_classes):
    c_emb = torch.load(f'../joint/results/{dataset}/{fold}/embedding_encoding.pt')
    c_scores = torch.load(f'../joint/results/{dataset}/{fold}/concept_scores.pt')
    train_mask = torch.load(f'../joint/results/{dataset}/{fold}/train_mask.pt')
    # test_mask = torch.load(f'../joint/results/{dataset}/{fold}/test_mask.pt')
    y = torch.load(f'../joint/results/{dataset}/{fold}/y_graph.pt')
    train_mask = train_mask == 1
    test_mask = ~train_mask

    y = torch.nn.functional.one_hot(y).float()

    c_scores = torch.clamp(c_scores - (torch.rand(c_scores.shape) > 1).int(), 0, 1)

    c_emb_train = c_emb[train_mask]
    c_scores_train = c_scores[train_mask].float()
    y_train = y[train_mask]

    c_emb_test = c_emb[test_mask]
    c_scores_test = c_scores[test_mask].float()
    y_test = y[test_mask]

    n_concepts_all = c_scores_train.shape[1]

    return c_emb_train, c_scores_train, y_train, c_emb_test, c_scores_test, y_test, n_concepts_all


# def main():
random_state = 42
datasets = ['xor', 'trig', 'vec', 'mutag', 'celeba']
train_epochs = [500, 500, 500, 500, 200]
n_epochs = [3000, 3000, 3000, 7000, 3000]
temperatures = [100, 100, 100, 100, 1]

learning_rate = 0.001
logic = GodelTNorm()
folds = [i+1 for i in range(5)]
loss_form = torch.nn.CrossEntropyLoss()

# we will save all results in this directory
results_dir = f"results/dcr/"
os.makedirs(results_dir, exist_ok=True)

competitors = [
    # GridSearchCV(XGBClassifier(random_state=random_state), cv=3,
    #              param_grid={'booster': ['gbtree', 'gblinear', 'dart']}),
    # GridSearchCV(DecisionTreeClassifier(random_state=random_state), cv=3,
    #              param_grid={'max_depth': [2, 4, 10, None], 'min_samples_split': [2, 4, 10],
    #                          'min_samples_leaf': [1, 2, 5, 10]}),
    # GridSearchCV(LogisticRegression(random_state=random_state), cv=3,
    #              param_grid={'solver': ['liblinear'], 'penalty': ['l1', 'l2', 'elasticnet']}),
    partial(XReluNN, loss=loss_form),
]

model_names = {
    'DCR (ours)': "DCR",
    'DecisionTreeClassifier': "Tree",
    'LogisticRegression': "LG",
    'XGBClassifier': "XGBoost",
    'XReluNN': "ReluNN",
    'DecisionTreeClassifier (Emb.)': "Tree \n(Emb.)",
    'LogisticRegression (Emb.)': "LG \n(Emb.)",
    'XGBClassifier (Emb.)': "XGBoost \n(Emb.)",
    'XReluNN (Emb.)': "ReluNN \n(Emb.)"
}
roc = RocAUC()
wrong_uncertain_df = []
cols = ['rules', 'accuracy', 'fold', 'Model', 'dataset', "embedding"]
for dataset, train_epoch, epochs, temperature in zip(datasets, train_epochs, n_epochs, temperatures):
    if dataset != "celeba":
        continue
    print("\nDataset:", dataset)
    results = []
    local_explanations_df = pd.DataFrame()
    global_explanations_df = pd.DataFrame()
    counterfactuals_df = pd.DataFrame()
    sensitivity_df = pd.DataFrame()
    for fold in folds:
        print(f"Fold {fold}/{folds[-1]}", dataset)
        # load data
        if dataset == 'mutag':
            c_emb_train, c_scores_train, y_train, c_emb_test, \
            c_scores_test, y_test, n_concepts_all = load_ba_shapes_data(dataset, fold - 1, num_classes=2)
            n_classes = y_train.shape[1]
        elif dataset == "celeba":
            c_emb_train, c_scores_train, y_train, c_emb_test, \
            c_scores_test, y_test, n_concepts_all = load_celeba_data(dataset, fold - 1)
            n_classes = y_train.shape[1]
        else:
            c_emb_train, c_scores_train, y_train, c_emb_test, \
            c_scores_test, y_test, n_concepts_all = load_data(dataset, fold,train_epoch)
            n_classes = y_train.shape[1]
        emb_size = c_emb_train.shape[2]

        # train model
        model = ConceptReasoningLayer(emb_size, n_classes, logic, temperature)
        model_path = os.path.join(results_dir, f"{dataset}_{fold}_dcr.pt")
        try:
            raise NotImplementedError()
            model.load_state_dict(torch.load(model_path))
        except:
            optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
            loss_form = torch.nn.CrossEntropyLoss()
            model.train()
            for epoch in (pbar := tqdm(range(epochs))):
                optimizer.zero_grad()
                y_pred = model.forward(c_emb_train, c_scores_train)
                loss = loss_form(y_pred, y_train.argmax(dim=1))
                loss.backward()
                optimizer.step()

                # monitor AUC
                if epoch % 100 == 0:
                    # train_auc = roc_auc_score(y_train.detach(), y_pred.detach())
                    train_auc = roc(y_pred.detach(), y_train.detach())
                    pbar.set_description(f'Epoch {epoch}: loss {loss:.4f} train AUC: {train_auc:.4f}')
            torch.save(model.state_dict(), model_path)

        # make predictions on test set and evaluate results
        y_pred = model(c_emb_test, c_scores_test)
        # test_auc = roc_auc_score(y_test.detach(), y_pred.detach())
        test_auc = roc(y_pred.detach(), y_test.detach())
        print(f'DCR (ours): Test accuracy: {test_auc:.4f}')
        results.append(['', test_auc, fold, 'DCR (ours)', dataset, False])

        # n_non_preds_error, percentage_non_preds_error = np.nan, np.nan
        # wrong_predictions = torch.where(y_test.argmax(dim=1) != y_pred.argmax(dim=1))[0]
        # unc_preds = torch.where(y_pred.sum(dim=1) < 0.1)[0]
        # if len(wrong_predictions != 0):
        #     n_non_preds_error = np.sum([1 for unc in unc_preds if unc in wrong_predictions])
        #     percentage_non_preds_error = n_non_preds_error / len(wrong_predictions)
        # wrong_uncertain_df.append({
        #     "fold": fold,
        #     "dataset": dataset,
        #     "n_wrong_unc": n_non_preds_error,
        #     "percentage_wrong_unc": percentage_non_preds_error,
        # })

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
        counterfactuals['Model'] = 'DCR (ours)'
        counterfactuals_df = pd.concat([counterfactuals_df, counterfactuals], axis=0)

        sensitivity = sensitivity_dcr(model, c_emb_test, c_scores_test)
        sensitivity = pd.DataFrame(sensitivity)
        sensitivity['fold'] = fold
        sensitivity['dataset'] = dataset
        sensitivity['Model'] = 'DCR (ours)'
        sensitivity_df = pd.concat([sensitivity_df, sensitivity], axis=0)

        # if dataset == 'celeba':
        #     # here we simulate that concept 0 and concept 1 are not available at test time
        #     # by setting them to a default value of zero
        #     c_emb_empty = torch.zeros((c_emb_test.shape[0], c_emb_train.shape[1], c_emb_test.shape[2]))
        #     c_scores_empty = torch.zeros((c_scores_test.shape[0], c_scores_train.shape[1]))
        #     c_emb_empty[:, 1:] = c_emb_test
        #     c_scores_empty[:, 1:] = c_scores_test
        #     c_emb_test = c_emb_empty
        #     c_scores_test = c_scores_empty

        # print('\nAnd now run competitors!\n')
        for classifier in competitors:
            if isinstance(classifier, partial):
                n_concepts = c_scores_train.shape[1]
                train_set = TensorDataset(c_scores_train.float(), y_train.argmax(dim=-1).detach())
                test_set = TensorDataset(c_scores_test.float(), y_test.argmax(dim=-1).detach())

                model_path = os.path.join(results_dir, f"{dataset}_{fold}_relunn.pt")
                classifier = classifier(n_classes=n_classes, n_features=n_concepts, hidden_neurons=[emb_size,],
                                        name=model_path)
                try:
                    raise NotImplementedError()
                    classifier.load(device=torch.device("cpu"))
                except:
                    classifier.fit(train_set, test_set, epochs=1000, l_r=learning_rate, metric=RocAUC(),
                                   save=True, early_stopping=False)
                test_accuracy = classifier.evaluate(dataset=test_set, metric=RocAUC())
            else:
                classifier.fit(c_scores_train, y_train.argmax(dim=-1).detach())
                try:
                    y_pred = classifier.predict_proba(c_scores_test)
                except:
                    y_pred = classifier.predict(c_scores_test)
                # test_accuracy = roc_auc_score(y_test.detach(), y_pred)
                test_accuracy = roc(y_pred, y_test.detach())

                classifier = classifier.best_estimator_

            classifier_name = classifier.__class__.__name__
            print(f'{classifier_name}: Test accuracy: {test_accuracy:.4f}')
            results.append(['', test_accuracy, fold, classifier.__class__.__name__, dataset, False])

            counterfactuals = counterfactual_functions[classifier_name](classifier, c_scores_test)
            counterfactuals = pd.DataFrame(counterfactuals)
            counterfactuals['fold'] = fold
            counterfactuals['dataset'] = dataset
            counterfactuals['Model'] = classifier_name
            counterfactuals_df = pd.concat([counterfactuals_df, counterfactuals], axis=0)

            if classifier_name == "XGBClassifier":
                counterfactuals = counterfactual_lime(classifier, c_scores_test)
                counterfactuals = pd.DataFrame(counterfactuals)
                counterfactuals['fold'] = fold
                counterfactuals['dataset'] = dataset
                counterfactuals['Model'] = "Lime"
                counterfactuals_df = pd.concat([counterfactuals_df, counterfactuals], axis=0)

            sensitivity = sensitivity_functions[classifier_name](classifier, c_scores_test.numpy())
            sensitivity = pd.DataFrame(sensitivity)
            sensitivity['fold'] = fold
            sensitivity['dataset'] = dataset
            sensitivity['Model'] = classifier_name if classifier_name != "XGBClassifier" else "Lime"
            sensitivity_df = pd.concat([sensitivity_df, sensitivity], axis=0)

        # print('\nAnd now run competitors with embeddings!\n')
        for classifier in competitors:
            if isinstance(classifier, partial):
                emb_size_flat = c_emb_train.shape[1]*c_emb_train.shape[2]
                train_set = TensorDataset(c_emb_train.reshape(c_emb_train.shape[0], -1),
                                          y_train.argmax(dim=-1).detach())
                test_set = TensorDataset(c_emb_test.reshape(c_emb_test.shape[0], -1),
                                         y_test.argmax(dim=-1).detach())

                model_path = os.path.join(results_dir, f"{dataset}_{fold}_relunn_emb.pt")
                classifier = classifier(n_classes=n_classes, n_features=emb_size_flat,
                                        hidden_neurons=[emb_size_flat], name=model_path)
                try:
                    raise NotImplementedError()
                    classifier.load(device=torch.device("cpu"))
                except:
                    classifier.fit(train_set, train_set, epochs=1000, l_r=learning_rate, metric=RocAUC(),
                                   save=True, early_stopping=True)
                test_accuracy = classifier.evaluate(dataset=test_set, metric=RocAUC())
            else:
                classifier.fit(c_emb_train.reshape(c_emb_train.shape[0], -1), y_train.argmax(dim=-1).detach())
                try:
                    y_pred = classifier.predict_proba(c_emb_test.reshape(c_emb_test.shape[0], -1))
                except:
                    y_pred = classifier.predict(c_emb_test.reshape(c_emb_test.shape[0], -1))

                # test_accuracy = f1_score(y_test.argmax(dim=-1).detach(), y_pred, average='weighted')
                # test_accuracy = roc_auc_score(y_test.detach(), y_pred)
                test_accuracy = roc(y_pred, y_test.detach())
                classifier = classifier.best_estimator_

            classifier_name = classifier.__class__.__name__
            print(f'{classifier_name} (Emb.): Test accuracy: {test_accuracy:.4f}')
            results.append(['', test_accuracy, fold, classifier_name + ' (Emb.)', dataset, True])

    results_df = pd.DataFrame(results, columns=cols)
    results_df.to_csv(os.path.join(results_dir, f'{dataset}_accuracy.csv'))
    counterfactuals_df = counterfactuals_df.reset_index()
    counterfactuals_df.to_csv(os.path.join(results_dir, f'{dataset}_counterfactuals.csv'))
    sensitivity_df = sensitivity_df.reset_index()
    sensitivity_df.to_csv(os.path.join(results_dir, f'{dataset}_sensitivity.csv'))

    plt.figure(figsize=(8, 4))
    sns.barplot(data=results_df, y="accuracy", x="Model")
    plt.xticks([*range(9)], model_names.values())
    plt.title(f"{dataset}")
    plt.savefig(os.path.join(results_dir, f"accuracy_{dataset}"))
    plt.show()

    sns.lineplot(data=counterfactuals_df, x="iteration", y="counterfactual_preds", hue="Model")
    # plt.xticks([0,1,2], ["0", "1", "2"])
    plt.ylabel("$f(x)$")
    plt.title(f"{dataset}")
    plt.savefig(os.path.join(results_dir, f"counterfactual_{dataset}"))
    plt.show()

    sns.lineplot(data=counterfactuals_df, x="iteration", y="counterfactual_preds_norm", hue="Model")
    # plt.xticks([0,1,2], ["0", "1", "2"])
    plt.ylabel("$f(x^\star)$")
    plt.title(f"{dataset}")
    plt.savefig(os.path.join(results_dir, f"counterfactual_norm_{dataset}"))
    plt.show()

    sns.lineplot(data=sensitivity_df, x="distance", y="explanation_dist", hue="Model")
    # plt.xticks([0,1,2], ["0", "1", "2"])
    plt.ylabel("$f(x)$")
    plt.title(f"Sensitivity {dataset}")
    plt.savefig(os.path.join(results_dir, f"sensitivity_{dataset}"))
    plt.show()

    sns.lineplot(data=sensitivity_df, x="distance", y="explanation_dist_norm", hue="Model")
    # plt.xticks([0,1,2], ["0", "1", "2"])
    plt.ylabel("$f(x)$")
    plt.title(f"Normalized Sensitivity {dataset}")
    plt.savefig(os.path.join(results_dir, f"sensitivity_norm_{dataset}"))
    plt.show()


# wrong_uncertain_df = pd.DataFrame(wrong_uncertain_df)
# wrong_uncertain_df.to_csv(os.path.join(results_dir, f'wrong_uncertain.csv'))
# sns.barplot(data=wrong_uncertain_df, y="percentage_wrong_unc", x="dataset")
# plt.savefig(os.path.join(results_dir, f"wrong_uncertain"))
# plt.show()
#
# #
#
#
# if __name__ == '__main__':
#     main()
