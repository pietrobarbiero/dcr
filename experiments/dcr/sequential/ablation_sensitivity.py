import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
from xgboost import XGBClassifier

from lime import lime_tabular
from causality.counterfactuals import counterfact
from dcr.semantics import GodelTNorm
import pandas as pd
import torch
import os
from dcr.nn import ConceptReasoningLayer
from experiments.dcr.sequential.load_data import load_data
# torch.autograd.set_detect_anomaly(True)

def sensitivity(e1, e2):
    return torch.norm(e1 - e2, p=1) / len(e1)

def sensitivity_predictions(p1, p2):
    return torch.sum(p1 != p2) / len(p1)


def perturb_concepts(c_emb_test, c_scores_test, radius):
    # generate random perturbations
    perturbation = radius * torch.randn_like(c_scores_test)
    c_scores_test_pertubed = torch.clamp(perturbation + c_scores_test, min=0, max=1)
    c_emb_test_perturbed = c_emb_test
    return c_emb_test_perturbed, c_scores_test_pertubed


def get_explanations_dcr(model, c_emb_test, c_scores_test, n_classes, logic):
    # make predictions
    y_pred, sign_attn, filter_attn = model(c_emb_test, c_scores_test, return_attn=True)
    pred_class = y_pred.argmax(dim=1)

    # and find new explanations
    values = c_scores_test.unsqueeze(-1).repeat(1, 1, n_classes)
    sign_terms = logic.iff_pair(sign_attn, values)
    filtered_values = logic.disj_pair(sign_terms, logic.neg(filter_attn))
    explanations_perturbed = torch.vstack([filtered_values[j, :, i] for j, i in enumerate(pred_class)])
    return pred_class, explanations_perturbed


def train_dcr(model, epochs, optimizer, c_emb_train, c_scores_train, loss_form, y_train):
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
    return model

def main():
    random_state = 42
    datasets = ['xor', 'trig', 'vec']
    train_epochs = [500, 500, 500]
    n_epochs = [3000, 3000, 3000]

    loss_form = torch.nn.CrossEntropyLoss()
    perturbation_radius = torch.arange(0, 1, 0.1)
    learning_rate = 0.001
    logic = GodelTNorm()
    folds = [i+1 for i in range(5)]

    # we will save all results in this directory
    results_dir = f"results/ablation/"
    os.makedirs(results_dir, exist_ok=True)

    competitors = [
        DecisionTreeClassifier(random_state=random_state),
        LogisticRegression(random_state=random_state),
    ]

    results = []
    cols = ['rules', 'accuracy', 'fold', 'model', 'dataset', 'radius', 'explanation sensitivity', 'prediction sensitivity']
    for dataset, train_epoch, epochs in zip(datasets, train_epochs, n_epochs):
        for radius in perturbation_radius:
            for fold in folds:
                # load data
                c_emb_train, c_scores_train, y_train, c_emb_test, c_scores_test, y_test, n_concepts_all = load_data(dataset, fold, train_epoch)
                emb_size = c_emb_train.shape[2]
                n_classes = y_train.shape[1]

                # train model
                model = ConceptReasoningLayer(emb_size, n_classes, logic, temperature=100)
                optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
                model = train_dcr(model, epochs, optimizer, c_emb_train, c_scores_train, loss_form, y_train)
                # get explanations
                pred_class, explanations = get_explanations_dcr(model, c_emb_test, c_scores_test, n_classes, logic)

                # perturb concepts
                c_emb_train_perturbed, c_scores_train_perturbed = perturb_concepts(c_emb_train, c_scores_train, radius)
                # retrain model on perturbed concepts
                model = ConceptReasoningLayer(emb_size, n_classes, logic, temperature=100)
                optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
                model = train_dcr(model, epochs, optimizer, c_emb_train_perturbed, c_scores_train_perturbed, loss_form, y_train)
                # get perturbed explanations
                pred_class_perturbed, explanations_perturbed = get_explanations_dcr(model, c_emb_test, c_scores_test, n_classes, logic)

                # compute sensitivity
                explanation_distance = sensitivity(explanations, explanations_perturbed)
                prediction_distance = sensitivity_predictions(pred_class, pred_class_perturbed)
                results.append(['', None, fold, 'DCR (ours)', dataset, radius.item(), explanation_distance.item(), prediction_distance.item()])
                pd.DataFrame(results, columns=cols).to_csv(os.path.join(results_dir, 'sensitivity.csv'))

                # repeat for competitors
                for classifier in competitors:
                    # get explanations
                    classifier.fit(c_scores_train, y_train.argmax(dim=-1).detach())
                    pred_class = classifier.predict(c_scores_test)
                    try:
                        explanations = classifier.coef_
                    except:
                        explanations = classifier.feature_importances_

                    # get perturbed explanations
                    classifier.fit(c_scores_train_perturbed, y_train.argmax(dim=-1).detach())
                    pred_class_perturbed = classifier.predict(c_scores_test)
                    try:
                        explanations_perturbed = classifier.coef_
                    except:
                        explanations_perturbed = classifier.feature_importances_

                    # compute sensitivity
                    explanation_distance = sensitivity(torch.FloatTensor(explanations), torch.FloatTensor(explanations_perturbed))
                    prediction_distance = sensitivity_predictions(torch.FloatTensor(pred_class), torch.FloatTensor(pred_class_perturbed))
                    results.append(['', None, fold, f'{classifier.__class__.__name__}', dataset, radius.item(), explanation_distance.item(), prediction_distance.item()])
                    pd.DataFrame(results, columns=cols).to_csv(os.path.join(results_dir, 'sensitivity.csv'))


if __name__ == '__main__':
    main()
