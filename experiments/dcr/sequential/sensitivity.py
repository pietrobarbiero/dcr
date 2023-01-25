# In this file we perform the sensitivity experiments.
# We compare the the sensitivity explanations provided by:
#       - CBM (logistic regression over embedding score)
#       - ReluNN
#       - DCR
#       - MLP with Anchors explanations?

# In terms of L(x,x′,y′,λ)=λ⋅(^f(x′)−y′)2+d(x,x′),  d(x,x′)=∑^p_{j=1}(|x_j−x′_j|)/MAD_j
import copy

import numpy as np
import pandas as pd
import sympy
import torch
from lime.lime_tabular import LimeTabularExplainer
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier, _tree
from xgboost import XGBClassifier

from dcr.nn import ConceptReasoningLayer
from lens.logic import test_explanation
from lens.models import XReluNN
from lens.utils.base import tree_to_formula, collect_parameters
from lens.utils.relu_nn import get_reduced_model


def sensitivity_dcr(model: ConceptReasoningLayer, c_emb: torch.Tensor, c_score: torch.Tensor, eps=[0.1, 0.2, 0.3, 0.4, 0.5]):
    df = {
        "sensitivity_samples": [],
        "explanation": [],
        "explanation_dist": [],
        "explanation_dist_norm": [],
        "sample_id": [],
        "distance": [],
    }

    explanations = model.explain(c_emb, c_score, 'local')
    explanations = pd.DataFrame(explanations)['attention'].tolist()
    for i, (x, c, explanation) in enumerate(zip(c_emb, c_score, explanations)):
        if len(explanation) != 0:
            explanation = torch.as_tensor(explanation)
            orig_class = model(x.unsqueeze(0), c.unsqueeze(0))[0].argmax()
            df["sensitivity_samples"].append(copy.deepcopy(c))
            df["explanation"].append(explanation)
            df["explanation_dist"].append(0)
            df["explanation_dist_norm"].append(0)
            df["sample_id"].append(i)
            df["distance"].append(0)

            for d in eps:
                while True:
                    new_c = perturb_input(c.numpy(), d)
                    new_c = torch.as_tensor(new_c)
                    new_class = model(x.unsqueeze(0), new_c.unsqueeze(0))[0].argmax()
                    if new_class == orig_class:
                        break

                new_exp = model.explain(x.unsqueeze(0), new_c.unsqueeze(0), 'local')
                new_exp = pd.DataFrame(new_exp)['attention'].tolist()
                new_exp = torch.as_tensor(new_exp)
                if len(new_exp.squeeze().shape) > 0:
                    if new_exp.squeeze().shape[0] == 0:
                        new_exp = torch.zeros_like(explanation)

                exp_dist = (explanation - new_exp).abs().sum().item()
                exp_dist_norm = exp_dist/torch.norm(explanation)

                df["sensitivity_samples"].append(copy.deepcopy(new_c))
                df["explanation"].append(new_exp)
                df["explanation_dist"].append(exp_dist)
                df["explanation_dist_norm"].append(exp_dist_norm.item())
                df["sample_id"].append(i)
                df["distance"].append(d)

    return df


def sensitivity_xrelu(model: XReluNN, x: np.ndarray, eps=[0.1, 0.2, 0.3, 0.4, 0.5]):
    x = torch.as_tensor(x)
    df = {
        "sensitivity_samples": [],
        "explanation": [],
        "explanation_dist": [],
        "explanation_dist_norm": [],
        "sample_id": [],
        "distance": [],
    }
    for i, sample in enumerate(x):
        orig_class = model(sample.unsqueeze(0))[0].argmax()
        reduced_model = get_reduced_model(model.model, sample.squeeze())
        explanation = collect_parameters(reduced_model)[0][0]
        explanation = explanation[orig_class]

        df["sensitivity_samples"].append(copy.deepcopy(sample))
        df["explanation"].append(explanation)
        df["explanation_dist"].append(0)
        df["explanation_dist_norm"].append(0)
        df["sample_id"].append(i)
        df["distance"].append(0)

        for d in eps:
            while True:
                new_sample = perturb_input(sample, d)
                new_sample = torch.as_tensor(new_sample, dtype=torch.float)
                new_class = model(new_sample.unsqueeze(0))[0].argmax()
                if new_class == orig_class:
                    break

            reduced_model = get_reduced_model(model.model, new_sample.squeeze())
            new_exp = collect_parameters(reduced_model)[0][0]
            new_exp = new_exp[orig_class]
            exp_dist = np.abs(explanation - new_exp).sum().item()
            exp_dist_norm = exp_dist / np.linalg.norm(explanation)

            df["sensitivity_samples"].append(copy.deepcopy(new_sample))
            df["explanation"].append(new_exp)
            df["explanation_dist"].append(exp_dist)
            df["explanation_dist_norm"].append(exp_dist_norm.item())
            df["sample_id"].append(i)
            df["distance"].append(d)

    return df


def sensitivity_logistic(model: LogisticRegression, x: np.ndarray, eps=[0.1, 0.2, 0.3, 0.4, 0.5]):
    df = {
        "sensitivity_samples": [],
        "explanation": [],
        "explanation_dist": [],
        "explanation_dist_norm": [],
        "sample_id": [],
        "distance": [],
    }
    explanations = model.coef_
    if len(explanations) == 1:
        explanations = [- explanations[0], explanations[0]]
    for i, sample in enumerate(x):
        orig_class = model.predict(np.expand_dims(sample,0))[0]
        explanation = explanations[orig_class]

        df["sensitivity_samples"].append(copy.deepcopy(sample))
        df["explanation"].append(explanation)
        df["explanation_dist"].append(0)
        df["explanation_dist_norm"].append(0)
        df["sample_id"].append(i)
        df["distance"].append(0)

        for d in eps:
            while True:
                new_sample = perturb_input(sample, d)
                new_class = model.predict(np.expand_dims(new_sample,0))[0]
                if new_class == orig_class:
                    break

            new_exp = explanations[new_class]
            exp_dist = np.abs(explanation - new_exp).sum().item()
            exp_dist_norm = exp_dist / np.linalg.norm(explanation)

            df["sensitivity_samples"].append(copy.deepcopy(new_sample))
            df["explanation"].append(new_exp)
            df["explanation_dist"].append(exp_dist)
            df["explanation_dist_norm"].append(exp_dist_norm.item())
            df["sample_id"].append(i)
            df["distance"].append(d)

    return df


def sensitivity_tree(model: DecisionTreeClassifier, x: np.ndarray, eps=[0.1, 0.2, 0.3, 0.4, 0.5]):
    n_features = x.shape[1]
    concept_names = [f"c_{i:2d}" for i in range(n_features)]
    df = {
        "sensitivity_samples": [],
        "explanation": [],
        "explanation_dist": [],
        "explanation_dist_norm": [],
        "sample_id": [],
        "distance": [],
    }
    for i, sample in enumerate(x):
        orig_class = model.predict(np.expand_dims(sample,0))[0]
        explanation = find_tree_local_explanation(model, sample, concept_names, orig_class)

        df["sensitivity_samples"].append(copy.deepcopy(sample))
        df["explanation"].append(explanation)
        df["explanation_dist"].append(0)
        df["explanation_dist_norm"].append(0)
        df["sample_id"].append(i)
        df["distance"].append(0)

        for d in eps:
            while True:
                new_sample = perturb_input(sample, d)
                new_class = model.predict(np.expand_dims(new_sample,0))[0]
                if new_class == orig_class:
                    break

            new_exp = find_tree_local_explanation(model, new_sample, concept_names, orig_class)

            exp_dist, exp_dist_norm = logic_exp_distance(explanation, new_exp)

            df["sensitivity_samples"].append(copy.deepcopy(new_sample))
            df["explanation"].append(new_exp)
            df["explanation_dist"].append(exp_dist)
            df["explanation_dist_norm"].append(exp_dist_norm)
            df["sample_id"].append(i)
            df["distance"].append(d)

    return df


def find_tree_local_explanation(model, sample, concept_names, orig_class):
    sample = torch.as_tensor(sample)
    global_explanation = tree_to_formula(model, concept_names, orig_class)
    orig_class = torch.as_tensor(orig_class).unsqueeze(0)
    explanation = ""
    for exp in global_explanation.split(" | "):
        exp_accuracy = test_explanation(exp, orig_class, sample.unsqueeze(0), orig_class,
                                        concept_names=concept_names, inequalities=True)[0]
        if exp_accuracy == 100:
            explanation = exp
            break

    assert explanation != "", "Error in founding an explanation"
    return explanation


def logic_exp_distance(exp1: str, exp2: str):
    assert not "|" in exp1, "Only minterm allowed"
    assert not "|" in exp2, "Only minterm allowed"
    dist = 0
    exp1 = exp1.replace(" ", "")
    exp2 = exp2.replace(" ", "")
    for term1 in exp1.split("&"):
        if term1 not in exp2:
            dist += 1
    for term2 in exp2.split("&"):
        if term2 not in exp1:
            dist += 1

    return dist, dist / len(exp1.split("&"))

# from matplotlib import pyplot as plt
# from sklearn import tree
#
# fig = plt.figure(figsize=(25, 20))
# _ = tree.plot_tree(model,
#                    class_names=["0", "1"])
# plt.show()


def sensitivity_lime(model: XGBClassifier, x: np.ndarray, eps=[0.1, 0.2, 0.3, 0.4, 0.5]):
    df = {
        "sensitivity_samples": [],
        "explanation": [],
        "explanation_dist": [],
        "explanation_dist_norm": [],
        "sample_id": [],
        "distance": [],
    }
    num_features = x.shape[1]
    explainer = LimeTabularExplainer(x, categorical_features=[*range(num_features)],
                                     feature_names=[f"x_{i:2d}" for i in range(num_features)])
    for i, sample in enumerate(x):
        orig_class = model.predict(np.expand_dims(sample, 0))[0]
        explanation = get_lime_explanation(explainer, model, sample, num_features, orig_class)

        df["sensitivity_samples"].append(copy.deepcopy(sample))
        df["explanation"].append(explanation)
        df["explanation_dist"].append(0)
        df["explanation_dist_norm"].append(0)
        df["sample_id"].append(i)
        df["distance"].append(0)

        for d in eps:
            while True:
                new_sample = perturb_input(sample, d)
                new_class = model.predict(np.expand_dims(new_sample,0))[0]
                if new_class == orig_class:
                    break

            new_exp = get_lime_explanation(explainer, model, new_sample, num_features, orig_class)
            exp_dist = np.abs(explanation - new_exp).sum().item()
            exp_dist_norm = exp_dist / np.linalg.norm(explanation)

            df["sensitivity_samples"].append(copy.deepcopy(new_sample))
            df["explanation"].append(new_exp)
            df["explanation_dist"].append(exp_dist)
            df["explanation_dist_norm"].append(exp_dist_norm.item())
            df["sample_id"].append(i)
            df["distance"].append(d)

    return df


def get_lime_explanation(explainer, model, sample, num_features, orig_class):
    lime_explanation= explainer.explain_instance(sample, model.predict_proba,
                               num_features=num_features, top_labels=1).as_list(label=orig_class)
    explanation = np.zeros(num_features)
    for feat, imp in lime_explanation:
        d = int(feat[2:4])
        sign = int(feat[5])
        if not sign:
            imp = - imp
        explanation[d] = imp

    return explanation


def perturb_input(sample: np.ndarray, eps=0.25) -> np.ndarray:
    n_features = len(sample)
    new_sample = copy.deepcopy(sample)
    # d_i = torch.randint(n_features, (1,))
    # feat = new_sample[d_i]
    # if feat > 0.:
    #     new_feat = np.max([0, feat - perturbation])
    # else:
    #     new_feat = np.min([1, feat + perturbation])
    # new_sample[d_i] = new_feat
    perturbation = np.random.uniform(-eps, eps, n_features)
    new_sample = new_sample + perturbation
    new_sample = np.clip(new_sample, 0, 1)

    return new_sample


sensitivity_functions = {
    "XReluNN": sensitivity_xrelu,
    "LogisticRegression": sensitivity_logistic,
    "DecisionTreeClassifier": sensitivity_tree,
    "XGBClassifier": sensitivity_lime,
}

