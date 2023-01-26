import itertools

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import os
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch

labelsize = 15
plt.rc('font', family='serif', serif='Times')
plt.rc('text', usetex=True)
plt.rc('figure', labelsize=23)
plt.rc('xtick', labelsize=15)
plt.rc('ytick', labelsize=18)
plt.rc('axes', labelsize=23)

# def main():
results = []
for dataset in ['xor', 'trig', 'vec', 'mutag', 'celeba']:
    results.append(pd.read_csv(f'results/dcr/{dataset}_sensitivity.csv', index_col=None))
results = pd.concat(results)
results['model'] = results['Model']
res_dir = f'results/dcr/'

os.makedirs(res_dir, exist_ok=True)
out_file = os.path.join(res_dir, f'results_sensitivity.png')
out_file_pdf = os.path.join(res_dir, f'results_sensitivity.pdf')

# results['model'] = results['model'].str.replace('DCR+', 'DCR+ (ours)')

datasets = results['dataset'].unique()
models = results['model'].unique()
datasets_names = ['XOR', 'Trigonometry', 'Vector', 'Mutagenicity', 'CelebA']
model_dict = {
    'DCR (ours)': 'CEM+DCR (ours)',
    'DecisionTreeClassifier': 'CBM+Decision Tree',
    'LogisticRegression': 'CBM+Logistic Regression',
    'XGBClassifier': 'CBM+XGBoost*',
    "XReluNN": "CBM+ReluNN",
    "Lime": "CBM+XGBoost (LIME)",
    'DecisionTreeClassifier (Emb.)': 'CEM+Decision Tree*',
    'LogisticRegression (Emb.)': 'CEM+Logistic Regression*',
    'XGBClassifier (Emb.)': 'CEM+XGBoost*',
    "XReluNN (Emb.)": "CEM+ReluNN"
}
model_df = [model_dict[m] for m in models]
model_df = pd.DataFrame(model_df, columns=['Model', ])

# Tabular results
table_results = []
for dataset in datasets:
    results_dataset = results[results['dataset'] == dataset]
    mean = results_dataset.groupby(["model"]).mean()['explanation_dist_norm'].to_frame().T
    std = results_dataset.groupby(["model"]).std()['explanation_dist_norm'].to_frame().T
    for m_i in mean.columns:
        mean[m_i] = f"{mean[m_i].item():.3f}" + "{\\tiny $\pm " + f"{std[m_i].item():.3f}" + "$ }"
    row = mean.rename(columns=model_dict)
    row.index = [dataset]
    table_results.append(row)
table_results = pd.concat(table_results).T
print(table_results.to_latex(escape=False))

sns.set_style('whitegrid')
cols = sns.color_palette('colorblind')[0:1] + sns.color_palette('colorblind')[3:4] + sns.color_palette('colorblind')[4:5]

fig = plt.figure(figsize=[18, 4])
lines = []
for i, dataset in enumerate(datasets):
    ax = plt.subplot(1, len(datasets), i + 1)
    plt.title(rf'{datasets_names[i]}', fontsize=30)
    res_in_dataset = results[results['dataset'] == dataset]
    res_in_dataset = res_in_dataset[res_in_dataset['Model'] != "LogisticRegression"][res_in_dataset['Model'] != "DecisionTreeClassifier"]
    sns.lineplot(ax=ax, data=res_in_dataset, x="distance", y="explanation_dist_norm",
                 hue="Model", palette=cols)
    sns.despine()
    if i == 0:
        # plt.ylabel("$\mathcal{D}$")
        plt.ylabel("Explanation Sensitivity")
    else:
        plt.ylabel("")
    # plt.xlabel("$\epsilon$")
    plt.xlabel("Perturbation Radius")
    lines.append(ax.get_legend_handles_labels())
    ax.legend().set_visible(False)

models_ordered = model_df['Model'].array[[0,4,1]]
fig.legend(lines[0][0], models_ordered, loc='upper center',
          bbox_to_anchor=(0.5, 0.05),
           fontsize=16, frameon=False,
           fancybox=False, shadow=False, ncol=len(models))
plt.tight_layout()
plt.savefig(out_file, bbox_inches='tight')
plt.savefig(out_file_pdf, bbox_inches='tight')
plt.show()

asd = "asd"
#
# if __name__ == '__main__':
#     main()
