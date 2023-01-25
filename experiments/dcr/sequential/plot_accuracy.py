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

def main():
    if os.path.exists(f'results/dcr/accuracy.csv'):
        results = pd.read_csv(f'results/dcr/accuracy.csv', index_col=None)
    else:
        results = []
        for dataset in ['xor', 'trig', 'vec', 'mutag', 'celeba']:
            results.append(pd.read_csv(f'results/dcr/{dataset}_accuracy.csv', index_col=None))
        results = pd.concat(results)
        results['model'] = results['Model']
    res_dir = f'results/dcr/'

    os.makedirs(res_dir, exist_ok=True)
    out_file = os.path.join(res_dir, 'results_auc.png')
    out_file_pdf = os.path.join(res_dir, 'results_auc.pdf')

    # results['model'] = results['model'].str.replace('DCR+', 'DCR+ (ours)')

    datasets = results['dataset'].unique()
    models = results['Model'].unique()
    datasets_names = ['XOR', 'Trigonometry', 'Vector', 'Mutagenicity', 'CelebA']
    model_dict = {}
    for m in models:
        if m == 'DCR (ours)': model_dict[m] = ['CE+DCR (ours)', True]
        if m == 'DecisionTreeClassifier': model_dict[m] = ['CT+Decision Tree', True]
        if m == 'LogisticRegression': model_dict[m] = ['CT+Logistic Regression', True]
        if m == 'XGBClassifier': model_dict[m] = ['CT+XGBoost*', False]
        if m == "XReluNN": model_dict[m] = ["CT+ReluNet", True]
        if m == 'DecisionTreeClassifier (Emb.)': model_dict[m] = ['CE+Decision Tree', False]
        if m == 'LogisticRegression (Emb.)': model_dict[m] = ['CE+Logistic Regression', False]
        if m == 'XGBClassifier (Emb.)': model_dict[m] = ['CE+XGBoost', False]
        if m == "XReluNN (Emb.)": model_dict[m] = ["CE+ReluNet", False]
    model_df = pd.DataFrame(model_dict, index=['model', 'Interpretable']).T



    sns.set_style('whitegrid')
    sns.despine()
    hatches = itertools.cycle(['', '', '', '',  '\\/','\\/', '\\/', '\\/', '\\/'])
    all_cols = ListedColormap(sns.color_palette('colorblind')).colors
    old_cols = all_cols[0:4] + [all_cols[7]] + all_cols[1:4] + [all_cols[7]]
    cols = []
    for i, c in enumerate(old_cols):
        if i > 3:
            cols.append((np.array(c)*0.6).tolist())
        else:
            cols.append(np.array(c).tolist())
    cols = np.array(cols)

    fig = plt.figure(figsize=[18, 4])
    lines = []
    for i, dataset in enumerate(datasets):
        ax = plt.subplot(1, len(datasets), i+1)
        plt.title(rf'{datasets_names[i]}', fontsize=30)

        res_in_dataset = results[results['dataset'] == dataset][['model', 'accuracy']]
        # make 2 bar groups one for interpretable and one for non-interpretable predictions
        groups = model_df['Interpretable'].unique()
        offset = 0
        sep = 0
        xticks = []
        ymin = 100
        for g in groups:
            models_in_g = model_df[model_df['Interpretable']==g]
            res_in_g = res_in_dataset[res_in_dataset['model'].isin(pd.Series(models_in_g['model'].index))]
            accs_mean = res_in_g.groupby(['model']).mean()
            accs_mean = accs_mean.loc[models_in_g.index].values.ravel() * 100
            accs_sem = res_in_g.groupby(['model']).sem()
            accs_sem = accs_sem.loc[models_in_g.index].values.ravel() * 100

            xidx = np.arange(len(models_in_g)) + offset
            xticks.append(xidx + sep)
            line = ax.bar(xidx + sep, accs_mean, color=cols[xidx], width=0.95)
            ax.errorbar(xidx + sep, accs_mean, accs_sem,
                        capsize=8, elinewidth=2, capthick=2,
                        fmt='.', ecolor='k', markerfacecolor='k',
                        markeredgecolor='k')
            lines.append(line)

            offset += len(models_in_g)
            sep += 0.5
            ymin = min(ymin, round(accs_mean.min()-15, -1))

        for bar, hatch in zip(ax.patches, hatches):
            bar.set_hatch(hatch)
        if i == 0:
            ax.set_ylabel(r'Task AUC (\%)')
        plt.xlabel('Interpretable')
        xt_pos, xt_label = [], ['yes', 'no']
        for xt in xticks:
            xt_pos.append(xt.mean())
        plt.xticks(xt_pos, xt_label)
        plt.gca().xaxis.grid(False)
        plt.ylim([ymin, 101.])
        yt = np.arange(ymin+10, 101., 10).astype(int)
        if len(yt) < 4:
            yt = np.arange(ymin+5, 101., 5).astype(int)
        plt.yticks(yt, yt)
    patches = [l for l in lines[0]] + [l for l in lines[1]]
    fig.legend(patches, model_df['model'].values, loc='upper center',
              # bbox_to_anchor=(-0.8, -0.2),
              bbox_to_anchor=(0.5, 0.05),
               fontsize=14, frameon=False,
               fancybox=False, shadow=False, ncol=len(models)/2)
    plt.tight_layout()
    plt.savefig(out_file, bbox_inches='tight')
    plt.savefig(out_file_pdf, bbox_inches='tight')
    plt.show()


if __name__ == '__main__':
    main()
