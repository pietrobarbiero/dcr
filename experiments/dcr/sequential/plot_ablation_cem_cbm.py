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
    results = pd.read_csv(f'results/ablation/cem_cbm.csv', index_col=None)
    res_dir = f'results/ablation/'

    os.makedirs(res_dir, exist_ok=True)
    out_file = os.path.join(res_dir, 'results_cem_cbm.png')
    out_file_pdf = os.path.join(res_dir, 'results_cem_cbm.pdf')

    # results['model'] = results['model'].str.replace('DCR+', 'DCR+ (ours)')

    datasets = results['dataset'].unique()
    models = results['setup'].unique()
    models_sorted = sorted(models.tolist())
    datasets_names = ['XOR', 'Trigonometry', 'Vector', 'CelebA']
    models_names = [
        'CEM+DCR',
        'CBM+DCR',
    ]
    model_idx = [1, 0]

    sns.set_style('whitegrid')
    sns.despine()
    all_cols = ListedColormap(sns.color_palette('colorblind')).colors
    old_cols = all_cols[0:3]

    fig = plt.figure(figsize=[12, 3])
    lines = []
    for i, dataset in enumerate(datasets):
        ax = plt.subplot(1, len(datasets), i+1)
        plt.title(rf'{datasets_names[i]}', fontsize=30)
        res = results[results['dataset'] == dataset][['setup', 'accuracy']] * 100
        accs_mean = res.groupby(['setup']).mean()
        accs_mean.index = models_sorted
        accs_mean = accs_mean.iloc[model_idx].values.ravel()
        accs_sem = res.groupby(['setup']).sem()
        accs_sem.index = models_sorted
        accs_sem = accs_sem.iloc[model_idx].values.ravel()
        line = ax.bar(np.arange(len(models)), accs_mean, color=all_cols, width=0.95)
        ax.errorbar(np.arange(len(models)), accs_mean, accs_sem,
                    capsize=10, elinewidth=2, capthick=2, fmt='.', ecolor='k')
        if i == 0:
            ax.set_ylabel(r'Task AUC (\%)')
        lines.append(line)
        plt.xlabel('')
        plt.xticks([], [])
        ymin = round(accs_mean.min()-15, -1)
        plt.ylim([ymin, 101.])
        yt = np.arange(ymin+10, 101., 10).astype(int)
        if len(yt) < 4:
            yt = np.arange(ymin+5, 101., 5).astype(int)
        plt.yticks(yt, yt)
    fig.legend(line.patches, models_names, loc='upper center',
              # bbox_to_anchor=(-0.8, -0.2),
              bbox_to_anchor=(0.5, 0.05),
               fontsize=20, frameon=False,
               fancybox=False, shadow=False, ncol=len(datasets))
    plt.tight_layout()
    plt.savefig(out_file, bbox_inches='tight')
    plt.savefig(out_file_pdf, bbox_inches='tight')
    plt.show()


if __name__ == '__main__':
    main()
