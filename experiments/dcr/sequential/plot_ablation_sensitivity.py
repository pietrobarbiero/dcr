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
    results = pd.read_csv(f'results/ablation/sensitivity.csv', index_col=None)
    res_dir = f'results/ablation/'

    os.makedirs(res_dir, exist_ok=True)
    radius = results['radius'].unique()
    models = results['model'].unique()
    datasets = {'XOR': 'xor', 'Trigonometry': 'trig', 'Vector': 'vec'}

    sns.set_style('whitegrid')
    sns.despine()
    all_cols = ListedColormap(sns.color_palette('colorblind')).colors[:3]

    out_file = os.path.join(res_dir, 'results_sensitivity.png')
    out_file_pdf = os.path.join(res_dir, 'results_sensitivity.pdf')
    fig = plt.figure(figsize=[10, 4])
    for q, (dataset, dataset_id) in enumerate(datasets.items()):
        ax = plt.subplot(1, len(datasets), q+1)
        plt.title(rf'{dataset}', fontsize=28, pad=15)
        for mi, model in enumerate(models):
            gb = results[(results['model']==model) & (results['dataset']==dataset_id)].groupby(['radius'])
            accs_mean = gb.mean()['explanation sensitivity'].values.ravel()
            accs_sem = gb.sem()['explanation sensitivity'].values.ravel()
            ax.errorbar(radius, accs_mean, accs_sem,
                        capsize=3, elinewidth=2, capthick=2, fmt='.',
                        ecolor=all_cols[mi], label=model)
        plt.xlabel('Perturbation Radius', fontsize=20)
        plt.yticks(fontsize=15)
        plt.xticks(fontsize=15)
        if q == 0:
            plt.ylabel(r'Explanation Sensitivity', fontsize=20, labelpad=10)
            fig.legend(loc='upper center',
                      bbox_to_anchor=(0.5, 0.05),
                      fontsize=18, frameon=False,
                      fancybox=False, shadow=False, ncol=len(models))
    plt.tight_layout()
    plt.savefig(out_file, bbox_inches='tight')
    plt.savefig(out_file_pdf, bbox_inches='tight')
    plt.show()

    out_file = os.path.join(res_dir, 'results_sensitivity_predictions.png')
    out_file_pdf = os.path.join(res_dir, 'results_sensitivity_predictions.pdf')
    fig = plt.figure(figsize=[10, 4])
    for q, (dataset, dataset_id) in enumerate(datasets.items()):
        ax = plt.subplot(1, len(datasets), q+1)
        plt.title(rf'{dataset}', fontsize=28, pad=15)
        for mi, model in enumerate(models):
            gb = results[(results['model']==model) & (results['dataset']==dataset_id)].groupby(['radius'])
            accs_mean = gb.mean()['prediction sensitivity'].values.ravel()
            accs_sem = gb.sem()['prediction sensitivity'].values.ravel()
            ax.errorbar(radius, accs_mean, accs_sem,
                        capsize=3, elinewidth=2, capthick=2, fmt='.',
                        ecolor=all_cols[mi], label=model)
        plt.xlabel('Perturbation Radius', fontsize=20)
        plt.yticks(fontsize=15)
        plt.xticks(fontsize=15)
        if q == 0:
            plt.ylabel(r'Prediction Sensitivity (\%)', fontsize=20, labelpad=10)
            fig.legend(loc='upper center',
                      bbox_to_anchor=(0.5, 0.05),
                      fontsize=18, frameon=False,
                      fancybox=False, shadow=False, ncol=len(models))
    plt.tight_layout()
    plt.savefig(out_file, bbox_inches='tight')
    plt.savefig(out_file_pdf, bbox_inches='tight')
    plt.show()


if __name__ == '__main__':
    main()
