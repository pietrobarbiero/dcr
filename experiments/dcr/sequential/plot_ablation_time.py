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
    results = pd.read_csv(f'results/ablation/time.csv', index_col=None)
    res_dir = f'results/ablation/'

    os.makedirs(res_dir, exist_ok=True)

    n_concepts = results['n concepts'].unique()

    sns.set_style('whitegrid')
    sns.despine()
    all_cols = ListedColormap(sns.color_palette('colorblind')).colors[0]

    out_file = os.path.join(res_dir, 'results_train_time.png')
    out_file_pdf = os.path.join(res_dir, 'results_train_time.pdf')
    fig = plt.figure(figsize=[4, 3])
    ax = plt.subplot(1, 1, 1)
    plt.title(rf'Training Time', fontsize=20)
    gb = results.groupby(['n concepts'])
    accs_mean = gb.mean()
    accs_mean = accs_mean['eta train'].values.ravel()
    accs_sem = gb.sem()
    accs_sem = accs_sem['eta train'].values.ravel()
    ax.errorbar(n_concepts, accs_mean, accs_sem,
                capsize=3, elinewidth=2, capthick=2, fmt='.', ecolor=all_cols)
    ax.set_ylabel(r'Time (sec.)', fontsize=18, labelpad=10)
    plt.xlabel('Number of concepts', fontsize=18)
    plt.xticks(n_concepts, n_concepts, fontsize=15)
    yt = [i for i in range(0, 9, 2)]
    plt.yticks(yt, yt, fontsize=15)
    plt.tight_layout()
    plt.savefig(out_file, bbox_inches='tight')
    plt.savefig(out_file_pdf, bbox_inches='tight')
    plt.show()

    out_file = os.path.join(res_dir, 'results_test_time.png')
    out_file_pdf = os.path.join(res_dir, 'results_test_time.pdf')
    fig = plt.figure(figsize=[4.5, 3])
    ax = plt.subplot(1, 1, 1)
    plt.title(rf'Test Time', fontsize=20)
    gb = results.groupby(['n concepts'])
    accs_mean = gb.mean()
    accs_mean = accs_mean['eta test'].values.ravel()
    accs_sem = gb.sem()
    accs_sem = accs_sem['eta test'].values.ravel()
    ax.errorbar(n_concepts, accs_mean, accs_sem,
                capsize=3, elinewidth=2, capthick=2, fmt='.', ecolor=all_cols)
    ax.set_ylabel(r'Time (sec.)', fontsize=18, labelpad=10)
    plt.xlabel('Number of concepts', fontsize=18)
    plt.xticks(n_concepts, n_concepts, fontsize=15)
    yt = [i for i in np.arange(0, 0.002, 0.0005)]
    yts = [f'{i:.1E}' for i in yt]
    plt.yticks(yt, yts, fontsize=15)
    plt.tight_layout()
    plt.savefig(out_file, bbox_inches='tight')
    plt.savefig(out_file_pdf, bbox_inches='tight')
    plt.show()


if __name__ == '__main__':
    main()
