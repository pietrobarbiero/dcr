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
    results = pd.read_csv(f'results/ablation/temperature.csv', index_col=None)
    res_dir = f'results/ablation/'

    os.makedirs(res_dir, exist_ok=True)
    out_file = os.path.join(res_dir, 'results_temperature.png')
    out_file_pdf = os.path.join(res_dir, 'results_temperature.pdf')

    temperatures = results['temperature'].unique()

    sns.set_style('whitegrid')
    sns.despine()
    all_cols = ListedColormap(sns.color_palette('colorblind')).colors[0]

    fig = plt.figure(figsize=[4, 3])
    lines = []
    ax = plt.subplot(1, 1, 1)
    plt.title(rf'Temperature ablation', fontsize=20)
    # res = results[results['dataset'] == dataset][['model', 'accuracy']] * 100
    gb = results.groupby(['temperature'])
    accs_mean = gb.mean() * 100
    accs_mean = accs_mean['relevant concepts'].values.ravel()
    accs_sem = gb.sem() * 100
    accs_sem = accs_sem['relevant concepts'].values.ravel()
    # line = ax.errorbar(np.arange(len(temperatures)), accs_mean, color=errorbar, width=0.95)
    ax.errorbar(temperatures, accs_mean, accs_sem,
                capsize=3, elinewidth=2, capthick=2, fmt='.', ecolor=all_cols)
    # ax.plot(temperatures, accs_mean, '-', color=all_cols)
    ax.set_ylabel(r'Relevant Concepts (\%)', fontsize=18, labelpad=10)
    ax.set_xscale('log')
    plt.xlabel('Temperature', fontsize=18)
    xt = [0, 25, 50, 75, 100]
    plt.yticks(xt, xt, fontsize=15)
    xt = [1E-1, 3E-1, 1E0, 3E0, 1E1]
    plt.xticks(xt, xt, fontsize=15)
    plt.ylim([0, 101.])
    plt.tight_layout()
    plt.savefig(out_file, bbox_inches='tight')
    plt.savefig(out_file_pdf, bbox_inches='tight')
    plt.show()


if __name__ == '__main__':
    main()
