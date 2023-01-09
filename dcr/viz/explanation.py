import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from matplotlib.colors import ListedColormap

all_cols = ListedColormap(sns.color_palette('colorblind').as_hex()).colors
green = all_cols[2]
red = all_cols[3]


def plot_explanation(img, weights, concept_names, class_name, max_c=5):
    weights = np.array(weights)
    weights_idx = np.argsort(np.abs(weights))[::-1]
    weights_sorted = weights[weights_idx][:max_c]

    available_concepts = len(weights_sorted)
    attn_sorted = np.zeros(max_c)
    attn_sorted[:available_concepts] = weights_sorted
    cns = np.array(concept_names)[weights_idx][:available_concepts].tolist()
    concept_names = cns + ['' for i in range(max_c - len(cns))]

    y_pos = np.arange(len(weights_sorted))[::-1]
    colors = [green if i > 0 else red for i in weights_sorted]

    sns.set_style('whitegrid')

    plt.figure(figsize=[5, 10])

    plt.subplot(2, 1, 1)
    plt.imshow(img.permute(1, 2, 0))
    plt.xticks([])
    plt.yticks([])
    plt.box(False)

    plt.subplot(2, 1, 2)
    plt.title(class_name, fontsize=25, pad=25)
    plt.grid(None)
    plt.hlines(y=y_pos, xmin=0, xmax=weights_sorted, colors=colors, linewidth=25)
    # for pos in range(max_c):
    for x, y, tex, cname in zip(weights_sorted, y_pos, weights_sorted, concept_names):
        if abs(round(tex, 2)) > 0.01:
            t = plt.text(x, y, round(tex, 2), horizontalalignment='right' if x < 0 else 'left',
                         verticalalignment='center',
                         fontdict={'color': red if x < 0 else green, 'size': 14})
            t = plt.text(0, y + 0.25, cname, horizontalalignment='center', verticalalignment='center',
                         fontdict={'color': 'black', 'size': 15})
    plt.xticks([])
    plt.yticks([])
    plt.box(False)
    plt.xlim([-1, 1])
    plt.tight_layout()
