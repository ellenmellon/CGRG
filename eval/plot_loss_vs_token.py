'''
Reads "gen_loss.txt" file and for each example, plot
token_level log probability for different settings in the txt file.

This gen_loss.txt file should be generated from running eval_gpt2.py 

'''


import sys
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict

def save_plot(folder, x, ys):
    raw_xticks = x.split('|')[:-3]
    xticks = []
    merge_indices = []
    for i in range(len(raw_xticks)):
        xtick = raw_xticks[i]
        # if len(xticks) > 0 and ((not xtick.startswith(' ')) or (xticks[-1] == 'new' and xtick.strip() in 'jersey')):
        if len(xticks) > 0 and (not xtick.startswith(' ')):
            xticks[-1] = (xticks[-1]+xtick).strip()
            merge_indices += [i]
        else:
            xticks.append(xtick.strip())
    x = np.array(list(range(len(xticks))))
    plt.xticks(x, xticks, rotation=30)
    print(len(ys))
    for label in ys:
        raw_y = ys[label].split('|')[:-1]
        y = []
        for i in range(len(raw_y)):
            logp = raw_y[i]
            if i in merge_indices:
                y[-1] = y[-1]+float(logp)
            else:
                y.append(float(logp))
        plt.plot(x, y, label=label)
    plt.ylim(-8.0, 1.0)
    plt.xlabel('token')
    plt.ylabel('log(P)')
    plt.legend(loc='upper left')
    title = '{}/{}.png'.format(folder, ' '.join(xticks))
    plt.savefig(title)
    plt.clf()


def read_plots_from_file(fname):
    plots = defaultdict(dict)
    with open(fname) as fin:
        content = fin.read()
        lines = content.split('\n\n')
        for lines in content.split('\n\n'):
            if len(lines.split('\n')) < 4:
                continue
            label, tokens, losses, ave_loss = lines.split('\n')
            if label in plots[tokens]:
                print('Warning: setting {} for target response \'{}\' appears more than once'.format(label, tokens))
            plots[tokens][label] = losses
    return plots


fname = sys.argv[1]
folder = '.'
if '/' in fname:
    folder = '/'.join(fname.split('/')[:-1])
plots = read_plots_from_file(fname)
for x in plots:
    ys = plots[x]
    save_plot(folder, x, ys)
