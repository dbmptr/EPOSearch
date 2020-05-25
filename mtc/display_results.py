import matplotlib.pyplot as plt
from brokenaxes import brokenaxes
import numpy as np
import pickle as pkl
import pandas as pd
from latex_utils import latexify
import os


baseline = "indiv"
methods = ["EPO", "PMTL", "LinScalar"]
markers = {"EPO": "*", "PMTL": "^", "LinScalar": "s", "indiv": "o"}
marker_szs = {"EPO": 5, "PMTL": 3.5, "LinScalar": 2}
colors = {"EPO": "g", "PMTL": "b", "LinScalar": "r"}
dataset = "emotion"
model = 'fnn'
niters, nprefs = 200, 10
folder = f"results"
m = 6

latexify(fig_width=4, fig_height=1.8)
fig = plt.figure()
fig.subplots_adjust(left=0.12, bottom=0.1, right=.78, top=1)
# Baseline stem plot
file = f"{baseline}_{dataset}_{model}_{niters}.pkl"
results = pkl.load(open(os.path.join(folder, file), 'rb'))
baseline_l = []
for j in results:
    res = results[j]["res"]
    lj = res["training_losses"][-1][j] / m
    baseline_l.append(lj)

markerline, stemlines, baseline = plt.stem(baseline_l, label='Baseline',
                                           use_line_collection=True,
                                           basefmt=' ')
plt.setp(stemlines, 'linewidth', 8, 'alpha', 0.5, 'color', 'gray', 'zorder', -5)
plt.setp(markerline, 'ms', 8, 'color', 'gray', 'zorder', -5, 'marker', '_')

# Other methods
for mid, method in enumerate(methods):
    file = f"{method}_{dataset}_{model}_{niters}_{nprefs}.pkl"
    results = pkl.load(open(os.path.join(folder, file), 'rb'))
    rls = []
    for i in results:
        r, res = results[i]["r"], results[i]["res"]
        method_l = res["training_losses"][-1]
        r /= r.sum()
        # r *= m
        rls.append(r * method_l)
    rls = np.stack(rls)
    df = pd.DataFrame(rls)
    low, high = 0.05, .95
    quant = df.quantile([low, high])
    rl_mean, rl_std = [], []
    for j in range(6):
        # rljs = rls[:, j]
        rljs = [rlj_ for rlj_ in rls[:, j]
                if rlj_ > quant.loc[low, j] and rlj_ < quant.loc[high, j]]
        rl_mean.append(np.mean(rljs))
        rl_std.append(np.std(rljs))
    # rl_mean = rls.mean(axis=0)
    # rl_std = rls.std(axis=0)
    # print(method)
    # print(f"stds={rl_std}")
    plt.plot(np.arange(m) + 0.1 * (mid - 1), rl_mean, c=colors[method], lw=0.5,
             marker=markers[method], ms=marker_szs[method], label=method)
    plt.errorbar(np.arange(len(rl_mean)) + 0.1 * (mid - 1), rl_mean,
                 elinewidth=0.9, yerr=rl_std, fmt=' ', color=colors[method])

plt.xlabel('Tasks')
plt.ylabel(r'$r \odot l$')
# plt.ylim([-0.1, 5])
plt.xticks(ticks=range(6), labels=[f'E{i}' for i in range(1, 9)])
# plt.text(-.5, 7, r'$\times 10^3$', fontsize=6)
ax = plt.gca()
ax.xaxis.set_label_coords(1.07, -0.0)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
plt.legend(loc=(0.99, 0.5), handletextpad=0.01)       # (0.05, 0.4)

# plt.savefig(f'{dataset}.pdf')
plt.show()
