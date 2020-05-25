import numpy as np
import pickle as pkl
import os

import matplotlib.pyplot as plt
from brokenaxes import brokenaxes
from latex_utils import latexify

baseline = "indiv"
methods = ["EPO", "PMTL", "LinScalar"]
markers = {"EPO": "*", "PMTL": "^", "LinScalar": "s", "indiv": "o"}
msz = {"EPO": 45, "PMTL": 30, "LinScalar": 25, "indiv": "o"}
datasets = ["mnist", "fashion", "fashion_and_mnist"]
model = 'lenet'
niters, nprefs = 100, 5
folder = f"results"

data = dict()
for dataset in datasets:
    print(dataset)
    data[dataset] = dict()
    for method in ["EPO", "PMTL", "indiv", "LinScalar"]:
        print(f"\t{method}")
        file_pfx = f"{method}_{dataset}_{model}_{niters}"
        file_sfx = "" if method == 'indiv' else f"_{nprefs}_from_0-{nprefs-1}"
        file = file_pfx + file_sfx + ".pkl"
        results = pkl.load(open(os.path.join(folder, file), 'rb'))
        last_ls, last_acs, rs = [], [], []
        for i in results:
            r, res = results[i]["r"], results[i]["res"]
            ls, acs = res["training_losses"], res["training_accuracies"]
            rs.append(r)
            last_ls.append(ls[-1])
            last_acs.append(acs[-1])

        last_ls, last_acs = np.asarray(last_ls), np.asarray(last_acs)
        data[dataset][method] = {"last_ls": last_ls,
                                 "last_acs": last_acs,
                                 "rs": rs}
        if method == "LinScalar":
            data[dataset]["rlens"] = [1.2 * np.linalg.norm(l) for l in last_ls]
            data[dataset]["max1"] = np.max(last_ls, axis=0)
            if dataset == "fashion":
                print(last_ls)
                print(data[dataset]["max1"])
        if method == "EPO":
            data[dataset]["max2"] = np.max(last_ls, axis=0)
            data[dataset]["rs"] = rs
            if dataset == "fashion":
                print(last_ls)

        if method == "indiv":
            data[dataset]["baseline_loss"] = []
            data[dataset]["baseline_acc"] = []
            for i, r in enumerate(rs):
                if r[0] == 0:
                    lxs, lys = [0, 10], [min(last_ls[i]), min(last_ls[i])]
                    axs, ays = [0, 1], [max(last_acs[i]), max(last_acs[i])]
                else:
                    lxs, lys = [min(last_ls[i]), min(last_ls[i])], [0, 10]
                    axs, ays = [max(last_acs[i]), max(last_acs[i])], [0, 1]
                data[dataset]["baseline_loss"].append((lxs, lys))
                data[dataset]["baseline_acc"].append((axs, ays))


latexify(fig_width=2.25, fig_height=1.5)
for dataset in datasets:
    fig = plt.figure()
    max1x, max1y = data[dataset]["max1"]
    max2x, max2y = data[dataset]["max2"]
    ax = brokenaxes(xlims=((-.05, max2x + .05), (max1x - .1, max1x + .15)),
                    ylims=((-.05, max2y + .05), (max1y - .1, max1y + .1)),
                    hspace=0.05, wspace=0.05, fig=fig)
    ax.set_xlabel(r"task 1 loss")
    ax.set_ylabel(r"task 2 loss")
    for i, (xs, ys) in enumerate(data[dataset]["baseline_loss"]):
        label = "baseline\nloss" if i == 0 and dataset == "fashion" else ""
        ax.plot(xs, ys, lw=2, alpha=0.4, c='k')

    colors = []
    for i, (r, l) in enumerate(zip(data[dataset]["rs"],
                                   data[dataset]["rlens"])):
        label = r"$r^{-1}$ Ray" if i == 0 and dataset == "fashion" else ""
        r_inv = np.sqrt(1 - r**2)
        lines = ax.plot([0, .9 * l * r_inv[0]], [0, .9 * l * r_inv[1]],
                        lw=1, alpha=0.5, ls='--', dashes=(10, 2), label=label)
        colors.append(lines[0][0].get_color())
        if i in range(1, len(data[dataset]["rs"]) - 1):
            ax0 = 0.85 * l * r_inv[0]
            ay0 = 0.85 * l * r_inv[1]
        else:
            ax0 = 0.9 * r_inv[0]
            ay0 = 0.9 * r_inv[1]
        ax.arrow(ax0, ay0, .05 * r_inv[0], .05 * r_inv[1],
                 color=colors[-1], lw=1, head_width=.04, alpha=0.5)

    for method in methods:
        last_ls = data[dataset][method]["last_ls"]
        s = 40 if method == "EPO" else 30
        ax.scatter(last_ls[:, 0], last_ls[:, 1], marker=markers[method],
                   c=colors, s=msz[method])  # , label=label)

    if dataset == "fashion":
        ax.legend()

    fig.savefig(f"figures/{dataset}_loss.pdf")

latexify(fig_width=2.25, fig_height=1.5)
for dataset in datasets:
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_xlabel(r"task 1 accuracies")
    ax.set_ylabel(r"task 2 accuracies")
    # Hide the right and top spines
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)

    # Only show ticks on the left and bottom spines
    ax.yaxis.set_ticks_position('right')
    ax.xaxis.set_ticks_position('top')
    ax.yaxis.set_label_position('right')
    ax.xaxis.set_label_position('top')
    for i, (xs, ys) in enumerate(data[dataset]["baseline_acc"]):
        label = "baseline" if i == 0 and dataset == "mnist" else ""
        ax.plot(xs, ys, lw=2, alpha=0.4, c='k', label=label)

    for method in methods:
        last_acs = data[dataset][method]["last_acs"]
        label = method if dataset == "fashion_and_mnist" else ""
        s = 40 if method == "EPO" else 30
        ax.scatter(last_acs[:, 0], last_acs[:, 1], marker=markers[method],
                   c=colors, s=msz[method], label=label)

    if dataset in ["mnist", "fashion_and_mnist"]:
        ax.legend()
    fig.savefig(f"figures/{dataset}_acc.pdf")

plt.show()
