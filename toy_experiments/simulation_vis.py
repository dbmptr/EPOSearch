import numpy as np
import pickle
import pandas as pd

from solvers.epo_lp import mu

import matplotlib.pyplot as plt
import seaborn as sns


data, timings, mus, times = pickle.load(open("simulaiton.pkl", 'rb'))
method_dfs = []
for method in data:
    dfs = []
    for m in data[method]:
        df = pd.DataFrame()
        df[r"$\mu_r(l)$"] = [mu(rl) for rl in data[method][m]]
        df[r"$m$"] = m
        if method == "PMTL":
            mu_rl = df[r"$\mu_r(l)$"]
            df = df[mu_rl.between(mu_rl.quantile(.1), mu_rl.quantile(.9))]
        dfs.append(df)
    method_df = pd.concat(dfs, ignore_index=True)
    method_df["Method"] = method
    method_dfs.append(method_df)
data_df = pd.concat(method_dfs, ignore_index=True)

fig = plt.figure(figsize=(3, 2.7))
fig.subplots_adjust(left=0.11, bottom=0.1, right=.96, top=.98)
g = sns.pointplot(x=r"$m$", y=r"$\mu_r(l)$", hue="Method", data=data_df,
                  markers=['*', '^'], capsize=.15, scale=1)
plt.ylabel(r"$\mu_r(l)$", rotation=0)
ax = plt.gca()
plt.setp(ax.lines, linewidth=2)
ax.xaxis.set_label_coords(1.02, -0.02)
ax.yaxis.set_label_coords(-0.07, 0.92)
ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2])
ax.spines['right'].set_color('none')  # turn off the right spine/ticks
ax.spines['top'].set_color('none')
handles, labels = ax.get_legend_handles_labels()
ax.legend(handles=handles, labels=labels, title='', loc=(0.3, 0.8))

plt.savefig("figures/simulation.pdf")

method_dfs = []
for method in timings:
    dfs = []
    for m in timings[method]:
        df = pd.DataFrame()
        df["Run Time in seconds"] = timings[method][m]
        df[r"$m$"] = m
        if method == "PMTL":
            tm = df["Run Time in seconds"]
            df = df[tm.between(tm.quantile(.1), tm.quantile(.9))]
        dfs.append(df)
    method_df = pd.concat(dfs, ignore_index=True)
    method_df["Method"] = method
    method_dfs.append(method_df)
times_df = pd.concat(method_dfs, ignore_index=True)

fig = plt.figure(figsize=(3, 2.7))
fig.subplots_adjust(left=0.16, bottom=0.1, right=.96, top=.98)
g = sns.pointplot(x=r"$m$", y="Run Time in seconds", hue="Method",
                  data=times_df, markers=['*', '^'], capsize=.15, scale=1)
ax = plt.gca()
ax.get_legend().set_visible(False)
plt.setp(ax.lines, linewidth=2)
ax.xaxis.set_label_coords(1.02, -0.02)
ax.yaxis.set_label_coords(-0.13, 0.5)
ax.spines['right'].set_color('none')  # turn off the right spine/ticks
ax.spines['top'].set_color('none')
plt.savefig("figures/run_time.pdf")
plt.show()
