import numpy as np

from solvers import epo_search
from problems.toy_biobjective import concave_fun_eval, create_pf, circle_points

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from latex_utils import latexify


def get_shifted_concave_fun(shift=0):
    def shifted_concave_fun_eval(x):
        l, dl = concave_fun_eval(x)
        return l + shift, dl
    return shifted_concave_fun_eval


if __name__ == '__main__':
    K = 4       # Number of trajectories
    n = 20      # dim of solution space
    m = 2       # dim of objective space
    rs = circle_points(K)
    max_iters = 100
    ss = 0.3
    colors = list(mcolors.TABLEAU_COLORS.values())

    latexify(fig_width=2., fig_height=1.8)
    fig = plt.figure()
    fig.subplots_adjust(left=.12, bottom=.12, right=.97, top=.97)

    pf, ls_left, ls_right = create_pf(side_nonpf=True)
    shift = 0.4
    for ovs in [pf, ls_left, ls_right]:
        ovs += shift
    label = 'Pareto\nFront'
    plt.plot(pf[:, 0], pf[:, 1], lw=5, c='k',  # label=label,
             zorder=-2, alpha=0.4)
    plt.plot(ls_left[:, 0], ls_left[:, 1], lw=1.5,
             ls='-', c='k', zorder=-2, alpha=0.4)
    plt.plot(ls_right[:, 0], ls_right[:, 1], lw=1.5,
             ls='-', c='k', zorder=-2, alpha=0.4)

    shifted_concave_fun_eval = get_shifted_concave_fun(shift)
    x0 = np.random.uniform(-0.4, 0.4, n)    # Easy initialization
    l0, _ = shifted_concave_fun_eval(x0)
    for k, r in enumerate(rs):
        if k == 0:
            label = r'$l(\theta^0)$'
            plt.scatter([l0[0]], [l0[1]], c=colors[k], s=40, label=label)

        r_inv = 1. / r
        ep_ray = r_inv * (r_inv @ l0) / np.linalg.norm(r_inv)**2
        ep_ray_line = np.stack([np.zeros(m), ep_ray])
        plt.plot(ep_ray_line[:, 0], ep_ray_line[:, 1], color=colors[k],
                 lw=1, ls='--', dashes=(8, 3), zorder=-1)
        plt.arrow(.95 * ep_ray[0], .95 * ep_ray[1],
                  .05 * ep_ray[0], .05 * ep_ray[1],
                  color=colors[k], lw=.75, head_width=.04)

        x0, res = epo_search(shifted_concave_fun_eval, r=r, x=x0,
                             step_size=ss, max_iters=max_iters)
        ls = res['ls']
        if k > 0:
            traj_label = ''
        plt.plot(ls[:, 0], ls[:, 1], c=colors[k], lw=1, alpha=0.8)
        plt.scatter(ls[[-1], 0], ls[[-1], 1], marker='*', c=colors[k], s=70,
                    label=r'$l^{*' + f'({k+1})' + r'}$')

    plt.xlabel(r'$l_1$')
    plt.ylabel(r'$l_2$')
    plt.xlim(0, 1.6)
    plt.ylim(0, 1.5)
    plt.legend(loc='lower left', handletextpad=.3, framealpha=0.9)
    ax = plt.gca()
    ax.xaxis.set_label_coords(1.015, -0.03)
    ax.yaxis.set_label_coords(-0.01, 1.01)
    ax.spines['right'].set_color('none')  # turn off the right spine/ticks
    ax.spines['top'].set_color('none')
    plt.savefig("figures/empty_epo.pdf")
    plt.show()
