import numpy as np

from problems.toy_biobjective import concave_fun_eval, create_pf, circle_points
from solvers import epo_search

import matplotlib.pyplot as plt
from latex_utils import latexify

if __name__ == '__main__':
    K = 4       # Number of trajectories
    n = 20      # dim of solution space
    m = 2       # dim of objective space
    rs = circle_points(K)
    x0 = np.zeros(n)
    x0[range(0, n, 2)] = 0.3
    x0[range(1, n, 2)] = -.3
    latexify(fig_width=2., fig_height=1.8)
    l0, _ = concave_fun_eval(x0)
    max_iters = 70
    ss = 0.1
    for c, des_type, relax in zip(['b', 'g'],
                                  ['Relaxed_descent', 'Restricted_descent'],
                                  [True, False]):
        pf = create_pf()
        fig = plt.figure()
        fig.subplots_adjust(left=.12, bottom=.12, right=.975, top=.975)
        label = 'Pareto\nFront' if relax else ''
        plt.plot(pf[:, 0], pf[:, 1], lw=2, c='k', label=label)
        label = r'$l(\theta^0)$' if not relax else ''
        plt.scatter([l0[0]], [l0[1]], c='r', s=40, label=label)
        for k, r in enumerate(rs):
            r_inv = 1. / r
            ep_ray = r_inv * (r_inv @ l0) / np.linalg.norm(r_inv)**2
            ep_ray_line = np.stack([np.zeros(m), ep_ray])
            label = r'$r^{-1} Ray$' if k == 0 and not relax else ''
            plt.plot(ep_ray_line[:, 0], ep_ray_line[:, 1], color='k',
                     label=label, lw=1, ls='--', dashes=(8, 3), zorder=-1)
            plt.arrow(.95 * ep_ray[0], .95 * ep_ray[1],
                      .05 * ep_ray[0], .05 * ep_ray[1],
                      color='k', lw=.75, head_width=.02)

        ss = 0.25 if relax else 0.1
        for k, r in enumerate(rs):
            _, res = epo_search(concave_fun_eval, r=r, x=x0,
                                relax=relax, eps=1e-4,
                                step_size=ss, max_iters=max_iters)

            ls = res['ls']
            if k > 0:
                traj_label = ''
            alpha = 0.8
            zorder = 1 if relax else 0
            plt.plot(ls[:, 0], ls[:, 1], c=c, lw=1.5,
                     alpha=alpha, zorder=zorder)
            plt.scatter(ls[[-1], 0], ls[[-1], 1], c=c, s=40)

        plt.xlabel(r'$l_1$')
        plt.ylabel(r'$l_2$')
        plt.legend(loc='lower left', handletextpad=.3, framealpha=0.9)
        ax = plt.gca()
        ax.xaxis.set_label_coords(1.015, -0.03)
        ax.yaxis.set_label_coords(-0.01, 1.01)
        ax.spines['right'].set_color('none')
        ax.spines['top'].set_color('none')
        plt.savefig('figures/' + des_type + '.pdf')
    plt.show()
