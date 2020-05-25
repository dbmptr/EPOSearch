import numpy as np

from problems.toy_biobjective import circle_points, concave_fun_eval, create_pf
from solvers import epo_search, pareto_mtl_search, linscalar, moo_mtl_search

import matplotlib.pyplot as plt
from latex_utils import latexify

if __name__ == '__main__':
    K = 4       # Number of trajectories
    n = 20      # dim of solution space
    m = 2       # dim of objective space
    rs = circle_points(K)  # preference

    pmtl_K = 5
    pmtl_refs = circle_points(pmtl_K, 0, np.pi / 2)
    methods = ['EPO', 'PMTL', 'MOOMTL', 'LinScalar']
    latexify(fig_width=2., fig_height=1.55)
    ss, mi = 0.1, 100
    pf = create_pf()
    for method in methods:
        fig, ax = plt.subplots()
        fig.subplots_adjust(left=.12, bottom=.12, right=.97, top=.97)
        ax.plot(pf[:, 0], pf[:, 1], lw=2, c='k', label='Pareto Front')
        last_ls = []
        for k, r in enumerate(rs):
            r_inv = 1. / r
            ep_ray = 1.1 * r_inv / np.linalg.norm(r_inv)
            ep_ray_line = np.stack([np.zeros(m), ep_ray])
            label = r'$r^{-1}$ ray' if k == 0 else ''
            ax.plot(ep_ray_line[:, 0], ep_ray_line[:, 1], color='k',
                    lw=1, ls='--', dashes=(15, 5), label=label)
            ax.arrow(.95 * ep_ray[0], .95 * ep_ray[1],
                     .05 * ep_ray[0], .05 * ep_ray[1],
                     color='k', lw=1, head_width=.02)
            # x0 = np.random.randn(n) * 0.4
            x0 = np.zeros(n)
            x0[range(0, n, 2)] = 0.3
            x0[range(1, n, 2)] = -.3
            x0 += 0.1 * np.random.randn(n)
            x0 = np.random.uniform(-0.6, 0.6, n) if method in ["MOOMTL", "LinScalar"] else x0
            if method == 'EPO':
                _, res = epo_search(concave_fun_eval, r=r, x=x0,
                                    step_size=ss, max_iters=100)
            if method == 'PMTL':
                _, res = pareto_mtl_search(concave_fun_eval,
                                           ref_vecs=pmtl_refs, r=r_inv, x=x0,
                                           step_size=0.2, max_iters=150)
            if method == 'LinScalar':
                _, res = linscalar(concave_fun_eval, r=r, x=x0,
                                   step_size=ss, max_iters=mi)
            if method == 'MOOMTL':
                _, res = moo_mtl_search(concave_fun_eval, x=x0,
                                        step_size=0.2, max_iters=150)
            last_ls.append(res['ls'][-1])
        last_ls = np.stack(last_ls)
        ax.scatter(last_ls[:, 0], last_ls[:, 1], s=40, c='b', alpha=1)
        ax.set_xlabel(r'$l_1$')
        ax.set_ylabel(r'$l_2$')
        ax.xaxis.set_label_coords(1.015, -0.03)
        ax.yaxis.set_label_coords(-0.01, 1.01)
        ax.spines['right'].set_color('none')
        ax.spines['top'].set_color('none')
        fig.savefig('figures/' + method + '.pdf')

    plt.show()
