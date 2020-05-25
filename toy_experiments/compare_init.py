import numpy as np

from problems.toy_biobjective import circle_points, concave_fun_eval, create_pf
from solvers import epo_search, pareto_mtl_search

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from latex_utils import latexify


if __name__ == "__main__":
    init_type = "hard"

    K = 4       # Number of trajectories
    n = 20      # dim of solution space
    m = 2       # dim of objective space
    rs = circle_points(K)  # preference

    pmtl_refs = 1. / rs
    pf = create_pf()

    methods = ["EPO", "PMTL"]
    colors = list(mcolors.TABLEAU_COLORS.values())
    print(colors)
    markers = {"EPO": '*', "PMTL": '^'}
    msz = {"EPO": 90, "PMTL": 35}

    for method in methods:
        latexify(fig_width=2.25, fig_height=1.8)
        fig, ax = plt.subplots()
        fig.subplots_adjust(left=.12, bottom=.12, right=.975, top=.975)

        label = 'Pareto Front' if init_type == "easy" and method == "EPO" else ''
        ax.plot(pf[:, 0], pf[:, 1], lw=2, c='k', label=label)

        for k, r in enumerate(rs):
            if init_type == "hard":
                x0 = np.random.uniform(0.15 * (-1)**(k), .5 * (-1)**(k), n)
            else:
                x0 = np.random.uniform(-0.4, 0.4, n)
            l0, _ = concave_fun_eval(x0)
            label = r'$l(\theta^0)$' if init_type == "hard" and k==0 and method == "PMTL" else ''
            plt.scatter([l0[0]], [l0[1]], c=colors[k], s=35,
                        label=label, zorder=2)
            r_inv = 1. / r
            if init_type == "hard":
                ss_epo, ss_pmtl, mi_epo, mi_pmtl = 0.2, 0.5, 120, 200
                ep_ray = r_inv * np.linalg.norm(l0) / np.linalg.norm(r_inv)
            else:
                ss_epo, ss_pmtl, mi_epo, mi_pmtl = 0.1, 0.25, 60, 100
                ep_ray = r_inv * (r_inv @ np.array([0.9, 0.9])) / np.linalg.norm(r_inv)**2

            ep_ray_line = np.stack([np.zeros(m), ep_ray])
            label = r'$r^{-1}$ ray' if k == 0 and init_type == "easy" and method == "EPO" else ''
            ax.plot(ep_ray_line[:, 0], ep_ray_line[:, 1], color=colors[k],
                    lw=1, ls='--', dashes=(10, 3), label=label)
            ax.arrow(.95 * ep_ray[0], .95 * ep_ray[1],
                     .05 * ep_ray[0], .05 * ep_ray[1],
                     color=colors[k], lw=1, head_width=.02)

            if method == "EPO":
                _, res = epo_search(concave_fun_eval, r=r, x=x0,
                                    step_size=ss_epo, max_iters=mi_epo)
                ls = res['ls']
            else:
                _, res = pareto_mtl_search(concave_fun_eval,
                                           ref_vecs=pmtl_refs, r=r_inv, x=x0,
                                           step_size=ss_pmtl,
                                           max_iters=mi_pmtl)
                ls = res['ls']
                ls_init = res['ls_init']
                ls_init = np.r_[ls_init, ls[[0], :]]
                label = 'PMTL Phase 1' if k == 0 and init_type == "easy" else ''
                ax.plot(ls_init[:, 0], ls_init[:, 1], c=colors[k],
                        ls=':', lw=2, label=label, zorder=1)

            label = "PMTL Phase 2" if k == 0 and init_type == "easy" and method == "PMTL" else ''
            ax.plot(ls[:, 0], ls[:, 1], c=colors[k], lw=2, label=label)
            label = f"{method} Search\n" if method == 'EPO' else f"{method} "
            label = label + "output" if k == 0 and init_type == "hard" else ""
            ax.scatter(ls[[-1], 0], ls[[-1], 1], marker=markers[method],
                       c=colors[k], s=msz[method], label=label, zorder=3)

        ax.set_xlabel(r'$l_1$')
        ax.xaxis.set_label_coords(1.015, -0.03)
        ax.yaxis.set_label_coords(-0.01, 1.01)
        ax.spines['right'].set_color('none')  # turn off the right spine/ticks
        ax.spines['top'].set_color('none')  # turn off the top spine/ticks
        ax.set_ylabel(r'$l_2$')
        ax.legend(loc="lower left", framealpha=.9)
        fig.savefig('figures/' + f'{method}_{init_type}_init.pdf')

    plt.show()
