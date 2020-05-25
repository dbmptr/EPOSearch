import numpy as np

from problems.toy_triobjective import concave_fun_eval, create_pf, sphere_points
from solvers import epo_search

import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.colors as mcolors
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import proj3d
from latex_utils import latexify


class Arrow3D(FancyArrowPatch):
    def __init__(self, xs, ys, zs, *args, **kwargs):
        FancyArrowPatch.__init__(self, (0, 0), (0, 0), *args, **kwargs)
        self._verts3d = xs, ys, zs

    def draw(self, renderer):
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, renderer.M)
        self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))
        FancyArrowPatch.draw(self, renderer)


if __name__ == '__main__':
    K = 3       # Number of trajectories
    n = 20      # dim of solution space
    m = 3       # dim of objective space
    rs = sphere_points(K)
    max_iters = 500
    ss = [0.05, 0.2, 0.05, 0.05, 0.02, 0.05]
    colors = list(mcolors.TABLEAU_COLORS.values())
    pf, pf_tri, ls, tri = create_pf()

    latexify(fig_width=4., fig_height=3.6)
    params = {'axes.labelsize': 16,
              'axes.titlesize': 16,
              'font.size': 15,
              'legend.fontsize': 18,
              'xtick.labelsize': 12,
              'ytick.labelsize': 12
              }
    mpl.rcParams.update(params)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_trisurf(pf[:, 0], pf[:, 1], pf[:, 2], triangles=pf_tri.triangles,
                    color='k', alpha=0.3, shade=True,
                    edgecolor='none', linewidth=0.)
    ax.plot_trisurf(ls[:, 0], ls[:, 1], ls[:, 2], triangles=tri.triangles,
                    color='k', linewidth=0., edgecolor='none', alpha=0.1)

    x0 = np.random.uniform(-0.7, 0.3, n)   # np.random.uniform(-0.5, 0.5, n)
    l0, _ = concave_fun_eval(x0)
    for k, r in enumerate(rs):
        if k == 0:
            label = r'$l(\theta^0)$'
            ax.scatter([l0[0]], [l0[1]], [l0[2]], c=colors[k],
                       s=80, label=label)
        r_inv = 1. / r
        ep_ray = 0.9 * r_inv * np.linalg.norm(l0) / np.linalg.norm(r_inv)
        a = Arrow3D([0, ep_ray[0]],
                    [0, ep_ray[1]],
                    [0, ep_ray[2]],
                    mutation_scale=15, alpha=0.6,
                    lw=2, arrowstyle="-|>", color=colors[k])
        ax.add_artist(a)

        x0, res = epo_search(concave_fun_eval, r=r, x=x0, grad_tol=1e-5,
                             step_size=ss[k], max_iters=max_iters)
        ls = res['ls']
        ax.plot(ls[:, 0], ls[:, 1], ls[:, 2], c=colors[k], lw=3)
        ax.scatter(ls[[-1], 0], ls[[-1], 1], ls[[-1], 2],
                   label=r'$l^{*' + f'({k+1})' + r'}$',
                   marker='*', c=colors[k], s=200)

    # ax.legend([fake2Dline], ['Pareto Front'], numpoints=1)
    ax.set_xlabel(r'$l_1$')
    ax.set_ylabel(r'$l_2$')
    ax.set_zlabel(r'$l_3$')
    # ax.text(0, 0, 1.1, r'$l_3$')
    ax.text(-0.05, -0.05, -0.05, r'$0$')
    ax.xaxis.set_rotate_label(False)
    ax.yaxis.set_rotate_label(False)
    ax.zaxis.set_rotate_label(False)
    ax.xaxis._axinfo['label']['space_factor'] = 0.1

    ax.xaxis.labelpad = -2
    ax.yaxis.labelpad = -2
    ax.zaxis.labelpad = -2
    ax.xaxis.tickpad = -5
    ax.yaxis.tickpad = -5
    ax.zaxis.tickpad = -5
    ax.dist = -2

    ax.grid(False)
    # ax.xaxis.pane.set_edgecolor("gray")
    # ax.yaxis.pane.set_edgecolor("gray")
    # ax.zaxis.pane.set_edgecolor("gray")
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    # ax.xaxis._axinfo['juggled'] = (1, 0, 2)
    # ax.yaxis._axinfo['juggled'] = (1, 1, 2)
    # ax.zaxis._axinfo['juggled'] = (1, 2, 2)

    ax.set_xticks([0.2, 0.4, 0.6, 0.8])
    ax.set_yticks([0.2, 0.4, 0.6, 0.8])
    ax.set_zticks([0.2, 0.4, 0.6, 0.8])

    [t.set_va('bottom') for t in ax.get_yticklabels()]
    [t.set_ha('center') for t in ax.get_yticklabels()]
    [t.set_va('bottom') for t in ax.get_xticklabels()]
    [t.set_ha('center') for t in ax.get_xticklabels()]
    [t.set_va('center') for t in ax.get_zticklabels()]
    [t.set_ha('center') for t in ax.get_zticklabels()]

    ax.xaxis._axinfo['tick']['inward_factor'] = 0
    ax.xaxis._axinfo['tick']['outward_factor'] = 0.4
    ax.yaxis._axinfo['tick']['inward_factor'] = 0
    ax.yaxis._axinfo['tick']['outward_factor'] = 0.4
    ax.zaxis._axinfo['tick']['inward_factor'] = 0
    ax.zaxis._axinfo['tick']['outward_factor'] = 0.4

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_zlim(0, 1)

    fig.subplots_adjust(left=0.0, bottom=-0.05, right=1.1, top=1.2)
    azim, elev = 45, 25
    ax.view_init(elev=elev, azim=azim)
    plt.savefig("figures/" + f"trace_epo_azim{azim}_elev{elev}" + ".pdf")

    # ax.xaxis._axinfo['juggled'] = (1, 0, 2)
    # ax.yaxis._axinfo['juggled'] = (1, 1, 2)
    ax.zaxis._axinfo['juggled'] = (1, 2, 2)

    ax.legend(loc=(0.17, 0.24), handletextpad=0.01)
    azim, elev = -46, 18
    ax.view_init(elev=elev, azim=azim)
    plt.savefig("figures/" + f"trace_epo_azim{azim}_elev{elev}" + ".pdf")

    # azim, elev = -46, 26
    # ax.view_init(elev=elev, azim=azim)
    # plt.savefig(f"trace_epo_azim{azim}_elev{elev}" + ".pdf")
    plt.show()
