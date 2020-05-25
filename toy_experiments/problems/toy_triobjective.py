import autograd.numpy as np
from autograd import grad
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.tri as mtri
from mpl_toolkits.mplot3d import Axes3D


def f1(x):
    n = len(x)
    dx = np.linalg.norm(x - 1. / np.sqrt(n))
    return 1. - np.exp(-dx**2)


def f2(x):
    n = len(x)
    dx = np.linalg.norm(x + 1. / np.sqrt(n))
    return 1. - np.exp(-dx**2)


def f3(x):
    n = len(x)
    idx = range(1, len(x), 2)
    shift = np.ones_like(x)
    shift[idx] = -1.
    dx = np.linalg.norm(x + shift / np.sqrt(n))
    return 1. - np.exp(-dx**2)


# calculate the gradients using autograd
f1_dx = grad(f1)
f2_dx = grad(f2)
f3_dx = grad(f3)


def concave_fun_eval(x):
    """
    return the function values and gradient values
    """
    return np.stack([f1(x), f2(x), f3(x)]), \
        np.stack([f1_dx(x), f2_dx(x), f3_dx(x)])


# ### create the ground truth Pareto front ###
def create_pf():
    u = np.linspace(-1.2 / np.sqrt(2), 1.2 / np.sqrt(2), endpoint=True, num=60)
    v = np.linspace(-1.2 / np.sqrt(2), 1.2 / np.sqrt(2), endpoint=True, num=60)
    U, V = np.meshgrid(u, v)
    u, v = U.flatten(), V.flatten()
    uv = np.stack([u, v]).T
    print(f"uv.shape={uv.shape}")
    ls = []
    for x in uv:
        # generate solutions on the Pareto front:
        # x = np.array([x1, x1])

        f, f_dx = concave_fun_eval(x)
        ls.append(f)
    ls = np.stack(ls)

    po, pf = [], []
    for i, x in enumerate(uv):
        l_i = ls[i]
        if np.any(np.all(l_i > ls, axis=1)):
            continue
        else:
            po.append(x)
            pf.append(l_i)

    po = np.stack(po)
    pf = np.stack(pf)
    pf_tri = mtri.Triangulation(po[:, 0], po[:, 1])
    tri = mtri.Triangulation(u, v)

    return pf, pf_tri, ls, tri


def sphere_points(K, min_angle=None, max_angle=None):
    # generate evenly distributed preference vector
    # ang0 = np.pi / 20. if min_angle is None else min_angle
    # ang1 = np.pi * 9 / 20. if max_angle is None else max_angle
    # azim = np.linspace(ang0, ang1, endpoint=True, num=K)
    # elev = np.linspace(ang0, ang1, endpoint=True, num=K)

    azim = [np.linspace(np.pi / 8, np.pi * 3 / 8, endpoint=True, num=3),
            np.linspace(np.pi / 5, np.pi * 3. / 10, endpoint=True, num=2),
            [np.pi / 4]]
    elev = [np.pi / 3, np.pi / 6, np.pi / 12]
    rs = []
    for i, el in enumerate(elev):
        azs = azim[i] if i % 2 != 0 else azim[i][::-1]
        for az in azs:
            rs.append(np.array([np.sin(el) * np.cos(az),
                                np.sin(el) * np.sin(az),
                                np.cos(el)]))

    # for az in azim:
    #     for el in elev:
    #         rs.append(np.array([np.cos(el) * np.cos(az),
    #                             np.cos(el) * np.sin(az),
    #                             np.sin(el)]))

    # rs.append(np.array([1., 1., 1.]))
    return np.stack(rs)


if __name__ == '__main__':

    pf, pf_tri, ls, tri = create_pf()
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1, projection='3d')
    ax.plot_trisurf(pf[:, 0], pf[:, 1], pf[:, 2], triangles=pf_tri.triangles,
                    color='k', alpha=0.5, shade=True)
    ax.plot_trisurf(ls[:, 0], ls[:, 1], ls[:, 2], triangles=tri.triangles,
                    color='k', linewidth=0., edgecolor='none', alpha=0.1, shade=True)

    fake2Dline = mpl.lines.Line2D([0], [0], linestyle="none", c='k',
                                  marker='s', alpha=0.5)
    ax.legend([fake2Dline], ['Pareto Front'], numpoints=1)
    ax.set_xlabel(r'$l_1$')
    ax.set_ylabel(r'$l_2$')
    ax.set_zlabel(r'$l_3$')
    # ax.grid(False)

    ax.set_xticks([0.2, 0.4, 0.6, 0.8])
    ax.set_yticks([0.2, 0.4, 0.6, 0.8])
    ax.set_zticks([0.2, 0.4, 0.6, 0.8])

    # ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    # ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    # ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))

    # ax.xaxis._axinfo["grid"]['color'] = (1, 1, 1, 0)
    # ax.yaxis._axinfo["grid"]['color'] = (1, 1, 1, 0)
    # ax.zaxis._axinfo["grid"]['color'] =  (1,1,1,0)

    ax.grid(False)
    ax.xaxis.pane.set_edgecolor('black')
    ax.yaxis.pane.set_edgecolor('black')
    ax.zaxis.pane.set_edgecolor('black')
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False

    # change fontsize
    # for t in ax.zaxis.get_major_ticks(): t.label.set_fontsize(10)
    # disable auto rotation
    ax.zaxis.set_rotate_label(False)
    ax.xaxis.set_rotate_label(False)
    # ax.xaxis._axinfo['label']['space_factor'] = 0.5
    # ax.set_xticklabels(['0.2', '0.4', '0.6', '0.8'], rotation=0,
    #                    verticalalignment='bottom',
    #                    horizontalalignment='center')
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

    # ax.xaxis._axinfo['juggled'] = (1, 0, 2)
    # ax.yaxis._axinfo['juggled'] = (1, 1, 2)
    # ax.zaxis._axinfo['juggled'] = (1, 2, 2)
    ax.view_init(elev=15., azim=100.)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_zlim(0, 1)
    # plt.savefig('moo_synthetic.pdf')
    plt.show()
