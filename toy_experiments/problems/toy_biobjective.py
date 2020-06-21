import autograd.numpy as np
from autograd import grad

import matplotlib.pyplot as plt
import matplotlib as mpl
from labellines import labelLines   # , labelLine,
from latex_utils import latexify


def f1(x):
    n = len(x)
    dx = np.linalg.norm(x - 1. / np.sqrt(n))
    return 1 - np.exp(-dx**2)


def f2(x):
    n = len(x)
    dx = np.linalg.norm(x + 1. / np.sqrt(n))
    return 1 - np.exp(-dx**2)


# calculate the gradients using autograd
f1_dx = grad(f1)
f2_dx = grad(f2)


def concave_fun_eval(x):
    """
    return the function values and gradient values
    """
    return np.stack([f1(x), f2(x)]), np.stack([f1_dx(x), f2_dx(x)])


# ### create the ground truth Pareto front ###
def create_pf(side_nonpf=False):
    """
    if `side_nonpf` is True, then the boundary of attainable objectives,
    which lie adjacent to the PF, is also returned.
    """
    def map_to_objective_space(xs):
        fs = []
        for x1 in xs:
            x = np.array([x1, x1])
            f, f_dx = concave_fun_eval(x)
            fs.append(f)

        return np.array(fs)

    if side_nonpf:
        ps = np.linspace(-1. / np.sqrt(2), 1. / np.sqrt(2), 30, endpoint=True)
    else:
        ps = np.linspace(-1 / np.sqrt(2), 1 / np.sqrt(2))

    pf = map_to_objective_space(ps)

    if side_nonpf:
        s_left = np.linspace(-1.5 / np.sqrt(2), -1. / np.sqrt(2), 10,
                             endpoint=True)
        fs_left = map_to_objective_space(s_left)

        s_right = np.linspace(1. / np.sqrt(2), 1.5 / np.sqrt(2), 10,
                              endpoint=True)
        fs_right = map_to_objective_space(s_right)

        return pf, fs_left, fs_right

    return pf


def circle_points(K, min_angle=None, max_angle=None):
    # generate evenly distributed preference vector
    ang0 = np.pi / 20. if min_angle is None else min_angle
    ang1 = np.pi * 9 / 20. if max_angle is None else max_angle
    angles = np.linspace(ang0, ang1, K, endpoint=True)
    x = np.cos(angles)
    y = np.sin(angles)
    return np.c_[x, y]


def add_interval(ax, xdata, ydata,
                 color="k", caps="  ", label='', side="both", lw=2):
    line = ax.add_line(mpl.lines.Line2D(xdata, ydata))
    line.set_label(label)
    line.set_color(color)
    line.set_linewidth(lw)
    anno_args = {
        'ha': 'center',
        'va': 'center',
        'size': 12,
        'color': line.get_color()
    }
    a = []
    if side in ["left", "both"]:
        a0 = ax.annotate(caps[0], xy=(xdata[0], ydata[0]), zorder=2, **anno_args)
        a.append(a0)
    if side in ["right", "both"]:
        a1 = ax.annotate(caps[1], xy=(xdata[1], ydata[1]), zorder=2, **anno_args)
        a.append(a1)
    return (line, tuple(a))


if __name__ == '__main__':
    theta = np.linspace(-3, 3, 100)
    l1 = [f1(np.array([x])) for x in theta]
    l2 = [f2(np.array([x])) for x in theta]

    latexify(fig_width=2.25, fig_height=1.8)
    fig, ax = plt.subplots()
    fig.subplots_adjust(left=0.025, bottom=.12, right=.975, top=.975)
    ax.spines['left'].set_position('zero')
    ax.spines['right'].set_color('none')    # turn off the right spine/ticks
    ax.spines['top'].set_color('none')      # turn off the top spine/ticks
    ax.xaxis.tick_bottom()

    ax.plot(theta, l1, label=r'$l_1(\theta)$', lw=2)
    ax.plot(theta, l2, label=r'$l_2(\theta)$', lw=2)

    # Comment out the twin axis code below for ppt
    axt = ax.twinx()
    axt.axis("off")
    add_interval(axt, (-3, -1.05), (0.02, 0.02), "r", "()",
                 r"$\theta^0 \preccurlyeq -\mathbf{1}/\sqrt{n}$ or " +
                 r"$\theta^0 \succcurlyeq \mathbf{1}/\sqrt{n}$", side="right")
    add_interval(axt, (1.05, 3), (0.02, 0.02), "r", "()", side="left")
    add_interval(axt, (-.95, .95), (0.02, 0.02), "g", "()",
                 r"$-\mathbf{1}/\sqrt{n} \preccurlyeq \theta^0" +
                 r"\preccurlyeq \mathbf{1}/\sqrt{n}$", side="both")
    axt.legend(loc=(0.09, 0.2))

    labelLines(ax.get_lines(), xvals=[2, -2], align=False)
    ax.set_xlabel(r'$\theta$')
    ax.xaxis.set_label_coords(0.99, -0.03)

    plt.savefig('../figures/moo_synthetic.pdf')   # for paper
    # plt.savefig('../figures/moo_synthetic_ppt_just_losses.pdf')     # for ppt
    plt.show()

