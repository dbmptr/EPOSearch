import matplotlib
from math import sqrt


SPINE_COLOR = 'black'


def latexify(fig_width=None, fig_height=None, columns=1):
    """Set up matplotlib's RC params for LaTeX plotting.
    Call this before plotting a figure.

    Parameters
    ----------
    fig_width : float, optional, inches
    fig_height : float,  optional, inches
    columns : {1, 2}
    """

    # code adapted from http://www.scipy.org/Cookbook/Matplotlib/LaTeX_Examples

    # Width and max height in inches for IEEE journals taken from
    # computer.org/cms/Computer.org/Journal%20templates/transactions_art_guide.pdf

    assert(columns in [1,2])

    if fig_width is None:
        fig_width = 3.487 if columns==1 else 6.9 # width in inches

    if fig_height is None:
        golden_mean = (sqrt(5)-1.0)/2.0    # Aesthetic ratio
        fig_height = fig_width*golden_mean # height in inches

    MAX_HEIGHT_INCHES = 8.0
    if fig_height > MAX_HEIGHT_INCHES:
        print("WARNING: fig_height too large:" + fig_height + 
              "so will reduce to" + MAX_HEIGHT_INCHES + "inches.")
        fig_height = MAX_HEIGHT_INCHES
    # , '\usepackage{amsmath, amsfonts}',
    params = {'backend': 'pdf',
              'text.latex.preamble': [r'\usepackage{amsmath}',
                                      r'\usepackage{amssymb}',
                                      r'\usepackage{gensymb}',
                                      # r'\usepackage{mathabx}',
                                      r'\usepackage{amsfonts}',
                                      r'\usepackage{newtxmath}'],
              'axes.labelsize': 8,  # fontsize for x and y labels (was 10)
              'axes.titlesize': 9,
              'font.size': 10,   # was 10
              'legend.fontsize': 9,     # was 10
              'legend.shadow': False,
              'legend.fancybox': True,
              'xtick.labelsize': 6,
              'ytick.labelsize': 6,
              'text.usetex': True,
              'figure.figsize': [fig_width, fig_height],
              'font.family': 'serif',
              'font.serif': 'times new roman',
              'patch.linewidth': 0.5
              }

    matplotlib.rcParams.update(params)


def format_axes(ax, title=None, xlabel=None, ylabel=None, leg_loc=None, grid=None):

    for spine in ['top', 'right']:
        # ax.spines[spine].set_visible(False)
        ax.spines[spine].set_color(SPINE_COLOR)
        ax.spines[spine].set_linewidth(0.7)

    for spine in ['left', 'bottom']:
        ax.spines[spine].set_color(SPINE_COLOR)
        ax.spines[spine].set_linewidth(0.7)

    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')

    for axis in [ax.xaxis, ax.yaxis]:
        axis.set_tick_params(direction='out', color=SPINE_COLOR)

    if title is not None:
        ax.set_title(title)
    if xlabel is not None:
        ax.set_xlabel(xlabel, labelpad=0.4)
    if ylabel is not None:
        ax.set_ylabel(ylabel, labelpad=0.3)
    if leg_loc is not None:
        ax.legend(loc=leg_loc)
    if grid is not None:
        ax.grid(grid, lw=0.3)
    return ax
