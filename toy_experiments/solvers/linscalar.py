import numpy as np


def linscalar(multi_obj_fg, r, x=None, max_iters=100,
              n_dim=20, step_size=.1, debug=False):
    # randomly generate one solution
    x = np.random.randn(n_dim) if x is None else x
    m = 2       # number of objectives
    ls = []

    # find the Pareto optimal solution
    for t in range(max_iters):
        l, G = multi_obj_fg(x)
        d = r @ G
        if np.linalg.norm(d, ord=np.inf) < 1e-4:
            print('converged', end=', ')
            break
        x = x - step_size * d
        ls.append(l)

    res = {'ls': np.stack(ls)}
    return x, res
