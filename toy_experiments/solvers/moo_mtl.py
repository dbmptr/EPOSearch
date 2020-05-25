# This code is from
# Multi-Task Learning as Multi-Objective Optimization
# Ozan Sener, Vladlen Koltun
# Neural Information Processing Systems (NeurIPS) 2018 
# https://github.com/intel-isl/MultiObjectiveOptimization

import numpy as np

from .min_norm_solvers_numpy import MinNormSolver


def moo_mtl_search(multi_obj_fg, x=None,
                   max_iters=200, n_dim=20, step_size=1):
    """
    MOO-MTL
    """
    # x = np.random.uniform(-0.5,0.5,n_dim)
    x = np.random.randn(n_dim) if x is None else x
    fs = []
    for t in range(max_iters):
        f, f_dx = multi_obj_fg(x)

        weights = get_d_moomtl(f_dx)

        x = x - step_size * np.dot(weights.T, f_dx).flatten()
        fs.append(f)

    res = {'ls': np.stack(fs)}
    return x, res


def get_d_moomtl(grads):
    """
    calculate the gradient direction for MOO-MTL
    """

    nobj, dim = grads.shape
    if nobj <= 1:
        return np.array([1.])

#    # use cvxopt to solve QP
#    P = np.dot(grads , grads.T)
#
#    q = np.zeros(nobj)
#
#    G =  - np.eye(nobj)
#    h = np.zeros(nobj)
#
#
#    A = np.ones(nobj).reshape(1,2)
#    b = np.ones(1)
#
#    cvxopt.solvers.options['show_progress'] = False
#    sol = cvxopt_solve_qp(P, q, G, h, A, b)
    # print(f'grad.shape: {grads.shape}')
    # use MinNormSolver to solve QP
    sol, nd = MinNormSolver.find_min_norm_element(grads)

    return sol
