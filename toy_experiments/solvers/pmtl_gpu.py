# import numpy as np

import torch
import torch.utils.data

from .min_norm_solvers import MinNormSolver
# from time import time


def get_d_paretomtl_init(grads, value, weights, i):
    """
    calculate the gradient direction for ParetoMTL initialization
    """

    flag = False
    nobj = value.shape

    # check active constraints
    current_weight = weights[i]
    rest_weights = weights
    w = rest_weights - current_weight

    gx = torch.matmul(w, value / torch.norm(value))
    idx = gx > 0

    # calculate the descent direction
    if torch.sum(idx) <= 0:
        flag = True
        return flag, torch.zeros(nobj)
    if torch.sum(idx) == 1:
        sol = torch.ones(1).cuda().float()
    else:
        vec = torch.matmul(w[idx], grads)
        sol, nd = MinNormSolver.find_min_norm_element(
            [[vec[t]] for t in range(len(vec))])

    # weight0 =  torch.sum(torch.stack([sol[j] * w[idx][j ,0] for j in torch.arange(0, torch.sum(idx))]))
    # weight1 =  torch.sum(torch.stack([sol[j] * w[idx][j ,1] for j in torch.arange(0, torch.sum(idx))]))
    # weight = torch.stack([weight0,weight1])

    new_weights = []
    for t in range(len(value)):
        new_weights.append(torch.sum(torch.stack(
            [sol[j] * w[idx][j, t] for j in torch.arange(0, torch.sum(idx))])))

    return flag, torch.stack(new_weights)


def get_d_paretomtl(grads, value, weights, i):
    """ calculate the gradient direction for ParetoMTL """

    # check active constraints
    current_weight = weights[i]
    rest_weights = weights
    w = rest_weights - current_weight

    gx = torch.matmul(w, value / torch.norm(value))
    idx = gx > 0

    # calculate the descent direction
    if torch.sum(idx) <= 0:
        sol, nd = MinNormSolver.find_min_norm_element(
            [[grads[t]] for t in range(len(grads))])
        return torch.tensor(sol).cuda().float()

    vec = torch.cat((grads, torch.matmul(w[idx], grads)))
    sol, nd = MinNormSolver.find_min_norm_element(
        [[vec[t]] for t in range(len(vec))])

    # weight0 =  sol[0] + torch.sum(torch.stack([sol[j] * w[idx][j - 2 ,0] for j in torch.arange(2, 2 + torch.sum(idx))]))
    # weight1 =  sol[1] + torch.sum(torch.stack([sol[j] * w[idx][j - 2 ,1] for j in torch.arange(2, 2 + torch.sum(idx))]))
    # weight = torch.stack([weight0,weight1])

    new_weights = []
    for t in range(len(value)):
        new_weights.append(sol[t] + torch.sum(torch.stack([sol[j] * w[idx][j, t]
                                                           for j in torch.arange(0, torch.sum(idx))])))

    return torch.stack(new_weights)


def pareto_mtl_search(multi_obj_fg, ref_vecs, r, x=None,
                      max_iters=200, n_dim=20, step_size=1):
    """
    Pareto MTL
    """
    # m = len(r)
    # randomly generate one solution
    x = torch.randn(n_dim).cuda() / n_dim if x is None else x
    # x = np.random.uniform(0.1,.5,n_dim) if x is None else x

    fs_init, fs = [], []

    # find the initial solution
    for t in range(int(max_iters * 0.2)):
        f, f_dx = multi_obj_fg(x)
        flag, weights = get_d_paretomtl_init(f_dx, f, ref_vecs, r)
        if flag:
            # print("fealsible solution is obtained.")
            break
        x = x - step_size * (weights @ f_dx)
        fs_init.append(f)

    # find the Pareto optimal solution
    for t in range(int(max_iters * 0.8)):
        f, f_dx = multi_obj_fg(x)
        weights = get_d_paretomtl(f_dx, f, ref_vecs, r)
        x = x - step_size * (weights @ f_dx)
        fs.append(f)

    res = {'ls': torch.stack(fs)}   # 'ls_init': torch.stack(fs_init),
    return x, res
