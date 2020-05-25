# This code is from 
# Pareto Multi-Task LearningLin
# Xi Lin, Hui-Ling Zhen, Zhenhua Li, Qingfu Zhang, Sam Kwong
# Neural Information Processing Systems (NeurIPS) 2019
# https://github.com/Xi-L/ParetoMTL

import numpy as np
# import cvxpy as cp

from .min_norm_solvers_numpy import MinNormSolver


def get_d_paretomtl(grads,value,weights,r):
    # calculate the gradient direction for Pareto MTL
    nobj, dim = grads.shape
    
    # check active constraints
    normalized_current_weight = r/np.linalg.norm(r)
    normalized_rest_weights = weights / np.linalg.norm(weights, axis=1, keepdims=True)
    w = normalized_rest_weights - normalized_current_weight
    
    
    # solve QP 
    gx =  np.dot(w,value/np.linalg.norm(value))
    idx = gx >  0
   
    
    vec =  np.concatenate((grads, np.dot(w[idx],grads)), axis = 0)
    # use MinNormSolver to solve QP
    sol, nd = MinNormSolver.find_min_norm_element(vec)
   
    
    # reformulate ParetoMTL as linear scalarization method, return the weights
    weight0 =  sol[0] + np.sum(np.array([sol[j] * w[idx][j - 2,0] for j in np.arange(2,2 + np.sum(idx))]))
    weight1 = sol[1] + np.sum(np.array([sol[j] * w[idx][j - 2,1] for j in np.arange(2,2 + np.sum(idx))]))
    weight = np.stack([weight0,weight1])
   

    return weight


def get_d_paretomtl_init(grads,value,weights,r):
    # calculate the gradient direction for Pareto MTL initialization
    nobj, dim = grads.shape
    
    # check active constraints
    normalized_current_weight = r/np.linalg.norm(r)
    normalized_rest_weights = weights / np.linalg.norm(weights, axis=1, keepdims=True)
    w = normalized_rest_weights - normalized_current_weight
    
    gx =  np.dot(w,value/np.linalg.norm(value))
    idx = gx >  0
    
    if np.sum(idx) <= 0:
        return np.zeros(nobj)
    if np.sum(idx) == 1:
        sol = np.ones(1)
    else:
        vec =  np.dot(w[idx],grads)
        sol, nd = MinNormSolver.find_min_norm_element(vec)

    # calculate the weights
    weight0 =  np.sum(np.array([sol[j] * w[idx][j ,0] for j in np.arange(0, np.sum(idx))]))
    weight1 =  np.sum(np.array([sol[j] * w[idx][j ,1] for j in np.arange(0, np.sum(idx))]))
    weight = np.stack([weight0,weight1])
   

    return weight


def pareto_mtl_search(multi_obj_fg, ref_vecs, r, x=None,
                      max_iters = 200, n_dim = 20, step_size = 1):
    """
    Pareto MTL
    """

    # randomly generate one solution
    x = np.random.randn(n_dim) if x is None else x
    # x = np.random.uniform(0.1,.5,n_dim) if x is None else x

    fs_init, fs = [], []
    
    # find the initial solution
    for t in range(int(max_iters * 0.2)):
        f, f_dx = multi_obj_fg(x)
        weights =  get_d_paretomtl_init(f_dx,f,ref_vecs,r)
        x = x - step_size * np.dot(weights.T,f_dx).flatten()
        fs_init.append(f)
    
    # find the Pareto optimal solution
    for t in range(int(max_iters * 0.8)):
        f, f_dx = multi_obj_fg(x)
        weights =  get_d_paretomtl(f_dx,f,ref_vecs,r)
        x = x - step_size * np.dot(weights.T,f_dx).flatten()
        fs.append(f)

    res = {'ls_init': np.stack(fs_init),
           'ls': np.stack(fs)}
    return x, res
