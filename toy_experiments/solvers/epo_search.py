import numpy as np

from .epo_lp import EPO_LP


def epo_search(multi_obj_fg, r, x=None, relax=False, eps=1e-4, max_iters=100,
               n_dim=20, step_size=.1, grad_tol=1e-4, store_xs=False):
    if relax:
        print('relaxing')
    else:
        print('Restricted')
    # randomly generate one solution
    x = np.random.randn(n_dim) if x is None else x
    m = len(r)       # number of objectives
    lp = EPO_LP(m, n_dim, r, eps=eps)
    ls, mus, adjs, gammas, lambdas = [], [], [], [], []
    if store_xs:
        xs = [x]

    # find the Pareto optimal solution
    desc, asce = 0, 0
    for t in range(max_iters):
        l, G = multi_obj_fg(x)
        alpha = lp.get_alpha(l, G, relax=relax)
        if lp.last_move == "dom":
            desc += 1
        else:
            asce += 1
        ls.append(l)
        lambdas.append(np.min(r * l))
        mus.append(lp.mu_rl)
        adjs.append(lp.a.value)
        gammas.append(lp.gamma)

        d_nd = alpha @ G
        if np.linalg.norm(d_nd, ord=np.inf) < grad_tol:
            print('converged, ', end=',')
            break
        x = x - 10. * max(lp.mu_rl, 0.1) * step_size * d_nd
        if store_xs:
            xs.append(x)

    print(f'# iterations={asce+desc}; {100. * desc/(desc+asce)} % descent')
    res = {'ls': np.stack(ls),
           'mus': np.stack(mus),
           'adjs': np.stack(adjs),
           'gammas': np.stack(gammas),
           'lambdas': np.stack(lambdas)}
    if store_xs:
        res['xs': xs]
    return x, res
