import numpy as np
from time import time
import pickle
import torch

from solvers import epo_search
from solvers.epo_lp import mu
from solvers.pmtl_gpu import pareto_mtl_search


def get_concave_func_eval(n, m):
    shifts = np.random.uniform(-1. / n, 1. / n, (m, n))

    def concave_func_eval(x):
        xshifted = x.reshape(1, n) - shifts
        xshifted_norm = np.linalg.norm(xshifted, axis=1)
        exshifted = np.exp(-xshifted_norm**2)
        f = 1. - exshifted
        df = 2 * exshifted.reshape(m, 1) * xshifted
        # print(f"xshifted: {xshifted.shape}")
        # print(f"xshifted_norm: {xshifted_norm.shape}")
        # print(f"exshifted: {exshifted.shape}")
        # print(f"f: {f.shape}")
        # print(f"df: {df.shape}")
        return f, df

    return concave_func_eval


def get_concave_func_eval_gpu(n, m):
    shifts = (2. / n) * torch.rand(m, n).cuda() - 1. / n

    def concave_func_eval(x):
        xshifted = x.reshape(1, n) - shifts
        xshifted_norm = torch.norm(xshifted, dim=1)
        exshifted = torch.exp(-xshifted_norm**2)
        f = 1. - exshifted
        df = 2 * exshifted.reshape(m, 1) * xshifted
        # print(f"xshifted: {xshifted.shape}")
        # print(f"xshifted_norm: {xshifted_norm.shape}")
        # print(f"exshifted: {exshifted.shape}")
        # print(f"f: {f.shape}")
        # print(f"df: {df.shape}")
        return f, df

    return concave_func_eval


if __name__ == "__main__":
    n_ns = 10
    n_ms = 40

    ms = list(range(2, n_ms, 4))
    data = {"EPO Search": {m: [] for m in ms},
            "PMTL": {m: [] for m in ms}}
    timings = {"EPO Search": {m: [] for m in ms},
               "PMTL": {m: [] for m in ms}}
    for m in ms:
        print(f"m={m}")
        ns = np.random.randint(20, 50, n_ns)
        for n in ns:
            print(f"\tn={n}", end=': ')
            ss = 1. / n
            print("EPO", end=":")
            f_df = get_concave_func_eval(n, m)
            r = np.random.rand(m)
            r /= np.sum(r)
            t0 = time()
            _, res = epo_search(f_df, r=r, step_size=ss, n_dim=n,
                                max_iters=1000)
            t1 = time()
            timings["EPO Search"][m].append(t1 - t0)
            ls = res['ls']
            data["EPO Search"][m].append(r * ls[-1])

            print(f"({t1-t0}); PMTL", end=": ")
            f_df_gpu = get_concave_func_eval_gpu(n, m)
            K = 2 * m
            r_invs = torch.rand(K, m).cuda()
            k = np.random.choice(K)
            t0 = time()
            x, res = pareto_mtl_search(f_df_gpu, ref_vecs=r_invs, r=k,
                                       step_size=ss, n_dim=n, max_iters=100)
            t1 = time()
            timings["PMTL"][m].append(t1 - t0)
            ls = res['ls']
            r = 1. / r_invs[k]
            r /= torch.sum(r)
            rl = np.asarray(r.cpu() * ls[-1].cpu(), dtype=np.float)
            data["PMTL"][m].append(rl)
            print(f"({t1-t0})")

    mus = {"mean": {"EPO Search": [], "PMTL": []},
           "std": {"EPO Search": [], "PMTL": []}}
    times = {"mean": {"EPO Search": [], "PMTL": []},
             "std": {"EPO Search": [], "PMTL": []}}
    for method in data:
        for m in data[method]:
            m_mus = [mu(rl) for rl in data[method][m]]
            mus["mean"][method].append(np.mean(m_mus))
            mus["std"][method].append(np.std(m_mus))

            m_times = [t for t in timings[method][m]]
            times["mean"][method].append(np.mean(m_times))
            times["std"][method].append(np.std(m_times))
    pickle.dump((data, timings, mus, times), open("simulaiton.pkl", 'wb'))
