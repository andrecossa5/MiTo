"""Simulations: QC carriers = guarded flat calls vs AD>=2 vs AD>=1 (alpha=0.05, no split), guarded genotyper downstream."""
import sys, warnings, logging; sys.setrecursionlimit(10000); warnings.filterwarnings('ignore'); logging.disable(logging.INFO)
import numpy as np, pandas as pd, mito as mt
from mito.pp.filters import annotate_vars, filter_baseline, filter_MiTo
from geno2 import em_genotype
from grid import FK, SIM
from joincount import carrier_nonrandomness
from carriers import binomial_carriers
from qc2_bench import downstream
from refined_bench import cluster, sim_front_metrics
shard, n_shards = int(sys.argv[1]), int(sys.argv[2])
configs = [(nk, topo, seed) for nk in [5, 10, 30, 50] for topo in ['polytomy', 'depth3'] for seed in [0, 1, 2]]
import os
MINC = None if os.environ.get('MINC', '0.5') == 'None' else float(os.environ.get('MINC', '0.5'))
TAG = os.environ.get('TAG', '')
rows = []
for i, (nk, topo, seed) in enumerate(configs):
    if i % n_shards != shard:
        continue
    kw = dict(n_cells=1000, n_clones=nk, frac_double_variants=0.3, frac_noisy_variants=0.2, min_max_ratio_clones=0.2, random_seed=seed, **SIM)
    kw.update(dict(n_root_clones=nk, max_depth=1) if topo == 'polytomy' else dict(n_root_clones=max(2, nk//3), max_depth=3))
    s = mt.ut.simulate_afm(**kw); b = s.copy(); annotate_vars(b); b = filter_baseline(b); c = filter_MiTo(b, **FK)
    X = c.X.toarray(); AD = c.layers['AD'].toarray(); COV = c.layers['site_coverage'].toarray().astype(np.int64); n, m = X.shape
    V = s.var.loc[c.var_names]; per_clone = V[~V.is_noise].groupby('clone_of_origin').size()
    g_flat, _ = em_genotype(AD, COV, np.full((n, n), 1/n), mode='bb')
    _ALL = {'guard flat': lambda: (g_flat > 0.7).astype(np.int8), 'AD>=2': lambda: (AD >= 2).astype(np.int8), 'AD>=1': lambda: (AD >= 1).astype(np.int8),
            'binom0.001': lambda: binomial_carriers(AD, COV, alpha_cell=0.001), 'binom0.01': lambda: binomial_carriers(AD, COV, alpha_cell=0.01)}
    for qc_name, Bq in [(k_, _ALL[k_]()) for k_ in os.environ.get('QCDEFS', 'guard flat,AD>=2,AD>=1').split(',')]:
        keep, _, _ = carrier_nonrandomness(X, Bq, cell_depth=COV.mean(1), alpha=0.05, min_concentration=MINC)
        cols = np.flatnonzero(keep)
        if cols.size < 2:
            continue
        cols_f, Bs, Xs, Xk = downstream(cols, X, AD, COV, split=False)
        vars_ = list(dict.fromkeys(c.var_names[cols_f]))
        n_sole = sum(1 for v in vars_ if not V.loc[v, 'is_noise'] and per_clone[V.loc[v, 'clone_of_origin']] == 1)
        front = dict(qc_carriers=qc_name, clones=nk, topo=topo, seed=seed, qc_kept=int(cols.size),
                     qc_noise_kept=int(V.is_noise.values[cols].sum()), n_sole_total=int((per_clone == 1).sum()), n_sole_kept=n_sole,
                     **sim_front_metrics(s, c, cols_f, Bs))
        for r in cluster(c, cols_f, Bs, Xs, Xk, 'clone', 1000):
            rows.append({**front, **r})
    pd.DataFrame(rows).to_csv(f'adqc_sim{TAG}_{shard}.csv', index=False)
    print(f'done {nk}{topo}/{seed}', flush=True)
print('ALL DONE', flush=True)
