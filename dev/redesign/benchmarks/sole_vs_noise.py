"""
Sole-marker clones vs noise: (1) graph-free discriminating features, (2) what graph-prior
genotyping + imputation do to each, (3) oracle variant sets -> final ARI.

Usage: python sole_vs_noise.py <shard> <n_shards>
"""
import sys, warnings, logging
sys.setrecursionlimit(10000)
warnings.filterwarnings('ignore'); logging.disable(logging.INFO)
import numpy as np, pandas as pd, mito as mt
from mito.pp.filters import annotate_vars, filter_baseline, filter_MiTo
from geno2 import em_genotype
from grid import kernel, maxcompat, FK, SIM
from impute import impute_dropouts
from joincount import join_count_test
from refined_bench import cluster

N_PERM = 999


def exclusivity(B, j, rng):
    """Mean number of OTHER calls in carriers of j vs random sets of the same size (two-sided p, z)."""
    Bb = B > 0
    other = Bb.sum(1) - Bb[:, j]
    C = np.flatnonzero(Bb[:, j])
    obs = other[C].mean()
    null = np.array([other[rng.choice(len(other), C.size, replace=False)].mean() for _ in range(N_PERM)])
    z = (obs - null.mean())/max(null.std(), 1e-9)
    p_low = (1 + (null <= obs).sum())/(1 + N_PERM)      # depletion: fewer other calls than random
    return obs, null.mean(), z, p_low


shard, n_shards = int(sys.argv[1]), int(sys.argv[2])
configs = [(nk, topo, seed) for nk in [5, 10, 30, 50] for topo in ['polytomy', 'depth3'] for seed in [0, 1]]
feat_rows, geno_rows, ari_rows = [], [], []
for i, (nk, topo, seed) in enumerate(configs):
    if i % n_shards != shard:
        continue
    rng = np.random.default_rng(seed)
    kw = dict(n_cells=1000, n_clones=nk, frac_double_variants=0.3, frac_noisy_variants=0.2,
              min_max_ratio_clones=0.2, random_seed=seed, **SIM)
    kw.update(dict(n_root_clones=nk, max_depth=1) if topo == 'polytomy'
              else dict(n_root_clones=max(2, nk//3), max_depth=3))
    s = mt.ut.simulate_afm(**kw); b = s.copy(); annotate_vars(b); b = filter_baseline(b); c = filter_MiTo(b, **FK)
    V = s.var.loc[c.var_names]
    per_clone = V[~V.is_noise].groupby('clone_of_origin').size()
    kind = np.array(['noise' if V.loc[v, 'is_noise'] else
                     ('sole' if per_clone[V.loc[v, 'clone_of_origin']] == 1 else 'multi') for v in c.var_names])
    G = s[c.obs_names, c.var_names].layers['genotype']; G = (G.toarray() if hasattr(G, 'toarray') else np.asarray(G)) > 0
    X = c.X.toarray(); AD = c.layers['AD'].toarray(); COV = c.layers['site_coverage'].toarray().astype(np.int64)
    n = X.shape[0]
    tag = dict(cfg=f'{nk}{topo[:4]}', seed=seed)

    # ---- (1) features on flat-prior calls
    g_flat, _ = em_genotype(AD, COV, np.full((n, n), 1/n), mode='bb')
    Bf = (g_flat > 0.7).astype(np.int8)
    _, _, p_jc = join_count_test(X, Bf, graph='loo', n_perm=N_PERM)
    for j in range(len(kind)):
        C = Bf[:, j] > 0
        if C.sum() < 2:
            continue
        obs, exp, z, p_low = exclusivity(Bf, j, rng)
        feat_rows.append(dict(**tag, kind=kind[j], carriers=int(C.sum()), jc_p=p_jc[j],
                              other_calls=obs, other_calls_random=exp, excl_z=z, excl_p=p_low,
                              frac_carriers_no_other=float(((Bf[C].sum(1) - 1) == 0).mean()),
                              frac_all_cells_no_other=float((np.delete(Bf, j, 1).sum(1) == 0).mean()),
                              median_AF_carriers=float(np.median(X[C, j])),
                              median_AD_carriers=float(np.median(AD[C, j])),
                              mean_posterior_carriers=float(g_flat[C, j].mean())))

    # ---- (2) genotype quality by stage, all variants kept
    g_graph, _ = em_genotype(AD, COV, kernel(X), mode='bb'); Bg = (g_graph > 0.7).astype(np.int8)
    Bi = impute_dropouts(X, Bg, k=30, thr=0.6, min_support=1)[0]
    for stage, Bs in [('flat prior', Bf), ('graph prior', Bg), ('graph + imputation', Bi)]:
        for kd in ['sole', 'multi', 'noise']:
            sel = kind == kd
            if not sel.any():
                continue
            p_, t_ = Bs[:, sel] > 0, G[:, sel]
            geno_rows.append(dict(**tag, stage=stage, kind=kd, recall=(p_ & t_).sum()/max(t_.sum(), 1),
                                  precision=(p_ & t_).sum()/max(p_.sum(), 1), calls=int(p_.sum()), true=int(t_.sum())))

    # ---- (3) oracle variant sets through the same downstream pipeline
    jc_keep = p_jc <= 0.01
    sets = {'all variants (no QC)': np.ones(len(kind), bool),
            'true clonal only (oracle: no noise)': kind != 'noise',
            'no noise, no sole markers': kind == 'multi',
            'noise kept, sole markers removed': kind != 'sole',
            'join count LOO a=0.01': jc_keep,
            'join count + sole markers rescued': jc_keep | (kind == 'sole')}
    for name, keep in sets.items():
        cols = np.flatnonzero(keep)
        if cols.size < 2:
            continue
        Xk = X[:, cols]; ADk = AD[:, cols]; COVk = COV[:, cols]
        gk, _ = em_genotype(ADk, COVk, kernel(Xk), mode='bb')
        Bk = impute_dropouts(Xk, (gk > 0.7).astype(np.int8), k=30, thr=0.6, min_support=1)[0]
        kv = np.flatnonzero(Bk.astype(bool).mean(0) <= 0.5)
        al = maxcompat(Bk[:, kv]); final = kv[al]
        meta = dict(**tag, variant_set=name, n_vars=int(cols.size), n_noise=int((kind[cols] == 'noise').sum()),
                    n_sole=int((kind[cols] == 'sole').sum()), n_final=int(final.size))
        for r in cluster(c, cols[final], Bk[:, final], Xk[:, final], Xk, 'clone', 1000):
            ari_rows.append({**meta, **r})
    pd.DataFrame(feat_rows).to_csv(f'svn_feat_{shard}.csv', index=False)
    pd.DataFrame(geno_rows).to_csv(f'svn_geno_{shard}.csv', index=False)
    pd.DataFrame(ari_rows).to_csv(f'svn_ari_{shard}.csv', index=False)
    print(f'done {nk}{topo}/{seed}', flush=True)
print('ALL DONE', flush=True)
