"""
End-to-end benchmark of the carrier non-randomness QC (join count OR exclusivity).

Front ends compared (all followed by the same tree + evidence_cut, one_sided off/on):
  DEV current            graph-prior genotyping + imputation on all, prevalence cap, four-gamete
  no QC                  graph-prior + imputation on all, cap, split recurrent, four-gamete
  join count a=0.01      flat-prior QC (join count only) -> graph-prior + imputation on kept -> cap, split, 4-gamete
  QC2 a=0.01 / a=0.05    flat-prior QC (join count OR exclusivity) -> same downstream

Usage: python qc2_bench.py mda
       python qc2_bench.py sim <shard> <n_shards>
"""
import sys, warnings, logging
sys.setrecursionlimit(10000)
warnings.filterwarnings('ignore'); logging.disable(logging.INFO)
import numpy as np, pandas as pd, scanpy as sc, mito as mt
from mito.pp.filters import annotate_vars, filter_baseline, filter_MiTo
from geno2 import em_genotype
from grid import kernel, maxcompat, FK, SIM
from impute import impute_dropouts
from refine import split_recurrent_graph
from joincount import carrier_nonrandomness
from refined_bench import cluster, sim_front_metrics, as_variants


def graph_genotype(X, AD, COV, cols):
    Xk = X[:, cols]
    g, _ = em_genotype(AD[:, cols], COV[:, cols], kernel(Xk), mode='bb')
    return Xk, impute_dropouts(Xk, (g > 0.7).astype(np.int8), k=30, thr=0.6, min_support=1)[0]


def downstream(cols, X, AD, COV, split=True):
    Xk, Bk = graph_genotype(X, AD, COV, cols)
    kv = np.flatnonzero(Bk.astype(bool).mean(0) <= 0.5)
    Bp, Xp, cp = Bk[:, kv], Xk[:, kv], cols[kv]
    if split:
        Bp, origin, _ = split_recurrent_graph(Xp, Bp)
    else:
        origin = np.arange(Bp.shape[1])
    al = maxcompat(Bp)
    Bs, origin = Bp[:, al], origin[al]
    return cp[origin], Bs, np.where(Bs > 0, Xp[:, origin], 0.0), Xk


def front_ends(afm):
    b = afm.copy(); annotate_vars(b); b = filter_baseline(b); c = filter_MiTo(b, **FK)
    X = c.X.toarray(); AD = c.layers['AD'].toarray(); COV = c.layers['site_coverage'].toarray().astype(np.int64)
    n, m = X.shape
    allc = np.arange(m)
    out = {}
    out['DEV current'] = (c,) + downstream(allc, X, AD, COV, split=False)
    out['no QC (+split)'] = (c,) + downstream(allc, X, AD, COV)
    g_flat, _ = em_genotype(AD, COV, np.full((n, n), 1/n), mode='bb')
    Bf = (g_flat > 0.7).astype(np.int8)
    keep, p_join, p_excl = carrier_nonrandomness(X, Bf, cell_depth=COV.mean(1), alpha=0.01)
    qc = {}
    for name, kk, split in [('join count a=0.01', p_join <= 0.01, True),
                            ('QC2 a=0.01', np.minimum(p_join, p_excl) <= 0.005, True),
                            ('QC2 a=0.05', np.minimum(p_join, p_excl) <= 0.025, True),
                            ('QC2 a=0.05 no split', np.minimum(p_join, p_excl) <= 0.025, False)]:
        cols = np.flatnonzero(kk)
        qc[name] = dict(n_kept=int(cols.size), n_join=int((p_join <= 0.01).sum()), n_excl_only=int(((p_excl <= 0.005) & (p_join > 0.005)).sum()))
        if cols.size >= 2:
            out[name] = (c,) + downstream(cols, X, AD, COV, split=split)
    return c, out, qc, (p_join, p_excl)


if __name__ == '__main__':
    target = sys.argv[1]
    rows = []
    if target == 'mda':
        D = '/Users/cossa/Desktop/projects/MiTo/data_test/afm_unfiltered.h5ad'
        base = mt.pp.filter_cells(sc.read(D), cell_filter='filter2')
        gbc = base.obs['GBC'].astype(str)
        GT = set(mt.pp.filter_afm(base.copy(), filtering='GT_enriched', lineage_column='GBC', ncores=4).var_names)
        c, out, qc, (pj, pe) = front_ends(base)
        print('QC:', qc, flush=True)
        pd.DataFrame(dict(var=c.var_names, GT=[v in GT for v in c.var_names], p_join=pj, p_excl=pe)).to_csv('qc2_mda_pvals.csv', index=False)
        for name, (c_, cols, B, Xc, Xg) in out.items():
            vars_, M = as_variants(c_, cols, B)
            gb = gbc.loc[c_.obs_names].values
            spec, clones = [], set()
            for j, v in enumerate(vars_):
                pos = M[:, j]
                if pos.sum() < 3:
                    continue
                modal = pd.Series(gb[pos]).value_counts().index[0]
                spec.append(pos[gb == modal].sum()/pos.sum())
                if v in GT:
                    clones.add(modal)
            front = dict(pipeline=name, n_vars=len(vars_), GT_enriched_kept=len(set(vars_) & GT),
                         call_clone_specificity=round(float(np.mean(spec)), 3), GBC_clones_marked=len(clones))
            for r in cluster(c_, cols, B, Xc, Xg, 'GBC', base.shape[0]):
                rows.append({**front, **r})
        pd.DataFrame(rows).to_csv('qc2_mda.csv', index=False)
    else:
        shard, n_shards = int(sys.argv[2]), int(sys.argv[3])
        configs = [(nk, topo, seed) for nk in [5, 10, 30, 50] for topo in ['polytomy', 'depth3'] for seed in [0, 1, 2]]
        for i, (nk, topo, seed) in enumerate(configs):
            if i % n_shards != shard:
                continue
            kw = dict(n_cells=1000, n_clones=nk, frac_double_variants=0.3, frac_noisy_variants=0.2,
                      min_max_ratio_clones=0.2, random_seed=seed, **SIM)
            kw.update(dict(n_root_clones=nk, max_depth=1) if topo == 'polytomy'
                      else dict(n_root_clones=max(2, nk//3), max_depth=3))
            s = mt.ut.simulate_afm(**kw)
            c, out, qc, _ = front_ends(s)
            V = s.var.loc[c.var_names]
            per_clone = V[~V.is_noise].groupby('clone_of_origin').size()
            for name, (c_, cols, B, Xc, Xg) in out.items():
                vars_ = list(dict.fromkeys(c_.var_names[cols]))
                n_sole = sum(1 for v in vars_ if not V.loc[v, 'is_noise'] and per_clone[V.loc[v, 'clone_of_origin']] == 1)
                front = dict(pipeline=name, clones=nk, topo=topo, seed=seed, n_sole_total=int((per_clone == 1).sum()),
                             n_sole_kept=n_sole, **qc.get(name, {}), **sim_front_metrics(s, c_, cols, B))
                for r in cluster(c_, cols, B, Xc, Xg, 'clone', 1000):
                    rows.append({**front, **r})
            pd.DataFrame(rows).to_csv(f'qc2guard_sim_{shard}.csv', index=False)
            print(f'done {nk}{topo}/{seed}', flush=True)
    print('ALL DONE', flush=True)
