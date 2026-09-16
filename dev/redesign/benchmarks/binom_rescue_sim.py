"""
Simulations: guarded pipeline vs binomial genotypes vs binomial genotypes + pooled rescue.

Usage (from dev/redesign/results): PYTHONPATH=..:../benchmarks python ../benchmarks/binom_rescue_sim.py <shard> <n_shards>
"""
import sys, warnings, logging
sys.setrecursionlimit(100000); warnings.filterwarnings('ignore'); logging.disable(logging.INFO)
import numpy as np, pandas as pd, mito as mt, anndata as ad
from scipy.sparse import csr_matrix
from mito.pp.filters import annotate_vars, filter_baseline, filter_MiTo
from geno2 import em_genotype
from grid import FK, SIM, kernel, maxcompat, character_af
from joincount import carrier_nonrandomness
from carriers import binomial_carriers
from qc2_bench import graph_genotype
from refined_bench import sim_front_metrics
from cutter import evidence_cut
from compat import cell_membership
from rescue import pooled_rescue

shard, n_shards = int(sys.argv[1]), int(sys.argv[2])
configs = [(nk, topo, seed) for nk in [5, 10, 30, 50] for topo in ['polytomy', 'depth3'] for seed in [0, 1, 2]]
rows = []
for i_cfg, (nk, topo, seed) in enumerate(configs):
    if i_cfg % n_shards != shard:
        continue
    kw = dict(n_cells=1000, n_clones=nk, frac_double_variants=0.3, frac_noisy_variants=0.2, min_max_ratio_clones=0.2, random_seed=seed, **SIM)
    kw.update(dict(n_root_clones=nk, max_depth=1) if topo == 'polytomy' else dict(n_root_clones=max(2, nk//3), max_depth=3))
    s = mt.ut.simulate_afm(**kw); b = s.copy(); annotate_vars(b); b = filter_baseline(b); c = filter_MiTo(b, **FK)
    X = c.X.toarray(); AD = c.layers['AD'].toarray(); COV = c.layers['site_coverage'].toarray().astype(np.int64); n, m = X.shape
    names = np.array(c.var_names)
    truth = c.obs['clone'].astype(str).values
    sizes = pd.Series(truth).value_counts()

    def tree_and_cut(cols, Bk):
        Bk = Bk > 0
        kv = np.flatnonzero((Bk.mean(0) <= 0.5) & (Bk.sum(0) >= 2))
        al = maxcompat(Bk[:, kv].astype(np.int8)); fc = kv[al]
        Bs = Bk[:, fc].astype(np.int8); Xk = X[:, cols]
        kc = Bs.sum(1) >= 1
        out = np.full(n, 'unassigned', dtype=object)
        if kc.sum() < 10 or fc.size < 2:
            return out, fc, Bs
        a = ad.AnnData(X=csr_matrix(character_af(Bs, Xk[:, fc])[kc]), obs=c.obs.iloc[np.flatnonzero(kc)][['clone']].copy(), layers={'bin': csr_matrix(Bs[kc])})
        a.var_names = [f'{names[cols[j]]}#{i}' for i, j in enumerate(fc)]
        a.uns['genotyping'] = {'bin_method': 'x', 'binarization_kwargs': {}}
        for key in ('scLT_system', 'pp_method'):
            if key in c.uns: a.uns[key] = c.uns[key]
        mt.pp.compute_distances(a, precomputed=True, metric='weighted_jaccard', ncores=4, verbose=False)
        tree = mt.tl.build_tree(a, precomputed=True, solver='UPMGA')
        Bt = a.layers['bin'].toarray() > 0
        lab = evidence_cut(tree, Bt.astype(float), list(a.obs_names), min_in=0.25, ratio=3.0, self_min_in=0.85, ladder='descend')
        _, core = cell_membership(kernel(Xk, raw=True), Bs > 0, tau=0.0)
        lab = lab.mask(~(Bt & (core[kc] >= 0.25)).any(1), 'unassigned')
        out[np.flatnonzero(kc)] = lab.values
        return out, fc, Bs

    def evaluate(name, lab, cols, fc, Bs, extra=None):
        ok = lab != 'unassigned'
        rec = 0
        for cl, size in sizes.items():
            x = lab[truth == cl]; x = x[x != 'unassigned']
            if len(x) and len(x)/size >= .5:
                vc = pd.Series(x).value_counts()
                rec += int(vc.iloc[0]/len(x) >= .8 and (truth[lab == vc.index[0]] == cl).mean() >= .8)
        row = dict(pipeline=name, clones=nk, topo=topo, seed=seed, cells_pct=round(100*ok.mean(), 1), labels=int(pd.Series(lab[ok]).nunique()),
                   ARI=round(mt.ut.custom_ARI(pd.Series(truth[ok]), pd.Series(lab[ok])), 3) if ok.sum() > 10 else np.nan,
                   NMI=round(mt.ut.normalized_mutual_info_score(pd.Series(truth[ok]), pd.Series(lab[ok])), 3) if ok.sum() > 10 else np.nan,
                   clones_recovered=rec, clones_true=int(sizes.size), **(extra or {}))
        if fc.size:
            row.update(sim_front_metrics(s, c, cols[fc], Bs))
        rows.append(row)
        return lab

    # guarded pipeline
    g_flat, _ = em_genotype(AD, COV, np.full((n, n), 1/n), mode='bb')
    keep, _, _ = carrier_nonrandomness(X, (g_flat > 0.7).astype(np.int8), cell_depth=COV.mean(1), alpha=0.05)
    cols = np.flatnonzero(keep)
    lab, fc, Bs = tree_and_cut(cols, graph_genotype(X, AD, COV, cols)[1])
    guarded = evaluate('guarded (current)', lab, cols, fc, Bs)
    # binomial QC + binomial genotypes
    Bq, p0 = binomial_carriers(AD, COV, alpha_cell=0.001, return_background=True)
    keep, _, _ = carrier_nonrandomness(X, Bq, cell_depth=COV.mean(1), alpha=0.05)
    cols = np.flatnonzero(keep)
    Bb = Bq[:, cols]
    lab, fc, Bs = tree_and_cut(cols, Bb)
    evaluate('binomial', lab, cols, fc, Bs)
    if fc.size:
        ch = cols[fc]
        lab_r, det = pooled_rescue(lab, Bb[:, fc] > 0, AD[:, ch], COV[:, ch], p0[ch], alpha=0.01)
        evaluate('binomial + pooled rescue', lab_r, cols, fc, Bs, extra=dict(rescued=int(det.assigned.sum()) if len(det) else 0))
    # pooled rescue on the guarded labels too (background from binomial_carriers on all candidates)
    g_cols = np.flatnonzero(carrier_nonrandomness(X, (g_flat > 0.7).astype(np.int8), cell_depth=COV.mean(1), alpha=0.05)[0])
    lab_g, fc_g, Bs_g = tree_and_cut(g_cols, graph_genotype(X, AD, COV, g_cols)[1])
    if fc_g.size:
        ch = g_cols[fc_g]
        lab_gr, det = pooled_rescue(lab_g, Bs_g > 0, AD[:, ch], COV[:, ch], p0[ch], alpha=0.01)
        evaluate('guarded + pooled rescue', lab_gr, g_cols, fc_g, Bs_g, extra=dict(rescued=int(det.assigned.sum()) if len(det) else 0))
    pd.DataFrame(rows).to_csv(f'binom_rescue_sim_{shard}.csv', index=False)
    print(f'done {nk}{topo}/{seed}', flush=True)
print('ALL DONE', flush=True)
