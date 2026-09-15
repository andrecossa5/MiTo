"""
Local variant QC (refine.py) vs the current DEV front end: variant/genotype metrics and ARI.

Usage: python refined_bench.py mda | sim
"""
import sys, os, time, warnings, logging
sys.setrecursionlimit(10000)
warnings.filterwarnings('ignore'); logging.disable(logging.INFO)
import numpy as np, pandas as pd, scanpy as sc, mito as mt
import anndata as ad
from scipy.sparse import csr_matrix
from mito.pp.filters import annotate_vars, filter_baseline, filter_MiTo
from geno2 import em_genotype
from grid import kernel, maxcompat, FK, SIM
from impute import impute_dropouts
from cutter import evidence_cut
from compat import cell_membership
from refine import neighbourhood_concordance, split_recurrent_graph

ALPHA = 0.01
TAU = 0.25


def genotype(c, cols):
    AD = c.layers['AD'].toarray()[:, cols]; COV = c.layers['site_coverage'].toarray()[:, cols].astype(np.int64)
    X = c.X.toarray()[:, cols]
    g, _ = em_genotype(AD, COV, kernel(X), mode='bb')
    B = (g > 0.7).astype(np.int8)
    return X, impute_dropouts(X, B, k=30, thr=0.6, min_support=1)[0]


def front_ends(afm):
    """Each variant returns (cells mask over c, char names, origin variant per char, B chars, AF chars, X graph)."""
    b = afm.copy(); annotate_vars(b); b = filter_baseline(b); c = filter_MiTo(b, **FK)
    allcols = np.arange(c.shape[1])
    out = {}

    # current DEV: genotype + impute on all, prevalence cap, max-conflict four-gamete
    X, B = genotype(c, allcols)
    kv = np.flatnonzero(B.astype(bool).mean(0) <= 0.5)
    al = maxcompat(B[:, kv]); cols = kv[al]
    out['DEV imp.6 (current)'] = (c, cols, B[:, cols], X[:, cols], X)

    # concordance rejection on first-pass calls, then one refinement (re-genotype + impute without rejected)
    Bfirst = B if False else None
    X0 = c.X.toarray()
    AD = c.layers['AD'].toarray(); COV = c.layers['site_coverage'].toarray().astype(np.int64)
    g0, _ = em_genotype(AD, COV, kernel(X0), mode='bb')
    dec = neighbourhood_concordance(X0, (g0 > 0.7).astype(np.int8), n_rounds=3)[0]
    keep = np.flatnonzero(dec != 'reject')
    X2, B2 = genotype(c, keep)
    kv2 = np.flatnonzero(B2.astype(bool).mean(0) <= 0.5)
    cols2 = keep[kv2]; Bp = B2[:, kv2]; Xp = X2[:, kv2]

    out['CONC, no 4-gamete'] = (c, cols2, Bp, Xp, X2)
    al = maxcompat(Bp)
    out['CONC + 4-gamete'] = (c, cols2[al], Bp[:, al], Xp[:, al], X2)
    Bs, origin, info = split_recurrent_graph(Xp, Bp)
    al = maxcompat(Bs)
    Bs, origin = Bs[:, al], origin[al]
    out['CONC + split + 4-gamete'] = (c, cols2[origin], Bs, np.where(Bs > 0, Xp[:, origin], 0.0), X2)
    kvc = np.flatnonzero(B.astype(bool).mean(0) <= 0.5)
    Bc, oc, info_c = split_recurrent_graph(X[:, kvc], B[:, kvc])
    alc = maxcompat(Bc); Bc, oc = Bc[:, alc], oc[alc]
    out['DEV imp.6 + split'] = (c, kvc[oc], Bc, np.where(Bc > 0, X[:, kvc][:, oc], 0.0), X)
    print('rejected:', list(c.var_names[dec == 'reject']), '| split (CONC):',
          {str(c.var_names[cols2[j]]): v for j, v in info.items()}, flush=True)
    out['_qc'] = dict(n_mito=c.shape[1], n_kept_conc=len(keep))
    return out


def as_variants(c, cols, B):
    """Merge split characters back to their source variant (OR), for comparison with truth."""
    names = c.var_names[cols]
    uniq = list(dict.fromkeys(names))
    M = np.column_stack([(B[:, names == v] > 0).any(1) for v in uniq]) if uniq else np.zeros((B.shape[0], 0), bool)
    return uniq, M


def cluster(c, cols, B, Xc, Xg, truth_col, n_tot, **kw):
    keep = (B > 0).sum(1) >= 1
    if keep.sum() < 30 or B.shape[1] < 2:
        return []
    names = [f'{v}#{i}' for i, v in enumerate(c.var_names[cols])]
    a = ad.AnnData(X=csr_matrix(Xc[keep]), obs=c.obs.iloc[np.flatnonzero(keep)].copy())
    a.var_names = names
    a.layers['bin'] = csr_matrix((B[keep] > 0).astype(np.int8))
    a.uns['genotyping'] = {'bin_method': 'knn', 'binarization_kwargs': {}}
    for key in ('scLT_system', 'pp_method'):
        if key in c.uns:
            a.uns[key] = c.uns[key]
    mt.pp.compute_distances(a, precomputed=True, metric='weighted_jaccard', ncores=4, verbose=False)
    tree = mt.tl.build_tree(a, precomputed=True, solver='UPMGA')
    Bk = a.layers['bin'].toarray() > 0
    gt = pd.Series(np.asarray(a.obs[truth_col].astype(str)), index=a.obs_names)
    _, core = cell_membership(kernel(Xg, raw=True), B > 0, tau=0.0)
    bad = ~(Bk & (core[keep] >= TAU)).any(1)
    rows = []
    for one_sided in (False, True):
        lab = evidence_cut(tree, Bk.astype(float), list(a.obs_names), min_in=0.25, ratio=3.0,
                           self_min_in=0.85, ladder='descend', one_sided=one_sided)
        lab[bad] = 'unassigned'
        ok = (lab != 'unassigned').values
        rows.append(dict(one_sided=one_sided, cells_pct=round(100*ok.sum()/n_tot, 1), labels=int(lab[ok].nunique()),
                         ARI=round(mt.ut.custom_ARI(gt[ok], lab[ok]), 3),
                         NMI=round(mt.ut.normalized_mutual_info_score(gt[ok], lab[ok]), 3), **kw))
    return rows


def sim_front_metrics(s, c, cols, B):
    V = s.var
    vars_, M = as_variants(c, cols, B)
    keep = M.sum(1) >= 1
    cells = c.obs_names[keep]; M = M[keep]
    G = s[cells, vars_].layers['genotype']
    G = (G.toarray() if hasattr(G, 'toarray') else np.asarray(G)) > 0
    cl = ~V.loc[vars_, 'is_noise'].values
    clonal_all = set(V.index[~V.is_noise])
    own = V.loc[[v for v in vars_ if v in clonal_all], 'clone_of_origin'].unique()
    t, p = G[:, cl], M[:, cl]
    return dict(n_vars=len(vars_), n_chars=B.shape[1], var_recall=len(set(vars_) & clonal_all)/len(clonal_all),
                var_precision=cl.mean() if len(vars_) else np.nan, noise_vars_kept=int((~cl).sum()),
                geno_recall=(p & t).sum()/max(t.sum(), 1), geno_precision=(p & t).sum()/max(p.sum(), 1),
                noise_calls_share=M[:, ~cl].sum()/max(M.sum(), 1),
                clones_with_marker_pct=round(100*len(own)/s.obs['clone'].nunique(), 1),
                front_cells_pct=round(100*keep.sum()/s.shape[0], 1))


if __name__ == '__main__':
    target = sys.argv[1]
    rows = []
    if target == 'mda':
        D = '/Users/cossa/Desktop/projects/MiTo/data_test/afm_unfiltered.h5ad'
        base = mt.pp.filter_cells(sc.read(D), cell_filter='filter2')
        gbc = base.obs['GBC'].astype(str)
        GT = set(mt.pp.filter_afm(base.copy(), filtering='GT_enriched', lineage_column='GBC', ncores=4).var_names)
        fe = front_ends(base)
        print('QC:', fe.pop('_qc'), flush=True)
        for name, (c, cols, B, Xc, Xg) in fe.items():
            vars_, M = as_variants(c, cols, B)
            keep = M.sum(1) >= 1
            gb = gbc.loc[c.obs_names].values
            spec, clones = [], set()
            for j, v in enumerate(vars_):
                pos = M[:, j]
                if pos.sum() < 3:
                    continue
                modal = pd.Series(gb[pos]).value_counts().index[0]
                spec.append(pos[gb == modal].sum()/pos.sum())
                if v in GT:
                    clones.add(modal)
            front = dict(pipeline=name, n_vars=len(vars_), n_chars=B.shape[1], GT_enriched_kept=len(set(vars_) & GT),
                         var_precision_vs_GT=round(len(set(vars_) & GT)/max(len(vars_), 1), 2),
                         call_clone_specificity=round(np.mean(spec), 3), GBC_clones_marked=len(clones),
                         front_cells_pct=round(100*keep.sum()/base.shape[0], 1),
                         noise_like_vars=int(sum(s_ < 0.6 for s_ in spec)))
            for r in cluster(c, cols, B, Xc, Xg, 'GBC', base.shape[0]):
                rows.append({**front, **r})
            print(pd.DataFrame(rows).to_string(), flush=True)
        pd.DataFrame(rows).to_csv('conc_mda.csv', index=False)
    else:
        path = 'conc_sim.csv'
        for nk in [5, 10, 30, 50]:
            for topo in ['polytomy', 'depth3']:
                for seed in [0, 1, 2]:
                    kw = dict(n_cells=1000, n_clones=nk, frac_double_variants=0.3, frac_noisy_variants=0.2,
                              min_max_ratio_clones=0.2, random_seed=seed, **SIM)
                    kw.update(dict(n_root_clones=nk, max_depth=1) if topo == 'polytomy'
                              else dict(n_root_clones=max(2, nk//3), max_depth=3))
                    s = mt.ut.simulate_afm(**kw)
                    fe = front_ends(s); qc = fe.pop('_qc')
                    for name, (c, cols, B, Xc, Xg) in fe.items():
                        front = dict(pipeline=name, clones=nk, topo=topo, seed=seed, **qc,
                                     **sim_front_metrics(s, c, cols, B))
                        for r in cluster(c, cols, B, Xc, Xg, 'clone', 1000):
                            rows.append({**front, **r})
                    pd.DataFrame(rows).to_csv(path, index=False)
                    print(f'done {nk}{topo}/{seed}', flush=True)
    print('ALL DONE', flush=True)
