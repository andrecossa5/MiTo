"""
Development MiTo preprocessing + clonal inference pipeline (not part of the `mito` package yet).

    filter_MiTo (loose thresholds)
      -> guarded flat-prior genotyping                     (geno2.em_genotype, guard=True)
      -> carrier non-randomness QC: join count OR exclusivity, one alpha
                                                           (joincount.carrier_nonrandomness)
      -> guarded graph-prior genotyping + dropout imputation on kept variants
      -> prevalence cap -> [optional split of recurrent variants] -> four-gamete
      -> UPGMA tree -> evidence-balanced cut (+ membership abstention)

Usage
-----
    import scanpy as sc, mito as mt
    from pipeline import run_pipeline
    afm = mt.pp.filter_cells(sc.read('afm_unfiltered.h5ad'), cell_filter='filter2')
    res = run_pipeline(afm)
    res['labels']          # pd.Series over afm.obs_names, 'unassigned' where not called
"""
import sys

import anndata as ad
import mito as mt
import numpy as np
import pandas as pd
from mito.pp.filters import annotate_vars, filter_baseline, filter_MiTo
from scipy.sparse import csr_matrix

from compat import cell_membership
from cutter import evidence_cut
from geno2 import em_genotype
from grid import FK, kernel, maxcompat
from impute import impute_dropouts
from joincount import carrier_nonrandomness
from refine import split_recurrent_graph

sys.setrecursionlimit(100000)


def run_pipeline(
    afm,
    alpha=0.05,
    one_sided=False,
    split=False,
    tau=0.25,
    k_qc=15,
    n_perm=999,
    seed=0,
    filter_kwargs=None,
    impute_thr=0.6,
    max_prevalence=0.5,
    ncores=8,
):
    """
    Run the development pipeline on a cell-filtered AFM.

    Parameters
    ----------
    afm : AnnData
        Allele frequency matrix after cell filtering (layers AD, site_coverage).
    alpha : float
        Single threshold of the variant QC (Bonferroni over its two tests).
    one_sided : bool
        Let evidence_cut split a nested clone off its parent (resolves subclones).
    split : bool
        Split recurrent variants into lineage-specific characters. Off by default:
        it hurt MDA_PT (-0.07 ARI) and was neutral elsewhere.
    tau : float or None
        Membership abstention threshold; None disables abstention.

    Returns
    -------
    dict with labels, labels_no_abstention, final AnnData, tree, per-variant QC table.
    """
    filter_kwargs = FK if filter_kwargs is None else filter_kwargs
    b = afm.copy()
    annotate_vars(b)
    b = filter_baseline(b)
    c = filter_MiTo(b, **filter_kwargs)
    X = c.X.toarray()
    AD = c.layers['AD'].toarray()
    COV = c.layers['site_coverage'].toarray().astype(np.int64)
    n, m = X.shape

    # 1. variant QC on guarded flat-prior calls (a graph prior here would imprint
    #    lineage structure on noise calls and blind the test)
    g_flat, _ = em_genotype(AD, COV, np.full((n, n), 1/n), mode='bb', guard=True)
    B_flat = (g_flat > 0.7).astype(np.int8)
    keep, p_join, p_excl = carrier_nonrandomness(X, B_flat, cell_depth=COV.mean(1), k=k_qc, alpha=alpha,
                                                 n_perm=n_perm, seed=seed)
    qc = pd.DataFrame(dict(carriers=B_flat.sum(0), p_join=p_join, p_excl=p_excl, keep=keep), index=c.var_names)
    cols = np.flatnonzero(keep)
    if cols.size < 2:
        raise ValueError(f'Variant QC kept {cols.size} variants: nothing to build a tree from.')

    # 2. guarded graph-prior genotyping + imputation on kept variants
    Xk = X[:, cols]
    g, _ = em_genotype(AD[:, cols], COV[:, cols], kernel(Xk), mode='bb', guard=True)
    Bk = impute_dropouts(Xk, (g > 0.7).astype(np.int8), k=30, thr=impute_thr, min_support=1)[0]

    # 3. character filters
    kv = np.flatnonzero(Bk.astype(bool).mean(0) <= max_prevalence)
    Bp, Xp, cp = Bk[:, kv], Xk[:, kv], cols[kv]
    origin = np.arange(Bp.shape[1])
    if split:
        Bp, origin, _ = split_recurrent_graph(Xp, Bp)
    alive = maxcompat(Bp)
    Bs, origin = Bp[:, alive], origin[alive]
    Xs = np.where(Bs > 0, Xp[:, origin], 0.0)
    char_vars = c.var_names[cp[origin]]
    qc['in_final'] = qc.index.isin(char_vars)

    # 4. tree + cut
    kc = (Bs > 0).sum(1) >= 1
    a = ad.AnnData(X=csr_matrix(Xs[kc]), obs=c.obs.iloc[np.flatnonzero(kc)].copy(),
                   layers={'bin': csr_matrix((Bs[kc] > 0).astype(np.int8))})
    a.var_names = [f'{v}#{i}' for i, v in enumerate(char_vars)]
    a.uns['genotyping'] = {'bin_method': 'dev_guarded_bb', 'binarization_kwargs': {}}
    for key in ('scLT_system', 'pp_method'):
        if key in afm.uns:
            a.uns[key] = afm.uns[key]
    mt.pp.compute_distances(a, precomputed=True, metric='weighted_jaccard', ncores=ncores, verbose=False)
    tree = mt.tl.build_tree(a, precomputed=True, solver='UPMGA')
    Bt = a.layers['bin'].toarray() > 0
    lab = evidence_cut(tree, Bt.astype(float), list(a.obs_names), min_in=0.25, ratio=3.0, self_min_in=0.85,
                       ladder='descend', one_sided=one_sided)
    lab_abst = lab.copy()
    if tau is not None:
        _, core = cell_membership(kernel(Xk, raw=True), Bs > 0, tau=0.0)
        lab_abst[~(Bt & (core[kc] >= tau)).any(1)] = 'unassigned'

    def full(x):
        out = pd.Series('unassigned', index=afm.obs_names, dtype=object)
        out.loc[x.index] = x.values
        return out

    return dict(labels=full(lab_abst), labels_no_abstention=full(lab), afm=a, tree=tree, qc=qc)


if __name__ == '__main__':
    import scanpy as sc
    path, truth = sys.argv[1], (sys.argv[2] if len(sys.argv) > 2 else None)
    afm = mt.pp.filter_cells(sc.read(path), cell_filter='filter2')
    res = run_pipeline(afm)
    lab = res['labels']
    ok = lab != 'unassigned'
    print(f'cells assigned {ok.mean():.1%}, labels {lab[ok].nunique()}, '
          f'variants kept by QC {int(res["qc"].keep.sum())}, characters in tree {res["afm"].shape[1]}')
    if truth:
        gt = afm.obs[truth].astype(str)
        print(f'ARI {mt.ut.custom_ARI(gt[ok], lab[ok]):.3f}  NMI {mt.ut.normalized_mutual_info_score(gt[ok], lab[ok]):.3f}')
