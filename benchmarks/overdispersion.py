"""
A generic, assay-agnostic candidate stage: the variant-level part of redeemR Filter-2,
applied to any AFM (AD sparse, DP dense), and tested as a replacement for MiTo's
threshold-based candidate filter.

    1. qualified cells: mean site coverage >= 10
    2. variants carried (AD > 0) by >= 2 qualified cells
    3. homoplasmy flag (CellNPCT > .75 & mean AF in carriers > .75 & var/mean < .01)
    4. noise test, keep q <= .05:
         chi2    redeemR goodness of fit to ONE binomial at the site's mean depth
         tarone  Tarone's overdispersion score test with each cell's own depth (BH)
    5. blacklist regions (rCRS 302-315, 513-525, 3105-3109, 16182-16187), ancestral RSRS50
    6. median depth >= 5, and CellNPCT <= .75 again
    7. cap: at most `max_candidates`, ranked by the test statistic

No edge trimming: that needs molecule coordinates, available from redeemV only (the
reader already trims). Downstream, filter_afm runs with its candidate thresholds opened.

Usage: python overdispersion.py redeem | redeem_qcells | maester | sims
    redeem_qcells   as redeem, but only cells with mean coverage >= 10 go downstream
    sims            the 24 MAESTER simulations (no RSRS50: simulated positions)
"""

import logging
import os
import sys
import time
import warnings

sys.setrecursionlimit(50000)
warnings.filterwarnings('ignore')
logging.disable(logging.INFO)

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests

import mito as mt
from mito.ut.phylo_utils import get_clades

sys.path.insert(0, os.path.dirname(__file__))
import datasets as ds  # noqa: E402
import filter2_port as fp  # noqa: E402
import metrics as me  # noqa: E402
from mgatk_e2e import OPEN  # noqa: E402
from small_clones import small_clone_pass  # noqa: E402

RESULTS = os.path.join(os.path.dirname(__file__), 'results')


def overdispersion_filter(afm, test='tarone', min_cells=2, q_max=0.05, min_median_depth=5,
                          max_candidates=1000, min_mean_cov=10, rsrs50=None):
    """Returns (variant mask, per-variant table); the AFM is not modified."""
    AD = afm.layers['AD'].toarray().astype(np.int64)
    DP = np.asarray(afm.layers['DP'], dtype=np.int64)
    DP = np.maximum(DP, AD)
    cov = afm.obs['mean_site_coverage'].values if 'mean_site_coverage' in afm.obs else DP.mean(1)
    q = cov >= min_mean_cov
    A, D = AD[q], DP[q]
    car = A > 0
    n_car = car.sum(0)
    with np.errstate(all='ignore'):
        het = np.where(car, A / np.maximum(D, 1), np.nan)
        pos_mean = np.nanmean(het, 0)
        cv = np.nanvar(het, 0, ddof=1) / pos_mean
    cellnpct = n_car / np.maximum((D > 0).sum(0), 1)
    homo = (cellnpct > 0.75) & (pos_mean > 0.75) & (cv < 0.01)
    tested = (n_car >= min_cells) & ~homo

    stat = np.full(A.shape[1], -np.inf)
    pv = np.ones(A.shape[1])
    if test == 'chi2':
        # redeemR: the counts of the cells carrying any tested variant, one binomial at the
        # site's mean depth
        rows = car[:, tested].any(1)
        for j in np.flatnonzero(tested):
            c2, p = fp.gof(A[rows, j], D[:, j].mean())
            stat[j], pv[j] = c2, p
        qv = np.ones_like(pv)
        qv[tested] = fp.qvalue(pv[tested])
    else:
        n, x = D.astype(float), A.astype(float)
        p_hat = x.sum(0) / np.maximum(n.sum(0), 1)
        with np.errstate(all='ignore'):
            S = ((x - n * p_hat) ** 2).sum(0) / (p_hat * (1 - p_hat)) - n.sum(0)
            V = 2 * (n * (n - 1)).sum(0)
            Z = S / np.sqrt(V)
        Z = np.where(np.isfinite(Z) & (p_hat > 0) & (p_hat < 1), Z, -np.inf)
        stat = np.where(tested, Z, -np.inf)
        pv = np.where(tested, stats.norm.sf(Z), 1.0)
        qv = np.ones_like(pv)
        qv[tested] = multipletests(pv[tested], method='fdr_bh')[1]

    pos = afm.var['pos'].values
    keep = tested & (qv <= q_max) & ~np.isin(pos, list(fp.BLACKLIST))
    if rsrs50 is not None:
        ids = afm.var_names.str.replace('>', '_')
        keep &= ~ids.isin(rsrs50)
    med = np.median(D, 0)
    keep &= (med >= min_median_depth) & (n_car / np.maximum((D > 0).sum(0), 1) <= 0.75)
    n_pass = int(keep.sum())
    if max_candidates and n_pass > max_candidates:
        cut = np.sort(stat[keep])[::-1][max_candidates - 1]
        keep &= stat >= cut

    tab = pd.DataFrame(dict(n_carriers=n_car, homo=homo, stat=stat, q=qv, median_depth=med,
                            keep=keep), index=afm.var_names)
    tab.attrs['n_pass_before_cap'] = n_pass
    return keep, tab


def run(afm, truth, strata, name, front, test=None, rsrs50=None, **meta):
    t0 = time.time()
    afm = afm.copy()
    n_in = afm.shape[1]
    if front == 'mito_default':
        fk = {}
        n_front = np.nan
    else:
        keep, tab = overdispersion_filter(afm, test=test, rsrs50=rsrs50)
        afm = afm[:, keep].copy()
        fk = OPEN
        n_front = int(keep.sum())
        meta['n_pass_before_cap'] = tab.attrs['n_pass_before_cap']
    mt.pp.filter_afm(afm, **fk)
    tree = mt.tl.build_tree(afm, solver='UPMGA')
    mt.tl.annotate_clones(tree, afm)
    lab0 = afm.obs['MiTo_clone'].astype(str)
    sets_tree = [set(v) for v in get_clades(tree, with_root=False).values()]
    rows = []
    for d2 in (False, True):
        lab, n_new = small_clone_pass(afm, tree, lab0, 4, 0) if d2 else (lab0, 0)
        row = dict(dataset=name, front=front if test is None else f'{front}_{test}', D2=d2,
                   n_vars_in=n_in, n_front=n_front, n_chars=afm.shape[1], cells_kept=afm.shape[0],
                   n_new=n_new, secs=round(time.time() - t0), **meta, **me.partition_scores(lab, truth))
        sets = [set(lab.index[lab == L]) for L in lab.unique() if L != me.UNASSIGNED]
        if strata is not None and sets:
            pr = me.clade_pr(truth, sets, strata=strata).iloc[0]
            row.update({k: pr[k] for k in pr.index if k.startswith(('recall', 'precision'))})
            tp = me.clade_pr(truth, sets_tree, strata=strata).iloc[0]
            row.update({f'tree_{k}': tp[k] for k in tp.index if k.startswith(('recall', 'precision'))})
        rows.append(row)
        print({k: row.get(k) for k in ('dataset', 'front', 'D2', 'n_front', 'n_chars', 'yield_',
                                         'n_labels', 'ARI', 'ARI_all', 'recall')}, flush=True)
    return rows


def main():
    target = sys.argv[1]
    rs = fp.rsrs50_variants()
    rows = []
    out = os.path.join(RESULTS, f'overdispersion_{target}.csv')
    if target == 'redeem':
        truth = ds.pl1000_truth()
        strata = me.size_strata(truth)
        afm, _ = ds.pl1000_redeemv(edge_trim=9)
        for front, test in (('mito_default', None), ('overdisp', 'chi2'), ('overdisp', 'tarone')):
            rows += run(afm, truth, strata, 'pL1000_redeemv_trim9', front, test, rsrs50=rs)
            pd.DataFrame(rows).to_csv(out, index=False)
    elif target == 'redeem_qcells':
        truth = ds.pl1000_truth()
        strata = me.size_strata(truth)
        afm, _ = ds.pl1000_redeemv(edge_trim=9)
        afm = afm[afm.obs['mean_site_coverage'].values >= 10].copy()
        for front, test in (('mito_default', None), ('overdisp', 'chi2')):
            rows += run(afm, truth, strata, 'pL1000_redeemv_trim9_qcells', front, test, rsrs50=rs)
            pd.DataFrame(rows).to_csv(out, index=False)
    elif target == 'sims':
        for nk, topo, seed in ds.SIM_GRID:
            afm, truth = ds.sim(nk, topo, seed)
            for front, test in (('mito_default', None), ('overdisp', 'chi2'), ('overdisp', 'tarone')):
                try:
                    rows += run(afm, truth, None, 'sim_maester', front, test, clones=nk, topo=topo, seed=seed)
                except Exception as e:  # noqa: BLE001
                    print('FAILED', nk, topo, seed, front, test, repr(e), flush=True)
                    rows.append(dict(dataset='sim_maester', clones=nk, topo=topo, seed=seed,
                                     front=f'{front}_{test}', error=repr(e)))
                pd.DataFrame(rows).to_csv(out, index=False)
    else:
        for name in ('MDA_clones', 'MDA_lung', 'MDA_PT'):
            for reader in ('shipped', 'current'):
                afm, truth = ds.mda(name, reader=reader)
                strata = me.size_strata(truth, bins=((0.05, 1.0), (0.01, 0.05), (0.0, 0.01)))
                for front, test in (('mito_default', None), ('overdisp', 'tarone'), ('overdisp', 'chi2')):
                    try:
                        rows += run(afm, truth, strata, name, front, test, rsrs50=rs, reader=reader)
                    except Exception as e:  # noqa: BLE001
                        print('FAILED', name, reader, front, test, repr(e), flush=True)
                        rows.append(dict(dataset=name, reader=reader, front=f'{front}_{test}', error=repr(e)))
                    pd.DataFrame(rows).to_csv(out, index=False)
    print('ALL DONE', flush=True)


if __name__ == '__main__':
    main()
