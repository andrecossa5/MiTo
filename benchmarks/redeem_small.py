"""
Open point 1: recover low-prevalence variants (and the small clones they mark) on ReDeeM.

Front end fixed: redeemR Filter-2 port (chi2) on raw redeemV, all qualified cells
(~4k), candidate thresholds opened inside filter_afm. Layers varied:

    genotyping   binom1e-3 (current) | binom1e-2 | any (AD >= 1 on a Filter-2 variant)
    clonality    on (qc_alpha .05) | off (Filter-2's noise test already vetted variants)
    annotation   tau .25 / .1 / 0  x  min_cells 5 / 3  x  D2 small-clone pass off / on

Scored on the 1,362 LARRY cells: partition metrics, clone recall by size stratum, and the
tree's own clade recall (every internal node, MitoDrift's metric).

Usage: python redeem_small.py            -> results/redeem_small.csv
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
import scanpy as sc
from scipy.sparse import csr_matrix

import mito as mt
import mito.pp.preprocessing as pre
from mito.ut.phylo_utils import get_clades

sys.path.insert(0, os.path.dirname(__file__))
import datasets as ds  # noqa: E402
import filter2_port as fp  # noqa: E402
import metrics as me  # noqa: E402
from mgatk_e2e import OPEN  # noqa: E402
from redeem_e2e import port_afm, to_gex  # noqa: E402
from small_clones import small_clone_pass  # noqa: E402

RESULTS = os.path.join(os.path.dirname(__file__), 'results')
_orig_call = pre.call_genotypes
_orig_qc = pre.filter_non_clonal_variants


def port_all_cells():
    path = os.path.join(ds.PL1000, 'cache', 'port_chi2_all.h5ad')
    if os.path.exists(path):
        return sc.read(path)
    keep, _, gt = fp.filter2(test='chi2', median_depth=5, rsrs50=fp.rsrs50_variants())
    _, cell_mean, dcells, DP = fp.depth_tables('S')
    qualified = list(cell_mean.index[cell_mean >= 10])
    afm = to_gex(port_afm(keep, gt, dcells, DP, qualified))
    afm.write(path)
    return afm


def caller(mode):
    def call(afm, alpha=1e-3, max_iter=50, copy=False):
        if mode == 'binom1e-2':
            alpha = alpha * 10           # filter_afm passes geno_alpha and geno_alpha/10
        _orig_call(afm, alpha=alpha, max_iter=max_iter)
        if mode == 'any':
            B = (afm.layers['AD'].toarray() > 0).astype(np.int8)
            afm.layers['bin'] = csr_matrix(B)
            afm.var['n_carriers'] = B.sum(0)
            afm.var['prevalence'] = B.mean(0)
        return afm if copy else None
    return call


def no_qc(afm, alpha=0.05, seed=0, copy=False, **kw):
    return afm if copy else None


def score(lab, truth, strata, **meta):
    row = dict(**meta, **me.partition_scores(lab, truth))
    sets = [set(lab.index[lab == L]) for L in lab.unique() if L != me.UNASSIGNED]
    if sets:
        pr = me.clade_pr(truth, sets, strata=strata).iloc[0]
        row.update({k: pr[k] for k in pr.index if k.startswith(('recall', 'precision'))})
    return row


def main():
    truth = ds.pl1000_truth()
    strata = me.size_strata(truth)
    afm0 = port_all_cells()
    print(f'port AFM {afm0.shape}', flush=True)
    rows = []
    for geno in ('binom1e-3', 'binom1e-2', 'any'):
        for qc in ('on', 'off'):
            t0 = time.time()
            afm = afm0.copy()
            pre.call_genotypes = caller(geno)
            pre.filter_non_clonal_variants = _orig_qc if qc == 'on' else no_qc
            try:
                mt.pp.filter_afm(afm, **OPEN)
            except Exception as e:  # noqa: BLE001
                print('FAILED', geno, qc, repr(e), flush=True)
                rows.append(dict(geno=geno, qc=qc, error=repr(e)))
                continue
            finally:
                pre.call_genotypes, pre.filter_non_clonal_variants = _orig_call, _orig_qc
            tree = mt.tl.build_tree(afm, solver='UPMGA')
            meta = dict(geno=geno, qc=qc, n_chars=afm.shape[1], cells_kept=afm.shape[0],
                        labelled_kept=int(afm.obs_names.isin(truth.index).sum()))

            # the tree itself, scored like MitoDrift (every internal node)
            sets = [set(v) for v in get_clades(tree, with_root=False).values()]
            pr = me.clade_pr(truth, sets, strata=strata).iloc[0]
            meta.update({f'tree_{k}': pr[k] for k in pr.index if k.startswith(('recall', 'precision'))})

            for min_cells in (5, 3):
                for tau in (0.25, 0.1, 0.0):
                    mt.tl.annotate_clones(tree, afm, tau=tau, min_cells=min_cells)
                    lab = afm.obs['MiTo_clone'].astype(str)
                    for d2 in (None, (4, 0)):
                        L, n_new = (lab, 0) if d2 is None else small_clone_pass(afm, tree, lab, *d2)
                        rows.append(score(L, truth, strata, **meta, min_cells=min_cells, tau=tau,
                                          D2=str(d2), n_new=n_new, secs=round(time.time() - t0)))
            last = rows[-12:]
            best = max(last, key=lambda r: r.get('ARI_all', -1))
            print({k: best[k] for k in ('geno', 'qc', 'n_chars', 'labelled_kept', 'tree_recall',
                                        'min_cells', 'tau', 'D2', 'yield_', 'ARI', 'ARI_all',
                                        'recall', 'recall_0.001-0.0025')}, flush=True)
            pd.DataFrame(rows).to_csv(os.path.join(RESULTS, 'redeem_small.csv'), index=False)
    print('ALL DONE', flush=True)


if __name__ == '__main__':
    main()
