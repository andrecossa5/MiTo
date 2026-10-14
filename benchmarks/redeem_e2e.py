"""
Phase 1B, end to end: MiTo on ReDeeM pL1000 from different front ends, scored on LARRY.

    filter2_deposited     redeemR Filter-2 matrix (MitoDrift's input), 1,362 cells
    port_chi2_all         our Filter-2 port on raw redeemV, all qualified cells
    port_chi2_labelled    the same, restricted to the labelled cells (sanity: ~ deposited)
    port_tarone_all       port with Tarone's depth-aware overdispersion test
    reader_default        current package: make_afm(redeem-v, edge_trim=9) + filter_afm defaults

The port AFMs carry the trimmed molecule counts and depths of filter2_port.read_trim, with
the site depth of QualifiedTotalCts (Sensitive) for cells without the variant. Candidate
thresholds inside filter_afm are opened for the port AFMs, since Filter-2 has selected them.
"""

import logging
import os
import sys
import warnings

sys.setrecursionlimit(50000)
warnings.filterwarnings('ignore')
logging.disable(logging.INFO)

import numpy as np
import pandas as pd
from anndata import AnnData
from scipy.sparse import csr_matrix

sys.path.insert(0, os.path.dirname(__file__))
import datasets as ds  # noqa: E402
import filter2_port as fp  # noqa: E402
import metrics as me  # noqa: E402
from mgatk_e2e import OPEN  # noqa: E402
from run import run_mito  # noqa: E402

RESULTS = os.path.join(os.path.dirname(__file__), 'results')


def port_afm(keep, gt, cells_all, DP, cells):
    """AFM over `cells` (raw ATAC names) x variants in `keep`."""
    var = pd.DataFrame(index=keep.index)
    var['pos'] = keep['pos'].values
    var['ref'] = [v.split('_')[1] for v in var.index]
    var['alt'] = [v.split('_')[2] for v in var.index]
    g = gt[gt['Variants'].isin(var.index) & gt['Cell'].isin(cells)]
    ci = pd.Index(cells).get_indexer(g['Cell'])
    vi = pd.Index(var.index).get_indexer(g['Variants'])
    AD = np.zeros((len(cells), len(var)), dtype=np.int32)
    AD[ci, vi] = g['Freq'].values
    di = pd.Index(cells_all).get_indexer(cells)
    D = DP[di][:, var['pos'].values - 1].astype(np.int32)
    D[ci, vi] = np.maximum(g['depth'].values, g['Freq'].values).astype(np.int32)
    D = np.maximum(D, AD)
    var.index = [f'{p}_{r}>{a}' for p, r, a in var[['pos', 'ref', 'alt']].values]
    obs = pd.DataFrame(index=cells)
    obs['mean_site_coverage'] = DP[di].mean(1)
    return AnnData(X=csr_matrix((AD / np.maximum(D, 1)).astype(np.float32)), obs=obs, var=var,
                   layers={'AD': csr_matrix(AD), 'DP': D},
                   uns={'scLT_system': 'ReDeeM', 'pp_method': 'redeem-v'})


def to_gex(afm):
    bmap = ds.barcode_map()
    afm.obs_names = [f'{bmap.get(c, c)}-1' for c in afm.obs_names]
    return afm


def main():
    truth = ds.pl1000_truth()
    strata = me.size_strata(truth)
    annot = [{'tau': 0.25}, {'tau': 0.1}]
    rows = []

    afm, _ = ds.pl1000_filter2()
    rows += run_mito(afm, truth, annot_kwargs=annot, strata=strata, dataset='pL1000_redeem',
                     front='filter2_deposited')
    rows += run_mito(afm, truth, filter_kwargs=OPEN, annot_kwargs=annot, strata=strata,
                     dataset='pL1000_redeem', front='filter2_deposited_open')
    pd.DataFrame(rows).to_csv(os.path.join(RESULTS, 'redeem_e2e.csv'), index=False)

    rs = fp.rsrs50_variants()
    pos_mean, cell_mean, dcells, DP = fp.depth_tables('S')
    qualified = list(cell_mean.index[cell_mean >= 10])
    bmap = ds.barcode_map()
    inv = {v: k for k, v in bmap.items()}
    labelled = [inv[c.replace('-1', '')] for c in truth.index if c.replace('-1', '') in inv]
    labelled = [c for c in labelled if c in set(qualified)]

    for test in ('chi2', 'tarone'):
        keep, f, gt = fp.filter2(test=test, median_depth=5, rsrs50=rs)
        sets = [('all', qualified)] + ([('labelled', labelled)] if test == 'chi2' else [])
        for cname, cells in sets:
            afm = to_gex(port_afm(keep, gt, dcells, DP, cells))
            if cname == 'labelled':   # redeemR's final subset: >= 2 carriers among these cells
                afm = afm[:, (afm.layers['AD'] > 0).sum(0).A1 >= 2].copy()
            for fk, tag in ((OPEN, 'open'), ({}, 'default')):
                rows += run_mito(afm, truth, filter_kwargs=fk, annot_kwargs=annot, strata=strata,
                                 dataset='pL1000_redeem', front=f'port_{test}_{cname}_{tag}',
                                 n_port=afm.shape[1])
                pd.DataFrame(rows).to_csv(os.path.join(RESULTS, 'redeem_e2e.csv'), index=False)

    afm, _ = ds.pl1000_redeemv(edge_trim=9)
    rows += run_mito(afm, truth, annot_kwargs=annot, strata=strata, dataset='pL1000_redeem',
                     front='reader_default')
    pd.DataFrame(rows).to_csv(os.path.join(RESULTS, 'redeem_e2e.csv'), index=False)
    print('ALL DONE', flush=True)


if __name__ == '__main__':
    main()
