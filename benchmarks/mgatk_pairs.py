"""
mtscATAC without a fudge factor: count alt molecules seen on both strands only.

On pL1000 (mgatk_inflation.csv), alt evidence of a real molecule is on both strands in
87% of cells (overlapping mates: one forward + one reverse read), background in 23%;
single reads are 70% of the background. So:

    AD  = min(fw, rev) of the alt allele     molecules confirmed by both mates
    DP  = rint(reads / f)                    coverage in molecules, f = 1.66 estimated
                                             from strand variance (mgatk_dupfactor.py)

and the package pipeline runs unchanged on these molecule units (over-dispersion stage,
binomial genotyping), on the same 3,721 cells as the ReDeeM result.

    pairs        AD = min(fw, rev)
    pairs_f2     same, DP = rint(reads / 2), to see how much the coverage factor matters
    reads_f1.66  AD = rint(reads / 1.66) (reference: one factor for both)

Usage: python mgatk_pairs.py        -> results/mgatk_pairs.csv
"""

import logging
import os
import sys
import warnings

sys.setrecursionlimit(50000)
warnings.filterwarnings('ignore')
logging.disable(logging.WARNING)

import numpy as np
import pandas as pd
from anndata import AnnData
from scipy.sparse import csr_matrix

sys.path.insert(0, os.path.dirname(__file__))
import ablation as ab  # noqa: E402
import datasets as ds  # noqa: E402
import metrics as me  # noqa: E402
import mgatk_strand as ms  # noqa: E402

RESULTS = os.path.join(os.path.dirname(__file__), 'results')
F_EST = 1.66


def molecule_afm(d, cells, alt='pairs', f=F_EST):
    keep = np.isin(d['cells'], cells)
    sites = d['alt'] != 'N'
    FW = d['FW'][keep][:, sites].toarray()
    REV = d['REV'][keep][:, sites].toarray()
    R = d['DP'][keep][:, sites]
    if alt == 'pairs':
        AD = np.minimum(FW, REV)
    else:
        AD = np.rint((FW + REV) / f)
    DP = np.maximum(np.rint(R / f), AD).astype(np.int32)
    AD = AD.astype(np.int32)
    var = pd.DataFrame(dict(pos=d['sites'][sites], ref=d['ref'][sites], alt=d['alt'][sites]))
    var.index = [f'{p}_{r}>{a}' for p, r, a in var.values]
    obs = pd.DataFrame(index=d['cells'][keep])
    obs['mean_site_coverage'] = d['DP'][keep].mean(1) / f
    nz = AD.sum(0) > 0
    return AnnData(X=csr_matrix((AD / np.maximum(DP, 1)).astype(np.float32))[:, nz], obs=obs,
                   var=var[nz], layers={'AD': csr_matrix(AD[:, nz]), 'DP': DP[:, nz]},
                   uns={'scLT_system': 'mtscATAC', 'pp_method': 'mgatk'})


def main():
    rv, truth = ds.pl1000_redeemv(edge_trim=9)
    cells = rv.obs_names[rv.obs['mean_site_coverage'].values >= 10]
    d = ms.load()
    strata = me.size_strata(truth)
    ab.CONFIGS = {'unified': ({'candidates': 'overdispersion'}, {})}
    rows = []
    for name, alt, f in (('pairs', 'pairs', F_EST), ('pairs_f2', 'pairs', 2.0), ('reads_f1.66', 'reads', F_EST)):
        afm = molecule_afm(d, cells, alt, f)
        print(f'{name}: AFM {afm.shape}, alt entries {afm.layers["AD"].nnz}', flush=True)
        rows += ab.run(afm, truth, strata, dataset='pL1000_mgatk_redeem_cells', units=name)
        pd.DataFrame(rows).to_csv(os.path.join(RESULTS, 'mgatk_pairs.csv'), index=False)
    print('ALL DONE', flush=True)


if __name__ == '__main__':
    main()
