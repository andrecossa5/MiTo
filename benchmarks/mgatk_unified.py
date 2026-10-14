"""
mtscATAC (mgatk, pL1000) through the unified pipeline, on the same cells as the ReDeeM
result (redeemV coverage >= 10, 3,721 cells), so the two assays compare directly.

    thresholds         package default for mtscATAC (read-level thresholds, SNR 10)
    od_raw             over-dispersion stage on raw read counts, binomial calls
    od_raw+scaled      ... calls on duplicate-scaled counts (AD/f, DP/f, continuous tail)
    od_scaled+scaled   over-dispersion test on rounded AD/f, DP/f too; then as above

f = 2 by default (any f in 1.5-4 worked for genotyping, mgatk_dup.py); `--f 1.66` is the
value estimated from the strand counts (mgatk_dupfactor.py).

Usage: python mgatk_unified.py [--f F]     -> results/mgatk_unified[_fF].csv
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
from scipy.sparse import csr_matrix

import mito as mt

sys.path.insert(0, os.path.dirname(__file__))
import ablation as ab  # noqa: E402
import datasets as ds  # noqa: E402
import metrics as me  # noqa: E402
from mgatk_e2e import OPEN  # noqa: E402
from mgatk_mechanism import cont_sf, generic_caller  # noqa: E402

RESULTS = os.path.join(os.path.dirname(__file__), 'results')
F = float(sys.argv[sys.argv.index('--f') + 1]) if '--f' in sys.argv else 2.0
OUT = os.path.join(RESULTS, 'mgatk_unified.csv' if F == 2.0 else f'mgatk_unified_f{F}.csv')
scaled_caller = generic_caller(lambda A, N, p: cont_sf(A / F, N / F, p))


def scaled_test_afm(afm):
    """Raw AFM restricted to the variants passing the test on rounded AD/f, DP/f."""
    s = afm.copy()
    s.layers['AD'] = csr_matrix(np.rint(s.layers['AD'].toarray() / F).astype(np.int32))
    s.layers['DP'] = np.rint(np.asarray(s.layers['DP']) / F).astype(np.int32)
    s.obs['mean_site_coverage'] = s.obs['mean_site_coverage'] / F
    mt.pp.filter_overdispersed_variants(s, min_cell_coverage=10 / F)
    out = afm[:, s.var_names].copy()
    out.var['od_stat'] = s.var['od_stat'].values
    print(f'scaled test: {s.shape[1]} variants pass', flush=True)
    return out


def main():
    rv, truth = ds.pl1000_redeemv(edge_trim=9)
    cells = rv.obs_names[rv.obs['mean_site_coverage'].values >= 10]
    afm, _ = ds.pl1000_mgatk()
    afm = afm[afm.obs_names.isin(cells)].copy()
    print(f'mgatk AFM on the ReDeeM cells: {afm.shape}', flush=True)
    strata = me.size_strata(truth)
    meta = dict(dataset='pL1000_mgatk_redeem_cells')

    rows = []
    if F != 2.0:                     # the thresholds / od_raw rows do not depend on f
        ab.CONFIGS = {'od_raw+scaled': ({'candidates': 'overdispersion'}, {'call_genotypes': scaled_caller})}
        rows += ab.run(afm, truth, strata, f=F, **meta)
        pd.DataFrame(rows).to_csv(OUT, index=False)
        pre = scaled_test_afm(afm)
        ab.CONFIGS = {'od_scaled+scaled': ({'candidates': 'thresholds', 'min_snr': 0, **OPEN},
                                           {'call_genotypes': scaled_caller})}
        rows += ab.run(pre, truth, strata, f=F, **meta)
        pd.DataFrame(rows).to_csv(OUT, index=False)
        print('ALL DONE', flush=True)
        return
    ab.CONFIGS = {
        'thresholds': ({'candidates': 'thresholds'}, {}),
        'od_raw': ({'candidates': 'overdispersion'}, {}),
        'od_raw+scaled': ({'candidates': 'overdispersion'}, {'call_genotypes': scaled_caller}),
    }
    rows += ab.run(afm, truth, strata, **meta)
    pd.DataFrame(rows).to_csv(OUT, index=False)

    # Test on scaled counts, then the downstream with candidate thresholds open
    pre = scaled_test_afm(afm)
    ab.CONFIGS = {
        'od_scaled+scaled': ({'candidates': 'thresholds', 'min_snr': 0, **OPEN},
                             {'call_genotypes': scaled_caller}),
    }
    rows += ab.run(pre, truth, strata, **meta)
    pd.DataFrame(rows).to_csv(OUT, index=False)
    print('ALL DONE', flush=True)


if __name__ == '__main__':
    main()
