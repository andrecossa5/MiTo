"""
Phase 1A follow-up: duplicate-aware genotyping without UMIs.

1. Estimator. If each molecule is read f times on average, alternative counts at a site
   are compound Poisson and their dispersion index (variance / mean, depth-normalised)
   is ~ E[f^2]/E[f] instead of 1. Estimated per site over all cells,
       D_j = sum_i (x_ij - n_ij p_j)^2 / (n_ij p_j) / (N - 1),
   at sites that are background-like (no clonal structure expected: low bulk AF), and
   summarised by the median. Checked on ReDeeM (consensus molecules, expect ~1) and
   mgatk (reads, expect ~2).
2. Sensitivity: MiTo end-to-end on mgatk (conf3 candidates, both+dedup genotyping) for
   f in 1..4 and for the estimated f.
"""

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
import datasets as ds  # noqa: E402
import metrics as me  # noqa: E402
import mgatk_e2e as e2e  # noqa: E402
import mgatk_strand as ms  # noqa: E402
import mito.pp.preprocessing as pre  # noqa: E402
from run import run_mito  # noqa: E402

RESULTS = os.path.join(os.path.dirname(__file__), 'results')


def dispersion_index(AD, DP, max_bulk_af=0.002, min_mean_alt=0.02):
    """Per-site depth-normalised dispersion of alt counts; returns (median, per-site)."""
    AD = AD.toarray() if hasattr(AD, 'toarray') else np.asarray(AD)
    DP = np.asarray(DP, float)
    p = AD.sum(0) / np.maximum(DP.sum(0), 1)
    sel = (p > 0) & (p <= max_bulk_af) & (AD.mean(0) >= min_mean_alt)
    mu = DP[:, sel] * p[sel]
    ok = mu > 0
    r = np.where(ok, (AD[:, sel] - mu) ** 2 / np.where(ok, mu, 1), 0)
    Dj = r.sum(0) / np.maximum(ok.sum(0) - 1, 1)
    return float(np.median(Dj)), Dj


def main():
    d = ms.load()
    truth = ds.pl1000_truth()
    lab = np.isin(d['cells'], truth.index) & (d['DP'].mean(1) >= 10)

    # Estimator on both assays (all cells for mgatk; the Filter-2 grid for ReDeeM)
    ALT = (d['FW'] + d['REV']).tocsc()
    chunks = [dispersion_index(ALT[:, s:s + 2000], d['DP'][:, s:s + 2000])[1]
              for s in range(0, ALT.shape[1], 2000)]
    f_mgatk = float(np.median(np.concatenate(chunks)))
    afm, _ = ds.pl1000_filter2()
    f_redeem, _ = dispersion_index(afm.layers['AD'], afm.layers['DP'])
    print(f'dispersion index: mgatk {f_mgatk:.3f} | ReDeeM Filter-2 {f_redeem:.3f} '
          f'| measured read/molecule ratio on shared entries ~2.0-2.1', flush=True)

    st = pd.read_csv(os.path.join(ds.PL1000, 'cache', 'mgatk_site_table.csv'), index_col=0)
    n_all = d['DP'].shape[0]
    cand = ((st['mean_DP'] >= 5) & (st['n_carriers'] >= 2) & (st['n_carriers'] <= 0.5 * n_all)
            & (st['n_cells_conf_detected'] >= 3)).values
    afm = e2e.build_afm(d, cand, lab)
    strata = me.size_strata(truth)
    rows = []
    for f in (1.0, 1.5, 2.0, 2.5, 3.0, 4.0, round(f_mgatk, 2)):
        e2e.DUP = f
        pre.call_genotypes = e2e.make_caller('both+dedup' if f > 1 else 'both_strands')
        try:
            rows += run_mito(afm, truth, filter_kwargs=e2e.OPEN,
                             annot_kwargs=[{'tau': 0.25}, {'tau': 0.1}], strata=strata,
                             dataset='pL1000_mgatk_labelled', dup=f, f_estimated=round(f_mgatk, 3))
        finally:
            pre.call_genotypes = e2e._orig_call
        pd.DataFrame(rows).to_csv(os.path.join(RESULTS, 'mgatk_dup.csv'), index=False)
    print('ALL DONE', flush=True)


if __name__ == '__main__':
    main()
