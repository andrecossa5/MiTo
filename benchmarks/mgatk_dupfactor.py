"""
Estimate the reads-per-molecule factor f of mgatk output from the count tables alone, and
check it against ReDeeM molecule counts on the same cells (pL1000).

Ground truth (results/mgatk_inflation.csv, labelled cells x Filter-2 variants):
mgatk/ReDeeM coverage 2.13 (per cell IQR 2.10-2.17); one alt molecule -> 2.08 alt reads,
and 97.6% of the 2-read cases are one forward + one reverse read: overlapping mates.

Strand-variance estimator. Per cell and site, molecules are read by both mates (one
forward + one reverse read, fraction o) or by one (a single read, forward with
probability s). With R reads and fw forward reads,

    Var(fw - s R | R) = S s (1 - s),   S = R (1 - o) / (1 + o) single reads

so kappa = sum (fw - s R)^2 / sum R s (1 - s) = (1 - o) / (1 + o), and
f = 1 + o = 2 / (1 + kappa). s is the site's forward fraction over all cells. PCR
duplicates (same-strand copies) would inflate the variance instead: kappa > 1, f < 1
signals them.

Usage: python mgatk_dupfactor.py   -> results/mgatk_dupfactor.csv (per site), stdout
"""

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
import datasets as ds  # noqa: E402

BASES = ['A', 'C', 'G', 'T']
RESULTS = os.path.join(os.path.dirname(__file__), 'results')


def strand_coverage():
    """Forward and reverse reads over all four bases, (cells x sites), cached."""
    path = os.path.join(ds.PL1000, 'cache', 'mgatk_strand_coverage.npz')
    if os.path.exists(path):
        z = np.load(path, allow_pickle=True)
        return z['cells'], z['FW'], z['REV']
    cells = None
    FW = REV = None
    for base in BASES:
        df = pd.read_csv(os.path.join(ds.PL1000, f'pL1000_mgatk.{base}.txt.gz'), header=None,
                         names=['pos', 'cell', 'fw', 'rev'],
                         dtype={'pos': np.int32, 'cell': str, 'fw': np.int32, 'rev': np.int32})
        if cells is None:
            cov = pd.read_csv(os.path.join(ds.PL1000, 'pL1000_mgatk.coverage.txt.gz'), header=None,
                              names=['pos', 'cell', 'DP'], usecols=['cell'], dtype={'cell': 'category'})
            cells = cov['cell'].cat.categories.values.astype(str)
            FW = np.zeros((cells.size, 16569), np.int32)
            REV = np.zeros_like(FW)
        r = pd.Index(cells).get_indexer(df['cell'])
        ok = r >= 0
        np.add.at(FW, (r[ok], df['pos'].values[ok] - 1), df['fw'].values[ok])
        np.add.at(REV, (r[ok], df['pos'].values[ok] - 1), df['rev'].values[ok])
        print(f'{base} done', flush=True)
    np.savez(path, cells=cells, FW=FW, REV=REV)
    return cells, FW, REV


def kappa_per_site(FW, REV, min_reads=1):
    R = (FW + REV).astype(float)
    s = FW.sum(0) / np.maximum(R.sum(0), 1)
    used = R >= min_reads
    num = np.where(used, (FW - s * R) ** 2, 0).sum(0)
    den = np.where(used, R * s * (1 - s), 0).sum(0)
    with np.errstate(all='ignore'):
        return num / den, s, R.mean(0)


def main():
    cells, FW, REV = strand_coverage()
    R = FW + REV
    keep_cells = R.mean(1) >= 10
    FW, REV = FW[keep_cells], REV[keep_cells]
    print(f'cells with mean coverage >= 10: {keep_cells.sum()}', flush=True)

    k, s, depth = kappa_per_site(FW, REV)
    tab = pd.DataFrame(dict(pos=np.arange(1, k.size + 1), kappa=k, fw_frac=s, mean_reads=depth))
    tab['f'] = 2 / (1 + tab['kappa'])
    tab.to_csv(os.path.join(RESULTS, 'mgatk_dupfactor.csv'), index=False)

    ok = tab['mean_reads'] >= 5
    for lo, hi in ((0.0, 1.0), (0.2, 0.8), (0.35, 0.65)):
        m = ok & tab['fw_frac'].between(lo, hi)
        print(f'sites with fw fraction in [{lo}, {hi}]: {m.sum():5d}  median kappa '
              f'{tab.loc[m, "kappa"].median():.3f}  f = {2 / (1 + tab.loc[m, "kappa"].median()):.2f}  '
              f'(IQR of per-site f {tab.loc[m, "f"].quantile(.25):.2f}-{tab.loc[m, "f"].quantile(.75):.2f})',
              flush=True)

    # Per cell, over balanced sites
    bal = (ok & tab['fw_frac'].between(0.35, 0.65)).values
    Rb = (FW + REV)[:, bal].astype(float)
    sb = s[bal]
    kc = ((FW[:, bal] - sb * Rb) ** 2).sum(1) / (Rb * sb * (1 - sb)).sum(1)
    fc = 2 / (1 + kc)
    print(f'per cell f: median {np.median(fc):.2f}, IQR {np.quantile(fc, .25):.2f}-{np.quantile(fc, .75):.2f}',
          flush=True)
    print('truth (ReDeeM, same cells): coverage ratio 2.13 (per cell IQR 2.10-2.17); '
          '2.08 alt reads per alt molecule', flush=True)


if __name__ == '__main__':
    main()
