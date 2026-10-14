"""
Phase 1B: redeemR Filter-2 (redeemR2.0 `preprocessed_redeemr`) ported to Python, on raw
redeemV output, validated against the deposited Filter-2 matrix (MitoDrift's input).

Stages, mirrored from chenweng1991/redeemR@redeemR2.0 (R/variant_summary.R, R/filter2.R,
R/buid_tree.R, R/variant_annotation.R):

 1. redeemR.read.trim   RawGenotypes.<thr>.StrandBalance; molecules within `edge_trim` bp
                        of a fragment end dropped; Freq = molecules per (cell, variant);
                        depth = redeemV depth - trimmed molecules; variants in an N
                        reference context dropped (rCRS 3107).
 2. DepthSummary        per-position and per-cell mean of the TOTAL column of
                        QualifiedTotalCts, over the rows present.
 3. Vfilter_v4          qualified cells: mean coverage >= 10. Per variant over qualified
                        cells: CellN, PositiveMean (mean hetero), maxcts, CV = var/mean of
                        hetero (a variance-mean ratio), CellNPCT = CellN / cells with
                        coverage at the site. Homoplasmic: CellNPCT > .75 & PositiveMean >
                        .75 & CV < .01. Keep CellN >= 2.
 4. Make_matrix         counts over the cells carrying any heteroplasmic candidate.
 5. binomial noise      chi2 goodness of fit of a variant's counts to ONE binomial at the
                        site's mean (Total) coverage, df = coverage; q-values (Storey).
 6. clean_redeem        Hetero, q <= 0.05, CellN >= 2.
 7. blacklist / RSRS50  rCRS 302-315, 513-525, 3105-3109, 16182-16187; heteroplasmic
                        variants flagged RSRS50 == "Yes" in MITOMAP.
 8. median depth        median per-cell depth >= 5 (and CellNPCT > .75 removed again).

Alternative to 5 ("tarone"): Tarone's score test for binomial overdispersion with each
cell's own depth, over every qualified cell, BH-corrected.

Usage: python filter2_port.py
"""

import os
import sys

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests

sys.path.insert(0, os.path.dirname(__file__))
import datasets as ds  # noqa: E402

RESULTS = os.path.join(os.path.dirname(__file__), 'results')
CACHE = os.path.join(ds.PL1000, 'cache')
BLACKLIST = set(range(302, 316)) | set(range(513, 526)) | set(range(3105, 3110)) | set(range(16182, 16188))
N_CONTEXT = {3106, 3107, 3108}     # trinucleotide context containing rCRS 3107N
THR_COL = {'T': 'Total', 'LS': 'VerySensitive', 'S': 'Sensitive', 'VS': 'Specific'}


def qvalue(p):
    """Storey q-values, pi0 from the bootstrap-free smoother approximation (lambda = .5)."""
    p = np.asarray(p, float)
    pi0 = min(1.0, np.mean(p > 0.5) / 0.5) if p.size else 1.0
    return np.minimum(multipletests(p, method='fdr_bh')[1] * pi0, 1.0)


def depth_tables(thr='S'):
    """Total-column means (per position, per cell) and the <thr> depth as a dense matrix."""
    path = os.path.join(CACHE, f'qtc_{thr}.npz')
    if os.path.exists(path):
        z = np.load(path, allow_pickle=True)
        return (pd.Series(z['pos_mean'], index=z['pos_idx']), pd.Series(z['cell_mean'], index=z['cells']),
                z['cells'], z['DP'])
    cols = ['Cell', 'Pos', 'Total', 'VerySensitive', 'Sensitive', 'Specific']
    q = pd.read_csv(os.path.join(ds.PL1000, 'QualifiedTotalCts'), sep='\t', header=None, names=cols,
                    usecols=['Cell', 'Pos', 'Total', THR_COL[thr]],
                    dtype={'Cell': 'category', 'Pos': np.int32, 'Total': np.float32, THR_COL[thr]: np.float32})
    pos_mean = q.groupby('Pos')['Total'].mean()
    cell_mean = q.groupby('Cell', observed=True)['Total'].mean()
    cells = q['Cell'].cat.categories.values.astype(str)
    DP = np.zeros((cells.size, 16569), dtype=np.int32)
    DP[q['Cell'].cat.codes.values, q['Pos'].values - 1] = q[THR_COL[thr]].values.astype(np.int32)
    np.savez(path, pos_mean=pos_mean.values, pos_idx=pos_mean.index.values,
             cell_mean=cell_mean.reindex(cells).values, cells=cells, DP=DP)
    return pos_mean, cell_mean.reindex(cells), cells, DP


def read_trim(thr='S', edge_trim=9):
    cols = ['UMI', 'Cell', 'Pos', 'Variants', 'Call', 'Ref', 'FamSize', 'GT_Cts', 'CSS',
            'DB_Cts', 'SG_Cts', 'Plus', 'Minus', 'Depth']
    raw = pd.read_csv(os.path.join(ds.PL1000, f'RawGenotypes.{THR_COL[thr]}.StrandBalance'), sep='\t',
                      header=None, names=cols, usecols=['UMI', 'Cell', 'Pos', 'Variants', 'Depth'],
                      dtype={'UMI': str, 'Cell': str, 'Pos': np.int32, 'Variants': str, 'Depth': np.float32})
    raw = raw[~raw['Pos'].isin(N_CONTEXT)]
    se = raw['UMI'].str.rsplit('_', n=2, expand=True)
    a, b = se[1].astype(np.int64), se[2].astype(np.int64)
    start, end = np.minimum(a, b), np.maximum(a, b)
    raw['edge_dist'] = np.minimum((raw['Pos'] - start).abs(), (raw['Pos'] - end).abs())

    before = raw.groupby(['Cell', 'Variants']).size().rename('Freq_before')
    after = raw[raw['edge_dist'] >= edge_trim].groupby(['Cell', 'Variants']).size().rename('Freq')
    depth = raw.groupby(['Cell', 'Pos'])['Depth'].first()
    gt = after.to_frame().join(before, how='inner').reset_index()
    gt['Pos'] = gt['Variants'].str.split('_').str[0].astype(int)
    gt['depth'] = depth.reindex(pd.MultiIndex.from_frame(gt[['Cell', 'Pos']])).values - (gt['Freq_before'] - gt['Freq'])
    gt['hetero'] = gt['Freq'] / gt['depth']
    return gt


def tarone(x, n):
    """Tarone (1979) score test of binomial overdispersion; one-sided p-value."""
    p = x.sum() / max(n.sum(), 1)
    if p <= 0 or p >= 1:
        return 1.0
    S = ((x - n * p) ** 2).sum() / (p * (1 - p)) - n.sum()
    V = 2 * (n * (n - 1)).sum()
    return float(stats.norm.sf(S / np.sqrt(V))) if V > 0 else 1.0


def gof(trials, coverage):
    """redeemR goodness_of_fit_test, verbatim semantics."""
    trials = np.asarray(trials, int)
    coverage = max(int(coverage), int(trials.max()))
    obs = np.bincount(trials, minlength=coverage + 1)[:coverage + 1]
    p_hat = trials.sum() / (trials.size * coverage)
    exp = trials.size * stats.binom.pmf(np.arange(coverage + 1), coverage, p_hat)
    exp[exp == 0] = 1e-318
    chi2 = ((obs - exp) ** 2 / exp).sum()
    return chi2, stats.chi2.sf(chi2, df=coverage)


def filter2(thr='S', edge_trim=9, test='chi2', median_depth=5, rsrs50=None):
    gt = read_trim(thr, edge_trim)
    pos_mean, cell_mean, dcells, DP = depth_tables(thr)

    qualified = set(cell_mean.index[cell_mean >= 10])
    g = gt[gt['Cell'].isin(qualified)]
    f = g.groupby('Variants').agg(CellN=('Cell', 'size'), PositiveMean=('hetero', 'mean'),
                                  maxcts=('Freq', 'max'),
                                  var=('hetero', lambda h: h.var(ddof=1)))
    f['CV'] = f['var'] / f['PositiveMean']
    f['pos'] = f.index.str.split('_').str[0].astype(int)
    qmask = np.isin(dcells, list(qualified))
    nonzero = (DP[qmask] > 0).sum(0)
    f['CellNPCT'] = f['CellN'] / nonzero[f['pos'].values - 1]
    f['Homo'] = (f['CellNPCT'] > 0.75) & (f['PositiveMean'] > 0.75) & (f['CV'] < 0.01)
    f = f[(f['CellN'] >= 2) & (f['maxcts'] >= 1)].copy()

    # Count matrix over the cells carrying any heteroplasmic candidate
    het = f.index[~f['Homo']]
    gh = g[g['Variants'].isin(het)]
    mat_cells = gh['Cell'].unique()
    M = gh.pivot_table(index='Cell', columns='Variants', values='Freq', aggfunc='sum', fill_value=0)
    M = M.reindex(index=mat_cells, columns=het, fill_value=0)

    if test == 'chi2':
        pv = np.array([gof(M[v].values, pos_mean.get(f.loc[v, 'pos'], 0))[1] for v in het])
        f['q'] = np.nan
        f.loc[het, 'q'] = qvalue(pv)
    else:  # tarone over every qualified cell, with its own depth
        ci = pd.Index(dcells).get_indexer(sorted(qualified))
        counts = g[g['Variants'].isin(het)].pivot_table(index='Cell', columns='Variants', values='Freq',
                                                        aggfunc='sum', fill_value=0)
        counts = counts.reindex(index=sorted(qualified), columns=het, fill_value=0)
        pv = np.array([tarone(counts[v].values, np.maximum(DP[ci, f.loc[v, 'pos'] - 1], counts[v].values))
                       for v in het])
        f['q'] = np.nan
        f.loc[het, 'q'] = multipletests(pv, method='fdr_bh')[1]

    keep = f[(~f['Homo']) & (f['q'] <= 0.05) & (f['CellN'] >= 2)].copy()
    keep = keep[~keep['pos'].isin(BLACKLIST)]
    if rsrs50 is not None:
        keep = keep[~keep.index.isin(rsrs50)]

    # median depth over the object's cells (cells carrying any kept variant)
    obj_cells = gh[gh['Variants'].isin(keep.index)]['Cell'].unique()
    oi = pd.Index(dcells).get_indexer(obj_cells)
    Dk = DP[oi][:, keep['pos'].values - 1]
    keep['median_depth'] = np.median(Dk, axis=0)
    keep['CellNPCT_depth'] = keep['CellN'] / np.maximum((Dk > 0).sum(0), 1)
    if median_depth:
        keep = keep[(keep['median_depth'] >= median_depth) & (keep['CellNPCT_depth'] <= 0.75)]

    return keep, f, gt


def rsrs50_variants():
    import pyreadr
    m = next(iter(pyreadr.read_r(os.path.join(ds.PL1000, 'mitomap_freq.rda')).values()))
    m = m[m['RSRS50'].astype(str) == 'Yes']
    return set(m['ID'].astype(str))


def validate_entries():
    """
    Entry-level check: the port's trimmed alt counts and depths against the deposited
    matrix, over every cell x variant of it (raw redeemV cells are ATAC barcodes).
    Result on 2026-10-08: 949,314 entries, alt and depth identical in 100% of them.
    """
    dep = pd.read_csv(os.path.join(ds.PL1000, 'pL1000_full_mut_dat.csv'))
    gt = read_trim()
    _, _, dcells, DP = depth_tables('S')
    inv = {v: k for k, v in ds.barcode_map().items()}
    dep['atac'] = dep['cell'].str.replace('-1', '').map(inv)
    m = dep.merge(gt[['Cell', 'Variants', 'Freq', 'depth']], left_on=['atac', 'variant'],
                  right_on=['Cell', 'Variants'], how='left')
    m['Freq'] = m['Freq'].fillna(0)
    di = pd.Index(dcells).get_indexer(m['atac'])
    pos = m['variant'].str.split('_').str[0].astype(int).values
    m['port_d'] = np.where(m['depth'].notna(), m['depth'], DP[di, pos - 1])
    out = dict(entries=len(m), alt_identical=float((m['a'] == m['Freq']).mean()),
               depth_identical=float((m['d'] == m['port_d']).mean()))
    print(out, flush=True)
    return out


def main():
    dep = pd.read_csv(os.path.join(ds.PL1000, 'pL1000_full_mut_dat.csv'), usecols=['variant'])
    D = set(dep['variant'].unique())
    rs = rsrs50_variants()
    rows = []
    for test in ('chi2', 'tarone'):
        for md in (5, 0):
            keep, f, _ = filter2(test=test, median_depth=md, rsrs50=rs)
            K = set(keep.index)
            row = dict(test=test, median_depth=md, n=len(K), deposited=len(D), shared=len(K & D),
                       jaccard=round(len(K & D) / len(K | D), 4),
                       only_port=len(K - D), only_deposited=len(D - K))
            rows.append(row)
            print(row, flush=True)
            keep.to_csv(os.path.join(RESULTS, f'filter2_port_{test}_md{md}.csv'))
    pd.DataFrame(rows).to_csv(os.path.join(RESULTS, 'filter2_port_validation.csv'), index=False)


if __name__ == '__main__':
    validate_entries() if '--entries' in sys.argv else main()
