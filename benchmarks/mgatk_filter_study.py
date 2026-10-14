"""
Phase 1A: which read-level candidate filter recovers the informative mgatk variants?

Same pL1000 cells, two assays. Filters are evaluated on statistics computed over ALL
cells of the sample (what a user has), and scored on the 1,362 LARRY-labelled cells:

    R : variants of the redeemR Filter-2 matrix (MitoDrift's ReDeeM input)
    D : variants of MitoDrift's own mgatk input (fr2_mc3)
    L : LARRY-informative mgatk variants - carriers (>= `min_reads` alt reads) enriched in
        one LARRY clone (Fisher, Bonferroni over clones, BH over variants, FDR <= 0.05,
        >= 2 carriers in the clone)

Outputs results/mgatk_filters.csv (one row per filter setting) and
results/mgatk_inflation.csv (read vs UMI counts on the same cell x variant entries).
"""

import itertools
import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import hypergeom
from statsmodels.stats.multitest import multipletests

sys.path.insert(0, os.path.dirname(__file__))
import datasets as ds  # noqa: E402
import mgatk_strand as ms  # noqa: E402

RESULTS = os.path.join(os.path.dirname(__file__), 'results')


def informative(alt, truth_codes, n_clones, min_reads=2, fdr=0.05):
    """Per site: best-clone enrichment of carriers (hypergeometric upper tail)."""
    C = (alt >= min_reads)
    n = C.shape[0]
    clone_size = np.bincount(truth_codes, minlength=n_clones)
    n_car = np.asarray(C.sum(0)).ravel()
    # carriers per clone, per site
    onehot = np.zeros((n, n_clones))
    onehot[np.arange(n), truth_codes] = 1
    K = (onehot.T @ C).T if not hasattr(C, 'toarray') else np.asarray((C.T @ onehot))
    best = K.argmax(1)
    k_best = K[np.arange(K.shape[0]), best]
    p = hypergeom.sf(k_best - 1, n, clone_size[best], n_car)
    p = np.minimum(p * n_clones, 1.0)
    p[(n_car < 2) | (k_best < 2)] = 1.0
    q = multipletests(p, method='fdr_bh')[1]
    return pd.DataFrame(dict(n_carriers_lab=n_car, k_best=k_best,
                             purity=np.where(n_car > 0, k_best / np.maximum(n_car, 1), 0),
                             q=q, L=q <= fdr))


def main():
    d = ms.load()
    sites, ref, alt_base = d['sites'], d['ref'], d['alt']
    vid = np.array([f'{p}_{r}>{a}' for p, r, a in zip(sites, ref, alt_base)])

    # Statistics on all cells (filters), and the deposited Signac stats for vmr/strand
    st = ms.signac_stats(d['FW'], d['REV'], d['DP'])
    st.index = vid
    dep = pd.read_csv(os.path.join(ds.PL1000, 'pL1000_mgatk.variant_stats.tsv.gz'), sep='\t')
    dep['vid'] = dep['position'].astype(str) + '_' + dep['nucleotide']
    dep = dep.set_index('vid')
    st['vmr_dep'] = dep['vmr'].reindex(st.index).values
    st['strand_dep'] = dep['strand_correlation'].reindex(st.index).values
    n_all = d['DP'].shape[0]
    AF = None  # not materialised; confident-detection counts from the stats
    st['n_neg'] = n_all - st['n_carriers']

    # Labelled cells
    truth = ds.pl1000_truth()
    lab = pd.Index(d['cells']).get_indexer(truth.index)
    print(f'labelled cells found in mgatk: {(lab >= 0).sum()} / {truth.size}', flush=True)
    keep = lab[lab >= 0]
    tcodes, tnames = pd.factorize(truth.values[lab >= 0])
    ALT = (d['FW'] + d['REV'])[keep].tocsc()
    inf = informative(ALT, tcodes, len(tnames))
    inf.index = vid
    st = st.join(inf)

    # Reference sets
    f2 = pd.read_csv(os.path.join(ds.PL1000, 'pL1000_full_mut_dat.csv'), usecols=['variant'])
    R = set(f2['variant'].unique().astype(str))
    R = {v.replace('_', '>', 2).replace('>', '_', 1) for v in R}   # 123_A_G -> 123_A>G
    fr2 = pd.read_csv(os.path.join(ds.PL1000, 'pL1000_mgatk.mutations.fr2_mc3.tsv'), usecols=['variant'])
    Dset = {v.replace('_', '>', 2).replace('>', '_', 1) for v in fr2['variant'].unique().astype(str)}
    st['R'] = st.index.isin(R)
    st['D'] = st.index.isin(Dset)
    R_pos = {int(v.split('_')[0]) for v in R}
    st['R_pos'] = [int(v.split('_')[0]) in R_pos for v in st.index]
    print(f'R {len(R)} ({st["R"].sum()} same allele in mgatk, {st["R_pos"].sum()} same site) | '
          f'D {len(Dset)} ({st["D"].sum()} matched) | L {int(st["L"].sum())}', flush=True)
    st.to_csv(os.path.join(ds.PL1000, 'cache', 'mgatk_site_table.csv'))

    # Filter grid
    rows = []
    grid = itertools.product(
        [None, 2, 3, 5, 10],          # n_cells_conf_detected >=
        [None, 0.5, 0.65, 0.8],       # strand correlation (deposited) >=
        [None, 0.001, 0.01, 0.1],     # vmr (deposited) >
        [None, 1.5, 2, 3],            # mean AD in positives >=
    )
    base = (st['mean_DP'] >= 5) & (st['n_neg'] >= 0.5 * n_all) & (st['n_carriers'] >= 2)
    for conf, sc, vmr, ad in grid:
        m = base.copy()
        if conf is not None:
            m &= st['n_cells_conf_detected'] >= conf
        if sc is not None:
            m &= st['strand_dep'] >= sc
        if vmr is not None:
            m &= st['vmr_dep'] > vmr
        if ad is not None:
            m &= st['mean_AD_in_positives'] >= ad
        n = int(m.sum())
        rows.append(dict(conf=conf, strand=sc, vmr=vmr, mean_AD=ad, n_cand=n,
                         recall_L=round(st.loc[m, 'L'].sum() / max(st['L'].sum(), 1), 4),
                         precision_L=round(st.loc[m, 'L'].mean(), 4) if n else np.nan,
                         recall_R=round(st.loc[m, 'R_pos'].sum() / max(st['R_pos'].sum(), 1), 4),
                         recall_D=round(st.loc[m, 'D'].sum() / max(st['D'].sum(), 1), 4),
                         median_purity=round(st.loc[m & (st['n_carriers_lab'] >= 2), 'purity'].median(), 4) if n else np.nan))
    df = pd.DataFrame(rows)
    os.makedirs(RESULTS, exist_ok=True)
    df.to_csv(os.path.join(RESULTS, 'mgatk_filters.csv'), index=False)
    print(df.sort_values('recall_L', ascending=False).query('n_cand<=1000').head(15).to_string(), flush=True)

    # Read vs UMI inflation on shared entries (labelled cells x Filter-2 variants)
    f2 = pd.read_csv(os.path.join(ds.PL1000, 'pL1000_full_mut_dat.csv'))
    f2['vid'] = f2['variant'].str.replace(r'^(\d+)_(\w)_(\w)$', r'\1_\2>\3', regex=True)
    f2 = f2[f2['vid'].isin(st.index)]
    ci = pd.Index(d['cells']).get_indexer(f2['cell'])
    si = pd.Index(st.index).get_indexer(f2['vid'])
    ok = ci >= 0
    ALTall = (d['FW'] + d['REV']).tocsr()
    f2 = f2[ok].assign(
        mg_alt=np.asarray(ALTall[ci[ok], si[ok]]).ravel(),
        mg_dp=d['DP'][ci[ok], si[ok]],
        mg_fw=np.asarray(d['FW'].tocsr()[ci[ok], si[ok]]).ravel(),
        mg_rev=np.asarray(d['REV'].tocsr()[ci[ok], si[ok]]).ravel(),
    )
    f2.to_csv(os.path.join(RESULTS, 'mgatk_inflation.csv'), index=False)
    pos = f2[f2['a'] > 0]
    print('depth ratio mgatk/redeem median', round((f2['mg_dp'] / f2['d'].clip(lower=1)).median(), 3))
    print('alt ratio (redeem a>0) median', round((pos['mg_alt'] / pos['a']).median(), 3),
          '| redeem carriers seen in mgatk', round((pos['mg_alt'] > 0).mean(), 3),
          '| on both strands', round(((pos['mg_fw'] > 0) & (pos['mg_rev'] > 0)).mean(), 3))
    neg = f2[f2['a'] == 0]
    print('redeem a=0 entries with mgatk alt>0', round((neg['mg_alt'] > 0).mean(), 4),
          '| of which both strands', round(((neg['mg_fw'] > 0) & (neg['mg_rev'] > 0)).mean() / max((neg['mg_alt'] > 0).mean(), 1e-9), 3))


if __name__ == '__main__':
    main()
