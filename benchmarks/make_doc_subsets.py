"""
Small real inputs for the "Loading data" tutorial (docs/source/data), in each tool's own
layout, cut from the benchmark datasets:

    maester_MDA_clones/   maegatk tables (nf-MiTo layout), N_CELLS cells, GBC truth
    mtscATAC_pL1000/      mgatk tables (sample-prefixed), N_CELLS LARRY-labelled cells
    redeem_pL1000/        RedeemV RawGenotypes.Sensitive.StrandBalance + QualifiedTotalCts
                          (gzipped), the same cells

Cells are drawn from the largest ground-truth clones (equally many from each), so the
examples have clonal signal. The genome-wide assays (mgatk, RedeemV) are cut to
positions <= MAX_POS to keep the files small (~4 MB); the MAESTER panel is kept whole.
The tables are otherwise untouched.

Usage: python make_doc_subsets.py [n_cells=50] [max_pos=3000]
"""

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
import datasets as ds  # noqa: E402

OUT = os.path.join(os.path.dirname(__file__), '..', 'docs', 'source', 'data')
N_CELLS = int(sys.argv[1]) if len(sys.argv) > 1 else 50
MAX_POS = int(sys.argv[2]) if len(sys.argv) > 2 else 3000    # genome-wide assays only
N_CLONES = 5


def pick(truth, n, k=N_CLONES, seed=0):
    top = truth.value_counts().index[:k]
    rng = np.random.default_rng(seed)
    cells = [rng.choice(truth.index[truth == c], n // k, replace=False) for c in top]
    return list(np.concatenate(cells))


def subset_table(src, dst, cells, cell_col, pos_col=None, sep=',', header=None):
    """Rows of `src` whose cell is in `cells` (and position <= MAX_POS), streamed in chunks."""
    keep = set(cells)
    first = True
    for df in pd.read_csv(src, sep=sep, header=header, chunksize=2_000_000, dtype=str):
        df = df[df[cell_col].isin(keep)]
        if pos_col is not None:
            df = df[df[pos_col].astype(int) <= MAX_POS]
        df.to_csv(dst, sep=sep, header=False, index=False, mode='w' if first else 'a',
                  compression='gzip' if dst.endswith('.gz') else None)
        first = False


def main():
    os.makedirs(OUT, exist_ok=True)

    # MAESTER: MDA_clones maegatk tables, cells named {CB}_{sample} in the metadata
    meta = pd.read_csv(os.path.join(ds.MDA, '..', 'cells_meta.csv'), index_col=0)
    meta = meta[meta['sample'] == 'MDA_clones'] if 'sample' in meta.columns else meta[meta.index.str.endswith('_MDA_clones')]
    cells = pick(meta['GBC'].astype(str), N_CELLS)
    cbs = [c.split('_', 1)[0] for c in cells]
    out = os.path.join(OUT, 'maester_MDA_clones')
    os.makedirs(out, exist_ok=True)
    for t in ('A', 'C', 'G', 'T', 'coverage'):
        subset_table(os.path.join(ds.MDA, 'MDA_clones', f'{t}.txt.gz'), os.path.join(out, f'{t}.txt.gz'), cbs, 1)
    meta.loc[cells, ['GBC']].to_csv(os.path.join(out, 'cells_meta.csv'))

    # pL1000: LARRY truth on GEX barcodes; mgatk / redeemV use ATAC barcodes
    truth = ds.pl1000_truth()
    gex = pick(truth, N_CELLS)
    to_atac = {v: k for k, v in ds.barcode_map().items()}
    atac = [to_atac[g.replace('-1', '')] for g in gex]
    lab = pd.DataFrame({'LARRY': truth[gex].values}, index=atac)

    out = os.path.join(OUT, 'mtscATAC_pL1000')
    os.makedirs(out, exist_ok=True)
    for t in ('A', 'C', 'G', 'T', 'coverage'):
        subset_table(os.path.join(ds.PL1000, f'pL1000_mgatk.{t}.txt.gz'), os.path.join(out, f'pL1000.{t}.txt.gz'),
                     list(lab.index), 1, pos_col=0)
    ref = pd.read_csv(os.path.join(ds.PL1000, 'chrM_refAllele.txt'), sep='\t', header=None)
    ref.to_csv(os.path.join(out, 'pL1000.refAllele.txt'), sep='\t', header=False, index=False)
    lab.to_csv(os.path.join(out, 'cells_meta.csv'))

    out = os.path.join(OUT, 'redeem_pL1000')
    os.makedirs(out, exist_ok=True)
    subset_table(os.path.join(ds.PL1000, 'RawGenotypes.Sensitive.StrandBalance'),
                 os.path.join(out, 'RawGenotypes.Sensitive.StrandBalance.gz'), atac, 1, pos_col=2, sep='\t')
    subset_table(os.path.join(ds.PL1000, 'QualifiedTotalCts'), os.path.join(out, 'QualifiedTotalCts.gz'),
                 atac, 0, pos_col=1, sep='\t')
    pd.DataFrame({'LARRY': truth[gex].values}, index=atac).to_csv(os.path.join(out, 'cells_meta.csv'))

    for root, _, files in os.walk(OUT):
        for f in sorted(files):
            p = os.path.join(root, f)
            print(f'{os.path.relpath(p, OUT):55s} {os.path.getsize(p)/1e6:6.2f} MB')


if __name__ == '__main__':
    main()
