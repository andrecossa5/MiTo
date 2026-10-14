"""
Strand-aware view of the pL1000 mgatk output, for the read-level filter study.

`mt.io.make_afm` sums forward and reverse counts; here they are kept, for the dominant
alternative allele of each site (the same allele the reader keeps), together with the
site coverage, as (cells x sites) matrices over GEX barcodes. Signac's per-variant
statistics (IdentifyVariants) are recomputed from them.
"""

import os

import numpy as np
import pandas as pd
from scipy import sparse

import datasets as ds

BASES = ['A', 'C', 'G', 'T']


def _read_base(path):
    return pd.read_csv(path, header=None, names=['pos', 'cell', 'fw', 'rev'],
                       dtype={'pos': np.int32, 'cell': 'category', 'fw': np.int32, 'rev': np.int32})


def load(cache=True):
    """
    Returns dict with:
        cells (GEX names), sites (positions), ref, alt (dominant allele per site),
        FW, REV : sparse int32 (cells x sites), alt counts of the dominant allele per strand
        DP : dense int32 (cells x sites), coverage
    """

    path = os.path.join(ds.PL1000, 'cache', 'mgatk_strand.npz')
    if cache and os.path.exists(path):
        z = np.load(path, allow_pickle=True)
        return dict(cells=z['cells'], sites=z['sites'], ref=z['ref'], alt=z['alt'],
                    FW=sparse.csr_matrix((z['fw_data'], z['fw_ind'], z['fw_ptr']), shape=tuple(z['shape'])),
                    REV=sparse.csr_matrix((z['rv_data'], z['rv_ind'], z['rv_ptr']), shape=tuple(z['shape'])),
                    DP=z['DP'])

    ref = pd.read_csv(os.path.join(ds.PL1000, 'chrM_refAllele.txt'), sep='\t', header=None,
                      names=['pos', 'ref']).set_index('pos')['ref'].str.upper()
    cov = pd.read_csv(os.path.join(ds.PL1000, 'pL1000_mgatk.coverage.txt.gz'), header=None,
                      names=['pos', 'cell', 'DP'],
                      dtype={'pos': np.int32, 'cell': 'category', 'DP': np.int32})
    cells = cov['cell'].cat.categories
    sites = np.arange(1, ref.index.max() + 1)
    ci = pd.Index(cells)

    # Alt reads of every non-reference base, per strand
    tables = {}
    support = np.zeros((len(BASES), sites.size))
    for b, base in enumerate(BASES):
        df = _read_base(os.path.join(ds.PL1000, f'pL1000_mgatk.{base}.txt.gz'))
        df = df[df['pos'].map(ref).values != base]
        tables[base] = df
        np.add.at(support[b], df['pos'].values - 1, (df['fw'] + df['rev']).values)
    dominant = np.array(BASES)[support.argmax(0)]
    dominant[support.max(0) == 0] = 'N'

    def mat(col):
        parts = []
        for base, df in tables.items():
            d = df[dominant[df['pos'].values - 1] == base]
            r = ci.get_indexer(d['cell'].astype(str))
            ok = r >= 0
            parts.append(sparse.csr_matrix((d[col].values[ok], (r[ok], d['pos'].values[ok] - 1)),
                                           shape=(len(cells), sites.size)))
        return sum(parts).tocsr()

    FW, REV = mat('fw'), mat('rev')
    DP = np.zeros((len(cells), sites.size), dtype=np.int32)
    DP[cov['cell'].cat.codes.values, cov['pos'].values - 1] = cov['DP'].values

    # ATAC -> GEX names
    bmap = ds.barcode_map()
    gex = np.array([f'{bmap.get(c.replace("-1", ""), c.replace("-1", ""))}-1' for c in cells])

    out = dict(cells=gex, sites=sites, ref=ref.reindex(sites).values, alt=dominant,
               FW=FW, REV=REV, DP=DP)
    if cache:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        np.savez(path, cells=gex, sites=sites, ref=out['ref'], alt=dominant, DP=DP,
                 shape=np.array(FW.shape),
                 fw_data=FW.data, fw_ind=FW.indices, fw_ptr=FW.indptr,
                 rv_data=REV.data, rv_ind=REV.indices, rv_ptr=REV.indptr)
    return out


def signac_stats(FW, REV, DP, low_coverage_threshold=10, cells=None, chunk=2000):
    """
    Signac IdentifyVariants statistics per site (columns of FW/REV/DP), optionally over a
    subset of cells (boolean mask). Mirrors stuart-lab/signac R/mito.R. Computed in site
    chunks, so memory stays bounded on genome-wide input.
    """

    if cells is not None:
        FW, REV, DP = FW[cells], REV[cells], DP[cells]
    FW, REV = FW.tocsc(), REV.tocsc()
    out = []
    for s in range(0, DP.shape[1], chunk):
        e = min(s + chunk, DP.shape[1])
        fw = FW[:, s:e].toarray().astype(float)
        rv = REV[:, s:e].toarray().astype(float)
        cov = DP[:, s:e].astype(float)
        alt = fw + rv

        bulk = alt.sum(0) / np.maximum(cov.sum(0), 1)
        af = np.divide(alt, cov, out=np.zeros_like(alt), where=cov > 0)
        # stabilize_variance: low-coverage cells take the bulk frequency
        af_s = np.where(cov < low_coverage_threshold, bulk[None, :], af)
        vmr = af_s.var(0, ddof=1) / (bulk + 1e-11)

        # Pearson fw vs rev over cells with any count on either strand
        any_ = (fw > 0) | (rv > 0)
        n = any_.sum(0)
        with np.errstate(all='ignore'):
            mf = np.where(any_, fw, 0).sum(0) / n
            mr = np.where(any_, rv, 0).sum(0) / n
            dfw = np.where(any_, fw - mf, 0)
            drv = np.where(any_, rv - mr, 0)
            corr = (dfw * drv).sum(0) / np.sqrt((dfw ** 2).sum(0) * (drv ** 2).sum(0))

        n_pos = (alt > 0).sum(0)
        out.append(pd.DataFrame(dict(
            mean=bulk, vmr=vmr, strand_correlation=corr,
            n_cells_conf_detected=((fw >= 2) & (rv >= 2)).sum(0),
            n_cells_over_5=(af >= 0.05).sum(0), n_cells_over_10=(af >= 0.10).sum(0),
            n_cells_over_20=(af >= 0.20).sum(0), n_carriers=n_pos,
            mean_AD_in_positives=alt.sum(0) / np.maximum(n_pos, 1),
            mean_DP=cov.mean(0),
        )))

    return pd.concat(out, ignore_index=True)
