"""
Benchmark datasets, each returned as (afm, truth): an AFM in the current MiTo contract and
a pd.Series of ground-truth clone labels over cell names (NaN = unlabelled cell).

    MDA_clones, MDA_lung, MDA_PT   MAESTER, GBC lentiviral barcodes (MiTo paper)
    sim                            simulate_afm panel, MAESTER or ReDeeM regime
    pL1000_filter2                 ReDeeM, redeemR Filter-2 matrix (MitoDrift's input)
    pL1000_redeemv                 ReDeeM, raw redeemV output
    pL1000_mgatk                   mtscATAC, raw mgatk output (same cells)

pL1000 files come from benchmarks/fetch_pl1000.py. The read-heavy AFMs are cached as
.h5ad next to the data, keyed by their reader arguments.
"""

import os

import numpy as np
import pandas as pd
import scanpy as sc
from anndata import AnnData
from scipy.sparse import csr_matrix

import mito as mt

ROOT = os.path.join(os.path.dirname(__file__), '..', 'data_test')
MDA = os.path.join(ROOT, 'source_data', 'data', 'general', 'AFMs')
PL1000 = os.path.join(ROOT, 'bench', 'pL1000')

# Simulation regimes. MAESTER: the reference configuration of the redesign benchmarks.
# ReDeeM: calibrated on pL1000 (depth median ~63, carrier AF median ~0.03, half of the
# carrier observations a single molecule).
SIM_PRESETS = {
    'maester': dict(
        af_beta=(3, 57), af_beta_min=0.04, af_depth_decay=0.85,
        af_beta_noise=(2, 1500), af_negative_frac=0.002,
        mean_coverage=200, coverage_cell_cv=0.2, coverage_site_cv=0.8,
        mean_molecules=40, molecule_cv=0.5, molecule_coverage_exponent=0.0,
        overdispersion_background=1.2, overdispersion_noise_frac=0.5,
        noise_prevalence=(0.05, 0.2),
    ),
    'redeem': dict(
        mean_coverage=70, coverage_cell_cv=0.55, coverage_site_cv=0.15,
        mean_molecules=12, molecule_cv=0.6, af_beta=(0.9, 30), af_beta_min=0.005,
        af_beta_noise=(2, 1500), af_negative_frac=0.002, af_depth_decay=0.85,
        overdispersion_background=1.2, overdispersion_noise_frac=0.5,
        noise_prevalence=(0.02, 0.1),
    ),
}
SIM_GRID = [(nk, topo, seed) for nk in (5, 10, 30, 50)
            for topo in ('polytomy', 'depth3') for seed in (0, 1, 2)]


##


def mda(name, reader='shipped'):
    """
    reader='shipped': the AFM written by MiTo < 0.3 (migrated); 'current': re-read from the
    raw maegatk tables with the reader in the working tree (dominant alt allele per site).
    """
    if reader == 'shipped':
        afm = sc.read(os.path.join(MDA, name, 'afm_unfiltered.h5ad'))
        if 'site_coverage' in afm.layers or 'qual' in afm.layers:
            mt.io.migrate_afm(afm)
    else:
        path = os.path.join(ROOT, 'bench', 'mda_current_reader', f'{name}.h5ad')
        if os.path.exists(path):
            afm = sc.read(path)
        else:
            afm = mt.io.make_afm(os.path.join(MDA, name),
                                 path_meta=os.path.join(MDA, '..', 'cells_meta.csv'),
                                 sample=name, scLT_system='MAESTER', pp_method='maegatk')
            os.makedirs(os.path.dirname(path), exist_ok=True)
            afm.write(path)
    mt.pp.filter_cells(afm, cell_filter='filter2')
    return afm, afm.obs['GBC'].astype(str)


def sim(n_clones, topo, seed, preset='maester', n_cells=1000):
    kw = dict(n_cells=n_cells, n_clones=n_clones, frac_double_variants=0.3,
              frac_noisy_variants=0.2, min_max_ratio_clones=0.2, random_seed=seed,
              **SIM_PRESETS[preset])
    kw.update(dict(n_root_clones=n_clones, max_depth=1) if topo == 'polytomy'
              else dict(n_root_clones=max(2, n_clones // 3), max_depth=3))
    if preset == 'redeem':
        kw.update(scLT_system='ReDeeM', pp_method='redeem-v')
    afm = mt.ut.simulate_afm(**kw)
    return afm, afm.obs['clone'].astype(str)


##


def pl1000_truth(which='final'):
    """LARRY clone per cell (GEX barcodes). 'final': 1,362 cells / 256 clones."""
    df = pd.read_csv(os.path.join(PL1000, f'pL1000_barcode_annot_{which}.csv'), index_col=0)
    return df['annot'].astype(str)


def barcode_map():
    """10x Multiome ATAC -> GEX barcode pairing (redeemR ATACWhite / RNAWhite, by position)."""
    import pyreadr
    atac = pyreadr.read_r(os.path.join(PL1000, 'ATACWhite.rda'))
    rna = pyreadr.read_r(os.path.join(PL1000, 'RNAWhite.rda'))
    a = next(iter(atac.values())).iloc[:, 0].astype(str).values
    r = next(iter(rna.values())).iloc[:, 0].astype(str).values
    return dict(zip(a, r))


def _to_gex(afm, truth):
    """Rename cells ATAC -> GEX barcodes when the AFM does not already use GEX names."""
    if afm.obs_names.isin(truth.index).any():
        return afm
    bmap = barcode_map()
    core = afm.obs_names.str.replace('-1$', '', regex=True)
    new = [f'{bmap[c]}-1' if c in bmap else n for c, n in zip(core, afm.obs_names)]
    afm.obs_names = new
    afm.obs_names_make_unique()
    return afm


def _cached(key, build):
    path = os.path.join(PL1000, 'cache', f'{key}.h5ad')
    if os.path.exists(path):
        return sc.read(path)
    afm = build()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    afm.write(path)
    return afm


##


def pl1000_filter2():
    """
    redeemR Filter-2 matrix (pL1000_full_mut_dat.csv): complete cell x variant grid of alt
    (a) and depth (d) counts. Depth is per variant here, already corrected for trimming.
    """

    def build():
        df = pd.read_csv(os.path.join(PL1000, 'pL1000_full_mut_dat.csv'))
        df['mut'] = df['variant'].str.replace(r'^(\d+)_([ACGTN])_([ACGTN])$', r'\1_\2>\3', regex=True)
        AD = df.pivot(index='cell', columns='mut', values='a').fillna(0)
        DP = df.pivot(index='cell', columns='mut', values='d').reindex_like(AD).fillna(0)
        var = AD.columns.to_series().to_frame('mut')
        var['pos'] = var['mut'].map(lambda x: int(x.split('_')[0]))
        var['ref'] = var['mut'].map(lambda x: x.split('_')[1].split('>')[0])
        var['alt'] = var['mut'].map(lambda x: x.split('>')[1])
        var = var.sort_values('pos')[['pos', 'ref', 'alt']]
        AD, DP = AD[var.index], DP[var.index]
        dp = np.maximum(DP.values, AD.values).astype(np.int32)
        afm = AnnData(
            X=csr_matrix((AD.values / np.maximum(dp, 1)).astype(np.float32)),
            obs=pd.DataFrame(index=AD.index), var=var,
            layers={'AD': csr_matrix(AD.values.astype(np.int16)), 'DP': dp},
            uns={'scLT_system': 'ReDeeM', 'pp_method': 'redeem-v'},
        )
        afm.obs['mean_site_coverage'] = dp.mean(1)
        return afm

    afm = _cached('filter2', build)
    truth = pl1000_truth()
    afm.obs['LARRY'] = truth.reindex(afm.obs_names).values
    return afm, truth


def pl1000_redeemv(treshold='Sensitive', edge_trim=9):
    afm = _cached(
        f'redeemv_{treshold}_trim{edge_trim}',
        lambda: mt.io.make_afm(PL1000, scLT_system='ReDeeM', pp_method='redeem-v',
                               treshold=treshold, edge_trim=edge_trim),
    )
    truth = pl1000_truth()
    afm = _to_gex(afm, truth)
    afm.obs['LARRY'] = truth.reindex(afm.obs_names).values
    return afm, truth


def pl1000_mgatk():
    """mgatk tables are named pL1000_mgatk.<X>.txt.gz; the reader finds them by suffix."""
    afm = _cached(
        'mgatk',
        lambda: mt.io.make_afm(PL1000, scLT_system='mtscATAC', pp_method='mgatk'),
    )
    truth = pl1000_truth()
    afm = _to_gex(afm, truth)
    afm.obs['LARRY'] = truth.reindex(afm.obs_names).values
    return afm, truth
