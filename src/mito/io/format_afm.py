"""
Assemble an Allele Frequency Matrix (AFM) from MT-scLT pre-processing output.

The AFM is the single object every MiTo function consumes, and it holds the minimum
needed to genotype cells and filter MT-SNVs:

    afm.X                 allele frequencies, float32, sparse
    afm.layers['AD']      alternative allele (consensus UMI) counts, int, sparse
    afm.layers['DP']      coverage at the variant's site, int, DENSE
    afm.obs               cell metadata, plus per-cell coverage metrics
    afm.var               'pos', 'ref', 'alt' and, where available, 'quality'
    afm.uns               'scLT_system', 'pp_method'

DP is the coverage of the SITE, defined for every cell, not only for the cells with an
alternative basecall: a cell covered 200x with no alt read is evidence of absence, and
storing 0 there (as a per-variant depth would) is what makes genotyping unable to tell
"not covered" from "covered, no alt". It is dense on purpose - coverage is non-zero
almost everywhere, and a CSR matrix of a 92%-dense matrix costs more than the values.
"""

import glob
import logging
import os
import warnings

import numpy as np
import pandas as pd
from anndata import AnnData
from scipy.sparse import csr_matrix

from mito.ut.positions import mask_mt_sites
from mito.ut.utils import Timer, path_assets

warnings.filterwarnings("ignore")

##


SCLT_SYSTEMS = ['MAESTER', 'mtscATAC', 'ReDeeM']
PP_METHODS = {'MAESTER':['maegatk', 'mgatk'], 'mtscATAC':['mgatk', 'maegatk'], 'ReDeeM':['redeem-v']}

# MAESTER enriches a panel of MT sites: the coverage of those sites is what qualifies a
# cell. The other systems cover the genome uniformly, so only the genome-wide metric applies.
TARGETED_SYSTEMS = ['MAESTER']


##


def _coverage_dtype(cov):
    """Smallest integer type that holds the coverage values."""
    return np.int16 if np.nanmax(cov)<=np.iinfo(np.int16).max else np.int32


##


def _resolve_dir(path):
    """
    Both mgatk and RedeemV publish into a `final/` sub-folder; nf-MiTo publishes the
    tables directly. Accept either, so the user can pass the pipeline's output folder.
    """

    final = os.path.join(path, 'final')

    return final if os.path.isdir(final) else path


##


def _find(path, suffix):
    """
    One file ending in `suffix`, with or without a sample-name prefix and with or without
    gzip compression: nf-MiTo writes "A.txt.gz", mgatk "<sample>.A.txt.gz".
    """

    hits = sorted(glob.glob(os.path.join(path, f'*{suffix}'))
                  + glob.glob(os.path.join(path, f'*{suffix}.gz')))
    hits = [ x for x in hits if os.path.basename(x).endswith((suffix, f'{suffix}.gz')) ]
    if not hits:
        return None
    if len(hits)>1:
        raise ValueError(f'Several files match *{suffix} in {path}: {[os.path.basename(x) for x in hits]}')

    return hits[0]


##


def _read_reference(ref='rCRS', path=None):
    """Reference base at every MT position, 1-based."""

    # mgatk writes the reference it was run against: prefer it, so that a run on a
    # non-human MT genome is not silently read against rCRS.
    path_ref_allele = _find(path, 'refAllele.txt') if path is not None else None
    if path_ref_allele is not None:
        logging.info(f'Use the reference alleles of the pre-processing run: {os.path.basename(path_ref_allele)}')
        df = pd.read_csv(path_ref_allele, sep='\t', header=None, names=['pos', 'ref'])
        return dict(zip(df['pos'].astype(int), df['ref'].str.upper(), strict=True))

    if ref == 'rCRS':
        chrM_path = os.path.join(path_assets, 'chrM.fa')
    elif os.path.exists(ref) and ref.endswith('.fa'):
        chrM_path = ref
    else:
        raise ValueError('Provide "rCRS" or a path to a custom MT reference (FASTA file).')

    with open(chrM_path) as f:
        lines = f.readlines()
    seq = ''.join([ x.strip() for x in lines[1:] ])

    return { pos+1:base for pos, base in enumerate(seq) }


##


def _build_afm(AD, COV, cell_meta, scLT_system, pp_method, quality=None):
    """
    Assemble the AnnData from aligned (cells x variants) AD and (cells x sites) coverage.

    AD : pd.DataFrame, columns named '<pos>_<ref>><alt>'
    COV : pd.DataFrame, columns are MT positions, defined for every cell
    quality : pd.Series over AD's columns, or None
    """

    # Variant metadata, and multi-allelic sites out: a site with two alternative alleles
    # cannot be genotyped one allele at a time (their AFs are not independent).
    var = AD.columns.to_series().to_frame('mut')
    var['pos'] = var['mut'].map(lambda x: int(x.split('_')[0]))
    var['ref'] = var['mut'].map(lambda x: x.split('_')[1].split('>')[0])
    var['alt'] = var['mut'].map(lambda x: x.split('_')[1].split('>')[1])
    n0 = var.shape[0]
    var = var.loc[var['pos'].map(var['pos'].value_counts())==1].sort_values('pos')
    if n0>var.shape[0]:
        logging.info(f'Exclude {n0-var.shape[0]} MT-SNVs at multi-allelic sites')
    AD = AD[var.index].copy()

    # Per-variant coverage: the coverage of its site, for every cell
    missing = set(var['pos']) - set(COV.columns)
    if missing:
        raise ValueError(f'Coverage table misses {len(missing)} of the sites carrying MT-SNVs.')
    DP = COV[var['pos'].values].values
    DP = np.maximum(DP, AD.values).astype(_coverage_dtype(COV.values))

    AF = np.divide(AD.values, np.maximum(DP, 1)).astype(np.float32)
    var = var[['pos', 'ref', 'alt']]
    if quality is not None:
        var['quality'] = quality.reindex(var.index).values

    afm = AnnData(
        X=csr_matrix(AF),
        obs=cell_meta,
        var=var,
        layers={'AD':csr_matrix(AD.values.astype(np.int16)), 'DP':DP},
        uns={'scLT_system':scLT_system, 'pp_method':pp_method}
    )

    # Per-cell coverage metrics: genome-wide always, target-panel where one applies
    afm.obs['mean_site_coverage'] = COV.mean(axis=1).loc[afm.obs_names].values
    if scLT_system in TARGETED_SYSTEMS:
        target = mask_mt_sites(COV.columns)
        afm.obs['median_target_site_coverage'] = COV.loc[:,target].median(axis=1).loc[afm.obs_names].values
        afm.obs['frac_target_site_covered'] = (
            (COV.loc[:,target]>0).sum(axis=1) / target.sum()
        ).loc[afm.obs_names].values

    return afm


##


def _read_allelic_tables(path_ch_matrix, path_meta=None, sample=None,
                         scLT_system='MAESTER', pp_method='maegatk', ref='rCRS'):
    """
    Read maegatk / mgatk per-base allelic tables.

    `path_ch_matrix` (or its `final/` sub-folder) must hold the per-base tables and the
    coverage table, named `A.txt[.gz]` … `coverage.txt[.gz]` with or without a sample-name
    prefix (nf-MiTo writes the former, mgatk the latter).

    Table layouts differ between tools and modes, and are detected from the column count:

    * 4 columns - pos, cell, forward count, reverse count. This is what mgatk writes for
      10x data unless it is run with `--emit-base-qualities`: there are no base qualities,
      so the AFM gets no `var["quality"]` and the filters that would use it are skipped.
    * 6 columns - the same, with a mean base quality per strand (maegatk; mgatk with
      `-eb`).
    * more - extra consensus/group-size columns (mito_preprocessing-style), ignored.
    """

    path_ch_matrix = _resolve_dir(path_ch_matrix)
    ref_base = _read_reference(ref, path=path_ch_matrix)

    L = []
    has_quality = True
    for base in ['A', 'C', 'T', 'G']:
        path_base = _find(path_ch_matrix, f'{base}.txt')
        if path_base is None:
            raise ValueError(
                f'Missing the {base} allelic table in {path_ch_matrix}. Expected '
                f'"{base}.txt(.gz)", with or without a sample-name prefix.'
            )
        logging.info(f'Process table: {os.path.basename(path_base)}')
        df = pd.read_csv(path_base, header=None)
        if df.shape[1] == 4:                      # no base qualities (mgatk, 10x mode)
            df.columns = ['pos', 'cell', 'count_fw', 'count_rev']
            df['qual_fw'] = df['qual_rev'] = np.nan
            has_quality = False
        elif df.shape[1] == 6:
            df.columns = ['pos', 'cell', 'count_fw', 'qual_fw', 'count_rev', 'qual_rev']
        elif df.shape[1]>6:                       # extra consensus/group-size columns
            df = pd.concat([df.iloc[:,:2], df.iloc[:,[2,3,6,7]]], axis=1)
            df.columns = ['pos', 'cell', 'count_fw', 'qual_fw', 'count_rev', 'qual_rev']
        else:
            raise ValueError(
                f'Unexpected layout in {os.path.basename(path_base)}: {df.shape[1]} columns.'
            )
        df['counts'] = df['count_fw'] + df['count_rev']
        qual = df[['qual_fw', 'qual_rev']].values
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', RuntimeWarning)   # basecalls with no quality
            df['qual'] = np.nanmean(np.where(qual>0, qual, np.nan), axis=1)
        L.append(df[['pos', 'cell', 'counts', 'qual']].assign(base=base))

    if not has_quality:
        logging.info('No base qualities in the allelic tables: var["quality"] is not set.')

    logging.info('Format all basecalls in a long table')
    long = pd.concat(L)
    if sample is not None:
        long['cell'] = long['cell'].map(lambda x: f'{x}_{sample}')
    long['ref'] = long['pos'].map(ref_base)

    # Alternative basecalls only, one alternative allele per (cell, site)
    logging.info('Filter variant allele basecalls')
    long = long.query('base!=ref').copy()
    long['nunique'] = long.groupby(['cell', 'pos'])['base'].transform('nunique')
    long = (
        long.query('nunique==1')
        .drop(columns=['nunique'])
        .rename(columns={'counts':'AD', 'base':'alt'})
    )
    logging.info(f'Unique variant basecalls: {long.shape[0]}')

    # Coverage, for every cell and site
    logging.info('Retrieve cell-site total coverage')
    path_cov = _find(path_ch_matrix, 'coverage.txt')
    if path_cov is None:
        raise ValueError(f'Missing the coverage table in {path_ch_matrix}.')
    cov = pd.read_csv(path_cov, header=None)
    cov.columns = ['pos', 'cell', 'DP']
    if sample is not None:
        cov['cell'] = cov['cell'].map(lambda x: f'{x}_{sample}')

    cell_meta, cells = _select_cells(path_meta, set(long['cell']) & set(cov['cell']))
    long = long.query('cell in @cells').copy()

    long['mut'] = long['pos'].astype(str) + '_' + long['ref'] + '>' + long['alt']
    AD = long.pivot(index='cell', columns='mut', values='AD').fillna(0).reindex(cells).fillna(0)
    quality = long.groupby('mut')['qual'].mean() if has_quality else None
    COV = (
        cov.query('cell in @cells')
        .pivot(index='cell', columns='pos', values='DP')
        .fillna(0).reindex(cells).fillna(0)
    )

    return _build_afm(AD, COV, cell_meta, scLT_system, pp_method, quality=quality)


##


REDEEM_THRESHOLDS = ['Total', 'VerySensitive', 'Sensitive', 'Specific']


def _read_redeem(path_ch_matrix, path_meta=None, sample=None, scLT_system='ReDeeM',
                 pp_method='redeem-v', edge_trim=4, treshold='Sensitive', **kwargs):
    """
    Read RedeemV output (Weng et al., 2024).

    `path_ch_matrix` (or its `final/` sub-folder) must hold
    `RawGenotypes.<treshold>.StrandBalance` and `QualifiedTotalCts`. RedeemV emits four
    consensus-stringency levels, from least to most stringent: Total, VerySensitive,
    Sensitive, Specific; `treshold` selects one, and the same level is read from the
    coverage table so that alternative counts and depth agree.

    Basecalls within `edge_trim` bp of a fragment end are discarded, and the coverage of
    the trimmed basecalls is removed from the site's depth.
    """

    if treshold not in REDEEM_THRESHOLDS:
        raise ValueError(f'treshold must be one of {REDEEM_THRESHOLDS}. Provided: {treshold}')

    path_ch_matrix = _resolve_dir(path_ch_matrix)
    path_basecalls = os.path.join(path_ch_matrix, f'RawGenotypes.{treshold}.StrandBalance')
    if not os.path.exists(path_basecalls):
        raise ValueError(f'Missing RedeemV basecalls: {path_basecalls}')
    logging.info(f'Process RedeemV basecalls from: {path_basecalls}')

    # 14 columns: MoleculeID (CB_start_end), CellBC, Pos, Variant, V, Ref, FamSize,
    # V-counts, CSS, DB_Cts, SG_Cts, Is+, Is-, TotalDepth
    cols = ['UMI', 'Cell', 'Pos', 'Variants', 'Call', 'Ref', 'FamSize',
            'GT_Cts', 'CSS', 'DB_Cts', 'SG_Cts', 'Plus', 'Minus', 'Depth']
    basecalls = pd.read_csv(path_basecalls, sep='\t', header=None, names=cols)
    basecalls['Variants'] = (
        basecalls['Pos'].astype(str) + '_' + basecalls['Ref'] + '>' + basecalls['Call']
    )

    logging.info('Count AD before edge-trimming')
    long = (
        basecalls.groupby(['Cell', 'Variants'])['UMI'].nunique().reset_index()
        .rename(columns={'UMI':'AD_raw'})
        .merge(basecalls[['Cell', 'Variants', 'Depth']].drop_duplicates(),
               on=['Cell', 'Variants'], how='left')
    )

    logging.info(f'Trim basecalls at <{edge_trim}bp from fragment start/end')
    # MoleculeID is <cell barcode>_<start>_<end>: the fragment ends are its last two fields
    split = basecalls['UMI'].str.rsplit('_', n=2, expand=True)
    if split.shape[1]<3:
        raise ValueError(
            'RedeemV MoleculeIDs are not in <barcode>_<start>_<end> format, so basecalls '
            'cannot be edge-trimmed. Pass edge_trim=0 to skip trimming.'
        )
    start, end = split[1].astype(int), split[2].astype(int)
    basecalls['Edge_dist'] = np.minimum(
        (basecalls['Pos']-np.minimum(start, end)).abs(),
        (np.maximum(start, end)-basecalls['Pos']).abs()
    )
    basecalls = basecalls.loc[basecalls['Edge_dist']>=edge_trim].copy()
    long_trim = (
        basecalls.groupby(['Cell', 'Variants'])['UMI'].nunique().reset_index()
        .rename(columns={'UMI':'AD_trimmed'})
    )

    long = (
        long.merge(long_trim, on=['Cell', 'Variants'], how='outer')
        .fillna({'AD_raw':0, 'AD_trimmed':0})
        .assign(n_trimmed=lambda x: x['AD_raw']-x['AD_trimmed'],
                DP=lambda x: x['Depth']-x['n_trimmed'])
        .query('AD_trimmed>0')
    )

    # One alternative allele per (cell, site)
    long['Pos'] = long['Variants'].map(lambda x: int(x.split('_')[0]))
    long['Alt'] = long['Variants'].map(lambda x: x.split('_')[1].split('>')[1])
    long['nunique'] = long.groupby(['Cell', 'Pos'])['Alt'].transform('nunique')
    long = long.query('nunique==1').copy()
    logging.info(f'Unique variant basecalls: {long.shape[0]}')
    if sample is not None:
        long['Cell'] = long['Cell'].map(lambda x: f'{x}_{sample}')

    # Coverage, for every cell and site
    path_cov = os.path.join(path_ch_matrix, 'QualifiedTotalCts')
    if not os.path.exists(path_cov):
        raise ValueError(f'Missing RedeemV coverage table: {path_cov}')
    logging.info('Add full site-coverage matrix')
    cov = pd.read_csv(path_cov, sep='\t', header=None)
    cov.columns = ['Cell', 'Pos'] + REDEEM_THRESHOLDS
    cov = cov[['Cell', 'Pos', treshold]].rename(columns={treshold:'DP'})
    if sample is not None:
        cov['Cell'] = cov['Cell'].map(lambda x: f'{x}_{sample}')

    cell_meta, cells = _select_cells(path_meta, set(long['Cell']) & set(cov['Cell']))
    long = long.query('Cell in @cells').copy()
    AD = (
        long.pivot(index='Cell', columns='Variants', values='AD_trimmed')
        .fillna(0).reindex(cells).fillna(0)
    )
    COV = (
        cov.query('Cell in @cells').pivot(index='Cell', columns='Pos', values='DP')
        .fillna(0).reindex(cells).fillna(0)
    )

    afm = _build_afm(AD, COV, cell_meta, scLT_system, pp_method)

    # Edge-trimming removes reads from the site's depth: put the corrected values back,
    # where they exist, so that AF and coverage refer to the same basecalls.
    corrected = (
        long.pivot(index='Cell', columns='Variants', values='DP').reindex(cells)
        .reindex(columns=afm.var_names)
    )
    test = corrected.notna().values
    afm.layers['DP'][test] = corrected.values[test].astype(afm.layers['DP'].dtype)

    return afm


##


def _select_cells(path_meta, observed):
    """
    Cells to assemble the AFM for: those in the metadata, if provided, else all observed.
    """

    if path_meta not in [None, 'null'] and os.path.exists(path_meta):
        logging.info('Filter for annotated cells (i.e., sample CBs in cell_meta)')
        cell_meta = pd.read_csv(path_meta, index_col=0)
        cells = [ x for x in cell_meta.index if x in observed ]
        if not cells:
            raise ValueError(
                'No cell in the metadata matches the pre-processing output. Cell names '
                'must agree: pass sample=<name> if the metadata uses {CB}_{sample}.'
            )
        cell_meta = cell_meta.loc[cells].copy()
    else:
        cells = sorted(observed)
        cell_meta = pd.DataFrame(index=cells)

    logging.info(f'Cells: {len(cells)}')

    return cell_meta, cells


##


def make_afm(
    path_ch_matrix: str,
    path_meta: str = None,
    sample: str = None,
    scLT_system: str = 'MAESTER',
    pp_method: str = 'maegatk',
    ref: str = 'rCRS',
    **kwargs
    ) -> AnnData:
    """
    Assemble an Allele Frequency Matrix from MT-scLT pre-processing output.

    Parameters
    ----------
    path_ch_matrix : str
        Folder with the pre-processing output. maegatk/mgatk: A|C|T|G.txt.gz and
        coverage.txt.gz (as published by nf-MiTo). RedeemV: RawGenotypes.<treshold>
        .StrandBalance and QualifiedTotalCts.
    path_meta : str, optional
        .csv with cell metadata, indexed by cell name. Only its cells are retained.
        Cell names must match the pre-processing output, i.e. {CB}_{sample} if `sample`
        is given. Default is None (all observed cells).
    sample : str, optional
        Sample name appended to the cell barcodes. Default is None.
    scLT_system : str, optional
        MT-scLT assay: "MAESTER", "mtscATAC" or "ReDeeM". Default is "MAESTER".
    pp_method : str, optional
        Pre-processing pipeline: "maegatk", "mgatk" or "redeem-v". Default is "maegatk".
    ref : str, optional
        "rCRS", or a path to a custom MT reference (FASTA). Default is "rCRS".
    **kwargs
        Reader-specific options (RedeemV: `edge_trim`, `treshold`).

    Returns
    -------
    AnnData
        The AFM: AF in .X, "AD" and "DP" layers, "pos"/"ref"/"alt" in .var.
    """

    if not os.path.exists(path_ch_matrix):
        raise ValueError(f'{path_ch_matrix} does not exist. Specify a valid path_ch_matrix!')
    if scLT_system not in SCLT_SYSTEMS:
        raise ValueError(f'scLT_system must be one of {SCLT_SYSTEMS}. Provided: {scLT_system}')
    if pp_method not in PP_METHODS[scLT_system]:
        raise ValueError(
            f'pp_method for {scLT_system} must be one of {PP_METHODS[scLT_system]}. '
            f'Provided: {pp_method}'
        )

    logging.info(f'Allele Frequency Matrix generation: {scLT_system} system, {pp_method} pre-processing')
    T = Timer()
    T.start()

    if pp_method == 'redeem-v':
        afm = _read_redeem(path_ch_matrix, path_meta, sample, scLT_system, pp_method, **kwargs)
    else:
        afm = _read_allelic_tables(path_ch_matrix, path_meta, sample, scLT_system, pp_method, ref)

    logging.info(f'Allele Frequency Matrix: cell x char {afm.shape}. {T.stop()}')

    return afm


##


def migrate_afm(afm: AnnData, copy: bool = False) -> AnnData | None:
    """
    Bring an AFM written by MiTo < 0.3 up to the current contract.

    Before 0.3 an AFM carried two coverage layers: "site_coverage", the coverage of the
    variant's site, and "DP", the same values *masked* to the cells with an alternative
    read - so DP was 0 for the majority of the matrix. The pipeline now expects a single
    "DP" layer holding the site's coverage for every cell, and base quality as a .var
    column, so an old object read from disk has to be converted: used as it is, every
    negative cell would have a zero denominator and the genotyping would see no evidence
    of absence anywhere.

    Parameters
    ----------
    afm : AnnData
        AFM read from a file written by an earlier version.
    copy : bool, optional
        Return a converted copy instead of updating `afm` in place. Default is False.

    Returns
    -------
    AnnData | None
        The converted AFM if `copy` is True, otherwise None.
    """

    afm = afm.copy() if copy else afm

    if 'site_coverage' not in afm.layers:
        logging.info('AFM already follows the current contract: nothing to migrate.')
        return afm if copy else None

    logging.info('Migrate AFM: site_coverage -> DP (dense), qual -> var["quality"]')
    AD = afm.layers['AD'].toarray() if hasattr(afm.layers['AD'], 'toarray') else np.asarray(afm.layers['AD'])
    site = afm.layers['site_coverage']
    site = site.toarray() if hasattr(site, 'toarray') else np.asarray(site)
    DP = np.maximum(site, AD)
    afm.layers['DP'] = DP.astype(_coverage_dtype(DP))
    del afm.layers['site_coverage']

    if 'qual' in afm.layers:
        qual = afm.layers['qual']
        qual = qual.toarray() if hasattr(qual, 'toarray') else np.asarray(qual)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', RuntimeWarning)
            afm.var['quality'] = np.nanmean(np.where(qual>0, qual, np.nan), axis=0)
        del afm.layers['qual']

    # Per-position tables are cells x 16569 and nothing reads them any more
    for key in ('per_position_coverage', 'per_position_quality', 'raw_basecalls_metrics'):
        afm.uns.pop(key, None)

    # AF is recomputed, since its denominator has just changed for the negative cells
    afm.X = csr_matrix((AD/np.maximum(DP, 1)).astype(np.float32))

    return afm if copy else None


##


def read_coverage(afm: AnnData, path_coverage: str, sample: str = None) -> pd.DataFrame:
    """
    Read the per-position coverage table of a sample, for the cells of `afm`.

    Returned in long format (cell, pos, coverage), as the coverage plots expect it.
    """

    cov = pd.read_csv(path_coverage, header=None)
    cov.columns = ['pos', 'cell', 'coverage']
    if sample is not None:
        cov['cell'] = cov['cell'].map(lambda x: f'{x}_{sample}')

    return cov.loc[cov['cell'].isin(afm.obs_names)].copy()


##
