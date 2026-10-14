"""
Assemble an Allele Frequency Matrix (AFM) from MT-scLT pre-processing output.

The AFM is the single object every MiTo function consumes, and it holds the minimum
needed to genotype cells and filter MT-SNVs:

    afm.X                 allele frequencies, float32, sparse
    afm.layers['AD']      alternative allele molecules (consensus UMIs; for mtscATAC,
                          molecules read by both mates), int, sparse
    afm.layers['DP']      coverage at the variant's site, in molecules, int, DENSE
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


def _dominant_alt(long, pos_col='pos', base_col='alt', count_col='AD'):
    """
    Keep one alternative allele per MT site: the one with the most alternative reads
    across cells.

    Every site shows all three alternative alleles once a dataset is large enough -
    sequencing error alone puts single reads everywhere (on a 4,637-cell mgatk run,
    16,120 of 16,568 sites had all three). Dropping such sites as "multi-allelic" would
    discard the entire genome, and picking by mere presence is meaningless; the dominant
    allele is the only candidate worth genotyping, and whether it is real is decided
    downstream by the candidate and clonality filters.

    Sites where a second allele also carries real signal are reported in the log: a
    genuine one will be rare, and is a sign of a repeat or an alignment artefact.
    """

    support = long.groupby([pos_col, base_col])[count_col].sum()
    dominant = support.groupby(level=0).idxmax().map(lambda x: x[1])

    runner_up = (
        support.groupby(level=0)
        .apply(lambda x: x.sort_values(ascending=False).iloc[1] / x.max() if len(x) > 1 else 0.0)
    )
    n_ambiguous = int((runner_up >= 0.5).sum())
    if n_ambiguous:
        logging.info(
            f'{n_ambiguous} site(s) where a second alternative allele carries >=50% of the '
            f'dominant one\'s reads: the dominant allele is kept for all of them.'
        )

    keep = long[base_col] == long[pos_col].map(dominant)
    logging.info(f'Keep the dominant alternative allele per site: {int(keep.sum())} of {len(long)} basecalls')

    return long.loc[keep].copy()


##


def _build_afm(AD, COV, cell_meta, scLT_system, pp_method, quality=None):
    """
    Assemble the AnnData from aligned (cells x variants) AD and (cells x sites) coverage.

    AD : pd.DataFrame, columns named '<pos>_<ref>><alt>'
    COV : pd.DataFrame, columns are MT positions, defined for every cell
    quality : pd.Series over AD's columns, or None
    """

    # How many alleles per site is the reader's decision (`multi_allelic_sites`): by
    # default one, the dominant alternative allele.
    var = AD.columns.to_series().to_frame('mut')
    var['pos'] = var['mut'].map(lambda x: int(x.split('_')[0]))
    var['ref'] = var['mut'].map(lambda x: x.split('_')[1].split('>')[0])
    var['alt'] = var['mut'].map(lambda x: x.split('_')[1].split('>')[1])
    var = var.sort_values(['pos', 'alt'])
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


def _reads_per_molecule(fw, rev, pos, min_reads=5):
    """
    Reads per molecule of a read-count assay (mtscATAC), from the strand counts alone.

    A molecule is read by both mates (one forward and one reverse read, fraction `o`) or by
    one (a single read, forward with the site's probability `s`). Paired reads leave
    fw - s * R unchanged; single reads add binomial noise, so per site

        kappa = sum (fw - s R)^2 / sum R s (1 - s) = (1 - o) / (1 + o)

    over cells, and f = 1 + o = 2 / (1 + kappa). The median over well-covered,
    strand-balanced sites is returned, within [1, 2]: PCR duplicates (same-strand copies)
    would push kappa above 1, a sign that duplicates were not removed upstream.

    fw, rev, pos: per (cell, site) forward reads, reverse reads (all bases) and position.
    On pL1000 (Multiome), 1.66, against 1.73 from the ReDeeM molecules of the same cells.
    """

    df = pd.DataFrame({'fw':fw, 'R':fw + rev, 'pos':pos})
    tot = df.groupby('pos')[['fw', 'R']].sum()
    s = (tot['fw'] / tot['R'].clip(lower=1)).rename('s')
    df = df.join(s, on='pos')
    df['num'] = (df['fw'] - df['s'] * df['R'])**2
    df['den'] = df['R'] * df['s'] * (1 - df['s'])
    site = df.groupby('pos')[['num', 'den', 'R']].agg({'num':'sum', 'den':'sum', 'R':'mean'})
    ok = (site['R'] >= min_reads) & s.reindex(site.index).between(0.2, 0.8) & (site['den'] > 0)
    if not ok.any():
        logging.warning('Too little coverage to estimate reads per molecule: assuming 1.')
        return 1.0
    kappa = float(np.median(site.loc[ok, 'num'] / site.loc[ok, 'den']))
    f = 2 / (1 + kappa)
    if f < 1:
        logging.warning(
            f'Strand counts are over-dispersed (kappa={kappa:.2f}): PCR duplicates were '
            f'probably not removed upstream. Reads per molecule set to 1.'
        )
    return float(np.clip(f, 1.0, 2.0))


##


def _read_allelic_tables(path_ch_matrix, path_meta=None, sample=None,
                         scLT_system='MAESTER', pp_method='maegatk', ref='rCRS',
                         multi_allelic_sites=False, molecules=None):
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

    `molecules` (default: for mtscATAC) converts read counts to molecules. mtscATAC reads
    are not UMI-collapsed: a molecule covered by both mates of a short fragment is counted
    twice, and sequencing errors are single reads. Alternative molecules are those seen by
    both mates, min(forward, reverse), and coverage is reads / f, with f the reads per
    molecule estimated from the strand counts (`_reads_per_molecule`), stored in
    .uns["reads_per_molecule"].
    """

    if molecules is None:
        molecules = scLT_system == 'mtscATAC'

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
        L.append(df[['pos', 'cell', 'counts', 'count_fw', 'count_rev', 'qual']].assign(base=base))

    if not has_quality:
        logging.info('No base qualities in the allelic tables: var["quality"] is not set.')

    logging.info('Format all basecalls in a long table')
    long = pd.concat(L)
    if sample is not None:
        long['cell'] = long['cell'].map(lambda x: f'{x}_{sample}')
    long['ref'] = long['pos'].map(ref_base)

    f = 1.0
    if molecules:
        strands = long.groupby(['pos', 'cell'], observed=True)[['count_fw', 'count_rev']].sum().reset_index()
        f = _reads_per_molecule(strands['count_fw'].values, strands['count_rev'].values,
                                strands['pos'].values)
        logging.info(f'Reads per molecule, from the strand counts: {f:.2f}')
        del strands

    # Alternative basecalls only, one alternative allele per (cell, site)
    logging.info('Filter variant allele basecalls')
    long = long.query('base!=ref').rename(columns={'counts':'AD', 'base':'alt'}).copy()
    logging.info(f'Alternative basecalls: {long.shape[0]}')
    if not multi_allelic_sites:
        long = _dominant_alt(long)
    if molecules:
        long['AD'] = np.minimum(long['count_fw'], long['count_rev'])
        long = long.query('AD>0').copy()
        logging.info(f'Alternative molecules read by both mates: {long.shape[0]} (cell, variant) pairs')

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
    if molecules:
        COV = np.rint(COV / f)

    afm = _build_afm(AD, COV, cell_meta, scLT_system, pp_method, quality=quality)
    if molecules:
        afm.uns['reads_per_molecule'] = f

    return afm


##


REDEEM_THRESHOLDS = ['Total', 'VerySensitive', 'Sensitive', 'Specific']


def _read_redeem(path_ch_matrix, path_meta=None, sample=None, scLT_system='ReDeeM',
                 pp_method='redeem-v', edge_trim=9, treshold='Sensitive',
                 multi_allelic_sites=False, **kwargs):
    """
    Read RedeemV output (Weng et al., 2024).

    `path_ch_matrix` (or its `final/` sub-folder) must hold
    `RawGenotypes.<treshold>.StrandBalance` and `QualifiedTotalCts`. RedeemV emits four
    consensus-stringency levels, from least to most stringent: Total, VerySensitive,
    Sensitive, Specific; `treshold` selects one, and the same level is read from the
    coverage table so that alternative counts and depth agree.

    Basecalls within `edge_trim` bp of a fragment end are discarded, and the coverage of
    the trimmed basecalls is removed from the site's depth. The default, 9, is redeemR's
    (Filter-2, the input of MitoDrift): fragment ends concentrate damage and
    end-repair artefacts.
    """

    if treshold not in REDEEM_THRESHOLDS:
        raise ValueError(f'treshold must be one of {REDEEM_THRESHOLDS}. Provided: {treshold}')

    path_ch_matrix = _resolve_dir(path_ch_matrix)
    path_basecalls = os.path.join(path_ch_matrix, f'RawGenotypes.{treshold}.StrandBalance')
    if not os.path.exists(path_basecalls):
        raise ValueError(f'Missing RedeemV basecalls: {path_basecalls}')
    logging.info(f'Process RedeemV basecalls from: {path_basecalls}')

    # 14 columns: MoleculeID (CB_start_end), CellBC, Pos, Variant, V, Ref, FamSize,
    # V-counts, CSS, DB_Cts, SG_Cts, Is+, Is-, TotalDepth. Only six are used, and the
    # table runs to hundreds of millions of rows on a real run, so the rest are not read.
    cols = ['UMI', 'Cell', 'Pos', 'Variants', 'Call', 'Ref', 'FamSize',
            'GT_Cts', 'CSS', 'DB_Cts', 'SG_Cts', 'Plus', 'Minus', 'Depth']
    used = ['UMI', 'Cell', 'Pos', 'Call', 'Ref', 'Depth']
    basecalls = pd.read_csv(
        path_basecalls, sep='\t', header=None, names=cols,
        usecols=[cols.index(c) for c in used],
        # NB: RedeemV writes the count columns as floats ("29.0"), so Depth is read as
        # float and cast once the arithmetic is done.
        dtype={'UMI':'string', 'Cell':'category', 'Pos':np.int32,
               'Call':'category', 'Ref':'category', 'Depth':np.float32}
    )
    # NB: Ref/Call are categorical (memory), so they are cast before concatenation
    basecalls['Variants'] = (
        basecalls['Pos'].astype(str) + '_'
        + basecalls['Ref'].astype(str) + '>' + basecalls['Call'].astype(str)
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

    # One alternative allele per site: the dominant one, as in the read-level readers
    long['pos'] = long['Variants'].map(lambda x: int(x.split('_')[0]))
    long['alt'] = long['Variants'].map(lambda x: x.split('_')[1].split('>')[1])
    if not multi_allelic_sites:
        long = _dominant_alt(long.rename(columns={'AD_trimmed':'AD'}), count_col='AD')
        long = long.rename(columns={'AD':'AD_trimmed'})
    logging.info(f'Alternative basecalls kept: {long.shape[0]}')
    if sample is not None:
        long['Cell'] = long['Cell'].map(lambda x: f'{x}_{sample}')

    # Coverage, for every cell and site
    path_cov = os.path.join(path_ch_matrix, 'QualifiedTotalCts')
    if not os.path.exists(path_cov):
        raise ValueError(f'Missing RedeemV coverage table: {path_cov}')
    logging.info('Add full site-coverage matrix')
    cov_cols = ['Cell', 'Pos'] + REDEEM_THRESHOLDS
    cov = pd.read_csv(
        path_cov, sep='\t', header=None, names=cov_cols,
        usecols=[0, 1, cov_cols.index(treshold)],
        dtype={'Cell':'category', 'Pos':np.int32, treshold:np.int32}
    ).rename(columns={treshold:'DP'})
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
    multi_allelic_sites: bool = False,
    molecules: bool = None,
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
    multi_allelic_sites : bool, optional
        Keep every alternative allele observed at a site, one variant each (as redeemR).
        By default only the dominant one is kept: sequencing error alone puts all three
        alleles at nearly every site of a large dataset. Default is False.
    molecules : bool, optional
        maegatk / mgatk tables only: convert read counts to molecules (alternative
        molecules read by both mates; coverage divided by the reads per molecule, estimated
        from the strand counts). None (default) decides from the assay: on for mtscATAC,
        whose reads are not UMI-collapsed, off for MAESTER (UMI consensus counts).
    **kwargs
        Reader-specific options (RedeemV: `edge_trim`, default 9; `treshold`, default
        "Sensitive").

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
        afm = _read_redeem(path_ch_matrix, path_meta, sample, scLT_system, pp_method,
                           multi_allelic_sites=multi_allelic_sites, **kwargs)
    else:
        afm = _read_allelic_tables(path_ch_matrix, path_meta, sample, scLT_system, pp_method, ref,
                                   multi_allelic_sites=multi_allelic_sites, molecules=molecules)

    logging.info(f'Allele Frequency Matrix: cell x char {afm.shape}. {T.stop()}')

    return afm


##


def migrate_afm(afm: AnnData, multi_allelic_sites: bool = False, copy: bool = False) -> AnnData | None:
    """
    Bring an AFM written by MiTo < 0.3 up to the current contract.

    Before 0.3 an AFM carried two coverage layers: "site_coverage", the coverage of the
    variant's site, and "DP", the same values *masked* to the cells with an alternative
    read - so DP was 0 for the majority of the matrix. The pipeline now expects a single
    "DP" layer holding the site's coverage for every cell, and base quality as a .var
    column, so an old object read from disk has to be converted: used as it is, every
    negative cell would have a zero denominator and the genotyping would see no evidence
    of absence anywhere.

    Earlier readers also kept every alternative allele of a site; as `make_afm` does now,
    only the dominant one (most alternative reads across cells) is kept by default.

    Parameters
    ----------
    afm : AnnData
        AFM read from a file written by an earlier version.
    multi_allelic_sites : bool, optional
        Keep every alternative allele of a site. Default is False.
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

    if not multi_allelic_sites:
        pos = afm.var_names.str.split('_').str[0]
        support = pd.Series(AD.sum(0), index=afm.var_names)
        dominant = support.groupby(pos.values).idxmax()
        n0 = afm.shape[1]
        afm._inplace_subset_var(afm.var_names.isin(dominant.values))
        logging.info(f'Keep the dominant alternative allele per site: {afm.shape[1]} of {n0} MT-SNVs')

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
