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

    pos = np.asarray(pos, dtype=np.int64)
    fw = np.asarray(fw, dtype=float)
    R = fw + np.asarray(rev, dtype=float)
    n = int(pos.max()) + 1
    s = np.bincount(pos, weights=fw, minlength=n) / np.maximum(np.bincount(pos, weights=R, minlength=n), 1)
    num = np.bincount(pos, weights=(fw - s[pos]*R)**2, minlength=n)
    den = np.bincount(pos, weights=R*s[pos]*(1 - s[pos]), minlength=n)
    cnt = np.bincount(pos, minlength=n)
    with np.errstate(all='ignore'):
        mean_R = np.bincount(pos, weights=R, minlength=n) / cnt
    return _kappa_to_f(num, den, mean_R, s, min_reads)


def _reads_per_molecule_dense(FW, REV, min_reads=5, block=512):
    """`_reads_per_molecule` on (cells x positions) strand-count arrays, in column blocks."""

    n = FW.shape[1]
    num, den, mean_R, s = (np.zeros(n) for _ in range(4))
    for a in range(0, n, block):
        f_ = FW[:, a:a+block].astype(float)
        R = f_ + REV[:, a:a+block]
        seen = R > 0
        tot = R.sum(0)
        s[a:a+block] = f_.sum(0) / np.maximum(tot, 1)
        sb = s[a:a+block]
        num[a:a+block] = np.where(seen, (f_ - sb*R)**2, 0).sum(0)
        den[a:a+block] = (R*sb*(1 - sb)).sum(0)
        with np.errstate(all='ignore'):
            mean_R[a:a+block] = tot / seen.sum(0)

    return _kappa_to_f(num, den, mean_R, s, min_reads)


def _kappa_to_f(num, den, mean_R, s, min_reads):
    """Median kappa over well-covered, strand-balanced sites, as reads per molecule."""

    ok = (mean_R >= min_reads) & (s >= 0.2) & (s <= 0.8) & (den > 0)
    if not ok.any():
        logging.warning('Too little coverage to estimate reads per molecule: assuming 1.')
        return 1.0
    kappa = float(np.median(num[ok] / den[ok]))
    f = 2 / (1 + kappa)
    if f < 1:
        logging.warning(
            f'Strand counts are over-dispersed (kappa={kappa:.2f}): PCR duplicates were '
            f'probably not removed upstream. Reads per molecule set to 1.'
        )
    return float(np.clip(f, 1.0, 2.0))


##


def _table_layout(path):
    """
    Columns of a maegatk / mgatk per-base table, from its first line:

    * 4 columns - pos, cell, forward count, reverse count. This is what mgatk writes for
      10x data unless it is run with `--emit-base-qualities`: there are no base qualities,
      so the AFM gets no `var["quality"]` and the filters that would use it are skipped.
    * 6 columns - the same, with a mean base quality per strand (maegatk; mgatk with
      `-eb`).
    * more - extra consensus/group-size columns (mito_preprocessing-style), ignored.
    """

    n = pd.read_csv(path, header=None, nrows=1).shape[1]
    if n == 4:
        return [0, 1, 2, 3], ['pos', 'cell', 'fw', 'rev']
    if n == 6:
        return [0, 1, 2, 3, 4, 5], ['pos', 'cell', 'fw', 'qfw', 'rev', 'qrev']
    if n > 6:
        return [0, 1, 2, 3, 6, 7], ['pos', 'cell', 'fw', 'qfw', 'rev', 'qrev']
    raise ValueError(f'Unexpected layout in {os.path.basename(path)}: {n} columns.')


_CHUNK = 5_000_000


def _read_base_table(path):
    """One per-base table, typed and restricted to the columns used, in chunks of rows."""

    usecols, names = _table_layout(path)
    dtype = {'pos':np.int32, 'cell':'category', 'fw':np.int32, 'rev':np.int32,
             'qfw':np.float32, 'qrev':np.float32}
    for df in pd.read_csv(path, header=None, usecols=usecols, chunksize=_CHUNK,
                          dtype={i:dtype[n] for i, n in zip(usecols, names, strict=True)}):
        df.columns = names
        yield df


def _assemble_alt_rows(alt, counts, names, n_cov_cells, COV_all, listed, ref_base, n_pos, *,
                       sample=None, path_meta=None, multi_allelic_sites=False, molecules=False,
                       f=1.0, has_quality=False, scLT_system='MAESTER', pp_method='maegatk'):
    """
    AFM from alternative basecalls and a coverage matrix, shared by `make_afm` (maegatk /
    mgatk tables) and `make_afm_from_counts`.

    alt : dict of aligned arrays - pos, cell (index into `names`), base (0-3: A C G T),
        fw, rev, qual (NaN if none); `counts` the alternative reads (or molecules) per row.
    COV_all, listed : (n_cov_cells, n_pos) coverage, and whether the table lists the pair.
    """

    BASES = np.array(['A', 'C', 'G', 'T'])
    # One alternative allele per site: the dominant one (most alternative reads)
    if not multi_allelic_sites:
        support = np.bincount(alt['pos'].astype(np.int64)*4 + alt['base'], weights=counts,
                              minlength=n_pos*4).reshape(n_pos, 4)
        dominant = support.argmax(1)                        # first maximum: A < C < G < T
        srt = np.sort(support, 1)
        n_ambiguous = int(((srt[:,-2] >= 0.5*srt[:,-1]) & (srt[:,-2] > 0)).sum())
        if n_ambiguous:
            logging.info(
                f'{n_ambiguous} site(s) where a second alternative allele carries >=50% of the '
                f'dominant one\'s reads: the dominant allele is kept for all of them.'
            )
        keep = alt['base'] == dominant[alt['pos']]
        logging.info(f'Keep the dominant alternative allele per site: {int(keep.sum())} of {keep.size} basecalls')
        alt = {k:v[keep] for k, v in alt.items()}
        counts = counts[keep]

    rule = None
    if molecules:
        rule = 'both_strands' if f - 1 >= 0.5 else 'reads_over_f'
        AD = np.minimum(alt['fw'], alt['rev']) if rule == 'both_strands' else np.rint(counts / f)
        logging.info(
            f'Alternative molecules ({rule}, {f - 1:.0%} of molecules read by both mates): '
            f'{int((AD>0).sum())} (cell, variant) pairs'
        )
    else:
        AD = counts
    nz = AD > 0
    alt = {k:v[nz] for k, v in alt.items()}
    AD = AD[nz].astype(np.int64)

    # Cells: observed with an alternative basecall and in the coverage table, as annotated
    if sample is not None:
        names = [f'{x}_{sample}' for x in names]
    observed = {names[i] for i in np.unique(alt['cell']) if i < n_cov_cells}
    cell_meta, cells = _select_cells(path_meta, observed)
    row = np.full(len(names), -1, dtype=np.int64)
    name_pos = {c:i for i, c in enumerate(names)}
    row[[name_pos[c] for c in cells]] = np.arange(len(cells))

    in_cells = row[alt['cell']] >= 0
    alt = {k:v[in_cells] for k, v in alt.items()}
    AD = AD[in_cells]
    r = row[alt['cell']]

    # Variants carried by the selected cells, ordered by position and allele
    vkey = alt['pos'].astype(np.int64)*4 + alt['base']
    uvar, col = np.unique(vkey, return_inverse=True)
    v_pos = (uvar // 4).astype(np.int64)
    v_alt = BASES[uvar % 4]
    v_ref = np.array([ref_base.get(int(p), 'N') for p in v_pos])
    var = pd.DataFrame({'pos':v_pos, 'ref':v_ref, 'alt':v_alt},
                       index=[f'{p}_{a}>{b}' for p, a, b in zip(v_pos, v_ref, v_alt, strict=True)])
    if has_quality:
        q = alt['qual']
        ok = np.isfinite(q)
        s_ = np.bincount(col[ok], weights=q[ok], minlength=uvar.size)
        n_ = np.bincount(col[ok], minlength=uvar.size)
        with np.errstate(all='ignore'):
            var['quality'] = s_ / np.where(n_>0, n_, np.nan)

    # Coverage of the selected cells, over the positions the table holds for them
    src = np.array([name_pos[c] for c in cells], dtype=np.int64)
    present = np.flatnonzero(listed[src].any(0))
    col_of = np.full(n_pos, -1, dtype=np.int64)
    col_of[present] = np.arange(present.size)
    COV = COV_all[np.ix_(src, present)]
    if molecules:
        COV = np.rint(COV / f).astype(np.int32)
    if (col_of[v_pos] < 0).any():
        raise ValueError(f'Coverage table misses {int((col_of[v_pos]<0).sum())} of the sites carrying MT-SNVs.')

    dtype = _coverage_dtype(COV)
    DP = COV[:, col_of[v_pos]]
    DP[r, col] = np.maximum(DP[r, col], AD)
    DP = DP.astype(dtype)
    AF = (AD / np.maximum(DP[r, col], 1)).astype(np.float32)
    shape = (len(cells), uvar.size)
    ad_dtype = np.int16 if AD.max(initial=0) <= np.iinfo(np.int16).max else np.int32

    afm = AnnData(
        X=csr_matrix((AF, (r, col)), shape=shape),
        obs=cell_meta,
        var=var,
        layers={'AD':csr_matrix((AD.astype(ad_dtype), (r, col)), shape=shape), 'DP':DP},
        uns={'scLT_system':scLT_system, 'pp_method':pp_method}
    )
    afm.obs['mean_site_coverage'] = COV.mean(axis=1)
    if scLT_system in TARGETED_SYSTEMS:
        target = np.asarray(mask_mt_sites(pd.Index(present)))
        afm.obs['median_target_site_coverage'] = np.median(COV[:, target], axis=1)
        afm.obs['frac_target_site_covered'] = (COV[:, target]>0).sum(1) / target.sum()
    if molecules:
        afm.uns['molecules'] = {'reads_per_molecule':f, 'frac_both_mates':f - 1, 'alt_rule':rule}

    return afm


##


def _read_allelic_tables(path_ch_matrix, path_meta=None, sample=None,
                         scLT_system='MAESTER', pp_method='maegatk', ref='rCRS',
                         multi_allelic_sites=False, molecules=None):
    """
    Read maegatk / mgatk per-base allelic tables.

    `path_ch_matrix` (or its `final/` sub-folder) must hold the per-base tables and the
    coverage table, named `A.txt[.gz]` … `coverage.txt[.gz]` with or without a sample-name
    prefix (nf-MiTo writes the former, mgatk the latter). Table layouts differ between
    tools and modes, and are detected from the column count (`_table_layout`).

    The tables are read typed and column-selective, reduced to alternative basecalls as
    they are read, and assembled straight into sparse matrices: no cell x variant table
    is ever built dense, and the tables are parsed in chunks (a 4,800-cell MAESTER sample:
    ~10 GB and 65 s before, under 2 GB and ~20 s now).

    `molecules` (default: for mtscATAC) converts read counts to molecules. mtscATAC reads
    are not UMI-collapsed: a molecule covered by both mates of a short fragment is counted
    twice, and sequencing errors are single reads. From the strand counts alone:

    1. f, the reads per molecule, and o = f - 1, the fraction of molecules read by both
       mates, are estimated (`_reads_per_molecule`; PCR duplicates must have been removed
       upstream, as mgatk does by default);
    2. coverage = reads / f;
    3. alternative molecules = min(forward, reverse) if most molecules are read by both
       mates (o >= 0.5): an alternative base seen on one read only is, most of the time,
       a sequencing error (on pL1000, 70% of the background against 11% of the real
       molecules), and a real one is kept with probability ~o. With fewer overlapping
       mates (single-end reads, long fragments) strand confirmation would discard most
       real molecules, so alternative molecules = rint(alternative reads / f).

    f, o and the rule used are stored in .uns["molecules"].
    """

    if molecules is None:
        molecules = scLT_system == 'mtscATAC'

    path_ch_matrix = _resolve_dir(path_ch_matrix)
    ref_base = _read_reference(ref, path=path_ch_matrix)
    n_pos = max(ref_base) + 2
    BASES = np.array(['A', 'C', 'G', 'T'])
    ref_code = np.full(n_pos, -1, dtype=np.int8)                     # -1: N or unknown
    for p, b in ref_base.items():
        ref_code[p] = {'A':0, 'C':1, 'G':2, 'T':3}.get(b, -1)

    paths = {}
    for base in BASES:
        paths[base] = _find(path_ch_matrix, f'{base}.txt')
        if paths[base] is None:
            raise ValueError(
                f'Missing the {base} allelic table in {path_ch_matrix}. Expected '
                f'"{base}.txt(.gz)", with or without a sample-name prefix.'
            )
    path_cov = _find(path_ch_matrix, 'coverage.txt')
    if path_cov is None:
        raise ValueError(f'Missing the coverage table in {path_ch_matrix}.')

    # Coverage first: it fixes the cell universe
    logging.info('Retrieve cell-site total coverage')
    names = []
    index = {}

    def codes(cat):
        """Cell codes of a categorical column in the shared cell index."""
        cats = list(cat.cat.categories)
        for c in cats:
            if c not in index:
                index[c] = len(names)
                names.append(c)
        lut = np.array([index[c] for c in cats], dtype=np.int64)
        return lut[cat.cat.codes.values]

    # Coverage of every cell at every position, and which positions the table lists for it
    cov_chunks = []
    for cov in pd.read_csv(path_cov, header=None, names=['pos', 'cell', 'DP'], chunksize=_CHUNK,
                           dtype={'pos':np.int32, 'cell':'category', 'DP':np.int32}):
        cov_chunks.append((codes(cov['cell']).astype(np.int32), cov['pos'].values, cov['DP'].values))
    n_cov_cells = len(names)
    COV_all = np.zeros((n_cov_cells, n_pos), dtype=np.int32)
    listed = np.zeros((n_cov_cells, n_pos), dtype=bool)
    for c, p_, d in cov_chunks:
        COV_all[c, p_] = d
        listed[c, p_] = True
    del cov_chunks

    # Per-base tables, chunk by chunk: alternative basecalls kept, forward / reverse reads
    # of every base added up per (cell, position) if read counts are to become molecules.
    # (cell, position) pairs are unique within a table, so plain fancy addition is exact.
    alt = {k:[] for k in ('pos', 'cell', 'base', 'fw', 'rev', 'qual')}
    if molecules:
        strand_fw = np.zeros((n_cov_cells, n_pos), dtype=np.int32)
        strand_rev = np.zeros((n_cov_cells, n_pos), dtype=np.int32)
    has_quality = True
    for b, base in enumerate(BASES):
        logging.info(f'Process table: {os.path.basename(paths[base])}')
        for df in _read_base_table(paths[base]):
            cell = codes(df['cell'])
            pos = df['pos'].values
            fw = df['fw'].values
            rev = df['rev'].values
            if 'qfw' in df.columns:
                q = np.stack([df['qfw'].values, df['qrev'].values], 1).astype(float)
                with warnings.catch_warnings():
                    warnings.simplefilter('ignore', RuntimeWarning)   # basecalls with no quality
                    qual = np.nanmean(np.where(q>0, q, np.nan), axis=1)
            else:
                has_quality = False
                qual = np.full(pos.size, np.nan)
            del df

            if molecules:
                ok = cell < n_cov_cells
                strand_fw[cell[ok], pos[ok]] += fw[ok]
                strand_rev[cell[ok], pos[ok]] += rev[ok]

            is_alt = ref_code[pos] != b
            for k, v in (('pos', pos), ('cell', cell.astype(np.int32)), ('fw', fw), ('rev', rev),
                         ('qual', qual.astype(np.float32))):
                alt[k].append(v[is_alt])
            alt['base'].append(np.full(int(is_alt.sum()), b, dtype=np.int8))

    if not has_quality:
        logging.info('No base qualities in the allelic tables: var["quality"] is not set.')
    alt = {k:np.concatenate(v) for k, v in alt.items()}
    counts = alt['fw'].astype(np.int64) + alt['rev']
    logging.info(f'Alternative basecalls: {counts.size}')

    f = 1.0
    if molecules:
        f = _reads_per_molecule_dense(strand_fw, strand_rev)
        logging.info(f'Reads per molecule, from the strand counts: {f:.2f}')
        del strand_fw, strand_rev

    return _assemble_alt_rows(
        alt, counts, names, n_cov_cells, COV_all, listed, ref_base, n_pos,
        sample=sample, path_meta=path_meta, multi_allelic_sites=multi_allelic_sites,
        molecules=molecules, f=f, has_quality=has_quality,
        scLT_system=scLT_system, pp_method=pp_method,
    )


##


REDEEM_THRESHOLDS = ['Total', 'VerySensitive', 'Sensitive', 'Specific']


def _read_redeem(path_ch_matrix, path_meta=None, sample=None, scLT_system='ReDeeM',
                 pp_method='redeem-v', edge_trim=9, treshold='Sensitive',
                 multi_allelic_sites=False, **kwargs):
    """
    Read RedeemV output (Weng et al., 2024).

    `path_ch_matrix` (or its `final/` sub-folder) must hold
    `RawGenotypes.<treshold>.StrandBalance` and `QualifiedTotalCts` (either gzipped). RedeemV emits four
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
    path_basecalls = _find(path_ch_matrix, f'RawGenotypes.{treshold}.StrandBalance')
    if path_basecalls is None:
        raise ValueError(f'Missing RedeemV basecalls: RawGenotypes.{treshold}.StrandBalance(.gz) in {path_ch_matrix}')
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
    path_cov = _find(path_ch_matrix, 'QualifiedTotalCts')
    if path_cov is None:
        raise ValueError(f'Missing RedeemV coverage table: QualifiedTotalCts(.gz) in {path_ch_matrix}')
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
    Cells to assemble the AFM for: those in the metadata (a .csv path or a DataFrame),
    if provided, else all observed.
    """

    if isinstance(path_meta, pd.DataFrame) or (path_meta not in [None, 'null'] and os.path.exists(path_meta)):
        logging.info('Filter for annotated cells (i.e., sample CBs in cell_meta)')
        cell_meta = path_meta.copy() if isinstance(path_meta, pd.DataFrame) else pd.read_csv(path_meta, index_col=0)
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
        maegatk / mgatk tables only: convert read counts to molecules, from the strand
        counts (reads per molecule, then alternative molecules read by both mates, or
        alternative reads / reads per molecule when few fragments have overlapping mates).
        None (default) decides from the assay: on for mtscATAC, whose reads are not
        UMI-collapsed, off for MAESTER (UMI consensus counts).
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


def make_afm_from_counts(
    counts,
    coverage,
    cell_meta: pd.DataFrame = None,
    scLT_system: str = 'MAESTER',
    pp_method: str = 'counts',
    ref: str = 'rCRS',
    multi_allelic_sites: bool = False,
    ) -> AnnData:
    """
    Assemble an Allele Frequency Matrix from long tables of counts, for data that did not
    go through maegatk / mgatk / RedeemV (another pipeline, a published matrix).

    Parameters
    ----------
    counts : pd.DataFrame or str
        Alternative allele counts, one row per cell and variant (a .csv path is read):
        columns "cell", "pos" (1-based), "alt" and "AD" (alternative molecules), and
        optionally "ref" (checked against the reference) and "quality" (mean base quality).
        Rows with AD = 0 are ignored.
    coverage : pd.DataFrame or str
        Coverage, one row per cell and site: columns "cell", "pos" and "DP". Every site
        carrying a variant must be listed; a site a cell has no row for counts as not
        covered (DP = 0) - which is how genotyping tells "not covered" from "covered, no
        alternative molecule", so include the covered sites without alternative molecules.
    cell_meta : pd.DataFrame, optional
        Cell metadata, indexed by cell name; only its cells are kept, in its order.
        Default is None (every cell with at least one alternative molecule).
    scLT_system : str, optional
        Assay, as in `make_afm`: "MAESTER", "mtscATAC" or "ReDeeM". Default is "MAESTER".
    pp_method : str, optional
        Label stored in .uns["pp_method"]. Default is "counts".
    ref : str, optional
        "rCRS", or a path to a custom MT reference (FASTA). Default is "rCRS".
    multi_allelic_sites : bool, optional
        Keep every alternative allele of a site; by default only the one with the most
        alternative molecules. Default is False.

    Returns
    -------
    AnnData
        The AFM, with the same contract as `make_afm`'s.
    """

    if scLT_system not in SCLT_SYSTEMS:
        raise ValueError(f'scLT_system must be one of {SCLT_SYSTEMS}. Provided: {scLT_system}')
    counts = pd.read_csv(counts) if isinstance(counts, str) else counts.copy()
    coverage = pd.read_csv(coverage) if isinstance(coverage, str) else coverage.copy()
    for name, df, cols in (('counts', counts, ['cell', 'pos', 'alt', 'AD']), ('coverage', coverage, ['cell', 'pos', 'DP'])):
        missing = [c for c in cols if c not in df.columns]
        if missing:
            raise ValueError(f'{name} needs the columns {cols}; missing: {missing}.')

    ref_base = _read_reference(ref)
    n_pos = max(ref_base) + 2
    counts = counts.loc[counts['AD'] > 0]
    pos = counts['pos'].to_numpy(np.int64)
    if (pos < 1).any() or (pos >= n_pos).any():
        raise ValueError(f'Positions must be 1-based MT coordinates (1-{n_pos-2}).')
    code = {'A':0, 'C':1, 'G':2, 'T':3}
    base = counts['alt'].astype(str).str.upper().map(code)
    if base.isna().any():
        raise ValueError(f'alt must be one of A, C, G, T; found {sorted(set(counts["alt"][base.isna()]))}.')
    base = base.to_numpy(np.int8)
    ref_of_pos = np.array([ref_base.get(int(p), 'N') for p in pos])
    if (np.array(['A', 'C', 'G', 'T'])[base] == ref_of_pos).any():
        raise ValueError('Some rows have alt equal to the reference base: they are not variants.')
    if 'ref' in counts.columns:
        bad = counts['ref'].astype(str).str.upper().to_numpy() != ref_of_pos
        if bad.any():
            raise ValueError(f'{int(bad.sum())} row(s) have a ref base that differs from the {ref} reference.')
    if counts.duplicated(['cell', 'pos', 'alt']).any():
        raise ValueError('counts has several rows for the same cell, pos and alt.')

    # One index of cells, coverage first (as the table readers)
    names = list(pd.unique(coverage['cell'].astype(str)))
    index = {c:i for i, c in enumerate(names)}
    n_cov_cells = len(names)
    for c in pd.unique(counts['cell'].astype(str)):
        if c not in index:
            index[c] = len(names)
            names.append(c)
    COV_all = np.zeros((n_cov_cells, n_pos), dtype=np.int32)
    listed = np.zeros((n_cov_cells, n_pos), dtype=bool)
    cr = coverage['cell'].astype(str).map(index).to_numpy(np.int64)
    cp = coverage['pos'].to_numpy(np.int64)
    COV_all[cr, cp] = coverage['DP'].to_numpy(np.int64)
    listed[cr, cp] = True

    AD = counts['AD'].to_numpy(np.int64)
    alt = {
        'pos':pos, 'cell':counts['cell'].astype(str).map(index).to_numpy(np.int64), 'base':base,
        'fw':AD, 'rev':np.zeros_like(AD),
        'qual':counts['quality'].to_numpy(float) if 'quality' in counts.columns else np.full(AD.size, np.nan),
    }

    return _assemble_alt_rows(
        alt, AD, names, n_cov_cells, COV_all, listed, ref_base, n_pos,
        path_meta=cell_meta, multi_allelic_sites=multi_allelic_sites,
        has_quality='quality' in counts.columns, scLT_system=scLT_system, pp_method=pp_method,
    )


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
