"""
Create a clean allelic frequency matrix as AnnData object.
"""

import logging
import os
import numpy as np
import pandas as pd
import warnings
import cassiopeia as cs
from scipy.io import mmread
from scipy.sparse import csr_matrix
from anndata import AnnData
from ..ut.utils import Timer, path_assets
from ..ut.positions import MAESTER_genes_positions, mask_mt_sites
warnings.filterwarnings("ignore")


##


def read_from_AD_DP(
    path_ch_matrix: str, path_meta: str = None, sample: str = None, 
    pp_method: str = None, cell_col: str ='cell', scLT_system: str = 'MAESTER'
    ) -> AnnData :
    """
    Create AFM as as AnnData object from a path_ch_matrix folder with AD, DP tables. AD and DP columns must have:
    1) <cell_col> column; 2) <char> column; 3) AD/DP columns, respectively.
    N.B. <char> columns must be formatted in "pos>_ref>alt" fashion and cell_meta index must be in {CB}_{sample} format.

    Example file input:

    ,CHROM,POS,ID,REF,ALT,AD,DP,cell
    4,chrM,1438,.,A,G,19,19,GTGCACGTCCATCTGC
    5,chrM,1719,.,G,A,17,17,GTGCACGTCCATCTGC
    6,chrM,2706,.,A,G,19,19,GTGCACGTCCATCTGC
    8,chrM,6221,.,T,C,10,10,GTGCACGTCCATCTGC
    9,chrM,6371,.,C,T,16,16,GTGCACGTCCATCTGC
    ...

    """

    table = pd.read_csv(path_ch_matrix, index_col=0)
    if path_meta not in ['null',None]:
        cell_meta = pd.read_csv(path_meta, index_col=0)
    if sample is not None:
        table['cell'] = table['cell'].map(lambda x: f'{x}_{sample}')
    table['MUT'] = table['POS'].astype(str) + '_' + table['REF'] + '>' + table['ALT']
    AD = table.pivot(index=cell_col, columns='MUT', values='AD').fillna(0)
    DP = table.pivot(index=cell_col, columns='MUT', values='DP').fillna(0)
    
    if path_meta not in ['null',None] and os.path.exists(path_meta):
        cells = list(set(cell_meta.index) & set(DP.index))
        AD = AD.loc[cells].copy()
        DP = DP.loc[cells].copy()
        cell_meta = cell_meta.loc[cells].copy()
    else:
        cell_meta = None

    assert (AD.index == DP.index).all()

    char_meta = DP.columns.to_series().to_frame('mut')
    char_meta['pos'] = char_meta['mut'].map(lambda x: int(x.split('_')[0]))
    char_meta['ref'] = char_meta['mut'].map(lambda x: x.split('_')[1].split('>')[0])
    char_meta['alt'] = char_meta['mut'].map(lambda x: x.split('_')[1].split('>')[1])
    char_meta = char_meta[['pos', 'ref', 'alt']]
    AF = csr_matrix(np.divide(AD,(DP+.00000001)).values.astype(np.float32))
    AD = csr_matrix(AD.values).astype(np.int16)
    DP = csr_matrix(DP.values).astype(np.int16)

    afm = AnnData(X=AF, obs=cell_meta, var=char_meta, layers={'AD':AD, 'DP':DP}, uns={'pp_method':pp_method,'scLT_system':scLT_system})
    sorted_vars = afm.var['pos'].sort_values().index
    afm = afm[:,sorted_vars].copy()

    return afm


##


def read_from_cellsnp(
    path_ch_matrix: str, path_meta: str = None, sample: str = None, 
    pp_method: str = None,  scLT_system: str = 'MAESTER'
    ) -> AnnData :

    """
    Create AFM as as AnnData object from cellsnp output tables. The path_ch_matrix folder must contain the four default output from cellsnp-lite:
    * 1: 'cellSNP.tag.AD.mtx.gz'
    * 2: 'cellSNP.tag.AD.mtx.gz'
    * 3. 'cellSNP.base.vcf.gz'
    * 4: 'cellSNP.samples.tsv.gz'
    N.B. cell_meta index must be in {CB}_{sample} format.
    """

    path_AD = os.path.join(path_ch_matrix, 'cellSNP.tag.AD.mtx')
    path_DP = os.path.join(path_ch_matrix, 'cellSNP.tag.DP.mtx')
    path_vcf = os.path.join(path_ch_matrix, 'cellSNP.base.vcf')
    path_cells = os.path.join(path_ch_matrix, 'cellSNP.samples.tsv')

    if sample is not None:
        cells = [ f'{x}_{sample}' for x in pd.read_csv(path_cells, header=None)[0].to_list() ]
    else:
        cells = pd.read_csv(path_cells, header=None)[0].to_list()

    vcf = pd.read_csv(path_vcf, sep='\t', skiprows=1)
    variants = vcf['POS'].astype(str) + '_' + vcf['REF'] + '>' + vcf['ALT']
    AD = pd.DataFrame(mmread(path_AD).toarray().T, index=cells, columns=variants)
    DP = pd.DataFrame(mmread(path_DP).toarray().T, index=cells, columns=variants)
    
    if path_meta is not None and os.path.exists(path_meta):
        cell_meta = pd.read_csv(path_meta, index_col=0)
        cells = list(set(cell_meta.index) & set(DP.index))
    else:
        cell_meta = None
        cells = list(set(DP.index))

    AD = AD.loc[cells].copy()
    DP = DP.loc[cells].copy()
    if cell_meta is not None:
        cell_meta = cell_meta.loc[cells].copy()

    assert (AD.index == DP.index).all()

    char_meta = DP.columns.to_series().to_frame('mut')
    char_meta['pos'] = char_meta['mut'].map(lambda x: int(x.split('_')[0]))
    char_meta['ref'] = char_meta['mut'].map(lambda x: x.split('_')[1].split('>')[0])
    char_meta['alt'] = char_meta['mut'].map(lambda x: x.split('_')[1].split('>')[1])
    char_meta = char_meta[['pos', 'ref', 'alt']]

    AF = csr_matrix(np.divide(AD,(DP+.00000001)).values.astype(np.float32))
    AD = csr_matrix(AD.values).astype(np.int16)
    DP = csr_matrix(DP.values).astype(np.int16)

    afm = AnnData(X=AF, obs=cell_meta, var=char_meta, layers={'AD':AD, 'DP':DP}, uns={'pp_method':pp_method,'scLT_system':scLT_system})
    sorted_vars = afm.var['pos'].sort_values().index
    afm = afm[:,sorted_vars].copy()

    return afm


##


def sparse_from_long(df, covariate, nrow, ncol, cell_order):
    """
    Make a long df a sparse matrix more efficiently.
    """

    df['code'] = pd.Categorical(df['cell'], categories=cell_order).codes
    sparse_matrix = csr_matrix(
        (df[covariate].values, (df['code'].values, df['pos'].values-1)), 
        shape=(nrow, ncol)
    )

    return sparse_matrix


##


def read_from_scmito(
    path_ch_matrix: str, path_meta: str = None, sample: str = None, 
    pp_method: str = None, scLT_system: str = 'MAESTER', ref='rCRS'
    ) -> AnnData :
    
    """
    Create AFM as as AnnData object from cellsnp output tables. 
    The path_ch_matrix folder must contain the default output from mito_preprocessing/maegatk:
    * 1: 'A|C|T|G.txt.gz'
    * 2: 'coverage.txt.gz'
    * 3. 'refAllele.txt'

    Additional outputs from mito_preprocessing can be used for separate analyses.
    N.B. cell_meta index must be in {CB}_{sample} format. mito_preprocessing and maegatk CBs are
    plain, not in {CB}_{sample} format.
    """

    # Metrics
    metrics = {}

    # Process each base table
    path_A = os.path.join(path_ch_matrix, 'A.txt.gz')
    path_C = os.path.join(path_ch_matrix, 'C.txt.gz')
    path_T = os.path.join(path_ch_matrix, 'T.txt.gz')
    path_G = os.path.join(path_ch_matrix, 'G.txt.gz')
    path_cov = os.path.join(path_ch_matrix, 'coverage.txt.gz')              

    # Get ref
    if ref == 'rCRS':
        chrM_path = os.path.join(path_assets, 'chrM.fa')
    elif os.path.exists(ref) and ref.endswith('.fa'):
        chrM_path = ref
    else:
        raise ValueError('Provide a path to your custom genome ref (FASTA file).')

    with open(chrM_path, 'r') as f:
        _ = f.readlines()
    seq = ''.join([ x.strip() for x in _[1:] ])
    ref = { pos+1:ref for pos,ref in enumerate(seq) }

    # Here we go
    L = []
    for base, path_base in zip(['A', 'C', 'T', 'G'], [path_A, path_C, path_T, path_G]):

        logging.info(f'Process table: {base}')
        base_df = pd.read_csv(path_base, header=None)

        if pp_method == 'maegatk':
            base_df.columns = ['pos', 'cell', 'count_fw', 'qual_fw', 'count_rev', 'qual_rev']
        elif pp_method == 'mito_preprocessing':
            base_df.columns = ['pos', 'cell', 
                               'count_fw', 'qual_fw', 'cons_fw', 'gs_fw', 
                               'count_rev', 'qual_rev', 'cons_rev', 'gs_rev']
            
        base_df['counts'] = base_df['count_fw'] + base_df['count_rev']
        qual = base_df[['qual_fw', 'qual_rev']].values
        base_df['qual'] = np.nanmean(np.where(qual>0,qual,np.nan), axis=1)
        L.append(base_df[['pos', 'cell', 'counts', 'qual']].assign(base=base))
    
    # Concat in long format
    logging.info(f'Format all basecalls in a long table')
    long = pd.concat(L)
    if sample is not None:
        long['cell'] = long['cell'].map(lambda x: f'{x}_{sample}')

    # Annotate ref and alt base calls
    long['ref'] = long['pos'].map(ref)
    # s = long.groupby(['cell', 'pos', 'base']).size()
    # assert all(s == 1)
    metrics['total_basecalls'] = long.shape[0]
    logging.info(f'Total basecalls: {long.shape[0]}')

    # Filter only variant basecalls
    logging.info(f'Filter variant allele basecalls')
    long = long.query('base!=ref').copy()
    long['nunique'] = long.groupby(['cell', 'pos'])['base'].transform('nunique')
    long = (
        long.query('nunique==1')
        .drop(columns=['nunique'])
        .rename(columns={'counts':'AD', 'base':'alt'})
        .copy()
    )

    # s = long.groupby(['cell', 'pos']).size()
    # assert all(s == 1)
    metrics['variant_basecalls'] = long.shape[0]
    logging.info(f'Unique variant basecalls: {long.shape[0]}')
 
    # Filter basecalls of annotated cells only (i.e., we have cell metadata)
    if path_meta not in [None,'null']:
        logging.info(f'Filter for annotated cells (i.e., sample CBs in cell_meta)')
        cell_meta = pd.read_csv(path_meta, index_col=0)
        cells = list(set(cell_meta.index) & set(long['cell'].unique()))
        long = long.query('cell in @cells').copy()
        metrics['variant_basecalls_for_annot_cells'] = long.shape[0]
        logging.info(f'Unique variant basecalls for annotated cells: {long.shape[0]}')
    else:
        cell_meta = None
        cells = list(long['cell'].unique())
 
    # Add site coverage
    logging.info(f'Retrieve cell-site total coverage')
    cov = pd.read_csv(path_cov, header=None)
    cov.columns = ['pos', 'cell', 'DP']
    if sample is not None:
        cov['cell'] = cov['cell'].map(lambda x: f'{x}_{sample}')
    long = long.merge(cov, on=['pos', 'cell'], how='left')
  
    # Matrices
    logging.info(f'Format AD/DP/qual matrices')
    long['mut'] = long['pos'].astype(str) + '_' + long['ref'] + '>' + long['alt']
    AD = long.pivot(index='cell', columns='mut', values='AD').fillna(0)
    DP = long.pivot(index='cell', columns='mut', values='DP').fillna(0)
    qual = long.pivot(index='cell', columns='mut', values='qual').fillna(0)
 
    assert (AD.index.value_counts()==1).all()
 
    # Ensure common cell index for each matrix
    AD = AD.loc[cells].copy()
    DP = DP.loc[cells].copy()
    qual = qual.loc[cells].copy()
    if path_meta is not None and os.path.exists(path_meta):
        cell_meta = cell_meta.loc[cells].copy()
    else:
        cell_meta = pd.DataFrame(index=cells)
 
    # At least one unique variant basecall for each cell
    assert (np.sum(DP>0, axis=1)>0).all()
    assert (np.sum(DP>0, axis=1)>0).all() 
    assert (AD.index == DP.index).all()
    assert (AD.columns == DP.columns).all()
 
    # Char and cell metadata
    char_meta = DP.columns.to_series().to_frame('mut')
    char_meta['pos'] = char_meta['mut'].map(lambda x: int(x.split('_')[0]))
    char_meta['ref'] = char_meta['mut'].map(lambda x: x.split('_')[1].split('>')[0])
    char_meta['alt'] = char_meta['mut'].map(lambda x: x.split('_')[1].split('>')[1])
    char_meta = char_meta[['pos', 'ref', 'alt']]
 
    """
    We have selected relevant info (alt and ref counts, quality and allelic frequency) 
    about all interesting variant basecalls in the data. We have just excluded:
        1- basecalls of un-annotated cells
        2- basecalls for which more than one alternative allele has been observed (same cell, same site).
    """
 
    # To sparse and AnnData
    logging.info('Build AnnData object')
    AF = csr_matrix(np.divide(AD.values,(DP.values+.00000001)).astype(np.float32))
    AD = csr_matrix(AD.values).astype(np.int16)
    DP = csr_matrix(DP.values).astype(np.int16)
    qual = csr_matrix(qual.values).astype(np.int16)
    afm = AnnData(
        X=AF, 
        obs=cell_meta, 
        var=char_meta, 
        layers={'AD':AD, 'DP':DP, 'qual':qual}, 
        uns={'pp_method':pp_method, 'scLT_system':scLT_system, 'raw_basecalls_metrics':metrics}
    )
    sorted_vars = afm.var['pos'].sort_values().index
    assert sorted_vars.size == afm.shape[1]
    afm = afm[:,sorted_vars].copy()
 
    # Add complete site coverage info
    logging.info('Add site-coverage matrix and cell-coverage metrics')
    cov = cov.pivot(index='cell', columns='pos', values='DP').fillna(0)
    cov = cov.loc[cells]
    mapping = afm.var['pos'].to_dict()
    df_ = pd.DataFrame({ mut : cov[mapping[mut]].values for mut in mapping }, index=cells)
    assert all(df_.columns == afm.var_names)
    afm.layers['site_coverage'] = csr_matrix(df_.values)
    afm.obs['mean_site_coverage'] = cov.mean(axis=1)   
    test_sites = mask_mt_sites(range(cov.shape[1]))
    afm.obs['median_target_site_coverage'] = cov.loc[:,test_sites].median(axis=1)
    afm.obs['median_untarget_site_coverage'] = cov.loc[:,~test_sites].median(axis=1)
    afm.obs['frac_target_site_covered'] = np.sum(cov.loc[:,test_sites]>0, axis=1) / test_sites.sum()

    return afm


##


def read_redeem(
    path_ch_matrix: str, 
    path_meta: str = None, 
    sample: str = None, 
    pp_method: str = None, 
    scLT_system: str = 'RedeeM',
    edge_trim: int = 4, 
    treshold: str = 'Sensitive'
    ) -> AnnData :
    """
    Utility to assemble an AFM from RedeeM (see Weng et al., 2024) MT-SNVs data.
    """

    # Intantiate metrics dictionary
    metrics = {}

    ##

    if pp_method == 'RedeemV' or pp_method is None:

        # Read basecalls
        path_basecalls = os.path.join(path_ch_matrix, f"RawGenotypes.{treshold}.StrandBalance")
        logging.info(f'Process RedeemV basecalls from: {path_basecalls}')
        cols = [
            "UMI","Cell","Pos","Variants","Call","Ref","FamSize",
            "GT_Cts","CSS","DB_Cts","SG_Cts","Plus","Minus","Depth"
        ]
        basecalls = pd.read_csv(path_basecalls, sep='\t', header=None, names=cols)

        # Fix variants names
        basecalls['Variants'] = basecalls['Pos'].astype(str) + '_' + \
                                basecalls['Ref'] + '>' + \
                                basecalls['Call']

        # AD counts, before trimming
        logging.info(f'Count AD before edge-trimming')
        long = (
            basecalls.groupby(['Cell','Variants'])
            ['UMI'].nunique().reset_index()
            .rename(columns={'UMI':'AD_raw'})
            .merge(
                basecalls[['Cell','Variants','Depth']].drop_duplicates(), 
                on=['Cell','Variants'], 
                how='left'
            )
        )

        # Trim edge basecalls
        logging.info(f'Trim basecalls at <{edge_trim}bp distance from DNA fragments start/end')
        splitted = basecalls['UMI'].str.split('_', expand=True)
        start_raw = splitted[1].astype(int)
        end_raw   = splitted[2].astype(int)
        basecalls['Start'] = np.minimum(start_raw, end_raw)
        basecalls['End']   = np.maximum(start_raw, end_raw)
        basecalls['Edge_dist'] = np.minimum(
            (basecalls['Pos'] - basecalls['Start']).abs(),
            (basecalls['End'] - basecalls['Pos']).abs()
        )
        basecalls = basecalls.loc[basecalls['Edge_dist'] >= edge_trim].copy()

        # Count AD after trimming
        logging.info(f'Count AD after edge-trimming')
        long_trim = (
            basecalls.groupby(['Cell','Variants'])
            ['UMI'].nunique().reset_index()
            .rename(columns={'UMI':'AD_trimmed'})
        )

        # Correct Depth --> DP
        logging.info(f'Correct DP for trimmed basecalls')
        long = (
            long
            .merge(long_trim, on=['Cell','Variants'], how='outer')
            .fillna({'AD_raw':0, 'AD_trimmed':0})
            .assign(
                n_trimmed=lambda x: x['AD_raw']-x['AD_trimmed'],
                DP=lambda x: x['Depth']-x['n_trimmed']
            )
            .query('AD_trimmed>0') # Remove basecalls with no trimmed AD left
        )

    elif pp_method == 'RedeemR':

        if os.path.exists(os.path.join(path_ch_matrix, 'FilteredCounts')) and edge_trim==0:
            long = pd.read_csv(os.path.join(path_ch_matrix, 'FilteredCounts'), sep='\t')
        elif os.path.exists(os.path.join(path_ch_matrix, 'FilteredCounts.trimmed')) and edge_trim>0:
            long = pd.read_csv(os.path.join(path_ch_matrix, 'FilteredCounts.trimmed'), sep='\t')
        else:
            raise ValueError(f'With pp_method={pp_method}, specified checkout your path_ch_matrix {path_ch_matrix} kwarg (edge_trim={edge_trim})')
    else:
        raise ValueError(f'pp_method should be either RedeemV, RedeemR or None. Provided: {pp_method}')

    ##

    # Filter for unique variant basecalls (i.e., no multi-allelic calls)
    long['Pos'] = long['Variants'].map(lambda x: x.split('_')[0]).astype(int)
    long['Ref'] = long['Variants'].map(lambda x: x.split('_')[1].split('>')[0])
    long['Alt'] = long['Variants'].map(lambda x: x.split('_')[1].split('>')[1])
    long['nunique'] = long.groupby(['Cell', 'Pos'])['Alt'].transform('nunique')
    long = (
        long.query('nunique==1')
        .drop(columns=['nunique', 'Pos', 'Ref', 'Alt'])
        .copy()
    )

    # Add metrics
    metrics['variant_basecalls'] = long.shape[0]
    logging.info(f'Unique variant basecalls: {long.shape[0]}')

    ##

    # Filter basecalls of annotated cells only (i.e., we have cell metadata)
    if path_meta not in [None,'null']:
        logging.info(f'Filter for annotated cells (i.e., sample CBs in cell_meta)')
        cell_meta = pd.read_csv(path_meta, index_col=0)
        cells = list(set(cell_meta.index.map(lambda x: x.split('_')[0])) & set(long['Cell'].unique()))
        long = long.query('Cell in @cells').copy()
        metrics['variant_basecalls_for_annot_cells'] = long.shape[0]
        logging.info(f'Unique variant basecalls for annotated cells: {long.shape[0]}')
    else:
        cell_meta = None
        cells = list(long['Cell'].unique())

    # Matrices
    logging.info(f'Format AD/DP matrices')
    AD = long.pivot(index='Cell', columns='Variants', values='AD_trimmed').fillna(0)
    DP = long.pivot(index='Cell', columns='Variants', values='DP').fillna(0)
    assert (AD.index.value_counts()==1).all()
    AD = AD.loc[cells].copy()
    DP = DP.loc[cells].copy()

    # Cell meta
    if path_meta is not None and os.path.exists(path_meta):
        cell_meta = cell_meta.loc[cells].copy()
    else:
        cells = cells if sample is None else [ f"{cell}_{sample}" for cell in cells ]
        cell_meta = pd.DataFrame(index=cells)

    # Ensure at least one unique variant basecall for each cell
    assert (np.sum(DP>0, axis=1)>0).all()
    assert (np.sum(DP>0, axis=1)>0).all() 
    assert (AD.index == DP.index).all()
    assert (AD.columns == DP.columns).all()

    # Char and cell metadata
    char_meta = DP.columns.to_series().to_frame('mut')
    char_meta['pos'] = char_meta['mut'].map(lambda x: int(x.split('_')[0]))
    char_meta['ref'] = char_meta['mut'].map(lambda x: x.split('_')[1].split('>')[0])
    char_meta['alt'] = char_meta['mut'].map(lambda x: x.split('_')[1].split('>')[1])
    char_meta = char_meta[['pos', 'ref', 'alt']]

    # To sparse and AnnData
    logging.info('Build AnnData object')
    AF = csr_matrix(np.divide(AD.values,(DP.values+.00000001)).astype(np.float32))
    AD = csr_matrix(AD.values).astype(np.int16)
    DP = csr_matrix(DP.values).astype(np.int16)
    afm = AnnData(
        X=AF, 
        obs=cell_meta, 
        var=char_meta, 
        layers={'AD':AD, 'DP':DP}, 
        uns={'pp_method':pp_method, 'scLT_system':scLT_system, 'raw_basecalls_metrics':metrics}
    )

    # Remove MT-SNVs with multi-allelic calls (different cells this time)
    var_sites = afm.var_names.map(lambda x: x.split('_')[0])
    test = var_sites.value_counts()[var_sites] == 1
    afm = afm[:,afm.var_names[test]].copy()

    # Sort vars
    sorted_vars = afm.var['pos'].sort_values().index
    assert sorted_vars.size == afm.shape[1]
    afm = afm[:,sorted_vars].copy()

    ##

    # Add coverage info
    if os.path.exists(os.path.join(path_ch_matrix, 'QualifiedTotalCts')):

        # Raw cell-MT_genome position consensus UMI counts
        logging.info('Add full site-coverage matrix')
        cov = pd.read_csv(os.path.join(path_ch_matrix, 'QualifiedTotalCts'), sep='\t', header=None)
        cov.columns = ['Cell', 'Pos', 'Total', 'VerySensitive', 'Sensitive', 'Stringent']
        cov = cov[['Cell', 'Pos', treshold]].rename(columns={treshold:'coverage'})   

        # Pivot and align
        cov = cov.pivot(index='Cell', columns='Pos', values='coverage').fillna(0)
        cov = cov.loc[cells]
        mapping = afm.var['pos'].to_dict()
        df_ = pd.DataFrame({ mut : cov[mapping[mut]].values for mut in mapping }, index=cells)
        assert all(df_.columns == afm.var_names)

        # Correct trimmed DPs in site coverage layer
        logging.info('Correct trimmed DPs in site_coverage layer')
        x,y = afm.layers['DP'].nonzero()
        df_.values[x,y] = afm.layers['DP'][x,y].A1
        afm.layers['site_coverage'] = csr_matrix(df_.values)

        # Add cell mean_site_coverage to cell meta
        afm.obs['mean_site_coverage'] = cov.mean(axis=1)   

    elif os.path.exists(os.path.join(path_ch_matrix, 'meanCellCov')) and \
        os.path.exists(os.path.join(path_ch_matrix, 'meanSiteCov')):

        # Read mean coverage (per cell and site)
        logging.info('Add mean site and cell coverage across MT-genome')
        cell_cov = pd.read_csv(os.path.join(path_ch_matrix, 'meanCellCov'), sep='\t', index_col=0)
        site_cov = pd.read_csv(os.path.join(path_ch_matrix, 'meanSiteCov'), sep='\t', index_col=0)

        # Add to cell and vars meta
        afm.obs['mean_site_coverage'] = cell_cov.loc[afm.obs_names, 'coverage'].values
        afm.var['mean_cov'] = site_cov.loc[afm.var['pos'].values, 'coverage'].values

    else:
        raise ValueError(f"""
            Coverage files at {path_ch_matrix}. Provide either QualifiedTotalCts or meanCellCov/meanSiteCov files.
            """
        )

    return afm


##


def _add_priors(afm, priors, key='priors'):

    n_targets = len(priors)
    max_state_encoding = 0
    for i in priors:
        new_max = max(priors[i].keys())
        if new_max>max_state_encoding:
            max_state_encoding = new_max

    W = np.zeros((n_targets, max_state_encoding+1))

    for i in range(W.shape[0]):
        W[i,:] = -1
        d = priors[i]
        for j in d:
            W[i,j] = d[j]
    
    afm.varm[key] = W


##


def read_cas9(
    path_ch_matrix: str, path_meta: str = None, sample: str = None, 
    pp_method: str = None, scLT_system: str = 'Cas9', 
    priors_groupby: str = 'MetFamily',
    sample_col: str = 'Tumor'
    ) -> AnnData :
    """
    Utility to assemble an AFM from Cas9, output from Cassiopeia.
    e.g. KP tracer mice data from Yang et al., 2022.
    https://www.sc-best-practices.org/trajectories/lineage_tracing.html#
    """

    # Read allele table from Cassiopeia
    allele_table = pd.read_csv(path_ch_matrix, sep='\t', index_col=0).dropna()
    # Compute priors
    indel_priors = cs.pp.compute_empirical_indel_priors(
        allele_table, grouping_variables=["intBC", priors_groupby]
    )
    allele_table = allele_table[allele_table[sample_col] == sample]
    # Conver to character matrix
    (
        char_matrix,
        priors,
        _,
    ) = cs.pp.convert_alleletable_to_character_matrix(
        allele_table, allele_rep_thresh=0.9, mutation_priors=indel_priors
    )

    # Handle cell meta, if present
    if path_meta is not None and os.path.exists(path_meta):
      cell_meta = pd.read_csv(path_meta, index_col=0)
      cell_meta = cell_meta.query('sample==@sample').copy()
      cells = list( set(cell_meta.index) & set(char_matrix.index) )
      char_matrix = char_matrix.loc[cells,:].copy()
      cell_meta = cell_meta.loc[cells,:].copy()
    else:
        logging.info('No cell-metadata present...')
        cell_meta = pd.DataFrame(index=char_matrix.index)

    # Build AFM
    afm = AnnData(
        X=csr_matrix(char_matrix.values), 
        obs=cell_meta, 
        var=pd.DataFrame(index=char_matrix.columns),
        uns={
           'pp_method':pp_method, 
           'scLT_system':scLT_system, 
        }
    )
    _add_priors(afm, priors)
    afm.layers['bin'] = afm.X.copy()

    return afm


##


def read_scwgs(
    path_ch_matrix: str, path_meta: str = None, sample: str = None, 
    pp_method: str = None, scLT_system: str = 'scWGS'
    ) -> AnnData :
    """
    Utility to assemble an AFM from scWGS data (Fabre et al., 2022) single-colony WGS data.
    """

    # Read ch matrix
    char_matrix = pd.read_csv(path_ch_matrix, index_col=0)

    # Handle cell meta, if present
    if path_meta is not None and os.path.exists(path_meta):
      cell_meta = pd.read_csv(path_meta, index_col=0)
      cell_meta = cell_meta.query('sample==@sample').copy()
      cells = list( set(cell_meta.index) & set(char_matrix.index) )
      char_matrix = char_matrix.loc[cells,:].copy()
      cell_meta = cell_meta.loc[cells,:].copy()
    else:
        logging.info('No cell (i.e., single-cell colony) metadata present...')
        cell_meta = pd.DataFrame(index=char_matrix.index)

    afm = AnnData(
        X=csr_matrix(char_matrix.values), 
        obs=cell_meta, 
        var=pd.DataFrame(index=char_matrix.columns),
        uns={'pp_method':pp_method, 'scLT_system':scLT_system}
    )
    afm.uns['genotyping'] = {'layer':'bin', 'bin_method':None, 'binarization_kwargs':{}}
    afm.layers['bin'] = afm.X.copy()

    return afm


##


def read_epiclone(
    path_ch_matrix: str, path_meta: str = None, sample: str = None, 
    pp_method: str = None, scLT_system: str = 'EPI-clone'
    ) -> AnnData :
    """
    Utility to assemble an AFM from EPI-clone data (Scherer et al., 2025) DNA-methylation data.
    """

    # Check path_ch_matrix folder contents
    required_files = ['DNAm_binary.csv', 'panel_info_dropout_pwm.tsv', 'cpg_selection.csv']
    for f in required_files:
        if not os.path.exists(os.path.join(path_ch_matrix, f)):
            raise FileNotFoundError(f'{f} not found in {path_ch_matrix}. Check your inputs...')
        
    # Read data
    char_matrix = pd.read_csv(os.path.join(path_ch_matrix, 'DNAm_binary.csv'), index_col=0)
    vars_meta = pd.read_csv(os.path.join(path_ch_matrix, 'panel_info_dropout_pwm.tsv'), sep='\t', index_col=0)
    variant_selection = pd.read_csv(os.path.join(path_ch_matrix, 'cpg_selection.csv'), index_col=0)

    # Handle cell meta, if present
    if path_meta is not None and os.path.exists(path_meta):
      cell_meta = pd.read_csv(path_meta, index_col=0)
      cell_meta = cell_meta.query('ProcessingBatch==@sample').copy()
      cells = list( set(cell_meta.index) & set(char_matrix.index) )
      char_matrix = char_matrix.loc[cells,:].copy()
      cell_meta = cell_meta.loc[cells,:].copy()
    else:
        logging.info('No cell cell metadata present...')
        cell_meta = pd.DataFrame(index=char_matrix.index)
    
    # Select DNAm features
    VOIs = variant_selection.query('Type=="Static"').index
    char_matrix = char_matrix[VOIs].copy()
    char_matrix = csr_matrix(char_matrix.values).astype(np.int16)
    vars_meta = vars_meta.loc[VOIs].copy()

    # Assemble AFM
    afm = AnnData(X=char_matrix, 
                  layers={'bin':char_matrix},
                  uns={'pp_method':pp_method,'scLT_system':scLT_system})

    # Check cells and vars
    cells = afm.obs_names[(afm.layers['bin']==1).sum(axis=1).A1>0]
    variants = afm.var_names[(afm.layers['bin']==1).sum(axis=0).A1>0]
    afm = afm[cells,variants].copy()

    return afm


##


def make_afm(
    path_ch_matrix: str, 
    path_meta: str = None, 
    sample: str = None, 
    pp_method: str = 'maegatk', 
    scLT_system: str = 'MAESTER', 
    ref: str ='rCRS',
    kwargs: dict = {}
    ) -> AnnData :
    """
    Creates an annotated Allele Frequency Matrix from different 
    scLT_system and pre-processing pipelines outputs.

    Parameters
    ----------
    path_ch_matrix : str
        Path to folder with necessary data for provided scLT_system.
    path_meta : str, optional
        Path to .csv file with cell meta-data. Default is None.
    sample : str, optional
        Sample name to append at preprocessed CBs. Default is None.
    pp_method : str, optional
        Preprocessing method (MAESTER data only). Available options:
        mito_preprocessing, maegatk, cellsnp-lite, freebayes, samtools.
        Default is 'maegatk'.
    scLT_system : str, optional
        scLT system (i.e., marker) used for tracing. Available options:
        MAESTER, RedeeM, Cas9, scWGS.
        Default is 'MAESTER'.
    ref : str, optional
        Path to MT-reference genome. THe user can provide a custom FASTA file.
        Default is 'rCRS'.
    kwargs : dict, optional
        Optional arguments for specific scLT_system readers.
        Default is {}.

    Returns
    -------
    afm : AnnData
        The assembled Allele Frequency Matrix (AFM).
    """

    if os.path.exists(path_ch_matrix):

        logging.info(f'Allele Frequency Matrix generation: {scLT_system} system')

        T = Timer()
        T.start()

        if scLT_system == 'MAESTER':

            logging.info(f'Pre-processing pipeline used: {pp_method}')

            if pp_method in ['samtools', 'freebayes']:
                afm = read_from_AD_DP(
                    path_ch_matrix=path_ch_matrix, 
                    path_meta=path_meta, 
                    sample=sample, 
                    pp_method=pp_method, 
                    cell_col='cell', 
                    scLT_system=scLT_system
                )
            elif pp_method == 'cellsnp-lite':
                afm = read_from_cellsnp(path_ch_matrix, path_meta, sample, pp_method, scLT_system)
            elif pp_method in ['mito_preprocessing', 'maegatk']:
                afm = read_from_scmito(path_ch_matrix, path_meta, sample, pp_method, scLT_system, ref)
        
        else:
            
            logging.info('Public dataset. Character matrix already pre-processed. Just assembling the AFM...')
            logging.info('#TODO: include preprocessing entry-points for other scLT methods in nf-MiTo pipeline.')

            if scLT_system == 'RedeeM':
                afm = read_redeem(path_ch_matrix, path_meta, sample, pp_method, scLT_system, **kwargs)
            elif scLT_system == 'Cas9':
                afm = read_cas9(path_ch_matrix, path_meta, sample, 'Cassiopeia', scLT_system, **kwargs)
            elif scLT_system == 'scWGS':
                afm = read_scwgs(path_ch_matrix, path_meta, sample, 'Sequoia', scLT_system)
            elif scLT_system == 'EPI-clone':
                afm = read_epiclone(path_ch_matrix, path_meta, sample, 'EPI-clone', scLT_system)
            else:
                raise ValueError(f'Unknown {scLT_system}. Check your inputs...')
        
        logging.info(f'Allele Frequency Matrix: cell x char {afm.shape}. {T.stop()}')

        return afm

    else:
        raise ValueError('Specify a valid path_ch_matrix!')


##


def filter_multiallelic_sites(afm: AnnData, verbose: bool = True) -> AnnData:
    """
    Remove variants associated to sites (.var['pos']) that have multiple variant alleles.
    This ensures each genomic position has only one variant type across all cells.
    
    Parameters
    ----------
    afm : AnnData
        The Allele Frequency Matrix.
    verbose : bool, optional
        Whether to print filtering statistics. Default is True.
        
    Returns
    -------
    afm : AnnData
        Filtered AFM with only single-allelic sites.
    """
    
    if 'pos' not in afm.var.columns:
        raise ValueError("AFM must have 'pos' column in .var to identify genomic positions")
    
    # Count number of variants per site
    site_variant_counts = afm.var['pos'].value_counts()
    multiallelic_sites = site_variant_counts[site_variant_counts > 1].index
    
    if verbose:
        logging.info(f'Found {len(multiallelic_sites)} sites with multiple variant alleles')
        logging.info(f'Total variants before filtering: {afm.shape[1]}')
    
    # Create test for sites with only one variant allele
    test = ~afm.var['pos'].isin(multiallelic_sites)
    
    # Filter AFM
    afm_filtered = afm[:, test].copy()
    
    if verbose:
        n_removed = afm.shape[1] - afm_filtered.shape[1]
        logging.info(f'Removed {n_removed} variants from {len(multiallelic_sites)} multiallelic sites')
        logging.info(f'Total variants after filtering: {afm_filtered.shape[1]}')
        
        if len(multiallelic_sites) > 0:
            logging.info(f'Example multiallelic sites: {list(multiallelic_sites[:5])}')
    
    return afm_filtered


##


def read_coverage(afm_raw: AnnData, path_coverage: str, sample: str) -> pd.DataFrame:
    """
    Read coverage table from mito_preprocessing/maegatk output.
    """
    cov = pd.read_csv(path_coverage, header=None)
    cov.columns = ['pos', 'cell', 'n'] 
    cov['cell'] = cov['cell'].map(lambda x: f'{x}_{sample}')
    cov = cov.query('cell in @afm_raw.obs_names')
    cov['cell'] = pd.Categorical(cov['cell'], categories=afm_raw.obs_names)
    cov['pos'] = pd.Categorical(cov['pos'], categories=range(1,16569+1))
    cov = cov.pivot_table(index='cell', columns='pos', values='n', fill_value=0)

    return cov


##