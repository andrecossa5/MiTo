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
    path_ch_matrix: str, path_meta: str = None, sample: str = None, 
    pp_method: str = None, scLT_system: str = 'RedeeM'
    ) -> AnnData :
    """
    Utility to assemble an AFM from RedeeM (Weng et al., 2024) MT-SNVs data.
    """

    # Check presence of all data tables in path_ch_matrix
    files = ['QualifiedTotalCts.gz', 'filtered_basecalls.csv']
    if any([ f not in os.listdir(path_ch_matrix) for f in files ]):
        raise ValueError(f'Missing files! Check {path_ch_matrix} structure...')

    # Filtered basecalls (redeemR v2: trimming, V4 filter and binomial modeling, cells with mean MT-genome coverage>=10)
    basecalls = pd.read_csv(os.path.join(path_ch_matrix, 'filtered_basecalls.csv'), index_col=0)
    basecalls.columns = ['genotype', 'cell', 'mut', 'AD', 'DP', 'type', 'context', 'AD_before_trim', 'AF']
    basecalls['pos'] = basecalls['mut'].map(lambda x: x.split('_')[0])
    basecalls['type'] = basecalls['type'].map(lambda x: x.replace('_', '>'))
    basecalls = basecalls.drop(columns=['genotype']).assign(mut=lambda x: x['pos']+'_'+x['type'])
    basecalls = basecalls[['cell', 'mut', 'AD', 'DP', 'AF']]

    ## Raw cell-MT_genome position consensus UMI counts
    cov = pd.read_csv(os.path.join(path_ch_matrix, 'QualifiedTotalCts.gz'), sep='\t', header=None)
    cov.columns = ['cell', 'pos', 'total', 'less stringent', 'stringent', 'very stringent']
    cov = (
        ## Retain total coverage consensus UMI counts for DP values,
        ## as in https://github.com/chenweng1991/redeemR/blob/master/R/VariantSummary.R
        cov[['cell', 'pos', 'total']].rename(columns={'total':'DP'})        
    )

    # Handle cell meta, if present
    if path_meta is not None and os.path.exists(path_meta):
        cell_meta = pd.read_csv(path_meta, index_col=0)
        cell_meta = cell_meta.query('sample==@sample').copy()
        cells = list(set(cell_meta.index))
        # Filter only good quality cells (Mean MT-total coverage >=10) as in redeemR (i.e., cells in filtered basecalls)
        filtered_cells = list( set(basecalls['cell'].unique()) & set(cells) ) 
    else:
        cell_meta = pd.DataFrame({'mean_cov' : cov.groupby('cell')['DP'].mean()})
        filtered_cells = list(set(basecalls['cell'].unique())) 

    # Pivot filtered basecalls, and filter cell_meta and cov for cells
    AD = basecalls.pivot(index='cell', columns='mut', values='AD').fillna(0).loc[filtered_cells].copy()
    DP = basecalls.pivot(index='cell', columns='mut', values='DP').fillna(0).loc[filtered_cells].copy()
    cell_meta = cell_meta.loc[filtered_cells].copy()

    # Checks
    assert (AD.index.value_counts()==1).all()
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

    # Create sparse matrices, and store into the AnnData object
    logging.info('Build AnnData object...')
    AF = csr_matrix(np.divide(AD.values,(DP.values+.00000001)).astype(np.float32))
    AD = csr_matrix(AD.values).astype(np.int16)
    DP = csr_matrix(DP.values).astype(np.int16)
    afm = AnnData(
        X=AF, 
        obs=cell_meta, 
        var=char_meta, 
        layers={'AD':AD, 'DP':DP}, 
        uns={'pp_method':pp_method, 'scLT_system':scLT_system, 'raw_basecalls_metrics':None}
    )
    sorted_vars = afm.var['pos'].sort_values().index
    assert sorted_vars.size == afm.shape[1]
    afm = afm[:,sorted_vars].copy()

    # Add complete site coverage info
    logging.info('Add site-coverage matrix and cell-coverage metrics')
    cov = cov.query('cell in @filtered_cells').pivot(index='cell', columns='pos', values='DP').fillna(0)
    mapping = afm.var['pos'].to_dict()
    df_ = pd.DataFrame({ mut : cov[mapping[mut]].values for mut in mapping }, index=filtered_cells)
    assert all(df_.columns == afm.var_names)
    afm.layers['site_coverage'] = csr_matrix(df_.values)
    afm.obs['mean_site_coverage'] = cov.mean(axis=1)   

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
    pp_method: str = None, scLT_system: str = 'Cas9'
    ) -> AnnData :
    """
    Utility to assemble an AFM from Cas9 (e.g. KP tracer mice data from Yang et al., 2022) data.
    https://www.sc-best-practices.org/trajectories/lineage_tracing.html#
    """

    # Read KP-tracer allele table
    allele_table = pd.read_csv(path_ch_matrix, sep='\t', index_col=0)
    # Compute priors
    indel_priors = cs.pp.compute_empirical_indel_priors(
        allele_table, grouping_variables=["intBC", "MetFamily"]
    )
    tumor_allele_table = allele_table[allele_table["Tumor"] == sample]
    # Conver to character matrix
    (
        char_matrix,
        priors,
        _,
    ) = cs.pp.convert_alleletable_to_character_matrix(
        tumor_allele_table, allele_rep_thresh=0.9, mutation_priors=indel_priors
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
    Utility to assemble an AFM from scWGS data (Weng et al., 2024) MT-SNVs data.
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


def make_afm(
    path_ch_matrix: str, path_meta: str = None, sample: str = None, 
    pp_method: str = 'maegatk', scLT_system: str = 'MAESTER', ref: str ='rCRS'
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
            logging.info('TODO: include (mito_preprocessing Nextflow pipeline) preprocessing entry-points for other scLT methods...')

            if scLT_system == 'RedeeM':
                afm = read_redeem(path_ch_matrix, path_meta, sample, pp_method, scLT_system)
            elif scLT_system == 'Cas9':
                afm = read_cas9(path_ch_matrix, path_meta, sample, pp_method, scLT_system)
            elif scLT_system == 'scWGS':
                afm = read_scwgs(path_ch_matrix, path_meta, sample, pp_method, scLT_system)
            else:
                raise ValueError(f'Unknown {scLT_system}. Check your inputs...')
        
        logging.info(f'Allele Frequency Matrix: cell x char {afm.shape}. {T.stop()}')

        return afm

    else:
        raise ValueError('Specify a valid path_ch_matrix!')


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