"""
Bootstrap utils.
"""

import numpy as np
from scipy.sparse import issparse, csr_matrix
from anndata import AnnData


##


def bootstrap_allele_tables(afm, layer='AD', frac_char_resampling=.8):
    """
    Bootstrap of an Allele Frequency Matrix (AFM) layer.
    Both sparse matrices and dense .layers can be passed.
    """

    # Get layer
    if layer in afm.layers:
        X = afm.layers[layer]
        X = X if not issparse(X) else X.toarray()
    else:
        raise KeyError(f'{layer} not present in afm! Check your inputs...')

    # Resample afm.var index
    n = X.shape[1]
    if frac_char_resampling == 1:
        resampled_idx = np.random.choice(np.arange(n), n, replace=True) 
    else:    
        resampled_idx = np.random.choice(np.arange(n), round(n*frac_char_resampling), replace=False)   

    return X[:,resampled_idx], resampled_idx
    


##


def jackknife_allele_tables(afm, layer='AD'):
    """
    Jackknife of an Allele Frequency Matrix (AFM) layer.
    Both sparse matrices and dense .layers can be passed.
    """

    # Get layer
    if layer in afm.layers:
        X = afm.layers[layer]
        X = X if not issparse(X) else X.toarray()
    else:
        raise KeyError(f'{layer} not present in afm! Check your inputs...')

    # Resample afm.var index
    n = X.shape[1]
    to_exclude = np.random.choice(np.arange(n), 1)[0]
    resampled_idx = [ x for x in np.arange(n) if x != to_exclude ]

    return X[:,resampled_idx], resampled_idx


##


def bootstrap_MiTo(
    afm: AnnData, 
    boot_replicate: str = 'observed', 
    boot_strategy: str ='feature_resampling', 
    frac_char_resampling: float = .8
    ) -> AnnData:
    """
    Bootstrap MAESTER/RedeeM Allele Frequency matrices.
    """

    if boot_replicate != 'observed':

        if boot_strategy == 'jacknife':
            AD, _ = jackknife_allele_tables(afm, layer='AD')
            cov, idx = jackknife_allele_tables(afm, layer='site_coverage')                                              # USE SITE, NBBB
        elif boot_strategy == 'feature_resampling':
            AD, _ = bootstrap_allele_tables(afm, layer='AD', frac_char_resampling=frac_char_resampling)
            cov, idx = bootstrap_allele_tables(afm, layer='site_coverage', frac_char_resampling=frac_char_resampling)    # USE SITE, NBBB
        elif boot_strategy == 'counts_resampling':
            raise ValueError(f'#TODO: {boot_strategy} boot_strategy. This strategy is not supported yet.')
        else:
            raise ValueError(f'{boot_strategy} boot_strategy is not supported...')
        
        AF = csr_matrix(np.divide(AD, (cov+.0000001)))
        AD = csr_matrix(AD)
        cov = csr_matrix(cov)
        afm_new = AnnData(X=AF, obs=afm.obs, var=afm.var.iloc[idx,:], uns=afm.uns, layers={'AD':AD, 'site_coverage':cov})

    else:
        afm_new = afm.copy()

    return afm_new


##


def bootstrap_bin(
    afm: AnnData, 
    boot_replicate: str = 'observed', 
    boot_strategy: str ='feature_resampling', 
    frac_char_resampling: float = .8
    ) -> AnnData:
    """
    Bootstrap scWGS/Cas9 AFMs.
    """

    if boot_replicate != 'observed':

        if boot_strategy == 'jacknife':
            X_new, idx = jackknife_allele_tables(afm, layer='bin')                                             
        elif boot_strategy == 'feature_resampling':
            X_new, idx = bootstrap_allele_tables(afm, layer='bin', frac_char_resampling=frac_char_resampling)  
        else:
            raise ValueError(f'{boot_strategy} boot_strategy is not supported...')
        
        X_new = csr_matrix(X_new)
        afm_new = AnnData(
            obs=afm.obs, 
            var=afm.var.iloc[idx,:], 
            uns=afm.uns, 
            layers={'bin':X_new}
        )

        if 'priors' in afm.varm:
            afm_new.varm['priors'] = afm.varm['priors'][idx,:]

    else:
        afm_new = afm.copy()

    return afm_new


##
