"""
Custom distance function among cell AF profiles.
"""

import logging
import numpy as np
import pandas as pd
import sklearn.preprocessing as pp
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import (
    pairwise_distances, 
    PAIRWISE_BOOLEAN_FUNCTIONS, PAIRWISE_DISTANCE_FUNCTIONS
)
from anndata import AnnData
from bbmix.models import MixtureBinomial
from cassiopeia.solver.solver_utilities import transform_priors
from .kNN import kNN_graph
from ..ut.utils import Timer
from ..ut.stats_utils import genotype_mix, get_posteriors


##


discrete_metrics = PAIRWISE_BOOLEAN_FUNCTIONS + ['weighted_jaccard', 'weighted_hamming']
continuous_metrics = list(PAIRWISE_DISTANCE_FUNCTIONS.keys()) + ['correlation', 'sqeuclidean']


##


def genotype_mixtures(AD, DP, t_prob=.75, t_vanilla=.001, min_AD=2, debug=False):
    """
    Single-cell MT-SNVs genotyping with binomial mixtures posterior 
    probabilities thresholding (Kwock et al., 2022).
    """

    X = np.zeros(AD.shape)
    for idx in range(AD.shape[1]):
        X[:,idx] = genotype_mix(
            AD[:,idx], 
            DP[:,idx], 
            t_prob=t_prob, 
            t_vanilla=t_vanilla, 
            min_AD=min_AD, 
            debug=debug
        )

    return X


##


def genotype_MiTo(AD, DP, t_prob=.7, t_vanilla=0, min_AD=1, min_cell_prevalence=.1, debug=False):
    """
    Hybrid genotype calling strategy: if a mutation has prevalence (AD>=min_AD and AF>=t_vanilla) 
    >= min_cell_prevalence, use probabilistic modeling as in 'bin_mixtures'. Else, use simple 
    tresholding as in 'vanilla' method.
    """
    X = np.zeros(AD.shape)
    n_binom = 0
    
    for idx in range(AD.shape[1]):
        test = (AD[:,idx]/(DP[:,idx]+.0000001)>t_vanilla)
        prevalence = test.sum() / test.size
        if prevalence >= min_cell_prevalence:
            X[:,idx] = genotype_mix(AD[:,idx], DP[:,idx], t_prob=t_prob, t_vanilla=t_vanilla, min_AD=min_AD, debug=debug)
            n_binom += 1
        else:
            X[:,idx] = np.where(test & (AD[:,idx]>=min_AD), 1, 0)

    logging.info(f'n MT-SNVs genotyped with binomial mixtures: {n_binom}')

    return X


##


def genotype_MiTo_smooth(
    AD, DP, t_prob=.7, t_vanilla=0, min_AD=2, min_cell_prevalence=.05, k=5, 
    gamma=.25, n_samples=100, resample=False
    ):
    """
    Single-cell MT-SNVs genotyping with binomial mixtures posterior probabilities thresholding
    (readapted from  MQuad, Kwock et al., 2022) and kNN smoothing (readapted from Phylinsic, 
    Liu et al., 2022).
    """

    # kNN 
    t = Timer()
    t.start()

    # 1. Resampling strategy, as in Phylinsic
    if resample:

        logging.info(f'Phylinsic-like procedure to obtain the cell kNN graph')

        L = []
        for _ in range(1,n_samples+1):

            logging.info(f'Resampling AD counts for kNN: sample {_}/{n_samples}')
            AD_sample = np.zeros(AD.shape)
            for idx in range(AD.shape[1]):
                model = MixtureBinomial(n_components=2, tor=1e-20)
                _ = model.fit((AD[:,idx], DP[:,idx]), max_iters=500, early_stop=True)
                AD_sample[:,idx] = model.sample(DP[:,idx])
            afm_ = AnnData(
                X=csr_matrix(np.divide(AD_sample, (DP+0.0000001))), 
                layers={'AD':csr_matrix(AD_sample), 'site_coverage':csr_matrix(DP)},
                uns={'scLT_system':'MAESTER'}
            )
            compute_distances(
                afm_, 
                bin_method='vanilla', 
                binarization_kwargs={'min_AD':1, 't_vanilla':0}, # Loose genotyping.
                verbose=False
            )
            L.append(afm_.obsp['distances'].A)

        D = np.mean(np.stack(L, axis=0), axis=0)
        logging.info(f'Compute kNN graph for smoothing')
        index, _, _ = kNN_graph(D=D, k=k, from_distances=True)
    
    # 2. Direct kNN computation (weighted jaccard on MiTo binary genotypes), no resampling
    else:

        logging.info(f'Direct kNN graph calculation (weighted jaccard on MiTo binary genotypes)')

        afm_ = AnnData(
            X=csr_matrix(np.divide(AD, (DP+0.0000001))), 
            layers={'AD':csr_matrix(AD), 'site_coverage':csr_matrix(DP)},
            uns={'scLT_system':'MAESTER'}
        )
        compute_distances(
            afm_, 
            bin_method='MiTo', 
            binarization_kwargs={'min_AD':min_AD, 't_vanilla':t_vanilla, 't_prob':t_prob, 'min_cell_prevalence':min_cell_prevalence},
            verbose=True,
        )
        logging.info(f'Compute kNN graph for smoothing')
        index, _, _ = kNN_graph(D=afm_.obsp['distances'].A, k=k, from_distances=True)

    ##

    # Compute posteriors
    logging.info(f'Compute posteriors...')
    P0 = np.zeros(AD.shape)
    P1 = np.zeros(AD.shape)
    for idx in range(P0.shape[1]):
        ad = AD[:,idx]
        dp = DP[:,idx]
        positive_idx = np.where(dp>0)[0]
        p = get_posteriors(ad[positive_idx], dp[positive_idx])
        P0[positive_idx,idx] = p[:,0]
        P1[positive_idx,idx] = p[:,1]

    # Smooth posteriors
    logging.info(f'Smooth each cell posteriors using neighbors values')
    P0_smooth = np.zeros(P0.shape)
    P1_smooth = np.zeros(P1.shape)
    for i in range(index.shape[0]):
        neighbors = index[i,1:]
        P0_smooth[i,:] = (1-gamma) * P0[i,:] + gamma * (P0[neighbors,:].mean(axis=0))
        P1_smooth[i,:] = (1-gamma) * P1[i,:] + gamma * (P1[neighbors,:].mean(axis=0))

    # Assign final genotypes
    logging.info(f'Final genotyping: {t.stop()}')
    tests = [ 
        (P1_smooth>t_prob) & (P0_smooth<(1-t_prob)), 
        (P1_smooth<(1-t_prob)) & (P0_smooth>t_prob) 
    ]
    X = np.select(tests, [1,0], default=0)

    return X



##


def call_genotypes(afm, bin_method='MiTo', t_vanilla=.0, min_AD=2, 
                   t_prob=.75, min_cell_prevalence=.1, k=5, gamma=.25, n_samples=100, resample=False):
    """
    Call genotypes using simple thresholding or th MiTo binomial mixtures approachm (w/i or w/o kNN smoothing).
    """

    assert 'AD' in afm.layers 
    assert 'site_coverage' in afm.layers or 'DP' in afm.layers
    cov_layer = 'site_coverage' if 'site_coverage' in afm.layers else 'DP' 

    X = afm.X.A.copy()
    AD = afm.layers['AD'].A.copy()
    DP = afm.layers[cov_layer].A.copy()
    
    if bin_method == 'vanilla':
        X = np.where((X>=t_vanilla) & (AD>=min_AD), 1, 0)
    elif bin_method == 'MiTo':
        X = genotype_MiTo(AD, DP, t_prob=t_prob, t_vanilla=t_vanilla, min_AD=min_AD, min_cell_prevalence=min_cell_prevalence)
    elif bin_method == 'MiTo_smooth':
        X = genotype_MiTo_smooth(AD, DP, t_prob=t_prob, t_vanilla=t_vanilla, min_AD=min_AD, min_cell_prevalence=min_cell_prevalence, 
                                 k=k, gamma=gamma, n_samples=n_samples, resample=resample)
    else:
        raise ValueError("""
                Provide one of the following genotype calling methods: 
                vanilla, MiTo, MiTo_smooth
                """
            )

    afm.layers['bin'] = csr_matrix(X)
    afm.uns['genotyping'] = {
        'bin_method':bin_method, 
        'binarization_kwargs': {
            't_prob':t_prob, 't_vanilla':t_vanilla, 
            'min_AD':min_AD, 'min_cell_prevalence':min_cell_prevalence
        }
    }


##


def weighted_jaccard(M, w):
    """
    Vectorized weighted jaccard index from Weng et al., 2024.
    """

    total = M @ w 
    M_weighted = M * w 
    a = M_weighted @ M.T 
    b = np.expand_dims(total, axis=1) - a  
    c = np.expand_dims(total, axis=0) - a 
    denom = a + b + c
    S = np.where(denom != 0, a / denom, 0.0)
    D = 1.0 - S

    return D


##


def weighted_hamming(X, weights, missing_state_indicator=-1):
    """
    Cassiopeia-like (but vectorized and faster) weighted hamming distance.
    """

    n, m = X.shape
    valid = (X != missing_state_indicator)
    
    # Pairwise comparisons via broadcasting.
    X1 = X[:, None, :]  # shape: (n, 1, m)
    X2 = X[None, :, :]  # shape: (1, n, m)
    valid_pair = valid[:, None, :] & valid[None, :, :]
    count = valid_pair.sum(axis=2)
    same = (X1 == X2)
    diff_mask = valid_pair & (~same)
    
    # Lookups
    lookup = []
    for i in range(m):
        col_weights = weights[i]
        max_state = max(col_weights.keys())
        table = np.zeros(max_state + 1, dtype=float)
        for state, w in col_weights.items():
            table[state] = w
        lookup.append(table)
        
    # Build a weight matrix W of shape (n, m) using vectorized lookup.
    W = np.empty((n, m), dtype=float)
    for i in range(m):
        col = X[:, i].astype(int)
        # For missing or 0, assign 0; otherwise, look up the weight.
        W[:, i] = np.where(
            (col == missing_state_indicator) | (col == 0),
            0,
            np.take(lookup[i], col)
        )
        # Expand weights to pairwise matrices.
        W1 = W[:, None, :]   # shape: (n, 1, m)
        W2 = W[None, :, :]   # shape: (1, n, m)
        
        # Explicitly broadcast these arrays to full shape (n, n, m)
        X1_full = np.broadcast_to(X1, (n, n, m))
        W1_full = np.broadcast_to(W1, (n, n, m))
        W2_full = np.broadcast_to(W2, (n, n, m))
        zero_mask1 = np.broadcast_to((X1 == 0), (n, n, m))
        zero_mask2 = np.broadcast_to((X2 == 0), (n, n, m))
        
        # Create masks for pairs where one sample is 0.
        mask_one_zero = diff_mask & (zero_mask1 | zero_mask2)
        mask_nonzero = diff_mask & ~(zero_mask1 | zero_mask2)
        contrib = np.zeros((n, n, m), dtype=float)
        # For pairs with one zero, use the weight from the nonzero sample.
        contrib[mask_one_zero] = np.where(
            X1_full[mask_one_zero] == 0,
            W2_full[mask_one_zero],
            W1_full[mask_one_zero]
        )
        # For pairs where both are nonzero, sum both weights.
        contrib[mask_nonzero] = W1_full[mask_nonzero] + W2_full[mask_nonzero]
    
    # Sum contributions over features.
    D_total = contrib.sum(axis=2)
    
    # Normalize by the number of valid comparisons.
    with np.errstate(divide='ignore', invalid='ignore'):
        D = np.where(count != 0, D_total / count, 0)
    
    return D


##


def preprocess_feature_matrix(
    afm, distance_key='distances', precomputed=False, metric='jaccard', bin_method='MiTo', binarization_kwargs={}, verbose=True
    ):
    """
    Preprocess a feature matrix for cell-cell distance computations.
    """

    layer = None
    scLT_system = afm.uns['scLT_system'] 
    afm.uns['distance_calculations'] = {}

    if scLT_system in ['RedeeM', 'scWGS', 'MAESTER']:

        if metric in continuous_metrics:
            layer = 'scaled'
            if layer in afm.layers and precomputed:
                if verbose:
                    logging.info('Use precomputed scaled layer...')
            else:
                logging.info('Scale raw AFs in afm.X')
                afm.layers['scaled'] = csr_matrix(pp.scale(afm.X.A))

        elif metric in discrete_metrics:
            layer = 'bin'
            if layer in afm.layers and precomputed:
                bin_method = afm.uns['genotyping']['bin_method']
                binarization_kwargs = afm.uns['genotyping']['binarization_kwargs']
                if verbose: 
                    logging.info(f'Use precomputed bin layer: bin_method={bin_method}, binarization_kwargs={binarization_kwargs}')
            else:
                if verbose:
                    logging.info(f'Call genotypes with bin_method={bin_method}, binarization_kwargs={binarization_kwargs}: update afm.uns.genotyping')
                call_genotypes(afm, bin_method=bin_method, **binarization_kwargs)

        else:
            raise ValueError(f'{metric} is not a valid metric! Specify for a valid metric in {continuous_metrics} or {discrete_metrics}')

    elif scLT_system == 'Cas9':

        if metric in continuous_metrics:
            raise ValueError(f'For {scLT_system} only discrete metrics are available!')
        elif metric in discrete_metrics:     
            if 'bin' in afm.layers:
                layer = 'bin'
                if verbose:
                    logging.info(f'Use precomputed bin layer.')
            else:
                raise ValueError(f'With the {scLT_system} system, provide an AFM with Cas9 INDELS character matrix in afm.layers, under the "bin" key!')
        else:
            raise ValueError(f'{metric} is not a valid metric! Specify for a valid metric in {discrete_metrics}')
    else:
        raise ValueError(f'{scLT_system} is not a valid scLT system. Choose one between MAESTER, scWGS, RedeeM, and Cas9.')

    afm.uns['distance_calculations'][distance_key] = {'metric':metric}
    afm.uns['distance_calculations'][distance_key]['layer'] = layer


##


# TO FIX!!!
def compute_distances(
    afm, distance_key='distances', metric='weighted_jaccard', precomputed=False,
    bin_method='MiTo', binarization_kwargs={}, ncores=1, rescale=True, verbose=True
    ):
    """
    Calculates pairwise cell-cell (or sample-) distances in some character space (e.g., MT-SNVs mutation space).

    Args:
        afm (AnnData): An annotated cell x character matrix with .X slot and bin or scaled layers.
        distance_key (str, optional): Key in .obsp at which the new distances will be stored. Default: distances.
        metric ((str, callable), optional): distance metric. Default: 'jaccard'.
        bin_method (str, optional): method to binarize the provided character matrix, if the chosen metric 
            involves comparison of discrete character vectors. Default: MiTo.
        ncores (int, optional): n processors for parallel computation. Default: 8.
        weights (np.array, optional): numerical weights to use for each MT-SNVs. Default None.
        binarization_kwargs (dict, optional): **kwargs of the discretization function. Default: {}.
        verbose (bool, optional): Level of verbosity. Default: True

    Returns:
        Updates inplace .obsp slot in afm AnnData with key 'distances'.
    """
    
    # Preprocess afm
    preprocess_feature_matrix(
        afm, distance_key=distance_key, metric=metric, precomputed=precomputed,
        bin_method=bin_method, binarization_kwargs=binarization_kwargs, verbose=verbose
    )
    layer = afm.uns['distance_calculations'][distance_key]['layer']
    metric = afm.uns['distance_calculations'][distance_key]['metric']
    X = afm.layers[layer].A.copy()

    if verbose:
        logging.info(f'Compute distances: ncores={ncores}, metric={metric}.')

    # Calculate distances (handle weights, if necessary)
    if metric=='weighted_jaccard':
        w = np.nanmedian(np.where(afm.X.A>0, afm.X.A, np.nan), axis=0)
        D = weighted_jaccard(X, w)
    elif metric=='weighted_hamming':
        # Reformat, keys as integers
        d = afm.uns['indel_priors']
        afm.uns['indel_priors'] = { 
            int(k) : { int(k_) : d[k][k_] for k_ in  d[k] } \
            for k in d
        }
        w = transform_priors(afm.uns['indel_priors'])
        D = weighted_hamming(X, w)
    else:
        D = pairwise_distances(X, metric=metric, n_jobs=ncores, force_all_finite=False)

    # Optional: rescale distances (min-max)
    if rescale:
        min_dist = D[~np.eye(D.shape[0], dtype=bool)].min()
        max_dist = D[~np.eye(D.shape[0], dtype=bool)].max()
        D = (D-min_dist)/(max_dist-min_dist)
        np.fill_diagonal(D, 0)

    afm.obsp[distance_key] = csr_matrix(D)
    

##

























