"""
Custom distance function among cell AF profiles.
"""

import inspect
import logging
from typing import Any

import numpy as np
import sklearn.preprocessing as pp
from anndata import AnnData
from cassiopeia.solver.solver_utilities import transform_priors
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import PAIRWISE_BOOLEAN_FUNCTIONS, PAIRWISE_DISTANCE_FUNCTIONS, pairwise_distances

from mito.ut.stats_utils import genotype_mix

##


discrete_metrics = PAIRWISE_BOOLEAN_FUNCTIONS + ['weighted_jaccard', 'weighted_hamming']
continuous_metrics = list(PAIRWISE_DISTANCE_FUNCTIONS.keys()) + ['correlation', 'sqeuclidean']

# scikit-learn renamed `force_all_finite` to `ensure_all_finite` in 1.6 and dropped
# the old spelling in 1.8; detect which one this installation accepts.
_ALLOW_NON_FINITE_KWARG = (
    'ensure_all_finite'
    if 'ensure_all_finite' in inspect.signature(pairwise_distances).parameters
    else 'force_all_finite'
)


##


def genotype_mixtures(
    AD: np.array,
    DP: np.array,
    t_prob: float = .75,
    t_vanilla: float = .001,
    min_AD: int = 2,
    debug: bool = False
    ) -> np.array:
    """
    Single-cell MT-SNVs probabilistic genotyping.
    Thresholds the binomial mixtures posterior
    probabilities obtained from the model defined in
    (Kwock et al., 2022).
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


def genotype_MiTo(
    AD: np.array,
    DP: np.array,
    t_prob: float = .7,
    t_vanilla: float = 0,
    min_AD: int = 1,
    min_cell_prevalence: float = .1,
    debug: bool = False
    ) -> np.array:
    """
    MiTo genotype calling strategy.

    If a mutation has prevalence (i.e., fraction of positive cells with AD >= min_AD
    and AF >= t_vanilla) greater than or equal to min_cell_prevalence, MiTo uses
    bin_mixtures probabilistic modeling to assign each cell-variant genotype.
    Otherwise, each cell's mutational status is assigned using simple thresholding,
    as in the "vanilla" method.

    Parameters
    ----------
    AD : np.array
        Alternative allele (consensus) UMI counts. Expected shape is Cell x variant.
    DP : np.array
        Total UMI counts. Expected shape is Cell x variant.
    t_prob : float, optional
        Threshold on posterior probabilities. Default is 0.7.
    t_vanilla : float, optional
        Threshold on raw allele frequency. Default is 0.
    min_AD : int, optional
        Minimum number of alternative UMI counts to assign the "mut" (1) genotype.
        Default is 1.
    min_cell_prevalence : float, optional
        Minimum cell prevalence to use probabilistic genotyping. Default is 0.1.
    debug : bool, optional
        If True, print additional debugging information. Default is False.

    Returns
    -------
    X : np.array
        Binary genotype array.
    """

    X = np.zeros(AD.shape)
    n_binom = 0

    for idx in range(AD.shape[1]):
        test = (AD[:,idx]/(DP[:,idx]+.0000001)>t_vanilla)
        prevalence = test.sum() / test.size
        if prevalence >= min_cell_prevalence:
            X[:,idx] = genotype_mix(
                AD[:,idx], DP[:,idx], t_prob=t_prob,
                t_vanilla=t_vanilla, min_AD=min_AD, debug=debug
            )
            n_binom += 1
        else:
            X[:,idx] = np.where(test & (AD[:,idx]>=min_AD), 1, 0)

    logging.info(f'n MT-SNVs genotyped with binomial mixtures: {n_binom}')

    return X


##



##


def call_genotypes(
    afm: AnnData,
    bin_method: str = 'MiTo',
    t_vanilla: float = .0,
    min_AD: int = 2,
    t_prob: float = .7,
    min_cell_prevalence: float = .1
    ):
    """
    Call genotypes. The 'bin' layer is added in-place.

    Two strategies are implemented:
    * "vanilla": simple, hard thresholding on raw AF values or number of alternative allele counts.
    * "MiTo": hybrid MiTo genotype calling strategy (see mito.pp.genotype_MiTo).

    Parameters
    ----------
    afm : AnnData
        Allele Frequency Matrix.
    bin_method : str, optional
        Genotyping strategy. Default is "MiTo".
    t_prob : float, optional
        Threshold on posterior probabilities. Default is 0.7.
    t_vanilla : float, optional
        Threshold on raw allele frequencies. Default is 0.
    min_AD : int, optional
        Minimum number of alternative UMI counts to assign the 'mut' (1) genotype. Default is 1.
    min_cell_prevalence : float, optional
        Minimum cell prevalence to use probabilistic genotyping. Default is 0.1.
    """

    assert 'AD' in afm.layers
    assert 'site_coverage' in afm.layers or 'DP' in afm.layers
    cov_layer = 'site_coverage' if 'site_coverage' in afm.layers else 'DP'

    X = afm.X.toarray()
    AD = afm.layers['AD'].toarray()
    DP = afm.layers[cov_layer].toarray()

    if bin_method == 'vanilla':
        X = np.where((X>=t_vanilla) & (AD>=min_AD), 1, 0)
    elif bin_method == 'MiTo':
        if cov_layer == 'site_coverage':
            X = genotype_MiTo(AD, DP, t_prob=t_prob, t_vanilla=t_vanilla,
                              min_AD=min_AD, min_cell_prevalence=min_cell_prevalence)
        else:
            raise ValueError("""
                    MiTo genotyping requires total site coverage info
                    in afm.layers["site_coverage"]
                    """
            )
    else:
        raise ValueError(
            f'"{bin_method}" is not a valid genotype calling method. '
            f'Choose one of: vanilla, MiTo.'
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


def _get_priors(afm, key='priors'):

    if key not in afm.varm:
        raise ValueError(
            f'The "weighted_hamming" metric needs per-character priors in '
            f'afm.varm["{key}"], which this AFM does not have. Provide them, or '
            f'choose another metric (e.g. "weighted_jaccard").'
        )
    W = afm.varm[key]
    priors = {}
    for i in range(W.shape[0]):
        priors[i] = {}
        for j in range(W.shape[1]):
            if W[i,j] != -1:
                priors[i][j] = W[i,j]

    return priors


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
    afm, distance_key='distances', precomputed=False, metric='jaccard',
    bin_method='MiTo', binarization_kwargs=None, verbose=True
    ):
    """
    Preprocess a feature matrix for cell-cell distance computations.
    """

    if binarization_kwargs is None:
        binarization_kwargs = {}
    layer = None
    scLT_system = afm.uns['scLT_system']
    if 'distance_calculations' not in afm.uns:
        afm.uns['distance_calculations'] = { distance_key: {}}
    else:
        afm.uns['distance_calculations'][distance_key] = {}

    if scLT_system in ['RedeeM', 'MAESTER', 'Smart-seq2']:

        if metric in continuous_metrics:
            layer = 'scaled'
            if layer in afm.layers and precomputed:
                if verbose:
                    logging.info('Use precomputed scaled layer...')
            else:
                logging.info('Scale raw AFs in afm.X')
                afm.layers['scaled'] = csr_matrix(pp.scale(afm.X.toarray()))

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

    elif scLT_system in ['scWGS', 'Cas9', 'EPI-clone']:

        if metric in continuous_metrics:
            raise ValueError(f'For {scLT_system} only discrete metrics are available!')
        elif metric in discrete_metrics:
            if 'bin' in afm.layers:
                layer = 'bin'
                if verbose:
                    logging.info('Use precomputed bin layer.')
            else:
                raise ValueError(f'With the {scLT_system} system, provide an AFM with binary character matrix in afm.layers, under the "bin" key!')
        else:
            raise ValueError(f'{metric} is not a valid metric! Specify for a valid metric in {discrete_metrics}')
    else:
        raise ValueError(f'{scLT_system} is not a valid scLT system. Choose one between MAESTER, scWGS, RedeeM, Cas9, and EPI-clone.')

    afm.uns['distance_calculations'][distance_key]['metric'] = metric
    afm.uns['distance_calculations'][distance_key]['layer'] = layer


##


def compute_distances(
    afm: AnnData,
    distance_key: str = 'distances',
    metric: str = 'weighted_jaccard',
    precomputed: bool = False,
    bin_method: str = 'MiTo',
    binarization_kwargs: dict[str,Any] = None,
    ncores: int = 1,
    rescale: bool = True,
    verbose: bool = True
    ):
    """
    Pairwise cell-cell (or sample-) distance computation in some character space
    (e.g., MT-SNVs mutation space). Updates the afm.obsp slot in-place.

    Parameters
    ----------
    afm : AnnData
        Allele Frequency Matrix (.X slot or 'bin' layer present).
    distance_key : str, optional
        Key in afm.obsp at which distances will be stored. Default is "distances".
    metric : str, optional
        Distance metric to use. Default is "weighted_jaccard".
    precomputed : bool, optional
        If True, use precomputed genotypes; otherwise, recompute from scratch.
        Default is False.
    bin_method : str, optional
        Genotyping method. Default is "MiTo".
    binarization_kwargs : dict, optional
        Keyword arguments for the discretization function. Default is {}.
    ncores : int, optional
        Number of cores for parallel computation. Default is 1.
    rescale : bool, optional
        Whether to apply min-max rescaling to distance values. Default is True.
    verbose : bool, optional
        Whether to print verbose logging. Default is True.
    """

    # Preprocess afm
    if binarization_kwargs is None:
        binarization_kwargs = {}
    preprocess_feature_matrix(
        afm, distance_key=distance_key, metric=metric, precomputed=precomputed,
        bin_method=bin_method, binarization_kwargs=binarization_kwargs, verbose=verbose
    )
    layer = afm.uns['distance_calculations'][distance_key]['layer']
    metric = afm.uns['distance_calculations'][distance_key]['metric']
    X = afm.layers[layer].toarray()

    if verbose:
        logging.info(f'Compute distances: ncores={ncores}, metric={metric}.')

    if X.shape[0] == 0:
        raise ValueError(
            'Cannot compute distances: no cells left in the AFM. '
            'Relax the cell and variant filters.'
        )
    if X.shape[1] == 0:
        raise ValueError(
            'Cannot compute distances: no characters left in the AFM. '
            'The chosen filtering strategy retained no MT-SNVs; relax its thresholds '
            'or pick a different one.'
        )

    # Calculate distances (handle weights, if necessary)
    if metric=='weighted_jaccard':
        af = afm.X.toarray()
        w = np.nanmedian(np.where(af>0, af, np.nan), axis=0)
        D = weighted_jaccard(X, w)
    elif metric=='weighted_hamming':
        w = _get_priors(afm)
        w = transform_priors(w)
        D = weighted_hamming(X, w)
    elif metric in PAIRWISE_BOOLEAN_FUNCTIONS:
        # scipy's boolean metrics reject float input, and skipping the finiteness
        # check also skips the cast scikit-learn would otherwise apply.
        D = pairwise_distances(X.astype(bool), metric=metric, n_jobs=ncores)
    else:
        D = pairwise_distances(
            X, metric=metric, n_jobs=ncores, **{_ALLOW_NON_FINITE_KWARG: False}
        )

    # Optional: rescale distances (min-max)
    if rescale and D.shape[0] > 1:
        off_diagonal = D[~np.eye(D.shape[0], dtype=bool)]
        min_dist = off_diagonal.min()
        max_dist = off_diagonal.max()
        if max_dist > min_dist:
            D = (D-min_dist)/(max_dist-min_dist)
        np.fill_diagonal(D, 0)

    afm.obsp[distance_key] = csr_matrix(D)


##

























