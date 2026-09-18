"""
Custom distance function among cell AF profiles.
"""

import inspect
import logging
import warnings

import numpy as np
import sklearn.preprocessing as pp
from anndata import AnnData
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import PAIRWISE_BOOLEAN_FUNCTIONS, PAIRWISE_DISTANCE_FUNCTIONS, pairwise_distances

##


discrete_metrics = PAIRWISE_BOOLEAN_FUNCTIONS + ['weighted_jaccard']
continuous_metrics = list(PAIRWISE_DISTANCE_FUNCTIONS.keys()) + ['correlation', 'sqeuclidean']

# scikit-learn renamed `force_all_finite` to `ensure_all_finite` in 1.6 and dropped
# the old spelling in 1.8; detect which one this installation accepts.
_ALLOW_NON_FINITE_KWARG = (
    'ensure_all_finite'
    if 'ensure_all_finite' in inspect.signature(pairwise_distances).parameters
    else 'force_all_finite'
)


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


def preprocess_feature_matrix(
    afm, distance_key='distances', metric='jaccard', verbose=True
    ):
    """
    Preprocess a feature matrix for cell-cell distance computations.

    Discrete metrics need genotypes in afm.layers["bin"] (see `mito.pp.call_genotypes`,
    or `mito.pp.filter_afm` which calls it); continuous ones need scaled AFs, which are
    computed here if missing.
    """

    layer = None
    if 'distances' not in afm.uns:
        afm.uns['distances'] = { distance_key: {}}
    else:
        afm.uns['distances'][distance_key] = {}

    # What can be computed depends on what the AFM holds, not on which assay produced it.
    if metric in discrete_metrics:
        layer = 'bin'
        if 'bin' not in afm.layers:
            raise ValueError(
                f'The "{metric}" metric needs genotypes in afm.layers["bin"]. '
                f'Run mito.pp.call_genotypes (or mito.pp.filter_afm) first.'
            )
        if verbose:
            logging.info('Use the genotypes in the bin layer.')

    elif metric in continuous_metrics:
        layer = 'scaled'
        if 'scaled' in afm.layers:
            if verbose:
                logging.info('Use precomputed scaled layer...')
        else:
            logging.info('Scale raw AFs in afm.X')
            afm.layers['scaled'] = csr_matrix(pp.scale(afm.X.toarray()))

    else:
        raise ValueError(
            f'{metric} is not a valid metric. Choose one of {discrete_metrics} '
            f'(on genotypes) or {continuous_metrics} (on allele frequencies).'
        )

    afm.uns['distances'][distance_key]['metric'] = metric
    afm.uns['distances'][distance_key]['layer'] = layer


##


def compute_distances(
    afm: AnnData,
    distance_key: str = 'distances',
    metric: str = 'weighted_jaccard',
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
    ncores : int, optional
        Number of cores for parallel computation. Default is 1.
    rescale : bool, optional
        Whether to apply min-max rescaling to distance values. Default is True.
    verbose : bool, optional
        Whether to print verbose logging. Default is True.
    """

    # Preprocess afm
    preprocess_feature_matrix(
        afm, distance_key=distance_key, metric=metric, verbose=verbose
    )
    layer = afm.uns['distances'][distance_key]['layer']
    metric = afm.uns['distances'][distance_key]['metric']
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
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', category=RuntimeWarning)   # all-NaN slices
            # The weight of a character is the typical AF of the cells that CARRY it, so
            # when genotypes are available the median is taken over called cells only.
            # Over all cells with any read, background single reads in non-carriers drag
            # the weight of a clean marker down towards that of a noisy, prevalent variant.
            positive = (af>0) & (afm.layers['bin'].toarray()>0) if layer=='bin' else (af>0)
            w = np.nanmedian(np.where(positive, af, np.nan), axis=0)
        # A character with no positive AF (e.g. all its calls are imputed, or it was
        # carried over with no positive cell) gets a NaN weight, and a single NaN weight
        # makes EVERY cell-cell distance NaN. Fall back to the median of the defined
        # weights for those characters.
        bad = ~np.isfinite(w)
        if bad.any():
            fallback = np.nanmedian(w[~bad]) if (~bad).any() else 1.0
            fallback = fallback if np.isfinite(fallback) else 1.0
            w = np.where(bad, fallback, w)
            logging.warning(
                f'{bad.sum()} character(s) have no cell with AF > 0: their weighted_jaccard '
                f'weight is undefined and was set to {fallback:.3g} (median of the others).'
            )
        D = weighted_jaccard(X, w)
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

























