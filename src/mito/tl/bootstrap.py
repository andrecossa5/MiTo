"""
Bootstrap utils.
"""

import warnings
from copy import deepcopy

import numpy as np
from anndata import AnnData

##


def _resample_characters(n, strategy, frac_char_resampling, rng):
    """
    Character (i.e. .var) indices of one bootstrap replicate.

    "feature_resampling" draws a fraction of the characters without replacement;
    "jacknife" drops exactly one.
    """

    if strategy == 'feature_resampling':
        if frac_char_resampling == 1:
            return rng.choice(np.arange(n), n, replace=True)
        return rng.choice(np.arange(n), round(n*frac_char_resampling), replace=False)

    if strategy == 'jacknife':
        excluded = rng.choice(np.arange(n), 1)[0]
        return np.array([ x for x in np.arange(n) if x != excluded ])

    raise ValueError(
        f'{strategy} boot_strategy is not supported. Choose "feature_resampling" or "jacknife".'
    )


##


def _subset_characters(afm, idx):
    """
    A new AFM holding the resampled characters, with every layer, .X and .var subset
    consistently, and its own .uns (so replicates cannot write into each other's).
    """

    # Resampling with replacement draws a character more than once: AnnData warns about
    # the duplicate names, which are made unique right after (the tree solvers index
    # characters by name).
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', UserWarning)
        afm_new = afm[:, idx].copy()
    afm_new.uns = deepcopy(dict(afm.uns))
    afm_new.var_names_make_unique()

    return afm_new


##


def bootstrap_MiTo(
    afm: AnnData,
    boot_replicate: str = 'observed',
    boot_strategy: str = 'feature_resampling',
    frac_char_resampling: float = .8,
    seed: int = None
    ) -> AnnData:
    """
    One bootstrap replicate of an AFM, resampling MT-SNVs (characters).

    Read counts, allele frequencies and genotypes are all carried over for the resampled
    characters, so the replicate can go straight into `mito.pp.compute_distances` and
    `mito.tl.build_tree`. `boot_replicate="observed"` returns the AFM unchanged, which is
    how the point estimate is obtained in the same loop as the replicates.

    Parameters
    ----------
    afm : AnnData
        Filtered AFM.
    boot_replicate : str, optional
        Replicate name; "observed" returns a copy of the input. Default is "observed".
    boot_strategy : str, optional
        "feature_resampling" or "jacknife". Default is "feature_resampling".
    frac_char_resampling : float, optional
        Fraction of characters to draw (1 resamples with replacement). Default is 0.8.
    seed : int, optional
        Random seed of this replicate. Pass one (e.g. the replicate index) for a
        reproducible bootstrap. Default is None.

    Returns
    -------
    AnnData
        The replicate.
    """

    if boot_replicate == 'observed':
        return afm.copy()

    rng = np.random.default_rng(seed)
    idx = _resample_characters(afm.shape[1], boot_strategy, frac_char_resampling, rng)

    return _subset_characters(afm, idx)


##


def bootstrap_bin(
    afm: AnnData,
    boot_replicate: str = 'observed',
    boot_strategy: str = 'feature_resampling',
    frac_char_resampling: float = .8,
    seed: int = None
    ) -> AnnData:
    """
    Bootstrap replicate of an AFM whose characters are already binary (i.e. only
    .layers["bin"] is meaningful). Same semantics as `bootstrap_MiTo`.
    """

    if boot_replicate == 'observed':
        return afm.copy()
    if 'bin' not in afm.layers:
        raise ValueError('bootstrap_bin needs genotypes in afm.layers["bin"].')

    rng = np.random.default_rng(seed)
    idx = _resample_characters(afm.shape[1], boot_strategy, frac_char_resampling, rng)

    return _subset_characters(afm, idx)


##
