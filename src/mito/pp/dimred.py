"""
Dimensionality reduction utils to reduce a (pre-filtered) AFMs.
"""

import logging

import numpy as np
import sklearn.preprocessing as pp
from anndata import AnnData
from scipy.linalg import eigh
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import PAIRWISE_BOOLEAN_FUNCTIONS, PAIRWISE_DISTANCE_FUNCTIONS
from umap.umap_ import find_ab_params, simplicial_set_embedding

from mito.ut.provenance import record

from .distances import compute_distances
from .kNN import _kNN_graph

# UMAP only uses `metric` for the spectral initialisation of a disconnected
# graph, where it forwards the name to sklearn.pairwise_distances. MiTo's own
# metrics (weighted_jaccard) are not known there, so they are
# mapped to a safe default -- the connectivity graph is precomputed either way.
_UMAP_SAFE_METRICS = (
    set(PAIRWISE_DISTANCE_FUNCTIONS) | set(PAIRWISE_BOOLEAN_FUNCTIONS)
    | {'correlation', 'sqeuclidean'}
)

##


def find_diffusion_matrix(D):
    """
    Symmetrised diffusion operator of a distance matrix, and the scaling that maps its
    eigenvectors back to diffusion coordinates.
    """
    alpha = D.flatten().std()
    K = np.exp(-D**2 / alpha**2)
    r = np.sum(K, axis=0)
    P = np.matmul(np.diag(1/r), K)
    D_left = np.diag((r)**-0.5)
    P_prime = np.matmul(np.diag((r)**0.5), np.matmul(P, D_left))

    return P_prime, D_left


##


def find_diffusion_map(P_prime, D_left, n_eign=10):
    """
    Function to find the diffusion coordinates in the diffusion space.
    """
    eigenValues, eigenVectors = eigh(P_prime)
    idx = eigenValues.argsort()[::-1]
    eigenValues = eigenValues[idx]
    eigenVectors = eigenVectors[:,idx]
    diffusion_coordinates = np.matmul(D_left, eigenVectors)

    return diffusion_coordinates[:,:n_eign]


##


def find_pca(X, n_pcs=30, random_state=1234):
    """
    Get PCA embeddings with fbpca.
    """
    model = PCA(n_components=n_pcs, random_state=random_state)
    X_pca = model.fit_transform(X)

    return X_pca


##


def _umap_from_X_conn(X, conn, ncomps=2, metric='cosine', metric_kwargs=None, seed=1234):
    """
    Wrapper around umap.umap_.simplicial_set_embedding() to create a umap embedding of the
    feature matrix X using a precomputed fuzzy graph.
    """
    if metric_kwargs is None:
        metric_kwargs = {}
    # NB: `parallel=False` (umap's default, passed explicitly here) is what makes the
    # embedding reproducible: the layout optimisation is numba-compiled, and with
    # parallel=True its updates race, so two runs of the same seed differ.
    a, b = find_ab_params(1.0, 0.5)
    X_umap, _ = simplicial_set_embedding(
        X, conn, ncomps, 1.0, a, b, 1.0, 5, 200, 'spectral',
        random_state=np.random.RandomState(seed), metric=metric, metric_kwds=metric_kwargs,
        densmap=None, densmap_kwds=None, output_dens=None, parallel=False
    )
    return X_umap


##


def _get_X(afm, layer):

    if layer in afm.layers:
        logging.info(f'Use {layer} layer')
        X = afm.layers[layer].toarray()
    else:
        logging.info(f'{layer} layer not found. Fall back to scaled .X raw AF...')
        X = pp.scale(afm.X.toarray())

    return X


##


def _get_D(afm, distance_key, metric, ncores):
    """
    Distances in .obsp[distance_key], computed with `metric` if they are not there yet.
    """

    computed = afm.uns.get('distances', {}).get(distance_key, {}).get('metric')
    if distance_key in afm.obsp and computed == metric:
        logging.info(f'Use precomputed {distance_key}')
    else:
        compute_distances(afm, distance_key=distance_key, metric=metric, ncores=ncores)

    return afm.obsp[distance_key].toarray()


##


def reduce_dimensions(
    afm: AnnData,
    method: str = 'UMAP',
    n_comps: int = 2,
    k: int = 10,
    layer: str = 'bin',
    distance_key: str = 'distances',
    metric: str = 'weighted_jaccard',
    seed: int = 1234,
    ncores: int = 8,
    copy: bool = False
    ) -> AnnData | None:
    """
    Dimensionality reduction for an Allele Frequency Matrix.

    Parameters
    ----------
    afm : AnnData
        Allele Frequency Matrix.
    layer : str, optional
        Layer to use. Default is "bin".
    distance_key : str, optional
        afm.obsp key to append distances. Default is "distances".
    seed : int, optional
        Random seed. Default is 1234.
    method : str, optional
        Dimensionality reduction method. Default is "UMAP".
    k : int, optional
        Number of neighbors to use for kNN search. Default is 10.
    n_comps : int, optional
        Number of dimensions of the output embedding. Default is 2.
    metric : str, optional
        Dissimilarity metric, if distances have to be computed. Default is "weighted_jaccard".
    ncores : int, optional
        Cores for the distance computation. Default is 8.
    copy : bool, optional
        Return a modified copy instead of updating `afm` in place. Default is False.

    Returns
    -------
    AnnData | None
        Updated AFM if `copy` is True, otherwise None. The embedding is added to
        .obsm["X_pca" | "X_umap" | "X_diffmap"], with its parameters in
        .uns["mito"]["reduce_dimensions"].

    Notes
    -----
    Embeddings are reproducible: PCA, the UMAP layout and the diffusion map are all
    seeded by `seed`, and the UMAP optimisation runs single-threaded (see
    `_umap_from_X_conn`).
    """

    afm = afm.copy() if copy else afm

    if method == 'PCA':
        X = _get_X(afm, layer)
        afm.obsm['X_pca'] = find_pca(X, n_pcs=n_comps, random_state=seed)

    elif method == 'UMAP':
        X = _get_X(afm, layer)
        D = _get_D(afm, distance_key, metric, ncores)
        _, _, conn = _kNN_graph(D=D, k=k, from_distances=True)
        umap_metric = metric if metric in _UMAP_SAFE_METRICS else 'euclidean'
        if umap_metric != metric:
            logging.info(
                f'UMAP spectral init does not support metric "{metric}": using "euclidean". '
                f'Cell-cell distances are unaffected -- the kNN graph is precomputed.'
            )
        afm.obsm['X_umap'] = _umap_from_X_conn(X, conn, ncomps=n_comps, metric=umap_metric, seed=seed)

    elif method == 'diffmap':
        D = _get_D(afm, distance_key, metric, ncores)
        P_prime, D_left = find_diffusion_matrix(D)
        afm.obsm['X_diffmap'] = find_diffusion_map(P_prime, D_left, n_eign=n_comps)

    else:
        raise ValueError(f'Method {method} not recognized. Please use "PCA", "UMAP" or "diffmap".')

    record(afm, 'reduce_dimensions', {
        'method':method, 'n_comps':n_comps, 'k':k, 'layer':layer, 'metric':metric, 'seed':seed
    })

    return afm if copy else None


