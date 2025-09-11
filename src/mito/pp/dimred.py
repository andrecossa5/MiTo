"""
Dimensionality reduction utils to reduce a (pre-filtered) AFMs.
"""

import logging
import numpy as np
from typing import Dict, Any
from scipy.linalg import eigh
from sklearn.decomposition import PCA
from umap.umap_ import simplicial_set_embedding, find_ab_params
from .kNN import *
from .distances import *


##


def find_diffusion_matrix(D):
    """
    Function to find the diffusion matrix P.
    """
    alpha = D.flatten().std()
    K = np.exp(-D**2 / alpha**2) # alpha is the variance of the distance matrix, here
    r = np.sum(K, axis=0)
    Di = np.diag(1/r)
    P = np.matmul(Di, K)
    D_right = np.diag((r)**0.5)
    D_left = np.diag((r)**-0.5)
    P_prime = np.matmul(D_right, np.matmul(P,D_left))

    return P_prime, P, Di, K, D_left


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


def _umap_from_X_conn(X, conn, ncomps=2, metric='cosine', metric_kwargs={}, seed=1234):
    """
    Wrapper around umap.umap_.simplicial_set_embedding() to create a umap embedding of the 
    feature matrix X using a precomputed fuzzy graph.
    """
    a, b = find_ab_params(1.0, 0.5)
    X_umap, _ = simplicial_set_embedding(
        X, conn, ncomps, 1.0, a, b, 1.0, 5, 200, 'spectral', 
        random_state=np.random.RandomState(seed), metric=metric, metric_kwds=metric_kwargs,
        densmap=None, densmap_kwds=None, output_dens=None
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


def _get_D(afm, distance_key, **kwargs):

    if 'distance_calculations' in afm.uns:
        if distance_key in afm.uns['distance_calculations']:
            if afm.uns['distance_calculations'][distance_key]['metric'] == kwargs['metric']:
                logging.info(f'Use precomputed {distance_key}')
                D = afm.obsp[distance_key].toarray()
                return D
            
    compute_distances(afm, distance_key=distance_key, **kwargs)
    D = afm.obsp[distance_key].toarray()

    return D


##


def reduce_dimensions(
    afm: AnnData, 
    layer: str = 'bin',
    distance_key: str = 'distances', 
    seed: int = 1234, 
    method: str = 'UMAP', 
    k: int = 10, 
    n_comps: int = 2, 
    ncores: int = 8,
    metric: str = 'weighted_jaccard', 
    bin_method: str = 'MiTo', 
    binarization_kwargs: Dict[str,Any] = {}
    ):
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
        Dissimilarity metric to use. Default is "weightde_jaccard".
    bin_method : str, optional
        Genotyping method. Default is "MiTo".
    binarization_kwargs : dict, optional
        Keyword arguments for binarization. Default is {}.
    """

    kwargs = dict(metric=metric, bin_method=bin_method, ncores=ncores, binarization_kwargs=binarization_kwargs)
    
    if method == 'PCA':
        X = _get_X(afm, layer)
        afm.obsm['X_pca'] = find_pca(X, n_pcs=n_comps, random_state=seed)

    elif method == 'UMAP':
        X = _get_X(afm, layer)
        D = _get_D(afm, distance_key, **kwargs)
        _, _, conn = kNN_graph(D=D, k=k, from_distances=True)
        afm.obsm['X_umap'] = _umap_from_X_conn(X, conn, ncomps=n_comps, metric=metric, seed=seed)

    elif method == 'diffmap':
        D = _get_D(afm, distance_key, **kwargs)
        P_prime, _,_,_, D_left = find_diffusion_matrix(D)
        afm.obsm['X_diffmap'] = find_diffusion_map(P_prime, D_left, n_eign=n_comps)
    
    else:
        raise ValueError(f'Method {method} not recognized. Please use "PCA", "UMAP" or "diffmap".')

    return


