"""
Nearest neighbors utils.
"""

import numpy as np
from typing import Tuple, Dict, Any
from umap.umap_ import nearest_neighbors  
from umap.umap_ import fuzzy_simplicial_set 
from scipy.sparse import coo_matrix, csr_matrix, issparse


##



def _get_sparse_matrix_from_indices_distances_umap(
    knn_indices, knn_dists, n_obs, n_neighbors
    ):
    """
    Legacy function from scanpy neighbors.
    """
    rows = np.zeros((n_obs * n_neighbors), dtype=np.int64)
    cols = np.zeros((n_obs * n_neighbors), dtype=np.int64)
    vals = np.zeros((n_obs * n_neighbors), dtype=np.float64)

    for i in range(knn_indices.shape[0]):
        for j in range(n_neighbors):
            if knn_indices[i, j] == -1:
                continue
            if knn_indices[i, j] == i:
                val = 0.0
            else:
                val = knn_dists[i, j]

            rows[i * n_neighbors + j] = i
            cols[i * n_neighbors + j] = knn_indices[i, j]
            vals[i * n_neighbors + j] = val

    result = coo_matrix((vals, (rows, cols)), shape=(n_obs, n_obs))
    result.eliminate_zeros()

    return result.tocsr()


##


def _NN(
    X: np.array, 
    k: int = 15, 
    metric: str = 'cosine', 
    implementation: str = 'pyNNDescent', 
    random_state: int = 1234, 
    metric_kwds: Dict[str,Any] = {}
    ) -> Tuple[csr_matrix,csr_matrix]:
    """
    kNN search over an X obs x features matrix. pyNNDescent and hsnwlib implementation available.
    """

    if k <= 500 and implementation == 'pyNNDescent':
        knn_indices, knn_dists, _ = nearest_neighbors(
            X,
            k,
            metric=metric, 
            metric_kwds=metric_kwds,
            angular=False,
            random_state=random_state
        )
    else:
        raise Exception(f'Incorrect options: {metric}, {metric_kwds}, {implementation}')

    return (knn_indices, knn_dists)


##


def get_idx_from_simmetric_matrix(X, k=15):
    """
    Given a simmetric affinity matrix, get its k NN indeces and their values.
    """
    if issparse(X):
        X = X.toarray()
        
    assert X.shape[0] == X.shape[1]

    idx_all = []
    for i in range(X.shape[0]):
        x = np.delete(X[i,:], i)
        idx_all.append(x.argsort())

    idx_all = np.concatenate([
        np.arange(X.shape[0]).reshape(X.shape[0],1),
        np.vstack(idx_all)
        ], axis=1
    )

    idx = idx_all[:,:k]
    dists = X[np.arange(X.shape[0])[:, None], idx]

    return idx, dists


##


def kNN_graph(
    X: np.array = None, 
    D: np.array = None,
    k: int = 10, 
    from_distances: bool = False, 
    nn_kwargs: Dict[str,Any] = {}
    ) -> Tuple[np.array,csr_matrix,csr_matrix]:
    """
    kNN graph computation.

    Parameters
    ----------
    X : np.array
        Feature matrix (observations x features).
    D : np.array, optional
        Pairwise distance matrix. Default is None.
    k : int, optional
        Number of neighbors. Default is 10.
    from_distances : bool, optional
        Whether to start from precomputed distances. Default is False.
    nn_kwargs : dict, optional
        Additional keyword arguments for kNN search.

    Returns
    -------
    tuple of (np.array, csr_matrix, csr_matrix)
        A tuple containing:
        - A numpy array of shape (n_samples, k) with the indices of the k-nearest neighbors.
        - A csr_matrix representing the connectivity matrix of the kNN graph.
        - A csr_matrix representing the distances corresponding to the kNN graph.
    """

    if from_distances:
        knn_indices, knn_dists = get_idx_from_simmetric_matrix(D, k=k)
        n = D.shape[0]
    else:
        knn_indices, knn_dists = _NN(X, k, **nn_kwargs)
        n = X.shape[0]
    
    # Compute connectivities
    connectivities = fuzzy_simplicial_set(
        coo_matrix(([], ([], [])), 
        shape=(n, 1)),
        k,
        None,
        None,
        knn_indices=knn_indices,
        knn_dists=knn_dists,
        set_op_mix_ratio=1.0,
        local_connectivity=1.0,
    )
    connectivities = connectivities[0]
    
    # Sparsiy
    distances = _get_sparse_matrix_from_indices_distances_umap(
        knn_indices, knn_dists, n, k
    )

    return (knn_indices, distances, connectivities)


##


def spatial_w_from_idx(idx):
    n = idx.shape[0]
    spw = np.zeros((n,n))
    for i in range(n):
        spw[i,idx[i,1:]] = 1
    return spw


##
