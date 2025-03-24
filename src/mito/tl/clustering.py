"""
Clustering: leiden/vireoSNP.
"""

import numpy as np
import pandas as pd
import scanpy as sc
import leidenalg


##


def leiden_clustering(A: np.array, res: float = 0.5):
    """
    Compute leiden clustering, at some resolution.
    """

    g = sc._utils.get_igraph_from_adjacency(A, directed=True)
    part = leidenalg.find_partition(
        g,
        leidenalg.RBConfigurationVertexPartition,
        resolution_parameter=res,
        seed=1234
    )
    labels = np.array(part.membership)

    return labels


##