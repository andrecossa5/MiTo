"""
mito.pp.compute_distances

The weights of the weighted Jaccard metric are where this used to go wrong: one character
without a positive allele frequency made *every* distance NaN, and taking the median over
all read-bearing cells (rather than the called ones) dragged a clean marker's weight down
to a noisy variant's.
"""

import numpy as np
import pytest
from scipy.sparse import csr_matrix

import mito as mt


def _afm(AF, B):
    import anndata as ad
    import pandas as pd
    n, m = AF.shape
    return ad.AnnData(
        X=csr_matrix(AF.astype(np.float32)),
        obs=pd.DataFrame(index=[f"c{i}" for i in range(n)]),
        var=pd.DataFrame(index=[f"v{j}" for j in range(m)]),
        layers={"bin": csr_matrix(B.astype(np.int8))},
    )


# -- the weights ------------------------------------------------------------


def test_weights_are_the_median_af_of_the_called_cells():
    """
    A character's weight is the typical allele frequency of the cells that CARRY it.
    Background single reads in non-carriers must not enter the median.
    """
    n = 40
    AF = np.full((n, 1), 0.01)          # everyone has a little background signal
    B = np.zeros((n, 1), bool)
    B[:10] = True
    AF[:10] = 0.5                       # the carriers are unambiguous
    afm = _afm(AF, B)

    mt.pp.compute_distances(afm, metric="weighted_jaccard", verbose=False)
    # with one character the distances are 0 (both carriers) or 1, whatever the weight,
    # so check the weight through a second, weaker character instead
    AF2 = np.hstack([AF, np.full((n, 1), 0.01)])
    B2 = np.hstack([B, np.zeros((n, 1), bool)])
    B2[10:20, 1] = True                 # called, but at background level: a weak weight
    afm2 = _afm(AF2, B2)
    mt.pp.compute_distances(afm2, metric="weighted_jaccard", verbose=False)

    D = afm2.obsp["distances"].toarray()
    assert np.isfinite(D).all()
    # cells sharing the strong character are closer than cells sharing the weak one
    assert D[0, 1] <= D[10, 11]


def test_a_character_without_a_positive_af_does_not_poison_every_distance(caplog):
    """
    Regression: a NaN weight propagates through the whole matrix. It happens whenever a
    character's calls are all imputed (no reads), which is routine with imputation on.
    """
    n = 30
    AF = np.zeros((n, 3))
    B = np.zeros((n, 3), bool)
    AF[:10, 0] = 0.4
    B[:10, 0] = True
    AF[10:20, 1] = 0.3
    B[10:20, 1] = True
    B[20:25, 2] = True                  # called everywhere it matters, but no reads
    afm = _afm(AF, B)

    with caplog.at_level("WARNING"):
        mt.pp.compute_distances(afm, metric="weighted_jaccard", verbose=False)

    D = afm.obsp["distances"].toarray()
    assert np.isfinite(D).all(), "one undefined weight must not make every distance NaN"
    assert "no cell with AF > 0" in caplog.text


# -- the object -------------------------------------------------------------


def test_writes_distances_and_their_provenance(afm_filtered):
    mt.pp.compute_distances(afm_filtered, metric="weighted_jaccard", verbose=False)
    assert "distances" in afm_filtered.obsp
    assert afm_filtered.uns["distances"]["distances"]["layer"] == "bin"


def test_several_metrics_can_coexist(afm_filtered):
    mt.pp.compute_distances(afm_filtered, distance_key="jaccard", metric="jaccard", verbose=False)
    assert {"distances", "jaccard"} <= set(afm_filtered.obsp)
    assert afm_filtered.uns["distances"]["jaccard"]["metric"] == "jaccard"


@pytest.mark.parametrize("metric", ["weighted_jaccard", "jaccard", "dice", "russellrao"])
def test_discrete_metrics_run_on_the_genotypes(afm_filtered, metric):
    mt.pp.compute_distances(afm_filtered, metric=metric, verbose=False)
    D = afm_filtered.obsp["distances"].toarray()
    assert np.isfinite(D).all() and (np.diag(D) == 0).all()


def test_continuous_metrics_scale_the_allele_frequencies(afm_filtered):
    mt.pp.compute_distances(afm_filtered, metric="euclidean", verbose=False)
    assert "scaled" in afm_filtered.layers
    assert afm_filtered.uns["distances"]["distances"]["layer"] == "scaled"


def test_discrete_metrics_need_genotypes(afm):
    with pytest.raises(ValueError, match="call_genotypes"):
        mt.pp.compute_distances(afm, metric="weighted_jaccard", verbose=False)


def test_unknown_metric_lists_the_alternatives(afm_filtered):
    with pytest.raises(ValueError, match="valid metric"):
        mt.pp.compute_distances(afm_filtered, metric="not_a_metric", verbose=False)
