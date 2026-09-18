"""
mito.pp.filter_non_clonal_variants

The variant QC that replaced Moran's I. Its two tests answer different questions, and
both are needed: the join count sees clones marked by several variants, exclusivity sees
clones marked by a single one. Scattered noise is neither, and is what must be rejected.
"""

import numpy as np
import pytest
from scipy.sparse import csr_matrix

import mito as mt


def _add_variant(afm, carriers, af=0.15, error_rate=0.002, seed=0):
    """Append one variant carried by `carriers`, at `af`, on a sequencing-error floor."""
    rng = np.random.default_rng(seed)
    DP = np.asarray(afm.layers["DP"])
    AD = afm.layers["AD"].toarray()

    depth = np.clip(rng.poisson(DP.mean(), afm.shape[0]), 5, None).astype(np.int16)
    counts = np.where(carriers, rng.binomial(depth, af), rng.binomial(depth, error_rate))

    new = afm[:, [0]].copy()
    new.var_names = ["9999_A>T"]
    new.var["pos"] = 9999
    new.var["ref"], new.var["alt"] = "A", "T"
    new.layers["AD"] = csr_matrix(counts.reshape(-1, 1).astype(np.int16))
    new.layers["DP"] = depth.reshape(-1, 1)
    new.X = csr_matrix((counts / depth).reshape(-1, 1).astype(np.float32))
    new.layers["genotype"] = csr_matrix(carriers.reshape(-1, 1).astype(np.int8))

    import anndata as ad
    out = ad.concat([afm, new], axis=1, merge="first", uns_merge="first")
    out.obs = afm.obs.copy()
    out.uns = dict(afm.uns)
    return out


# -- what it keeps and what it rejects --------------------------------------


def test_keeps_the_planted_clone_markers(afm):
    mt.pp.call_genotypes(afm, alpha=1e-4)
    planted = set(afm.var_names[afm.layers["genotype"].toarray().any(axis=0)])
    mt.pp.filter_non_clonal_variants(afm, alpha=0.05)
    kept = set(afm.var_names)
    assert len(planted & kept) / len(planted) >= 0.8


def test_rejects_scattered_noise(afm):
    """
    A variant whose carriers are a random set of cells is not a lineage: it must fail
    both tests. This is the case Moran's I passed 96% of the time, because it included
    the variant under test in the cell-cell distances.
    """
    rng = np.random.default_rng(1)
    scattered = rng.random(afm.shape[0]) < 0.25         # same prevalence as a real clone
    a = _add_variant(afm, scattered, seed=1)

    mt.pp.call_genotypes(a, alpha=1e-4)
    mt.pp.filter_non_clonal_variants(a, alpha=0.05)
    assert "9999_A>T" not in a.var_names


def test_keeps_a_clone_marked_by_a_single_variant(afm):
    """
    A sole marker has no correlated variant to cluster with, so the join count has little
    power; exclusivity catches it, because its carriers lack every other clone's markers.
    """
    clone = (afm.obs["GBC"].astype(str) == "clone_0").values
    a = afm[:, ~afm.layers["genotype"].toarray()[clone].any(axis=0)].copy()   # drop its markers
    a = _add_variant(a, clone, seed=2)

    mt.pp.call_genotypes(a, alpha=1e-4)
    mt.pp.filter_non_clonal_variants(a, alpha=0.05)
    assert "9999_A>T" in a.var_names


# -- statistics and bookkeeping ---------------------------------------------


def test_writes_the_p_values_of_both_tests(afm):
    mt.pp.call_genotypes(afm, alpha=1e-4)
    a = mt.pp.filter_non_clonal_variants(afm, alpha=1.0, copy=True)     # keep everything
    for column in ("p_join", "p_exclusive", "p_clonal", "clonal"):
        assert column in a.var.columns
    assert np.allclose(a.var["p_clonal"], 2 * np.minimum(a.var["p_join"], a.var["p_exclusive"]))
    assert ((a.var["p_join"] > 0) & (a.var["p_join"] <= 1)).all()


def test_alpha_sets_how_much_is_kept(afm):
    mt.pp.call_genotypes(afm, alpha=1e-4)
    strict = mt.pp.filter_non_clonal_variants(afm, alpha=1e-3, copy=True)
    loose = mt.pp.filter_non_clonal_variants(afm, alpha=0.2, copy=True)
    assert strict.shape[1] <= loose.shape[1]


def test_is_reproducible_given_a_seed(afm):
    mt.pp.call_genotypes(afm, alpha=1e-4)
    first = mt.pp.filter_non_clonal_variants(afm, seed=3, copy=True)
    second = mt.pp.filter_non_clonal_variants(afm, seed=3, copy=True)
    assert list(first.var_names) == list(second.var_names)
    assert np.allclose(first.var["p_join"], second.var["p_join"])


def test_records_its_parameters(afm):
    mt.pp.call_genotypes(afm, alpha=1e-4)
    mt.pp.filter_non_clonal_variants(afm, alpha=0.05, k=10, n_perm=199)
    record = afm.uns["clonality"]
    assert record["alpha"] == 0.05 and record["k"] == 10 and record["n_perm"] == 199
    assert record["n_clonal"] == afm.shape[1]


def test_requires_genotypes(afm):
    with pytest.raises(ValueError, match="call_genotypes"):
        mt.pp.filter_non_clonal_variants(afm)


def test_copy_semantics(afm):
    mt.pp.call_genotypes(afm, alpha=1e-4)
    n0 = afm.shape[1]
    out = mt.pp.filter_non_clonal_variants(afm, alpha=1e-6, copy=True)
    assert out is not afm and afm.shape[1] == n0
    assert out.shape[1] <= n0
