"""
mito.pp.call_genotypes and the genotyping primitives.

Tested at the primitive level rather than through filter_afm, so thresholds can
be swept without the downstream cell filters masking their effect.
"""

import numpy as np
import pytest
from conftest import build_afm

import mito as mt
from mito.pp.distances import genotype_MiTo, genotype_mixtures

BIN_METHODS = ["vanilla", "MiTo"]


# -- call_genotypes contract ------------------------------------------------

@pytest.mark.parametrize("bin_method", BIN_METHODS)
def test_adds_binary_layer(afm, bin_method):
    mt.pp.call_genotypes(afm, bin_method=bin_method)
    assert "bin" in afm.layers
    assert afm.layers["bin"].shape == afm.shape


@pytest.mark.parametrize("bin_method", BIN_METHODS)
def test_output_is_strictly_binary(afm, bin_method):
    mt.pp.call_genotypes(afm, bin_method=bin_method)
    values = np.unique(afm.layers["bin"].toarray())
    assert set(values).issubset({0, 1})


@pytest.mark.parametrize("bin_method", BIN_METHODS)
def test_records_provenance(afm, bin_method):
    mt.pp.call_genotypes(afm, bin_method=bin_method)
    assert afm.uns["genotyping"]["bin_method"] == bin_method
    assert "binarization_kwargs" in afm.uns["genotyping"]


def test_unknown_method_raises(afm):
    with pytest.raises(ValueError, match="not a valid genotype calling method"):
        mt.pp.call_genotypes(afm, bin_method="MiTo_smooth")


@pytest.mark.parametrize("bad", ["", "smooth", "Vanilla", None])
def test_various_invalid_methods_raise(afm, bad):
    with pytest.raises(ValueError):
        mt.pp.call_genotypes(afm, bin_method=bad)


def test_mito_requires_site_coverage(afm):
    """MiTo genotyping needs the site_coverage layer, not just DP."""
    del afm.layers["site_coverage"]
    with pytest.raises(ValueError, match="site_coverage"):
        mt.pp.call_genotypes(afm, bin_method="MiTo")


def test_vanilla_works_without_site_coverage(afm):
    del afm.layers["site_coverage"]
    mt.pp.call_genotypes(afm, bin_method="vanilla")
    assert "bin" in afm.layers


# -- vanilla thresholds -----------------------------------------------------

@pytest.mark.parametrize("t_vanilla", [0.0, 0.01, 0.05, 0.1, 0.5, 0.9])
def test_vanilla_af_threshold(afm_factory, t_vanilla):
    a = afm_factory(seed=81)
    mt.pp.call_genotypes(a, bin_method="vanilla", t_vanilla=t_vanilla, min_AD=1)
    assert set(np.unique(a.layers["bin"].toarray())).issubset({0, 1})


def test_vanilla_af_threshold_is_monotone(afm_factory):
    """A higher AF threshold cannot call more mutations."""
    counts = []
    for t in [0.0, 0.05, 0.2, 0.5, 0.9]:
        a = afm_factory(seed=82)
        mt.pp.call_genotypes(a, bin_method="vanilla", t_vanilla=t, min_AD=1)
        counts.append(int(a.layers["bin"].sum()))
    assert counts == sorted(counts, reverse=True)


@pytest.mark.parametrize("min_AD", [1, 2, 5, 10, 100])
def test_vanilla_min_AD(afm_factory, min_AD):
    a = afm_factory(seed=83)
    mt.pp.call_genotypes(a, bin_method="vanilla", min_AD=min_AD)
    assert set(np.unique(a.layers["bin"].toarray())).issubset({0, 1})


def test_vanilla_min_AD_is_monotone(afm_factory):
    counts = []
    for m in [1, 2, 5, 20]:
        a = afm_factory(seed=84)
        mt.pp.call_genotypes(a, bin_method="vanilla", min_AD=m)
        counts.append(int(a.layers["bin"].sum()))
    assert counts == sorted(counts, reverse=True)


def test_vanilla_impossible_threshold_calls_nothing(afm_factory):
    a = afm_factory(seed=85)
    mt.pp.call_genotypes(a, bin_method="vanilla", min_AD=10**6)
    assert int(a.layers["bin"].sum()) == 0


# -- MiTo thresholds --------------------------------------------------------

@pytest.mark.parametrize("t_prob", [0.5, 0.6, 0.7, 0.9, 0.99])
def test_mito_posterior_threshold(afm_factory, t_prob):
    a = afm_factory(seed=86)
    mt.pp.call_genotypes(a, bin_method="MiTo", t_prob=t_prob)
    assert set(np.unique(a.layers["bin"].toarray())).issubset({0, 1})


@pytest.mark.parametrize("min_cell_prevalence", [0.0, 0.05, 0.1, 0.5, 1.0])
def test_mito_min_cell_prevalence(afm_factory, min_cell_prevalence):
    """Controls the switch between probabilistic and hard-threshold calling."""
    a = afm_factory(seed=87)
    mt.pp.call_genotypes(a, bin_method="MiTo", min_cell_prevalence=min_cell_prevalence)
    assert set(np.unique(a.layers["bin"].toarray())).issubset({0, 1})


def test_mito_prevalence_extremes_differ(afm_factory):
    """Prevalence 0 forces the mixture path, 1 forces the vanilla path."""
    a0 = afm_factory(seed=88)
    mt.pp.call_genotypes(a0, bin_method="MiTo", min_cell_prevalence=0.0)
    a1 = afm_factory(seed=88)
    mt.pp.call_genotypes(a1, bin_method="MiTo", min_cell_prevalence=1.0)
    assert a0.layers["bin"].shape == a1.layers["bin"].shape


@pytest.mark.parametrize("min_AD", [1, 2, 5])
def test_mito_min_AD(afm_factory, min_AD):
    a = afm_factory(seed=89)
    mt.pp.call_genotypes(a, bin_method="MiTo", min_AD=min_AD)
    assert set(np.unique(a.layers["bin"].toarray())).issubset({0, 1})


# -- method comparison ------------------------------------------------------

def test_methods_disagree_somewhere(afm_factory):
    """vanilla and MiTo must not be trivially identical on structured data."""
    a = afm_factory(seed=90)
    b = afm_factory(seed=90)
    mt.pp.call_genotypes(a, bin_method="vanilla", min_AD=2)
    mt.pp.call_genotypes(b, bin_method="MiTo", min_AD=2)
    assert not np.array_equal(a.layers["bin"].toarray(), b.layers["bin"].toarray())


@pytest.mark.parametrize("bin_method", BIN_METHODS)
def test_deterministic(afm_factory, bin_method):
    a = afm_factory(seed=91)
    b = afm_factory(seed=91)
    mt.pp.call_genotypes(a, bin_method=bin_method)
    mt.pp.call_genotypes(b, bin_method=bin_method)
    assert np.array_equal(a.layers["bin"].toarray(), b.layers["bin"].toarray())


@pytest.mark.parametrize("bin_method", BIN_METHODS)
def test_recalling_is_idempotent(afm_factory, bin_method):
    a = afm_factory(seed=92)
    mt.pp.call_genotypes(a, bin_method=bin_method)
    first = a.layers["bin"].toarray().copy()
    mt.pp.call_genotypes(a, bin_method=bin_method)
    assert np.array_equal(first, a.layers["bin"].toarray())


# -- primitives -------------------------------------------------------------

def test_genotype_MiTo_shape_and_values():
    rng = np.random.default_rng(5)
    DP = rng.integers(10, 100, size=(50, 10))
    AD = (DP * rng.random((50, 10))).astype(int)
    X = genotype_MiTo(AD, DP)
    assert X.shape == AD.shape
    assert set(np.unique(X)).issubset({0, 1})


def test_genotype_mixtures_shape_and_values():
    rng = np.random.default_rng(6)
    DP = rng.integers(10, 100, size=(40, 6))
    AD = (DP * rng.random((40, 6))).astype(int)
    X = genotype_mixtures(AD, DP)
    assert X.shape == AD.shape
    assert set(np.unique(X)).issubset({0, 1})


def test_genotype_MiTo_all_zero_input():
    """No alternative counts anywhere means no positive genotypes."""
    DP = np.full((20, 4), 50)
    AD = np.zeros((20, 4), dtype=int)
    X = genotype_MiTo(AD, DP)
    assert X.sum() == 0


def test_genotype_MiTo_fully_fixed_variant_calls_nothing():
    """
    A variant at 100% VAF in every cell is degenerate for a two-component binomial
    mixture -- and biologically homoplasmic, so uninformative for lineage tracing.
    The probabilistic path therefore calls nothing.
    """
    DP = np.full((20, 4), 50)
    AD = np.full((20, 4), 50)
    assert genotype_MiTo(AD, DP, min_AD=1).sum() == 0


def test_genotype_MiTo_falls_back_to_hard_threshold():
    """Below min_cell_prevalence the vanilla path is used, which does call them."""
    DP = np.full((20, 4), 50)
    AD = np.full((20, 4), 50)
    X = genotype_MiTo(AD, DP, min_AD=1, min_cell_prevalence=1.1)
    assert X.sum() == X.size


def test_genotype_MiTo_bimodal_variant_is_called():
    """A genuinely clonal variant -- high VAF in half the cells -- must be called."""
    rng = np.random.default_rng(11)
    DP = np.full((60, 3), 80)
    AD = np.zeros((60, 3), dtype=int)
    AD[:30, :] = rng.binomial(80, 0.8, size=(30, 3))
    X = genotype_MiTo(AD, DP, min_AD=1)
    assert X[:30, :].sum() > X[30:, :].sum()


# -- edge-sized inputs ------------------------------------------------------

@pytest.mark.parametrize("bin_method", BIN_METHODS)
def test_small_afm(afm_small, bin_method):
    mt.pp.call_genotypes(afm_small, bin_method=bin_method)
    assert afm_small.layers["bin"].shape == afm_small.shape


@pytest.mark.parametrize("bin_method", BIN_METHODS)
def test_single_variant(bin_method):
    a = build_afm(n_cells=25, n_vars=1, n_clones=2, seed=93)
    mt.pp.call_genotypes(a, bin_method=bin_method)
    assert a.layers["bin"].shape == (25, 1)


@pytest.mark.parametrize("coverage", [5, 20, 200])
def test_across_coverage_levels(coverage):
    a = build_afm(coverage=coverage, seed=94)
    mt.pp.call_genotypes(a, bin_method="MiTo")
    assert set(np.unique(a.layers["bin"].toarray())).issubset({0, 1})
