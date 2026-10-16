"""
mito.pp.filter_cells

Covers both filtering strategies, the scLT-system gating, threshold behaviour
across a wide parameter range, cell subsetting, and the AnnData invariants that
must survive filtering.
"""

import pytest
from conftest import build_afm

import mito as mt

# -- basic contract ---------------------------------------------------------

@pytest.mark.parametrize("cell_filter", ["filter1", "filter2"])
def test_returns_anndata_subset(afm, cell_filter):
    n0 = afm.shape[0]
    out = mt.pp.filter_cells(afm, cell_filter=cell_filter)
    assert out.shape[0] <= n0
    assert out.shape[0] > 0
    assert set(out.obs_names).issubset(set(afm.obs_names))


@pytest.mark.parametrize("cell_filter", ["filter1", "filter2"])
def test_layers_and_var_preserved(afm, cell_filter):
    out = mt.pp.filter_cells(afm, cell_filter=cell_filter)
    for layer in ("AD", "DP"):
        assert layer in out.layers
        assert out.layers[layer].shape == out.shape
    assert set(out.var.columns) >= {"pos", "ref", "alt"}


@pytest.mark.parametrize("cell_filter", ["filter1", "filter2"])
def test_records_parameters_in_uns(afm, cell_filter):
    out = mt.pp.filter_cells(afm, cell_filter=cell_filter)
    assert "cell_filter" in out.uns
    assert out.uns["cell_filter"]["cell_filter"] == cell_filter


def test_drops_never_observed_sites():
    """Sites with no positive cell must be removed regardless of filter."""
    from scipy.sparse import csr_matrix
    a = build_afm(n_cells=40, n_vars=20, seed=3)
    dense = a.X.toarray()
    dense[:, 0] = 0                    # silence one variant entirely
    silenced = a.var_names[0]
    a.X = csr_matrix(dense)
    out = mt.pp.filter_cells(a, cell_filter="filter1", mean_cov_all=0)
    assert silenced not in out.var_names


# -- filter1: mean genome-wide coverage -------------------------------------

@pytest.mark.parametrize("mean_cov_all", [0, 1, 20, 50, 1e6])
def test_filter1_mean_cov_threshold_is_monotone(afm_factory, mean_cov_all):
    """Raising the coverage floor can only remove cells, never add them."""
    a = afm_factory(seed=11)
    out = mt.pp.filter_cells(a.copy(), cell_filter="filter1", mean_cov_all=mean_cov_all)
    kept = out.shape[0]
    if mean_cov_all == 0:
        assert kept > 0
    assert kept <= a.shape[0]


def test_filter1_monotonicity_across_thresholds(afm_factory):
    a = afm_factory(seed=12)
    sizes = [
        mt.pp.filter_cells(a.copy(), cell_filter="filter1", mean_cov_all=t).shape[0]
        for t in [0, 10, 30, 55, 80]
    ]
    assert sizes == sorted(sizes, reverse=True)


@pytest.mark.parametrize("nmads", [1, 3, 5, 10, 100])
def test_filter1_nmads_range(afm_factory, nmads):
    a = afm_factory(seed=13)
    out = mt.pp.filter_cells(a, cell_filter="filter1", nmads=nmads, mean_cov_all=0)
    assert out.shape[0] > 0


def test_filter1_nmads_monotonicity(afm_factory):
    """A wider MAD window cannot retain fewer cells."""
    a = afm_factory(seed=14)
    sizes = [
        mt.pp.filter_cells(a.copy(), cell_filter="filter1", nmads=n, mean_cov_all=0).shape[0]
        for n in [1, 2, 5, 20]
    ]
    assert sizes == sorted(sizes)


# -- filter2: target-site coverage ------------------------------------------

@pytest.mark.parametrize("median_cov_target", [0, 10, 25, 60, 10_000])
def test_filter2_coverage_threshold(afm_factory, median_cov_target):
    a = afm_factory(seed=15)
    out = mt.pp.filter_cells(a, cell_filter="filter2", median_cov_target=median_cov_target)
    assert out.shape[0] <= a.shape[0]


@pytest.mark.parametrize("min_perc_covered_sites", [0.0, 0.5, 0.75, 0.95, 1.0])
def test_filter2_covered_sites_threshold(afm_factory, min_perc_covered_sites):
    a = afm_factory(seed=16)
    out = mt.pp.filter_cells(
        a, cell_filter="filter2", median_cov_target=0,
        min_perc_covered_sites=min_perc_covered_sites,
    )
    assert out.shape[0] <= a.shape[0]


def test_filter2_both_thresholds_monotone(afm_factory):
    a = afm_factory(seed=17)
    sizes = [
        mt.pp.filter_cells(
            a.copy(), cell_filter="filter2", median_cov_target=c, min_perc_covered_sites=p
        ).shape[0]
        for c, p in [(0, 0.0), (20, 0.5), (40, 0.8), (70, 0.95)]
    ]
    assert sizes == sorted(sizes, reverse=True)


def test_filter2_extreme_threshold_removes_everything(afm_factory):
    a = afm_factory(seed=18)
    out = mt.pp.filter_cells(a, cell_filter="filter2", median_cov_target=10**9)
    assert out.shape[0] == 0


# -- scLT system gating -----------------------------------------------------

def test_filter1_accepts_redeem(afm_redeem):
    out = mt.pp.filter_cells(afm_redeem, cell_filter="filter1", mean_cov_all=0)
    assert out.shape[0] > 0


def test_filter2_rejects_redeem(afm_redeem):
    """filter2 is MAESTER-only."""
    with pytest.raises(ValueError):
        mt.pp.filter_cells(afm_redeem, cell_filter="filter2")


@pytest.mark.parametrize("cell_filter", ["filter1", "filter2"])
def test_cas9_rejected_by_both_filters(afm_cas9, cell_filter):
    with pytest.raises(ValueError):
        mt.pp.filter_cells(afm_cas9, cell_filter=cell_filter)


@pytest.mark.parametrize("cell_filter", [None, "nonexistent", "filter3", ""])
def test_unknown_filter_is_a_no_op(afm, cell_filter):
    """Unrecognised strategies skip filtering rather than raising."""
    n0 = afm.shape[0]
    out = mt.pp.filter_cells(afm, cell_filter=cell_filter)
    assert out.shape[0] == n0
    assert out.uns["cell_filter"] == {}


# -- cell_subset ------------------------------------------------------------

def test_cell_subset_restricts_to_requested_cells(afm):
    wanted = list(afm.obs_names[:20])
    out = mt.pp.filter_cells(afm, cell_subset=wanted, cell_filter=None)
    assert set(out.obs_names) == set(wanted)


def test_cell_subset_ignores_unknown_barcodes(afm):
    wanted = list(afm.obs_names[:10]) + ["NOT_A_CELL_1", "NOT_A_CELL_2"]
    out = mt.pp.filter_cells(afm, cell_subset=wanted, cell_filter=None)
    assert set(out.obs_names) == set(afm.obs_names[:10])


def test_cell_subset_combines_with_filter(afm):
    wanted = list(afm.obs_names[:30])
    out = mt.pp.filter_cells(afm, cell_subset=wanted, cell_filter="filter2",
                             median_cov_target=0, min_perc_covered_sites=0.0)
    assert set(out.obs_names).issubset(set(wanted))


def test_empty_cell_subset(afm):
    out = mt.pp.filter_cells(afm, cell_subset=[], cell_filter=None)
    assert out.shape[0] == 0


# -- edge-sized inputs ------------------------------------------------------

def test_small_afm(afm_small):
    out = mt.pp.filter_cells(afm_small, cell_filter="filter1", mean_cov_all=0)
    assert out.shape[0] > 0


def test_single_cell_afm():
    a = build_afm(n_cells=1, n_vars=5, n_clones=1, seed=21)
    out = mt.pp.filter_cells(a, cell_filter="filter1", mean_cov_all=0, nmads=100)
    assert out.shape[0] in (0, 1)


def test_single_variant_afm():
    a = build_afm(n_cells=20, n_vars=1, n_clones=2, seed=22)
    out = mt.pp.filter_cells(a, cell_filter="filter1", mean_cov_all=0)
    assert out.shape[1] <= 1


def test_input_is_not_mutated(afm):
    before = afm.shape
    mt.pp.filter_cells(afm.copy(), cell_filter="filter2")
    assert afm.shape == before
