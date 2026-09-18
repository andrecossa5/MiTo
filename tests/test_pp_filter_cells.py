"""
mito.pp.filter_cells

Which criterion applies is decided by the coverage metrics the AFM carries, not by the
assay's name, so that a new pre-processing pipeline only has to produce the contract.
"""

import numpy as np
import pytest

import mito as mt

# -- filter1: genome-wide coverage -----------------------------------------


def test_filter1_keeps_well_covered_cells(afm):
    n0 = afm.shape[0]
    mt.pp.filter_cells(afm, cell_filter="filter1", mean_cov_all=1, nmads=100)
    assert afm.shape[0] == n0


def test_filter1_drops_shallow_cells(afm):
    afm.obs.loc[afm.obs_names[:10], "mean_site_coverage"] = 1.0
    mt.pp.filter_cells(afm, cell_filter="filter1", mean_cov_all=20)
    assert afm.shape[0] == 110


def test_filter1_drops_outliers_above_the_mad_threshold(afm):
    """A cell with a huge MT library is as suspect as a shallow one (doublets)."""
    outlier = afm.obs_names[0]
    afm.obs.loc[outlier, "mean_site_coverage"] = 1e6
    mt.pp.filter_cells(afm, cell_filter="filter1", mean_cov_all=1, nmads=3)
    assert outlier not in afm.obs_names


# -- filter2: target panel --------------------------------------------------


def test_filter2_uses_the_panel_metrics(afm):
    afm.obs.loc[afm.obs_names[:15], "frac_target_site_covered"] = 0.1
    mt.pp.filter_cells(afm, cell_filter="filter2", min_perc_covered_sites=0.75)
    assert afm.shape[0] == 105


def test_filter2_explains_itself_when_the_panel_metrics_are_missing(afm_no_quality):
    """
    An assay that covers the MT genome uniformly has no target-panel metrics, so filter2
    is not applicable: that must be a message, not a KeyError.
    """
    with pytest.raises(ValueError, match="filter1"):
        mt.pp.filter_cells(afm_no_quality, cell_filter="filter2")


def test_filter1_works_on_a_genome_wide_assay(afm_no_quality):
    mt.pp.filter_cells(afm_no_quality, cell_filter="filter1", mean_cov_all=1)
    assert afm_no_quality.shape[0] > 0


# -- generic behaviour ------------------------------------------------------


def test_unknown_filter_is_a_no_op_with_a_log(afm):
    n0 = afm.shape[0]
    mt.pp.filter_cells(afm, cell_filter="nonexistent")
    assert afm.shape[0] == n0


def test_cell_subset_is_intersected(afm):
    subset = list(afm.obs_names[:20]) + ["NOT_A_CELL"]
    mt.pp.filter_cells(afm, cell_subset=subset, cell_filter="filter1", mean_cov_all=1)
    assert afm.shape[0] == 20


def test_variants_never_seen_are_dropped(afm):
    afm.X[:, 0] = 0
    afm.X.eliminate_zeros()
    n_vars = afm.shape[1]
    mt.pp.filter_cells(afm, cell_filter="filter1", mean_cov_all=1)
    assert afm.shape[1] == n_vars - 1


def test_modifies_in_place_and_returns_none(afm):
    out = mt.pp.filter_cells(afm, cell_filter="filter2")
    assert out is None
    assert "mito" in afm.uns


def test_copy_leaves_the_input_untouched(afm):
    n0 = afm.shape[0]
    afm.obs.loc[afm.obs_names[:30], "frac_target_site_covered"] = 0.0
    out = mt.pp.filter_cells(afm, cell_filter="filter2", copy=True)
    assert out is not afm
    assert afm.shape[0] == n0 and out.shape[0] == n0 - 30
    assert "mito" not in afm.uns


def test_records_what_it_did(afm):
    mt.pp.filter_cells(afm, cell_filter="filter2")
    record = afm.uns["mito"]["filter_cells"]
    assert record["cell_filter"] == "filter2"
    assert record["n_cells_in"] == 120
    assert record["n_cells_out"] == afm.shape[0]
    assert np.isclose(record["median_cov_target"], 25)
