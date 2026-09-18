"""
mito.io.make_afm

The readers are the only place where MiTo meets another tool's file format, so these
tests are written against the layouts maegatk, mgatk and RedeemV actually produce, and
they check the object contract every downstream function relies on.
"""

import numpy as np
import pandas as pd
import pytest
from scipy.sparse import issparse

import mito as mt

# -- the contract -----------------------------------------------------------


def _assert_contract(afm):
    """Everything mito.pp assumes about a freshly built AFM."""

    assert issparse(afm.X) and afm.X.dtype == np.float32
    assert issparse(afm.layers["AD"])
    # DP is dense on purpose: it is non-zero almost everywhere, so CSR would cost more
    assert not issparse(afm.layers["DP"])
    assert np.issubdtype(afm.layers["DP"].dtype, np.integer)

    AD = afm.layers["AD"].toarray()
    DP = np.asarray(afm.layers["DP"])
    assert (AD <= DP).all(), "alt counts cannot exceed the site's coverage"
    assert np.allclose(afm.X.toarray(), AD / np.maximum(DP, 1), atol=1e-6)

    # coverage is defined for every cell, not only for those with an alternative read:
    # "covered, no alt read" is evidence of absence and must be distinguishable from
    # "not covered"
    assert (DP > 0).mean() > 0.5
    assert ((DP > 0) & (AD == 0)).any()

    for column in ("pos", "ref", "alt"):
        assert column in afm.var.columns
    assert afm.var["pos"].is_monotonic_increasing
    assert afm.var_names.to_series().str.match(r"^\d+_[ACGT]>[ACGT]$").all()
    assert afm.var["pos"].value_counts().max() == 1, "multi-allelic sites must be dropped"

    assert "mean_site_coverage" in afm.obs.columns
    assert {"scLT_system", "pp_method"} <= set(afm.uns)


def test_maegatk_tables_build_a_valid_afm(maegatk_tables):
    folder, meta = maegatk_tables
    afm = mt.io.make_afm(folder, scLT_system="MAESTER", pp_method="maegatk")
    _assert_contract(afm)
    assert afm.shape[0] == len(meta["cells"])
    assert "quality" in afm.var.columns, "maegatk tables carry base qualities"


def test_mgatk_tables_build_a_valid_afm(mgatk_tables):
    """mgatk in 10x mode: a final/ folder, sample-prefixed names, 4-column tables."""
    folder, meta = mgatk_tables
    afm = mt.io.make_afm(folder, scLT_system="mtscATAC", pp_method="mgatk")
    _assert_contract(afm)
    assert afm.shape[0] == len(meta["cells"])
    assert "quality" not in afm.var.columns, "mgatk 10x tables carry no base qualities"


def test_redeem_tables_build_a_valid_afm(redeem_tables):
    folder, meta = redeem_tables
    afm = mt.io.make_afm(folder, scLT_system="ReDeeM", pp_method="redeem-v")
    _assert_contract(afm)
    assert afm.shape[0] <= len(meta["cells"])
    assert afm.shape[1] > 0


# -- the differences between assays -----------------------------------------


def test_target_panel_metrics_only_for_targeted_assays(maegatk_tables, mgatk_tables):
    """
    The cell filters choose their criterion from the metrics present, so a targeted assay
    must get the panel columns and a genome-wide one must not.
    """
    targeted = mt.io.make_afm(maegatk_tables[0], scLT_system="MAESTER", pp_method="maegatk")
    genomewide = mt.io.make_afm(mgatk_tables[0], scLT_system="mtscATAC", pp_method="mgatk")

    panel = {"median_target_site_coverage", "frac_target_site_covered"}
    assert panel <= set(targeted.obs.columns)
    assert not (panel & set(genomewide.obs.columns))


def test_reference_alleles_of_the_run_are_preferred(mgatk_tables):
    """
    mgatk writes the reference it was run against (it ships a mouse MT genome too), so a
    refAllele table in the output must win over the bundled rCRS.
    """
    folder, meta = mgatk_tables
    afm = mt.io.make_afm(folder, scLT_system="mtscATAC", pp_method="mgatk")
    for name in afm.var_names:
        pos, sub = name.split("_")
        assert sub.split(">")[0] == meta["ref"][int(pos)]


def test_sample_name_is_appended_to_barcodes(maegatk_tables):
    folder, _ = maegatk_tables
    afm = mt.io.make_afm(folder, scLT_system="MAESTER", pp_method="maegatk", sample="s1")
    assert afm.obs_names.str.endswith("_s1").all()


def test_uncompressed_tables_are_read(tmp_path):
    from conftest import _write_allelic_tables

    folder = tmp_path / "plain"
    _write_allelic_tables(folder, compress=False)
    afm = mt.io.make_afm(folder, scLT_system="MAESTER", pp_method="maegatk")
    assert afm.shape[1] > 0


# -- cell metadata ----------------------------------------------------------


def test_metadata_restricts_the_cells(maegatk_tables, tmp_path):
    folder, meta = maegatk_tables
    keep = meta["cells"][:20]
    path_meta = tmp_path / "meta.csv"
    pd.DataFrame({"label": ["x"] * len(keep)}, index=keep).to_csv(path_meta)

    afm = mt.io.make_afm(folder, path_meta=str(path_meta), scLT_system="MAESTER",
                         pp_method="maegatk")
    assert list(afm.obs_names) == keep
    assert "label" in afm.obs.columns


def test_metadata_that_matches_nothing_explains_the_cell_name_convention(maegatk_tables, tmp_path):
    folder, _ = maegatk_tables
    path_meta = tmp_path / "meta.csv"
    pd.DataFrame({"label": ["x"]}, index=["NOT_A_BARCODE"]).to_csv(path_meta)

    with pytest.raises(ValueError, match="sample="):
        mt.io.make_afm(folder, path_meta=str(path_meta), scLT_system="MAESTER",
                       pp_method="maegatk")


# -- refusals ---------------------------------------------------------------


def test_unknown_scLT_system_is_refused(maegatk_tables):
    with pytest.raises(ValueError, match="scLT_system"):
        mt.io.make_afm(maegatk_tables[0], scLT_system="Cas9", pp_method="maegatk")


def test_pp_method_must_match_the_assay(maegatk_tables):
    with pytest.raises(ValueError, match="pp_method"):
        mt.io.make_afm(maegatk_tables[0], scLT_system="MAESTER", pp_method="redeem-v")


def test_missing_path_is_refused(tmp_path):
    with pytest.raises(ValueError, match="path_ch_matrix"):
        mt.io.make_afm(str(tmp_path / "nope"))


def test_missing_tables_name_what_is_expected(tmp_path):
    (tmp_path / "empty").mkdir()
    with pytest.raises(ValueError, match=r"A\.txt"):
        mt.io.make_afm(tmp_path / "empty", scLT_system="MAESTER", pp_method="maegatk")


def test_redeem_threshold_is_validated(redeem_tables):
    with pytest.raises(ValueError, match="treshold"):
        mt.io.make_afm(redeem_tables[0], scLT_system="ReDeeM", pp_method="redeem-v",
                       treshold="Stringent")          # RedeemV calls this level "Specific"


# -- the AFM feeds the pipeline ---------------------------------------------


@pytest.mark.parametrize("fixture,system,method,cell_filter", [
    ("maegatk_tables", "MAESTER", "maegatk", "filter2"),
    ("mgatk_tables", "mtscATAC", "mgatk", "filter1"),
    ("redeem_tables", "ReDeeM", "redeem-v", "filter1"),
])
def test_every_reader_output_runs_the_pipeline(request, fixture, system, method, cell_filter):
    folder, _ = request.getfixturevalue(fixture)
    afm = mt.io.make_afm(folder, scLT_system=system, pp_method=method)
    mt.pp.filter_cells(afm, cell_filter=cell_filter, mean_cov_all=5)
    mt.pp.filter_afm(afm, cand_min_n_positive=3, cand_min_DP_in_positives=5,
                     qc_alpha=0.1, min_snr=2, ncores=1)
    tree = mt.tl.build_tree(afm, solver="UPMGA")
    mt.tl.annotate_clones(tree, afm, min_cells=3)
    assert afm.obs["MiTo_clone"].nunique() >= 2
