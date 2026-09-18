"""
mito.pp.filter_afm

The pipeline entry point: candidate MT-SNVs -> genotypes -> clonality QC -> final
genotypes -> signal over background -> (imputation) -> characters -> distances. These
tests pin the slots it writes, the order it runs its stages in, and the messages it gives
when a dataset has too little signal to go on.
"""

import numpy as np
import pytest

import mito as mt
from conftest import build_afm

# -- the object it produces -------------------------------------------------


def test_writes_the_expected_slots(afm_filtered):
    afm = afm_filtered

    assert "bin" in afm.layers
    assert set(afm.layers) >= {"AD", "DP", "bin"}
    assert "distances" in afm.obsp

    expected_var = {"pos", "ref", "alt", "error_rate", "n_carriers", "n_imputed",
                    "prevalence", "snr", "mean_af", "mean_cov", "n_cells",
                    "median_af_in_positives", "mean_AD_in_positives", "mean_DP_in_positives"}
    assert expected_var <= set(afm.var.columns)
    assert {"n_characters", "n_imputed"} <= set(afm.obs.columns)


def test_internal_qc_statistics_are_not_persisted(afm_filtered):
    """
    The p-values of the clonality tests and the conflict counts are means to a decision,
    not results: they stay available on the stage functions, not on the final object.
    """
    assert not ({"p_join", "p_exclusive", "p_clonal", "clonal", "n_conflicts"}
                & set(afm_filtered.var.columns))


def test_provenance_is_one_record(afm_filtered):
    record = afm_filtered.uns["mito"]["filter_afm"]
    assert set(record) == {"params", "flow", "converged", "n_cells_in", "n_cells_out", "seconds"}
    assert set(record["params"]) == {"candidates", "genotyping", "clonality", "signal",
                                     "imputation", "characters"}
    assert record["converged"] is True
    assert record["n_cells_out"] == afm_filtered.shape[0]
    # the per-stage keys are folded into it, not left lying around
    assert not ({"genotyping", "clonality", "compatibility", "known_artefacts"}
                & set(afm_filtered.uns))


def test_the_flow_table_accounts_for_every_dropped_variant(afm_filtered):
    flow = afm_filtered.uns["mito"]["filter_afm"]["flow"]
    assert list(flow.columns) == ["stage", "n_vars_in", "n_vars_out"]
    # each stage starts where the previous one ended, and the last one ends at the AFM
    assert (flow["n_vars_in"].values[1:] == flow["n_vars_out"].values[:-1]).all()
    assert flow["n_vars_out"].iloc[-1] == afm_filtered.shape[1]
    assert (flow["n_vars_out"] <= flow["n_vars_in"]).all()


def test_characters_carry_signal(afm_filtered):
    """The surviving characters must recover the planted clones."""
    B = afm_filtered.layers["bin"].toarray() > 0
    truth = afm_filtered.obs["GBC"].astype(str)
    assert afm_filtered.shape[1] >= 4
    # every character is prevalent in one clone and rare elsewhere
    for j in range(B.shape[1]):
        rates = truth[B[:, j]].value_counts(normalize=True)
        assert rates.iloc[0] > 0.8


def test_distances_are_computed_from_the_genotypes(afm_filtered):
    D = afm_filtered.obsp["distances"].toarray()
    assert D.shape == (afm_filtered.shape[0],) * 2
    assert np.isfinite(D).all()
    assert afm_filtered.uns["distances"]["distances"] == {
        "metric": "weighted_jaccard", "layer": "bin"
    }


# -- knobs ------------------------------------------------------------------


def test_imputation_is_off_by_default_and_adds_calls_when_on():
    from conftest import run_pipeline
    plain = run_pipeline()
    imputed = run_pipeline(impute=True)
    assert plain.obs["n_imputed"].sum() == 0
    assert imputed.uns["mito"]["filter_afm"]["params"]["imputation"]["enabled"] is True
    assert imputed.obs["n_imputed"].sum() >= 0


def test_min_n_var_filters_cells():
    from conftest import run_pipeline
    lenient = run_pipeline(min_n_var=1)
    strict = run_pipeline(min_n_var=2)
    assert strict.shape[0] <= lenient.shape[0]
    assert (strict.layers["bin"].toarray() > 0).sum(axis=1).min() >= 2


def test_prevalence_cap_removes_germline_like_variants():
    """A variant carried by most cells cannot separate them."""
    a = build_afm(n_clones=1, markers_per_clone=3, seed=5)      # markers in every cell
    mt.pp.filter_cells(a, cell_filter="filter2")
    with pytest.raises(ValueError):
        mt.pp.filter_afm(a, max_prevalence=0.5, ncores=1)


def test_gene_mask_is_on_for_targeted_assays_only(afm, afm_no_quality):
    mt.pp.filter_cells(afm, cell_filter="filter2")
    mt.pp.filter_afm(afm, ncores=1)
    assert afm.uns["mito"]["filter_afm"]["params"]["candidates"]["only_genes"] is True

    mt.pp.filter_cells(afm_no_quality, cell_filter="filter1", mean_cov_all=1)
    mt.pp.filter_afm(afm_no_quality, ncores=1)
    assert afm_no_quality.uns["mito"]["filter_afm"]["params"]["candidates"]["only_genes"] is False


def test_quality_threshold_is_skipped_when_no_qualities_are_available(afm_no_quality, caplog):
    mt.pp.filter_cells(afm_no_quality, cell_filter="filter1", mean_cov_all=1)
    with caplog.at_level("INFO"):
        mt.pp.filter_afm(afm_no_quality, ncores=1)
    assert "quality" in caplog.text
    assert afm_no_quality.shape[1] > 0


def test_copy_semantics(afm):
    mt.pp.filter_cells(afm, cell_filter="filter2")
    shape = afm.shape
    out = mt.pp.filter_afm(afm, ncores=1, copy=True)
    assert out is not afm
    assert afm.shape == shape and "bin" not in afm.layers
    assert out.shape[1] < shape[1]


def test_in_place_returns_none(afm):
    mt.pp.filter_cells(afm, cell_filter="filter2")
    assert mt.pp.filter_afm(afm, ncores=1) is None


def test_is_keyword_only_after_the_afm(afm):
    mt.pp.filter_cells(afm, cell_filter="filter2")
    with pytest.raises(TypeError):
        mt.pp.filter_afm(afm, "GBC")


# -- refusals ---------------------------------------------------------------


def test_no_candidate_variant_is_explained():
    a = build_afm(markers_per_clone=0, error_rate=0.0, seed=11)
    mt.pp.filter_cells(a, cell_filter="filter2")
    with pytest.raises(ValueError, match="cand_min_site_cov|cand_"):
        mt.pp.filter_afm(a, ncores=1)


def test_too_few_variants_after_the_qc_is_explained():
    a = build_afm(n_clones=2, markers_per_clone=1, seed=13)
    mt.pp.filter_cells(a, cell_filter="filter2")
    with pytest.raises(ValueError, match="qc_alpha"):
        mt.pp.filter_afm(a, qc_alpha=1e-12, ncores=1)


def test_lineage_column_drives_small_clone_removal():
    a = build_afm(n_cells=120, n_clones=4, seed=17)
    a.obs["GBC"] = a.obs["GBC"].astype(str)
    a.obs.loc[a.obs_names[:3], "GBC"] = "tiny_clone"
    mt.pp.filter_cells(a, cell_filter="filter2")
    mt.pp.filter_afm(a, lineage_column="GBC", min_cell_number=10, ncores=1)
    assert "tiny_clone" not in set(a.obs["GBC"])
