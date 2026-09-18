"""
The public API: what MiTo exports, and the one-screen workflow the documentation promises.

A rename that forgets ``__all__``, or a module that only imports because something else
imported it first, shows up here.
"""

import importlib

import numpy as np
import pandas as pd
import pytest

import mito as mt
from conftest import build_afm

MODULES = ["io", "pp", "tl", "pl", "ut"]

EXPORTS = {
    "io": ["make_afm", "read_coverage", "read_newick", "write_newick"],
    "pp": [
        "filter_cells", "filter_afm",
        "annotate_vars", "filter_low_quality_variants", "filter_candidate_variants",
        "filter_known_artefacts", "call_genotypes", "filter_non_clonal_variants",
        "filter_low_signal_variants", "impute_dropouts", "filter_incompatible_variants",
        "compute_distances", "kNN_graph", "reduce_dimensions",
        "filter_small_clones", "compute_lineage_biases", "select_gt_enriched_variants",
    ],
    "tl": [
        "build_tree", "coarse_grained_tree", "AFM_to_seqs",
        "annotate_clones", "evidence_cut", "clone_support", "rescue_unassigned",
        "compute_clonal_fate_bias", "compute_scPlasticity", "compute_fitness",
        "compute_expansions", "bootstrap_MiTo", "bootstrap_bin", "leiden_clustering",
    ],
    "pl": [
        "plot_tree", "draw_embedding", "heatmap_distances", "heatmap_variants",
        "vars_AF_spectrum", "plot_ncells_nAD", "mut_profile", "packed_circle_plot",
        "MT_coverage_polar", "MT_coverage_by_gene_polar",
    ],
    "ut": [
        "simulate_afm", "dataset_metrics", "custom_ARI", "normalized_mutual_info_score",
        "kbet", "CI", "RI", "distance_AUPRC", "NN_entropy", "NN_purity",
        "calculate_corr_distances", "mask_mt_sites", "get_clades",
        "get_internal_node_stats", "extract_kwargs", "Timer",
    ],
}


# -- exports ----------------------------------------------------------------


@pytest.mark.parametrize("module", MODULES)
def test_module_is_importable_on_its_own(module):
    importlib.import_module(f"mito.{module}")


@pytest.mark.parametrize("module", MODULES)
def test_everything_in_all_exists(module):
    mod = getattr(mt, module)
    missing = [name for name in mod.__all__ if not hasattr(mod, name)]
    assert not missing


@pytest.mark.parametrize("module,names", EXPORTS.items())
def test_the_documented_api_is_exported(module, names):
    mod = getattr(mt, module)
    missing = [name for name in names if name not in mod.__all__]
    assert not missing


@pytest.mark.parametrize("gone", [
    "filter_MiTo", "filter_MQuad", "filter_CV", "filter_miller2022", "filter_weng2024",
    "filter_variant_moransI", "genotype_mixtures", "nans_as_zeros",
])
def test_retired_functions_are_gone(gone):
    """MT-only scope: the superseded feature-selection strategies were removed in 0.3."""
    assert not hasattr(mt.pp, gone)


def test_retired_annotator_is_gone():
    assert not hasattr(mt.tl, "MiToTreeAnnotator")


def test_version_is_exposed():
    assert isinstance(mt.__version__, str) and mt.__version__.count(".") >= 1


# -- the documented workflow ------------------------------------------------


def test_the_getting_started_workflow():
    """
    The four steps of the tutorial, on synthetic data: cells, variants, tree, clones.
    """
    afm = build_afm()

    mt.pp.filter_cells(afm, cell_filter="filter2")
    mt.pp.filter_afm(afm, ncores=1)
    tree = mt.tl.build_tree(afm, solver="UPMGA")
    mt.tl.annotate_clones(tree, afm)

    labels = afm.obs["MiTo_clone"].astype(str)
    assigned = labels != "unassigned"
    assert mt.ut.custom_ARI(afm.obs["GBC"].astype(str)[assigned], labels[assigned]) > 0.9
    mt.pl.plot_tree(tree, annot="MiTo_clone")


def test_provenance_records_every_step(afm_filtered):
    """One namespace holds the whole history of the object."""
    record = afm_filtered.uns["mito"]
    assert "version" in record
    assert {"filter_cells", "filter_afm"} <= set(record)

    mt.pp.reduce_dimensions(afm_filtered, method="PCA", ncores=1)
    assert "reduce_dimensions" in afm_filtered.uns["mito"]


def test_uns_stays_small(afm_filtered):
    """
    The object carries what functions read plus one provenance namespace - not a dump of
    every intermediate statistic.
    """
    assert set(afm_filtered.uns) == {"scLT_system", "pp_method", "distances", "mito"}


# -- utilities used by nf-MiTo ----------------------------------------------


def test_extract_kwargs_only_returns_arguments_the_functions_accept():
    out = mt.ut.extract_kwargs({
        "cell_filter": "filter2", "qc_alpha": 0.01, "impute": True, "solver": "UPMGA",
        "filtering": "MQuad",            # retired: must be dropped, not forwarded
        "nonsense": 1,
    })
    assert out["filter_cells"] == {"cell_filter": "filter2"}
    assert out["filter_afm"] == {"qc_alpha": 0.01, "impute": True}
    assert out["build_tree"] == {"solver": "UPMGA"}
    assert "filtering" not in out["filter_afm"] and "nonsense" not in out["filter_afm"]


def test_dataset_metrics_describes_a_filtered_afm(afm_filtered):
    metrics = mt.ut.dataset_metrics(afm_filtered)
    assert isinstance(metrics, pd.Series)
    assert metrics["n_cells"] == afm_filtered.shape[0]
    assert metrics["n_vars"] == afm_filtered.shape[1]
    assert 0 < metrics["density"] <= 1
    assert "transitions_vs_transversions_ratio" in metrics.index


def test_dataset_metrics_needs_genotypes(afm):
    with pytest.raises(ValueError, match="bin"):
        mt.ut.dataset_metrics(afm)


def test_metrics_agree_with_themselves(annotated):
    afm, _ = annotated
    truth = afm.obs["GBC"].astype(str)
    labels = afm.obs["MiTo_clone"].astype(str)
    assert mt.ut.custom_ARI(truth, truth) == pytest.approx(1.0)
    assert mt.ut.normalized_mutual_info_score(truth, truth) == pytest.approx(1.0)
    assert 0 <= mt.ut.custom_ARI(truth, labels) <= 1


def test_newick_round_trip(tmp_path, annotated_tree):
    path = tmp_path / "tree.newick"
    mt.io.write_newick(annotated_tree, str(path))
    back = mt.io.read_newick(str(path))
    assert set(back.leaves) == set(annotated_tree.leaves)


def test_afm_to_seqs_needs_genotypes(afm):
    with pytest.raises(ValueError, match="call_genotypes"):
        mt.tl.AFM_to_seqs(afm)


def test_afm_to_seqs_returns_one_sequence_per_cell(afm_filtered):
    seqs = mt.tl.AFM_to_seqs(afm_filtered)
    assert len(seqs) == afm_filtered.shape[0]
    assert all(len(s) == afm_filtered.shape[1] for s in seqs.values())


def test_mask_mt_sites_is_a_boolean_mask():
    """Regression: an empty list used to give a float array, which AnnData rejects."""
    assert mt.ut.mask_mt_sites([3300, 1]).dtype == np.bool_
    assert mt.ut.mask_mt_sites([]).dtype == np.bool_
