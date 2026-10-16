"""
Smoke coverage for the rest of the public API.

The core preprocessing, tree-building and tree-plotting functions have dedicated
suites. Everything else is checked here: that it is importable, callable on
well-formed input, and returns the documented kind of object.
"""

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

import mito as mt

SUBMODULES = ["io", "pp", "tl", "pl", "ut"]


def _ax():
    fig, ax = plt.subplots(figsize=(4, 4))
    return ax


# -- package surface --------------------------------------------------------

def test_version_is_a_string():
    assert isinstance(mt.__version__, str)
    assert mt.__version__.count(".") >= 1


@pytest.mark.parametrize("name", SUBMODULES)
def test_submodules_are_exposed(name):
    assert hasattr(mt, name)
    assert name in mt.__all__


def test_ut_exports_all_resolve():
    missing = [n for n in mt.ut.__all__ if not hasattr(mt.ut, n)]
    assert not missing, f"mt.ut.__all__ lists names that do not exist: {missing}"


@pytest.mark.parametrize("submodule,expected", [
    ("io", ["make_afm", "read_coverage", "read_newick", "write_newick"]),
    ("pp", ["annotate_vars", "call_genotypes", "compute_distances", "filter_afm",
            "filter_cells", "kNN_graph", "reduce_dimensions"]),
    ("tl", ["build_tree", "MiToTreeAnnotator", "leiden_clustering"]),
    ("pl", ["plot_tree", "heatmap_distances", "heatmap_variants", "draw_embedding"]),
])
def test_expected_names_present(submodule, expected):
    mod = getattr(mt, submodule)
    for name in expected:
        assert hasattr(mod, name), f"mt.{submodule}.{name} is missing"


# -- io ---------------------------------------------------------------------

def test_newick_roundtrip(tree, tmp_path):
    path = tmp_path / "tree.newick"
    mt.io.write_newick(tree, str(path))
    assert path.exists() and path.stat().st_size > 0
    reloaded = mt.io.read_newick(str(path))
    assert reloaded is not None


def test_make_afm_rejects_a_missing_path():
    with pytest.raises(ValueError):
        mt.io.make_afm("/definitely/not/a/real/path")


# -- pp ---------------------------------------------------------------------

def test_filter_baseline(afm_annotated):
    out = mt.pp.filter_baseline(afm_annotated)
    assert out.shape[1] <= afm_annotated.shape[1]


def test_filter_cell_clones(afm):
    out = mt.pp.filter_cell_clones(afm, column="GBC", min_cell_number=5)
    assert out.shape[0] <= afm.shape[0]


def test_compute_distances(afm_filtered):
    mt.pp.compute_distances(afm_filtered, metric="weighted_jaccard", ncores=1)
    D = afm_filtered.obsp["distances"].toarray()
    assert np.allclose(D, D.T, atol=1e-6)


def test_compute_lineage_biases(afm_filtered):
    out = mt.pp.compute_lineage_biases(
        afm_filtered, lineage_column="GBC",
        target_lineage=str(afm_filtered.obs["GBC"].iloc[0]),
    )
    assert out is not None


def test_filter_MiTo(afm_annotated):
    out = mt.pp.filter_MiTo(mt.pp.filter_baseline(afm_annotated))
    assert out.shape[1] <= afm_annotated.shape[1]


# -- tl ---------------------------------------------------------------------

def test_leiden_clustering(afm_filtered):
    mt.pp.reduce_dimensions(afm_filtered, method="UMAP", ncores=1)
    labels = mt.tl.leiden_clustering(afm_filtered.obsp["distances"], res=0.5)
    assert len(labels) == afm_filtered.shape[0]


def test_AFM_to_seqs(afm_filtered):
    seqs = mt.tl.AFM_to_seqs(afm_filtered)
    assert len(seqs) == afm_filtered.shape[0]


def test_bootstrap_bin(afm_filtered):
    out = mt.tl.bootstrap_bin(afm_filtered)
    assert out is not None


def test_bootstrap_MiTo(afm_filtered):
    out = mt.tl.bootstrap_MiTo(afm_filtered)
    assert out is not None


def test_compute_clonal_fate_bias(annotated_tree):
    """Takes a CassiopeiaTree, not an AFM."""
    target = str(annotated_tree.cell_meta["GBC"].iloc[0])
    df = mt.tl.compute_clonal_fate_bias(
        annotated_tree, state_column="GBC", clone_column="MiTo clone",
        target_state=target,
    )
    assert df is not None


# -- pl ---------------------------------------------------------------------

def test_heatmap_distances(afm_filtered):
    ax = _ax()
    mt.pl.heatmap_distances(afm_filtered, ax=ax)
    assert len(ax.collections) + len(ax.images) > 0


def test_heatmap_variants(afm_filtered):
    ax = _ax()
    mt.pl.heatmap_variants(afm_filtered, ax=ax)
    assert len(ax.collections) + len(ax.images) > 0


def test_draw_embedding(afm_filtered):
    mt.pp.reduce_dimensions(afm_filtered, method="UMAP", ncores=1)
    ax = _ax()
    mt.pl.draw_embedding(afm_filtered, ax=ax)
    assert len(ax.collections) + len(ax.lines) > 0


def test_vars_AF_spectrum(afm_filtered):
    ax = _ax()
    mt.pl.vars_AF_spectrum(afm_filtered, ax=ax)
    assert len(ax.lines) + len(ax.collections) > 0


def test_plot_ncells_nAD(afm_annotated):
    ax = _ax()
    mt.pl.plot_ncells_nAD(afm_annotated, ax=ax)
    assert len(ax.collections) + len(ax.lines) > 0


def test_mut_profile(afm_annotated):
    """Builds and returns its own Figure."""
    fig = mt.pl.mut_profile(afm_annotated.var_names.to_list())
    assert isinstance(fig, matplotlib.figure.Figure)
    plt.close(fig)


def test_MT_coverage_polar(afm):
    """Takes the per-position coverage table, not the AFM."""
    fig, ax = plt.subplots(subplot_kw={"projection": "polar"})
    mt.pl.MT_coverage_polar(afm.uns["per_position_coverage"], ax=ax)
    assert len(ax.lines) + len(ax.collections) > 0
    plt.close(fig)


def test_packed_circle_plot(afm_filtered):
    """Takes a DataFrame plus the column to size circles by."""
    ax = _ax()
    df = afm_filtered.obs["GBC"].value_counts().to_frame("n")
    mt.pl.packed_circle_plot(df, ax=ax, covariate="n")
    assert len(ax.patches) + len(ax.collections) > 0


# -- ut ---------------------------------------------------------------------

def test_metrics_on_a_tree(tree):
    """CI and RI are reported per character, despite the -> float annotation."""
    ci = np.asarray(mt.ut.CI(tree))
    ri = np.asarray(mt.ut.RI(tree))
    assert ci.size > 0 and ri.size > 0
    assert np.isfinite(ci).any() and np.isfinite(ri).any()


def test_calculate_corr_distances(tree):
    out = mt.ut.calculate_corr_distances(tree)
    assert out is not None


def test_neighbourhood_metrics(afm_filtered):
    D = afm_filtered.obsp["distances"].toarray()
    idx, _, _ = mt.pp.kNN_graph(D=D, k=5, from_distances=True)
    labels = afm_filtered.obs["GBC"]
    assert np.isfinite(mt.ut.kbet(idx, labels))
    assert np.isfinite(mt.ut.NN_entropy(idx, labels))
    assert np.isfinite(mt.ut.NN_purity(idx, labels))


def test_distance_AUPRC(afm_filtered):
    D = afm_filtered.obsp["distances"].toarray()
    out = mt.ut.distance_AUPRC(D, afm_filtered.obs["GBC"])
    assert out is not None


def test_clustering_agreement_metrics():
    a = pd.Series(["x", "x", "y", "y", "z", "z"])
    b = pd.Series(["1", "1", "2", "2", "3", "3"])
    assert 0.0 <= mt.ut.custom_ARI(a, b) <= 1.0
    assert 0.0 <= mt.ut.normalized_mutual_info_score(a, b) <= 1.0


def test_asset_loaders_return_data():
    assert mt.ut.load_mut_spectrum_ref().shape[0] > 0
    assert mt.ut.load_mt_gene_annot().shape[0] > 0
    assert len(mt.ut.load_common_dbSNP()) > 0
    assert len(mt.ut.load_edits_REDIdb()) > 0


def test_positions_helpers():
    assert len(mt.ut.transitions) > 0
    assert len(mt.ut.transversions) > 0
    assert len(mt.ut.MAESTER_genes_positions) > 0


def test_small_helpers():
    assert mt.ut.ji({1, 2, 3}, {2, 3, 4}) == pytest.approx(0.5)
    assert mt.ut.rescale(np.array([0.0, 5.0, 10.0])).max() == pytest.approx(1.0)
    assert mt.ut.update_params({"a": 1}, {"b": 2}) == {"a": 1, "b": 2}
    assert mt.ut.flatten_dict({"a": {"b": 1}}) is not None


def test_timer():
    t = mt.ut.Timer()
    t.start()
    assert t.stop().endswith(" s")
