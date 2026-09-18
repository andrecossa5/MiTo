"""
mito.pl, other than plot_tree

Plotting is checked for the contract a user relies on: it draws on the axes it is given,
creates one when it is not, accepts the data shapes the rest of MiTo produces, and colours
categorical and continuous features alike.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

import mito as mt


@pytest.fixture
def coverage_table(afm_filtered):
    """The long table mito.io.read_coverage returns."""
    rng = np.random.default_rng(0)
    cells = list(afm_filtered.obs_names[:20])
    positions = np.arange(1, 61) * 270
    return pd.DataFrame({
        "cell": np.repeat(cells, positions.size),
        "pos": np.tile(positions, len(cells)),
        "coverage": rng.integers(20, 60, len(cells) * positions.size),
    })


@pytest.fixture
def embedded(annotated):
    afm, tree = annotated
    afm.obs["state"] = pd.Categorical(np.where(afm.obs["clone_support"] > 0.9, "high", "low"))
    mt.pp.reduce_dimensions(afm, method="UMAP", ncores=1)
    mt.pp.reduce_dimensions(afm, method="PCA", ncores=1)
    return afm, tree


# -- embeddings -------------------------------------------------------------


@pytest.mark.parametrize("feature", [None, "MiTo_clone", "clone_support", "state"])
def test_draw_embedding_handles_every_feature_kind(embedded, feature):
    afm, _ = embedded
    _, ax = plt.subplots()
    assert mt.pl.draw_embedding(afm, feature=feature, ax=ax) is not None


def test_draw_embedding_legend_for_a_categorical_feature(embedded):
    """
    Regression: the legend used to receive the palette (a list of colors) instead of the
    {category: color} mapping, so every categorical embedding with a legend failed.
    """
    afm, _ = embedded
    _, ax = plt.subplots()
    mt.pl.draw_embedding(afm, feature="MiTo_clone", legend=True, ax=ax)


@pytest.mark.parametrize("cmap", ["tab10", {"high": "r", "low": "grey"}])
def test_draw_embedding_accepts_a_palette_or_a_mapping(embedded, cmap):
    afm, _ = embedded
    _, ax = plt.subplots()
    mt.pl.draw_embedding(afm, feature="state", legend=True, categorical_cmap=cmap, ax=ax)


def test_draw_embedding_incomplete_mapping_names_the_missing_categories(embedded):
    afm, _ = embedded
    with pytest.raises(ValueError, match="No color for"):
        mt.pl.draw_embedding(afm, feature="state", categorical_cmap={"high": "r"})


def test_draw_embedding_uses_the_requested_basis(embedded):
    afm, _ = embedded
    _, ax = plt.subplots()
    mt.pl.draw_embedding(afm, basis="X_pca", feature="MiTo_clone", ax=ax)


# -- heatmaps ---------------------------------------------------------------


def test_heatmap_distances(annotated):
    afm, tree = annotated
    _, ax = plt.subplots()
    assert mt.pl.heatmap_distances(afm, tree=tree, ax=ax) is not None


def test_heatmap_distances_needs_distances(afm):
    with pytest.raises(ValueError, match="compute_distances"):
        mt.pl.heatmap_distances(afm)


@pytest.mark.parametrize("layer,vmax", [(None, 0.1), ("bin", 1), ("AD", 30), ("DP", 300)])
def test_heatmap_variants_on_every_layer(annotated, layer, vmax):
    """
    Regression: a sparse layer was handed straight to pandas, so every `layer=` call
    failed and only the default allele-frequency path worked.
    """
    afm, tree = annotated
    _, ax = plt.subplots()
    mt.pl.heatmap_variants(afm, tree=tree, layer=layer, vmax=vmax, ax=ax)


def test_heatmap_variants_with_a_cell_annotation(annotated):
    afm, tree = annotated
    _, ax = plt.subplots()
    mt.pl.heatmap_variants(afm, tree=tree, annot="MiTo_clone", ax=ax)


def test_heatmap_variants_unknown_annotation(annotated):
    afm, tree = annotated
    with pytest.raises(KeyError):
        mt.pl.heatmap_variants(afm, tree=tree, annot="not_a_column")


def test_heatmap_variants_unknown_layer(annotated):
    afm, tree = annotated
    with pytest.raises(KeyError):
        mt.pl.heatmap_variants(afm, tree=tree, layer="not_a_layer")


# -- diagnostics ------------------------------------------------------------


def test_vars_AF_spectrum(afm_filtered):
    _, ax = plt.subplots()
    assert mt.pl.vars_AF_spectrum(afm_filtered, ax=ax, color="g") is not None


def test_plot_ncells_nAD(afm_filtered):
    _, ax = plt.subplots()
    assert mt.pl.plot_ncells_nAD(afm_filtered, ax=ax, title="synthetic") is not None


def test_mut_profile(afm_filtered):
    assert mt.pl.mut_profile(list(afm_filtered.var_names)) is not None


def test_coverage_plots_accept_the_long_table(coverage_table, afm_filtered):
    """Both polar plots take what mito.io.read_coverage returns, unchanged."""
    _, ax = plt.subplots(subplot_kw={"projection": "polar"})
    mt.pl.MT_coverage_polar(coverage_table, var_subset=list(afm_filtered.var_names), ax=ax)

    _, ax = plt.subplots(subplot_kw={"projection": "polar"})
    mt.pl.MT_coverage_by_gene_polar(coverage_table, sample="synthetic", ax=ax)


def test_coverage_by_gene_can_be_restricted_to_a_cell_subset(coverage_table):
    cells = list(coverage_table["cell"].unique()[:5])
    _, ax = plt.subplots(subplot_kw={"projection": "polar"})
    mt.pl.MT_coverage_by_gene_polar(coverage_table, sample="s", subset=cells, ax=ax)


def test_coverage_plots_create_their_own_polar_axes(coverage_table):
    assert mt.pl.MT_coverage_polar(coverage_table) is not None
    assert mt.pl.MT_coverage_by_gene_polar(coverage_table, sample="s") is not None


# -- circles ----------------------------------------------------------------


def test_packed_circle_plot_accepts_a_series(annotated):
    """value_counts() is the natural input, and used to raise."""
    afm, _ = annotated
    counts = afm.obs["MiTo_clone"].astype(str).value_counts()
    assert mt.pl.packed_circle_plot(counts) is not None


def test_packed_circle_plot_accepts_a_frame(annotated):
    afm, _ = annotated
    counts = afm.obs["MiTo_clone"].astype(str).value_counts().to_frame("n")
    _, ax = plt.subplots()
    mt.pl.packed_circle_plot(counts, covariate="n", ax=ax, annotate=True)


def test_packed_circle_plot_ambiguous_frame_is_refused(annotated):
    afm, _ = annotated
    df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
    with pytest.raises(ValueError, match="covariate"):
        mt.pl.packed_circle_plot(df)


# -- every exported plot draws without an axes ------------------------------


def test_all_plotting_functions_are_exported_and_callable(embedded, coverage_table):
    afm, tree = embedded
    assert set(mt.pl.__all__) == {
        "vars_AF_spectrum", "MT_coverage_by_gene_polar", "MT_coverage_polar", "mut_profile",
        "plot_ncells_nAD", "draw_embedding", "heatmap_distances", "heatmap_variants",
        "plot_tree", "packed_circle_plot",
    }
    mt.pl.plot_tree(tree)
    mt.pl.draw_embedding(afm, feature="MiTo_clone")
    mt.pl.heatmap_distances(afm, tree=tree)
    mt.pl.heatmap_variants(afm, tree=tree)
    mt.pl.vars_AF_spectrum(afm)
    mt.pl.plot_ncells_nAD(afm)
    mt.pl.mut_profile(list(afm.var_names))
    mt.pl.packed_circle_plot(afm.obs["MiTo_clone"].astype(str).value_counts())
    mt.pl.MT_coverage_polar(coverage_table)
    mt.pl.MT_coverage_by_gene_polar(coverage_table, sample="s")
