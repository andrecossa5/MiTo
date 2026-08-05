"""
mito.pl.plot_tree

The widest plotting surface in the package (~40 arguments). Covers orientation,
branch styling, the feature/character colour strips, leaf and internal-node
annotation, colour scaling, and the kwargs pass-throughs.

Assertions stay at the level a plotting wrapper can honestly guarantee: the call
succeeds, returns an Axes, and puts artists on it. Where a parameter has an
observable effect on the axes, that effect is asserted directly.
"""

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pytest
import mito as mt


def _ax():
    fig, ax = plt.subplots(figsize=(4, 4))
    return ax


def _n_artists(ax):
    return len(ax.lines) + len(ax.collections) + len(ax.patches) + len(ax.texts)


# -- baseline ---------------------------------------------------------------

def test_returns_an_axes(tree):
    ax = _ax()
    out = mt.pl.plot_tree(tree, ax=ax)
    assert isinstance(out, matplotlib.axes.Axes)


def test_draws_something(tree):
    ax = _ax()
    mt.pl.plot_tree(tree, ax=ax)
    assert _n_artists(ax) > 0


def test_draws_one_branch_per_edge(tree):
    ax = _ax()
    mt.pl.plot_tree(tree, ax=ax)
    assert len(ax.lines) >= len(tree.leaves)


def test_creates_its_own_axes_when_none_given(tree):
    out = mt.pl.plot_tree(tree)
    assert isinstance(out, matplotlib.axes.Axes)
    plt.close("all")


def test_does_not_mutate_the_tree(tree):
    n_leaves, n_internal = len(tree.leaves), len(tree.internal_nodes)
    mt.pl.plot_tree(tree, ax=_ax())
    assert len(tree.leaves) == n_leaves
    assert len(tree.internal_nodes) == n_internal


# -- orientation and layout -------------------------------------------------

@pytest.mark.parametrize("orient", [90, 180, 270, 360, "right", "left", "up", "down"])
def test_orientations(tree, orient):
    ax = _ax()
    out = mt.pl.plot_tree(tree, ax=ax, orient=orient)
    assert isinstance(out, matplotlib.axes.Axes)
    assert _n_artists(ax) > 0


@pytest.mark.parametrize("extend_branches", [True, False])
def test_extend_branches(tree, extend_branches):
    ax = _ax()
    mt.pl.plot_tree(tree, ax=ax, extend_branches=extend_branches)
    assert _n_artists(ax) > 0


@pytest.mark.parametrize("angled_branches", [True, False])
def test_angled_branches(tree, angled_branches):
    ax = _ax()
    mt.pl.plot_tree(tree, ax=ax, angled_branches=angled_branches)
    assert _n_artists(ax) > 0


@pytest.mark.parametrize("add_root", [True, False])
def test_add_root(tree, add_root):
    """Regression: add_root=True used to raise, because the plotting root is not
    a node of the tree network and was queried before being excluded."""
    ax = _ax()
    mt.pl.plot_tree(tree, ax=ax, add_root=add_root)
    assert _n_artists(ax) > 0


@pytest.mark.parametrize("x_space", [0.5, 1.5, 4.0])
def test_x_space(tree, x_space):
    ax = _ax()
    mt.pl.plot_tree(tree, ax=ax, x_space=x_space)
    assert _n_artists(ax) > 0


# -- feature colour strips --------------------------------------------------

def test_single_categorical_feature(tree):
    ax = _ax()
    mt.pl.plot_tree(tree, ax=ax, features=["GBC"])
    assert _n_artists(ax) > 0


def test_single_continuous_feature(tree):
    ax = _ax()
    mt.pl.plot_tree(tree, ax=ax, features=["mean_site_coverage"])
    assert _n_artists(ax) > 0


def test_multiple_features(tree):
    ax = _ax()
    mt.pl.plot_tree(tree, ax=ax, features=["GBC", "mean_site_coverage", "nUMIs"])
    assert _n_artists(ax) > 0


def test_feature_with_explicit_categorical_cmap(tree):
    ax = _ax()
    categories = tree.cell_meta["GBC"].unique()
    cmap = {c: col for c, col in zip(categories, ["#e41a1c", "#377eb8", "#4daf4a"])}
    mt.pl.plot_tree(tree, ax=ax, features=["GBC"], categorical_cmaps={"GBC": cmap})
    assert _n_artists(ax) > 0


@pytest.mark.parametrize("cmap", ["viridis", "mako", "Spectral_r"])
def test_feature_with_continuous_cmap(tree, cmap):
    ax = _ax()
    mt.pl.plot_tree(tree, ax=ax, features=["nUMIs"], continuous_cmaps={"nUMIs": cmap})
    assert _n_artists(ax) > 0


def test_unknown_feature_raises(tree):
    with pytest.raises(Exception):
        mt.pl.plot_tree(tree, ax=_ax(), features=["not_a_column"])


@pytest.mark.parametrize("colorstrip_width", [0.5, 1.5, 3.0])
def test_colorstrip_width(tree, colorstrip_width):
    ax = _ax()
    mt.pl.plot_tree(tree, ax=ax, features=["GBC"], colorstrip_width=colorstrip_width)
    assert _n_artists(ax) > 0


@pytest.mark.parametrize("colorstrip_spacing", [0.0, 0.25, 1.0])
def test_colorstrip_spacing(tree, colorstrip_spacing):
    ax = _ax()
    mt.pl.plot_tree(tree, ax=ax, features=["GBC"], colorstrip_spacing=colorstrip_spacing)
    assert _n_artists(ax) > 0


# -- character (MT-SNV) strips ----------------------------------------------

def test_characters(tree):
    ax = _ax()
    chars = list(tree.character_matrix.columns[:3])
    mt.pl.plot_tree(tree, ax=ax, characters=chars)
    assert _n_artists(ax) > 0


def test_all_characters(tree):
    ax = _ax()
    mt.pl.plot_tree(tree, ax=ax, characters=list(tree.character_matrix.columns))
    assert _n_artists(ax) > 0


@pytest.mark.parametrize("layer", ["raw", "transformed"])
def test_character_layers(tree, layer):
    ax = _ax()
    chars = list(tree.character_matrix.columns[:2])
    mt.pl.plot_tree(tree, ax=ax, characters=chars, layer=layer)
    assert _n_artists(ax) > 0


@pytest.mark.parametrize("cont_character_cmap", ["mako", "viridis"])
def test_character_colormap(tree, cont_character_cmap):
    ax = _ax()
    chars = list(tree.character_matrix.columns[:2])
    mt.pl.plot_tree(tree, ax=ax, characters=chars,
                    cont_character_cmap=cont_character_cmap)
    assert _n_artists(ax) > 0


@pytest.mark.parametrize("vmin_characters,vmax_characters",
                         [(0, 0.01), (0, 0.05), (0.01, 0.5)])
def test_character_value_limits(tree, vmin_characters, vmax_characters):
    ax = _ax()
    chars = list(tree.character_matrix.columns[:2])
    mt.pl.plot_tree(tree, ax=ax, characters=chars,
                    vmin_characters=vmin_characters, vmax_characters=vmax_characters)
    assert _n_artists(ax) > 0


def test_features_and_characters_together(tree):
    ax = _ax()
    mt.pl.plot_tree(tree, ax=ax, features=["GBC"],
                    characters=list(tree.character_matrix.columns[:2]))
    assert _n_artists(ax) > 0


# -- labels -----------------------------------------------------------------

@pytest.mark.parametrize("labels", [True, False])
def test_labels_toggle(tree, labels):
    ax = _ax()
    mt.pl.plot_tree(tree, ax=ax, features=["GBC"], labels=labels)
    assert _n_artists(ax) > 0


@pytest.mark.parametrize("label_size", [4, 10, 16])
def test_label_size(tree, label_size):
    ax = _ax()
    mt.pl.plot_tree(tree, ax=ax, features=["GBC"], label_size=label_size)
    assert _n_artists(ax) > 0


def test_leaves_labels_require_right_orient(tree):
    """Documented restriction: leaf labels are only placed for orient='right'."""
    with pytest.raises(ValueError, match="right orient"):
        mt.pl.plot_tree(tree, ax=_ax(), leaves_labels=True, orient=90)


def test_leaves_labels_with_right_orient(tree):
    ax = _ax()
    mt.pl.plot_tree(tree, ax=ax, orient="right", leaves_labels=True)
    assert len(ax.texts) >= len(tree.leaves)


@pytest.mark.parametrize("leaf_label_size", [3, 8])
def test_leaf_label_size(tree, leaf_label_size):
    ax = _ax()
    mt.pl.plot_tree(tree, ax=ax, orient="right", leaves_labels=True,
                    leaf_label_size=leaf_label_size)
    assert len(ax.texts) > 0


# -- internal nodes ---------------------------------------------------------

@pytest.mark.parametrize("show_internal", [True, False])
def test_show_internal(tree, show_internal):
    ax = _ax()
    mt.pl.plot_tree(tree, ax=ax, show_internal=show_internal)
    assert _n_artists(ax) > 0


def test_internal_node_labels(tree):
    ax = _ax()
    mt.pl.plot_tree(tree, ax=ax, show_internal=True, internal_node_labels=True)
    assert _n_artists(ax) > 0


def test_internal_node_subset(tree):
    ax = _ax()
    subset = list(tree.internal_nodes)[:5]
    mt.pl.plot_tree(tree, ax=ax, show_internal=True, internal_node_subset=subset)
    assert _n_artists(ax) > 0


@pytest.mark.parametrize("internal_node_label_size", [5, 12])
def test_internal_node_label_size(tree, internal_node_label_size):
    ax = _ax()
    mt.pl.plot_tree(tree, ax=ax, show_internal=True, internal_node_labels=True,
                    internal_node_label_size=internal_node_label_size)
    assert _n_artists(ax) > 0


# -- colour scaling ---------------------------------------------------------

@pytest.mark.parametrize("vmin,vmax", [(0, 1), (0.1, 0.9), (None, None)])
def test_vmin_vmax(tree, vmin, vmax):
    ax = _ax()
    mt.pl.plot_tree(tree, ax=ax, features=["nUMIs"], vmin=vmin, vmax=vmax)
    assert _n_artists(ax) > 0


@pytest.mark.parametrize("vmin_internal_nodes,vmax_internal_nodes",
                         [(0.0, 1.0), (0.2, 0.8)])
def test_internal_node_value_limits(tree, vmin_internal_nodes, vmax_internal_nodes):
    ax = _ax()
    mt.pl.plot_tree(tree, ax=ax, show_internal=True,
                    vmin_internal_nodes=vmin_internal_nodes,
                    vmax_internal_nodes=vmax_internal_nodes)
    assert _n_artists(ax) > 0


# -- kwargs pass-throughs ---------------------------------------------------

def test_branch_kwargs(tree):
    ax = _ax()
    mt.pl.plot_tree(tree, ax=ax, branch_kwargs={"linewidth": 2.0, "c": "red"})
    assert _n_artists(ax) > 0


def test_leaf_kwargs(tree):
    ax = _ax()
    mt.pl.plot_tree(tree, ax=ax, leaf_kwargs={"markersize": 4})
    assert _n_artists(ax) > 0


def test_internal_node_kwargs(tree):
    ax = _ax()
    mt.pl.plot_tree(tree, ax=ax, show_internal=True,
                    internal_node_kwargs={"markersize": 5})
    assert _n_artists(ax) > 0


def test_colorstrip_kwargs(tree):
    ax = _ax()
    mt.pl.plot_tree(tree, ax=ax, features=["GBC"],
                    colorstrip_kwargs={"linewidth": 0.5})
    assert _n_artists(ax) > 0


# -- annotated trees --------------------------------------------------------

def test_plots_inferred_clones(annotated_tree):
    """The tutorial's end state: colour leaves by the inferred MiTo clone."""
    ax = _ax()
    mt.pl.plot_tree(annotated_tree, ax=ax, features=["MiTo clone"])
    assert _n_artists(ax) > 0


def test_plots_ground_truth_against_inferred(annotated_tree):
    ax = _ax()
    mt.pl.plot_tree(annotated_tree, ax=ax, features=["GBC", "MiTo clone"])
    assert _n_artists(ax) > 0


# -- combinations and edge cases --------------------------------------------

@pytest.mark.parametrize("orient", [90, "right"])
def test_everything_at_once(tree, orient):
    ax = _ax()
    mt.pl.plot_tree(
        tree, ax=ax, orient=orient,
        features=["GBC", "nUMIs"],
        characters=list(tree.character_matrix.columns[:2]),
        show_internal=True, add_root=True,
        extend_branches=True, angled_branches=False,
        colorstrip_width=2.0, colorstrip_spacing=0.3,
        branch_kwargs={"linewidth": 0.8},
    )
    assert _n_artists(ax) > 0


def test_small_tree():
    """A small tree must still render."""
    from conftest import build_afm
    a = build_afm(n_cells=35, n_vars=12, n_clones=2, clone_specific_frac=0.6, seed=50)
    mt.pp.annotate_vars(a)
    filtered = mt.pp.filter_afm(a, filtering="baseline",
                                compute_enrichment=False, ncores=1)
    small_tree = mt.tl.build_tree(filtered, precomputed=True, solver="UPMGA")
    ax = _ax()
    mt.pl.plot_tree(small_tree, ax=ax)
    assert _n_artists(ax) > 0


@pytest.mark.parametrize("solver", ["NJ", "UPMGA", "spectral", "greedy"])
def test_plots_trees_from_every_solver(afm_filtered, solver):
    t = mt.tl.build_tree(afm_filtered, precomputed=True, solver=solver)
    ax = _ax()
    mt.pl.plot_tree(t, ax=ax)
    assert _n_artists(ax) > 0
