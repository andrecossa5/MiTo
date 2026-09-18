"""
mito.pl.plot_tree

Annotations are named once in ``annot`` (a cell_meta column or a character), their colors
and ranges come from ``cmaps`` / ``limits``, and each tree element is configured by one
dictionary. These tests cover every colour-bearing argument, because colour resolution is
where this function used to rot: each element had its own rules, and paths nobody
exercised broke silently.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

import mito as mt


@pytest.fixture
def annotated_with_states(annotated):
    """A tree carrying a categorical, a continuous and a NaN-bearing annotation."""
    afm, tree = annotated
    support = tree.cell_meta["clone_support"]
    tree.cell_meta["state"] = pd.Categorical(np.where(support > 0.9, "high", "low"))
    tree.cell_meta["clone_str"] = tree.cell_meta["MiTo_clone"].astype(str)
    tree.cell_meta["with_na"] = pd.Categorical(
        [x if x == "high" else None for x in tree.cell_meta["state"]]
    )
    return afm, tree


def _ax(polar=False):
    _, ax = plt.subplots(figsize=(3, 3), subplot_kw={"projection": "polar"} if polar else None)
    return ax


# -- the basics -------------------------------------------------------------


def test_returns_the_axes_it_drew_on(annotated_tree):
    ax = _ax()
    assert mt.pl.plot_tree(annotated_tree, ax=ax) is ax


def test_creates_an_axes_when_none_is_given(annotated_tree):
    assert mt.pl.plot_tree(annotated_tree) is not None


@pytest.mark.parametrize("orient", [90, 0, "down", "right", "up", "left"])
def test_every_orientation_draws(annotated_tree, orient):
    mt.pl.plot_tree(annotated_tree, ax=_ax(), orient=orient)


@pytest.mark.parametrize("extend,angled,root", [(True, True, False), (False, False, True)])
def test_branch_layouts(annotated_tree, extend, angled, root):
    mt.pl.plot_tree(annotated_tree, ax=_ax(), extend_branches=extend,
                    angled_branches=angled, add_root=root)


# -- annot: one name, resolved wherever it lives ----------------------------


def test_a_single_annotation_can_be_a_string(annotated_tree):
    mt.pl.plot_tree(annotated_tree, annot="MiTo_clone", ax=_ax())


def test_features_and_characters_can_be_mixed(annotated, annotated_tree):
    afm, _ = annotated
    character = str(afm.var_names[0])
    mt.pl.plot_tree(annotated_tree, annot=["MiTo_clone", character], ax=_ax())


def test_categorical_and_continuous_annotations_together(annotated_with_states):
    _, tree = annotated_with_states
    mt.pl.plot_tree(tree, annot=["state", "clone_support"], ax=_ax())


def test_an_annotation_with_missing_values_is_drawn(annotated_with_states):
    _, tree = annotated_with_states
    mt.pl.plot_tree(tree, annot=["with_na"], ax=_ax())


@pytest.mark.parametrize("layer", ["raw", "transformed"])
def test_characters_from_either_layer(annotated, annotated_tree, layer):
    afm, _ = annotated
    mt.pl.plot_tree(annotated_tree, annot=list(afm.var_names[:3]), layer=layer, ax=_ax())


def test_an_unknown_annotation_names_both_namespaces(annotated_tree):
    with pytest.raises(KeyError, match="cell_meta"):
        mt.pl.plot_tree(annotated_tree, annot=["not_a_thing"], ax=_ax())


def test_a_missing_layer_says_which_characters_it_could_not_draw(annotated, annotated_tree):
    afm, _ = annotated
    with pytest.raises(KeyError, match="tree.layers"):
        mt.pl.plot_tree(annotated_tree, annot=list(afm.var_names[:2]), layer="nope", ax=_ax())


# -- cmaps and limits -------------------------------------------------------


@pytest.mark.parametrize("spec", [
    {"state": "tab10"},                               # a palette name
    {"state": ["#111111", "#eeeeee"]},                # a list of colors
    {"state": {"high": "r", "low": "grey"}},          # an explicit mapping
    {"state": {"high": "r"}},                         # a partial mapping: filled in
])
def test_every_way_of_specifying_categorical_colors(annotated_with_states, spec):
    _, tree = annotated_with_states
    mt.pl.plot_tree(tree, annot=["state"], cmaps=spec, ax=_ax())


def test_continuous_colors_and_limits(annotated_with_states):
    _, tree = annotated_with_states
    mt.pl.plot_tree(tree, annot=["clone_support"], cmaps={"clone_support": "viridis"},
                    limits={"clone_support": (0, 1)}, ax=_ax())


def test_binary_characters_take_a_state_palette(annotated, annotated_tree):
    afm, _ = annotated
    characters = list(afm.var_names[:2])
    mt.pl.plot_tree(annotated_tree, annot=characters, layer="transformed",
                    cmaps={c: {1: "k", 0: "w", -1: "grey"} for c in characters}, ax=_ax())


# -- the element dictionaries ----------------------------------------------


def test_colorstrip_geometry_and_labels(annotated_tree):
    mt.pl.plot_tree(annotated_tree, annot=["MiTo_clone"], orient="down",
                    colorstrips={"width": 2, "spacing": 0.4, "labels": True,
                                 "label_size": 8, "label_offset": 3}, ax=_ax())


def test_colorstrip_style_is_forwarded_to_matplotlib(annotated_tree):
    mt.pl.plot_tree(annotated_tree, annot=["MiTo_clone"],
                    colorstrips={"alpha": 0.5, "linewidth": 0.5}, ax=_ax())


@pytest.mark.parametrize("feature", ["state", "clone_support"])
def test_branches_coloured_by_a_covariate(annotated_with_states, feature):
    _, tree = annotated_with_states
    mt.pl.plot_tree(tree, branches={"feature": feature, "meta": tree.cell_meta[[feature]],
                                    "linewidth": 2}, ax=_ax())


def test_branches_without_metadata_say_what_is_missing(annotated_with_states):
    _, tree = annotated_with_states
    with pytest.raises(KeyError, match="meta"):
        mt.pl.plot_tree(tree, branches={"feature": "state"}, ax=_ax())


@pytest.mark.parametrize("spec", [
    {"feature": "MiTo_clone"},
    {"feature": "state", "cmap": {"high": "r", "low": "grey"}},
    {"feature": "clone_support", "cmap": "viridis", "limits": (0, 1)},
    {"feature": "MiTo_clone", "labels": True, "markersize": 3},
])
def test_leaves_coloured_and_labelled(annotated_with_states, spec):
    _, tree = annotated_with_states
    mt.pl.plot_tree(tree, orient="right", leaves=spec, ax=_ax())


def test_leaf_labels_outside_the_right_orientation_are_refused(annotated_tree):
    with pytest.raises(ValueError, match="orient"):
        mt.pl.plot_tree(annotated_tree, leaves={"feature": "MiTo_clone", "labels": True},
                        ax=_ax())


def test_internal_nodes_need_the_attribute_and_say_where_to_get_it(annotated_tree):
    with pytest.raises(ValueError, match="compute_fitness"):
        mt.pl.plot_tree(annotated_tree, internal_nodes={"feature": "support"}, ax=_ax())


@pytest.mark.parametrize("feature", ["fitness", "expansion_pvalue"])
def test_internal_nodes_coloured_by_a_computed_attribute(annotated_tree, feature):
    mt.tl.compute_fitness(annotated_tree)
    mt.tl.compute_expansions(annotated_tree)
    mt.pl.plot_tree(annotated_tree, internal_nodes={"feature": feature, "show": True,
                                                    "labels": True}, ax=_ax())


def test_internal_nodes_can_be_subset(annotated_tree):
    mt.tl.compute_fitness(annotated_tree)
    subset = list(annotated_tree.internal_nodes[:5])
    mt.pl.plot_tree(annotated_tree, internal_nodes={"feature": "fitness", "subset": subset},
                    ax=_ax())


# -- usability: the errors have to teach -----------------------------------


@pytest.mark.parametrize("old,new", [
    ("features", "annot"),
    ("characters", "annot"),
    ("cov_leaves", "leaves"),
    ("feature_internal_nodes", "internal_nodes"),
    ("colorstrip_width", "colorstrips"),
    ("show_internal", "internal_nodes"),
])
def test_renamed_arguments_point_at_their_replacement(annotated_tree, old, new):
    with pytest.raises(TypeError, match=new):
        mt.pl.plot_tree(annotated_tree, ax=_ax(), **{old: "MiTo_clone"})


def test_a_typo_in_a_spec_suggests_the_intended_key(annotated_tree):
    with pytest.raises(ValueError, match="did you mean"):
        mt.pl.plot_tree(annotated_tree, leaves={"featrue": "MiTo_clone"}, ax=_ax())


def test_an_unknown_argument_is_reported_as_unknown(annotated_tree):
    with pytest.raises(TypeError, match="Unknown"):
        mt.pl.plot_tree(annotated_tree, ax=_ax(), nonsense=1)
