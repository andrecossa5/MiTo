"""
mito.tl.bootstrap_MiTo, bootstrap_bin

A replicate must be usable: the old implementation rebuilt the object from the counts
only, dropping the genotypes, so the very next call in a bootstrap loop
(``compute_distances``) failed.
"""

import numpy as np
import pytest

import mito as mt


def test_observed_replicate_is_the_input(afm_filtered):
    out = mt.tl.bootstrap_MiTo(afm_filtered, boot_replicate="observed")
    assert out.shape == afm_filtered.shape
    assert list(out.var_names) == list(afm_filtered.var_names)
    assert out is not afm_filtered


@pytest.mark.parametrize("strategy", ["feature_resampling", "jacknife"])
def test_a_replicate_resamples_characters(afm_filtered, strategy):
    out = mt.tl.bootstrap_MiTo(afm_filtered, boot_replicate="1", boot_strategy=strategy,
                               frac_char_resampling=0.8, seed=0)
    assert out.shape[0] == afm_filtered.shape[0]
    assert out.shape[1] <= afm_filtered.shape[1]
    assert set(out.var_names) <= set(afm_filtered.var_names)


def test_a_replicate_can_be_used_downstream(afm_filtered):
    """
    Regression: the replicate must keep every layer, genotypes included, or the next step
    of a bootstrap loop cannot compute distances.
    """
    out = mt.tl.bootstrap_MiTo(afm_filtered, boot_replicate="1", seed=0)
    assert {"AD", "DP", "bin"} <= set(out.layers)

    mt.pp.compute_distances(out, verbose=False)
    tree = mt.tl.build_tree(out)
    assert set(tree.leaves) == set(out.obs_names)


def test_the_same_seed_gives_the_same_replicate(afm_filtered):
    first = mt.tl.bootstrap_MiTo(afm_filtered, boot_replicate="1", seed=3)
    second = mt.tl.bootstrap_MiTo(afm_filtered, boot_replicate="1", seed=3)
    assert list(first.var_names) == list(second.var_names)


def test_different_seeds_give_different_replicates(afm_filtered):
    first = mt.tl.bootstrap_MiTo(afm_filtered, boot_replicate="1", seed=1)
    second = mt.tl.bootstrap_MiTo(afm_filtered, boot_replicate="1", seed=2)
    assert list(first.var_names) != list(second.var_names)


def test_frac_1_resamples_with_replacement(afm_filtered):
    out = mt.tl.bootstrap_MiTo(afm_filtered, boot_replicate="1", frac_char_resampling=1, seed=0)
    assert out.shape[1] == afm_filtered.shape[1]


def test_jacknife_drops_exactly_one_character(afm_filtered):
    out = mt.tl.bootstrap_MiTo(afm_filtered, boot_replicate="1", boot_strategy="jacknife", seed=0)
    assert out.shape[1] == afm_filtered.shape[1] - 1


def test_replicates_do_not_share_state(afm_filtered):
    """Each replicate owns its .uns, so writing provenance into one cannot leak."""
    out = mt.tl.bootstrap_MiTo(afm_filtered, boot_replicate="1", seed=0)
    out.uns["mito"]["marker"] = True
    assert "marker" not in afm_filtered.uns["mito"]


def test_unknown_strategy_is_refused(afm_filtered):
    with pytest.raises(ValueError, match="feature_resampling"):
        mt.tl.bootstrap_MiTo(afm_filtered, boot_replicate="1", boot_strategy="magic")


def test_bootstrap_bin_needs_a_character_matrix(afm):
    with pytest.raises(ValueError, match="bin"):
        mt.tl.bootstrap_bin(afm, boot_replicate="1")


def test_bootstrap_bin_resamples_the_characters(afm_filtered):
    out = mt.tl.bootstrap_bin(afm_filtered, boot_replicate="1", seed=0)
    assert out.shape[1] <= afm_filtered.shape[1]
    assert np.isin(out.layers["bin"].toarray(), [0, 1]).all()
