"""
mito.pp.filter_incompatible_variants

The four-gamete filter, and above all *which* of two conflicting variants it drops. That
choice used to be made by conflict count alone, which favours sparse noise over prevalent
markers: on one of our datasets it deleted three real markers and left 613 of 1272
genotyped cells with no character at all.
"""

import numpy as np
import pytest
from scipy.sparse import csr_matrix

import mito as mt


def _afm_from_calls(B, snr=None):
    """A minimal AFM carrying a given call matrix (and optionally per-variant SNRs)."""
    import anndata as ad
    n, m = B.shape
    afm = ad.AnnData(
        X=csr_matrix(np.where(B, 0.1, 0.0).astype(np.float32)),
        obs=__import__("pandas").DataFrame(index=[f"c{i}" for i in range(n)]),
        var=__import__("pandas").DataFrame(
            {"pos": np.arange(1, m + 1)}, index=[f"v{j}" for j in range(m)]
        ),
        layers={"bin": csr_matrix(B.astype(np.int8))},
    )
    if snr is not None:
        afm.var["snr"] = snr
    return afm


def test_compatible_characters_are_all_kept():
    """Two nested clades: no four-gamete conflict, nothing to resolve."""
    B = np.zeros((40, 2), bool)
    B[:20, 0] = True          # a clade
    B[:10, 1] = True          # nested inside it
    afm = _afm_from_calls(B)
    mt.pp.filter_incompatible_variants(afm, min_gamete=2)
    assert afm.shape[1] == 2


def test_an_incompatible_pair_is_resolved():
    """All four gametes present in numbers: under infinite sites, one has to go."""
    B = np.zeros((80, 2), bool)
    B[:40, 0] = True
    B[20:60, 1] = True        # overlaps half of the first: 1/1, 1/0, 0/1, 0/0 all present
    afm = _afm_from_calls(B)
    mt.pp.filter_incompatible_variants(afm, min_gamete=5)
    assert afm.shape[1] == 1


def test_the_variant_with_less_signal_is_the_one_dropped():
    """
    Regression for the tie-breaking. Two conflicting variants, the second carried by far
    fewer cells but with a much weaker signal over its own background: the prevalent,
    clean marker must survive.
    """
    B = np.zeros((80, 2), bool)
    B[:40, 0] = True          # prevalent marker
    B[20:60, 1] = True        # conflicts with it
    afm = _afm_from_calls(B, snr=np.array([500.0, 12.0]))
    mt.pp.filter_incompatible_variants(afm, min_gamete=5)
    assert list(afm.var_names) == ["v0"]


def test_without_snr_the_most_conflicted_variant_goes_first():
    """
    With no signal to compare, the variant in conflict with the most others is the one
    dropped - and ties go to the least prevalent, so the outcome never depends on the
    column order of the AFM.
    """
    B = np.zeros((100, 3), bool)
    B[:50, 0] = True                      # two nested, compatible markers
    B[:25, 1] = True
    B[20:70:2, 2] = True                  # sparse and interleaved: conflicts with both
    afm = _afm_from_calls(B)
    mt.pp.filter_incompatible_variants(afm, min_gamete=5)
    assert list(afm.var_names) == ["v0", "v1"]


def test_min_gamete_tolerates_dropouts_and_stray_calls():
    """
    A handful of cells in each gamete is what dropouts and stray calls produce, not
    recurrence: `min_gamete` is the tolerance that keeps such a pair compatible.
    """
    B = np.zeros((100, 2), bool)
    B[:50, 0] = True
    B[5:55, 1] = True         # 5 cells in each of the two "impossible" gametes
    tolerant = _afm_from_calls(B)
    mt.pp.filter_incompatible_variants(tolerant, min_gamete=10)
    assert tolerant.shape[1] == 2

    strict = _afm_from_calls(B)
    mt.pp.filter_incompatible_variants(strict, min_gamete=1)
    assert strict.shape[1] == 1


def test_conflict_counts_and_record_are_written():
    B = np.zeros((80, 2), bool)
    B[:40, 0] = True
    B[20:60, 1] = True
    afm = _afm_from_calls(B)
    mt.pp.filter_incompatible_variants(afm, min_gamete=5)
    assert "n_conflicts" in afm.var.columns
    assert afm.uns["compatibility"] == {"min_gamete": 5, "n_removed": 1}


def test_requires_genotypes(afm):
    with pytest.raises(ValueError, match="call_genotypes"):
        mt.pp.filter_incompatible_variants(afm)


def test_copy_semantics():
    B = np.zeros((80, 2), bool)
    B[:40, 0] = True
    B[20:60, 1] = True
    afm = _afm_from_calls(B)
    out = mt.pp.filter_incompatible_variants(afm, min_gamete=5, copy=True)
    assert out is not afm and afm.shape[1] == 2 and out.shape[1] == 1
