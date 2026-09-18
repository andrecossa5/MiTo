"""
mito.pp.call_genotypes, filter_low_signal_variants, impute_dropouts

The genotyper tests a cell's alternative reads against the variant's own error rate.
Two properties matter most and are regression-tested here: a cell with no alternative
read is never called (the failure mode of the mixture genotyper it replaced), and the
background is estimated from non-carriers only (so a clone cannot inflate the background
of its own marker).
"""

import numpy as np
import pytest

import mito as mt

# -- calls ------------------------------------------------------------------


def test_writes_the_genotyping_slots(afm):
    mt.pp.call_genotypes(afm)
    assert "bin" in afm.layers and afm.layers["bin"].shape == afm.shape
    assert "imputed" in afm.layers
    for column in ("error_rate", "n_carriers", "n_imputed", "prevalence"):
        assert column in afm.var.columns
    assert afm.uns["genotyping"]["method"] == "binomial"


def test_a_cell_with_no_alternative_read_is_never_called(afm):
    """
    Regression: the binomial-mixture genotyper called cells with AD = 0 from the prior
    alone (61% of the calls on one of our datasets). Absence of reads is absence of
    evidence.
    """
    mt.pp.call_genotypes(afm)
    AD = afm.layers["AD"].toarray()
    B = afm.layers["bin"].toarray()
    assert not ((AD == 0) & (B > 0)).any()


def test_recovers_the_planted_genotypes(afm):
    mt.pp.call_genotypes(afm)
    truth = afm.layers["genotype"].toarray() > 0
    called = afm.layers["bin"].toarray() > 0
    planted = truth.any(axis=0)
    recall = called[:, planted][truth[:, planted]].mean()
    precision = truth[:, planted][called[:, planted]].mean()
    assert recall > 0.9 and precision > 0.9


def test_error_rate_is_estimated_from_non_carriers(afm):
    """
    The planted markers sit at AF 0.15 in a quarter of the cells, on a background of
    0.002. A background taken over all cells would be inflated towards the carriers'.
    """
    mt.pp.call_genotypes(afm)
    planted = afm.layers["genotype"].toarray().any(axis=0)
    assert afm.var.loc[planted, "error_rate"].max() < 0.02


def test_a_stricter_alpha_calls_no_more_cells(afm):
    strict = mt.pp.call_genotypes(afm, alpha=1e-6, copy=True)
    loose = mt.pp.call_genotypes(afm, alpha=1e-2, copy=True)
    assert strict.layers["bin"].sum() <= loose.layers["bin"].sum()


def test_convergence_is_reported(afm):
    mt.pp.call_genotypes(afm)
    assert afm.uns["genotyping"]["converged"] is True
    assert afm.uns["genotyping"]["n_iter"] >= 1


def test_truncated_background_is_flagged(afm, caplog):
    """Stopping the iteration early is legitimate but must not be silent."""
    with caplog.at_level("WARNING"):
        mt.pp.call_genotypes(afm, max_iter=1)
    assert afm.uns["genotyping"]["converged"] is False
    assert "converge" in caplog.text


def test_needs_read_counts(afm):
    del afm.layers["AD"]
    with pytest.raises(ValueError, match="AD"):
        mt.pp.call_genotypes(afm)


def test_copy_semantics(afm):
    out = mt.pp.call_genotypes(afm, copy=True)
    assert out is not afm
    assert "bin" in out.layers and "bin" not in afm.layers


# -- signal over background -------------------------------------------------


def test_snr_is_the_median_af_of_read_bearing_cells_over_the_error_rate(afm):
    mt.pp.call_genotypes(afm)
    AD = afm.layers["AD"].toarray()
    X = afm.X.toarray()
    mt.pp.filter_low_signal_variants(afm, min_snr=0)

    with np.errstate(all="ignore"):
        expected = np.nanmedian(np.where(AD >= 1, X, np.nan), axis=0) / afm.var["error_rate"]
    assert np.allclose(afm.var["snr"], expected, rtol=1e-6)


def test_planted_markers_stand_above_their_background(afm):
    mt.pp.call_genotypes(afm)
    planted = afm.layers["genotype"].toarray().any(axis=0)
    mt.pp.filter_low_signal_variants(afm, min_snr=0)
    assert afm.var.loc[planted, "snr"].min() > 10


def test_a_broad_heteroplasmic_variant_is_dropped(afm):
    """
    A variant present at a similar AF in every cell has a random-looking subset called
    and no signal over its own background: it must not survive.
    """
    rng = np.random.default_rng(0)
    AD = afm.layers["AD"].toarray()
    DP = np.asarray(afm.layers["DP"])
    AD[:, -1] = rng.binomial(DP[:, -1], 0.02)          # everywhere, weakly
    afm.layers["AD"] = type(afm.layers["AD"])(AD)
    afm.X = type(afm.X)((AD / DP).astype(np.float32))

    mt.pp.call_genotypes(afm)
    name = afm.var_names[-1]
    mt.pp.filter_low_signal_variants(afm, min_snr=10)
    assert name not in afm.var_names


def test_requires_genotypes(afm):
    with pytest.raises(ValueError, match="call_genotypes"):
        mt.pp.filter_low_signal_variants(afm)


# -- imputation -------------------------------------------------------------


def test_imputation_only_adds_calls(afm):
    mt.pp.call_genotypes(afm)
    before = afm.layers["bin"].toarray() > 0
    mt.pp.impute_dropouts(afm, k=10, thr=0.5)
    after = afm.layers["bin"].toarray() > 0
    assert (after >= before).all()
    assert after.sum() >= before.sum()


def test_imputed_calls_are_flagged_and_counted(afm):
    mt.pp.call_genotypes(afm)
    mt.pp.impute_dropouts(afm, k=10, thr=0.5)
    imputed = afm.layers["imputed"].toarray() > 0
    called = afm.layers["bin"].toarray() > 0
    assert (imputed <= called).all(), "an imputed call must be a call"
    assert afm.var["n_imputed"].sum() == imputed.sum()
    assert afm.uns["imputation"]["n_imputed"] == int(imputed.sum())


def test_min_support_protects_cells_with_no_evidence(afm):
    """
    Without it, a cell with no call at all is handed a genotype by its neighbourhood
    alone, which is how imputation starts inventing clones.
    """
    AD = afm.layers["AD"].toarray()
    AD[0] = 0                                   # a cell with no alternative read anywhere
    afm.layers["AD"] = type(afm.layers["AD"])(AD)
    afm.X = type(afm.X)((AD / np.asarray(afm.layers["DP"])).astype(np.float32))

    mt.pp.call_genotypes(afm)
    assert not (afm.layers["bin"].toarray()[0] > 0).any()

    mt.pp.impute_dropouts(afm, k=10, thr=0.5, min_support=1)
    assert not (afm.layers["bin"].toarray()[0] > 0).any(), "no call, no imputation"


def test_imputation_requires_genotypes(afm):
    with pytest.raises(ValueError, match="call_genotypes"):
        mt.pp.impute_dropouts(afm)
