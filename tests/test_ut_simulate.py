"""
mito.ut.simulate_afm

Checks the structure of the simulated AFM, and that the clonal process does what it
claims: one unique variant per clone, a polytomy of independent clones, and a
minority of clones arising as lineage splits that inherit their parent's variant.
"""

import numpy as np
import pytest
from anndata import AnnData

import mito as mt


# -- structure --------------------------------------------------------------

def test_returns_anndata_with_one_variant_per_clone():
    afm = mt.ut.simulate_afm(n_cells=200, n_clones=8, frac_noisy_variants=0.0)
    assert isinstance(afm, AnnData)
    assert afm.shape == (200, 8)


def test_noisy_variants_are_added_on_top_of_the_clonal_ones():
    afm = mt.ut.simulate_afm(n_cells=200, n_clones=10, frac_noisy_variants=0.3)
    s = afm.uns["simulation"]
    assert s["n_clonal_vars"] == 10
    assert s["n_noisy_vars"] / s["n_vars"] == pytest.approx(0.3, abs=0.05)
    assert afm.var["is_noise"].sum() == s["n_noisy_vars"]


def test_noisy_variants_ignore_the_tree():
    """A noisy variant sits at its own prevalence in every clone alike."""
    afm = mt.ut.simulate_afm(n_cells=600, n_clones=5, frac_noisy_variants=0.5,
                             noise_prevalence=(0.2, 0.2), random_seed=0)
    G = afm.layers["genotype"].toarray()
    labels = afm.obs["clone"].astype(str).values
    for j in np.flatnonzero(afm.var["is_noise"].values):
        rates = [G[labels == c, j].mean() for c in np.unique(labels)]
        assert min(rates) > 0.05 and max(rates) < 0.45


def test_has_the_layers_the_pipeline_needs():
    afm = mt.ut.simulate_afm(n_cells=100, n_clones=5)
    for layer in ("AD", "DP", "genotype"):
        assert layer in afm.layers
        assert afm.layers[layer].shape == afm.shape
    # the same contract mito.io.make_afm produces: one coverage layer, dense
    from scipy.sparse import issparse
    assert not issparse(afm.layers["DP"])
    assert "site_coverage" not in afm.layers and "qual" not in afm.layers


def test_has_the_obs_columns_the_cell_filters_need():
    afm = mt.ut.simulate_afm(n_cells=100, n_clones=5)
    for column in ("mean_site_coverage", "median_target_site_coverage",
                   "frac_target_site_covered"):
        assert column in afm.obs.columns


def test_has_the_uns_slots_the_pipeline_needs():
    afm = mt.ut.simulate_afm(n_cells=100, n_clones=5)
    for key in ("scLT_system", "pp_method", "simulation"):
        assert key in afm.uns


def test_var_names_follow_the_pos_ref_alt_convention():
    afm = mt.ut.simulate_afm(n_cells=100, n_clones=10)
    for name, row in zip(afm.var_names, afm.var.itertuples(), strict=True):
        assert name == f"{row.pos}_{row.ref}>{row.alt}"
        assert row.ref != row.alt


def test_positions_fall_inside_maester_target_sites():
    """The baseline filter masks anything outside gene bodies, so simulated
    variants must live there or they would all be discarded."""
    afm = mt.ut.simulate_afm(n_cells=100, n_clones=40, frac_noisy_variants=0.0)
    assert mt.ut.mask_mt_sites(afm.var["pos"]).all()


# -- genotypes are exact presence/absence -----------------------------------

def test_base_quality_is_a_variant_property():
    afm = mt.ut.simulate_afm(n_cells=100, n_clones=5)
    assert "quality" in afm.var.columns
    assert (afm.var["quality"] > 0).all()


def test_genotype_layer_is_binary():
    afm = mt.ut.simulate_afm(n_cells=300, n_clones=10, n_root_clones=7)
    assert set(np.unique(afm.layers["genotype"].toarray())) <= {0, 1}


def test_counts_are_consistent():
    afm = mt.ut.simulate_afm(n_cells=300, n_clones=10)
    AD = afm.layers["AD"].toarray()
    DP = np.asarray(afm.layers["DP"])
    assert (DP >= 1).all()
    assert (AD <= DP).all()
    assert np.allclose(afm.X.toarray(), AD / DP, atol=1e-6)


def test_coverage_is_defined_for_every_cell():
    """
    DP is the coverage of the SITE, not a per-variant depth masked to the cells with an
    alternative read: a cell covered 100x with no alt read is evidence of absence, and
    the genotyping needs that denominator. A simulator that zeroed DP there would hide
    exactly the bug this contract exists to prevent.
    """
    afm = mt.ut.simulate_afm(n_cells=300, n_clones=10)
    AD = afm.layers["AD"].toarray()
    DP = np.asarray(afm.layers["DP"])
    assert (DP > 0).all()
    assert ((DP > 0) & (AD == 0)).mean() > 0.1


def test_positive_cells_carry_more_alternative_reads_than_negative_ones():
    afm = mt.ut.simulate_afm(n_cells=400, n_clones=6, af_positive=0.05,
                             af_negative=0.001, frac_noisy_variants=0.0)
    G = afm.layers["genotype"].toarray().astype(bool)
    X = afm.X.toarray()
    assert X[G].mean() == pytest.approx(0.05, abs=0.01)
    assert X[~G].mean() == pytest.approx(0.001, abs=0.001)


def test_noisy_variants_sit_at_their_own_allele_frequency():
    """Noise is weak as well as tree-blind: its positive cells sit at
    af_positive_noise, not at the af_positive of a real clonal variant."""
    afm = mt.ut.simulate_afm(n_cells=500, n_clones=8, frac_noisy_variants=0.4,
                             af_positive=0.1, af_positive_noise=0.01,
                             af_negative=0.001, random_seed=0)
    G = afm.layers["genotype"].toarray().astype(bool)
    X = afm.X.toarray()
    noise = afm.var["is_noise"].values

    assert X[:, ~noise][G[:, ~noise]].mean() == pytest.approx(0.1, abs=0.02)
    assert X[:, noise][G[:, noise]].mean() == pytest.approx(0.01, abs=0.005)
    assert X[~G].mean() == pytest.approx(0.001, abs=0.001)


def test_noise_allele_frequency_is_what_separates_noise_from_signal():
    """The two components must be far enough apart for the AD-based variant
    filters to have something to discriminate on."""
    afm = mt.ut.simulate_afm(n_cells=500, n_clones=8, frac_noisy_variants=0.4,
                             random_seed=0)
    mt.pp.annotate_vars(afm, overwrite=True)
    noise = afm.var["is_noise"].values
    ad = afm.var["mean_AD_in_positives"].values

    assert ad[noise].max() < ad[~noise].min()


def test_af_beta_noise_makes_artefacts_weaker_than_clonal_variants():
    """Real artefacts are ~4x weaker than real clonal variants. Without a separate
    Beta the two are identical by construction, which makes the benchmark strictly
    harder than reality."""
    afm = mt.ut.simulate_afm(n_cells=500, n_clones=10, frac_noisy_variants=0.5,
                             af_beta=(2, 40), af_beta_noise=(2, 800), random_seed=0)
    p = np.array(afm.uns["simulation"]["p_positive"])
    noise = afm.var["is_noise"].values
    assert np.median(p[noise]) < np.median(p[~noise]) / 4


def test_af_beta_noise_defaults_to_the_worst_case():
    """Left unset, noise draws from the same Beta as signal."""
    afm = mt.ut.simulate_afm(n_cells=500, n_clones=10, frac_noisy_variants=0.5,
                             af_beta=(2, 40), random_seed=0)
    p = np.array(afm.uns["simulation"]["p_positive"])
    noise = afm.var["is_noise"].values
    assert np.median(p[noise]) == pytest.approx(np.median(p[~noise]), rel=0.6)


def test_af_beta_noise_requires_af_beta():
    with pytest.raises(ValueError, match="af_beta_noise"):
        mt.ut.simulate_afm(af_beta_noise=(2, 800))


def test_overdispersion_is_component_specific():
    """Measured on MAESTER data: carriers phi~6 (heteroplasmy drifts between cells
    of a clone), non-carriers phi~1.2 (sequencing error is a fixed-rate process).
    One factor for both makes the background far heavier-tailed than reality."""
    afm = mt.ut.simulate_afm(n_cells=800, n_clones=8, frac_noisy_variants=0.0,
                             mean_coverage=100, overdispersion=6.0,
                             overdispersion_background=1.2, random_seed=0)
    G = afm.layers["genotype"].toarray().astype(bool)
    AD = afm.layers["AD"].toarray()
    COV = np.asarray(afm.layers["DP"]).astype(float)

    def phi(ad, cov):
        p = ad.sum()/max(cov.sum(), 1e-9)
        return float(np.sum((ad-cov*p)**2/np.maximum(cov*p*(1-p), 1e-12))/(len(ad)-1))

    carr = np.median([phi(AD[G[:, j], j], COV[G[:, j], j])
                      for j in range(afm.n_vars) if G[:, j].sum() > 20])
    non = np.median([phi(AD[~G[:, j], j], COV[~G[:, j], j]) for j in range(afm.n_vars)])
    assert carr > 3.0                      # carriers overdispersed
    assert non < 2.0                       # background near-binomial
    assert carr > 2 * non


def test_overdispersion_noise_frac_keeps_artefacts_tight():
    """An artefact has no heteroplasmy to drift; at the full clonal dispersion it
    occasionally draws a high rate and erodes the count separation."""
    kw = dict(n_cells=800, n_clones=8, frac_noisy_variants=0.5, mean_coverage=100,
              overdispersion=6.0, overdispersion_background=1.2, random_seed=0)

    def noise_reads(frac):
        afm = mt.ut.simulate_afm(overdispersion_noise_frac=frac, **kw)
        G = afm.layers["genotype"].toarray().astype(bool)
        AD = afm.layers["AD"].toarray()
        n = afm.var["is_noise"].values
        return np.nanmean(np.where(G[:, n] & (AD[:, n] > 0), AD[:, n], np.nan))

    assert noise_reads(0.5) < noise_reads(1.0)


@pytest.mark.parametrize("kwargs,match", [
    ({"overdispersion_background": 0.5}, "overdispersion_background"),
    ({"overdispersion_noise_frac": 1.5}, "overdispersion_noise_frac"),
    ({"af_beta": (3, 57), "af_depth_decay": 0}, "af_depth_decay"),
])
def test_invalid_dispersion_parameters_are_rejected(kwargs, match):
    with pytest.raises(ValueError, match=match):
        mt.ut.simulate_afm(**kwargs)


def test_noise_at_the_same_af_is_indistinguishable_by_counts():
    """Guard on the parameter meaning: raising af_positive_noise to af_positive
    collapses the two components back into one."""
    afm = mt.ut.simulate_afm(n_cells=500, n_clones=8, frac_noisy_variants=0.4,
                             af_positive=0.1, af_positive_noise=0.1,
                             random_seed=0)
    G = afm.layers["genotype"].toarray().astype(bool)
    X = afm.X.toarray()
    noise = afm.var["is_noise"].values
    clonal_af = X[:, ~noise][G[:, ~noise]].mean()
    noise_af = X[:, noise][G[:, noise]].mean()

    assert noise_af == pytest.approx(clonal_af, abs=0.02)


def test_coverage_follows_the_requested_distribution():
    afm = mt.ut.simulate_afm(n_cells=400, n_clones=6, mean_coverage=50, sd_coverage=5)
    SC = np.asarray(afm.layers["DP"])
    assert SC.mean() == pytest.approx(50, abs=1)
    assert SC.std() == pytest.approx(5, abs=1)


def test_coverage_cvs_make_the_depth_heterogeneous_and_skewed():
    """Real target-site coverage varies between cells and between sites, and is
    right-skewed; the CVs switch the depth model from normal to gamma-Poisson."""
    flat = np.asarray(mt.ut.simulate_afm(n_cells=300, n_clones=8, mean_coverage=200,
                                         random_seed=0).layers["DP"])
    var = np.asarray(mt.ut.simulate_afm(n_cells=300, n_clones=8, mean_coverage=200,
                                        coverage_cell_cv=0.6, coverage_site_cv=0.8,
                                        random_seed=0).layers["DP"])
    assert var.std()/var.mean() > 4 * (flat.std()/flat.mean())
    assert np.median(var) < var.mean()                       # right-skewed
    assert var.mean(axis=1).std()/var.mean() > 0.3           # per-cell spread
    assert var.mean(axis=0).std()/var.mean() > 0.3           # per-site spread


def test_overdispersion_widens_counts_without_moving_their_mean():
    """NB: measure within carriers only. Across the whole matrix the carrier /
    non-carrier mixture dominates the variance and masks the effect."""
    kw = dict(n_cells=400, n_clones=8, frac_noisy_variants=0.0, mean_coverage=200,
              random_seed=0)

    def carrier_counts(od):
        afm = mt.ut.simulate_afm(overdispersion=od, **kw)
        G = afm.layers["genotype"].toarray().astype(bool)
        return afm.layers["AD"].toarray()[G]

    binom, bbinom = carrier_counts(1.0), carrier_counts(10.0)
    assert bbinom.mean() == pytest.approx(binom.mean(), rel=0.3)
    assert bbinom.var() > 2 * binom.var()


@pytest.mark.parametrize("kwargs,match", [
    ({"overdispersion": 0.5}, "overdispersion"),
    ({"coverage_cell_cv": -1}, "coverage_cell_cv"),
    ({"coverage_site_cv": -1}, "coverage_site_cv"),
])
def test_invalid_noise_parameters_are_rejected(kwargs, match):
    with pytest.raises(ValueError, match=match):
        mt.ut.simulate_afm(**kwargs)


def test_every_cell_carries_its_own_clone_variant():
    afm = mt.ut.simulate_afm(n_cells=300, n_clones=10, n_root_clones=7)
    X = afm.layers["genotype"].toarray()
    origin = list(afm.var["clone_of_origin"])
    for i, clone in enumerate(afm.obs["clone"].astype(str)):
        assert X[i, origin.index(clone)] == 1


def test_a_variant_is_carried_by_exactly_its_clade():
    afm = mt.ut.simulate_afm(n_cells=300, n_clones=10, n_root_clones=7,
                             frac_noisy_variants=0.0)
    X = afm.layers["genotype"].toarray()
    labels = afm.obs["clone"].astype(str).values
    for j, clade in enumerate(afm.var["clade"]):
        assert np.array_equal(X[:, j] > 0, np.isin(labels, clade.split(";")))


# -- clonal structure -------------------------------------------------------

@pytest.mark.parametrize("n_clones", [1, 2, 5, 20])
def test_n_clones(n_clones):
    afm = mt.ut.simulate_afm(n_cells=200, n_clones=n_clones, frac_noisy_variants=0.0)
    assert afm.obs["clone"].nunique() == n_clones
    assert afm.n_vars == n_clones


def test_pure_polytomy_shares_no_variant():
    afm = mt.ut.simulate_afm(n_cells=300, n_clones=10, n_root_clones=10,
                             frac_noisy_variants=0.0)
    assert (afm.var["n_clones_carrying"] == 1).all()
    assert (afm.layers["genotype"].toarray().sum(axis=1) == 1).all()
    assert afm.uns["simulation"]["n_lineage_splits"] == 0


@pytest.mark.parametrize("n_root_clones", [1, 5, 10, 20])
def test_n_root_clones_is_honoured(n_root_clones):
    n_clones = 20
    afm = mt.ut.simulate_afm(n_cells=300, n_clones=n_clones,
                             n_root_clones=n_root_clones, max_depth=8)
    parents = afm.uns["simulation"]["parent"]
    assert sum(1 for p in parents if p == "root") == n_root_clones
    assert afm.uns["simulation"]["n_lineage_splits"] == n_clones - n_root_clones


def test_lineage_splits_inherit_the_parent_variant():
    """A clone that split off another must carry its parent's variant too."""
    afm = mt.ut.simulate_afm(n_cells=300, n_clones=12, n_root_clones=6)
    parents = afm.uns["simulation"]["parent"]
    origin = list(afm.var["clone_of_origin"])
    X = afm.layers["genotype"].toarray()
    labels = afm.obs["clone"].astype(str).values

    split = [i for i, p in enumerate(parents) if p != "root"]
    assert split, "expected at least one lineage split"
    for i in split:
        cells = labels == f"clone_{i}"
        assert X[cells, origin.index(parents[i])].all()


@pytest.mark.parametrize("max_depth", [1, 2, 3, 4, 6])
def test_max_depth_is_reached_and_never_exceeded(max_depth):
    n_clones = 30
    afm = mt.ut.simulate_afm(n_cells=500, n_clones=n_clones, split_size=1,
                             n_root_clones=-(-n_clones // max_depth),
                             max_depth=max_depth)
    assert afm.uns["simulation"]["realised_depth"] == max_depth


def test_max_depth_one_forces_a_polytomy():
    afm = mt.ut.simulate_afm(n_cells=300, n_clones=12, n_root_clones=12,
                             max_depth=1, frac_noisy_variants=0.0)
    assert (afm.var["n_clones_carrying"] == 1).all()
    assert afm.uns["simulation"]["n_lineage_splits"] == 0


def test_impossible_depth_budget_is_rejected():
    """More clones than the depth cap can hold, whatever the splits."""
    with pytest.raises(ValueError, match="Cannot fit"):
        mt.ut.simulate_afm(n_cells=300, n_clones=16, n_root_clones=1,
                           split_size=1, max_depth=3)


def test_a_clone_carries_one_variant_per_level_of_its_depth():
    afm = mt.ut.simulate_afm(n_cells=500, n_clones=20, n_root_clones=4,
                             max_depth=4, frac_noisy_variants=0.0)
    X = afm.layers["genotype"].toarray()
    depth = dict(zip(afm.var["clone_of_origin"], afm.var["depth"]))
    for i, clone in enumerate(afm.obs["clone"].astype(str)):
        assert X[i].sum() == depth[clone]


def test_a_parent_sits_exactly_one_level_above_its_child():
    afm = mt.ut.simulate_afm(n_cells=500, n_clones=20, n_root_clones=4,
                             max_depth=4)
    s = afm.uns["simulation"]
    for i, p in enumerate(s["parent"]):
        if p != "root":
            assert s["depth"][int(p.split("_")[1])] == s["depth"][i] - 1
        else:
            assert s["depth"][i] == 1


def test_clades_are_nested():
    """Two clades either nest or are disjoint -- never partially overlap."""
    afm = mt.ut.simulate_afm(n_cells=300, n_clones=15, n_root_clones=6)
    clades = [set(c.split(";")) for c in afm.uns["simulation"]["clades"]]
    for a in clades:
        for b in clades:
            assert a <= b or b <= a or not (a & b)


@pytest.mark.parametrize("min_max_ratio", [1.0, 0.5, 0.1])
def test_min_max_ratio_clones(min_max_ratio):
    afm = mt.ut.simulate_afm(n_cells=1000, n_clones=4,
                             min_max_ratio_clones=min_max_ratio)
    sizes = afm.obs["clone"].value_counts().values
    assert sizes.min() / sizes.max() == pytest.approx(min_max_ratio, abs=0.05)


def test_all_cells_are_assigned():
    afm = mt.ut.simulate_afm(n_cells=317, n_clones=7)
    assert afm.obs["clone"].notna().all()
    assert sum(afm.uns["simulation"]["clone_sizes"]) == 317


# -- reproducibility and validation -----------------------------------------

def test_is_reproducible():
    a = mt.ut.simulate_afm(n_cells=100, n_clones=6, random_seed=7)
    b = mt.ut.simulate_afm(n_cells=100, n_clones=6, random_seed=7)
    assert np.array_equal(a.X.toarray(), b.X.toarray())
    assert (a.obs["clone"].values == b.obs["clone"].values).all()


def test_different_seeds_give_different_trees():
    trees = {
        tuple(mt.ut.simulate_afm(n_cells=100, n_clones=12, n_root_clones=4,
                                 random_seed=s).uns["simulation"]["parent"])
        for s in range(10)
    }
    assert len(trees) > 1


@pytest.mark.parametrize("kwargs,match", [
    ({"n_clones": 0}, "n_clones"),
    ({"n_cells": 5, "n_clones": 10}, "n_cells"),
    ({"n_clones": 5, "n_root_clones": 9}, "n_root_clones"),
    ({"n_root_clones": 0}, "n_root_clones"),
    ({"min_max_ratio_clones": 0}, "min_max_ratio_clones"),
    ({"min_max_ratio_clones": 2}, "min_max_ratio_clones"),
    ({"max_depth": 0}, "max_depth"),
    ({"af_positive_noise": 1.5}, "af_positive_noise"),
    ({"af_positive_noise": -0.1}, "af_positive_noise"),
])
def test_invalid_parameters_are_rejected(kwargs, match):
    with pytest.raises(ValueError, match=match):
        mt.ut.simulate_afm(**kwargs)


def test_too_many_clones_for_the_target_sites():
    with pytest.raises(ValueError, match="target sites"):
        mt.ut.simulate_afm(n_cells=200_000, n_clones=100_000)


# -- molecule-level draw ----------------------------------------------------

def _carrier_stats(afm):
    """Per-variant dispersion, median coverage and dropout, over true carriers."""
    AD = afm.layers["AD"].toarray()
    COV = np.asarray(afm.layers["DP"]).astype(float)
    G = afm.layers["genotype"].toarray().astype(bool)
    phi, cov, drop = [], [], []
    for j in range(afm.shape[1]):
        m = G[:, j]
        if m.sum() < 20:
            continue
        ad, cv = AD[m, j], COV[m, j]
        p = ad.sum() / max(cv.sum(), 1e-9)
        if not 0 < p < 1:
            continue
        phi.append(float(((ad - cv * p) ** 2 / np.maximum(cv * p * (1 - p), 1e-12)).mean()))
        cov.append(float(np.median(cv)))
        drop.append(float((ad == 0).mean()))
    return np.array(phi), np.array(cov), np.array(drop)


MOL = dict(
    n_cells=1500, n_clones=20, n_root_clones=20, max_depth=1, frac_noisy_variants=0.2,
    af_beta=(3, 57), mean_coverage=200, coverage_cell_cv=0.2, coverage_site_cv=0.8,
    mean_molecules=40, molecule_cv=0.5, random_seed=0,
)


def test_mean_molecules_is_opt_in():
    """Leaving it None must not perturb the legacy beta-binomial draw."""
    kw = dict(n_cells=400, n_clones=8, overdispersion=4.0, random_seed=0)
    a = mt.ut.simulate_afm(**kw)
    b = mt.ut.simulate_afm(mean_molecules=None, **kw)
    assert (a.layers["AD"].toarray() == b.layers["AD"].toarray()).all()


def test_molecule_draw_produces_carrier_dropout_without_a_dropout_parameter():
    """
    A carrier that captures no ALT molecule reads as zero however deeply it is
    sequenced. Dropout is a consequence of the draw, not a calibrated target.
    """
    _, _, drop = _carrier_stats(mt.ut.simulate_afm(**MOL))
    assert 0.2 < drop.mean() < 0.55


def test_molecule_draw_makes_dispersion_grow_with_coverage():
    """
    Regression on the whole point of the model: reads are redundant copies of few
    molecules, so effective sample size does not track read depth and phi rises
    with coverage. The beta-binomial draw pins it flat by construction.
    """
    phi, cov, _ = _carrier_stats(mt.ut.simulate_afm(**MOL))
    slope = np.polyfit(np.log10(cov), np.log10(np.maximum(phi, 1e-3)), 1)[0]
    assert slope > 0.35, f"phi should grow with coverage, got slope {slope:.2f}"


def test_molecule_coverage_exponent_flattens_the_relation():
    """e = 1 ties molecules to reads, which removes the coverage dependence."""
    phi0, cov0, _ = _carrier_stats(mt.ut.simulate_afm(**MOL))
    kw = dict(MOL); kw["molecule_coverage_exponent"] = 1.0
    phi1, cov1, _ = _carrier_stats(mt.ut.simulate_afm(**kw))
    s0 = np.polyfit(np.log10(cov0), np.log10(np.maximum(phi0, 1e-3)), 1)[0]
    s1 = np.polyfit(np.log10(cov1), np.log10(np.maximum(phi1, 1e-3)), 1)[0]
    assert s1 < s0


def test_fewer_molecules_means_more_dispersion_and_more_dropout():
    """Concentration is the molecule count, so lowering it does both at once."""
    lo = _carrier_stats(mt.ut.simulate_afm(**{**MOL, "mean_molecules": 15}))
    hi = _carrier_stats(mt.ut.simulate_afm(**{**MOL, "mean_molecules": 120}))
    assert np.median(lo[0]) > np.median(hi[0])
    assert lo[2].mean() > hi[2].mean()


def test_molecule_draw_leaves_the_background_near_binomial():
    """Sequencing error is read-level; it has no molecules and must stay binomial."""
    afm = mt.ut.simulate_afm(**MOL)
    AD = afm.layers["AD"].toarray()
    COV = np.asarray(afm.layers["DP"]).astype(float)
    G = afm.layers["genotype"].toarray().astype(bool)
    ad, cv = AD[~G], COV[~G]
    p = ad.sum() / cv.sum()
    phi = float(((ad - cv * p) ** 2 / np.maximum(cv * p * (1 - p), 1e-12)).mean())
    assert phi < 2.0, f"background should be near-binomial, got phi={phi:.2f}"


def test_molecule_draw_records_its_parameters():
    uns = mt.ut.simulate_afm(**MOL).uns["simulation"]
    assert uns["mean_molecules"] == 40
    assert uns["molecule_cv"] == 0.5
    assert uns["molecule_coverage_exponent"] == 0.0


@pytest.mark.parametrize("kwargs,match", [
    ({"mean_molecules": 0}, "mean_molecules"),
    ({"mean_molecules": -5}, "mean_molecules"),
    ({"molecule_cv": -0.1}, "molecule_cv"),
    ({"molecule_coverage_exponent": 1.5}, "molecule_coverage_exponent"),
    ({"molecule_coverage_exponent": -0.1}, "molecule_coverage_exponent"),
])
def test_invalid_molecule_parameters_are_rejected(kwargs, match):
    with pytest.raises(ValueError, match=match):
        mt.ut.simulate_afm(**kwargs)
