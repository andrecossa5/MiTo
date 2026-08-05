"""
mito.pp.filter_afm

The widest parameter surface in the package (22 arguments). Covers every
variant-filtering strategy, both binarisation methods, the distance metrics,
lineage-based cell filtering, explicit variant selection, and the structural
invariants the filtered AFM must satisfy.
"""

import numpy as np
import pandas as pd
import pytest
import mito as mt

from conftest import build_afm, build_rich_annotated


# Every variant-filtering strategy, as advertised in mito.ut.utils._var_filters.
FILTERING_METHODS = ["CV", "MiTo", "MQuad", "weng2024", "miller2022"]
BIN_METHODS = ["MiTo", "vanilla"]
CONTINUOUS_METRICS = ["correlation", "cosine", "euclidean"]
DISCRETE_METRICS = ["jaccard", "weighted_jaccard", "dice"]


def _annotated(**kwargs):
    a = build_afm(**kwargs)
    mt.pp.annotate_vars(a)
    return a


def _filter(a, **kwargs):
    """filter_afm with the slow/optional machinery off unless asked for."""
    kwargs.setdefault("compute_enrichment", False)
    kwargs.setdefault("ncores", 1)
    return mt.pp.filter_afm(a, **kwargs)


# -- structural contract ----------------------------------------------------

def test_returns_filtered_anndata(afm_annotated):
    out = _filter(afm_annotated)
    assert out.shape[0] > 0 and out.shape[1] > 0
    assert out.shape[1] <= 40


def test_layers_survive_filtering(afm_annotated):
    out = _filter(afm_annotated)
    for layer in ("AD", "DP"):
        assert layer in out.layers
        assert out.layers[layer].shape == out.shape


def test_adds_binary_layer(afm_annotated):
    out = _filter(afm_annotated)
    assert "bin" in out.layers
    values = np.unique(out.layers["bin"].toarray())
    assert set(values).issubset({0, 1})


def test_records_provenance_in_uns(afm_annotated):
    out = _filter(afm_annotated)
    for key in ("char_filter", "genotyping", "distance_calculations"):
        assert key in out.uns


def test_computes_distances(afm_annotated):
    out = _filter(afm_annotated)
    assert "distances" in out.obsp
    D = out.obsp["distances"].toarray()
    assert D.shape == (out.shape[0], out.shape[0])
    assert np.allclose(D, D.T, atol=1e-6), "distance matrix must be symmetric"
    assert np.allclose(np.diag(D), 0, atol=1e-6), "diagonal must be zero"


def test_subsets_original_variants(afm_annotated):
    original = set(afm_annotated.var_names)
    out = _filter(afm_annotated)
    assert set(out.var_names).issubset(original)


# -- filtering strategies ---------------------------------------------------

@pytest.mark.parametrize("filtering", FILTERING_METHODS)
def test_every_filtering_method_with_defaults(filtering, tmp_path, monkeypatch):
    """
    Each strategy must run to completion on a well-formed AFM using only its own
    default thresholds -- no per-strategy tuning.
    """
    monkeypatch.chdir(tmp_path)              # MQuad writes BIC files to cwd
    out = _filter(build_rich_annotated(), filtering=filtering)
    assert out.shape[0] > 0 and out.shape[1] > 0
    assert out.uns["char_filter"]["filtering"] == filtering
    assert "bin" in out.layers
    assert "distances" in out.obsp


@pytest.mark.parametrize("filtering", ["baseline", "CV", "MiTo"])
def test_lightweight_strategies_on_default_afm(filtering):
    a = _annotated(seed=31)
    out = _filter(a, filtering=filtering)
    assert out.shape[1] >= 1
    assert out.uns["char_filter"]["filtering"] == filtering


@pytest.mark.parametrize("filtering", ["miller2022", "weng2024"])
def test_literature_strategies_on_rich_input(filtering):
    """
    These strategies select rare, high-VAF variants: weng2024 in particular wants
    >=90% negative cells per variant, so the input needs many small clones rather
    than a few large ones.
    """
    a = _annotated(n_cells=200, n_vars=60, n_clones=25, clone_specific_frac=0.8,
                   coverage=100, seed=61)
    out = _filter(a, filtering=filtering)
    assert out.shape[1] >= 1


def test_mquad_strategy(tmp_path, monkeypatch):
    """
    MQuad writes BIC files into the working directory, so it is run in a tmpdir.
    It either selects variants or explains why it could not pick a cutoff.
    """
    monkeypatch.chdir(tmp_path)
    a = _annotated(n_cells=120, n_vars=40, n_clones=6, coverage=80,
                   clone_specific_frac=0.5, seed=65)
    try:
        out = _filter(a, filtering="MQuad")
    except ValueError as err:
        assert any(s in str(err) for s in
                   ("deltaBIC cutoff", "no cells left", "retained no MT-SNVs", "min_n_var"))
    else:
        assert out.shape[1] >= 1


def test_mquad_small_input_explains_cutoff_failure(tmp_path, monkeypatch):
    """Regression: this used to surface as `'>' not supported between float and NoneType`."""
    monkeypatch.chdir(tmp_path)
    a = _annotated(n_cells=40, n_vars=15, seed=32)
    with pytest.raises(ValueError) as excinfo:
        _filter(a, filtering="MQuad")
    assert "deltaBIC cutoff" in str(excinfo.value) or "no cells left" in str(excinfo.value)


@pytest.mark.parametrize("filtering", ["miller2022", "weng2024"])
def test_stringent_strategies_report_empty_selection(filtering):
    """When a strategy keeps nothing, the error must say so intelligibly."""
    a = _annotated(n_cells=25, n_vars=8, coverage=8, seed=62)
    with pytest.raises(ValueError, match="retained no MT-SNVs"):
        _filter(a, filtering=filtering)


def test_unknown_filtering_raises():
    a = _annotated(seed=33)
    with pytest.raises((ValueError, KeyError, AssertionError)):
        _filter(a, filtering="not_a_strategy")


def test_every_filtering_strategy_is_covered():
    """Guard: a new strategy in _var_filters must get an explicit test."""
    from mito.ut.utils import _var_filters
    covered = {"baseline", "CV", "miller2022", "weng2024", "MQuad", "MiTo", "GT_enriched"}
    assert set(_var_filters) - covered == set()


@pytest.mark.parametrize("filtering", ["baseline", "CV", "MiTo"])
def test_strategies_are_deterministic(filtering):
    """Same input and seed must give the same variant set."""
    v1 = _filter(_annotated(seed=34), filtering=filtering).var_names
    v2 = _filter(_annotated(seed=34), filtering=filtering).var_names
    assert list(v1) == list(v2)


# -- explicit variant selection ---------------------------------------------

def test_explicit_variants_need_the_custom_branch():
    """`variants` is only honoured with filtering=None; the baseline filter still applies."""
    a = _annotated(seed=35)
    wanted = list(a.var_names[:8])
    out = _filter(a, filtering=None, variants=wanted)
    assert set(out.var_names).issubset(set(wanted))


def test_explicit_variants_ignored_by_named_strategies():
    """With a named strategy the argument is silently unused -- documented here."""
    a = _annotated(seed=36)
    wanted = list(a.var_names[:3])
    out = _filter(a, filtering="baseline", variants=wanted)
    assert out.shape[1] >= 1


def test_variants_marked_as_predefined_in_uns():
    a = _annotated(seed=36)
    out = _filter(a, filtering=None, variants=list(a.var_names))
    assert out.uns["char_filter"]["filtering"] == "predefined_sets"


@pytest.mark.parametrize("min_n_var", [1, 2])
def test_min_n_var(min_n_var):
    a = _annotated(seed=37)
    out = _filter(a, min_n_var=min_n_var)
    assert out.shape[1] >= 1


def test_min_n_var_too_high_raises_clearly():
    a = _annotated(seed=37)
    with pytest.raises(ValueError, match="min_n_var"):
        _filter(a, min_n_var=1000)


# -- binarisation -----------------------------------------------------------

@pytest.mark.parametrize("bin_method", BIN_METHODS)
def test_binarisation_methods(bin_method):
    a = _annotated(seed=38)
    out = _filter(a, bin_method=bin_method)
    assert "bin" in out.layers
    assert set(np.unique(out.layers["bin"].toarray())).issubset({0, 1})
    assert out.uns["genotyping"]["bin_method"] == bin_method


@pytest.mark.parametrize("bin_method", BIN_METHODS)
@pytest.mark.parametrize("metric", CONTINUOUS_METRICS + DISCRETE_METRICS)
def test_genotyping_by_metric_matrix(bin_method, metric):
    """Every genotyping method must combine with every supported metric."""
    a = _annotated(seed=68)
    out = _filter(a, bin_method=bin_method, metric=metric)
    D = out.obsp["distances"].toarray()
    assert D.shape == (out.shape[0], out.shape[0])
    assert np.allclose(D, D.T, atol=1e-6)
    assert (D >= 0).all()
    assert out.uns["distance_calculations"]["distances"]["metric"] == metric


@pytest.mark.parametrize("filtering", ["CV", "MiTo", "weng2024", "miller2022"])
@pytest.mark.parametrize("bin_method", BIN_METHODS)
def test_filtering_by_genotyping_matrix(filtering, bin_method):
    """Filtering strategy and genotyping method must be independently selectable."""
    out = _filter(build_rich_annotated(), filtering=filtering, bin_method=bin_method)
    assert out.shape[1] >= 1
    assert out.uns["char_filter"]["filtering"] == filtering
    assert out.uns["genotyping"]["bin_method"] == bin_method


def test_bin_method_changes_genotypes():
    """The two binarisation strategies must not produce identical output."""
    a1 = _filter(_annotated(seed=39), bin_method="vanilla")
    a2 = _filter(_annotated(seed=39), bin_method="MiTo")
    same_shape = a1.shape == a2.shape
    if same_shape:
        assert not np.array_equal(a1.layers["bin"].toarray(), a2.layers["bin"].toarray())
    else:
        assert True    # differing cell/variant counts is itself a difference


@pytest.mark.parametrize("max_AD_counts", [1, 2, 5, 20])
def test_max_AD_counts_range(max_AD_counts):
    a = _annotated(seed=40)
    out = _filter(a, max_AD_counts=max_AD_counts)
    assert out.shape[1] >= 1


# -- MiTo hyper-parameters --------------------------------------------------
# The MiTo filter and the MiTo genotyping method are the package defaults, so
# their tunables are swept across plausible ranges rather than spot-checked.

@pytest.mark.parametrize("min_n_positive", [2, 5, 10])
def test_mito_filter_min_n_positive(min_n_positive):
    out = _filter(build_rich_annotated(), filtering="MiTo",
                  filtering_kwargs={"min_n_positive": min_n_positive})
    assert out.shape[1] >= 1


@pytest.mark.parametrize("af_confident_detection", [0.01, 0.05, 0.1])
def test_mito_filter_af_confident_detection(af_confident_detection):
    out = _filter(build_rich_annotated(), filtering="MiTo",
                  filtering_kwargs={"af_confident_detection": af_confident_detection})
    assert out.shape[1] >= 1


def test_mito_filter_stricter_thresholds_keep_fewer_variants():
    """Tightening the MiTo filter must be monotone in the number of variants."""
    counts = []
    for n in [2, 8, 20]:
        out = _filter(build_rich_annotated(), filtering="MiTo",
                      filtering_kwargs={"min_n_positive": n})
        counts.append(out.shape[1])
    assert counts == sorted(counts, reverse=True)


@pytest.mark.parametrize("t_prob", [0.5, 0.7, 0.9])
def test_mito_genotyping_t_prob(t_prob):
    out = _filter(build_rich_annotated(), bin_method="MiTo",
                  binarization_kwargs={"t_prob": t_prob})
    assert set(np.unique(out.layers["bin"].toarray())).issubset({0, 1})


@pytest.mark.parametrize("min_AD", [1, 2, 5])
def test_mito_genotyping_min_AD(min_AD):
    out = _filter(build_rich_annotated(), bin_method="MiTo",
                  binarization_kwargs={"min_AD": min_AD})
    assert out.shape[1] >= 1


@pytest.mark.parametrize("min_cell_prevalence", [0.01, 0.05, 0.1, 0.5])
def test_mito_genotyping_min_cell_prevalence(min_cell_prevalence):
    """Switches between probabilistic and hard-threshold calling per variant."""
    out = _filter(build_rich_annotated(), bin_method="MiTo",
                  binarization_kwargs={"min_cell_prevalence": min_cell_prevalence})
    assert out.shape[1] >= 1


@pytest.mark.parametrize("t_vanilla", [0.0, 0.01, 0.05])
def test_vanilla_genotyping_thresholds(t_vanilla):
    out = _filter(build_rich_annotated(), bin_method="vanilla",
                  binarization_kwargs={"t_vanilla": t_vanilla, "min_AD": 1})
    assert out.shape[1] >= 1


@pytest.mark.parametrize("min_frac_negative", [0.1, 0.2, 0.5])
def test_mito_filter_min_frac_negative(min_frac_negative):
    out = _filter(build_rich_annotated(), filtering="MiTo",
                  filtering_kwargs={"min_frac_negative": min_frac_negative})
    assert out.shape[1] >= 1


@pytest.mark.parametrize("min_mean_DP_in_positives", [5, 25, 60])
def test_mito_filter_min_mean_DP(min_mean_DP_in_positives):
    out = _filter(build_rich_annotated(), filtering="MiTo",
                  filtering_kwargs={"min_mean_DP_in_positives": min_mean_DP_in_positives})
    assert out.shape[1] >= 1


@pytest.mark.parametrize("n_top", [5, 20, 100])
def test_cv_filter_n_top(n_top):
    """filter_CV keeps the n_top most variable MT-SNVs."""
    out = _filter(build_rich_annotated(), filtering="CV",
                  filtering_kwargs={"n_top": n_top})
    assert 1 <= out.shape[1] <= max(n_top, 1)


@pytest.mark.parametrize("min_site_cov", [10, 50, 100])
def test_miller2022_min_site_cov(min_site_cov):
    out = _filter(build_rich_annotated(), filtering="miller2022",
                  filtering_kwargs={"min_site_cov": min_site_cov})
    assert out.shape[1] >= 1


@pytest.mark.parametrize("min_n_positive", [2, 5])
def test_weng2024_min_n_positive(min_n_positive):
    out = _filter(build_rich_annotated(), filtering="weng2024",
                  filtering_kwargs={"min_n_positive": min_n_positive})
    assert out.shape[1] >= 1


# -- custom obs / var sets --------------------------------------------------

def test_custom_cells_and_variants_together():
    """The documented escape hatch: hand-picked cells and variants."""
    a = build_rich_annotated()
    cells = list(a.obs_names[:150])
    variants = list(a.var_names[:40])
    out = _filter(a, filtering=None, cells=cells, variants=variants)
    assert set(out.obs_names).issubset(set(cells))
    assert set(out.var_names).issubset(set(variants))
    assert out.uns["char_filter"]["filtering"] == "predefined_sets"


def test_custom_sets_survive_the_full_pipeline():
    a = build_rich_annotated()
    out = _filter(a, filtering=None,
                  cells=list(a.obs_names[:200]), variants=list(a.var_names[:50]))
    assert "bin" in out.layers
    assert "distances" in out.obsp
    D = out.obsp["distances"].toarray()
    assert np.allclose(D, D.T, atol=1e-6)


@pytest.mark.parametrize("n_variants", [10, 30, 60])
def test_custom_variant_set_sizes(n_variants):
    a = build_rich_annotated()
    out = _filter(a, filtering=None, variants=list(a.var_names[:n_variants]))
    assert out.shape[1] <= n_variants


def test_custom_cells_only():
    a = build_rich_annotated()
    cells = list(a.obs_names[:180])
    out = _filter(a, filtering=None, cells=cells, variants=list(a.var_names))
    assert set(out.obs_names).issubset(set(cells))


# -- boolean toggles --------------------------------------------------------
# Each flag must be selectable in both states and mean what the docstring says.

@pytest.mark.parametrize("compute_enrichment", [True, False])
def test_toggle_compute_enrichment(compute_enrichment):
    a = build_rich_annotated()
    out = mt.pp.filter_afm(a, lineage_column="GBC", ncores=1,
                           compute_enrichment=compute_enrichment)
    assert out.uns["char_filter"]["compute_enrichment"] is compute_enrichment


@pytest.mark.parametrize("filter_dbs", [True, False])
def test_toggle_filter_dbs(filter_dbs):
    out = _filter(build_rich_annotated(), filter_dbs=filter_dbs)
    assert out.uns["char_filter"]["filter_dbs"] is filter_dbs
    if filter_dbs:
        assert not np.isnan(out.uns["char_filter"]["n_dbSNP"])
    else:
        assert np.isnan(out.uns["char_filter"]["n_dbSNP"])


@pytest.mark.parametrize("filter_moran", [True, False])
def test_toggle_filter_moran(filter_moran):
    out = _filter(build_rich_annotated(), filter_moran=filter_moran)
    assert out.uns["char_filter"]["filter_moran"] is filter_moran


@pytest.mark.parametrize("spatial_metrics", [True, False])
def test_toggle_spatial_metrics(spatial_metrics):
    out = _filter(build_rich_annotated(), spatial_metrics=spatial_metrics)
    assert out.uns["char_filter"]["spatial_metrics"] is spatial_metrics
    if spatial_metrics:
        assert "average_degree" in out.uns["dataset_metrics"]


@pytest.mark.parametrize("fit_mixtures,only_positive_deltaBIC",
                         [(True, False), (True, True)])
def test_toggle_mixture_fitting(fit_mixtures, only_positive_deltaBIC, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    out = _filter(build_rich_annotated(), fit_mixtures=fit_mixtures,
                  only_positive_deltaBIC=only_positive_deltaBIC)
    assert "deltaBIC" in out.var.columns
    if only_positive_deltaBIC:
        assert (out.var["deltaBIC"] > 0).all()


# -- distance metrics -------------------------------------------------------

@pytest.mark.parametrize(
    "metric",
    ["weighted_jaccard", "jaccard", "dice",
     "russellrao", "sokalsneath", "yule", "rogerstanimoto"],
)
def test_discrete_metrics(metric):
    """All advertised discrete metrics must work, including scipy's boolean ones."""
    a = _annotated(seed=41)
    out = _filter(a, metric=metric)
    D = out.obsp["distances"].toarray()
    assert D.shape == (out.shape[0], out.shape[0])
    assert np.allclose(D, D.T, atol=1e-6)
    assert (D >= 0).all()
    assert np.allclose(np.diag(D), 0, atol=1e-6)


@pytest.mark.parametrize(
    "metric",
    ["euclidean", "cosine", "correlation", "cityblock", "l1", "l2",
     "manhattan", "sqeuclidean", "nan_euclidean"],
)
def test_continuous_metrics(metric):
    """Every usable continuous metric must produce a valid distance matrix."""
    a = _annotated(seed=41)
    out = _filter(a, metric=metric)
    D = out.obsp["distances"].toarray()
    assert D.shape == (out.shape[0], out.shape[0])
    assert np.allclose(D, D.T, atol=1e-6)
    assert (D >= 0).all()


def test_matching_metric_is_not_available():
    """
    'matching' was a deprecated alias for 'hamming' and was removed from SciPy,
    so MiTo rejects it; 'weighted_hamming' is the supported relative.
    """
    a = _annotated(seed=41)
    with pytest.raises(ValueError, match="not a valid metric"):
        _filter(a, metric="matching")


@pytest.mark.parametrize("metric", ["haversine", "precomputed"])
def test_structurally_invalid_metrics_raise(metric):
    """
    These come from sklearn's metric registry but cannot apply to an AFM:
    haversine needs 2-D coordinates, precomputed needs a square input.
    """
    a = _annotated(seed=41)
    with pytest.raises(ValueError):
        _filter(a, metric=metric)


def test_every_advertised_metric_is_covered():
    """Guard: if MiTo gains a metric, this test forces a decision about it."""
    from mito.pp.distances import discrete_metrics, continuous_metrics
    covered = {
        "weighted_jaccard", "weighted_hamming", "jaccard", "dice", "russellrao",
        "sokalsneath", "yule", "rogerstanimoto", "euclidean", "cosine",
        "correlation", "cityblock", "l1", "l2", "manhattan", "sqeuclidean",
        "nan_euclidean", "haversine", "precomputed",
    }
    advertised = set(discrete_metrics) | set(continuous_metrics)
    assert advertised - covered == set(), f"untested metrics: {advertised - covered}"


def test_invalid_metric_raises():
    a = _annotated(seed=42)
    with pytest.raises(ValueError):
        _filter(a, metric="definitely_not_a_metric")


def test_weighted_hamming_without_priors_explains_itself():
    """MAESTER AFMs carry no per-character priors; the error must say so."""
    a = _annotated(seed=42)
    with pytest.raises(ValueError, match="priors"):
        _filter(a, metric="weighted_hamming")


# -- lineage-aware filtering ------------------------------------------------

@pytest.mark.parametrize("min_cell_number", [0, 5, 20])
def test_min_cell_number_with_lineage_column(min_cell_number):
    a = _annotated(seed=43)
    out = _filter(a, lineage_column="GBC", min_cell_number=min_cell_number)
    if min_cell_number > 0:
        counts = out.obs["GBC"].value_counts()
        assert (counts[counts > 0] >= min_cell_number).all()


def test_min_cell_number_too_large_raises_clearly():
    """Filtering away every cell must report why, not fail with a ZeroDivisionError."""
    a = _annotated(seed=44)
    with pytest.raises(ValueError, match="no cells left"):
        _filter(a, lineage_column="GBC", min_cell_number=10_000)


def test_cells_argument_needs_the_custom_branch():
    a = _annotated(seed=45)
    wanted = list(a.obs_names[:60])
    out = _filter(a, filtering=None, cells=wanted, variants=list(a.var_names))
    assert set(out.obs_names).issubset(set(wanted))


def test_gt_enriched_requires_lineage_column():
    a = _annotated(seed=46)
    out = _filter(a, filtering="GT_enriched", lineage_column="GBC")
    assert out.shape[1] >= 1


# -- optional machinery -----------------------------------------------------

def test_compute_enrichment(afm_annotated):
    out = _filter(afm_annotated, lineage_column="GBC", compute_enrichment=True)
    assert out.shape[1] >= 1


@pytest.mark.parametrize("filter_dbs", [True, False])
def test_filter_dbs_toggle(filter_dbs):
    """Database filtering uses the packaged dbSNP/REDIdb assets."""
    a = _annotated(seed=47)
    out = _filter(a, filter_dbs=filter_dbs)
    assert out.shape[1] >= 1


@pytest.mark.parametrize("filter_moran,moran_I_pvalue", [(False, 0.01), (True, 0.5)])
def test_moran_filter(filter_moran, moran_I_pvalue):
    a = _annotated(seed=48)
    out = _filter(a, filter_moran=filter_moran, moran_I_pvalue=moran_I_pvalue)
    assert out.shape[1] >= 1


def test_return_tree_returns_a_tree():
    """return_tree=True must yield an actual tree, not None."""
    a = _annotated(seed=49)
    result = _filter(a, return_tree=True)
    assert isinstance(result, tuple) and len(result) == 2
    afm_out, tree = result
    assert tree is not None
    assert hasattr(tree, "leaves")
    assert len(tree.leaves) == afm_out.shape[0]


def test_return_tree_false_returns_bare_anndata():
    from anndata import AnnData
    a = _annotated(seed=49)
    out = _filter(a, return_tree=False)
    assert isinstance(out, AnnData)


# -- edge conditions --------------------------------------------------------

def test_small_input():
    a = _annotated(n_cells=35, n_vars=12, n_clones=2, clone_specific_frac=0.6, seed=50)
    out = _filter(a, filtering="baseline")
    assert out.shape[1] >= 1


def test_tiny_input_fails_intelligibly():
    """Too little data to filter must produce a readable error, not a stack trace."""
    a = _annotated(n_cells=8, n_vars=4, n_clones=2, clone_specific_frac=0.0, seed=63)
    with pytest.raises(ValueError, match="no cells left|retained no MT-SNVs|min_n_var"):
        _filter(a, filtering="baseline")


def test_no_clonal_structure():
    """Pure-noise input either yields a well-formed AFM or a clear error."""
    a = _annotated(n_cells=30, n_vars=12, n_clones=1, clone_specific_frac=0.0, seed=51)
    try:
        out = _filter(a, filtering="baseline")
    except ValueError as err:
        assert "retained no MT-SNVs" in str(err) or "min_n_var" in str(err)
    else:
        assert "distances" in out.obsp


def test_median_af_in_positives_is_not_constant():
    """Regression: this column previously averaged the positivity mask, always giving 1."""
    a = _annotated(n_cells=60, n_vars=25, seed=71)
    values = a.var["median_af_in_positives"].dropna().values
    assert len(np.unique(np.round(values, 6))) > 1
    assert ((values > 0) & (values <= 1)).all()


def test_high_signal_input():
    a = _annotated(n_cells=45, n_vars=20, n_clones=3, clone_specific_frac=0.9, seed=52)
    out = _filter(a, filtering="MiTo")
    assert out.shape[1] >= 1


@pytest.mark.parametrize("coverage", [10, 30, 100])
def test_across_coverage_levels(coverage):
    a = _annotated(coverage=coverage, seed=53)
    out = _filter(a, filtering="baseline")
    assert out.shape[1] >= 1
