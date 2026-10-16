"""
mito.ut.extract_kwargs

The helper the nf-MiTo pipeline scripts use to turn command-line arguments (or a tuning table)
into the arguments of `filter_afm` and `build_tree`. What it returns must be accepted by them.
"""

from argparse import Namespace

import pandas as pd
import pytest
from conftest import build_afm

import mito as mt

# The command line of nf-MiTo's afm_preprocess/MiTo.py, including `k` and `gamma`, the options of
# the removed "MiTo_smooth" method, which older command lines and tuning tables still carry.
CLI = dict(
    path_tuning=None, job_id=None, cell_filter="filter2", min_cell_number=0, lineage_column=None,
    filtering="MiTo", bin_method="MiTo", min_n_var=1, filter_dbs="false", ncores=1,
    metric="weighted_jaccard", spatial_metrics="false", filter_moran="false",
    min_cov=5, min_var_quality=30, min_frac_negative=.2, min_n_positive=5,
    af_confident_detection=.05, min_n_confidently_detected=2, min_mean_AD_in_positives=1.,
    min_mean_DP_in_positives=25., t_prob=.7, t_vanilla=0., min_AD=1, min_cell_prevalence=.05,
    k=5, gamma=.25, solver="UPMGA",
)
SMOOTHING = {"k", "gamma", "resample"}


def test_cli_binarization_kwargs_are_those_of_the_genotyping_methods():
    _, _, _, binarization_kwargs, tree_kwargs = mt.ut.extract_kwargs(Namespace(**CLI))
    assert set(binarization_kwargs) == {"t_prob", "t_vanilla", "min_AD", "min_cell_prevalence"}
    assert tree_kwargs == {"solver": "UPMGA", "metric": "weighted_jaccard"}


def test_cli_without_the_smoothing_options_is_accepted():
    cli = {k: v for k, v in CLI.items() if k not in SMOOTHING}
    _, _, _, binarization_kwargs, _ = mt.ut.extract_kwargs(Namespace(**cli))
    assert not SMOOTHING & set(binarization_kwargs)


@pytest.mark.parametrize("with_smoothing", [True, False])
def test_tuning_table_branch(tmp_path, with_smoothing):
    options = {
        "cell_filter": "filter2", "min_cell_number": 0, "lineage_column": "GBC", "filtering": "MiTo",
        "bin_method": "MiTo", "metric": "weighted_jaccard", "min_n_var": 1, "filter_dbs": "false",
        "filter_moran": "false", "min_cov": 5, "min_var_quality": 30, "min_frac_negative": .2,
        "min_n_positive": 5, "af_confident_detection": .05, "min_n_confidently_detected": 2,
        "min_mean_AD_in_positives": 1., "min_mean_DP_in_positives": 25., "t_prob": .7,
        "t_vanilla": 0., "min_AD": 1, "min_cell_prevalence": .05, "solver": "UPMGA",
    }
    if with_smoothing:
        options.update(k=5, gamma=.25)
    table = pd.DataFrame({"job_id": "job1", "option": list(options), "value": list(options.values())})
    table.to_csv(tmp_path / "all_options_final.csv", index=False)
    args = Namespace(path_tuning=str(tmp_path), job_id="job1", ncores=1, spatial_metrics="false")
    _, kwargs, _, binarization_kwargs, tree_kwargs = mt.ut.extract_kwargs(args)
    assert not SMOOTHING & set(binarization_kwargs)
    assert kwargs["bin_method"] == "MiTo" and tree_kwargs["solver"] == "UPMGA"


@pytest.mark.parametrize("filtering", ["MiTo", None])
def test_what_it_returns_runs_through_filter_afm(filtering):
    """The call nf-MiTo makes: filter_afm(**the helper's dictionaries)."""
    afm = build_afm()
    mt.pp.annotate_vars(afm)
    cli = {**CLI, "filtering": filtering}
    cell_filter, kwargs, filtering_kwargs, binarization_kwargs, tree_kwargs = mt.ut.extract_kwargs(Namespace(**cli))
    afm = mt.pp.filter_cells(afm, cell_filter=cell_filter)
    out = mt.pp.filter_afm(
        afm, filtering_kwargs=filtering_kwargs, binarization_kwargs=binarization_kwargs,
        tree_kwargs=tree_kwargs, return_tree=False, compute_enrichment=False, **kwargs
    )
    assert out.n_obs > 0 and "bin" in out.layers
    assert out.uns["genotyping"]["bin_method"] == "MiTo"


def test_only_tree_returns_the_tree_arguments():
    cell_filter, kwargs, filtering_kwargs, binarization_kwargs, tree_kwargs = mt.ut.extract_kwargs(
        Namespace(**CLI), only_tree=True
    )
    assert (cell_filter, kwargs, filtering_kwargs, binarization_kwargs) == (None, None, None, None)
    assert tree_kwargs == {"solver": "UPMGA", "metric": "weighted_jaccard"}
