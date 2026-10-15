"""
mito.io.make_afm

The readers are the only place where MiTo meets another tool's file format, so these
tests are written against the layouts maegatk, mgatk and RedeemV actually produce, and
they check the object contract every downstream function relies on.
"""

import numpy as np
import pandas as pd
import pytest
from scipy.sparse import issparse

import mito as mt

# -- the contract -----------------------------------------------------------


def _assert_contract(afm):
    """Everything mito.pp assumes about a freshly built AFM."""

    assert issparse(afm.X) and afm.X.dtype == np.float32
    assert issparse(afm.layers["AD"])
    # DP is dense on purpose: it is non-zero almost everywhere, so CSR would cost more
    assert not issparse(afm.layers["DP"])
    assert np.issubdtype(afm.layers["DP"].dtype, np.integer)

    AD = afm.layers["AD"].toarray()
    DP = np.asarray(afm.layers["DP"])
    assert (AD <= DP).all(), "alt counts cannot exceed the site's coverage"
    assert np.allclose(afm.X.toarray(), AD / np.maximum(DP, 1), atol=1e-6)

    # coverage is defined for every cell, not only for those with an alternative read:
    # "covered, no alt read" is evidence of absence and must be distinguishable from
    # "not covered"
    assert (DP > 0).mean() > 0.5
    assert ((DP > 0) & (AD == 0)).any()

    for column in ("pos", "ref", "alt"):
        assert column in afm.var.columns
    assert afm.var["pos"].is_monotonic_increasing
    assert afm.var_names.to_series().str.match(r"^\d+_[ACGT]>[ACGT]$").all()
    assert afm.var["pos"].value_counts().max() == 1, "multi-allelic sites must be dropped"

    assert "mean_site_coverage" in afm.obs.columns
    assert {"scLT_system", "pp_method"} <= set(afm.uns)


def test_maegatk_tables_build_a_valid_afm(maegatk_tables):
    folder, meta = maegatk_tables
    afm = mt.io.make_afm(folder, scLT_system="MAESTER", pp_method="maegatk")
    _assert_contract(afm)
    assert afm.shape[0] == len(meta["cells"])
    assert "quality" in afm.var.columns, "maegatk tables carry base qualities"


def test_mgatk_tables_build_a_valid_afm(mgatk_tables):
    """mgatk in 10x mode: a final/ folder, sample-prefixed names, 4-column tables."""
    folder, meta = mgatk_tables
    afm = mt.io.make_afm(folder, scLT_system="mtscATAC", pp_method="mgatk")
    _assert_contract(afm)
    assert afm.shape[0] == len(meta["cells"])
    assert "quality" not in afm.var.columns, "mgatk 10x tables carry no base qualities"


def test_redeem_tables_build_a_valid_afm(redeem_tables):
    folder, meta = redeem_tables
    afm = mt.io.make_afm(folder, scLT_system="ReDeeM", pp_method="redeem-v")
    _assert_contract(afm)
    assert afm.shape[0] <= len(meta["cells"])
    assert afm.shape[1] > 0


# -- the differences between assays -----------------------------------------


def test_target_panel_metrics_only_for_targeted_assays(maegatk_tables, mgatk_tables):
    """
    The cell filters choose their criterion from the metrics present, so a targeted assay
    must get the panel columns and a genome-wide one must not.
    """
    targeted = mt.io.make_afm(maegatk_tables[0], scLT_system="MAESTER", pp_method="maegatk")
    genomewide = mt.io.make_afm(mgatk_tables[0], scLT_system="mtscATAC", pp_method="mgatk")

    panel = {"median_target_site_coverage", "frac_target_site_covered"}
    assert panel <= set(targeted.obs.columns)
    assert not (panel & set(genomewide.obs.columns))


def test_reference_alleles_of_the_run_are_preferred(mgatk_tables):
    """
    mgatk writes the reference it was run against (it ships a mouse MT genome too), so a
    refAllele table in the output must win over the bundled rCRS.
    """
    folder, meta = mgatk_tables
    afm = mt.io.make_afm(folder, scLT_system="mtscATAC", pp_method="mgatk")
    for name in afm.var_names:
        pos, sub = name.split("_")
        assert sub.split(">")[0] == meta["ref"][int(pos)]


def test_sample_name_is_appended_to_barcodes(maegatk_tables):
    folder, _ = maegatk_tables
    afm = mt.io.make_afm(folder, scLT_system="MAESTER", pp_method="maegatk", sample="s1")
    assert afm.obs_names.str.endswith("_s1").all()


def test_uncompressed_tables_are_read(tmp_path):
    from conftest import _write_allelic_tables

    folder = tmp_path / "plain"
    _write_allelic_tables(folder, compress=False)
    afm = mt.io.make_afm(folder, scLT_system="MAESTER", pp_method="maegatk")
    assert afm.shape[1] > 0


# -- cell metadata ----------------------------------------------------------


def test_metadata_restricts_the_cells(maegatk_tables, tmp_path):
    folder, meta = maegatk_tables
    keep = meta["cells"][:20]
    path_meta = tmp_path / "meta.csv"
    pd.DataFrame({"label": ["x"] * len(keep)}, index=keep).to_csv(path_meta)

    afm = mt.io.make_afm(folder, path_meta=str(path_meta), scLT_system="MAESTER",
                         pp_method="maegatk")
    assert list(afm.obs_names) == keep
    assert "label" in afm.obs.columns


def test_metadata_that_matches_nothing_explains_the_cell_name_convention(maegatk_tables, tmp_path):
    folder, _ = maegatk_tables
    path_meta = tmp_path / "meta.csv"
    pd.DataFrame({"label": ["x"]}, index=["NOT_A_BARCODE"]).to_csv(path_meta)

    with pytest.raises(ValueError, match="sample="):
        mt.io.make_afm(folder, path_meta=str(path_meta), scLT_system="MAESTER",
                       pp_method="maegatk")


# -- refusals ---------------------------------------------------------------


def test_unknown_scLT_system_is_refused(maegatk_tables):
    with pytest.raises(ValueError, match="scLT_system"):
        mt.io.make_afm(maegatk_tables[0], scLT_system="Cas9", pp_method="maegatk")


def test_pp_method_must_match_the_assay(maegatk_tables):
    with pytest.raises(ValueError, match="pp_method"):
        mt.io.make_afm(maegatk_tables[0], scLT_system="MAESTER", pp_method="redeem-v")


def test_missing_path_is_refused(tmp_path):
    with pytest.raises(ValueError, match="path_ch_matrix"):
        mt.io.make_afm(str(tmp_path / "nope"))


def test_missing_tables_name_what_is_expected(tmp_path):
    (tmp_path / "empty").mkdir()
    with pytest.raises(ValueError, match=r"A\.txt"):
        mt.io.make_afm(tmp_path / "empty", scLT_system="MAESTER", pp_method="maegatk")


def test_multi_allelic_sites_keep_every_allele(maegatk_tables):
    folder, meta = maegatk_tables
    # A second, weaker alternative allele at the first marker site
    p = meta["positions"][0]
    other = next(b for b in "ACGT" if b not in (meta["ref"][p], meta["alt"][p]))
    path = folder / f"{other}.txt.gz"
    table = pd.read_csv(path, header=None)
    extra = pd.DataFrame([[p, c, 1, 35.0, 1, 35.0] for c in meta["cells"][:3]])
    pd.concat([table, extra]).to_csv(path, header=False, index=False)

    one = mt.io.make_afm(folder, scLT_system="MAESTER", pp_method="maegatk")
    every = mt.io.make_afm(folder, scLT_system="MAESTER", pp_method="maegatk", multi_allelic_sites=True)
    assert one.var["pos"].value_counts().max() == 1
    assert f"{p}_{meta['ref'][p]}>{meta['alt'][p]}" in one.var_names
    assert set(every.var_names) - set(one.var_names) == {f"{p}_{meta['ref'][p]}>{other}"}


def test_redeem_edge_trim_defaults_to_redeemR(redeem_tables):
    import inspect

    from mito.io.format_afm import _read_redeem
    assert inspect.signature(_read_redeem).parameters["edge_trim"].default == 9
    folder, _ = redeem_tables
    trimmed = mt.io.make_afm(folder, scLT_system="ReDeeM", pp_method="redeem-v")
    loose = mt.io.make_afm(folder, scLT_system="ReDeeM", pp_method="redeem-v", edge_trim=0)
    assert trimmed.layers["AD"].sum() <= loose.layers["AD"].sum()


def test_redeem_threshold_is_validated(redeem_tables):
    with pytest.raises(ValueError, match="treshold"):
        mt.io.make_afm(redeem_tables[0], scLT_system="ReDeeM", pp_method="redeem-v",
                       treshold="Stringent")          # RedeemV calls this level "Specific"


# -- the AFM feeds the pipeline ---------------------------------------------


@pytest.mark.parametrize("fixture,system,method,cell_filter", [
    ("maegatk_tables", "MAESTER", "maegatk", "filter2"),
    ("mgatk_tables", "mtscATAC", "mgatk", "filter1"),
    ("redeem_tables", "ReDeeM", "redeem-v", "filter1"),
])
def test_every_reader_output_runs_the_pipeline(request, fixture, system, method, cell_filter):
    folder, _ = request.getfixturevalue(fixture)
    afm = mt.io.make_afm(folder, scLT_system=system, pp_method=method)
    mt.pp.filter_cells(afm, cell_filter=cell_filter, mean_cov_all=5)
    mt.pp.filter_afm(afm, qc_alpha=0.1, ncores=1)
    tree = mt.tl.build_tree(afm, solver="UPMGA")
    mt.tl.annotate_clones(tree, afm, min_cells=3)
    assert afm.obs["MiTo_clone"].nunique() >= 2


# -- mtscATAC: reads to molecules -------------------------------------------

@pytest.mark.parametrize("o", [0.3, 0.66, 0.9])
def test_reads_per_molecule_is_recovered_from_the_strands(o):
    from mito.io.format_afm import _reads_per_molecule
    rng = np.random.default_rng(0)
    n_cells, n_sites = 300, 60
    s = rng.uniform(0.3, 0.7, n_sites)                    # forward fraction of single reads
    molecules = rng.poisson(30, (n_cells, n_sites))
    paired = rng.binomial(molecules, o)
    single_fw = rng.binomial(molecules - paired, s)
    fw = paired + single_fw
    rev = paired + (molecules - paired - single_fw)
    pos = np.tile(np.arange(n_sites), (n_cells, 1))
    f = _reads_per_molecule(fw.ravel(), rev.ravel(), pos.ravel())
    assert f == pytest.approx(1 + o, abs=0.05)


def test_mtscatac_counts_molecules_read_by_both_mates(mgatk_tables):
    folder, _ = mgatk_tables
    reads = mt.io.make_afm(folder, scLT_system="mtscATAC", pp_method="mgatk", molecules=False)
    mols = mt.io.make_afm(folder, scLT_system="mtscATAC", pp_method="mgatk")
    assert "molecules" in mols.uns and "molecules" not in reads.uns
    assert 1 <= mols.uns["molecules"]["reads_per_molecule"] <= 2
    common = reads.var_names.intersection(mols.var_names)
    assert (mols[:, common].layers["AD"].toarray() <= reads[:, common].layers["AD"].toarray()).all()
    assert (mols.layers["DP"] <= reads[:, mols.var_names].layers["DP"]).all()


def test_maester_counts_are_left_as_they_are(maegatk_tables):
    folder, _ = maegatk_tables
    afm = mt.io.make_afm(folder, scLT_system="MAESTER", pp_method="maegatk")
    assert "molecules" not in afm.uns


# -- AFMs written by MiTo < 0.3 ---------------------------------------------

def _legacy_afm():
    from anndata import AnnData
    from scipy.sparse import csr_matrix
    AD = np.array([[3, 1, 0], [2, 0, 0], [0, 0, 4]])
    site = np.array([[50, 50, 40], [60, 60, 30], [55, 55, 20]])
    DP_masked = np.where(AD > 0, site, 0)
    qual = np.where(AD > 0, 35.0, 0.0)
    return AnnData(X=csr_matrix(AD / np.maximum(DP_masked, 1)),
                   var=pd.DataFrame(index=["100_A>G", "100_A>T", "200_C>T"]),
                   layers={"AD": csr_matrix(AD), "DP": csr_matrix(DP_masked),
                           "site_coverage": csr_matrix(site), "qual": csr_matrix(qual)})


def test_migrate_afm_restores_site_coverage_and_one_allele_per_site():
    afm = _legacy_afm()
    mt.io.migrate_afm(afm)
    assert list(afm.var_names) == ["100_A>G", "200_C>T"]          # 100_A>T: fewer alt reads
    assert {"site_coverage", "qual"}.isdisjoint(afm.layers)
    assert (afm.layers["DP"] == np.array([[50, 40], [60, 30], [55, 20]])).all()
    assert "quality" in afm.var.columns


def test_migrate_afm_can_keep_every_allele():
    afm = _legacy_afm()
    mt.io.migrate_afm(afm, multi_allelic_sites=True)
    assert afm.shape[1] == 3


def test_without_overlapping_mates_alternative_reads_are_scaled_not_dropped(mgatk_single_end_tables):
    folder, _ = mgatk_single_end_tables
    reads = mt.io.make_afm(folder, scLT_system="mtscATAC", pp_method="mgatk", molecules=False)
    mols = mt.io.make_afm(folder, scLT_system="mtscATAC", pp_method="mgatk")
    info = mols.uns["molecules"]
    assert info["alt_rule"] == "reads_over_f"
    assert info["frac_both_mates"] < 0.5
    # strand confirmation would have kept almost nothing; scaling keeps the evidence
    assert mols.layers["AD"].sum() >= 0.5 * reads.layers["AD"].sum() / info["reads_per_molecule"]


def test_with_overlapping_mates_alternative_molecules_need_both_strands(mgatk_tables):
    folder, _ = mgatk_tables
    mols = mt.io.make_afm(folder, scLT_system="mtscATAC", pp_method="mgatk")
    assert mols.uns["molecules"]["alt_rule"] == "both_strands"


# -- from long count tables -------------------------------------------------

def _long_tables(afm):
    """An AFM exported as the counts + coverage tables make_afm_from_counts reads."""
    AD = afm.layers["AD"].tocoo()
    counts = pd.DataFrame({"cell": afm.obs_names[AD.row], "pos": afm.var["pos"].values[AD.col],
                           "ref": afm.var["ref"].values[AD.col], "alt": afm.var["alt"].values[AD.col],
                           "AD": AD.data})
    DP = np.asarray(afm.layers["DP"])
    sites = afm.var.drop_duplicates("pos")
    j = [list(afm.var_names).index(v) for v in sites.index]
    coverage = pd.DataFrame({"cell": np.repeat(afm.obs_names, len(j)),
                             "pos": np.tile(sites["pos"].values, afm.shape[0]),
                             "DP": DP[:, j].ravel()})
    return counts, coverage


def test_counts_tables_give_back_the_same_matrices(maegatk_tables):
    folder, _ = maegatk_tables
    afm = mt.io.make_afm(folder, scLT_system="MAESTER", pp_method="maegatk")
    counts, coverage = _long_tables(afm)
    back = mt.io.make_afm_from_counts(counts, coverage, scLT_system="MAESTER")
    _assert_contract(back)
    assert list(back.var_names) == list(afm.var_names)
    cells = back.obs_names
    assert (back.layers["AD"].toarray() == afm[cells].layers["AD"].toarray()).all()
    assert (np.asarray(back.layers["DP"]) == np.asarray(afm[cells].layers["DP"])).all()
    assert np.allclose(back.X.toarray(), afm[cells].X.toarray())


def test_counts_tables_keep_the_cells_of_the_metadata_in_its_order(maegatk_tables):
    folder, _ = maegatk_tables
    afm = mt.io.make_afm(folder, scLT_system="MAESTER", pp_method="maegatk")
    counts, coverage = _long_tables(afm)
    meta = pd.DataFrame({"group": "a"}, index=list(afm.obs_names[10:30][::-1]))
    back = mt.io.make_afm_from_counts(counts, coverage, cell_meta=meta)
    assert list(back.obs_names) == list(meta.index)
    assert "group" in back.obs.columns


@pytest.mark.parametrize("bad, match", [
    (lambda c, v: (c.drop(columns="AD"), v), "AD"),
    (lambda c, v: (c.assign(alt="N"), v), "alt must be"),
    (lambda c, v: (c.assign(alt=c["ref"]), v), "reference base"),
    (lambda c, v: (pd.concat([c, c.iloc[:1]]), v), "several rows"),
])
def test_counts_tables_are_validated(maegatk_tables, bad, match):
    folder, _ = maegatk_tables
    afm = mt.io.make_afm(folder, scLT_system="MAESTER", pp_method="maegatk")
    counts, coverage = _long_tables(afm)
    with pytest.raises(ValueError, match=match):
        mt.io.make_afm_from_counts(*bad(counts, coverage))
