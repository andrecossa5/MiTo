"""
Shared fixtures for the MiTo test-suite.

Everything is synthesised: the suite must pass from a clean checkout with no external
data and no network access, since scverse reviewers run ``pip install ".[test]" &&
pytest`` themselves.

The synthetic AFM is the object contract every MiTo function consumes, i.e. what
``mito.io.make_afm`` produces:

* ``X``            : CSR float32 allele frequencies (AD / DP)
* ``layers['AD']`` : CSR int16 alternative-allele counts
* ``layers['DP']`` : DENSE int16 coverage of the variant's site, defined for every cell
* ``var``          : ``pos`` / ``ref`` / ``alt`` / ``quality``, indexed ``{pos}_{REF}>{ALT}``
* ``obs``          : the coverage metrics the cell filters test, plus a ``GBC`` ground truth
* ``uns``          : ``scLT_system``, ``pp_method``

Fixtures also write pre-processing output to disk (``maegatk_tables``, ``mgatk_tables``,
``redeem_tables``), so the readers are tested against the layouts the upstream tools and
nf-MiTo actually produce.
"""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from anndata import AnnData
from scipy.sparse import csr_matrix

N_CELLS = 120
N_VARS = 30
N_CLONES = 4
MT_GENOME_SIZE = 16569


@pytest.fixture(autouse=True)
def close_figures():
    yield
    plt.close("all")


# -- synthetic AFM ----------------------------------------------------------

def _positions(n_vars, rng):
    """Unique MT positions inside MT-gene bodies, with plausible ref/alt bases."""
    bases = np.array(["A", "C", "G", "T"])
    pos = np.sort(rng.choice(np.arange(3200, 15000), size=n_vars, replace=False))
    ref = rng.choice(bases, size=n_vars)
    alt = np.array([rng.choice(bases[bases != r]) for r in ref])
    names = [f"{p}_{r}>{a}" for p, r, a in zip(pos, ref, alt, strict=True)]
    return pd.DataFrame({"pos": pos, "ref": ref, "alt": alt}, index=names)


def build_afm(
    n_cells=N_CELLS,
    n_vars=N_VARS,
    n_clones=N_CLONES,
    markers_per_clone=2,
    coverage=120,
    af_positive=0.15,
    dropout=0.1,
    error_rate=0.002,
    quality=True,
    targeted=True,
    scLT_system="MAESTER",
    pp_method="maegatk",
    seed=1234,
):
    """
    A synthetic AFM with real clonal signal: `markers_per_clone` variants per clone at
    `af_positive`, the rest sequencing-error noise at `error_rate`.

    `quality=False` / `targeted=False` reproduce the AFMs of assays that report no base
    qualities (mgatk in 10x mode) and enrich no panel (mtscATAC, ReDeeM), which is what
    the capability-based dispatch has to cope with.
    """
    rng = np.random.default_rng(seed)

    var = _positions(n_vars, rng)
    cells = [f"CELL{i:04d}" for i in range(n_cells)]
    clones = np.array([f"clone_{i}" for i in range(n_clones)])
    labels = clones[np.arange(n_cells) % n_clones]

    DP = np.clip(rng.poisson(coverage, size=(n_cells, n_vars)), 5, None)

    # Background: sequencing error everywhere. Signal: a few markers per clone.
    AD = rng.binomial(DP, error_rate)
    genotype = np.zeros((n_cells, n_vars), bool)
    for j in range(min(n_clones * markers_per_clone, n_vars)):
        carriers = (labels == clones[j // markers_per_clone]) & (rng.random(n_cells) > dropout)
        genotype[:, j] = carriers
        AD[carriers, j] = rng.binomial(DP[carriers, j], af_positive)

    AF = (AD / DP).astype(np.float32)

    obs = pd.DataFrame(
        {
            "GBC": pd.Categorical(labels),
            "mean_site_coverage": DP.mean(axis=1),
            "nUMIs": DP.sum(axis=1),
        },
        index=cells,
    )
    if targeted:
        obs["median_target_site_coverage"] = np.median(DP, axis=1)
        obs["frac_target_site_covered"] = rng.uniform(0.85, 1.0, n_cells)
    if quality:
        var["quality"] = rng.uniform(32, 40, n_vars)

    return AnnData(
        X=csr_matrix(AF),
        obs=obs,
        var=var,
        layers={
            "AD": csr_matrix(AD.astype(np.int16)),
            "DP": DP.astype(np.int16),              # dense, on purpose
            "genotype": csr_matrix(genotype.astype(np.int8)),
        },
        uns={"scLT_system": scLT_system, "pp_method": pp_method},
    )


@pytest.fixture
def afm():
    """Default MAESTER AFM: 120 cells x 30 variants, 4 clones, 2 markers each."""
    return build_afm()


@pytest.fixture
def afm_factory():
    """Callable returning a fresh AFM, for parametrised tests."""
    return build_afm


@pytest.fixture
def afm_small():
    """Minimal AFM, for edge-condition tests."""
    return build_afm(n_cells=24, n_vars=8, n_clones=2, seed=7)


@pytest.fixture
def afm_no_quality():
    """What mgatk writes in 10x mode: counts, no base qualities, no target panel."""
    return build_afm(quality=False, targeted=False, scLT_system="mtscATAC", pp_method="mgatk")


@pytest.fixture
def afm_redeem():
    return build_afm(quality=False, targeted=False, scLT_system="ReDeeM", pp_method="redeem-v")


@pytest.fixture
def distance_matrix():
    """Symmetric distance matrix with a zero diagonal."""
    rng = np.random.default_rng(0)
    X = rng.random((30, 30))
    D = (X + X.T) / 2
    np.fill_diagonal(D, 0.0)
    return D


# -- cached pipeline products ----------------------------------------------
# Filtering and tree building are slow: compute once per session and hand out copies,
# so no test can leak state into another.

def run_pipeline(**overrides):
    """Cell filter -> variant filter, with thresholds suited to the synthetic AFM."""
    import mito as mt

    a = build_afm(**overrides.pop("afm_kwargs", {}))
    mt.pp.filter_cells(a, cell_filter="filter2")
    mt.pp.filter_afm(a, ncores=1, **overrides)
    return a


@pytest.fixture(scope="session")
def _filtered_cache():
    return run_pipeline()


@pytest.fixture
def afm_filtered(_filtered_cache):
    """Filtered AFM, ready for distances, embeddings and tree building."""
    return _filtered_cache.copy()


@pytest.fixture(scope="session")
def _tree_cache(_filtered_cache):
    import mito as mt
    return mt.tl.build_tree(_filtered_cache.copy(), solver="UPMGA")


@pytest.fixture
def tree(_tree_cache):
    import copy
    return copy.deepcopy(_tree_cache)


@pytest.fixture(scope="session")
def _annotated_cache(_filtered_cache, _tree_cache):
    import copy

    import mito as mt
    afm = _filtered_cache.copy()
    tree = copy.deepcopy(_tree_cache)
    mt.tl.annotate_clones(tree, afm)
    return afm, tree


@pytest.fixture
def annotated(_annotated_cache):
    """(AFM, tree) after clonal annotation."""
    import copy
    afm, tree = _annotated_cache
    return afm.copy(), copy.deepcopy(tree)


@pytest.fixture
def annotated_tree(annotated):
    return annotated[1]


# -- pre-processing output on disk ------------------------------------------

def _reference_bases(positions, random=False, rng=None):
    """
    Reference base at each position: rCRS, or - when the fixture writes its own
    refAllele table - deliberately different bases, so that a reader ignoring that
    table is caught.
    """

    if random:
        return {p: str(rng.choice(list("ACGT"))) for p in positions}

    from mito.io.format_afm import _read_reference
    rcrs = _read_reference("rCRS")

    return {p: rcrs[p] for p in positions}


##


def _write_allelic_tables(folder, prefix="", with_quality=True, n_cells=120, n_sites=40,
                          n_clones=4, seed=0, ref_allele=False, compress=True):
    """
    maegatk / mgatk output: one table per base, plus coverage. With `with_quality=False`
    the tables have 4 columns, as mgatk writes them for 10x data.
    """
    folder.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    cells = [f"CB{i:03d}" for i in range(n_cells)]
    positions = sorted(rng.choice(range(3300, 9000), n_sites, replace=False))
    # The reference bases must be the ones the reader will use, or every reference
    # basecall is read as a variant: rCRS unless the fixture writes its own refAllele.
    ref_of = _reference_bases(positions, random=ref_allele, rng=rng)
    alt_of = {p: str(rng.choice([b for b in "ACGT" if b != ref_of[p]])) for p in positions}
    cov = rng.integers(60, 160, size=(n_cells, n_sites))
    clone = np.arange(n_cells) % n_clones

    rows = {b: [] for b in "ACGT"}
    cov_rows = []
    for i, c in enumerate(cells):
        for j, p in enumerate(positions):
            ad = int(rng.binomial(cov[i, j], 0.2)) if (j < 2 * n_clones and clone[i] == j // 2) else 0
            ref_ct = int(cov[i, j]) - ad
            for base, count in ((ref_of[p], ref_ct), (alt_of[p], ad)):
                if count > 0:
                    fw, rev = count // 2, count - count // 2
                    row = ([p, c, fw, 35.0, rev, 35.0] if with_quality else [p, c, fw, rev])
                    rows[base].append(row)
            cov_rows.append((p, c, int(cov[i, j])))

    ext = ".txt.gz" if compress else ".txt"
    for base in "ACGT":
        pd.DataFrame(rows[base]).to_csv(folder / f"{prefix}{base}{ext}", header=False, index=False)
    pd.DataFrame(cov_rows).to_csv(folder / f"{prefix}coverage{ext}", header=False, index=False)
    if ref_allele:
        pd.DataFrame([(p, ref_of.get(p, "N")) for p in range(1, MT_GENOME_SIZE + 1)]).to_csv(
            folder / f"{prefix}refAllele.txt", sep="\t", header=False, index=False
        )

    return dict(cells=cells, positions=positions, ref=ref_of, alt=alt_of, clone=clone)


def _write_redeem_tables(folder, n_cells=160, n_sites=40, n_clones=4, seed=0):
    """RedeemV output: RawGenotypes.<level>.StrandBalance and QualifiedTotalCts."""
    folder.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    cells = [f"CB{i:03d}" for i in range(n_cells)]
    positions = sorted(rng.choice(range(3300, 9000), n_sites, replace=False))
    ref_of = _reference_bases(positions)
    alt_of = {p: str(rng.choice([b for b in "ACGT" if b != ref_of[p]])) for p in positions}
    cov = rng.integers(20, 60, size=(n_cells, n_sites))
    clone = np.arange(n_cells) % n_clones

    geno = []
    for i, c in enumerate(cells):
        for j, p in enumerate(positions[: 2 * n_clones]):
            if clone[i] == j // 2:
                for _ in range(int(rng.integers(3, 9))):
                    start = p - int(rng.integers(20, 150))
                    end = p + int(rng.integers(20, 150))
                    geno.append((f"{c}_{start}_{end}", c, p, f"{p}_{ref_of[p]}>{alt_of[p]}",
                                 alt_of[p], ref_of[p], 4, 4, 1.0, 2, 2, 1, 1, int(cov[i, j])))
    pd.DataFrame(geno).to_csv(folder / "RawGenotypes.Sensitive.StrandBalance",
                              sep="\t", header=False, index=False)
    pd.DataFrame([
        (c, p, int(cov[i, j] * 1.2), int(cov[i, j] * 1.1), int(cov[i, j]), int(cov[i, j] * 0.9))
        for i, c in enumerate(cells) for j, p in enumerate(positions)
    ]).to_csv(folder / "QualifiedTotalCts", sep="\t", header=False, index=False)

    return dict(cells=cells, positions=positions, clone=clone)


@pytest.fixture
def maegatk_tables(tmp_path):
    """nf-MiTo layout: bare table names, with base qualities."""
    folder = tmp_path / "maegatk"
    meta = _write_allelic_tables(folder, with_quality=True)
    return folder, meta


@pytest.fixture
def mgatk_tables(tmp_path):
    """mgatk layout: a final/ sub-folder, sample-prefixed names, no base qualities."""
    folder = tmp_path / "mgatk"
    meta = _write_allelic_tables(folder / "final", prefix="sampleX.", with_quality=False,
                                 ref_allele=True)
    return folder, meta


@pytest.fixture
def redeem_tables(tmp_path):
    folder = tmp_path / "redeem"
    meta = _write_redeem_tables(folder / "final")
    return folder, meta
