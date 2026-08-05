"""
Shared fixtures for the MiTo test-suite.

Everything is synthesised: the suite must pass from a clean checkout with no
external data and no network access, since scverse reviewers run
``pip install ".[test]" && pytest`` themselves.

The synthetic AFM mirrors the structure produced by ``mito.io.make_afm``:

* ``X``            : CSR float32 allele frequencies (AD / DP)
* ``layers['AD']`` : CSR int16 alternative-allele counts
* ``layers['DP']`` : CSR int16 total depth
* ``var``          : ``pos`` / ``ref`` / ``alt``, indexed as ``{pos}_{REF}>{ALT}``
* ``obs``          : coverage metrics used by the cell filters, plus a ``GBC``
                     ground-truth lineage label
* ``uns``          : ``scLT_system``, ``pp_method`` and the per-position tables
"""

import matplotlib
matplotlib.use("Agg")

import numpy as np
import pandas as pd
import pytest
from anndata import AnnData
from scipy.sparse import csr_matrix
import matplotlib.pyplot as plt


N_CELLS = 90
N_VARS = 40
N_CLONES = 3
MT_GENOME_SIZE = 16569


@pytest.fixture(autouse=True)
def close_figures():
    yield
    plt.close("all")


def _make_positions(n_vars, rng):
    """Unique MT positions with plausible ref/alt bases."""
    bases = np.array(["A", "C", "G", "T"])
    pos = rng.choice(np.arange(1, MT_GENOME_SIZE), size=n_vars, replace=False)
    pos.sort()
    ref = rng.choice(bases, size=n_vars)
    alt = np.array([rng.choice(bases[bases != r]) for r in ref])
    names = [f"{p}_{r}>{a}" for p, r, a in zip(pos, ref, alt)]
    var = pd.DataFrame({"pos": pos, "ref": ref, "alt": alt}, index=names)
    return var


def build_afm(
    n_cells=N_CELLS,
    n_vars=N_VARS,
    n_clones=N_CLONES,
    scLT_system="MAESTER",
    pp_method="maegatk",
    seed=1234,
    clone_specific_frac=0.4,
    coverage=60,
    add_per_position=True,
):
    """
    Build a synthetic AFM with clone-structured variants.

    A fraction of variants is made clone-specific so that filtering, distance
    computation and tree building have real signal to recover; the remainder is
    background noise.
    """
    rng = np.random.default_rng(seed)

    var = _make_positions(n_vars, rng)
    cells = [f"CELL{i:04d}" for i in range(n_cells)]
    clones = np.array([f"clone_{i}" for i in range(n_clones)])
    labels = clones[np.repeat(np.arange(n_clones), int(np.ceil(n_cells / n_clones)))[:n_cells]]

    # depth: over-dispersed around `coverage`
    DP = rng.poisson(coverage, size=(n_cells, n_vars)).astype(np.int64)
    DP = np.clip(DP, 1, None)

    # Allele frequencies. Real MT-SNV matrices are sparse -- most cells are exact
    # zeros for most variants -- and the variant filters depend on that sparsity,
    # so the background is explicitly zeroed rather than merely small.
    n_specific = max(1, int(n_vars * clone_specific_frac))
    af = rng.beta(0.2, 20, size=(n_cells, n_vars))          # background noise
    detected = rng.random((n_cells, n_vars)) < 0.05          # ~5% background positivity
    af = np.where(detected, af, 0.0)
    for j in range(n_specific):
        target = clones[j % n_clones]
        mask = labels == target
        af[mask, j] = rng.beta(6, 3, size=mask.sum())        # strong in-clone signal

    AD = np.rint(af * DP).astype(np.int64)
    AD = np.clip(AD, 0, DP)
    AF = (AD / (DP + 1e-8)).astype(np.float32)

    obs = pd.DataFrame(
        {
            "GBC": pd.Categorical(labels),
            "mean_site_coverage": rng.normal(coverage, coverage * 0.1, n_cells).clip(1),
            "median_target_site_coverage": rng.normal(coverage, coverage * 0.1, n_cells).clip(1),
            "median_untarget_site_coverage": rng.normal(coverage / 3, 2, n_cells).clip(1),
            "frac_target_site_covered": rng.uniform(0.8, 1.0, n_cells),
            "nUMIs": rng.integers(1000, 5000, n_cells),
        },
        index=cells,
    )

    uns = {"scLT_system": scLT_system, "pp_method": pp_method}

    # site-level coverage and base quality, as produced by the MAESTER readers
    site_cov = rng.poisson(coverage, size=(n_cells, n_vars)).clip(1)
    qual = rng.uniform(20, 40, size=(n_cells, n_vars))

    afm = AnnData(
        X=csr_matrix(AF),
        obs=obs,
        var=var,
        layers={
            "AD": csr_matrix(AD.astype(np.int16)),
            "DP": csr_matrix(DP.astype(np.int16)),
            "site_coverage": csr_matrix(site_cov.astype(np.int16)),
            "qual": csr_matrix(qual.astype(np.float32)),
        },
        uns=uns,
    )

    if add_per_position:
        positions = np.unique(var["pos"].values)
        afm.uns["per_position_coverage"] = pd.DataFrame(
            rng.normal(coverage, 5, size=(n_cells, positions.size)).clip(0),
            index=cells, columns=positions,
        )
        afm.uns["per_position_quality"] = pd.DataFrame(
            rng.uniform(20, 40, size=(n_cells, positions.size)),
            index=cells, columns=positions,
        )

    return afm


# -- ready-made fixtures ----------------------------------------------------

@pytest.fixture
def afm():
    """Default MAESTER AFM: 90 cells x 40 variants, 3 clones."""
    return build_afm()


@pytest.fixture
def afm_factory():
    """Callable returning a fresh AFM, for parametrised tests."""
    return build_afm


@pytest.fixture
def afm_small():
    """Minimal AFM, for edge-condition tests."""
    return build_afm(n_cells=12, n_vars=6, n_clones=2, seed=7)


@pytest.fixture
def afm_redeem():
    return build_afm(scLT_system="RedeeM")


@pytest.fixture
def afm_cas9():
    return build_afm(scLT_system="Cas9")


@pytest.fixture
def distance_matrix():
    """Symmetric distance matrix with a zero diagonal."""
    rng = np.random.default_rng(0)
    X = rng.random((30, 30))
    D = (X + X.T) / 2
    np.fill_diagonal(D, 0.0)
    return D


@pytest.fixture
def afm_annotated():
    """AFM with .var annotations, the state ``filter_afm`` expects."""
    import mito as mt
    a = build_afm()
    mt.pp.annotate_vars(a)
    return a


# Parameters for an AFM on which *every* variant-filtering strategy succeeds with
# its default thresholds. The strategies pull in opposite directions: MQuad needs
# strongly bimodal variants, while weng2024 requires >=90% negative cells per
# variant -- so the input needs many small clones, each with its own high-VAF
# variants.
RICH_AFM = dict(
    n_cells=320, n_vars=80, n_clones=16,
    coverage=150, clone_specific_frac=1.0, seed=67,
)


def build_rich_annotated(**overrides):
    """Annotated AFM with enough signal for every filtering strategy's defaults."""
    import mito as mt
    cfg = {**RICH_AFM, **overrides}
    a = build_afm(**cfg)
    mt.pp.annotate_vars(a)
    return a


@pytest.fixture
def afm_rich():
    return build_rich_annotated()


# -- cached pipeline products ----------------------------------------------
# Filtering and tree building are slow; compute once per session and hand out
# copies, so no test can leak state into another.

@pytest.fixture(scope="session")
def _filtered_cache():
    import mito as mt
    a = build_afm()
    mt.pp.annotate_vars(a)
    a = mt.pp.filter_cells(a, cell_filter="filter2")
    return mt.pp.filter_afm(a, filtering="MiTo", compute_enrichment=False, ncores=1)


@pytest.fixture
def afm_filtered(_filtered_cache):
    """Filtered AFM, ready for dimensionality reduction and tree building."""
    return _filtered_cache.copy()


@pytest.fixture(scope="session")
def _tree_cache(_filtered_cache):
    import mito as mt
    return mt.tl.build_tree(_filtered_cache.copy(), precomputed=True, solver="UPMGA")


@pytest.fixture
def tree(_tree_cache):
    """A built CassiopeiaTree."""
    import copy as _copy
    return _copy.deepcopy(_tree_cache)


@pytest.fixture(scope="session")
def _annotated_tree_cache(_tree_cache):
    """Tree with clonal inference already run -- the grid search is slow."""
    import copy as _copy
    import mito as mt
    tree = _copy.deepcopy(_tree_cache)
    annotator = mt.tl.MiToTreeAnnotator(tree)
    annotator.clonal_inference()
    return annotator


@pytest.fixture
def annotator(_annotated_tree_cache):
    import copy as _copy
    return _copy.deepcopy(_annotated_tree_cache)


@pytest.fixture
def annotated_tree(_annotated_tree_cache):
    """A CassiopeiaTree carrying MiTo clone assignments in cell_meta."""
    import copy as _copy
    return _copy.deepcopy(_annotated_tree_cache.tree)
