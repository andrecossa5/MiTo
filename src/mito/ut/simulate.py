"""
Simulation of Allele Frequency Matrices.

Produces AFMs with the same structure as those assembled by :func:`mito.io.make_afm`,
so they can be passed straight to the preprocessing, phylogeny and plotting APIs.
Useful for tutorials, tests and for exploring how filtering behaves as the clonal
structure changes.

The generative process is deliberately minimal: a population of cells is split into
clones, each clone is marked by one unique variant, and clones are mostly independent
of one another, with the rest arising as lineage splits from an existing clone. Read
counts are then drawn from a two-component binomial mixture around those genotypes.
"""

import numpy as np
import pandas as pd
from anndata import AnnData
from scipy.sparse import csr_matrix

from scipy.stats import beta as _beta_dist

from .positions import MAESTER_genes_positions


def _draw_beta(shape, size, lower, rng):
    """Beta draws, truncated below `lower` by inverse-CDF sampling."""
    if lower <= 0:
        return rng.beta(*shape, size=size)
    lo = float(_beta_dist.cdf(lower, *shape))
    if lo >= 1 - 1e-12:
        return np.full(size, lower, dtype=float)

    return _beta_dist.ppf(rng.uniform(lo, 1.0, size=size), *shape)

##


def _clone_sizes(n_cells, n_clones, min_max_ratio, rng):
    """
    Split `n_cells` into `n_clones` groups whose sizes span `min_max_ratio`.

    Sizes are geometrically spaced from the largest to the smallest clone, so that
    smallest / largest == min_max_ratio, then apportioned with largest remainders so
    they sum exactly to n_cells.

    NB: the sizes are finally shuffled across clones. Clones are numbered in the
    order the tree creates them, so handing them out in order would tie size to
    position, making every root clone large and every deep clone small. Shuffling
    leaves the spread intact while letting a big clone sit anywhere in the tree.
    """
    if n_clones == 1:
        return np.array([n_cells])

    weights = np.geomspace(1.0, min_max_ratio, n_clones)
    weights = weights / weights.sum()

    exact = weights * n_cells
    sizes = np.floor(exact).astype(int)
    sizes = np.maximum(sizes, 1)                       # no empty clones

    # distribute what floor() left over, largest fractional part first
    while sizes.sum() < n_cells:
        sizes[np.argmax(exact - sizes)] += 1
    while sizes.sum() > n_cells:
        shrinkable = np.where(sizes > 1)[0]
        sizes[shrinkable[np.argmin(exact[shrinkable] - sizes[shrinkable])]] -= 1

    return sizes[rng.permutation(n_clones)]


##


def _draw_from_range(low, high, skew, rng, size=None):
    """
    Draw integers from `low`..`high` with a geometric bias towards the low end.

    The weight of value k is ``skew ** (k - low)``, normalised. A skew of 1 leaves
    every value equally likely, below 1 concentrates the draw on the low end, and
    above 1 on the high end.
    """
    values = np.arange(low, high + 1)
    weights = np.asarray(skew, dtype=float) ** (values - low)

    return rng.choice(values, size=size, p=weights / weights.sum())


##


def _subtree_capacity(max_depth, high, n_clones):
    """
    How many clones can still be added below a node, by the depth of that node.

    A node at depth d can take up to `high` children, each of which can in turn host
    a whole subtree, so ``cap[d] = high * (1 + cap[d + 1])`` and ``cap[max_depth] =
    0``. Values are clamped at `n_clones`, since capacity beyond the number of clones
    left to place makes no difference to any decision and the raw figure grows
    exponentially with depth.
    """
    cap = [0] * (max_depth + 2)
    for d in range(max_depth - 1, 0, -1):
        cap[d] = min(high * (1 + cap[d + 1]), n_clones)

    return cap


##


def _clonal_tree(n_clones, n_root_clones, max_depth, split_size, range_skew, rng):
    """
    Relate the clones, and return the tree as a parent array plus the clone depths.

    `n_root_clones` of them descend straight from the root, forming a polytomy at
    depth 1: they share no variant and are entirely independent of one another. The
    remaining clones are added by splitting events, each of which takes one clone
    that has not split yet and gives it `split_size` children at once. A split of
    size 1 is a chain, in which the clone simply acquires a further variant; size 2
    is a bifurcation into two sibling subclones; larger sizes give a nested polytomy.

    The parent of each split is drawn uniformly among the clones that have not split
    yet, so the shape of the tree emerges rather than being imposed.

    When `max_depth` is given, the draw is made capacity-aware instead of free: at
    every step the split size is drawn only from those values that leave enough room
    below the remaining unsplit clones to place all the clones still owed, and any
    parent offering no such value is passed over. The tree therefore always fits
    within the cap, with the randomness confined to the choices that keep it
    feasible. Depth is a ceiling and not a target, so a tree may come out shallower.

    Clones are numbered in the order they are created, which guarantees
    `parent[i] < i` and so keeps the tree acyclic by construction. `parent[i] == -1`
    marks a clone hanging off the root.
    """
    low, high = (split_size, split_size) if isinstance(split_size, int) else split_size

    cap = None
    if max_depth is not None:
        cap = _subtree_capacity(max_depth, high, n_clones)
        room = n_root_clones * cap[1]
        if n_clones - n_root_clones > room:
            raise ValueError(
                f'Cannot fit {n_clones} clones within max_depth={max_depth}: '
                f'{n_root_clones} root clones splitting into at most {high} children '
                f'hold {n_root_clones + room} clones at the most. Raise max_depth or '
                f'n_root_clones, raise the top of split_size, or drop the cap with '
                f'max_depth=None.'
            )

    parent = np.full(n_clones, -1, dtype=int)
    depth = np.ones(n_clones, dtype=int)
    has_split = np.zeros(n_clones, dtype=bool)

    n_placed = n_root_clones
    while n_placed < n_clones:
        owed = n_clones - n_placed
        can_split = ~has_split[:n_placed]
        if max_depth is not None:
            can_split &= depth[:n_placed] < max_depth
        eligible = np.flatnonzero(can_split)

        if cap is None:
            p = int(rng.choice(eligible))
            k = min(int(_draw_from_range(low, high, range_skew, rng)), owed)
        else:
            # Splitting a parent at depth d spends cap[d] of the room available and
            # buys back k * cap[d + 1] from its children, so k has to be large enough
            # for what is left to still hold every clone still owed.
            total = sum(cap[depth[i]] for i in eligible)
            options = []
            for i in eligible:
                d = int(depth[i])
                shortfall = owed - total + cap[d]
                denominator = 1 + cap[d + 1]
                k_low = max(low, -(-shortfall // denominator)) if shortfall > 0 else low
                k_high = min(high, owed)
                if k_low <= k_high:
                    options.append((int(i), k_low, k_high))

            i, k_low, k_high = options[rng.integers(len(options))]
            p = i
            k = int(_draw_from_range(k_low, k_high, range_skew, rng))

        parent[n_placed:n_placed + k] = p
        depth[n_placed:n_placed + k] = depth[p] + 1
        has_split[p] = True
        n_placed += k

    return parent, depth


##


def _descendant_masks(parent):
    """
    For every clone, the boolean mask of clones descending from it, itself included.

    A clone's variant is carried by exactly these clones. Because `parent[i] < i` by
    construction, one reverse pass accumulates them.
    """
    n_clones = parent.size
    masks = np.eye(n_clones, dtype=bool)
    for i in range(n_clones - 1, -1, -1):
        if parent[i] >= 0:
            masks[parent[i]] |= masks[i]

    return masks


##


def _sample_target_positions(n_vars, rng):
    """
    Sample unique MT positions from within MAESTER target gene bodies.

    NB: the baseline variant filter masks anything outside these ranges, so a
    simulated AFM with positions drawn uniformly over the MT genome would lose most
    of its variants at the first filtering step.
    """
    sites = np.unique(np.concatenate([
        np.arange(start, end + 1) for _, start, end in MAESTER_genes_positions
    ]))
    if n_vars > sites.size:
        raise ValueError(
            f'Cannot simulate {n_vars} variants: only {sites.size} MAESTER target '
            f'sites are available.'
        )

    return np.sort(rng.choice(sites, size=n_vars, replace=False))


##


def simulate_afm(
    n_cells: int = 500,
    n_clones: int = 10,
    n_root_clones: int | None = None,
    frac_double_variants: float = 0.0,
    frac_noisy_variants: float = 0.3,
    noise_prevalence: tuple[float, float] = (0.05, 0.2),
    max_depth: int | None = None,
    split_size: int | tuple[int, int] = (1, 3),
    range_skew: float = 1.0,
    min_max_ratio_clones: float = 0.2,
    mean_coverage: float = 50,
    sd_coverage: float = 10,
    coverage_cell_cv: float = 0.0,
    coverage_site_cv: float = 0.0,
    overdispersion: float = 1.0,
    overdispersion_background: float = 1.0,
    overdispersion_noise_frac: float = 1.0,
    mean_molecules: float | None = None,
    molecule_cv: float = 0.0,
    molecule_coverage_exponent: float = 0.0,
    af_positive: float = 0.05,
    af_positive_noise: float = 0.005,
    af_negative: float = 0.0001,
    af_beta: tuple[float, float] | None = None,
    af_beta_noise: tuple[float, float] | None = None,
    af_beta_min: float = 0.0,
    af_depth_decay: float = 1.0,
    af_negative_frac: float | None = None,
    scLT_system: str = 'MAESTER',
    pp_method: str = 'maegatk',
    random_seed: int = 0,
) -> AnnData:
    """
    Simulate a noise-free Allele Frequency Matrix with a known clonal structure.

    A population of `n_cells` cells is partitioned into `n_clones` clones, and each
    clone is marked by exactly one unique variant, so the matrix has `n_clones`
    variants unless `frac_double_variants` is raised. `n_root_clones` of them descend straight from the root and are
    therefore entirely independent, sharing no variant: a polytomy. The rest descend
    from another clone, representing a lineage split in which a further variant was
    acquired late within an already established clone. Those splits nest up to
    `max_depth` levels, so a clone deep in the tree carries a whole chain of
    ancestral variants. A cell carries the variant of its own clone and those of all
    its ancestors.

    Read counts are then drawn from the two-component binomial mixture that MiTo's
    genotyping assumes, run in reverse. Depth is normal with mean `mean_coverage` and
    standard deviation `sd_coverage`, and alternative counts are binomial given that
    depth, at `af_negative` for a cell that does not carry the variant and at
    `af_positive` for one that does, or `af_positive_noise` if the variant is a noisy
    one. Noise is therefore weak as well as tree-blind, which is what lets the allele
    frequency itself carry information about which variants are real. The noiseless
    genotypes are kept in ``.layers['genotype']``, so the truth behind the counts
    stays available.

    The result carries everything the MiTo preprocessing API expects, so it can be
    passed directly to :func:`mito.pp.filter_cells` and :func:`mito.pp.filter_afm`.

    Parameters
    ----------
    n_cells : int, optional
        Number of cells. Default is 500.
    n_clones : int, optional
        Number of clones, and therefore also the number of variants, since each
        clone is marked by exactly one. Default is 10.
    frac_double_variants : float, optional
        Fraction of clones marked by a second variant. Every clone is marked by one
        unique variant; once the tree is built, this fraction of them is sampled and
        given one more, so the matrix has ``n_clones * (1 + frac_double_variants)``
        variants. The two variants of a doubled clone are carried by exactly the same
        cells and so are indistinguishable from one another by presence/absence
        alone, as they are in real data. Default is 0.
    frac_noisy_variants : float, optional
        Fraction of the variants in the final matrix that carry no clonal signal.
        These are present in a random set of cells drawn without regard to the tree,
        so they cut across lineages and mark no clade at all, and they sit at the
        lower `af_positive_noise` allele frequency in the cells that carry them.
        Default is 0.3.
    noise_prevalence : tuple of float, optional
        Range from which the prevalence of each noisy variant is drawn, i.e. the
        probability that any given cell carries it. Drawn once per variant, so noisy
        variants differ in how widespread they are. Default is ``(0.05, 0.2)``.
    n_root_clones : int, optional
        Number of clones descending straight from the root, i.e. the width of the
        top-level polytomy. This is the main dial on clonal structure: setting it to
        `n_clones` gives one big polytomy of fully independent clones sharing no
        variant, while setting it to 1 gives a single founding clone that splits
        recursively into a deep hierarchy. Defaults to half of `n_clones`, giving a
        mix of independent and nested clones.
    max_depth : int or None, optional
        Deepest level of nesting below the root, or None to leave it unconstrained
        and let the depth emerge from the splits. 1 forces a pure polytomy, 2 allows
        one round of lineage splits, and higher values allow splits of splits. With
        a cap in place the tree is grown depth-first so the cap is actually reached;
        without one, the parent of each split is drawn uniformly. Default is None.
    split_size : int or tuple of int, optional
        Number of children produced by a splitting event. 1 gives a chain, in which
        a clone simply acquires a further variant; 2 gives a bifurcation into two
        sibling subclones; larger values give a nested polytomy. Pass a ``(low,
        high)`` tuple to draw the size of each split uniformly from that range, and
        so mix chains, bifurcations and polytomies in one tree. Default is ``(1, 3)``.
    range_skew : float, optional
        Bias applied to the `split_size` draw. The weight of value k is ``range_skew ** (k - low)``, so
        1 leaves the draw uniform, values below 1 concentrate it on the low end, and
        values above 1 on the high end. Default is 1.
    min_max_ratio_clones : float, optional
        Ratio between the smallest and the largest clone. 1 gives clones of equal
        size; smaller values give a more skewed distribution, with the sizes spaced
        geometrically in between and handed out independently of where a clone sits
        in the tree. Default is 0.2, i.e. a five-fold spread.
    mean_coverage : float, optional
        Mean sequencing depth per cell and site, drawn from a normal distribution
        and clipped at 1. Default is 50, comfortably above the median target coverage
        of 25 that `mito.pp.filter_cells` requires under "filter2".
    sd_coverage : float, optional
        Standard deviation of the sequencing depth. Only used when both coverage
        CVs are 0. Default is 10.
    coverage_cell_cv : float, optional
        Coefficient of variation of a per-cell multiplicative coverage effect
        (library size). Setting either CV above 0 switches the depth model from a
        normal draw to gamma-Poisson, which is right-skewed as real coverage is.
        Measured at roughly 0.6 on MAESTER data. Default is 0, i.e. homogeneous.
    coverage_site_cv : float, optional
        The same, per site (capture efficiency). Measured at roughly 0.8 on
        MAESTER data. Default is 0.
    overdispersion : float, optional
        Ratio of the variance of the alternative-allele counts to the binomial
        variance, i.e. the beta-binomial dispersion factor. 1 draws from the
        binomial that MiTo's genotyping assumes; real MAESTER counts sit closer to
        3, with a long tail. Values above 1 widen the counts without changing their
        mean, so any test calibrated against a binomial null becomes anti-
        conservative. Applies to the CARRIER component only; measured at roughly 6
        on MAESTER data, where heteroplasmy drifts between cells of a clone.
        Default is 1.
    overdispersion_background : float, optional
        The same, for cells that do NOT carry the variant. Sequencing error is a
        fixed-rate process and measures at roughly 1.2, i.e. very nearly binomial,
        so this should stay close to 1 even when `overdispersion` is high. Default
        is 1.
    overdispersion_noise_frac : float, optional
        Fraction of the clonal overdispersion applied to the positive cells of the
        *noisy* variants: their dispersion is
        ``1 + frac * (overdispersion - 1)``. Artefacts have no heteroplasmy to
        drift, so giving them the full clonal dispersion lets them draw a high rate
        occasionally, pick up many reads, and erode the count separation from real
        variants. Default is 1, i.e. the same dispersion as clonal variants.
    mean_molecules : float or None, optional
        Mean number of captured transcripts per cell-site. When set, carrier counts
        are drawn from a molecule-level process instead of a beta-binomial, and
        `overdispersion` is ignored for them (the background and the artefacts keep
        their own settings). Reads are redundant copies of the few molecules
        actually captured, so the sampling unit is the molecule: dropout then
        emerges as ``(1 - af) ** mean_molecules`` rather than being calibrated, and
        the effective concentration is coverage-invariant, which is what real
        MAESTER data shows. The concentration measured on MDA_clones is ~50, so
        that is a reasonable starting value. Default is None (beta-binomial draw).
    molecule_cv : float, optional
        Coefficient of variation of the molecule count, as a gamma-Poisson mixture.
        0 gives a plain Poisson. Default is 0.0.
    molecule_coverage_exponent : float, optional
        How the molecule count scales with read depth, ``n_mol ~ depth ** e``. 0
        decouples capture from depth, 1 makes reads and molecules proportional (and
        recovers the depth-independent dispersion of the beta-binomial draw). The
        default 0 reproduces the measured ``log phi ~ 0.56 log coverage`` on
        MDA_clones (simulated 0.58); raising it flattens that relation. Default
        is 0.0.
    af_positive : float, optional
        Binomial success probability for a cell that carries a clonal variant, i.e.
        the expected allele frequency in a positive cell. Default is 0.05.
    af_positive_noise : float, optional
        The same, for a cell that carries a noisy variant. Kept well below
        `af_positive` so that noise is weak as well as scattered: a noisy variant is
        detectable but sits at a low allele frequency in the cells that carry it, as
        background and RNA-editing artefacts do in real data. Setting it equal to
        `af_positive` makes noise indistinguishable from clonal signal by allele
        frequency alone. Default is 0.005.
    af_negative : float, optional
        Binomial success probability for a cell that does not carry the variant,
        soaking up sequencing error and background. Kept an order of magnitude below
        `af_positive_noise` so that the three components stay ordered and separable:
        a clonal variant is strong within its clade and near-absent outside it, a
        noisy variant is weak but genuinely present in the cells that carry it, and
        the background is weaker still. Raising it towards `af_positive_noise` drowns
        the noisy variants in spurious calls, at which point their prevalence stops
        governing where they are detected. Default is 0.0001. Ignored when
        `af_negative_frac` is given.
    af_beta : tuple of float, optional
        Shape parameters ``(a, b)`` of a Beta distribution from which each variant
        draws its own positive rate, replacing the fixed `af_positive` /
        `af_positive_noise` pair. Heteroplasmy then varies between variants as it
        does in real data, and — crucially — noisy variants draw from the same Beta
        as the clonal ones. Noise is no longer weaker, only unstructured, so no
        statistic computed on a single variant's counts can separate the two and
        only where a variant sits on the tree distinguishes them -- see
        `af_beta_noise` to relax that. Default is None, i.e. fixed rates.
    af_beta_noise : tuple of float, optional
        Shape parameters of a separate Beta from which the *noisy* variants draw
        their positive rate, so that artefacts can be weaker than clonal variants
        as they are in real data. Requires `af_beta`. Leaving it at None makes
        noise exactly as strong as signal, which is a deliberate worst case rather
        than a realistic one. Default is None.
    af_negative_frac : float, optional
        Background rate expressed as a fraction of each variant's own positive rate,
        replacing the fixed `af_negative` floor. A strong variant then leaks
        proportionally more into the cells that do not carry it, as contamination
        and mis-mapping do. Default is None, i.e. a single global background.
    scLT_system : str, optional
        Value stored in ``.uns['scLT_system']``. Default is "MAESTER".
    pp_method : str, optional
        Value stored in ``.uns['pp_method']``. Default is "maegatk".
    random_seed : int, optional
        Seed for reproducibility. Default is 0.

    Returns
    -------
    AnnData
        The simulated Allele Frequency Matrix, of shape ``(n_cells, n_clones)``.
        ``X`` holds allele frequencies, 1 where a cell carries a variant and 0
        elsewhere, alongside the ``AD`` / ``DP``
        layers. Ground truth clone labels are in ``.obs['clone']``. ``.var`` records
        which clone each variant arose in and the clade carrying it, and
        ``.uns['simulation']`` holds the tree and every parameter.

    Examples
    --------
    >>> import mito as mt
    >>> afm = mt.ut.simulate_afm(n_cells=200, n_clones=5, random_seed=0)
    >>> afm.shape
    (200, 5)
    >>> sorted(afm.obs['clone'].unique())
    ['clone_0', 'clone_1', 'clone_2', 'clone_3', 'clone_4']
    """

    if n_clones < 1:
        raise ValueError(f'n_clones must be at least 1, got {n_clones}.')
    if n_cells < n_clones:
        raise ValueError(
            f'n_cells={n_cells} is not enough for {n_clones} clones. '
            f'Raise n_cells, or lower n_clones.'
        )
    if max_depth is not None and max_depth < 1:
        raise ValueError(f'max_depth must be at least 1 or None, got {max_depth}.')
    if isinstance(split_size, int):
        if split_size < 1:
            raise ValueError(f'split_size must be at least 1, got {split_size}.')
    elif not (len(split_size) == 2 and 1 <= split_size[0] <= split_size[1]):
        raise ValueError(
            f'split_size must be an int, or a (low, high) tuple with '
            f'1 <= low <= high, got {split_size}.'
        )
    if n_root_clones is None:
        n_root_clones = max(1, round(n_clones / 2))
    if range_skew <= 0:
        raise ValueError(f'range_skew must be positive, got {range_skew}.')
    if not 0 <= frac_double_variants <= 1:
        raise ValueError(
            f'frac_double_variants must be in [0, 1], got {frac_double_variants}.'
        )
    if mean_coverage <= 0:
        raise ValueError(f'mean_coverage must be positive, got {mean_coverage}.')
    if sd_coverage < 0:
        raise ValueError(f'sd_coverage must be non-negative, got {sd_coverage}.')
    for name, value in (('coverage_cell_cv', coverage_cell_cv),
                        ('coverage_site_cv', coverage_site_cv)):
        if value < 0:
            raise ValueError(f'{name} must be non-negative, got {value}.')
    for _n, _v in (('overdispersion', overdispersion),
                   ('overdispersion_background', overdispersion_background)):
        if _v < 1:
            raise ValueError(f'{_n} must be at least 1 (1 = binomial), got {_v}.')
    if mean_molecules is not None and mean_molecules <= 0:
        raise ValueError(
            f'mean_molecules must be positive or None, got {mean_molecules}.'
        )
    if molecule_cv < 0:
        raise ValueError(f'molecule_cv must be non-negative, got {molecule_cv}.')
    if not 0 <= molecule_coverage_exponent <= 1:
        raise ValueError(
            f'molecule_coverage_exponent must be in [0, 1], got '
            f'{molecule_coverage_exponent}.'
        )
    if not 0 <= overdispersion_noise_frac <= 1:
        raise ValueError(
            f'overdispersion_noise_frac must be in [0, 1], got '
            f'{overdispersion_noise_frac}.'
        )
    for name, value in (('af_positive', af_positive),
                        ('af_positive_noise', af_positive_noise),
                        ('af_negative', af_negative)):
        if not 0 <= value <= 1:
            raise ValueError(f'{name} must be in [0, 1], got {value}.')
    for _name, _v in (('af_beta', af_beta), ('af_beta_noise', af_beta_noise)):
        if _v is not None and not (len(_v) == 2 and all(x > 0 for x in _v)):
            raise ValueError(
                f'{_name} must be an (a, b) tuple of positive shape parameters, '
                f'got {_v}.'
            )
    if af_beta_noise is not None and af_beta is None:
        raise ValueError('af_beta_noise requires af_beta to be set as well.')
    if not 0 <= af_beta_min < 1:
        raise ValueError(f'af_beta_min must be in [0, 1), got {af_beta_min}.')
    if not 0 < af_depth_decay <= 1:
        raise ValueError(
            f'af_depth_decay must be in (0, 1], got {af_depth_decay}.'
        )
    if af_negative_frac is not None and not 0 <= af_negative_frac < 1:
        raise ValueError(
            f'af_negative_frac must be in [0, 1), got {af_negative_frac}.'
        )
    if not 0 <= frac_noisy_variants < 1:
        raise ValueError(
            f'frac_noisy_variants must be in [0, 1), got {frac_noisy_variants}.'
        )
    if not (len(noise_prevalence) == 2
            and 0 <= noise_prevalence[0] <= noise_prevalence[1] <= 1):
        raise ValueError(
            f'noise_prevalence must be a (low, high) tuple with '
            f'0 <= low <= high <= 1, got {noise_prevalence}.'
        )
    if not 1 <= n_root_clones <= n_clones:
        raise ValueError(
            f'n_root_clones must be in [1, n_clones={n_clones}], got {n_root_clones}.'
        )
    if not 0 < min_max_ratio_clones <= 1:
        raise ValueError(
            f'min_max_ratio_clones must be in (0, 1], got {min_max_ratio_clones}.'
        )

    rng = np.random.default_rng(random_seed)
    # Every clone is marked by one unique variant. A sample of clones is then given
    # a second one, carried by exactly the same cells as the first.
    muts_per_clone = np.ones(n_clones, dtype=int)
    n_doubled = int(round(n_clones * frac_double_variants))
    if n_doubled > 0:
        muts_per_clone[rng.choice(n_clones, size=n_doubled, replace=False)] = 2

    # variant j arose on clone var_clone[j], or is pure noise if var_clone[j] < 0
    n_clonal_vars = int(muts_per_clone.sum())
    # solve n_noise / (n_clonal + n_noise) == frac_noisy_variants
    n_noisy_vars = int(round(
        n_clonal_vars * frac_noisy_variants / (1 - frac_noisy_variants)
    ))
    var_clone = np.concatenate([
        np.repeat(np.arange(n_clones), muts_per_clone),
        np.full(n_noisy_vars, -1, dtype=int),
    ])
    n_vars = n_clonal_vars + n_noisy_vars

    # NB: sample the positions up front. This validates n_vars against the target
    # sites available before anything of size (n_cells, n_vars) is allocated.
    positions = _sample_target_positions(n_vars, rng)

    # -- clonal structure ---------------------------------------------------
    parent, depth = _clonal_tree(
        n_clones, n_root_clones, max_depth, split_size, range_skew, rng
    )
    clades = _descendant_masks(parent)

    sizes = _clone_sizes(n_cells, n_clones, min_max_ratio_clones, rng)
    clone_idx = np.repeat(np.arange(n_clones), sizes)
    labels = np.array([f'clone_{i}' for i in clone_idx])

    # -- genotypes ----------------------------------------------------------
    # A cell carries variant j if its own clone descends from the clone the variant
    # arose on. Reading the clade masks by cell gives the whole matrix at once.
    genotypes = np.zeros((n_cells, n_vars), dtype=bool)
    genotypes[:, :n_clonal_vars] = clades[var_clone[:n_clonal_vars]][:, clone_idx].T

    # Noisy variants ignore the tree entirely: each gets its own prevalence, and
    # every cell carries it independently with that probability.
    if n_noisy_vars > 0:
        prevalence = rng.uniform(*noise_prevalence, size=n_noisy_vars)
        genotypes[:, n_clonal_vars:] = rng.random((n_cells, n_noisy_vars)) < prevalence

    # -- read counts --------------------------------------------------------
    # `genotypes` is indexed exactly as the AFM is, so using it to pick between the
    # two mixture components keeps every cell on the right side of it.

    def _depth():
        """
        Per-cell, per-site sequencing depth.

        Real target-site coverage is not homogeneous: it varies systematically
        between cells (library size) and between sites (capture efficiency), and
        the resulting distribution is strongly right-skewed. Two multiplicative
        gamma effects reproduce that; with both CVs at 0 this collapses back to
        the old normal draw.
        """
        mu = np.full((n_cells, n_vars), float(mean_coverage))
        for cv, size in ((coverage_cell_cv, (n_cells, 1)),
                         (coverage_site_cv, (1, n_vars))):
            if cv > 0:
                k = 1.0 / cv**2
                mu = mu * rng.gamma(k, 1.0 / k, size=size)
        if coverage_cell_cv > 0 or coverage_site_cv > 0:
            return np.clip(rng.poisson(mu), 1, None).astype(int)

        return np.clip(
            np.rint(rng.normal(mu, sd_coverage)), 1, None,
        ).astype(int)

    def _draw_molecules(depth, mean_mol):
        """
        Captured transcripts per cell-site.

        Coupled to read depth as ``mean_mol * (depth / mean_coverage) ** e``.

        e = 0 decouples them, which is the mechanistic default: read depth is a
        sequencing decision, molecule capture is biology. That does NOT imply a
        log-log slope of 1 for phi against coverage -- from
        ``phi = 1 + (n - 1) / (n_mol + 1)`` the slope is ``(phi - 1) / phi``, so
        around phi ~ 5 it is ~0.6, which is what MAESTER data shows (0.56,
        r = 0.70). Raising e flattens the relation; e = 1 removes it entirely and
        recovers the depth-independent dispersion of the beta-binomial draw.
        """
        mu = mean_mol * (depth / max(float(mean_coverage), 1e-9)) ** molecule_coverage_exponent
        if molecule_cv > 0:
            k = 1.0 / molecule_cv**2
            mu = rng.gamma(k, mu / k)
        return np.maximum(rng.poisson(np.maximum(mu, 1e-9)), 1)

    def _draw_alt_mol(depth, p, mean_mol):
        """
        Alternative-allele counts from a molecule-level process.

        Reads are redundant copies of the few transcripts actually captured, so the
        sampling unit is the molecule, not the read:

            n_mol ~ Poisson/gamma-Poisson    captured transcripts
            alt   ~ Binomial(n_mol, h)       heteroplasmy at the molecule level
            AD    ~ Binomial(depth, alt / n_mol)   reads resample the molecules

        Two things fall out that the beta-binomial draw has to be told. Dropout is a
        consequence -- a carrier with ``alt == 0`` has AD == 0 no matter how deeply
        it is sequenced, with probability ``(1 - h) ** n_mol`` -- so there is no
        dropout parameter to calibrate. And the effective concentration is ``n_mol``,
        a property of the counts rather than of the read depth, so phi grows with
        coverage the way it does in real data instead of being pinned flat.
        """
        n_mol = _draw_molecules(depth, mean_mol)
        alt = rng.binomial(n_mol, np.broadcast_to(p, depth.shape))

        return rng.binomial(depth, alt / n_mol)

    def _draw_alt(depth, p, phi):
        """
        Alternative-allele counts, binomial or beta-binomial.

        NB: the two mixture components are not equally dispersed, and modelling them
        with one factor is wrong. Measured within homogeneous groups on MAESTER
        data, CARRIERS sit at phi ~ 6 -- mitochondrial heteroplasmy drifts between
        cells of a clone, so the rate itself varies -- while NON-CARRIERS sit at
        phi ~ 1.2, because sequencing error is a fixed-rate process and therefore
        very nearly binomial. Drawing both from one overdispersed distribution makes
        the background far heavier-tailed than reality.
        """
        if phi <= 1:
            return rng.binomial(depth, p)
        # phi = 1 + (n_bar - 1) * rho  ->  concentration of the Beta prior
        rho = np.clip((phi - 1) / max(depth.mean() - 1, 1e-9), 1e-6, 0.999)
        conc = (1 - rho) / rho
        pp = np.broadcast_to(p, depth.shape)
        theta = rng.beta(np.maximum(pp * conc, 1e-9),
                         np.maximum((1 - pp) * conc, 1e-9))

        return rng.binomial(depth, theta)

    # NB: a noisy variant is weaker in its positive cells than a real clonal one, not
    # merely more scattered across the tree. Drawing both at `af_positive` would make
    # the two indistinguishable from the counts alone, leaving the AF- and AD-based
    # variant filters nothing to discriminate on and only the phylogeny to tell them
    # apart. The positive component is therefore per-variant.
    if af_beta is None:
        p_positive = np.where(var_clone < 0, af_positive_noise, af_positive)
        p_background = np.full(n_vars, af_negative, dtype=float)
    else:
        # Heteroplasmy varies between variants rather than being shared, so each
        # variant draws its own rate. Noisy variants draw from the same Beta as the
        # clonal ones: they are then no weaker, only unstructured, and nothing in
        # the counts of a single cell distinguishes the two. Only where a variant
        # sits on the tree can tell them apart.
        p_positive = _draw_beta(af_beta, n_vars, af_beta_min, rng)
        if af_beta_noise is not None:
            # Real artefacts are not merely mis-placed clonal variants: they are
            # weaker as well as unstructured (on MAESTER data roughly a quarter of
            # the alternative reads per positive cell). Drawing them from the same
            # Beta as the clonal variants makes the benchmark strictly harder than
            # reality, because it removes the count-based evidence that separates
            # them and leaves only the phylogeny.
            noise_mask = var_clone < 0
            # NB: the floor applies to the clonal draw only. Noisy variants are
            # meant to be weak, so truncating them too would undo the separation.
            p_positive[noise_mask] = _draw_beta(
                af_beta_noise, int(noise_mask.sum()), 0.0, rng
            )
        p_background = np.full(n_vars, af_negative, dtype=float)

    if af_depth_decay != 1.0:
        # NB: an ancestral variant arose earlier and has had more generations to
        # expand within its lineage, so it sits at a higher heteroplasmy and is
        # correspondingly harder to lose. Recent variants are weaker and drop out
        # more. Depth 1 keeps its drawn rate; each further level scales it by
        # `af_depth_decay`. The floor still applies to the final rate, so decay
        # cannot push a variant below detectability.
        var_depth = np.ones(n_vars, dtype=float)
        clonal = var_clone >= 0
        var_depth[clonal] = depth[var_clone[clonal]]
        p_positive = np.where(clonal,
                              p_positive * af_depth_decay ** (var_depth - 1),
                              p_positive)
        if af_beta is not None and af_beta_min > 0:
            # NB: floor the CLONAL rates only. Applying it to the noisy variants
            # would raise them to the floor and undo their weakness entirely.
            p_positive = np.where(clonal,
                                  np.maximum(p_positive, af_beta_min),
                                  p_positive)

    if af_negative_frac is not None:
        # Background scales with the variant's own strength, as contamination and
        # mis-mapping do, instead of being one global floor.
        p_background = af_negative_frac * p_positive

    # NB: DP is the coverage of the SITE, defined for every cell whether or not an
    # alternative read is seen there -- the denominator the genotyping needs, and what
    # `mt.io.make_afm` assembles from the pre-processing coverage table.
    site_coverage = _depth()
    # NB: overdispersion is heteroplasmy drift, which is a property of a real
    # mitochondrial variant. An artefact (mis-mapping, RNA editing) has no
    # heteroplasmy to drift, so its positive cells are drawn at
    # `overdispersion_noise_frac` of the clonal dispersion -- otherwise noisy
    # variants occasionally draw a high rate, pick up many reads, and close the
    # count separation from real clonal variants.
    _od_noise = 1.0 + overdispersion_noise_frac * (overdispersion - 1.0)
    if mean_molecules is not None:
        # Artefacts carry no heteroplasmy, so they have no molecule-level variance
        # to inherit. `overdispersion_noise_frac` is preserved by giving them a
        # LARGER effective molecule count: phi - 1 = (n - 1) / (n_mol + 1), so
        # scaling phi - 1 by `frac` means dividing n_mol + 1 by it.
        _mol_noise = (mean_molecules + 1.0) / max(overdispersion_noise_frac, 1e-6) - 1.0
        _pos = np.where(
            (var_clone >= 0)[None, :],
            _draw_alt_mol(site_coverage, p_positive, mean_molecules),
            _draw_alt_mol(site_coverage, p_positive, _mol_noise),
        )
        # Sequencing error is a read-level, fixed-rate process -- no molecules
        # involved -- so the background stays where it was, near-binomial.
        AD = np.where(
            genotypes, _pos,
            _draw_alt(site_coverage, p_background, overdispersion_background),
        )
    else:
        AD = np.where(
            genotypes,
            np.where(
                (var_clone >= 0)[None, :],
                _draw_alt(site_coverage, p_positive, overdispersion),
                _draw_alt(site_coverage, p_positive, _od_noise),
            ),
            _draw_alt(site_coverage, p_background, overdispersion_background),
        )
    AF = (AD / site_coverage).astype(np.float32)
    DP = site_coverage.astype(np.int16)

    # -- assembly -----------------------------------------------------------
    bases = np.array(['A', 'C', 'G', 'T'])
    ref = rng.choice(bases, size=n_vars)
    alt = np.array([rng.choice(bases[bases != r]) for r in ref])
    var_names = [f'{p}_{r}>{a}' for p, r, a in zip(positions, ref, alt, strict=True)]

    obs = pd.DataFrame(
        {
            'clone': pd.Categorical(labels),
            'mean_site_coverage': site_coverage.mean(axis=1),
            'median_target_site_coverage': np.median(site_coverage, axis=1),
            'frac_target_site_covered': (site_coverage > 0).mean(axis=1),
            'nUMIs': DP.sum(axis=1),
        },
        index=[f'CELL{i:04d}' for i in range(n_cells)],
    )

    clade_names = [';'.join(f'clone_{i}' for i in np.flatnonzero(c)) for c in clades]
    var = pd.DataFrame(
        {
            'pos': positions, 'ref': ref, 'alt': alt,
            'clone_of_origin': [f'clone_{i}' if i >= 0 else '' for i in var_clone],
            'clade': [clade_names[i] if i >= 0 else '' for i in var_clone],
            'n_clones_carrying': [
                int(clades[i].sum()) if i >= 0 else 0 for i in var_clone
            ],
            'depth': [int(depth[i]) if i >= 0 else 0 for i in var_clone],
            'is_noise': var_clone < 0,
        },
        index=var_names,
    )

    afm = AnnData(
        X=csr_matrix(AF),
        obs=obs,
        var=var,
        layers={
            'AD': csr_matrix(AD.astype(np.int16)),
            'DP': DP,
            'genotype': csr_matrix(genotypes.astype(np.int8)),
        },
        uns={
            'scLT_system': scLT_system,
            'pp_method': pp_method,
            'simulation': {
                'n_cells': n_cells,
                'n_clones': n_clones,
                'n_vars': n_vars,
                'frac_double_variants': frac_double_variants,
                'frac_noisy_variants': frac_noisy_variants,
                'noise_prevalence': noise_prevalence,
                'n_clonal_vars': n_clonal_vars,
                'n_noisy_vars': n_noisy_vars,
                'n_doubled_clones': int((muts_per_clone == 2).sum()),
                'range_skew': range_skew,
                'muts_per_clone': muts_per_clone.tolist(),
                'n_root_clones': n_root_clones,
                'max_depth': max_depth,
                'split_size': split_size,
                'min_max_ratio_clones': min_max_ratio_clones,
                'mean_coverage': mean_coverage,
                'sd_coverage': sd_coverage,
                'coverage_cell_cv': coverage_cell_cv,
                'coverage_site_cv': coverage_site_cv,
                'overdispersion': overdispersion,
                'overdispersion_background': overdispersion_background,
                'overdispersion_noise_frac': overdispersion_noise_frac,
                'mean_molecules': mean_molecules,
                'molecule_cv': molecule_cv,
                'molecule_coverage_exponent': molecule_coverage_exponent,
                'af_positive': af_positive,
                'af_positive_noise': af_positive_noise,
                'af_negative': af_negative,
                'af_beta': af_beta,
                'af_beta_noise': af_beta_noise,
                'af_beta_min': af_beta_min,
                'af_depth_decay': af_depth_decay,
                'af_negative_frac': af_negative_frac,
                'p_positive': p_positive.tolist(),
                'p_background': p_background.tolist(),
                'random_seed': random_seed,
                'clone_sizes': sizes.tolist(),
                'parent': [
                    'root' if p < 0 else f'clone_{p}' for p in parent
                ],
                'clades': clade_names,
                'depth': depth.tolist(),
                'realised_depth': int(depth.max()),
                'n_lineage_splits': int((parent >= 0).sum()),
            },
        },
    )

    afm.var['quality'] = 35.0

    return afm


##