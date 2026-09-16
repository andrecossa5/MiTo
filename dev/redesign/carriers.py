"""Carrier calls for the variant QC: per-cell binomial test against a per-variant background."""
import numpy as np
from scipy import stats


def binomial_carriers(AD, COV, alpha_cell=0.001, n_iter=10, return_background=False):
    # NB: n_iter=10 truncates the fixed point on some datasets: convergence took 7-18
    # updates across MDA_clones / MDA_lung / MDA_PT (MDA_PT at alpha 1e-3 needs 18), so the
    # benchmarked calls are ~0.3% short of the fixed point. Carrier sets only ever grew
    # (monotone, no cycles observed), so raising the cap is safe.
    """
    A cell carries variant j if its alt reads exceed the variant's own background.

    No mixture, no prior. For each variant the background rate is the pooled alt/total
    read ratio over the cells NOT called carriers (re-estimated until the carrier set
    stops changing, so a clone does not inflate the background of its own marker), with
    a one-read pseudocount so a site that never shows an alt read has a finite rate.
    Each cell is tested one-sided, P(X >= AD | COV, background) <= alpha_cell.

    What this buys over the alternatives, on MDA_PT:
      AD > 0          counts single reads at error-prone sites: variants spread over
                      barcode clones at AF ~0.006 (the sequencing-error level) got carriers,
                      passed the QC and fragmented large clones
      guarded mixture capped the prior at the global detection rate, so real mid-clone
                      and subclonal markers (AF ~0.02) got no carriers at all
    Here a single read is enough at a clean site and not at a noisy one.

    Returns an int8 (cells x variants) carrier matrix [, background rates].
    """
    AD = np.asarray(AD, dtype=np.int64)
    COV = np.maximum(np.asarray(COV, dtype=np.int64), AD)
    carrier = np.zeros(AD.shape, bool)
    for _ in range(n_iter):
        bg_ad = np.where(carrier, 0, AD).sum(0)
        bg_cov = np.where(carrier, 0, COV).sum(0)
        p0 = (bg_ad + 1.0)/(bg_cov + 1.0)
        pval = np.where(AD > 0, stats.binom.sf(AD - 1, COV, np.clip(p0, 1e-12, 1 - 1e-12)[None, :]), 1.0)
        new = pval <= alpha_cell
        if (new == carrier).all():
            break
        carrier = new
    B = carrier.astype(np.int8)
    return (B, p0) if return_background else B


def signal_to_background(AD, X, background, min_ratio=10.0):
    """
    Keep variants whose carriers stand clearly above the variant's background.

    ratio = median AF of cells with >= 1 alt read / background rate.

    Median AF of CALLED cells does not work: calls are exactly the cells that beat the
    background, so their AF is high by construction (MDA_clones 12818_G>A: 0.44, ratio 12).
    Over all cells with reads the same variant has median AF 0.20 against a background of
    3.6e-2 (ratio 5.5): a broad heteroplasmic variant, present at similar AF in cells of every
    clone. Such a variant gets a random-looking subset of cells called, looks like scattered
    noise, and the greedy four-gamete filter deletes the prevalent markers of the largest
    clone instead of it (136-cell clone: 96% -> 57% of cells assigned).

    Clone markers usually score in the tens to thousands (median 86 / 228 / 244 on
    MDA_clones / MDA_lung / MDA_PT), but the margin is not uniform: on MDA_lung two
    GT-enriched variants score 8.5 and 9.7 and are dropped at min_ratio=10 (no ARI cost
    there). Background single reads in non-carrier cells enter the median, so the ratio
    is conservative. Only min_ratio=10 was benchmarked end to end.

    Returns (keep mask, ratios); variants with no read get ratio 0.
    """
    readers = np.asarray(AD) >= 1
    af = np.where(readers, X, np.nan)
    with np.errstate(all='ignore'):
        med = np.nanmedian(af, axis=0)
    ratio = np.where(np.isfinite(med), med/np.maximum(background, 1e-12), 0.0)
    return ratio >= min_ratio, ratio
