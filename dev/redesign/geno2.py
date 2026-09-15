"""Genotyping EM with a graph-smoothed local mixing weight (bb likelihood)."""
import numpy as np
from scipy import stats


def em_genotype(AD, COV, Wn, mode='bb', n_iter=40, tol=1e-4, phi_glob=None, guard=True):
    """
    Two-component mixture with the mixing weight made LOCAL: each cell's prior
    comes from its neighbours on the cosine graph. `bb` is the shared-dispersion
    beta-binomial that won the model comparison (asym / nb / zi were all wash or
    worse, and asym was actively harmful on real data).

    guard=True (default) prevents the mixture from calling cells without reads.
    Without it, when most cells have AD = 0 the two components are nearly
    indistinguishable, the mixing weight drifts (0.74-0.99 for some MDA_PT
    variants), and the prior alone pushes zero-read cells over the call threshold
    against their own likelihood: 61% of calls on MDA_PT and 50% on MDA_clones had
    no alternative read. Three changes:

      * initialise from the data (cells with AD >= 1), not from 0.5
      * cap each cell's prior at the detection rate of its neighbourhood (for a
        flat prior, the variant's global detection rate): the prior may not claim
        more carriers than there are cells with reads
      * no reads, no call: AD = 0 cells stay negative here; recovering dropouts is
        left to the explicit imputation step

    guard=False reproduces the previous behaviour.
    """
    nC, nV = AD.shape
    detect = AD >= 1
    g = detect.astype(float) if guard else np.full((nC, nV), 0.5)
    cap = np.maximum(Wn @ detect.astype(float), 1e-4) if guard else None
    phi = phi_glob if phi_glob is not None else 3.0
    rho = min(max((phi-1)/max(COV.mean()-1, 1e-9), 1e-6), .999)
    conc = (1-rho)/rho
    for _ in range(n_iter):
        p_pos = np.clip((g*AD).sum(0)/np.maximum((g*COV).sum(0), 1e-9), 1e-4, .5)
        p_neg = np.clip(((1-g)*AD).sum(0)/np.maximum(((1-g)*COV).sum(0), 1e-9), 1e-8, None)
        p_neg = np.minimum(p_neg, p_pos/5)
        L1 = stats.betabinom.pmf(AD, COV, p_pos[None, :]*conc, (1-p_pos[None, :])*conc)
        L0 = stats.betabinom.pmf(AD, COV, np.maximum(p_neg[None, :]*conc, 1e-9),
                                 np.maximum((1-p_neg[None, :])*conc, 1e-9))
        pi = np.clip(Wn @ g, 1e-4, 1-1e-4)
        if guard:
            pi = np.minimum(pi, cap)
        gn = (pi*L1)/np.maximum(pi*L1 + (1-pi)*L0, 1e-300)
        if guard:
            gn[~detect] = 0.0
        if np.abs(gn-g).max() < tol:
            g = gn; break
        g = gn
    return g, np.full(nV, phi)
