"""
Phase 1A follow-up: WHY does duplicate scaling fix mgatk genotyping?

Integer rounding of AD / f removes single-read observations (rint(1/2.1) = 0), so the
gain could come from that alone. Callers compared on the same candidates (conf3):

    binomial          current test on reads
    min2              binomial AND >= 2 alt reads
    min2_both         binomial AND >= 2 alt reads AND >= 1 read on each strand
    scaled_cont_f     continuous scaling: P(X >= AD/f | DP/f, p) via the regularised
                      incomplete beta (no rounding, single reads keep weight 1/f)
    betabinom_rho     beta-binomial test, intra-molecule correlation rho (variance
                      inflation 1 + (n-1) rho), p = variant error rate
"""

import os
import sys

import numpy as np
import pandas as pd
from scipy import special, stats
from scipy.sparse import csr_matrix

sys.path.insert(0, os.path.dirname(__file__))
import datasets as ds  # noqa: E402
import metrics as me  # noqa: E402
import mgatk_e2e as e2e  # noqa: E402
import mgatk_strand as ms  # noqa: E402
import mito.pp.preprocessing as pre  # noqa: E402
from mito.pp import genotyping as geno  # noqa: E402
from run import run_mito  # noqa: E402

RESULTS = os.path.join(os.path.dirname(__file__), 'results')


def cont_sf(k, n, p):
    """P(X >= k) for a binomial with non-integer k, n: I_p(k, n - k + 1)."""
    k = np.maximum(k, 1e-9)
    return special.betainc(k, np.maximum(n - k + 1, 1e-9), p)


def generic_caller(sf, alpha_scale=1.0, max_iter=50):
    """Binomial-style iterative background caller with a custom tail function sf(AD, DP, p)."""
    def call(afm, alpha=1e-3, max_iter=max_iter, copy=False):
        AD, COV = geno._counts(afm)
        carrier = np.zeros(AD.shape, bool)
        for _ in range(max_iter):
            p0 = geno._error_rates(AD, COV, carrier)
            pv = np.where(AD > 0, sf(AD, COV, np.clip(p0, 1e-12, 1 - 1e-12)[None, :]), 1.0)
            new = pv <= alpha * alpha_scale
            if (new == carrier).all():
                break
            carrier = new
        afm.layers['bin'] = csr_matrix(carrier.astype(np.int8))
        afm.layers['imputed'] = csr_matrix(np.zeros(carrier.shape, np.int8))
        afm.var['error_rate'] = p0
        afm.var['n_carriers'] = carrier.sum(0)
        afm.var['n_imputed'] = 0
        afm.var['prevalence'] = carrier.mean(0)
        afm.uns['genotyping'] = {'method': 'custom', 'converged': True}
        return afm if copy else None
    return call


def masked(base, rule):
    def call(afm, alpha=1e-3, max_iter=50, copy=False):
        base(afm, alpha=alpha, max_iter=max_iter)
        B = (afm.layers['bin'].toarray() > 0) & rule(afm)
        afm.layers['bin'] = csr_matrix(B.astype(np.int8))
        afm.var['n_carriers'] = B.sum(0)
        afm.var['prevalence'] = B.mean(0)
        return afm if copy else None
    return call


def main():
    d = ms.load()
    truth = ds.pl1000_truth()
    lab = np.isin(d['cells'], truth.index) & (d['DP'].mean(1) >= 10)
    st = pd.read_csv(os.path.join(ds.PL1000, 'cache', 'mgatk_site_table.csv'), index_col=0)
    n_all = d['DP'].shape[0]
    cand = ((st['mean_DP'] >= 5) & (st['n_carriers'] >= 2) & (st['n_carriers'] <= 0.5 * n_all)
            & (st['n_cells_conf_detected'] >= 3)).values
    afm = e2e.build_afm(d, cand, lab)
    strata = me.size_strata(truth)

    ad2 = lambda a: a.layers['AD'].toarray() >= 2                               # noqa: E731
    both = lambda a: (a.layers['FW'].toarray() > 0) & (a.layers['REV'].toarray() > 0)  # noqa: E731
    callers = {
        'binomial': e2e._orig_call,
        'min2': masked(e2e._orig_call, ad2),
        'min2_both': masked(e2e._orig_call, lambda a: ad2(a) & both(a)),
    }
    for f in (2.0, 3.0):
        callers[f'scaled_cont_{f}'] = generic_caller(lambda A, N, p, f=f: cont_sf(A / f, N / f, p))
    for rho in (0.05, 0.2, 0.5):
        def bb(A, N, p, rho=rho):
            ab = (1 - rho) / rho
            return stats.betabinom.sf(A - 1, N, p * ab, (1 - p) * ab)
        callers[f'betabinom_{rho}'] = generic_caller(bb)

    rows = []
    for name, caller in callers.items():
        pre.call_genotypes = caller
        try:
            rows += run_mito(afm, truth, filter_kwargs=e2e.OPEN,
                             annot_kwargs=[{'tau': 0.25}, {'tau': 0.1}], strata=strata,
                             dataset='pL1000_mgatk_labelled', caller=name)
        except Exception as e:  # noqa: BLE001
            rows.append(dict(caller=name, error=repr(e)))
            print('FAILED', name, repr(e), flush=True)
        finally:
            pre.call_genotypes = e2e._orig_call
        pd.DataFrame(rows).to_csv(os.path.join(RESULTS, 'mgatk_mechanism.csv'), index=False)
    print('ALL DONE', flush=True)


if __name__ == '__main__':
    main()
