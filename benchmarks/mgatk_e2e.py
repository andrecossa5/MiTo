"""
Phase 1A, end to end: MiTo on pL1000 mgatk with read-level candidate filters and
strand-aware genotyping, scored on the LARRY cells.

Candidate sets come from the site table written by mgatk_filter_study.py. Genotyping
variants are injected by wrapping `call_genotypes` inside filter_afm (no package change):

    binomial      current per-cell binomial test on fw+rev reads
    both_strands  binomial AND >= 1 read on each strand
    dedup         binomial on duplicate-scaled counts (AD, DP / `dup`), dup = 2.1 measured
    both+dedup    both of the above

Usage: python mgatk_e2e.py [--cells labelled|all]
"""

import argparse
import logging
import os
import sys
import warnings

sys.setrecursionlimit(50000)
warnings.filterwarnings('ignore')
logging.disable(logging.INFO)

import numpy as np
import pandas as pd
from anndata import AnnData
from scipy.sparse import csr_matrix

import mito as mt
import mito.pp.preprocessing as pre
from mito.pp import genotyping as geno

sys.path.insert(0, os.path.dirname(__file__))
import datasets as ds  # noqa: E402
import metrics as me  # noqa: E402
import mgatk_strand as ms  # noqa: E402
from run import run_mito  # noqa: E402

RESULTS = os.path.join(os.path.dirname(__file__), 'results')
_orig_call = geno.call_genotypes
DUP = 2.1

# No-op candidate thresholds: the candidate set is chosen before filter_afm
OPEN = dict(cand_min_site_cov=0, cand_min_frac_negative=0, cand_min_n_positive=1,
            cand_af_confident=0, cand_min_n_confident=0, cand_min_AD_in_positives=0,
            cand_min_DP_in_positives=0)


def make_caller(mode):
    def call(afm, alpha=1e-3, max_iter=50, copy=False):
        if mode in ('dedup', 'both+dedup'):
            AD, DP = afm.layers['AD'], afm.layers['DP']
            afm.layers['AD'] = csr_matrix(np.rint(AD.toarray() / DUP).astype(np.int32))
            afm.layers['DP'] = np.rint(np.asarray(DP) / DUP).astype(np.int32)
            _orig_call(afm, alpha=alpha, max_iter=max_iter)
            afm.layers['AD'], afm.layers['DP'] = AD, DP
        else:
            _orig_call(afm, alpha=alpha, max_iter=max_iter)
        if mode in ('both_strands', 'both+dedup'):
            both = (afm.layers['FW'].toarray() > 0) & (afm.layers['REV'].toarray() > 0)
            B = (afm.layers['bin'].toarray() > 0) & both
            afm.layers['bin'] = csr_matrix(B.astype(np.int8))
            afm.var['n_carriers'] = B.sum(0)
            afm.var['prevalence'] = B.mean(0)
        return afm if copy else None
    return call


def build_afm(d, site_mask, cell_mask):
    FW = d['FW'][cell_mask][:, site_mask]
    REV = d['REV'][cell_mask][:, site_mask]
    DP = d['DP'][cell_mask][:, site_mask]
    AD = (FW + REV).toarray()
    DP = np.maximum(DP, AD).astype(np.int32)
    var = pd.DataFrame(dict(pos=d['sites'][site_mask], ref=d['ref'][site_mask], alt=d['alt'][site_mask]))
    var.index = [f'{p}_{r}>{a}' for p, r, a in var.values]
    obs = pd.DataFrame(index=d['cells'][cell_mask])
    obs['mean_site_coverage'] = d['DP'][cell_mask].mean(1)
    return AnnData(X=csr_matrix((AD / np.maximum(DP, 1)).astype(np.float32)), obs=obs, var=var,
                   layers={'AD': csr_matrix(AD.astype(np.int32)), 'DP': DP,
                           'FW': csr_matrix(FW), 'REV': csr_matrix(REV)},
                   uns={'scLT_system': 'mtscATAC', 'pp_method': 'mgatk'})


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--cells', default='labelled', choices=['labelled', 'all'])
    args = p.parse_args()

    d = ms.load()
    st = pd.read_csv(os.path.join(ds.PL1000, 'cache', 'mgatk_site_table.csv'), index_col=0)
    truth = ds.pl1000_truth()
    strata = me.size_strata(truth)
    cells = np.isin(d['cells'], truth.index) if args.cells == 'labelled' else np.ones(len(d['cells']), bool)
    cells &= d['DP'].mean(1) >= 10

    n_all = d['DP'].shape[0]
    base = (st['mean_DP'] >= 5) & (st['n_carriers'] >= 2) & (st['n_carriers'] <= 0.5 * n_all)
    cands = {
        'conf3': base & (st['n_cells_conf_detected'] >= 3),
        'conf3_strand.5': base & (st['n_cells_conf_detected'] >= 3) & (st['strand_dep'] >= 0.5),
        'conf2_AD1.5': base & (st['n_cells_conf_detected'] >= 2) & (st['mean_AD_in_positives'] >= 1.5),
        'read_preset_AD3': base & (st['mean_AD_in_positives'] >= 3) & (st['mean_DP'] >= 20),
        'redeem_positions': base & st['R_pos'],
    }

    rows = []
    for cname, cmask in cands.items():
        site_mask = cmask.values
        if site_mask.sum() > 1000:
            top = st.loc[cmask, 'n_cells_conf_detected'].sort_values(ascending=False).index[:1000]
            site_mask = st.index.isin(top)
        afm = build_afm(d, site_mask, cells)
        for mode in ('binomial', 'both_strands', 'dedup', 'both+dedup'):
            pre.call_genotypes = make_caller(mode)
            try:
                rows += run_mito(afm, truth, filter_kwargs=OPEN,
                                 annot_kwargs=[{'tau': 0.25}, {'tau': 0.1}], strata=strata,
                                 dataset=f'pL1000_mgatk_{args.cells}', candidates=cname,
                                 n_cand=int(site_mask.sum()), genotyping=mode)
            except Exception as e:  # noqa: BLE001
                rows.append(dict(dataset='pL1000_mgatk', candidates=cname, genotyping=mode, error=repr(e)))
                print('FAILED', cname, mode, repr(e), flush=True)
            finally:
                pre.call_genotypes = _orig_call
            pd.DataFrame(rows).to_csv(os.path.join(RESULTS, f'mgatk_e2e_{args.cells}.csv'), index=False)
    print('ALL DONE', flush=True)


if __name__ == '__main__':
    main()
