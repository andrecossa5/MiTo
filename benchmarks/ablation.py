"""
Which stages after the over-dispersion candidate stage still pay for themselves?

Package filter_afm defaults (candidates="overdispersion", max_candidates=1000), with one
downstream stage switched off at a time:

    full            all stages
    no_artefacts    filter_artefacts=False (dbSNP / REDIdb)
    prov_eq_final   clonality filter run on calls at geno_alpha, not geno_alpha/10
    no_clonality    clonality filter skipped
    no_snr          min_snr=0
    no_compat       four-gamete filter skipped
    minimal         candidates -> calls -> prevalence / >= 2 calls only
    thresholds      0.3.0 candidate stage (reference)

Each run is scored with and without the small-clone pass (annotate_clones(small_clones=True)).

Usage: python ablation.py mda | redeem | sims     -> results/ablation_<target>.csv
"""

import logging
import os
import sys
import time
import warnings

sys.setrecursionlimit(50000)
warnings.filterwarnings('ignore')
logging.disable(logging.WARNING)

import pandas as pd

import mito as mt
import mito.pp.preprocessing as pre

sys.path.insert(0, os.path.dirname(__file__))
import datasets as ds  # noqa: E402
import metrics as me  # noqa: E402

RESULTS = os.path.join(os.path.dirname(__file__), 'results')
_orig = {k: getattr(pre, k) for k in ('call_genotypes', 'filter_non_clonal_variants', 'filter_incompatible_variants')}


def _noop(afm, *a, copy=False, **kw):
    return afm if copy else None


def _calls_at_final(afm, alpha=1e-3, **kw):
    # filter_afm calls geno_alpha/10 first: raise it back to geno_alpha
    return _orig['call_genotypes'](afm, alpha=alpha * 10 if alpha < 1e-3 else alpha, **kw)


CONFIGS = {
    'full': ({}, {}),
    'no_artefacts': ({'filter_artefacts': False}, {}),
    'prov_eq_final': ({}, {'call_genotypes': _calls_at_final}),
    'no_clonality': ({}, {'filter_non_clonal_variants': _noop}),
    'no_snr': ({'min_snr': 0}, {}),
    'no_compat': ({}, {'filter_incompatible_variants': _noop}),
    'minimal': ({'filter_artefacts': False, 'min_snr': 0},
                {'filter_non_clonal_variants': _noop, 'filter_incompatible_variants': _noop}),
    'thresholds': ({'candidates': 'thresholds'}, {}),
}


def run(afm, truth, strata, **meta):
    rows = []
    for name, (kw, patch) in CONFIGS.items():
        t0 = time.time()
        a = afm.copy()
        for k, f in patch.items():
            setattr(pre, k, f)
        try:
            mt.pp.filter_afm(a, **kw)
        except Exception as e:  # noqa: BLE001
            rows.append(dict(**meta, config=name, error=repr(e)))
            print('FAILED', meta, name, repr(e), flush=True)
            continue
        finally:
            for k, f in _orig.items():
                setattr(pre, k, f)
        flow = a.uns['mito']['filter_afm']['flow'].set_index('stage')['n_vars_out']
        tree = mt.tl.build_tree(a, solver='UPMGA')
        for d2 in (False, True):
            mt.tl.annotate_clones(tree, a, small_clones=d2)
            lab = a.obs['MiTo_clone'].astype(str)
            row = dict(**meta, config=name, D2=d2, n_cand=int(flow.get('candidate cap', flow.iloc[0])),
                       n_chars=a.shape[1], cells_kept=a.shape[0], secs=round(time.time() - t0),
                       **me.partition_scores(lab, truth))
            sets = [set(lab.index[lab == L]) for L in lab.unique() if L != me.UNASSIGNED]
            if strata is not None and sets:
                pr = me.clade_pr(truth, sets, strata=strata).iloc[0]
                row.update({k: pr[k] for k in pr.index if k.startswith(('recall', 'precision'))})
            rows.append(row)
        print({k: row.get(k) for k in ('dataset', 'seed', 'config', 'n_cand', 'n_chars', 'yield_',
                                         'ARI', 'ARI_all', 'recall')}, flush=True)
    return rows


def main():
    target = sys.argv[1]
    out = os.path.join(RESULTS, f'ablation_{target}.csv')
    rows = []
    if target == 'mda':
        for name in ('MDA_clones', 'MDA_lung', 'MDA_PT'):
            afm, truth = ds.mda(name, reader='current')
            strata = me.size_strata(truth, bins=((0.05, 1.0), (0.01, 0.05), (0.0, 0.01)))
            rows += run(afm, truth, strata, dataset=name)
            pd.DataFrame(rows).to_csv(out, index=False)
    elif target == 'redeem':
        truth = ds.pl1000_truth()
        afm, _ = ds.pl1000_redeemv(edge_trim=9)
        afm = afm[afm.obs['mean_site_coverage'].values >= 10].copy()
        rows += run(afm, truth, me.size_strata(truth), dataset='pL1000_redeemv_trim9_qcells')
        pd.DataFrame(rows).to_csv(out, index=False)
    else:
        for nk, topo, seed in ds.SIM_GRID:
            afm, truth = ds.sim(nk, topo, seed)
            rows += run(afm, truth, None, dataset='sim_maester', clones=nk, topo=topo, seed=seed)
            pd.DataFrame(rows).to_csv(out, index=False)
    print('ALL DONE', flush=True)


if __name__ == '__main__':
    main()
