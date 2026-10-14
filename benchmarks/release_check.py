"""
MiTo 0.5.0 release check: package defaults only, from the readers, on every benchmark.

    make_afm -> filter_cells (defaults) -> filter_afm (defaults) -> UPGMA ->
    clade_support (100 count replicates) -> annotate_clones (defaults; small_clones=True;
    min_support=0.5)

    MDA_clones / lung / PT   MAESTER, GBC truth (current reader, filter2 cells)
    pL1000 ReDeeM            redeemV (edge_trim 9 default), LARRY truth
    pL1000 mtscATAC          mgatk, molecules from the strand counts, LARRY truth
    24 MAESTER sims

Expected (from the benchmark runs): MDA .877/.746, .958/.903, .841/.453; ReDeeM
~.818/.644; mgatk ~.846/.644; sims mean .788.

Usage: python release_check.py [mda|pl1000|sims]   -> results/release_check_<target>.csv
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

sys.path.insert(0, os.path.dirname(__file__))
import datasets as ds  # noqa: E402
import metrics as me  # noqa: E402

RESULTS = os.path.join(os.path.dirname(__file__), 'results')


def run(afm, truth, strata, **meta):
    t0 = time.time()
    afm = afm.copy()
    mt.pp.filter_afm(afm)
    t_filter = time.time() - t0
    tree = mt.tl.build_tree(afm, solver='UPMGA')
    t_tree = time.time() - t0 - t_filter
    sup = mt.tl.clade_support(tree, afm, n_boot=100)
    t_support = time.time() - t0 - t_filter - t_tree
    rows = []
    for d2, ms in ((False, None), (True, None), (False, 0.5)):
        mt.tl.annotate_clones(tree, afm, small_clones=d2, min_support=ms)
        lab = afm.obs['MiTo_clone'].astype(str)
        row = dict(**meta, D2=d2, min_support=ms, version=mt.__version__, n_chars=afm.shape[1],
                   cells_kept=afm.shape[0], secs_filter=round(t_filter, 1), secs_tree=round(t_tree, 1),
                   secs_support=round(t_support, 1), median_support=round(float(sup.median()), 3),
                   **me.partition_scores(lab, truth))
        sets = [set(lab.index[lab == L]) for L in lab.unique() if L != me.UNASSIGNED]
        if strata is not None and sets:
            pr = me.clade_pr(truth, sets, strata=strata).iloc[0]
            row.update({k: pr[k] for k in pr.index if k.startswith(('recall', 'precision'))})
        rows.append(row)
        print({k: row.get(k) for k in ('dataset', 'seed', 'D2', 'min_support', 'secs_filter', 'secs_support',
                                         'n_chars', 'cells_kept', 'yield_',
                                         'ARI', 'ARI_all', 'recall')}, flush=True)
    return rows


def main():
    target = sys.argv[1]
    out = os.path.join(RESULTS, f'release_check_{target}.csv')
    rows = []
    if target == 'mda':
        for name in ('MDA_clones', 'MDA_lung', 'MDA_PT'):
            afm, truth = ds.mda(name, reader='current')
            strata = me.size_strata(truth, bins=((0.05, 1.0), (0.01, 0.05), (0.0, 0.01)))
            rows += run(afm, truth, strata, dataset=name)
            pd.DataFrame(rows).to_csv(out, index=False)
    elif target == 'pl1000':
        truth = ds.pl1000_truth()
        strata = me.size_strata(truth)
        afm = mt.io.make_afm(ds.PL1000, scLT_system='ReDeeM', pp_method='redeem-v')
        afm = ds._to_gex(afm, truth)
        mt.pp.filter_cells(afm)
        rows += run(afm, truth, strata, dataset='pL1000_ReDeeM', cells_in=afm.shape[0])
        pd.DataFrame(rows).to_csv(out, index=False)
        cache = os.path.join(ds.PL1000, 'cache', 'mgatk_molecules_reader.h5ad')
        if os.path.exists(cache):
            import scanpy as sc
            afm = sc.read(cache)
        else:
            afm = ds._to_gex(mt.io.make_afm(ds.PL1000, scLT_system='mtscATAC', pp_method='mgatk'), truth)
            afm.write(cache)
        mt.pp.filter_cells(afm)
        rows += run(afm, truth, strata, dataset='pL1000_mtscATAC', cells_in=afm.shape[0],
                    reads_per_molecule=afm.uns['molecules']['reads_per_molecule'],
                    alt_rule=afm.uns['molecules']['alt_rule'])
        pd.DataFrame(rows).to_csv(out, index=False)
    else:
        for nk, topo, seed in ds.SIM_GRID:
            afm, truth = ds.sim(nk, topo, seed)
            rows += run(afm, truth, None, dataset='sim_maester', clones=nk, topo=topo, seed=seed)
            pd.DataFrame(rows).to_csv(out, index=False)
    print('ALL DONE', flush=True)


if __name__ == '__main__':
    main()
