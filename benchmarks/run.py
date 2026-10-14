"""
Run MiTo on the benchmark datasets and score it.

    python benchmarks/run.py baseline            # MDA x3, MAESTER sims, pL1000 Filter-2
    python benchmarks/run.py mitodrift           # MitoDrift's published tree, same metrics
    python benchmarks/run.py <target> --tag X    # results/<target>[_X].csv

Every row carries the dataset, the configuration and the metrics of metrics.py, so runs
with different settings can be concatenated and compared.
"""

import argparse
import logging
import os
import sys
import time
import warnings

sys.setrecursionlimit(50000)
warnings.filterwarnings('ignore')
logging.disable(logging.INFO)

import pandas as pd

import mito as mt

sys.path.insert(0, os.path.dirname(__file__))
import datasets as ds  # noqa: E402
import metrics as me  # noqa: E402

RESULTS = os.path.join(os.path.dirname(__file__), 'results')


def run_mito(afm, truth, filter_kwargs=None, annot_kwargs=None, strata=None, **meta):
    """filter_afm -> UPGMA -> annotate_clones; one row per annotate_clones setting."""
    t0 = time.time()
    afm = afm.copy()
    n_in = afm.shape[0]
    mt.pp.filter_afm(afm, **(filter_kwargs or {}))
    tree = mt.tl.build_tree(afm, solver='UPMGA')
    rows = []
    for ak in (annot_kwargs or [{}]):
        mt.tl.annotate_clones(tree, afm, **ak)
        lab = afm.obs['MiTo_clone'].astype(str)
        row = dict(**meta, **{f'f_{k}': v for k, v in (filter_kwargs or {}).items()},
                   **{f'a_{k}': v for k, v in ak.items()},
                   cells_in=n_in, cells_kept=afm.shape[0], n_chars=afm.shape[1],
                   **me.partition_scores(lab, truth))
        clones = [set(lab.index[lab == L]) for L in lab.unique() if L != me.UNASSIGNED]
        if strata is not None and clones:
            pr = me.clade_pr(truth, clones, strata=strata).iloc[0]
            row.update({k: pr[k] for k in pr.index if k.startswith(('recall', 'precision'))})
        row['secs'] = round(time.time() - t0)
        rows.append(row)
        print({k: row[k] for k in ('dataset', 'yield_', 'n_labels', 'ARI', 'ARI_all') if k in row},
              flush=True)
    return rows


def baseline():
    rows = []
    for name in ('MDA_clones', 'MDA_lung', 'MDA_PT'):
        afm, truth = ds.mda(name)
        rows += run_mito(afm, truth, dataset=name)
    afm, truth = ds.pl1000_filter2()
    rows += run_mito(afm, truth, annot_kwargs=[{'tau': 0.25}, {'tau': 0.1}],
                     strata=me.size_strata(truth), dataset='pL1000_filter2')
    for nk, topo, seed in ds.SIM_GRID:
        afm, truth = ds.sim(nk, topo, seed)
        rows += run_mito(afm, truth, dataset='sim_maester', clones=nk, topo=topo, seed=seed)
    return rows


def reader():
    """MDA x3 with the shipped (migrated) AFMs vs re-read by the current reader."""
    rows = []
    for name in ('MDA_clones', 'MDA_lung', 'MDA_PT'):
        for r in ('shipped', 'current'):
            afm, truth = ds.mda(name, reader=r)
            rows += run_mito(afm, truth, annot_kwargs=[{'tau': 0.25}, {'tau': 0.1}],
                             dataset=name, reader=r)
    return rows


def mitodrift():
    truth = ds.pl1000_truth()
    t = me.read_mitodrift_tree(os.path.join(ds.PL1000, 'tree_annotated.newick'))
    sets, conf = me.tree_clades(t)
    strata = me.size_strata(truth)
    rows = []
    for c in (0.05, 0.125, 0.25, 0.5):
        lab = me.assign_clones_polytomy(me.trim_tree(t, c, collapse_trivial=False))
        lab = lab.where(~lab.str.startswith('_root_single_'), me.UNASSIGNED)
        row = dict(dataset='pL1000', method='MitoDrift', conf=c, **me.partition_scores(lab, truth),
                   ARI_gt1pct=round(me.mitodrift_ari(t, truth, c, 0.01), 4),
                   ARI_gt025pct=round(me.mitodrift_ari(t, truth, c, 0.0025), 4))
        pr = me.clade_pr(truth, sets, conf, cutoffs=(c,), strata=strata).iloc[0]
        row.update({k: pr[k] for k in pr.index if k.startswith(('recall', 'precision'))})
        rows.append(row)
        print(row, flush=True)
    return rows


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('target', choices=['baseline', 'mitodrift', 'reader'])
    p.add_argument('--tag', default='')
    args = p.parse_args()
    rows = globals()[args.target]()
    os.makedirs(RESULTS, exist_ok=True)
    out = os.path.join(RESULTS, f'{args.target}{"_" + args.tag if args.tag else ""}.csv')
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f'-> {out}', flush=True)
