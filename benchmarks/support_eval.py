"""
Count-resampling clade support (mt.tl.clade_support): is it informative, and how should
annotate_clones use it?

Per dataset (package defaults: filter_cells -> filter_afm -> UPGMA), 100 replicates:

    tree level   every internal node a predicted clade, support as confidence: precision
                 / recall at support cutoffs (MitoDrift's metric; MitoDrift on pL1000:
                 .77 / .75 at .125, all 1,362 cells)
    partitions   evidence    annotate_clones defaults (reference)
                 evidence_s  same, clones whose clade support < s dissolved (unassigned)
                 maximal_s   MitoDrift-style: clones = the maximal clades with support >= s
                             (>= 2 cells), the other cells unassigned
                 with and without the small-clone pass (restricted to support >= s)

Usage: python support_eval.py [pl1000|mda]   -> results/support_eval_<target>.csv
"""

import logging
import os
import sys
import time
import warnings

sys.setrecursionlimit(50000)
warnings.filterwarnings('ignore')
logging.disable(logging.WARNING)

import numpy as np
import pandas as pd
import scanpy as sc

import mito as mt
from mito.tl.support import clade_support

sys.path.insert(0, os.path.dirname(__file__))
import datasets as ds  # noqa: E402
import metrics as me  # noqa: E402

RESULTS = os.path.join(os.path.dirname(__file__), 'results')
CUTS = (0.25, 0.5, 0.75, 0.9)


def maximal_clades(tree, support, s, min_cells=2):
    lab = pd.Series(me.UNASSIGNED, index=tree.leaves, dtype=object)
    stack = list(tree.children(tree.root))
    while stack:
        node = stack.pop()
        if tree.is_leaf(node):
            continue
        if support[node] >= s:
            leaves = tree.leaves_in_subtree(node)
            if len(leaves) >= min_cells:
                lab[leaves] = str(node)
        else:
            stack += tree.children(node)
    return lab


def score(lab, truth, strata, **meta):
    row = dict(**meta, **me.partition_scores(lab, truth))
    sets = [set(lab.index[lab == L]) for L in lab.unique() if L != me.UNASSIGNED]
    if sets:
        pr = me.clade_pr(truth, sets, strata=strata).iloc[0]
        row.update({k: pr[k] for k in pr.index if k.startswith(('recall', 'precision'))})
    return row


def evaluate(afm, truth, strata, name):
    rows = []
    t0 = time.time()
    tree = mt.tl.build_tree(afm, solver='UPMGA')
    sup = clade_support(tree, afm, n_boot=100)
    secs = round(time.time() - t0, 1)
    print(f'{name}: {afm.shape}, tree + 100 replicates {secs}s', flush=True)

    # tree level
    nodes = [x for x in tree.internal_nodes if x != tree.root]
    sets = [set(tree.leaves_in_subtree(x)) for x in nodes]
    conf = sup[nodes].values
    for c in (0.0,) + CUTS:
        pr = me.clade_pr(truth, sets, conf, cutoffs=(c,), strata=strata).iloc[0]
        rows.append(dict(dataset=name, level='tree', method='support', s=c, secs=secs,
                         **{k: pr[k] for k in pr.index if k.startswith(('recall', 'precision'))}))

    # partitions
    mt.tl.annotate_clones(tree, afm)
    ev = afm.obs['MiTo_clone'].astype(str)
    node_of = afm.uns['mito']['annotate_clones']['clones']['node'].to_dict() if len(ev.unique()) > 1 else {}
    mt.tl.annotate_clones(tree, afm, small_clones=True)
    ev_d2 = afm.obs['MiTo_clone'].astype(str)
    node_of_d2 = afm.uns['mito']['annotate_clones']['clones']['node'].to_dict()
    rows.append(score(ev, truth, strata, dataset=name, level='partition', method='evidence', s=np.nan, D2=False))
    rows.append(score(ev_d2, truth, strata, dataset=name, level='partition', method='evidence', s=np.nan, D2=True))
    for s in CUTS:
        for d2, lab, nodes_ in ((False, ev, node_of), (True, ev_d2, node_of_d2)):
            keep = {L for L, nd in nodes_.items() if sup.get(nd, 0) >= s}
            L = lab.where(lab.isin(keep), me.UNASSIGNED)
            rows.append(score(L, truth, strata, dataset=name, level='partition', method='evidence_s', s=s, D2=d2))
        rows.append(score(maximal_clades(tree, sup, s), truth, strata, dataset=name, level='partition',
                          method='maximal_s', s=s, D2=False))
    for r in rows[-(2 + 3 * len(CUTS)):]:
        print({k: r.get(k) for k in ('method', 's', 'D2', 'yield_', 'n_labels', 'ARI', 'ARI_all',
                                     'precision', 'recall')}, flush=True)
    return rows


def main():
    target = sys.argv[1]
    rows = []
    if target == 'pl1000':
        truth = ds.pl1000_truth()
        strata = me.size_strata(truth)
        afm = sc.read(os.path.join(ds.PL1000, 'cache', 'redeem_filtered_050.h5ad'))
        rows += evaluate(afm, truth, strata, 'pL1000_ReDeeM')
        m = sc.read(os.path.join(ds.PL1000, 'cache', 'mgatk_molecules_reader.h5ad'))
        mt.pp.filter_cells(m)
        mt.pp.filter_afm(m)
        rows += evaluate(m, truth, strata, 'pL1000_mtscATAC')
    else:
        for name in ('MDA_clones', 'MDA_lung', 'MDA_PT'):
            afm, truth = ds.mda(name, reader='current')
            mt.pp.filter_afm(afm)
            strata = me.size_strata(truth, bins=((0.05, 1.0), (0.01, 0.05), (0.0, 0.01)))
            rows += evaluate(afm, truth, strata, name)
    pd.DataFrame(rows).to_csv(os.path.join(RESULTS, f'support_eval_{target}.csv'), index=False)
    print('ALL DONE', flush=True)


if __name__ == '__main__':
    main()
