"""
ReDeeM pL1000: MitoDrift (published) vs MiTo 0.3.0 (package defaults) vs MiTo proposed.

Every row is scored on the 1,362 LARRY cells with the same metrics: partition (yield,
ARI on assigned, ARI_all with abstention as singletons, clone precision / recall by
size stratum) and tree level (every internal node a candidate clade, as in the paper).

    MitoDrift tau=0.125 / 0.25   deposited tree (paper run); partition = top-level clades
    MiTo-0.3.0 raw redeemV       make_afm(redeem-v) package defaults (edge_trim=4) +
                                 filter_afm + UPGMA + annotate_clones defaults
    MiTo-0.3.0 Filter-2 input    the same MiTo on MitoDrift's exact input matrix
    MiTo-proposed labelled       Filter-2 matrix (= port on labelled cells), MiTo candidate
                                 thresholds opened, clonality filter on, tau .25, D2
    MiTo-proposed all cells      Filter-2 port on raw redeemV over all qualified cells (~4k),
                                 same downstream

Usage: python redeem_compare.py -> results/redeem_compare.csv
"""

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
from mito.ut.phylo_utils import get_clades

sys.path.insert(0, os.path.dirname(__file__))
import datasets as ds  # noqa: E402
import metrics as me  # noqa: E402
from mgatk_e2e import OPEN  # noqa: E402
from redeem_small import port_all_cells, score  # noqa: E402
from small_clones import small_clone_pass  # noqa: E402

RESULTS = os.path.join(os.path.dirname(__file__), 'results')


def tree_level(sets, truth, strata, conf=None, cutoff=0.0):
    pr = me.clade_pr(truth, sets, conf, cutoffs=(cutoff,), strata=strata).iloc[0]
    return {f'tree_{k}': pr[k] for k in pr.index if k.startswith(('recall', 'precision'))}


def mito_row(afm, truth, strata, name, filter_kwargs, d2):
    t0 = time.time()
    afm = afm.copy()
    mt.pp.filter_afm(afm, **filter_kwargs)
    tree = mt.tl.build_tree(afm, solver='UPMGA')
    mt.tl.annotate_clones(tree, afm)                       # tau .25, min_cells 5
    lab = afm.obs['MiTo_clone'].astype(str)
    n_new = 0
    if d2:
        lab, n_new = small_clone_pass(afm, tree, lab, max_cells=4, slack=0)
    sets = [set(v) for v in get_clades(tree, with_root=False).values()]
    row = score(lab, truth, strata, method=name, cells_input=afm.uns['mito']['filter_afm']['n_cells_in'],
                cells_kept=afm.shape[0], n_chars=afm.shape[1], n_small_clones_D2=n_new,
                secs=round(time.time() - t0))
    row.update(tree_level(sets, truth, strata))
    print({k: row[k] for k in ('method', 'yield_', 'n_labels', 'ARI', 'ARI_all', 'precision', 'recall')}, flush=True)
    return row


def main():
    truth = ds.pl1000_truth()
    strata = me.size_strata(truth)
    rows = []

    t = me.read_mitodrift_tree(os.path.join(ds.PL1000, 'tree_annotated.newick'))
    sets, conf = me.tree_clades(t)
    for c in (0.125, 0.25):
        lab = me.assign_clones_polytomy(me.trim_tree(t, c, collapse_trivial=False))
        lab = lab.where(~lab.str.startswith('_root_single_'), me.UNASSIGNED)
        row = score(lab, truth, strata, method=f'MitoDrift tau={c}', cells_input=1362, cells_kept=1362)
        row.update(tree_level(sets, truth, strata, conf, c))
        rows.append(row)

    afm, _ = ds.pl1000_redeemv(edge_trim=4)               # package default edge trim
    rows.append(mito_row(afm, truth, strata, 'MiTo-0.3.0 raw redeemV', {}, False))
    afm, _ = ds.pl1000_filter2()
    rows.append(mito_row(afm, truth, strata, 'MiTo-0.3.0 Filter-2 input', {}, False))
    rows.append(mito_row(afm, truth, strata, 'MiTo-proposed labelled cells', OPEN, True))
    rows.append(mito_row(port_all_cells(), truth, strata, 'MiTo-proposed all cells', OPEN, True))

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(RESULTS, 'redeem_compare.csv'), index=False)
    print('ALL DONE', flush=True)


if __name__ == '__main__':
    main()
