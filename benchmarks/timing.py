"""
End-to-end time of the package defaults on one MAESTER sample, from the maegatk tables
to annotated clones, step by step (fresh process; imports not counted).

Usage: python timing.py [MDA_PT]   -> results/timing_<sample>.csv
"""

import logging
import os
import resource
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


def main():
    name = sys.argv[1] if len(sys.argv) > 1 else 'MDA_PT'
    steps = []
    t = time.time()

    def lap(step, **info):
        nonlocal t
        now = time.time()
        steps.append(dict(step=step, secs=round(now - t, 2), **info))
        print(steps[-1], flush=True)
        t = now

    afm = mt.io.make_afm(os.path.join(ds.MDA, name), path_meta=os.path.join(ds.MDA, '..', 'cells_meta.csv'),
                         sample=name, scLT_system='MAESTER', pp_method='maegatk')
    lap('make_afm (maegatk tables)', cells=afm.shape[0], variants=afm.shape[1])
    mt.pp.filter_cells(afm, cell_filter='filter2')
    lap('filter_cells', cells=afm.shape[0], variants=afm.shape[1])
    mt.pp.filter_afm(afm)
    flow = afm.uns['mito']['filter_afm']['flow']
    lap('filter_afm', cells=afm.shape[0], variants=afm.shape[1])
    tree = mt.tl.build_tree(afm)
    lap('build_tree (UPGMA)')
    mt.tl.annotate_clones(tree, afm)
    sup_secs = afm.uns['mito']['clade_support']['seconds']
    lap('annotate_clones (incl. 100 support replicates)', clones=int(afm.obs['MiTo_clone'].nunique() - 1),
        support_secs=sup_secs)
    df = pd.DataFrame(steps)
    total = df['secs'].sum()
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024**2 if sys.platform == 'darwin' else 1024)
    print(f'TOTAL {total:.1f} s, peak memory {rss:.0f} MB', flush=True)
    print(flow.to_string(), flush=True)
    df.to_csv(os.path.join(os.path.dirname(__file__), 'results', f'timing_{name}.csv'), index=False)


if __name__ == '__main__':
    main()
