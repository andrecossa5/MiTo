"""
Clonality filter: the 0.5.0 implementation (full n x n leave-one-out distances per
variant, benchmarks/legacy/clonality_050.py) against the incremental one (only the
held-out variant's carriers change, one bank of stratified permutations).

    real      the AFMs entering the clonality filter on MDA_PT, pL1000 ReDeeM and
              pL1000 mtscATAC (saved by filter_afm; pass their paths): time, peak
              memory, and agreement of the retained variants and p-values
    scaling   simulated MAESTER AFMs of 2,500 / 5,000 / 10,000 / 20,000 cells, after
              filter_afm's stages up to the clonality filter

Peak memory is measured in a fresh process per run (ru_maxrss).

Usage: python clonality_speed.py real <afm.h5ad>... | scaling   -> results/clonality_speed_*.csv
"""

import json
import os
import subprocess
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, 'results')
PY = sys.executable

CHILD = r'''
import json, resource, sys, time, warnings, logging
warnings.filterwarnings('ignore'); logging.disable(logging.WARNING)
sys.path.insert(0, %(here)r)
import numpy as np, scanpy as sc
impl, path, out = sys.argv[1], sys.argv[2], sys.argv[3]
afm = sc.read(path)
if impl == 'old':
    from legacy.clonality_050 import filter_non_clonal_variants as f
else:
    from mito.pp.clonality import filter_non_clonal_variants as f
t = time.time()
f(afm)
secs = time.time() - t
rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024**2 if sys.platform == 'darwin' else 1024)
json.dump(dict(secs=secs, peak_mb=rss, kept=list(afm.var_names),
               p=afm.var['p_clonal'].to_dict()), open(out, 'w'))
'''


def run(impl, path):
    out = path + f'.{impl}.json'
    subprocess.run([PY, '-c', CHILD % dict(here=HERE), impl, path, out], check=True)
    return json.load(open(out))


def compare(name, path, n, m, impls=('old', 'new')):
    r = {i: run(i, path) for i in impls}
    row = dict(dataset=name, n_cells=n, n_vars=m)
    for i in impls:
        row[f'{i}_secs'] = round(r[i]['secs'], 1)
        row[f'{i}_peak_mb'] = round(r[i]['peak_mb'])
        row[f'{i}_kept'] = len(r[i]['kept'])
    if len(impls) == 2:
        a, b = set(r['old']['kept']), set(r['new']['kept'])
        row['kept_jaccard'] = round(len(a & b) / max(len(a | b), 1), 3)
        pa = pd.Series(r['old']['p']); pb = pd.Series(r['new']['p'])
        row['p_spearman'] = round(pa.corr(pb.reindex(pa.index), method='spearman'), 3)
        row['speedup'] = round(row['old_secs'] / max(row['new_secs'], 1e-3), 1)
    print(row, flush=True)
    return row


def main():
    import scanpy as sc
    target = sys.argv[1]
    rows = []
    if target == 'real':
        for path in sys.argv[2:]:
            a = sc.read(path)
            rows.append(compare(os.path.basename(path).split('_')[-1].replace('.h5ad', ''), path, *a.shape))
    else:
        import logging
        import warnings
        warnings.filterwarnings('ignore')
        logging.disable(logging.WARNING)
        import mito as mt
        import mito.pp.preprocessing as pre
        tmp = os.path.join(HERE, 'logs')
        for n in (2500, 5000, 10000, 20000):
            afm = mt.ut.simulate_afm(n_cells=n, n_clones=50, n_root_clones=50, max_depth=1,
                                     frac_double_variants=.3, frac_noisy_variants=.2, random_seed=0)
            saved = {}

            def stop(a, **kw):
                saved['afm'] = a.copy()
                raise StopIteration

            pre.filter_non_clonal_variants = stop
            try:
                mt.pp.filter_afm(afm)
            except StopIteration:
                pass
            path = os.path.join(tmp, f'clonality_input_sim{n}.h5ad')
            saved['afm'].uns.pop('simulation', None)
            saved['afm'].write(path)
            impls = ('old', 'new') if n <= 10000 else ('new',)
            rows.append(compare(f'sim_{n}', path, *saved['afm'].shape, impls=impls))
    pd.DataFrame(rows).to_csv(os.path.join(RESULTS, f'clonality_speed_{target}.csv'), index=False)
    print('ALL DONE', flush=True)


if __name__ == '__main__':
    main()
