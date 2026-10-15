"""
maegatk / mgatk reader: the 0.5.0 implementation (benchmarks/legacy/format_afm_050.py)
against the typed, sparse one. The AFMs must be identical; time and peak memory are
measured in a fresh process per run.

    real   MDA_PT (MAESTER, maegatk) and pL1000 (mtscATAC, mgatk, molecules)

Usage: python reader_check.py   -> results/reader_check.csv
"""

import os
import subprocess
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import datasets as ds  # noqa: E402

CHILD = r'''
import resource, sys, time, warnings, logging
warnings.filterwarnings('ignore'); logging.disable(logging.WARNING)
sys.path.insert(0, %(here)r)
impl, out, path, meta, sample, system, method = sys.argv[1:8]
meta = None if meta == '-' else meta
sample = None if sample == '-' else sample
if impl == 'old':
    from legacy.format_afm_050 import make_afm
else:
    from mito.io import make_afm
t = time.time()
afm = make_afm(path, path_meta=meta, sample=sample, scLT_system=system, pp_method=method)
secs = time.time() - t
rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024**2 if sys.platform == 'darwin' else 1024)
afm.uns['timing'] = {'secs': secs, 'peak_mb': rss}
afm.write(out)
'''


def run(impl, out, path, meta, sample, system, method):
    subprocess.run([sys.executable, '-c', CHILD % dict(here=HERE), impl, out, path,
                    meta or '-', sample or '-', system, method], check=True)
    import scanpy as sc
    return sc.read(out)


def same(a, b):
    """Every slot of two AFMs, as a dict of booleans."""
    def eq_sparse(x, y):
        return x.shape == y.shape and (x != y).nnz == 0
    q = {}
    q['obs_names'] = list(a.obs_names) == list(b.obs_names)
    q['obs'] = q['obs_names'] and list(a.obs.columns) == list(b.obs.columns) and all(
        np.allclose(a.obs[c].values.astype(float), b.obs[c].values.astype(float), equal_nan=True)
        if pd.api.types.is_numeric_dtype(a.obs[c]) else (a.obs[c].astype(str).values == b.obs[c].astype(str).values).all()
        for c in a.obs.columns)
    q['var'] = list(a.var_names) == list(b.var_names) and all(
        (a.var[c].values == b.var[c].values).all() if c != 'quality'
        else np.allclose(a.var[c].values, b.var[c].values, equal_nan=True) for c in a.var.columns) \
        and list(a.var.columns) == list(b.var.columns)
    q['X'] = eq_sparse(a.X, b.X)
    q['AD'] = eq_sparse(a.layers['AD'], b.layers['AD'])
    q['DP'] = np.array_equal(np.asarray(a.layers['DP']), np.asarray(b.layers['DP'])) and \
        np.asarray(a.layers['DP']).dtype == np.asarray(b.layers['DP']).dtype
    ua = {k: v for k, v in a.uns.items() if k != 'timing'}
    ub = {k: v for k, v in b.uns.items() if k != 'timing'}
    q['uns'] = str(ua) == str(ub)
    return q


def main():
    tmp = os.path.join(HERE, 'logs')
    cases = [
        ('MDA_PT', os.path.join(ds.MDA, 'MDA_PT'), os.path.join(ds.MDA, '..', 'cells_meta.csv'), 'MDA_PT',
         'MAESTER', 'maegatk'),
        ('pL1000_mgatk', ds.PL1000, None, None, 'mtscATAC', 'mgatk'),
    ]
    rows = []
    for name, path, meta, sample, system, method in cases:
        r = {}
        for impl in ('old', 'new'):
            import time
            time.sleep(30)                 # let the previous process's memory be reclaimed
            r[impl] = run(impl, os.path.join(tmp, f'reader_{name}_{impl}.h5ad'), path, meta, sample, system, method)
        q = same(r['old'], r['new'])
        row = dict(dataset=name, shape=str(r['new'].shape),
                   old_secs=round(r['old'].uns['timing']['secs'], 1), new_secs=round(r['new'].uns['timing']['secs'], 1),
                   old_peak_mb=round(r['old'].uns['timing']['peak_mb']), new_peak_mb=round(r['new'].uns['timing']['peak_mb']),
                   identical=all(q.values()), **q)
        print(row, flush=True)
        rows.append(row)
    pd.DataFrame(rows).to_csv(os.path.join(HERE, 'results', 'reader_check.csv'), index=False)
    print('ALL DONE', flush=True)


if __name__ == '__main__':
    main()
