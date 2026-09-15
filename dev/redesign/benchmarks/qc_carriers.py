"""How do QC carrier definitions change the number of variants kept (GT vs non-GT)?"""
import sys, time, warnings, logging; warnings.filterwarnings('ignore'); logging.disable(logging.INFO)
import numpy as np, pandas as pd, scanpy as sc, mito as mt
from mito.pp.filters import annotate_vars, filter_baseline, filter_MiTo
from geno2 import em_genotype
from grid import FK
from joincount import carrier_nonrandomness
ds = sys.argv[1]
ROOT = '/Users/cossa/Desktop/projects/MiTo/data_test/source_data/data/general/AFMs'
base = mt.pp.filter_cells(sc.read(f'{ROOT}/{ds}/afm_unfiltered.h5ad'), cell_filter='filter2')
GT = set(pd.read_csv(f'real_{ds}_pvals.csv').query('GT')['var'])
leg = mt.pp.filter_afm(base.copy(), filtering='MiTo', ncores=8); SH = set(leg.var_names)
b = base.copy(); annotate_vars(b); b = filter_baseline(b); c = filter_MiTo(b, **FK)
X = c.X.toarray(); AD = c.layers['AD'].toarray(); COV = c.layers['site_coverage'].toarray().astype(np.int64); n, m = X.shape
isGT = np.array([v in GT for v in c.var_names]); isSH = np.array([v in SH for v in c.var_names])
flat = np.full((n, n), 1/n)
g_guard, _ = em_genotype(AD, COV, flat, mode='bb', guard=True)
g_old, _ = em_genotype(AD, COV, flat, mode='bb', guard=False)
defs = {'guard flat g>0.7 (current)': g_guard > 0.7, 'guard flat g>0.5': g_guard > 0.5, 'AD>=1': AD >= 1, 'AD>=2': AD >= 2,
        'old flat g>0.7 (unguarded)': g_old > 0.7}
rows = []
for name, B in defs.items():
    t = time.time()
    keep, pj, pe = carrier_nonrandomness(X, B.astype(np.int8), cell_depth=COV.mean(1), alpha=0.05)
    car = B.sum(0)
    rows.append(dict(dataset=ds, carriers_def=name, calls=int(B.sum()),
                     median_carriers_GT=float(np.median(car[isGT])), median_carriers_nonGT=float(np.median(car[~isGT])),
                     kept=int(keep.sum()), GT_kept=int(keep[isGT].sum()), GT_total=int(isGT.sum()),
                     nonGT_kept=int(keep[~isGT].sum()), shipped_vars_kept=int(keep[isSH].sum()), shipped_total=int(isSH.sum()),
                     secs=round(time.time()-t)))
    print(rows[-1], flush=True)
pd.DataFrame(rows).to_csv(f'qc_carriers_{ds}.csv', index=False)
print('ALL DONE')
