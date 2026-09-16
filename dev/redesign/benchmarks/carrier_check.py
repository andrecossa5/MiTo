"""
Carrier definitions for the QC, before any pipeline run:
  - MDA_PT: carriers and QC keep rate per oracle variant group (spread_nature_MDA_PT.csv groups)
  - every dataset: variants kept by the QC, GT-enriched vs not

Usage (from dev/redesign/results): PYTHONPATH=..:../benchmarks python ../benchmarks/carrier_check.py <dataset>
"""
import sys, warnings, logging, os
warnings.filterwarnings('ignore'); logging.disable(logging.INFO)
import numpy as np, pandas as pd, scanpy as sc, mito as mt
from mito.pp.filters import annotate_vars, filter_baseline, filter_MiTo
from geno2 import em_genotype
from grid import FK
from joincount import carrier_nonrandomness
from carriers import binomial_carriers
pd.set_option('display.width', 250)
ds = sys.argv[1]
ROOT = '/Users/cossa/Desktop/projects/MiTo/data_test/source_data/data/general/AFMs'
base = mt.pp.filter_cells(sc.read(f'{ROOT}/{ds}/afm_unfiltered.h5ad'), cell_filter='filter2')
GT = set(pd.read_csv(f'real_{ds}_pvals.csv').query('GT')['var'])
b = base.copy(); annotate_vars(b); b = filter_baseline(b); c = filter_MiTo(b, **FK)
X = c.X.toarray(); AD = c.layers['AD'].toarray(); COV = c.layers['site_coverage'].toarray().astype(np.int64); n, m = X.shape
names = np.array(c.var_names); isGT = np.isin(names, list(GT))
g_flat, _ = em_genotype(AD, COV, np.full((n, n), 1/n), mode='bb', guard=True)
DEFS = {'guard flat (current)': (g_flat > 0.7).astype(np.int8), 'AD>0': (AD >= 1).astype(np.int8),
        'binomial a=0.01': binomial_carriers(AD, COV, alpha_cell=0.01), 'binomial a=0.001': binomial_carriers(AD, COV, alpha_cell=0.001)}
groups = None
if os.path.exists(f'spread_nature_{ds}.csv'):
    groups = pd.read_csv(f'spread_nature_{ds}.csv').set_index('var')['group'].reindex(names).fillna('other').values
rows, grows = [], []
for name, B in DEFS.items():
    keep, pj, pe = carrier_nonrandomness(X, B, cell_depth=COV.mean(1), alpha=0.05)
    rows.append(dict(dataset=ds, carriers=name, calls=int(B.sum()), kept=int(keep.sum()), GT_kept=int(keep[isGT].sum()),
                     GT_total=int(isGT.sum()), nonGT_kept=int(keep[~isGT].sum()),
                     median_carriers_GT=float(np.median(B[:, isGT].sum(0))), median_carriers_nonGT=float(np.median(B[:, ~isGT].sum(0)))))
    if groups is not None:
        for gname in pd.unique(groups):
            if gname == 'other':
                continue
            sel = groups == gname
            grows.append(dict(group=gname, n=int(sel.sum()), carriers=name, median_carriers=float(np.median(B[:, sel].sum(0))),
                              kept=round(float(keep[sel].mean()), 2)))
    print(rows[-1], flush=True)
pd.DataFrame(rows).to_csv(f'carrier_check_{ds}.csv', index=False)
print(pd.DataFrame(rows).to_string(index=False))
if grows:
    G = pd.DataFrame(grows)
    G.to_csv(f'carrier_check_groups_{ds}.csv', index=False)
    print('\n-- per oracle group: median carriers | share kept by QC')
    print(G.pivot_table(index=['group', 'n'], columns='carriers', values=['median_carriers', 'kept'], aggfunc='first').round(2).to_string())
print('ALL DONE')
