"""Does carrier concentration flag the spread variants of the MDA_PT flow table (oracle categories)?"""
import sys, warnings, logging; warnings.filterwarnings('ignore'); logging.disable(logging.INFO)
import numpy as np, pandas as pd, scanpy as sc, mito as mt
from mito.pp.filters import annotate_vars, filter_baseline, filter_MiTo
from grid import FK
from joincount import carrier_nonrandomness
pd.set_option('display.width', 250)
ds = sys.argv[1] if len(sys.argv) > 1 else 'MDA_PT'
ROOT = '/Users/cossa/Desktop/projects/MiTo/data_test/source_data/data/general/AFMs'
base = mt.pp.filter_cells(sc.read(f'{ROOT}/{ds}/afm_unfiltered.h5ad'), cell_filter='filter2')
F = pd.read_csv(f'flow_{ds}.csv').set_index('var')
b = base.copy(); annotate_vars(b); b = filter_baseline(b); c = filter_MiTo(b, **FK)
X = c.X.toarray(); AD = c.layers['AD'].toarray(); COV = c.layers['site_coverage'].toarray().astype(np.int64)
keep, pj, pe, conc = carrier_nonrandomness(X, (AD >= 1).astype(np.int8), cell_depth=COV.mean(1), alpha=0.05, return_concentration=True)
_, _, _, _ = None, None, None, None
D = F.copy()
D['p_join'] = pj; D['p_excl'] = pe; D['concentration'] = np.round(conc, 2); D['keep_new'] = keep
D['kept_before'] = (pj <= 0.025) | (pe <= 0.025)
D['by'] = np.select([pe <= 0.025, pj <= 0.025], ['exclusivity', 'join count'], 'rejected')
k = D[D.kept_before]
print(f'{ds}: kept before {int(D.kept_before.sum())} -> with concentration {int(D.keep_new.sum())}')
print('\n-- variants kept before, by oracle category: kept before / kept now / median concentration')
print(k.groupby('marks').agg(n=('keep_new', 'size'), kept_now=('keep_new', 'sum'), median_conc=('concentration', 'median'),
                             GT=('GT', 'sum')).to_string())
print('\n-- same, for AD>0-only final variants (the ablation groups)')
a = k[k.flow == 'AD>0 only']
print(a.groupby('marks').agg(n=('keep_new', 'size'), kept_now=('keep_new', 'sum'), median_conc=('concentration', 'median')).to_string())
print('\n-- dropped by concentration')
print(k[~k.keep_new][['GT', 'shipped', 'carriers_AD0', 'modal_clone_size', 'modal_share', 'n_clones_3plus', 'flow', 'marks', 'by', 'concentration']].sort_values('marks').to_string())
print('\n-- concentration of spread variants still kept')
print(k[k.keep_new & (k.marks == 'spread over clones (share<0.5)')][['GT', 'carriers_AD0', 'modal_share', 'n_clones_3plus', 'flow', 'by', 'concentration']].to_string())
D.to_csv(f'concentration_{ds}.csv')
