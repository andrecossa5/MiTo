"""SNR definitions on QC-kept variants (QC 1e-4): median AF of CALLED cells vs of cells with >=1 read, over background."""
import sys, warnings, logging; warnings.filterwarnings('ignore'); logging.disable(logging.INFO)
import numpy as np, pandas as pd, scanpy as sc, mito as mt
from mito.pp.filters import annotate_vars, filter_baseline, filter_MiTo
from grid import FK
from joincount import carrier_nonrandomness
from carriers import binomial_carriers
pd.set_option('display.width', 250); pd.set_option('display.max_rows', 200)
ds = sys.argv[1]
ROOT = '/Users/cossa/Desktop/projects/MiTo/data_test/source_data/data/general/AFMs'
base = mt.pp.filter_cells(sc.read(f'{ROOT}/{ds}/afm_unfiltered.h5ad'), cell_filter='filter2')
GT = set(pd.read_csv(f'real_{ds}_pvals.csv').query('GT')['var'])
b = base.copy(); annotate_vars(b); b = filter_baseline(b); c = filter_MiTo(b, **FK)
X = c.X.toarray(); AD = c.layers['AD'].toarray(); COV = c.layers['site_coverage'].toarray().astype(np.int64)
gbc = c.obs['GBC'].astype(str).values; sizes = pd.Series(gbc).value_counts()
keep, _, _ = carrier_nonrandomness(X, binomial_carriers(AD, COV, alpha_cell=1e-4), cell_depth=COV.mean(1), alpha=0.05)
cols = np.flatnonzero(keep)
B, p0 = binomial_carriers(AD[:, cols], COV[:, cols], alpha_cell=1e-3, return_background=True)
rows = []
for k, j in enumerate(cols):
    calls = B[:, k] > 0; readers = AD[:, j] >= 1
    vc = pd.Series(gbc[calls]).value_counts() if calls.any() else pd.Series(dtype=int)
    share = vc.iloc[0]/calls.sum() if calls.any() else np.nan
    rows.append(dict(var=c.var_names[j], GT=c.var_names[j] in GT, calls=int(calls.sum()), readers=int(readers.sum()),
                     background=p0[k], medAF_calls=np.median(X[calls, j]) if calls.any() else np.nan,
                     medAF_readers=np.median(X[readers, j]) if readers.any() else np.nan, calls_top_clone_share=round(share, 2)))
D = pd.DataFrame(rows)
D['ratio_calls'] = (D.medAF_calls/D.background).round(1); D['ratio_readers'] = (D.medAF_readers/D.background).round(1)
D['kind'] = np.where(D.GT, 'GT-enriched', np.where(D.calls_top_clone_share < 0.5, 'non-GT, calls spread (<0.5 in one clone)', 'non-GT, calls concentrated'))
print(f'== {ds}: {len(D)} QC-kept variants')
print(D.groupby('kind')[['ratio_calls', 'ratio_readers']].describe(percentiles=[.1, .5]).round(1).to_string())
for thr in (5, 10, 20):
    print(f'ratio_readers < {thr}: dropped', D[D.ratio_readers < thr].groupby('kind').size().to_dict())
print(D.sort_values('ratio_readers').head(15)[['var', 'kind', 'calls', 'readers', 'background', 'medAF_readers', 'medAF_calls', 'ratio_readers', 'ratio_calls']].to_string(index=False))
D.to_csv(f'snr_check_{ds}.csv', index=False)
