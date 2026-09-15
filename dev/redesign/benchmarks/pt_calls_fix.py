import sys, warnings, logging; warnings.filterwarnings('ignore'); logging.disable(logging.INFO)
import numpy as np, pandas as pd, scanpy as sc, mito as mt
from mito.pp.filters import annotate_vars, filter_baseline, filter_MiTo
from geno2 import em_genotype
from grid import FK, kernel
from impute import impute_dropouts
ds = sys.argv[1]
ROOT = '/Users/cossa/Desktop/projects/MiTo/data_test/source_data/data/general/AFMs'
base = mt.pp.filter_cells(sc.read(f'{ROOT}/{ds}/afm_unfiltered.h5ad'), cell_filter='filter2')
leg = mt.pp.filter_afm(base.copy(), filtering='MiTo', ncores=8)
Bm = leg.layers['bin'].toarray() > 0
b = base.copy(); annotate_vars(b); b = filter_baseline(b); c = filter_MiTo(b, **FK)
c = c[leg.obs_names, leg.var_names].copy()
X = c.X.toarray(); AD = c.layers['AD'].toarray(); COV = c.layers['site_coverage'].toarray().astype(np.int64); n = X.shape[0]
gbc = leg.obs['GBC'].astype(str).values
rows = []
def row(name, B):
    spec = []
    for j in range(B.shape[1]):
        pos = B[:, j]
        if pos.sum() >= 3:
            spec.append(pd.Series(gbc[pos]).value_counts().iloc[0]/pos.sum())
    rows.append(dict(ds=ds, calls=name, n_calls=int(B.sum()), calls_per_cell=round(B.sum(1).mean(), 2),
                     AD0_calls=int((B & (AD == 0)).sum()), shared_with_MiTo=int((B & Bm).sum()),
                     only_ours=int((B & ~Bm).sum()), only_MiTo=int((Bm & ~B).sum()),
                     median_call_specificity=round(float(np.median(spec)), 3), vars_prev_gt_0_5=int((B.mean(0) > 0.5).sum())))
row('MiTo (shipped)', Bm)
for guard in [False, True]:
    tag = 'guard' if guard else 'old'
    gf, _ = em_genotype(AD, COV, np.full((n, n), 1/n), mode='bb', guard=guard); row(f'bb flat ({tag})', gf > 0.7)
    gg, _ = em_genotype(AD, COV, kernel(X), mode='bb', guard=guard); row(f'bb graph ({tag})', gg > 0.7)
    Bi = impute_dropouts(X, (gg > 0.7).astype(np.int8), k=30, thr=0.6, min_support=1)[0] > 0
    row(f'bb graph + imputation ({tag})', Bi)
print(pd.DataFrame(rows).to_string(index=False), flush=True)
