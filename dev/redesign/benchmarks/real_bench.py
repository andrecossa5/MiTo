"""
Real-data benchmark: SHIPPED vs DEV current vs QC2 (carrier non-randomness) on the MiTo AFMs.

Usage: python real_bench.py <dataset>      (MDA_clones | MDA_lung | MDA_PT)
Writes real_<dataset>.csv (one row per pipeline x one_sided), checkpointed after each pipeline.
"""
import sys, time, warnings, logging
sys.setrecursionlimit(100000)
warnings.filterwarnings('ignore'); logging.disable(logging.INFO)
import numpy as np, pandas as pd, scanpy as sc, mito as mt
from mito.pp.filters import annotate_vars, filter_baseline, filter_MiTo
from geno2 import em_genotype
from grid import FK
from joincount import carrier_nonrandomness
from qc2_bench import downstream
from refined_bench import cluster, as_variants

ROOT = '/Users/cossa/Desktop/projects/MiTo/data_test/source_data/data/general/AFMs'
ds = sys.argv[1]
out_csv = f'real_{ds}.csv'
rows = []


def log(msg):
    print(f'[{time.strftime("%H:%M:%S")}] {ds}: {msg}', flush=True)


def score(lab, gt, n_tot):
    ok = (lab != 'unassigned').values
    return dict(cells=int(ok.sum()), cells_pct=round(100*ok.sum()/n_tot, 1), labels=int(lab[ok].nunique()),
                ARI=round(mt.ut.custom_ARI(gt[ok], lab[ok]), 3),
                NMI=round(mt.ut.normalized_mutual_info_score(gt[ok], lab[ok]), 3))


def front_metrics(c, cols, B, gbc, GT):
    vars_, M = as_variants(c, cols, B)
    gb = gbc.loc[c.obs_names].values
    spec, clones = [], set()
    for j, v in enumerate(vars_):
        pos = M[:, j]
        if pos.sum() < 3:
            continue
        modal = pd.Series(gb[pos]).value_counts().index[0]
        spec.append(pos[gb == modal].sum()/pos.sum())
        if v in GT:
            clones.add(modal)
    return dict(n_vars=len(vars_), n_chars=int(B.shape[1]), GT_enriched_kept=len(set(vars_) & GT),
                GT_enriched_total=len(GT), call_clone_specificity=round(float(np.mean(spec)), 3) if spec else np.nan,
                GBC_clones_marked=len(clones))


t0 = time.time()
base = mt.pp.filter_cells(sc.read(f'{ROOT}/{ds}/afm_unfiltered.h5ad'), cell_filter='filter2')
N = base.shape[0]
gbc = base.obs['GBC'].astype(str)
n_gbc = gbc.nunique()
log(f'{N} cells after filter2, {n_gbc} GBC clones ({time.time()-t0:.0f}s)')
meta = dict(dataset=ds, n_cells=N, n_gbc=n_gbc)

t = time.time()
try:
    GT = set(mt.pp.filter_afm(base.copy(), filtering='GT_enriched', lineage_column='GBC', ncores=8).var_names)
except Exception as e:  # noqa: BLE001
    log(f'GT_enriched failed: {e}'); GT = set()
log(f'{len(GT)} GT-enriched variants ({time.time()-t:.0f}s)')

# ---- SHIPPED
t = time.time()
try:
    leg = mt.pp.filter_afm(base.copy(), filtering='MiTo', ncores=8)
    tr = mt.tl.build_tree(leg, precomputed=True, solver='UPMGA')
    m0 = mt.tl.MiToTreeAnnotator(tr); m0.clonal_inference()
    cm = tr.cell_meta
    lab = pd.Series(np.where(~cm['MiTo clone'].isna(), cm['MiTo clone'].astype(str), 'unassigned'), index=cm.index)
    gt = pd.Series(cm['GBC'].astype(str).values, index=cm.index)
    Bl = leg.layers['bin'].toarray() > 0
    vl = list(leg.var_names)
    rows.append(dict(**meta, pipeline='SHIPPED', one_sided=np.nan, n_vars=len(vl), GT_enriched_kept=len(set(vl) & GT),
                     GT_enriched_total=len(GT), secs=round(time.time()-t), **score(lab, gt, N)))
except Exception as e:  # noqa: BLE001
    log(f'SHIPPED failed: {type(e).__name__}: {e}')
    rows.append(dict(**meta, pipeline='SHIPPED', error=str(e)))
pd.DataFrame(rows).to_csv(out_csv, index=False)
log(f'SHIPPED done ({time.time()-t:.0f}s)')

# ---- shared DEV preprocessing
t = time.time()
b = base.copy(); annotate_vars(b); b = filter_baseline(b); c = filter_MiTo(b, **FK)
X = c.X.toarray(); AD = c.layers['AD'].toarray(); COV = c.layers['site_coverage'].toarray().astype(np.int64)
n, m = X.shape
log(f'filter_MiTo: {m} variants, {n} cells ({time.time()-t:.0f}s)')


def run(name, cols, split=True):
    t = time.time()
    try:
        cols_f, Bs, Xs, Xk = downstream(cols, X, AD, COV, split=split)
        fm = front_metrics(c, cols_f, Bs, gbc, GT)
        for r in cluster(c, cols_f, Bs, Xs, Xk, 'GBC', N):
            rows.append(dict(**meta, pipeline=name, **fm, secs=round(time.time()-t), **r))
    except Exception as e:  # noqa: BLE001
        log(f'{name} failed: {type(e).__name__}: {e}')
        rows.append(dict(**meta, pipeline=name, error=str(e)))
    pd.DataFrame(rows).to_csv(out_csv, index=False)
    log(f'{name} done ({time.time()-t:.0f}s)')


run('DEV current', np.arange(m), split=False)

t = time.time()
g_flat, _ = em_genotype(AD, COV, np.full((n, n), 1/n), mode='bb')
Bf = (g_flat > 0.7).astype(np.int8)
_, p_join, p_excl = carrier_nonrandomness(X, Bf, cell_depth=COV.mean(1), alpha=0.01)
minp = np.minimum(p_join, p_excl)
pd.DataFrame(dict(var=c.var_names, GT=[v in GT for v in c.var_names], carriers=Bf.sum(0),
                  p_join=p_join, p_excl=p_excl)).to_csv(f'real_{ds}_pvals.csv', index=False)
log(f'QC tests done ({time.time()-t:.0f}s): keep a.01={int((minp<=0.005).sum())}, a.05={int((minp<=0.025).sum())}, '
    f'excl-only a.05={int(((p_excl<=0.025)&(p_join>0.025)).sum())}')

run('QC2 a=0.01', np.flatnonzero(minp <= 0.005))
run('QC2 a=0.05', np.flatnonzero(minp <= 0.025))
log(f'ALL DONE ({time.time()-t0:.0f}s)')
