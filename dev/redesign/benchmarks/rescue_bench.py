"""
Recovering unassigned cells after the binomial pipeline:
  option 2  pooled lineage rescue (rescue.pooled_rescue)
  option 1  imputation into cells with no call (impute_dropouts min_support=0)
and both, vs binomial baseline, guarded pipeline and SHIPPED (cells, clones, shared-cell ARI).

Usage (from dev/redesign/results): PYTHONPATH=..:../benchmarks python ../benchmarks/rescue_bench.py <dataset>
"""
import sys, warnings, logging
sys.setrecursionlimit(100000); warnings.filterwarnings('ignore'); logging.disable(logging.INFO)
import numpy as np, pandas as pd, scanpy as sc, mito as mt, anndata as ad
from scipy.sparse import csr_matrix
from mito.pp.filters import annotate_vars, filter_baseline, filter_MiTo
from geno2 import em_genotype
from grid import FK, kernel, maxcompat, character_af
from impute import impute_dropouts
from joincount import carrier_nonrandomness
from carriers import binomial_carriers
from qc2_bench import graph_genotype
from cutter import evidence_cut
from compat import cell_membership
from rescue import pooled_rescue
pd.set_option('display.width', 250)
ds = sys.argv[1]
MFU = {'MDA_clones': 0.05, 'MDA_lung': 0.1, 'MDA_PT': 0.1}[ds]
ROOT = '/Users/cossa/Desktop/projects/MiTo/data_test/source_data/data/general/AFMs'
base = mt.pp.filter_cells(sc.read(f'{ROOT}/{ds}/afm_unfiltered.h5ad'), cell_filter='filter2'); N = base.shape[0]
gbc = base.obs['GBC'].astype(str); sizes = gbc.value_counts()
b = base.copy(); annotate_vars(b); b = filter_baseline(b); c = filter_MiTo(b, **FK)
X = c.X.toarray(); AD = c.layers['AD'].toarray(); COV = c.layers['site_coverage'].toarray().astype(np.int64); n, m = X.shape
names = np.array(c.var_names)
labels = {}


def full_series(lab_idx, lab_vals):
    s = pd.Series('unassigned', index=base.obs_names, dtype=object); s.loc[lab_idx] = lab_vals; return s


def tree_and_cut(cols, Bk):
    """Final characters, tree, evidence cut + tau. Returns labels over ALL c cells, final char indices (into cols)."""
    Bk = Bk > 0
    kv = np.flatnonzero((Bk.mean(0) <= 0.5) & (Bk.sum(0) >= 2))
    al = maxcompat(Bk[:, kv].astype(np.int8)); fc = kv[al]
    Bs = Bk[:, fc].astype(np.int8); Xk = X[:, cols]
    kc = Bs.sum(1) >= 1
    a = ad.AnnData(X=csr_matrix(character_af(Bs, Xk[:, fc])[kc]), obs=c.obs.iloc[np.flatnonzero(kc)][['GBC']].copy(), layers={'bin': csr_matrix(Bs[kc])})
    a.var_names = [f'{names[cols[j]]}#{i}' for i, j in enumerate(fc)]
    a.uns['genotyping'] = {'bin_method': 'x', 'binarization_kwargs': {}}
    for key in ('scLT_system', 'pp_method'): a.uns[key] = base.uns[key]
    mt.pp.compute_distances(a, precomputed=True, metric='weighted_jaccard', ncores=8, verbose=False)
    tree = mt.tl.build_tree(a, precomputed=True, solver='UPMGA')
    Bt = a.layers['bin'].toarray() > 0
    lab = evidence_cut(tree, Bt.astype(float), list(a.obs_names), min_in=0.25, ratio=3.0, self_min_in=0.85, ladder='descend')
    _, core = cell_membership(kernel(Xk, raw=True), Bs > 0, tau=0.0)
    lab = lab.mask(~(Bt & (core[kc] >= 0.25)).any(1), 'unassigned')
    out = np.full(n, 'unassigned', dtype=object); out[np.flatnonzero(kc)] = lab.values
    return out, fc, Bs


# ---- references
leg = mt.pp.filter_afm(base.copy(), filtering='MiTo', ncores=8)
tr = mt.tl.build_tree(leg, precomputed=True, solver='UPMGA')
m0 = mt.tl.MiToTreeAnnotator(tr); m0.clonal_inference(max_fraction_unassigned=MFU)
cm = tr.cell_meta
labels['SHIPPED'] = full_series(cm.index, np.where(cm['MiTo clone'].isna(), 'unassigned', cm['MiTo clone'].astype(str)))
g_flat, _ = em_genotype(AD, COV, np.full((n, n), 1/n), mode='bb')
keep_g, _, _ = carrier_nonrandomness(X, (g_flat > 0.7).astype(np.int8), cell_depth=COV.mean(1), alpha=0.05)
cols_g = np.flatnonzero(keep_g)
lab_g, _, _ = tree_and_cut(cols_g, graph_genotype(X, AD, COV, cols_g)[1])
labels['guarded (current)'] = full_series(c.obs_names, lab_g)

# ---- binomial baseline
Bq, p0 = binomial_carriers(AD, COV, alpha_cell=0.001, return_background=True)
keep, _, _ = carrier_nonrandomness(X, Bq, cell_depth=COV.mean(1), alpha=0.05)
cols = np.flatnonzero(keep)
Bb = Bq[:, cols]
lab_b, fc_b, Bs_b = tree_and_cut(cols, Bb)
labels['binomial'] = full_series(c.obs_names, lab_b)
info = {}
# option 2: pooled rescue on top of the binomial labels
for a_pool in (0.01, 0.001):
    ch = cols[fc_b]
    lab_r, det = pooled_rescue(lab_b, Bb[:, fc_b] > 0, AD[:, ch], COV[:, ch], p0[ch], alpha=a_pool)
    labels[f'binomial + pooled rescue a={a_pool}'] = full_series(c.obs_names, lab_r)
    info[f'binomial + pooled rescue a={a_pool}'] = f'rescued {int(det.assigned.sum()) if len(det) else 0} of {len(det)} unassigned cells'
# option 1: imputation into cells without calls
Bi1, add1 = impute_dropouts(X[:, cols], Bb, k=30, thr=0.6, min_support=0)
lab_i, fc_i, Bs_i = tree_and_cut(cols, Bi1)
labels['binomial + imputation min_support=0'] = full_series(c.obs_names, lab_i)
info['binomial + imputation min_support=0'] = f'imputed {int(add1.sum())} calls; cells gaining a first call {int(((Bb.sum(1) == 0) & (Bi1.sum(1) > 0)).sum())}'
# both: imputation min_support=0, then pooled rescue
ch = cols[fc_i]
lab_ir, det = pooled_rescue(lab_i, Bi1[:, fc_i] > 0, AD[:, ch], COV[:, ch], p0[ch], alpha=0.01)
labels['binomial + imputation min_support=0 + pooled rescue a=0.01'] = full_series(c.obs_names, lab_ir)
info['binomial + imputation min_support=0 + pooled rescue a=0.01'] = f'rescued {int(det.assigned.sum()) if len(det) else 0} of {len(det)}'
for k_, v_ in info.items():
    print(f'{k_}: {v_}', flush=True)

# ---- evaluation
rows = []
for name, lab in labels.items():
    ok = lab != 'unassigned'
    rec = 0; rec_mid = 0
    for cl, size in sizes.items():
        x = lab[gbc == cl]; x = x[x != 'unassigned']
        if len(x) and len(x)/size >= .5:
            top = x.value_counts()
            r_ = top.iloc[0]/len(x) >= .8 and (gbc[lab == top.index[0]] == cl).mean() >= .8
            rec += int(r_); rec_mid += int(r_ and 10 <= size < 100)
    row = dict(pipeline=name, cells=int(ok.sum()), cells_pct=round(100*ok.sum()/N, 1), labels=int(lab[ok].nunique()),
               ARI=round(mt.ut.custom_ARI(gbc[ok], lab[ok]), 3), NMI=round(mt.ut.normalized_mutual_info_score(gbc[ok], lab[ok]), 3),
               clones_recovered=rec, mid_clones_recovered=rec_mid)
    for ref in ('SHIPPED', 'guarded (current)'):
        if name == ref:
            continue
        common = lab.index[ok & (labels[ref] != 'unassigned')]
        row[f'shared_cells_vs_{ref.split()[0]}'] = len(common)
        row[f'ARI_shared_vs_{ref.split()[0]} (this / ref)'] = f'{mt.ut.custom_ARI(gbc[common], lab[common]):.3f} / {mt.ut.custom_ARI(gbc[common], labels[ref][common]):.3f}'
    rows.append(row)
R = pd.DataFrame(rows)
R.to_csv(f'rescue_{ds}.csv', index=False)
print(R.to_string(index=False))
print('ALL DONE')
