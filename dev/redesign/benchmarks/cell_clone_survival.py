"""
Cell and clone survival: binomial genotypes (with/without imputation) vs guarded pipeline vs shipped.

Cells: after filter2 -> with >=1 call on final characters -> assigned by the cut -> after abstention.
Clones (GBC): represented (>=1 / >=5 assigned cells), recovered (>=50% of cells assigned, >=80% in one
label, label >=80% pure), by clone size.

Usage (from dev/redesign/results): PYTHONPATH=..:../benchmarks python ../benchmarks/cell_clone_survival.py <dataset>
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
pd.set_option('display.width', 250)
ds = sys.argv[1]
MFU = {'MDA_clones': 0.05, 'MDA_lung': 0.1, 'MDA_PT': 0.1}[ds]
ROOT = '/Users/cossa/Desktop/projects/MiTo/data_test/source_data/data/general/AFMs'
base = mt.pp.filter_cells(sc.read(f'{ROOT}/{ds}/afm_unfiltered.h5ad'), cell_filter='filter2'); N = base.shape[0]
gbc = base.obs['GBC'].astype(str); sizes = gbc.value_counts()
b = base.copy(); annotate_vars(b); b = filter_baseline(b); c = filter_MiTo(b, **FK)
X = c.X.toarray(); AD = c.layers['AD'].toarray(); COV = c.layers['site_coverage'].toarray().astype(np.int64); n, m = X.shape
names = np.array(c.var_names)
labels, funnel = {}, []


def run_dev(name, cols, Bk):
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
    lab_t = lab.mask(~(Bt & (core[kc] >= 0.25)).any(1), 'unassigned')
    full = pd.Series('unassigned', index=base.obs_names, dtype=object); full.loc[lab_t.index] = lab_t.values
    labels[name] = full
    funnel.append(dict(pipeline=name, cells_filter2=N, cells_with_call_on_final_chars=int(kc.sum()),
                       assigned_by_cut=int((lab != 'unassigned').sum()), after_abstention=int((full != 'unassigned').sum()),
                       final_chars=int(fc.size), calls=int(Bs.sum())))


# shipped
leg = mt.pp.filter_afm(base.copy(), filtering='MiTo', ncores=8)
tr = mt.tl.build_tree(leg, precomputed=True, solver='UPMGA')
m0 = mt.tl.MiToTreeAnnotator(tr); m0.clonal_inference(max_fraction_unassigned=MFU)
cm = tr.cell_meta
full = pd.Series('unassigned', index=base.obs_names, dtype=object)
full.loc[cm.index] = np.where(cm['MiTo clone'].isna(), 'unassigned', cm['MiTo clone'].astype(str))
labels['SHIPPED'] = full
funnel.append(dict(pipeline='SHIPPED', cells_filter2=N, cells_with_call_on_final_chars=int(leg.shape[0]),
                   assigned_by_cut=int((full != 'unassigned').sum()), after_abstention=int((full != 'unassigned').sum()),
                   final_chars=int(leg.shape[1]), calls=int(leg.layers['bin'].sum())))
# guarded pipeline
g_flat, _ = em_genotype(AD, COV, np.full((n, n), 1/n), mode='bb')
keep, _, _ = carrier_nonrandomness(X, (g_flat > 0.7).astype(np.int8), cell_depth=COV.mean(1), alpha=0.05)
cols = np.flatnonzero(keep)
run_dev('guarded (current)', cols, graph_genotype(X, AD, COV, cols)[1])
# binomial
Bq = binomial_carriers(AD, COV, alpha_cell=0.001)
keep, _, _ = carrier_nonrandomness(X, Bq, cell_depth=COV.mean(1), alpha=0.05)
cols = np.flatnonzero(keep)
Bb = Bq[:, cols]
run_dev('binomial 0.001', cols, Bb)
Bi, added = impute_dropouts(X[:, cols], Bb, k=30, thr=0.6, min_support=1)
run_dev('binomial 0.001 + imputation', cols, Bi)
cells_without_any_call = int((Bb.sum(1) == 0).sum())
print(f'binomial: cells with no call on ANY QC-kept variant: {cells_without_any_call} of {N}; imputed calls added: {int(added.sum())}; '
      f'cells gaining their first call through imputation: {int(((Bb.sum(1) == 0) & (Bi.sum(1) > 0)).sum())}', flush=True)

print('\n==== cell funnel'); print(pd.DataFrame(funnel).to_string(index=False))
bins = [0, 10, 20, 50, 100, 10**6]; blab = ['<10', '10-19', '20-49', '50-99', '>=100']
rows = []
for name, lab in labels.items():
    for cl, size in sizes.items():
        x = lab[gbc == cl]; x = x[x != 'unassigned']
        rec = False
        if len(x) and len(x)/size >= .5:
            top = x.value_counts(); rec = bool(top.iloc[0]/len(x) >= .8 and (gbc[lab == top.index[0]] == cl).mean() >= .8)
        rows.append(dict(pipeline=name, clone=cl, size=size, size_bin=blab[np.searchsorted(bins, size, side='right') - 1],
                         assigned=len(x), frac_assigned=len(x)/size, recovered=rec))
R = pd.DataFrame(rows)
R.to_csv(f'survival_{ds}.csv', index=False)
print('\n==== barcode clones by size: n | represented (>=5 assigned cells) | recovered | mean share of cells assigned')
t = R.groupby(['size_bin', 'pipeline']).agg(n=('clone', 'size'), represented=('assigned', lambda a: int((a >= 5).sum())),
                                            recovered=('recovered', 'sum'), frac_assigned=('frac_assigned', 'mean')).round(2)
print(t.unstack('pipeline').to_string())
print('\n==== totals')
print(R.groupby('pipeline').agg(clones_with_any_cell=('assigned', lambda a: int((a >= 1).sum())), clones_5plus=('assigned', lambda a: int((a >= 5).sum())),
                                recovered=('recovered', 'sum')).to_string())
# pairwise shared-cell ARI
print('\n==== ARI on cells assigned by both (row pipeline / column pipeline)')
names_p = list(labels)
M = pd.DataFrame(index=names_p, columns=names_p, dtype=object)
for p1 in names_p:
    for p2 in names_p:
        common = labels[p1].index[(labels[p1] != 'unassigned') & (labels[p2] != 'unassigned')]
        M.loc[p1, p2] = f'{mt.ut.custom_ARI(gbc[common], labels[p1][common]):.3f} (n={len(common)})'
print(M.to_string())
print('ALL DONE')
