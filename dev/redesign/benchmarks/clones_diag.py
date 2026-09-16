"""
Why does the binomial pipeline underperform on MDA_clones? Guarded vs binomial (QC 1e-4, genotypes 1e-3, tau).

Usage (from dev/redesign/results): PYTHONPATH=..:../benchmarks python ../benchmarks/clones_diag.py MDA_clones
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
from cutter import evidence_cut
from compat import cell_membership
pd.set_option('display.width', 260); pd.set_option('display.max_rows', 200); pd.set_option('display.max_columns', 40)
ds = sys.argv[1] if len(sys.argv) > 1 else 'MDA_clones'
ROOT = '/Users/cossa/Desktop/projects/MiTo/data_test/source_data/data/general/AFMs'
base = mt.pp.filter_cells(sc.read(f'{ROOT}/{ds}/afm_unfiltered.h5ad'), cell_filter='filter2')
N = base.shape[0]
b = base.copy(); annotate_vars(b); b = filter_baseline(b); c = filter_MiTo(b, **FK)
X = c.X.toarray(); AD = c.layers['AD'].toarray(); COV = c.layers['site_coverage'].toarray().astype(np.int64); n, m = X.shape
names = np.array(c.var_names); gbc = c.obs['GBC'].astype(str).values; sizes = pd.Series(gbc).value_counts()
print(f'{ds}: {N} cells; clone sizes {sizes.to_dict()}')
print(f'mean site coverage {COV.mean():.0f}; median per-cell mean coverage {np.median(COV.mean(1)):.0f}')


def tree_cut(cols, Bk):
    Bk = Bk > 0
    prev = Bk.mean(0); ncall = Bk.sum(0)
    kv = np.flatnonzero((prev <= 0.5) & (ncall >= 2))
    al = maxcompat(Bk[:, kv].astype(np.int8)); fc = kv[al]
    fate = pd.Series('final', index=names[cols])
    fate[names[cols][prev > 0.5]] = 'lost: prevalence>0.5'
    fate[names[cols][(ncall < 2)]] = 'lost: <2 calls'
    fate[names[cols[kv[~al]]]] = 'lost: four-gamete'
    Bs = Bk[:, fc].astype(np.int8); Xk = X[:, cols]; kc = Bs.sum(1) >= 1
    a = ad.AnnData(X=csr_matrix(character_af(Bs, Xk[:, fc])[kc]), obs=pd.DataFrame(index=c.obs_names[kc]), layers={'bin': csr_matrix(Bs[kc])})
    a.var_names = [f'v{i}' for i in range(fc.size)]
    a.uns['genotyping'] = {'bin_method': 'x', 'binarization_kwargs': {}}
    for key in ('scLT_system', 'pp_method'): a.uns[key] = base.uns[key]
    mt.pp.compute_distances(a, precomputed=True, metric='weighted_jaccard', ncores=8, verbose=False)
    tree = mt.tl.build_tree(a, precomputed=True, solver='UPMGA')
    Bt = a.layers['bin'].toarray() > 0
    lab = evidence_cut(tree, Bt.astype(float), list(a.obs_names), min_in=0.25, ratio=3.0, self_min_in=0.85, ladder='descend')
    _, core = cell_membership(kernel(Xk, raw=True), Bs > 0, tau=0.0)
    lab = lab.mask(~(Bt & (core[kc] >= 0.25)).any(1), 'unassigned')
    out = np.full(n, 'unassigned', dtype=object); out[np.flatnonzero(kc)] = lab.values
    return out, Bk, fate


P = {}
g_flat, _ = em_genotype(AD, COV, np.full((n, n), 1/n), mode='bb')
keep, _, _ = carrier_nonrandomness(X, (g_flat > 0.7).astype(np.int8), cell_depth=COV.mean(1), alpha=0.05)
cols = np.flatnonzero(keep)
g, _ = em_genotype(AD[:, cols], COV[:, cols], kernel(X[:, cols]), mode='bb')
Bg = impute_dropouts(X[:, cols], (g > 0.7).astype(np.int8), k=30, thr=0.6, min_support=1)[0]
P['guarded'] = (cols,) + tree_cut(cols, Bg)
Bq = binomial_carriers(AD, COV, alpha_cell=1e-4)
keep, _, _ = carrier_nonrandomness(X, Bq, cell_depth=COV.mean(1), alpha=0.05)
cols = np.flatnonzero(keep)
Bgen, p0 = binomial_carriers(AD[:, cols], COV[:, cols], alpha_cell=1e-3, return_background=True)
P['binomial'] = (cols,) + tree_cut(cols, Bgen)
bg_est = pd.Series(p0, index=names[cols])

# ---- 1. clones: fraction assigned and label composition
print('\n==== 1. per GBC clone: fraction assigned | labels holding its cells (top 3, share of the clone)')
rows = []
for cl, size in sizes.items():
    r = dict(clone=cl, size=size)
    for pipe, (_, lab, _, _) in P.items():
        x = pd.Series(lab[gbc == cl]); asg = x[x != 'unassigned']
        r[f'{pipe}_assigned'] = round(len(asg)/size, 2)
        vc = asg.value_counts()
        r[f'{pipe}_labels'] = ' | '.join(f'{L[-6:]}:{v/size:.2f}(pur {np.mean(gbc[lab == L] == cl):.2f})' for L, v in vc.head(3).items())
    rows.append(r)
print(pd.DataFrame(rows).to_string(index=False))

# ---- 2. variants
print('\n==== 2. variants kept by either QC: marked clone, in-clone recall, outside calls, fate, background')
vrows = []
allv = sorted(set(names[P['guarded'][0]]) | set(names[P['binomial'][0]]))
for v in allv:
    j = list(names).index(v)
    reads = AD[:, j] >= 1
    vc = pd.Series(gbc[reads]).value_counts()
    cl = vc.index[0] if len(vc) else None
    in_cl = gbc == cl
    true_bg = AD[~in_cl, j].sum()/max(COV[~in_cl, j].sum(), 1)
    r = dict(var=v, clone=cl, clone_size=int(sizes.get(cl, 0)), frac_clone_with_read=round(reads[in_cl].mean(), 2),
             median_AF_in_clone_readers=round(float(np.median(X[in_cl & reads, j])) if (in_cl & reads).any() else np.nan, 3),
             true_bg_outside_clone=f'{true_bg:.1e}')
    for pipe, (cols_p, lab, Bk, fate) in P.items():
        if v in set(names[cols_p]):
            k = list(names[cols_p]).index(v)
            calls = Bk[:, k]
            r[f'{pipe}_in_clone_recall'] = round(calls[in_cl].mean(), 2)
            r[f'{pipe}_calls_outside'] = int(calls[~in_cl].sum())
            r[f'{pipe}_fate'] = fate[v]
        else:
            r[f'{pipe}_in_clone_recall'] = np.nan; r[f'{pipe}_calls_outside'] = np.nan; r[f'{pipe}_fate'] = 'not kept by QC'
    r['binomial_bg_estimate'] = f'{bg_est[v]:.1e}' if v in bg_est else ''
    vrows.append(r)
V = pd.DataFrame(vrows).sort_values(['clone_size', 'clone'], ascending=False)
print(V.to_string(index=False))
V.to_csv(f'clones_diag_variants_{ds}.csv', index=False)

# ---- 3. summary per clone of best marker recall under each pipeline (final variants only)
print('\n==== 3. per clone: best in-clone recall among FINAL variants')
for pipe in P:
    fin = V[V[f'{pipe}_fate'] == 'final']
    print(pipe, fin.groupby('clone')[f'{pipe}_in_clone_recall'].max().round(2).to_dict())
print('ALL DONE')
