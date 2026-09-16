"""
What is thrown out on one AFM (variants, clones, cells), and what could still be recovered.

GBC barcodes are used ONLY as a diagnostic reference.
Clone markers are defined from significant per-cell reads (binomial_carriers alpha 0.001) over ALL
variants passing filter_baseline:
    marker       >= 30% of one clone's cells carry it, >= 60% of its carriers are in that clone
    weak marker  10-30% of the clone, same specificity
Pipelines: guarded (current default) and binomial; SHIPPED as reference.

Usage (from dev/redesign/results): PYTHONPATH=..:../benchmarks python ../benchmarks/pt_loss.py <dataset>
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
pd.set_option('display.width', 260); pd.set_option('display.max_rows', 300); pd.set_option('display.max_columns', 40)
ds = sys.argv[1] if len(sys.argv) > 1 else 'MDA_PT'
MFU = {'MDA_clones': 0.05, 'MDA_lung': 0.1, 'MDA_PT': 0.1}[ds]
MIN_CLONE, MARKER_PREV, WEAK_PREV, MARKER_SPEC = 10, 0.30, 0.10, 0.60
ROOT = '/Users/cossa/Desktop/projects/MiTo/data_test/source_data/data/general/AFMs'
base = mt.pp.filter_cells(sc.read(f'{ROOT}/{ds}/afm_unfiltered.h5ad'), cell_filter='filter2')
N = base.shape[0]; cells = base.obs_names
gbc = base.obs['GBC'].astype(str); sizes = gbc.value_counts()
GT = set(pd.read_csv(f'real_{ds}_pvals.csv').query('GT')['var'])

# ---------------------------------------------------------------- candidate variants at each filter
b = base.copy(); annotate_vars(b); b = filter_baseline(b)
names_base = np.array(b.var_names)
AD_b = b.layers['AD'].toarray(); COV_b = b.layers['site_coverage'].toarray().astype(np.int64)
B_ref = binomial_carriers(AD_b, COV_b, alpha_cell=0.001).astype(bool)          # reference carriers, all baseline variants
c = filter_MiTo(b.copy(), **FK)
names = np.array(c.var_names); idx_c = np.searchsorted(names_base, names) if (np.sort(names_base) == names_base).all() else np.array([list(names_base).index(v) for v in names])
X = c.X.toarray(); AD = c.layers['AD'].toarray(); COV = c.layers['site_coverage'].toarray().astype(np.int64); n, m = X.shape
leg = mt.pp.filter_afm(base.copy(), filtering='MiTo', ncores=8); SH = set(leg.var_names)
print(f'{ds}: {N} cells, {sizes.size} GBC clones ({int((sizes >= MIN_CLONE).sum())} with >= {MIN_CLONE} cells, '
      f'{int(sizes[sizes >= MIN_CLONE].sum())} cells); variants: baseline {len(names_base)}, filter_MiTo {m}, shipped {len(SH)}', flush=True)

# ---------------------------------------------------------------- clone markers in the raw data
big_clones = sizes.index[sizes >= MIN_CLONE]
gb = gbc.values
mk_rows = []
for j, v in enumerate(names_base):
    car = B_ref[:, j]
    if car.sum() < 3:
        continue
    vc = pd.Series(gb[car]).value_counts()
    for cl in vc.index[:3]:
        if cl not in big_clones:
            continue
        prev = vc[cl]/sizes[cl]; spec = vc[cl]/car.sum()
        if spec >= MARKER_SPEC and prev >= WEAK_PREV:
            mk_rows.append(dict(var=v, clone=cl, clone_size=int(sizes[cl]), prev_in_clone=round(prev, 2), specificity=round(spec, 2),
                                carriers=int(car.sum()), kind='marker' if prev >= MARKER_PREV else 'weak marker',
                                median_AF_carriers=round(float(np.median(b.X[car, j].toarray())), 3)))
MK = pd.DataFrame(mk_rows)


# ---------------------------------------------------------------- pipelines with per-variant stage + labels
def tree_and_cut(cols, Bk):
    Bk = Bk > 0
    few = Bk.sum(0) < 2; high = Bk.mean(0) > 0.5
    kv = np.flatnonzero(~few & ~high)
    al = maxcompat(Bk[:, kv].astype(np.int8)); fc = kv[al]
    stage = {}
    for i_, j in enumerate(cols):
        stage[names[j]] = 'lost: <2 calls' if few[i_] else ('lost: prevalence>0.5' if high[i_] else 'lost: four-gamete')
    for j in cols[fc]:
        stage[names[j]] = 'final'
    Bs = Bk[:, fc].astype(np.int8); Xk = X[:, cols]
    kc = Bs.sum(1) >= 1
    a = ad.AnnData(X=csr_matrix(character_af(Bs, Xk[:, fc])[kc]), obs=c.obs.iloc[np.flatnonzero(kc)][['GBC']].copy(), layers={'bin': csr_matrix(Bs[kc])})
    a.var_names = [f'{names[cols[j]]}#{i}' for i, j in enumerate(fc)]
    a.uns['genotyping'] = {'bin_method': 'x', 'binarization_kwargs': {}}
    for key in ('scLT_system', 'pp_method'): a.uns[key] = base.uns[key]
    mt.pp.compute_distances(a, precomputed=True, metric='weighted_jaccard', ncores=8, verbose=False)
    tree = mt.tl.build_tree(a, precomputed=True, solver='UPMGA')
    Bt = a.layers['bin'].toarray() > 0
    lab_cut = evidence_cut(tree, Bt.astype(float), list(a.obs_names), min_in=0.25, ratio=3.0, self_min_in=0.85, ladder='descend')
    _, core = cell_membership(kernel(Xk, raw=True), Bs > 0, tau=0.0)
    lab = lab_cut.mask(~(Bt & (core[kc] >= 0.25)).any(1), 'unassigned')
    full = pd.Series('unassigned', index=cells, dtype=object); full.loc[lab.index] = lab.values
    cut = pd.Series('unassigned', index=cells, dtype=object); cut.loc[lab_cut.index] = lab_cut.values
    has_call = pd.Series(False, index=cells); has_call.loc[c.obs_names[kc]] = True
    call_mat = pd.DataFrame(Bs.astype(bool), index=c.obs_names, columns=names[cols[fc]])
    return full, cut, has_call, stage, call_mat


P = {}
g_flat, _ = em_genotype(AD, COV, np.full((n, n), 1/n), mode='bb')
keep, _, _ = carrier_nonrandomness(X, (g_flat > 0.7).astype(np.int8), cell_depth=COV.mean(1), alpha=0.05)
cols = np.flatnonzero(keep)
g, _ = em_genotype(AD[:, cols], COV[:, cols], kernel(X[:, cols]), mode='bb')
Bg = impute_dropouts(X[:, cols], (g > 0.7).astype(np.int8), k=30, thr=0.6, min_support=1)[0]
P['guarded'] = (cols,) + tree_and_cut(cols, Bg)
Bq = binomial_carriers(AD, COV, alpha_cell=0.001)
keep, _, _ = carrier_nonrandomness(X, Bq, cell_depth=COV.mean(1), alpha=0.05)
cols = np.flatnonzero(keep)
P['binomial'] = (cols,) + tree_and_cut(cols, Bq[:, cols])
cm = mt.tl.MiToTreeAnnotator(tr := mt.tl.build_tree(leg, precomputed=True, solver='UPMGA'))
cm.clonal_inference(max_fraction_unassigned=MFU)
shp = pd.Series('unassigned', index=cells, dtype=object)
shp.loc[tr.cell_meta.index] = np.where(tr.cell_meta['MiTo clone'].isna(), 'unassigned', tr.cell_meta['MiTo clone'].astype(str))


def variant_stage(v, pipe):
    if v not in set(names):
        return 'lost: filter_MiTo'
    return P[pipe][4].get(v, 'lost: QC')


# ---------------------------------------------------------------- 1. variants
V = pd.DataFrame(dict(var=names_base))
V['GT'] = V['var'].isin(GT); V['shipped'] = V['var'].isin(SH)
for pipe in P:
    V[f'stage[{pipe}]'] = V['var'].map(lambda v: variant_stage(v, pipe))
best = MK.sort_values(['kind', 'prev_in_clone'], ascending=[True, False]).drop_duplicates('var').set_index('var') if len(MK) else pd.DataFrame()
V['marker_kind'] = V['var'].map(best['kind']) if len(best) else np.nan
V['marker_kind'] = V['marker_kind'].fillna('not a clone marker')
V['marker_clone_size'] = V['var'].map(best['clone_size']) if len(best) else np.nan
V.to_csv(f'loss_variants_{ds}.csv', index=False)
print('\n==== 1. VARIANTS: where each class is lost')
for pipe in P:
    print(f'\n-- {pipe}')
    print(pd.crosstab(V['marker_kind'], V[f'stage[{pipe}]'], margins=True).to_string())
print('\n-- shipped keeps, by marker class:'); print(V[V.shipped].marker_kind.value_counts().to_string())

# ---------------------------------------------------------------- 2. clones
def clone_status(lab, cl):
    x = lab[gbc == cl]; x = x[x != 'unassigned']
    frac = len(x)/sizes[cl]
    if len(x) == 0:
        return 'lost (no cell assigned)', frac
    top = x.value_counts(); purity = (gbc[lab == top.index[0]] == cl).mean()
    if frac >= .5 and top.iloc[0]/len(x) >= .8 and purity >= .8:
        return 'recovered', frac
    return 'partial', frac


C_rows = []
for cl in big_clones:
    mks = MK[(MK.clone == cl) & (MK.kind == 'marker')]['var'].tolist()
    weak = MK[(MK.clone == cl) & (MK.kind == 'weak marker')]['var'].tolist()
    r = dict(clone=cl, size=int(sizes[cl]), n_markers=len(mks), n_weak=len(weak), best_marker_prev=MK[MK.clone == cl].prev_in_clone.max() if len(MK[MK.clone == cl]) else 0)
    for pipe in P:
        st, frac = clone_status(P[pipe][1], cl)
        final_mk = [v for v in mks + weak if variant_stage(v, pipe) == 'final']
        r[f'{pipe}_status'] = st; r[f'{pipe}_frac_assigned'] = round(frac, 2); r[f'{pipe}_markers_final'] = len(final_mk)
        lost = [variant_stage(v, pipe) for v in mks + weak if variant_stage(v, pipe) != 'final']
        r[f'{pipe}_markers_lost_at'] = ';'.join(f'{k_}:{v_}' for k_, v_ in pd.Series(lost).value_counts().items()) if lost else ''
    st, frac = clone_status(shp, cl); r['shipped_status'] = st; r['shipped_frac_assigned'] = round(frac, 2)
    C_rows.append(r)
C = pd.DataFrame(C_rows).sort_values('size', ascending=False)
C.to_csv(f'loss_clones_{ds}.csv', index=False)
print('\n==== 2. CLONES (>= %d cells)' % MIN_CLONE)
print(C.to_string(index=False))
for pipe in list(P) + ['shipped']:
    print(f'\n-- {pipe}: status x has marker in data')
    print(pd.crosstab(C[f'{pipe}_status'], np.where(C.n_markers > 0, 'has marker', np.where(C.n_weak > 0, 'weak marker only', 'no marker'))).to_string())

# ---------------------------------------------------------------- 3. unassigned cells, by cause
cell_rows = []
marker_of = MK.groupby('clone')['var'].apply(list).to_dict()
for pipe in P:
    cols_p, full, cut, has_call, stage, call_mat = P[pipe]
    for cell in cells[full.loc[cells] == 'unassigned']:
        cl = gbc[cell]; size = sizes[cl]
        mks = marker_of.get(cl, [])
        kept = [v for v in mks if stage.get(v) == 'final']
        if size < MIN_CLONE:
            cause = 'a. clone < 10 cells'
        elif not mks:
            cause = 'b. clone has no marker in the data'
        elif not kept:
            cause = 'c. clone markers lost by the pipeline'
        elif not (cell in call_mat.index and call_mat.loc[cell, kept].any()):
            cause = 'd. marker kept, cell has no call on it (dropout)'
        elif cut[cell] != 'unassigned':
            cause = 'e. called on marker, removed by abstention'
        else:
            cause = 'f. called on marker, not assigned by the cut'
        cell_rows.append(dict(pipeline=pipe, cell=cell, clone=cl, clone_size=size, cause=cause,
                              reads_on_clone_markers=int(AD_b[base.obs_names.get_loc(cell), [list(names_base).index(v) for v in mks]].sum()) if mks else 0))
U = pd.DataFrame(cell_rows)
U.to_csv(f'loss_cells_{ds}.csv', index=False)
print('\n==== 3. UNASSIGNED CELLS by cause')
print(pd.crosstab(U.cause, U.pipeline, margins=True).to_string())
print('\n-- cause d (dropout): reads on the clone markers (any significance)')
print(U[U.cause.str.startswith('d.')].groupby('pipeline').reads_on_clone_markers.describe().round(1).to_string())
print('\n-- cause c: which clones, and at which stage their markers are lost')
cc = U[U.cause.str.startswith('c.')].groupby(['pipeline', 'clone', 'clone_size']).size().rename('cells').reset_index()
for pipe in P:
    x = cc[cc.pipeline == pipe].merge(C[['clone', f'{pipe}_markers_lost_at', 'n_markers', 'n_weak']], on='clone')
    print(f'\n[{pipe}]'); print(x.sort_values('cells', ascending=False).to_string(index=False))
print('ALL DONE')
