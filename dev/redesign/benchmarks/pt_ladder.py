"""
Oracle ladder on one AFM: how far can cell/clone recovery go, and which stage loses it?

Clone markers (GBC used ONLY as diagnostic reference), for clones >= MIN_CLONE cells, over variants
passing filter_baseline, carriers = significant reads (binomial_carriers alpha 0.001):
    marker  >= 30% of the clone's cells, >= 60% of carriers in the clone (weak: 10-30%)
    strict  a variant carried (AD >= 1) by ALL cells of the clone, >= 60% of carriers in the clone

L0  cells in clones with >= 1 marker                                  (upper bound)
L1  cells carrying >= 1 marker of their clone (binomial call / AD >= 1)  (genotyping ceiling)
L2  oracle markers + binomial calls, cell -> clone of the marker it carries (perfect assignment)
L3  oracle markers -> genotyping (binomial | guarded graph EM + imputation) -> tree + cut (+ tau)
L4  QC-selected variants (binomial carriers) -> same

Usage (from dev/redesign/results): PYTHONPATH=..:../benchmarks python ../benchmarks/pt_ladder.py <dataset> <min_clone>
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
pd.set_option('display.width', 250)
ds = sys.argv[1] if len(sys.argv) > 1 else 'MDA_PT'
MIN_CLONE = int(sys.argv[2]) if len(sys.argv) > 2 else 10
ROOT = '/Users/cossa/Desktop/projects/MiTo/data_test/source_data/data/general/AFMs'
base = mt.pp.filter_cells(sc.read(f'{ROOT}/{ds}/afm_unfiltered.h5ad'), cell_filter='filter2')
N = base.shape[0]; cells = base.obs_names
gbc = base.obs['GBC'].astype(str).values; sizes = pd.Series(gbc).value_counts()
b = base.copy(); annotate_vars(b); b = filter_baseline(b)
vb = np.array(b.var_names)
ADb = b.layers['AD'].toarray(); COVb = b.layers['site_coverage'].toarray().astype(np.int64); Xb = b.X.toarray()
Bref = binomial_carriers(ADb, COVb, alpha_cell=0.001).astype(bool)
eligible = sizes.index[sizes >= MIN_CLONE]

# ---- clone markers
mk = {}          # clone -> list of (variant index, kind)
strict = {}
for j in range(len(vb)):
    car = Bref[:, j]
    if car.sum() < 2:
        continue
    vc = pd.Series(gbc[car]).value_counts()
    cl = vc.index[0]
    if cl not in eligible:
        continue
    prev, spec = vc.iloc[0]/sizes[cl], vc.iloc[0]/car.sum()
    if spec >= 0.6 and prev >= 0.1:
        mk.setdefault(cl, []).append((j, 'marker' if prev >= 0.3 else 'weak'))
    in_cl = gbc == cl
    if (ADb[in_cl, j] >= 1).all() and ((ADb[:, j] >= 1) & in_cl).sum()/max((ADb[:, j] >= 1).sum(), 1) >= 0.6:
        strict.setdefault(cl, []).append(j)
with_marker = [cl for cl in eligible if any(k == 'marker' for _, k in mk.get(cl, []))]
with_any = [cl for cl in eligible if cl in mk]
print(f'{ds}: {N} cells; clones >= {MIN_CLONE} cells: {len(eligible)} ({int(sizes[eligible].sum())} cells); '
      f'with marker (>=30%): {len(with_marker)} ({int(sizes[with_marker].sum())} cells); with marker or weak: {len(with_any)} '
      f'({int(sizes[with_any].sum())} cells); with a STRICT variant (in all cells): {len(strict)} ({int(sizes[list(strict)].sum())} cells)', flush=True)
rows = []


def recovered(lab, clones_eval):
    lab = pd.Series(lab, index=cells)
    rec = 0
    for cl in clones_eval:
        x = lab[gbc == cl]; x = x[x != 'unassigned']
        if len(x) and len(x)/sizes[cl] >= .5:
            top = x.value_counts()
            rec += int(top.iloc[0]/len(x) >= .8 and (gbc[lab.values == top.index[0]] == cl).mean() >= .8)
    return rec


def add(step, desc, lab=None, n_cells=None, n_clones=None):
    if lab is not None:
        lab = np.asarray(lab, dtype=object); ok = lab != 'unassigned'
        n_cells = int(ok.sum())
        n_clones = recovered(lab, with_any)
        ari = round(mt.ut.custom_ARI(pd.Series(gbc[ok]), pd.Series(lab[ok])), 3) if ok.sum() > 10 else np.nan
        n_labels = int(pd.Series(lab[ok]).nunique())
    else:
        ari = n_labels = np.nan
    rows.append(dict(step=step, description=desc, cells=n_cells, cells_pct=round(100*n_cells/N, 1), clones_recovered=n_clones,
                     clones_with_marker_or_weak=len(with_any), labels=n_labels, ARI=ari))
    print(rows[-1], flush=True)


add('L0', f'cells in clones >= {MIN_CLONE} with a marker (>=30%)', n_cells=int(sizes[with_marker].sum()), n_clones=len(with_marker))
add('L0b', f'cells in clones >= {MIN_CLONE} with marker or weak marker', n_cells=int(sizes[with_any].sum()), n_clones=len(with_any))
add('L0c', f'cells in clones >= {MIN_CLONE} with a strict variant (all cells)', n_cells=int(sizes[list(strict)].sum()), n_clones=len(strict))

# L1: cells carrying a marker of their own clone
lab_call = np.full(N, 'unassigned', dtype=object); lab_read = np.full(N, 'unassigned', dtype=object)
for cl, lst in mk.items():
    js = [j for j, _ in lst]; in_cl = gbc == cl
    lab_call[in_cl & Bref[:, js].any(1)] = cl
    lab_read[in_cl & (ADb[:, js] >= 1).any(1)] = cl
add('L1a', 'cells with a significant call on a marker of their clone (oracle assignment)', lab=lab_call)
add('L1b', 'cells with >=1 read on a marker of their clone (oracle assignment)', lab=lab_read)

# L2: oracle markers, binomial calls, assignment by carried marker only (no GBC for assignment)
oracle_js = sorted({j for lst in mk.values() for j, _ in lst})
marker_clone = {j: cl for cl, lst in mk.items() for j, _ in lst}
lab_l2 = np.full(N, 'unassigned', dtype=object)
Bo = Bref[:, oracle_js]
for i in range(N):
    carried = [marker_clone[oracle_js[k]] for k in np.flatnonzero(Bo[i])]
    if carried and len(set(carried)) == 1:
        lab_l2[i] = carried[0]
add('L2', 'oracle markers + binomial calls; cell -> clone of the marker(s) it carries (unambiguous only)', lab=lab_l2)


def tree_cut(Xk, Bk, tau=True):
    Bk = Bk > 0
    kv = np.flatnonzero((Bk.mean(0) <= 0.5) & (Bk.sum(0) >= 2))
    al = maxcompat(Bk[:, kv].astype(np.int8)); fc = kv[al]
    Bs = Bk[:, fc].astype(np.int8); kc = Bs.sum(1) >= 1
    a = ad.AnnData(X=csr_matrix(character_af(Bs, Xk[:, fc])[kc]), obs=pd.DataFrame(index=cells[kc]), layers={'bin': csr_matrix(Bs[kc])})
    a.var_names = [f'v{i}' for i in range(fc.size)]
    a.uns['genotyping'] = {'bin_method': 'x', 'binarization_kwargs': {}}
    for key in ('scLT_system', 'pp_method'): a.uns[key] = base.uns[key]
    mt.pp.compute_distances(a, precomputed=True, metric='weighted_jaccard', ncores=8, verbose=False)
    tree = mt.tl.build_tree(a, precomputed=True, solver='UPMGA')
    Bt = a.layers['bin'].toarray() > 0
    lab = evidence_cut(tree, Bt.astype(float), list(a.obs_names), min_in=0.25, ratio=3.0, self_min_in=0.85, ladder='descend')
    if tau:
        _, core = cell_membership(kernel(Xk, raw=True), Bs > 0, tau=0.0)
        lab = lab.mask(~(Bt & (core[kc] >= 0.25)).any(1), 'unassigned')
    out = np.full(N, 'unassigned', dtype=object); out[np.flatnonzero(kc)] = lab.values
    return out, fc.size


# L3: oracle markers through our genotyping + tree + cut
Xo = Xb[:, oracle_js]
lab, nc = tree_cut(Xo, Bref[:, oracle_js])
add('L3a', f'oracle markers, binomial genotypes, tree + cut + tau ({nc} chars)', lab=lab)
lab, nc = tree_cut(Xo, Bref[:, oracle_js], tau=False)
add('L3b', f'oracle markers, binomial genotypes, tree + cut, no abstention ({nc} chars)', lab=lab)
g, _ = em_genotype(ADb[:, oracle_js], COVb[:, oracle_js], kernel(Xo), mode='bb')
Bg = impute_dropouts(Xo, (g > 0.7).astype(np.int8), k=30, thr=0.6, min_support=1)[0]
lab, nc = tree_cut(Xo, Bg)
add('L3c', f'oracle markers, guarded graph EM + imputation, tree + cut + tau ({nc} chars)', lab=lab)

# L4: real variant selection (filter_MiTo + binomial QC)
c = filter_MiTo(b.copy(), **FK)
jc = np.array([list(vb).index(v) for v in c.var_names])
keep, _, _ = carrier_nonrandomness(Xb[:, jc], Bref[:, jc].astype(np.int8), cell_depth=COVb[:, jc].mean(1), alpha=0.05)
sel = jc[keep]
n_oracle_in_sel = len(set(sel) & set(oracle_js))
lab, nc = tree_cut(Xb[:, sel], Bref[:, sel])
add('L4a', f'QC-selected ({sel.size} vars, {n_oracle_in_sel}/{len(oracle_js)} oracle markers), binomial genotypes, tree + cut + tau', lab=lab)
lab, nc = tree_cut(Xb[:, sel], Bref[:, sel], tau=False)
add('L4b', 'QC-selected, binomial genotypes, tree + cut, no abstention', lab=lab)
# L4c: QC-selected restricted to oracle markers (isolates noise variants in the selection)
sel_o = np.array(sorted(set(sel) & set(oracle_js)))
lab, nc = tree_cut(Xb[:, sel_o], Bref[:, sel_o])
add('L4c', f'QC-selected AND oracle markers only ({sel_o.size}), binomial genotypes, tree + cut + tau', lab=lab)

R = pd.DataFrame(rows)
R.to_csv(f'ladder_{ds}_min{MIN_CLONE}.csv', index=False)
print(R.to_string(index=False))
print('ALL DONE')
