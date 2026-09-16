"""
Last-resort test: QC alpha_cell 1e-4, binomial genotypes 1e-3, then read-level rescue (vs pooled rescue).

Real data:   python ../benchmarks/final_rescue.py real <dataset>
Simulations: python ../benchmarks/final_rescue.py sim <shard> <n_shards>
(run from dev/redesign/results with PYTHONPATH=..:../benchmarks)
"""
import sys, warnings, logging
sys.setrecursionlimit(100000); warnings.filterwarnings('ignore'); logging.disable(logging.INFO)
import numpy as np, pandas as pd, scanpy as sc, mito as mt, anndata as ad
from scipy.sparse import csr_matrix
from mito.pp.filters import annotate_vars, filter_baseline, filter_MiTo
from geno2 import em_genotype
from grid import FK, SIM, kernel, maxcompat, character_af
from impute import impute_dropouts
from joincount import carrier_nonrandomness
from carriers import binomial_carriers
from cutter import evidence_cut
from compat import cell_membership
from rescue import pooled_rescue, read_rescue

mode = sys.argv[1]


def run_one(afm, truth_col, cells_all, shipped=None, clone_sets=None, tag=None):
    """All pipelines on one AFM. Returns rows."""
    b = afm.copy(); annotate_vars(b); b = filter_baseline(b); c = filter_MiTo(b, **FK)
    X = c.X.toarray(); AD = c.layers['AD'].toarray(); COV = c.layers['site_coverage'].toarray().astype(np.int64); n, m = X.shape
    truth = pd.Series(afm.obs[truth_col].astype(str).values, index=afm.obs_names)
    sizes = truth.value_counts()
    N = afm.shape[0]
    labels = {}

    def tree_cut(cols, Bk):
        Bk = Bk > 0
        kv = np.flatnonzero((Bk.mean(0) <= 0.5) & (Bk.sum(0) >= 2))
        al = maxcompat(Bk[:, kv].astype(np.int8)); fc = kv[al]
        Bs = Bk[:, fc].astype(np.int8); Xk = X[:, cols]; kc = Bs.sum(1) >= 1
        out = np.full(n, 'unassigned', dtype=object); out_t = out.copy()
        if kc.sum() < 10 or fc.size < 2:
            return out, out_t, fc, Bs
        a = ad.AnnData(X=csr_matrix(character_af(Bs, Xk[:, fc])[kc]), obs=pd.DataFrame(index=c.obs_names[kc]), layers={'bin': csr_matrix(Bs[kc])})
        a.var_names = [f'v{i}' for i in range(fc.size)]
        a.uns['genotyping'] = {'bin_method': 'x', 'binarization_kwargs': {}}
        for key in ('scLT_system', 'pp_method'):
            if key in afm.uns: a.uns[key] = afm.uns[key]
        mt.pp.compute_distances(a, precomputed=True, metric='weighted_jaccard', ncores=4, verbose=False)
        tree = mt.tl.build_tree(a, precomputed=True, solver='UPMGA')
        Bt = a.layers['bin'].toarray() > 0
        lab = evidence_cut(tree, Bt.astype(float), list(a.obs_names), min_in=0.25, ratio=3.0, self_min_in=0.85, ladder='descend')
        _, core = cell_membership(kernel(Xk, raw=True), Bs > 0, tau=0.0)
        lab_t = lab.mask(~(Bt & (core[kc] >= 0.25)).any(1), 'unassigned')
        out[np.flatnonzero(kc)] = lab.values; out_t[np.flatnonzero(kc)] = lab_t.values
        return out, out_t, fc, Bs

    # guarded reference (current default)
    g_flat, _ = em_genotype(AD, COV, np.full((n, n), 1/n), mode='bb')
    keep, _, _ = carrier_nonrandomness(X, (g_flat > 0.7).astype(np.int8), cell_depth=COV.mean(1), alpha=0.05)
    cols = np.flatnonzero(keep)
    g, _ = em_genotype(AD[:, cols], COV[:, cols], kernel(X[:, cols]), mode='bb')
    Bg = impute_dropouts(X[:, cols], (g > 0.7).astype(np.int8), k=30, thr=0.6, min_support=1)[0]
    _, lab_g, _, _ = tree_cut(cols, Bg)
    labels['guarded (current default)'] = lab_g
    # new: QC 1e-4, genotypes 1e-3
    Bq = binomial_carriers(AD, COV, alpha_cell=1e-4)
    keep, _, _ = carrier_nonrandomness(X, Bq, cell_depth=COV.mean(1), alpha=0.05)
    cols = np.flatnonzero(keep)
    Bgen, p0 = binomial_carriers(AD[:, cols], COV[:, cols], alpha_cell=1e-3, return_background=True)
    lab, lab_t, fc, Bs = tree_cut(cols, Bgen)
    labels['QC 1e-4 + binom 1e-3'] = lab
    labels['QC 1e-4 + binom 1e-3 + tau'] = lab_t
    info = {}
    if fc.size:
        args = (Bgen[:, fc] > 0, AD[:, cols[fc]], COV[:, cols[fc]], p0[fc])
        for base_name, base_lab in (('', lab), (' + tau', lab_t)):
            r, det = read_rescue(base_lab, *args, alpha=0.05)
            labels[f'QC 1e-4 + binom 1e-3{base_name} + read rescue'] = r
            info[f'QC 1e-4 + binom 1e-3{base_name} + read rescue'] = int(det.assigned.sum()) if len(det) else 0
            r, det = pooled_rescue(base_lab, *args, alpha=0.01)
            labels[f'QC 1e-4 + binom 1e-3{base_name} + pooled rescue'] = r
            info[f'QC 1e-4 + binom 1e-3{base_name} + pooled rescue'] = int(det.assigned.sum()) if len(det) else 0
    if shipped is not None:
        labels['SHIPPED'] = shipped.loc[c.obs_names].values

    rows = []
    for name, lab_ in labels.items():
        lab_ = pd.Series(np.asarray(lab_, dtype=object), index=c.obs_names)
        ok = lab_ != 'unassigned'
        t = truth.loc[c.obs_names]
        row = dict(pipeline=name, cells_pct=round(100*ok.sum()/N, 1), labels=int(lab_[ok].nunique()),
                   ARI=round(mt.ut.custom_ARI(t[ok], lab_[ok]), 3) if ok.sum() > 10 else np.nan,
                   NMI=round(mt.ut.normalized_mutual_info_score(t[ok], lab_[ok]), 3) if ok.sum() > 10 else np.nan,
                   rescued=info.get(name, np.nan), **(tag or {}))
        sets = clone_sets if clone_sets is not None else {'all': list(sizes.index)}
        for key, clist in sets.items():
            rec = 0
            for cl in clist:
                x = lab_[t == cl]; x = x[x != 'unassigned']
                if len(x) and len(x)/sizes[cl] >= .5:
                    top = x.value_counts()
                    rec += int(top.iloc[0]/len(x) >= .8 and (t[lab_ == top.index[0]] == cl).mean() >= .8)
            row[f'clones_rec_{key}'] = rec; row[f'of_{key}'] = len(clist)
        if 'SHIPPED' in labels and name != 'SHIPPED':
            s_ = pd.Series(np.asarray(labels['SHIPPED'], dtype=object), index=c.obs_names)
            common = lab_.index[ok & (s_ != 'unassigned')]
            row['shared_cells_vs_shipped'] = len(common)
            row['ARI_shared (this/shipped)'] = f'{mt.ut.custom_ARI(t[common], lab_[common]):.3f}/{mt.ut.custom_ARI(t[common], s_[common]):.3f}'
        rows.append(row)
    return rows


if mode == 'real':
    ds = sys.argv[2]
    MFU = {'MDA_clones': 0.05, 'MDA_lung': 0.1, 'MDA_PT': 0.1}[ds]
    ROOT = '/Users/cossa/Desktop/projects/MiTo/data_test/source_data/data/general/AFMs'
    base = mt.pp.filter_cells(sc.read(f'{ROOT}/{ds}/afm_unfiltered.h5ad'), cell_filter='filter2')
    gbc = base.obs['GBC'].astype(str).values; sizes = pd.Series(gbc).value_counts()
    # clone sets with marker/weak marker (as in pt_ladder.py) for min 10 and 5
    bb = base.copy(); annotate_vars(bb); bb = filter_baseline(bb)
    Bref = binomial_carriers(bb.layers['AD'].toarray(), bb.layers['site_coverage'].toarray().astype(np.int64), alpha_cell=1e-3).astype(bool)
    clone_sets = {}
    for mc in (10, 5):
        eligible = set(sizes.index[sizes >= mc]); found = set()
        for j in range(Bref.shape[1]):
            car = Bref[:, j]
            if car.sum() < 2:
                continue
            vc = pd.Series(gbc[car]).value_counts(); cl = vc.index[0]
            if cl in eligible and vc.iloc[0]/car.sum() >= 0.6 and vc.iloc[0]/sizes[cl] >= 0.1:
                found.add(cl)
        clone_sets[f'min{mc}'] = sorted(found)
    leg = mt.pp.filter_afm(base.copy(), filtering='MiTo', ncores=8)
    tr = mt.tl.build_tree(leg, precomputed=True, solver='UPMGA')
    m0 = mt.tl.MiToTreeAnnotator(tr); m0.clonal_inference(max_fraction_unassigned=MFU)
    shp = pd.Series('unassigned', index=base.obs_names, dtype=object)
    shp.loc[tr.cell_meta.index] = np.where(tr.cell_meta['MiTo clone'].isna(), 'unassigned', tr.cell_meta['MiTo clone'].astype(str))
    rows = run_one(base, 'GBC', base.obs_names, shipped=shp, clone_sets=clone_sets, tag=dict(dataset=ds))
    R = pd.DataFrame(rows); R.to_csv(f'final_rescue_{ds}.csv', index=False)
    pd.set_option('display.width', 260); print(R.to_string(index=False))
else:
    shard, n_shards = int(sys.argv[2]), int(sys.argv[3])
    configs = [(nk, topo, seed) for nk in [5, 10, 30, 50] for topo in ['polytomy', 'depth3'] for seed in [0, 1, 2]]
    rows = []
    for i, (nk, topo, seed) in enumerate(configs):
        if i % n_shards != shard:
            continue
        kw = dict(n_cells=1000, n_clones=nk, frac_double_variants=0.3, frac_noisy_variants=0.2, min_max_ratio_clones=0.2, random_seed=seed, **SIM)
        kw.update(dict(n_root_clones=nk, max_depth=1) if topo == 'polytomy' else dict(n_root_clones=max(2, nk//3), max_depth=3))
        s = mt.ut.simulate_afm(**kw)
        rows += run_one(s, 'clone', s.obs_names, tag=dict(clones=nk, topo=topo, seed=seed))
        pd.DataFrame(rows).to_csv(f'final_rescue_sim_{shard}.csv', index=False)
        print(f'done {nk}{topo}/{seed}', flush=True)
print('ALL DONE')
