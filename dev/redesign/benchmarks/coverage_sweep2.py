"""
Coverage levers on the binomial pipeline (QC 1e-4 + SNR >= 10): genotype threshold x imputation x abstention.
Also reports, for the default, where cells are lost per clone.

real: python ../benchmarks/coverage_sweep.py real <dataset>
sim : python ../benchmarks/coverage_sweep.py sim <shard> <n_shards>
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
from carriers import binomial_carriers, signal_to_background
from cutter import evidence_cut
from compat import cell_membership
from rescue import pooled_rescue
pd.set_option('display.width', 260)
mode = sys.argv[1]


def run_one(afm, truth_col, shipped=None, tag=None, per_clone=False):
    b = afm.copy(); annotate_vars(b); b = filter_baseline(b); c = filter_MiTo(b, **FK)
    X = c.X.toarray(); AD = c.layers['AD'].toarray(); COV = c.layers['site_coverage'].toarray().astype(np.int64); n, m = X.shape
    truth = pd.Series(afm.obs[truth_col].astype(str).values, index=afm.obs_names).loc[c.obs_names]
    sizes = truth.value_counts(); N = afm.shape[0]
    rows, labels, funnel = [], {}, []

    def tree_cut(cols, Bk, tau=True):
        Bk = Bk > 0
        kv = np.flatnonzero((Bk.mean(0) <= 0.5) & (Bk.sum(0) >= 2))
        al = maxcompat(Bk[:, kv].astype(np.int8)); fc = kv[al]
        Bs = Bk[:, fc].astype(np.int8); Xk = X[:, cols]; kc = Bs.sum(1) >= 1
        out = np.full(n, 'unassigned', dtype=object)
        if kc.sum() < 10 or fc.size < 2:
            return out, fc, Bs, int(kc.sum()), 0
        a = ad.AnnData(X=csr_matrix(character_af(Bs, Xk[:, fc])[kc]), obs=pd.DataFrame(index=c.obs_names[kc]), layers={'bin': csr_matrix(Bs[kc])})
        a.var_names = [f'v{i}' for i in range(fc.size)]
        a.uns['genotyping'] = {'bin_method': 'x', 'binarization_kwargs': {}}
        for key in ('scLT_system', 'pp_method'):
            if key in afm.uns: a.uns[key] = afm.uns[key]
        mt.pp.compute_distances(a, precomputed=True, metric='weighted_jaccard', ncores=4, verbose=False)
        tree = mt.tl.build_tree(a, precomputed=True, solver='UPMGA')
        Bt = a.layers['bin'].toarray() > 0
        lab = evidence_cut(tree, Bt.astype(float), list(a.obs_names), min_in=0.25, ratio=3.0, self_min_in=0.85, ladder='descend')
        n_cut = int((lab != 'unassigned').sum())
        if tau:
            _, core = cell_membership(kernel(Xk, raw=True), Bs > 0, tau=0.0)
            lab = lab.mask(~(Bt & (core[kc] >= 0.25)).any(1), 'unassigned')
        out[np.flatnonzero(kc)] = lab.values
        return out, fc, Bs, int(kc.sum()), n_cut

    def score(name, lab, **kw):
        lab = pd.Series(np.asarray(lab, dtype=object), index=c.obs_names); ok = lab != 'unassigned'
        rec = 0
        for cl, size in sizes.items():
            x = lab[truth == cl]; x = x[x != 'unassigned']
            if len(x) and len(x)/size >= .5:
                top = x.value_counts()
                rec += int(top.iloc[0]/len(x) >= .8 and (truth[lab == top.index[0]] == cl).mean() >= .8)
        row = dict(pipeline=name, cells_pct=round(100*ok.sum()/N, 1), labels=int(lab[ok].nunique()),
                   ARI=round(mt.ut.custom_ARI(truth[ok], lab[ok]), 3) if ok.sum() > 10 else np.nan,
                   clones_rec=rec, of_clones=int(sizes.size), **(tag or {}), **kw)
        if shipped is not None:
            s_ = shipped.loc[c.obs_names]
            common = lab.index[ok & (s_ != 'unassigned')]
            row['ARI_shared (this/shipped)'] = f'{mt.ut.custom_ARI(truth[common], lab[common]):.3f}/{mt.ut.custom_ARI(truth[common], s_[common]):.3f}'
        rows.append(row); labels[name] = lab
        return row

    # QC once
    Bq = binomial_carriers(AD, COV, alpha_cell=1e-4)
    keep, _, _ = carrier_nonrandomness(X, Bq, cell_depth=COV.mean(1), alpha=0.05)
    cols = np.flatnonzero(keep)
    for a_geno in (1e-3,):
        Bg, p0 = binomial_carriers(AD[:, cols], COV[:, cols], alpha_cell=a_geno, return_background=True)
        ok_snr, _ = signal_to_background(AD[:, cols], X[:, cols], p0, min_ratio=10.0)
        cs = cols[ok_snr]; Bg = Bg[:, ok_snr]; p0s = p0[ok_snr]
        IMPS = [('', None), ('imp.6', dict(thr=0.6)), ('imp.8', dict(thr=0.8)),
                ('imp.6+coh.5', dict(thr=0.6, min_coherence=0.5)), ('imp.8+coh.5', dict(thr=0.8, min_coherence=0.5))]
        for iname, ikw in IMPS:
            Bk = impute_dropouts(X[:, cs], Bg, k=30, min_support=1, **ikw)[0] if ikw else Bg
            for tau in (True,):
                lab, fc, Bs, n_call, n_cut = tree_cut(cs, Bk, tau=tau)
                nm = f'geno {a_geno:g}{" + " + iname if iname else ""}{" + tau" if tau else ""}'
                score(nm, lab, qc_kept=int(cols.size), snr_kept=int(cs.size), calls=int((Bk > 0).sum()),
                      cells_with_call=n_call, cells_after_cut=n_cut)
                if tau and fc.size:
                    r, det = pooled_rescue(lab[lab != 'x'].values if False else np.asarray(lab, dtype=object),
                                           Bk[:, fc] > 0, AD[:, cs[fc]], COV[:, cs[fc]], p0s[fc], alpha=0.01)
                    score(nm + ' + pooled', r, qc_kept=int(cols.size), snr_kept=int(cs.size), calls=int((Bk > 0).sum()),
                          cells_with_call=n_call, cells_after_cut=n_cut, rescued=int(det.assigned.sum()) if len(det) else 0)
    if per_clone:
        print('\n-- per clone: share of cells assigned')
        pc = pd.DataFrame({nm: [round((lab[truth == cl] != 'unassigned').mean(), 2) for cl in sizes.index] for nm, lab in labels.items()},
                          index=[f'{cl[:8]} (n={v})' for cl, v in sizes.items()])
        print(pc.to_string())
    return rows


if mode == 'real':
    ds = sys.argv[2]
    MFU = {'MDA_clones': 0.05, 'MDA_lung': 0.1, 'MDA_PT': 0.1}[ds]
    ROOT = '/Users/cossa/Desktop/projects/MiTo/data_test/source_data/data/general/AFMs'
    base = mt.pp.filter_cells(sc.read(f'{ROOT}/{ds}/afm_unfiltered.h5ad'), cell_filter='filter2')
    leg = mt.pp.filter_afm(base.copy(), filtering='MiTo', ncores=8)
    tr = mt.tl.build_tree(leg, precomputed=True, solver='UPMGA')
    m0 = mt.tl.MiToTreeAnnotator(tr); m0.clonal_inference(max_fraction_unassigned=MFU)
    shp = pd.Series('unassigned', index=base.obs_names, dtype=object)
    shp.loc[tr.cell_meta.index] = np.where(tr.cell_meta['MiTo clone'].isna(), 'unassigned', tr.cell_meta['MiTo clone'].astype(str))
    rows = run_one(base, 'GBC', shipped=shp, tag=dict(dataset=ds), per_clone=True)
    R = pd.DataFrame(rows); R.to_csv(f'coverage2_{ds}.csv', index=False)
    print(R.to_string(index=False))
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
        rows += run_one(s, 'clone', tag=dict(clones=nk, topo=topo, seed=seed))
        pd.DataFrame(rows).to_csv(f'coverage2_sim_{shard}.csv', index=False)
        print(f'done {nk}{topo}/{seed}', flush=True)
print('ALL DONE')
