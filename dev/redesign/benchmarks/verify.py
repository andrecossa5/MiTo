"""
Verification: LEGACY (shipped) vs DEVELOPMENT pipelines on MDA_clones and simulations.

Usage: python verify.py mda | sim
Results are checkpointed to verify_<target>.csv (resumable).
"""
import sys, os, time, warnings, logging
sys.setrecursionlimit(10000)
warnings.filterwarnings('ignore'); logging.disable(logging.INFO)
import numpy as np, pandas as pd, scanpy as sc, mito as mt
from scipy.sparse import csr_matrix
from mito.pp.filters import annotate_vars, filter_baseline, filter_MiTo
from geno2 import em_genotype
from grid import kernel, votes, maxcompat, FK, SIM
from cutter import evidence_cut
from compat import cell_membership
from impute import impute_dropouts

TAU = 0.25
IMPUTATION = {'none': None, 'imp.6': dict(thr=0.6), 'imp.6+coh.5': dict(thr=0.6, min_coherence=0.5)}
# (imputation, cutter) combinations to report
DEV = [('none', 'evidF'), ('none', 'evid1'), ('imp.6', 'evidF'), ('imp.6', 'evid1')]


def score(lab, gt, n_tot, **kw):
    ok = (lab != 'unassigned').values
    n = int(ok.sum())
    if n < 10:
        return dict(cells=n, pct=round(100*n/n_tot), labels=0, ARI=np.nan, NMI=np.nan, **kw)
    g, l = gt[ok], lab[ok]
    return dict(cells=n, pct=round(100*n/n_tot), labels=int(l.nunique()),
                ARI=round(mt.ut.custom_ARI(g, l), 3),
                NMI=round(mt.ut.normalized_mutual_info_score(g, l), 3), **kw)


def legacy(afm_raw, truth, n_tot, **kw):
    t = time.time()
    leg = mt.pp.filter_afm(afm_raw.copy(), filtering='MiTo')
    tr = mt.tl.build_tree(leg, precomputed=True, solver='UPMGA')
    m0 = mt.tl.MiToTreeAnnotator(tr); m0.clonal_inference()
    cm = tr.cell_meta; ok = ~cm['MiTo clone'].isna()
    lab = pd.Series(np.where(ok, cm['MiTo clone'].astype(str), 'unassigned'), index=cm.index)
    gt = pd.Series(cm[truth].astype(str).values, index=cm.index)
    return [score(lab, gt, n_tot, pipeline='LEGACY', abstain='-', vars=leg.shape[1],
                  secs=round(time.time()-t), **kw)]


def development(afm_raw, truth, n_tot, **kw):
    t0 = time.time()
    b = afm_raw.copy(); annotate_vars(b); b = filter_baseline(b)
    c = filter_MiTo(b, **FK)
    AD = c.layers['AD'].toarray(); COV = c.layers['site_coverage'].toarray().astype(np.int64)
    X = c.X.toarray(); K = kernel(X, raw=True)
    g, _ = em_genotype(AD, COV, kernel(X), mode='bb')
    B0 = (g > 0.7).astype(np.int8)
    t_geno = time.time() - t0
    rows = []
    for imp_name in dict.fromkeys(i for i, _ in DEV):
        t = time.time()
        kwargs = IMPUTATION[imp_name]
        B = B0.copy() if kwargs is None else impute_dropouts(X, B0, k=30, min_support=1, **kwargs)[0]
        kv = B.astype(bool).mean(0) <= 0.5
        Bs = B[:, kv]; al = maxcompat(Bs); Bs = Bs[:, al]
        keep = Bs.sum(1) >= 1
        a0 = c[:, kv].copy()[:, al].copy()
        a0.layers['bin'] = csr_matrix(Bs.astype(np.int8))
        a0.uns['genotyping'] = {'bin_method': 'knn', 'binarization_kwargs': {}}
        a = a0[keep, :].copy()
        if a.shape[0] < 30 or a.shape[1] < 2:
            continue
        mt.pp.compute_distances(a, precomputed=True, metric='weighted_jaccard', ncores=4, verbose=False)
        tree = mt.tl.build_tree(a, precomputed=True, solver='UPMGA')
        Bk = a.layers['bin'].toarray() > 0
        gt = pd.Series(np.asarray(a.obs[truth].astype(str)), index=a.obs_names)
        _, core_full = cell_membership(K, Bs > 0, tau=0.0)
        bad = ~(Bk & (core_full[keep] >= TAU)).any(1)
        for imp_c, cutter in DEV:
            if imp_c != imp_name:
                continue
            lab = votes(a) if cutter == 'votes' else evidence_cut(
                tree, Bk.astype(float), list(a.obs_names), min_in=0.25, ratio=3.0, self_min_in=0.85,
                ladder='descend' if cutter in ('evidF', 'evid1') else 'stop', one_sided=cutter == 'evid1')
            secs = round(t_geno + time.time() - t)
            meta = dict(pipeline=f'DEV {imp_name}+{cutter}', vars=a.shape[1], secs=secs, **kw)
            rows.append(score(lab, gt, n_tot, abstain='none', **meta))
            lab_a = lab.copy(); lab_a[bad] = 'unassigned'
            rows.append(score(lab_a, gt, n_tot, abstain=f'tau{TAU}', **meta))
    return rows


def checkpoint(rows, path):
    pd.DataFrame(rows).to_csv(path, index=False)


if __name__ == '__main__':
    target = sys.argv[1]
    path = f'verify_{target}{os.environ.get("TAG", "")}.csv'
    only = os.environ.get('ONLY')
    rows = pd.read_csv(path).to_dict('records') if os.path.exists(path) else []

    if target == 'mda':
        D = '/Users/cossa/Desktop/projects/MiTo/data_test/afm_unfiltered.h5ad'
        base = mt.pp.filter_cells(sc.read(D), cell_filter='filter2')
        N = base.shape[0]
        print(f'MDA_clones: {N} cells, {base.obs["GBC"].nunique()} GBC clones', flush=True)
        for fn in (legacy, development):
            rows += fn(base, 'GBC', N, dataset='MDA_clones')
            checkpoint(rows, path)

    elif target == 'sim':
        done = {(r['clones'], r['topo'], r['seed'], r['pipeline'].split()[0]) for r in rows}
        for nk in [5, 10, 30, 50]:
            for topo in ['polytomy', 'depth3']:
                for seed in [0, 1, 2]:
                    kw = dict(n_cells=1000, n_clones=nk, frac_double_variants=0.3,
                              frac_noisy_variants=0.2, min_max_ratio_clones=0.2,
                              random_seed=seed, **SIM)
                    kw.update(dict(n_root_clones=nk, max_depth=1) if topo == 'polytomy'
                              else dict(n_root_clones=max(2, nk//3), max_depth=3))
                    s = mt.ut.simulate_afm(**kw)
                    base = dict(dataset='sim', clones=nk, topo=topo, seed=seed)
                    for name, fn in (('LEGACY', legacy), ('DEV', development)):
                        if (nk, topo, seed, name) in done or (only and name != only):
                            continue
                        try:
                            rows += fn(s, 'clone', 1000, **base)
                        except Exception as e:  # noqa: BLE001
                            print(f'  {name} FAILED {nk}/{topo}/{seed}: {type(e).__name__}: {e}', flush=True)
                            rows.append(dict(pipeline=name, abstain='-', error=f'{type(e).__name__}: {e}', **base))
                        checkpoint(rows, path)
                    print(f'done {nk}cl/{topo}/s{seed}', flush=True)
    print('ALL DONE', flush=True)
