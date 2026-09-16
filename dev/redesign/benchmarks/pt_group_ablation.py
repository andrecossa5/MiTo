"""
Oracle ablation on one AFM: which groups of variants, kept by the AD>0-carrier QC, hurt clone calling?

Groups come from flow_<ds>.csv (pt_flow.py), i.e. from GBC ground truth. Variants are removed from
the QC keep set BEFORE genotyping, so genotyping, prevalence cap, four-gamete, tree and cut are redone.

Usage (from dev/redesign/results): PYTHONPATH=..:../benchmarks python ../benchmarks/pt_group_ablation.py MDA_PT
"""
import sys, warnings, logging
sys.setrecursionlimit(100000); warnings.filterwarnings('ignore'); logging.disable(logging.INFO)
import numpy as np, pandas as pd, scanpy as sc, mito as mt, anndata as ad
from scipy.sparse import csr_matrix
from sklearn.metrics import homogeneity_completeness_v_measure as hcv
from mito.pp.filters import annotate_vars, filter_baseline, filter_MiTo
from grid import FK, kernel
from joincount import carrier_nonrandomness
from qc2_bench import downstream
from cutter import evidence_cut
from compat import cell_membership
pd.set_option('display.width', 250)

ds = sys.argv[1] if len(sys.argv) > 1 else 'MDA_PT'
ROOT = '/Users/cossa/Desktop/projects/MiTo/data_test/source_data/data/general/AFMs'
base = mt.pp.filter_cells(sc.read(f'{ROOT}/{ds}/afm_unfiltered.h5ad'), cell_filter='filter2'); N = base.shape[0]
gbc_all = base.obs['GBC'].astype(str); sizes = gbc_all.value_counts(); big = sizes.index[sizes > 100]
F = pd.read_csv(f'flow_{ds}.csv').set_index('var')
b = base.copy(); annotate_vars(b); b = filter_baseline(b); c = filter_MiTo(b, **FK)
X = c.X.toarray(); AD = c.layers['AD'].toarray(); COV = c.layers['site_coverage'].toarray().astype(np.int64)
names = np.array(c.var_names)
keep_ad, _, _ = carrier_nonrandomness(X, (AD >= 1).astype(np.int8), cell_depth=COV.mean(1), alpha=0.05)
qc_ad = set(names[keep_ad])
final_ad = set(F.index[F['stage[AD>0]'] == 'final'])
final_guard = set(F.index[F['stage[guard]'] == 'final'])
ad_only = set(F.index[F.flow == 'AD>0 only'])
spread = set(F.index[F.marks == 'spread over clones (share<0.5)'])
subclonal = set(F.index[F.marks == 'within a large clone (>100): SUBCLONAL'])
mid = set(F.index[F.marks == 'marks a mid clone (10-100)'])
print(f'AD>0 QC keeps {len(qc_ad)}; final {len(final_ad)}; AD>0-only final {len(ad_only)} '
      f'(spread {len(ad_only & spread)}, subclonal {len(ad_only & subclonal)}, mid {len(ad_only & mid)}); '
      f'spread among AD>0 final {len(final_ad & spread)}, subclonal among AD>0 final {len(final_ad & subclonal)}', flush=True)

qc_guard_final_plus = final_guard  # guard final variants all pass the AD>0 QC? report overlap
print('guard-final variants also kept by AD>0 QC:', len(final_guard & qc_ad), 'of', len(final_guard), flush=True)

CONDITIONS = [
    ('1 AD>0: all kept', qc_ad),
    ('2 - AD>0-only spread', qc_ad - (ad_only & spread)),
    ('3 - all spread', qc_ad - spread),
    ('4 - AD>0-only subclonal', qc_ad - (ad_only & subclonal)),
    ('5 - all subclonal', qc_ad - subclonal),
    ('6 - AD>0-only spread & subclonal', qc_ad - (ad_only & (spread | subclonal))),
    ('7 - all AD>0-only', qc_ad - ad_only),
    ('8 guard final + AD>0-only mid-clone markers', final_guard | (ad_only & mid)),
    ('9 guard final (reference, re-run)', final_guard),
]

rows = []
for name, vs in CONDITIONS:
    cols = np.flatnonzero(np.isin(names, list(vs)))
    cols_f, Bs, Xs, Xk = downstream(cols, X, AD, COV, split=False)
    kc = (Bs > 0).sum(1) >= 1
    a = ad.AnnData(X=csr_matrix(Xs[kc]), obs=c.obs.iloc[np.flatnonzero(kc)][['GBC']].copy(), layers={'bin': csr_matrix((Bs[kc] > 0).astype(np.int8))})
    a.var_names = [f'{names[j]}#{i}' for i, j in enumerate(cols_f)]
    a.uns['genotyping'] = {'bin_method': 'x', 'binarization_kwargs': {}}
    for key in ('scLT_system', 'pp_method'): a.uns[key] = base.uns[key]
    mt.pp.compute_distances(a, precomputed=True, metric='weighted_jaccard', ncores=8, verbose=False)
    tree = mt.tl.build_tree(a, precomputed=True, solver='UPMGA')
    Bt = a.layers['bin'].toarray() > 0
    lab = evidence_cut(tree, Bt.astype(float), list(a.obs_names), min_in=0.25, ratio=3.0, self_min_in=0.85, ladder='descend')
    _, core = cell_membership(kernel(Xk, raw=True), Bs > 0, tau=0.0)
    lab_t = lab.mask(~(Bt & (core[kc] >= 0.25)).any(1), 'unassigned')
    fv = set(names[cols_f])
    for abst, l in [('none', lab), ('tau', lab_t)]:
        full = pd.Series('unassigned', index=base.obs_names, dtype=object); full.loc[l.index] = l.values
        ok = full != 'unassigned'
        h, cp, _ = hcv(gbc_all[ok], full[ok])
        rec, nl = 0, []
        for cl in big:
            x = full[gbc_all == cl]; x = x[x != 'unassigned']
            nl.append(x.nunique())
            if len(x) and len(x)/sizes[cl] >= .5:
                top = x.value_counts(); best = top.index[0]
                rec += int(top.iloc[0]/len(x) >= .8 and (gbc_all[full == best] == cl).mean() >= .8)
        rows.append(dict(condition=name, abstain=abst, qc_vars=len(vs), final_vars=len(fv),
                         spread_in_final=len(fv & spread), subclonal_in_final=len(fv & subclonal), mid_in_final=len(fv & mid),
                         cells_pct=round(100*ok.sum()/N, 1), labels=int(full[ok].nunique()),
                         ARI=round(mt.ut.custom_ARI(gbc_all[ok], full[ok]), 3), homog=round(h, 3), compl=round(cp, 3),
                         big_recovered=f'{rec}/{len(big)}', big_labels_per_clone=round(float(np.mean(nl)), 1)))
    print(pd.DataFrame(rows[-2:]).to_string(index=False, header=len(rows) == 2), flush=True)
    pd.DataFrame(rows).to_csv(f'group_ablation_{ds}.csv', index=False)
print('ALL DONE')
