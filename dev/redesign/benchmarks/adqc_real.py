"""Guarded genotyper: QC2 (alpha=0.05, one_sided=False) with/without split, evidence_cut and MiToTreeAnnotator, vs SHIPPED."""
import sys, time, warnings, logging; sys.setrecursionlimit(100000); warnings.filterwarnings('ignore'); logging.disable(logging.INFO)
import numpy as np, pandas as pd, scanpy as sc, mito as mt, anndata as ad
from scipy.sparse import csr_matrix
from sklearn.metrics import homogeneity_completeness_v_measure as hcv
from mito.pp.filters import annotate_vars, filter_baseline, filter_MiTo
from geno2 import em_genotype
from grid import FK, kernel
from joincount import carrier_nonrandomness
from qc2_bench import downstream
from cutter import evidence_cut
from compat import cell_membership
ds = sys.argv[1]
ROOT = '/Users/cossa/Desktop/projects/MiTo/data_test/source_data/data/general/AFMs'
MFU = {'MDA_clones': 0.05, 'MDA_lung': 0.1, 'MDA_PT': 0.1}[ds]
base = mt.pp.filter_cells(sc.read(f'{ROOT}/{ds}/afm_unfiltered.h5ad'), cell_filter='filter2'); N = base.shape[0]
gbc_all = base.obs['GBC'].astype(str); sizes = gbc_all.value_counts(); big = sizes.index[sizes > 100]
GT = set(pd.read_csv(f'real_{ds}_pvals.csv').query('GT')['var'])
rows, labels = [], {}

def full(lab):
    out = pd.Series('unassigned', index=base.obs_names, dtype=object); out.loc[lab.index] = lab.values; return out

def metrics(lab, **kw):
    ok = lab != 'unassigned'; g, l = gbc_all[ok], lab[ok]
    h, cp, _ = hcv(g, l)
    rec = []
    for cl in big:
        x = lab[gbc_all == cl]; x = x[x != 'unassigned']
        if len(x) == 0: rec.append(False); continue
        top = x.value_counts(); best = top.index[0]
        rec.append(len(x)/sizes[cl] >= .5 and top.iloc[0]/len(x) >= .8 and (gbc_all[lab == best] == cl).mean() >= .8)
    r = dict(cells_pct=round(100*ok.sum()/N, 1), labels=int(l.nunique()), ARI=round(mt.ut.custom_ARI(g, l), 3),
             NMI=round(mt.ut.normalized_mutual_info_score(g, l), 3), homog=round(h, 3), compl=round(cp, 3),
             big_recovered=f'{sum(rec)}/{len(big)}' if len(big) else '-', **kw)
    if 'SHIPPED' in labels and kw.get('pipeline', '').find('SHIPPED') < 0:
        s_ = labels['SHIPPED']; common = lab.index[(lab != 'unassigned') & (s_ != 'unassigned')]
        r.update(common_cells=len(common), ARI_common=round(mt.ut.custom_ARI(gbc_all[common], lab[common]), 3),
                 SHIPPED_ARI_common=round(mt.ut.custom_ARI(gbc_all[common], s_[common]), 3))
    return r

t = time.time()
leg = mt.pp.filter_afm(base.copy(), filtering='MiTo', ncores=8)
tr = mt.tl.build_tree(leg, precomputed=True, solver='UPMGA')
m0 = mt.tl.MiToTreeAnnotator(tr); m0.clonal_inference(max_fraction_unassigned=MFU)
cm = tr.cell_meta
labels['SHIPPED'] = full(pd.Series(np.where(cm['MiTo clone'].isna(), 'unassigned', cm['MiTo clone'].astype(str)), index=cm.index))
rows.append(metrics(labels['SHIPPED'], pipeline=f'SHIPPED (mfu={MFU})', caller='MiToTreeAnnotator', n_vars=leg.shape[1], GT_kept=len(set(leg.var_names) & GT)))
print(rows[-1], flush=True)

b = base.copy(); annotate_vars(b); b = filter_baseline(b); c = filter_MiTo(b, **FK)
X = c.X.toarray(); AD = c.layers['AD'].toarray(); COV = c.layers['site_coverage'].toarray().astype(np.int64); n, m = X.shape
import os
MINC = None if os.environ.get('MINC', '0.5') == 'None' else float(os.environ.get('MINC', '0.5'))
TAG = os.environ.get('TAG', '')
from carriers import binomial_carriers
_ALL = {'AD>=2': lambda: (AD >= 2).astype(np.int8), 'AD>=1': lambda: (AD >= 1).astype(np.int8),
        'binom0.001': lambda: binomial_carriers(AD, COV, alpha_cell=0.001), 'binom0.01': lambda: binomial_carriers(AD, COV, alpha_cell=0.01)}
QC_CARRIERS = {k_: _ALL[k_]() for k_ in os.environ.get('QCDEFS', 'AD>=2,AD>=1').split(',')}
for qc_name, Bq in QC_CARRIERS.items():
  keep, pj, pe = carrier_nonrandomness(X, Bq, cell_depth=COV.mean(1), alpha=0.05, min_concentration=MINC)
  cols = np.flatnonzero(keep)
  print(f'QC [{qc_name}] keeps {cols.size} ({sum(v in GT for v in c.var_names[cols])} GT of {len(GT)})', flush=True)
  for split in (False,):
      cols_f, Bs, Xs, Xk = downstream(cols, X, AD, COV, split=split)
      kc = (Bs > 0).sum(1) >= 1
      a = ad.AnnData(X=csr_matrix(Xs[kc]), obs=c.obs.iloc[np.flatnonzero(kc)][['GBC']].copy(), layers={'bin': csr_matrix((Bs[kc] > 0).astype(np.int8))})
      a.var_names = [f'{v}#{i}' for i, v in enumerate(c.var_names[cols_f])]
      a.uns['genotyping'] = {'bin_method': 'knn', 'binarization_kwargs': {}}
      for key in ('scLT_system', 'pp_method'): a.uns[key] = base.uns[key]
      mt.pp.compute_distances(a, precomputed=True, metric='weighted_jaccard', ncores=8, verbose=False)
      tree = mt.tl.build_tree(a, precomputed=True, solver='UPMGA')
      Bk = a.layers['bin'].toarray() > 0
      vk = set(c.var_names[cols_f]); info = dict(n_vars=len(vk), n_chars=int(Bs.shape[1]), GT_kept=len(vk & GT))
      name = f'QC2 a=0.05 [{qc_name}] conc={MINC} no split'
      lab = evidence_cut(tree, Bk.astype(float), list(a.obs_names), min_in=0.25, ratio=3.0, self_min_in=0.85, ladder='descend', one_sided=False)
      _, core = cell_membership(kernel(Xk, raw=True), Bs > 0, tau=0.0)
      bad = ~(Bk & (core[kc] >= 0.25)).any(1)
      for cal, l_ in [('evidence_cut', lab), ('evidence_cut + tau', lab.mask(bad, 'unassigned'))]:
          rows.append(metrics(full(l_), pipeline=name, caller=cal, **info)); print(rows[-1], flush=True)
      try:
          m1 = mt.tl.MiToTreeAnnotator(tree); m1.clonal_inference(max_fraction_unassigned=MFU)
          cm = tree.cell_meta
          l_ = pd.Series(np.where(cm['MiTo clone'].isna(), 'unassigned', cm['MiTo clone'].astype(str)), index=cm.index)
          rows.append(metrics(full(l_), pipeline=name, caller='MiToTreeAnnotator', **info)); print(rows[-1], flush=True)
      except Exception as e:
          print(name, 'annotator failed:', str(e).strip()[:60], flush=True)
      pd.DataFrame(rows).to_csv(f'adqc_real{TAG}_{ds}.csv', index=False)
print('ALL DONE', flush=True)
