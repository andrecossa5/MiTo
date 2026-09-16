"""
Binomial carriers for the QC AND as final genotypes (instead of graph-prior EM), with/without imputation.

Usage (from dev/redesign/results): PYTHONPATH=..:../benchmarks python ../benchmarks/binom_genotypes.py <dataset>
"""
import sys, warnings, logging
sys.setrecursionlimit(100000); warnings.filterwarnings('ignore'); logging.disable(logging.INFO)
import numpy as np, pandas as pd, scanpy as sc, mito as mt, anndata as ad
from scipy.sparse import csr_matrix
from sklearn.metrics import homogeneity_completeness_v_measure as hcv
from mito.pp.filters import annotate_vars, filter_baseline, filter_MiTo
from grid import FK, kernel, maxcompat, character_af
from impute import impute_dropouts
from joincount import carrier_nonrandomness
from carriers import binomial_carriers
from qc2_bench import graph_genotype
from cutter import evidence_cut
from compat import cell_membership
pd.set_option('display.width', 250)
ds = sys.argv[1]
ROOT = '/Users/cossa/Desktop/projects/MiTo/data_test/source_data/data/general/AFMs'
base = mt.pp.filter_cells(sc.read(f'{ROOT}/{ds}/afm_unfiltered.h5ad'), cell_filter='filter2'); N = base.shape[0]
gbc_all = base.obs['GBC'].astype(str); sizes = gbc_all.value_counts(); big = sizes.index[sizes > 100]
GT = set(pd.read_csv(f'real_{ds}_pvals.csv').query('GT')['var'])
b = base.copy(); annotate_vars(b); b = filter_baseline(b); c = filter_MiTo(b, **FK)
X = c.X.toarray(); AD = c.layers['AD'].toarray(); COV = c.layers['site_coverage'].toarray().astype(np.int64)
names = np.array(c.var_names)
rows = []
for a_cell in (0.001, 0.01):
    Bq = binomial_carriers(AD, COV, alpha_cell=a_cell)
    keep, _, _ = carrier_nonrandomness(X, Bq, cell_depth=COV.mean(1), alpha=0.05)
    cols = np.flatnonzero(keep)
    Xk = X[:, cols]
    B_binom = Bq[:, cols]
    B_binom_imp = impute_dropouts(Xk, B_binom, k=30, thr=0.6, min_support=1)[0]
    _, B_em = graph_genotype(X, AD, COV, cols)
    for geno, Bk in [('graph EM + imputation (current)', B_em), ('binomial', B_binom), ('binomial + imputation', B_binom_imp)]:
        Bk = Bk > 0
        kv = np.flatnonzero((Bk.mean(0) <= 0.5) & (Bk.sum(0) >= 2))
        al = maxcompat(Bk[:, kv].astype(np.int8)); fc = kv[al]
        Bs = Bk[:, fc].astype(np.int8)
        kc = Bs.sum(1) >= 1
        a = ad.AnnData(X=csr_matrix(character_af(Bs, Xk[:, fc])[kc]), obs=c.obs.iloc[np.flatnonzero(kc)][['GBC']].copy(),
                       layers={'bin': csr_matrix(Bs[kc])})
        a.var_names = [f'{names[cols[j]]}#{i}' for i, j in enumerate(fc)]
        a.uns['genotyping'] = {'bin_method': 'x', 'binarization_kwargs': {}}
        for key in ('scLT_system', 'pp_method'): a.uns[key] = base.uns[key]
        mt.pp.compute_distances(a, precomputed=True, metric='weighted_jaccard', ncores=8, verbose=False)
        tree = mt.tl.build_tree(a, precomputed=True, solver='UPMGA')
        Bt = a.layers['bin'].toarray() > 0
        lab = evidence_cut(tree, Bt.astype(float), list(a.obs_names), min_in=0.25, ratio=3.0, self_min_in=0.85, ladder='descend')
        _, core = cell_membership(kernel(Xk, raw=True), Bs > 0, tau=0.0)
        lab = lab.mask(~(Bt & (core[kc] >= 0.25)).any(1), 'unassigned')
        full = pd.Series('unassigned', index=base.obs_names, dtype=object); full.loc[lab.index] = lab.values
        ok = full != 'unassigned'
        rec = 0
        for cl in big:
            x = full[gbc_all == cl]; x = x[x != 'unassigned']
            if len(x) and len(x)/sizes[cl] >= .5:
                top = x.value_counts(); rec += int(top.iloc[0]/len(x) >= .8 and (gbc_all[full == top.index[0]] == cl).mean() >= .8)
        h, cp, _ = hcv(gbc_all[ok], full[ok])
        rows.append(dict(dataset=ds, alpha_cell=a_cell, genotypes=geno, qc_kept=int(cols.size), final_vars=int(fc.size),
                         GT_final=int(np.isin(names[cols[fc]], list(GT)).sum()), calls=int(Bs.sum()),
                         cells_pct=round(100*ok.sum()/N, 1), labels=int(full[ok].nunique()), ARI=round(mt.ut.custom_ARI(gbc_all[ok], full[ok]), 3),
                         homog=round(h, 3), compl=round(cp, 3), big_recovered=f'{rec}/{len(big)}'))
        print(rows[-1], flush=True)
        pd.DataFrame(rows).to_csv(f'binom_genotypes_{ds}.csv', index=False)
print(pd.DataFrame(rows).to_string(index=False))
print('ALL DONE')
