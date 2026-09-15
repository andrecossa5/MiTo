"""Where does the pipeline fail on MDA_PT? Marker funnel, reference-graph test, cut diagnosis per clone size."""
import sys, time, warnings, logging
sys.setrecursionlimit(100000); warnings.filterwarnings('ignore'); logging.disable(logging.INFO)
import numpy as np, pandas as pd, scanpy as sc, mito as mt, anndata as ad
from scipy.sparse import csr_matrix
from sklearn.metrics import homogeneity_completeness_v_measure as hcv
from mito.pp.filters import annotate_vars, filter_baseline, filter_MiTo
from geno2 import em_genotype
from grid import FK, kernel, maxcompat
from impute import impute_dropouts
from joincount import carrier_nonrandomness
from qc2_bench import downstream
from cutter import evidence_cut
pd.set_option('display.width', 250); pd.set_option('display.max_rows', 200)
ds = sys.argv[1] if len(sys.argv) > 1 else 'MDA_PT'
ROOT = '/Users/cossa/Desktop/projects/MiTo/data_test/source_data/data/general/AFMs'
base = mt.pp.filter_cells(sc.read(f'{ROOT}/{ds}/afm_unfiltered.h5ad'), cell_filter='filter2')
N = base.shape[0]
gbc = base.obs['GBC'].astype(str)
sizes = gbc.value_counts()
print(f'{ds}: {N} cells, {sizes.size} GBC clones; clone size quantiles', sizes.quantile([.1, .25, .5, .75, .9]).round(0).to_dict(),
      '; clones >=10 cells:', int((sizes >= 10).sum()), flush=True)

# ---------------- part 1: marker funnel
gt_afm = mt.pp.filter_afm(base.copy(), filtering='GT_enriched', lineage_column='GBC', ncores=8)
GT = list(gt_afm.var_names)
Gb = gt_afm.layers['bin'].toarray() > 0
gt_gbc = gt_afm.obs['GBC'].astype(str).values
leg = mt.pp.filter_afm(base.copy(), filtering='MiTo', ncores=8)
SH = set(leg.var_names)
b = base.copy(); annotate_vars(b); b = filter_baseline(b); after_base = set(b.var_names)
c = filter_MiTo(b, **FK)
X = c.X.toarray(); AD = c.layers['AD'].toarray(); COV = c.layers['site_coverage'].toarray().astype(np.int64)
n, m = X.shape
names = list(c.var_names); isGT = np.array([v in set(GT) for v in names]); isSH = np.array([v in SH for v in names])
g_flat, _ = em_genotype(AD, COV, np.full((n, n), 1/n), mode='bb')
Bf = (g_flat > 0.7).astype(np.int8)
cgbc = c.obs['GBC'].astype(str).values
t = time.time()
keep_all, pj_all, pe_all = carrier_nonrandomness(X, Bf, cell_depth=COV.mean(1), alpha=0.05)
keep_gt, pj_gt, pe_gt = carrier_nonrandomness(X, Bf, cell_depth=COV.mean(1), alpha=0.05, ref=isGT)
keep_sh, pj_sh, pe_sh = carrier_nonrandomness(X, Bf, cell_depth=COV.mean(1), alpha=0.05, ref=isSH)
print(f'QC with 3 reference sets done ({time.time()-t:.0f}s)', flush=True)
cols_f, Bs, Xs, Xk = downstream(np.flatnonzero(keep_all), X, AD, COV)
final = set(c.var_names[cols_f])

rows = []
for j, v in enumerate(GT):
    pos = Gb[:, j]
    vc = pd.Series(gt_gbc[pos]).value_counts()
    modal = vc.index[0] if len(vc) else None
    r = dict(var=v, GT_carriers=int(pos.sum()), modal_clone=modal, modal_clone_size=int(sizes.get(modal, 0)),
             modal_share=round(vc.iloc[0]/pos.sum(), 2) if len(vc) else np.nan,
             n_clones_3plus=int((vc >= 3).sum()), in_shipped=v in SH)
    if v not in after_base:
        r['stage'] = 'lost: baseline'
    elif v not in names:
        r['stage'] = 'lost: filter_MiTo'
    else:
        jj = names.index(v)
        r.update(flat_carriers=int(Bf[:, jj].sum()), p_join=pj_all[jj], p_excl=pe_all[jj],
                 p_join_refGT=pj_gt[jj], p_join_refShipped=pj_sh[jj],
                 keep=bool(keep_all[jj]), keep_refGT=bool(keep_gt[jj]), keep_refShipped=bool(keep_sh[jj]))
        r['stage'] = 'lost: QC' if not keep_all[jj] else ('lost: downstream' if v not in final else 'kept')
    rows.append(r)
F = pd.DataFrame(rows)
F.to_csv(f'pt_diag_funnel_{ds}.csv', index=False)
print('\n==== 1. FUNNEL of GT-enriched markers (n=%d)' % len(F))
print(F.stage.value_counts().to_string())
print('shipped keeps', int(F.in_shipped.sum()), 'GT markers; of those lost by QC2 at QC:', int((F.in_shipped & (F.stage == 'lost: QC')).sum()))
q = F[F.stage.isin(['kept', 'lost: QC', 'lost: downstream'])]
print('\n-- markers reaching QC: kept vs rejected (medians)')
print(q.groupby('stage')[['GT_carriers', 'flat_carriers', 'modal_clone_size', 'modal_share', 'n_clones_3plus', 'p_join', 'p_excl']].median().round(3).to_string())
print(q.groupby('stage').size().to_string())
print('\n-- rejected markers by modal share (1 clone vs spread over clones)')
qq = q.assign(spread=pd.cut(q.modal_share, [0, .5, .8, 1.01], labels=['<0.5', '0.5-0.8', '>0.8']))
print(qq.groupby('spread', observed=True).agg(n=('var', 'size'), kept=('keep', 'mean'), kept_refGT=('keep_refGT', 'mean'),
                                              kept_refShipped=('keep_refShipped', 'mean'), in_shipped=('in_shipped', 'mean')).round(2).to_string())
print('\n==== 2. REFERENCE GRAPH: share of GT markers kept / non-GT candidates kept')
for lab_, kk in [('all candidates', keep_all), ('ref = GT-enriched (oracle)', keep_gt), ('ref = shipped variants', keep_sh)]:
    print(f'  {lab_:<30} GT kept {kk[isGT].mean():.2f} ({int(kk[isGT].sum())}/{int(isGT.sum())})   non-GT kept {kk[~isGT].mean():.2f} ({int(kk[~isGT].sum())}/{int((~isGT).sum())})')


# ---------------- part 3: cut diagnosis per clone size
def evaluate(a, front, rows_):
    mt.pp.compute_distances(a, precomputed=True, metric='weighted_jaccard', ncores=8, verbose=False)
    tree = mt.tl.build_tree(a, precomputed=True, solver='UPMGA')
    gt = a.obs['GBC'].astype(str)
    Bk = a.layers['bin'].toarray() > 0
    out = {}
    m0 = mt.tl.MiToTreeAnnotator(tree); m0.clonal_inference(max_fraction_unassigned=0.1)
    cm = tree.cell_meta
    out['MiToTreeAnnotator'] = pd.Series(np.where(cm['MiTo clone'].isna(), 'unassigned', cm['MiTo clone'].astype(str)), index=cm.index).loc[a.obs_names]
    out['evidence_cut'] = evidence_cut(tree, Bk.astype(float), list(a.obs_names), min_in=0.25, ratio=3.0, self_min_in=0.85, ladder='descend')
    for caller, lab in out.items():
        ok = (lab != 'unassigned').values
        h, cpl, _ = hcv(gt[ok], lab[ok])
        full = pd.Series('unassigned', index=base.obs_names, dtype=object); full.loc[lab.index] = lab.values
        rows_.append(dict(front=front, caller=caller, cells_pct=round(100*ok.sum()/N, 1), labels=int(lab[ok].nunique()),
                          ARI=round(mt.ut.custom_ARI(gt[ok], lab[ok]), 3), homogeneity=round(h, 3), completeness=round(cpl, 3), labels_full=full))
    return rows_


rows3 = []
a = ad.AnnData(X=leg.X.copy(), obs=leg.obs[['GBC']].copy(), layers={'bin': leg.layers['bin'].copy()}); a.var_names = leg.var_names
a.uns['genotyping'] = leg.uns.get('genotyping', {'bin_method': 'MiTo', 'binarization_kwargs': {}})
for key in ('scLT_system', 'pp_method'): a.uns[key] = base.uns[key]
evaluate(a, 'SHIPPED variants', rows3)
kc = (Bs > 0).sum(1) >= 1
a = ad.AnnData(X=csr_matrix(Xs[kc]), obs=c.obs.iloc[np.flatnonzero(kc)][['GBC']].copy(), layers={'bin': csr_matrix((Bs[kc] > 0).astype(np.int8))})
a.var_names = [f'{v}#{i}' for i, v in enumerate(c.var_names[cols_f])]
a.uns['genotyping'] = {'bin_method': 'knn', 'binarization_kwargs': {}}
for key in ('scLT_system', 'pp_method'): a.uns[key] = base.uns[key]
evaluate(a, 'QC2 a=0.05 variants', rows3)

print('\n==== 3. CUT: purity (homogeneity) vs fragmentation (completeness)')
print(pd.DataFrame([{k_: v_ for k_, v_ in r.items() if k_ != 'labels_full'} for r in rows3]).to_string(index=False))

bins = [0, 10, 20, 50, 100, 10000]; blab = ['<10', '10-20', '20-50', '50-100', '>100']
per = []
for r in rows3:
    lab = r['labels_full']
    for clone, size in sizes.items():
        cells = gbc.index[gbc == clone]
        l = lab.loc[cells]; assigned = l[l != 'unassigned']
        frac_assigned = len(assigned)/size
        if len(assigned):
            top = assigned.value_counts()
            best = top.index[0]
            best_label_cells = lab.index[lab == best]
            purity = (gbc.loc[best_label_cells] == clone).mean()
            share = top.iloc[0]/len(assigned)
            n_labels = assigned.nunique()
        else:
            purity = share = np.nan; n_labels = 0
        per.append(dict(front=r['front'], caller=r['caller'], clone=clone, size=size, size_bin=pd.cut([size], bins, labels=blab, right=False)[0],
                        frac_assigned=frac_assigned, n_labels=n_labels, top_label_share=share, top_label_purity=purity,
                        recovered=bool(frac_assigned >= 0.5 and share >= 0.8 and purity >= 0.8)))
P = pd.DataFrame(per)
P.to_csv(f'pt_diag_perclone_{ds}.csv', index=False)
print('\n==== 3b. PER CLONE SIZE: cells assigned / labels per clone / share in top label / purity of top label / clones recovered')
print(P.groupby(['size_bin', 'front', 'caller'], observed=True).agg(n=('clone', 'size'), frac_assigned=('frac_assigned', 'mean'),
      n_labels=('n_labels', 'mean'), top_share=('top_label_share', 'mean'), purity=('top_label_purity', 'mean'),
      recovered=('recovered', 'mean')).round(2).to_string())
print('ALL DONE')
