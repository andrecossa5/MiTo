"""
Variant flow: QC carriers = AD>0 vs guarded flat prior (current dev pipeline), on one AFM.

Per candidate variant: QC decision under both carrier definitions, fate downstream
(min calls, prevalence cap, four-gamete), what it marks (GT-enriched, shipped, modal GBC
clone, clone size, share). Then, for large barcode clones fragmented by the AD>0 pipeline,
the characters that best explain the split and whether they are AD>0-only.

Usage (from dev/redesign/results): PYTHONPATH=..:../benchmarks python ../benchmarks/pt_flow.py MDA_PT
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
from cutter import evidence_cut
from compat import cell_membership
pd.set_option('display.width', 250); pd.set_option('display.max_rows', 300); pd.set_option('display.max_columns', 40)

ds = sys.argv[1] if len(sys.argv) > 1 else 'MDA_PT'
ROOT = '/Users/cossa/Desktop/projects/MiTo/data_test/source_data/data/general/AFMs'
base = mt.pp.filter_cells(sc.read(f'{ROOT}/{ds}/afm_unfiltered.h5ad'), cell_filter='filter2')
gbc_all = base.obs['GBC'].astype(str); sizes = gbc_all.value_counts()
GT = set(pd.read_csv(f'real_{ds}_pvals.csv').query('GT')['var'])
SH = set(mt.pp.filter_afm(base.copy(), filtering='MiTo', ncores=8).var_names)
b = base.copy(); annotate_vars(b); b = filter_baseline(b); c = filter_MiTo(b, **FK)
X = c.X.toarray(); AD = c.layers['AD'].toarray(); COV = c.layers['site_coverage'].toarray().astype(np.int64)
n, m = X.shape
names = np.array(c.var_names); gbc = c.obs['GBC'].astype(str).values

g_flat, _ = em_genotype(AD, COV, np.full((n, n), 1/n), mode='bb', guard=True)
CARRIERS = {'guard': (g_flat > 0.7).astype(np.int8), 'AD>0': (AD >= 1).astype(np.int8)}


def run(Bq):
    keep, pj, pe = carrier_nonrandomness(X, Bq, cell_depth=COV.mean(1), alpha=0.05)
    cols = np.flatnonzero(keep)
    Xk = X[:, cols]
    g, _ = em_genotype(AD[:, cols], COV[:, cols], kernel(Xk), mode='bb')
    Bk = impute_dropouts(Xk, (g > 0.7).astype(np.int8), k=30, thr=0.6, min_support=1)[0] > 0
    stage = pd.Series('lost: QC', index=names, dtype=object)
    calls = pd.Series(0, index=names)
    calls.iloc[cols] = Bk.sum(0)
    few = Bk.sum(0) < 2; high = Bk.mean(0) > 0.5
    stage.iloc[cols[few]] = 'lost: <2 calls after genotyping'
    stage.iloc[cols[~few & high]] = 'lost: prevalence > 0.5'
    kv = np.flatnonzero(~few & ~high)
    alive = maxcompat(Bk[:, kv].astype(np.int8))
    stage.iloc[cols[kv[~alive]]] = 'lost: four-gamete'
    stage.iloc[cols[kv[alive]]] = 'final'
    fc = cols[kv[alive]]
    Bs = Bk[:, kv[alive]].astype(np.int8)
    return dict(keep=keep, p_join=pj, p_excl=pe, stage=stage, calls=calls, final_cols=fc, B=Bs, Xs=character_af(Bs, X[:, fc]), Xk=Xk)


res = {k: run(B) for k, B in CARRIERS.items()}

# ---- per-variant table
rows = []
for j, v in enumerate(names):
    Bref = res['AD>0']['B'][:, list(res['AD>0']['final_cols']).index(j)] > 0 if j in res['AD>0']['final_cols'] else CARRIERS['AD>0'][:, j] > 0
    vc = pd.Series(gbc[Bref]).value_counts()
    modal = vc.index[0] if len(vc) else None
    r = dict(var=v, GT=v in GT, shipped=v in SH, carriers_guard=int(CARRIERS['guard'][:, j].sum()), carriers_AD0=int(CARRIERS['AD>0'][:, j].sum()),
             modal_clone_size=int(sizes.get(modal, 0)), modal_share=round(vc.iloc[0]/Bref.sum(), 2) if len(vc) else np.nan,
             frac_of_modal_clone=round(vc.iloc[0]/sizes.get(modal, 1), 2) if len(vc) else np.nan, n_clones_3plus=int((vc >= 3).sum()))
    for k in res:
        r[f'p_join[{k}]'] = res[k]['p_join'][j]; r[f'p_excl[{k}]'] = res[k]['p_excl'][j]
        r[f'stage[{k}]'] = res[k]['stage'].iloc[j]; r[f'calls[{k}]'] = int(res[k]['calls'].iloc[j])
    rows.append(r)
F = pd.DataFrame(rows)
F['flow'] = np.select([(F['stage[guard]'] == 'final') & (F['stage[AD>0]'] == 'final'),
                       (F['stage[guard]'] != 'final') & (F['stage[AD>0]'] == 'final'),
                       (F['stage[guard]'] == 'final') & (F['stage[AD>0]'] != 'final')],
                      ['final in both', 'AD>0 only', 'guard only'], 'final in neither')


def kind(r):
    if pd.isna(r.modal_share):
        return 'no carriers'
    if r.modal_share < 0.5:
        return 'spread over clones (share<0.5)'
    if r.modal_clone_size > 100:
        return 'within a large clone (>100): whole clone' if r.frac_of_modal_clone >= 0.5 else 'within a large clone (>100): SUBCLONAL'
    if r.modal_clone_size >= 10:
        return 'marks a mid clone (10-100)'
    return 'marks a tiny clone (<10)'


F['marks'] = F.apply(kind, axis=1)
F.to_csv(f'flow_{ds}.csv', index=False)

print(f'==== {ds}: {m} candidates, {len(GT & set(names))} GT-enriched among them')
print('\n-- final variant counts:', {k: int(len(res[k]['final_cols'])) for k in res})
print('\n-- flow categories x GT / shipped')
print(F.groupby('flow').agg(n=('var', 'size'), GT=('GT', 'sum'), shipped=('shipped', 'sum')).to_string())
print('\n-- where the guard pipeline loses the variants that AD>0 carries to the final set')
print(F[F.flow == 'AD>0 only']['stage[guard]'].value_counts().to_string())
print('\n-- carriers under each definition for AD>0-only variants (median)')
print(F[F.flow == 'AD>0 only'][['carriers_guard', 'carriers_AD0', 'calls[AD>0]', 'p_join[guard]', 'p_excl[guard]', 'p_join[AD>0]', 'p_excl[AD>0]']].median().round(3).to_string())
print('\n-- what the AD>0-only (and guard-only) variants mark')
print(pd.crosstab(F.marks, F.flow).to_string())
print('\n-- AD>0-only variants')
print(F[F.flow == 'AD>0 only'][['var', 'GT', 'shipped', 'carriers_guard', 'carriers_AD0', 'calls[AD>0]', 'modal_clone_size', 'modal_share', 'frac_of_modal_clone', 'n_clones_3plus', 'stage[guard]', 'marks']].sort_values('marks').to_string(index=False))

# ---- fragmentation of large clones
big = sizes.index[sizes > 100]
frag_rows, chars_rows = [], []
for k in res:
    B = res[k]['B']; fc = res[k]['final_cols']
    kc = B.sum(1) >= 1
    a = ad.AnnData(X=csr_matrix(res[k]['Xs'][kc]), obs=c.obs.iloc[np.flatnonzero(kc)][['GBC']].copy(), layers={'bin': csr_matrix(B[kc])})
    a.var_names = [f'{names[j]}#{i}' for i, j in enumerate(fc)]
    a.uns['genotyping'] = {'bin_method': 'x', 'binarization_kwargs': {}}
    for key in ('scLT_system', 'pp_method'): a.uns[key] = base.uns[key]
    mt.pp.compute_distances(a, precomputed=True, metric='weighted_jaccard', ncores=8, verbose=False)
    tree = mt.tl.build_tree(a, precomputed=True, solver='UPMGA')
    Bt = a.layers['bin'].toarray() > 0
    lab = evidence_cut(tree, Bt.astype(float), list(a.obs_names), min_in=0.25, ratio=3.0, self_min_in=0.85, ladder='descend')
    _, core = cell_membership(kernel(res[k]['Xk'], raw=True), B > 0, tau=0.0)
    lab[~(Bt & (core[kc] >= 0.25)).any(1)] = 'unassigned'
    ok = lab != 'unassigned'
    g_ok = a.obs['GBC'].astype(str)
    print(f'\n[{k}] cells {ok.sum()} labels {lab[ok].nunique()} ARI {mt.ut.custom_ARI(g_ok[ok], lab[ok]):.3f}', flush=True)
    for cl in big:
        in_cl = (g_ok == cl).values & ok.values
        l = lab[in_cl]
        vc = l.value_counts()
        frag_rows.append(dict(pipeline=k, clone=cl, size=int(sizes[cl]), assigned=int(in_cl.sum()), labels=int(vc.size),
                              top_share=round(vc.iloc[0]/vc.sum(), 2) if vc.size else np.nan))
        if vc.size >= 2:
            # characters whose presence best separates the clone's labels (Cramer's V)
            Bc = Bt[in_cl]; y = pd.Categorical(l.values)
            for i in range(Bc.shape[1]):
                x = Bc[:, i]
                if x.sum() < 3 or x.sum() > len(x) - 3:
                    continue
                tab = pd.crosstab(x, y.codes).values
                chi2 = ((tab - tab.sum(1, keepdims=True)*tab.sum(0, keepdims=True)/tab.sum())**2 /
                        np.maximum(tab.sum(1, keepdims=True)*tab.sum(0, keepdims=True)/tab.sum(), 1e-12)).sum()
                V = np.sqrt(chi2/(tab.sum()*max(min(tab.shape)-1, 1)))
                v = names[fc[i]]
                chars_rows.append(dict(pipeline=k, clone=cl, var=v, cramers_V=round(V, 2), prev_in_clone=round(x.mean(), 2),
                                       flow=F.set_index('var').loc[v, 'flow'], GT=v in GT, marks=F.set_index('var').loc[v, 'marks']))
FR = pd.DataFrame(frag_rows); CH = pd.DataFrame(chars_rows)
print('\n==== large clones (>100 cells): labels per clone')
print(FR.pivot_table(index=['clone', 'size'], columns='pipeline', values=['labels', 'top_share'], aggfunc='first').to_string())
if len(CH):
    top = CH[CH.cramers_V >= 0.5].sort_values(['pipeline', 'clone', 'cramers_V'], ascending=[True, True, False])
    print('\n==== characters splitting large clones (Cramer V >= 0.5 between character and label inside the clone)')
    print(top.to_string(index=False))
    print('\n-- splitting characters by flow category')
    print(pd.crosstab([top.pipeline], [top.flow]).to_string())
    print(pd.crosstab([top.pipeline], [top.marks]).to_string())
CH.to_csv(f'flow_splitters_{ds}.csv', index=False); FR.to_csv(f'flow_bigclones_{ds}.csv', index=False)
print('ALL DONE')
