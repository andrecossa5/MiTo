"""Join count (LOO / fixed graph) vs shipped-style Moran's I on the three target cases."""
import warnings, logging; warnings.filterwarnings('ignore'); logging.disable(logging.INFO)
import numpy as np, pandas as pd, scanpy as sc, mito as mt
from mito.pp.filters import annotate_vars, filter_baseline, filter_MiTo
from mito.pp.distances import weighted_jaccard
from geno2 import em_genotype
from grid import kernel, FK, SIM
from joincount import join_count_test, morans_i_test
pd.set_option('display.width', 250)
ALPHA = 0.01


def flat_calls(c):
    X = c.X.toarray(); AD = c.layers['AD'].toarray(); COV = c.layers['site_coverage'].toarray().astype(np.int64)
    n = X.shape[0]
    g, _ = em_genotype(AD, COV, np.full((n, n), 1/n), mode='bb')
    return X, (g > 0.7).astype(np.int8)


def tests(X, B):
    out = {}
    for graph, eps in [('loo', None), ('eps', 0.05), ('eps', 0.2), ('eps', 0.5), ('fixed', None)]:
        _, _, p = join_count_test(X, B, graph=graph, eps=eps if eps else 0.1)
        out[f'JC {graph}' + (f' {eps}' if eps else '')] = p
    Bb = B > 0
    w = np.nanmedian(np.where(Bb, X, np.nan), axis=0); w = np.nan_to_num(w, nan=0.0)
    D = weighted_jaccard(Bb.astype(int), w)
    out["Moran"] = morans_i_test(D, B)[1]
    return out


rows = []
for nk in [5, 10, 30, 50]:
    for topo in ['polytomy', 'depth3']:
        for seed in [0, 1]:
            kw = dict(n_cells=1000, n_clones=nk, frac_double_variants=0.3, frac_noisy_variants=0.2,
                      min_max_ratio_clones=0.2, random_seed=seed, **SIM)
            kw.update(dict(n_root_clones=nk, max_depth=1) if topo == 'polytomy'
                      else dict(n_root_clones=max(2, nk//3), max_depth=3))
            s = mt.ut.simulate_afm(**kw); b = s.copy(); annotate_vars(b); b = filter_baseline(b)
            c = filter_MiTo(b, **FK)
            V = s.var.loc[c.var_names]
            per_clone = V[~V.is_noise].groupby('clone_of_origin').size()
            G = s[c.obs_names, c.var_names].layers['genotype']
            G = (G.toarray() if hasattr(G, 'toarray') else np.asarray(G)) > 0
            X, B = flat_calls(c)
            P = tests(X, B)
            for j, v in enumerate(c.var_names):
                pos = B[:, j] > 0
                fp = int((pos & ~G[:, j]).sum())
                noise = bool(V.loc[v, 'is_noise'])
                sole = (not noise) and per_clone[V.loc[v, 'clone_of_origin']] == 1
                if noise:
                    case = '1 prevalent scattered (noise >=20)' if pos.sum() >= 20 else 'noise <20 carriers'
                elif 5 <= pos.sum() <= 10 and fp == 0:
                    case = '2 rare clean clone (5-10, no outside)'
                elif pos.sum() >= 30 and fp >= 3:
                    case = '3 prevalent clone + outside noise'
                else:
                    case = 'other clonal'
                rows.append(dict(cfg=f'{nk}{topo[:4]}', seed=seed, var=v, case=case,
                                 marker='sole' if sole else ('noise' if noise else '>1'),
                                 carriers=int(pos.sum()), outside=fp, **{k_: P[k_][j] for k_ in P}))
            print(f'done {nk}{topo}/{seed}', flush=True)

S = pd.DataFrame(rows); S.to_csv('joincount_eps_sim.csv', index=False)
cols = ['JC loo', 'JC eps 0.05', 'JC eps 0.2', 'JC eps 0.5', 'JC fixed', 'Moran']
print(f'\n===== SIMULATIONS: share KEPT at p <= {ALPHA}')
print(S.groupby(['case', 'marker'])[cols].agg(lambda p: (p <= ALPHA).mean()).round(2)
      .join(S.groupby(['case', 'marker']).size().rename('n')).to_string())

# MDA
D_ = '/Users/cossa/Desktop/projects/MiTo/data_test/afm_unfiltered.h5ad'
base = mt.pp.filter_cells(sc.read(D_), cell_filter='filter2')
GT = set(mt.pp.filter_afm(base.copy(), filtering='GT_enriched', lineage_column='GBC', ncores=4).var_names)
b = base.copy(); annotate_vars(b); b = filter_baseline(b); c = filter_MiTo(b, **FK)
gbc = c.obs['GBC'].astype(str).values
X, B = flat_calls(c)
P = tests(X, B)
mrows = []
for j, v in enumerate(c.var_names):
    pos = B[:, j] > 0
    if pos.sum() == 0:
        continue
    vc = pd.Series(gbc[pos]).value_counts()
    share = vc.iloc[0]/pos.sum()
    outside = int(pos.sum() - vc.iloc[0])
    if v in GT and pos.sum() <= 12 and outside <= 1:
        case = '2 rare clean clone'
    elif v in GT and pos.sum() >= 30 and outside >= 3:
        case = '3 prevalent clone + outside'
    elif v in GT:
        case = 'other GT-enriched'
    elif share < 0.6 and pos.sum() >= 20:
        case = '1 prevalent scattered'
    elif share < 0.6:
        case = 'rare scattered (<20)'
    else:
        case = 'non-GT clone-specific'
    mrows.append(dict(var=v, case=case, carriers=int(pos.sum()), top_clone_share=round(share, 2), outside=outside,
                      **{k_: round(P[k_][j], 3) for k_ in P}))
M = pd.DataFrame(mrows).sort_values(['case', 'carriers'])
M.to_csv('joincount_eps_mda.csv', index=False)
print(f'\n===== MDA (flat-prior calls), p-values; kept if <= {ALPHA}')
print(M[M.case.isin(['2 rare clean clone', '3 prevalent clone + outside', '1 prevalent scattered'])].to_string(index=False))
print(M.groupby('case')[cols].agg(lambda p: (p <= ALPHA).mean()).round(2).join(M.groupby('case').size().rename('n')).to_string())
