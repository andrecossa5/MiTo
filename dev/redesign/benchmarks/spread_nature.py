"""
Are variants spread over several barcode clones noise, recurrent, or shared ancestral MT variants?

Per variant (candidates after filter_MiTo; categories from flow_<ds>.csv):
  carrier clones   GBC clones (>= MIN_CLONE cells) where >= CLONE_PREV of cells carry the variant (AD >= 1)
  sporadic share   carriers outside those clones
  read evidence    median AF / AD / base quality in carriers
  relatedness      are the carrier clones closer to each other on the OTHER variants than random sets
                   of clones of the same size? (cosine distance of clone-level prevalence profiles,
                   permutation null) -> shared ancestry if closer, recurrence if not

Usage (from dev/redesign/results): PYTHONPATH=..:../benchmarks python ../benchmarks/spread_nature.py MDA_PT
"""
import sys, warnings, logging
warnings.filterwarnings('ignore'); logging.disable(logging.INFO)
import numpy as np, pandas as pd, scanpy as sc, mito as mt
from scipy.spatial.distance import pdist, squareform
from mito.pp.filters import annotate_vars, filter_baseline, filter_MiTo
from grid import FK
pd.set_option('display.width', 250); pd.set_option('display.max_rows', 200)
ds = sys.argv[1] if len(sys.argv) > 1 else 'MDA_PT'
MIN_CLONE, CLONE_PREV, N_PERM = 10, 0.3, 2000
rng = np.random.default_rng(0)
ROOT = '/Users/cossa/Desktop/projects/MiTo/data_test/source_data/data/general/AFMs'
base = mt.pp.filter_cells(sc.read(f'{ROOT}/{ds}/afm_unfiltered.h5ad'), cell_filter='filter2')
F = pd.read_csv(f'flow_{ds}.csv').set_index('var')
b = base.copy(); annotate_vars(b); b = filter_baseline(b); c = filter_MiTo(b, **FK)
names = np.array(c.var_names)
X = c.X.toarray(); AD = c.layers['AD'].toarray(); Q = c.layers['qual'].toarray() if 'qual' in c.layers else None
car = AD >= 1
gbc = c.obs['GBC'].astype(str).values
vc = pd.Series(gbc).value_counts()
clones = vc.index[vc >= MIN_CLONE].tolist()
masks = {cl: gbc == cl for cl in clones}
P = np.vstack([car[masks[cl]].mean(0) for cl in clones])          # clones x variants: within-clone prevalence
print(f'{ds}: {len(clones)} clones with >= {MIN_CLONE} cells, holding {int(sum(m.sum() for m in masks.values()))} of {len(gbc)} cells')

rows = []
for j, v in enumerate(names):
    C = car[:, j]
    if C.sum() < 3:
        continue
    S = [i for i, cl in enumerate(clones) if P[i, j] >= CLONE_PREV]
    in_S = np.zeros(len(gbc), bool)
    for i in S:
        in_S |= masks[clones[i]]
    r = dict(var=v, carriers=int(C.sum()), n_carrier_clones=len(S),
             carrier_clone_sizes=';'.join(str(int(vc[clones[i]])) for i in S),
             within_clone_prev=round(float(np.median(P[S, j])), 2) if S else np.nan,
             sporadic_share=round(float((C & ~in_S).sum()/C.sum()), 2),
             median_AF=round(float(np.median(X[C, j])), 3), median_AD=float(np.median(AD[C, j])),
             median_qual=round(float(np.median(Q[C, j])), 1) if Q is not None else np.nan)
    if len(S) >= 2:
        others = np.delete(P, j, axis=1)
        D = squareform(pdist(others, metric='cosine'))
        iu = np.triu_indices(len(S), 1)
        obs = D[np.ix_(S, S)][iu].mean()
        null = np.array([D[np.ix_(R, R)][iu].mean() for R in (rng.choice(len(clones), len(S), replace=False) for _ in range(N_PERM))])
        r.update(carrier_clone_dist=round(float(obs), 3), random_clone_dist=round(float(null.mean()), 3),
                 p_related=round(float((1 + (null <= obs).sum())/(1 + N_PERM)), 4))
    rows.append(r)
T = pd.DataFrame(rows).set_index('var').join(F[['GT', 'shipped', 'flow', 'marks', 'stage[AD>0]']])
T['group'] = np.select(
    [T.marks.eq('spread over clones (share<0.5)') & T.flow.eq('AD>0 only'),
     T.marks.eq('spread over clones (share<0.5)') & T['stage[AD>0]'].eq('final'),
     T.marks.eq('spread over clones (share<0.5)') & T['stage[AD>0]'].eq('lost: QC'),
     T.marks.eq('within a large clone (>100): whole clone'),
     T.marks.eq('within a large clone (>100): SUBCLONAL'),
     T.marks.eq('marks a mid clone (10-100)')],
    ['spread, AD>0-only final (harmful set)', 'spread, final in both', 'spread, rejected by QC (noise reference)',
     'whole large clone marker', 'subclonal marker', 'mid clone marker'], 'other')
T.to_csv(f'spread_nature_{ds}.csv')

print('\n==== group medians')
print(T[T.group != 'other'].groupby('group')[['carriers', 'n_carrier_clones', 'within_clone_prev', 'sporadic_share', 'median_AF',
      'median_AD', 'median_qual', 'carrier_clone_dist', 'random_clone_dist', 'p_related']].median().round(3).to_string())
print(T[T.group != 'other'].groupby('group').size().to_string())


def verdict(r):
    if r.n_carrier_clones == 0 or r.sporadic_share >= 0.7:
        return 'noise-like (sporadic, no clone-level carriers)'
    if r.n_carrier_clones == 1:
        return 'one clone + sporadic calls'
    if r.p_related <= 0.05:
        return 'shared ancestry (carrier clones related)'
    return 'recurrence (carrier clones unrelated)'


S_ = T[T.marks.eq('spread over clones (share<0.5)') & T['stage[AD>0]'].eq('final')].copy()
S_['verdict'] = S_.apply(verdict, axis=1)
print('\n==== spread variants in the AD>0 final set')
print(S_[['GT', 'flow', 'carriers', 'n_carrier_clones', 'carrier_clone_sizes', 'within_clone_prev', 'sporadic_share', 'median_AF',
          'median_AD', 'median_qual', 'carrier_clone_dist', 'random_clone_dist', 'p_related', 'verdict']].sort_values('verdict').to_string())
print('\n', S_.groupby(['flow', 'verdict']).size().to_string())
R = T[T.group == 'spread, rejected by QC (noise reference)'].copy(); R['verdict'] = R.apply(verdict, axis=1)
print('\n-- verdicts for spread variants rejected by QC (reference):'); print(R.verdict.value_counts().to_string())
C_ = T[T.group.isin(['whole large clone marker', 'mid clone marker', 'subclonal marker'])].copy(); C_['verdict'] = C_.apply(verdict, axis=1)
print('\n-- verdicts for clone markers (reference):'); print(pd.crosstab(C_.group, C_.verdict).to_string())
print('ALL DONE')
