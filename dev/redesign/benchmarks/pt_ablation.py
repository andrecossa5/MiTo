"""MDA_PT: which downstream step mixes clones? variant set x genotyping x split, evaluated with both callers."""
import sys, warnings, logging; sys.setrecursionlimit(100000); warnings.filterwarnings('ignore'); logging.disable(logging.INFO)
import numpy as np, pandas as pd, scanpy as sc, mito as mt, anndata as ad
from scipy.sparse import csr_matrix
from sklearn.metrics import homogeneity_completeness_v_measure as hcv
from mito.pp.filters import annotate_vars, filter_baseline, filter_MiTo
from geno2 import em_genotype
from grid import FK, kernel, maxcompat
from impute import impute_dropouts
from joincount import carrier_nonrandomness
from refine import split_recurrent_graph
from cutter import evidence_cut
pd.set_option('display.width', 250)
ds = sys.argv[1] if len(sys.argv) > 1 else 'MDA_PT'
ROOT = '/Users/cossa/Desktop/projects/MiTo/data_test/source_data/data/general/AFMs'
base = mt.pp.filter_cells(sc.read(f'{ROOT}/{ds}/afm_unfiltered.h5ad'), cell_filter='filter2'); N = base.shape[0]
gbc = base.obs['GBC'].astype(str); sizes = gbc.value_counts(); big = sizes.index[sizes > 100]
leg = mt.pp.filter_afm(base.copy(), filtering='MiTo', ncores=8)
b = base.copy(); annotate_vars(b); b = filter_baseline(b); c = filter_MiTo(b, **FK)
X = c.X.toarray(); AD = c.layers['AD'].toarray(); COV = c.layers['site_coverage'].toarray().astype(np.int64); n, m = X.shape
g_flat, _ = em_genotype(AD, COV, np.full((n, n), 1/n), mode='bb'); Bf = (g_flat > 0.7).astype(np.int8)
keep, _, _ = carrier_nonrandomness(X, Bf, cell_depth=COV.mean(1), alpha=0.05)
qc_cols = np.flatnonzero(keep)
ship_cols = np.array([i for i, v in enumerate(c.var_names) if v in set(leg.var_names)])
print(f'QC2 keeps {qc_cols.size}; shipped variants present among candidates: {ship_cols.size}/{leg.shape[1]}', flush=True)

def build(cols, geno, split, cap=True, compat=True):
    Xk = X[:, cols]
    if geno == 'flat':
        Bk = Bf[:, cols]
    else:
        g, _ = em_genotype(AD[:, cols], COV[:, cols], kernel(Xk), mode='bb'); Bk = (g > 0.7).astype(np.int8)
        if geno == 'graph+imp':
            Bk = impute_dropouts(Xk, Bk, k=30, thr=0.6, min_support=1)[0]
    kv = np.flatnonzero(Bk.astype(bool).mean(0) <= 0.5) if cap else np.arange(Bk.shape[1])
    Bp, Xp = Bk[:, kv], Xk[:, kv]
    origin = np.arange(Bp.shape[1])
    if split:
        Bp, origin, _ = split_recurrent_graph(Xp, Bp)
    if compat:
        al = maxcompat(Bp); Bp, origin = Bp[:, al], origin[al]
    return Bp, np.where(Bp > 0, Xp[:, origin], 0.0)

def evaluate(name, B, Xc, rows):
    kc = (B > 0).sum(1) >= 1
    a = ad.AnnData(X=csr_matrix(Xc[kc]), obs=c.obs.iloc[np.flatnonzero(kc)][['GBC']].copy(), layers={'bin': csr_matrix((B[kc] > 0).astype(np.int8))})
    a.var_names = [f'c{i}' for i in range(B.shape[1])]
    a.uns['genotyping'] = {'bin_method': 'x', 'binarization_kwargs': {}}
    for key in ('scLT_system', 'pp_method'): a.uns[key] = base.uns[key]
    mt.pp.compute_distances(a, precomputed=True, metric='weighted_jaccard', ncores=8, verbose=False)
    tree = mt.tl.build_tree(a, precomputed=True, solver='UPMGA')
    Bk = a.layers['bin'].toarray() > 0; gt = a.obs['GBC'].astype(str)
    labs = {}
    try:
        m0 = mt.tl.MiToTreeAnnotator(tree); m0.clonal_inference(max_fraction_unassigned=0.1)
        cm = tree.cell_meta; labs['annotator'] = pd.Series(np.where(cm['MiTo clone'].isna(), 'unassigned', cm['MiTo clone'].astype(str)), index=cm.index).loc[a.obs_names]
    except Exception as e:
        print(name, 'annotator failed', str(e).strip()[:60], flush=True)
    labs['evid_cut'] = evidence_cut(tree, Bk.astype(float), list(a.obs_names), min_in=0.25, ratio=3.0, self_min_in=0.85, ladder='descend')
    for caller, lab in labs.items():
        ok = (lab != 'unassigned').values
        h, cp, _ = hcv(gt[ok], lab[ok])
        rec, nl = [], []
        for cl in big:
            l = lab[gt == cl]; l = l[l != 'unassigned']
            if len(l) == 0: rec.append(False); nl.append(0); continue
            top = l.value_counts(); best = top.index[0]
            pur = (gt[lab == best] == cl).mean()
            rec.append(len(l)/sizes[cl] >= 0.5 and top.iloc[0]/len(l) >= 0.8 and pur >= 0.8); nl.append(l.nunique())
        rows.append(dict(setting=name, caller=caller, chars=B.shape[1], cells_pct=round(100*ok.sum()/N, 1), labels=int(lab[ok].nunique()),
                         ARI=round(mt.ut.custom_ARI(gt[ok], lab[ok]), 3), homog=round(h, 3), compl=round(cp, 3),
                         big_recovered=f'{sum(rec)}/{len(big)}', big_labels_per_clone=round(np.mean(nl), 1)))
    print(pd.DataFrame(rows[-len(labs):]).to_string(index=False, header=len(rows) <= 2), flush=True)

rows = []
settings = [
    ('SHIPPED front as shipped (MiTo calls)', None),
    ('shipped vars | flat calls | no split', (ship_cols, 'flat', False)),
    ('shipped vars | graph+imp | no split', (ship_cols, 'graph+imp', False)),
    ('shipped vars | graph+imp | split', (ship_cols, 'graph+imp', True)),
    ('QC2 vars | flat calls | no split', (qc_cols, 'flat', False)),
    ('QC2 vars | graph, no imp | no split', (qc_cols, 'graph', False)),
    ('QC2 vars | graph+imp | no split', (qc_cols, 'graph+imp', False)),
    ('QC2 vars | graph+imp | split (current)', (qc_cols, 'graph+imp', True)),
]
for name, spec in settings:
    if spec is None:
        evaluate(name, leg.layers['bin'].toarray()[np.searchsorted(leg.obs_names, c.obs_names)] if False else None, None, rows) if False else None
        a_leg = leg
        kc_idx = [list(c.obs_names).index(x) for x in leg.obs_names if x in set(c.obs_names)]
        B = np.zeros((n, leg.shape[1]), np.int8); Xc = np.zeros((n, leg.shape[1]))
        pos = {x: i for i, x in enumerate(c.obs_names)}
        li = [pos[x] for x in leg.obs_names]
        B[li] = leg.layers['bin'].toarray(); Xc[li] = leg.X.toarray()
        evaluate(name, B, Xc, rows)
    else:
        B, Xc = build(*spec)
        evaluate(name, B, Xc, rows)
    pd.DataFrame(rows).to_csv(f'pt_ablation_{ds}.csv', index=False)
print('\n'); print(pd.DataFrame(rows).to_string(index=False))
print('ALL DONE')
