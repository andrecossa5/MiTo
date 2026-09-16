"""
Sweep: QC carrier threshold (binomial alpha_cell) x final genotype threshold on kept variants x abstention.

Clones recovered are counted against GBC clones with a marker or weak marker (see pt_ladder.py), for
clones >= 10 and >= 5 cells; the ladder ceilings (L1a significant call, L1b any read) are reported too.

Usage (from dev/redesign/results): PYTHONPATH=..:../benchmarks python ../benchmarks/geno_sweep.py <dataset>
"""
import sys, warnings, logging
sys.setrecursionlimit(100000); warnings.filterwarnings('ignore'); logging.disable(logging.INFO)
import numpy as np, pandas as pd, scanpy as sc, mito as mt, anndata as ad
from scipy.sparse import csr_matrix
from mito.pp.filters import annotate_vars, filter_baseline, filter_MiTo
from grid import FK, kernel, maxcompat, character_af
from joincount import carrier_nonrandomness
from carriers import binomial_carriers
from cutter import evidence_cut
from compat import cell_membership
pd.set_option('display.width', 260); pd.set_option('display.max_rows', 200)
ds = sys.argv[1]
MFU = {'MDA_clones': 0.05, 'MDA_lung': 0.1, 'MDA_PT': 0.1}[ds]
ROOT = '/Users/cossa/Desktop/projects/MiTo/data_test/source_data/data/general/AFMs'
base = mt.pp.filter_cells(sc.read(f'{ROOT}/{ds}/afm_unfiltered.h5ad'), cell_filter='filter2')
N = base.shape[0]; cells = base.obs_names
gbc = base.obs['GBC'].astype(str).values; sizes = pd.Series(gbc).value_counts()
b = base.copy(); annotate_vars(b); b = filter_baseline(b)
vb = list(b.var_names)
ADb = b.layers['AD'].toarray(); COVb = b.layers['site_coverage'].toarray().astype(np.int64); Xb = b.X.toarray()
Bref = binomial_carriers(ADb, COVb, alpha_cell=0.001).astype(bool)

# ---- clone sets with a marker / weak marker (diagnostic reference), for min clone 10 and 5
clone_sets, ceilings = {}, {}
for mc in (10, 5):
    eligible = set(sizes.index[sizes >= mc]); mk = {}
    for j in range(len(vb)):
        car = Bref[:, j]
        if car.sum() < 2:
            continue
        vc = pd.Series(gbc[car]).value_counts(); cl = vc.index[0]
        if cl in eligible and vc.iloc[0]/car.sum() >= 0.6 and vc.iloc[0]/sizes[cl] >= 0.1:
            mk.setdefault(cl, []).append(j)
    clone_sets[mc] = list(mk)
    for kind, M in (('L1a call', Bref), ('L1b read', ADb >= 1)):
        lab = np.full(N, 'unassigned', dtype=object)
        for cl, js in mk.items():
            lab[(gbc == cl) & M[:, js].any(1)] = cl
        ceilings[(mc, kind)] = lab


def n_recovered(lab, mc):
    lab = np.asarray(lab, dtype=object); rec = 0
    for cl in clone_sets[mc]:
        x = lab[gbc == cl]; x = x[x != 'unassigned']
        if len(x) and len(x)/sizes[cl] >= .5:
            top = pd.Series(x).value_counts()
            rec += int(top.iloc[0]/len(x) >= .8 and (gbc[lab == top.index[0]] == cl).mean() >= .8)
    return rec


rows = []


def record(name, lab, **kw):
    lab = np.asarray(lab, dtype=object); ok = lab != 'unassigned'
    rows.append(dict(dataset=ds, pipeline=name, cells=int(ok.sum()), cells_pct=round(100*ok.mean(), 1), labels=int(pd.Series(lab[ok]).nunique()),
                     ARI=round(mt.ut.custom_ARI(pd.Series(gbc[ok]), pd.Series(lab[ok])), 3) if ok.sum() > 10 else np.nan,
                     clones_rec_min10=n_recovered(lab, 10), of_min10=len(clone_sets[10]),
                     clones_rec_min5=n_recovered(lab, 5), of_min5=len(clone_sets[5]), **kw))
    print({k_: rows[-1][k_] for k_ in ('pipeline', 'cells_pct', 'labels', 'ARI', 'clones_rec_min10', 'clones_rec_min5')} | kw, flush=True)


for (mc, kind), lab in ceilings.items():
    if mc == 10:
        record(f'CEILING {kind} (clones>=10 markers)', lab)
    else:
        record(f'CEILING {kind} (clones>=5 markers)', lab)

leg = mt.pp.filter_afm(base.copy(), filtering='MiTo', ncores=8)
tr = mt.tl.build_tree(leg, precomputed=True, solver='UPMGA')
m0 = mt.tl.MiToTreeAnnotator(tr); m0.clonal_inference(max_fraction_unassigned=MFU)
shp = pd.Series('unassigned', index=cells, dtype=object)
shp.loc[tr.cell_meta.index] = np.where(tr.cell_meta['MiTo clone'].isna(), 'unassigned', tr.cell_meta['MiTo clone'].astype(str))
record('SHIPPED', shp.values)

c = filter_MiTo(b.copy(), **FK)
jc = np.array([vb.index(v) for v in c.var_names])
Xc, ADc, COVc = Xb[:, jc], ADb[:, jc], COVb[:, jc]
GENO = {'binom 1e-3': lambda A, C: binomial_carriers(A, C, alpha_cell=1e-3), 'binom 1e-2': lambda A, C: binomial_carriers(A, C, alpha_cell=1e-2),
        'binom 5e-2': lambda A, C: binomial_carriers(A, C, alpha_cell=5e-2), 'AD>=1': lambda A, C: (A >= 1).astype(np.int8)}
for qa in (1e-2, 1e-3, 1e-4):
    Bq = binomial_carriers(ADc, COVc, alpha_cell=qa)
    keep, _, _ = carrier_nonrandomness(Xc, Bq, cell_depth=COVc.mean(1), alpha=0.05)
    cols = np.flatnonzero(keep)
    print(f'QC alpha_cell={qa}: kept {cols.size}', flush=True)
    Xk = Xc[:, cols]
    for gname, gfun in GENO.items():
        Bk = gfun(ADc[:, cols], COVc[:, cols]) > 0
        kv = np.flatnonzero((Bk.mean(0) <= 0.5) & (Bk.sum(0) >= 2))
        al = maxcompat(Bk[:, kv].astype(np.int8)); fc = kv[al]
        Bs = Bk[:, fc].astype(np.int8); kc = Bs.sum(1) >= 1
        if kc.sum() < 10 or fc.size < 2:
            continue
        a = ad.AnnData(X=csr_matrix(character_af(Bs, Xk[:, fc])[kc]), obs=pd.DataFrame(index=cells[kc]), layers={'bin': csr_matrix(Bs[kc])})
        a.var_names = [f'v{i}' for i in range(fc.size)]
        a.uns['genotyping'] = {'bin_method': 'x', 'binarization_kwargs': {}}
        for key in ('scLT_system', 'pp_method'): a.uns[key] = base.uns[key]
        mt.pp.compute_distances(a, precomputed=True, metric='weighted_jaccard', ncores=8, verbose=False)
        tree = mt.tl.build_tree(a, precomputed=True, solver='UPMGA')
        Bt = a.layers['bin'].toarray() > 0
        lab = evidence_cut(tree, Bt.astype(float), list(a.obs_names), min_in=0.25, ratio=3.0, self_min_in=0.85, ladder='descend')
        _, core = cell_membership(kernel(Xk, raw=True), Bs > 0, tau=0.0)
        lab_t = lab.mask(~(Bt & (core[kc] >= 0.25)).any(1), 'unassigned')
        for abst, l_ in (('none', lab), ('tau', lab_t)):
            out = np.full(N, 'unassigned', dtype=object); out[np.flatnonzero(kc)] = l_.values
            record(f'QC {qa:g} | geno {gname} | abstain {abst}', out, qc_alpha_cell=qa, genotypes=gname, abstain=abst,
                   qc_kept=int(cols.size), final_chars=int(fc.size), calls=int(Bs.sum()))
    pd.DataFrame(rows).to_csv(f'geno_sweep_{ds}.csv', index=False)
R = pd.DataFrame(rows)
print(R[['pipeline', 'qc_kept', 'final_chars', 'calls', 'cells_pct', 'labels', 'ARI', 'clones_rec_min10', 'of_min10', 'clones_rec_min5', 'of_min5']].to_string(index=False))
print('ALL DONE')
