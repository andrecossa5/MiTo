"""Fragments of several samples -> QCed, merged snapATAC2 AnnData (cluster-ready: no hardcoded paths).

Usage:
  atac_preprocess.py --gsm GSM9096489 ... GSM9096495 --fragments-dir <dir> --out-dir <out>
                     [--whitelist-template '/path/{gsm}/depth.txt.gz'] [--tmp-dir $TMPDIR] [--n-jobs 8]
                     [--min-counts 1000] [--min-tsse 20] [--doublet-prob 0.5] [--bin-size 5000] [--n-features 250000]
                     [--resume] [--remove-intermediate]
A fragment file is found as <fragments-dir>/<GSM>_*_fragments.tsv.gz (GEO naming, e.g. GSM9096491_CKDL210028090_fragments.tsv.gz).

Per sample (one after the other; peak memory ~6-9 GB, set by the biggest sample, not by the number of samples):
  1. import      fragments -> AnnData. --whitelist-template (optional; txt/csv/gz, barcode = first column) restricts to the CellRanger
                 cells, e.g. the mgatk depth.txt.gz, so ATAC and MT cells match. Without it: every barcode with >= 200 fragments.
                 The files are position-sorted, so snapATAC2 sorts them internally (sorted_by_barcode=False).
  2. metrics     TSS enrichment, unique fragments, duplicate fraction, fragment-size distribution, for ALL imported barcodes.
  3. QC filter   unique fragments >= --min-counts and TSSE >= --min-tsse.
  4. doublets    5-kb tile matrix, feature selection, scrublet on the QC-passing cells; cells with probability >= --doublet-prob dropped.
  5. rename      cells become {barcode}_{GSM} (the MiTo AFM naming).
Then, across samples:
  6. merge       snap.concat streams the samples into ONE AnnData (tile matrix + fragments of the surviving cells, obs: sample + QC),
                 then features are re-selected on the pooled cells (var['selected']).

Outputs in <out-dir>/:
  merged.h5ad          the object for local analysis: spectral / kNN / Harmony / Leiden / peaks / gene scores start from here
  metrics_all.csv      QC metrics and fate (pass_qc, is_doublet, kept) of ALL imported barcodes of all samples
  summary.json         cell counts at each step, parameters, timings, peak memory per sample
  qc.pdf               TSSE vs fragments per sample, fragment-size profiles, cells kept per step
  samples/<GSM>/       per-sample intermediates (<GSM>.h5ad, metrics, json); --remove-intermediate deletes them after the merge"""
import argparse, glob, gzip, json, os, resource, shutil, sys, tempfile, time
import numpy as np, pandas as pd, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import snapatac2 as snap

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument('--gsm', nargs='+', required=True)
ap.add_argument('--fragments-dir', required=True)
ap.add_argument('--out-dir', required=True)
ap.add_argument('--whitelist-template', default=None, help="path with a {gsm} placeholder, e.g. '/data/mgatk/{gsm}/depth.txt.gz'")
ap.add_argument('--tmp-dir', default=None)
ap.add_argument('--n-jobs', type=int, default=8)
ap.add_argument('--min-frags-import', type=int, default=200)
ap.add_argument('--min-counts', type=int, default=1000)
ap.add_argument('--min-tsse', type=float, default=20.)
ap.add_argument('--doublet-prob', type=float, default=0.5)
ap.add_argument('--bin-size', type=int, default=5000)
ap.add_argument('--n-features', type=int, default=250000)
ap.add_argument('--resume', action='store_true', help='skip samples whose per-sample outputs already exist')
ap.add_argument('--remove-intermediate', action='store_true')
a = ap.parse_args()
OUT = a.out_dir
os.makedirs(f'{OUT}/samples', exist_ok=True)
T0 = time.time()


def peak_gb():       # ru_maxrss: bytes on macOS, kB on Linux
    r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return round(r / (1e9 if sys.platform == 'darwin' else 1e6), 2)


def to_pandas(adata):
    obs = adata.obs[:]
    obs = obs.to_pandas() if hasattr(obs, 'to_pandas') else pd.DataFrame(obs)
    obs.index = list(adata.obs_names)
    return obs


def process_sample(gsm):
    d = f'{OUT}/samples/{gsm}'
    os.makedirs(d, exist_ok=True)
    if a.resume and os.path.exists(f'{d}/{gsm}.json'):
        print(f'[{gsm}] resume: found {d}/{gsm}.json', flush=True)
        return json.load(open(f'{d}/{gsm}.json'))
    hits = glob.glob(f'{a.fragments_dir}/{gsm}_*_fragments.tsv.gz')
    if len(hits) != 1:
        sys.exit(f'expected one fragment file for {gsm} in {a.fragments_dir}, found {hits}')
    frag, tmp, steps, t = hits[0], tempfile.mkdtemp(prefix=f'{gsm}_', dir=a.tmp_dir), {}, time.time()

    def done(name):
        nonlocal t
        steps[name] = dict(seconds=round(time.time() - t), peak_gb=peak_gb()); t = time.time()
        print(f'[{gsm}] {name}: {steps[name]}', flush=True)

    whitelist = None
    if a.whitelist_template:
        wl = a.whitelist_template.format(gsm=gsm)
        with (gzip.open if wl.endswith('.gz') else open)(wl, 'rt') as h:
            whitelist = [l.replace(',', '\t').split('\t')[0].strip() for l in h if l.strip()]
    print(f'[{gsm}] fragments {os.path.getsize(frag) / 1e9:.2f} GB; whitelist: {len(whitelist) if whitelist else "none"}', flush=True)

    # 1-2. import, metrics for all imported barcodes
    adata = snap.pp.import_fragments(frag, snap.genome.hg38, file=f'{tmp}/raw.h5ad', sorted_by_barcode=False, whitelist=whitelist,
                                     min_num_fragments=a.min_frags_import, tempdir=tmp, n_jobs=a.n_jobs)
    n_imported = adata.n_obs; done('import')
    snap.metrics.tsse(adata, snap.genome.hg38)
    snap.metrics.frag_size_distr(adata)
    m = to_pandas(adata)[['n_fragment', 'tsse', 'frac_dup']]
    pd.Series(np.asarray(adata.uns['frag_size_distr']), name='n_fragments').rename_axis('size_bp').to_csv(f'{d}/{gsm}_frag_size.csv')
    m['pass_qc'] = ((m['n_fragment'] >= a.min_counts) & (m['tsse'] >= a.min_tsse)).to_numpy()
    done('metrics')

    # 3. QC filter
    final = f'{d}/{gsm}.h5ad'
    if os.path.exists(final): os.remove(final)
    adata.subset(obs_indices=np.flatnonzero(m['pass_qc'].to_numpy()), out=final, inplace=False)
    adata.close(); os.remove(f'{tmp}/raw.h5ad')
    adata = snap.read(final); n_pass = adata.n_obs; done('qc_filter')

    # 4. doublets (scrublet on the QC-passing cells only)
    snap.pp.add_tile_matrix(adata, bin_size=a.bin_size)
    snap.pp.select_features(adata, n_features=a.n_features)
    snap.pp.scrublet(adata)
    o = to_pandas(adata)
    m['doublet_score'] = o['doublet_score'].reindex(m.index); m['doublet_probability'] = o['doublet_probability'].reindex(m.index)
    m['is_doublet'] = m['doublet_probability'] >= a.doublet_prob
    snap.pp.filter_doublets(adata, probability_threshold=a.doublet_prob)
    done('doublets')

    # 5. rename and annotate; metrics of all barcodes
    adata.obs['sample'] = [gsm] * adata.n_obs
    adata.obs_names = [f'{b}_{gsm}' for b in adata.obs_names]
    kept = set(adata.obs_names)
    m.index = [f'{b}_{gsm}' for b in m.index]
    m['sample'] = gsm
    m['kept'] = m.index.isin(kept)
    m.rename_axis('cell').to_csv(f'{d}/{gsm}_metrics.csv')
    n_final = adata.n_obs
    adata.close(); shutil.rmtree(tmp, ignore_errors=True)
    S = dict(gsm=gsm, fragments_gb=round(os.path.getsize(frag) / 1e9, 2), n_whitelist=len(whitelist) if whitelist else None,
             n_imported=n_imported, n_pass_qc=n_pass, n_doublets=n_pass - n_final, n_kept=n_final, steps=steps,
             seconds=round(sum(s['seconds'] for s in steps.values())), peak_gb=peak_gb())
    json.dump(S, open(f'{d}/{gsm}.json', 'w'), indent=1)
    print(f'[{gsm}]', {k: v for k, v in S.items() if k != 'steps'}, flush=True)
    return S


summ = [process_sample(g) for g in a.gsm]

# 6. merge
t = time.time()
print('merging', a.gsm, flush=True)
merged_f = f'{OUT}/merged.h5ad'
if os.path.exists(merged_f): os.remove(merged_f)
parts = [snap.read(f'{OUT}/samples/{g}/{g}.h5ad') for g in a.gsm]
merged = snap.concat(parts, join='inner', file=merged_f)     # obs['sample'] already set per sample
snap.pp.select_features(merged, n_features=a.n_features)        # on the pooled cells
n_cells, n_vars = merged.n_obs, merged.n_vars
obs_cols = list(to_pandas(merged).columns)
has_fragments = 'fragment_paired' in list(merged.obsm.keys())
merged.close()
for p in parts: p.close()

M = pd.concat([pd.read_csv(f'{OUT}/samples/{g}/{g}_metrics.csv', index_col=0) for g in a.gsm])
M.to_csv(f'{OUT}/metrics_all.csv')
fs = {g: pd.read_csv(f'{OUT}/samples/{g}/{g}_frag_size.csv', index_col=0)['n_fragments'] for g in a.gsm}

# QC figure: one row of TSSE-vs-fragments hexbins (all imported barcodes), fragment sizes, cells kept per step
n = len(a.gsm)
fig, axs = plt.subplots(2, max(n, 3), figsize=(3.2 * max(n, 3), 6.6), gridspec_kw=dict(wspace=.45, hspace=.5), squeeze=False)
for ax in axs.ravel(): ax.set_visible(False)
for i, g in enumerate(a.gsm):
    ax, mm = axs[0, i], M[M['sample'] == g]; ax.set_visible(True)
    ax.hexbin(np.log10(mm['n_fragment']), mm['tsse'], gridsize=50, bins='log', cmap='viridis', mincnt=1)
    ax.axvline(np.log10(a.min_counts), c='r', lw=.8, ls='--'); ax.axhline(a.min_tsse, c='r', lw=.8, ls='--')
    ax.set(xlabel='log10 fragments', ylabel='TSSE', title=f'{g[-7:]}: {int(mm.kept.sum()):,}/{len(mm):,} kept')
ax = axs[1, 0]; ax.set_visible(True)
for g in a.gsm: ax.plot(fs[g].index, fs[g] / fs[g].sum(), lw=1, label=g[-7:])
ax.set(xlim=(0, 800), xlabel='Fragment size (bp)', ylabel='Fraction', title='Fragment sizes'); ax.legend(fontsize=6, frameon=False)
ax = axs[1, 1]; ax.set_visible(True)
S_ = pd.DataFrame(summ).set_index('gsm')[['n_imported', 'n_pass_qc', 'n_kept']]
S_.index = [g[-7:] for g in S_.index]; S_.plot.bar(ax=ax, width=.8, legend=True, rot=45); ax.legend(fontsize=6, frameon=False)
ax.set(ylabel='Cells', title='Cells per step')
fig.savefig(f'{OUT}/qc.pdf', bbox_inches='tight'); plt.close(fig)

if a.remove_intermediate:
    shutil.rmtree(f'{OUT}/samples', ignore_errors=True)
J = dict(samples=summ, merged=dict(n_cells=n_cells, n_bins=n_vars, obs_columns=obs_cols, has_fragments=has_fragments,
         h5ad_gb=round(os.path.getsize(merged_f) / 1e9, 2), merge_seconds=round(time.time() - t)),
         params=vars(a), total_seconds=round(time.time() - T0), peak_gb=peak_gb())
json.dump(J, open(f'{OUT}/summary.json', 'w'), indent=1, default=str)
print(json.dumps(J['merged']), 'peak GB', peak_gb(), 'total s', J['total_seconds'], flush=True)
