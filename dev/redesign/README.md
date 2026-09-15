# Preprocessing and clonal inference redesign (development)

Prototype of a new MiTo front end: variant QC, genotyping and clone calling. It is **not part of the
`mito` package** (only `src/mito` is packaged and tested) and is kept here so the work is versioned
while it is being debugged. Modules import each other by name: run scripts from this folder.

## Pipeline (`pipeline.py::run_pipeline`)

1. `filter_MiTo` with loose thresholds (`grid.FK`).
2. **Guarded flat-prior genotyping** (`geno2.em_genotype(..., guard=True)`): beta-binomial mixture
   initialised from cells with AD >= 1, prior capped at the observed detection rate, cells with
   AD = 0 never called. Without the guard the mixture degenerates when most cells have no reads
   and calls zero-read cells (61% of calls on MDA_PT).
3. **Variant QC** (`joincount.carrier_nonrandomness`), a local replacement for Moran's I with one
   threshold. Each variant is tested against the rest of the data, leaving itself out, with a
   depth-stratified permutation null; keep if `min(p_join, p_excl) <= alpha / 2`:
   - *join count*: carriers joined by more kNN edges than random cell sets (clones with several
     markers, recurrent variants);
   - *exclusivity*: carriers carry fewer other calls than random sets (clones marked by one variant).
   The QC must run on flat-prior calls: a graph prior imprints lineage structure on noise calls.
4. Guarded **graph-prior genotyping + dropout imputation** (`impute.impute_dropouts`) on kept variants.
5. Prevalence cap (0.5) -> optional split of recurrent variants (`refine.split_recurrent_graph`,
   off by default) -> four-gamete filter (`grid.maxcompat`).
6. UPGMA tree -> `cutter.evidence_cut(ladder='descend', one_sided=False)` -> membership abstention
   (`compat.cell_membership`, tau = 0.25).

Defaults: `alpha=0.05`, `one_sided=False`, `split=False`, `tau=0.25`.

```bash
python pipeline.py <afm_unfiltered.h5ad> GBC     # cell filter2, prints ARI/NMI against GBC
```

## Results

Real data: `data_test/source_data/data/general/AFMs` (git-ignored). Shipped = `filter_afm('MiTo')`
+ UPGMA + `MiToTreeAnnotator.clonal_inference` (`max_fraction_unassigned=0.1` on MDA_lung and
MDA_PT, where the default finds no solution). "Shared cells" = cells both pipelines assign.

| Dataset (cells / GBC clones) | Pipeline | Cells | Labels | ARI | ARI on shared cells (dev / shipped) |
|---|---|---|---|---|---|
| MDA_clones (375 / 8) | shipped | 74% | 9 | 0.956 | |
| | dev | 73% | 6 | 0.930 | 0.967 / 0.962 |
| MDA_lung (1537 / 28) | shipped | 76.5% | 11 | 0.953 | |
| | dev | 81% | 6 | 0.945 | 0.962 / 0.955 |
| MDA_PT (2757 / 216) | shipped | 57% | 33 | 0.755 | |
| | dev | 56% | 27 | 0.773 | 0.874 / 0.847 |

Simulations (24 datasets: 5/10/30/50 clones x polytomy/depth-3 x 3 seeds, `grid.SIM`): dev 0.770,
0.805 with `one_sided=True`; shipped 0.614.

Tables: `results/fix_real_*.csv` (real data), `results/qc2guard_sim_*.csv` and `results/adqc_sim_*.csv`
(simulations), `results/pt_ablation_MDA_PT.csv` (MDA_PT ablation).

## Open issues

- **Few variants kept on real data**: 19 / 13 / 32 on MDA_clones / lung / PT vs 28 / 57 / 128
  shipped. With a flat prior the guard caps the prior at the global detection rate, so clone
  markers get very few carriers (median 2 on MDA_lung and MDA_PT) and lose QC power
  (`results/qc_carriers_*.csv`).
- Raw read carriers (AD >= 1 or AD >= 2) for the QC restore the variant count but degrade ARI on
  MDA_PT (0.61-0.63) and **produce a degenerate tree on MDA_lung (ARI 0, 170 / 1 labels)**,
  most likely a bug downstream (`results/adqc_real_*.csv`). Under investigation.
- `evidence_cut` has no merge step: with `one_sided=True` it splits MT subclones inside large
  barcode clones.
- Clones of fewer than ~10 cells are not recovered by any pipeline on MDA_PT.
