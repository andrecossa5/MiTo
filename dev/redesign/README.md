# Preprocessing and clonal inference redesign (development)

Prototype front end for MiTo: variant QC, genotyping and clone calling. **Not part of the `mito`
package** (only `src/mito` is packaged and tested); kept here so the work is versioned while it is
still being changed. Modules import each other by file name, so run scripts from this folder
(`PYTHONPATH=.:benchmarks` when running from `results/`). Data paths point at `data_test/`, which is
git-ignored; `results/*.log` is ignored too.

## The published pipeline, for reference

`mt.pp.filter_afm(afm, filtering='MiTo')` -> `mt.tl.build_tree` -> `MiToTreeAnnotator.clonal_inference()`

| Stage | What it does | Key parameters |
|---|---|---|
| cell filter | median target coverage, covered sites | 25, 0.75 (`filter2`) |
| baseline | site coverage, quality, positive cells, genes only | 5, 30, 2 |
| `filter_MiTo` | negative fraction, positive cells, confident detection, AD/DP in positives | 0.2, 5, 0.02 in >=2 cells, 1.25, 25 |
| dbSNP / REDIdb | drop common SNPs and RNA edits | |
| genotyping | binomial mixture where prevalence >= 10%, else AD threshold | `t_prob` 0.7, `min_AD` 2, `min_cell_prevalence` 0.1 |
| max-AD | >=1 cell with AD >= 2 | `max_AD_counts` 2 |
| distances | weighted Jaccard, weights = median AF of positives | |
| Moran's I | dense weights `1 - distance` over all pairs, permutation p | `moran_I_pvalue` 0.01 |
| tree | UPGMA | |
| clone calling | grid over similarity percentile / mutation enrichment / merging, scored 0.3 silhouette + 0.4 (-n clones) + 0.3 similarity | `max_fraction_unassigned` 0.05 |

## The development pipeline

Stages 1-2 as published. `pipeline.py::run_pipeline` implements stages 3-12 with the guarded
genotyper and imputation at 0.6; the best configuration found in the last runs (binomial genotyping,
SNR filter, imputation 0.8) is **not yet wired into that entry point** -- see `benchmarks/snr_bench.py`
and `benchmarks/coverage_sweep2.py`.

| # | Stage | What it does | Parameters |
|---|---|---|---|
| 3 | `filter_MiTo(**FK)` | loosened candidate filter; the QC decides, not fixed read cutoffs | DP in positives 10, negative fraction 0.5, confident detection 0.01 |
| 4 | `carriers.binomial_carriers` | per-cell one-sided binomial test against the variant's own background (pooled alt/coverage over non-carriers, +1 pseudocount, re-estimated iteratively) | `alpha_cell` 1e-4 (QC input) |
| 5 | `joincount.carrier_nonrandomness` | Moran's I replacement. Per variant, kNN graph built WITHOUT it; **join count** (carrier-carrier edges) or **exclusivity** (carriers carry fewer other calls); nulls = random cell sets stratified by coverage; keep if min p <= alpha/2 | `alpha` 0.05, `k` 15, 999 permutations, 5 strata |
| 6 | `binomial_carriers` | final genotypes on kept variants | `alpha_cell` 1e-3 |
| 7 | `carriers.signal_to_background` | median AF of cells with >=1 read / background; drops variants whose carriers do not stand above their own error rate | `min_ratio` 10 |
| 8 | `impute.impute_dropouts` | kNN (leave-one-out, Gaussian weights) dropout imputation; needs >= 1 other call in the cell | `k` 30, `thr` 0.8, `min_support` 1 |
| 9 | character filters | prevalence <= 0.5, >= 2 calls, greedy four-gamete | cap 0.5, `min_calls` 2, `min_gamete` 10 |
| 10 | distances + tree | weighted Jaccard + UPGMA; `grid.character_af` gives imputed calls the character's median positive AF | |
| 11 | `cutter.evidence_cut` | split a clade only when >= 2 children carry markers specific against their SIBLINGS; `ladder='descend'` passes through nodes with one large child | `min_in` 0.25, `ratio` 3.0, `self_min_in` 0.85, `alpha` 0.01, `min_cells` 5, `min_supported` 2, `one_sided` False |
| 12 | `compat.cell_membership` | abstention: keep a cell only if one of its calls is corroborated by its neighbourhood | `tau` 0.25 |
| 13 | `rescue.pooled_rescue` (optional) | unassigned cells joined to a label when their reads over that label's markers beat background and no other label competes | `alpha` 0.01, markers >= 0.5 in / <= 0.1 out |

### Why the published stages were replaced

* **Moran's I** weights every cell pair and includes the variant under test, so variants confirm
  themselves: 96% of scattered noise passed. Excluding the variant and truncating to kNN rejects 98%
  of simulated noise while keeping 98% of multi-marker and 93% of sole-marker clonal variants.
* **Mixture genotyping** is unidentifiable when most cells have AD = 0: the mixing weight drifts and
  the prior alone calls cells with **zero alt reads** (61% of calls on MDA_PT, 50% on MDA_clones).
* **A graph prior on genotyping** (our own first attempt) spreads noise calls along lineages -- noise
  call precision 0.96 -> 0.50 -- which blinds any graph-based QC downstream.
* **No post-genotyping noise control**: a broad heteroplasmic variant gets a random-looking subset of
  cells called, and the greedy four-gamete filter then deletes the largest clone's real markers
  (MDA_clones: that clone fell to 57% of cells assigned, ARI 0.834; with stage 7, 0.908).
* **Grid-search clone calling** needs a manual override to run on MDA_lung and gives 33 labels on
  MDA_PT; the evidence cut is threshold-driven and runs unchanged on all datasets.

### Stage order

Stage 5 first (it decides which variants exist, and keeps noise out of the graphs used later);
stage 7 after 6 (it consumes that stage's background estimate) but **before** 9, because protecting
the greedy four-gamete filter is its purpose; stage 9 last, on the final call matrix, since
imputation changes prevalences and conflicts.

## Results

Real data: `data_test/source_data/data/general/AFMs`. Shipped uses `max_fraction_unassigned=0.1` on
MDA_lung and MDA_PT (the default finds no solution). "Shared cells" = cells both pipelines assign.

| Dataset (cells / GBC clones) | Pipeline | Cells | Labels | ARI | ARI shared (dev/shipped) |
|---|---|---|---|---|---|
| MDA_clones (375 / 8) | shipped | 75.5% | 9 | 0.945 | |
| | dev, binomial + SNR + imputation 0.8 | 72.5% | 6 | 0.937 | 0.935/0.947 |
| | dev, no imputation | 57.9% | 6 | 0.908 | 0.903/0.946 |
| MDA_lung (1537 / 28) | shipped | 76.5% | 11 | 0.953 | |
| | dev, binomial + SNR (+ imputation 0.8) | 80.1% | 11 | 0.960 (0.956) | 0.968/0.958 |
| MDA_PT (2757 / 216) | shipped | 56.9% | 33 | 0.755 | |
| | dev, binomial + SNR | 46.8% | 37 | 0.897 | 0.900/0.838 |
| | dev, + imputation 0.8 | 46.9% | 37 | 0.891 | 0.894/0.839 |
| | dev, + pooled rescue | 47.9% | 37 | 0.896 | 0.899/0.834 |

Simulations (24 datasets: 5/10/30/50 clones x polytomy/depth-3 x 3 seeds, `grid.SIM`): shipped 0.614;
dev guarded 0.770 on 86% of cells; dev binomial + SNR 0.818 on 72%; with imputation 0.8, 0.775 on 77%.

Clones recovered on MDA_PT (>= 50% of cells assigned, >= 80% in one label, label >= 80% pure):
shipped 12, dev guarded 8, dev binomial 13-15. Ladder ceilings there: ~52% of cells and 26 clones of
>= 10 cells (39 of >= 5 cells) -- about 640 cells are in clones under 10 cells or clones with no
marker at all.

Tables: `results/snr_*.csv` (final real/sim runs), `results/coverage2_*.csv` (imputation sweep),
`results/ladder_MDA_PT_min*.csv` (ceilings), `results/loss_*_MDA_PT.csv` (what is thrown out),
`results/flow_MDA_PT.csv`, `results/spread_nature_MDA_PT.csv`, `results/geno_sweep_*.csv`.

## Tried and rejected (all measured)

| Idea | Outcome |
|---|---|
| graph-prior genotyping | imprints lineage structure on noise calls; MDA_PT large clones fragment |
| looser genotypes (1e-2, any read) | MDA_PT clones recovered 11 -> 2-4; single reads land on other clones' markers |
| read-level rescue | +300 cells on MDA_PT but ARI 0.90 -> 0.78 |
| imputation with `min_support=0` | +55 cells on MDA_PT, ARI 0.897 -> 0.846 |
| imputation coherence gate | blocks nearly every imputation (634 -> 635 calls) |
| `split_recurrent_graph` (recurrent variants -> per-lineage characters) | -0.07 ARI on MDA_PT, neutral elsewhere; off by default |
| carrier concentration in the QC (`min_concentration`) | removes real markers too (a whole-clone marker at 0.38); no ARI gain; off by default |
| QC rounds (`n_rounds` > 1) | no help |
| `one_sided=True` in the cut | resolves nested subclones, oversplits barcode-level truth |
| AD>0 / AD>=2 QC carriers | restore variant counts but MDA_PT 0.61-0.63 |

MDA_PT's variants "spread over several barcode clones" were shown to be **noise, not recurrence**:
carriers at ~0.006 AF (the sequencing-error level), no clone carrying them at clone level, and their
carrier clones are not related to each other.

## Open issues

* **The iterative background truncates.** `binomial_carriers(n_iter=10)`; convergence needed 7-18
  updates (MDA_PT at 1e-3: 18), so benchmarked calls are ~0.3% short of the fixed point. Carrier sets
  only grew (monotone, no cycles), so raising the cap is safe. More principled options: iterate to
  convergence with an assertion, empirical-Bayes shrinkage of the background across variants, a
  beta-binomial background, or a one-pass quantile estimator.
* **`min_ratio=10` (stage 7) is only benchmarked at one value** and drops two plausible MDA_lung
  markers (ratios 8.5, 9.7). The cleaner fix is the four-gamete filter's tie-breaking (it removes the
  variant with most conflicts, favouring sparse variants over prevalent markers).
* **Imputation is dataset-dependent**: needed on MDA_clones (+14.6 points of coverage), neutral on
  MDA_lung, costs 0.04 ARI in simulations (labels 20.2 -> 23.9).
* **No merge step in the cut**, so nested subclones cannot be resolved without oversplitting.
* **Coverage on MDA_PT** stays at 47% vs shipped 57% (ceiling ~52%).
* **`mito.pp.distances.compute_distances` still has the NaN-weight hazard**: one character without a
  positive AF makes every distance NaN. Worked around here by `grid.character_af` + `min_calls`.
* **Kernel-weighted join count** and **SNR before the QC** are plausible variants, untested.
* Validation is three real datasets and 24 simulations with a simulator that does not reproduce
  MDA_PT's regime (200+ clones, noise-dominated candidates).
