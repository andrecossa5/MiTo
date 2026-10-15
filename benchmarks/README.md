# MiTo benchmarks — status and how to resume

Last updated: 2026-10-14. Branch `scverse`. **MiTo 0.5.0** is implemented (one pipeline for MAESTER, ReDeeM and mtscATAC). Its release check is `release_check.py`; results are below.
Approved plan: `~/.claude/plans/process-plan-md-take-your-parallel-fox.md`; user notes in `PLAN.md` (repo root, untracked).

**API note:** the scripts before `release_check.py` were written against the development API. That API still had the read-threshold candidate stage (`candidates=`, `cand_min_*`, `min_snr`), which 0.5.0 removes.
- They reproduce at commit `373cbb3` (`git checkout 373cbb3 -- src` in a scratch worktree).
- `release_check.py`, `final_compare.py` and `mgatk_pairs.py` run on 0.5.0.

## Release check (0.5.0, package defaults only, from the readers)

`python benchmarks/release_check.py mda|pl1000|sims`. Values are ARI / ARI_all.

| Dataset | 0.5.0 | + `min_support=0.5` | 0.3.0 | Time: filter_afm / tree / 100 support replicates |
|---|---|---|---|---|
| MDA_clones | .895 / .758 | .938 / .756 | .929 / .647 | 1.0 / 0.0 / 0.3 s |
| MDA_lung | .960 / .905 | .961 / .905 | .960 / .896 | 3.1 / 0.0 / 2.7 s |
| MDA_PT | .842 / .456 | .856 / .456 | .830 / .370 | 6.1 / 0.1 / 3.8 s |
| 24 MAESTER sims | mean .792 / .565 | .797 / .565 | .755 | 0.4 / — / 1.0 s |
| pL1000 ReDeeM | .840 / .660, recall 49% | .859 / .646, precision .80 | .785 / .593 | 8.6 / 0.1 / 12.9 s |
| pL1000 mtscATAC (f = 1.66, both strands) | .841 / .629, recall 34% | .871 / .600, precision .80 | .34 / .27 | 7.6 / 0.1 / 8.0 s |

- **Comparisons:** MitoDrift on pL1000 scores .679 / .632 (τ .125) and .800 / .635 (τ .25). The paper reports .967 / .964 / .847 on MDA.
- **Same cells and barcodes, ReDeeM vs mtscATAC** (`final_compare.py`): .818/.648 vs .846/.644.
- **Clonality filter** (`clonality_speed.py`):
  - 10–26× faster: 1,000 candidates × 3,300 cells, 87 s → 3.3 s; 9,700 cells, 70 s / 4.6 GB → 6.8 s / 1.3 GB.
  - Fixes a self-confirmation bug (float32 Gram downdate).
  - Kept variants vs the 0.5.0 code: Jaccard .95–.97. Against the bug-fixed old code, .965–.995, which is Monte Carlo level.
  - Bit-identical across re-runs.
- **Clade support** (`support_eval.py`), transfer support from count resampling, tree clade precision at support ≥ .5: ReDeeM .49 → .81 (recall .65 → .59), MDA_lung .21 → .69, mtscATAC .45 → .81.
- **UPGMA** is now SciPy (32.5 s → 0.18 s at 2,400 cells). It breaks ties between identical cells differently from Cassiopeia, so trees are not identical to before (ReDeeM: 72% of clades shared); the metrics above are equal or better.

## Resume in 3 commands

```bash
conda activate mito-docs                       # mito 0.5.0 editable; numpy 2.4.6, pandas 3.0.5, scanpy 1.12.3,
pip install pyreadr                            # sklearn 1.9.0, scipy 1.18.0, statsmodels 0.14.6, ete3 3.1.3
python benchmarks/fetch_pl1000.py              # -> data_test/bench/pL1000 (5.6 GB, md5-checked, skips intact files)
python benchmarks/run.py mitodrift             # gate: must print recall .7539 / precision .7728 / ARI_gt1pct .9184
```

- **Data and caches are gitignored** (`data_test/`).
  - Raw: `data_test/bench/pL1000/`.
  - Caches: `.../pL1000/cache/` (strand-aware mgatk npz, QualifiedTotalCts npz, AFMs).
  - MDA re-read with the current reader: `data_test/bench/mda_current_reader/`.
  - All are rebuilt on demand if missing.
- **MAESTER inputs** come from `data_test/source_data/data/general/AFMs/` (MiTo paper source data).

## Scripts

| Script | Purpose | Output |
|---|---|---|
| `fetch_pl1000.py` | figshare 10.6084/m9.figshare.31323481 + redeemR `data/{ATACWhite,RNAWhite,mitomap_freq}.rda` | data |
| `datasets.py` | loaders: `mda(name, reader)`, `sim(n, topo, seed, preset)`, `pl1000_filter2/redeemv/mgatk`, `barcode_map` (ATAC→GEX) | — |
| `metrics.py` | partition scores (ARI on assigned, ARI_all = abstention as singletons) and MitoDrift clade metrics | — |
| `run.py baseline \| mitodrift \| reader` | MDA×3 + pL1000 Filter-2 + 24 MAESTER sims; MitoDrift published tree; MDA shipped vs current reader | `results/baseline.csv`, `mitodrift.csv`, `reader.csv` |
| `mgatk_strand.py` | per-strand alt counts of the dominant allele + Signac `IdentifyVariants` stats | cache |
| `mgatk_filter_study.py` | candidate-filter grid vs reference sets (ReDeeM Filter-2, MitoDrift fr2_mc3, LARRY-informative) | `mgatk_filters.csv`, `mgatk_inflation.csv` |
| `mgatk_e2e.py` | MiTo on mgatk: candidate sets × genotyping (binomial / both strands / rounded dedup) | `mgatk_e2e_labelled.csv` |
| `mgatk_dup.py` | duplication-factor sweep (+ failed dispersion-index estimator) | `mgatk_dup.csv` |
| `mgatk_mechanism.py` | why dedup works: min2 reads, continuous scaling, beta-binomial | `mgatk_mechanism.csv` |
| `filter2_port.py [--entries]` | redeemR Filter-2 in Python on raw redeemV; validation vs the deposited matrix | `filter2_port_*.csv` |
| `redeem_e2e.py` | MiTo on ReDeeM: deposited Filter-2, port (χ²/Tarone, all vs labelled cells), current reader | `redeem_e2e.csv` |
| `redeem_small.py` | ReDeeM small clones: genotyping (binom 1e-3/1e-2/any molecule) x clonality on/off x tau x min_cells x D2, Filter-2 port on all cells | `redeem_small.csv` |
| `redeem_compare.py` | MitoDrift vs MiTo 0.3.0 vs proposed MiTo, same LARRY cells and metrics | `redeem_compare.csv` |
| `overdispersion.py redeem \| redeem_qcells \| maester \| sims` | generic Filter-2-style candidate stage (χ² or Tarone, blacklist/RSRS50, median depth, cap 1000) on any AFM vs MiTo 0.3.0 candidate filter | `overdispersion_*.csv` |
| `ablation.py mda \| redeem \| sims` | package `filter_afm` with one downstream stage off at a time (artefacts, provisional calls, clonality, SNR, four-gamete, all), vs 0.3.0 thresholds; D2 off/on | `ablation_*.csv` |
| `mgatk_unified.py [--f F]` | mgatk through the unified pipeline with a duplication factor (thresholds / raw / scaled) | `mgatk_unified*.csv` |
| `mgatk_dupfactor.py` | reads per molecule from strand variance (1.66), checked against ReDeeM molecules (1.73) | `mgatk_dupfactor.csv` |
| `mgatk_pairs.py` | mgatk as both-strand molecules (= the 0.5.0 reader) | `mgatk_pairs.csv` |
| `final_compare.py` | ReDeeM vs mtscATAC vs MitoDrift, same cells and LARRY barcodes, per-clone recovery | `final_compare*.csv` |
| `release_check.py mda \| pl1000 \| sims` | 0.5.0 defaults end to end, from the readers | `release_check_*.csv` |
| `clonality_speed.py real <afm>... \| scaling` | old vs incremental clonality filter: time, peak memory, agreement | `clonality_speed_*.csv` |
| `support_eval.py pl1000 \| mda` | count-resampling clade support: tree-level precision/recall, support as a clone filter | `support_eval_*.csv` |
| `reader_check.py` | maegatk/mgatk reader, 0.5.0 vs typed/sparse/chunked: identical AFMs, time and memory | `reader_check.csv` |
| `timing.py [sample]` | end-to-end time, maegatk tables -> annotated clones | `timing_*.csv` |
| `small_clones.py [--ceiling]` | where small LARRY clones are lost; D1 (size-aware tau) and D2 (small-clone pass); ceiling and tree-clade recall | `small_clones_*.csv` |

### Upstream code mirrored (read, not vendored)

- **redeemR** (`github.com/chenweng1991/redeemR`, branch `redeemR2.0`):
  - `scripts/redeemR2.0_preprocess.R`;
  - `R/filter2.R`: `goodness_of_fit_test`, `run_binomial_noise_removal`, `Create_redeemR_model`, `clean_redeem*`;
  - `R/variant_summary.R`: `redeemR.read.trim`, `GTSummary`, `DepthSummary`, `Vfilter_v4`;
  - `R/buid_tree.R`: `Make_matrix`;
  - `R/variant_annotation.R`: blacklist, `add_median_depth_to_redeemR`.
- **MitoDrift** (`github.com/sankaranlab/mitodrift`, master):
  - `R/utils.R`: `trim_tree`, `assign_clones_polytomy`;
  - `R/diagnostics.R`: `compute_pr_from_jaccard_mat`.
- **Signac** (`stuart-lab/signac` `R/mito.R`): `IdentifyVariants` (strand_correlation, vmr, n_cells_conf_detected).
- **MitoDrift paper** (PMC12918977):
  - mgatk arm "fr2_mc3" = ≥2 reads per strand in ≥3 cells;
  - MAESTER: ≥5 cells with VAF >5%.

## Results (2026-10-08)

**Gates passed**
- **MitoDrift reproduced exactly** from its deposited tree:
  - clade recall 75.4%, precision 77.3% at τ=0.125;
  - strata 85/79/71%;
  - Fig 2F ARI 0.918 / 0.750. This needs root singletons excluded and no trivial-root collapse.
- **MiTo 0.3.0 reproduced exactly:**
  - MDA_clones 0.929 @70.1%, lung 0.960 @79.9%, PT 0.830 @45.3%;
  - 24 sims 0.755.
  - pL1000 Filter-2 matrix → MiTo 0.863 @47.9% (Sep's hand-built object gave 0.879).
- **Paper MiTo (median):** 0.967 / 0.964 / 0.847.

**mtscATAC (pL1000 mgatk, same 1,362 LARRY cells)**
- **The cause is genotyping.** mgatk counts reads, not molecules: alt-bearing entries are 10.7× those of ReDeeM, and depth is 2.1×. The binomial test then treats duplicates as independent evidence.
- **Fix: continuous read scaling** (binomial tail on AD/f, DP/f via `betainc`, f=2) → ARI **0.887**, ARI_all **0.662**, clone precision **0.97** at 46% yield. Current genotyping gives 0.46 / 0.38 / 0.45.
- **Robust to the choice of f:** any f in 1.5–4 works.
- **What doesn't work:**
  - requiring ≥2 reads does nothing;
  - beta-binomial is far too conservative;
  - estimating f by dispersion index failed (1.18 on mgatk, 12 on ReDeeM).
- **Candidates:** ≥2 reads on each strand in ≥3 cells (fr2_mc3) → **732 sites**, covering 94% of LARRY-informative variants. VMR is useless; strand correlation adds little.

**ReDeeM**
- **Filter-2 port is EXACT.** Its variant set equals the deposited matrix's once restricted to the labelled cells with ≥2 carriers. All **949,314** entries have identical alt count and depth.
- **The noise test does the selection:** 12,310 candidates → ~1.1k variants.
- **Tarone ≈ χ²** downstream (no gain): keep χ².
- **On all ~4k cells, Filter-2 port → MiTo:** ARI 0.813, ARI_all 0.629 (MitoDrift 0.632), yield 65%, recall 49% (MitoDrift 75%), smallest clones 33% (MitoDrift 71%).
- **Current reader + defaults:** 0.792 / 0.610, recall 50%, but lower precision (0.70 vs 0.76).

**Small clones** (deposited matrix, labelled cells)
- **Of the 126 clones of 2–3 cells:** 69 are lost at the candidate filter, 35 at genotyping/QC, 17 at the cut, and 2 at tau.
- **Observable ceiling:** 81/126 have a ≥2-carrier, mostly private marker. MitoDrift recovers 90/126.
- **MiTo tree** (all internal nodes) reaches 15% on these; MiTo's labels reach 2%.
- **D2 (small-clone pass):** recall 26→33%. D1 (size-aware tau) has a small effect.
- **Why MitoDrift recovers them:**
  1. no n_pos/AD candidate floor;
  2. single molecules count. Ours are never called: depth ~60 × error 1e-4 → p≈0.006 > geno_alpha 1e-3;
  3. every cell stays a tip (MiTo keeps 749/1362);
  4. the clade metric credits any confident cherry.

**MAESTER reader**
- The current reader is better on ARI_all on all three datasets: clones .647→.746, lung .896→.904, PT .370→.394.
- ARI on assigned cells drops on clones (.929→.877) and PT (.830→.759), for +18 and +8 points of yield.

## Results (2026-10-14): ReDeeM small clones and the three-way comparison

- **Partition vs clade scoring.** MitoDrift's 75% recall counts every confident tree node. Scored as a partition (as MiTo is), MitoDrift gets 53.5% at tau .125 (smallest clones 44%, precision .80, ARI_all .632) and 62.5% at tau .25 (54%, .89, .635).
- **Single-molecule binary calls are a dead end.** Binomial 1e-2 or AD>=1 drops ARI to ~0.50 and breaks the large clones.
- **The clonality filter is redundant after Filter-2 and costs small clones.** Turning it off raises tree recall 68%→77% (smallest 64%→79%). Large clones split more, so it stays on by default.
- **D2 on ReDeeM:** +6–10 pts recall, +13–20 pts on the smallest clones, ~-0.06 precision, ARI_all neutral. min_cells=3 is worse than D2.
- **Proposed MiTo** (Filter-2 front end, MiTo candidate thresholds opened, clonality on, tau .25, D2) on MitoDrift's exact input: ARI .866, ARI_all **.662** (MitoDrift .632/.635), precision .75, recall 51% (77/61/39% by stratum), yield 63%, 13 s.
- **On all ~4k cells:** ARI_all .612, recall 57%.
- **MiTo 0.3.0 defaults from raw redeemV:** ARI_all .593, recall 51%, precision .71.
- **Remaining gap:** yield and small-clone precision. MiTo's tree has no clade confidence (tree precision .45 vs .77), so this is the bootstrap clade-support question.

## Results (2026-10-14): one overdispersion candidate stage for all UMI assays

`overdispersion.py`: Filter-2's variant-level steps applied to any AFM, then `filter_afm` with candidate thresholds opened. Values are D2 off unless stated.
- **MAESTER, current reader:**
  - MDA_PT: ARI .759→.841, ARI_all .394→.453, precision .29→.41;
  - MDA_clones and MDA_lung: unchanged (.877/.746, .958/.903);
  - χ² is better than Tarone here (Tarone passes about 2× as many variants, hitting the cap on lung).
- **ReDeeM, raw redeemV trim 9:**
  - all 4,630 cells downstream: no gain (.791/.610);
  - only cells with mean coverage ≥10 (3,721): ARI .814, ARI_all .625, tree recall 68%; with D2, recall 55% and ARI_all .628. This recovers the exact port (.813/.629).
  - MiTo defaults on the same cells stay at .794/.613, so the gain needs both steps.
  - Still just under MitoDrift's ARI_all (.632/.635).
- **24 MAESTER sims:** mean ARI .755→.788. All of the gain comes from one sim (5 clones, depth3, seed 0: .118→.924). Every other sim is within ±.003, so there is no regression.

## Results (2026-10-14): stage ablation, package implementation

`ablation.py` runs the package's `filter_afm` (over-dispersion, cap 1000), with one downstream stage off at a time. Values are ARI / ARI_all, D2 off.

| Stage switched off | MDA_clones | MDA_lung | MDA_PT | ReDeeM | Sims (24) |
|---|---|---|---|---|---|
| none | .877/.746 | .958/.903 | .841/.453 | .814/.626 | .788/.558 |
| SNR | .895/.758 | .958/.903 | .841/.453 | .814/.626 | .787/.557 |
| dbSNP/REDIdb | .877/.746 | .956/.904 | .841/.453 | .818/.644 | .788/.559 |
| provisional α/10 → α | .877/.746 | .957/.904 | .835/.454 | .813/.613 | .779/.551 |
| clonality | .853/.706 | .954/.893 | .818/.424 | .810/.609 | .742/.539 |
| four-gamete | .885/.804 | .945/.902 | .719/.509 | .800/.616 | .705/.569 |
| 0.3.0 thresholds | .877/.746 | .959/.904 | .759/.394 | .794/.613 | .755/.551 |

- **Defaults adopted:**
  - SNR filter skipped after the over-dispersion stage;
  - dbSNP/REDIdb on for MAESTER only.
- **Checked with the new defaults** (D2 on): MDA_PT .845/.453, MDA_clones .896/.758, ReDeeM (cells with coverage ≥10) .813/**.644**, recall 57%. MitoDrift: .632/.635.

## Decisions taken

- Plan:
  - reader decided after Phase 1;
  - recall ideas limited to D1/D2;
  - hard cap of 1000 candidates;
  - data in `data_test/bench`, harness in `benchmarks/`.
- 2026-10-14:
  - ReDeeM front end = faithful Filter-2 port (χ²);
  - reader: dominant allele default + `multi_allelic_sites=True` option.
  - over-dispersion candidate stage = default for MAESTER/ReDeeM (implemented);
  - `max_candidates=1000` (implemented);
  - D2 = `annotate_clones(small_clones=False)` option (implemented);
  - SNR off after over-dispersion; artefacts on for MAESTER only (implemented);
  - `multi_allelic_sites` option, redeemV `edge_trim=9`, `filter_cells(mean_cov_all=10)` (implemented).
- Next: mtscATAC.

## Open (after 0.5.0)

- **Clade confidence** (bootstrap support on the tree): the remaining gap to MitoDrift is tree precision (.46 vs .90) and small-clone recall.
- **mtscATAC cell coverage:** 36% of barcoded cells carry no called variant (ReDeeM 22%). The user's lighter alternative is still untested: weighted cosine distances on allele frequencies, with calls used for annotation only.
- **Scalability:** incremental leave-one-out kNN for the clonality filter (plan, Phase 2).
- Push / release: not done (user's call).
