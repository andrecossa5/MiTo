# MiTo finalization benchmarks — status and how to resume

Last updated: 2026-10-14. Branch `scverse`, on top of `c441938` (0.3.0).
Approved plan: `~/.claude/plans/process-plan-md-take-your-parallel-fox.md` (Phases 0–3); user notes in `PLAN.md` (repo root, untracked).

**Where we are:** Phase 0 (data + harness) and Phase 1 (analyses) are done. Phase 2 (package changes) has **not** started. It waits on the open questions at the bottom.

---

## Resume in 3 commands

```bash
conda activate mito-docs                       # mito 0.3.0 editable; numpy 2.4.6, pandas 3.0.5, scanpy 1.12.3,
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

## Working tree (uncommitted)

- `src/mito/io/format_afm.py`:
  - dominant-alt-allele reader (`_dominant_alt`);
  - memory-safe redeemV reader (6 of 14 columns, typed, float `Depth`).

  **Decision:** keep dominant-allele as the default, and add `multi_allelic_sites=True` as an option that keeps every allele per site, like redeemR. Not implemented yet.

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

## Decisions taken

- Plan:
  - reader decided after Phase 1;
  - recall ideas limited to D1/D2;
  - hard cap of 1000 candidates;
  - data in `data_test/bench`, harness in `benchmarks/`.
- 2026-10-14:
  - ReDeeM front end = faithful Filter-2 port (χ²);
  - reader: dominant allele default + `multi_allelic_sites=True` option.

## Open questions (user) — answer before Phase 2

1. **mtscATAC:** a lighter alternative to `read_scale`? Proposed test:
   - site-level overdispersion filter on reads/f;
   - weighted cosine on AF for distances;
   - calls only for the annotation.

   Compare with `read_scale` alone. Not run yet.
2. **D2:** undecided; it helps on ReDeeM (see above). Also open: whether to add clade confidence (bootstrap support).
   Earlier note: Proposed test first: let single molecules count at clean sites on ReDeeM (point 2 above), and report small private-marker cherries.

## Next (Phase 2, after the open questions)

- **io:**
  - `multi_allelic_sites` option;
  - per-strand AD layers for read-level input;
  - DP = A+C+G+T fallback;
  - redeemV `edge_trim=9` default;
  - `barcode_map=`;
  - `read_counts_table`.
- **pp:**
  - Filter-2 front end for ReDeeM;
  - `read_scale` in `call_genotypes`;
  - per-`scLT_system` presets in `filter_afm(preset=)`;
  - `max_candidates=1000`;
  - drop the redundant multi-allelic drop in `filter_low_quality_variants`.
- **Scalability:**
  - incremental leave-one-out kNN (only carrier rows/cols change), per-variant seeds, float32, threads;
  - shared with `impute_dropouts`.
- **Tests** for all of the above; then the Phase 3 re-baseline, docs and release 0.4.0.
