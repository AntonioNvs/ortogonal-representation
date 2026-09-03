# Validation-Rigor Enhancement — Design

**Branch:** `sage-position-regression`
**Date:** 2026-09-03
**Status:** implementation spec (no code written yet)

---

## Motivation

The career-validation framework (`docs/career_validation_framework.md`) is the primary
gate for the MIT Sloan submission. Its headline numbers currently favor
`orthogonal_shapley` over the Bradley–Terry baseline (partial ρ 0.27 vs 0.14,
underrated resolution 0.80 vs 0.67). A critical review of the implementation
(`src/validation/`, `src/explain/`, `output/validation_benchmark/benchmark.json`)
found six defects that would be surfaced in peer review. This document specifies
their fixes.

The review is independent of the *interpretability* reframe (the additive-readout /
Shapley-share story, which is a separate track) and of the *external market*
validation (salary/contract data). This document covers **statistical rigor only**:

1. Endogenous underrated cohort → fixed-cohort comparison + paired bootstrap difference
2. Coarse 3-bin car control → continuous car-quality control
3. Binary resolution → censored survival analysis
4. Pooled 1950–2025 → modern / hybrid era windows
5. Norm-correlation leakage → supervised recoverability probe
6. Hard-coded thresholds → sensitivity grid

---

## 1. Fixed-cohort comparison + paired bootstrap difference

### Problem

`mark_underrated` (`src/validation/inconsistency.py:21`) computes the within-season
skill percentile from **each model's own** `skill_score`. The underrated set is
therefore model-dependent: BT flags 49 rows, OrthShapley flags 15, and the two sets
are **different drivers**. Comparing `resolution_rate` 0.673 vs 0.80 across different
populations is not a model comparison.

`compare_resolution_rates` (`src/validation/inference.py:474`) then reports only the
point difference `rate − baseline_rate` with `beats_baseline: bool(diff >= 0)` — no
CI, no p-value, on a shared-driver structure that should permit a paired test.

### Fix

**(a) Model-agnostic cohort.** Define the underrated set from a *model-free* input:
the `teammate_residual` export (driver minus teammate mean — pure data, already a
baseline in `src/baselines/teammate_residual.py`). A row `(driver, T)` is underrated
iff:

```
teammate_residual within-season percentile >= 0.75   # ranking uses cohort_skill_col
AND constructor_tier_score_at_T <= 1                 # B-tier
```

Every model is then evaluated on its *own* `skill_score` **over the identical row
set**. Change `mark_underrated` to accept a `cohort_skill_col` (used only for the
percentile) distinct from `skill_col` (used for evaluation). Default
`cohort_skill_col = "teammate_residual_score"` when present.

**(b) Paired cluster-bootstrap difference.** Add
`paired_cluster_bootstrap_diff(df_a, df_b, stat_fn)` to `inference.py`: for each
bootstrap replicate, resample `driverId` clusters with replacement **once**, evaluate
`stat_fn` on both models over the *same* resampled drivers, and record `θ_a − θ_b`.
Report:

- percentile CI on the difference (95%),
- one-sided p `P(θ_a ≤ θ_b)` (fraction of replicates where Orth does not beat BT).

Because the two models share drivers, this is a paired test — strictly more powerful
than comparing independent CIs, and it answers "does Orth beat BT on the same people."

**Important correction (implementation).** On a model-free fixed cohort the
`underrated_flag` and `promoted` labels are *identical* across models, so the
resolution rate is no longer a model discriminator — pairing on it is degenerate.
The paired test therefore targets the quantities that still differ across models:
**within-cohort AUROC** and **within-cohort Spearman**. `compare_resolution_rates`
is kept for the endogenous (legacy) per-model cohort but is annotated as a point
comparison only.

### Deliverables

- `mark_underrated(..., cohort_skill_col=...)`
- `paired_cluster_bootstrap_diff(...)` and `fixed_cohort_paired_comparison(...)` in
  `src/validation/inference.py`
- `compare_resolution_rates` annotated as point-comparison only (legacy endogenous cohort).
- `--fixed-cohort` flag on `run_validation_benchmark.py`; benchmark report surfaces a
  `fixed_cohort` block with paired AUROC / Spearman differences.

---

## 2. Continuous car-quality control

### Problem

`partial_spearman` (`inference.py:165`) residualizes on
`constructor_tier_score_at_T` ∈ {1,2,3}. The tier is a percentile bin of the rolling
points-share `score`; two B-tier drivers share a control value even when one team
scored 4% of points and the other 0.5%, yet their forward trajectories differ
systematically. Collapsing a continuous confound to 3 bins leaves residual
confounding — the exact thing "skill adds signal above the car" must rule out.

### Fix

Thread the continuous rolling constructor `score` (already computed in
`compute_team_tiers`, but dropped before `join_career`) into the join and residualize
on it.

1. `join_career` (`src/experiments/evaluate_skill_model.py:37`): add
   `constructor_score_at_T` to the `ds_t` merge.
2. Call sites: pass `z_col="constructor_score_at_T"` to `partial_spearman` for the
   **primary** headline; keep `z_col="constructor_tier_score_at_T"` as a sensitivity
   row. The rank-residualization is unchanged (ranks are monotone-invariant to the
   score's scale, so it is robust to outliers).
3. Report both `partial_rho_continuous` (control = score) and the existing
   `partial_rho` (control = tier).

Add a monotonicity sanity check (`rho(skill, score)` should be modest) so a
near-collinear control does not silently wipe the effect.

### Deliverables

- `constructor_score_at_T` in the join.
- `partial_rho_continuous` in `career_metrics` and the benchmark.

---

## 3. Survival analysis for time-to-promotion

### Problem

`rest_of_career_outcome` (`src/validation/career_labels.py:79`) collapses a driver's
future into `outcome_score = mean(future tiers)` and a binary
`promoted = outcome > tier_at_T`. Magnitude of the promotion (1 vs 9 seasons) is
discarded, and drivers active at data cutoff are treated as **never promoted** rather
than **censored** — biasing resolution down for recent drivers. The inputs for a
proper survival treatment already exist: `seasons_to_promotion`, `n_future_seasons`,
`first_promotion_season`, `peak_tier_score`.

### Fix

Add `src/validation/survival.py` (pure numpy/scipy — `requirements.txt` has no
survival library, so no new dependency). Semantics:

- **Time origin** = `season_T`.
- **Event** = first season `k > 0` with `tier(T+k) > tier(T)`.
- **Censoring time** = `n_future_seasons` (active but not yet promoted at cutoff).
- **Ties** handled by Breslow approximation.

Estimators:

1. **Kaplan–Meier** curve of time-to-first-promotion, stratified by skill tertile.
2. **Log-rank** test between skill strata — permutation version (shuffle skill labels
   within season, mirroring `permutation_within_season`) to honor driver clustering.
3. **Univariate Cox** (Breslow partial likelihood, ~40 lines) with `skill_score` as
   covariate; hazard ratio + cluster-bootstrap CI.

**Cohorts (both, per user decision):**

- **Primary:** all eligible drivers (tier at T < S).
- **Secondary / diagnostic:** underrated drivers only.

### Deliverables

- `src/validation/survival.py` with `km_curve`, `logrank_perm`, `cox_univariate`.
- A `survival` block in `benchmark.json`.
- A survival-curve plot in the publication set.

---

## 4. Modern / hybrid era windows

### Problem

Headline metrics pool 1950–2025 (68 seasons). The efficient-market hypothesis is
coherent only after free agency and a competitive multi-team driver market existed —
not in the privateer / pay-driver 1950s–1960s. Pooling regimes dilutes the signal and
invites the "your market test isn't a market test" objection. `resolution_by_decade`
already shows instability (0.4 → 1.0 → 0.5 by decade).

### Fix

Stratify headline metrics into windows, per user decision:

- **Modern (≥ 2010):** primary for the market-efficiency claim.
- **Hybrid (≥ 1990):** robustness.
- **Full history:** demoted to sensitivity only.

Implementation: add a `--min-year` flag to `career_validation.py` and
`run_validation_benchmark.py`; a thin `window=` filter at the top of
`compute_career_metrics`; emit `era_windows = {">=2010": {...}, ">=1990": {...},
"all": {...}}`.

**Critical:** recompute the skill percentile / underrated flag **within each window**
(do not inherit the full-history flag), or the cohort drifts again (Section 1 bug).

### Deliverables

- `--min-year` flag; `era_windows` block in the benchmark JSON.
- Doc update marking modern window primary.

---

## 5. Supervised leakage probe (replace norm-correlation)

### Problem

`constructor_leakage_probe` (`src/explain/orthogonal_shapley_probes.py:64`) computes
`Spearman(driver_skill, ‖constructor_emb‖)`. Embedding **norm** is not constructor
**quality**; the gate `|ρ| < 0.3` answers "do big constructor embeddings coincide with
high driver skill," not "does the driver state leak the constructor." The current
0.238 measures the wrong quantity.

### Fix

Replace with a supervised recoverability probe:

1. **Logistic/linear probe** (sklearn) predicting `constructor_tier` (the thing the
   career test controls on) from the `driver_state` embedding, fit on train rows,
   evaluated held-out (multiclass / macro-AUC).
2. **Null threshold:** shuffle constructor labels, refit, get the null AUC
   distribution; gate = held-out AUC ≤ null 95th percentile (or ≤ chance + margin).
3. Keep the old norm-correlation as a *diagnostic* (renamed), but remove it as a gate.

### Deliverables

- `constructor_recoverability_probe` in `orthogonal_shapley_probes.py`.
- `constructor_tier_auc` + `null_auc_ci` in the benchmark `xai` block, replacing
  `constructor_leakage_rho` as the gated quantity.

---

## 6. Sensitivity grid (threshold robustness)

### Problem

`mark_underrated` hard-codes `skill_pct_threshold=0.75`; tier cut is
`P_S=0.30, P_A=0.35` (`team_tiers.py:8`). The doc's Limitations #6 promises "sensitivity
to 20%/30% should be reported" — not implemented. Every headline number is
conditional on two arbitrary cut-points.

### Fix

A grid runner recomputing headline metrics over:

- `skill_pct_threshold ∈ {0.70, 0.75, 0.80}`
- `p_S ∈ {0.25, 0.30, 0.35}` (holding `p_A` fixed at 0.35)

for each source, reporting resolution rate, underrated AUROC, and partial ρ per cell.
Claim to make: **the Orth > BT ordering is stable across the grid** (report the
min-difference cell and "how often Orth beats BT").

### Deliverables

- `src/experiments/sensitivity_grid.py` (or `--sweep` flag on the benchmark).
- `sensitivity` block in the benchmark JSON.
- A grid heatmap / table plot.

---

## Ordering & dependencies

1. **Section 1** (fixed cohort) is a prerequisite for 3 and 6, which consume the
   underrated flag.
2. **Section 2** (continuous control) is independent; lands alongside 1.
3. **Section 4** (era windows) is independent; can land anytime.
4. **Section 5** (leakage) touches the XAI path only; independent.
5. **Section 6** (grid) depends on 1 (cohort) and 4 (windows).

Suggested sequence: 1 → 2 → 4 → 3 → 6 → 5.

## Acceptance criteria

- `benchmark.json` gains: `paired_diffs` (resolution + partial ρ with CI and
  one-sided p), `partial_rho_continuous`, `era_windows`, `survival`, `sensitivity`,
  and the leakage gate is `constructor_tier_auc` against a null threshold.
- All headline claims rest on the **modern** window, **fixed** cohort, and
  **continuous** car control.
- No new runtime dependencies (survival is hand-rolled).
