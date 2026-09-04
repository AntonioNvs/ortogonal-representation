# Career Validation Framework

**Version:** 3.1 (2026-09-04)  
**Status:** Primary validation gate for MIT Sloan Sports 2027 submission

This document is the methodological reference for career validation. The operational contract lives in [`model_contract.md`](model_contract.md); implementation in [`src/validation/`](../src/validation/). The six rigor fixes specified in [`docs/plans/2026-09-03-validation-rigor-design.md`](plans/2026-09-03-validation-rigor-design.md) are implemented; v3 reflects them (fixed cohort, continuous car control, censored survival, era windows, supervised leakage probe, sensitivity grid). v3.1 splits the leakage probe into a **season-state** and a **career-channel** probe, reflecting the hard-identification split (`docs/plans/2026-09-03-hard-identification-design.md`) where the driver effect = car-free career embedding + per-season offset.

---

## 1. Research hypothesis

Formula 1 operates as a **partially efficient driver market**: teams with better resources tend to hire drivers who have demonstrated higher car-adjusted performance. If a skill score truly isolates driver contribution from car context, it should predict **future team-tier trajectories** beyond what the driver's current team tier alone explains.

The central test is not "does skill correlate with points?" (trivially confounded by car) but:

> Among drivers currently in weak teams, does the model identify those who will eventually reach stronger teams?

This is the **underrated-driver** cohort: high skill signal, low current team tier.

---

## 2. Estimand

For driver **D** at season **T**:

| Symbol | Definition |
|--------|------------|
| `skill(D,T)` | Season-mean car-adjusted performance readout from the model |
| `tier(D,T)` | Ordinal team tier of D's primary constructor at T ∈ {B=1, A=2, S=3} |
| `outcome(D,T)` | Mean tier score over **all future active seasons** until career end |
| `promoted(D,T)` | `outcome(D,T) > tier(D,T)` |

We do **not** claim `skill` measures "pure intrinsic talent." The estimand is **retrospective car-adjusted performance** (`f(D,T,R)` in the model contract). Career validation tests whether this quantity carries **incremental predictive content** for market outcomes.

---

## 3. Team tier construction

Implemented in [`src/validation/team_tiers.py`](../src/validation/team_tiers.py).

1. **Points share** per constructor-season from end-of-season standings.
2. **Rolling score:** 3-season rolling mean grouped by **lineage** (rebrands pooled via [`team_lineage.py`](../src/validation/team_lineage.py)).
3. **Per-season percentile cut:** top 30% → S, next 35% → A, rest → B.

Tiers are **relative within each season**, not absolute across eras. This absorbs structural grid-size changes and dominance shifts (e.g. Mercedes 2014–2020).

**Driver-season constructor assignment:** mode `constructorId` across races that season. Mid-season transfers are collapsed to the most frequent team.

---

## 4. Underrated cohort

A driver-season `(D,T)` is **underrated** when:

```
skill_percentile(D, T) >= 0.75   # within-season rank
AND tier(D, T) == B              # constructor_tier_score_at_T == 1
```

**Rationale:** S- and A-tier drivers are already at competitive teams; the market-efficiency test is sharpest for drivers in backmarker seats who the model scores highly.

**Fixed-cohort correction (v3).** Ranking `skill_percentile` on *each model's own* score makes the underrated set model-dependent — BT flags one set of drivers, OrthShapley another, and comparing resolution across *different populations* is not a model comparison. The benchmark therefore supports a **model-free fixed cohort**: the percentile is computed from `teammate_residual` (driver minus teammate mean, pure data), so the underrated **row set is identical** across models and only the *score* differs. On a fixed cohort the `underrated_flag` and `promoted` labels are shared, so the resolution rate is **no longer a discriminator**; the model comparison moves to within-cohort AUROC, within-cohort Spearman (paired cluster-bootstrap), and censored survival (Section 8). `--fixed-cohort` toggles this in `sensitivity_grid.py` and the benchmark.

**Examples (illustrative):**
- Charles Leclerc at Sauber (2018): high BT skill, B-tier team → promoted to Ferrari.
- Lando Norris at McLaren (2019): rising skill in recovering B/A-tier team.
- Esteban Ocon at Force India/Racing Point (2018–2019): strong teammate differential, eventual Renault/Alpine seat.

---

## 5. Forward outcome: infinite horizon

### Fixed horizon (legacy, sensitivity only)

`forward_tier_outcome(horizon=3)` averages tier scores at T+1, T+2, T+3 with `require_full_horizon=True`. This drops retirees and end-of-data drivers and smooths single-season noise but **misses late promotions**.

### Rest-of-career (default since v2)

`rest_of_career_outcome()` averages tier scores over **every future season** where the driver remains active:

```
outcome(D,T) = mean{ tier(D, T+k) : k = 1, 2, ... until no more seasons }
```

Additional fields:
- `n_future_seasons` — number of future seasons observed
- `peak_tier_score` — maximum tier score achieved post-T
- `first_promotion_season` — first season where tier > tier-at-T
- `seasons_to_promotion` — offset to first promotion

**Censoring:** Drivers with no future seasons are excluded from the join. Drivers with 1+ future seasons are included regardless of count (no minimum horizon).

---

## 6. Inconsistency and resolution

| Term | Definition |
|------|------------|
| **Inconsistency** | Driver is underrated at T: high skill, B-tier team |
| **Resolution** | `promoted(D,T) = 1`: rest-of-career mean tier exceeds tier-at-T |
| **Resolution rate** | Fraction of underrated rows with `promoted == 1` |

A strong skill model should achieve a **high resolution rate** among underrated drivers: the market eventually "catches up" to the model's assessment.

**Time to promotion:** For resolved cases, `seasons_to_promotion` measures how quickly the correction occurs. Reported as median/mean over the underrated-promoted subset.

---

## 7. Skill trajectory features

From season-level skill history ([`skill_trajectory.py`](../src/validation/skill_trajectory.py)):

| Feature | Definition |
|---------|------------|
| `skill_slope_3yr` | Linear slope of skill over up to 3 seasons ending at T |
| `skill_delta` | `skill(T) - skill(T-1)` |
| `career_phase` | debut (≤2 seasons), mid (3–6), veteran (>6) |

**Rising underrated partial ρ:** Partial Spearman restricted to underrated drivers with `skill_slope_3yr > 0`. Tests whether improving skill trajectory adds signal beyond the snapshot.

---

## 8. Metrics and inference

All career rows are **clustered by `driverId`** (one driver contributes many seasons). Naive p-values on stacked rows are invalid.

| Metric | Formula / method | Primary? |
|--------|------------------|----------|
| **Survival Cox HR** | Univariate Cox (Breslow) of `skill` on time-to-first-promotion, eligible cohort (tier at T < S); cluster-bootstrap HR CI | **Yes** |
| **Log-rank (tertiles)** | Permutation log-rank, top vs bottom skill tertile | **Yes** |
| **Partial Spearman (all, continuous control)** | `ρ(skill, outcome)` residualized on continuous rolling constructor score | **Yes** |
| Partial Spearman (all, tier control) | Same, residualized on `tier_at_T` ∈ {1,2,3} | Diagnostic |
| **Underrated Spearman (stratum)** | `ρ(skill, outcome)` within fixed underrated stratum | **Yes** |
| Underrated promotion AUROC | `AUROC(skill, promoted \| underrated)` | Diagnostic (fixed cohort) |
| Resolution rate | `mean(promoted \| underrated)` | **Legacy** — identical across models on a fixed cohort |
| Cluster-bootstrap Spearman CI | Resample drivers with replacement | Diagnostic |
| Within-season permutation | Shuffle skill within season blocks | Diagnostic |
| Paired cluster-bootstrap difference | Orth − BT on identical drivers, one-sided p | **Yes** (fixed cohort) |

**Cluster-bootstrap:** 2000 replicates; resampling unit is the driver, not the row. Percentile CI at 95%.

### 8.1 Censored survival (primary since v3)

The binary `promoted` label discards time and misclassifies active drivers at cutoff as never-promoted. `src/validation/survival.py` models time-to-first-promotion directly (hand-rolled, no new dependency):

- **Time origin** = `season_T`; **event** = first `tier(T+k) > tier(T)`; **censoring** = `n_future_seasons`.
- **KM** curve stratified by skill tertile; **log-rank** (permutation) top-vs-bottom tertile; **Cox** HR with cluster-bootstrap CI.

### 8.2 Continuous car control (primary since v3)

Residualizing on 3-bin tier leaves residual confounding (two B-tier teams can differ 8× in points share). The primary headline now residualizes on the continuous rolling constructor **score** (`partial_rho_continuous`); the tier control is retained as a sensitivity row.

### 8.3 Era windows

Headline metrics stratify into **Modern (≥ 2010, primary)**, **Hybrid (≥ 1990)**, **Common (≥ 2014)**, and **Full history (sensitivity only)**. The market-efficiency claim is coherent only after free agency; pooling 1950–2025 dilutes it. Skill percentiles are recomputed **within** each window (not inherited from full history).

**Common window (≥ 2014)** is the fair like-for-like comparison across *all* models: it is the intersection of full-history coverage (BT/Orthogonal) with the Bayesian SSM's `start_year=2014` floor. The Bayesian SSM now fits through 2025 (the `end_year=2021` cap was removed), but it still has no coverage before 2014 — so a common-≥2014 window is where its HR / partial ρ can be compared to BT/Orthogonal on the *same* seasons. Treat Bayesian's modern-2010 / hybrid-1990 cells with caution: those windows extend earlier than its own fit start and will silently drop its pre-2014 rows.

### 8.4 Sensitivity grid

`src/experiments/sensitivity_grid.py` sweeps `skill_pct_threshold ∈ {0.70, 0.75, 0.80}` × `p_S ∈ {0.25, 0.30, 0.35}`. Claim: the Orth > BT ordering is stable across the grid. On a fixed cohort, only AUROC and within-stratum Spearman are reported (resolution is shared, hence non-discriminating).

### 8.5 Supervised leakage probe (interpretability gate)

Replaces the old norm-correlation gate. Two supervised probes predict the constructor from a driver embedding, held-out, against a permuted-label null (`src/explain/orthogonal_shapley_probes.py`):

1. **Season-state probe** (`constructor_recoverability_probe`) — predicts `constructorId` from the **`driver_state`** (per-season) embedding, deduplicated by season-long state, `StratifiedKFold`. This is the *season* channel, which under hard identification is **allowed** to carry constructor level (the offset is what absorbs team strength), and is further confounded by driver identity (each driver drives one team per season). **Current result: gate failed** — held-out macro-AUC 0.988 vs null 95th pct 0.515 → `leakage: true`. This is *expected* by design, not a bug: it tests the wrong channel for the car-free claim.

2. **Career-channel probe** (`constructor_recoverability_career_probe`) — the honest falsification for hard identification. The driver effect is split into a **career** embedding (car-free by construction: `nn.Embedding` outside `HeteroConv`, receives no constructor messages) plus a per-season offset. The probe predicts `constructorId` from the **career** embedding **restricted to team-switchers** (drivers with ≥2 constructors), aggregated to one `(driver, constructor)` pair each with equal weight, and split by **`GroupKFold` on driver** so a driver's embedding is never seen at train time. AUC ≈ chance (0.5) ⇒ the career channel is car-free; AUC ≫ null p95 ⇒ it still encodes the car. **Result: pending re-run** — the prior career probe was degenerate (sampled only the 2024–2025 test window, leaving 8 team-switchers) and has been rewritten against the full results table; the number must be regenerated before any conclusion about the car-free claim.

**Gate thresholds** (honest fixed-cohort numbers, 2026-09-03; career probe pending):

**Gate thresholds** (honest fixed-cohort numbers, 2026-09-03):

| Gate | Threshold | BT / Orthogonal / Bayesian |
|------|-----------|----------------------------------------|
| Survival Cox HR (eligible) | HR > 1; CI excludes 1 | 1.09 [0.98, 1.22] / **1.43 [1.16, 1.79]** / 5.72† |
| Survival log-rank p | < 0.05 | 0.0108 / **0.0136** / — |
| Partial ρ (continuous control) | CI low > 0 | 0.114 / **0.162** / — |
| Partial ρ (tier control) | ρ ≥ 0.15; CI low > 0 | 0.143 / **0.211** / 0.434 |
| Underrated Spearman (stratum) | CI low > −0.1 | 0.141 / **0.249** / — |
| Paired diff (fixed cohort, n=100) | p_one_sided < 0.05 | Spearman +0.108 (p=0.19, n.s.) / AUROC +0.018 (p=0.41, n.s.) |
| Constructor recoverability (season) | held-out AUC ≤ null p95 | — / **0.988 vs 0.515 (FAIL — expected)** / — |
| Constructor recoverability (career) | held-out AUC ≤ null p95 | — / **pending re-run** / — |

† Bayesian's HR/partial-ρ are estimated on a 2014–2025 window only (no pre-2014 coverage); at small n its HR CI can be wide. Compare it to BT/Orthogonal on the **common ≥ 2014** window, not the full-history cells.

**Reading:** Orthogonal Shapley beats Bradley–Terry on the *censored survival* test (HR 1.43 excluding 1, BT's straddling 1) and on *continuous car control* (CI excludes 0). The *strictest paired test* — within-cohort Spearman on the same 100 drivers — favors Orth (+0.108) but is **not significant** (p=0.19). The claim is therefore "Orth predicts promotion *timing* better" (survival), not "Orth discriminates the underrated cohort better."

---

## 9. Limitations

1. **Relative tiers** — S/A/B are within-season percentiles; cross-era comparison of raw tier labels is invalid.
2. **Constructor assignment** — Mode constructor per season; mid-season moves are lost.
3. **Censoring at data cutoff** — Active drivers at end of dataset have incomplete futures. The v3 survival analysis models this as right-censoring; the legacy `promoted`/`resolution` path still treats them as never-promoted (retained only as sensitivity).
4. **Bayesian SSM window** — Export now fits 2014–2025 (the `end_year=2021` cap was removed), so it is directly comparable to BT/Orthogonal on the **common ≥ 2014** window. It still has no coverage before 2014, so its `hybrid_1990`/`modern_2010`/`full` cells undercount relative to BT/Orthogonal — compare those with caution. Its high partial ρ and HR are *within-window* on a modern-only sample; the HR CI can still be wide at small n.
5. **No telemetry** — Claims are "car-adjusted performance," not strategy/reliability isolation.
6. **Underrated definition** — Top-25% threshold is a design choice; sensitivity to 20%/30% is reported in the grid (§8.4).
7. **Paired test power** — On the fixed cohort the within-cohort Orth-vs-BT difference (+0.108 Spearman) is not significant at n=100 drivers. The Orth edge is in survival *timing* and continuous car control, not within-cohort discrimination.
8. **Constructor recoverability (open)** — The *season-state* interpretability probe fails (held-out AUC 0.988 vs null p95 0.515). Under hard identification this is expected: the season channel is supposed to carry constructor level, and it is further confounded by driver identity. The clean falsification is the **career-channel probe** (Section 8.5.2), which tests the car-free career embedding restricted to team-switchers with driver-grouped folds. Its result is **pending re-run** (the prior version was degenerate). Until it passes (AUC ≤ null p95), the strongest defensible claim is **"car-adjusted performance"**, not "pure driver skill."

---

## 10. Canonical commands

```bash
# Single source, rest-of-career (default)
python src/experiments/career_validation.py --skill-source bradley_terry
python src/experiments/career_validation.py --skill-source orthogonal_shapley
python src/experiments/career_validation.py --skill-source bayesian_ssm

# Legacy fixed horizon (sensitivity)
python src/experiments/career_validation.py --skill-source bradley_terry --horizon 3

# Multi-source benchmark (era windows incl. common >=2014 for fair cross-model comparison)
python src/experiments/run_validation_benchmark.py \
  --sources bradley_terry bayesian_ssm orthogonal_shapley \
  --horizon inf --era-windows

# Bayesian SSM now fits through 2025 (drop the old 2021 cap)
python src/experiments/run_bayesian_ssm.py --start-year 2014 --end-year 2025

# Sensitivity grid (fixed cohort — model-free underrated set)
python src/experiments/sensitivity_grid.py \
  --sources bradley_terry orthogonal_shapley \
  --baseline bradley_terry --fixed-cohort

# Publication figures (post-hoc, from JSON — no DB/model re-run)
python src/experiments/plots/plot_validation_figures.py \
  --benchmark-json output/validation_benchmark/benchmark.json \
  --sensitivity-json output/sensitivity_grid/sensitivity_grid.json
```

---

## 11. Output artifacts

Per source under `output/career_validation/{source}/`:

| File | Contents |
|------|----------|
| `correlation.json` | Full metrics report |
| `resolution_report.json` | Inconsistency metrics only |
| `joined.csv` | All driver-season rows with outcomes and trajectory |
| `inconsistencies.csv` | Underrated rows only |
| `skill_scores.csv` | Season skill input |
| `team_tiers.csv` | Tier assignments |

Unified benchmark: `output/validation_benchmark/benchmark.json` includes `resolution_comparison` across sources.

---

## 12. Interpreting results

**Cox HR > 1 with CI excluding 1 (eligible cohort):** Higher skill predicts *faster* time-to-first-promotion. This is the primary fair-market claim — the skill score carries signal about who the market will promote.

**Log-rank p < 0.05 (top vs bottom tertile):** The KM curves separate — high-skill drivers promote faster than low-skill drivers at the same career stage.

**Partial ρ (continuous control) CI excluding 0:** Skill adds forward-tier signal *above* the continuous car-quality control — the "skill is more than the car" test.

**Paired diff significant (p < 0.05):** Orth beats BT on the *same* drivers. When non-significant, the model edge is only in survival timing / car control, not within-cohort discrimination.

**Recoverability AUC ≤ null p95:** The relevant embedding does not leak the constructor. For the hard-identification claim the gate is the **career-channel** probe (Section 8.5.2); the season-state probe is expected to leak by design. **Career AUC ≫ p95:** the "isolated skill" claim must be downgraded to "car-adjusted performance" (see Limitations #8).

**Low n_underrated:** Sparse cohort; widen CI or pool decades for stability.

Compare sources on equal footing: note year coverage (Bayesian 2014–2025 vs full history for BT/Orthogonal) and window — **common ≥ 2014** is the fair cross-model window; modern ≥ 2010 is primary for the market claim.
