# Career Validation Framework

**Version:** 4.0 (2026-09-04)
**Status:** Primary validation gate for MIT Sloan Sports 2027 submission

This document is the methodological reference for career validation. The operational
contract lives in [`model_contract.md`](model_contract.md); implementation in
[`src/validation/`](../src/validation/).

**What changed in v4.** v3.x was statistically complete but overloaded (~12 metrics +
2 leakage probes), which buried the message. v4 leads with **three headline statistics**
run on **one fixed protocol** (common era ≥ 2014 + a model-free fixed cohort), so the
*same* career transitions are scored across all four models. Every other metric from
v3.x is retained but demoted to the **Supporting metrics** appendix (§7). The prior
rigor fixes (fixed cohort, continuous car control, censored survival, era windows,
leakage probes) still underpin everything — see
[`docs/plans/2026-09-03-validation-rigor-design.md`](plans/2026-09-03-validation-rigor-design.md)
and [`docs/plans/2026-09-03-hard-identification-design.md`](plans/2026-09-03-hard-identification-design.md).

---

## 1. Research hypothesis

Formula 1 operates as a **partially efficient driver market**: teams with better
resources tend to hire drivers who have demonstrated higher car-adjusted performance. If
a skill score isolates driver contribution from car context, it should predict **future
team-tier trajectories beyond what the driver's current team tier alone explains**.

The estimand is **retrospective car-adjusted performance** (`f(D,T,R)` in the model
contract), *not* "pure intrinsic talent." Career validation tests whether this quantity
carries **incremental predictive content** for market outcomes.

The four models under comparison:

| Model | Role | Level |
|-------|------|-------|
| `bradley_terry` | weak baseline | race-level pairwise |
| `plackett_luce` | ok baseline | race-level listwise |
| `bayesian_ssm` | strong baseline | season-level state-space (Lindner et al.), fits 2014–2025 |
| **`orthogonal_shapley`** | **candidate** | race-level GNN + coalition Shapley |

---

## 2. The fixed protocol (like-for-like across all four models)

Two choices make the comparison fair, and are applied identically to every model:

1. **Common era window ≥ 2014.** This is the intersection of full-history coverage
   (BT / PL / Orthogonal) with the Bayesian SSM's fit start. On this window all four
   models cover the *same seasons*, so their headline numbers are directly comparable.
   Skill percentiles are recomputed **within** the window.
2. **Model-free fixed cohort.** The underrated cohort and the `promoted` label are
   defined once, from `teammate_residual` (pure data: driver minus teammate mean), so
   the **row set and career transitions are identical across models** — only each model's
   *score* differs. This is what "fixing the transfers/checks" means: BT, PL, Bayesian
   and Orthogonal are all judged on the same drivers moving between the same teams.

**One canonical command produces the artifact all figures read:**

```bash
python src/experiments/run_validation_benchmark.py \
  --sources bradley_terry plackett_luce bayesian_ssm orthogonal_shapley \
  --horizon inf --min-year 2014 --fixed-cohort --era-windows
```

`--min-year 2014` filters **before** the underrated flag is assigned and propagates into
both the `career` and the `survival` blocks; `--fixed-cohort` defines the cohort on
`teammate_residual`; `--era-windows` also emits per-window cells (incl. `common_2014`)
for the sensitivity appendix.

---

## 3. Headline: the three statistics

These three carry the abstract. Each answers a distinct, plain-language question.

### Statistic 1 — Partial Spearman ρ, continuous car control

- **JSON:** `sources.<m>.career.partial_rho_continuous` (+ `partial_rho_continuous_ci_low/high`)
- **What it measures:** Does a driver's skill score predict their **forward team-tier
  trajectory** *above and beyond the quality of the car they are in right now*? We
  rank-residualize both skill and the forward outcome on the **continuous** rolling
  constructor points-share (tighter than a 3-bin tier: two B-tier teams can differ 8× in
  pace) and correlate the residuals.
- **Why continuous, not tier:** residualizing on 3 bins leaves the car half-controlled;
  the continuous score is the honest "skill is more than the car" test.
- **Pass:** ρ > 0 with 95% cluster-bootstrap CI (by driver) excluding 0.
- **Implementation:** `partial_spearman(..., z_col="constructor_score_at_T")` in
  [`src/validation/inference.py`](../src/validation/inference.py).

### Statistic 2 — Cox hazard ratio, censored time-to-promotion

- **JSON:** `sources.<m>.survival.eligible.cox.{hazard_ratio, hr_lo, hr_hi}`
- **What it measures:** Among drivers who *can* still move up (below the top tier at T),
  does higher skill predict reaching a stronger team **sooner**? Time origin = season T,
  event = first season with `tier(T+k) > tier(T)`, censoring = data cutoff. Unlike a
  binary "promoted?" label, this uses *when* the promotion happens and correctly treats
  active-but-not-yet-promoted drivers as right-censored.
- **Pass:** HR > 1 with cluster-bootstrap CI (by driver) **excluding 1**. This is the
  decisive discriminator: it rewards *timing*, not just direction.
- **Companion:** permutation log-rank, top vs bottom skill tertile (drives Figure 2).
- **Implementation:** [`src/validation/survival.py`](../src/validation/survival.py)
  (`eligible_survival`, `cox_univariate`, `cox_cluster_bootstrap_ci`, hand-rolled — no
  new dependency).

### Statistic 3 — Locked-test ranking fidelity (2024–2025)

- **JSON:** `sources.<m>.locked_test.{pl_nll, pairwise_acc}` (lower NLL / higher acc better)
- **What it measures:** Does the *same* skill readout reproduce true race finishing order
  on the held-out 2024–25 seasons, scored in causal `filtered` mode (only races 1…R
  known when scoring round R)? This checks the score is a genuine performance signal, not
  just a career-outcome correlate.
- **Comparability caveat:** BT, PL and Orthogonal are `filtered` (walk-forward, honest
  held-out). The **Bayesian SSM is `smoothed` / in-sample** (`walk_forward=False`), so
  its locked-test number is **not a fair held-out comparison** — report it annotated as
  in-sample, or omit it from this row. It is *not* NaN on a current run.
- **Pass:** Orthogonal ≤ BT NLL (within ~0.01) and ≥ BT − 0.01 pairwise accuracy, i.e.
  the car-adjusted readout does not sacrifice ranking to gain career signal.

---

## 4. The three figures

All render post-hoc from the canonical artifacts (no DB / model re-run for figures 1–2).

| Figure | File stem | Statistic shown | Renderer |
|--------|-----------|-----------------|----------|
| **1. Fair-market forest** | `fair_market_forest` | Stats 1 & 2: two rows (partial ρ, Cox HR), 4 models each, 95% CIs, null lines at 0 and 1 | `visualization/benchmark_forest.py` |
| **2. Time-to-promotion KM** | `survival_km_<m>` | Stat 2 mechanism: KM curves by skill tertile; top tertile promotes faster; HR + log-rank annotated | `visualization/survival_curve.py` |
| **3. Shapley attribution** | `shapley_attribution_<yr>` | The method's identity: driver / constructor / context shares per driver (Orthogonal **only**) | `visualization/entity_attribution.py` |

Stat 3 (locked-test ranking) stays **tabular** — `locked_test` has no CI, so it is not a
forest row.

Figure 3 is Orthogonal-only by design: BT and PL hard-code `contrib_context = 0.0` and
have no genuine coalition, so a stacked decomposition is degenerate for them. The
attribution decomposition is what no baseline can produce — it is the paper's signature.

Render:

```bash
# Figures 1 & 2 (from benchmark.json)
python src/experiments/plots/plot_validation_figures.py \
  --benchmark-json output/validation_benchmark/benchmark.json

# Figure 3 (needs the race parquet + a representative season)
python src/experiments/plots/plot_entity_attribution.py \
  --source orthogonal_shapley --season 2024 \
  --output output/plots/validation/shapley_attribution_2024
```

---

## 5. Reading the result

The abstract-level claim is:

> **Orthogonal Shapley predicts *who* the market promotes and *how fast* it does so,
> better than a race-level (BT/PL) or a season-level Bayesian baseline — while isolating
> the driver's share of each race, which no baseline can.**

Concretely, on the ≥ 2014 fixed-cohort protocol we expect:

- **Stat 1:** Orthogonal ≥ baselines on partial ρ, CI excluding 0. (Bayesian may post a
  higher point ρ; note its inference mode and season-level smoothing — see §6.)
- **Stat 2:** Orthogonal HR > 1 with CI **excluding 1**, where BT's CI straddles 1. This
  is the cleanest single discriminator.
- **Stat 3:** Orthogonal best `pl_nll` / `pairwise_acc` among held-out race-level models;
  Bayesian annotated in-sample.

The honest framing: the Orthogonal edge is in **promotion timing (survival) + continuous
car control + ranking fidelity + attribution**, not necessarily in a single point ρ.

---

## 6. Estimand, tiers, cohort, outcome (definitions)

**Estimand** for driver **D** at season **T**:

| Symbol | Definition |
|--------|------------|
| `skill(D,T)` | season-mean car-adjusted performance readout from the model |
| `tier(D,T)` | ordinal team tier of D's primary constructor at T ∈ {B=1, A=2, S=3} |
| `outcome(D,T)` | mean tier score over **all future active seasons** until career end |
| `promoted(D,T)` | `outcome(D,T) > tier(D,T)` |

**Team tiers** ([`src/validation/team_tiers.py`](../src/validation/team_tiers.py)):
points share per constructor-season → 3-season rolling mean grouped by **lineage**
(rebrands pooled via [`team_lineage.py`](../src/validation/team_lineage.py)) → per-season
percentile cut (top 30% → S, next 35% → A, rest → B). Tiers are **relative within each
season**. Driver-season constructor = mode `constructorId` across that season's races.

**Underrated (fixed) cohort:** a driver-season is underrated when its **model-free**
percentile (from `teammate_residual`) is ≥ 0.75 **and** its tier is B. Because the flag
is model-free, the underrated **row set and `promoted` labels are identical across
models**; only the score differs. `mark_underrated(..., cohort_skill_col=...)` in
[`src/validation/inconsistency.py`](../src/validation/inconsistency.py).

**Forward outcome (rest-of-career, default):** `rest_of_career_outcome()` averages tier
scores over **every** future active season (no fixed horizon → captures late promotions).
Drivers with no future seasons are excluded; the survival model treats data-cutoff
censoring explicitly. Fixed horizon=3 is retained for sensitivity only.

**Bayesian comparability note (corrected in v4):** the SSM now fits **2014–2025** (the
old `end_year=2021` cap is removed), so it participates in **all** headline metrics on
the ≥ 2014 window on equal footing. It still has **no pre-2014 coverage** — so its
`modern_2010` / `hybrid_1990` / `full` era cells undercount and must be read with caution;
its `common_2014` cell is the comparable one. It remains season-level and `smoothed`,
which is why its locked-test ranking (Stat 3) is in-sample.

---

## 7. Supporting metrics & diagnostics (retained, non-headline)

These stay in the benchmark for rigor but do **not** appear in the abstract. All career
rows are **clustered by `driverId`**; naive p-values on stacked rows are invalid.

| Metric | Method | Notes |
|--------|--------|-------|
| Partial ρ (tier control) | residualize on `tier_at_T ∈ {1,2,3}` | looser than Stat 1's continuous control |
| Underrated Spearman (stratum) | ρ(skill, outcome) within the fixed underrated stratum | small-n; wide CI |
| Underrated promotion AUROC | AUROC(skill, promoted \| underrated) | diagnostic on fixed cohort |
| Resolution rate | mean(promoted \| underrated) | **identical across models on a fixed cohort** — non-discriminating |
| Paired cluster-bootstrap diff | Orth − BT on identical drivers, one-sided p | the strictest within-cohort test; underpowered at n≈100 |
| Cluster-bootstrap Spearman CI | resample drivers with replacement, 2000–5000 reps | CI primitive used everywhere |
| Within-season permutation | shuffle skill within season blocks | era-robust null |
| Fisher-z pooled ρ | per-season ρ → z → variance-weighted mean | within-season independence |
| Sensitivity grid | sweep `skill_pct ∈ {.70,.75,.80}` × `p_S ∈ {.25,.30,.35}` | claim: Orth ≥ BT ordering stable |

### 7.1 Interpretability / leakage probes (hard-identification gate)

The driver effect is split into a **career** embedding (car-free by construction:
`nn.Embedding` outside `HeteroConv`) + a per-season **offset**
([`src/explain/orthogonal_shapley_probes.py`](../src/explain/orthogonal_shapley_probes.py)):

1. **Season-state probe** — predicts `constructorId` from the per-season `driver_state`.
   **Expected to leak** (the offset is *supposed* to carry team level; also confounded by
   driver identity). A failing gate here is by design, not a bug.
2. **Career-channel probe** — the honest falsification: predicts `constructorId` from the
   **career** embedding, restricted to team-switchers, split by `GroupKFold` on driver.
   AUC ≈ chance ⇒ car-free; AUC ≫ null p95 ⇒ still encodes the car. **Status: pending
   re-run** (the prior version was degenerate). Until it passes, the defensible claim is
   **"car-adjusted performance,"** not "pure driver skill."

---

## 8. Limitations

1. **Relative tiers** — S/A/B are within-season percentiles; cross-era raw-label
   comparison is invalid.
2. **Constructor assignment** — mode constructor per season; mid-season moves are lost.
3. **Censoring** — headline Stat 2 models data-cutoff censoring correctly; the legacy
   binary `promoted` path (§7) treats active drivers as never-promoted (sensitivity only).
4. **Bayesian window / mode** — fits 2014–2025 and joins the ≥ 2014 headline, but is
   season-level and `smoothed`; its pre-2014 era cells undercount and its locked-test
   ranking is in-sample.
5. **No telemetry** — claims are "car-adjusted performance," not strategy/reliability
   isolation.
6. **Career-channel probe open** — until it passes (AUC ≤ null p95), do not claim "pure
   skill."

---

## 9. Output artifacts

Unified benchmark: `output/validation_benchmark/benchmark.json` — per source: `career`
(incl. `partial_rho_continuous`), `survival` (incl. `eligible.cox`), `locked_test`,
`shapley_season_mean`, `era_windows`, plus a top-level `fixed_cohort` paired-comparison
block and `resolution_comparison`. Per-source detail under
`output/career_validation/{source}/` (`correlation.json`, `joined.csv`,
`inconsistencies.csv`, `skill_scores.csv`, `team_tiers.csv`).

Figures: `output/plots/validation/` (`fair_market_forest`, `survival_km_<m>`,
`shapley_attribution_<yr>`), each as PNG + SVG + PDF with a sidecar metadata JSON.
