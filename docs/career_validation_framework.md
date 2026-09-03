# Career Validation Framework

**Version:** 2.0 (2026-09-02)  
**Status:** Primary validation gate for MIT Sloan Sports 2027 submission

This document is the methodological reference for career validation. The operational contract lives in [`model_contract.md`](model_contract.md); implementation in [`src/validation/`](../src/validation/).

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
| **Resolution rate** | `mean(promoted \| underrated)` | **Yes** |
| **Underrated promotion AUROC** | `AUROC(skill, promoted \| underrated)` | **Yes** |
| **Partial Spearman (underrated)** | `ρ(skill, outcome)` within underrated stratum (tier-at-T is constant B; raw Spearman with cluster-bootstrap) | **Yes** |
| Partial Spearman (all) | Same, all drivers | Diagnostic |
| Cluster-bootstrap Spearman CI | Resample drivers with replacement | Diagnostic |
| Within-season permutation | Shuffle skill within season blocks | Diagnostic |
| Eligible promotion AUROC | Below S-tier at T, all drivers | Diagnostic |
| Fisher-z pooled ρ | Per-season ρ pooled across eras | Diagnostic |

**Cluster-bootstrap:** 2000 replicates; resampling unit is the driver, not the row. Percentile CI at 95%.

**Gate thresholds** (calibrated against Bradley–Terry baseline, 2026-09-02):

| Gate | Threshold | Empirical (BT / Orthogonal / Bayesian) |
|------|-----------|----------------------------------------|
| Underrated resolution rate | ≥ BT baseline; CI low > 0.5 | 0.67 / **0.80** / 1.00 (n=6) |
| Underrated promotion AUROC | ≥ BT baseline; CI low > 0.45 | 0.57 / **0.67** / n/a |
| Underrated Spearman (stratum) | ≥ BT baseline; CI low > −0.1 | 0.00 / **0.63** / 0.14 |
| All-driver partial Spearman | ρ ≥ 0.15; CI low > 0 (diagnostic) | 0.14 / **0.27** / 0.43 |

Orthogonal Shapley beats BT on resolution rate and underrated-stratum Spearman. Bayesian SSM has strong in-window signal but only 6 underrated rows (2014–2021); treat separately.

---

## 9. Limitations

1. **Relative tiers** — S/A/B are within-season percentiles; cross-era comparison of raw tier labels is invalid.
2. **Constructor assignment** — Mode constructor per season; mid-season moves are lost.
3. **Censoring at data cutoff** — Active drivers at end of dataset have incomplete futures; their `outcome` uses available seasons only.
4. **Bayesian SSM window** — Export covers 2014–2021 only; career joins exclude later seasons for that source.
5. **No telemetry** — Claims are "car-adjusted performance," not strategy/reliability isolation.
6. **Underrated definition** — Top-25% threshold is a design choice; sensitivity to 20%/30% should be reported in robustness.

---

## 10. Canonical commands

```bash
# Single source, rest-of-career (default)
python src/experiments/career_validation.py --skill-source bradley_terry
python src/experiments/career_validation.py --skill-source orthogonal_shapley
python src/experiments/career_validation.py --skill-source bayesian_ssm

# Legacy fixed horizon (sensitivity)
python src/experiments/career_validation.py --skill-source bradley_terry --horizon 3

# Multi-source benchmark
python src/experiments/run_validation_benchmark.py \
  --sources bradley_terry bayesian_ssm orthogonal_shapley \
  --horizon inf
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

**High resolution rate + high underrated AUROC:** The model identifies hidden talent in weak teams, and higher skill scores discriminate who gets promoted.

**High partial ρ (underrated) with CI excluding zero:** Skill adds forward-tier signal beyond current team tier within the target cohort.

**Resolution rate below BT baseline:** Model's underrated flags are less predictive of career promotion than the benchmark.

**Low n_underrated:** Sparse cohort; widen CI or pool decades for stability.

Compare sources on equal footing: note year coverage (Bayesian 2014–2021 vs full history for BT/Orthogonal).
