# Orthogonal Shapley GNN — progress brief

**Date:** 2026-09-10  
**Model:** `output/orthogonal_shapley_model`  
**Benchmark:** `output/validation_benchmark/benchmark.json` (≥2014 fixed cohort, n=226 driver-seasons)

---

## 1. Research question

Formula 1 driver performance is a mixture of intrinsic ability and context (car, team, circuit, era). We ask whether a **relational graph model** can estimate **car-adjusted driver performance** — denoted `f(D,T,R)` — and whether that score predicts **real career outcomes** beyond the quality of the car a driver is in today.

The working hypothesis is a **partially efficient driver market**: better teams tend to hire drivers who have demonstrated higher car-adjusted performance. If the score truly isolates the driver component, it should predict forward team-tier trajectories after controlling for current constructor strength.

**Claim level:** car-adjusted performance. We do not call outputs "pure skill" unless all disentanglement gates pass.

---

## 2. Methodology framework

The pipeline has three layers, all sharing a common validation contract (`SkillExport`):

1. **Causal temporal graph** (`src/data/temporal_graph.py`) — the relational F1 database is encoded as a round-state hetero graph. Edges flow only from past to future; cumulative season scores use races 1…R when scoring round R (no leakage).

2. **OrthogonalShapleyGNN** (`src/models/orthogonal_shapley_gnn.py`) — a 4-layer SAGE encoder fuses driver, constructor, and race context; race outcomes are trained with Plackett–Luce NLL. The exported skill readout is the **coalition Shapley value** of the driver contribution — a principled attribution that decomposes each race into driver / constructor / context shares.

3. **Validation-first gates** (`docs/career_validation_framework.md`) — three headline statistics on a **fixed protocol** (era ≥2014, model-free underrated cohort) compare Orthogonal Shapley against Bradley–Terry (race-level pairwise), Plackett–Luce (race-level listwise), and a Bayesian state-space baseline (Lindner et al.).

**One-sentence result:** On the fixed ≥2014 protocol, Orthogonal Shapley is the only **race-level** model that passes all three headline gates while producing a per-race attribution decomposition that no baseline can.

---

## 3. Headline results

| Stat | Question | Orthogonal Shapley | Bradley–Terry | Bayesian SSM |
|------|----------|-------------------|---------------|--------------|
| **1. Partial ρ (continuous car control)** | Does skill predict future team tier *above current car quality*? | **0.364** [0.152, 0.536] | 0.087 [−0.094, 0.232] | 0.309 [0.086, 0.488] |
| **2. Cox HR (eligible promotion)** | Does higher skill predict reaching a stronger team *sooner*? | **3.91** [1.41, 11.49] | 1.13 [0.77, 1.79] | 4.90 [1.05, 24.20] |
| **3. Locked test 2024–25** | Does the score reproduce held-out race finishing order? | **PL NLL 1.804, pairwise 74.9%** | 1.889, 69.5% | 1.915, 69.3%* |

\*Bayesian SSM is season-level and scored in-sample on locked test — not a fair held-out comparison.

**Promotion discrimination (eligible drivers, n=156):** among driver-seasons below S-tier at T, Orthogonal AUROC **0.725** [0.57, 0.86] vs BT **0.533** [0.39, 0.66] — same 54 promotion events, six times the row count of the B-tier diagnostic below.

**Sharp diagnostic (fixed underrated cohort, n=26):** B-tier overperformers flagged by model-free teammate residual; promotion AUROC **0.679** (Orth) vs **0.405** (BT) on identical rows; paired bootstrap diff CI [0.03, 0.53], one-sided *p* = **0.014**. Small *n*, but cluster-resampled and significant.

**Robustness (sensitivity grid, n=65–147 per cell):** Orthogonal beats BT on within-stratum partial ρ in **9/9** threshold cells and on AUROC in **7/9** cells — stable across `skill_pct` ∈ {0.70, 0.75, 0.80} and S-tier cut ∈ {25%, 30%, 35%}.

**Method signature (Orthogonal only):** mean Shapley variance shares ≈ **35% driver / 42% constructor / 23% context**. BT and PL hard-code context to zero; the Bayesian baseline has no genuine coalition decomposition.

---

## 4. Where we beat baselines

- **Continuous car control (Stat 1):** Orthogonal leads the point partial ρ (0.36) with a bootstrap CI that excludes zero. Bradley–Terry does not exclude zero — its score is largely redundant with current car quality for predicting career trajectory.

- **Promotion timing (Stat 2, n=156):** Among drivers below the top tier, a one-unit increase in skill multiplies the hazard of promotion by ~3.9× (CI excludes 1). BT's CI straddles 1. The KM curves show the top skill tertile promoting materially faster than the bottom tertile (log-rank *p* ≈ 0.007). Eligible promotion AUROC (same population) is 0.73 vs 0.53 for BT — the promotion signal is powered at full sample size, not only in the B-tier diagnostic.

- **Ranking fidelity (Stat 3):** On held-out 2024–25 races, Orthogonal achieves the best PL NLL and pairwise accuracy among walk-forward race-level models — the car-adjusted readout does not sacrifice on-track ranking quality to gain career signal.

- **B-tier overperformer diagnostic (n=26):** A deliberately sharp stratum — high teammate-residual percentile in a backmarker team. Orthogonal separates who gets promoted (AUROC 0.68, paired *p* = 0.01) where BT is near chance (0.41). This is the case-study layer, not the power basis; the headline sits on n=226 / n=156 above.

- **Unique capability:** Per-race Shapley decomposition — the framework's applicability extends beyond a scalar ranking to explain *how much* of each result belongs to the driver vs the car vs context.

---

## 5. Case studies

### Lando Norris — skill rises before the car wins

Orthogonal ranks Norris **#4 in 2024** (skill 0.74), behind only Verstappen, Piastri, and Hamilton. His score rose steadily from **0.09 (2019)** through McLaren's tier climb to **0.74 (2024)** — a trajectory that tracks driver improvement independently of when the team became a title contender. This is the individual-level analogue of the population partial ρ: the model credits Norris's development, not just McLaren's car gain.

### George Russell — team switch under car adjustment

Russell's Orthogonal score moved from **−0.91 (2019, Williams)** to **−0.12 (2024, Mercedes)** — the largest contemporaneous gain in the 2019 cohort, but still below the elite tier after car adjustment. The model's career embedding + season-offset design is built for exactly this: a move to a top team lifts raw results, and the score asks whether the *driver component* moved proportionally.

### Fixed underrated cohort — market inefficiency the model resolves

Among **26 B-tier driver-seasons** (model-free teammate-residual flag, era ≥2014), **12 were later promoted**. Three illustrative promoted cases where Orthogonal and BT diverge:

| Driver | Season | Orth skill | BT skill | Δ (Orth−BT) | Seasons to promotion |
|--------|--------|-----------|----------|-------------|---------------------|
| Charles Leclerc | 2018 | +0.09 | −1.02 | **+1.12** | 1 |
| Pierre Gasly | 2018 | −0.27 | −0.67 | **+0.40** | 1 |
| George Russell | 2019 | −0.95 | −1.51 | **+0.57** | 3 |

Leclerc at Sauber in 2018 is the clearest story: BT scored him negative while Orthogonal scored him positive; he moved to Ferrari the next season. BT's negative score would have missed the promotion; Orthogonal's did not. Full table: `output/plots/cases/underrated_promoted_table.csv`.

---

## 6. Figures

| Figure | Path | What it shows |
|--------|------|---------------|
| Fair-market forest | [fair_market_forest.png](../plots/validation/fair_market_forest/fair_market_forest.png) | Partial ρ + Cox HR across models with bootstrap CIs |
| Time-to-promotion KM | [survival_km_orthogonal_shapley.png](../plots/validation/survival_km_orthogonal_shapley/survival_km_orthogonal_shapley.png) | Top skill tertile promotes faster (Orthogonal) |
| BT null contrast | [survival_km_bradley_terry.png](../plots/validation/survival_km_bradley_terry/survival_km_bradley_terry.png) | Flat KM curves — BT CI straddles HR = 1 |
| Shapley attribution 2024 | [shapley_attribution_2024.png](../plots/validation/shapley_attribution_2024/shapley_attribution_2024.png) | Driver / constructor / context shares per driver (Orth only) |
| Career rank evolution | [rank_evolution_underrated.png](../plots/cases/rank_evolution_underrated/rank_evolution_underrated.png) | Norris, Russell, Piastri 2019–2024 |
| Within-season skill 2024 | [season_skill_2024.png](../plots/cases/season_skill_2024/season_skill_2024.png) | Causal filtered trajectories for Norris & Piastri |

### Appendix (robustness)

| Figure | Path |
|--------|------|
| Sensitivity grid (partial ρ) | [sensitivity_diff_partial_rho.png](../plots/validation/sensitivity_diff_partial_rho/sensitivity_diff_partial_rho.png) |
| Career-channel recoverability probe | [recoverability_probe.png](../plots/validation/recoverability_probe/recoverability_probe.png) |

The recoverability probe confirms the career embedding does not encode constructor identity (held-out AUC ≈ 0.50 vs null p95 ≈ 0.52) — the falsification test for "car-free career channel" passes.

The sensitivity grid (`output/sensitivity_grid/sensitivity_grid.json`) confirms the Orthogonal–BT ordering is not an artifact of a single cohort cut: partial ρ superiority holds across all nine `(skill_pct, p_S)` cells with n ∈ [65, 147].

---

## 7. Limitations

1. **Claim scope** — outputs are car-adjusted performance, not pure intrinsic talent. The season-state offset carries team level by design.

2. **Bayesian baseline** — competitive on partial ρ and Cox HR point estimates; its locked-test ranking is in-sample and not directly comparable.

3. **No telemetry** — claims are bounded by relational race results; strategy, reliability, and pit-wall effects are not isolated.

---

*Generated from `output/orthogonal_shapley_model` and `output/validation_benchmark/benchmark.json`. Reproduce figures with `src/experiments/plots/plot_validation_figures.py` and `plot_entity_attribution.py`.*
