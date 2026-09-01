# Retrospective Driver-Skill Ranking — Design

**Branch:** `sage-position-regression` (extends with skill-ranking modules)  
**Date:** 2026-08-31  
**Status:** implementation spec

---

## Estimand

For driver D, team T, race R (round k of season year):

```
f(D,T,R) = posterior driver contribution after R
         = driver_state_pre(R) + shrunk_driver_innovation(R)
```

- **Higher f = better performance** (ranks transformed accordingly).
- **Cumulative season skill** at round r: mean of race-level f over rounds 1…r only.
- **Uncertainty:** bootstrap or posterior interval; never report a point estimate alone for low-support drivers.

## What this is NOT

| Quantity | Meaning |
|----------|---------|
| SAGE qualifying prediction | Pre-race expected grid position (predictive, not retrospective) |
| Post-hoc Shapley on mixed embedding | Model attribution, not identified driver effect |
| Beat-teammate head on contaminated embedding | Constructor signal may live in driver embedding |
| Raw points / constructor tier | Confounded by car |

## Identification (current relational data only)

Requires:
1. **Teammate comparisons** — same constructor, different drivers, same race.
2. **Transfers** — same driver, different constructors across seasons.
3. **Sum-to-zero / centering** — per-race or per-season constraints on driver + constructor effects.
4. **Separate channels** — driver state never receives constructor messages in the skill readout path.

Cannot remove without telemetry: strategy, tire deg, weather, damage, pit ops, reliability DNFs.

## Model architecture (primary)

Round-state causal graph (`driver_state`, `constructor_state`, …) with:

```
utility(D,T,R) = u_driver(D,R) + u_constructor(T,R) + u_context(R) + u_grid + interaction_centered
```

- Joint **qualifying** and **race** ranking likelihoods (Plackett–Luce / pairwise).
- Race conditions on grid position.
- Gated state updates after each race; driver innovation = within-team differential vs constructor mean.

## DNF policy

- **Primary:** classified finishers only (`position` not null) for race ranking.
- **Sensitivity:** all entries via `positionOrder`; status-filtered reliability subset.

## Splits

Extended mode (config): train 1950–2021, val 2022–2023, test 2024–2025 (headline).  
2026 = demo/live only until season complete.

## Validation gates (must pass before strong claims)

1. Beat dynamic Bradley–Terry on held-out ranking NLL / teammate ordering.
2. Driver score stable under constructor swap intervention.
3. Constructor not recoverable from driver state above leakage threshold.
4. Acceptable uncertainty coverage; robust across seeds and DNF policies.

## Related work (mandatory comparators)

- Dynamic Bayesian driver/constructor state-space model (Aug 2026, arXiv:2608.04629)
- Bayesian rank-ordered logit (Van Kesteren & Bergkamp, JQAS 2022)
- Walk-forward Bradley–Terry (this repo, `src/baselines/bradley_terry.py`)

**Status:** implementation spec — **implement on `temporal_graph.py`**, not a tabular ranker.

## Implementation note (2026-08-31 GNN reset)

The primary model lives on the causal round-state graph:

```
src/data/temporal_graph.py       causal hetero graph (shared substrate)
src/models/skill_gnn.py          primary SkillGNN (PL on race results)
src/models/sage_regressor.py     SAGE qualifying baseline
src/models/ranking_likelihood.py Plackett-Luce losses
src/baselines/                   BT, bayesian_comparator, teammate_residual
src/validation/                  career framework + inference
src/explain/                     attribution + falsification (SkillGNN hooks pending)
src/skill/scoring.py             cumulative season skill helpers
src/visualization/driver_rankings.py  plots
```
