# Counterfactual Driver-in-Car Swap — Design

**Branch:** `counterfactual-swap` (from `master`)
**Date:** 2026-08-17
**Sibling design:** `2026-08-17-marginal-attribution-design.md`
**Motivating question (MIT Sloan Sports 2027):** "If driver X had driven team Y's car in season T, what outcome would we expect?"

---

## Positioning

The prior `new-architecture` branch trained a Kalman-GNN with an orthogonality prior between `skill_head(v_drivers)` and `context_encoder(qualifying, grid)`. Post-hoc diagnostics showed the `skill_head` barely moved from random init (norm ≈ 0.51 vs. ≈ 0.57 baseline), so the observed `partial_rho = +0.19` could not be honestly attributed to a learned skill readout. This branch abandons that architecture and reframes the driver-skill question as an explicit **causal counterfactual over a heterogeneous temporal graph**.

The branch is one of two parallel exploratory branches (see `marginal-attribution-design.md` for the other). Both branches share the same graph and the same base predictive model; they differ in the causal quantity they extract from it.

## Question and identification

For each `(driver X, season T)`, we estimate the expected outcome of X in an **average car of season T**, marginalising over the constructors actually active in T. Formally:

```
skill(X, T) = mean over Y in constructors_active(T),
              over c in circuits_active(T):
                  f(driver = X@T, constructor = Y@T, circuit = c)
```

The counterfactual is valid to the extent that the graph provides **empirical support** for varying the driver node while holding the constructor slot fixed. Identification rests on the classical *positivity* assumption from causal inference: there must be observed pairs `(X, Y)` and `(X, Y')` (or, symmetrically, `(X, Y)` and `(X', Y)`) somewhere in the training data. Career transfers act as our natural experiments.

## Graph schema

A **single graph spans 2000–2026**, with time encoded via meta-nodes rather than a recurrent state:

**Node types**
- `driver_season` — one node per `(driver, season)` pair that exists. `Hamilton@2015`, `Hamilton@2016`, `Antonelli@2025`.
- `constructor_season` — one node per `(constructor, season)`.
- `circuit` — static; a circuit does not need per-season copies (surface, layout are stable).
- `race` — one node per race, with contextual features (round, calendar position).

> **No engine node.** The enriched Ergast/Jolpica schema has no engine table —
> engine supplier is folded into the constructor entry. If engine-level
> decomposition is ever wanted, it must be injected from an external source.

**Edge types**
- `drives_for(driver_season → constructor_season)` — the modal team of the season.
- `same_driver(driver_season_T → driver_season_{T+1})` — directional, carries a driver's identity forward in time.
- `same_constructor(constructor_season_T → constructor_season_{T+1})` — carries a team's identity forward.
- `raced_in(driver_season → race)` — participation edge; this is the edge the model regresses on.
- `held_at(race → circuit)`.

The `same_driver` and `same_constructor` edges are the mechanism by which knowledge propagates across seasons *without* a recurrent state and *without* forcing every year of a driver into a single embedding. Antonelli@2025 has no `same_driver` predecessor (his first F1 year), which is correct: he has no prior F1 history to carry.

## Base model

**Architecture**: Heterogeneous GNN with 2–3 SAGE layers over the schema above. One propagation step diffuses via `same_driver` and `same_constructor` (temporal); another via `drives_for` (contextual).

**Target**: for each `raced_in(driver_season, race)` edge in the training set, predict `positionOrder / n_racers ∈ [0, 1]` (regression). Simple, dense signal, works even for DNFs (position given at the end).

**Readout**: MLP over `[emb(driver_season), emb(constructor_season), emb(race), emb(circuit)]`. The readout is the point of intervention for the counterfactual.

**Splits**: from `cfg.TRAIN_YEARS` / `VAL_YEARS` / `TEST_YEARS` (extended mode: train 2000–2021, val 2022–2023, test 2024–2026). Directional `same_driver`/`same_constructor` edges enforce causal ordering.

## The counterfactual operation

At **inference** — no retraining — for `skill(X, T)`:

1. Freeze the trained embeddings `emb(X@T)`, `emb(circuit_c)`, `emb(race_r)`.
2. Iterate over `Y in constructors_active(T)` and `c in circuits_active(T)`.
3. For each `(Y, c)`, run the readout with `constructor_season = Y@T` substituted in.
4. Return the mean prediction.

**No embedding is re-learned**; we only redirect the readout to consume a different `constructor_season` node whose embedding was learned from Y's *actual* drivers.

## Support score (v1) and stratified reporting

For each `(X, T)`, compute:

```
support(X, T) =   n_constructors_in_history(X)          [transfers X has made]
                + 0.5 * n_seasons_in_history(X)          [career length]
                + graph_neighborhood_diversity(X@T)      [entropy of Y in 2-hop]
```

Scale is ordinal; bucket into `high / medium / low` via quantiles.

**Paper strategy**:
- **Main ranking table**: `support == high` only. Alonso, Hamilton, Verstappen, Vettel — the veterans on which the swap is well-identified.
- **Secondary table (rookies & one-team drivers)**: `support == low`, marked explicitly as "model projection beyond empirical support". Antonelli, Bearman, Doohan appear here.
- `partial_rho_by_support` bucketed correlations in the validation report so reviewers see the effect within each stratum.

## v2 (documented, not implemented in v1)

Neighbourhood-bootstrap counterfactuals for `support == low` drivers:

- For Antonelli@2025, sample K neighbours in the graph (same age cohort, same F2 background if that feature is available, same current team).
- Run the counterfactual K times using each neighbour's `driver_season` embedding as a proxy prior.
- Report median + IQR instead of a point estimate.

This converts "no data" into "measured uncertainty" and gives reviewers a defensible ranking for rookies.

## Contract with the validation framework

The scorer exposes the standard interface:

```python
load_counterfactual_swap_skill() -> DataFrame[
    driverId: int,
    season: int,
    skill_score: float,          # counterfactual mean over Y, c
    support_score: float,
    support_bucket: str,          # high / medium / low
]
```

Plugs directly into `src/experiments/career_validation.py --skill-source counterfactual_swap` and `career_validation_grid.py`. The framework provides:

- Spearman ρ vs. forward tier outcome.
- Cluster-bootstrap CI by `driverId` (honest, given clustered rows).
- Fisher-z pooled ρ per season.
- Partial ρ residualising on `constructor_tier_score_at_T` — the decisive number.
- Within-season permutation p-value.
- AUROC of "moved up a tier".

## Validation strategy

1. **Held-out swap reconstruction** — reserve 10% of *observed transfers* from the test window. Train without them. Then, at inference, "place" the driver into their real destination team and see whether the predicted outcome matches the observed outcome. Metric: R² over the held-out transfer set.
2. **Framework metrics** — partial ρ ≥ +0.15 with CI excluding zero, above the `constructor_tier` baseline.
3. **Face validity** — Alonso, Hamilton, Verstappen in the top-10 all-time (support == high). Bandeira vermelha if not.

## Exit criteria

Continue if **all three** hold:

- Held-out swap R² ≥ 0.60 on the transfer test set.
- Partial ρ ≥ +0.15 on the validation framework, with cluster-bootstrap CI excluding 0.
- Top-10 ranking passes face validity.

If any of the three fails, the architecture does not decompose driver from car well enough to justify the paper's central claim; kill the branch and record why.

## Files to create (in this branch)

```
src/data/temporal_graph.py            # meta-node graph builder
src/models/hetero_race_predictor.py   # HeteroGNN + edge readout
train_counterfactual.py               # training loop (repo root, like train_kalman.py)
src/counterfactual/swap.py            # inference-time swap and aggregation
src/counterfactual/support.py         # support score
src/validation/counterfactual_skill.py  # adapter for career_validation.load_skill
```

Nothing under `src/models/kalman_*` or `src/data/kalman_dataset.py` is carried over.

## Timeline (indicative)

- **Week 1**: `temporal_graph.py` + graph sanity checks (node/edge counts per season).
- **Week 2**: `hetero_race_predictor.py` + training + baseline (predict-mean-position) comparison.
- **Week 3**: swap inference + support score + first validation run.
- **Week 4 (buffer)**: iterate on identifiability failures, edge cases.

Total: ~3–4 weeks to exit criteria.
