# Marginal Attribution over F1 Relational Graph — Design

**Branch:** `marginal-attribution` (from `master`)
**Date:** 2026-08-17
**Sibling design:** `2026-08-17-counterfactual-swap-design.md`
**Motivating question (MIT Sloan Sports 2027):** "For each race, how much of the outcome came from the driver, the constructor, the engine, and the circuit?"

---

## Positioning

Same starting context as the sibling branch: the Kalman-GNN skill readout did not train, so the `partial_rho = +0.19` from `new-architecture` could not be attributed to a learned driver signal. This branch takes a different angle: instead of intervening on the input (as `counterfactual-swap` does), we decompose the model's output into per-node contributions post hoc via Shapley values.

Where **counterfactual-swap** answers *"what if the driver were someone else"* (intervention), **marginal-attribution** answers *"how much did the driver contribute here"* (attribution). The two are complementary; if both work, they unify into a single MIT Sloan submission where swap acts as the semantic validation of the attribution.

## Question and decomposition

For every race prediction:

```
y_hat = f(driver@T, constructor@T, engine@T, circuit)
```

Decompose into additive per-node contributions using Shapley values:

```
y_hat  ≈  y_baseline + phi_driver + phi_constructor + phi_engine + phi_circuit
```

`phi_i` is the average marginal contribution of node `i` over all coalitions of the other nodes, satisfying the Shapley axioms (efficiency, symmetry, dummy, linearity).

**Driver skill** at the season level is then

```
skill(X, T) = mean over races r in T of phi_driver(X, r)
```

— the average "amount the driver contributed" across the season.

## Base model — shared with sibling branch

The **graph schema and prediction model are identical** to `counterfactual-swap`:

- Meta-nodes `driver_season`, `constructor_season`, `engine_season`, plus static `circuit` and per-race `race`.
- Edges `drives_for`, `same_driver`, `same_constructor`, `raced_in`, `held_at`, `uses_engine`.
- HeteroGNN with 2–3 SAGE/GAT layers.
- Readout: MLP over `[emb(driver_season), emb(constructor_season), emb(engine_season), emb(race), emb(circuit)]` predicting `positionOrder / n_racers`.

**Discipline**: this component is implemented and stabilised first in `counterfactual-swap`, then cherry-picked into `marginal-attribution`. We do not maintain a shared branch — synchronisation cost is not worth the reuse gain at this scale.

## Attribution — primary method (Shapley Monte Carlo)

**Players**: `N = {driver, constructor, engine, circuit}`. Four players is small — Monte Carlo Shapley converges fast.

**Coalition semantics — what does "remove node i" mean?**
Replace `emb(i)` with the **marginal-mean embedding of its type** (the average of all `driver_season` embeddings in the training pool for a driver, etc.). This is the standard SHAP baseline (Lundberg & Lee, 2017) applied to graph embeddings. A "removed" node contributes the type-average signal, not zero — that would bias attribution toward whichever node has a high-magnitude embedding.

**Algorithm** (per prediction):

```
for k = 1..K:
    sample a random permutation pi of N
    for each i in pi (in order):
        S_before = {j in pi placed before i}
        S_after  = S_before ∪ {i}
        phi_i[k] = f(baseline embeddings for N \ S_after, real embeddings for S_after)
                 - f(baseline embeddings for N \ S_before, real embeddings for S_before)
phi_i = mean_k phi_i[k]
```

`K = 200` permutations. Cost: `4 * 200 = 800` forward passes per prediction. With ~10,000 test predictions total, ~8M forwards — feasible on a single GPU in a few hours.

**Efficiency axiom check**: `sum_i phi_i` should equal `y_hat - y_baseline` up to Monte Carlo noise. Assert on a sample of predictions during development.

## Attribution — sanity-check method (structural ablation)

For each node `i`:

```
delta_i = f(real) - f(real with emb(i) replaced by type-mean)
```

Not additive by construction (unlike Shapley), but transparent. Serves as a validator: if Spearman correlation between `phi_i` (Shapley) and `delta_i` (ablation) across the test set is **≥ 0.7**, the two methods agree qualitatively and Shapley is not producing artefacts. If **< 0.5**, the attribution is fragile and the branch is in trouble.

## Aggregation and secondary insight

**Season-level driver skill**:

```
skill(X, T) = mean over r in races(T) where driver_season(X, T) raced_in r:
                  phi_driver(X, r)
```

**Bonus paper insight — decomposition of variance by era**:

```
for each era in {1990s, 2000s, 2010s, 2020s}:
    R2_driver      = Var[phi_driver]      / Var[y_hat]
    R2_constructor = Var[phi_constructor] / Var[y_hat]
    R2_engine      = Var[phi_engine]      / Var[y_hat]
    R2_circuit     = Var[phi_circuit]     / Var[y_hat]
```

Expected pattern (a testable prediction): `R2_constructor` rises across eras (F1 becomes more team-dominated); `R2_driver` falls. This is a *quantitative* claim about how F1 has evolved — Sloan reviewers respond well to headlines like "we measure that the modern F1 result is 65% team, 20% driver, 10% engine, 5% circuit". Whether or not that specific number is right, the framework produces one.

## Support score — shared with sibling branch

Same `support(X, T)` definition and `high/medium/low` bucket as `counterfactual-swap`. Reason: consistency in the paper — the same drivers are flagged as low-support in both branches, so the two attribution methods can be compared honestly on the same subset.

## Contract with the validation framework

```python
load_marginal_attribution_skill() -> DataFrame[
    driverId: int,
    season: int,
    skill_score: float,           # mean phi_driver over season
    support_score: float,
    support_bucket: str,          # high / medium / low
]
```

Plugs into `career_validation.py --skill-source marginal_attribution` unchanged.

## Validation strategy

1. **Shapley ↔ ablation consistency** — Spearman(phi_i, delta_i) ≥ 0.7 across the test set. Gate check; run early.
2. **Framework metrics** — partial ρ ≥ +0.15 with CI excluding zero, above the `constructor_tier` baseline.
3. **Additivity check** — `sum_i phi_i ≈ y_hat - y_baseline` on a held-out sample (efficiency axiom).
4. **Face validity** — top-10 season skill scores dominated by known great drivers on their prime years.
5. **Era decomposition sanity** — R² decomposition curves should be smooth across eras, not noisy year-by-year jumps.

## Exit criteria

Continue if **all four** hold:

- Shapley vs. ablation Spearman ≥ 0.7.
- Additivity residual `mean_r |sum_i phi_i - (y_hat_r - y_baseline)|` < 5% of `Var[y_hat]`.
- Partial ρ ≥ +0.15 on validation framework.
- Era decomposition is monotonic-ish and interpretable (subjective; document what you see).

Kill the branch if any two fail.

## Files to create (in this branch)

```
src/data/temporal_graph.py            # cherry-picked from counterfactual-swap
src/models/hetero_race_predictor.py   # cherry-picked from counterfactual-swap
src/attribution/shapley.py            # Monte Carlo Shapley over graph readout
src/attribution/ablation.py           # single-node structural ablation
src/attribution/aggregate.py          # per-race phi → per-season skill
src/validation/marginal_skill.py      # adapter for career_validation.load_skill
src/experiments/marginal_run.py       # end-to-end runner
src/experiments/era_decomposition.py  # variance decomposition per era
```

## Timeline (indicative)

- **Week 3**: cherry-pick graph and model from `counterfactual-swap` (assumes that branch has stabilised).
- **Week 4**: implement Shapley + ablation; consistency check.
- **Week 5**: aggregation + validation run + era decomposition + face-validity ranking.
- **Week 6 (buffer)**: iterate.

Total: ~3 weeks after `counterfactual-swap` reaches its exit criteria.

## Unification note (if both branches survive)

If `counterfactual-swap` and `marginal-attribution` both exit successfully, the joint paper positions them as:

> "We estimate driver skill via a graph-based counterfactual swap (§Method A). We independently attribute race outcomes to driver, car, engine, and circuit via Shapley decomposition over the same graph (§Method B). The two methods agree at Spearman ρ ≥ 0.X on the driver ranking, validating that the model has learned a consistent driver-vs-car decomposition."

If only one survives, that one is the paper and the other is a rejected-branch note in the appendix or a companion tech report.
