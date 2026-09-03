# Hard Identification for the Driver-Skill Readout — Design

**Branch:** `sage-position-regression`
**Date:** 2026-09-03
**Status:** drafted and compile-checked; not yet run on A100

---

## Motivation

The career-validation framework measures whether a skill score predicts a driver's
future team-tier promotion **above** what the current car explains (`partial_rho`,
Cox hazard ratio). On that criterion the Bayesian SSM (`a` — the Lindner driver
ability) currently leads OrthogonalShapleyGNN. The gap is structural, not tuning:

1. The Bayesian identifies driver vs car **hard** (sum-to-zero + team-switch
   leverage); OrthogonalShapleyGNN separates them **soft** (orthogonality penalty
   + a soft target share). The result is a driver state that leaks the constructor
   — the recoverability probe fails (`macro_auc` 0.988 vs null p95 0.515).
2. The Bayesian uses qualifying as a clean pace signal; OrthogonalShapleyGNN uses
   race finish order only.
3. The Bayesian's ability is a GP random-walk (career trajectory);
   OrthogonalShapleyGNN's skill is a static `(driver, season)` node with no
   trajectory prior.

This document specifies the first lever only — **hard identification** — leaving
qualifying and temporal smoothness for later. The goal is to make the driver-skill
readout win on **partial ρ / Cox HR**, not on raw prediction (PL NLL / pairwise
accuracy), which is explicitly out of scope.

---

## 1. Career-shared driver effect (identification by team switch)

### Idea

The "pure" driver skill becomes one vector **per driver**, constant across a
career, disconnected from the constructor graph. The existing GNN node becomes a
per-season *offset*.

```
skill(driver, T) = career_skill(driver) + season_offset(driver, T)
```

Team switches are the causal lever: the same driver performs well at team A and
poorly at team B, so the ranking gradient forces the *constant* part (the driver)
to absorb the driver, and the *constructor* to absorb the team level. This is the
Lindner `a` in GNN form.

### Implementation

1. **New `driver_career` embedding.** `nn.Embedding(n_drivers, hidden_dim)`,
   indexed by `driverId`. It does **not** participate in `HeteroConv`, so it
   receives no constructor edges via message-passing — car-free by construction.
2. **Two-part skill readout.** Replace the single `aux_driver` with
   `aux_driver_career(driver_career_emb[driverId])` and
   `aux_driver_season(driver_state_emb[driver, T])`. The driver *player* value is
   their sum; the player count stays 3 (driver / constructor / context), not 4.
3. **Export plumbing.** `driverId` is already in `res.driver_id`; map it to the
   career embedding index in `export_race_skills`.

### Files

- `src/models/orthogonal_shapley_gnn.py` — new embedding + split heads.
- `src/explain/coalition_shapley.py` — driver player = career + offset.
- `src/baselines/orthogonal_shapley_skill.py` — export the combined driver value.
- `src/explain/orthogonal_shapley_probes.py` — `constructor_recoverability_career_probe`
  (team-switcher-only career-channel leak test).
- `src/experiments/plots/plot_validation_figures.py` — prefer the career probe in
  the recoverability figure.

---

## 2. Hard within-race centering (remove the level)

### Idea

Center the driver skill within each race so it becomes an **intra-race deviation**,
not an absolute utility:

```
skill_centered(i) = skill(i) − mean_{j ∈ race} skill(j)
```

This removes the car/era/circuit *level* by construction — the same demeaning the
Lindner grid already does. It makes the score comparable across seasons and leaves
the partial-ρ residualizer with almost nothing to remove (no statistical inflation,
only structural cleanup).

### Scope (decision)

Center **the driver channel only**. Constructor and context keep their level, which
the residualizer needs. Centering all three would strip the constructor level the
residualizer controls on and artificially inflate ρ.

### Implementation

- Apply in `export_race_skills` as a per-`raceId` mean-subtraction of the driver
  value. Plackett-Luce is invariant to per-race translation, so this is a
  post-hoc export transform that does not conflict with what was optimized. It
  does not break Shapley efficiency (a per-race translation of a player is a
  translation of `v`, not a re-weighting).

### Files

- `src/baselines/orthogonal_shapley_skill.py` — center `raw_skill` by `raceId`;
  keep `contrib_driver/constructor/context` un-centered so attribution stays exact.

---

## 3. Acceptance criteria

Measured on the **common ≥ 2014** window against the Bayesian `a`:

| # | Gate | Pass if | Priority |
|---|------|---------|----------|
| 1 | Career-channel constructor recoverability (team-switchers only) | held-out AUC ≤ null p95 (leakage gone) | **Must** |
| 2 | `partial_rho_continuous` | Orth ≥ Bayesian on the same window | **Must** |
| 3 | Cox HR (eligible) | HR > 1, CI excludes 1, tighter than Bayesian | Should |

Gate 1 is the hard stop: it probes the **career** (car-free) embedding, not the
per-season node, and is restricted to team-switchers (for a career-constant
embedding, "recover the constructor" is only meaningful when the driver changed
teams). If that leak does not collapse, Section 1 did not work and there is no
point reading ρ. Gate 2/3 are the actual "beat the Bayesian" claim.

### Out of scope (not a criterion)

PL NLL and pairwise accuracy. Centering is ranking-invariant, so accuracy should
not move materially; if it does, that is acceptable.

### Known risks

| Risk | Symptom | Mitigation |
|------|---------|------------|
| `career_skill` collapses (~0) | skill variance → 0, ρ → 0 | detach/regularize the head, or amplify team-switch signal |
| Centering removes little | leak AUC stays high | leak was not level-only; re-examine graph edges |
| Offset dominates career part | reverts to static per-season skill | weight the sum or decay the offset |

---

## 4. Honest A/B protocol

- Same seed, same hyperparameters; change **only** the structure (Sections 1–2).
- Report side-by-side: recoverability AUC, `partial_rho_continuous`, Cox HR — on
  `common_2014`.
- One change at a time so the effect is attributable to design, not tuning.

---

## 5. Canonical commands (A100)

```bash
git checkout sage-position-regression && git pull

# Re-fit Bayesian through 2025 (baseline for the common >=2014 window)
python -m src.experiments.run_bayesian_ssm --start-year 2014 --end-year 2025

# Train the hard-identification variant
python src/experiments/train_orthogonal_shapley_gnn.py --seed 42 --use-additive-readout

# Benchmark all three on the common window
python src/experiments/run_validation_benchmark.py \
  --sources bradley_terry bayesian_ssm orthogonal_shapley \
  --horizon inf --era-windows
```

---

## 6. Not yet specified (deferred)

- Qualifying as a second pace signal (Section "add qualifying").
- Temporal smoothness prior on the driver skill (GP random-walk analogue).
- Re-specifying the Shapley story if the driver player becomes career-based (the
  attribution figure must narrate "career skill + season offset", not a static
  season node).
