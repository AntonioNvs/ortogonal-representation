# Model contract — GNN-first driver skill (2026-08-31 reset)

Single source of truth for **what the model predicts**, **what it exports**, and **how we evaluate it**.

---

## Estimand

For driver **D**, team **T**, race **R** (round k of season year):

```text
utility(D,T,R) = driver_channel(driver_state(D,R)) + constructor_channel(constructor_state(T,R)) + grid_effect
f(D,T,R)       = driver_channel(driver_state(D,R))   # exported skill readout
```

- **Higher f = better car-adjusted retrospective performance.**
- **Cumulative season skill** at round r: mean of f(D,T,R) over rounds 1…r only (causal as-of-round).
- Do **not** call outputs "pure skill" unless disentanglement gates pass.

## Architecture (non-negotiable)

All models share the **causal round-state graph** built by [`src/data/temporal_graph.py`](src/data/temporal_graph.py). Message passing uses SAGE convolutions over hetero edges — no tabular race-by-race Python replay.

| Role | Model | File |
|------|-------|------|
| Graph substrate | Causal round-state `HeteroData` | `src/data/temporal_graph.py` |
| **Primary skill GNN** | SkillGNN — PL ranking on race results | `src/models/skill_gnn.py` |
| **GNN baseline (predictive)** | SAGE qualifying regressor (4 layers / 128 hidden) | `src/models/sage_regressor.py` |
| Explainability baseline | Walk-forward Bradley–Terry | `src/baselines/bradley_terry_skill.py` |
| Statistical comparator | Alternating-effects IPF | `src/baselines/bayesian_comparator.py` |
| Simple baseline | Teammate-residual | `src/baselines/teammate_residual.py` |

## What the primary model learns

SkillGNN is trained to predict **race finishing order** via Plackett–Luce NLL on classified finishers. One full-graph forward pass per epoch (batched SAGE, same substrate as SAGE qualifying). Grid enters as a monotonic pre-race covariate. The **driver readout** on `driver_state` embeddings is the exported skill score — trained end-to-end, no detached post-hoc formula.

## What this is NOT

| Quantity | Role |
|----------|------|
| Relational ranker (removed) | Tabular GRU loop — not a GNN; slow; weak convergence |
| Legacy pseudo-GNN (removed) | Tabular embeddings; detached f formula; skill head did not train |
| Raw points / constructor tier | Confounded by car |
| Post-hoc Shapley on mixed embedding | Attribution, not identified skill |

## Lessons (do not repeat)

1. **Pseudo-GNN tabular model** — skill head stayed at random init; export used detached formula.
2. **Relational ranker reset** — violated GNN contract; ~PL 1.78→1.72 over 35 epochs; Python replay per epoch.
3. **Primary career metric** — partial Spearman ρ (skill vs forward tier, controlling tier-at-T), not raw Spearman or all-driver AUROC.
4. **SAGE 4/128** — validated GNN substrate; fast convergence on qualifying grid.
5. **Claims** — "car-adjusted performance" without telemetry; not "pure skill isolated from strategy/reliability".

## Evaluation (headline)

1. **Locked test (2024–2025):** race Plackett–Luce NLL and pairwise accuracy vs walk-forward BT.
2. **Career decomposition:** partial Spearman of season-end f vs forward team tier, controlling tier-at-T (cluster-bootstrap by driver). **Primary gate:** partial ρ ≥ 0.15 and CI low > 0.
3. **Eligible promotion AUROC:** drivers **below top tier at T** only.
4. **Disentanglement (XAI):** constructor leakage probe (`|ρ| < 0.3`), swap invariance on driver readout (`skill_diff < 0.05`). Swap invariance is an **architectural invariant** (separate driver/constructor channels); leakage is the substantive XAI gate.
5. **Robustness:** ≥5 seeds; classified-finish DNF sensitivity.

**Claim levels** (from `evaluate_skill_model.py`):

| Level | Condition |
|-------|-----------|
| `strong_skill` | partial ρ gate **and** leakage gate pass |
| `car_adjusted_performance` | partial ρ passes, leakage fails |
| `insufficient` | partial ρ fails |

**Do not use as career gates:** raw Spearman alone, all-driver moved-up AUROC > 0.7.

## XAI falsification

Probes live in [`src/explain/skill_gnn_probes.py`](src/explain/skill_gnn_probes.py). They run on the test window (2024–2025) over classified finishers.

| Probe | Metric | Gate |
|-------|--------|------|
| Constructor leakage | Spearman(driver readout, ‖constructor_emb‖) | `\|ρ\| < 0.3` |
| Swap invariance | Mean \|Δ skill\| under constructor swap at readout | `< 0.05` (architectural) |
| Channel decomposition | Mean driver share `\|u_d\|/(\|u_d\|+\|u_c\|)` | diagnostic only |

**`xai_report.json` schema:**

```json
{
  "skill_source": "skill_gnn",
  "constructor_leakage_rho": 0.0,
  "swap_invariance": {"skill_diff": 0.0, "utility_swap_delta": 0.0, "n_swaps": 200},
  "channel_decomposition": {"driver_share_mean": 0.0, "constructor_dominates": false},
  "gates": {"constructor_leakage": true, "swap_invariance": true},
  "n_samples": 1000,
  "seed": 42
}
```

Combined career + XAI output: `output/skill_evaluation/evaluation.json` includes `xai`, `gates`, and `claim_level` when `skill_gnn` is evaluated and a checkpoint exists.

## Canonical commands

```bash
# SAGE — GNN predictive baseline (fast, convergent)
python src/experiments/sage_qualifying_run.py --num-layers 4 --hidden-dim 128 --epochs 200 --seed 42

# Primary skill GNN
python src/experiments/train_skill_gnn.py --seed 42

# Explainability baselines — career validation
python src/experiments/career_validation.py --skill-source bradley_terry
python src/experiments/career_validation.py --skill-source bayesian_comparator

# Multi-source evaluation (career + XAI gates)
python src/experiments/evaluate_skill_model.py

# Standalone XAI falsification on SkillGNN
python src/experiments/run_xai_falsification.py --seed 42
```

Outputs: `output/skill_model/skill_gnn.pth`, `output/skill_model/skill_gnn_encoder.pt`, `output/skill_evaluation/evaluation.json`, `output/skill_evaluation/xai_report.json`.

**Checkpoint loading:** inference scripts use `get_skill_gnn_db()` (same `get_dataset` + year window as training). The encoder sidecar `skill_gnn_encoder.pt` is written at train time and pins categorical schema for reload.
