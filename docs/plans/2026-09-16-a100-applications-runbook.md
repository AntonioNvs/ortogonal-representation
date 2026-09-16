# A100 runbook — GOAT + Undervalued applications (frozen Model A)

**Date:** 2026-09-16
**Model:** frozen Model A — `output/orthogonal_shapley_model/orthogonal_shapley.pth`
(seed 42), the same checkpoint behind `output/validation_benchmark/benchmark.json`.
**These are *applications* of the frozen model, not model changes.** No retraining.

Two scripts were written locally and are already in the branch:

- `src/experiments/run_goat_equal_car.py` — equal-car GOAT counterfactual.
- `src/experiments/run_undervalued_backtest.py` — undervalued backtest + 2026 watchlist.

Both import only existing modules (`baselines.orthogonal_shapley_skill`,
`utils.naming`, `validation.*`) and need the enriched DB + the frozen checkpoint,
which live on the A100.

---

## 0. Sync + preflight (A100)

```bash
git checkout sage-position-regression && git pull

# Only if the enriched DB is absent:
python -m src.data.pipeline build

# Confirm the frozen checkpoint + metadata exist:
ls -la output/orthogonal_shapley_model/orthogonal_shapley.pth \
        output/orthogonal_shapley_model/orthogonal_shapley_meta.json \
        output/orthogonal_shapley_model/coalition_baselines.json
```

**Note on the DB window:** the model was trained over the full enriched graph
(`cfg.MAX_YEAR = 2026`), with the loss masked to train years. The two scripts
control which *years* are exported via `--max-year`. Use `2025` for anything
retrospective; use `2026` only for the prospective watchlist, and see the
cold-start caveat in §4.

---

## 1. Equal-car GOAT

```bash
python src/experiments/run_goat_equal_car.py \
  --checkpoint output/orthogonal_shapley_model/orthogonal_shapley.pth \
  --meta      output/orthogonal_shapley_model/orthogonal_shapley_meta.json \
  --baselines output/orthogonal_shapley_model/coalition_baselines.json \
  --max-year 2025 \
  --min-champ-year 1980 \
  --n-sim 4000 \
  --seed 42
```

**What it does (locked rules):**
1. Detects post-1980 world champions from `standings` (`position == 1` in any
   season ≥ 1980) — no hardcoded list.
2. Picks each champion's *peak season* by an **external** rule (max points share
   of the season, tie-break wins → position) — never by the model's own skill.
3. Reads the frozen model's season skill at that peak (within-race-centred
   driver Shapley contribution).
4. Runs a Plackett–Luce common-car Monte Carlo over the champions' peak skills:
   `utility_i = skill_i + beta·Gumbel(0,1)`, `beta` = median within-(driver,
   season) race-level residual std of the model (data-driven), modern points
   table, independent DNF at 5%.
5. Always runs and reports three negative controls (see `goat_equal_car.json`):
   shift-invariance, symmetry, and a `beta`/DNF sensitivity grid with top-5
   Kendall-tau.

**Outputs** → `output/applications/goat_equal_car/`
- `goat_equal_car.json` — ranking (title prob, expected wins/points with 95% CI),
  controls, sensitivity.
- `goat_equal_car.png` — top-6 expected-points bar chart.

**Read the result:** the title-probability ranking is the GOAT panel. The
negative controls must read `pass: true` before the panel is abstract-grade.

---

## 2. Undervalued drivers — backtest + 2026 watchlist

```bash
python src/experiments/run_undervalued_backtest.py \
  --checkpoint output/orthogonal_shapley_model/orthogonal_shapley.pth \
  --meta      output/orthogonal_shapley_model/orthogonal_shapley_meta.json \
  --baselines output/orthogonal_shapley_model/coalition_baselines.json \
  --max-year 2026 \
  --min-backtest-year 2000 \
  --horizon 3
```

**What it does:**
1. Builds a `(driver, season)` panel with model skill + constructor
   points-share strength (trailing 3-season, lineage-aware).
2. **Rolling-origin backtest:** for every season `T` from 2000, computes
   `uv = z(skill_T) − z(car_T)` and checks whether `uv` predicts moving to a
   stronger team over `T+1..T+horizon`. Reports mean AUROC, precision@k, lift vs
   the base promotion rate.
3. **Prospective watchlist:** same `uv` on the most recent season, reported as a
   forecast.

**Outputs** → `output/applications/undervalued/undervalued.json`

**Gate to pass:** the backtest summary must show `mean_lift > 1` (and ideally
`mean_auroc > 0.5`) before the 2026 watchlist is treated as evidence. A
watchlist with no backtest support is not a result.

---

## 3. Wiring results back into the abstract

The GOAT panel and the 2026 watchlist are *additions* to the already-frozen
abstract core (figure + table). After the A100 runs:

1. Copy the two JSON artifacts back into this repo's `output/applications/`.
2. Regenerate the abstract figure to add a GOAT panel (or keep the current
   3-panel hero figure and add `goat_equal_car.png` as a standalone second
   figure — the submission allows two combined elements).
3. Update `docs/plans/2026-09-16-sloan-abstract.md` Results/Conclusion with the
   GOAT top-5 and (only if the backtest passes) a one-line prospective watchlist
   phrased explicitly as a forecast.

---

## 4. Caveats to keep honest

- **Cold start (2026 rookies):** any driver with no 2026-season state that the
  encoder saw at training time gets an untrained career embedding. The watchlist
  may need a documented exclusion for such drivers, or a `n_races` cutoff.
- **In-sample vs held-out:** the Bayesian SSM row in the table remains
  `smoothed/in-sample`; GOAT and the watchlist are *applications* of the frozen
  model and carry no new held-out claim on their own.
- **β / DNF are fixed choices**, surfaced in the JSON `config` and the
  sensitivity grid — do not tune them against the result.
- If `--max-year 2026` fails at export (checkpoint built over a graph without
  2026 rows), rerun the undervalued script with `--max-year 2025`; the script
  degrades gracefully to the latest available season for the watchlist.
