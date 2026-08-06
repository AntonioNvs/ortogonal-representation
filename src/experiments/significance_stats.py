import itertools
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np


def _percentile_interval(samples: np.ndarray, ci: float) -> Tuple[float, float]:
    alpha = 1.0 - ci
    low = float(np.quantile(samples, alpha / 2.0))
    high = float(np.quantile(samples, 1.0 - alpha / 2.0))
    return low, high


def bootstrap_mean_ci(
    values: Sequence[float],
    n_bootstrap: int = 5000,
    ci: float = 0.95,
    seed: int = 0,
) -> Dict[str, float]:
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return {"mean": float("nan"), "std": float("nan"), "ci_low": float("nan"), "ci_high": float("nan")}

    rng = np.random.default_rng(seed)
    idx = rng.integers(0, arr.size, size=(n_bootstrap, arr.size))
    means = arr[idx].mean(axis=1)
    ci_low, ci_high = _percentile_interval(means, ci=ci)
    return {
        "mean": float(arr.mean()),
        "std": float(arr.std(ddof=1)) if arr.size > 1 else 0.0,
        "ci_low": ci_low,
        "ci_high": ci_high,
        "n": int(arr.size),
    }


def paired_bootstrap_delta_ci(
    a_values: Sequence[float],
    b_values: Sequence[float],
    n_bootstrap: int = 5000,
    ci: float = 0.95,
    seed: int = 0,
) -> Dict[str, float]:
    a = np.asarray(a_values, dtype=float)
    b = np.asarray(b_values, dtype=float)
    if a.size != b.size:
        raise ValueError("paired_bootstrap_delta_ci expects equal-length paired samples.")

    if a.size == 0:
        return {
            "delta_mean": float("nan"),
            "delta_std": float("nan"),
            "ci_low": float("nan"),
            "ci_high": float("nan"),
            "n": 0,
        }

    diffs = a - b
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, diffs.size, size=(n_bootstrap, diffs.size))
    boot_means = diffs[idx].mean(axis=1)
    ci_low, ci_high = _percentile_interval(boot_means, ci=ci)
    return {
        "delta_mean": float(diffs.mean()),
        "delta_std": float(diffs.std(ddof=1)) if diffs.size > 1 else 0.0,
        "ci_low": ci_low,
        "ci_high": ci_high,
        "n": int(diffs.size),
    }


def paired_sign_flip_test(
    a_values: Sequence[float],
    b_values: Sequence[float],
    n_permutations: int = 20000,
    seed: int = 0,
) -> Dict[str, float]:
    a = np.asarray(a_values, dtype=float)
    b = np.asarray(b_values, dtype=float)
    if a.size != b.size:
        raise ValueError("paired_sign_flip_test expects equal-length paired samples.")
    if a.size == 0:
        return {"p_value": float("nan"), "observed_delta": float("nan"), "n": 0}

    diffs = a - b
    obs = abs(float(diffs.mean()))
    n = diffs.size

    rng = np.random.default_rng(seed)
    signs = rng.choice(np.array([-1.0, 1.0]), size=(n_permutations, n))
    permuted = (signs * diffs[None, :]).mean(axis=1)
    count = int(np.sum(np.abs(permuted) >= obs))

    # Add-one smoothing for finite permutation sampling.
    p_value = (count + 1.0) / (n_permutations + 1.0)
    return {"p_value": float(p_value), "observed_delta": float(diffs.mean()), "n": int(n)}


def holm_bonferroni(p_values: Dict[str, float], alpha: float = 0.05) -> Dict[str, Dict[str, float]]:
    ordered = sorted(p_values.items(), key=lambda kv: kv[1])
    m = len(ordered)
    out: Dict[str, Dict[str, float]] = {}

    for i, (name, p_raw) in enumerate(ordered, start=1):
        threshold = alpha / (m - i + 1)
        p_adjusted = min(1.0, p_raw * (m - i + 1))
        out[name] = {
            "p_raw": float(p_raw),
            "p_adjusted": float(p_adjusted),
            "threshold": float(threshold),
            "reject_null": bool(p_raw <= threshold),
        }
    return out


def _extract_metric(run_row: Dict, metric: str) -> float:
    return float(run_row["test_metrics"][metric])


def summarize_models(
    run_rows: Iterable[Dict],
    metric: str = "mae",
    n_bootstrap: int = 5000,
    ci: float = 0.95,
    seed: int = 0,
) -> Dict[str, Dict[str, float]]:
    by_model: Dict[str, List[float]] = {}
    for row in run_rows:
        model = row.get("model_level", row["model_name"])
        by_model.setdefault(model, []).append(_extract_metric(row, metric))

    summary = {}
    for i, (model, values) in enumerate(sorted(by_model.items())):
        summary[model] = bootstrap_mean_ci(
            values,
            n_bootstrap=n_bootstrap,
            ci=ci,
            seed=seed + i,
        )
    return summary


def _paired_metric_vectors(
    run_rows: Iterable[Dict],
    model_a: str,
    model_b: str,
    metric: str = "mae",
) -> Tuple[np.ndarray, np.ndarray]:
    grouped: Dict[Tuple[str, str], Dict[str, float]] = {}
    for row in run_rows:
        metadata = row.get("run_metadata", {})
        split_id = str(metadata.get("split_id", "default"))
        run_id = str(metadata.get("run_id", metadata.get("seed", "0")))
        key = (split_id, run_id)
        grouped.setdefault(key, {})
        grouped[key][row.get("model_level", row["model_name"])] = _extract_metric(row, metric)

    a_vals: List[float] = []
    b_vals: List[float] = []
    for key in sorted(grouped.keys()):
        point = grouped[key]
        if model_a in point and model_b in point:
            a_vals.append(point[model_a])
            b_vals.append(point[model_b])

    return np.asarray(a_vals, dtype=float), np.asarray(b_vals, dtype=float)


def pairwise_significance(
    run_rows: Iterable[Dict],
    model_levels: Sequence[str],
    metric: str = "mae",
    n_bootstrap: int = 5000,
    n_permutations: int = 20000,
    ci: float = 0.95,
    seed: int = 0,
) -> Dict[str, Dict]:
    pairwise: Dict[str, Dict] = {}
    pvals_for_holm: Dict[str, float] = {}

    for i, (model_a, model_b) in enumerate(itertools.combinations(model_levels, 2)):
        pair_name = f"{model_a}_vs_{model_b}"
        a_vals, b_vals = _paired_metric_vectors(run_rows, model_a=model_a, model_b=model_b, metric=metric)
        if a_vals.size == 0:
            pairwise[pair_name] = {"error": "No paired runs found between model levels.", "n": 0}
            continue

        delta = paired_bootstrap_delta_ci(
            a_vals,
            b_vals,
            n_bootstrap=n_bootstrap,
            ci=ci,
            seed=seed + i,
        )
        test = paired_sign_flip_test(
            a_vals,
            b_vals,
            n_permutations=n_permutations,
            seed=seed + 100 + i,
        )
        pairwise[pair_name] = {"delta": delta, "test": test}
        pvals_for_holm[pair_name] = test["p_value"]

    if pvals_for_holm:
        corrections = holm_bonferroni(pvals_for_holm, alpha=0.05)
        for pair_name, correction in corrections.items():
            pairwise[pair_name]["holm"] = correction
    return pairwise


def build_significance_summary(
    run_rows: Iterable[Dict],
    model_levels: Sequence[str],
    metric: str = "mae",
    n_bootstrap: int = 5000,
    n_permutations: int = 20000,
    ci: float = 0.95,
    seed: int = 0,
) -> Dict[str, Dict]:
    model_summary = summarize_models(
        run_rows,
        metric=metric,
        n_bootstrap=n_bootstrap,
        ci=ci,
        seed=seed,
    )
    pairwise = pairwise_significance(
        run_rows,
        model_levels=model_levels,
        metric=metric,
        n_bootstrap=n_bootstrap,
        n_permutations=n_permutations,
        ci=ci,
        seed=seed,
    )
    return {
        "metric": metric,
        "model_summary": model_summary,
        "pairwise": pairwise,
    }
