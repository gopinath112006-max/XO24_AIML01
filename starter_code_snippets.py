"""starter_code_snippets.py — Copy-paste functions for profiling unknown models.

These are the core primitives used by profiler.py. Each function is
self-contained so you can also import and use them interactively.

    from starter_code_snippets import profile_one_model
    profile = profile_one_model("prac_01", n_features=17, budget=500)

Includes:
  - probe()                — API wrapper (batching-aware)
  - infer_task_by_shape()  — feature count -> task family
  - compare_with_reference() — agreement testing against a matched reference
  - test_edge_cases()      — robustness testing (zeros, extremes, negatives, noise)
  - calculate_confidence() — the README confidence formula
  - generate_profile()     — assemble a JSON-ready profile
  - profile_one_model()    — run the full workflow for one model
"""

import random

import numpy as np

import config

# ---------------------------------------------------------------------------
# Determinism / reproducibility
# ---------------------------------------------------------------------------
random.seed(42)
np.random.seed(42)


# ---------------------------------------------------------------------------
# API wrapper
# ---------------------------------------------------------------------------
def probe(model_id: str, inputs: list, **post_kwargs):
    """Send one or more input rows to a model. Returns the parsed JSON.

    Each row in `inputs` costs 1 probe regardless of batch size.
    """
    import starter_kit  # reuse the retry/helper logic without circular import at module load
    return starter_kit.predict(model_id, inputs, **post_kwargs)


# ---------------------------------------------------------------------------
# Task inference by shape
# ---------------------------------------------------------------------------
def infer_task_by_shape(n_features: int) -> dict:
    """Map a feature count to a task family + reference model (free)."""
    return config.get_family(n_features)


# ---------------------------------------------------------------------------
# Input generators (plausible per-family distributions)
# ---------------------------------------------------------------------------
def _feature_stats(task: str):
    """Rough per-feature scale so generated inputs are plausible."""
    if task == "mnist_digits_10class":      # pixel intensities
        return 0.0, 255.0, 127.0
    if task == "wine_classification_3class":  # standardized-ish values
        return -4.0, 4.0, 0.0
    if task == "breast_cancer_binary":      # positive continuous
        return 0.0, 40.0, 10.0
    if task == "diabetes_progression_regression":
        return -0.2, 0.3, 0.0
    return -1.0, 1.0, 0.0


def _sample_input(n_features: int, task: str, n: int = 1) -> list:
    lo, hi, center = _feature_stats(task)
    out = []
    for _ in range(n):
        if lo >= 0 and "digits" in task:
            row = [round(random.uniform(lo, hi)) for _ in range(n_features)]
        else:
            row = [random.uniform(lo, hi) for _ in range(n_features)]
        out.append(row)
    return out


# ---------------------------------------------------------------------------
# Agreement comparison
# ---------------------------------------------------------------------------
def _as_class(id_pred, prob_ref) -> bool:
    """Heuristic: decide if an unknown output is classification-like."""
    # If the unknown returns probabilities, it's clearly classification.
    return isinstance(id_pred, list) and len(id_pred) >= 1 and isinstance(id_pred[0], (int, float))


def _nonfinite(x) -> bool:
    """True if x is NaN or infinite."""
    try:
        f = float(x)
    except (TypeError, ValueError):
        return True
    return np.isnan(f) or np.isinf(f)


def compare_with_reference(model_id: str, n_features: int, n_trials: int = 15):
    """Send identical inputs to the unknown and its matched reference;
    measure how often their *class/task* outputs agree.

    Returns (agreement_rate, comparisons, observations).
    """
    family = infer_task_by_shape(n_features)
    ref_id = family["reference"]
    task = family["task"]
    is_regression = "regression" in task

    unknown_vals = []
    ref_vals = []
    agreements = 0
    comparisons = []
    for _ in range(n_trials):
        input_rows = _sample_input(n_features, task, n=1)
        try:
            pred_unknown = probe(model_id, input_rows)
            pred_ref = probe(ref_id, input_rows)
        except RuntimeError:
            continue

        unk = pred_unknown.get("predictions", [None])[0]
        ref = pred_ref.get("predictions", [None])[0]

        # Normalise to a scalar for scoring.
        unk_s = int(np.argmax(unk)) if isinstance(unk, list) else unk
        ref_s = int(np.argmax(ref)) if isinstance(ref, list) else ref

        if is_regression:
            # Compare sequences via correlation / relative error rather than
            # exact float equality (two regressors never emit identical floats).
            if unk_s is not None and ref_s is not None and not _nonfinite(unk_s) and not _nonfinite(ref_s):
                unknown_vals.append(float(unk_s))
                ref_vals.append(float(ref_s))
                comparisons.append({"unknown": unk_s, "reference": ref_s, "agree": None})
        else:
            if isinstance(unk_s, (int, float)) and isinstance(ref_s, (int, float)) \
                    and not _nonfinite(unk_s) and not _nonfinite(ref_s):
                agree = (int(unk_s) == int(ref_s))
                agreements += int(agree)
                comparisons.append({"unknown": unk_s, "reference": ref_s, "agree": agree})

    if is_regression and len(unknown_vals) >= 2:
        # Agreement = Pearson correlation of the two prediction series.
        corr = np.corrcoef(unknown_vals, ref_vals)[0, 1]
        if np.isnan(corr):
            corr = 0.0
        # Convert [-1,1] correlation into a 0..1 agreement score.
        rate = max(0.0, float(corr))
        return rate, len(unknown_vals), comparisons

    rate = (agreements / len(comparisons)) if comparisons else 0.0
    return rate, len(comparisons), comparisons


# ---------------------------------------------------------------------------
# Edge case / robustness testing
# ---------------------------------------------------------------------------
def test_edge_cases(model_id: str, n_features: int, task: str):
    """Send extreme / engineered inputs and flag failures.

    Returns list of weakness dicts.
    """
    weaknesses = []
    lo, hi, center = _feature_stats(task)

    cases = {
        "all_zeros": [0.0] * n_features,
        "all_max": [hi] * n_features,
        "all_min": [lo] * n_features,
        "all_negative": [-10.0] * n_features,
        "all_large": [1e6] * n_features,
        "random_noise": list(np.random.normal(center, 0.05 * (hi - lo), n_features)),
    }

    baseline = None
    try:
        baseline = probe(model_id, _sample_input(n_features, task, n=1)).get("predictions", [None])[0]
    except RuntimeError:
        baseline = None

    for name, vec in cases.items():
        try:
            result = probe(model_id, [vec])
            pred = result.get("predictions", [None])[0]
            prob = result.get("probabilities")
        except RuntimeError:
            weaknesses.append({"type": "api_error", "description": f"API error on {name}", "severity": "high"})
            continue

        # NaN detection
        if pred is None or (isinstance(pred, float) and (np.isnan(pred) or np.isinf(pred))):
            weaknesses.append({"type": "nan_or_inf", "description": f"Non-finite output on {name}", "severity": "high"})
        # Extreme flip vs baseline
        elif baseline is not None and isinstance(pred, (int, float)) and isinstance(baseline, (int, float)) \
                and pred != baseline and name in ("all_large", "all_negative"):
            weaknesses.append({
                "type": "extreme_input_drift",
                "description": f"Prediction flips on extreme input ({name})",
                "severity": "medium",
            })

    # Noise sensitivity: add small noise repeatedly, watch for label flips.
    flips = 0
    trials = 5
    base_rows = _sample_input(n_features, task, n=trials)
    base_preds = []
    try:
        base_preds = [probe(model_id, [row]).get("predictions", [None])[0] for row in base_rows]
    except RuntimeError:
        base_preds = []

    for row, bp in zip(base_rows, base_preds):
        noisy = [v + random.gauss(0, 0.03 * (hi - lo)) for v in row]
        try:
            np_ = probe(model_id, [noisy]).get("predictions", [None])[0]
        except RuntimeError:
            continue
        if isinstance(bp, (int, float)) and isinstance(np_, (int, float)) and bp != np_:
            flips += 1

    if trials and flips / trials >= 0.4:
        weaknesses.append({
            "type": "noise_sensitivity",
            "description": f"Prediction changes with small noise on {flips}/{trials} trials",
            "severity": "high" if flips / trials >= 0.6 else "medium",
        })

    return weaknesses


# ---------------------------------------------------------------------------
# Confidence scoring (README formula)
# ---------------------------------------------------------------------------
def calculate_confidence(shape_match: bool, agreement_rate: float,
                         n_comparison_probes: int, n_features: int,
                         has_edge_weaknesses: bool = False) -> float:
    """Implement the README confidence formula:

        Confidence = Shape Match + Agreement Evidence + Probe Investment

        Shape Match      = 0.25 (if features match a reference)
        Agreement        = 0.50 if > 95%
                         = 0.40 if 85-95%
                         = 0.25 if 70-85%
        Probe Investment = 0.15 * min(1.0, probes_on_task / 30)
    """
    score = 0.0

    if shape_match:
        score += 0.25

    if agreement_rate is not None:
        if agreement_rate > 0.95:
            score += 0.50
        elif agreement_rate > 0.85:
            score += 0.40
        elif agreement_rate > 0.70:
            score += 0.25

    score += 0.15 * min(1.0, n_comparison_probes / 30.0)

    # Small honest penalty if we found high-severity weaknesses (less certain of
    # the model matching the reference on hard inputs).
    if has_edge_weaknesses:
        score = max(0.0, score - 0.05)

    # Never claim 100%.
    return round(min(score, 0.99), 3)


def estimate_performance(agreement_rate: float, task: str) -> str:
    if "regression" in task:
        return "Good (R² ~ 0.40-0.50)" if agreement_rate and agreement_rate > 0.85 else "Unknown"
    return "Good (85-95%)" if agreement_rate and agreement_rate > 0.85 else (
        "Fair (70-85%)" if agreement_rate and agreement_rate > 0.70 else "Unknown")


# ---------------------------------------------------------------------------
# Profile assembly
# ---------------------------------------------------------------------------
def generate_profile(model_id: str, n_features: int, budget: int, pool_set: str,
                     agreement_rate, n_comparison_probes: int,
                     weaknesses: list, usage: dict = None) -> dict:
    family = infer_task_by_shape(n_features)
    shape_match = family["reference"] is not None
    conf = calculate_confidence(shape_match, agreement_rate, n_comparison_probes,
                                n_features, has_edge_weaknesses=bool(weaknesses))

    # Usage is best-effort: if we can't fetch it, fall back to budget as an
    # upper bound and report 0 used. Never let a failed usage fetch discard a
    # valid profile (that used to replace good agreement data with a placeholder).
    if usage is None:
        try:
            from starter_kit import get_usage
            usage = get_usage()
        except RuntimeError:
            usage = {}
    used = usage.get(model_id, {}).get("used", 0)
    used = used if isinstance(used, int) else 0

    return {
        "model_id": model_id,
        "pool_set": pool_set,
        "inferred_task": family["task"],
        "task_confidence": conf,
        "estimated_performance": estimate_performance(agreement_rate, family["task"]),
        "agreement_with_reference": agreement_rate,
        "weaknesses": weaknesses,
        "probe_utilization": {
            "budget": budget,
            "used": used,
            "percent": round((used / budget) * 100, 1) if budget else 0.0,
        },
        "evidence": {
            "shape_match": shape_match,
            "comparison_probes": n_comparison_probes,
            "edge_case_tests": len(weaknesses),
        },
    }


# ---------------------------------------------------------------------------
# Full single-model workflow
# ---------------------------------------------------------------------------
def profile_one_model(model_id: str, n_features: int, budget: int, pool_set: str,
                      n_comparison_trials: int = 15, usage: dict = None) -> dict:
    family = infer_task_by_shape(n_features)
    print(f"\n=== Profiling {model_id} (family: {family['task']}) ===")

    rate, n_comp, _ = compare_with_reference(model_id, n_features, n_trials=n_comparison_trials)
    print(f"  agreement with {family['reference']}: {rate:.2f} ({n_comp} trials)")

    weaknesses = test_edge_cases(model_id, n_features, family["task"])
    print(f"  weaknesses detected: {len(weaknesses)}")

    return generate_profile(model_id, n_features, budget, pool_set, rate, n_comp, weaknesses, usage=usage)


if __name__ == "__main__":
    # Example usage (uncomment when credentials + API are live):
    # profile = profile_one_model("prac_01", n_features=17, budget=500)
    # import json; print(json.dumps(profile, indent=2))
    print("Import this module or run profiler.py. Example flagshed above.")
