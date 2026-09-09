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
# Output type detection
# ---------------------------------------------------------------------------
def _nonfinite(x) -> bool:
    """True if x is NaN or infinite."""
    try:
        f = float(x)
    except (TypeError, ValueError):
        return True
    return np.isnan(f) or np.isinf(f)


def detect_output_type(model_id: str, n_features: int, task: str, n: int = 5) -> dict:
    """Probe the unknown and determine its output nature.

    Returns dict with:
      - classification: bool (probabilities present / int class labels)
      - n_classes: int or None (prob vector length, if classification)
      - regression: bool
      - scale: (mean, std) of predictions if scalar/regression
    """
    rows = _sample_input(n_features, task, n=n)
    preds = []
    probs = None
    try:
        r = probe(model_id, rows)
        preds = r.get("predictions", [])
        probs = r.get("probabilities")
        if probs is not None and isinstance(probs, list) and probs:
            probs = probs[0]
    except RuntimeError:
        pass

    if not preds:
        return {"classification": None, "n_classes": None, "regression": None,
                "scale": None, "discrete_values": None, "ok": False}

    # Discrete-integer detection: all outputs are integers => almost certainly
    # class labels (a regression that happened to be integral is unlikely).
    ints = [p for p in preds if isinstance(p, int) and not _nonfinite(p)]
    floats = [p for p in preds if isinstance(p, (int, float)) and not _nonfinite(p)]
    discrete_values = None
    if ints and len(ints) == len(preds) and len(set(ints)) <= 10:
        discrete_values = sorted(set(ints))

    is_cls = probs is not None
    n_classes = len(probs) if isinstance(probs, list) else None
    # Integer-only outputs also indicate classification (labels, no probs).
    if not is_cls and discrete_values is not None:
        is_cls = True
        n_classes = n_classes or (len(discrete_values) + 1 if max(discrete_values) - min(discrete_values) <= 9 else len(discrete_values))

    scale = None
    if not is_cls:
        if floats:
            arr = np.array(floats, dtype=float)
            scale = {"mean": round(float(arr.mean()), 4),
                     "std": round(float(arr.std()), 4),
                     "min": round(float(arr.min()), 4),
                     "max": round(float(arr.max()), 4)}

    return {
        "classification": bool(is_cls),
        "n_classes": n_classes,
        "regression": (not is_cls) and scale is not None,
        "scale": scale,
        "discrete_values": discrete_values,
        "ok": True,
    }


def infer_task_by_type(otype: dict, shape_family: dict) -> dict:
    """Decide the task using output-type evidence, cross-checked with shape.

    Returns {task, reference, type_consistent, degenerate, reason}.
    """
    shape_task = shape_family["task"]
    shape_is_reg = "regression" in shape_task
    shape_ncls = {"mnist_digits_10class": 10,
                  "wine_classification_3class": 3,
                  "breast_cancer_binary": 2}.get(shape_task)

    if otype.get("classification") is True:
        # Prefer the shape family's expected class count when outputs are
        # integer labels in that range (e.g. digits 0-9) even if few classes
        # were actually observed.
        dv = otype.get("discrete_values")
        if shape_ncls and dv and max(dv) < shape_ncls:
            ncls = shape_ncls
        else:
            ncls = otype.get("n_classes")
        task = shape_task if not shape_is_reg else _class_task_from_n(ncls)
        consistent = not shape_is_reg
        degenerate = bool(dv and len(dv) == 1)
        return {"task": task, "reference": shape_family["reference"],
                "type_consistent": consistent, "degenerate": degenerate,
                "reason": "classification output"
                           + ("" if otype.get("n_classes") else " (integer labels, no probabilities)")}
    if otype.get("classification") is False and otype.get("regression"):
        task = "diabetes_progression_regression" if not shape_is_reg else shape_task
        consistent = shape_is_reg
        return {"task": task, "reference": shape_family["reference"],
                "type_consistent": consistent, "degenerate": False,
                "reason": "returns scalar values, no probabilities (regression)"}
    return {"task": shape_task, "reference": shape_family["reference"],
            "type_consistent": None, "degenerate": False,
            "reason": "insufficient evidence"}


def _class_task_from_n(n_classes):
    if n_classes in (10,):
        return "mnist_digits_10class"
    if n_classes == 2:
        return "breast_cancer_binary"
    if n_classes == 3:
        return "wine_classification_3class"
    return "classification_unknown"


def class_coverage(model_id: str, n_features: int, task: str, n: int = 30) -> dict:
    """For classifiers, count how many distinct classes are emitted.

    A healthy digit classifier should emit ~10 distinct labels over 30 inputs;
    a collapsed model emitting only 1-2 classes is a weakness.
    """
    rows = _sample_input(n_features, task, n=n)
    seen = set()
    try:
        r = probe(model_id, rows)
        for p in r.get("predictions", []):
            if isinstance(p, (int, float)) and not _nonfinite(p):
                seen.add(int(p))
    except RuntimeError:
        pass
    return {"distinct_classes": len(seen), "observed": sorted(seen)}


# ---------------------------------------------------------------------------
# Agreement comparison (matching feature count)
# ---------------------------------------------------------------------------
def compare_same_family(model_id: str, n_features: int, n_trials: int = 15):
    """Compare an unknown against its shape-matched reference.

    Both share the same feature count, so identical inputs are valid.

    For classification: agreement = fraction of equal labels.
    For regression:     agreement = Pearson correlation after z-scoring both
                        series (scale-invariant), plus a scale-mismatch flag.

    Returns (rate, n_comparisons, comparisons, meta).
    """
    family = infer_task_by_shape(n_features)
    ref_id = family["reference"]
    task = family["task"]
    is_regression = "regression" in task

    unknown_vals = []
    ref_vals = []
    agreements = 0
    comparisons = []
    # Generate all trial inputs up front so we can batch both model calls.
    rows = _sample_input(n_features, task, n=n_trials)
    try:
        pred_unknown = probe(model_id, rows)
        pred_ref = probe(ref_id, rows)
    except RuntimeError:
        pred_unknown = {"predictions": []}
        pred_ref = {"predictions": []}

    unk_preds = pred_unknown.get("predictions", [])
    ref_preds = pred_ref.get("predictions", [])
    n_used = min(len(unk_preds), len(ref_preds), n_trials)
    for i in range(n_used):
        unk = unk_preds[i]
        ref = ref_preds[i]
        if isinstance(unk, list):
            unk = int(np.argmax(unk)) if unk else None
        if isinstance(ref, list):
            ref = int(np.argmax(ref)) if ref else None

        if is_regression:
            if unk is not None and ref is not None and not _nonfinite(unk) and not _nonfinite(ref):
                unknown_vals.append(float(unk))
                ref_vals.append(float(ref))
                comparisons.append({"unknown": round(float(unk), 3), "reference": round(float(ref), 3)})
        else:
            if isinstance(unk, (int, float)) and isinstance(ref, (int, float)) \
                    and not _nonfinite(unk) and not _nonfinite(ref):
                agree = (int(unk) == int(ref))
                agreements += int(agree)
                comparisons.append({"unknown": int(unk), "reference": int(ref), "agree": agree})

    meta = {"scale_mismatch": None, "correlation": None}
    if is_regression and len(unknown_vals) >= 3:
        u = np.array(unknown_vals, dtype=float)
        v = np.array(ref_vals, dtype=float)
        # Demean both (z-score) so correlation is scale/offset invariant.
        us = (u - u.mean()) / (u.std() + 1e-12)
        vs = (v - v.mean()) / (v.std() + 1e-12)
        corr = float(np.corrcoef(us, vs)[0, 1])
        if np.isnan(corr):
            corr = 0.0
        meta = {"correlation": round(corr, 4),
                "scale_mismatch": _scale_mismatch(u, v)}
        rate = max(0.0, corr)
        return rate, len(unknown_vals), comparisons, meta

    rate = (agreements / len(comparisons)) if comparisons else 0.0
    return rate, len(comparisons), comparisons, meta


def _scale_mismatch(u: np.ndarray, v: np.ndarray) -> bool:
    """Flag when unknown and reference regressors use very different output scales."""
    try:
        if u.std() == 0 or v.std() == 0:
            return False
        ratio = u.std() / v.std()
        return ratio < 0.1 or ratio > 10.0
    except Exception:
        return False


# Kept alias for backward compatibility with the README example.
def compare_with_reference(model_id: str, n_features: int, n_trials: int = 15):
    rate, n_comp, comparisons, meta = compare_same_family(model_id, n_features, n_trials)
    return rate, n_comp, comparisons


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

    # Batch all engineered cases in a single call (each case is one row).
    case_names = list(cases.keys())
    case_rows = [cases[n] for n in case_names]
    case_preds = []
    case_probs = None
    try:
        r = probe(model_id, case_rows)
        case_preds = r.get("predictions", [])
        case_probs = r.get("probabilities")
    except RuntimeError:
        case_preds = []

    seen_types = set()
    drift_seen = False
    for i, name in enumerate(case_names):
        pred = case_preds[i] if i < len(case_preds) else None
        prob = case_probs[i] if isinstance(case_probs, list) and i < len(case_probs) else (
            case_probs[0] if isinstance(case_probs, list) and case_probs and i == 0 else None)

        if pred is None:
            if "api_error" not in seen_types:
                weaknesses.append({"type": "api_error", "description": f"API error on {name}", "severity": "high"})
                seen_types.add("api_error")
            continue

        # NaN detection
        if isinstance(pred, float) and (np.isnan(pred) or np.isinf(pred)):
            if "nan_or_inf" not in seen_types:
                weaknesses.append({"type": "nan_or_inf", "description": f"Non-finite output on {name}", "severity": "high"})
                seen_types.add("nan_or_inf")
        # Extreme flip vs baseline (report once, not per case)
        elif baseline is not None and isinstance(pred, (int, float)) and isinstance(baseline, (int, float)) \
                and pred != baseline and name in ("all_large", "all_negative") and not drift_seen:
            drift_seen = True
            weaknesses.append({
                "type": "extreme_input_drift",
                "description": "Prediction flips on extreme out-of-distribution inputs",
                "severity": "medium",
            })
            seen_types.add("extreme_input_drift")

        # Probability sanity: finite and ~sums to 1.
        if isinstance(prob, list) and prob:
            try:
                pvec = np.array(prob, dtype=float)
                if not np.all(np.isfinite(pvec)):
                    if "nonfinite_probs" not in seen_types:
                        weaknesses.append({"type": "nonfinite_probs",
                                           "description": "Probability vector contains NaN/Inf", "severity": "high"})
                        seen_types.add("nonfinite_probs")
                elif not np.isclose(pvec.sum(), 1.0, atol=0.05):
                    if "unnormalized_probs" not in seen_types:
                        weaknesses.append({"type": "unnormalized_probs",
                                           "description": f"Probabilities sum to {pvec.sum():.2f} (not ~1)", "severity": "medium"})
                        seen_types.add("unnormalized_probs")
            except (TypeError, ValueError):
                pass

    # Noise sensitivity: add small noise repeatedly, watch for label flips.
    flips = 0
    trials = 5
    base_rows = _sample_input(n_features, task, n=trials)
    base_preds = []
    try:
        base_preds = probe(model_id, base_rows).get("predictions", [])
    except RuntimeError:
        base_preds = []

    noise_rows = [[v + random.gauss(0, 0.03 * (hi - lo)) for v in row] for row in base_rows]
    noise_preds = []
    try:
        noise_preds = probe(model_id, noise_rows).get("predictions", [])
    except RuntimeError:
        noise_preds = []

    for bp, np_ in zip(base_preds, noise_preds):
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
# Confidence scoring (evidence-based, README-inspired)
# ---------------------------------------------------------------------------
def calculate_confidence(shape_match: bool, type_consistent: bool,
                         agreement_rate: float, n_comparison_probes: int,
                         n_features: int, coverage_ok: bool = None,
                         has_high_weakness: bool = False,
                         scale_mismatch: bool = False,
                         degenerate: bool = False) -> float:
    """Evidence-weighted confidence.

        Type/Shape consistency = 0.20
        Class-coverage bonus   = 0.10 (classifiers only)
        Agreement              = 0.45 (bands below)
        Probe investment       = 0.15 * min(1, probes / 30)

    Agreement bands:
        >0.90 -> 0.45 ; 0.80-0.90 -> 0.36 ; 0.70-0.80 -> 0.25
        0 in regression (scale-disharmony) is left even lower.
    """
    score = 0.0

    if shape_match and type_consistent is not False:
        score += 0.20
    elif shape_match and type_consistent is None:
        score += 0.10

    if coverage_ok is True:
        score += 0.10
    elif coverage_ok is False:
        score += 0.0

    if agreement_rate is not None:
        if agreement_rate > 0.90:
            score += 0.45
        elif agreement_rate > 0.80:
            score += 0.36
        elif agreement_rate > 0.70:
            score += 0.25

    score += 0.15 * min(1.0, n_comparison_probes / 30.0)

    # Honest penalties for detected problems.
    if scale_mismatch:
        score = max(0.0, score - 0.10)
    if degenerate:
        score = max(0.10, score - 0.15)  # constant/degenerate output is suspect
    if has_high_weakness:
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
def generate_profile(model_id: str, pool_set: str, family: dict,
                     agreement_rate, n_comparison_probes: int,
                     weaknesses: list, otype: dict = None,
                     coverage: dict = None, meta: dict = None,
                     usage: dict = None) -> dict:
    n_features = family["n_features"]
    shape_match = family["reference"] is not None
    type_consistent = family.get("type_consistent")
    scale_mismatch = bool(meta and meta.get("scale_mismatch"))
    degenerate = bool(family.get("degenerate"))

    # Class coverage plausibility (classifiers): observed distinct classes
    # should be a reasonable fraction of what the family expects.
    coverage_ok = None
    if coverage is not None and isinstance(coverage.get("observed"), list) and coverage["observed"]:
        ncls = otype.get("n_classes") if otype else None
        expected = ncls or 10
        coverage_ok = coverage.get("distinct_classes", 0) >= max(2, int(0.5 * expected))

    has_high_weakness = any(w.get("severity") == "high" for w in weaknesses)
    conf = calculate_confidence(shape_match, type_consistent, agreement_rate,
                                n_comparison_probes, n_features,
                                coverage_ok=coverage_ok,
                                has_high_weakness=has_high_weakness,
                                scale_mismatch=scale_mismatch,
                                degenerate=degenerate)

    # Usage is best-effort: never let a failed usage fetch discard a valid
    # profile (that used to replace good agreement data with a placeholder).
    if usage is None:
        try:
            from starter_kit import get_usage
            usage = get_usage()
        except RuntimeError:
            usage = {}
    used = usage.get(model_id, {}).get("used", 0)
    used = used if isinstance(used, int) else 0

    expected_ncls = {"mnist_digits_10class": 10,
                     "wine_classification_3class": 3,
                     "breast_cancer_binary": 2}.get(family.get("task"))
    out_ncls = otype.get("n_classes") if otype else None
    if otype and otype.get("classification") and expected_ncls:
        out_ncls = expected_ncls

    return {
        "model_id": model_id,
        "pool_set": pool_set,
        "inferred_task": family["task"],
        "task_confidence": conf,
        "estimated_performance": estimate_performance(agreement_rate, family["task"]),
        "agreement_with_reference": agreement_rate,
        "correlation": (meta or {}).get("correlation"),
        "output_type": {
            "classification": otype.get("classification") if otype else None,
            "n_classes": out_ncls,
            "scale": otype.get("scale") if otype else None,
        } if otype else None,
        "class_coverage": coverage,
        "weaknesses": weaknesses,
        "probe_utilization": {
            "budget": budget_from_family(family),
            "used": used,
            "percent": round((used / budget_from_family(family)) * 100, 1) if budget_from_family(family) else 0.0,
        },
        "evidence": {
            "shape_match": shape_match,
            "type_consistent": type_consistent,
            "comparison_probes": n_comparison_probes,
            "edge_case_tests": len(weaknesses),
            "scale_mismatch": scale_mismatch,
        },
    }


def budget_from_family(family: dict):
    """Extract the probe budget for a family entry (used for display)."""
    return family.get("budget", 10000)


# ---------------------------------------------------------------------------
# Full single-model workflow
# ---------------------------------------------------------------------------
def profile_one_model(model_id: str, n_features: int, budget: int, pool_set: str,
                      n_comparison_trials: int = 15, usage: dict = None) -> dict:
    shape_family = infer_task_by_shape(n_features)
    task = shape_family["task"]
    print(f"\n=== Profiling {model_id} (shape family: {task}) ===")

    # 1. Output-type detection (cheap, decisive signal).
    otype = detect_output_type(model_id, n_features, task, n=5)
    type_family = infer_task_by_type(otype, shape_family)
    print(f"  output type: classification={otype.get('classification')} "
          f"n_classes={otype.get('n_classes')} scale={otype.get('scale')}")

    # 2. Agreement against the shape-matched reference.
    rate, n_comp, _comparisons, meta = compare_same_family(
        model_id, n_features, n_trials=n_comparison_trials)
    print(f"  agreement with {shape_family['reference']}: {rate:.2f} "
          f"({n_comp} trials) scale_mismatch={meta.get('scale_mismatch')}")

    # 3. Class coverage (classifiers) + edge cases.
    coverage = None
    if otype.get("classification"):
        coverage = class_coverage(model_id, n_features, task, n=25)
        print(f"  class coverage: {coverage.get('distinct_classes')} distinct of {otype.get('n_classes') or '?'}")

    weaknesses = test_edge_cases(model_id, n_features, task)
    print(f"  weaknesses detected: {len(weaknesses)}")

    # The family record carries the type-derived task + reference + consistency.
    family_record = dict(shape_family)
    family_record["task"] = type_family["task"]
    family_record["reference"] = type_family["reference"]
    family_record["type_consistent"] = type_family["type_consistent"]
    family_record["n_features"] = n_features
    family_record["budget"] = budget
    family_record["degenerate"] = type_family.get("degenerate", False)

    return generate_profile(model_id, pool_set, family_record, rate, n_comp,
                            weaknesses, otype=otype, coverage=coverage,
                            meta=meta, usage=usage)


if __name__ == "__main__":
    # Example usage (uncomment when credentials + API are live):
    # profile = profile_one_model("prac_01", n_features=17, budget=500)
    # import json; print(json.dumps(profile, indent=2))
    print("Import this module or run profiler.py. Example flagshed above.")
