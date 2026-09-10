"""strategy.py — Cross-reference "Compare & Infer" probing strategy.

Extends starter_code_snippets with a systematic approach to comparing unknown
models against references so task inference is evidence-backed and weaknesses
are surfaced early:

  Phase 0 (free)   : shape matching + candidate-reference ranking.
  Phase 1 (cheap)  : output-type detection, then a DEGENERACY SCREEN before any
                     agreement testing. A collapsed model (e.g. always class 4)
                     can otherwise "agree" with a reference by luck and burn
                     deep probes on a meaningless comparison.
  Phase 2 (screen) : ONE shared batch of inputs is sent to the unknown, its
                     shape-matched reference, sub-sliced controls from smaller
                     families (row[:k]), and sibling unknowns with the same
                     feature count. This produces an apples-to-apples agreement
                     vector per candidate, plus probability-level agreement
                     where the API returns probability vectors.
  Phase 3 (decide): a deterministic decision tree turns that agreement vector
                     into a verdict: degenerate | confirmed | re-anchored |
                     ambiguous | off-corpus | unclassified.
  Phase 4 (deep)  : budget-aware follow-up comparison on the adopted family
                     (converged depth when decisive, escalated when ambiguous).
  Phase 5 (weak)  : weakness suite v2 — noise sweep at multiple perturbation
                     levels, cross-family drift inputs, regression constant
                     output check, on top of the original extreme-input tests.
  Phase 6 (report): profile assembled via generate_profile() and enriched with
                     the full strategy audit trail ("strategy" key).

The API-independent helpers (compute_agreement, decide_family, _encode_rows,
_cap_deep_trials) are unit-tested in test_strategy.py without any API calls.

Run standalone:
    python -m src.strategy --model prac_01
    python -m src.strategy --model prac_01 --deep   # enable deep evidence phase
"""

import json
import random
from typing import Any, cast

import numpy as np

from src import config
from src.starter_code_snippets import (
    _feature_stats,
    _nonfinite,
    _sample_input,
    _scale_mismatch,
    class_coverage,
    deep_compare_same_family,
    detect_output_type,
    generate_profile,
    infer_task_by_shape,
    infer_task_by_type,
    probe,
    test_edge_cases,
)

# Re-seed so strategy runs are reproducible with the rest of the pipeline.
random.seed(42)
np.random.seed(42)

# Task families used to pick a "foreign" distribution for the drift test.
_TASK_SCALES = (
    "mnist_digits_10class",
    "wine_classification_3class",
    "breast_cancer_binary",
    "diabetes_progression_regression",
)

# ValueError("unknown encoding: ...") is raised for unsupported encodings.
_ENCODING_NAMES = ("identity", "log1p", "zscore", "minmax")


# ---------------------------------------------------------------------------
# Pure helpers (no API calls — fully unit-testable)
# ---------------------------------------------------------------------------
def compute_agreement(unk_preds, ref_preds, task: str) -> dict:
    """Pure agreement between two prediction lists.

    Classification (task has "regression" is False): fraction of equal labels.
    Regression: Pearson correlation after z-scoring both series (scale/offset
    invariant), clamped to [0, 1] as the agreement rate, with the raw
    correlation kept separately for the record.

    Returns {"agreement", "n", "correlation"}; n counts the paired, finite
    predictions actually compared.
    """
    is_reg = "regression" in task
    pairs = [
        (u, r) for u, r in zip(unk_preds, ref_preds, strict=False)
        if u is not None and r is not None and not _nonfinite(u) and not _nonfinite(r)
    ]
    if not pairs:
        return {"agreement": 0.0, "n": 0, "correlation": None}

    if is_reg:
        u = np.array([float(a) for a, _ in pairs])
        v = np.array([float(b) for _, b in pairs])
        if len(u) < 3:
            return {"agreement": 0.0, "n": len(pairs), "correlation": None}
        us = (u - u.mean()) / (u.std() + 1e-12)
        vs = (v - v.mean()) / (v.std() + 1e-12)
        c = float(np.corrcoef(us, vs)[0, 1])
        if np.isnan(c):
            c = 0.0
        return {"agreement": round(max(0.0, c), 4), "n": len(pairs),
                "correlation": round(c, 4)}

    same = sum(1 for u, r in pairs if int(u) == int(r))
    return {"agreement": round(same / len(pairs), 4), "n": len(pairs),
            "correlation": None}


def decide_family(screen: dict, shape_family: dict) -> dict:
    """Decision tree turning a cross-ref screen into a verdict (pure).

    ``screen`` keys used (all with defaults so it degrades gracefully):
      primary_agreement, max_slice_agreement, max_sibling_agreement,
      degenerate, slices (for re-anchoring).

    Returns {"verdict", "family", "reason", "deep_trials"}. The adopted
    ``family`` carries {task, reference, n_features, ref_n_features} so the
    deep phase knows the reference's slice width.
    """
    def _fam(ref=None, task=None, ref_n=None):
        return {
            "task": task or shape_family.get("task", "unknown"),
            "reference": ref if ref is not None else shape_family.get("reference"),
            "n_features": shape_family.get("n_features"),
            "ref_n_features": ref_n if ref_n is not None else shape_family.get("n_features"),
        }

    primary = screen.get("primary_agreement", 0.0) or 0.0
    slice_ag = screen.get("max_slice_agreement", 0.0) or 0.0
    sib = screen.get("max_sibling_agreement", 0.0) or 0.0
    degenerate = bool(screen.get("degenerate", False))

    confirm = config.STRATEGY_CONFIRM_AGREEMENT
    band_lo = config.STRATEGY_REJECT_AGREEMENT
    ctrl = config.STRATEGY_CONTROL_MAX
    sib_conf = config.STRATEGY_SIBLING_CORROBORATE

    if degenerate:
        return {
            "verdict": "degenerate",
            "family": _fam(),
            "reason": ("model emits a single output class for every screened "
                       "input; agreement-vs-reference is meaningless"),
            "deep_trials": 0,
        }

    if primary >= confirm and slice_ag < ctrl:
        return {
            "verdict": "confirmed",
            "family": _fam(),
            "reason": (f"primary agreement {primary:.2f} \u2265 {confirm} vs "
                       f"{shape_family.get('reference')} and slice controls "
                       f"{slice_ag:.2f} < {ctrl}"),
            "deep_trials": config.STRATEGY_DEEP_CONVERGED,
        }

    if slice_ag >= confirm:
        best = max(screen.get("slices") or [], key=lambda s: s.get("agreement", 0.0))
        fam = _fam(ref=best.get("reference") if best else shape_family.get("reference"),
                   task=best.get("task") if best else shape_family.get("task"),
                   ref_n=best.get("n_features") if best else shape_family.get("n_features"))
        return {
            "verdict": "re-anchored",
            "family": fam,
            "reason": (f"primary agreement {primary:.2f} but sub-slice control "
                       f"{slice_ag:.2f} \u2265 {confirm}; adopting "
                       f"{fam.get('reference')} ({fam.get('task')}) family"),
            "deep_trials": config.STRATEGY_DEEP_AMBIGUOUS,
        }

    if band_lo <= primary < confirm:
        return {
            "verdict": "ambiguous",
            "family": _fam(),
            "reason": f"primary agreement {primary:.2f} in the ambiguous [{band_lo}, {confirm}) band",
            "deep_trials": config.STRATEGY_DEEP_AMBIGUOUS,
        }

    if primary >= confirm and slice_ag >= ctrl:
        return {
            "verdict": "ambiguous",
            "family": _fam(),
            "reason": (f"primary agreement {primary:.2f} is high but slice controls "
                       f"{slice_ag:.2f} \u2265 {ctrl} also elevate; cannot confirm the "
                       f"primary family"),
            "deep_trials": config.STRATEGY_DEEP_AMBIGUOUS,
        }

    if primary < band_lo and sib >= sib_conf:
        return {
            "verdict": "off-corpus",
            "family": _fam(),
            "reason": (f"low reference agreement {primary:.2f} but sibling agreement "
                       f"{sib:.2f} \u2265 {sib_conf}: a same-family cluster outside the "
                       f"reference corpus"),
            "deep_trials": config.STRATEGY_DEEP_CONVERGED,
        }

    return {
        "verdict": "unclassified",
        "family": _fam(),
        "reason": (f"no decisive signal (primary {primary:.2f}, slice {slice_ag:.2f}, "
                   f"sibling {sib:.2f})"),
        "deep_trials": 0,
    }


def _encode_rows(arr: np.ndarray, encoding: str):
    """Apply an input encoding to a (n_rows, n_features) array (pure)."""
    if encoding == "identity":
        return arr
    if encoding == "log1p":
        return np.sign(arr) * np.log1p(np.abs(arr))
    if encoding == "zscore":
        m = arr.mean(axis=0, keepdims=True)
        sd = arr.std(axis=0, keepdims=True)
        return (arr - m) / (sd + 1e-9)
    if encoding == "minmax":
        mn = arr.min(axis=0, keepdims=True)
        mx = arr.max(axis=0, keepdims=True)
        return (arr - mn) / ((mx - mn) + 1e-9)
    raise ValueError(f"unknown encoding: {encoding}")


def _cap_deep_trials(wanted: int, ref_id, usage: dict | None = None) -> int:
    """Bound deep trials so the shared reference budget stays safe (pure)."""
    if not wanted or not ref_id:
        return 0
    if not usage:
        return int(wanted)
    used = usage.get(ref_id, {}).get("used", 0)
    budget = usage.get(ref_id, {}).get("budget", 10000)
    try:
        used = int(used)
        budget = int(budget)
    except (TypeError, ValueError):
        return int(wanted)
    remaining = max(0, budget - used)
    cap = max(0, remaining - config.REFERENCE_BUDGET_MARGIN)
    return min(int(wanted), cap)


# ---------------------------------------------------------------------------
# Cheap probe helpers (API-bound, but small)
# ---------------------------------------------------------------------------
def _probe_labels(model_id: str, rows: list) -> list:
    """Return a list of numeric labels (argmax of probs when needed)."""
    try:
        r = probe(model_id, rows)
    except RuntimeError:
        return []
    out: list = []
    for p in r.get("predictions", []):
        if isinstance(p, list):
            p = int(np.argmax(p)) if p else None
        if p is None or _nonfinite(p):
            out.append(None)
        else:
            out.append(p)
    return out


def _probe_with_probs(model_id: str, rows: list):
    """Return (labels, prob_vectors). prob_vectors is None unless the API
    returns per-row probability lists matching the row count."""
    try:
        r = probe(model_id, rows)
    except RuntimeError:
        return [], None
    labels = []
    for p in r.get("predictions", []):
        if isinstance(p, list):
            labels.append(int(np.argmax(p)) if p else None)
        elif p is None or _nonfinite(p):
            labels.append(None)
        else:
            labels.append(p)
    probs = r.get("probabilities")
    if (isinstance(probs, list) and probs and isinstance(probs[0], list)
            and len(probs) == len(labels)):
        return labels, probs
    return labels, None


def _prob_agree(unk_vec, ref_vec):
    """Mean per-row Pearson correlation between probability vectors (pure)."""
    if not unk_vec or not ref_vec:
        return None
    corrs = []
    for a, b in zip(unk_vec, ref_vec, strict=False):
        x = np.asarray(a, dtype=float)
        y = np.asarray(b, dtype=float)
        if x.shape != y.shape:
            continue
        if x.std() == 0 and y.std() == 0:
            corrs.append(1.0)
            continue
        c = np.corrcoef(x, y)[0, 1]
        if not np.isnan(c):
            corrs.append(float(c))
    return round(float(np.mean(corrs)), 3) if corrs else None


# ---------------------------------------------------------------------------
# Phase 1 — degeneracy screen and type detection
# ---------------------------------------------------------------------------
def screen_degenerate(model_id: str, n_features: int) -> dict:
    """Cheap first check: does the model ever change its output?

    Sends deliberately diverse inputs (family-plausible rows plus all-zeros,
    all-max, all-min, ramp, and reverse-ramp) and counts distinct outputs.
    A constant output means agreement metrics are meaningless, so the caller
    short-circuits before any reference comparison.

    Returns {distinct_classes, observed, constant_class, degenerate, probes}.
    """
    task = infer_task_by_shape(n_features)["task"]
    family_rows = _sample_input(n_features, task, n=config.STRATEGY_DEGENERATE_SCREEN_INPUTS)
    lo, hi, _c = _feature_stats(task)
    extreme = [
        [0.0] * n_features,
        [hi] * n_features,
        [lo] * n_features,
        [float(i) for i in range(n_features)],
        [float(n_features - i) for i in range(n_features)],
    ]
    rows = family_rows + extreme
    preds = [p for p in _probe_labels(model_id, rows) if p is not None]
    observed = sorted({int(p) for p in preds})
    return {
        "distinct_classes": len(observed),
        "observed": observed,
        "constant_class": observed[0] if len(observed) == 1 else None,
        "degenerate": len(observed) == 1,
        "probes": len(rows),
    }


# ---------------------------------------------------------------------------
# Phase 2 — cross-reference screening
# ---------------------------------------------------------------------------
def cross_ref_screen(unknown_id: str, n_features: int,
                     n_trials: int | None = None,
                     siblings: list | None = None) -> dict:
    """Send ONE shared batch of inputs to the unknown and to candidate refs.

    Candidates probed with the SAME rows (apples-to-apples):
      - primary  : the shape-matched reference (exact feature-count family).
      - slices   : references from SMALLER families, fed ``row[:k]`` — catches
                   70-feature unknowns that are really smaller models padded to
                   70 features (a 34-feature model would agree with ref_03 on
                   its first 34 features).
      - siblings : other unknown models sharing the same feature count, which
                   corroborate a same-family cluster even when the reference
                   disagrees (off-corpus detection).

    Returns the agreement vector plus per-candidate probability-level
    agreement where available.
    """
    family = infer_task_by_shape(n_features)
    task = family["task"]
    ref_id = family["reference"]
    n_trials = n_trials or config.STRATEGY_SCREEN_TRIALS
    rows = _sample_input(n_features, task, n=n_trials)

    unk_lbl, unk_vec = _probe_with_probs(unknown_id, rows)

    primary_lbl, primary_vec = _probe_with_probs(ref_id, rows)
    primary = {"reference": ref_id, "task": task,
               **compute_agreement(unk_lbl, primary_lbl, task),
               "prob_agreement": _prob_agree(unk_vec, primary_vec)}

    slices = []
    for cand in config.rank_references(n_features, limit=len(config.FEATURE_TO_FAMILY)):
        if cand["is_primary"] or not cand["sliceable"]:
            continue
        k = cand["n_features"]
        c_rows = [row[:k] for row in rows]
        c_lbl, c_vec = _probe_with_probs(cand["reference"], c_rows)
        slices.append({"reference": cand["reference"], "task": cand["task"],
                       "n_features": k,
                       **compute_agreement(unk_lbl, c_lbl, task),
                       "prob_agreement": _prob_agree(unk_vec, c_vec)})

    if siblings is None:
        siblings = [m for m, s in config.UNKNOWN_MODELS.items()
                    if s["n_features"] == n_features and m != unknown_id]
    siblings_out = []
    for sid in siblings[:2]:
        s_lbl, s_vec = _probe_with_probs(sid, rows)
        siblings_out.append({"model_id": sid,
                             **compute_agreement(unk_lbl, s_lbl, task),
                             "prob_agreement": _prob_agree(unk_vec, s_vec)})

    return {
        "unknown_id": unknown_id,
        "n_features": n_features,
        "task": task,
        "n_trials": n_trials,
        "degenerate_unknown": len({p for p in unk_lbl if p is not None}) == 1,
        "primary": primary,
        "primary_agreement": primary["agreement"],
        "slices": slices,
        "max_slice_agreement": round(max([s["agreement"] for s in slices], default=0.0), 3),
        "siblings": siblings_out,
        "max_sibling_agreement": round(max([s["agreement"] for s in siblings_out], default=0.0), 3),
    }


def encode_sweep(unknown_id: str, n_features: int, ref_id: str, task: str,
                 n_trials: int | None = None, encodings: tuple = _ENCODING_NAMES) -> dict:
    """Rescreen agreement under several input encodings (preprocessing check).

    Only used when the default screen produced no decisive agreement (e.g. the
    unknown expects standardized inputs while the family generator sends raw
    plausible values). The SAME transform is applied to the shared rows before
    they go to both the unknown and the reference, so any rise in agreement is
    attributable to the encoding.

    Returns per-encoding results plus the best encoding and its agreement.
    """
    n_trials = n_trials or config.STRATEGY_SCREEN_TRIALS
    rows = _sample_input(n_features, task, n=n_trials)
    arr = np.asarray(rows, dtype=float)
    per = []
    for enc in encodings:
        x = _encode_rows(arr, enc)
        unk_lbl = _probe_labels(unknown_id, x.tolist())
        ref_lbl = _probe_labels(ref_id, x.tolist())
        agg = compute_agreement(unk_lbl, ref_lbl, task)
        per.append({"encoding": enc, **agg})
    best = max(per, key=lambda e: e.get("agreement", 0.0))
    return {"per_encoding": per,
            "best_encoding": best["encoding"],
            "best_agreement": round(best["agreement"], 3)}


# ---------------------------------------------------------------------------
# Phase 4 — deep, budget-aware agreement on the adopted family
# ---------------------------------------------------------------------------
def deep_get_agreement(model_id: str, n_features: int, task: str, ref_id: str,
                       ref_n_features: int | None = None, n_trials: int = 150,
                       batch: int = 96) -> dict:
    """Stable batched agreement between an unknown and one reference.

    Rows are generated with the unknown's feature count; the reference consumes
    ``row[:ref_n_features]`` when it expects fewer (sub-slice/re-anchored
    families), else the full row. Mirrors deep_compare_same_family's return
    shape so generate_profile() can consume it directly.

    Reuses deep_compare_same_family when no slicing is needed.
    """
    ref_n_features = ref_n_features or n_features
    if ref_n_features >= n_features:
        return dict(deep_compare_same_family(model_id, n_features, task,
                                             ref_id, n_trials=n_trials, batch=batch))

    from collections import Counter, defaultdict

    is_reg = "regression" in task
    k = ref_n_features
    rows_all = _sample_input(n_features, task, n=n_trials)
    unk_vals, ref_vals = [], []
    agree = 0
    n_used = 0
    conf: defaultdict = defaultdict(Counter)
    dis: list = []
    h1a, h1n, h2a, h2n = 0, 0, 0, 0

    idx = 0
    while idx < len(rows_all):
        chunk = rows_all[idx: idx + batch]
        try:
            ru = probe(model_id, chunk)
            rr = probe(ref_id, [row[:k] for row in chunk])
        except RuntimeError:
            break
        pu = ru.get("predictions", [])
        pr = rr.get("predictions", [])
        for i, (u, v) in enumerate(zip(pu, pr, strict=False)):
            if isinstance(u, list):
                u = int(np.argmax(u)) if u else None
            if isinstance(v, list):
                v = int(np.argmax(v)) if v else None
            if u is None or v is None or _nonfinite(u) or _nonfinite(v):
                continue
            first = idx + i < len(rows_all) / 2
            n_used += 1
            if is_reg:
                unk_vals.append(float(u))
                ref_vals.append(float(v))
                continue
            same = int(u) == int(v)
            agree += int(same)
            conf[int(u)][int(v)] += 1
            if first:
                h1a += int(same)
                h1n += 1
            else:
                h2a += int(same)
                h2n += 1
            if not same and len(dis) < 12:
                dis.append({"unknown": int(u), "reference": int(v),
                            "input": [round(float(x), 3) for x in chunk[i][:8]]})
        idx += batch

    if is_reg:
        if n_used >= 3:
            u = np.array(unk_vals, dtype=float)
            v = np.array(ref_vals, dtype=float)
            us = (u - u.mean()) / (u.std() + 1e-12)
            vs = (v - v.mean()) / (v.std() + 1e-12)
            c = float(np.corrcoef(us, vs)[0, 1])
            if np.isnan(c):
                c = 0.0
            mae = float(np.mean(np.abs(u - v)))
            return {"agreement": round(max(0.0, c), 4), "n": n_used,
                    "half1": round(max(0.0, c), 4), "half2": round(max(0.0, c), 4),
                    "confusion": {}, "disagreement_examples": None,
                    "meta": {"correlation": round(c, 4),
                             "mae_vs_reference": round(mae, 4),
                             "scale_mismatch": _scale_mismatch(u, v)}}
        return {"agreement": 0.0, "n": n_used, "half1": None, "half2": None,
                "confusion": {}, "disagreement_examples": None, "meta": {}}

    if n_used == 0:
        return {"agreement": 0.0, "n": 0, "half1": None, "half2": None,
                "confusion": {}, "disagreement_examples": [], "meta": {}}

    rate = agree / n_used
    conf_out = {int(kk): {int(k2): int(c2) for k2, c2 in d.items()}
                for kk, d in conf.items()}
    return {"agreement": round(rate, 4), "n": n_used,
            "half1": round(h1a / h1n, 4) if h1n else None,
            "half2": round(h2a / h2n, 4) if h2n else None,
            "confusion": conf_out,
            "disagreement_examples": dis,
            "meta": {"correlation": None, "scale_mismatch": None}}


# ---------------------------------------------------------------------------
# Phase 5 — weakness suite v2
# ---------------------------------------------------------------------------
def weakness_suite_v2(model_id: str, n_features: int, task: str) -> list:
    """Weakness detection v2.

    Extends test_edge_cases() with:
      * a noise-sensitivity SWEEP at 1% / 5% / 10% of the feature span,
      * a cross-family drift test (inputs drawn from another task's scale),
      * a regression constant-output check,
    plus the original extreme-input / probability-sanity checks. Uses the same
    weakness schema so profiles stay uniform.
    """
    weaknesses = list(test_edge_cases(model_id, n_features, task))
    seen = {w["type"] for w in weaknesses}

    lo, hi, _c = _feature_stats(task)
    span = (hi - lo) or 1.0
    trials = 5
    base_rows = _sample_input(n_features, task, n=trials)
    base_preds = _probe_labels(model_id, base_rows)
    if len([b for b in base_preds if b is not None]) < 2:
        return weaknesses

    for level in (0.01, 0.05, 0.10):
        noise_rows = [[v + random.gauss(0, level * span) for v in row] for row in base_rows]
        noise_preds = _probe_labels(model_id, noise_rows)
        flips = sum(1 for b, n in zip(base_preds, noise_preds, strict=False)
                    if b is not None and n is not None and b != n)
        wtype = f"noise_sensitivity_{int(level * 100)}pct"
        if wtype not in seen and flips >= max(2, int(0.4 * trials)):
            weaknesses.append({
                "type": wtype,
                "description": (f"Prediction changes under {int(level * 100)}% perturbation "
                                f"on {flips}/{trials} trials"),
                "severity": "high" if level >= 0.05 and flips / trials >= 0.6 else "medium",
            })
            seen.add(wtype)

    other = next((t for t in _TASK_SCALES if t != task), task)
    flo, fhi, _ = _feature_stats(other)
    drift_rows = [[random.uniform(flo, fhi) for _ in range(n_features)] for _ in range(trials)]
    drift_preds = _probe_labels(model_id, drift_rows)
    drift_flips = sum(1 for b, n in zip(base_preds, drift_preds, strict=False)
                      if b is not None and n is not None and b != n)
    if "family_drift_sensitivity" not in seen and drift_flips >= max(2, int(0.5 * trials)):
        weaknesses.append({
            "type": "family_drift_sensitivity",
            "description": (f"Output changes on {drift_flips}/{trials} trials under "
                            f"out-of-family distribution ({other})"),
            "severity": "medium",
        })
        seen.add("family_drift_sensitivity")

    if "regression" in task:
        vals = [float(b) for b in base_preds if b is not None and not _nonfinite(b)]
        if vals and np.std(vals) == 0 and "constant_regression_output" not in seen:
            weaknesses.append({
                "type": "constant_regression_output",
                "description": f"Regression output is constant ({vals[0]:.3f}) across input trials",
                "severity": "high",
            })

    return weaknesses


# ---------------------------------------------------------------------------
# Phase 6 — full strategy orchestrator
# ---------------------------------------------------------------------------
def run_strategy(model_id: str, n_features: int, budget: int, pool_set: str,
                 usage: dict | None = None, deep_ok: bool = True,
                 siblings: list | None = None) -> dict:
    """Full "Compare & Infer" profile for one model.

    Phases: type -> degeneracy -> cross-ref screen -> decide -> deep (if
    warranted and enabled) -> weakness suite v2 -> coverage -> report.

    Returns a profile schema-compatible with profile_one_model() and enriched
    with a "strategy" audit trail (verdict, screen, deep, probe ledger).
    """
    shape_family = infer_task_by_shape(n_features)
    shape_family["n_features"] = n_features
    task = shape_family["task"]
    deep = None
    sweep = None
    n_compare = 0
    probes_ledger = {}

    print(f"\n=== [strategy] Profiling {model_id} (shape family: {task}) ===")

    # Phase 1a: output-type detection (cheap, decisive signal).
    otype = detect_output_type(model_id, n_features, task, n=5)
    type_family = infer_task_by_type(otype, shape_family)
    probes_ledger["type_detect"] = 5
    print(f"  output type: classification={otype.get('classification')} "
          f"n_classes={otype.get('n_classes')} scale={otype.get('scale')}")

    # Phase 1b: degeneracy screen BEFORE any agreement probing.
    degen = screen_degenerate(model_id, n_features)
    probes_ledger["degeneracy"] = degen["probes"]
    degenerate = degen["degenerate"]
    print(f"  degeneracy screen: {degen['distinct_classes']} distinct output(s) "
          f"-> {'DEGENERATE' if degenerate else 'ok'}")

    # Phase 2: cross-reference screen (skipped entirely for degenerate models —
    # agreement is meaningless there and would waste reference probes).
    screen = None
    if degenerate:
        screen = {
            "unknown_id": model_id, "n_features": n_features, "task": task,
            "n_trials": 0, "degenerate_unknown": True,
            "primary_agreement": 0.0, "max_slice_agreement": 0.0,
            "max_sibling_agreement": 0.0,
            "primary": {"reference": shape_family.get("reference"), "agreement": 0.0,
                        "prob_agreement": None, "n": 0},
            "slices": [], "siblings": [],
            "degenerate": True,
        }
    else:
        screen = cross_ref_screen(model_id, n_features, siblings=siblings)
        screen["degenerate"] = False
        probes_ledger["screen"] = (screen["n_trials"]
                                   * (1 + len(screen["slices"]) + len(screen["siblings"])))
        print(f"  cross-ref screen vs {screen['primary']['reference']}: "
              f"agreement={screen['primary']['agreement']:.3f} "
              f"(slices max={screen['max_slice_agreement']:.2f}, "
              f"siblings max={screen['max_sibling_agreement']:.2f})")

    # Phase 3: decisions.
    decision = decide_family(screen, shape_family)
    verdict = decision["verdict"]
    family = decision["family"]
    print(f"  decision: {verdict} -> {family.get('task')}/ref={family.get('reference')}")

    # Phase 4: budget-aware deep follow-up.
    deep_trials_used = 0
    if deep_ok and verdict in ("confirmed", "ambiguous", "off-corpus", "re-anchored"):
        wanted = decision["deep_trials"]
        ref_id = family.get("reference")
        deep_trials = _cap_deep_trials(wanted, ref_id, usage)
        if deep_trials and ref_id:
            deep_task = family.get("task", task)
            deep = deep_get_agreement(model_id, n_features, deep_task, ref_id,
                                      family.get("ref_n_features"), deep_trials)
            deep_trials_used = deep_trials
            probes_ledger["deep"] = deep.get("n", 0) * 2
            print(f"  deep agree vs {ref_id}: {deep['agreement']:.4f} on {deep.get('n')} rows "
                  f"(half1={deep.get('half1')}, half2={deep.get('half2')})")
            if deep.get("meta", {}).get("scale_mismatch"):
                print(f"  !! scale mismatch flagged vs {ref_id}")

    # Phase 4b: encoding sweep for otherwise-unclassified models.
    if verdict == "unclassified" and shape_family.get("reference"):
        sweep = encode_sweep(model_id, n_features, shape_family["reference"], task)
        probes_ledger["encode_sweep"] = len(sweep["per_encoding"]) * screen["n_trials"]
        print(f"  encoding sweep best: {sweep['best_encoding']} "
              f"(agreement {sweep['best_agreement']:.3f})")

    # Effective agreement + comparison-probe count feeding the confidence.
    eff_agreement = deep["agreement"] if deep else (
        screen["primary"]["agreement"] if screen and screen.get("primary") else 0.0)
    n_compare = deep["n"] if deep else (
        (screen.get("primary") or {}).get("n", 0))
    if sweep and sweep["best_agreement"] > eff_agreement:
        eff_agreement = sweep["best_agreement"]

    # Phase 5: weaknesses (v2 suite) + class coverage for classifiers.
    weaknesses = weakness_suite_v2(model_id, n_features, task)
    coverage = None
    if otype.get("classification"):
        if degenerate:
            coverage = {"distinct_classes": 1,
                        "observed": degen.get("observed", [])}
            if not any(w["type"] == "degenerate_output" for w in weaknesses):
                weaknesses.append({
                    "type": "degenerate_output",
                    "description": (f"Model only outputs class "
                                    f"{degen.get('observed', ['?'])[0]} for all tested inputs"),
                    "severity": "high",
                })
        else:
            coverage = class_coverage(model_id, n_features, task, n=25)
            probes_ledger["coverage"] = 25
    probes_ledger["weaknesses"] = len(weaknesses)

    # Collapsed coverage penalty mirroring profile_one_model's logic.
    expected_cls = {"mnist_digits_10class": 10, "wine_classification_3class": 3,
                    "breast_cancer_binary": 2}.get(family["task"], 10)
    coverage_collapse = False
    if (otype.get("classification") and coverage is not None
            and not degenerate and expected_cls
            and coverage.get("distinct_classes", 0) <= max(1, int(0.3 * expected_cls))):
        coverage_collapse = True
        if not any(w["type"] == "low_class_coverage" for w in weaknesses):
            weaknesses.append({
                "type": "low_class_coverage",
                "description": (f"Only {coverage['distinct_classes']} of ~{expected_cls} "
                                f"classes observed (collapsed output)"),
                "severity": "medium",
            })

    # Build the family record the profile generator expects.
    family_record = {
        "task": family["task"],
        "reference": family["reference"],
        "type_consistent": type_family["type_consistent"],
        "n_features": n_features,
        "budget": budget,
        "degenerate": degenerate,
        "coverage_collapse": coverage_collapse,
    }

    meta = {"correlation": (deep or {}).get("meta", {}).get("correlation"),
            "scale_mismatch": bool((deep or {}).get("meta", {}).get("scale_mismatch"))}
    profile = generate_profile(model_id, pool_set, family_record,
                               eff_agreement, n_compare, weaknesses,
                               otype=cast(Any, otype), coverage=cast(Any, coverage),
                               meta=cast(Any, meta), usage=cast(Any, usage),
                               deep=cast(Any, deep))

    probes_total = sum(v for v in probes_ledger.values())
    profile["strategy"] = {
        "verdict": verdict,
        "family_adopted": family,
        "reason": decision["reason"],
        "decision": decision,
        "screen": screen,
        "deep": deep,
        "encode_sweep": sweep,
        "deep_trials_used": deep_trials_used,
        "probe_ledger": probes_ledger,
        "probes_spent_estimate": probes_total,
    }
    return profile


def main():
    import argparse
    import sys

    parser = argparse.ArgumentParser(description="Run the Compare & Infer probing strategy.")
    parser.add_argument("--model", required=True, help="model_id to profile")
    parser.add_argument("--deep", action="store_true",
                        help="enable the budget-aware deep agreement phase")
    args = parser.parse_args()

    manifest = config.load_manifest_from_api()
    entry = (manifest or {"models": {}}).get("models", {}).get(args.model)
    if entry:
        n_features = entry["n_features"]
        budget = entry["budget"]
        pool_set = entry["pool_set"]
    else:
        spec = cast("dict[str, object] | None",
                    {**config.REFERENCE_MODELS, **config.UNKNOWN_MODELS}.get(args.model))
        if not spec:
            print(f"Unknown model: {args.model}", file=sys.stderr)
            sys.exit(1)
        n_features = cast("int", spec["n_features"])
        budget = cast("int", spec["budget"])
        if args.model in config.REFERENCE_MODELS:
            pool_set = "reference"
        elif args.model.startswith("prac"):
            pool_set = "practice"
        else:
            pool_set = "held_out"

    profile = run_strategy(args.model, n_features, budget, pool_set,
                           usage=None, deep_ok=args.deep)
    print(json.dumps(profile, indent=2))


if __name__ == "__main__":
    main()
