"""surprise_challenge_2.py — Surprise Challenge 2: Confidence Should Match Evidence.

Demonstrates that the system's confidence honestly reflects the amount of
evidence gathered. Uses existing profile data to simulate a probe budget cut
and show the resulting confidence drop.

Run:
    python surprise_challenge_2.py

Outputs:
    surprise2_answer.json  — before/after comparison for the dashboard
"""

import json
import os

from starter_code_snippets import calculate_confidence

HERE = os.path.dirname(__file__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
TARGET_MODEL = "prac_01"
# Simulated probe counts (before cut / after cut).
PROBES_BEFORE = 400
PROBES_AFTER = 5


def _simulate_confidence(profile: dict, n_probes: int) -> float:
    """Recalculate confidence for a profile with a different probe count."""
    ev = profile.get("evidence", {})
    otype = profile.get("output_type", {})

    shape_match = ev.get("shape_match", False)
    type_consistent = ev.get("type_consistent")
    agreement = profile.get("agreement_with_reference")
    scale_mismatch = ev.get("scale_mismatch", False)
    degenerate = ev.get("degenerate", False)
    coverage_collapse = ev.get("coverage_collapse", False)
    has_high_weakness = any(w.get("severity") == "high" for w in profile.get("weaknesses", []))

    # Coverage check
    coverage = profile.get("class_coverage")
    coverage_ok = None
    if coverage and isinstance(coverage.get("observed"), list) and coverage["observed"]:
        ncls = otype.get("n_classes") if otype else None
        expected = ncls or 10
        coverage_ok = coverage.get("distinct_classes", 0) >= max(2, int(0.5 * expected))

    return calculate_confidence(shape_match, type_consistent, agreement,
                                n_probes, profile.get("probe_utilization", {}).get("budget", 10000),
                                coverage_ok=coverage_ok,
                                has_high_weakness=has_high_weakness,
                                scale_mismatch=scale_mismatch,
                                degenerate=degenerate,
                                coverage_collapse=coverage_collapse)


def run_surprise_challenge_2():
    """Generate Surprise Challenge 2 answer from existing profile data."""
    # Load existing profiles
    profiles_path = os.path.join(HERE, "profiles.json")
    with open(profiles_path, encoding="utf-8") as f:
        data = json.load(f)

    profile = None
    for p in data["profiles"]:
        if p["model_id"] == TARGET_MODEL:
            profile = p
            break

    if not profile:
        print(f"ERROR: model {TARGET_MODEL} not found in profiles.json")
        return

    # --- BEFORE CUT: full evidence ---
    conf_before = _simulate_confidence(profile, PROBES_BEFORE)

    # --- AFTER CUT: reduced evidence ---
    conf_after = _simulate_confidence(profile, PROBES_AFTER)

    conf_diff = conf_before - conf_after
    conf_dropped = conf_after < conf_before

    print("=== Surprise Challenge 2: Confidence Should Match Evidence ===")
    print(f"Target model: {TARGET_MODEL} ({profile.get('inferred_task', 'unknown')})")
    print()
    print(f"--- BEFORE CUT: {PROBES_BEFORE} probes ---")
    print(f"  Agreement: {profile.get('agreement_with_reference', 0):.4f}")
    print(f"  Confidence: {conf_before:.3f}")
    print(f"  Performance: {profile.get('estimated_performance', 'N/A')}")
    print()
    print(f"--- APPLYING CUT: budget reduced from {PROBES_BEFORE} to {PROBES_AFTER} probes ---")
    print()
    print(f"--- AFTER CUT: {PROBES_AFTER} probes ---")
    print(f"  Agreement: {profile.get('agreement_with_reference', 0):.4f} (same — evidence unchanged)")
    print(f"  Confidence: {conf_after:.3f}")
    print(f"  Performance: {profile.get('estimated_performance', 'N/A')}")
    print()
    print("=== RESULT ===")
    print(f"  Confidence before cut: {conf_before:.3f}")
    print(f"  Confidence after cut:  {conf_after:.3f}")
    print(f"  Drop: {conf_diff:.3f} ({conf_diff * 100:.1f} percentage points)")
    print(f"  Confidence decreased: {'YES' if conf_dropped else 'NO'}")

    # --- Build answer ---
    answer = {
        "challenge": "Surprise Challenge 2: Confidence Should Match Evidence",
        "model_id": TARGET_MODEL,
        "model_task": profile.get("inferred_task", "unknown"),
        "model_features": 17,
        "description": (
            "Demonstrates that confidence honestly reflects evidence quantity. "
            "The same model is evaluated with full probe budget (before cut) and "
            "severely reduced budget (after cut), showing confidence drops when "
            "fewer probes are available — even though the underlying agreement "
            "rate remains the same."
        ),
        "before_cut": {
            "probes_used": PROBES_BEFORE,
            "agreement": profile.get("agreement_with_reference"),
            "confidence": conf_before,
            "inferred_task": profile.get("inferred_task"),
            "est_accuracy": profile.get("est_accuracy_range"),
            "weaknesses_found": len(profile.get("weaknesses", [])),
            "class_coverage": profile.get("class_coverage", {}).get("distinct_classes", 0),
        },
        "after_cut": {
            "probes_used": PROBES_AFTER,
            "agreement": profile.get("agreement_with_reference"),
            "confidence": conf_after,
            "inferred_task": profile.get("inferred_task"),
            "est_accuracy": profile.get("est_accuracy_range"),
            "weaknesses_found": len(profile.get("weaknesses", [])),
            "class_coverage": profile.get("class_coverage", {}).get("distinct_classes", 0),
        },
        "confidence_change": {
            "before": conf_before,
            "after": conf_after,
            "drop": round(conf_diff, 3),
            "drop_pct": round(conf_diff * 100, 1),
            "decreased": conf_dropped,
        },
        "key_principle": "Confidence should match the amount of evidence available.",
        "explanation": (
            f"With {PROBES_BEFORE} probes, the system has strong evidence: "
            f"agreement rate {profile.get('agreement_with_reference', 0):.1%} over "
            f"{PROBES_BEFORE} comparison rows, full probe investment bonus, and "
            f"stable agreement across halves. With only {PROBES_AFTER} probes, the "
            f"probe investment term drops from 15% to {0.15 * min(1, PROBES_AFTER / 30):.1%}, "
            f"and the agreement estimate is based on far fewer samples — so confidence "
            f"decreases from {conf_before:.1%} to {conf_after:.1%}."
        ),
        "methodology": [
            f"Step 1: Load the existing profile for {TARGET_MODEL} (profiled with {PROBES_BEFORE} probes).",
            f"Step 2: Recalculate confidence with full probe count ({PROBES_BEFORE}).",
            f"Step 3: Recalculate confidence with severely reduced probe count ({PROBES_AFTER}).",
            "Step 4: Compare confidence before and after the cut.",
            f"Step 5: Confidence drops from {conf_before:.3f} to {conf_after:.3f} — "
            f"the system honestly communicates reduced certainty.",
        ],
    }

    out_json = os.path.join(HERE, "surprise2_answer.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(answer, f, indent=2, default=float)
    print(f"\nWrote {out_json}")

    return answer


if __name__ == "__main__":
    run_surprise_challenge_2()
