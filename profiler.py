"""profiler.py — Main orchestration script.

Profiles all 13 models following the 6-phase strategy from the README,
respecting per-model probe budgets, and writes:

  - profiles.json      (all model profiles, replaces sample placeholders)
  - probe_usage.json   (per-model probe utilization tracker)

Run:
    python profiler.py            # profile everything
    python profiler.py --quick    # few comparison trials, low probe spend (dry-run/live check)
    python profiler.py --models prac_01 prac_02   # profile only specific models

Phase mapping:
  Phase 0-1: verify API access (uses starter_kit).
  Phase 2:   shape matching (free) + reference comparison (probes).
  Phase 3:   edge case / robustness testing (probes).
  Phase 4:   confidence scoring -> profiles.json.
"""

import argparse
import json
import os
import sys

import config
import starter_kit
from starter_code_snippets import profile_one_model


# ---------------------------------------------------------------------------
# Budget-aware probe allocation
# ---------------------------------------------------------------------------
# How many comparison trials to run per unknown model. Comparison uses
# 2 probes per trial (unknown + reference). Edge-case testing costs a few
# more probes. These stay well inside the 500/150 budgets.
DEFAULT_COMPARISON_TRIALS = 15
QUICK_COMPARISON_TRIALS = 6


def build_model_manifest(quick: bool):
    """Return list of (model_id, n_features, budget, pool_set, trials)."""
    trials = QUICK_COMPARISON_TRIALS if quick else DEFAULT_COMPARISON_TRIALS
    manifest = []

    # References first (calibration phase) — we still probe them lightly to
    # confirm documented behavior, but keep spend tiny relative to 10k budget.
    for mid, spec in config.REFERENCE_MODELS.items():
        manifest.append((mid, spec["n_features"], spec["budget"], "reference", min(trials, 8)))

    # Then practice models (primary characterization targets).
    for mid, spec in config.UNKNOWN_MODELS.items():
        pool = "practice" if mid.startswith("prac") else "held_out"
        t = trials if pool == "practice" else min(trials, 8)
        manifest.append((mid, spec["n_features"], spec["budget"], pool, t))

    return manifest


# ---------------------------------------------------------------------------
# Output writers
# ---------------------------------------------------------------------------
def write_profiles(profiles: list):
    path = os.path.join(os.path.dirname(__file__), "profiles.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(profiles, f, indent=2)
    print(f"\nWrote {len(profiles)} profiles -> {path}")


def record_usage(manifest):
    """Query live usage and persist to probe_usage.json."""
    try:
        usage = starter_kit.get_usage()
    except RuntimeError as exc:
        print(f"WARN: could not fetch usage ({exc}); writing manifest budget info only.")
        usage = {}
    record = {}
    for mid, nfeat, budget, pool, _ in manifest:
        u = usage.get(mid, {})
        used = u.get("used", 0)
        record[mid] = {
            "pool_set": pool,
            "n_features": nfeat,
            "budget": budget,
            "used": used,
            "remaining": max(0, budget - used),
            "percent_used": round((used / budget) * 100, 1) if budget else 0.0,
        }
    path = os.path.join(os.path.dirname(__file__), "probe_usage.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(record, f, indent=2)
    print(f"Wrote probe usage -> {path}")
    return record


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Profile all 13 hackathon models.")
    parser.add_argument("--quick", action="store_true",
                        help="Use fewer probes (quick live check / dry run).")
    parser.add_argument("--models", nargs="*", default=None,
                        help="Only profile these model ids, e.g. --models prac_01 prac_02")
    args = parser.parse_args()

    if config.TEAM_ID == "your_team_id" or config.API_KEY == "your_api_key":
        print("ERROR: Fill in TEAM_ID and API_KEY in config.py first (or set "
              "HACKATHON_TEAM_ID / HACKATHON_API_KEY env vars).", file=sys.stderr)
        sys.exit(1)

    # Phase 0-1: verify connectivity.
    print(">>> Phase 0/1: verifying API access...")
    try:
        models = starter_kit.list_models()
        print(f"    Connected. {len(models)} models advertised by the API.")
    except RuntimeError as exc:
        print(f"ERROR: Could not reach API: {exc}", file=sys.stderr)
        sys.exit(1)

    manifest = build_model_manifest(args.quick)
    if args.models:
        wanted = set(args.models)
        manifest = [m for m in manifest if m[0] in wanted]
        if not manifest:
            print("No matching model ids. Available:", [m[0] for m in build_model_manifest(args.quick)])
            sys.exit(1)

    mode = "QUICK" if args.quick else "FULL"
    print(f">>> Profiling {len(manifest)} models in {mode} mode "
          f"(comparison trials per model)")
    print("    budget summary (probes):")
    total = sum(b for _, _, b, _, _ in manifest)
    print(f"    total probe budget = {total}")
    print(f"    committed comparison probes ~= {sum(t for _, _, _, _, t in manifest) * 2}")

    profiles = []
    for mid, nfeat, budget, pool, trials in manifest:
        try:
            prof = profile_one_model(mid, nfeat, budget, pool, n_comparison_trials=trials)
            profiles.append(prof)
            # Live progress
            print(f"    -> {mid}: {prof['inferred_task']} "
                  f"(conf {prof['task_confidence']:.2f}, "
                  f"{prof['evidence']['comparison_probes']} cmp trials, "
                  f"{len(prof['weaknesses'])} weaknesses)")
        except RuntimeError as exc:
            # Don't crash the whole run; record an honest low-confidence stub.
            print(f"    !! {mid} failed ({exc}); recording placeholder profile")
            profiles.append({
                "model_id": mid,
                "pool_set": pool,
                "inferred_task": "unknown",
                "task_confidence": 0.10,
                "estimated_performance": "Unknown (probe error)",
                "agreement_with_reference": None,
                "weaknesses": [{"type": "probe_error", "description": str(exc), "severity": "high"}],
                "probe_utilization": {"budget": budget, "used": 0, "percent": 0.0},
                "evidence": {"shape_match": False, "comparison_probes": 0, "edge_case_tests": 0},
            })

    write_profiles(profiles)
    record_usage(manifest)
    print("\nDone. Regenerate the dashboard view by reloading profiles.json.")


if __name__ == "__main__":
    main()
