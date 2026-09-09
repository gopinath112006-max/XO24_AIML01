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
import socket
import sys
import time

import config
import starter_kit
from starter_code_snippets import profile_one_model


# ---------------------------------------------------------------------------
# Budget-aware probe allocation
# ---------------------------------------------------------------------------
# How many comparison trials to run per unknown model. Comparison uses
# 2 probes per trial (unknown + reference). Edge-case testing and type
# detection add a few more. Balanced spend: stays well inside the 10k budgets.
DEFAULT_COMPARISON_TRIALS = 35
QUICK_COMPARISON_TRIALS = 10

# Deep mode: stable agreement + accuracy estimate. Generous budgets (10k per
# model) afford hundreds of trials; cap still leaves the reference budget out.
DEEP_COMPARISON_TRIALS = 400
# Keep this many probes untouched on the shared reference model (safety margin).
REFERENCE_BUDGET_MARGIN = 300

# Number of times to retry a whole model before recording a failure stub.
PROFILER_ATTEMPTS = 3
# Seconds to wait between connectivity re-checks / retries.
CONNECTIVITY_RECHECK_S = 15
CONNECTIVITY_MAX_WAIT_S = 120


def _dns_resolves() -> bool:
    """True if the API hostname currently resolves via DNS."""
    try:
        host = config.GENERAL_POOL_URL.split("//")[1].split("/")[0]
        socket.getaddrinfo(host, 443)
        return True
    except socket.gaierror:
        return False


def _wait_for_connectivity():
    """If DNS is down (transient outage), poll until it recovers (up to a cap)."""
    waited = 0
    while not _dns_resolves() and waited < CONNECTIVITY_MAX_WAIT_S:
        print(f"    ... waiting for DNS/network to recover (elapsed {waited}s)")
        time.sleep(CONNECTIVITY_RECHECK_S)
        waited += CONNECTIVITY_RECHECK_S
    # Give the app a moment once DNS is back.
    time.sleep(2)


def build_model_manifest(quick: bool):
    """Return list of (model_id, n_features, budget, pool_set, trials).

    Uses the live API manifest when available (authoritative), falling back
    to the static config tables otherwise.
    """
    trials = QUICK_COMPARISON_TRIALS if quick else DEFAULT_COMPARISON_TRIALS
    manifest = []

    api_manifest = config.load_manifest_from_api()
    if api_manifest:
        # References, then practice, then held-out, preserving API order.
        order = {"reference": 0, "practice": 1, "held_out": 2}
        entries = sorted(
            api_manifest["by_pool"]["reference"]
            + api_manifest["by_pool"]["practice"]
            + api_manifest["by_pool"]["held_out"],
            key=lambda e: (order.get(e["pool_set"], 9), e["model_id"]),
        )
        # References get fewer probing trials (calibration only); practice and
        # held-out unknowns get the full trial count to stabilise agreement.
        t = trials if not quick else min(trials, 8)
        for e in entries:
            is_ref = e["pool_set"] == "reference"
            per = min(t, 8) if is_ref else t
            manifest.append((e["model_id"], e["n_features"], e["budget"], e["pool_set"], per))
        return manifest

    # Fallback to static config tables.
    for mid, spec in config.REFERENCE_MODELS.items():
        manifest.append((mid, spec["n_features"], spec["budget"], "reference", min(trials, 8)))
    for mid, spec in config.UNKNOWN_MODELS.items():
        pool = "practice" if mid.startswith("prac") else "held_out"
        t = trials if pool == "practice" else min(trials, 8)
        manifest.append((mid, spec["n_features"], spec["budget"], pool, t))
    return manifest


# ---------------------------------------------------------------------------
# Output writers
# ---------------------------------------------------------------------------
def _load_existing_profiles(path):
    """Return a list of existing profiles, unwrapping the envelope if present."""
    if not os.path.exists(path):
        return []
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except (ValueError, OSError):
        return []
    if isinstance(data, dict) and isinstance(data.get("profiles"), list):
        return data["profiles"]
    return data if isinstance(data, list) else []


def write_profiles(profiles: list):
    """Persist profiles, preserving already-profiled models on partial runs.

    Writes an envelope {generated_at, profiles} so the dashboard can show a
    freshness timestamp. On ``--models`` runs, only the requested models are
    replaced; everything else keeps its prior profile (deterministic order).
    """
    path = os.path.join(os.path.dirname(__file__), "profiles.json")
    existing = _load_existing_profiles(path)
    replaced = {p["model_id"] for p in profiles}
    merged = [p for p in existing if p["model_id"] not in replaced]
    merged += profiles
    # Deterministic order: reference, practice, held-out, then by id.
    order = {"reference": 0, "practice": 1, "held_out": 9}
    merged.sort(key=lambda p: (order.get(p.get("pool_set"), 9), p.get("model_id", "")))
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                   "profiles": merged}, f, indent=2)
    print(f"\nWrote {len(merged)} profiles -> {path}")
    return merged


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
    parser.add_argument("--deep", action="store_true",
                        help="Deep comparison trials (~400/unknown) for stable agreement "
                             "and accuracy estimates.")
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
    print(f">>>> Profiling {len(manifest)} models in {mode} mode "
          f"(comparison trials per model)")
    print("    budget summary (probes):")
    total = sum(b for _, _, b, _, _ in manifest)
    print(f"    total probe budget = {total}")
    print(f"    committed comparison probes ~= {sum(t for _, _, _, _, t in manifest) * 2}")

    # Fetch usage once, best-effort, so a failure here doesn't hit every model.
    try:
        usage = starter_kit.get_usage()
        print(f"    usage fetched for {len(usage)} models")
    except RuntimeError as exc:
        print(f"    WARN: initial usage fetch failed ({exc}); proceeding without it.")
        usage = {}

    profiles = []
    for mid, nfeat, budget, pool, trials in manifest:
        # Deep trials are budget-aware: keep the shared reference model safe.
        deep_trials = 0
        if args.deep and pool != "reference":
            family = config.get_family(nfeat)
            ref_id = family.get("reference")
            if ref_id:
                ref_used = usage.get(ref_id, {}).get("used", 0)
                ref_remaining = max(0, budget - ref_used)
                deep_trials = min(DEEP_COMPARISON_TRIALS,
                                  max(0, ref_remaining - REFERENCE_BUDGET_MARGIN))
                print(f"    {mid}: deep trials {deep_trials} (ref {ref_id} remaining ~{ref_remaining})")

        last_exc = None
        prof = None
        for attempt in range(1, PROFILER_ATTEMPTS + 1):
            try:
                candidate = profile_one_model(mid, nfeat, budget, pool,
                                              n_comparison_trials=trials,
                                              usage=usage, deep_trials=deep_trials)
                # A successful run should yield at least one comparison probe.
                # Zero probes means the API was unreachable for the whole model
                # (low-level code swallows per-call errors), so treat it as a
                # failed attempt and retry after waiting for connectivity.
                if candidate["evidence"]["comparison_probes"] == 0:
                    prof = None
                    raise RuntimeError("0 successful comparison probes (API unreachable?)")
                prof = candidate
                break
            except RuntimeError as exc:
                last_exc = exc
                if attempt < PROFILER_ATTEMPTS:
                    print(f"    !! {mid} attempt {attempt}/{PROFILER_ATTEMPTS} failed "
                          f"({type(exc).__name__}: {exc}); waiting and retrying...")
                    _wait_for_connectivity()

        if prof is not None:
            profiles.append(prof)
            # Live progress
            print(f"    -> {mid}: {prof['inferred_task']} "
                  f"(conf {prof['task_confidence']:.2f}, "
                  f"{prof['evidence']['comparison_probes']} cmp trials, "
                  f"{len(prof['weaknesses'])} weaknesses)")
        else:
            # All attempts failed; record an honest low-confidence stub.
            print(f"    !! {mid} failed after {PROFILER_ATTEMPTS} attempts "
                  f"({last_exc}); recording infrastructure-error profile")
            profiles.append({
                "model_id": mid,
                "pool_set": pool,
                "inferred_task": "unknown",
                "task_confidence": 0.05,
                "estimated_performance": "Unknown (API unreachable)",
                "agreement_with_reference": None,
                "weaknesses": [{"type": "api_unreachable",
                                "description": f"Could not reach API after retries: {last_exc}",
                                "severity": "high"}],
                "probe_utilization": {"budget": budget, "used": 0, "percent": 0.0},
                "evidence": {"shape_match": False, "comparison_probes": 0, "edge_case_tests": 0},
            })

        # Incremental save protects against a late run-killer loss.
        write_profiles(profiles)

    record_usage(manifest)
    print("\nDone. Regenerate the dashboard view by reloading profiles.json.")


if __name__ == "__main__":
    main()
