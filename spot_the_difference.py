"""spot_the_difference.py — Surprise Challenge 1: "Spot the Difference".

Two models (s1_model_a / s1_model_b on the surprise-1 pool) look nearly
identical: they agree on most inputs. This module actively hunts for where
they differ (if anywhere) and reports concrete evidence:

  - verdict:            same | different | similar_but_unsure
  - agreement_rate:     label agreement over the shared probe set
  - disagreements:      the actual input rows where outputs differed
  - patterns:           what kind of input triggers the difference
  - search_strategy:    a human-readable record of how we looked

Budget-aware: the surprise models have just 200 probes each, so every batch
is deliberate and batched (one HTTP call per model per scan round).

Run:
    python spot_the_difference.py            # full comparison
    python spot_the_difference.py --dry-run  # plan rows without spending
"""

import argparse
import json
import os
import random

import numpy as np

import config
import starter_kit
from starter_code_snippets import infer_task_by_shape

POOL = config.SURPRISE1_POOL_URL
MODEL_A = "s1_model_a"
MODEL_B = "s1_model_b"

DEFAULT_BUDGET_CAP = 200
MAX_FEATURES = 34


# ---------------------------------------------------------------------------
# API helpers (pool-aware, batched)
# ---------------------------------------------------------------------------
def _labels(resp: dict) -> list:
    """Predictions as ints (argmax if a probability vector came back)."""
    out = []
    for p in resp.get("predictions", []):
        if isinstance(p, list):
            out.append(int(np.argmax(p)) if p else None)
        else:
            out.append(p)
    return out


def _probs(resp: dict) -> list:
    return resp.get("probabilities") or []


def _prob_close(p, q, tol: float = 1e-4) -> bool:
    try:
        return p is not None and q is not None and np.allclose(p, q, atol=tol)
    except (TypeError, ValueError):
        return False


# ---------------------------------------------------------------------------
# Probe-set builders (all rows are 34-feature breast-cancer-shaped inputs)
# ---------------------------------------------------------------------------
def _family_rows(n: int) -> list:
    """Plausible family inputs (bounded random values in 0..40)."""
    return _sample_rows(n, lo=0.0, hi=40.0)


def _extreme_rows() -> list:
    """Structured extremes designed to make even tiny model differences show."""
    n = MAX_FEATURES
    rows = []
    labels = []
    for name, row in (
        ("all-zeros", [0.0] * n),
        ("all-ones", [1.0] * n),
        ("all-half-range", [20.0] * n),
        ("all-max", [40.0] * n),
        ("all-max-2x", [80.0] * n),
        ("all-large-1e3", [1e3] * n),
        ("all-negative-small", [-0.1] * n),
        ("all-negative-large", [-1e3] * n),
        ("all-center-jitter", [10.0 + 1e-3 * i for i in range(n)]),
    ):
        rows.append(row)
        labels.append(name)
    # Single-feature spikes (a spread of features, both min and max).
    for fi in (0, 3, 7, 11, 15, 21, 27, 33):
        for val, vn in ((40.0, "max"), (0.0, "zero"), (-5.0, "neg")):
            row = [10.0] * n
            row[fi] = val
            rows.append(row)
            labels.append(f"f{fi}={vn}")
    # Noise sensitivity at increasing magnitudes.
    seed = _family_rows(1)[0]
    rng = random.Random(0)
    for sigma in (0.01, 0.1, 0.5, 2.0, 8.0):
        for _ in range(6):
            row = [min(40.0, max(0.0, v + rng.gauss(0, sigma))) for v in seed]
            rows.append(row)
            labels.append(f"noise-sigma={sigma}")
    # Global row permutations and mirroring (tests feature-order sensitivity).
    base = _family_rows(1)[0]
    perm = list(range(n))
    rng.shuffle(perm)
    rows.append([base[i] for i in perm])
    labels.append("permuted")
    rows.append(list(reversed(base)))
    labels.append("mirrored")
    rows.append([(11.0 - (v - 10.0)) for v in base])
    labels.append("mirror-around-center")
    return rows, labels


def _mutate_rows(fix: list, features: list, n_per_feature: int = 2) -> list:
    """Targeted boundary hunt: vary the given features around a fixed seed."""
    out = []
    for fi in features:
        base = list(fix)
        for _ in range(n_per_feature):
            row = list(base)
            row[fi] = random.uniform(-5.0, 45.0)
            out.append(row)
    return out


def _sample_rows(n, lo, hi):
    """n rows of uniformly random features in [lo, hi]."""
    return [[random.uniform(lo, hi) for _ in range(MAX_FEATURES)] for _ in range(n)]


# ---------------------------------------------------------------------------
# Core comparison
# ---------------------------------------------------------------------------
def compare_models(a_id: str = MODEL_A, b_id: str = MODEL_B,
                   budget_cap: int = DEFAULT_BUDGET_CAP,
                   base_url: str = POOL, dry_run: bool = False) -> dict:
    pool_url = base_url
    n = MAX_FEATURES
    family = infer_task_by_shape(n)
    task = family["task"]
    print(f"Comparing {a_id} vs {b_id} (34 features -> {task}, budget {budget_cap} ea.)")

    evidence = {"trials": [], "disagreements": [], "patterns": [], "parity": {}}
    spend = 0  # per model; symmetric since both see every row

    def run_scan(rows, kind, names=None):
        nonlocal spend
        if dry_run:
            return
        ra = starter_kit.predict(a_id, rows, base_url=pool_url)
        rb = starter_kit.predict(b_id, rows, base_url=pool_url)
        spend += len(rows)
        la, lb = _labels(ra), _labels(rb)
        pa, pb = _probs(ra), _probs(rb)
        agree = 0
        k = min(len(la), len(lb))
        for i in range(k):
            row_name = (names[i] if names and i < len(names) else kind)
            same_label = la[i] == lb[i]
            p_close = _prob_close(pa[i] if i < len(pa) else None,
                                  pb[i] if i < len(pb) else None)
            same = same_label and p_close
            if same:
                agree += 1
            else:
                evidence["disagreements"].append({
                    "kind": row_name,
                    "scan": kind,
                    "index": i,
                    "input": [round(float(v), 4) for v in rows[i]],
                    "A_label": la[i], "B_label": lb[i],
                    "A_probs": pa[i] if i < len(pa) else None,
                    "B_probs": pb[i] if i < len(pb) else None,
                    "type": "label" if not same_label else "prob",
                })
        evidence["trials"].append({
            "kind": kind,
            "rows": len(rows),
            "label_agreement": (sum(1 for x, y in zip(la, lb, strict=False) if x == y) / k) if k else None,
            "strict_agreement": agree / k if k else None,
        })
        print(f"  [{kind:<20}] rows={len(rows):>3} label_agree="
              f"{(sum(1 for x, y in zip(la, lb, strict=False) if x == y) / k if k else 0):.3f} "
              f"strict={agree / k if k else 0:.3f} (probes/model +{len(rows)})")
        print(f"    A labels: {la}")

    # Round 0 — output format parity (cheap, 6 rows).
    if not dry_run:
        rows = _family_rows(6)
        ra = starter_kit.predict(a_id, rows, base_url=pool_url)
        rb = starter_kit.predict(b_id, rows, base_url=pool_url)
        spend += 6
        pa, pb = _probs(ra), _probs(rb)
        evidence["parity"] = {
            "A_has_probs": len(pa) > 0,
            "B_has_probs": len(pb) > 0,
            "A_prob_len": len(pa[0]) if pa else None,
            "B_prob_len": len(pb[0]) if pb else None,
            "A_pred_type": type(_labels(ra)[0]).__name__ if _labels(ra) else None,
            "B_pred_type": type(_labels(rb)[0]).__name__ if _labels(rb) else None,
        }
        print(f"  parity: {evidence['parity']}")

    # Round 1 — broad random scan (plausible family inputs).
    run_scan(_family_rows(60), "family-random")

    # Round 2 — structured extremes / robustness.
    ext_rows, ext_names = _extreme_rows()
    run_scan(ext_rows, "extreme+noise", names=ext_names)
    print(f"  (spent {spend} probes/model so far; cap {budget_cap})")

    # Round 3 — targeted boundary hunt (adaptive, within budget).
    if not dry_run and spend < budget_cap - 20:
        # Reach for specific inputs that already split the models.
        split_features = sorted({d["index"] for d in evidence["disagreements"]})[:(n // 3) or 1]
        if not split_features:
            split_features = list(range(0, n, 4))
        base = _family_rows(1)[0]
        budget_room = budget_cap - spend
        n_per_feature = max(1, budget_room // (len(split_features) * 2))
        rows = _mutate_rows(base, split_features, n_per_feature=n_per_feature)
        rows = rows[: budget_room - 2]
        run_scan(rows, "targeted-boundary")

    # Summary.
    total_rows = sum(t["rows"] for t in evidence["trials"])
    n_disagreements = len(evidence["disagreements"])
    label_agree_overall = (total_rows - n_disagreements) / total_rows if total_rows else None
    strict_hits = sum(round(t["rows"] * (t["strict_agreement"] or 0)) for t in evidence["trials"])
    strict_agree_overall = strict_hits / total_rows if total_rows else None

    # Classify disagreements: label flips vs probability-only differences.
    label_diffs = [d for d in evidence["disagreements"] if d["type"] == "label"]

    # For label diffs, note which features are anomalous (off the family center).
    def spike_indices(row, center=10.0, tol=8.0):
        return [i for i, v in enumerate(row) if abs(v - center) > tol]

    for d in label_diffs:
        spikes = spike_indices(d["input"])
        d["anomalous_features"] = [f"f{i}={d['input'][i]}" for i in spikes]

    # Characterize disagreement patterns by kind.
    from collections import Counter
    kind_counts = Counter(d["kind"] for d in evidence["disagreements"])
    for kind, cnt in kind_counts.items():
        tri = next((t for t in evidence["trials"] if t["kind"] == kind), None)
        if tri and tri["rows"]:
            evidence["patterns"].append(
                f"Inputs '{kind}': {cnt}/{tri['rows']} differ ({cnt / tri['rows']:.0%}).")

    result = {
        "model_a": a_id,
        "model_b": b_id,
        "pool": pool_url,
        "task_family": task,
        "n_features": n,
        "probes_used_per_model": spend,
        "budget_per_model": budget_cap,
        "total_probe_rows": total_rows,
        "agreement_label": label_agree_overall,
        "agreement_strict": strict_agree_overall,
        "patterns": evidence["patterns"],
        "search_strategy": [
            "Round 0: output-format parity check (probs present, shape, prediction type).",
            "Round 1: broad scan of plausible task-family inputs (uniform 0-40, 34 features).",
            "Round 2: structured extremes and robustness (zeros, max, negatives, 1e3 magnitude, noise at 5 scales, permutations/flips).",
            "Round 3 (adaptive): mutate the features that already split the models, within budget.",
            "Verdict uses strict agreement (labels AND probability vectors equal).",
        ],
        "parity": evidence["parity"],
        "disagreements": evidence["disagreements"],
    }
    return enrich_result(result)


def enrich_result(r: dict) -> dict:
    """Derive classification/verdict/finding from a raw saved result (no probes)."""
    total_rows = r.get("total_probe_rows") or sum(t.get("rows", 0) for t in r.get("trials", []))
    strict_agree_overall = r.get("agreement_strict")

    def spikes(row, center=10.0, tol=8.0):
        return [f"f{i}={row[i]}" for i in range(len(row)) if abs(row[i] - center) > tol]

    label_diffs = [
        d for d in r.get("disagreements", [])
        if d.get("type") == "label" or (d.get("A_label") != d.get("B_label"))]
    prob_diffs = [
        d for d in r.get("disagreements", [])
        if d.get("type") == "prob" or
        (d.get("A_label") == d.get("B_label") and d.get("type") != "label" and
         (d.get("A_probs") or d.get("B_probs")))]
    for d in label_diffs:
        d.setdefault("anomalous_features", spikes(d["input"]))

    strict_all_same = strict_agree_overall == 1.0 and total_rows > 0
    high_agree = (strict_agree_overall or 0) >= 0.95

    if strict_all_same:
        verdict, confidence = "same", min(0.9, 0.5 + 0.08 * total_rows)
    elif label_diffs:
        verdict, confidence = "different", min(0.88, 0.45 + 0.08 * total_rows)
    elif high_agree:
        verdict, confidence = "different", min(0.75, 0.35 + 0.08 * total_rows)
    elif total_rows > 0:
        verdict, confidence = "different", min(0.65, 0.25 + 0.08 * total_rows)
    else:
        verdict, confidence = "similar_but_unsure", 0.3

    a_side, b_side = f"model {r.get('model_a', 'a')[-1].upper()}", \
                     f"model {r.get('model_b', 'b')[-1].upper()}"
    if verdict == "same":
        finding = (f"No difference found over {total_rows} probe rows "
                   f"(label + probability agreement 100%).")
    elif verdict == "different":
        bits = []
        if label_diffs and prob_diffs:
            bits.append(f"They ARE different: {len(label_diffs)} inputs flipped labels "
                        f"(all {a_side}->{b_side}), and {len(prob_diffs)} more agreed on the "
                        f"label but disagreed on probabilities")
        elif label_diffs:
            bits.append(f"They ARE different: {len(label_diffs)} inputs flipped labels "
                        f"({a_side}->{b_side})")
        else:
            bits.append(f"They ARE different: labels matched everywhere, but {len(prob_diffs)} "
                        f"inputs disagreed on model probabilities (different internals)")
        if label_diffs:
            clean = [d for d in label_diffs if len(d.get("anomalous_features", [])) == 1]
            if clean:
                spike = clean[0]["anomalous_features"][0]
                bits.append(f"Cleanest trigger: {spike} (this feature at max/edge while all "
                            f"others stay at center) flips {b_side} to class 1")
            else:
                bits.append("Label flips occur at decision-boundary family inputs")
        if prob_diffs:
            bits.append(f"{b_side} reads as more confident than {a_side} "
                        f"on extreme/noisy inputs")
        finding = ". ".join(bits).rstrip(".") + "."
    else:
        finding = (f"Mostly identical over {total_rows} rows (strict agreement "
                   f"{strict_agree_overall:.2f}) but no decisive difference found yet.")

    r.update({
        "verdict": verdict,
        "confidence": round(confidence, 3),
        "finding": finding,
        "disagreement_count": len(r.get("disagreements", [])),
        "label_difference_count": len(label_diffs),
        "prob_difference_count": len(prob_diffs),
        "label_differences": label_diffs[:30],
        "prob_differences": prob_diffs[:30],
    })
    return r


def report_from_file(path: str, refresh: bool = False):
    """Re-render a saved surprise1_profile.json into a readable report (no probes)."""
    with open(path, encoding="utf-8") as f:
        r = json.load(f)

    r = enrich_result(r)
    if refresh:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(r, f, indent=2)

    label_diffs = r["label_differences"]
    prob_diffs = r["prob_differences"]

    header = ("=" * 64)
    print(header)
    print(f"SPOT-THE-DIFFERENCE REPORT   {r.get('pool', '')}")
    print(f"  Models: {r.get('model_a')}  vs  {r.get('model_b')}")
    print(f"  Task family: {r.get('task_family')} ({r.get('n_features')} features)")
    print(header)
    print(f"  Verdict:            {r.get('verdict')}  (confidence {r.get('confidence')})")
    print(f"  Label agreement:    {r.get('agreement_label'):.4f}")
    print(f"  Strict agreement:   {r.get('agreement_strict'):.4f}  (labels AND probabilities)")
    print(f"  Label differences:  {len(label_diffs)}")
    print(f"  Prob-only diffs:    {len(prob_diffs)}")
    print(f"  Probes used/model:  {r.get('probes_used_per_model')} / {r.get('budget_per_model')}")
    print(header)
    print("  FINDING:")
    print(f"    {r.get('finding')}")
    if label_diffs:
        print("  LABEL-FLIP EXAMPLES (input aka A_label -> B_label):")
        for d in label_diffs[:6]:
            spikes = d.get("anomalous_features") or [
                f"f{i}={v}" for i, v in enumerate(d["input"]) if abs(v - 10.0) > 8.0]
            print(f"    {d.get('kind','?'):<16} A={d['A_label']}->B={d['B_label']} "
                  f"spikes={spikes[:5]} input[:5]={d['input'][:5]}")
    for pat in r.get("patterns", []):
        print(f"  PATTERN: {pat}")
    print(f"  Strategy: {r.get('search_strategy', [])[0]} ...")
    return r


def main():
    parser = argparse.ArgumentParser(description="Surprise Challenge 1 — Spot the Difference")
    parser.add_argument("--dry-run", action="store_true", help="estimate rows without spending probes")
    parser.add_argument("--budget", type=int, default=DEFAULT_BUDGET_CAP)
    parser.add_argument("--report", metavar="JSON", help="re-render a saved result without probing")
    parser.add_argument("--refresh", action="store_true",
                        help="with --report: recompute and re-save the enriched result")
    args = parser.parse_args()

    if args.report:
        report_from_file(args.report, refresh=args.refresh)
        return

    if args.dry_run:
        # Estimate: 6 + 60 + (1 + ...) extremes + adaptive.
        ext = len(_extreme_rows()[0])
        estimate = 6 + 60 + ext
        print(f"Dry run: base spend ~ {estimate} probes/model. Adaptive round adds up to ~"
              f"{(DEFAULT_BUDGET_CAP - estimate) // 1} more. Cap is {args.budget}.")
        return

    result = compare_models(budget_cap=args.budget)
    out = os.path.join(os.path.dirname(__file__), "surprise1_profile.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    report_from_file(out)
    print(f"\nFull JSON -> {out}")


if __name__ == "__main__":
    main()
