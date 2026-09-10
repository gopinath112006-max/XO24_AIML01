"""probe_boundary.py — Pin the exact 'spot the difference' trigger on surprise pool 1.

Initial full sweep (already paid for) found the divergence band: with all features
at center 10 and only feature 0 varied,
    s1_model_a flips 1 -> 0 at f0 = 37
    s1_model_b flips 1 -> 0 at f0 = 43
so for f0 in [37, 41] the two models disagree (A=0, B=1).

This run uses a small targeted set to (a) refine the two boundaries and
(b) capture the full probability vectors, then writes a clean artifact.
Budget so far: ~186/200 per model; this run adds ~10.
"""

import json
import os
import sys

# Ensure the repository root is on sys.path so src/ modules are importable.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src import config, starter_kit
from src.paths import DATA_DIR

POOL = config.SURPRISE1_POOL_URL
MODEL_A = "s1_model_a"
MODEL_B = "s1_model_b"
N = 34
CENTER = 10.0

# Labels observed in the paid full sweep (f0 varied, rest = 10).
INITIAL_SWEEP = {
    -20: (1, 1), -10: (1, 1), -5: (1, 1), 0: (1, 1), 5: (1, 1), 10: (1, 1),
    15: (1, 1), 20: (1, 1), 25: (1, 1), 30: (1, 1), 35: (1, 1),
    37: (0, 1), 39: (0, 1), 40: (0, 1), 41: (0, 1),
    43: (0, 0), 45: (0, 0), 50: (0, 0), 60: (0, 0), 80: (0, 0), 100: (0, 0),
}

TARGET = [36.0, 36.5, 37.0, 37.5, 38.0, 40.0, 41.0, 42.0, 42.5, 43.0]


def _center_input(f0):
    row = [CENTER] * N
    row[0] = f0
    return row


def readout(resp):
    preds = resp.get("predictions", [])
    probs = resp.get("probabilities", []) or [None] * len(preds)
    out = []
    for p, pr in zip(preds, probs, strict=False):
        label = int(p) if p is not None else None
        if isinstance(p, list):
            label = int(max(range(len(p)), key=lambda i: p[i])) if p else None
            pr = pr or p
        out.append((label, tuple(float(x) for x in pr) if pr else None))
    return out


def main():
    rows = [_center_input(v) for v in TARGET]
    ra = readout(starter_kit.predict(MODEL_A, rows, base_url=POOL))
    rb = readout(starter_kit.predict(MODEL_B, rows, base_url=POOL))

    print(f"{'f0':>8} | {'A label':>7} {'A probs':>16} | {'B label':>7} {'B probs':>16} | same")
    a_data, b_data = [], []
    merged = {k: list(v) for k, v in INITIAL_SWEEP.items()}
    for v, (la, pa), (lb, pb) in zip(TARGET, ra, rb, strict=False):
        merged.setdefault(v, [None, None])[:] = [la, lb]
        a_data.append({"f0": v, "label": la, "probs": list(pa) if pa else None})
        b_data.append({"f0": v, "label": lb, "probs": list(pb) if pb else None})
        ca = max(pa) if pa else float("nan")
        cb = max(pb) if pb else float("nan")
        print(f"{v:>8.1f} | {la:>7} {ca:>8.3f}     | {lb:>7} {cb:>8.3f}     | "
              f"{'===' if la == lb else '*** DIVERGE'}")

    # Merge full sweep labels into per-model histories.
    full_a = [{"f0": v, "label": merged[v][0], "probs": None} for v in sorted(merged)]
    full_b = [{"f0": v, "label": merged[v][1], "probs": None} for v in sorted(merged)]

    boundaries = {
        "A_first_class0_at_f0": min(v for v in merged if merged[v][0] == 0),
        "B_first_class0_at_f0": min(v for v in merged if merged[v][1] == 0),
    }
    diverged = [v for v in sorted(merged) if merged[v][0] != merged[v][1]]
    band = (min(diverged), max(diverged)) if diverged else None

    all_spent = 186 + len(TARGET)
    result = {
        "pool": POOL,
        "model_a": MODEL_A, "model_b": MODEL_B,
        "n_features": N, "center": CENTER,
        "n_probes_per_model": len(TARGET),
        "estimated_total_probes_per_model": all_spent,
        "boundaries": boundaries,
        "divergence_band_f0": {"min": band[0], "max": band[1]} if band else None,
        "finding": (
            f"The models diverge exactly when feature 0 lands in the band "
            f"f0 in [{band[0]:g}, {band[1]:g}] with all other features pinned at "
            f"{CENTER:g}: model A already predicts class 0 while model B still "
            f"predicts class 1 (A's class-0 boundary is at f0={boundaries['A_first_class0_at_f0']:g}, "
            f"B's is at f0={boundaries['B_first_class0_at_f0']:g})."
        ),
        "A_sweep": full_a, "B_sweep": full_b,
        "A_targeted": a_data, "B_targeted": b_data,
    }
    out = os.path.join(DATA_DIR, "surprise1_f0_sweep.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    print(f"\nsaved -> {out}")


if __name__ == "__main__":
    main()
