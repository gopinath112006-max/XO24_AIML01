"""build_submission.py — Build Challenge-1 submission files from saved artifacts."""

import json
import os

HERE = os.path.dirname(__file__)


def main():
    profile = json.load(open(os.path.join(HERE, "surprise1_profile.json"), encoding="utf-8"))
    sweep = json.load(open(os.path.join(HERE, "surprise1_f0_sweep.json"), encoding="utf-8"))

    label_examples = []
    for d in profile.get("label_differences", [])[:4]:
        label_examples.append({
            "input": d["input"],
            "model_A_prediction": d["A_label"],
            "model_B_prediction": d["B_label"],
            "model_A_probs": d["A_probs"],
            "model_B_probs": d["B_probs"],
            "scan": d.get("scan", d.get("kind")),
        })

    answer = {
        "challenge": "surprise1",
        "pool": profile["pool"],
        "models": {"A": profile["model_a"], "B": profile["model_b"]},
        "task_family": profile["task_family"],
        "n_features": profile["n_features"],
        "verdict": profile["verdict"],            # same | different | similar_but_unsure
        "confidence": profile["confidence"],
        "are_they_different": profile["verdict"] == "different",
        "evidence": {
            "probes_used_per_model": 196,
            "budget_per_model": 200,
            "label_agreement": profile["agreement_label"],
            "strict_agreement_labels_and_probs": profile["agreement_strict"],
            "label_flips_count": profile["label_difference_count"],
            "prob_only_disagreement_count": profile["prob_difference_count"],
            "calibration_note": "model B is consistently more confident than model A on extreme and noisy inputs",
        },
        "trigger": {
            "kind": "single-feature spike on feature 0 with all other features pinned at center (10)",
            "feature": 0,
            "divergence_band": sweep["divergence_band_f0"],
            "A_class0_boundary": sweep["boundaries"]["A_first_class0_at_f0"],
            "B_class0_boundary": sweep["boundaries"]["B_first_class0_at_f0"],
            "meaning": "In the band f0 in [37, 41], model A predicts class 0 while model B predicts class 1",
            "example_inputs": label_examples,
        },
        "finding": profile["finding"],
        "methodology": [
            "Round 0: output-format parity check (probs present, shape, prediction type).",
            "Round 1: broad scan of plausible task-family inputs (uniform 0-40, 34 features).",
            "Round 2: structured extremes and robustness (zeros, max, negatives, 1e3 magnitude, noise at 5 scales, permutations/flips).",
            "Round 3: adaptive boundary hunt on the feature that already split the models.",
            "f0 sweep: pin all features at 10, vary feature 0 across [-20, 100] to map both class-0 boundaries.",
            "Verdict uses strict agreement (labels AND probability vectors equal).",
        ],
    }

    out_json = os.path.join(HERE, "surprise1_answer.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(answer, f, indent=2, default=float)

    md = f"""# Surprise Challenge 1 — Spot the Difference: Answer

**Pool:** `{answer["pool"]}`  
**Models:** `{answer["models"]["A"]}` vs `{answer["models"]["B"]}`  
**Task family:** {answer["task_family"]} ({answer["n_features"]} features, class 0/1)

## Verdict

**{answer["verdict"].upper()}** — the two models are NOT identical (confidence {answer["confidence"]}).

- Label agreement over the scan: **{answer["evidence"]["label_agreement"]:.2%}**
- Strict agreement (labels **and** probability vectors equal): **{answer["evidence"]["strict_agreement_labels_and_probs"]:.2%}**
- **{answer["evidence"]["label_flips_count"]} inputs flipped labels** — always the same direction: `model A -> class 0`, `model B -> class 1`.
- **{answer["evidence"]["prob_only_disagreement_count"]} more inputs** agreed on the label but returned different probabilities.

**Calibration note:** {answer["evidence"]["calibration_note"]}.

## Exactly what kind of input splits them

With all 34 features pinned at center value **10** and only **feature 0** varied:

| f0 | model A | model B |
|----|---------|---------|
| ≤ 36.5 | class 1 | class 1 |
| **37 – 41** | **class 0** | **class 1** ← **DIVERGE** |
| ≥ 42 | class 0 | class 0 |

- A's class-0 boundary sits between f0 = 36.5 and 37.0.
- B's class-0 boundary sits between f0 = 41.0 and 42.0.

**Trigger:** raise feature 0 into the band **[37, 41]** while other features are near center → `A=0, B=1`.

### Example disagreement inputs

{"".join(
    f"- f0 spike row `{ex['input'][:4]}...`: A predicts **{ex['model_A_prediction']}** (probs {[round(x,3) for x in ex['model_A_probs']] or 'n/a'}), B predicts **{ex['model_B_prediction']}** (probs {[round(x,3) for x in ex['model_B_probs']] or 'n/a'})\n"
    for ex in answer["trigger"]["example_inputs"])
}

## Finding

> {answer["finding"]}

## Methodology / budget

- Probes used: ~196 / 200 per model (manual tracking; surprise pool does not expose /usage).
- {"; ".join(answer["methodology"])}

_Evidence files: `surprise1_profile.json`, `surprise1_f0_sweep.json`._
"""
    out_md = os.path.join(HERE, "surprise1_answer.md")
    with open(out_md, "w", encoding="utf-8") as f:
        f.write(md)

    print("wrote", out_json)
    print("wrote", out_md)


if __name__ == "__main__":
    main()