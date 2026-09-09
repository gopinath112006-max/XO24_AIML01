# Surprise Challenge 1 — Spot the Difference: Answer

**Pool:** `https://hackathon-surprise1-pool.onrender.com`  
**Models:** `s1_model_a` vs `s1_model_b`  
**Task family:** breast_cancer_binary (34 features, class 0/1)

## Verdict

**DIFFERENT** — the two models are NOT identical (confidence 0.88).

- Label agreement over the scan: **98.11%**
- Strict agreement (labels **and** probability vectors equal): **89.31%**
- **3 inputs flipped labels** — always the same direction: `model A -> class 0`, `model B -> class 1`.
- **14 more inputs** agreed on the label but returned different probabilities.

**Calibration note:** model B is consistently more confident than model A on extreme and noisy inputs.

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

- f0 spike row `[17.423, 29.241, 10.7358, 34.0685]...`: A predicts **0** (probs [0.902, 0.098]), B predicts **1** (probs [0.036, 0.964])
- f0 spike row `[27.2291, 1.2628, 37.9282, 4.3958]...`: A predicts **0** (probs [0.928, 0.072]), B predicts **1** (probs [0.446, 0.554])
- f0 spike row `[40.0, 10.0, 10.0, 10.0]...`: A predicts **0** (probs [0.819, 0.181]), B predicts **1** (probs [0.292, 0.708])


## Finding

> They ARE different: 3 inputs flipped labels (all model A->model B), and 14 more agreed on the label but disagreed on probabilities. Cleanest trigger: f0=40.0 (this feature at max/edge while all others stay at center) flips model B to class 1. model B reads as more confident than model A on extreme/noisy inputs.

## Methodology / budget

- Probes used: ~196 / 200 per model (manual tracking; surprise pool does not expose /usage).
- Round 0: output-format parity check (probs present, shape, prediction type).; Round 1: broad scan of plausible task-family inputs (uniform 0-40, 34 features).; Round 2: structured extremes and robustness (zeros, max, negatives, 1e3 magnitude, noise at 5 scales, permutations/flips).; Round 3: adaptive boundary hunt on the feature that already split the models.; f0 sweep: pin all features at 10, vary feature 0 across [-20, 100] to map both class-0 boundaries.; Verdict uses strict agreement (labels AND probability vectors equal).

_Evidence files: `surprise1_profile.json`, `surprise1_f0_sweep.json`._
