# Methodology — How We Profiled the 13 Black-Box Models

A 1-page summary of our approach, evidence, and confidence model.
We interact with the models **only** through the REST API (black box) —
no weights, architecture, or training data.

---

## 1. Probe strategy (budget-aware)

Every input row costs **1 probe**. We spend deliberately and track usage per
model (`probe_usage.json`). Order of operations matters:

1. **Calibrate references (light spend).** We probe the 4 documented reference
   models just enough to confirm their documented behavior and learn the
   output format (label + probability vector for classifiers, scalar for
   regression).
2. **Shape matching (free).** The API advertises each model's expected feature
   count. We map feature count → candidate task family:
   - 70 features → digits (10-class)
   - 17 features → wine (3-class)
   - 34 features → cancer (binary)
   - 14 features → diabetes (regression)
3. **Output-type verification (cheap, decisive).** Before trusting the
   shape-based guess, we probe each unknown and inspect its output:
   - Classification → probability vector present (`n_classes` tells us 2/3/10).
   - Classification (no probs) → integer-only labels (e.g. digit `0-9`) are
     still treated as classification even when the API omits probabilities
     (`held_05` is a 10-class digit classifier that exposes no probs).
   - Regression → scalar values, no probabilities.
   - Note: 14-feature `prac_02` returns small scalars (~O(1)) while the
     diabetes reference `ref_04` returns values ~O(100) — a genuine **scale
     mismatch**, which we flag rather than hide.
4. **Agreement vs. matched reference.** Send identical inputs to the unknown
   and its feature-matched reference; compare outputs.
   - Classifiers: fraction of equal predicted labels.
   - Regression: Pearson correlation after **z-scoring** both prediction
     series (scale/offset invariant).
5. **Weakness / robustness probing.** Extreme inputs (zeros, all-max, all-min,
   all-negative, huge values), small-Gaussian-noise flips, class-coverage
   (does a classifier emit all its classes?), and probability sanity
   (finite, ~sums to 1).

### Digit-family inputs (70 features)

Random pixel noise collapses every digit model onto a single class, which makes
coverage and agreement uninformative. We isolate this problem and fix it:

- We tested real MNIST digits downsampled to several 70-cell grids
  (7×10, 10×7, 5×14, 14×5) against `ref_01` — none were recognized, so the
  70-dim representation is **not a simple raster** of MNIST.
- Instead we feed a **structured generator**: synthetic 7×10 digit renderings
  plus geometric primitives (bars, gradients, quadrants, inversions) with
  random shifts/scaling. These elicit multiple classes from healthy models
  (observed: `ref_01` → 5 classes, `prac_03`/`held_02` → 5-6) and exposed
  **`held_05` as a collapsed classifier** that returns class 4 (or 1-2 classes)
  for virtually all inputs — exactly the kind of honest surprise we report
  instead of hiding.

## 2. Confidence formula (evidence-backed)

Confidence is not a guess — it is a sum of observable evidence:

```
Confidence =
   0.20  Type/Shape consistency   (output type matches shape family)
  +0.10  Class-coverage bonus     (classifier emits a plausible spread of classes)
  +0.45  Agreement vs reference   (>0.90 → 0.45, >0.80 → 0.36, >0.70 → 0.25)
  +0.15  Probe investment         (0.15 × min(1, comparison_probes / 30))

Penalties:
  −0.10 if regression scale mismatch detected
  −0.15 if degenerate output (only one class ever observed)
  −0.10 if collapsed class coverage (far fewer classes than the family expects)
  −0.05 if any high-severity weakness
Cap at 0.99 — we never claim 100% certainty.
```

Every claim in `profiles.json` is tied to concrete evidence
(`comparison_probes`, `edge_case_tests`, `output_type`, `class_coverage`,
`correlation`) so a reviewer can audit any number.

## 3. Honest handling of surprises

Where shape and behavior disagree we report **lower confidence** and surface
the reason instead of forcing a confident guess:
- `prac_02` (14-feature): regression confirmed, but correlation vs `ref_04` ≈ 0
  and scale differs → low confidence that it matches the reference family.
- Same-feature models can diverge (e.g. two 70-feature models with very
  different agreement vs `ref_01`), showing feature count alone is not
  authoritative — evidence is.

## 4. Files

- `profiler.py` — runs the full workflow for all 13 models (resilient to
  transient network drops; saves progress incrementally).
- `starter_code_snippets.py` — the core primitives (probe, type detection,
  agreement, edge cases, confidence).
- `profiles.json` — the output this dashboard renders.
- `probe_usage.json` — per-model probe budget tracking.

*Build fast. Profile smart. Ship harder.*
