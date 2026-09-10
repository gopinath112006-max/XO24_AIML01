# ML Model Profiling Dashboard — XO24 AIML01

**Black-box profile all 13 undocumented ML models, prove the inferences with probe
evidence, and ship a live dashboard.**

**Live dashboard → https://gopinath112006-max.github.io/XO24_AIML01/**
(`index.html` redirects to `dashboard.html`; deployed automatically from `main`.)

---

## 🎯 What This Project Does

Thirteen machine learning models are exposed behind a simple REST API. They are
black boxes — no weights, no architecture, no training data. This repository:

1. **Infers each model's task** (classification vs regression, 2/3/10-class, wine/digits/cancer/diabetes).
2. **Estimates performance** by comparing agreement with documented reference models.
3. **Detects weaknesses** (edge cases, noise sensitivity, collapsed outputs, scale mismatches).
4. **Scores every claim honestly** — confidence is derived from probe evidence, never guessed.
5. **Renders it all** as a static dashboard on GitHub Pages.

**Grading:** accuracy of inferences + honesty of confidence + evidence quality.

---

## 📊 The Model Pool

| Pool | Count | Models | Probe budget | Known? |
|------|-------|--------|--------------|--------|
| Reference | 4 | `ref_01` … `ref_04` | 10,000 each (docs) | ✅ Documented |
| Practice | 4 | `prac_01` … `prac_04` | 10,000 each (docs) | ❌ Unknown |
| Held-Out | 5 | `held_01` … `held_05` | 10,000 each (docs) | ❌ Unknown |

### Reference models (calibration anchors)

| Model | Task | Features | Reference performance |
|-------|------|----------|-----------------------|
| `ref_01` | MNIST digits (10-class) | 70 | 95.7% accuracy |
| `ref_02` | Wine classification (3-class) | 17 | 96.3% accuracy |
| `ref_03` | Breast cancer (binary) | 34 | 97.7% accuracy |
| `ref_04` | Diabetes progression (regression) | 14 | R² = 0.466 |

Unknowns are matched to references **by feature count** (shape matching, free):

```
70 features → mnist/digits family     17 features → wine family
34 features → breast-cancer family    14 features → diabetes regression family
```

> ⚠️ **Live budget reality (read before profiling).** Historical runs under the
> documented 10,000/model budgets were heavy; the live API has since reported a
> **per-model `probe_budget` of 3,000**, and the reference models are currently
> **overspent server-side** (e.g. `ref_02` used ≈9,770). The API **rejects further
> probes** on exhausted models, so reference-comparison profiling returns empty
> predictions and the strategy honestly emits `unclassified` (or `degenerate` when
> a collapsed output is caught without needing a reference). Full comparison
> runs require a budget reset from the organizers. Check live state any time with:

```bash
python -c "import starter_kit as k, json; print(json.dumps(k.get_usage(), indent=2))"
```

---

## 📁 Repository Layout

```
.
├── README.md                       ← this file
├── METHODOLOGY.md                  ← profiling approach + confidence formula
│
├── config.py                       ← credentials, pools, families, strategy tuning
├── starter_kit.py                  ← API client (list_models / predict / get_usage)
├── starter_code_snippets.py        ← core profiling primitives (deep compare, profile gen)
├── strategy.py                     ← "Compare & Infer" probing strategy (--strategy)
├── profiler.py                     ← full 13-model pipeline (writes profiles.json)
│
├── profiles.json                   ← OUTPUT: all 13 model profiles (dashboard data)
├── probe_usage.json                ← OUTPUT: per-model probe budget utilization
├── dashboard.html                  ← OUTPUT: the dashboard (static, reads profiles.json)
├── index.html                      ← redirects the Pages root to dashboard.html
│
├── test_profiling.py               ← unit tests for core primitives (no API calls)
├── test_strategy.py                ← unit tests for the strategy (no API calls)
├── pyproject.toml                  ← deps + ruff/mypy/pytest config
├── requirements.txt                ← runtime deps (requests, numpy)
│
├── ─────────── Surprise Challenge 1: Spot the Difference ───────────
├── spot_the_difference.py          ← hunt where s1_model_a vs s1_model_b differ
├── probe_boundary.py               ← pin the exact trigger inputs
├── build_submission.py             ← build answer JSON/MD from saved artifacts
├── surprise1_answer.json           ← answer (machine-readable)
├── surprise1_answer.md             ← answer (human-readable)
├── surprise1_profile.json          ← raw evidence
├── surprise1_f0_sweep.json         ← boundary-sweep evidence
│
├── ─────────── Surprise Challenge 2: Confidence vs Evidence ─────────
├── surprise_challenge_2.py         ← shows confidence tracks evidence quantity
├── surprise2_answer.json           ← before/after budget-cut comparison
│
└── .github/workflows/pages.yml     ← GitHub Pages auto-deploy on push to main
```

---

## 🚀 Setup (5 minutes)

### 1. Prerequisites

- Python **3.10+**
- Runtime deps: `requests`, `numpy`

```bash
pip install -r requirements.txt
# optional dev tooling:
pip install ".[dev]"          # pytest, ruff, mypy
```

### 2. Credentials — never commit secrets

`config.py` reads your team credentials from (priority order):

1. Environment variables — `HACKATHON_TEAM_ID`, `HACKATHON_API_KEY`
2. A local, gitignored `.credentials.py`:

```python
# .credentials.py  (do NOT commit — already in .gitignore)
TEAM_ID = "your_team_id"
API_KEY = "your_api_key"
```

### 3. Verify API access

```bash
python starter_kit.py
python -c "import starter_kit as k; k.demo_usage()"   # free usage snapshot
```

If you see `13 models available` and a usage table, you're connected.

---

## 🧪 Running the Pipeline

### Recommended: Compare & Infer strategy

```bash
python profiler.py --strategy --deep        # full 13-model strategy run
```

`strategy.py` runs six evidence phases per model:

1. **Type detection** (cheap) — probe a few rows; classifier (probs / int labels) vs regression (scalars).
2. **Degeneracy screen** — diverse + extreme rows; single observed output ⇒ `degenerate` verdict (no comparison, no wasted probes).
3. **Cross-reference screen** — ONE shared batch sent to the unknown, its shape-matched reference, `row[:k]` sub-slice controls from smaller families, and same-feature sibling unknowns.
4. **Decision tree** (`decide_family`) — `degenerate | confirmed | re-anchored | ambiguous | off-corpus | unclassified`, each with a reason string.
5. **Budget-aware deep agreement** — `_cap_deep_trials()` keeps a 300-probe safety margin on shared references; adaptive depth (150 converged / 400 ambiguous).
6. **Weakness suite v2** — extreme inputs + 1/5/10% noise sweep + cross-family drift + regression-constant check + class-coverage.

Every profile carries a full `strategy` audit trail (verdict, screen results, deep
agreement, probe ledger). Standalone per-model run:

```bash
python strategy.py --model prac_01 --deep
```

### Legacy / targeted runs

```bash
python profiler.py                              # legacy profile_one_model pipeline
python profiler.py --quick                      # low probe spend (smoke/dry-run)
python profiler.py --models prac_01 held_05     # subset only
python profiler.py --strategy --deep --models prac_01 held_04
```

### Outputs

- `profiles.json` — all 13 profiles under an envelope `{"generated_at", "profiles": [...]}`. This is the dashboard's data source.
- `probe_usage.json` — per-model `used / budget / remaining / percent_used`.

---

## 🖥️ Dashboard

Static HTML (no build step) that fetches `profiles.json` relative to itself:

| Feature | Detail |
|---------|--------|
| 13 model cards | friendly `MODEL_DISPLAY_NAMES` + raw `.model-tag` (e.g. `prac_03` / 70 features / MNIST Digits 10-Class) |
| Pool filters | Reference / Practice / Held-Out |
| Surprise filters | Surprise 1 / Surprise 2 buttons cross-highlight the relevant model cards |
| Evidence | confidence, inferred task, agreement, weaknesses, probe utilization, strategy verdict |

Light warm theme (#FDF6EC cream / #36454F charcoal, burgundy/terracotta accents).

---

## 🚢 Deployment

### Primary: GitHub Pages (already configured)

A GitHub Actions workflow (`.github/workflows/pages.yml`) deploys the static
site to Pages **on every push to `main`** (and via manual `workflow_dispatch`).

```bash
git add .
git commit -m "Update profiles + readme"
git push origin main
```

1. Push the repo to GitHub.
2. Repo **Settings → Pages → Source: GitHub Actions**.
3. Live at `https://<user>.github.io/<repo>/` — `index.html` redirects to `dashboard.html`.

To publish fresh data:

```bash
python profiler.py --strategy --deep   # regenerate profiles.json + probe_usage.json
git add profiles.json probe_usage.json
git commit -m "Refresh profile data"
git push origin main                   # auto-deploys
```

> The dashboard is fully static (`fetch("profiles.json")`), so it works from
> GitHub Pages, a file server, or any static host — no server-side code.

### Alternatives

```bash
# Vercel
npm i -g vercel && vercel deploy --prod

# Netlify
npm i -g netlify-cli && netlify deploy --prod --dir=.
```

---

## 🔌 API Reference

| Endpoint | Cost | Purpose |
|----------|------|---------|
| `GET /models` | Free | List all models, expected features, budget, pool |
| `POST /model/{model_id}/predict` | **1 probe per input row** (batch size irrelevant) | Send `{"team_id", "inputs": [[...], ...]}` |
| `GET /team/{team_id}/usage` | Free | Per-model probe usage |

Auth header on every request: `x-team-key: YOUR_API_KEY`.

```jsonc
// GET /models → [ { "model_id", "pool_set", "n_features_expected", "probe_budget" }, ... ]
// POST /model/ref_02/predict →
{
  "team_id": "your_team_id",
  "inputs": [[17 floats]]
}
// → { "predictions": [1], "probabilities": [[0.1, 0.3, 0.6]], "probes_used_this_model": 8, ... }
```

---

## 📈 Confidence & Evidence

Full write-up in **`METHODOLOGY.md`** (read it first!). The formula is:

```
Confidence =
  0.20 Type/Shape consistency
+ 0.10 Class-coverage plausibility
+ up to 0.45 Agreement with reference (>90% … 70-80% tiers)
+ 0.15 × probe investment (min(1, comparison_probes / 30))
− penalties (scale mismatch, degenerate output, collapsed coverage, high-severity weakness)
(never > 0.99)
```

Implemented in `starter_code_snippets.calculate_confidence()`.

### Profile JSON shape (`profiles.json` → `profiles[]`)

```jsonc
{
  "model_id": "prac_01",
  "pool_set": "practice",
  "inferred_task": "wine_classification_3class",
  "task_confidence": 0.45,
  "estimated_performance": "…",
  "agreement_with_reference": 0.575,
  "weaknesses": [ { "type": "…", "description": "…", "severity": "low|medium|high" } ],
  "probe_utilization": { "budget": 3000, "used": 400, "percent": 13.3 },
  "evidence": { "shape_match": true, "comparison_probes": 400, "edge_case_tests": 8 },
  "strategy": { "verdict": "confirmed|re-anchored|…", "reason": "…",
                "screen": {…}, "deep": {…}, "probe_ledger": {…} }   // when run with --strategy
}
```

---

## 🎭 Surprise Challenges

### Surprise 1 — "Spot the Difference"

`s1_model_a` vs `s1_model_b` (surprise-1 pool) are **DIFFERENT**:

- Label agreement 98.1%, but labels **and** probabilities 89.3% over 165 probes.
- 3 inputs flip labels (A → class 0, B → class 1); 14 more differ only in probabilities (B overconfident).
- Precise trigger: with all 34 features at center **10**, raising **feature 0 into [37, 41]** makes A say class 0 while B says class 1.

```bash
python spot_the_difference.py --report surprise1_profile.json
python probe_boundary.py
python build_submission.py
```

### Surprise 2 — "Confidence Should Match Evidence"

Demonstrated on `prac_01` (wine, 3-class, 17 features): the **same agreement**
(0.575) yields **confidence 0.45 with 400 probes** but only **0.304 with 5 probes**
— confidence drops when evidence is cut, even though the underlying signal doesn't change.

```bash
python surprise_challenge_2.py
# → surprise2_answer.json (before/after budget-cut table)
```

---

## ✅ Quality Gates & Troubleshooting

```bash
python -m pytest                 # all unit tests (no API calls)
python -m pytest test_strategy.py -v
ruff check .                     # lint (E W F I B UP SIM)
```

> **mypy:** the project pins `python_version = 3.10`, but this environment's
> numpy 2.x stubs require Python 3.12 syntax. If `mypy` refuses to start, run it
> with `python -m mypy --python-version 3.12 strategy.py profiler.py config.py`
> (the legacy `starter_*` modules still have pre-existing type noise).

| Symptom | Likely cause |
|---------|--------------|
| `0 comparison probes` + `unclassified` everywhere | Reference probe budget exhausted server-side; API rejects ref probes. Check `probe_usage.json` `remaining`. Needs an organizer budget reset to run comparisons. |
| `used > budget` in usage | Cumulative counter across runs; API may have lowered per-model budget (10k → 3k). |
| DNS/connectivity churn | `profiler.py` auto-waits and retries (`CONNECTIVITY_MAX_WAIT_S`); confirm `GENERAL_POOL_URL` reachable. |
| Missing `profiles.json` on dashboard | Regenerate (`python profiler.py …`) and commit; Pages re-deploys on push. |
| `Type statement is only supported` (mypy + numpy 2) | Use the `--python-version 3.12` invocation above. |

---

## 🎓 Learning Outcomes

- Black-box model characterization under hard probe budgets
- Evidence-driven confidence scoring (no speculation)
- Compare-and-infer strategy design (degeneracy screening, cross-reference agreement, adaptive depth)
- Static dashboard rendering pre-computed JSON + one-command CI/CD deploy

---

**Made for the 24-Hour ML Model Profiling Challenge.**

*Build fast. Profile smart. Ship harder.* 🚀