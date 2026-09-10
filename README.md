# XO24 AIML01 — Black-Box ML Model Profiling

**Profile 13 undocumented ML models through a black-box REST API, back every
inference with probe evidence, score confidence honestly, and ship it as a
static dashboard.**

| | |
|---|---|
| Live dashboard | <https://gopinath112006-max.github.io/XO24_AIML01/> |
| Tests | 63/63 passing (no API calls) |
| Lint | `ruff check .` clean |
| Runtime | Python 3.10+ · `requests` · `numpy` |

---

## Table of contents

1. [What this project does](#what-this-project-does)
2. [The model pool](#the-model-pool)
3. [Repository layout](#repository-layout)
4. [Architecture](#architecture)
5. [Quickstart](#quickstart)
6. [Command reference](#command-reference)
7. [Working code](#working-code)
8. [Compare & Infer strategy](#compare--infer-strategy)
9. [Technical implementation](#technical-implementation)
10. [Innovation](#innovation)
11. [Future impact](#future-impact)
12. [Outputs & dashboard](#outputs--dashboard)
13. [Deployment](#deployment)
14. [Test, lint, type-check](#test-lint-type-check)
15. [Troubleshooting](#troubleshooting)
16. [Commit history](#commit-history)

---

## What this project does

The models are black boxes — no weights, no architecture, no training data.
`src/profiler.py` characterizes all 13 of them:

1. **Infers each model's task** — classification vs regression, 2/3/10-class,
   wine / digits / cancer / diabetes (feature-count shape matching, then
   output-type probing).
2. **Estimates performance** — agreement against documented reference models,
   converted to an accuracy range using each reference's declared performance.
3. **Detects weaknesses** — noise sensitivity, edge cases, collapsed/degenerate
   outputs, scale mismatches, cross-family drift.
4. **Scores claims honestly** — confidence is derived from probe evidence, never
   guessed; uncertainty is reported, not hidden.
5. **Renders it all** — pre-computed `data/profiles.json` feeds a zero-dependency
   static dashboard on GitHub Pages.

## The model pool

| Pool | Count | Budget / model | Known? |
|------|-------|----------------|--------|
| Reference | 4 (`ref_01`…`ref_04`) | 10,000 (documented) | ✅ Documented |
| Practice | 4 (`prac_01`…`prac_04`) | 10,000 (documented) | ❌ Unknown |
| Held-Out | 5 (`held_01`…`held_05`) | 10,000 (documented) | ❌ Unknown |

Unknowns are matched to references **by feature count** (free, deterministic):

| Features | Family | Reference |
|----------|--------|-----------|
| 70 | MNIST digits (10-class) | `ref_01` (95.7%) |
| 17 | Wine (3-class) | `ref_02` (96.3%) |
| 34 | Breast cancer (binary) | `ref_03` (97.7%) |
| 14 | Diabetes progression (regression) | `ref_04` (R² 0.466) |

> ⚠️ **Live budget reality.** Historical runs under the documented
> 10,000/model budgets were heavy; the live API now reports a **per-model
> `probe_budget` of 3,000** and the reference models are **overspent
> server-side** (e.g. `ref_02` ≈ 9,770 used). The API **rejects further probes**
> on exhausted models, so reference-comparison runs return empty predictions and
> the strategy honestly emits `unclassified` (or `degenerate` when the collapse
> is caught without a reference). Full comparison runs need a budget reset from
> the organizers. Check live state any time (from the repo root):

```bash
python -c "import src.starter_kit as k, json; print(json.dumps(k.get_usage(), indent=2))"
```

## Repository layout

```
.
├── index.html                 # web root: meta-refresh → dashboard.html
├── dashboard.html             # static dashboard (fetches data/*.json relatively)
├── data/                      # all JSON/MD artifacts the dashboard consumes
│   ├── profiles.json          #   13 model profiles (envelope + profiles[])
│   ├── probe_usage.json       #   per-model probe utilization
│   └── surprise*_answer.*     #   Surprise Challenge 1 & 2 submissions + evidence
├── src/                       # core package (importable via `from src import …`)
│   ├── paths.py               #   repo-relative path resolution (data/, .credentials.py)
│   ├── config.py              #   credentials, pools, families, strategy tuning
│   ├── starter_kit.py         #   API client: list_models / predict / get_usage
│   ├── starter_code_snippets.py # core profiling primitives
│   ├── strategy.py            #   "Compare & Infer" probing strategy
│   └── profiler.py            #   CLI: full 13-model pipeline
├── scripts/                   # challenge tooling (live probes + submission builders)
│   ├── spot_the_difference.py #   Surprise 1: hunt divergences A vs B
│   ├── probe_boundary.py      #   Surprise 1: pin exact trigger inputs
│   ├── build_submission.py    #   Surprise 1: assemble answer JSON/MD from data/
│   └── surprise_challenge_2.py#   Surprise 2: confidence vs evidence ('budget cut')
├── tests/                     # unit tests (no API calls)
│   ├── test_profiling.py
│   └── test_strategy.py
├── METHODOLOGY.md             # profiling approach + confidence formula
├── pyproject.toml             # deps + ruff / mypy / pytest config
├── requirements.txt           # runtime deps
└── .github/workflows/pages.yml # GitHub Pages auto-deploy on push to main
```

`data/`, `src/`, `scripts/`, `tests/` mirror the pipeline stages:
**probe → infer → evidence → artifacts → render**.

## Architecture

Two layers, cleanly separated by committed, reviewable JSON artifacts.

```
OFFLINE · Python pipeline (needs API + credentials)
  src/config.py ──────────────────────────┐
  (pools · families · strategy tuning)    ┴─▶ src/starter_kit.py — API client
                                             list_models() / predict() / get_usage()
                                                       │
                                                       ▼
  src/starter_code_snippets.py — core primitives:
     probe() → infer_task_by_shape/type() → compare_with_reference()
          → test_edge_cases() → calculate_confidence()
                                                       ▲
  src/strategy.py — Compare & Infer: type detection → degeneracy screen
     → cross-reference screen → decide_family → deep agreement
                                                       │ pairs with
  src/profiler.py — CLI: python -m src.profiler --strategy --deep  (all 13)
                                                       │ writes precomputed artifacts
                                                       ▼
                                              data/*.json
                              profiles.json · probe_usage.json · surprise*_answer.json

RUNTIME · static web (no API, no server)
  index.html ──▶ dashboard.html ── fetch('data/*.json') ──▶ 13 cards + pool & Surprise filters
  Hosts: GitHub Pages (auto-deploy on push) · Vercel (auto-deploy, cleanUrls)
```

| Module | Responsibility |
|--------|----------------|
| `src/config.py` | credentials, pool URLs, families, strategy thresholds |
| `src/starter_kit.py` | API client — connect, discover models, predict, usage |
| `src/starter_code_snippets.py` | profiling primitives — probe, inference, agreement, edge cases, confidence |
| `src/strategy.py` | six-phase Compare & Infer strategy |
| `src/profiler.py` | orchestration — whole-pool runs, budget ledger, JSON outputs |
| `scripts/*.py` | Surprise 1 & 2 tooling (live probes → submission artifacts) |
| `tests/` | 63 unit tests, no API calls |
| `data/*.json` | precomputed, versioned results the dashboard renders |
| `dashboard.html` / `index.html` | zero-dependency renderer + redirect |

**End-to-end flow:** connect (`starter_kit`) → discover (`list_models`) → probe
(`probe`, 1 probe per row) → infer + collect evidence (`infer_task_by_*`,
`compare_with_reference`, `test_edge_cases`) → honest confidence
(`calculate_confidence`) → serialize (`profiler` → `data/*.json`) → render
(`dashboard.html`, precomputed results only).

## Quickstart

```bash
# 1. Install
pip install -r requirements.txt

# 2. Credentials (never commit) — either set env vars …
#    HACKATHON_TEAM_ID / HACKATHON_API_KEY
# … or drop a gitignored .credentials.py at the repo root:
#    TEAM_ID = "your_team_id"
#    API_KEY = "your_api_key"

# 3. Verify API access (free endpoints; no probes spent)
python -c "import src.starter_kit as k; k.demo_usage()"

# 4. Run the full 13-model pipeline (Compare & Infer + deep agreement)
python -m src.profiler --strategy --deep

# 5. Open the dashboard locally (or push → auto-deploys)
python -m http.server 8000        # http://localhost:8000
```

## Command reference

| Task | Command |
|------|---------|
| Full strategy run | `python -m src.profiler --strategy --deep` |
| Quick smoke run (low spend) | `python -m src.profiler --quick` |
| Subset of models | `python -m src.profiler --models prac_01 held_04` |
| Single-model strategy | `python -m src.strategy --model prac_01 --deep` |
| Legacy pipeline | `python -m src.profiler` |
| Surprise 1 — full comparison | `python scripts/spot_the_difference.py` |
| Surprise 1 — boundary sweep | `python scripts/probe_boundary.py` |
| Surprise 1 — build submission | `python scripts/build_submission.py` |
| Surprise 2 — confidence vs evidence | `python scripts/surprise_challenge_2.py` |
| Run all unit tests | `python -m pytest` |
| Lint | `ruff check .` |
| Type-check (numpy 2 workaround) | `python -m mypy --python-version 3.12 src/` |
| Live budget snapshot | `python -c "import src.starter_kit as k, json; print(json.dumps(k.get_usage(), indent=2))"` |

Run `python -m <module> --help` for full flag lists. Commands assume the
repository root as the working directory.

## Working code

The entire pipeline in ~15 lines — run from the repo root with credentials
configured (see **Quickstart**). These are the exact functions `profiler.py`
uses; a handful of probes are spent per call.

```python
from src import starter_kit as kit
from src.starter_code_snippets import (
    calculate_confidence,
    compare_with_reference,
    infer_task_by_shape,
    probe,
)

# 1. Connect to the API and discover the available models (free)
models = kit.list_models()
print(models[0]["model_id"])                          # e.g. ref_01

# 2. Probe a black-box model (batched; 1 probe per row)
resp = probe("prac_01", [[5.1, 3.5, 1.4, 0.2], [6.2, 3.4, 5.4, 2.3]])
print(resp["predictions"])                            # e.g. [1, 2]

# 3. Shape matching (free) → task family + matched reference
family = infer_task_by_shape(n_features=17)           # wine_classification_3class

# 4. Agreement vs the matched reference (spends probes)
rate, n_comp, _comparisons = compare_with_reference("prac_01", 17, n_trials=15)

# 5. Confidence derived from evidence, never guessed
confidence = calculate_confidence(
    shape_match=True, type_consistent=True,
    agreement_rate=rate, n_comparison_probes=n_comp, n_features=17,
)
print(f"task={family['task']} · agreement={rate:.3f} · confidence={confidence:.3f}")
```

## Compare & Infer strategy

`src/strategy.py` runs six evidence phases per model:

1. **Type detection** (cheap) — probe a few rows; classifier vs regression.
2. **Degeneracy screen** — diverse + extreme inputs; a single observed output
   ⇒ `degenerate` verdict (skips wasted comparison probes).
3. **Cross-reference screen** — one shared batch to the unknown, its
   shape-matched reference, `row[:k]` sub-slice controls from smaller families,
   and same-feature sibling unknowns.
4. **Decision tree** (`decide_family`) — `degenerate | confirmed | re-anchored
   | ambiguous | off-corpus | unclassified`, each with a reason string.
5. **Budget-aware deep agreement** — `_cap_deep_trials()` keeps a 300-probe
   safety margin on shared references; adaptive depth (150 converged / 400
   ambiguous).
6. **Weakness suite v2** — extreme inputs, 1/5/10% noise sweep, cross-family
   drift, regression-constant check, class coverage.

Every profile carries a full `strategy` audit trail (verdict, screen results,
deep agreement, probe ledger). The confidence formula and accuracy-range
conversion are documented in [`METHODOLOGY.md`](METHODOLOGY.md).

## Technical implementation

**Stack.** Python 3.10+ with `requests` and `numpy` on the analysis side; a
hand-written static HTML/JS dashboard with no build step and no server at
runtime. All heavy lifting is done offline — the site only reads JSON.

**API contract** (the model pool, authenticated via `x-team-key`):

| Endpoint | Cost | Purpose |
|----------|------|---------|
| `GET /models` | free | list models, expected features, budget, pool |
| `POST /model/{id}/predict` | 1 probe per row (batch size irrelevant) | send `{team_id, inputs: [[…]]}` |
| `GET /team/{team}/usage` | free | per-model probe usage |

**Probe accounting.** Every prediction is counted in a per-model ledger
(`data/probe_usage.json`). The strategy keeps a **300-probe safety margin** on
shared reference models and adapts depth (150 trials when decisive, 400 when
ambiguous). `probe()` batches rows into single HTTP calls, so N rows cost N
probes but only one round-trip.

**Algorithmic core** (details in `METHODOLOGY.md`):
- six-phase **Compare & Infer** (`strategy.py`) — type detection → degeneracy
  screen → cross-reference screen → decision tree → budget-aware deep
  agreement → weakness suite;
- evidence-weighted **confidence** formula:
  `0.20 shape/type + 0.10 coverage + up to 0.45 agreement + 0.15 probe
  investment − penalties`, capped at 0.99;
- accuracy range from agreement: `E_min = A·r` and `E_max = A÷r`, using each
  reference's declared performance.

**Quality gates.** 63 unit tests with no API calls (`pytest`), `ruff check .`
clean, `mypy` clean for new modules. Deployment is CI-free: a GitHub Actions
workflow (`pages.yml`) and Vercel both rebuild from `main` on push.

## Innovation

- **Cross-reference screen, not naive agreement.** A single shared batch goes
  to the unknown, its shape-matched reference, `row[:k]` sub-slice controls
  from smaller families, and same-feature sibling unknowns — so a family
  verdict comes from a whole agreement vector, not one lucky score.
- **Degeneracy screen before spending.** A model that always emits one class
  (`held_04`) is caught with diverse + extreme inputs *before* any comparison
  probes are burned on it — honesty and budget efficiency in one move.
- **Probability-level evidence.** Labels on the Surprise-1 pair agree 98.1%,
  but labels *and* probability vectors agree only 89.3% over 165 probes. That
  gap is the proof they are different models — and it pins the trigger
  (`f0 ∈ [37, 41]` flips A→0, B→1).
- **Confidence tracks evidence.** Surprise 2 shows the *same* underlying
  signal scoring 0.45 with 400 probes but only 0.304 with 5 — confidence is a
  property of evidence, never a guess.
- **Honest `unclassified`.** When the shared reference budget is exhausted
  server-side, the pipeline reports "no verdict" instead of fabricating one.

## Future impact

- **Reproducible rerun.** On an organizer budget reset, `python -m src.profiler
  --strategy --deep` regenerates full evidence-backed profiles for all 13
  models and one push redeploys the dashboard.
- **A general black-box audit toolkit.** Point `config.py` at any pool, add
  new families, and the same strategy audits any undocumented model API — a
  "model card without weights" for ML-as-a-service vetting.
- **Continuous model health.** The offline/static split makes scheduled
  re-probing easy: re-run → regenerate `data/` → redeploy. Drift, collapse, or
  over-confidence in a served model becomes visible on the dashboard.
- **Auditable reporting.** Every conclusion carries its probe evidence and
  confidence, so human reviewers (or graders) can check any number.

## Outputs & dashboard

- `data/profiles.json` — all 13 profiles under
  `{"generated_at", "profiles": […]}`. This is the dashboard's data source.
- `data/probe_usage.json` — per-model `used / budget / remaining / percent_used`.

The dashboard is static HTML (no build step) that fetches `data/*.json`
**relative to itself**, so it works from GitHub Pages, Vercel, Netlify, or a
plain file server.

| Feature | Detail |
|---------|--------|
| 13 model cards | friendly `MODEL_DISPLAY_NAMES` + raw model tag (e.g. `prac_03`, 70 features) |
| Pool filters | Reference / Practice / Held-Out |
| Surprise filters | Surprise 1 / Surprise 2 cross-highlight relevant cards |
| Evidence | confidence, inferred task, agreement, weaknesses, probe utilization, strategy verdict |

Light warm theme with burgundy/terracotta accents.

## Deployment

The site is fully static: only `index.html`, `dashboard.html`, `data/`, and
`.github/workflows/pages.yml` are needed to host it.

### GitHub Pages (primary — already configured)

`.github/workflows/pages.yml` deploys **on every push to `main`** (and via
manual `workflow_dispatch`). Repo **Settings → Pages → Source: GitHub Actions**.

To publish fresh profile data:

```bash
python -m src.profiler --strategy --deep   # regenerate data/profiles.json + data/probe_usage.json
git add data/profiles.json data/probe_usage.json
git commit -m "Refresh profile data"
git push origin main                        # auto-deploys
```

### Vercel (optional)

Import the same GitHub repo at <https://vercel.com/new> → framework preset
**Other**, root `./`, no build/output commands. Vercel then auto-deploys every
push to `main` and gives preview URLs per PR. Both hosts can stay live.

> `.credentials.py` is gitignored and untracked, so it is **never** deployed by
> either host — secrets stay local.

## Test, lint, type-check

```bash
python -m pytest                 # 63 unit tests, no API calls
python -m pytest tests/test_strategy.py -v
ruff check .                     # E W F I B UP SIM
python -m mypy --python-version 3.12 src/   # numpy-2 workaround, see below
```

> **mypy + numpy 2:** this environment's numpy 2.x stubs require Python 3.12
> syntax even though the project pins `python_version = "3.10"`. Run with
> `--python-version 3.12` as above. The legacy `starter_*` modules still carry
> pre-existing type noise; new code in `strategy.py`, `profiler.py`, `tests/`
> is clean.

## Troubleshooting

| Symptom | Likely cause |
|---------|--------------|
| `0 comparison probes` + `unclassified` everywhere | Reference probe budget exhausted server-side; API rejects ref probes (see budget note). Check `data/probe_usage.json` `remaining`. Needs an organizer budget reset. |
| `used > budget` | Cumulative counter across runs; the API may have lowered per-model budgets (10k → 3k). |
| DNS/connectivity churn | `profiler.py` auto-waits and retries; confirm `GENERAL_POOL_URL` is reachable. |
| Dashboard: missing profiles | Run `python -m src.profiler --strategy --deep`, commit `data/profiles.json`; Pages re-deploys on push. |
| `No module named 'src'` / `starter_kit` | Run from the repo root (all documented commands assume it). |

## Commit history

All 25 commits on `main`, newest first.

| Commit | Phase | What changed |
|--------|-------|--------------|
| `45803cd` | Deploy | Add `vercel.json` for clean URLs on Vercel |
| `24ceb2c` | Cleanup | Remove `Given/` references; reword METHODOLOGY citation |
| `b5c734f` | Structure | Restructure: `src/` package, `scripts/`, `tests/`, `data/`; rebuild README; dashboard fetches `data/` |
| `3af1e36` | Docs | Final-deployment README refresh; fix dashboard header HTML |
| `d08ec52` | Strategy | Compare & Infer strategy + light-theme dashboard + Surprise filters |
| `e268603` | Compliance | Fix all Given-folder compliance gaps |
| `23c7e9c` | Surprises | Add Surprise Challenge 2; fix Given-analysis bugs |
| `c1e025d` | Polish | Lint, tests, dashboard surprise section, degenerate diagnosis |
| `ae0b6b4` | Docs | Correct live budgets (10k/model); Surprise-1 answer section |
| `5520e3d` | Surprises | Deep-profile unknowns (est. accuracy); Spot-the-Difference tooling + answer |
| `318e42e` | Data | Regenerate 13-model probe usage |
| `a844fe9` | Refine | Structured inputs; degenerate fix; envelope timestamp; dashboard confidence UI |
| `ef1a26e` | Deploy | Add root `index.html` redirect to dashboard |
| `7ce22a3` | Deploy | Add GitHub Pages workflow |
| `a28392c` | Refine | Type detection for no-prob classifiers; batch probes; degenerate penalty; refresh data |
| `a880940` | Pipeline | Network resilience; profile all 13 models with real data |
| `9aa1958` | Pipeline | Add profiling pipeline (config, starter kit, snippets, profiler) |
| `706f68d` | Merge | Merge remote; keep canonical README |
| `07ef1b5` | Dashboard | Add model profiling dashboard + sample profiles |
| `b215d68` | Docs | README: challenge details and resources |
| `d44e724` | Cleanup | Remove `README_PROJECT.md` |
| `dfdb63c` | Docs | README: project contents |
| `116b977` | Merge | Merge `main` (remote sync) |
| `4481889` | Docs | Add README |
| `99b4b92` | Boot | Initial commit |

---

*Made for the 24-Hour ML Model Profiling Challenge. Build fast. Profile smart.
Ship harder.*