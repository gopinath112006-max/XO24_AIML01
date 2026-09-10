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
4. [Quickstart](#quickstart)
5. [Command reference](#command-reference)
6. [Compare & Infer strategy](#compare--infer-strategy)
7. [Outputs & dashboard](#outputs--dashboard)
8. [Deployment](#deployment)
9. [Quality gates](#quality-gates)
10. [Troubleshooting](#troubleshooting)

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

---

*Made for the 24-Hour ML Model Profiling Challenge. Build fast. Profile smart.
Ship harder.*