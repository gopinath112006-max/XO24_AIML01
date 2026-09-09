# Participant Guide — Model Capability Profiling Challenge

## What you're building
For all 13 models in the general pool, produce a **capability profile**: what task you believe each model performs, an estimated performance/reliability level, your confidence in that estimate, and any notable weaknesses you've detected — all derived from probing the models through the API, not from any hidden documentation.

4 of the 13 models are documented for you (see `general_reference_docs.md`) — use these as your calibration points. The other 9 are unknowns you have to characterize yourselves.

## Getting started — how to explore the pool

**Step 1 — Ask the system what models exist.** Make this request (see `list_models()` in `starter_kit.py` for working code):
```
GET https://hackathon-general-pool.onrender.com/models
```
It replies with something like this, for **every one of the 13 models**:
```json
[
  {"model_id": "ref_02", "pool_set": "reference", "n_features_expected": 17, "probe_budget": 10000},
  {"model_id": "prac_01", "pool_set": "practice", "n_features_expected": 17, "probe_budget": 500},
  {"model_id": "held_01", "pool_set": "held_out", "n_features_expected": 17, "probe_budget": 150}
]
```
This one request tells you everything you need to get started, for free, before spending a single probe:
- **`pool_set`** tells you which group a model belongs to. `"reference"` = documented (look it up in `general_reference_docs.md` for its task and performance). `"practice"` and `"held_out"` = undocumented, yours to figure out.
- **`n_features_expected`** tells you exactly how many numbers to send it — required before you can send anything, or your request will just error out.
- **`probe_budget`** tells you your ceiling for that model.

**Step 2 — Send it some numbers and see what happens.** Once you know a model expects, say, 17 numbers, make this request:
```
POST https://hackathon-general-pool.onrender.com/model/prac_01/predict
Headers: x-team-key: <your api_key>
Body: {"team_id": "<your team_id>", "inputs": [[<17 numbers here>]]}
```
It replies with the model's prediction for whatever numbers you sent. What numbers to actually send — and what to learn from the replies — is the real challenge (see the strategy notes below). But the request format itself is always this same shape, for every single model.

`starter_kit.py` runs both of these steps for you automatically the first time you execute it, printing the full model list and one example prediction — run it first, before writing any of your own analysis code, just to confirm your credentials and connection work.

## Strategy notes — how to actually figure out an unlabeled model

There's no manual telling you what numbers to send an unlabeled model — working that out is the actual challenge. A reasonable approach:

1. **Match by shape.** If an unlabeled model expects the same number of features as a documented reference model (e.g., both want 17 numbers), they're very likely the same task family.
2. **Probe broadly first.** Try extreme inputs (all zeros, all large numbers) and a spread of "reasonable" values to see how the model responds — does it stay confident, or does it get uncertain on some inputs?
3. **Compare against the matching reference model.** Send the *same* inputs to both the unlabeled model and its likely-matching reference model. High agreement suggests similar task and quality; consistent disagreement on certain kinds of input is a weakness worth flagging, not just noise to ignore.
4. **Track your evidence.** How many probes you've actually spent on a model should directly affect how confident your final writeup claims to be about it.

## API Reference — every endpoint you'll use

| Method & Path | Auth required? | Purpose |
|---|---|---|
| `GET /models` | No | Lists all 13 models: `model_id`, `pool_set` (reference/practice/held_out), `n_features_expected`, `probe_budget` |
| `POST /model/{model_id}/predict` | Yes — `x-team-key` header | Sends input rows to a specific model, returns its predictions. Each row costs one probe against your budget for that model |
| `GET /team/{team_id}/usage` | Yes — `x-team-key` header | Shows how many probes your team has used vs. remaining, for every model |

All three exist on **both** pool URLs — the general pool (live from hour 0) and the surprise-1 pool (locked until announced). Once Surprise Challenge 1 is released, use the exact same three endpoints against the surprise-1 URL instead.

**`GET /models` — no credentials needed, anyone can check what's available:**
```
GET https://hackathon-general-pool.onrender.com/models
```
See the example response in "Getting started" above.

**`POST /model/{model_id}/predict` — requires your team's key:**
```
POST https://hackathon-general-pool.onrender.com/model/{model_id}/predict
Headers: x-team-key: <your api_key>
Body: {"team_id": "<your team_id>", "inputs": [[<n_features_expected numbers>]]}
```
Response:
```json
{"predictions": [...], "probabilities": [[...]], "probes_used_this_model": 12, "probes_remaining_this_model": 138}
```
`probabilities` is only present for classification models (not the regression one). `inputs` can contain multiple rows in one call — each row costs one probe.

**`GET /team/{team_id}/usage` — requires your team's key:**
```
GET https://hackathon-general-pool.onrender.com/team/{team_id}/usage
Headers: x-team-key: <your api_key>
```
Response: one entry per model, e.g. `{"ref_02": {"used": 12, "budget": 10000, "remaining": 9988}, ...}`

All three are already implemented for you in `starter_kit.py` — `list_models()`, `probe()`, and `check_my_usage()` map directly to these three endpoints.

## Your credentials
You've been given, privately:
- `team_id` and `api_key` — required on every request (see `starter_kit.py` for exact usage)
- Two URLs: the general pool (live now) and the surprise-1 pool (locked until announced)

**Do not share your `api_key`** with other teams — it's how the system tells your team's probes apart from everyone else's, and how your probe budget is protected from being drained by someone else.

## Rules / FAQ

**Is the probe budget shared across my whole team, or per person?**
Shared per team — it's tied to your `team_id`, regardless of which team member is making the request. Coordinate internally so you don't waste probes duplicating each other's work.

**What happens if I run out of probes on a model?**
You'll get an HTTP 429 error. You can still see how many you've used with `GET /team/{team_id}/usage`. Reference models have a generous budget (10,000) specifically so you can freely calibrate your approach on them — the practice (500) and held-out (150) models have tighter budgets, so probe deliberately rather than randomly once you move to those.

**Can multiple team members query at the same time?**
Yes — this has been tested under real concurrent load and works correctly. Your usage counts stay accurate even if several of you are probing simultaneously.



**Is a malformed request going to cost me a probe?**
Yes — if you send the wrong number of features, that request still counts against your budget (this mirrors how a real black-box API works: you don't get refunds for your own mistakes). Double-check `n_features_expected` from `GET /models` before sending a batch.

**What can I use to build this?** Any language/tooling you want — the API is plain HTTP/JSON. `starter_kit.py` is provided in Python for convenience, not as a requirement.

## Submission format

**Your final submission is a working dashboard**, deployed somewhere reachable by judges (e.g., a Streamlit app URL, though any live web dashboard is acceptable). It must show, for all 13 models:

- **Inferred task** — what you believe the model does (and, ideally, which reference model's task family it matches)
- **Estimated performance/quality** — your best estimate of how good the model is, with the reasoning/evidence behind it (e.g., agreement rate against a reference model)
- **Confidence level** — how much evidence backs your estimate (this should visibly reflect how many probes you actually spent — a model probed 15 times shouldn't get the same confidence label as one probed 300 times)
- **Flagged weaknesses**, if any — specific patterns where a model underperforms or behaves inconsistently, not just an overall score

A simple, honest, well-evidenced dashboard beats a polished one with unsupported claims — profiles are graded against ground truth we hold privately, including whether claimed confidence actually matches the amount of real evidence gathered.

**Quick way to build this:** Python's `streamlit` library turns a script into a shareable dashboard with very little code (`pip install streamlit`, then `streamlit run your_dashboard.py`). Not required, just a fast option if you're not sure where to start.
