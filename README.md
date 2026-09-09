# ML Model Profiling Challenge — 24-Hour Hackathon

**Profile 13 undocumented ML models through black-box API probing, build a dashboard, ship it in 24 hours.**

---

## 🎯 Challenge Overview

You're given access to 13 machine learning models through a simple REST API. You can only interact with them as a black box — no access to weights, architecture, or training data. Your job:

1. **Infer what each model does** (classification, regression, task type)
2. **Estimate how good it is** (accuracy, reliability)
3. **Detect weaknesses** (edge cases, failures, noise sensitivity)
4. **Show your work** (confidence backed by actual probe counts, not speculation)

**Deliverable:** A deployed dashboard displaying profiles for all 13 models.

**Grading:** Accuracy of inferences + honesty of confidence levels + evidence quality.

---

## 📊 What You're Working With

### Model Pool Composition

| Pool | Count | Models | Budget | Known? |
|------|-------|--------|--------|--------|
| Reference | 4 | ref_01 to ref_04 | 10,000 probes each | ✅ Documented |
| Practice | 5 | prac_01 to prac_05 | 10,000 probes each | ❌ Unknown |
| Held-Out | 4 | held_01 to held_04 | 10,000 probes each | ❌ Unknown |

> Budget note: the live API reports `probe_budget: 10000` for every model (the
> figures in the challenge docs — 500 practice / 150 held-out — are examples,
> not the actual caps on this server).

### Reference Models (Your Calibration Points)

| Model | Task | Features | Performance |
|-------|------|----------|-------------|
| **ref_01** | MNIST digits (10-class) | 70 | 95.7% accuracy |
| **ref_02** | Wine classification (3-class) | 17 | 96.3% accuracy |
| **ref_03** | Breast cancer (binary) | 34 | 97.7% accuracy |
| **ref_04** | Diabetes progression (regression) | 14 | R² = 0.466 |

**The unknowns are matched to references by feature count:**
- 17 features → Wine classification family
- 34 features → Breast cancer family
- 70 features → Digits family
- 14 features → Diabetes regression family

---

## 📁 Project Files

```
.
├── README.md                          ← This file
├── starter_kit.py                     ← API client (list_models / predict / get_usage)
├── starter_code_snippets.py           ← Core profiling primitives (incl. deep compare)
├── profiler.py                        ← Full 13-model profiling pipeline (--deep)
│
├── profiles.json                      ← OUTPUT: deep-profiled model profiles (est. accuracy)
├── probe_usage.json                   ← OUTPUT: Tracks probe budget utilization
├── dashboard.html                     ← OUTPUT: live dashboard (GitHub Pages)
├── index.html                         ← Root redirect to dashboard.html
│
├── spot_the_difference.py             ← Surprise-1: two-model diff hunter
├── probe_boundary.py                  ← Surprise-1: precise trigger isolator
├── build_submission.py                ← Surprise-1: answer generator
├── surprise1_answer.json              ← Surprise-1 answer (machine-readable)
├── surprise1_answer.md                ← Surprise-1 answer (human-readable)
├── surprise1_profile.json             ← Surprise-1 raw evidence
└── surprise1_f0_sweep.json            ← Surprise-1 boundary sweep evidence
```

---

## 🚀 Quick Start (5 minutes)

### 1. Set Up Credentials

Get your `TEAM_ID` and `API_KEY` from the organizers. Fill them into `starter_kit.py`:

```python
TEAM_ID = "your_team_id"
API_KEY = "your_api_key"
GENERAL_POOL_URL = "https://hackathon-general-pool.onrender.com"
```

### 2. Verify Connection

```bash
python starter_kit.py
```

Expected output:
```
13 models available:
  ref_01               expects  70 features, budget 10000, pool_set=reference
  ref_02               expects  17 features, budget 10000, pool_set=reference
  ...
  prac_01              expects  17 features, budget 10000, pool_set=practice
  ...
```

### 3. Read the Methodology

```bash
cat METHODOLOGY.md
```

Our profiling approach in 1 page.

### 4. Understand the Strategy

Read `METHODOLOGY.md` (10 min). The strategy is:

1. **Shape matching** (free) — Feature count → likely task
2. **Reference comparison** (30 probes) — Send same inputs to unknown + reference
3. **Edge case testing** (20 probes) — Detect failures
4. **Confidence scoring** (formula) — Confidence = evidence invested

### 5. Start Profiling

```bash
python starter_code_snippets.py
# Or uncomment the example at the bottom and run it
```

---

## 🔄 The Workflow (6 Phases, 24 Hours)

### Phase 0: Setup (0-1 hour)
- ✅ Confirm API access
- ✅ Assign team roles (Backend, Frontend, Strategy, DevOps)
- ✅ Decide on probe budget allocation

### Phase 1: Calibration (1-3 hours)
- ✅ Understand reference model behavior
- ✅ Learn output format (classifier vs regression, determinism, etc.)
- ✅ Cost: ~50 probes on ref_02

**Deliverable:** Understand what reference models look like.

### Phase 2: Shape Matching & Comparison (3-8 hours)
- ✅ Match unknowns to references by feature count
- ✅ Send identical inputs to unknown + reference, measure agreement
- ✅ Cost: ~2 probes per comparison trial × 15-20 trials = ~40 probes per unknown

**Deliverable:** For each unknown, you have:
- Inferred task (shape match + agreement evidence)
- Agreement rate (95% agreement = excellent, 85% = good, etc.)
- Task confidence (0-99%)

### Phase 3: Weakness Detection (8-12 hours)
- ✅ Send extreme/edge-case inputs (all-zeros, all-large, negatives)
- ✅ Test noise robustness (add Gaussian noise, check predictions)
- ✅ Identify failure patterns
- ✅ Cost: ~20-30 probes per unknown

**Deliverable:** List of weaknesses (edge cases, noise sensitivity, boundaries).

### Phase 4: Confidence Scoring (12-14 hours)
- ✅ Calculate confidence using formula: `shape_match + agreement_evidence + probe_investment`
- ✅ Generate profile JSON for each model
- ✅ Attach honest confidence levels and evidence

**Deliverable:** `profiles.json` with all 13 models fully profiled.

### Phase 5: Dashboard (14-20 hours)
- ✅ Build HTML dashboard that loads `profiles.json`
- ✅ Display task, confidence, performance, weaknesses for each model
- ✅ Deploy to public URL

**Deliverable:** Live dashboard at `https://yourname.github.io/repo-name/`

### Phase 6: Polish & Demo (20-24 hours)
- ✅ Fix bugs, test edge cases
- ✅ Practice 2-minute demo
- ✅ Final submission

---

## 🔌 API Reference

### 1. List Models (Free)
```bash
GET /models
```

Response:
```json
[
  {
    "model_id": "ref_01",
    "pool_set": "reference",
    "n_features_expected": 70,
    "probe_budget": 10000
  },
  ...
]
```

### 2. Make Predictions (Costs Probes)
```bash
POST /model/{model_id}/predict
Headers: x-team-key: YOUR_API_KEY
Body: {
  "team_id": "your_team_id",
  "inputs": [[70 numbers], [70 numbers], ...]
}
```

Response:
```json
{
  "predictions": [9, 5, ...],
  "probabilities": [[0.1, 0.05, ..., 0.8], ...],
  "probes_used_this_model": 42,
  "probes_remaining_this_model": 9958
}
```

**Note:** Each row costs 1 probe, regardless of batch size.

### 3. Check Budget (Free)
```bash
GET /team/{team_id}/usage
Headers: x-team-key: YOUR_API_KEY
```

Response:
```json
{
  "ref_01": {"used": 5, "budget": 10000, "remaining": 9995},
  "prac_01": {"used": 47, "budget": 10000, "remaining": 9953},
  ...
}
```

---

## 💻 Code Structure

### starter_kit.py
Working examples of all 3 API endpoints. **Run this first to confirm credentials.**

```python
python starter_kit.py
```

### starter_code_snippets.py
Copy-paste functions for your profiling workflow:

```python
# Example usage:
from starter_code_snippets import profile_one_model

profile = profile_one_model("prac_01", n_features=17, budget=500)
```

**Includes:**
- `probe()` — API wrapper
- `infer_task_by_shape()` — Feature count → task
- `compare_with_reference()` — Agreement testing
- `test_edge_cases()` — Robustness testing
- `calculate_confidence()` — Scoring formula
- `generate_profile()` — Output JSON

### Your Code (To Build)
1. **profiler.py** — Main script that profiles all 13 models
2. **profiles.json** — Output file with all profiles
3. **dashboard.html** — Web UI that loads profiles.json

---

## 📊 Confidence Scoring Formula

Every confidence score must be honest and evidence-backed.

```
Confidence = (Shape Match) + (Agreement Evidence) + (Probe Investment)

where:
  Shape Match        = 0.25 (if features match reference)
  Agreement          = 0.50 if agreement > 95%
                     = 0.40 if agreement 85-95%
                     = 0.25 if agreement 70-85%
  Probe Investment   = 0.15 × min(1.0, probes_on_task / 30)

Example:
  - Shape matches ref_02:       +0.25
  - 92% agreement with ref_02:  +0.40
  - 15 comparison probes:       +0.075
  = 0.725 → Display as 72% confidence
```

**Rule: Never claim 100% confidence. Always leave room for uncertainty.**

Implemented in `starter_code_snippets.py`:

```python
confidence = calculate_confidence(
    shape_match=True,
    agreement_rate=0.92,
    n_comparison_probes=15,
    task_name="wine_classification"
)
# Returns: 0.725
```

---

## 📈 Profile JSON Format

Your output for each model should look like this:

```json
{
  "model_id": "prac_01",
  "inferred_task": "wine_classification_3class",
  "task_confidence": 0.87,
  "estimated_performance": "Good (85-95%)",
  "agreement_with_reference": 0.92,
  "weaknesses": [
    {
      "type": "noise_sensitivity",
      "description": "Prediction changes with 5% Gaussian noise",
      "severity": "low"
    }
  ],
  "probe_utilization": {
    "budget": 500,
    "used": 47,
    "percent": 9.4
  },
  "evidence": {
    "shape_match": true,
    "comparison_probes": 15,
    "edge_case_tests": 8
  }
}
```

---

## 🖥️ Dashboard (Minimal Viable)

Simple HTML template that loads pre-computed `profiles.json`:

```html
<!DOCTYPE html>
<html>
<head>
  <title>ML Model Profiling Dashboard</title>
  <style>
    body { font-family: sans-serif; margin: 20px; background: #f5f5f5; }
    .model-card { background: white; padding: 20px; margin: 15px 0; 
                  border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
    .confidence { padding: 5px 10px; border-radius: 4px; font-weight: bold; }
    .confidence.high { background: #d4edda; color: #155724; }
    .confidence.medium { background: #fff3cd; color: #856404; }
    .confidence.low { background: #f8d7da; color: #721c24; }
  </style>
</head>
<body>
  <h1>🤖 ML Model Profiling Dashboard</h1>
  <div id="models"></div>

  <script>
    fetch('profiles.json')
      .then(r => r.json())
      .then(profiles => {
        profiles.forEach(p => {
          const confLevel = p.task_confidence > 0.8 ? 'high' 
                          : p.task_confidence > 0.6 ? 'medium' 
                          : 'low';
          
          document.getElementById('models').innerHTML += `
            <div class="model-card">
              <h2>${p.model_id}</h2>
              <p><b>Task:</b> ${p.inferred_task}</p>
              <p>
                <b>Confidence:</b> 
                <span class="confidence ${confLevel}">
                  ${Math.round(p.task_confidence * 100)}%
                </span>
              </p>
              <p><b>Performance:</b> ${p.estimated_performance}</p>
              <p><b>Probes used:</b> ${p.probe_utilization.used}/${p.probe_utilization.budget}</p>
              ${p.weaknesses.length > 0 ? `
                <h4>Weaknesses:</h4>
                <ul>
                  ${p.weaknesses.map(w => `<li>${w.description}</li>`).join('')}
                </ul>
              ` : '<p><i>No significant weaknesses detected</i></p>'}
            </div>
          `;
        });
      });
  </script>
</body>
</html>
```

Save as `dashboard.html`, then deploy.

---

## 🚢 Deployment

### Option A: GitHub Pages (Fastest)
```bash
# Create repo, commit dashboard.html + profiles.json
git add dashboard.html profiles.json
git commit -m "Add model profiles dashboard"
git push origin main

# Enable GitHub Pages in repo settings
# Live at: https://username.github.io/repo-name/dashboard.html
```

### Option B: Vercel
```bash
npm install -g vercel
vercel deploy --prod
# Live at: https://your-project.vercel.app/
```

### Option C: Netlify
```bash
npm install -g netlify-cli
netlify deploy --prod --dir=.
# Live at: https://your-site.netlify.app/
```

**Tip: Pick ONE platform and commit to it. You won't regret GitHub Pages for speed.**

---

## ⏰ Timeline with Commit Points

| Time | Phase | Commit Point |
|------|-------|--------------|
| 0:00 - 1:00 | Setup | Credentials confirmed, roles assigned |
| 1:00 - 3:00 | Calibration | Understand ref_02 behavior (50 probes used) |
| 3:00 - 8:00 | Shape matching & comparison | All unknowns have inferred tasks (200 probes used) |
| 8:00 - 12:00 | Weakness detection | Weaknesses identified for all models (500 probes used) |
| 12:00 - 14:00 | Confidence scoring | profiles.json complete with all 13 models |
| 14:00 - 20:00 | Dashboard | Live dashboard at public URL |
| 20:00 - 24:00 | Polish & demo | Final fixes, demo rehearsed, submitted |

---

## 📋 Probe Budget Allocation (Recommended)

**Total available: ~130,000 probes across all models (10,000 each × 13)**

```
Calibration (ref models):       500 probes
Shape matching (free):            0 probes
Reference comparison:           1,500 probes  (30 per unknown × 5 unknowns + 5 per ref)
Edge case testing:                800 probes  (40 per unknown × 9 + 80 on refs)
Robustness/noise testing:         800 probes
Follow-up/surprises:              500 probes
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Total used (conservative):      4,100 probes (~9.5% of budget)
Remaining buffer:              39,000 probes
```

**You won't run out of budget if you probe deliberately.**

---

## ⚠️ Common Pitfalls (Avoid These!)

### ❌ Pitfall: Claiming high confidence without evidence
- ❌ "Model X is definitely classification (95% confident)"
- ✅ "Model X is likely classification (78% confident): shape match + 92% agreement with ref_02 + 15 comparison probes"

### ❌ Pitfall: Building fancy dashboard before profiling is done
- ❌ Frontend starts on React before backend has any profiles
- ✅ Frontend builds against sample profiles.json while backend profiles models

### ❌ Pitfall: Running out of budget
- ❌ Allocate 500 probes per practice model, run out by hour 8
- ✅ Allocate 80 probes per model, save the rest for edge cases

### ❌ Pitfall: Hardcoding reference model results
- ❌ Assume ref_02 is 96.3% accurate without probing
- ✅ Actually probe ref_02, verify it matches documented performance

### ❌ Pitfall: Not tracking probes
- ❌ Profile generated, but no evidence of how many probes backed it
- ✅ Every profile includes: `probes_used`, `agreement_rate`, `comparison_trials`

### ❌ Pitfall: Trying to profile all 13 equally
- ❌ Spread 1,000 probes across 13 models = 77 per model = low confidence everywhere
- ✅ Focus on references first (calibration), then practice (characterization), then held-out (quick check)

---

## 🎯 Success Criteria

**By 24:00, you must have:**

- ✅ **Dashboard deployed** at a public URL (live, no broken links)
- ✅ **All 13 models displayed** (reference + practice + held-out)
- ✅ **Task inference** for each model (with evidence)
- ✅ **Confidence scores** 0-99% (honest, backed by probes)
- ✅ **At least 1 weakness per unknown** (detected via probing)
- ✅ **Probe budget tracking** visible (% used per model)
- ✅ **Evidence drill-down** (show queries used for top claims)
- ✅ **No crashes** (no NaN, null, or undefined values)
- ✅ **2-minute demo** ready (you can walk through in < 120 sec)
- ✅ **1-page writeup** (methodology: how did you profile? why this confidence formula?)

---

## 🔍 Grading (Estimated)

| Criterion | Weight |
|-----------|--------|
| **Correctness** (inferred tasks vs. ground truth) | 40% |
| **Confidence Calibration** (stated confidence matches accuracy) | 30% |
| **Evidence Quality** (all claims backed by probes) | 20% |
| **Dashboard Completeness** (all models, working, no crashes) | 10% |

**Key insight:** Polished UI beats technical correctness. **Honest uncertainty beats speculative confidence.**

---

## 🎓 Learning Outcomes

By the end of this hackathon, you'll have built:

1. ✅ A **probe strategy** under budget constraints
2. ✅ A **confidence scoring system** tied to evidence
3. ✅ A **dashboard** that displays pre-computed results
4. ✅ A **workflow** for characterizing black-box models

You'll learn:
- How to design an investigation strategy without ground truth
- How to measure confidence honestly
- How to build a product-grade dashboard in < 24 hours

---

## 📚 Files in This Repo

| File | Purpose | When to Read |
|------|---------|--------------|
| **README.md** (this file) | Project overview | First (you're reading it) |
| **METHODOLOGY.md** | Profiling approach & confidence formula | Hour 0 |
| **config.py** | Central configuration & credentials | Hour 0 |
| **starter_kit.py** | Working API examples | Hour 0 (run it) |
| **starter_code_snippets.py** | Copy-paste functions | Hour 1 (start profiling) |

---

## 🚀 Get Started Now

1. **Read this README** (you just did!)
2. **Read METHODOLOGY.md** — our profiling approach
3. **Fill in credentials** in config.py
4. **Run starter_kit.py** to confirm API access
5. **Read METHODOLOGY.md** for the strategy
6. **Assign team roles:** Backend, Frontend, Strategy, DevOps
7. **Start profiling** using starter_code_snippets.py

**You've got 24 hours. Ship it.** 🚀

---

## 🎭 Surprise Challenge 1 — Spot the Difference

Compare originals `s1_model_a` vs `s1_model_b` on the surprise-1 pool.

**Answer (see `surprise1_answer.md`): the two models are DIFFERENT.**
- Label agreement 98.1%, but strict agreement (labels **and** probabilities) 89.3% over 165 probes.
- 3 inputs flip labels (always A→class 0, B→class 1); 14 more differ only in probabilities (B is overconfident).
- **Precise trigger:** with all 34 features at center (10), raising **feature 0 into [37, 41]** makes A say class 0 while B says class 1 (f0 sweep, `probe_boundary.py`).

Reproduce: `python spot_the_difference.py --report surprise1_profile.json`

---

## ❓ FAQ

**Q: What if I run out of probes?**  
A: Report reduced confidence, not missing profiles. All 13 models must appear, even if confidence is low.

**Q: Can my team query simultaneously?**  
A: Yes. Budget is per team_id, regardless of who queries.

**Q: What if a model behaves unexpectedly?**  
A: Document it as a weakness. Honest surprises are worth more than confident BS.

**Q: Can I use other programming languages?**  
A: Yes. API is plain HTTP/JSON. Use anything you want.

**Q: Should I build a React dashboard?**  
A: Only if your team knows React well. HTML + vanilla JS is faster. Streamlit is also an option.

**Q: What if I don't finish?**  
A: Submit whatever you have. A complete profiles.json with low confidence is better than a polished UI with no profiles.

---

## 🤝 Contributing

If you find bugs or have suggestions, open an issue or PR. Good luck! 🎯

---

**Made for the 24-Hour ML Model Profiling Hackathon**

*Build fast. Profile smart. Ship harder.*
