"""Central configuration and credentials for the profiling pipeline.

Fill in your TEAM_ID and API_KEY from the organizers, then keep these
values out of version control (they are secrets).
"""

import os

from src.paths import CRED_FILE as _CRED_FILE

# ---------------------------------------------------------------------------
# Credentials & endpoint
# ---------------------------------------------------------------------------
# Secrets are read from (in priority order):
#   1. Environment variables HACKATHON_TEAM_ID / HACKATHON_API_KEY
#   2. An optional ignored .credentials.py file at the repo root (see .gitignore)
# Placeholder defaults keep the code importable until you add real values.
if os.path.exists(_CRED_FILE):
    _ns = {}
    with open(_CRED_FILE, encoding="utf-8") as _f:
        exec(compile(_f.read(), _CRED_FILE, "exec"), _ns)
    TEAM_ID = _ns.get("TEAM_ID", "your_team_id")
    API_KEY = _ns.get("API_KEY", "your_api_key")
else:
    TEAM_ID = os.environ.get("HACKATHON_TEAM_ID", "your_team_id")
    API_KEY = os.environ.get("HACKATHON_API_KEY", "your_api_key")

GENERAL_POOL_URL = "https://hackathon-general-pool.onrender.com"
SURPRISE1_POOL_URL = "https://hackathon-surprise1-pool.onrender.com"

# Named pools so tooling can select a base url without hardcoding strings.
POOL_URLS = {
    "general": GENERAL_POOL_URL,
    "surprise1": SURPRISE1_POOL_URL,
}

# Header used for auth on every request that costs probes.
AUTH_HEADERS = {"x-team-key": API_KEY}

# ---------------------------------------------------------------------------
# Model pool definitions
# ---------------------------------------------------------------------------
# Reference models are the documented calibration anchors.
REFERENCE_MODELS = {
    "ref_01": {"n_features": 70, "task": "mnist_digits_10class", "budget": 10000, "accuracy": 0.957},
    "ref_02": {"n_features": 17, "task": "wine_classification_3class", "budget": 10000, "accuracy": 0.963},
    "ref_03": {"n_features": 34, "task": "breast_cancer_binary", "budget": 10000, "accuracy": 0.977},
    "ref_04": {"n_features": 14, "task": "diabetes_progression_regression", "budget": 10000, "accuracy": 0.466},  # R^2
}

# Practice and held-out models (unknown) are matched to references by feature count.
# NOTE: These mirror the live API manifest (all budgets are 10,000 probes).
UNKNOWN_MODELS = {
    "prac_01": {"n_features": 17, "budget": 10000},
    "prac_02": {"n_features": 14, "budget": 10000},
    "prac_03": {"n_features": 70, "budget": 10000},
    "prac_04": {"n_features": 34, "budget": 10000},
    "held_01": {"n_features": 17, "budget": 10000},
    "held_02": {"n_features": 70, "budget": 10000},
    "held_03": {"n_features": 14, "budget": 10000},
    "held_04": {"n_features": 34, "budget": 10000},
    "held_05": {"n_features": 70, "budget": 10000},
}

# Feature count -> reference family mapping (shape matching, free of cost).
# Each family's reference is the model id used for agreement comparison.
FEATURE_TO_FAMILY = {
    70: {"task": "mnist_digits_10class", "reference": "ref_01"},
    17: {"task": "wine_classification_3class", "reference": "ref_02"},
    34: {"task": "breast_cancer_binary", "reference": "ref_03"},
    14: {"task": "diabetes_progression_regression", "reference": "ref_04"},
}

# Declared performance of each reference model (from general_reference_docs.md).
# Used to convert an unknown's agreement-vs-reference into an accuracy estimate.
REFERENCE_METADATA = {
    "mnist_digits_10class": {
        "reference": "ref_01", "n_classes": 10,
        "accuracy": 0.957, "metric": "accuracy", "metric_value": 0.957,
    },
    "wine_classification_3class": {
        "reference": "ref_02", "n_classes": 3,
        "accuracy": 0.963, "metric": "accuracy", "metric_value": 0.963,
    },
    "breast_cancer_binary": {
        "reference": "ref_03", "n_classes": 2,
        "accuracy": 0.977, "metric": "accuracy", "metric_value": 0.977,
    },
    "diabetes_progression_regression": {
        "reference": "ref_04", "r2": 0.466, "mae": 43.13,
        "metric": "r2", "metric_value": 0.466,
    },
}


def load_manifest_from_api():
    """Fetch the authoritative model list from the API.

    The live API is the source of truth for model ids, feature counts,
    budgets and pool sets. This replaces the hard-coded tables.
    """
    try:
        from src.starter_kit import list_models
        models = list_models()
    except Exception:
        return None
    manifest = {"models": {}, "by_pool": {"reference": [], "practice": [], "held_out": []}}
    for m in models:
        mid = m["model_id"]
        entry = {
            "model_id": mid,
            "pool_set": m["pool_set"],
            "n_features": m["n_features_expected"],
            "budget": m["probe_budget"],
        }
        manifest["models"][mid] = entry
        manifest["by_pool"][m["pool_set"]].append(entry)
    return manifest


def get_family(n_features: int):
    """Return the task family + reference model for a feature count."""
    return FEATURE_TO_FAMILY.get(n_features, {"task": "unknown", "reference": None})


def rank_references(n_features: int, limit: int = 3):
    """Candidate references ordered by feature-count distance (shape first).

    The exact shape-matched family's reference is ranked first; the remaining
    families are ranked by absolute feature-count distance. Their references
    act as ``row[:k]`` sub-slice controls in strategy.cross_ref_screen: a
    70-feature unknown whose real task came from a 34-feature family will agree
    strongly with ref_03 when fed only the first 34 features.

    Each entry: {reference, task, n_features, is_primary, sliceable}.
    """
    exact = get_family(n_features).get("reference")
    ranked = sorted(
        FEATURE_TO_FAMILY.items(),
        key=lambda kv: (0 if kv[1]["reference"] == exact else 1,
                        abs(kv[0] - n_features),
                        kv[1]["reference"]),
    )
    out = []
    for nfeat, fam in ranked[:limit]:
        ref = fam["reference"]
        out.append({
            "reference": ref,
            "task": fam["task"],
            "n_features": nfeat,
            "is_primary": ref == exact,
            "sliceable": nfeat < n_features,
        })
    return out


# ---------------------------------------------------------------------------
# Cross-reference probing strategy (strategy.py) tuning
# ---------------------------------------------------------------------------
# Shared-input screening trials per candidate reference.
STRATEGY_SCREEN_TRIALS = 12
# Diverse-input degeneracy screen (family rows + extreme rows).
STRATEGY_DEGENERATE_SCREEN_INPUTS = 8
# Agreements >= this (vs primary) confirm the family hypothesis.
STRATEGY_CONFIRM_AGREEMENT = 0.80
# Agreements below this (with no strong controls) mean "no match".
STRATEGY_REJECT_AGREEMENT = 0.55
# Sub-slice / control references must stay under this for a confirmation.
STRATEGY_CONTROL_MAX = 0.55
# Sibling (same-family unknown) agreement that corroborates an off-corpus cluster.
STRATEGY_SIBLING_CORROBORATE = 0.70
# Deep-trial depth to use when the screen was decisive vs. ambiguous.
STRATEGY_DEEP_CONVERGED = 150
STRATEGY_DEEP_AMBIGUOUS = 400


# ---------------------------------------------------------------------------
# Request tuning
# ---------------------------------------------------------------------------
REQUEST_TIMEOUT = 30  # seconds
MAX_RETRIES = 3
RETRY_BACKOFF = 1.0  # seconds
# Keep this many probes untouched on the shared reference models (safety margin).
REFERENCE_BUDGET_MARGIN = 300
