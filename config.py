"""Central configuration and credentials for the profiling pipeline.

Fill in your TEAM_ID and API_KEY from the organizers, then keep these
values out of version control (they are secrets).
"""

import os

# ---------------------------------------------------------------------------
# Credentials & endpoint
# ---------------------------------------------------------------------------
# Secrets are read from (in priority order):
#   1. Environment variables HACKATHON_TEAM_ID / HACKATHON_API_KEY
#   2. An optional ignored .credentials.py file (see .gitignore)
# Placeholder defaults keep the code importable until you add real values.
_CRED_FILE = os.path.join(os.path.dirname(__file__), ".credentials.py")
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
UNKNOWN_MODELS = {
    "prac_01": {"n_features": 17, "budget": 500},
    "prac_02": {"n_features": 34, "budget": 500},
    "prac_03": {"n_features": 70, "budget": 500},
    "prac_04": {"n_features": 14, "budget": 500},
    "prac_05": {"n_features": 34, "budget": 500},
    "held_01": {"n_features": 70, "budget": 150},
    "held_02": {"n_features": 17, "budget": 150},
    "held_03": {"n_features": 34, "budget": 150},
    "held_04": {"n_features": 14, "budget": 150},
}

# Feature count -> reference family mapping (shape matching, free of cost).
# Each family's reference is the model id used for agreement comparison.
FEATURE_TO_FAMILY = {
    70: {"task": "mnist_digits_10class", "reference": "ref_01"},
    17: {"task": "wine_classification_3class", "reference": "ref_02"},
    34: {"task": "breast_cancer_binary", "reference": "ref_03"},
    14: {"task": "diabetes_progression_regression", "reference": "ref_04"},
}


def get_family(n_features: int):
    """Return the task family + reference model for a feature count."""
    return FEATURE_TO_FAMILY.get(n_features, {"task": "unknown", "reference": None})


# ---------------------------------------------------------------------------
# Request tuning
# ---------------------------------------------------------------------------
REQUEST_TIMEOUT = 30  # seconds
MAX_RETRIES = 3
RETRY_BACKOFF = 1.0  # seconds
