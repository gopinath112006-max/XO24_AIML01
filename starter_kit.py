"""starter_kit.py — Working examples of all 3 API endpoints.

Run this first (after filling in credentials in config.py) to confirm
you can reach the API and that your TEAM_ID / API_KEY are correct.

    python starter_kit.py

Expected output (trimmed):
    13 models available:
      ref_01               expects  70 features, budget 10000, pool_set=reference
      ...
"""

import json
import random
import sys

import requests

import config

# ---------------------------------------------------------------------------
# 1. API helper functions
# ---------------------------------------------------------------------------


def api_get(path: str):
    """Generic GET helper with retries."""
    url = config.GENERAL_POOL_URL + path
    last_exc = None
    for _ in range(config.MAX_RETRIES):
        try:
            resp = requests.get(url, headers=config.AUTH_HEADERS, timeout=config.REQUEST_TIMEOUT)
            resp.raise_for_status()
            return resp.json()
        except (requests.RequestException, ValueError) as exc:
            last_exc = exc
    raise RuntimeError(f"GET {url} failed after {config.MAX_RETRIES} attempts: {last_exc}")


def api_post(path: str, body: dict):
    """Generic POST helper with retries."""
    url = config.GENERAL_POOL_URL + path
    last_exc = None
    for _ in range(config.MAX_RETRIES):
        try:
            resp = requests.post(url, json=body, headers=config.AUTH_HEADERS, timeout=config.REQUEST_TIMEOUT)
            resp.raise_for_status()
            return resp.json()
        except (requests.RequestException, ValueError) as exc:
            last_exc = exc
    raise RuntimeError(f"POST {url} failed after {config.MAX_RETRIES} attempts: {last_exc}")


# ---------------------------------------------------------------------------
# 2. The three endpoints
# ---------------------------------------------------------------------------


def list_models():
    """GET /models — Free. Returns all available models + feature counts."""
    return api_get("/models")


def predict(model_id: str, inputs: list):
    """POST /model/{model_id}/predict — Costs 1 probe per input row."""
    body = {"team_id": config.TEAM_ID, "inputs": inputs}
    return api_post(f"/model/{model_id}/predict", body)


def get_usage():
    """GET /team/{team_id}/usage — Free. Returns per-model probe usage."""
    return api_get(f"/team/{config.TEAM_ID}/usage")


# ---------------------------------------------------------------------------
# 3. Demo / sanity check
# ---------------------------------------------------------------------------


def demo_models():
    models = list_models()
    print(f"\n{len(models)} models available:")
    for m in models:
        print(f"  {m['model_id']:<18} expects {m['n_features_expected']:>4} features, "
              f"budget {m['probe_budget']:>6}, pool_set={m['pool_set']}")


def demo_predict(model_id="ref_02", n_features=17, n_probes=3):
    # Build random inputs in a plausible feature range.
    inputs = [[random.gauss(0, 1) for _ in range(n_features)] for _ in range(n_probes)]
    result = predict(model_id, inputs)
    print(f"\nPredicted on {model_id} with {n_probes} probes:")
    print(f"  predictions = {result.get('predictions')}")
    print(f"  probabilities present: {'probabilities' in result}")
    print(f"  probes_used_this_model = {result.get('probes_used_this_model')}")
    print(f"  probes_remaining_this_model = {result.get('probes_remaining_this_model')}")


def demo_usage():
    usage = get_usage()
    print("\nTeam usage:")
    for model_id, info in usage.items():
        print(f"  {model_id:<12} used {info['used']:>6} / budget {info['budget']:>6} / "
              f"remaining {info['remaining']:>6}")


def main():
    if config.TEAM_ID == "your_team_id" or config.API_KEY == "your_api_key":
        print("ERROR: Fill in TEAM_ID and API_KEY in config.py first (or set the "
              "HACKATHON_TEAM_ID / HACKATHON_API_KEY env vars).", file=sys.stderr)
        sys.exit(1)

    # Endpoint 1 (free)
    demo_models()

    # Endpoint 2 (costs 3 probes on ref_02 of 10,000)
    demo_predict()

    # Endpoint 3 (free)
    demo_usage()

    print("\nConnection OK — credentials verified.")


if __name__ == "__main__":
    main()
