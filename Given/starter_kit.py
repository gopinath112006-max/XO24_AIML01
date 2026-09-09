"""
STARTER KIT — Hackathon Model Pool

This script shows you the MECHANICS of talking to the API: authentication,
sending probes, reading responses, checking your remaining budget.

It does NOT contain any analysis logic -- figuring out what each model
does, how good it is, and how confident you should be is the actual
challenge. Build that part yourselves.

Fill in GENERAL_POOL_URL, TEAM_ID, and API_KEY below with your own values
before running.
"""
import requests

# --- Fill these in with YOUR team's details ---------------------------------
GENERAL_POOL_URL = "https://hackathon-general-pool.onrender.com"   # <-- organizer gives you this
SURPRISE1_POOL_URL = "https://hackathon-surprise1-pool.onrender.com"  # <-- and this
TEAM_ID = "your_team_id"       # <-- from your credentials sheet
API_KEY = "your_api_key"       # <-- from your credentials sheet


def probe(base_url: str, model_id: str, inputs: list[list[float]]) -> dict:
    """Sends a batch of inputs to a model and returns its response.

    `inputs` is a list of rows, where each row is a list of numbers. Check
    GET /models first to see how many numbers each model_id expects per row.

    Every call here costs probes against your team's budget for that model
    -- batching multiple rows in one call still costs one probe per row,
    it just saves you HTTP round-trips.
    """
    r = requests.post(
        f"{base_url}/model/{model_id}/predict",
        json={"team_id": TEAM_ID, "inputs": inputs},
        headers={"x-team-key": API_KEY},
    )
    r.raise_for_status()  # raises an error if something went wrong (bad key, budget exceeded, etc.)
    return r.json()


def list_models(base_url: str) -> list[dict]:
    """Returns every model_id currently visible on this pool, with how many
    features (numbers) each one expects and your remaining probe budget."""
    r = requests.get(f"{base_url}/models")
    r.raise_for_status()
    return r.json()


def check_my_usage(base_url: str) -> dict:
    """See exactly how many probes you've used per model so far."""
    r = requests.get(
        f"{base_url}/team/{TEAM_ID}/usage",
        headers={"x-team-key": API_KEY},
    )
    r.raise_for_status()
    return r.json()


if __name__ == "__main__":
    # --- Example 1: see what models exist ---
    models = list_models(GENERAL_POOL_URL)
    print(f"{len(models)} models available:")
    for m in models:
        print(f"  {m['model_id']:20s} expects {m['n_features_expected']:3d} features, "
              f"budget {m['probe_budget']}, pool_set={m['pool_set']}")

    # --- Example 2: probe one model with a single made-up input ---
    # (Replace this with real inputs and real analysis logic -- this is
    # just showing the mechanics work.)
    example_model = models[0]["model_id"]
    n_features = models[0]["n_features_expected"]
    fake_input = [[0.0] * n_features]  # one row of all-zero values, as a placeholder

    result = probe(GENERAL_POOL_URL, example_model, fake_input)
    print(f"\nProbed '{example_model}':")
    print(f"  predictions: {result['predictions']}")
    print(f"  probes used on this model so far: {result['probes_used_this_model']}")

    # --- Example 3: check your team's overall usage ---
    usage = check_my_usage(GENERAL_POOL_URL)
    print(f"\nYour usage on '{example_model}': {usage[example_model]}")

    # --- From here, it's up to you ---
    # Ideas to build on top of this (not required, just a starting point):
    #   - Compare a reference model's behavior against an unknown model of
    #     the same feature count, using the SAME inputs on both.
    #   - Vary inputs systematically and see what changes the prediction.
    #   - Track how many probes you've spent per model, and reflect that
    #     as a confidence level in your final writeup.
