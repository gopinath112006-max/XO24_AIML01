"""test_strategy.py — Unit tests for strategy.py pure helpers.

These tests exercise compute_agreement, decide_family, config.rank_references,
_encode_rows and _cap_deep_trials WITHOUT making any API calls.

Run:
    python -m pytest test_strategy.py -v
    # or
    python test_strategy.py
"""

import os
import sys
import unittest

import numpy as np

# Ensure the repository root is on sys.path so src/ modules are importable.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src import config
from src.config import STRATEGY_CONFIRM_AGREEMENT as CONFIRM
from src.config import STRATEGY_CONTROL_MAX as CTRL
from src.config import STRATEGY_DEEP_AMBIGUOUS as DEEP_AMB
from src.config import STRATEGY_DEEP_CONVERGED as DEEP_CONV
from src.config import STRATEGY_REJECT_AGREEMENT as LO
from src.config import STRATEGY_SIBLING_CORROBORATE as SIB
from src.strategy import _cap_deep_trials, _encode_rows, compute_agreement, decide_family


def _shape(task="mnist_digits_10class", reference="ref_01", n_features=70):
    return {"task": task, "reference": reference, "n_features": n_features}


class TestComputeAgreement(unittest.TestCase):
    """compute_agreement: classification + regression, edge cases."""

    def test_classification_identical(self):
        res = compute_agreement([1, 2, 3, 4], [1, 2, 3, 4], "mnist_digits_10class")
        self.assertEqual(res["agreement"], 1.0)
        self.assertEqual(res["n"], 4)

    def test_classification_half(self):
        res = compute_agreement([1, 2, 3, 4], [1, 1, 1, 1], "wine_classification_3class")
        self.assertEqual(res["agreement"], 0.25)
        self.assertEqual(res["n"], 4)

    def test_classification_empty(self):
        res = compute_agreement([None, None], [1, 2], "breast_cancer_binary")
        self.assertEqual(res["agreement"], 0.0)
        self.assertEqual(res["n"], 0)

    def test_regression_scale_invariant(self):
        a = [10.0, 20.0, 30.0, 40.0, 50.0]
        b = [2.0, 4.0, 6.0, 8.0, 10.0]
        res = compute_agreement(a, b, "diabetes_progression_regression")
        self.assertGreater(res["agreement"], 0.999)
        self.assertIsNotNone(res["correlation"])

    def test_regression_inverse_relation(self):
        a = [1.0, 2.0, 3.0, 4.0]
        b = [40.0, 30.0, 20.0, 10.0]
        res = compute_agreement(a, b, "diabetes_progression_regression")
        self.assertEqual(res["agreement"], 0.0)
        self.assertLess(res["correlation"], 0.0)

    def test_nonfinite_filtered(self):
        res = compute_agreement([1, float("nan"), 3, 4], [1, 2, 3, 4], "mnist_digits_10class")
        self.assertEqual(res["n"], 3)
        self.assertEqual(res["agreement"], 1.0)


class TestDecideFamily(unittest.TestCase):
    """All six verdict branches of the decision tree."""

    def _screen(self, primary=0.0, slice_=0.0, sib=0.0, degenerate=False, slices=None):
        return {
            "primary_agreement": primary,
            "max_slice_agreement": slice_,
            "max_sibling_agreement": sib,
            "degenerate": degenerate,
            "slices": slices if slices is not None else [],
        }

    def test_degenerate(self):
        d = decide_family(self._screen(degenerate=True), _shape())
        self.assertEqual(d["verdict"], "degenerate")
        self.assertEqual(d["deep_trials"], 0)

    def test_confirmed(self):
        d = decide_family(self._screen(primary=CONFIRM + 0.01, slice_=0.4), _shape())
        self.assertEqual(d["verdict"], "confirmed")
        self.assertEqual(d["deep_trials"], DEEP_CONV)
        self.assertEqual(d["family"]["reference"], "ref_01")

    def test_confirmed_requires_clean_cut_over_slices(self):
        # High primary but a slice control at/above the cap: NOT confirmed —
        # elevated controls mean the primary family can't be trusted outright.
        d = decide_family(self._screen(primary=CONFIRM + 0.01, slice_=CTRL + 0.1), _shape())
        self.assertEqual(d["verdict"], "ambiguous")
        self.assertEqual(d["deep_trials"], DEEP_AMB)

    def test_reanchored(self):
        slices = [{"reference": "ref_03", "task": "breast_cancer_binary",
                   "n_features": 34, "agreement": CONFIRM + 0.02}]
        d = decide_family(self._screen(primary=0.5, slice_=CONFIRM + 0.02,
                                       slices=slices), _shape())
        self.assertEqual(d["verdict"], "re-anchored")
        self.assertEqual(d["family"]["reference"], "ref_03")
        self.assertEqual(d["family"]["ref_n_features"], 34)
        self.assertEqual(d["deep_trials"], DEEP_AMB)

    def test_ambiguous_band(self):
        mid = (LO + CONFIRM) / 2.0
        d = decide_family(self._screen(primary=mid), _shape())
        self.assertEqual(d["verdict"], "ambiguous")
        self.assertEqual(d["deep_trials"], DEEP_AMB)

    def test_off_corpus(self):
        d = decide_family(self._screen(primary=0.3, sib=SIB + 0.01), _shape())
        self.assertEqual(d["verdict"], "off-corpus")
        self.assertEqual(d["deep_trials"], DEEP_CONV)

    def test_unclassified(self):
        d = decide_family(self._screen(primary=0.5, slice_=0.4, sib=0.4), _shape())
        self.assertEqual(d["verdict"], "unclassified")
        self.assertEqual(d["deep_trials"], 0)


class TestRankReferences(unittest.TestCase):
    """config.rank_references ordering + sliceability."""

    def test_exact_match_first(self):
        ranked = config.rank_references(70, limit=4)
        self.assertTrue(ranked[0]["is_primary"])
        self.assertEqual(ranked[0]["reference"], "ref_01")

    def test_ordered_by_distance(self):
        ranked = config.rank_references(17, limit=4)
        self.assertEqual(ranked[0]["reference"], "ref_02")
        # After the exact match, distance ordering: |70-17|=53 > |34-17|=17 > |14-17|=3
        distances = [abs(r["n_features"] - 17) for r in ranked[1:]]
        self.assertEqual(distances, sorted(distances))

    def test_sliceable_only_smaller(self):
        ranked = config.rank_references(70, limit=4)
        for r in ranked:
            if not r["is_primary"]:
                self.assertTrue(r["sliceable"], f"{r['reference']} should be sliceable < 70")
        ranked_17 = config.rank_references(17, limit=4)
        for r in ranked_17:
            if r["sliceable"]:
                self.assertLess(r["n_features"], 17)


class TestEncodeRows(unittest.TestCase):
    """_encode_rows: shape preservation + transforms."""

    def test_identity_shapes_match(self):
        arr = np.asarray([[1.0, 2.0], [3.0, 4.0]])
        out = _encode_rows(arr, "identity")
        self.assertEqual(out.shape, (2, 2))

    def test_log1p_positive(self):
        arr = np.asarray([[0.0], [5.0]])
        out = _encode_rows(arr, "log1p")
        self.assertAlmostEqual(out[0, 0], 0.0)
        self.assertGreater(out[1, 0], 0.0)

    def test_zscore_centers_and_scales(self):
        arr = np.asarray([[0.0], [10.0]])
        out = _encode_rows(arr, "zscore")
        self.assertAlmostEqual(float(out.mean()), 0.0, places=6)
        self.assertAlmostEqual(float(out.std()), 1.0, places=4)

    def test_minmax_in_range(self):
        arr = np.asarray([[2.0], [-2.0]])
        out = _encode_rows(arr, "minmax")
        self.assertAlmostEqual(float(out.min()), 0.0)
        self.assertAlmostEqual(float(out.max()), 1.0)

    def test_unknown_encoding_raises(self):
        with self.assertRaises(ValueError):
            _encode_rows(np.asarray([[1.0]]), "bogus")


class TestCapDeepTrials(unittest.TestCase):
    """_cap_deep_trials: budget boundary logic."""

    def test_no_usage_returns_wanted(self):
        self.assertEqual(_cap_deep_trials(150, "ref_01", None), 150)

    def test_empty_usage_returns_wanted(self):
        self.assertEqual(_cap_deep_trials(150, "ref_01", {}), 150)

    def test_cap_reserves_margin(self):
        usage = {"ref_01": {"used": 9600, "budget": 10000}}
        # remaining 400, margin 300 -> cap 100.
        self.assertEqual(_cap_deep_trials(150, "ref_01", usage), 100)

    def test_no_room_yields_zero(self):
        usage = {"ref_01": {"used": 9850, "budget": 10000}}
        self.assertEqual(_cap_deep_trials(200, "ref_01", usage), 0)

    def test_ref_missing_returns_zero_only_without_usage(self):
        self.assertEqual(_cap_deep_trials(0, "ref_01", None), 0)
        self.assertEqual(_cap_deep_trials(0, "ref_01", {}), 0)

    def test_corner_used_and_budget(self):
        # exactly budget - margin remaining -> full wanted passes
        usage = {"ref_01": {"used": 10000 - 400 - 150, "budget": 10000}}
        self.assertEqual(_cap_deep_trials(150, "ref_01", usage), 150)


if __name__ == "__main__":
    unittest.main()
