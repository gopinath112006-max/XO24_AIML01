"""test_profiling.py — Unit tests for core profiling primitives.

These tests exercise the pure/deterministic functions in starter_code_snippets
and config without making any API calls.

Run:
    python -m pytest test_profiling.py -v
    # or
    python test_profiling.py
"""

import json
import os
import sys
import unittest

# Ensure the repository root is on sys.path so src/ modules are importable.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src import config
from src.paths import DATA_DIR
from src.starter_code_snippets import (
    _class_task_from_n,
    _feature_stats,
    _nonfinite,
    _scale_mismatch,
    calculate_confidence,
    estimate_accuracy_range,
    estimate_performance,
    generate_profile,
    infer_task_by_shape,
    infer_task_by_type,
)


class TestConfigHelpers(unittest.TestCase):
    """Tests for config module helpers."""

    def test_get_family_known(self):
        fam = config.get_family(17)
        self.assertEqual(fam["task"], "wine_classification_3class")
        self.assertEqual(fam["reference"], "ref_02")

    def test_get_family_unknown(self):
        fam = config.get_family(999)
        self.assertEqual(fam["task"], "unknown")
        self.assertIsNone(fam["reference"])

    def test_feature_to_family_coverage(self):
        for nfeat in (14, 17, 34, 70):
            fam = config.get_family(nfeat)
            self.assertIsNotNone(fam["reference"], f"Missing reference for {nfeat} features")


class TestShapeInference(unittest.TestCase):
    """Tests for infer_task_by_shape."""

    def test_all_families(self):
        cases = {
            70: "mnist_digits_10class",
            17: "wine_classification_3class",
            34: "breast_cancer_binary",
            14: "diabetes_progression_regression",
        }
        for nfeat, expected_task in cases.items():
            result = infer_task_by_shape(nfeat)
            self.assertEqual(result["task"], expected_task)

    def test_unknown_family(self):
        result = infer_task_by_shape(42)
        self.assertEqual(result["task"], "unknown")
        self.assertIsNone(result["reference"])


class TestTypeDetection(unittest.TestCase):
    """Tests for infer_task_by_type."""

    def test_classification_output(self):
        otype = {"classification": True, "n_classes": 3, "regression": False, "scale": None}
        shape = {"task": "wine_classification_3class", "reference": "ref_02"}
        result = infer_task_by_type(otype, shape)
        self.assertEqual(result["task"], "wine_classification_3class")
        self.assertTrue(result["type_consistent"])

    def test_regression_output(self):
        otype = {"classification": False, "n_classes": None, "regression": True,
                 "scale": {"mean": 150.0, "std": 10.0}}
        shape = {"task": "diabetes_progression_regression", "reference": "ref_04"}
        result = infer_task_by_type(otype, shape)
        self.assertEqual(result["task"], "diabetes_progression_regression")
        self.assertTrue(result["type_consistent"])

    def test_classification_on_regression_shape(self):
        """A 14-feature model that outputs classification — task should be inferred from class count."""
        otype = {"classification": True, "n_classes": 2, "regression": False, "scale": None}
        shape = {"task": "diabetes_progression_regression", "reference": "ref_04"}
        result = infer_task_by_type(otype, shape)
        self.assertEqual(result["task"], "breast_cancer_binary")
        self.assertFalse(result["type_consistent"])


class TestClassTaskFromN(unittest.TestCase):
    def test_known_counts(self):
        self.assertEqual(_class_task_from_n(10), "mnist_digits_10class")
        self.assertEqual(_class_task_from_n(3), "wine_classification_3class")
        self.assertEqual(_class_task_from_n(2), "breast_cancer_binary")

    def test_unknown_count(self):
        self.assertEqual(_class_task_from_n(5), "classification_unknown")


class TestFeatureStats(unittest.TestCase):
    def test_all_families_have_bounds(self):
        for task in ("mnist_digits_10class", "wine_classification_3class",
                     "breast_cancer_binary", "diabetes_progression_regression"):
            lo, hi, center = _feature_stats(task)
            self.assertLess(lo, hi)
            self.assertLessEqual(lo, center)
            self.assertLessEqual(center, hi)


class TestNonfinite(unittest.TestCase):
    def test_normal(self):
        self.assertFalse(_nonfinite(1.0))
        self.assertFalse(_nonfinite(0))
        self.assertFalse(_nonfinite(-42.5))

    def test_nan(self):
        self.assertTrue(_nonfinite(float("nan")))

    def test_inf(self):
        self.assertTrue(_nonfinite(float("inf")))
        self.assertTrue(_nonfinite(float("-inf")))

    def test_none_like(self):
        self.assertTrue(_nonfinite(None))


class TestScaleMismatch(unittest.TestCase):
    def test_similar_scales(self):
        import numpy as np
        u = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        v = np.array([1.1, 2.1, 3.1, 4.1, 5.1])
        self.assertFalse(_scale_mismatch(u, v))

    def test_different_scales(self):
        import numpy as np
        u = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        v = np.array([100.0, 200.0, 300.0, 400.0, 500.0])
        self.assertTrue(_scale_mismatch(u, v))

    def test_zero_std(self):
        import numpy as np
        u = np.array([5.0, 5.0, 5.0])
        v = np.array([1.0, 2.0, 3.0])
        self.assertFalse(_scale_mismatch(u, v))


class TestConfidenceScoring(unittest.TestCase):
    """Tests for the evidence-weighted confidence formula."""

    def test_perfect_evidence(self):
        conf = calculate_confidence(
            shape_match=True, type_consistent=True,
            agreement_rate=0.95, n_comparison_probes=200,
            n_features=17, coverage_ok=True,
        )
        self.assertGreaterEqual(conf, 0.85)
        self.assertLessEqual(conf, 0.99)

    def test_no_shape_match(self):
        conf = calculate_confidence(
            shape_match=False, type_consistent=None,
            agreement_rate=0.95, n_comparison_probes=30,
            n_features=17,
        )
        self.assertLess(conf, 0.80)

    def test_low_agreement(self):
        conf = calculate_confidence(
            shape_match=True, type_consistent=True,
            agreement_rate=0.50, n_comparison_probes=30,
            n_features=17,
        )
        self.assertLess(conf, 0.50)

    def test_never_reaches_100(self):
        conf = calculate_confidence(
            shape_match=True, type_consistent=True,
            agreement_rate=1.0, n_comparison_probes=100,
            n_features=17, coverage_ok=True,
        )
        self.assertLessEqual(conf, 0.99)

    def test_degenerate_penalty(self):
        good = calculate_confidence(
            shape_match=True, type_consistent=True,
            agreement_rate=0.92, n_comparison_probes=30,
            n_features=17, coverage_ok=True,
        )
        bad = calculate_confidence(
            shape_match=True, type_consistent=True,
            agreement_rate=0.92, n_comparison_probes=30,
            n_features=17, coverage_ok=True, degenerate=True,
        )
        self.assertLess(bad, good)

    def test_scale_mismatch_penalty(self):
        good = calculate_confidence(
            shape_match=True, type_consistent=True,
            agreement_rate=0.92, n_comparison_probes=30,
            n_features=14,
        )
        bad = calculate_confidence(
            shape_match=True, type_consistent=True,
            agreement_rate=0.92, n_comparison_probes=30,
            n_features=14, scale_mismatch=True,
        )
        self.assertLess(bad, good)


class TestAccuracyEstimation(unittest.TestCase):
    def test_perfect_agreement(self):
        result = estimate_accuracy_range(1.0, 0.963)
        self.assertIsNotNone(result)
        self.assertAlmostEqual(result["best"], 0.973, places=2)

    def test_known_agreement(self):
        result = estimate_accuracy_range(0.85, 0.963)
        self.assertIsNotNone(result)
        self.assertGreater(result["best"], result["lower"])
        self.assertLess(result["best"], result["upper"])

    def test_none_inputs(self):
        self.assertIsNone(estimate_accuracy_range(None, 0.963))
        self.assertIsNone(estimate_accuracy_range(0.85, None))


class TestEstimatePerformance(unittest.TestCase):
    def test_good_classification(self):
        result = estimate_performance(0.92, "wine_classification_3class",
                                       est_range={"best": 0.92, "lower": 0.89, "upper": 0.95})
        self.assertIn("Good", result)

    def test_weak_classification(self):
        result = estimate_performance(0.55, "breast_cancer_binary",
                                       est_range={"best": 0.55, "lower": 0.53, "upper": 0.57})
        self.assertIn("Weak", result)

    def test_regression_with_correlation(self):
        result = estimate_performance(0.8, "diabetes_progression_regression", corr=0.8)
        self.assertIn("R", result)

    def test_regression_unknown(self):
        result = estimate_performance(0.1, "diabetes_progression_regression", corr=0.1)
        self.assertEqual(result, "Unknown")


class TestGenerateProfile(unittest.TestCase):
    def test_minimal_profile(self):
        family = {
            "task": "wine_classification_3class", "reference": "ref_02",
            "type_consistent": True, "n_features": 17, "budget": 10000,
            "degenerate": False, "coverage_collapse": False,
        }
        otype = {"classification": True, "n_classes": 3, "scale": None}
        coverage = {"distinct_classes": 3, "observed": [0, 1, 2]}
        profile = generate_profile(
            "prac_01", "practice", family,
            agreement_rate=0.85, n_comparison_probes=15,
            weaknesses=[], otype=otype, coverage=coverage,
        )
        self.assertEqual(profile["model_id"], "prac_01")
        self.assertEqual(profile["pool_set"], "practice")
        self.assertEqual(profile["inferred_task"], "wine_classification_3class")
        self.assertGreater(profile["task_confidence"], 0)
        self.assertLessEqual(profile["task_confidence"], 0.99)
        self.assertIn("probe_utilization", profile)
        self.assertIn("evidence", profile)

    def test_degenerate_profile_has_weakness(self):
        family = {
            "task": "mnist_digits_10class", "reference": "ref_01",
            "type_consistent": True, "n_features": 70, "budget": 10000,
            "degenerate": True, "coverage_collapse": False,
        }
        otype = {"classification": True, "n_classes": 10, "scale": None}
        coverage = {"distinct_classes": 1, "observed": [4]}
        weaknesses = [{
            "type": "degenerate_output",
            "description": "constant classifier; emits class 4",
            "severity": "high",
        }]
        profile = generate_profile(
            "held_05", "held_out", family,
            agreement_rate=0.785, n_comparison_probes=400,
            weaknesses=weaknesses, otype=otype, coverage=coverage,
        )
        self.assertTrue(profile["evidence"]["degenerate"])
        high_ws = [w for w in profile["weaknesses"] if w["severity"] == "high"]
        self.assertGreater(len(high_ws), 0)


class TestProfilesJsonStructure(unittest.TestCase):
    """Validate that the existing profiles.json has the expected schema."""

    PROFILES_PATH = os.path.join(DATA_DIR, "profiles.json")

    @unittest.skipUnless(
        os.path.exists(os.path.join(DATA_DIR, "profiles.json")),
        "profiles.json not found"
    )
    def test_profiles_json_valid(self):
        with open(self.PROFILES_PATH, encoding="utf-8") as f:
            data = json.load(f)
        self.assertIn("profiles", data)
        self.assertIsInstance(data["profiles"], list)
        self.assertEqual(len(data["profiles"]), 13)

    @unittest.skipUnless(
        os.path.exists(os.path.join(DATA_DIR, "profiles.json")),
        "profiles.json not found"
    )
    def test_each_profile_has_required_fields(self):
        with open(self.PROFILES_PATH, encoding="utf-8") as f:
            data = json.load(f)
        required = {"model_id", "pool_set", "inferred_task", "task_confidence",
                     "estimated_performance", "agreement_with_reference",
                     "weaknesses", "probe_utilization", "evidence"}
        for p in data["profiles"]:
            missing = required - set(p.keys())
            self.assertEqual(missing, set(), f"{p.get('model_id', '?')} missing {missing}")

    @unittest.skipUnless(
        os.path.exists(os.path.join(DATA_DIR, "profiles.json")),
        "profiles.json not found"
    )
    def test_confidence_bounds(self):
        with open(self.PROFILES_PATH, encoding="utf-8") as f:
            data = json.load(f)
        for p in data["profiles"]:
            c = p["task_confidence"]
            self.assertGreaterEqual(c, 0.0, f"{p['model_id']} confidence < 0")
            self.assertLessEqual(c, 0.99, f"{p['model_id']} confidence > 0.99")


if __name__ == "__main__":
    unittest.main()
