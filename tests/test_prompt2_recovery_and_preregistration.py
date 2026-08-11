from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from hdfa_rl_suite.evaluation.evidence import validate_report_payload
from hdfa_rl_suite.evaluation.launch import load_launch_definition
from hdfa_rl_suite.evaluation.next_steps import (
    CONFIRMATORY_V3_SEEDS, CONSUMED_SEEDS, run_all_post_amendment,
    run_one_interval_development, validate_rmst_support,
)


class Prompt2DevelopmentTests(unittest.TestCase):
    def test_familiar_recurrence_gate_does_not_change_interval_or_safety(self):
        with tempfile.TemporaryDirectory() as directory:
            report = run_one_interval_development(Path(directory))
        self.assertTrue(report["passed"])
        self.assertGreaterEqual(report["familiar_one_interval_fraction"], .90)
        self.assertEqual(report["cycles_per_interval"], 512)
        self.assertTrue(all(
            not row["interval_duration_changed"]
            and not row["threshold_definition_changed"]
            and not row["safety_constraints_changed"]
            for row in report["rows"]))
        self.assertTrue(all(
            row["residual_did_not_delay_first_prediction"]
            for row in report["rows"]))

    def test_estimator_validator_rejects_cross_attached_ci(self):
        ci = {"lower": 1.1, "estimate": 1.2, "upper": 1.3, "confidence": .95}
        estimators = {
            "worst_matched_ratio": .8, "median_matched_ratio": 1.0,
            "cluster_aggregate_ratio": 1.2, "cluster_aggregate_ci95": ci,
            "rmst_difference": None, "rmst_ci95": None,
            "tail_difference": None, "tail_ci95": None,
            "gate_decision_statistic": .8, "gate_threshold": .75,
            "gate_status": "pass",
        }
        payload = {
            "config": {"estimator_schema_version": "estimators.v2"},
            "evidence_records": [{"result_id": "x", "layer": "declared_surrogate",
                                  "description": "schema test", "measurement_role": "test",
                                  "source": "unit"}],
            "gates": [{"gate_id": "x", "status": "pass", "measured_ratio": .8,
                       "confidence_interval": ci, "estimators": estimators,
                       "estimand": {}}],
        }
        codes = {issue.code for issue in validate_report_payload(payload)}
        self.assertIn("estimator_ci_ambiguous", codes)

    def test_post_amendment_bundle_preregisters_without_consuming_seeds(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            # Protocol binding resolves relative to repository root, so execute the full
            # preregistration only in the real workspace in integration tests. Here the
            # seed firewall and fail-closed support configuration are unit tested.
            self.assertFalse(set(CONFIRMATORY_V3_SEEDS).intersection(CONSUMED_SEEDS))
            config = root/"config.json"
            config.write_text(json.dumps({
                "scenario_ids": ["s"],
                "benchmark": {"e2e_rmst_horizon_s": 8.0,
                              "minimum_e2e_followup_support_s": 7.9,
                              "seeds": [5001]}}), encoding="utf-8")
            report = validate_rmst_support(config_path=config)
            self.assertFalse(report["passed"])
            self.assertEqual(report["failure_rows"][0]["scenario_id"], "s")

    def test_checked_in_v3_launch_is_loadable_and_fresh(self):
        path = Path("configs/acceptance/confirmatory-v3.yaml")
        if not path.exists():
            self.skipTest("preregistration artifact not generated")
        definition = load_launch_definition(path)
        self.assertEqual(definition.config.e2e_rmst_horizon_s, 8.0)
        self.assertGreaterEqual(definition.config.minimum_e2e_followup_support_s, 9.0)
        self.assertEqual(definition.config.gate_reference_arm,
                         "predictive_hdfa_no_residual")
        self.assertFalse(set(definition.config.seeds).intersection(CONSUMED_SEEDS))


if __name__ == "__main__":
    unittest.main()
