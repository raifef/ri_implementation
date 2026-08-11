"""Deterministic regressions derived from immutable authoritative-comparison-v1.

These tests were deliberately added before changing recovery or rollback behaviour.
The legacy fingerprints are retained as development evidence; after the production
repair the behavioural assertions become the no-regression acceptance conditions.
"""
from __future__ import annotations

from dataclasses import replace
import unittest

from hdfa_rl_suite.evaluation.benchmark import BenchmarkRunner
from hdfa_rl_suite.evaluation.launch import load_launch_definition


class PostComparisonRepairTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        definition = load_launch_definition(
            "experiments/physical_validation/authoritative-comparison-v1.json")
        # Circuit-level logical evaluation is evaluation-only and cannot influence the
        # controller.  Eight shots retain the shared-state contract while making these
        # exact controller regressions inexpensive.
        cls.config = replace(
            definition.config, authoritative=False, logical_shots_per_interval=8)
        cls.scenarios = {item.scenario_id: item for item in definition.scenarios()}

    def _run_staged(self, scenario_id: str, seed: int):
        scenario = self.scenarios[scenario_id]
        runner = BenchmarkRunner(self.config, (scenario,))
        prepared = runner._prepare_matched_state(scenario, seed)
        return runner._run_arm(
            scenario, seed, "predictive_hdfa_residual_rl",
            runner.arm_factories["predictive_hdfa_residual_rl"], prepared)

    def test_nested_common_102_completes_via_certified_regional_recovery(self):
        metric, trajectories, baseline = self._run_staged("nested_common", 102)
        self.assertEqual(metric.completion_status, "completed")
        self.assertEqual(len(trajectories), 32)
        self.assertLessEqual(metric.rollback_count, 2)
        self.assertEqual(metric.bootstrap_count, 1)
        self.assertEqual(metric.lifecycle_violation_count, 0)
        self.assertEqual(metric.physical_rollback_failure_count, 0)
        self.assertEqual(baseline.observation_hash,
                         "2a66bb9fa0bb8b9d0296b7946cd465620c3d07157d48b2c795b1ca53ff67a20e")
        recoveries = [item.regional_recovery for item in trajectories
                      if item.regional_recovery]
        self.assertTrue(recoveries)
        self.assertTrue(all(item["passed"] for item in recoveries))
        self.assertTrue(all(item["gate_results"]["boundary_validation"]
                            and item["gate_results"]["unaffected_policy_frozen"]
                            for item in recoveries))
        requests = [item.reentry_request for item in trajectories
                    if item.reentry_request]
        self.assertTrue(any(str(item["scope"]).lower().endswith("regional")
                            for item in requests))

    def test_unknown_105_has_bounded_global_recovery_without_hidden_failure(self):
        metric, trajectories, baseline = self._run_staged("unknown", 105)
        self.assertEqual(metric.completion_status, "completed")
        self.assertEqual(len(trajectories), 32)
        self.assertLessEqual(metric.rollback_count, 4)
        self.assertLessEqual(metric.bootstrap_count, 5)
        self.assertEqual(metric.lifecycle_violation_count, 0)
        self.assertEqual(metric.physical_rollback_failure_count, 0)
        self.assertEqual(baseline.observation_hash,
                         "51f9ee0959e107549cc8d4a7a788663431e6ee6c99b89513a3e8ee6c0e03218d")
        self.assertFalse(any(item.physical_rollback_failures
                             for item in trajectories))
        requests = [item.reentry_request for item in trajectories
                    if item.reentry_request]
        self.assertTrue(all(not str(item["scope"]).lower().endswith("regional")
                            for item in requests))
        for trajectory in trajectories:
            for outcome in trajectory.rollback_outcomes:
                self.assertEqual(str(outcome["transaction_status"]),
                                 "TransactionRestorationStatus.CONFIRMED")
                self.assertEqual(outcome["expected_final_hash"],
                                 outcome["observed_final_hash"])


if __name__ == "__main__":
    unittest.main()
