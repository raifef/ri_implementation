from __future__ import annotations

import unittest

from hdfa_rl_suite.evaluation.benchmark import BenchmarkRunner
from hdfa_rl_suite.evaluation.launch import load_launch_definition
from hdfa_rl_suite.product import HDFAProductController, ProductLoopConfig
from hdfa_rl_suite.recovery import RecoveryScope, ReentryReason
from hdfa_rl_suite.simulator import (
    DriftKind, LatentProcessSpec, ScalableQECDevice, SimulatorConfig,
)
from hdfa_rl_suite.stage0 import ScalableBootstrapConfig
from hdfa_rl_suite.validation.compute_sanity import run_compute_accounting_validation
from hdfa_rl_suite.validation.post_comparison import _rollback_fault_evidence


class PostComparisonHardeningTests(unittest.TestCase):
    def _product(self) -> HDFAProductController:
        device = ScalableQECDevice(SimulatorConfig(
            qubit_count=3, seed=908, cycle_period_s=1e-4,
            processes=(LatentProcessSpec(
                "stationary", DriftKind.CONSTANT, {}, amplitude=0.0),)))
        return HDFAProductController(device, seed=908, config=ProductLoopConfig(
            residual_candidate_count=4, residual_candidate_cycles=4,
            bootstrap=ScalableBootstrapConfig(
                characterization_shots=96, validation_cycles=512,
                target_posterior_stddev=.07, qec_detector_rate_limit=.20)))

    def test_global_ood_recovery_is_not_relabelled_stationary_stage0(self):
        product = self._product()
        first = product.run_interval(64)
        self.assertEqual(first.bootstrap_count, 1)
        product.request_reentry(
            ReentryReason.OOD_RECALIBRATION, interval=1,
            scope=RecoveryScope.GLOBAL,
            triggering_evidence={"broad_causal_ood": True})
        recovered = product.run_interval(64, interval=1)
        self.assertIsNone(recovered.bootstrap)
        self.assertEqual(recovered.bootstrap_count, 1)
        self.assertEqual(recovered.recovery_count, 1)
        self.assertIn(
            "recovery:online_disturbance_aware_global", recovered.stage_path)
        self.assertEqual(recovered.stage_path[0], "stage0:validated_cache")
        self.assertTrue(recovered.regional_recovery.passed)
        self.assertEqual(recovered.regional_recovery.request.scope,
                         RecoveryScope.GLOBAL)

    def test_rollback_faults_are_separate_and_fail_closed(self):
        evidence = _rollback_fault_evidence()
        self.assertTrue(evidence["passed"])
        self.assertEqual(evidence["ack_hash_transaction_status"], "failed")
        self.assertEqual(evidence["ack_hash_physical_status"], "not_evaluated")
        self.assertEqual(evidence["unsafe_transaction_status"], "confirmed")
        self.assertEqual(evidence["unsafe_physical_status"], "failed")

    def test_compute_evidence_fault_matrix(self):
        report = run_compute_accounting_validation()
        check = next(item for item in report.checks
                     if item.check_id == "compute_evidence_fault_matrix")
        self.assertTrue(check.passed)
        self.assertEqual(report.metadata["fault_count"], 11)
        self.assertTrue(all(row["detected"] for row in check.observed.values()))

    def test_confirmatory_launch_is_protocol_bound_and_disjoint(self):
        definition = load_launch_definition(
            "experiments/authoritative_acceptance/compute-aware-confirmation-v2.json")
        self.assertEqual(definition.protocol_id,
                         "hdfa-rl-compute-aware-confirmation.v2")
        self.assertEqual(len(definition.protocol_sha256), 64)
        self.assertEqual(len(definition.launch_file_sha256), 64)
        self.assertTrue(set(definition.config.seeds).isdisjoint({101, 102, 103, 104, 105}))
        self.assertTrue(all(item.scenario_id.startswith("confirmatory_")
                            for item in definition.scenarios()))
        runner = BenchmarkRunner(
            definition.config, definition.scenarios(),
            launch_binding_hash=definition.configuration_hash)
        self.assertEqual(runner.launch_configuration_hash,
                         definition.configuration_hash)


if __name__ == "__main__":
    unittest.main()
