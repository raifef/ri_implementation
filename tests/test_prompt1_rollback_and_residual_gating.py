from pathlib import Path
import tempfile
import unittest

from hdfa_rl_suite.evaluation.residual_ablation import (
    run_residual_ablation, validate_residual_gating,
)
from hdfa_rl_suite.evaluation.rollback_reproduction import validate_rollback_semantics
from hdfa_rl_suite.product import HDFAProductController, ProductLoopConfig
from hdfa_rl_suite.simulator import (
    DriftKind, LatentProcessSpec, ScalableQECDevice, SimulatorConfig,
)
from hdfa_rl_suite.stage0 import ScalableBootstrapConfig
from hdfa_rl_suite.stage6 import ResidualRLDisposition


class PromptOneDevelopmentTests(unittest.TestCase):
    def test_four_state_rollback_semantics(self):
        with tempfile.TemporaryDirectory() as directory:
            self.assertTrue(validate_rollback_semantics(Path(directory))["passed"])

    def test_gate_and_deactivation_rules(self):
        with tempfile.TemporaryDirectory() as directory:
            self.assertTrue(validate_residual_gating(Path(directory))["passed"])

    def test_development_ablation_is_explicitly_nonconfirmatory(self):
        with tempfile.TemporaryDirectory() as directory:
            report = run_residual_ablation(Path(directory))
        self.assertTrue(report["claim_supported"])
        self.assertFalse(report["confirmatory_seeds_used"])
        self.assertIn("pure_shot_noise", report["scenarios"])

    def test_normal_product_path_may_abstain(self):
        processes = (LatentProcessSpec(
            "stationary", DriftKind.CONSTANT,
            {f"drive:q{i}": 0.0 for i in range(3)}, amplitude=0.0),)
        device = ScalableQECDevice(SimulatorConfig(
            qubit_count=3, seed=22, cycle_period_s=.001, processes=processes))
        config = ProductLoopConfig(
            residual_candidate_count=4, residual_candidate_cycles=4,
            bootstrap=ScalableBootstrapConfig(
                characterization_shots=64, validation_cycles=64,
                target_posterior_stddev=.07, qec_detector_rate_limit=.20))
        result = HDFAProductController(device, seed=22, config=config).run_interval(64)
        self.assertEqual(result.residual_gate_decision.disposition,
                         ResidualRLDisposition.ABSTAIN)
        self.assertEqual(result.residual_candidates, ())
        self.assertFalse(result.lifecycle_violations)


if __name__ == "__main__":
    unittest.main()

