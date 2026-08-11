import unittest

from hdfa_rl_suite.baselines.controllers import FullControlRLArm
from hdfa_rl_suite.simulator import DriftKind, LatentProcessSpec, ScalableQECDevice, SimulatorConfig
from hdfa_rl_suite.validation import (
    run_controller_validation,
    run_plant_validation,
    run_preflight,
    run_sample_budget_validation,
)


class PlantPhysicalInvariantTests(unittest.TestCase):
    def test_canonical_plant_ladder_passes(self):
        report = run_plant_validation()
        self.assertTrue(report.passed)
        self.assertTrue(report.trajectories)
        self.assertTrue(all(check.passed for check in report.checks))

    def test_each_declared_plant_fault_is_caught_by_its_gate(self):
        cases = {
            "no_disturbance_drift": "no_disturbance_stationarity",
            "step_reset": "persistent_step_degrades_fixed",
            "oracle_bias": "oracle_periodic_fixed_ordering",
            "nonmonotonic_response": "response_monotonicity",
            "sinusoid_phase": "sinusoid_period_phase_and_envelope",
            "rtn_alias": "rtn_state_and_dwell_statistics",
            "ou_clone_mismatch": "ou_persistence_and_clone_identity",
            "state_id_mismatch": "ou_persistence_and_clone_identity",
        }
        for fault, check_id in cases.items():
            with self.subTest(fault=fault):
                report = run_plant_validation(injected_faults=(fault,))
                check = next(item for item in report.checks if item.check_id == check_id)
                self.assertFalse(check.passed)
                self.assertFalse(report.passed)

    def test_physical_state_and_candidate_policy_ids_are_aligned(self):
        device = ScalableQECDevice(SimulatorConfig(
            qubit_count=3, cycle_period_s=.001, controller_latency_s=0., seed=9,
            processes=(LatentProcessSpec(
                "stationary", DriftKind.CONSTANT, {}, amplitude=0.),),
        ))
        result = FullControlRLArm(seed=9, candidate_count=4, candidate_cycles=8).run_interval(
            device, cycles=32, interval=0)
        self.assertTrue(result.candidate_trajectories)
        self.assertTrue(all(row["candidate_alignment_valid"]
                            for row in result.candidate_trajectories))
        self.assertTrue(result.observation.physical_state_id)
        self.assertTrue(result.observation.disturbance_state_id)
        self.assertIsNotNone(result.mean_policy_detector_rate)
        self.assertIsNotNone(result.aggregate_exploration_detector_rate)


class FullRLValidationTests(unittest.TestCase):
    def test_controller_ladder_and_budget_sweep_pass(self):
        controller = run_controller_validation()
        budget = run_sample_budget_validation()
        self.assertTrue(controller.passed)
        self.assertTrue(budget.passed)
        self.assertEqual(budget.metadata["selected_validated_reduced_budget"], 2048)
        self.assertEqual(budget.metadata["paper_scale_cycles_per_candidate"], 100000)

    def test_controller_faults_fail_closed(self):
        cases = {
            "reversed_reward_sign": "analytic_convergence_both_sides",
            "cumulative_perturbations": "candidate_centring_and_no_cumulative_error",
            "transposed_mask": "static_sparse_gradient_alignment",
            "calibrated_start_regression": "calibrated_start_no_regression",
        }
        for fault, check_id in cases.items():
            with self.subTest(fault=fault):
                report = run_controller_validation(injected_faults=(fault,))
                check = next(item for item in report.checks if item.check_id == check_id)
                self.assertFalse(check.passed)
                self.assertFalse(report.passed)

    def test_underpowered_budget_cannot_be_promoted(self):
        report = run_sample_budget_validation(injected_faults=("underpowered_budget_accepted",))
        check = next(item for item in report.checks
                     if item.check_id == "candidate_budget_adequacy")
        self.assertFalse(check.passed)
        self.assertFalse(report.passed)

    def test_benchmark_preflight_passes_and_rejects_stale_source_state(self):
        passing = run_preflight()
        stale = run_preflight(injected_faults=("stale_manifest",))
        self.assertTrue(passing.passed)
        self.assertTrue(passing.metadata["long_acquisition_authorized"])
        self.assertFalse(stale.passed)
        self.assertFalse(stale.metadata["long_acquisition_authorized"])


if __name__ == "__main__":
    unittest.main()
