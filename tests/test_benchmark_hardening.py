import unittest

from hdfa_rl_suite.evaluation.benchmark import (
    PRIMARY_ARMS, BenchmarkConfig, BenchmarkPreflightError, BenchmarkRunner,
    default_benchmark_scenarios,
)
from hdfa_rl_suite.validation import (
    build_preflight_manifest, run_fault_matrix_validation,
    run_performance_validation, run_preflight,
)


class BenchmarkLaunchHardeningTests(unittest.TestCase):
    @staticmethod
    def config(**changes):
        values = dict(
            qubit_count=3, intervals=2, cycles_per_interval=32, seeds=(1, 2),
            candidate_cycles=2048, logical_shots_per_interval=16,
            cycle_period_s=1e-5,
            bootstrap_characterization_shots=64, bootstrap_validation_cycles=64,
            bootstrap_target_stddev=.07, bootstrap_qec_rate_limit=.10,
        )
        values.update(changes)
        return BenchmarkConfig(**values)

    def test_missing_manifest_refuses_before_any_arm_is_constructed(self):
        constructed = []

        def factory(seed):
            constructed.append(seed)
            raise AssertionError("arm construction must occur after preflight")

        factories = {name: factory for name in PRIMARY_ARMS}
        runner = BenchmarkRunner(
            self.config(), default_benchmark_scenarios(3)[:1], factories)
        with self.assertRaises(BenchmarkPreflightError):
            runner.run()
        self.assertEqual(constructed, [])

    def test_manifest_is_bound_to_exact_launch_configuration(self):
        report = run_preflight()
        runner = BenchmarkRunner(
            self.config(), default_benchmark_scenarios(3)[:1])
        manifest = build_preflight_manifest(report, runner.launch_configuration_hash)
        runner.preflight_manifest = manifest
        self.assertEqual(runner._require_preflight(), manifest.manifest_hash)
        changed = BenchmarkRunner(
            self.config(cycles_per_interval=64),
            default_benchmark_scenarios(3)[:1],
            preflight_manifest=manifest)
        with self.assertRaises(BenchmarkPreflightError):
            changed._require_preflight()

    def test_matched_clone_fingerprint_and_mutable_state_isolation(self):
        config = self.config(authoritative=False)
        scenario = default_benchmark_scenarios(3)[0]
        runner = BenchmarkRunner(config, (scenario,))
        prepared = runner._prepare_matched_state(scenario, 1)
        clone = prepared.device.clone()
        self.assertEqual(prepared.device.counterfactual_state_fingerprint(),
                         clone.counterfactual_state_fingerprint())
        self.assertFalse(prepared.device.shares_mutable_state_with(clone))
        clone.apply_policy({"drive:q0": .01}, policy_id="counterfactual-change")
        clone.await_policy_acknowledgement()
        self.assertNotEqual(prepared.device.controller_state_hash,
                            clone.controller_state_hash)


class ScientificFailureCoverageTests(unittest.TestCase):
    def test_all_fifteen_targeted_failures_are_detected(self):
        report = run_fault_matrix_validation()
        self.assertTrue(report.passed)
        self.assertEqual(report.metadata["fault_count"], 15)
        self.assertTrue(all(row["detected"] for row in report.trajectories))

    def test_stage2_through_stage6_fast_kernels_preserve_reference_values(self):
        report = run_performance_validation()
        self.assertTrue(report.passed)
        self.assertTrue(all(row["maximum_absolute_error"] <= row["tolerance"]
                            for row in report.trajectories))


if __name__ == "__main__":
    unittest.main()
