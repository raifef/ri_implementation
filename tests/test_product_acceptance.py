from dataclasses import replace
import pickle
import unittest

from hdfa_rl_suite.evaluation.benchmark import (
    PRIMARY_ARMS,
    BenchmarkConfig,
    BenchmarkRunner,
    default_benchmark_scenarios,
)
from hdfa_rl_suite.baselines import FixedCalibrationArm
from hdfa_rl_suite.logical import RotatedSurfaceCodeEvaluator, SurfaceCodeMemoryConfig
from hdfa_rl_suite.product import (
    HDFAProductController, ProductLoopConfig, QECOperabilityError, ReentryReason,
)
from hdfa_rl_suite.simulator import DriftKind, LatentProcessSpec, ScalableQECDevice, SimulatorConfig
from hdfa_rl_suite.stage0 import ScalableBootstrapConfig
from hdfa_rl_suite.stage7.schema import Authorization, OperatingMode
from hdfa_rl_suite.validation import build_preflight_manifest, run_preflight


def fast_product_config(residual=True):
    return ProductLoopConfig(
        enable_residual_rl=residual,
        residual_candidate_count=4,
        residual_candidate_cycles=4,
        bootstrap=ScalableBootstrapConfig(
            characterization_shots=64,
            validation_cycles=64,
            target_posterior_stddev=.07,
            qec_detector_rate_limit=.20,
        ),
    )


class IntegratedProductLoopTests(unittest.TestCase):
    def make_device(self, seed=2):
        stationary = (LatentProcessSpec(
            "stationary", DriftKind.CONSTANT, {f"drive:q{i}": 0.0 for i in range(3)},
            amplitude=0.0),)
        return ScalableQECDevice(SimulatorConfig(
            qubit_count=3, seed=seed, cycle_period_s=.001, processes=stationary))

    def test_genuine_stage0_to_7_loop_executes_residual_through_supervisor(self):
        product = HDFAProductController(
            self.make_device(), seed=2,
            config=replace(fast_product_config(), residual_activation_mode="always_on"))
        result = product.run_interval(256)
        self.assertEqual(result.bootstrap_reason, ReentryReason.COLD_START)
        self.assertEqual(result.control.supervisor.mode, OperatingMode.RESIDUAL_LEARNING)
        self.assertEqual(len(result.residual_candidates), 4)
        self.assertIsNotNone(result.residual_result)
        self.assertTrue(result.completed_without_lifecycle_violations)
        self.assertTrue(all(decision.authorization is Authorization.APPROVED
                            for decision in result.authorization_log))
        self.assertEqual(result.stage_path, (
            "stage0:cold_start", "stage1:telemetry", "stage2:physical_inference",
            "stage3:joint_dynamics_hdfa", "stage4:forecast", "stage5:mpc",
            "stage6:residual_rl", "stage7:authorization_lifecycle",
            "device:atomic_apply", "stage1:feedback"))

    def test_stage0_is_cached_until_an_explicit_reentry_cause(self):
        product = HDFAProductController(self.make_device(seed=0), seed=0, config=fast_product_config())
        first = product.run_interval(256)
        second = product.run_interval(256)
        self.assertEqual(first.bootstrap_count, 1)
        self.assertIsNone(second.bootstrap)
        self.assertEqual(second.bootstrap_count, 1)
        product.request_reentry(ReentryReason.MAJOR_HARDWARE_RECONFIGURATION)
        third = product.run_interval(256)
        self.assertEqual(third.bootstrap_reason, ReentryReason.MAJOR_HARDWARE_RECONFIGURATION)
        self.assertEqual(third.bootstrap_count, 2)

    def test_qec_operability_error_round_trips_across_process_boundaries(self):
        product = HDFAProductController(self.make_device(), seed=2, config=fast_product_config())
        result = product.run_interval(64)
        error = QECOperabilityError(result.bootstrap, ReentryReason.COLD_START)
        restored = pickle.loads(pickle.dumps(error))
        self.assertEqual(restored.reason, ReentryReason.COLD_START)
        self.assertEqual(restored.result.replay_hash, result.bootstrap.replay_hash)


class LogicalEvidenceTests(unittest.TestCase):
    def test_named_stim_pymatching_stack_is_reproducible(self):
        evaluator = RotatedSurfaceCodeEvaluator(SurfaceCodeMemoryConfig(
            distance=3, rounds=3, shots=128))
        first = evaluator.evaluate({"drive:q0": .4, "drive:q1": -.2}, seed=19)
        second = evaluator.evaluate({"drive:q0": .4, "drive:q1": -.2}, seed=19)
        self.assertEqual(first, second)
        self.assertEqual(first.circuit_task, "surface_code:rotated_memory_z")
        self.assertEqual(first.decoder, "PyMatching MWPM (fixed nominal DEM)")
        self.assertEqual(first.shots, 128)
        self.assertGreaterEqual(first.logical_error_per_round, 0.0)


class AuthoritativeBenchmarkTests(unittest.TestCase):
    _preflight_report = None

    @classmethod
    def authorized_runner(cls, config, scenarios, factories=None):
        runner = BenchmarkRunner(config, scenarios, factories)
        if cls._preflight_report is None:
            cls._preflight_report = run_preflight()
        runner.preflight_manifest = build_preflight_manifest(
            cls._preflight_report, runner.launch_configuration_hash)
        return runner

    @classmethod
    def run_primary(cls, seeds=(1, 2)):
        config = BenchmarkConfig(
            qubit_count=3, intervals=4, cycles_per_interval=128, seeds=seeds,
            candidate_cycles=2048, logical_shots_per_interval=32,
            cycle_period_s=1e-5,
            bootstrap_characterization_shots=64, bootstrap_validation_cycles=64,
            bootstrap_target_stddev=.07, bootstrap_qec_rate_limit=.10,
        )
        initial = BenchmarkRunner(config, default_benchmark_scenarios(3)[:1])
        primary = {name: factory for name, factory in initial.arm_factories.items()
                   if name in PRIMARY_ARMS}
        return cls.authorized_runner(
            config, default_benchmark_scenarios(3)[:1], primary).run()

    def test_actual_primary_benchmark_is_complete_matched_and_evaluable(self):
        report = self.run_primary()
        report.assert_authoritative()
        self.assertEqual({metric.arm for metric in report.metrics}, set(PRIMARY_ARMS))
        central = [metric for metric in report.metrics if metric.arm in {
            "full_control_detector_rl", "predictive_hdfa_residual_rl"}]
        self.assertTrue(all(metric.completion_status == "completed" for metric in central))
        self.assertTrue(all(metric.completion_status in {"completed", "censored"}
                            for metric in report.metrics))
        self.assertTrue(all(metric.lifecycle_violation_count == 0 for metric in report.metrics))
        self.assertTrue(all(gate.status in {"pass", "fail"} for gate in report.gates))
        self.assertEqual(len(report.gates), 7)
        self.assertFalse(next(gate for gate in report.gates
                              if gate.gate_id == "sample_efficiency_to_observed_90pct_recovery").primary)
        self.assertTrue(next(gate for gate in report.gates
                             if gate.gate_id == "compute_aware_rmst_net_convergence_gain").primary)
        self.assertTrue(report.matched_statistics)
        self.assertTrue(report.recovery_summaries)
        self.assertTrue(report.design_audit.stationary_stage0)
        self.assertTrue(report.design_audit.held_out_native_qec_baseline)
        self.assertTrue(report.design_audit.matched_baseline_observations)
        self.assertTrue(report.design_audit.matched_disturbance_realizations)
        self.assertTrue(report.design_audit.synchronized_disturbance_onsets)
        self.assertEqual(len(report.pre_disturbance_baselines), len(PRIMARY_ARMS) * 2)
        self.assertTrue(all(trajectory.logical_evidence is not None for trajectory in report.trajectories))
        for seed in (1, 2):
            realization_ids = {metric.disturbance_realization_id for metric in report.metrics
                               if metric.seed == seed}
            self.assertEqual(len(realization_ids), 1)
            bootstrap_hashes = {
                trajectory.bootstrap_evidence["baseline_policy"]["policy_hash"]
                for trajectory in report.trajectories
                if trajectory.seed == seed and trajectory.interval == 0
                and trajectory.bootstrap_evidence is not None
            }
            self.assertEqual(len(bootstrap_hashes), 1)
            baseline_hashes = {
                baseline.observation_hash for baseline in report.pre_disturbance_baselines
                if baseline.seed == seed
            }
            self.assertEqual(len(baseline_hashes), 1)
            onset_times = {
                metric.disturbance_epoch_s for metric in report.metrics if metric.seed == seed
            }
            self.assertEqual(len(onset_times), 1)
        for metric in report.metrics:
            for endpoint in metric.recovery_endpoints:
                if endpoint.status == "reached":
                    self.assertIsNotNone(endpoint.candidate_evaluations)
        self.assertTrue(report.provenance.configuration_hash)
        self.assertTrue(report.provenance.vcs_revision)
        self.assertEqual(report.provenance.logical_stack_versions["stim"], "1.16.0")

    def test_missing_confidence_replication_fails_authoritative_assertion(self):
        report = self.run_primary(seeds=(1,))
        self.assertFalse(report.authoritative)
        self.assertTrue(any(gate.status == "not_evaluable" for gate in report.gates))
        with self.assertRaises(AssertionError):
            report.assert_authoritative()

    def test_declared_censoring_runs_to_the_limit_and_preserves_trajectories(self):
        config = BenchmarkConfig(
            qubit_count=3, intervals=4, censoring_limit_intervals=2,
            cycles_per_interval=32, seeds=(1, 2), logical_shots_per_interval=16,
            bootstrap_characterization_shots=64, bootstrap_validation_cycles=64,
            bootstrap_target_stddev=.07, bootstrap_qec_rate_limit=.10,
            authoritative=False,
        )
        scenario = default_benchmark_scenarios(3)[0]
        runner = BenchmarkRunner(config, (scenario,), {"fixed": lambda seed: FixedCalibrationArm()})
        metric, trajectories, baseline = runner._run_arm(
            scenario, 1, "fixed", runner.arm_factories["fixed"])
        self.assertEqual(metric.completion_status, "censored")
        self.assertEqual(metric.censoring_reason, "declared interval censoring limit reached")
        self.assertEqual(len(trajectories), 2)
        self.assertIsNotNone(baseline)

    def test_declared_primary_censoring_is_an_authoritative_failed_estimand(self):
        config = BenchmarkConfig(
            qubit_count=3, intervals=4, censoring_limit_intervals=2,
            cycles_per_interval=64, seeds=(1, 2), candidate_cycles=2048,
            logical_shots_per_interval=16,
            cycle_period_s=1e-5,
            bootstrap_characterization_shots=64, bootstrap_validation_cycles=64,
            bootstrap_target_stddev=.07, bootstrap_qec_rate_limit=.10,
        )
        scenario = default_benchmark_scenarios(3)[:1]
        initial = BenchmarkRunner(config, scenario)
        factories = {name: factory for name, factory in initial.arm_factories.items()
                     if name in PRIMARY_ARMS}
        report = self.authorized_runner(config, scenario, factories).run()
        self.assertTrue(report.authoritative)
        self.assertFalse(report.accepted)
        self.assertTrue(all(gate.status == "fail" for gate in report.gates))
        self.assertTrue(any("did not complete" in reason
                            for reason in report.acceptance_failure_reasons))

    def test_lifecycle_violation_is_valid_negative_evidence_not_design_invalidity(self):
        config = BenchmarkConfig(
            qubit_count=3, intervals=4, cycles_per_interval=128, seeds=(1, 2),
            candidate_cycles=2048, logical_shots_per_interval=32,
            cycle_period_s=1e-5,
            bootstrap_characterization_shots=64, bootstrap_validation_cycles=64,
            bootstrap_target_stddev=.07, bootstrap_qec_rate_limit=.10,
        )
        scenario = default_benchmark_scenarios(3)[:1]
        initial = BenchmarkRunner(config, scenario)
        factories = {name: factory for name, factory in initial.arm_factories.items()
                     if name in PRIMARY_ARMS}
        staged_factory = factories["predictive_hdfa_residual_rl"]

        class InjectedViolationArm:
            name = "predictive_hdfa_residual_rl"

            def __init__(self, seed):
                self.delegate = staged_factory(seed)

            def prepare(self, device, bootstrap):
                self.delegate.prepare(device, bootstrap)

            def run_interval(self, device, cycles, interval):
                result = self.delegate.run_interval(device, cycles, interval)
                return replace(result, lifecycle_violations=(
                    *result.lifecycle_violations, "injected lifecycle violation"))

        factories["predictive_hdfa_residual_rl"] = InjectedViolationArm
        report = self.authorized_runner(config, scenario, factories).run()
        self.assertTrue(report.authoritative)
        self.assertFalse(report.accepted)
        self.assertFalse(report.invalidity_reasons)
        self.assertTrue(any("lifecycle violation" in reason
                            for reason in report.acceptance_failure_reasons))


if __name__ == "__main__":
    unittest.main()
