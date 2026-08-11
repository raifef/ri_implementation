import math
from dataclasses import replace
import unittest

from hdfa_rl_suite.common import RecordEnvelope, deterministic_hash
from hdfa_rl_suite.evaluation import BenchmarkConfig, BenchmarkRunner, default_benchmark_scenarios
from hdfa_rl_suite.pipeline import build_default_loop
from hdfa_rl_suite.simulator import DriftKind, LatentProcessSpec, ScalableQECDevice, SimulatorConfig
from hdfa_rl_suite.stage0 import ScalableBootstrapCalibrator, ScalableBootstrapConfig
from hdfa_rl_suite.stage0.schema import ControlBound, DetectorDefinition, HardwareLimits, PolicySnapshot
from hdfa_rl_suite.stage1 import CircuitContext, PolicyActivation, RawMeasurementRecord, StreamingTelemetryProcessor, TelemetryProcessor
from hdfa_rl_suite.stage2 import InferenceConfig, LatentVariable, PhysicalInferenceEngine, QuadraticLogitObservationModel, StateSchema
from hdfa_rl_suite.stage2.schema import DetectorResponse
from hdfa_rl_suite.stage3 import DynamicsConfig, JointDynamicsEngine, default_model_bank
from hdfa_rl_suite.stage3.schema import ChangeAlarm, DynamicsModelKind, DynamicsModelSpec, DynamicsParticle, DynamicsPosterior, ModelEvidence
from hdfa_rl_suite.stage4 import ForecastConfig, ForecastEngine, LatencyModel, ResponseMap
from hdfa_rl_suite.stage4.schema import ForecastBundle, ForecastCalibration, ForecastRisk, ForecastScenario
from hdfa_rl_suite.stage5 import PredictiveController, SharedResourceConstraint
from hdfa_rl_suite.stage6 import ExplorationBudget, FullControlDetectorRL
from hdfa_rl_suite.stage7 import OperatingMode, SupervisoryController
from hdfa_rl_suite.stage7.schema import Authorization, SupervisorInput


def two_detector_view(count=96):
    definitions = (
        DetectorDefinition("d0", (0,), 0, ("g",), "r"),
        DetectorDefinition("d1", (1,), 0, ("g",), "r"),
    )
    records = tuple(RawMeasurementRecord(f"r{i}", i, 0, i, i * .01,
        (int(i % 4 == 0), int(i % 4 == 0)), "c", ("m0", "m1")) for i in range(count))
    context = CircuitContext("c", "ctx", "Z", 3, "memory", "active")
    policy = PolicyActivation("p", "h", -1., -1., 0., {"u": 0.})
    processor = TelemetryProcessor(definitions, {"d0": ("u",), "d1": ("u",)})
    return processor, records, policy, context, processor.process(records, (policy,), context).regional_views["r"]


class SharedAndSimulatorTests(unittest.TestCase):
    def test_canonical_envelope_is_order_independent(self):
        self.assertEqual(deterministic_hash({"b": 2, "a": 1}), deterministic_hash({"a": 1, "b": 2}))
        envelope = RecordEnvelope.wrap("x.v1", "id", 1., "test", {"a": 1})
        self.assertEqual(envelope.payload_hash, deterministic_hash({"a": 1}))

    def test_simulator_scales_and_keeps_truth_out_of_observations(self):
        first = ScalableQECDevice(SimulatorConfig(qubit_count=100, seed=7))
        second = ScalableQECDevice(SimulatorConfig(qubit_count=100, seed=7))
        batch_a, batch_b = first.acquire(5), second.acquire(5)
        self.assertEqual(batch_a.records, batch_b.records)
        self.assertEqual(len(first.circuit.detectors), 100)
        self.assertFalse(hasattr(batch_a, "latent_state"))
        with self.assertRaises(PermissionError):
            first.oracle_evaluation_view("controller")
        self.assertEqual(len(first.oracle_evaluation_view("evaluation:test").latent_state()), len(first.limits.controls))

    def test_mixed_nested_processes_and_dropout_preserve_exposures(self):
        processes = (
            LatentProcessSpec("slow", DriftKind.SINUSOID, {"drive:q0": 1.}, amplitude=.2, period_s=1.),
            LatentProcessSpec("fast", DriftKind.RANDOM_TELEGRAPH, {"drive:q0": 1.}, amplitude=.1,
                              rate_hz=2., parent_process_id="slow"),
        )
        device = ScalableQECDevice(SimulatorConfig(qubit_count=3, seed=4, processes=processes,
                                                    measurement_dropout_probability=.5))
        batch = device.acquire(64)
        self.assertLess(batch.detector_exposures, 64 * 3)
        self.assertGreater(device.oracle_evaluation_view("evaluation:nested").process_state()["slow"], -.21)

    def test_disarmed_calibration_does_not_consume_the_relative_disturbance_tape(self):
        process = (LatentProcessSpec(
            "ou", DriftKind.ORNSTEIN_UHLENBECK, {"drive:q0": 1.0},
            diffusion=.2, ou_kappa=.1),)
        first = ScalableQECDevice(SimulatorConfig(
            qubit_count=3, seed=17, cycle_period_s=.01,
            disturbance_resolution_s=.01, disturbances_enabled_at_start=False,
            processes=process))
        second = ScalableQECDevice(SimulatorConfig(
            qubit_count=3, seed=17, cycle_period_s=.01,
            disturbance_resolution_s=.01, disturbances_enabled_at_start=False,
            processes=process))
        first.acquire(7)
        second.acquire(19)
        self.assertFalse(first.disturbances_armed)
        self.assertEqual(first.oracle_evaluation_view("evaluation:pre-onset").process_state()["ou"], 0.0)
        first.arm_disturbances()
        second.arm_disturbances()
        first.acquire(20)
        second.acquire(20)
        self.assertAlmostEqual(first.disturbance_elapsed_s, second.disturbance_elapsed_s)
        self.assertAlmostEqual(
            first.oracle_evaluation_view("evaluation:relative-tape").process_state()["ou"],
            second.oracle_evaluation_view("evaluation:relative-tape").process_state()["ou"])
        self.assertEqual(first.disturbance_realization_id, second.disturbance_realization_id)

    def test_stationary_vectorized_acquisition_is_bit_exact_to_scalar_reference(self):
        config = SimulatorConfig(
            qubit_count=9, code_distance=3, seed=731,
            disturbances_enabled_at_start=False,
            correlation_probability=.07,
            measurement_dropout_probability=.11,
        )
        vectorized = ScalableQECDevice(config)
        scalar = ScalableQECDevice(replace(
            config, stationary_vectorized_acquisition=False))
        patch = {
            control: (.08 if index % 2 == 0 else -.06)
            for index, control in enumerate(vectorized.limits.controls)
        }
        vectorized.apply_policy(patch, policy_id="pending")
        scalar.apply_policy(patch, policy_id="pending")
        vector_batch = vectorized.acquire(23, shot=4)
        scalar_batch = scalar.acquire(23, shot=4)
        self.assertEqual(vector_batch, scalar_batch)
        self.assertEqual(vectorized.now_s, scalar.now_s)
        self.assertEqual(vectorized.confirmed_policy, scalar.confirmed_policy)
        vectorized.arm_disturbances()
        scalar.arm_disturbances()
        self.assertEqual(vectorized.acquire(17), scalar.acquire(17))

    def test_aggregate_only_acquisition_preserves_counts_and_sequence(self):
        config = SimulatorConfig(
            qubit_count=7, seed=37, disturbances_enabled_at_start=False)
        full = ScalableQECDevice(config)
        summary = ScalableQECDevice(config)
        retained = full.acquire(64)
        aggregate = summary.acquire(64, retain_records=False)
        self.assertFalse(aggregate.records)
        self.assertEqual(retained.detector_events, aggregate.detector_events)
        self.assertEqual(retained.detector_exposures, aggregate.detector_exposures)
        self.assertEqual(retained.logical_failures, aggregate.logical_failures)
        self.assertEqual(retained.detector_counts, aggregate.detector_counts)
        self.assertEqual(full.acquire(1).records[0].record_id,
                         summary.acquire(1).records[0].record_id)

    def test_dynamic_aggregate_fast_path_is_bit_exact_to_scalar_reference(self):
        processes = (
            LatentProcessSpec("ou", DriftKind.ORNSTEIN_UHLENBECK,
                              {"drive:q0": 1.0, "drive:q1": .4}, diffusion=.08),
            LatentProcessSpec("sin", DriftKind.SINUSOID,
                              {"drive:q2": .7}, amplitude=.2, period_s=.11),
        )
        config = SimulatorConfig(
            qubit_count=3, seed=73, cycle_period_s=.001,
            controller_latency_s=.002, disturbance_resolution_s=.003,
            processes=processes, dynamic_vectorized_acquisition=True)
        vectorized = ScalableQECDevice(config)
        scalar = ScalableQECDevice(replace(config, dynamic_vectorized_acquisition=False))
        for device in (vectorized, scalar):
            device.apply_policy({"drive:q0": .01}, policy_id="pending-dynamic")
        vector_batch = vectorized.acquire(257, retain_records=False)
        scalar_batch = scalar.acquire(257, retain_records=False)
        self.assertEqual(vector_batch, scalar_batch)
        self.assertEqual(vectorized.counterfactual_state_fingerprint(),
                         scalar.counterfactual_state_fingerprint())


class RichStageTests(unittest.TestCase):
    def test_scalable_bootstrap_joint_blocks_and_conflict_free_batches(self):
        device = ScalableQECDevice(SimulatorConfig(qubit_count=4, seed=2))
        result = ScalableBootstrapCalibrator(device).run()
        self.assertEqual(result.health.status.value, "passed")
        self.assertEqual(len(result.parameter_registry), 7)
        for batch in result.resource_batches:
            resources = [set(result.calibration_nodes[node].resources) for node in batch]
            self.assertTrue(all(resources[i].isdisjoint(resources[j]) for i in range(len(resources)) for j in range(i+1, len(resources))))

    def test_scalable_bootstrap_controls_false_rejection_across_regions(self):
        config = ScalableBootstrapConfig(
            characterization_shots=384, validation_cycles=512,
            target_posterior_stddev=.035, qec_detector_rate_limit=.10,
            block_predictive_familywise_alpha=1e-4)
        for seed in (103, 104):
            device = ScalableQECDevice(SimulatorConfig(
                qubit_count=5, seed=seed, disturbances_enabled_at_start=False))
            result = ScalableBootstrapCalibrator(device, config).run()
            self.assertEqual(result.health.status.value, "passed")
            for node, estimate in result.calibration_estimates.items():
                if node.startswith("block:"):
                    self.assertGreaterEqual(
                        estimate.model_scores["joint_block_familywise_tail_probability"],
                        estimate.model_scores["block_alpha_share"])

    def test_graph_coloured_sensitivity_preserves_per_control_exposure_and_jacobian(self):
        device = ScalableQECDevice(SimulatorConfig(
            qubit_count=17, seed=123, disturbances_enabled_at_start=False))
        result = ScalableBootstrapCalibrator(device, ScalableBootstrapConfig()).run()
        estimate = result.calibration_estimates["sensitivity"]
        diagnostics = estimate.diagnostics
        cycles_per_sign = diagnostics["per_control_cycles_per_sign"]
        self.assertTrue(diagnostics["interference_passed"])
        self.assertLess(diagnostics["batch_count"], len(device.limits.controls))
        self.assertEqual(diagnostics["qec_cycles"],
                         2 * diagnostics["batch_count"] * cycles_per_sign)
        self.assertEqual(set(result.sensitivity_scales), set(device.limits.controls))
        self.assertTrue(all(record.local_jacobian
                            for record in result.parameter_registry.values()))
        for batch in diagnostics["sensitivity_batches"]:
            supports = [
                set(result.parameter_registry[control].affected_detectors)
                for control in batch["batch"]
            ]
            self.assertTrue(all(
                supports[left].isdisjoint(supports[right])
                for left in range(len(supports))
                for right in range(left + 1, len(supports))))

    def test_streaming_and_offline_telemetry_are_identical(self):
        processor, records, policy, context, _ = two_detector_view(40)
        stream = StreamingTelemetryProcessor(processor, context)
        stream.append_policy(policy)
        online = stream.extend(records)
        offline = processor.process(records, (policy,), context)
        self.assertEqual(online.to_dict(), offline.to_dict())

    def test_sparse_correlation_likelihood_adds_only_a_dependence_correction(self):
        _, _, _, _, view = two_detector_view()
        schema = StateSchema("r", (LatentVariable("x", "error", "norm", -1, 1),))
        responses = (DetectorResponse("d0", -1.1), DetectorResponse("d1", -1.1))
        independent = QuadraticLogitObservationModel(schema, responses)
        correlated = QuadraticLogitObservationModel(schema, responses, {("d0", "d1"): .8})
        self.assertEqual(independent.pair_log_likelihood(view, {"x": 0.}, {}), 0.)
        self.assertGreater(correlated.pair_log_likelihood(view, {"x": 0.}, {}), 0.)

    def test_observability_reports_the_actual_null_variable(self):
        _, _, _, _, view = two_detector_view()
        schema = StateSchema("r", (LatentVariable("x", "unseen", "n", -1, 1),
                                    LatentVariable("y", "seen", "n", -1, 1)))
        model = QuadraticLogitObservationModel(schema, (
            DetectorResponse("d0", -2., {"y": 2.}), DetectorResponse("d1", -2., {"y": 1.}),))
        posterior = PhysicalInferenceEngine(schema, model, InferenceConfig(seed=1, particle_count=128)).infer(view, {})
        self.assertIn("x", posterior.observability.unresolved_variable_ids)
        self.assertNotIn("y", posterior.observability.unresolved_variable_ids)

    def test_composite_dynamics_and_noncausal_smoothing_are_real_paths(self):
        _, _, _, _, view = two_detector_view()
        schema = StateSchema("r", (LatentVariable("x", "error", "n", -1, 1),))
        model = QuadraticLogitObservationModel(schema, (DetectorResponse("d0", -2., {"x": 2.}),
                                                        DetectorResponse("d1", -2., {"x": 1.})))
        bank = default_model_bank("x")
        self.assertTrue(any(item.kind is DynamicsModelKind.ADDITIVE_COMPOSITE for item in bank))
        engine = JointDynamicsEngine(schema, model, bank, DynamicsConfig(seed=5, particle_count=160, fixed_lag=3))
        first = engine.update(view, {}, 1.)
        engine.update(view, {}, 2.)
        smoothed = engine.smooth_history()
        self.assertTrue(any(item.component_state for item in first.particles if item.model_id == "oscillator-plus-ou"))
        self.assertTrue(all(item.offline_divergence is not None for item in smoothed))

    def test_forecast_reduction_keeps_control_risk_tails(self):
        particles = tuple(DynamicsParticle("ou", {"x": -1 + 2*i/19}, {}, "r", False, 1/20) for i in range(20))
        posterior = DynamicsPosterior("v", "r", 1., particles, {"x": 0.}, ModelEvidence({"ou": 1.}, 0., 0.),
            ChangeAlarm(0., 1., "low", "r"), (), {"ou": {"kappa": 0., "sigma": 0.}}, 0., (), "test")
        schema = StateSchema("r", (LatentVariable("x", "error", "n", -2, 2),))
        observation = QuadraticLogitObservationModel(schema, (DetectorResponse("d", -2., {"x": 1.}),))
        bank = (DynamicsModelSpec("ou", DynamicsModelKind.ORNSTEIN_UHLENBECK, "x", 1., {"kappa": 0., "sigma": 0.}),)
        bundle = ForecastEngine(observation, bank, ResponseMap({"u": 0.}, {("u", "x"): -1.}),
                                ForecastConfig(seed=2, maximum_scenarios=4)).forecast(posterior, {}, (0.,), LatencyModel(0,0,0,0))
        states = [item.state["x"] for item in bundle.scenarios(0.)]
        self.assertLessEqual(len(states), 4)
        self.assertIn(-1., states)
        self.assertIn(1., states)

    def test_multistep_mpc_enforces_shared_resource_constraint(self):
        scenarios = {
            0.: (ForecastScenario(0., 0., "m", {"x": .3}, {"u": .3, "v": .3}, {"d": .02}, 1.),),
            1.: (ForecastScenario(1., 1., "m", {"x": .5}, {"u": .5, "v": .5}, {"d": .02}, 1.),),
        }
        risk = {h: ForecastRisk({"d": 0.}, 0., {"x": 0.}, {"u": 0., "v": 0.}, 0.) for h in scenarios}
        forecast = ForecastBundle("v", "r", 0., LatencyModel(0,0,0,0), scenarios, risk, 2., ForecastCalibration(0,None,None,None), ())
        limits = HardwareLimits({"u": ControlBound(-1,1,1,"n",1), "v": ControlBound(-1,1,1,"n",1)})
        schema = StateSchema("r", (LatentVariable("x", "error", "n", -1,1),))
        observation = QuadraticLogitObservationModel(schema, (DetectorResponse("d", -4.),))
        controller = PredictiveController(limits, observation, shared_constraints=(SharedResourceConstraint("sum", {"u":1.,"v":1.}, .4),))
        package = controller.solve_trajectory(forecast, (0.,1.), PolicySnapshot({"u":0.,"v":0.}, "p", 0.))
        self.assertEqual(len(package.trajectory), 2)
        self.assertTrue(all(step["u"] + step["v"] <= .4 + 1e-12 for step in package.trajectory))

    def test_full_control_baseline_reproduces_forty_candidate_batch(self):
        limits = HardwareLimits({"u": ControlBound(-1,1,.2,"n",.2)})
        baseline = FullControlDetectorRL(limits, {"d": ("u",)}, PolicySnapshot({"u":0.}, "p", 0.),
                                         ExplorationBudget(.2, 10.), candidate_count=40)
        candidates = baseline.propose()
        self.assertEqual(len(candidates), 40)
        self.assertEqual({item.sign for item in candidates}, {-1, 1})

    def test_supervisor_hysteresis_and_acknowledgement_mismatch(self):
        limits = HardwareLimits({"u": ControlBound(-1,1,1,"n",1)})
        supervisor = SupervisoryController(limits)
        supervisor.tick(SupervisorInput(1., (), local_change_probability=.99))
        held = supervisor.tick(SupervisorInput(2., (), local_change_probability=.5, forecast_valid=True))
        self.assertIs(held.mode, OperatingMode.LOCAL_RECOVERY)
        self.assertIs(held.authorization, Authorization.DELAYED)


class EndToEndTests(unittest.TestCase):
    def test_regional_forecasts_do_not_replicate_the_global_policy(self):
        device = ScalableQECDevice(SimulatorConfig(qubit_count=17, code_distance=3, seed=9))
        loop = build_default_loop(device, seed=9)
        self.assertGreater(len(device.confirmed_policy.controls), 3)
        for stack in loop.regions.values():
            self.assertEqual(set(stack.forecast.response_map.reference_controls), set(stack.controls))
            self.assertLessEqual(len(stack.forecast.response_map.reference_controls), 3)

    def test_default_loop_runs_all_regions_and_is_replay_deterministic(self):
        def run():
            device = ScalableQECDevice(SimulatorConfig(qubit_count=4, seed=9))
            output = build_default_loop(device, seed=9, horizons_s=(0., .05)).step(device.acquire(64))
            return output
        first, second = run(), run()
        self.assertEqual(first.replay_hash, second.replay_hash)
        self.assertEqual(len(first.regions), 4)
        self.assertIsNotNone(first.proposed_control)

    def test_benchmark_registers_all_required_comparison_arms(self):
        config = BenchmarkConfig(qubit_count=3, intervals=1, cycles_per_interval=8, seeds=(1,), candidate_cycles=4)
        runner = BenchmarkRunner(config, default_benchmark_scenarios(3)[:1])
        self.assertEqual(set(runner.arm_factories), {
            "fixed", "periodic_recalibration", "greedy_calibration", "state_only", "sequential_hdfa",
            "joint_hdfa_reactive", "full_control_detector_rl", "predictive_hdfa_no_residual",
            "predictive_hdfa_residual_rl", "oracle"})


if __name__ == "__main__":
    unittest.main()
