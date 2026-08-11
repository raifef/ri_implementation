import unittest

from hdfa_rl_suite.stage0.schema import DetectorDefinition
from hdfa_rl_suite.stage1 import CircuitContext, PolicyActivation, RawMeasurementRecord, TelemetryProcessor
from hdfa_rl_suite.stage2 import InferenceConfig, LatentVariable, PhysicalInferenceEngine, QuadraticLogitObservationModel, StateSchema
from hdfa_rl_suite.stage2.schema import DetectorResponse, InferenceValidity


def region_view(event_count: int, total: int = 96):
    definition = DetectorDefinition("d0", (0, 1), 0, ("g0",), "r0")
    records = tuple(RawMeasurementRecord(f"r{i}", i, 0, i, i * .001, (int(i < event_count), 0), "c", ("m0", "m1")) for i in range(total))
    context = CircuitContext("c", "ctx", "Z", 3, "memory", "active")
    policy = PolicyActivation("p", "hash", -1, -1, 0, {"drive": 0.0})
    return TelemetryProcessor((definition,), {"d0": ("drive",)}).process(records, (policy,), context).regional_views["r0"]


class InferenceTests(unittest.TestCase):
    def test_particle_inference_recovers_directional_operational_state(self):
        schema = StateSchema("r0", (LatentVariable("detuning", "effective detuning", "norm", -1, 1, intervention_control="drive", safe_intervention=.1),))
        model = QuadraticLogitObservationModel(schema, (DetectorResponse("d0", -2, {"detuning": 4}),))
        posterior = PhysicalInferenceEngine(schema, model, InferenceConfig(seed=4)).infer(region_view(60), {"drive": 0.0})
        self.assertGreater(posterior.mean["detuning"], 0.2)
        self.assertEqual(posterior.observability.rank, 1)

    def test_even_response_preserves_sign_ambiguity_and_requests_intervention(self):
        schema = StateSchema("r0", (LatentVariable("detuning", "effective detuning", "norm", -1, 1, intervention_control="drive", safe_intervention=.1),))
        model = QuadraticLogitObservationModel(schema, (DetectorResponse("d0", -3, state_quadratic={("detuning", "detuning"): 7}),))
        posterior = PhysicalInferenceEngine(schema, model, InferenceConfig(seed=8)).infer(region_view(46), {"drive": 0.0})
        values = [item.state["detuning"] for item in posterior.samples]
        self.assertTrue(any(value < 0 for value in values) and any(value > 0 for value in values))
        self.assertIsNotNone(posterior.intervention_request)
        self.assertIs(posterior.validity, InferenceValidity.LOW_OBSERVABILITY)

    def test_gaussian_path_and_predictive_mismatch_are_explicit(self):
        schema = StateSchema("r0", (LatentVariable("detuning", "effective detuning", "norm", -1, 1),))
        model = QuadraticLogitObservationModel(schema, (DetectorResponse("d0", -12),))
        posterior = PhysicalInferenceEngine(schema, model).infer(region_view(80), {}, method="gaussian")
        self.assertEqual(posterior.method, "gaussian")
        self.assertIs(posterior.validity, InferenceValidity.MODEL_MISMATCH)

    def test_state_conditioned_likelihood_kernels_are_numerically_equivalent(self):
        schema = StateSchema("r0", (LatentVariable("x", "effective error", "norm", -1, 1),))
        response = DetectorResponse("d0", -2.3, {"x": .4}, {"u": -.2},
            {("x", "x"): 1.7}, {("u", "u"): .8}, {("x", "u"): -1.1},
            context_intercepts={"ctx": .15})
        model = QuadraticLogitObservationModel(schema, (response,))
        state, controls = {"x": .37}, {"u": -.21}
        direct = model.probability("d0", state, controls, "ctx")
        prepared = model.prepare_state(state, "ctx")["d0"]
        indexed = dict(model.prepare_state_for_controls(state, ("u",), "ctx"))["d0"]
        self.assertAlmostEqual(model.probability_prepared(prepared, controls), direct, places=15)
        self.assertAlmostEqual(model.probability_prepared_values(indexed, (controls["u"],)),
                               direct, places=15)
