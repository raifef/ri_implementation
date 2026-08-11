import unittest

from hdfa_rl_suite.stage0.schema import DetectorDefinition
from hdfa_rl_suite.stage1 import CircuitContext, PolicyActivation, RawMeasurementRecord, TelemetryProcessor
from hdfa_rl_suite.stage2 import LatentVariable, QuadraticLogitObservationModel, StateSchema
from hdfa_rl_suite.stage2.schema import DetectorResponse
from hdfa_rl_suite.stage3 import DynamicsConfig, DynamicsModelKind, DynamicsModelSpec, JointDynamicsEngine, default_model_bank
from hdfa_rl_suite.stage3.sequential import segment_posterior_means


def view(batch: int, events: int = 35):
    detector = DetectorDefinition("d0", (0, 1), 0, ("g0",), "r0")
    records = tuple(RawMeasurementRecord(f"{batch}-{i}", i, batch, i, batch + i * .001, (int(i < events), 0), "c", ("m0", "m1")) for i in range(96))
    context = CircuitContext("c", "ctx", "Z", 3, "memory", "active")
    policy = PolicyActivation("p", "hash", -1, -1, 0, {"drive": 0.0})
    return TelemetryProcessor((detector,), {"d0": ("drive",)}).process(records, (policy,), context).regional_views["r0"]


class DynamicsTests(unittest.TestCase):
    def setUp(self):
        self.schema = StateSchema("r0", (LatentVariable("detuning", "effective detuning", "norm", -1, 1),))
        self.observation = QuadraticLogitObservationModel(self.schema, (DetectorResponse("d0", -2, {"detuning": 3}),))

    def test_joint_filter_returns_normalised_model_evidence_and_likelihood_particles(self):
        engine = JointDynamicsEngine(self.schema, self.observation, default_model_bank("detuning"), DynamicsConfig(seed=3, particle_count=128))
        output = engine.update(view(0), {"drive": 0.0}, 1.0)
        self.assertAlmostEqual(sum(output.model_evidence.model_probabilities.values()), 1.0, places=10)
        self.assertEqual(len(output.particles), 128)
        self.assertTrue(all(item.state for item in output.particles))
        engine.update(view(1), {"drive": 0.0}, 2.0)
        self.assertEqual(len(engine.smooth_history()), 2)

    def test_unknown_model_is_explicit_safety_path(self):
        bank = (DynamicsModelSpec("unknown", DynamicsModelKind.UNKNOWN, "detuning", 1.0, {"scale": .3}),)
        output = JointDynamicsEngine(self.schema, self.observation, bank, DynamicsConfig(seed=2, particle_count=64)).update(view(0), {}, 1.0)
        self.assertEqual(output.unknown_model_probability, 1.0)
        self.assertTrue(output.invalidity_reasons)

    def test_sequential_hdfa_baseline_remains_separate_ablation(self):
        segments = segment_posterior_means([0.] * 10 + [1.] * 10, minimum_length=4, jump_threshold=2.)
        self.assertGreaterEqual(len(segments), 2)
