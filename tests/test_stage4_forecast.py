import unittest

from hdfa_rl_suite.stage2 import LatentVariable, QuadraticLogitObservationModel, StateSchema
from hdfa_rl_suite.stage2.schema import DetectorResponse
from hdfa_rl_suite.stage3.schema import ChangeAlarm, DynamicsModelKind, DynamicsModelSpec, DynamicsParticle, DynamicsPosterior, ModelEvidence
from hdfa_rl_suite.stage4 import ForecastConfig, ForecastEngine, ForecastScorer, LatencyModel, ResponseMap


def posterior(unknown: float = 0.0):
    particles = (
        DynamicsParticle("rtn", {"detuning": -.6}, {}, "low", False, .5),
        DynamicsParticle("rtn", {"detuning": .6}, {}, "high", False, .5),
    )
    return DynamicsPosterior("stage3.v1", "r0", 1., particles, {"detuning": 0.}, ModelEvidence({"rtn": 1.}, unknown, -3.), ChangeAlarm(0., 1., "low", "r0"), (), {"rtn": {"amplitude": .6, "switch_rate": 0.}}, unknown, (), "test")


class ForecastTests(unittest.TestCase):
    def setUp(self):
        self.schema = StateSchema("r0", (LatentVariable("detuning", "detuning", "norm", -1, 1),))
        self.observation = QuadraticLogitObservationModel(self.schema, (DetectorResponse("d0", -2, {"detuning": 2}),))
        self.bank = (DynamicsModelSpec("rtn", DynamicsModelKind.RANDOM_TELEGRAPH, "detuning", 1., {"amplitude": .6, "switch_rate": 0.}),)
        self.response = ResponseMap({"drive": 0.}, {("drive", "detuning"): 1.})

    def test_multimodal_scenarios_are_not_collapsed_to_a_mean_action(self):
        bundle = ForecastEngine(self.observation, self.bank, self.response, ForecastConfig(seed=2)).forecast(posterior(), {}, (0.,), LatencyModel(0, 0, 0, 0))
        scenarios = bundle.scenarios(0.)
        self.assertEqual({round(item.optimum_controls["drive"], 2) for item in scenarios}, {-.6, .6})
        self.assertGreater(bundle.risk_by_horizon[0.].optimum_variance["drive"], 0.)

    def test_latency_is_included_in_activation_horizon(self):
        bundle = ForecastEngine(self.observation, self.bank, self.response).forecast(posterior(), {}, (.1,), LatencyModel(.01, .02, .03, .04))
        self.assertAlmostEqual(bundle.scenarios(.1)[0].activation_offset_s, .2)

    def test_unknown_dynamics_revoke_forecast_authority_and_scores_are_proper(self):
        bundle = ForecastEngine(self.observation, self.bank, self.response).forecast(posterior(.8), {}, (.1,), LatencyModel(0, 0, 0, 0))
        self.assertEqual(bundle.validity_horizon_s, 0.)
        self.assertTrue(bundle.invalidity_reasons)
        scorer = ForecastScorer()
        scorer.update_binary(.8, 1)
        self.assertAlmostEqual(scorer.summary().mean_brier_score, .04)
