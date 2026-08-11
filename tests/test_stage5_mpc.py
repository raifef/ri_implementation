import unittest

from hdfa_rl_suite.stage0.schema import ControlBound, HardwareLimits, PolicySnapshot
from hdfa_rl_suite.stage2 import LatentVariable, QuadraticLogitObservationModel, StateSchema
from hdfa_rl_suite.stage2.schema import DetectorResponse
from hdfa_rl_suite.stage4.schema import ForecastBundle, ForecastCalibration, ForecastRisk, ForecastScenario, LatencyModel
from hdfa_rl_suite.stage5 import MPCConfig, PredictiveController
from hdfa_rl_suite.stage5.schema import SolverStatus


def forecast(scenarios, validity=1., invalid=()):
    return ForecastBundle("stage4.v1", "r0", 1., LatencyModel(0, 0, 0, 0), {0.: tuple(scenarios)},
        {0.: ForecastRisk({"d0": 0.}, 0., {"detuning": 0.}, {"drive": 0.}, 0.)}, validity,
        ForecastCalibration(0, None, None, None), invalid)


class MPCTests(unittest.TestCase):
    def setUp(self):
        self.limits = HardwareLimits({"drive": ControlBound(-1., 1., 1., "norm", 1.)})
        schema = StateSchema("r0", (LatentVariable("detuning", "detuning", "norm", -1, 1),))
        self.observation = QuadraticLogitObservationModel(schema, (DetectorResponse("d0", -3., state_quadratic={("detuning", "detuning"): 4.}, control_quadratic={("drive", "drive"): 4.}, state_control={("detuning", "drive"): -8.}),))
        self.current = PolicySnapshot({"drive": 0.}, "current", 1.)

    def test_feedforward_mpc_corrects_predicted_offset_within_constraints(self):
        scenario = ForecastScenario(0., 0., "ou", {"detuning": .4}, {"drive": .4}, {"d0": .05}, 1.)
        package = PredictiveController(self.limits, self.observation).solve(forecast((scenario,)), 0., self.current)
        self.assertIs(package.status, SolverStatus.OPTIMAL)
        self.assertAlmostEqual(package.action["drive"], .4, places=3)
        self.assertLessEqual(package.action["drive"], 1.)
        self.assertIn("drive", package.residual_allocation.bounds)

    def test_chance_constraint_returns_certified_fallback(self):
        # No bounded drive action can offset a state far enough to make the detector safe.
        scenario = ForecastScenario(0., 0., "ou", {"detuning": 2.}, {"drive": 1.}, {"d0": .9}, 1.)
        package = PredictiveController(self.limits, self.observation).solve(forecast((scenario,)), 0., self.current)
        self.assertIs(package.status, SolverStatus.INFEASIBLE)
        self.assertEqual(package.action, self.current.values)
        self.assertIsNotNone(package.infeasibility)

    def test_invalid_forecast_never_executes_partial_solution(self):
        scenario = ForecastScenario(0., 0., "ou", {"detuning": .2}, {"drive": .2}, {"d0": .1}, 1.)
        package = PredictiveController(self.limits, self.observation).solve(forecast((scenario,), invalid=("unknown model",)), 0., self.current)
        self.assertIs(package.status, SolverStatus.EXPIRED_FORECAST)
        self.assertEqual(package.policy_hash, self.current.policy_hash)

    def test_vectorized_objective_matches_scalar_reference_controller(self):
        scenarios = (
            ForecastScenario(0., 0., "ou", {"detuning": -.35}, {"drive": -.35}, {"d0": .05}, .2, .01, .02),
            ForecastScenario(0., 0., "rtn", {"detuning": .15}, {"drive": .15}, {"d0": .04}, .3, .02, .01),
            ForecastScenario(0., 0., "oscillator", {"detuning": .55}, {"drive": .55}, {"d0": .06}, .5, .03, .04),
        )
        vectorized = PredictiveController(self.limits, self.observation,
            MPCConfig(vectorized_objective=True)).solve(forecast(scenarios), 0., self.current)
        scalar = PredictiveController(self.limits, self.observation,
            MPCConfig(vectorized_objective=False)).solve(forecast(scenarios), 0., self.current)
        self.assertIs(vectorized.status, scalar.status)
        self.assertEqual(vectorized.action, scalar.action)
        self.assertAlmostEqual(vectorized.cost_distribution.expected_cost,
                               scalar.cost_distribution.expected_cost, places=14)
        self.assertEqual(vectorized.cost_distribution.detector_violation_probability,
                         scalar.cost_distribution.detector_violation_probability)
