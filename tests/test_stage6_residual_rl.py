import unittest

from hdfa_rl_suite.stage0.schema import PolicySnapshot
from hdfa_rl_suite.stage5.schema import PredictedCostDistribution, PredictiveControlPackage, ResidualAllocation, SolverStatus
from hdfa_rl_suite.stage6 import ExplorationBudget, GaussianResidualPolicy, ResidualRLController
from hdfa_rl_suite.stage6.schema import CandidateObservation


def package():
    snapshot = PolicySnapshot({"drive": 0., "phase": 0.}, "baseline", 1.)
    allocation = ResidualAllocation(("drive", "phase"), {"drive": .1, "phase": .05}, ("drive",), {"drive": "uncertain", "phase": "uncertain"})
    return PredictiveControlPackage("stage5.v1", SolverStatus.OPTIMAL, dict(snapshot.values), (dict(snapshot.values),), {}, allocation, (), PredictedCostDistribution(0., 0., {}), snapshot, "action", 1., 2., snapshot)


class ResidualRLTests(unittest.TestCase):
    def setUp(self):
        self.controller = ResidualRLController(GaussianResidualPolicy.full_control_baseline(("drive", "phase"), .03),
            {"d_drive": ("drive",), "d_phase": ("phase",)}, ExplorationBudget(.05, 1.))

    def test_antithetic_candidates_respect_stage5_residual_bounds(self):
        candidates = self.controller.propose(package(), candidate_count=4)
        self.assertEqual(len(candidates), 4)
        for plus, minus in zip(candidates[::2], candidates[1::2]):
            self.assertEqual(plus.pair_id, minus.pair_id)
            self.assertAlmostEqual(plus.residual["drive"], -minus.residual["drive"])
            self.assertLessEqual(abs(plus.residual["drive"]), .1)
            self.assertLessEqual(abs(plus.residual["phase"]), .05)

    def test_masked_antithetic_update_exports_response_evidence_and_regime_replay(self):
        control_package = package()
        candidates = self.controller.propose(control_package, candidate_count=4)
        observations = tuple(CandidateObservation(item.candidate_id,
            {"d_drive": .1 if item.sign > 0 else .3, "d_phase": 999.}, {"d_drive": 100, "d_phase": 100},
            regime_id="r1", context_id="c1", model_version="m1") for item in candidates)
        result = self.controller.update(control_package, observations, current_regime="r1", current_context="c1", current_model_version="m1")
        self.assertEqual(result.policy_version, 1)
        self.assertTrue(result.response_evidence)
        self.assertNotEqual(result.gradient["drive"], 0.)
        self.assertEqual(len(self.controller.replay_for("r1", "c1", "m1")), len(candidates))
        self.assertEqual(self.controller.replay_for("other", "c1", "m1"), ())

    def test_damage_budget_requests_fallback_and_stops_future_exploration(self):
        controller = ResidualRLController(GaussianResidualPolicy.full_control_baseline(("drive",), .04), {"d": ("drive",)}, ExplorationBudget(.05, .001))
        small_package = package()
        candidates = controller.propose(small_package, candidate_count=4)
        observations = tuple(CandidateObservation(item.candidate_id, {"d": .2}, {"d": 10}, logical_risk=.01) for item in candidates)
        result = controller.update(small_package, observations)
        self.assertTrue(result.fallback_requested)
        self.assertEqual(controller.propose(small_package), ())

    def test_graph_colouring_and_shot_allocation_preserve_locality(self):
        colours = self.controller.graph_colours()
        self.assertEqual(colours["drive"], colours["phase"])
        self.assertEqual(len(self.controller.orthogonal_directions(("drive", "phase"))), 2)
        allocation = self.controller.allocate_shots({"a": .1, "b": 1.})
        self.assertGreater(allocation["b"], allocation["a"])
