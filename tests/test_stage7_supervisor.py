import unittest

from hdfa_rl_suite.stage0.schema import ControlBound, HardwareLimits, PolicySnapshot, stable_hash
from hdfa_rl_suite.stage5.schema import (PredictedCostDistribution, PredictiveControlPackage,
                                         ResidualAllocation, SolverStatus,
                                         bind_policy_lifecycle)
from hdfa_rl_suite.stage6.schema import ResidualCandidate, bind_candidate_lifecycle
from hdfa_rl_suite.stage7 import OperatingMode, SupervisoryController
from hdfa_rl_suite.stage7.schema import Authorization, DiagnosticOption, ModelLifecycle, StageHealth, SupervisorInput


def package(action=0.):
    baseline = {"drive": 0.}
    snapshot = PolicySnapshot(baseline, stable_hash(baseline), 1.)
    proposal = PredictiveControlPackage("stage5.v1", SolverStatus.OPTIMAL, {"drive": action}, ({"drive": action},), {},
        ResidualAllocation(("drive",), {"drive": .1}, ("drive",), {}), (), PredictedCostDistribution(0., 0., {}), snapshot,
        "action", 1., 2., snapshot)
    return bind_policy_lifecycle(
        proposal, policy_id="test-stage5", reference_policy_id="confirmed:test",
        reference_policy_hash=snapshot.policy_hash, created_from_state_id="state:test",
        controller_state_hash="controller:test")


def candidate_for(proposal, candidate):
    return bind_candidate_lifecycle(
        candidate,
        reference_policy_id=f"{proposal.policy_id}:candidate-reference:test",
        reference_policy_hash=proposal.policy_hash,
        created_from_state_id="state:candidate", controller_state_hash="controller:test")


class SupervisorTests(unittest.TestCase):
    def setUp(self):
        self.limits = HardwareLimits({"drive": ControlBound(-1., 1., 1., "norm", 1.)})

    def test_hard_invariant_forces_fail_safe_rollback(self):
        controller = SupervisoryController(self.limits)
        decision = controller.tick(SupervisorInput(1., (), hard_invariant_failed=True))
        self.assertIs(decision.mode, OperatingMode.FAIL_SAFE)
        self.assertIs(decision.authorization, Authorization.ROLLBACK)

    def test_decision_relevant_diagnostic_uses_explicit_value_and_downtime(self):
        controller = SupervisoryController(self.limits, (DiagnosticOption("slow", .5, 1., .1), DiagnosticOption("fast", .4, .1, .01)))
        decision = controller.tick(SupervisorInput(1., (), observation_nonidentifiable=True, diagnostic_decision_relevant=True))
        self.assertIs(decision.mode, OperatingMode.DIAGNOSTIC)
        self.assertEqual(decision.diagnostic.diagnostic_id, "fast")
        self.assertGreater(decision.diagnostic.downtime_s, 0.)

    def test_control_authorization_checks_mode_and_hard_bounds(self):
        controller = SupervisoryController(self.limits)
        controller.tick(SupervisorInput(1., (), forecast_valid=True, residual_small=True))
        self.assertIs(controller.authorize_control(package(.2), 1.1).authorization, Authorization.APPROVED)
        self.assertIs(controller.authorize_control(package(2.), 1.1).authorization, Authorization.ROLLBACK)

    def test_bad_rollback_becomes_unknown_event_and_lifecycle_requires_validation(self):
        controller = SupervisoryController(self.limits)
        decision = controller.verify_rollback(.4, .0, .1, 1.)
        self.assertIs(decision.mode, OperatingMode.UNKNOWN_EVENT)
        with self.assertRaises(ValueError):
            controller.set_model_lifecycle("model", ModelLifecycle.PROMOTED)
        controller.set_model_lifecycle("model", ModelLifecycle.PROMOTED, held_out_passed=True)

    def test_residual_candidate_requires_mode_projection_bounds_and_damage_budget(self):
        controller = SupervisoryController(self.limits)
        controller.tick(SupervisorInput(1., (), forecast_valid=True,
                                        residual_learning_safe=True, residual_small=False))
        proposal = package()
        candidate = candidate_for(proposal, ResidualCandidate(
            "c", "pair", 1, {"drive": .05}, {"drive": .05}, .01, 0))
        self.assertIs(controller.authorize_residual_candidate(
            proposal, candidate, 1.1).authorization, Authorization.APPROVED)
        escaped = candidate_for(proposal, ResidualCandidate(
            "bad", "pair", 1, {"other": .05}, {"drive": 0.}, .01, 0))
        self.assertIs(controller.authorize_residual_candidate(
            proposal, escaped, 1.1).authorization, Authorization.REJECTED)
