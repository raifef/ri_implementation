import unittest

from hdfa_rl_suite.stage0 import BootstrapCalibrator, BootstrapConfig, SimulatedCalibrationBackend, demo_topology
from hdfa_rl_suite.stage0.schema import HealthStatus, NodeStatus


def make_result(seed: int = 7):
    topology, limits, circuit = demo_topology()
    return BootstrapCalibrator(topology, limits, circuit, SimulatedCalibrationBackend(topology, limits, seed=seed), BootstrapConfig(seed=seed)).run()


class BootstrapTests(unittest.TestCase):
    def test_bootstrap_reaches_qec_operable_policy_and_replays(self):
        result = make_result()
        self.assertIs(result.health.status, HealthStatus.PASSED)
        self.assertTrue(all(status is NodeStatus.PASSED for status in result.calibration_dag.values()))
        self.assertTrue(BootstrapCalibrator.verify_replay(result))
        self.assertNotEqual(result.baseline_policy.policy_hash, result.rollback_snapshot.policy_hash)
        self.assertEqual(set(result.sensitivity_scales), set(result.baseline_policy.values))


    def test_bootstrap_is_seed_deterministic(self):
        first, second = make_result(13), make_result(13)
        self.assertEqual(first.replay_hash, second.replay_hash)


    def test_failed_qec_gate_is_explicitly_unhealthy(self):
        topology, limits, circuit = demo_topology()
        backend = SimulatedCalibrationBackend(topology, limits, seed=2)
        result = BootstrapCalibrator(topology, limits, circuit, backend, BootstrapConfig(qec_rate_limit=0.001)).run()
        self.assertIs(result.health.status, HealthStatus.FAILED)
        self.assertTrue("qec" in result.health.invalid_reasons or "final_validation" in result.health.invalid_reasons)
