import unittest

from hdfa_rl_suite.stage0.schema import DetectorDefinition
from hdfa_rl_suite.stage1 import CircuitContext, PolicyActivation, RawMeasurementRecord, TelemetryProcessor
from hdfa_rl_suite.stage1.schema import QualitySeverity


class TelemetryTests(unittest.TestCase):
    def setUp(self):
        self.definitions = (
            DetectorDefinition("d0", (0, 1), 0, ("g0",), "r0"),
            DetectorDefinition("d1", (1, 2), 1, ("g0",), "r0"),
        )
        self.processor = TelemetryProcessor(self.definitions, {"d0": ("p0",), "d1": ("p0",)})
        self.context = CircuitContext("c1", "ctx", "Z", 3, "mem", "active")
        self.policy = PolicyActivation("p", "hash", -1, -1, 0, {"p0": 0.0}, candidate_id="candidate")

    def test_exact_parity_exposure_counts_and_replay(self):
        records = tuple(RawMeasurementRecord(f"r{i}", i, 0, i, i * .01, values, "c1", ("m0", "m1", "m2"))
                        for i, values in enumerate(((0, 1, 1), (1, 1, 0), (0, None, 1))))
        batch = self.processor.process(records, (self.policy,), self.context)
        self.assertEqual(batch.event_tensor[(0, 0, "d0")], 1)
        self.assertEqual(batch.event_tensor[(0, 0, "d1")], 1)
        self.assertFalse(batch.exposure_mask[(0, 2, "d0")])
        factor = next(item for item in batch.count_factors if item.detector_id == "d0" and item.window_size == 8)
        self.assertEqual((factor.events, factor.exposures), (1, 2))
        self.assertTrue(TelemetryProcessor.verify_replay(batch))

    def test_ambiguous_policy_is_not_used_for_count_likelihood(self):
        record = RawMeasurementRecord("r0", 0, 0, 0, 1.0, (0, 0, 0), "c1", ("m0", "m1", "m2"))
        uncertain = PolicyActivation("p", "hash", 1.0, 1.0, .1, {"p0": 0.0})
        batch = self.processor.process((record,), (uncertain,), self.context)
        self.assertTrue(any(flag.code == "policy_ambiguous" for flag in batch.quality_flags))
        factor = next(item for item in batch.count_factors if item.detector_id == "d0" and item.window_size == 8)
        self.assertEqual(factor.exposures, 0)

    def test_sequence_gap_is_hard_invalid(self):
        records = (
            RawMeasurementRecord("r0", 0, 0, 0, 0., (0, 0, 0), "c1", ("m0", "m1", "m2")),
            RawMeasurementRecord("r2", 2, 0, 1, .01, (0, 0, 0), "c1", ("m0", "m1", "m2")),
        )
        batch = self.processor.process(records, (self.policy,), self.context)
        self.assertTrue(batch.hard_invalid)
        self.assertTrue(any(flag.severity is QualitySeverity.HARD_INVALID for flag in batch.quality_flags))
