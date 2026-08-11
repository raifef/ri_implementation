"""Labelled architecture-wide fault suite used by Stage-7 threshold validation."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from hdfa_rl_suite.simulator import ScalableQECDevice, SimulatorConfig
from hdfa_rl_suite.stage0.schema import PolicySnapshot
from hdfa_rl_suite.stage1 import TelemetryProcessor
from hdfa_rl_suite.stage1.schema import RawMeasurementRecord
from hdfa_rl_suite.stage5.schema import PredictedCostDistribution, PredictiveControlPackage, ResidualAllocation, SolverStatus
from hdfa_rl_suite.stage7 import OperatingMode, SupervisoryController
from hdfa_rl_suite.stage7.schema import Authorization, SupervisorInput


@dataclass(frozen=True)
class FaultCaseResult:
    fault_id: str
    detected: bool
    safe_response: bool
    response: str


@dataclass(frozen=True)
class FaultSuiteReport:
    cases: tuple[FaultCaseResult, ...]

    @property
    def passed(self) -> bool:
        return all(case.detected and case.safe_response for case in self.cases)


class FaultInjectionRunner:
    def __init__(self, seed: int = 0) -> None:
        self.seed = seed

    def _telemetry_case(self, fault_id: str, mutate: Callable[[tuple[RawMeasurementRecord, ...], ScalableQECDevice], tuple[RawMeasurementRecord, ...]]) -> FaultCaseResult:
        device = ScalableQECDevice(SimulatorConfig(qubit_count=3, seed=self.seed))
        batch = device.acquire(16)
        records = mutate(batch.records, device)
        telemetry = TelemetryProcessor(device.circuit.detectors, device.detector_control_graph).process(
            records, (batch.policy_activation,), batch.context)
        return FaultCaseResult(fault_id, bool(telemetry.quality_flags), telemetry.hard_invalid,
                               ",".join(flag.code for flag in telemetry.quality_flags))

    def run(self) -> FaultSuiteReport:
        cases = [
            self._telemetry_case("time_reversal", lambda records, device: records[:1] + (
                RawMeasurementRecord(records[1].record_id, records[1].sequence, records[1].shot,
                    records[1].cycle, records[0].device_timestamp_s - 1., records[1].measurements,
                    records[1].circuit_hash, records[1].channel_ids),) + records[2:]),
            self._telemetry_case("sequence_dropout", lambda records, device: records[:3] + records[4:]),
            self._telemetry_case("context_mismatch", lambda records, device: (
                RawMeasurementRecord(records[0].record_id, records[0].sequence, records[0].shot,
                    records[0].cycle, records[0].device_timestamp_s, records[0].measurements,
                    "wrong-circuit", records[0].channel_ids),) + records[1:]),
        ]
        device = ScalableQECDevice(SimulatorConfig(qubit_count=3, seed=self.seed))
        supervisor = SupervisoryController(device.limits)
        unknown = supervisor.tick(SupervisorInput(1., (), broad_ood=True, unknown_model_probability=.9))
        cases.append(FaultCaseResult("unknown_model", unknown.mode is OperatingMode.UNKNOWN_EVENT,
                                     unknown.authorization is Authorization.ROLLBACK, unknown.reason))
        snapshot = PolicySnapshot(dict(device.confirmed_policy.controls), device.confirmed_policy.policy_hash, 0.)
        package = PredictiveControlPackage("v", SolverStatus.OPTIMAL, dict(snapshot.values), (dict(snapshot.values),), {},
            ResidualAllocation(tuple(snapshot.values), {key: .01 for key in snapshot.values}, tuple(snapshot.values), {}), (),
            PredictedCostDistribution(0.,0.,{}), snapshot, "issued", 0., 2., snapshot,
            controller_acknowledged_hash="different")
        supervisor.tick(SupervisorInput(2., (), forecast_valid=True, residual_small=True))
        mismatch = supervisor.authorize_control(package, 1.)
        cases.append(FaultCaseResult("policy_hash_mismatch", "hash mismatch" in mismatch.reason,
                                     mismatch.authorization is Authorization.ROLLBACK, mismatch.reason))
        rollback = supervisor.verify_rollback(.5, 0., .1, 3.)
        cases.append(FaultCaseResult("failed_rollback", rollback.mode is OperatingMode.UNKNOWN_EVENT,
                                     rollback.authorization is Authorization.ROLLBACK, rollback.reason))
        return FaultSuiteReport(tuple(cases))
