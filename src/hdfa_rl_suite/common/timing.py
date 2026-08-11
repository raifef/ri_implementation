"""Symmetric online-control timing and critical-path accounting contracts."""
from __future__ import annotations

from dataclasses import dataclass
import math
import os
import platform
import sys
import time
from typing import Mapping, Sequence

from .records import deterministic_hash


TIMING_SCHEMA_VERSION = "online-critical-path-timing.v1"
HOST_CLOCK_DOMAIN = "host:perf_counter_ns"
DEVICE_CLOCK_DOMAIN = "simulated-device-time-s"


@dataclass(frozen=True)
class CriticalPathEvent:
    event_id: str
    sequence: int
    component: str
    stage: str
    clock_domain: str
    started: float
    ended: float
    duration_s: float
    on_critical_path: bool
    excluded_as_offline: bool = False
    overlaps_event_ids: tuple[str, ...] = ()


@dataclass(frozen=True)
class OnlineTimingBreakdown:
    schema_version: str
    clock_policy: str
    qec_acquisition_s: float
    diagnostic_downtime_s: float
    actuation_acknowledgement_s: float
    online_compute_critical_s: float
    simulator_kernel_host_s: float
    total_observed_host_wall_s: float
    offline_logical_evaluation_s: float
    offline_report_analysis_s: float
    stage_compute_s: Mapping[str, float]
    critical_path_events: tuple[CriticalPathEvent, ...]
    timing_complete: bool
    invalidity_reasons: tuple[str, ...] = ()

    @property
    def e2e_convergence_time_s(self) -> float:
        return (self.qec_acquisition_s + self.diagnostic_downtime_s
                + self.actuation_acknowledgement_s
                + self.online_compute_critical_s)

    @property
    def timing_hash(self) -> str:
        return deterministic_hash(self)

    def validate(self, *, expected_qec_s: float | None = None) -> tuple[str, ...]:
        reasons = list(self.invalidity_reasons)
        if self.schema_version != TIMING_SCHEMA_VERSION:
            reasons.append("unsupported timing schema")
        if self.clock_policy != "serial-hybrid-clock-critical-path.v1":
            reasons.append("changed or unsupported timing clock policy")
        numeric = (
            self.qec_acquisition_s, self.diagnostic_downtime_s,
            self.actuation_acknowledgement_s, self.online_compute_critical_s,
            self.simulator_kernel_host_s, self.total_observed_host_wall_s,
            self.offline_logical_evaluation_s, self.offline_report_analysis_s,
        )
        if any(value < 0 or not math.isfinite(value) for value in numeric):
            reasons.append("negative or non-finite timing duration")
        if abs(sum(self.stage_compute_s.values())-self.online_compute_critical_s) > 1e-8:
            reasons.append("online compute total differs from stage decomposition")
        if expected_qec_s is not None and abs(self.qec_acquisition_s-expected_qec_s) > 1e-8:
            reasons.append("QEC acquisition duration differs from cycles times period")
        sequences = [event.sequence for event in self.critical_path_events]
        if sequences != list(range(len(sequences))):
            reasons.append("critical-path event sequence is non-monotonic")
        for event in self.critical_path_events:
            expected_domain = (HOST_CLOCK_DOMAIN if event.component == "online_compute"
                               else DEVICE_CLOCK_DOMAIN)
            if event.clock_domain != expected_domain:
                reasons.append(f"changed clock domain:{event.event_id}")
            if (event.ended < event.started
                    or not all(math.isfinite(value) for value in (
                        event.started, event.ended, event.duration_s))
                    or abs(event.duration_s-(event.ended-event.started)) > 1e-8):
                reasons.append(f"invalid event duration:{event.event_id}")
            if event.overlaps_event_ids:
                reasons.append(f"unresolved critical-path overlap:{event.event_id}")
        component_totals = {
            name: sum(event.duration_s for event in self.critical_path_events
                      if event.component == name and event.on_critical_path
                      and not event.excluded_as_offline)
            for name in ("qec_acquisition", "diagnostic_downtime",
                         "actuation_acknowledgement", "online_compute")
        }
        declared = {
            "qec_acquisition": self.qec_acquisition_s,
            "diagnostic_downtime": self.diagnostic_downtime_s,
            "actuation_acknowledgement": self.actuation_acknowledgement_s,
            "online_compute": self.online_compute_critical_s,
        }
        for component, value in declared.items():
            if abs(component_totals[component]-value) > 1e-8:
                reasons.append(f"critical-path schedule differs from {component} total")
        if not self.timing_complete:
            reasons.append("timing record is incomplete")
        return tuple(dict.fromkeys(reasons))


@dataclass(frozen=True)
class TimingEnvironment:
    schema_version: str
    hardware_machine: str
    processor: str
    operating_system: str
    python_version: str
    package_version: str
    timer_name: str
    timer_resolution_s: float
    timer_monotonic: bool
    timer_adjustable: bool
    process_count: int
    thread_settings: Mapping[str, str]
    warmup_policy: str
    environment_hash: str = ""

    @classmethod
    def capture(cls, package_version: str, *, process_count: int = 1,
                warmup_policy: str = "one unreported controller warm-up per arm type") -> "TimingEnvironment":
        info = time.get_clock_info("perf_counter")
        thread_keys = (
            "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
            "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS")
        record = cls(
            "timing-environment.v1", platform.machine(), platform.processor(),
            platform.platform(), sys.version.split()[0], package_version,
            info.implementation, info.resolution, info.monotonic, info.adjustable,
            process_count, {key: os.environ.get(key, "unset") for key in thread_keys},
            warmup_policy, "")
        payload = dict(record.__dict__)
        return cls(**{**payload, "environment_hash": deterministic_hash(payload)})


class IntervalTimingRecorder:
    """Mutable local builder; final records are immutable and serializable."""

    def __init__(self) -> None:
        self.host_started_ns = time.perf_counter_ns()
        self._events: list[CriticalPathEvent] = []
        self._stage_compute: dict[str, float] = {}
        self._kernel_host_s = 0.0
        self._device_cursor_s = 0.0
        self._synthetic_host_cursor_s = self.host_started_ns/1e9

    def add_compute(self, stage: str, started_ns: int, ended_ns: int) -> None:
        duration = max(0.0, (ended_ns-started_ns)/1e9)
        self._stage_compute[stage] = self._stage_compute.get(stage, 0.0)+duration
        started = started_ns/1e9
        self._add("online_compute", stage, HOST_CLOCK_DOMAIN, started, started+duration)

    def add_compute_duration(self, stage: str, duration_s: float) -> None:
        """Add an already measured nested stage without measuring it twice."""
        duration = max(0.0, duration_s)
        self._stage_compute[stage] = self._stage_compute.get(stage, 0.0)+duration
        started = self._synthetic_host_cursor_s
        self._synthetic_host_cursor_s += duration
        self._add("online_compute", stage, HOST_CLOCK_DOMAIN, started, started+duration)

    def add_host_kernel(self, duration_s: float) -> None:
        self._kernel_host_s += max(0.0, duration_s)

    def add_physical(self, component: str, duration_s: float, *, stage: str) -> None:
        duration = max(0.0, duration_s)
        started = self._device_cursor_s
        self._device_cursor_s += duration
        self._add(component, stage, DEVICE_CLOCK_DOMAIN, started, self._device_cursor_s)

    def _add(self, component: str, stage: str, domain: str,
             started: float, ended: float) -> None:
        sequence = len(self._events)
        payload = (sequence, component, stage, domain, started, ended)
        self._events.append(CriticalPathEvent(
            deterministic_hash(payload)[:20], sequence, component, stage, domain,
            started, ended, ended-started, True, False, ()))

    def finalize(self, *, qec_acquisition_s: float,
                 diagnostic_downtime_s: float,
                 actuation_acknowledgement_s: float,
                 offline_logical_evaluation_s: float = 0.0,
                 offline_report_analysis_s: float = 0.0,
                 complete: bool = True,
                 invalidity_reasons: Sequence[str] = ()) -> OnlineTimingBreakdown:
        existing = {
            name: sum(event.duration_s for event in self._events
                      if event.component == name)
            for name in ("qec_acquisition", "diagnostic_downtime",
                         "actuation_acknowledgement")
        }
        declared_physical = {
            "qec_acquisition": qec_acquisition_s,
            "diagnostic_downtime": diagnostic_downtime_s,
            "actuation_acknowledgement": actuation_acknowledgement_s,
        }
        for component, total in declared_physical.items():
            remainder = total-existing[component]
            if remainder > 1e-15:
                self.add_physical(component, remainder, stage=component)
        host_wall = max(0.0, (time.perf_counter_ns()-self.host_started_ns)/1e9)
        online = sum(self._stage_compute.values())
        return OnlineTimingBreakdown(
            TIMING_SCHEMA_VERSION, "serial-hybrid-clock-critical-path.v1",
            qec_acquisition_s, diagnostic_downtime_s,
            actuation_acknowledgement_s, online,
            self._kernel_host_s, host_wall,
            offline_logical_evaluation_s, offline_report_analysis_s,
            dict(self._stage_compute), tuple(self._events), complete,
            tuple(invalidity_reasons),
        )
