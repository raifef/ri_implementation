"""Causal detector construction and policy-conditioned telemetry processing."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from collections import deque
from typing import Iterable, Mapping, Sequence

from hdfa_rl_suite.stage0.schema import DetectorDefinition, stable_hash

from .schema import (
    AlignedDetectorEvent, CircuitContext, ClockCalibration, CountFactor, PairCount, PolicyActivation,
    QualityFlag, QualitySeverity, RawMeasurementRecord, ReplayManifest, SCHEMA_VERSION,
    TelemetryBatch, TelemetryRegionView,
)


@dataclass(frozen=True)
class TelemetryConfig:
    window_sizes: tuple[int, ...] = (8, 32, 128)
    beta_prior_alpha: float = 0.5
    beta_prior_beta: float = 0.5
    maximum_clock_gap_s: float = 1.0
    maximum_retained_records: int = 4096
    frozen_bit_warning_records: int = 64


class TelemetryProcessor:
    """Transform append-only raw QEC records into exact, causally-labelled factors."""

    def __init__(self, detector_definitions: Sequence[DetectorDefinition],
                 detector_control_graph: Mapping[str, Sequence[str]], config: TelemetryConfig = TelemetryConfig(),
                 clock: ClockCalibration = ClockCalibration()) -> None:
        if not config.window_sizes or any(size <= 0 for size in config.window_sizes):
            raise ValueError("window sizes must be positive")
        self.definitions = tuple(detector_definitions)
        self.graph = {key: tuple(value) for key, value in detector_control_graph.items()}
        self.config, self.clock = config, clock

    @staticmethod
    def detector_parity(record: RawMeasurementRecord, definition: DetectorDefinition) -> int | None:
        """Reference parity implementation; missing or undefined operands have no exposure."""
        values = []
        for index in definition.measurement_indices:
            if index >= len(record.measurements) or record.measurements[index] is None:
                return None
            value = record.measurements[index]
            if value not in (0, 1):
                return None
            values.append(value)
        parity = definition.reference_parity
        for value in values:
            parity ^= value
        return parity

    def _policy_for(self, timestamp_s: float, timeline: Sequence[PolicyActivation]) -> tuple[PolicyActivation | None, bool]:
        if not timeline:
            return None, True
        ordered = sorted(timeline, key=lambda p: (p.nominal_activation_s, p.policy_id))
        ambiguous = any(low <= timestamp_s <= high for item in ordered for low, high in (item.ambiguity_interval(),))
        applicable = [item for item in ordered if item.nominal_activation_s <= timestamp_s]
        return (applicable[-1] if applicable else None), ambiguous

    def _integrity_flags(self, records: Sequence[RawMeasurementRecord], context: CircuitContext) -> list[QualityFlag]:
        flags: list[QualityFlag] = []
        previous_seq, previous_time = None, None
        seen: set[str] = set()
        for record in records:
            if record.record_id in seen:
                flags.append(QualityFlag("duplicate_record", QualitySeverity.HARD_INVALID, "duplicate record id", record.record_id))
            seen.add(record.record_id)
            if record.circuit_hash != context.circuit_hash:
                flags.append(QualityFlag("circuit_hash_mismatch", QualitySeverity.HARD_INVALID, "record circuit does not match context", record.record_id))
            if previous_seq is not None and record.sequence != previous_seq + 1:
                flags.append(QualityFlag("sequence_gap", QualitySeverity.HARD_INVALID, "dropped or reordered measurement record", record.record_id))
            if previous_time is not None and record.device_timestamp_s < previous_time:
                flags.append(QualityFlag("time_reversal", QualitySeverity.HARD_INVALID, "device timestamps are not acquisition ordered", record.record_id))
            if previous_time is not None and record.device_timestamp_s - previous_time > self.config.maximum_clock_gap_s:
                flags.append(QualityFlag("clock_gap", QualitySeverity.SOFT_INVALID, "excessive gap between records", record.record_id))
            previous_seq, previous_time = record.sequence, record.device_timestamp_s
        channel_counts = {len(record.measurements) for record in records}
        if len(channel_counts) > 1:
            flags.append(QualityFlag("missing_channel", QualitySeverity.HARD_INVALID, "measurement channel count changed inside context"))
        if len(records) >= self.config.frozen_bit_warning_records and records:
            width = min(len(record.measurements) for record in records)
            for index in range(width):
                values = {record.measurements[index] for record in records}
                if len(values) == 1 and None not in values:
                    flags.append(QualityFlag("frozen_bit", QualitySeverity.WARNING, f"measurement channel {index} is constant"))
        return flags

    def _count_factors(self, events: Sequence[AlignedDetectorEvent]) -> tuple[CountFactor, ...]:
        factors: list[CountFactor] = []
        for definition in self.definitions:
            detector_stream = [event for event in events if event.detector_id == definition.detector_id and not event.ambiguous_policy]
            keys = tuple(dict.fromkeys((event.policy_hash, event.context_id) for event in detector_stream)) or ((None, None),)
            for policy_hash, context_id in keys:
                stream = [event for event in detector_stream if event.policy_hash == policy_hash and event.context_id == context_id]
                for window in self.config.window_sizes:
                    subset = stream[-window:]
                    exposure = sum(event.exposure for event in subset)
                    count = sum(event.value or 0 for event in subset if event.exposure)
                    controls = dict(subset[-1].active_controls) if subset else {}
                    perturbation = dict(subset[-1].perturbation) if subset else {}
                    factors.append(CountFactor(window, definition.detector_id, count, exposure,
                                               self.config.beta_prior_alpha + count,
                                               self.config.beta_prior_beta + exposure - count,
                                               subset[0].timestamp_s if subset else None,
                                               subset[-1].timestamp_s if subset else None,
                                               policy_hash, context_id, controls, perturbation))
        return tuple(factors)

    def _pair_counts(self, events: Sequence[AlignedDetectorEvent]) -> tuple[PairCount, ...]:
        by_record: dict[str, dict[str, AlignedDetectorEvent]] = {}
        for event in events:
            by_record.setdefault(event.record_id, {})[event.detector_id] = event
        pairs: set[tuple[str, str]] = set()
        ids = [definition.detector_id for definition in self.definitions]
        for i, first in enumerate(ids):
            for second in ids[i + 1:]:
                if set(self.graph.get(first, ())) & set(self.graph.get(second, ())):
                    pairs.add((first, second))
        result: list[PairCount] = []
        for first, second in sorted(pairs):
            valid = []
            for record_events in by_record.values():
                left, right = record_events.get(first), record_events.get(second)
                if (left and right and left.exposure and right.exposure and not left.ambiguous_policy
                        and not right.ambiguous_policy and left.policy_hash == right.policy_hash):
                    valid.append((left, right))
            keys = tuple(dict.fromkeys((left.policy_hash, left.context_id) for left, _ in valid)) or ((None, None),)
            for policy_hash, context_id in keys:
                counts = [0, 0, 0, 0]
                for left, right in valid:
                    if left.policy_hash == policy_hash and left.context_id == context_id:
                        counts[2 * (left.value or 0) + (right.value or 0)] += 1
                result.append(PairCount(first, second, *counts, policy_hash, context_id))
        return tuple(result)

    def _regions(self, events: Sequence[AlignedDetectorEvent], factors: Sequence[CountFactor], pairs: Sequence[PairCount], context: CircuitContext) -> dict[str, TelemetryRegionView]:
        views: dict[str, TelemetryRegionView] = {}
        for definition in self.definitions:
            region = definition.region_id
            ids = tuple(d.detector_id for d in self.definitions if d.region_id == region)
            if region in views:
                continue
            controls = tuple(sorted({control for detector_id in ids for control in self.graph.get(detector_id, ())}))
            views[region] = TelemetryRegionView(region, ids, controls,
                tuple(event for event in events if event.detector_id in ids),
                tuple(factor for factor in factors if factor.detector_id in ids),
                tuple(pair for pair in pairs if pair.detector_a in ids and pair.detector_b in ids), context)
        return views

    def process(self, records: Iterable[RawMeasurementRecord], timeline: Sequence[PolicyActivation], context: CircuitContext) -> TelemetryBatch:
        """Process a causal batch without sorting records or assigning uncertain policy transitions."""
        raw = tuple(records)
        flags = self._integrity_flags(raw, context)
        events: list[AlignedDetectorEvent] = []
        tensor: dict[tuple[int, int, str], int | None] = {}
        exposure: dict[tuple[int, int, str], bool] = {}
        for record in raw:
            reference_time = self.clock.reference_time(record.device_timestamp_s)
            policy, ambiguous = self._policy_for(reference_time, timeline)
            ambiguous = ambiguous or self.clock.uncertainty_s > 0 and any(
                low - self.clock.uncertainty_s <= reference_time <= high + self.clock.uncertainty_s
                for item in timeline for low, high in (item.ambiguity_interval(),)
            )
            if ambiguous:
                flags.append(QualityFlag("policy_ambiguous", QualitySeverity.SOFT_INVALID, "record overlaps policy activation uncertainty", record.record_id))
            for definition in self.definitions:
                value = self.detector_parity(record, definition)
                key = (record.shot, record.cycle, definition.detector_id)
                tensor[key], exposure[key] = value, value is not None
                events.append(AlignedDetectorEvent(record.record_id, record.shot, record.cycle, definition.detector_id, value,
                    value is not None, reference_time, policy.policy_id if policy else None,
                    policy.policy_hash if policy else None, policy.candidate_id if policy else None,
                    context.context_id, definition.region_id, record.sequence, ambiguous,
                    dict(policy.controls) if policy else {}, dict(policy.perturbation) if policy else {},
                    policy.ambiguity_interval() if policy else None))
        factors = self._count_factors(events)
        pairs = self._pair_counts(events)
        definitions_hash = stable_hash([asdict(definition) for definition in self.definitions])
        manifest = ReplayManifest(tuple(record.record_id for record in raw), stable_hash([asdict(record) for record in raw]), definitions_hash,
                                  stable_hash([asdict(policy) for policy in timeline]), stable_hash(asdict(context)))
        return TelemetryBatch(SCHEMA_VERSION, tensor, exposure, tuple(events), factors, pairs, self._regions(events, factors, pairs, context), tuple(flags), manifest)

    @staticmethod
    def verify_replay(batch: TelemetryBatch) -> bool:
        """Check internal event/tensor consistency without inspecting future records or a backend."""
        return all(batch.event_tensor[(event.shot, event.cycle, event.detector_id)] == event.value
                   and batch.exposure_mask[(event.shot, event.cycle, event.detector_id)] == event.exposure
                   for event in batch.events) and bool(batch.replay_manifest.manifest_hash)


class StreamingTelemetryProcessor:
    """Bounded-memory online facade with deterministic offline equivalence.

    Recomputing over the fixed-size retention buffer gives O(1) cost with respect to total
    experiment duration while retaining exact events for every configured window.
    """

    def __init__(self, processor: TelemetryProcessor, context: CircuitContext) -> None:
        self.processor, self.context = processor, context
        self._records: deque[RawMeasurementRecord] = deque(maxlen=processor.config.maximum_retained_records)
        self._timeline: list[PolicyActivation] = []

    def append_policy(self, activation: PolicyActivation) -> None:
        if self._timeline and activation.nominal_activation_s < self._timeline[-1].nominal_activation_s:
            raise ValueError("policy timeline must remain acquisition ordered")
        self._timeline.append(activation)

    def append(self, record: RawMeasurementRecord) -> TelemetryBatch:
        self._records.append(record)
        return self.snapshot()

    def extend(self, records: Iterable[RawMeasurementRecord]) -> TelemetryBatch:
        for record in records:
            self._records.append(record)
        return self.snapshot()

    def snapshot(self) -> TelemetryBatch:
        return self.processor.process(tuple(self._records), tuple(self._timeline), self.context)
