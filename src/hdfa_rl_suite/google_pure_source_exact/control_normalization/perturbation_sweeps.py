"""Deterministic simultaneous-gate Gaussian perturbation sweeps."""
from __future__ import annotations

from dataclasses import asdict
from hashlib import sha256
import json
from pathlib import Path
from typing import Iterable, Sequence

from .contracts import (
    SCHEMA_VERSION,
    ControlTypeSpec,
    SweepPoint,
    SweepProtocol,
    SweepResult,
    atomic_json,
    canonical_hash,
)
from .edr_measurement import DetectorEventEvaluator


def _point_seed(seed: int, control_type: str, sigma_native: float, purpose: str) -> int:
    material = f"{int(seed)}:{control_type}:{float(sigma_native):.17g}:{purpose}"
    return int.from_bytes(sha256(material.encode("utf-8")).digest()[:8], "little") & ((1 << 63) - 1)


def _checkpoint_payload(result: SweepResult, complete: bool) -> dict:
    payload = {
        "schema_version": "google-pure-source-exact-sweep-checkpoint.v1",
        "complete": bool(complete),
        "identity_hash": canonical_hash({
            "control_type": result.control_type,
            "protocol": asdict(result.protocol),
            "reference": asdict(result.reference),
            "plant_hash": result.plant_hash,
            "shard_index": result.shard_index,
            "shard_count": result.shard_count,
        }),
        "sweep_result": result.to_dict(),
    }
    payload["checkpoint_hash"] = canonical_hash(payload)
    return payload


def run_control_type_sweep(evaluator: DetectorEventEvaluator, spec: ControlTypeSpec,
                           protocol: SweepProtocol, *, sigmas_native: Sequence[float] | None = None,
                           checkpoint_path: Path | None = None, resume: bool = False,
                           shard_index: int = 0, shard_count: int = 1) -> SweepResult:
    """Measure one control type without ever consulting an analytic response scale."""
    if spec.control_type not in {item.control_type for item in evaluator.control_specs}:
        raise ValueError("control type is not registered with the evaluator")
    requested = tuple(float(value) for value in (sigmas_native or spec.sweep_sigmas_native))
    if not requested:
        raise ValueError("at least one sigma is required")
    if len(requested) != len(set(requested)):
        raise ValueError("duplicate sweep points are forbidden")
    if any(value not in spec.sweep_sigmas_native for value in requested):
        raise ValueError("requested sigma is outside the frozen sweep grid")
    existing: dict[float, SweepPoint] = {}
    if checkpoint_path is not None and checkpoint_path.exists():
        if not resume:
            raise FileExistsError(f"checkpoint exists; pass resume=True: {checkpoint_path}")
        checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
        prior = SweepResult.from_dict(checkpoint["sweep_result"])
        expected_identity = canonical_hash({
            "control_type": spec.control_type,
            "protocol": asdict(protocol),
            "reference": asdict(evaluator.reference),
            "plant_hash": evaluator.plant_hash,
            "shard_index": shard_index,
            "shard_count": shard_count,
        })
        if checkpoint.get("identity_hash") != expected_identity:
            raise RuntimeError("checkpoint identity differs from the frozen run contract")
        existing = {point.sigma_native: point for point in prior.points}
        if any(sigma not in requested for sigma in existing):
            raise RuntimeError("checkpoint contains a point outside this shard")
    elif resume and checkpoint_path is not None:
        raise FileNotFoundError(f"cannot resume missing checkpoint: {checkpoint_path}")

    for sigma in requested:
        if sigma in existing:
            continue
        perturbation_seed = _point_seed(
            protocol.perturbation_seed, spec.control_type, sigma, "perturbation")
        detector_seed = _point_seed(protocol.detector_seed, spec.control_type, sigma, "detector")
        measurement = evaluator.measure_joint(
            {spec.control_type: sigma},
            candidates=protocol.candidates_per_sigma,
            shots_per_candidate=protocol.shots_per_candidate,
            perturbation_seed=perturbation_seed,
            detector_seed=detector_seed,
        )
        point = SweepPoint(
            control_type=spec.control_type,
            sigma_native=sigma,
            detector_events=measurement.detector_events,
            detector_opportunities=measurement.detector_opportunities,
            candidates=measurement.candidates,
            shots_per_candidate=measurement.shots_per_candidate,
            qec_cycles=measurement.qec_cycles,
            perturbation_seed=perturbation_seed,
            detector_seed=detector_seed,
            candidate_detector_events=measurement.candidate_detector_events,
            candidate_detector_opportunities=measurement.candidate_detector_opportunities,
        )
        if sigma in existing:
            raise RuntimeError("duplicate result rejected")
        existing[sigma] = point
        partial = SweepResult(
            schema_version=SCHEMA_VERSION,
            control_type=spec.control_type,
            native_unit=spec.native_unit,
            reference=evaluator.reference,
            protocol=protocol,
            fit_interval_native=spec.fit_interval_native,
            points=tuple(existing[key] for key in sorted(existing)),
            simultaneous_gate_ids=spec.gate_ids,
            plant_hash=evaluator.plant_hash,
            shard_index=shard_index,
            shard_count=shard_count,
        )
        if checkpoint_path is not None:
            atomic_json(checkpoint_path, _checkpoint_payload(partial, complete=False))
    result = SweepResult(
        schema_version=SCHEMA_VERSION,
        control_type=spec.control_type,
        native_unit=spec.native_unit,
        reference=evaluator.reference,
        protocol=protocol,
        fit_interval_native=spec.fit_interval_native,
        points=tuple(existing[key] for key in sorted(existing)),
        simultaneous_gate_ids=spec.gate_ids,
        plant_hash=evaluator.plant_hash,
        shard_index=shard_index,
        shard_count=shard_count,
    )
    if checkpoint_path is not None:
        atomic_json(checkpoint_path, _checkpoint_payload(result, complete=True))
    return result


def shard_sigmas(sigmas: Sequence[float], shard_index: int, shard_count: int) -> tuple[float, ...]:
    if not 0 <= shard_index < shard_count:
        raise ValueError("invalid shard identity")
    result = tuple(float(value) for index, value in enumerate(sigmas) if index % shard_count == shard_index)
    if not result:
        raise ValueError("shard has no sweep points")
    return result


def merge_sweep_shards(shards: Iterable[SweepResult], spec: ControlTypeSpec) -> SweepResult:
    records = tuple(shards)
    if not records:
        raise ValueError("no shards supplied")
    first = records[0]
    points: dict[float, SweepPoint] = {}
    seen_shards: set[int] = set()
    for record in records:
        identity = (
            record.control_type,
            record.native_unit,
            asdict(record.reference),
            asdict(record.protocol),
            record.plant_hash,
            record.shard_count,
        )
        first_identity = (
            first.control_type,
            first.native_unit,
            asdict(first.reference),
            asdict(first.protocol),
            first.plant_hash,
            first.shard_count,
        )
        if identity != first_identity:
            raise RuntimeError("shard contract mismatch")
        if record.shard_index in seen_shards:
            raise RuntimeError("duplicate shard rejected")
        seen_shards.add(record.shard_index)
        for point in record.points:
            if point.sigma_native in points:
                raise RuntimeError("duplicate sweep point rejected")
            points[point.sigma_native] = point
    missing = set(spec.sweep_sigmas_native) - set(points)
    extra = set(points) - set(spec.sweep_sigmas_native)
    if missing or extra:
        raise RuntimeError(f"incomplete shard merge; missing={sorted(missing)}, extra={sorted(extra)}")
    return SweepResult(
        schema_version=SCHEMA_VERSION,
        control_type=first.control_type,
        native_unit=first.native_unit,
        reference=first.reference,
        protocol=first.protocol,
        fit_interval_native=first.fit_interval_native,
        points=tuple(points[key] for key in sorted(points)),
        simultaneous_gate_ids=first.simultaneous_gate_ids,
        plant_hash=first.plant_hash,
        shard_index=0,
        shard_count=1,
    )
