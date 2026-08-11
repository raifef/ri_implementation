"""Production V15 source-normalized/native control boundary.

The public sensitivity law fixes one normalized variance unit to one detector-
event-rate percentage point.  Public analogues do not share the proprietary
control registry, so each analogue is calibrated against the exact connected-
detector objective consumed by its controller.  The resulting map is applied
once, immediately before the plant, and is accompanied by enough lineage to
reject stale, inverse, reordered, or repeated mappings.
"""
from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


IMPLEMENTATION_VERSION = "google_pure_v15"
BOUNDARY_TRANSFORM_NAME = "u = u0 + s*x"
BOUNDARY_SCHEMA = "google-pure-v15-production-boundary.v1"
KAPPA_EDR_FRACTION = 0.01


def canonical_hash(value: Any) -> str:
    return sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                             allow_nan=False).encode("utf-8")).hexdigest()


def file_hash(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _root() -> Path:
    return Path(__file__).resolve().parents[3]


def _required_json(relative: str) -> tuple[dict[str, Any], Path]:
    path = _root() / relative
    if not path.is_file():
        raise RuntimeError(f"missing mandatory V15 normalization input: {relative}")
    return json.loads(path.read_text(encoding="utf-8")), path


def source_normalization_inputs() -> dict[str, Any]:
    """Return hashes of the frozen source definition and calibration evidence."""
    source, source_path = _required_json(
        "artifacts/google_pure_source_exact/control_normalization/source_contract.json")
    bundle, bundle_path = _required_json(
        "artifacts/google_pure_source_exact/control_normalization/calibration_bundle.json")
    degree, degree_path = _required_json(
        "artifacts/google_pure_v15/sensitivity/detector_degree_audit.json")
    if not bundle.get("artifact_complete") or not bundle.get("mathematical_contract_pass"):
        raise RuntimeError("V15 calibration bundle is incomplete or mathematically invalid")
    if not degree.get("pass"):
        raise RuntimeError("V15 detector-degree audit did not pass")
    if degree.get("objective") != "sum of connected detector rewards, then one mean over K candidates":
        raise RuntimeError("calibration/training detector aggregation contract changed")
    return {
        "sensitivity_definition_hash": str(source["source_contract_hash"]),
        "sensitivity_definition_file_sha256": file_hash(source_path),
        "calibration_bundle_hash": canonical_hash(bundle),
        "calibration_bundle_file_sha256": file_hash(bundle_path),
        "detector_degree_audit_hash": canonical_hash(degree),
        "detector_degree_audit_file_sha256": file_hash(degree_path),
        "source_normalization_version": BOUNDARY_SCHEMA,
    }


def sensitivity_map_hash_for_family(family: str) -> str:
    """Hash the frozen rule and its mandatory inputs, independently of plant size."""
    inputs = source_normalization_inputs()
    return canonical_hash({
        "schema": BOUNDARY_SCHEMA,
        "family": str(family),
        "rule": "s_i=sqrt(0.01/native_connected_detector_objective_curvature_i)",
        "sensitivity_definition_hash": inputs["sensitivity_definition_hash"],
        "calibration_bundle_hash": inputs["calibration_bundle_hash"],
        "detector_degree_audit_hash": inputs["detector_degree_audit_hash"],
    })


def boundary_transform_hash() -> str:
    return canonical_hash({
        "schema": BOUNDARY_SCHEMA,
        "transform": BOUNDARY_TRANSFORM_NAME,
        "implementation_sha256": file_hash(Path(__file__)),
    })


@dataclass(frozen=True)
class BoundaryApplication:
    native: np.ndarray
    token: Mapping[str, Any]

    def __post_init__(self) -> None:
        value = np.asarray(self.native, dtype=float)
        if not np.all(np.isfinite(value)):
            raise ValueError("native controls must be finite")
        object.__setattr__(self, "native", value.copy())


@dataclass(frozen=True)
class SourceNormalizationBoundary:
    """A frozen exact-objective map with a typed, single-use application."""

    family: str
    control_ids: tuple[str, ...]
    native_origin: np.ndarray
    native_scale: np.ndarray
    native_objective_curvature: np.ndarray
    sensitivity_map_hash: str
    source_inputs: Mapping[str, Any]

    def __post_init__(self) -> None:
        origin = np.asarray(self.native_origin, dtype=float)
        scale = np.asarray(self.native_scale, dtype=float)
        curvature = np.asarray(self.native_objective_curvature, dtype=float)
        expected = (len(self.control_ids),)
        if origin.shape != expected or scale.shape != expected or curvature.shape != expected:
            raise ValueError("V15 boundary vectors must align with the control registry")
        if len(set(self.control_ids)) != len(self.control_ids):
            raise ValueError("V15 control ids must be unique")
        if (not np.all(np.isfinite(origin)) or not np.all(np.isfinite(scale)) or
                not np.all(np.isfinite(curvature)) or np.any(scale <= 0) or
                np.any(curvature <= 0)):
            raise ValueError("V15 boundary origin, scale, and curvature must be finite and positive")
        conditioned = curvature * np.square(scale)
        if not np.allclose(conditioned, KAPPA_EDR_FRACTION, rtol=0, atol=2e-15):
            raise RuntimeError("V15 map does not condition the training objective to one EDR percentage point")
        object.__setattr__(self, "native_origin", origin.copy())
        object.__setattr__(self, "native_scale", scale.copy())
        object.__setattr__(self, "native_objective_curvature", curvature.copy())
        self.native_origin.setflags(write=False)
        self.native_scale.setflags(write=False)
        self.native_objective_curvature.setflags(write=False)

    @property
    def control_order_hash(self) -> str:
        return canonical_hash(list(self.control_ids))

    @property
    def expanded_scale_hash(self) -> str:
        return canonical_hash(self.native_scale.tolist())

    @property
    def boundary_transform_hash(self) -> str:
        return boundary_transform_hash()

    @classmethod
    def from_training_objective(
        cls,
        family: str,
        curvature: Sequence[float] | np.ndarray,
        *,
        control_ids: Sequence[str] | None = None,
        native_origin: Sequence[float] | np.ndarray | None = None,
    ) -> "SourceNormalizationBoundary":
        values = np.asarray(curvature, dtype=float)
        if values.ndim != 1 or values.size == 0 or np.any(values <= 0) or not np.all(np.isfinite(values)):
            raise ValueError("training-objective curvature must be a positive finite vector")
        ids = tuple(control_ids or (f"{family}:control:{index}" for index in range(values.size)))
        if len(ids) != values.size:
            raise ValueError("control registry and curvature length differ")
        origin = np.zeros(values.size) if native_origin is None else np.asarray(native_origin, dtype=float)
        inputs = source_normalization_inputs()
        # The expanded scale hash separately identifies the condition-sized vector,
        # which can vary with distance and parameters per gate.
        map_hash = sensitivity_map_hash_for_family(str(family))
        return cls(str(family), ids, origin, np.sqrt(KAPPA_EDR_FRACTION / values),
                   values, map_hash, inputs)

    def apply(self, normalized: np.ndarray, *, application_count: int = 0,
              control_order_hash: str | None = None,
              sensitivity_map_hash: str | None = None) -> BoundaryApplication:
        if application_count != 0:
            raise RuntimeError("normalization boundary may be applied exactly once")
        if control_order_hash not in {None, self.control_order_hash}:
            raise RuntimeError("control order changed at the V15 plant boundary")
        if sensitivity_map_hash not in {None, self.sensitivity_map_hash}:
            raise RuntimeError("stale or different V15 sensitivity map")
        value = np.asarray(normalized, dtype=float)
        if value.ndim not in {1, 2} or value.shape[-1] != len(self.control_ids):
            raise ValueError("normalized controls do not match the V15 control registry")
        if not np.all(np.isfinite(value)):
            raise ValueError("normalized controls must be finite")
        native = self.native_origin + self.native_scale * value
        token = {
            "implementation_version": IMPLEMENTATION_VERSION,
            "boundary_transform_name": BOUNDARY_TRANSFORM_NAME,
            "boundary_transform_hash": self.boundary_transform_hash,
            "boundary_apply_count": 1,
            "input_space": "SOURCE_NORMALIZED",
            "output_space": "NATIVE_CONTROL",
            "sensitivity_map_hash": self.sensitivity_map_hash,
            "expanded_scale_hash": self.expanded_scale_hash,
            "control_order_hash": self.control_order_hash,
            "normalized_action_hash": canonical_hash(value.tolist()),
            "native_action_hash": canonical_hash(native.tolist()),
        }
        return BoundaryApplication(native, token)

    def target_to_native(self, normalized_target: np.ndarray) -> np.ndarray:
        """Use the same frozen map for an evaluation-only optimum."""
        value = np.asarray(normalized_target, dtype=float)
        if value.shape != (len(self.control_ids),) or not np.all(np.isfinite(value)):
            raise ValueError("target does not match the V15 control registry")
        return self.native_origin + self.native_scale * value

    def trace(self, normalized: np.ndarray, *, indices: Sequence[int] | None = None) -> dict[str, Any]:
        value = np.asarray(normalized, dtype=float)
        if value.shape != (len(self.control_ids),):
            raise ValueError("boundary trace requires one normalized candidate")
        selected = tuple(indices or np.flatnonzero(np.abs(value) > 0)[:16].tolist() or (0,))
        rows = []
        for index in selected:
            scaled = float(self.native_scale[index] * value[index])
            native = float(self.native_origin[index] + scaled)
            rows.append({
                "control_index": int(index), "control_id": self.control_ids[index],
                "x_i": float(value[index]), "s_i": float(self.native_scale[index]),
                "u0_i": float(self.native_origin[index]), "s_i_x_i": scaled,
                "u_i": native, "identity_error": float(native - self.native_origin[index] - scaled),
                "native_objective_curvature": float(self.native_objective_curvature[index]),
                "normalized_objective_curvature": float(
                    self.native_objective_curvature[index] * self.native_scale[index] ** 2),
            })
        return {
            "schema_version": "google-pure-v15-boundary-trace.v1",
            "family": self.family,
            "rows": rows,
            "exact_identity_pass": all(row["identity_error"] == 0.0 for row in rows),
            "same_control_order": True,
            "same_scale_for_mean_candidates_and_evaluation": True,
            "second_scaling_inside_plant": False,
            "inverse_scaling": False,
            "stale_v12_map": False,
            "unscaled_legacy_path": False,
            **self.provenance_fields(),
        }

    def provenance_fields(self) -> dict[str, Any]:
        return {
            "implementation_version": IMPLEMENTATION_VERSION,
            "sensitivity_map_hash": self.sensitivity_map_hash,
            "sensitivity_definition_hash": self.source_inputs["sensitivity_definition_hash"],
            "calibration_bundle_hash": self.source_inputs["calibration_bundle_hash"],
            "detector_degree_audit_hash": self.source_inputs["detector_degree_audit_hash"],
            "boundary_transform_hash": self.boundary_transform_hash,
            "boundary_transform_name": BOUNDARY_TRANSFORM_NAME,
            "boundary_apply_count": 1,
            "control_order_hash": self.control_order_hash,
            "expanded_scale_hash": self.expanded_scale_hash,
        }


def require_v15_boundary_provenance(value: Mapping[str, Any]) -> None:
    required = (
        "implementation_version", "sensitivity_map_hash", "sensitivity_definition_hash",
        "calibration_bundle_hash", "detector_degree_audit_hash", "boundary_transform_hash",
        "boundary_transform_name", "boundary_apply_count", "control_order_hash",
        "expanded_scale_hash",
    )
    missing = [name for name in required if value.get(name) in {None, ""}]
    if missing:
        raise RuntimeError("missing mandatory V15 boundary provenance: " + ", ".join(missing))
    if value["implementation_version"] != IMPLEMENTATION_VERSION:
        raise RuntimeError("a V15 result was generated by a non-V15 driver")
    if value["boundary_transform_name"] != BOUNDARY_TRANSFORM_NAME:
        raise RuntimeError("V15 boundary transform name changed")
    if int(value["boundary_apply_count"]) != 1:
        raise RuntimeError("V15 boundary must be applied exactly once")
