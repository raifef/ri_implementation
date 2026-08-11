"""Architecture-wide immutable records and deterministic serialization."""
from __future__ import annotations

from dataclasses import asdict, dataclass, is_dataclass
from enum import Enum
from hashlib import sha256
import json
import math
from typing import Any, Mapping, Sequence


def _jsonable(value: Any) -> Any:
    if is_dataclass(value):
        return _jsonable(asdict(value))
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in sorted(value.items(), key=lambda item: str(item[0]))}
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    if isinstance(value, float):
        if not math.isfinite(value):
            return str(value)
        return 0.0 if value == 0.0 else value
    return value


def canonical_json(value: Any) -> str:
    """Return a stable JSON representation used by every replayable protocol."""
    return json.dumps(_jsonable(value), sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def deterministic_hash(value: Any) -> str:
    return sha256(canonical_json(value).encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class Invalidity:
    code: str
    reason: str
    stage: str
    hard: bool = False
    affected_ids: tuple[str, ...] = ()


@dataclass(frozen=True)
class AuditEvent:
    sequence: int
    timestamp_s: float
    stage: str
    event_type: str
    payload: Mapping[str, Any]
    previous_hash: str = ""

    @property
    def event_hash(self) -> str:
        return deterministic_hash(asdict(self))


@dataclass(frozen=True)
class RecordEnvelope:
    """Transport envelope for immutable cross-stage artifacts."""
    schema_version: str
    record_id: str
    produced_at_s: float
    producer: str
    payload_hash: str
    upstream_hashes: tuple[str, ...] = ()
    invalidities: tuple[Invalidity, ...] = ()

    @classmethod
    def wrap(cls, schema_version: str, record_id: str, produced_at_s: float,
             producer: str, payload: Any, upstream_hashes: Sequence[str] = (),
             invalidities: Sequence[Invalidity] = ()) -> "RecordEnvelope":
        return cls(schema_version, record_id, produced_at_s, producer,
                   deterministic_hash(payload), tuple(upstream_hashes), tuple(invalidities))


def validate_finite_mapping(values: Mapping[str, float], *, name: str = "values") -> None:
    invalid = tuple(key for key, value in values.items() if not math.isfinite(value))
    if invalid:
        raise ValueError(f"{name} contains non-finite entries: {', '.join(invalid)}")
