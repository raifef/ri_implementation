"""Shared benchmark launch definitions and exact configuration hashing."""
from __future__ import annotations

from dataclasses import asdict, dataclass, fields
import hashlib
import json
from pathlib import Path
from typing import Mapping, Sequence

from hdfa_rl_suite.common import deterministic_hash

from .benchmark import (PRIMARY_ARMS, BenchmarkConfig, BenchmarkScenario,
                        benchmark_configuration_hash, benchmark_scenario_registry,
                        default_benchmark_scenarios)


@dataclass(frozen=True)
class BenchmarkLaunchDefinition:
    config: BenchmarkConfig
    scenario_ids: tuple[str, ...]
    primary_only: bool = True
    protocol_id: str = ""
    protocol_path: str = ""
    protocol_sha256: str = ""
    launch_file_sha256: str = ""

    def scenarios(self) -> tuple[BenchmarkScenario, ...]:
        available = {item.scenario_id: item
                     for item in benchmark_scenario_registry(self.config.qubit_count)}
        missing = set(self.scenario_ids) - set(available)
        if missing:
            raise ValueError(f"unknown benchmark scenarios: {sorted(missing)}")
        return tuple(available[item] for item in self.scenario_ids)

    @property
    def arm_names(self) -> tuple[str, ...]:
        return PRIMARY_ARMS if self.primary_only else ()

    @property
    def configuration_hash(self) -> str:
        scientific = benchmark_configuration_hash(
            self.config, self.scenarios(), self.arm_names)
        if not self.launch_file_sha256:
            return scientific
        return deterministic_hash({
            "scientific_configuration_hash": scientific,
            "launch_file_sha256": self.launch_file_sha256,
            "protocol_id": self.protocol_id,
            "protocol_path": self.protocol_path,
            "protocol_sha256": self.protocol_sha256,
        })


def load_launch_definition(path: str | Path) -> BenchmarkLaunchDefinition:
    launch_path = Path(path)
    launch_bytes = launch_path.read_bytes()
    payload = json.loads(launch_bytes.decode("utf-8"))
    if not isinstance(payload, Mapping) or not isinstance(payload.get("benchmark"), Mapping):
        raise ValueError("launch configuration requires a 'benchmark' object")
    allowed = {item.name for item in fields(BenchmarkConfig)}
    unknown = set(payload["benchmark"]) - allowed
    if unknown:
        raise ValueError(f"unknown benchmark configuration fields: {sorted(unknown)}")
    values = dict(payload["benchmark"])
    if "seeds" in values:
        values["seeds"] = tuple(int(item) for item in values["seeds"])
    config = BenchmarkConfig(**values)
    defaults = default_benchmark_scenarios(config.qubit_count)
    scenario_ids = tuple(payload.get("scenario_ids",
                                     [item.scenario_id for item in defaults]))
    protocol = payload.get("protocol", {})
    if protocol and not isinstance(protocol, Mapping):
        raise ValueError("launch protocol binding must be an object")
    protocol_id = str(protocol.get("protocol_id", "")) if protocol else ""
    protocol_path = str(protocol.get("path", "")) if protocol else ""
    protocol_sha256 = str(protocol.get("sha256", "")) if protocol else ""
    if protocol:
        if not protocol_id or not protocol_path or len(protocol_sha256) != 64:
            raise ValueError("launch protocol binding is incomplete")
        document = Path(protocol_path)
        if not document.is_absolute():
            repository_root = Path(__file__).resolve().parents[3]
            document = repository_root/document
        if not document.exists():
            raise ValueError(f"bound protocol does not exist: {protocol_path}")
        observed = hashlib.sha256(document.read_bytes()).hexdigest()
        if observed != protocol_sha256:
            raise ValueError("bound protocol SHA-256 does not match the frozen document")
    return BenchmarkLaunchDefinition(
        config, scenario_ids, bool(payload.get("primary_only", True)),
        protocol_id, protocol_path, protocol_sha256,
        hashlib.sha256(launch_bytes).hexdigest())
