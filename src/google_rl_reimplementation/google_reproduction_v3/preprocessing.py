"""Deterministic preprocessing for Stim text and archive paths."""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from typing import Iterable

from .schemas import CircuitShape, ExperimentRecord


_SWEEP_RE = re.compile(r"sweep\[(\d+)\]")
_REPEAT_RE = re.compile(r"^REPEAT\s+(\d+)\s*\{$", re.IGNORECASE)


@dataclass
class _Counts:
    measurements: int = 0
    detectors: int = 0
    observables: int = 0
    max_sweep_index: int = -1

    def add_repeated(self, other: "_Counts", repetitions: int) -> None:
        self.measurements += repetitions * other.measurements
        self.detectors += repetitions * other.detectors
        self.observables = max(self.observables, other.observables)
        self.max_sweep_index = max(self.max_sweep_index, other.max_sweep_index)


def stable_digest(value: str, *, length: int = 16) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:length]


def _measurement_targets(op: str, rest: list[str]) -> int:
    if op == "MPP":
        return sum(1 for token in rest if "*" in token or token[:1] in {"X", "Y", "Z"})
    return len(rest)


def summarize_stim_circuit(text: str) -> CircuitShape:
    """Count b8 dimensions without executing or rewriting a Stim circuit.

    The parser handles nested ``REPEAT`` blocks and the measurement operations
    used by the released memory circuits.  If Stim is installed, callers may
    additionally compare these counts with ``stim.Circuit`` in validation.
    """

    stack: list[tuple[int, _Counts]] = [(1, _Counts())]
    measurement_ops = {"M", "MX", "MY", "MR", "MRX", "MRY", "MPP", "MPAD"}
    for line_number, raw in enumerate(text.splitlines(), start=1):
        line = raw.split("#", 1)[0].strip()
        if not line:
            continue
        match = _REPEAT_RE.match(line)
        if match:
            stack.append((int(match.group(1)), _Counts()))
            continue
        if line == "}":
            if len(stack) == 1:
                raise ValueError(f"unexpected closing brace at Stim line {line_number}")
            repetitions, counts = stack.pop()
            stack[-1][1].add_repeated(counts, repetitions)
            continue

        tokens = line.split()
        op_token = tokens[0]
        op = op_token.split("(", 1)[0].upper()
        counts = stack[-1][1]
        if op in measurement_ops:
            counts.measurements += _measurement_targets(op, tokens[1:])
        elif op == "DETECTOR":
            counts.detectors += 1
        elif op == "OBSERVABLE_INCLUDE":
            try:
                observable = int(op_token[op_token.index("(") + 1 : op_token.index(")")])
            except (ValueError, IndexError) as exc:
                raise ValueError(f"malformed OBSERVABLE_INCLUDE at Stim line {line_number}") from exc
            counts.observables = max(counts.observables, observable + 1)
        for value in _SWEEP_RE.findall(line):
            counts.max_sweep_index = max(counts.max_sweep_index, int(value))

    if len(stack) != 1:
        raise ValueError("unterminated REPEAT block in Stim circuit")
    counts = stack[0][1]
    return CircuitShape(
        measurements=counts.measurements,
        detectors=counts.detectors,
        observables=counts.observables,
        sweep_bits=counts.max_sweep_index + 1,
    )


def parse_experiment_path(path: str, metadata: dict[str, object]) -> ExperimentRecord:
    parts = path.rstrip("/").split("/")
    if parts[-1] == "metadata.json":
        parts = parts[:-1]
    if len(parts) == 4 and parts[0] == "color_code_distance_5":
        family = "color_code"
        distance = 5
        condition, basis, rounds_dir = parts[1:]
        subgrid = "color_d5"
    elif len(parts) == 5 and parts[0] == "surface_code_distance_3_5_7":
        family = "surface_code"
        condition, subgrid, basis, rounds_dir = parts[1:]
        distance_match = re.match(r"d(\d+)_", subgrid)
        if distance_match is None:
            raise ValueError(f"cannot infer surface-code distance from {subgrid!r}")
        distance = int(distance_match.group(1))
    else:
        raise ValueError(f"unrecognized experiment path: {path}")

    rounds = int(rounds_dir.removeprefix("r"))
    shots = int(metadata["shots"])
    metadata_rounds = int(metadata["rounds"])
    metadata_basis = str(metadata["basis"])
    if rounds != metadata_rounds or basis != metadata_basis:
        raise ValueError(f"path/metadata mismatch for {path}")
    coords = tuple(tuple(int(y) for y in x) for x in metadata["qubit_coords"])  # type: ignore[arg-type]
    data_dir = "/".join(parts) + "/"
    return ExperimentRecord(
        experiment_id=stable_digest(data_dir),
        data_dir=data_dir,
        code_family=family,
        distance=distance,
        condition=condition,
        subgrid=subgrid,
        basis=basis,
        rounds=rounds,
        shots=shots,
        qubit_coords=coords,
    )


def contiguous_blocks(length: int, block_size: int) -> Iterable[tuple[int, int]]:
    if length < 0 or block_size <= 0:
        raise ValueError("length must be non-negative and block_size positive")
    for start in range(0, length, block_size):
        yield start, min(length, start + block_size)

