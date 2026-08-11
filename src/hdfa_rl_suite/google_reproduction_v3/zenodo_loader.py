"""Streaming, read-only access to the official Zenodo ZIP release."""

from __future__ import annotations

import io
import json
import math
import zipfile
from pathlib import Path
from typing import BinaryIO, Iterator

import numpy as np

from .preprocessing import parse_experiment_path, summarize_stim_circuit
from .schemas import CircuitShape, ExperimentRecord, ValidationIssue


REQUIRED_EXPERIMENT_MEMBERS = (
    "metadata.json",
    "circuit_ideal.stim",
    "circuit_noisy_si1000.stim",
    "measurements.b8",
    "sweep_bits.b8",
    "detection_events.b8",
    "obs_flips_actual.b8",
)


class ZenodoArchive:
    """A ZIP-backed dataset that never exposes a write operation."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        if not self.path.is_file():
            raise FileNotFoundError(self.path)
        self._archive: zipfile.ZipFile | None = None
        self._names: set[str] | None = None

    def __enter__(self) -> "ZenodoArchive":
        self._archive = zipfile.ZipFile(self.path, mode="r")
        self._names = set(self._archive.namelist())
        return self

    def __exit__(self, *_: object) -> None:
        if self._archive is not None:
            self._archive.close()
        self._archive = None
        self._names = None

    @property
    def archive(self) -> zipfile.ZipFile:
        if self._archive is None:
            raise RuntimeError("ZenodoArchive must be used as a context manager")
        return self._archive

    @property
    def names(self) -> set[str]:
        if self._names is None:
            raise RuntimeError("ZenodoArchive must be used as a context manager")
        return self._names

    def open_member(self, relative_path: str) -> BinaryIO:
        return self.archive.open(relative_path, mode="r")

    def read_text(self, relative_path: str) -> str:
        return self.archive.read(relative_path).decode("utf-8")

    def read_json(self, relative_path: str) -> dict[str, object]:
        value = json.loads(self.archive.read(relative_path))
        if not isinstance(value, dict):
            raise ValueError(f"expected JSON object in {relative_path}")
        return value

    def records(self) -> list[ExperimentRecord]:
        records: list[ExperimentRecord] = []
        for name in sorted(self.names):
            if name.endswith("/metadata.json"):
                records.append(parse_experiment_path(name, self.read_json(name)))
        return records

    def decoder_pathways(self, record: ExperimentRecord) -> tuple[str, ...]:
        prefix = record.data_dir + "decoding_results/"
        suffix = "/obs_flips_predicted.b8"
        return tuple(sorted(name[len(prefix) : -len(suffix)] for name in self.names if name.startswith(prefix) and name.endswith(suffix)))

    def circuit_shape(self, record: ExperimentRecord) -> CircuitShape:
        text = self.read_text(record.data_dir + "circuit_ideal.stim")
        shape = summarize_stim_circuit(text)
        try:
            import stim  # type: ignore[import-not-found]

            circuit = stim.Circuit(text)
            stim_shape = CircuitShape(
                measurements=int(circuit.num_measurements),
                detectors=int(circuit.num_detectors),
                observables=int(circuit.num_observables),
                sweep_bits=int(circuit.num_sweep_bits),
            )
            if stim_shape != shape:
                raise ValueError(f"internal circuit parser disagrees with Stim: {shape} != {stim_shape}")
        except ModuleNotFoundError:
            pass
        return shape

    def member_size(self, relative_path: str) -> int:
        return self.archive.getinfo(relative_path).file_size

    @staticmethod
    def _discard(source: BinaryIO, count: int) -> None:
        remaining = count
        while remaining:
            chunk = source.read(min(remaining, 1024 * 1024))
            if not chunk:
                raise EOFError("b8 member ended before requested shot block")
            remaining -= len(chunk)

    def read_b8_block(
        self,
        relative_path: str,
        *,
        bits_per_shot: int,
        start_shot: int = 0,
        stop_shot: int | None = None,
        total_shots: int | None = None,
    ) -> np.ndarray:
        """Read a contiguous shot block in Stim's little-endian b8 format."""

        if bits_per_shot <= 0 or start_shot < 0:
            raise ValueError("bits_per_shot must be positive and start_shot non-negative")
        if stop_shot is None:
            if total_shots is None:
                raise ValueError("stop_shot or total_shots is required")
            stop_shot = total_shots
        if stop_shot < start_shot:
            raise ValueError("stop_shot precedes start_shot")
        if total_shots is not None and stop_shot > total_shots:
            raise ValueError("requested b8 block exceeds declared shot count")
        bytes_per_shot = math.ceil(bits_per_shot / 8)
        rows = stop_shot - start_shot
        with self.open_member(relative_path) as source:
            self._discard(source, start_shot * bytes_per_shot)
            data = source.read(rows * bytes_per_shot)
        if len(data) != rows * bytes_per_shot:
            raise EOFError(f"truncated b8 block in {relative_path}")
        packed = np.frombuffer(data, dtype=np.uint8).reshape(rows, bytes_per_shot)
        return np.unpackbits(packed, axis=1, bitorder="little")[:, :bits_per_shot]

    def iter_b8_blocks(
        self,
        relative_path: str,
        *,
        bits_per_shot: int,
        total_shots: int,
        shots_per_block: int = 8192,
    ) -> Iterator[np.ndarray]:
        bytes_per_shot = math.ceil(bits_per_shot / 8)
        with self.open_member(relative_path) as source:
            completed = 0
            while completed < total_shots:
                rows = min(shots_per_block, total_shots - completed)
                data = source.read(rows * bytes_per_shot)
                if len(data) != rows * bytes_per_shot:
                    raise EOFError(f"truncated b8 stream in {relative_path}")
                packed = np.frombuffer(data, dtype=np.uint8).reshape(rows, bytes_per_shot)
                yield np.unpackbits(packed, axis=1, bitorder="little")[:, :bits_per_shot]
                completed += rows

    def logical_error_counts(self, record: ExperimentRecord, pathway: str) -> tuple[int, int]:
        actual_path = record.data_dir + "obs_flips_actual.b8"
        predicted_path = record.data_dir + f"decoding_results/{pathway}/obs_flips_predicted.b8"
        if predicted_path not in self.names:
            raise KeyError(f"decoder pathway is unavailable: {predicted_path}")
        errors = 0
        shots = 0
        with self.open_member(actual_path) as actual, self.open_member(predicted_path) as predicted:
            while shots < record.shots:
                count = min(64 * 1024, record.shots - shots)
                left = actual.read(count)
                right = predicted.read(count)
                if len(left) != count or len(right) != count:
                    raise EOFError(f"truncated observable b8 file for {record.experiment_id}")
                a = np.frombuffer(left, dtype=np.uint8)
                b = np.frombuffer(right, dtype=np.uint8)
                errors += int(np.count_nonzero((a ^ b) & 1))
                shots += count
        return errors, shots

    def detector_block(self, record: ExperimentRecord, start: int, stop: int) -> np.ndarray:
        shape = self.circuit_shape(record)
        return self.read_b8_block(
            record.data_dir + "detection_events.b8",
            bits_per_shot=shape.detectors,
            start_shot=start,
            stop_shot=stop,
            total_shots=record.shots,
        )

    def validate_record(self, record: ExperimentRecord) -> list[ValidationIssue]:
        issues: list[ValidationIssue] = []
        for member in REQUIRED_EXPERIMENT_MEMBERS:
            path = record.data_dir + member
            if path not in self.names:
                issues.append(ValidationIssue(path, "MISSING_MEMBER", "required experiment member is absent"))
        if issues:
            return issues
        shape = self.circuit_shape(record)
        dimensions = {
            "measurements.b8": shape.measurements,
            "sweep_bits.b8": shape.sweep_bits,
            "detection_events.b8": shape.detectors,
            "obs_flips_actual.b8": shape.observables,
        }
        for member, bits in dimensions.items():
            path = record.data_dir + member
            expected = record.shots * math.ceil(bits / 8)
            observed = self.member_size(path)
            if observed != expected:
                issues.append(
                    ValidationIssue(path, "B8_SIZE_MISMATCH", f"observed {observed} bytes, expected {expected}")
                )
        for pathway in self.decoder_pathways(record):
            path = record.data_dir + f"decoding_results/{pathway}/obs_flips_predicted.b8"
            expected = record.shots * math.ceil(shape.observables / 8)
            if self.member_size(path) != expected:
                issues.append(ValidationIssue(path, "B8_SIZE_MISMATCH", "predicted observable file has incorrect size"))
        return issues


def build_fixture_zip(path: Path) -> None:
    """Create a tiny deterministic fixture used only by unit tests."""

    base = "surface_code_distance_3_5_7/traditional_calibration/d3_0+0j/X/r002/"
    circuit = """CX sweep[0] 0\nM 0 1\nDETECTOR rec[-1]\nREPEAT 2 {\n M 0\n DETECTOR rec[-1]\n}\nOBSERVABLE_INCLUDE(0) rec[-1]\n"""
    metadata = {"basis": "X", "rounds": 2, "shots": 4, "qubit_coords": [[0, 0], [1, 0]]}
    detector_rows = bytes([0b000, 0b001, 0b010, 0b111])
    actual = bytes([0, 1, 0, 1])
    predicted = bytes([0, 0, 0, 1])
    with zipfile.ZipFile(path, mode="w", compression=zipfile.ZIP_STORED) as archive:
        archive.writestr("README/README.md", "Reinforcement Learning Control of Quantum Error Correction on Willow")
        archive.writestr(base + "metadata.json", json.dumps(metadata))
        archive.writestr(base + "circuit_ideal.stim", circuit)
        archive.writestr(base + "circuit_noisy_si1000.stim", circuit)
        archive.writestr(base + "measurements.b8", bytes(4))
        archive.writestr(base + "sweep_bits.b8", bytes(4))
        archive.writestr(base + "detection_events.b8", detector_rows)
        archive.writestr(base + "obs_flips_actual.b8", actual)
        archive.writestr(
            base + "decoding_results/test_decoder/obs_flips_predicted.b8",
            predicted,
        )


def bytes_stream(value: bytes) -> io.BytesIO:
    return io.BytesIO(value)
