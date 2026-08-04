from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest

from google_rl_reimplementation.google_reproduction_v3.dataset_manifest import hash_archive_once
from google_rl_reimplementation.google_reproduction_v3.estimators import (
    error_per_cycle_to_memory_failure,
    independent_nonlinear_decay_estimator,
    memory_failure_to_error_per_cycle,
    repository_decay_estimator,
)
from google_rl_reimplementation.google_reproduction_v3.preprocessing import summarize_stim_circuit
from google_rl_reimplementation.google_reproduction_v3.reporting import canonical_hash
from google_rl_reimplementation.google_reproduction_v3.schemas import (
    CERTIFICATION_SEEDS,
    ReproductionStatus,
    SurrogateValidationOutcome,
)
from google_rl_reimplementation.google_reproduction_v3.spectral import power_ratio_db
from google_rl_reimplementation.google_reproduction_v3.surrogate import EmpiricalStaticSurrogate
from google_rl_reimplementation.google_reproduction_v3.zenodo_loader import ZenodoArchive, build_fixture_zip


def test_allowed_status_vocabularies_are_frozen() -> None:
    assert {value.value for value in ReproductionStatus} == {
        "EXACTLY_REPRODUCED",
        "REPRODUCED_WITH_DOCUMENTED_APPROXIMATION",
        "NOT_REPRODUCIBLE_FROM_RELEASED_DATA",
        "ANALYSIS_DEFINITION_AMBIGUOUS",
    }
    assert {value.value for value in SurrogateValidationOutcome} == {
        "EMPIRICAL_SURROGATE_VALIDATED",
        "EMPIRICAL_SURROGATE_PARTIALLY_VALIDATED",
        "EMPIRICAL_SURROGATE_REJECTED",
    }


def test_circuit_parser_expands_repeats_and_preserves_b8_dimensions() -> None:
    text = """CX sweep[2] 0
M 0 1
DETECTOR rec[-1]
REPEAT 3 {
 M 0
 DETECTOR rec[-1]
}
OBSERVABLE_INCLUDE(0) rec[-1]
"""
    shape = summarize_stim_circuit(text)
    assert shape.measurements == 5
    assert shape.detectors == 4
    assert shape.observables == 1
    assert shape.sweep_bits == 3


def test_b8_little_endian_and_xor(tmp_path: Path) -> None:
    fixture = tmp_path / "fixture.zip"
    build_fixture_zip(fixture)
    with ZenodoArchive(fixture) as archive:
        [record] = archive.records()
        assert archive.validate_record(record) == []
        detections = archive.detector_block(record, 0, 4)
        assert detections.tolist() == [[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 1]]
        errors, shots = archive.logical_error_counts(record, "test_decoder")
        assert (errors, shots) == (1, 4)


def test_per_cycle_parity_inverse_not_linear_division() -> None:
    error_per_cycle = 0.0123
    rounds = 50
    failure = error_per_cycle_to_memory_failure(error_per_cycle, rounds)
    assert memory_failure_to_error_per_cycle(failure, rounds) == pytest.approx(error_per_cycle, abs=1e-14)
    assert memory_failure_to_error_per_cycle(failure, rounds) != pytest.approx(failure / rounds, rel=1e-3)


def test_decay_fit_recovers_known_error_rate() -> None:
    rounds = np.array([10, 30, 50, 70, 90])
    shots = np.full(len(rounds), 2_000_000)
    target = 0.0017
    contrast = 0.985
    probabilities = 0.5 * (1 - contrast * (1 - 2 * target) ** rounds)
    errors = np.rint(shots * probabilities).astype(int)
    repository = repository_decay_estimator(rounds, errors, shots)
    independent = independent_nonlinear_decay_estimator(rounds, errors, shots)
    assert repository.logical_error_per_cycle == pytest.approx(target, abs=2e-7)
    assert independent.logical_error_per_cycle == pytest.approx(target, abs=2e-7)
    assert repository.intercept == pytest.approx(math.log(contrast), abs=2e-5)


def test_power_db_conversion_uses_power_convention() -> None:
    assert power_ratio_db(4.0, 1.0) == pytest.approx(6.020599913, rel=1e-9)
    assert power_ratio_db(1.0, 4.0) == pytest.approx(-6.020599913, rel=1e-9)


def test_checksum_cache_avoids_rehash(tmp_path: Path) -> None:
    source = tmp_path / "archive.bin"
    source.write_bytes(b"zenodo-fixture" * 100)
    cache = tmp_path / "cache.json"
    first = hash_archive_once(source, cache)
    second = hash_archive_once(source, cache)
    assert first["cache_hit"] is False
    assert second["cache_hit"] is True
    assert first["sha256"] == second["sha256"]


def test_static_surrogate_rejects_unlogged_counterfactual_actions() -> None:
    surrogate = EmpiricalStaticSurrogate(-2.5, 0.2, 0.03, 0.1, None, None, "split")
    events = surrogate.sample_detection_events(shots=20, detectors=4, seed=7)
    assert events.shape == (20, 4)
    with pytest.raises(ValueError, match="no control-action support"):
        surrogate.sample_detection_events(shots=2, detectors=2, seed=7, action=np.zeros(2))


def test_split_hash_is_deterministic_and_certification_seeds_untouched() -> None:
    value = {"experiment": "abc", "shot_block": [0, 2048]}
    assert canonical_hash(value) == canonical_hash(json.loads(json.dumps(value)))
    assert CERTIFICATION_SEEDS == tuple(range(8101, 8113))
    config = json.loads(Path("configs/google_rl_v3/data_splits.yaml").read_text(encoding="utf-8"))
    assert config["final_certification"]["status"] == "LOCKED_UNCONSUMED"
    assert tuple(config["final_certification"]["certification_seeds"]) == CERTIFICATION_SEEDS


def test_faster_v2_results_are_not_labelled_algorithm_failure() -> None:
    source = Path("src/google_rl_reimplementation/google_reproduction_v3/reporting.py").read_text(encoding="utf-8")
    assert '"randomized_recovery"' in source
    assert '"classification": "TASK_NOT_COMMENSURABLE"' in source
    assert "526 epochs is faster, not worse" in source
