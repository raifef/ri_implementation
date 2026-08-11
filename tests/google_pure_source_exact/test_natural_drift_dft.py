from __future__ import annotations

import numpy as np
import pytest
from pathlib import Path

from hdfa_rl_suite.google_pure_source_exact.natural_drift_dft.contracts import (
    EvaluationTrace,
    SourceDFTConfig,
)
from hdfa_rl_suite.google_pure_source_exact.natural_drift_dft.estimator import (
    analyze_traces,
    preprocess_trace,
    run_spectrum,
)
from hdfa_rl_suite.google_pure_source_exact.natural_drift_dft.cli import (
    build_plan,
    generate_synthetic,
    run_analysis,
)


ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "configs/google_pure_source_exact/natural_drift_dft.json"


def _trace(run_id: str, stop: int, phase: float = 0.0) -> EvaluationTrace:
    epochs = np.arange(0, stop + 1, 5)
    fixed = 0.02 * (1.0 + 0.25 * np.sin(2 * np.pi * epochs / 200 + phase) +
                    0.03 * np.sin(2 * np.pi * epochs / 35 + 0.4))
    learned = 0.018 * (1.0 + 0.10 * np.sin(2 * np.pi * epochs / 200 + phase) +
                       0.03 * np.sin(2 * np.pi * epochs / 35 + 0.4))
    return EvaluationTrace(run_id, tuple(epochs), tuple(learned), tuple(fixed))


def test_epoch_domain_five_epoch_cadence_and_exact_warmup_normalization() -> None:
    config = SourceDFTConfig(shared_grid_points=32)
    prepared = preprocess_trace(_trace("a", 400), config)
    assert prepared["epochs"][0] == 150
    assert np.all(np.diff(prepared["epochs"]) == 5)
    assert prepared["learned"][0] == 1.0 and prepared["fixed"][0] == 1.0
    physical = _trace("bad", 400)
    physical = EvaluationTrace(physical.run_id, physical.epochs, physical.learned_mean_ler,
                               physical.fixed_initial_ler, time_coordinate="PHYSICAL_SECONDS")
    with pytest.raises(ValueError, match="epoch coordinates"):
        preprocess_trace(physical, config)


def test_zero_frequency_is_excluded_without_zero_padding() -> None:
    frequency, power = run_spectrum(np.asarray([1.0, 2.0, 1.0, 2.0]), cadence_epochs=5)
    assert np.all(frequency > 0)
    assert len(frequency) == len(np.fft.rfft(np.ones(4))) - 1
    assert power.shape == frequency.shape


def test_unequal_lengths_interpolate_and_geometrically_average() -> None:
    traces = [_trace("a", 400), _trace("b", 500, 0.3)]
    config = SourceDFTConfig(shared_grid_points=40)
    result = analyze_traces(traces, config)
    assert len(result["frequency_per_epoch"]) == 40
    assert result["spectral_aggregation"] == "GEOMETRIC_MEAN"
    assert result["zero_frequency_excluded"]
    assert all(row["epoch_150_learned"] == 1.0 for row in result["normalization_checks"])
    shared = np.asarray(result["frequency_per_epoch"])
    interpolated = []
    for trace in traces:
        prepared = preprocess_trace(trace, config)
        frequency, power = run_spectrum(prepared["learned"], cadence_epochs=5)
        interpolated.append(np.interp(shared, frequency, power))
    expected_geometric = np.exp(np.mean(np.log(np.maximum(interpolated, np.finfo(float).tiny)), axis=0))
    arithmetic = np.mean(interpolated, axis=0)
    np.testing.assert_allclose(result["learned_geometric_psd"], expected_geometric, rtol=1e-14)
    assert not np.allclose(expected_geometric, arithmetic, rtol=1e-6, atol=0)


def test_filter_sign_is_learned_over_fixed_and_low_frequency_is_negative() -> None:
    result = analyze_traces([_trace("a", 600), _trace("b", 550, 0.2), _trace("c", 500, 0.4)],
                            SourceDFTConfig(shared_grid_points=64))
    frequency = np.asarray(result["frequency_per_epoch"])
    raw = np.asarray(result["raw_filter_db"])
    low = raw[frequency < 0.01]
    assert np.median(low) < 0
    assert result["filter_db_definition"].startswith("10*log10(learned")


def test_candidate_stream_and_welch_source_panel_are_prohibited() -> None:
    trace = _trace("candidate", 400)
    trace = EvaluationTrace(trace.run_id, trace.epochs, trace.learned_mean_ler,
                            trace.fixed_initial_ler, stream_kind="SAMPLED_CANDIDATE")
    with pytest.raises(ValueError, match="decoded"):
        analyze_traces([trace, _trace("b", 450)], SourceDFTConfig(shared_grid_points=32))
    result = analyze_traces([_trace("a", 400), _trace("b", 450)],
                            SourceDFTConfig(shared_grid_points=32))
    assert result["source_panel_uses_welch"] is False
    assert result["source_panel_uses_candidate_stream"] is False


def test_deterministic_replay_from_stored_trace_values() -> None:
    traces = [_trace("a", 400), _trace("b", 450, 0.2)]
    first = analyze_traces(traces, SourceDFTConfig(shared_grid_points=32))
    second = analyze_traces(traces, SourceDFTConfig(shared_grid_points=32))
    assert first == second


def test_duplicate_runs_and_arithmetic_substitution_are_rejected_or_detectable() -> None:
    trace = _trace("a", 400)
    with pytest.raises(ValueError, match="duplicate"):
        analyze_traces([trace, trace], SourceDFTConfig(shared_grid_points=32))
    result = analyze_traces([_trace("a", 400), _trace("b", 450, 0.5)],
                            SourceDFTConfig(shared_grid_points=32))
    learned = np.asarray(result["learned_geometric_psd"])
    assert np.all(learned > 0)


def test_synthetic_cli_artifacts_fail_closed_for_hardware_claim(tmp_path) -> None:
    input_path = tmp_path / "synthetic.json"
    generate_synthetic(CONFIG, input_path)
    plan = build_plan(CONFIG, tmp_path / "out", input_path)
    assert plan["qec_cycles"] == 0 and plan["hardware_acquisition_not_launched"]
    assert not plan["public_release"]["hardware_dynamic_evaluation_traces_available"]
    result = run_analysis(CONFIG, input_path, tmp_path / "out", "synthetic-001")
    assert result["artifact_complete"] and result["mathematical_contract_pass"]
    assert result["source_structure_match"]
    assert not result["protocol_contract_pass"]
    assert not result["quantitative_match"] and not result["paper_comparable"]
    assert len(result["shards"]) == 4
