"""Candidate-cycle budget validation against an exact high-shot reference."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from time import perf_counter
from typing import Iterable

import numpy as np

from .common import ValidationCheck, ValidationReport, all_passed, finalize_report
from .gradient_diagnostics import cosine_similarity


# The public Figure-5a protocol uses 3.6e4 QEC cycles per candidate.  This is
# the only published per-candidate reference available to this clean-room
# validator; 100,000 was a repository-local convention and must not be called
# paper scale.
PAPER_SCALE_CYCLES_PER_CANDIDATE = 36_000


@dataclass(frozen=True)
class SampleBudgetConfig:
    seed: int = 161803
    budgets: tuple[int, ...] = (32, 128, 512, 2048, PAPER_SCALE_CYCLES_PER_CANDIDATE)
    candidates: int = 40
    controls: int = 6
    trials: int = 32
    convergence_steps: int = 10
    convergence_loss_fraction: float = .60
    minimum_ranking_accuracy: float = .70
    minimum_gradient_cosine: float = .70
    maximum_harmful_update_probability: float = .15
    # 0.75 made the 2,048-cycle classification hinge on one success out of 32
    # and even change across NumPy binomial implementations.  The stricter
    # threshold rejects that numerically brittle boundary case.
    minimum_convergence_probability: float = .80


def _detector_probabilities(action: np.ndarray, target: np.ndarray) -> np.ndarray:
    return np.clip(.02 + 1.25*(action-target)**2, 1e-6, .45)


def _ranking_accuracy(reference: np.ndarray, observed: np.ndarray) -> float:
    concordant = total = 0
    for left in range(len(reference)):
        for right in range(left+1, len(reference)):
            expected = np.sign(reference[left]-reference[right])
            if expected == 0:
                continue
            measured = np.sign(observed[left]-observed[right])
            concordant += int(expected == measured)
            total += 1
    return concordant/max(1, total)


def _one_gradient(rng: np.random.Generator, cycles: int, mean: np.ndarray,
                  target: np.ndarray, pairs: int, sigma: float = .055) -> tuple[np.ndarray, float]:
    directions = rng.normal(0., sigma, size=(pairs, len(mean)))
    plus = mean[None, :]+directions
    minus = mean[None, :]-directions
    expected_plus = _detector_probabilities(plus, target[None, :])
    expected_minus = _detector_probabilities(minus, target[None, :])
    sampled_plus = rng.binomial(cycles, expected_plus)/cycles
    sampled_minus = rng.binomial(cycles, expected_minus)/cycles
    denominators = 2*directions
    valid = np.abs(denominators) > 1e-10
    slopes = np.divide(sampled_plus-sampled_minus, denominators,
                       out=np.zeros_like(sampled_plus), where=valid)
    estimated_descent = -np.mean(slopes, axis=0)
    reference_candidate_loss = np.concatenate((expected_plus.mean(axis=1), expected_minus.mean(axis=1)))
    observed_candidate_loss = np.concatenate((sampled_plus.mean(axis=1), sampled_minus.mean(axis=1)))
    return estimated_descent, _ranking_accuracy(reference_candidate_loss, observed_candidate_loss)


def run_sample_budget_validation(config: SampleBudgetConfig = SampleBudgetConfig(),
                                 *, injected_faults: Iterable[str] = ()) -> ValidationReport:
    faults = set(injected_faults)
    if config.candidates < 4 or config.candidates % 2:
        raise ValueError("candidate count must be even and at least four")
    target = np.linspace(-.22, .24, config.controls)
    true_descent_initial = -2*1.25*(np.zeros(config.controls)-target)
    rows: list[dict[str, object]] = []
    for budget_index, cycles in enumerate(config.budgets):
        started = perf_counter()
        ranking, cosines, harmful, convergence, final_losses = [], [], [], [], []
        for trial in range(config.trials):
            rng = np.random.default_rng(config.seed + 10_000*budget_index + trial)
            gradient, rank = _one_gradient(rng, cycles, np.zeros(config.controls), target,
                                           config.candidates//2)
            cosine = cosine_similarity(
                {str(i): float(value) for i, value in enumerate(gradient)},
                {str(i): float(value) for i, value in enumerate(true_descent_initial)},
            )
            ranking.append(rank)
            cosines.append(cosine)
            harmful.append(float(np.dot(gradient, true_descent_initial)) <= 0.)
            mean = np.zeros(config.controls)
            initial_loss = float(np.mean(_detector_probabilities(mean, target))-.02)
            for _ in range(config.convergence_steps):
                update, _ = _one_gradient(rng, cycles, mean, target,
                                          config.candidates//2)
                mean = np.clip(mean+.30*update, -.5, .5)
            final_loss = float(np.mean(_detector_probabilities(mean, target))-.02)
            final_losses.append(final_loss)
            convergence.append(final_loss <= config.convergence_loss_fraction*initial_loss)
        elapsed = perf_counter()-started
        row = {
            "cycles_per_candidate": cycles,
            "candidate_count": config.candidates,
            "native_qec_cycles_per_epoch": cycles*config.candidates,
            "reward_ranking_accuracy": float(np.mean(ranking)),
            "gradient_cosine_similarity": float(np.mean(cosines)),
            "gradient_norm_bias": None,
            "harmful_update_probability": float(np.mean(harmful)),
            "convergence_probability": float(np.mean(convergence)),
            "final_mean_policy_excess_edr": float(np.mean(final_losses)),
            "runtime_s": elapsed,
            "classification": "paper-scale" if cycles >= PAPER_SCALE_CYCLES_PER_CANDIDATE else "candidate-reduced",
        }
        true_norm = float(np.linalg.norm(true_descent_initial))
        # The estimator norm is evaluated on a deterministic fresh diagnostic batch.
        diagnostic_rng = np.random.default_rng(config.seed+999_983+budget_index)
        diagnostic_gradient, _ = _one_gradient(
            diagnostic_rng, cycles, np.zeros(config.controls), target,
            config.candidates//2)
        row["gradient_norm_bias"] = float(np.linalg.norm(diagnostic_gradient)/max(true_norm, 1e-12)-1.)
        rows.append(row)

    if "underpowered_budget_accepted" in faults:
        rows[0]["reward_ranking_accuracy"] = 1.
        rows[0]["gradient_cosine_similarity"] = 1.
        rows[0]["harmful_update_probability"] = 0.
        rows[0]["convergence_probability"] = 1.
    passing = [row for row in rows if (
        float(row["reward_ranking_accuracy"]) >= config.minimum_ranking_accuracy
        and float(row["gradient_cosine_similarity"]) >= config.minimum_gradient_cosine
        and float(row["harmful_update_probability"]) <= config.maximum_harmful_update_probability
        and float(row["convergence_probability"]) >= config.minimum_convergence_probability
    )]
    selected = min((int(row["cycles_per_candidate"]) for row in passing), default=None)
    selected_reduced = min((int(row["cycles_per_candidate"]) for row in passing
                            if int(row["cycles_per_candidate"]) <
                            PAPER_SCALE_CYCLES_PER_CANDIDATE), default=None)
    checks = [ValidationCheck(
        "candidate_budget_adequacy", selected is not None and selected > min(config.budgets),
        {"selected_validated_budget": selected,
         "selected_validated_reduced_budget": selected_reduced,
         "paper_scale_budget": PAPER_SCALE_CYCLES_PER_CANDIDATE,
         "thresholds": {
             "ranking": config.minimum_ranking_accuracy,
             "gradient_cosine": config.minimum_gradient_cosine,
             "harmful_update_probability": config.maximum_harmful_update_probability,
             "convergence_probability": config.minimum_convergence_probability,
             "convergence_loss_fraction": config.convergence_loss_fraction,
         }},
        "at least one non-smoke budget passes all thresholds and the smallest tested budget is not accepted",
        "A candidate acquisition budget is accepted only after gradient and convergence adequacy is demonstrated.",
    )]
    paper_row = next((row for row in rows
                      if int(row["cycles_per_candidate"]) == PAPER_SCALE_CYCLES_PER_CANDIDATE), None)
    checks.append(ValidationCheck(
        "paper_scale_reference_available", paper_row is not None,
        paper_row or {}, "36,000-cycle public Figure-5a reference included",
        "The short validation does not silently substitute a smoke budget for the published-scale protocol.",
    ))
    return finalize_report(ValidationReport(
        "candidate-budget-validation.v1", "sample_budget_validation",
        all_passed(checks), tuple(checks), tuple(rows), {
            "config": asdict(config),
            "selected_validated_budget": selected,
            "selected_validated_reduced_budget": selected_reduced,
            "paper_scale_cycles_per_candidate": PAPER_SCALE_CYCLES_PER_CANDIDATE,
            "injected_faults": sorted(faults),
            "evidence_layer": "analytic detector model with finite-shot Monte Carlo",
        },
    ))
