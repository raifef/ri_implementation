"""Cycle and sparse-resource accounting kept separate from scientific metrics."""
from __future__ import annotations

def one_qubit_gates(distance: int) -> int: return 2 * distance * distance - 1
def two_qubit_gates(distance: int) -> int: return 4 * distance * distance - 4 * distance
def total_controls(distance: int, parameters_per_gate: int) -> int:
    return (one_qubit_gates(distance) + two_qubit_gates(distance)) * parameters_per_gate
def physical_qubits(distance: int) -> int: return 2 * distance * distance - 1
def detector_factors(distance: int) -> int: return distance * distance - 1

def acquisition_accounting(*, epochs: int, candidates: int, cycles_per_candidate: int,
                           evaluation_policies: int = 0, evaluation_cycles: int = 0) -> dict[str, int]:
    candidate = epochs * candidates * cycles_per_candidate
    evaluation = epochs * evaluation_policies * evaluation_cycles
    return {"epochs": epochs, "candidates_per_epoch": candidates,
            "effective_qec_cycles_per_candidate": cycles_per_candidate,
            "candidate_qec_cycles": candidate, "evaluation_qec_cycles": evaluation,
            "total_effective_qec_cycles": candidate + evaluation}
