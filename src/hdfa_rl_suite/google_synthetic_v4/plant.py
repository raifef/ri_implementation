"""Frozen sparse synthetic plants with a truth-isolated observation interface."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Iterable, Mapping

import numpy as np

from .config import canonical_hash, load_ensemble, load_priors


def surface_code_gate_count(distance: int) -> int:
    if distance < 3 or distance % 2 == 0:
        raise ValueError("surface-code distance must be odd and at least 3")
    return 6 * distance * distance - 4 * distance - 1


def surface_code_control_count(distance: int, controls_per_gate: int = 30) -> int:
    if controls_per_gate <= 0:
        raise ValueError("controls_per_gate must be positive")
    return surface_code_gate_count(distance) * controls_per_gate


@dataclass(frozen=True)
class PlantSpec:
    plant_id: str
    split: str
    family: str
    plant_draw_seed: int
    detector_count: int
    control_count: int
    graph_offset: int
    floor_mean: float
    floor_cv: float
    detector_covariance: float
    overdispersion: float
    curvature_mean: float
    asymmetry: float
    coupling: float
    drift_amplitude: float
    drift_frequency: float
    drift_phase: float
    step_epoch: int
    step_amplitude: float
    spoil_severity: float
    coupling_pattern: str
    evaluation_seed: int | None = None


def _draw_spec(row: Mapping[str, Any], *, detector_count: int, controls_per_cell: int) -> PlantSpec:
    rng = np.random.default_rng(int(row["plant_draw_seed"]))
    return PlantSpec(
        plant_id=str(row["plant_id"]), split=str(row["split"]), family=str(row["family"]),
        plant_draw_seed=int(row["plant_draw_seed"]), detector_count=detector_count,
        control_count=detector_count * controls_per_cell,
        graph_offset=int(rng.integers(1, detector_count)), floor_mean=float(rng.uniform(0.045, 0.105)),
        floor_cv=float(rng.uniform(0.12, 0.32)), detector_covariance=float(rng.uniform(0.00045, 0.0018)),
        overdispersion=float(rng.uniform(0.88, 1.10)), curvature_mean=float(rng.uniform(0.025, 0.070)),
        asymmetry=float(rng.uniform(-0.006, 0.006)), coupling=float(rng.uniform(0.0, 0.012)),
        drift_amplitude=float(rng.uniform(0.10, 0.28)), drift_frequency=float(rng.uniform(0.0015, 0.014)),
        drift_phase=float(rng.uniform(0.0, 2 * np.pi)), step_epoch=int(rng.integers(28, 55)),
        step_amplitude=float(rng.uniform(0.16, 0.34)), spoil_severity=float(rng.uniform(0.30, 0.90)),
        coupling_pattern=("alternating" if rng.random() < 0.5 else "signed_random"),
        evaluation_seed=(int(row["evaluation_seed"]) if "evaluation_seed" in row else None),
    )


def frozen_specs() -> tuple[PlantSpec, ...]:
    cfg = load_ensemble()
    sub = cfg["development_subgraph"]
    return tuple(_draw_spec(row, detector_count=int(sub["detectors"]), controls_per_cell=int(sub["controls_per_detector_cell"])) for row in cfg["draws"])


class SyntheticPlant:
    """Sparse local plant. Controllers only receive sampled detector counts."""

    def __init__(self, spec: PlantSpec):
        self.spec = spec
        rng = np.random.default_rng(spec.plant_draw_seed + 1_000_000)
        d, c = spec.detector_count, spec.control_count
        self.sensitivity = rng.uniform(0.75, 1.25, c)
        self.base_optimum = rng.uniform(-0.16, 0.16, c)
        self.initial_mean = np.clip(self.base_optimum + rng.normal(0.18, 0.025, c), -0.8, 0.8)
        self.floors = np.clip(rng.lognormal(np.log(spec.floor_mean), spec.floor_cv, d), 0.015, 0.18)
        self.curvature = rng.uniform(0.75, 1.25, d) * spec.curvature_mean
        self.local_phase = rng.uniform(0, 2 * np.pi, c)
        self.local_scale = rng.uniform(0.55, 1.0, c)
        self.step_mask = rng.random(c) < 0.35
        self.common_vector = rng.normal(size=d)
        self.common_vector /= max(np.std(self.common_vector), 1e-12)
        self.mask = np.zeros((d, c), dtype=bool)
        per = c // d
        for detector in range(d):
            for neighbor in (detector, (detector + spec.graph_offset) % d, (detector - 1) % d):
                self.mask[detector, neighbor * per:(neighbor + 1) * per] = True
        self._indices = tuple(np.flatnonzero(row) for row in self.mask)

    def controller_view(self) -> dict[str, Any]:
        """Public metadata only; excludes optimum and latent drift state."""
        return {"plant_id": self.spec.plant_id, "control_count": self.spec.control_count,
                "detector_count": self.spec.detector_count, "mask": self.mask.copy(),
                "sensitivity": self.sensitivity.copy(), "bound": 1.0}

    def optimum(self, epoch: int, *, family_override: str | None = None, frequency: float | None = None,
                amplitude: float | None = None, no_drift: bool = False) -> np.ndarray:
        if no_drift:
            return self.base_optimum.copy()
        family = family_override or self.spec.family
        f = self.spec.drift_frequency if frequency is None else float(frequency)
        a = self.spec.drift_amplitude if amplitude is None else float(amplitude)
        phase = self.spec.drift_phase
        opt = self.base_optimum.copy()
        if family == "local_quadratic_drift":
            opt += a * self.local_scale * np.sin(2 * np.pi * f * epoch + self.local_phase)
        elif family == "common_mode_plus_local":
            common = 0.72 * a * np.sin(2 * np.pi * f * epoch + phase)
            local = 0.28 * a * self.local_scale * np.sin(2 * np.pi * 1.7 * f * epoch + self.local_phase)
            opt += common + local
        elif family == "step_perturbation":
            if epoch >= self.spec.step_epoch:
                opt[self.step_mask] += self.spec.step_amplitude
        elif family == "sinusoidal_steering":
            opt += a * self.local_scale * np.sin(2 * np.pi * f * epoch + self.local_phase)
        elif family in {"randomized_policy_recovery", "large_sparse_scaling"}:
            pass
        else:
            raise ValueError(f"unknown synthetic family: {family}")
        return np.clip(opt, -0.85, 0.85)

    def detector_rates(self, actions_normalized: np.ndarray, optimum: np.ndarray) -> np.ndarray:
        actions = np.atleast_2d(np.asarray(actions_normalized, dtype=float))
        optimum = np.asarray(optimum, dtype=float)
        if actions.shape[1] != self.spec.control_count or optimum.shape != (self.spec.control_count,):
            raise ValueError("plant action/optimum shape mismatch")
        out = np.empty((actions.shape[0], self.spec.detector_count))
        for detector, idx in enumerate(self._indices):
            delta = actions[:, idx] - optimum[idx]
            quadratic = self.curvature[detector] * np.mean(delta * delta, axis=1)
            asymmetric = self.spec.asymmetry * np.mean(delta * delta * delta, axis=1)
            coupled = self.spec.coupling * np.mean(delta[:, :-1] * delta[:, 1:], axis=1)
            out[:, detector] = self.floors[detector] + quadratic + asymmetric + coupled
        return np.clip(out, 1e-6, 0.35)

    def logical_risk(self, actions_normalized: np.ndarray, optimum: np.ndarray) -> np.ndarray:
        rates = self.detector_rates(actions_normalized, optimum)
        excess = np.maximum(rates - self.floors[None, :], 0.0).mean(axis=1)
        floor = 0.0048 + 0.012 * (self.floors.mean() - 0.075)
        return np.clip(floor + 0.55 * excess, 1e-6, 0.25)

    def acquire_counts(self, actions_normalized: np.ndarray, optimum: np.ndarray, cycles: int,
                       rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
        rates = self.detector_rates(actions_normalized, optimum)
        common_sd = min(np.sqrt(self.spec.detector_covariance), 0.025)
        common = rng.normal(size=(len(rates), 1)) * common_sd
        perturbed = np.clip(rates + common * self.common_vector[None, :], 1e-6, 0.35)
        effective = max(1, int(round(cycles / self.spec.overdispersion)))
        counts_eff = rng.binomial(effective, perturbed)
        rates_observed = counts_eff / effective
        return np.rint(rates_observed * cycles).astype(np.int64), rates


def ensemble_contract() -> dict[str, Any]:
    specs = frozen_specs()
    ids = [s.plant_id for s in specs]
    if len(ids) != len(set(ids)):
        raise ValueError("plant identifiers are not unique")
    rows = [asdict(s) for s in specs]
    cert = [row for row in rows if row["split"] == "certification"]
    physical = [tuple(row[k] for k in ("drift_phase","drift_amplitude","drift_frequency","graph_offset","curvature_mean","detector_covariance","spoil_severity","coupling_pattern")) for row in cert]
    if len(physical) != len(set(physical)):
        raise ValueError("certification plants are not physically disjoint")
    d15 = surface_code_control_count(15)
    return {
        "schema_version":"google-synthetic-v4-plant-contract.v1", "frozen":True,
        "plant_draws":rows, "plant_count":len(rows), "family_count":len(set(s.family for s in specs)),
        "distance_15_control_count":d15, "distance_15_exact":d15 == 38670,
        "controller_truth_isolation":"controller_view excludes oracle optimum, drift phase, and future disturbances",
        "observation_constraints":load_priors()["zenodo_constraints"],
        "ensemble_hash":canonical_hash(rows), "certification_evaluation_seeds_consumed":False,
        "claim_boundary":"Frozen synthetic action-response ensemble; not reconstructed Google hardware dynamics.",
        "summary":{"status":"FROZEN","plants":len(rows),"families":len(set(s.family for s in specs)),"distance_15_controls":d15}
    }


def specs_for_split(split: str, families: Iterable[str] | None = None) -> tuple[PlantSpec, ...]:
    wanted = None if families is None else set(families)
    return tuple(s for s in frozen_specs() if s.split == split and (wanted is None or s.family in wanted))
