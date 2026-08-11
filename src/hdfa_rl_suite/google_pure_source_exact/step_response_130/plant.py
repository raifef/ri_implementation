"""Sparse public distance-5 analogue with a hidden, piecewise-constant XY target."""
from __future__ import annotations
from hashlib import sha256
import numpy as np

class SourceStepPlant:
    def __init__(self, controls: int = 924, detectors: int = 24, direction_coordinate: int = 0,
                 target_delta: float = .5, onset_epoch: int = 60):
        if controls != 924 or detectors != 24: raise ValueError("public analogue requires 924 controls and 24 distance-5 detectors")
        self.controls, self.detectors, self.direction_coordinate = controls, detectors, direction_coordinate
        self.target_delta, self.onset_epoch = float(target_delta), int(onset_epoch)
        self.mask = np.zeros((detectors, controls), dtype=bool)
        for coordinate in range(controls): self.mask[coordinate % detectors, coordinate] = True
        self.sensitivity = np.linspace(.00008, .00016, controls)
        self.base_edr = np.linspace(.012, .018, detectors)
        self.plant_hash = sha256(self.mask.tobytes()+self.sensitivity.tobytes()+self.base_edr.tobytes()).hexdigest()

    def hidden_target(self, epoch: int) -> np.ndarray:
        target = np.zeros(self.controls)
        if epoch >= self.onset_epoch: target[self.direction_coordinate] = self.target_delta
        return target

    def expected_edr(self, controls: np.ndarray, epoch: int, *, drift_enabled: bool = True,
                     target_controls: np.ndarray | None = None) -> np.ndarray:
        values = np.asarray(controls, dtype=float)
        if values.shape != (self.controls,): raise ValueError("control vector must have 924 native coordinates")
        if target_controls is None:
            target = self.hidden_target(epoch) if drift_enabled else np.zeros(self.controls)
        else:
            target = np.asarray(target_controls, dtype=float)
            if target.shape != (self.controls,): raise ValueError("target vector must align with controls")
        cost = self.sensitivity * np.square(values-target)
        return np.clip(self.base_edr + self.mask @ cost, 0, .49)

    def common_random_counts(self, control_vectors: dict[str, np.ndarray], epoch: int, qec_cycles: int, seed: int,
                             *, drift_enabled: bool = True,
                             target_controls: np.ndarray | None = None) -> dict[str, np.ndarray]:
        if qec_cycles <= 0: raise ValueError("qec_cycles must be positive")
        rng = np.random.default_rng(seed)
        names = tuple(control_vectors)
        probabilities = np.stack([self.expected_edr(
            control_vectors[name], epoch, drift_enabled=drift_enabled,
            target_controls=target_controls) for name in names])
        result = {name: np.zeros(self.detectors, dtype=np.int64) for name in names}
        # Exact common-uniform coupling, sampled by multinomial probability intervals without allocating one value per cycle.
        for detector in range(self.detectors):
            order = np.argsort(probabilities[:, detector]); sorted_p = probabilities[order, detector]
            interval_p = np.diff(np.concatenate(([0.0], sorted_p, [1.0])))
            bins = rng.multinomial(qec_cycles, interval_p); cumulative = np.cumsum(bins[:-1])
            for rank, condition_index in enumerate(order): result[names[condition_index]][detector] = cumulative[rank]
        return result
