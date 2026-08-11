"""Serializable log-hyperedge prior and fixed-shot offline objective."""
from dataclasses import dataclass
from hashlib import sha256
import json
from pathlib import Path
from typing import Callable
import numpy as np

@dataclass(frozen=True)
class DemPrior:
    log_probabilities: np.ndarray
    dem_hash: str
    decoder_backend: str
    def probabilities(self):
        values = np.exp(np.asarray(self.log_probabilities, dtype=float))
        if np.any(values <= 0) or np.any(values >= .5):
            raise ValueError("DEM hyperedge probabilities must lie strictly between 0 and 0.5")
        return values
    def save(self, path: Path) -> dict:
        self.probabilities()
        payload = {"schema": "dem-log-prior.v1", "dem_hash": self.dem_hash,
                   "decoder_backend": self.decoder_backend, "log_probabilities": self.log_probabilities.tolist()}
        payload["prior_hash"] = sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2, sort_keys=True)+"\n", encoding="utf-8")
        return {"prior_hash": payload["prior_hash"], "dem_hash": self.dem_hash}

def evaluate_candidates_offline(candidates: np.ndarray, decode_same_shots: Callable[[np.ndarray], int], shot_count: int) -> np.ndarray:
    if shot_count <= 0: raise ValueError("shot_count must be positive")
    candidates = np.asarray(candidates, dtype=float)
    if candidates.ndim != 2: raise ValueError("candidate log priors must be a matrix")
    failures = np.asarray([decode_same_shots(row) for row in candidates], dtype=int)
    if np.any(failures < 0) or np.any(failures > shot_count): raise ValueError("decoder returned an invalid logical failure count")
    return -np.log10((failures + .5) / (shot_count + 1.0))
