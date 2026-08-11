"""Immutable, hash-verified physical QEC shot artifacts."""
from __future__ import annotations
from dataclasses import dataclass
from hashlib import sha256
import json
from pathlib import Path
import numpy as np

def _array_hash(*arrays: np.ndarray) -> str:
    digest = sha256()
    for array in arrays:
        value = np.ascontiguousarray(array)
        digest.update(str(value.dtype).encode())
        digest.update(np.asarray(value.shape, dtype=np.int64).tobytes())
        digest.update(value.tobytes())
    return digest.hexdigest()

@dataclass(frozen=True)
class FrozenQecData:
    detection_events: np.ndarray
    logical_observables: np.ndarray
    shot_ids: np.ndarray
    epoch_ids: np.ndarray
    physical_arm: str
    physical_policy_hash: str
    data_hash: str

    def validate(self) -> None:
        arrays = (self.detection_events, self.logical_observables, self.shot_ids, self.epoch_ids)
        if any(len(a) != len(self.shot_ids) for a in arrays): raise ValueError("all QEC arrays must describe the same shots")
        if len(np.unique(self.shot_ids)) != len(self.shot_ids): raise ValueError("shot IDs must be unique")
        if self.physical_arm not in {"fixed_controls", "learned_controls"}: raise ValueError("unknown physical-control arm")
        if _array_hash(*arrays) != self.data_hash: raise ValueError("immutable QEC data hash mismatch")

def freeze_qec_data(path: Path, *, detection_events: np.ndarray, logical_observables: np.ndarray,
                    shot_ids: np.ndarray, epoch_ids: np.ndarray, physical_arm: str,
                    physical_policy_hash: str) -> dict:
    arrays = tuple(np.asarray(x) for x in (detection_events, logical_observables, shot_ids, epoch_ids))
    data = FrozenQecData(*arrays, physical_arm, physical_policy_hash, _array_hash(*arrays)); data.validate()
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, detection_events=arrays[0], logical_observables=arrays[1], shot_ids=arrays[2], epoch_ids=arrays[3])
    manifest = {"schema": "frozen-qec-shots.v1", "data_hash": data.data_hash, "physical_arm": physical_arm,
                "physical_policy_hash": physical_policy_hash, "shot_count": len(shot_ids),
                "contains_physical_rl_reward": False, "contains_controller_actions": False,
                "npz_sha256": sha256(path.read_bytes()).hexdigest()}
    path.with_suffix(path.suffix + ".manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True)+"\n", encoding="utf-8")
    return manifest

def load_qec_data(path: Path) -> FrozenQecData:
    manifest = json.loads(path.with_suffix(path.suffix + ".manifest.json").read_text(encoding="utf-8"))
    if sha256(path.read_bytes()).hexdigest() != manifest["npz_sha256"]: raise ValueError("QEC artifact bytes changed after freezing")
    with np.load(path, allow_pickle=False) as values:
        data = FrozenQecData(values["detection_events"], values["logical_observables"], values["shot_ids"],
            values["epoch_ids"], manifest["physical_arm"], manifest["physical_policy_hash"], manifest["data_hash"])
    data.validate(); return data

def chronological_split(data: FrozenQecData, train_epochs: set[int], evaluation_epochs: set[int]):
    data.validate()
    if train_epochs & evaluation_epochs: raise ValueError("training and evaluation epochs overlap")
    if train_epochs and evaluation_epochs and max(train_epochs) >= min(evaluation_epochs): raise ValueError("future leakage: training must precede evaluation")
    train = np.flatnonzero(np.isin(data.epoch_ids, tuple(train_epochs)))
    evaluation = np.flatnonzero(np.isin(data.epoch_ids, tuple(evaluation_epochs)))
    if train.size == 0 or evaluation.size == 0: raise ValueError("both split arms need shots")
    if np.intersect1d(data.shot_ids[train], data.shot_ids[evaluation]).size: raise ValueError("shot leakage between train and evaluation")
    return train, evaluation
