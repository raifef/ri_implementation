"""Four-arm physical/decoder factorial decomposition."""
from dataclasses import dataclass
from typing import Mapping

ARMS = ("fixed_controls_fixed_prior", "learned_controls_fixed_prior", "fixed_controls_steered_prior", "learned_controls_steered_prior")

@dataclass(frozen=True)
class FourArmResult:
    logical_error_rates: Mapping[str, float]
    physical_data_hashes: Mapping[str, str]
    shot_id_hashes: Mapping[str, str]

def decompose_four_arms(result: FourArmResult) -> dict:
    if set(result.logical_error_rates) != set(ARMS): raise ValueError("all and only the four preregistered arms are required")
    for prefix in ("fixed_controls", "learned_controls"):
        a, b = f"{prefix}_fixed_prior", f"{prefix}_steered_prior"
        if result.physical_data_hashes[a] != result.physical_data_hashes[b]: raise ValueError(f"physical data changed across decoder arms for {prefix}")
        if result.shot_id_hashes[a] != result.shot_id_hashes[b]: raise ValueError(f"shot identities changed across decoder arms for {prefix}")
    r = result.logical_error_rates
    if any(not 0 <= value <= 1 for value in r.values()): raise ValueError("logical error rates must be probabilities")
    l00, l10, l01, l11 = (r[name] for name in ARMS)
    return {"physical_contribution": l10-l00, "decoder_contribution": l01-l00, "joint_contribution": l11-l00,
            "interaction": l11-l10-l01+l00, "sign_convention": "negative is improvement", "rates": dict(r)}
