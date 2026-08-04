"""Disjoint development seed cohorts; certification seeds remain untouched."""
from __future__ import annotations

from itertools import combinations

from google_rl_reimplementation.google_pure_v7 import ACTIVE_CERTIFICATION_SEEDS, RETIRED_SEEDS
from .common import atomic_json, atomic_text, figure5_root

SEEDS = {
    "5a": {"smoke": (51001, 51002), "validation": (51101, 51102, 51103, 51104),
           "reference": tuple(range(51201, 51209)), "paper-scale": tuple(range(51301, 51313))},
    "5b": {"smoke": (52001, 52002), "validation": (52101, 52102, 52103, 52104),
           "reference": tuple(range(52201, 52209)), "paper-scale": tuple(range(52301, 52313))},
    "5c": {"smoke": (53001, 53002), "validation": (53101, 53102, 53103, 53104),
           "reference": tuple(range(53201, 53209)), "paper-scale": tuple(range(53301, 53313))},
}
BLACKLIST = frozenset((*ACTIVE_CERTIFICATION_SEEDS, *RETIRED_SEEDS, 7, 19, 43, 71, 101, 6201))


def validate_registry() -> dict:
    named = {(panel, mode): set(values) for panel, modes in SEEDS.items() for mode, values in modes.items()}
    overlap = []
    for (left, a), (right, b) in combinations(named.items(), 2):
        if a & b: overlap.append({"left": left, "right": right, "seeds": sorted(a & b)})
    forbidden = sorted(set().union(*named.values()) & BLACKLIST)
    if overlap or forbidden: raise RuntimeError(f"invalid Figure 5 seed registry: overlap={overlap}, forbidden={forbidden}")
    return {"schema_version": "google-pure-v7-figure5-seeds.v1", "seeds": SEEDS,
            "blacklist": sorted(BLACKLIST), "overlaps": overlap, "forbidden_used": forbidden,
            "certification_seeds_consumed": False, "status": "PASS"}


def write_registry() -> dict:
    result = validate_registry(); root = figure5_root() / "protocol_freezes"
    atomic_json(root / "seed_registry.json", result)
    lines = ["# Figure 5 Seed Registry", "", "Certification seeds are excluded and unconsumed.", ""]
    for panel, modes in SEEDS.items():
        for mode, seeds in modes.items(): lines.append(f"- {panel} / {mode}: {', '.join(map(str, seeds))}")
    atomic_text(root / "seed_registry.md", "\n".join(lines) + "\n")
    return result

