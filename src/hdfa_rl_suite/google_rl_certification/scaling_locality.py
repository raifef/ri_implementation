"""Level-7 detector-factor locality and software-work scaling."""
from __future__ import annotations

from typing import Any, Sequence

import numpy as np

from .agent import CandidateEvaluation, GaussianPolicyGradientAgent
from .config import GoogleRLConfig
from .static_detector_landscape import SparseDetectorLandscape


def _landscape(regions: int) -> SparseDetectorLandscape:
    controls = tuple(f"r{region}:u{axis}" for region in range(regions) for axis in range(2))
    detectors = tuple(f"r{region}:d{kind}" for region in range(regions) for kind in range(3))
    mask = np.zeros((len(detectors), len(controls)))
    weights = np.zeros_like(mask)
    coupling = np.zeros_like(mask)
    for region in range(regions):
        ci, di = 2*region, 3*region
        mask[di, ci] = mask[di+1, ci+1] = 1
        mask[di+2, ci:ci+2] = 1
        weights[di, ci] = weights[di+1, ci+1] = .30
        weights[di+2, ci:ci+2] = .08
        coupling[di+2, ci:ci+2] = (.7, .3)
    target = np.zeros(len(controls))
    target[:2] = (.32, -.26)
    return SparseDetectorLandscape(
        controls, detectors, mask, np.ones(len(controls)),
        np.full(len(detectors), .012), weights, coupling,
        np.asarray([0. if i % 3 != 2 else .08 for i in range(len(detectors))]),
        lambda _epoch: target.copy())


def run_scaling_locality(config: GoogleRLConfig, *, seed: int = 6601,
                         region_counts: Sequence[int] = (1, 2, 4, 8),
                         epochs: int = 32) -> dict[str, Any]:
    rows = []
    for index, regions in enumerate(region_counts):
        landscape = _landscape(int(regions))
        initial = np.zeros(len(landscape.control_ids))
        agent = GaussianPolicyGradientAgent(
            landscape.control_ids, landscape.detector_ids, landscape.mask,
            landscape.sensitivity_scales, initial, config, seed=seed+index)
        rng = np.random.default_rng(seed+90_000+index)
        initial_first = float(np.mean(
            landscape.expected_rates(initial[None, :])[0, :3]
            - landscape.irreducible_floors[:3]))
        for epoch in range(epochs):
            batch = agent.sample_candidates()
            observed = landscape.observe(
                batch.actions_native,
                config.sampling.effective_cycles_per_candidate, rng, epoch)
            agent.update(batch, tuple(CandidateEvaluation(identifier, observed[k])
                                      for k, identifier in enumerate(batch.candidate_ids)))
        final_rates = landscape.expected_rates(agent.mean_native[None, :])[0]
        first_excess = float(np.mean(final_rates[:3]-landscape.irreducible_floors[:3]))
        inactive_motion = (float(np.max(np.abs(agent.mean_native[2:])))
                           if len(agent.mean_native) > 2 else 0.)
        edges = int(np.count_nonzero(landscape.mask))
        rows.append({
            "regions": int(regions),
            "controls": len(landscape.control_ids),
            "detectors": len(landscape.detector_ids),
            "mask_edges": edges,
            "mask_density": edges/landscape.mask.size,
            "declared_work_units_per_epoch": edges*config.sampling.candidates_per_epoch,
            "initial_affected_region_excess_edr": initial_first,
            "final_affected_region_excess_edr": first_excess,
            "affected_region_remaining_fraction": first_excess/max(initial_first, 1e-15),
            "maximum_unrelated_region_motion": inactive_motion,
        })
    base_fraction = rows[0]["affected_region_remaining_fraction"]
    gates = {
        "inactive_regions_stable": max(row["maximum_unrelated_region_motion"] for row in rows) < .06,
        "affected_region_convergence_not_catastrophic": max(
            row["affected_region_remaining_fraction"] for row in rows)
        < max(.30, 3*base_fraction),
        "masks_remain_sparse": rows[-1]["mask_density"] < rows[0]["mask_density"],
        "declared_work_scales_with_graph_edges": all(
            row["declared_work_units_per_epoch"]
            == row["mask_edges"]*config.sampling.candidates_per_epoch for row in rows),
    }
    return {
        "schema_version": "google-rl-scaling-locality.v1",
        "evidence_layer": "software locality test; not hardware scalability evidence",
        "config_name": config.name,
        "gates": gates,
        "passed": all(gates.values()),
        "scaling_rows": rows,
    }
