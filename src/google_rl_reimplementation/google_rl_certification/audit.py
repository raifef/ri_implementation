"""Source extraction and end-to-end audit of the pre-Track-A baseline."""
from __future__ import annotations

from pathlib import Path
from typing import Any

from .common import write_json
from .config import repository_root


SOURCE_EXTRACTION = """# Google detector-RL public source extraction

## Evidence boundary

This extraction uses the Nature article, its Supplementary Information, the local QGSS-style tutorial archive, and the repository architecture documents. It is not a Willow reproduction. The publisher states that the custom experimental code is proprietary.

Primary sources: [Nature article](https://www.nature.com/articles/s41586-026-10759-2), [Supplementary Information](https://media.springernature.com/original/springer-static/esm/art%3A10.1038%2Fs41586-026-10759-2/MediaObjects/41586_2026_10759_MOESM1_ESM.pdf), DOI `10.1038/s41586-026-10759-2`.

## 1. Explicitly public algorithmic details

- A typical surface-code training epoch samples 40 policy candidates. Each candidate uses 4,000 shots of a 25-cycle memory circuit, or 100,000 effective QEC cycles. These settings are described as typical rather than rigid; the colour-code experiment used 15-cycle circuits.
- Every five training epochs, the experiment evaluates the learned mean and fixed initial policies with 200,000 shots each.
- The policy is a factorized multivariate Gaussian with learned mean and diagonal standard deviation/covariance.
- The reward vector is minus the per-detector error-detection-rate vector. Time-translation-equivalent detectors are combined.
- A sparse detector-control masking matrix is constructed through detecting regions and their gates. Sensitivity calibration rescales native control types so canonical Gaussian perturbations have comparable detector impact.
- The mathematical update is a parameter-exploring policy gradient with a detector-vector baseline, local importance ratios, elementwise PPO-style ratio clipping, baseline loss, entropy regularization, gradient clipping, and an optimizer step.
- The old policy from the most recent epoch is the collection policy. Several epochs may enter replay, but the supplement reports mixed success under non-stationarity.
- The public step-response anchor is about 130 epochs. Deliberately randomized controls recovered in about 1,000 epochs.
- The distance-3 steering simulation used 50 candidates, 36,000 cycles per candidate, 1,000 epochs, and 1.8 billion cycles. Learned-mean and stochastic/exploratory policies were evaluated separately. The reported stochastic-policy steerability transition is near 1/150 epochs.
- The public simulation plant uses an irreducible gate-error floor plus a quadratic penalty in control mismatch, with randomly chosen sensitivities.

## 2. Quantities inferred from public descriptions

- Independent Gaussian candidate draws are the natural reading of sampling a batch from the factorized policy, but the sources do not explicitly specify cross-candidate random-number construction. The high-shot repository reference therefore labels independent draws as an inference, not a disclosed controller detail.
- A per-detector moving baseline is a practical realization of the published learned baseline vector; its exact update rule is unavailable.
- The public source supports response-scale and steering-frequency anchors, not exact agreement on a different surrogate plant.

## 3. Proprietary or unavailable details

- Exact optimizer, learning rates, loss weights, gradient-clip magnitude, PPO clip, replay length/use schedule, initialization, entropy schedule, and stopping criteria.
- Numerical sensitivity coefficients, hardware detector-control graph, pulse compiler behavior, controller upload/acknowledgement path, safety projection, and orchestration latency.
- The Willow plant, transfer functions, uncontrolled device drift realization, hidden hardware state, and production implementation.
- Custom experimental code and the real-time closed-loop control stack.

## 4. Repository-specific approximations

- The high-shot reference uses dependency-free SGD, explicit configuration values, a moving detector baseline, one-batch on-policy updates, and bounded sensitivity-normalized controls.
- Hyperparameters were adjusted only on development seed 15501 to reproduce the qualitative public steering transition; certification uses different seeds.
- Certification plants are declared analytic or detector-likelihood surrogates with binomial finite-shot observations. They are not circuit/pulse-level Willow models.
- The reduced candidate uses 2,048 cycles and complete antithetic pairs. It remains experimental unless the matched Track-A equivalence artifact passes.
- Earlier reduced-budget antithetic variants remain historical development protocols; the high-shot public-structure implementation is the authoritative reference in this workflow.
"""


def current_implementation_audit() -> dict[str, Any]:
    return {
        "schema_version": "google-rl-current-implementation-audit.v1",
        "evidence_layer": "static source audit plus deterministic regression tests",
        "scope": [
            "google_rl_certification/agent.py", "google_rl_certification/config.py",
            "google_rl_certification/analytic_landscape.py", "google_rl_certification/static_detector_landscape.py",
            "google_rl_certification/drift_tracking.py", "google_rl_certification/sample_budget_equivalence.py"
        ],
        "trace": {
            "policy_parameterization": "factorized Gaussian with learned mean and diagonal standard deviation",
            "learned_mean_and_covariance": "relative mean, per-control stddev and optional sparse covariance",
            "candidate_generation": "40 independent Gaussian candidates for the high-shot reference; exact antithetic pairs only for the reduced-budget candidate",
            "detector_control_masking": "detector-to-control graph; each update averages only linked detector losses",
            "sensitivity_normalization": "explicit positive per-control sensitivity scales with shape and unit checks",
            "reward": "minus the local detector error-detection-rate vector",
            "detector_vector_baseline": "moving per-detector baseline",
            "policy_gradient_loss": "local importance-ratio PPO objective with elementwise clipping",
            "baseline_loss": "explicit detector-vector baseline update",
            "entropy_regularization": "explicit entropy term plus bounded standard deviation",
            "gradient_clipping": "mean update is bounded; no declared public-style gradient clip",
            "optimizer": "declared dependency-free SGD approximation because the proprietary optimizer is unavailable",
            "replay": "versioned current-policy batches; reduced replay claims remain fail-closed",
            "candidate_reward_association": "candidate IDs, policy versions, requested/activated hashes and immutable epoch reference",
            "policy_update_timing": "after a complete candidate batch; mean committed from the epoch reference",
            "policy_lifecycle": "confirmed -> proposed -> pending_validation -> authorized -> atomically_active -> acknowledged -> confirmed",
            "mean_policy_evaluation": "independent post-update device acquisition",
            "exploratory_policy_evaluation": "per-candidate trajectories and aggregate detector rate retained separately",
            "accounting": "candidate evaluations, candidate cycles, evaluation cycles, detector counts and exploration excess are separate fields",
            "truth_access": "typed oracle/evaluation capability; ordinary controller batches contain observations and provenance only",
        },
        "defects": [
            {
                "id": "GRC-001", "severity": "scientific-invalidity", "status": "corrected",
                "description": "Legacy antithetic finite differences were described as a faithful public policy-gradient reproduction.",
                "correction": "Dedicated public-structure agent added; legacy arm relabelled as an approximation.",
                "regression_test": "test_public_high_shot_sampling_structure_is_exact_and_versioned",
            },
            {
                "id": "GRC-002", "severity": "high", "status": "corrected",
                "description": "Over-age candidate observations entered the pair map before the age check and could affect the gradient.",
                "correction": "Age rejection now precedes pair association and replay; physical damage remains accounted.",
                "regression_test": "test_unknown_duplicate_and_stale_candidate_rewards_are_rejected",
            },
            {
                "id": "GRC-004", "severity": "scientific-invalidity", "status": "corrected",
                "description": "The 2,048-cycle path was labelled validated without full matched high-shot behavior tests.",
                "correction": "Runtime artifacts call it a reduced-budget candidate; equivalence is decided only by Track A.",
                "regression_test": "test_high_shot_is_independent_and_reduced_pairs_are_exactly_centred",
            },
        ],
        "tested_failure_modes": [
            "reversed reward/loss sign", "stale rewards", "shuffled reward order",
            "unknown/duplicate candidates", "transposed masks", "incorrect sensitivity shape/units",
            "cumulative rather than centred perturbations", "covariance floor/ceiling",
            "policy sharing", "wrong policy version", "hidden truth access",
            "detector-region and control-index mismatch"],
        "retained_comparison_evidence": [],
        "disposition": "standalone high-shot reference and reduced-budget candidate retained with distinct evidence labels",
    }


def audit_markdown(audit: dict[str, Any]) -> str:
    lines = [
        "# Current Google-style RL implementation audit", "",
        "This audit covers only the standalone public-structure policy-gradient implementation and its declared surrogate evidence.", "",
        "## End-to-end trace", "",
    ]
    lines.extend(f"- **{key.replace('_', ' ')}:** {value}"
                 for key, value in audit["trace"].items())
    lines.extend(["", "## Defects and regression coverage", "",
                  "| ID | Severity | Status | Finding | Regression test |",
                  "| --- | --- | --- | --- | --- |"])
    for item in audit["defects"]:
        lines.append(f"| {item['id']} | {item['severity']} | {item['status']} | {item['description']} | `{item['regression_test']}` |")
    lines.extend(["", "## Retained comparison evidence", ""])
    for item in audit["retained_comparison_evidence"]:
        lines.append(f"- `{item['artifact']}`: {item['finding']}")
    lines.extend(["", "## Disposition", "", audit["disposition"], ""])
    return "\n".join(lines)


def write_audit_artifacts(output: Path | None = None) -> dict[str, Any]:
    root = repository_root()
    destination = output or root / "artifacts" / "google_rl_certification"
    destination.mkdir(parents=True, exist_ok=True)
    (destination / "source_extraction.md").write_text(SOURCE_EXTRACTION, encoding="utf-8")
    audit = current_implementation_audit()
    write_json(destination / "current_implementation_audit.json", audit)
    (destination / "current_implementation_audit.md").write_text(
        audit_markdown(audit), encoding="utf-8")
    return audit
