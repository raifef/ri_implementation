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
- The older `FullControlDetectorRL` is an antithetic finite-difference controller sharing the Stage-6 residual core. It is retained as a legacy repository approximation and is not the high-shot public-structure implementation.
"""


def current_implementation_audit() -> dict[str, Any]:
    return {
        "schema_version": "google-rl-current-implementation-audit.v1",
        "evidence_layer": "static source audit plus deterministic regression tests",
        "scope": [
            "stage6/residual_rl.py", "baselines/controllers.py", "simulator/device.py",
            "common/policy_lifecycle.py", "validation/controller_sanity.py",
            "validation/sample_budget.py", "evaluation reporting contracts"],
        "trace": {
            "policy_parameterization": "legacy GaussianResidualPolicy; diagonal sparse identity initially, optional graph-local covariance",
            "learned_mean_and_covariance": "relative mean, per-control stddev and optional sparse covariance",
            "candidate_generation": "complete antithetic pairs around one immutable epoch mean; not the high-shot independent-Gaussian reference",
            "detector_control_masking": "detector-to-control graph; each update averages only linked detector losses",
            "sensitivity_normalization": "global normalized simulator units only; public per-control-type sensitivity calibration absent from the legacy arm",
            "reward": "detector rates plus observable logical/leakage/correlation penalties; differs from the public detector-only reward",
            "detector_vector_baseline": "absent in legacy core",
            "policy_gradient_loss": "absent; legacy core uses antithetic finite differences",
            "baseline_loss": "absent in legacy core",
            "entropy_regularization": "hard stddev floor, not the public entropy loss",
            "gradient_clipping": "mean update is bounded; no declared public-style gradient clip",
            "optimizer": "direct fixed-rate update; no disclosed optimizer abstraction",
            "replay": "compatible items are retained and counted but not applied to the legacy policy update",
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
                "regression_test": "test_legacy_full_control_is_not_labelled_faithful",
            },
            {
                "id": "GRC-002", "severity": "high", "status": "corrected",
                "description": "Over-age candidate observations entered the pair map before the age check and could affect the gradient.",
                "correction": "Age rejection now precedes pair association and replay; physical damage remains accounted.",
                "regression_test": "test_stale_candidate_rewards_do_not_update_legacy_policy",
            },
            {
                "id": "GRC-003", "severity": "high", "status": "corrected",
                "description": "Odd or too-small candidate counts were silently rounded, changing the declared protocol.",
                "correction": "Full-control candidate count is now validated as even and at least four.",
                "regression_test": "test_legacy_candidate_count_is_not_silently_changed",
            },
            {
                "id": "GRC-004", "severity": "scientific-invalidity", "status": "corrected",
                "description": "The 2,048-cycle path was labelled validated without full matched high-shot behavior tests.",
                "correction": "Runtime artifacts call it a reduced-budget candidate; equivalence is decided only by Track A.",
                "regression_test": "test_reduced_budget_label_requires_track_a",
            },
        ],
        "tested_failure_modes": [
            "reversed reward/loss sign", "stale rewards", "shuffled reward order",
            "unknown/duplicate candidates", "transposed masks", "incorrect sensitivity shape/units",
            "cumulative rather than centred perturbations", "covariance floor/ceiling",
            "policy sharing", "wrong policy version", "hidden truth access",
            "detector-region and control-index mismatch"],
        "retained_comparison_evidence": [
            {
                "artifact": "experiments/physical_validation/authoritative-comparison-v1.json",
                "finding": "launch manifest only; it declared 2,048 candidate cycles and contained no Track-A high-shot certification result",
            },
            {
                "artifact": "artifacts/acceptance/compute-aware-v2/authoritative-comparison-v2.json",
                "finding": "2,048-cycle configuration; authoritative=false and accepted=false because required metrics were not evaluable after a physical rollback-validation failure",
            },
            {
                "artifact": "artifacts/comparison/nature-2026-v5/authoritative-effectiveness.json",
                "finding": "32-cycle configuration; authoritative=true but accepted=false, with every effectiveness gate failed and a lifecycle/non-completion failure",
            },
        ],
        "disposition": "legacy approximation retained; high-shot reference implemented separately",
    }


def audit_markdown(audit: dict[str, Any]) -> str:
    lines = [
        "# Current Google-style RL implementation audit", "",
        "The pre-Track-A full-control arm is a legacy antithetic finite-difference approximation, not the public policy-gradient algorithm. The staged controller was not changed.", "",
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
