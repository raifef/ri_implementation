"""Deterministic and randomized policy transaction validation."""
from __future__ import annotations

from dataclasses import replace
import random
from typing import Iterable

from hdfa_rl_suite.common import (
    PolicyCertificate, PolicyLifecycleError, PolicyLifecycleState,
    PolicyTransactionLedger, deterministic_hash,
)

from .common import ValidationCheck, ValidationReport, all_passed, finalize_report


def _certificates(policy_hash: str, reference_policy_id: str, *,
                  projection: bool = True, bounds: bool = True,
                  slew: bool = True):
    return (
        PolicyCertificate.issue("projection", policy_hash, reference_policy_id,
                                projection, "projection certificate"),
        PolicyCertificate.issue("bounds", policy_hash, reference_policy_id,
                                bounds, "bounds certificate"),
        PolicyCertificate.issue("slew", policy_hash, reference_policy_id,
                                slew, "slew certificate"),
    )


def _prepare(ledger: PolicyTransactionLedger, policy_id: str, value: float,
             state_id: str, timestamp_s: float):
    reference = ledger.confirmed
    proposal = ledger.propose(
        policy_id, {"u": value}, reference_policy_id=reference.policy_id,
        reference_policy_hash=reference.policy_hash,
        created_from_state_id=state_id,
        expected_activation_state_id=deterministic_hash((state_id, policy_id)),
        created_at_s=timestamp_s)
    projection, bounds, slew = _certificates(
        proposal.policy_hash, proposal.reference_policy_id)
    pending = ledger.pending_validation(
        proposal.transaction_id, projection=projection, bounds=bounds, slew=slew)
    return ledger.authorize(pending.transaction_id, f"stage7:{policy_id}")


def run_lifecycle_validation(*, injected_faults: Iterable[str] = ()) -> ValidationReport:
    faults = set(injected_faults)
    checks: list[ValidationCheck] = []

    ledger = PolicyTransactionLedger("confirmed:0", {"u": 0.0}, "state:0")
    mpc = _prepare(ledger, "mpc:1", .1, "state:1", 1.0)
    confirmed_before_ack = ledger.confirmed.policy_id == "confirmed:0"
    active = ledger.mark_active(
        mpc.transaction_id, reference_policy_id=mpc.reference_policy_id,
        reference_policy_hash=mpc.reference_policy_hash, atomic=True)
    acknowledged = ledger.acknowledge(
        active.transaction_id, observed_policy_hash=active.policy_hash,
        observed_activation_state_id=active.expected_activation_state_id,
        acknowledged_at_s=1.1)
    delayed_ok = (confirmed_before_ack
                  and acknowledged.lifecycle_state is PolicyLifecycleState.CONFIRMED
                  and ledger.confirmed.policy_id == "mpc:1")
    checks.append(ValidationCheck(
        "delayed_acknowledgement_transaction", delayed_ok,
        {"confirmed_before_ack": confirmed_before_ack,
         "confirmed_after_ack": ledger.confirmed.policy_id},
        "a proposal becomes confirmed only after atomic activation and matching acknowledgement",
        "Delayed acknowledgements cannot silently promote a pending action."))

    # Permanent regression: a probe projected from the old confirmed policy while an
    # MPC action is pending must be rejected after MPC acknowledgement, then reprojected.
    race = PolicyTransactionLedger("confirmed:0", {"u": 0.0}, "state:0")
    pending_mpc = _prepare(race, "mpc:pending", .1, "state:mpc", 1.0)
    stale_probe = _prepare(race, "probe:stale", -.05, "state:probe", 1.01)
    active_mpc = race.mark_active(
        pending_mpc.transaction_id,
        reference_policy_id=pending_mpc.reference_policy_id,
        reference_policy_hash=pending_mpc.reference_policy_hash, atomic=True)
    race.acknowledge(
        active_mpc.transaction_id, observed_policy_hash=active_mpc.policy_hash,
        observed_activation_state_id=active_mpc.expected_activation_state_id,
        acknowledged_at_s=1.1)
    stale_rejected = False
    try:
        race.mark_active(
            stale_probe.transaction_id,
            reference_policy_id=stale_probe.reference_policy_id,
            reference_policy_hash=stale_probe.reference_policy_hash, atomic=True)
    except PolicyLifecycleError:
        stale_rejected = True
    if "wrong_policy_activation_reference" in faults:
        stale_rejected = False
    reprojected = _prepare(race, "probe:reprojected", .05, "state:probe:2", 1.2)
    active_probe = race.mark_active(
        reprojected.transaction_id,
        reference_policy_id=reprojected.reference_policy_id,
        reference_policy_hash=reprojected.reference_policy_hash, atomic=True)
    race.acknowledge(
        active_probe.transaction_id, observed_policy_hash=active_probe.policy_hash,
        observed_activation_state_id=active_probe.expected_activation_state_id,
        acknowledged_at_s=1.3)
    checks.append(ValidationCheck(
        "pending_mpc_probe_reference_race", stale_rejected and race.confirmed.policy_id == "probe:reprojected",
        {"stale_probe_rejected": stale_rejected,
         "confirmed": race.confirmed.policy_id},
        "a reference change rejects a pending intervention and requires reprojection",
        "Projection and validation use the policy that will actually be active."))

    rng = random.Random(271828)
    randomized = PolicyTransactionLedger("bootstrap:0", {"u": 0.0}, "state:0")
    accepted = rejected = rollbacks = 0
    invariant_ok = True
    for index in range(64):
        prior = randomized.confirmed
        candidate = max(-.8, min(.8, float(prior.controls["u"]) + rng.uniform(-.08, .08)))
        transaction = _prepare(randomized, f"random:{index}", candidate,
                               f"state:{index}", float(index+1))
        if index % 11 == 0 and index:
            try:
                randomized.mark_active(
                    transaction.transaction_id, reference_policy_id="stale",
                    reference_policy_hash=transaction.reference_policy_hash, atomic=True)
                invariant_ok = False
            except PolicyLifecycleError:
                rejected += 1
            continue
        active_transaction = randomized.mark_active(
            transaction.transaction_id,
            reference_policy_id=transaction.reference_policy_id,
            reference_policy_hash=transaction.reference_policy_hash, atomic=True)
        randomized.acknowledge(
            active_transaction.transaction_id,
            observed_policy_hash=active_transaction.policy_hash,
            observed_activation_state_id=active_transaction.expected_activation_state_id,
            acknowledged_at_s=index+1.1)
        accepted += 1
        invariant_ok &= randomized.confirmed.policy_id == transaction.policy_id
        if index % 17 == 0:
            invariant_ok &= randomized.rollback_target("bootstrap:0").policy_id == "bootstrap:0"
            rollbacks += 1
    checks.append(ValidationCheck(
        "randomized_lifecycle_state_machine", invariant_ok and accepted > 50 and rejected > 0,
        {"accepted": accepted, "rejected_stale": rejected,
         "named_rollback_checks": rollbacks},
        "randomized delayed/reference/rollback transitions preserve the transactional invariant",
        "Concurrent proposals, re-entry-like version changes, and named rollback are fail-closed."))

    return finalize_report(ValidationReport(
        "policy-lifecycle-validation.v1", "policy_lifecycle_validation",
        all_passed(checks), tuple(checks), (),
        {"seed": 271828, "injected_faults": sorted(faults),
         "evidence_layer": "executed repository transaction-state tests"},
    ))
