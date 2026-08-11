"""Architecture-wide transactional policy lifecycle records.

The ledger is deliberately independent of Stage 5, Stage 6, Stage 7, and the
simulator.  Hardware adapters can therefore implement the same reference and
acknowledgement protocol without importing a scientific controller.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from enum import Enum
from typing import Mapping

from .records import deterministic_hash


class PolicyLifecycleState(str, Enum):
    CONFIRMED = "confirmed"
    PROPOSED = "proposed"
    PENDING_VALIDATION = "pending_validation"
    AUTHORIZED = "authorized"
    ATOMICALLY_ACTIVE = "atomically_active"
    ACKNOWLEDGED = "acknowledged"
    REJECTED = "rejected"
    ROLLED_BACK = "rolled_back"


class PolicyLifecycleError(RuntimeError):
    """Raised when a policy transaction violates reference or state semantics."""


@dataclass(frozen=True)
class PolicyCertificate:
    certificate_id: str
    certificate_type: str
    policy_hash: str
    reference_policy_id: str
    passed: bool
    details: str

    @classmethod
    def issue(cls, certificate_type: str, policy_hash: str,
              reference_policy_id: str, passed: bool, details: str) -> "PolicyCertificate":
        payload = (certificate_type, policy_hash, reference_policy_id, bool(passed), details)
        return cls(deterministic_hash(payload), certificate_type, policy_hash,
                   reference_policy_id, bool(passed), details)


@dataclass(frozen=True)
class ActivationAcknowledgement:
    acknowledgement_id: str
    policy_id: str
    policy_hash: str
    expected_activation_state_id: str
    observed_activation_state_id: str
    acknowledged_at_s: float
    atomic: bool
    accepted: bool

    @classmethod
    def create(cls, policy_id: str, policy_hash: str,
               expected_activation_state_id: str, observed_activation_state_id: str,
               acknowledged_at_s: float, *, atomic: bool, accepted: bool) -> "ActivationAcknowledgement":
        payload = (policy_id, policy_hash, expected_activation_state_id,
                   observed_activation_state_id, acknowledged_at_s, atomic, accepted)
        return cls(deterministic_hash(payload), policy_id, policy_hash,
                   expected_activation_state_id, observed_activation_state_id,
                   acknowledged_at_s, atomic, accepted)


@dataclass(frozen=True)
class PolicyTransaction:
    transaction_id: str
    policy_id: str
    policy_hash: str
    controls: Mapping[str, float]
    reference_policy_id: str
    reference_policy_hash: str
    created_from_state_id: str
    expected_activation_state_id: str
    created_at_s: float
    lifecycle_state: PolicyLifecycleState
    projection_certificate: PolicyCertificate | None = None
    bounds_certificate: PolicyCertificate | None = None
    slew_certificate: PolicyCertificate | None = None
    supervisor_authorization: str = ""
    activation_acknowledgement: ActivationAcknowledgement | None = None
    rejection_reason: str | None = None

    @property
    def certificate_hashes(self) -> Mapping[str, str]:
        return {
            "projection": self.projection_certificate.certificate_id if self.projection_certificate else "",
            "bounds": self.bounds_certificate.certificate_id if self.bounds_certificate else "",
            "slew": self.slew_certificate.certificate_id if self.slew_certificate else "",
        }


class PolicyTransactionLedger:
    """Enforce reference-stable policy transitions and named rollback versions."""

    def __init__(self, initial_policy_id: str, controls: Mapping[str, float],
                 state_id: str, timestamp_s: float = 0.0) -> None:
        policy_hash = deterministic_hash(dict(controls))
        acknowledgement = ActivationAcknowledgement.create(
            initial_policy_id, policy_hash, state_id, state_id, timestamp_s,
            atomic=True, accepted=True)
        initial = PolicyTransaction(
            deterministic_hash(("initial", initial_policy_id, policy_hash, state_id)),
            initial_policy_id, policy_hash, dict(controls), initial_policy_id,
            policy_hash, state_id, state_id, timestamp_s,
            PolicyLifecycleState.CONFIRMED,
            supervisor_authorization="bootstrap:initial-confirmed",
            activation_acknowledgement=acknowledgement,
        )
        self._transactions: dict[str, PolicyTransaction] = {initial.transaction_id: initial}
        self._by_policy_id: dict[str, str] = {initial_policy_id: initial.transaction_id}
        self._confirmed_versions: dict[str, PolicyTransaction] = {initial_policy_id: initial}
        self._confirmed_transaction_id = initial.transaction_id
        self._events: list[PolicyTransaction] = [initial]

    @property
    def confirmed(self) -> PolicyTransaction:
        return self._transactions[self._confirmed_transaction_id]

    @property
    def events(self) -> tuple[PolicyTransaction, ...]:
        return tuple(self._events)

    @property
    def state_hash(self) -> str:
        return deterministic_hash({
            "confirmed": self.confirmed.transaction_id,
            "transactions": tuple((item.transaction_id, item.lifecycle_state.value,
                                   item.policy_hash, item.reference_policy_id)
                                  for item in self._events),
        })

    def _store(self, transaction: PolicyTransaction) -> PolicyTransaction:
        self._transactions[transaction.transaction_id] = transaction
        self._by_policy_id[transaction.policy_id] = transaction.transaction_id
        self._events.append(transaction)
        return transaction

    def _latest(self, transaction_id: str) -> PolicyTransaction:
        try:
            return self._transactions[transaction_id]
        except KeyError as error:
            raise PolicyLifecycleError(f"unknown policy transaction {transaction_id}") from error

    def _require_reference(self, transaction: PolicyTransaction) -> None:
        confirmed = self.confirmed
        if (transaction.reference_policy_id != confirmed.policy_id
                or transaction.reference_policy_hash != confirmed.policy_hash):
            raise PolicyLifecycleError(
                "active reference changed; reject and reproject against the current confirmed policy")

    def propose(self, policy_id: str, controls: Mapping[str, float], *,
                reference_policy_id: str, reference_policy_hash: str,
                created_from_state_id: str, expected_activation_state_id: str,
                created_at_s: float) -> PolicyTransaction:
        if policy_id in self._by_policy_id:
            raise PolicyLifecycleError(f"policy_id {policy_id!r} is not unique")
        policy_hash = deterministic_hash(dict(controls))
        transaction = PolicyTransaction(
            deterministic_hash((policy_id, policy_hash, reference_policy_id,
                                reference_policy_hash, created_from_state_id,
                                expected_activation_state_id, created_at_s)),
            policy_id, policy_hash, dict(controls), reference_policy_id,
            reference_policy_hash, created_from_state_id,
            expected_activation_state_id, created_at_s,
            PolicyLifecycleState.PROPOSED,
        )
        self._require_reference(transaction)
        return self._store(transaction)

    def pending_validation(self, transaction_id: str, *,
                           projection: PolicyCertificate,
                           bounds: PolicyCertificate,
                           slew: PolicyCertificate) -> PolicyTransaction:
        transaction = self._latest(transaction_id)
        self._require_reference(transaction)
        if transaction.lifecycle_state is not PolicyLifecycleState.PROPOSED:
            raise PolicyLifecycleError("only a proposed policy can enter validation")
        certificates = (projection, bounds, slew)
        if any(item.policy_hash != transaction.policy_hash
               or item.reference_policy_id != transaction.reference_policy_id
               for item in certificates):
            raise PolicyLifecycleError("certificate subject/reference does not match the proposed policy")
        updated = replace(transaction, lifecycle_state=PolicyLifecycleState.PENDING_VALIDATION,
                          projection_certificate=projection,
                          bounds_certificate=bounds, slew_certificate=slew)
        return self._store(updated)

    def authorize(self, transaction_id: str, authorization_id: str) -> PolicyTransaction:
        transaction = self._latest(transaction_id)
        self._require_reference(transaction)
        if transaction.lifecycle_state is not PolicyLifecycleState.PENDING_VALIDATION:
            raise PolicyLifecycleError("only a pending-validation policy can be authorized")
        certificates = (transaction.projection_certificate,
                        transaction.bounds_certificate, transaction.slew_certificate)
        if not authorization_id or any(item is None or not item.passed for item in certificates):
            raise PolicyLifecycleError("authorization requires passing projection, bounds, and slew certificates")
        return self._store(replace(
            transaction, lifecycle_state=PolicyLifecycleState.AUTHORIZED,
            supervisor_authorization=authorization_id))

    def mark_active(self, transaction_id: str, *, reference_policy_id: str,
                    reference_policy_hash: str, atomic: bool) -> PolicyTransaction:
        transaction = self._latest(transaction_id)
        self._require_reference(transaction)
        if (reference_policy_id != transaction.reference_policy_id
                or reference_policy_hash != transaction.reference_policy_hash):
            raise PolicyLifecycleError("activation reference differs from the validated reference")
        if transaction.lifecycle_state is not PolicyLifecycleState.AUTHORIZED:
            raise PolicyLifecycleError("only an authorized policy can become active")
        if not atomic:
            raise PolicyLifecycleError("partial policy activation is forbidden")
        return self._store(replace(
            transaction, lifecycle_state=PolicyLifecycleState.ATOMICALLY_ACTIVE))

    def acknowledge(self, transaction_id: str, *, observed_policy_hash: str,
                    observed_activation_state_id: str, acknowledged_at_s: float,
                    atomic: bool = True) -> PolicyTransaction:
        transaction = self._latest(transaction_id)
        if transaction.lifecycle_state is not PolicyLifecycleState.ATOMICALLY_ACTIVE:
            raise PolicyLifecycleError("acknowledgement requires an atomically active policy")
        accepted = atomic and observed_policy_hash == transaction.policy_hash
        acknowledgement = ActivationAcknowledgement.create(
            transaction.policy_id, observed_policy_hash,
            transaction.expected_activation_state_id, observed_activation_state_id,
            acknowledged_at_s, atomic=atomic, accepted=accepted)
        if not accepted:
            raise PolicyLifecycleError("activation acknowledgement did not confirm the complete policy hash")
        acknowledged = self._store(replace(
            transaction, lifecycle_state=PolicyLifecycleState.ACKNOWLEDGED,
            activation_acknowledgement=acknowledgement))
        confirmed = self._store(replace(
            acknowledged, lifecycle_state=PolicyLifecycleState.CONFIRMED))
        self._confirmed_transaction_id = confirmed.transaction_id
        self._confirmed_versions[confirmed.policy_id] = confirmed
        return confirmed

    def reject(self, transaction_id: str, reason: str) -> PolicyTransaction:
        transaction = self._latest(transaction_id)
        return self._store(replace(
            transaction, lifecycle_state=PolicyLifecycleState.REJECTED,
            rejection_reason=reason))

    def rollback_target(self, policy_id: str) -> PolicyTransaction:
        try:
            return self._confirmed_versions[policy_id]
        except KeyError as error:
            raise PolicyLifecycleError(f"rollback target {policy_id!r} is not a named confirmed version") from error
