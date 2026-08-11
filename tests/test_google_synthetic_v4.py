from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from hdfa_rl_suite.google_synthetic_v4.config import CERTIFICATION_SEEDS, load_controller_choices, load_splits, reject_certification_seed
from hdfa_rl_suite.google_synthetic_v4.controller import DetectorEvidence, MaskedGaussianPPO, clipped_objective_and_gradient, local_policy_ratios
from hdfa_rl_suite.google_synthetic_v4.plant import SyntheticPlant, frozen_specs, surface_code_control_count, surface_code_gate_count
from hdfa_rl_suite.google_synthetic_v4.studies import run_certification
from hdfa_rl_suite.google_synthetic_v4.validation import make_evidence, validate_ppo_reference


def choices(**overrides):
    value=dict(load_controller_choices()["baseline"])
    value.update(overrides)
    return value


def quadratic_evidence(batch, target=np.array([.1,-.1])):
    rates=np.clip(.05+.2*(batch.actions-target[None,:])**2,0,.9)
    return make_evidence(batch,rates,100_000)


def test_surface_code_counts_include_exact_distance_15():
    assert surface_code_gate_count(15)==1289
    assert surface_code_control_count(15)==38_670
    assert [surface_code_control_count(d) for d in (3,5,7,9,11,13,15)]==sorted(surface_code_control_count(d) for d in (3,5,7,9,11,13,15))


def test_frozen_splits_are_disjoint_and_certification_locked():
    splits=load_splits()
    names=("plant_construction","controller_development","development_validation","certification")
    ids=[set(splits[n]["plant_ids"]) for n in names]
    seeds=[set(splits[n]["evaluation_seeds"]) for n in names]
    assert all(not ids[i]&ids[j] for i in range(4) for j in range(i+1,4))
    assert all(not seeds[i]&seeds[j] for i in range(4) for j in range(i+1,4))
    assert tuple(splits["certification"]["evaluation_seeds"])==CERTIFICATION_SEEDS
    with pytest.raises(ValueError,match="forbidden"):
        reject_certification_seed(8101)


def test_certification_physical_draws_not_seed_only_clones():
    cert=[s for s in frozen_specs() if s.split=="certification"]
    signatures={(s.drift_phase,s.drift_amplitude,s.drift_frequency,s.graph_offset,s.curvature_mean,s.detector_covariance,s.spoil_severity,s.coupling_pattern) for s in cert}
    assert len(signatures)==len(cert)==12


def test_controller_view_is_truth_isolated_and_sensitivity_round_trips():
    plant=SyntheticPlant(next(s for s in frozen_specs() if s.split=="controller_development"))
    view=plant.controller_view()
    assert "base_optimum" not in view and "drift_phase" not in view
    normalized=np.linspace(-.2,.2,plant.spec.control_count)
    native=normalized*view["sensitivity"]
    assert np.allclose(native/view["sensitivity"],normalized)


def test_independent_reference_and_finite_difference_validation_passes():
    report=validate_ppo_reference()
    assert report["status"]=="PASS"
    assert all(report["checks"].values())
    assert report["maximum_absolute_errors"]["finite_difference_mean"]<2e-8


def test_mask_is_local_and_inactive_control_gradient_is_zero():
    actions=np.array([[.1,.2,.3],[-.2,.1,-.1]])
    advantages=np.array([[1.],[-.5]])
    mask=np.array([[1,0,0]],dtype=bool)
    _,gm,gs,_=clipped_objective_and_gradient(actions,advantages,mask,np.zeros(3),np.zeros(3),np.zeros(3),np.zeros(3),clip=.2)
    assert np.all(gm[1:]==0) and np.all(gs[1:]==0)
    ratios=local_policy_ratios(actions,np.array([.1,9.,9.]),np.zeros(3),np.zeros(3),np.zeros(3),mask)
    reference=local_policy_ratios(actions,np.array([.1,-9.,-9.]),np.zeros(3),np.zeros(3),np.zeros(3),mask)
    assert np.allclose(ratios,reference)


def test_multiple_passes_make_off_policy_ratios_nontrivial():
    agent=MaskedGaussianPPO(np.eye(2,dtype=bool),np.array([.4,-.4]),choices(optimizer_passes=5,replay_epochs=0,entropy_weight=0.0),seed=1)
    batch=agent.sample(40,regime_id="same")
    diag=agent.update(batch,quadratic_evidence(batch))
    assert diag["off_policy_fraction"]>0


def test_compatible_replay_used_and_incompatible_replay_rejected():
    agent=MaskedGaussianPPO(np.eye(2,dtype=bool),np.array([.4,-.4]),choices(replay_epochs=2),seed=2)
    first=agent.sample(20,regime_id="old")
    agent.update(first,quadratic_evidence(first))
    second=agent.sample(20,regime_id="new")
    diag=agent.update(second,quadratic_evidence(second))
    assert diag["replay_epochs_used"]==0 and diag["incompatible_replay_rejected"]==1
    third=agent.sample(20,regime_id="new")
    diag=agent.update(third,quadratic_evidence(third))
    assert diag["replay_epochs_used"]==1


def test_candidate_provenance_and_one_use_semantics_are_fail_closed():
    agent=MaskedGaussianPPO(np.eye(2,dtype=bool),np.array([.2,-.2]),choices(),seed=3)
    batch=agent.sample(10,regime_id="r")
    evidence=list(quadratic_evidence(batch))
    evidence[0]=replace(evidence[0],action_hash="wrong")
    with pytest.raises(ValueError,match="provenance"):
        agent.update(batch,evidence)
    agent.update(batch,quadratic_evidence(batch))
    with pytest.raises(ValueError,match="stale|consumed"):
        agent.update(batch,quadratic_evidence(batch))


def test_mean_and_scale_updates_can_be_decoupled():
    mean_only=MaskedGaussianPPO(np.eye(2,dtype=bool),np.array([.4,-.4]),choices(scale_learning_rate=0.0,entropy_weight=0.0,replay_epochs=0),seed=4)
    batch=mean_only.sample(40,regime_id="r")
    old_mean,old_scale=mean_only.mean.copy(),mean_only.log_std.copy()
    mean_only.update(batch,quadratic_evidence(batch))
    assert not np.allclose(mean_only.mean,old_mean) and np.array_equal(mean_only.log_std,old_scale)
    scale_only=MaskedGaussianPPO(np.eye(2,dtype=bool),np.array([.4,-.4]),choices(mean_learning_rate=0.0,scale_learning_rate=.1,entropy_weight=.1,replay_epochs=0),seed=4)
    batch=scale_only.sample(40,regime_id="r")
    old_mean,old_scale=scale_only.mean.copy(),scale_only.log_std.copy()
    scale_only.update(batch,quadratic_evidence(batch))
    assert np.array_equal(scale_only.mean,old_mean) and not np.allclose(scale_only.log_std,old_scale)


def test_certification_command_requires_explicit_seed_opening():
    with pytest.raises(RuntimeError,match="locked"):
        run_certification(confirm=False,epochs=8)
