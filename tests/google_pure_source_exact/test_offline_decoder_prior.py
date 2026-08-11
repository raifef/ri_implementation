from pathlib import Path
import numpy as np
import pytest
from hdfa_rl_suite.google_pure_source_exact.offline_decoder_prior.contracts import DecoderIdentity, PriorSteeringProtocol, PYMATCHING_PROXY, SPARSE_BLOSSOM, non_equivalent_proxy_identity, public_benchmark_contract
from hdfa_rl_suite.google_pure_source_exact.offline_decoder_prior.dataset import chronological_split, freeze_qec_data, load_qec_data
from hdfa_rl_suite.google_pure_source_exact.offline_decoder_prior.factorial import FourArmResult, decompose_four_arms
from hdfa_rl_suite.google_pure_source_exact.offline_decoder_prior.prior import DemPrior, evaluate_candidates_offline

def test_immutable_shots_and_no_future_leakage(tmp_path: Path):
    path = tmp_path/"shots.npz"; events=np.arange(60,dtype=np.uint8).reshape(10,6)%2
    manifest=freeze_qec_data(path,detection_events=events,logical_observables=np.arange(10)%2,
        shot_ids=np.arange(100,110),epoch_ids=np.repeat(np.arange(5),2),physical_arm="fixed_controls",physical_policy_hash="p"*64)
    loaded=load_qec_data(path); assert loaded.data_hash==manifest["data_hash"]
    train,test=chronological_split(loaded,{0,1},{3,4}); assert not set(loaded.shot_ids[train])&set(loaded.shot_ids[test])
    with pytest.raises(ValueError,match="future leakage"): chronological_split(loaded,{3},{2})

def test_decoder_identity_fails_closed_and_benchmark_is_first():
    proxy=non_equivalent_proxy_identity("2.4",graph_hash="g",dem_hash="d",boundary_hash="b")
    assert proxy.backend==PYMATCHING_PROXY
    with pytest.raises(RuntimeError,match="Sparse Blossom"): PriorSteeringProtocol(proxy,964).validate()
    exact=DecoderIdentity(SPARSE_BLOSSOM,"verified","g","d","b","i")
    contract=public_benchmark_contract(PriorSteeringProtocol(exact,964))
    assert contract["required_sequence"][0]=="reproduce_2024_public_benchmark"
    assert not contract["live_controller_coupling"] and not contract["logical_outcomes_allowed_in_physical_reward"]

def test_prior_serialization_scoring_and_four_arm_identity(tmp_path: Path):
    prior=DemPrior(np.log(np.array([.01,.02])),"dem",PYMATCHING_PROXY)
    assert prior.save(tmp_path/"a.json")==prior.save(tmp_path/"b.json")
    scores=evaluate_candidates_offline(np.zeros((3,2)),lambda _:2,100); assert np.all(scores==scores[0])
    rates={"fixed_controls_fixed_prior":.10,"learned_controls_fixed_prior":.08,
           "fixed_controls_steered_prior":.07,"learned_controls_steered_prior":.04}
    hashes={key:("fixed" if key.startswith("fixed_controls") else "learned") for key in rates}
    result=decompose_four_arms(FourArmResult(rates,hashes,hashes))
    assert result["interaction"]==pytest.approx(-.01)
    bad=dict(hashes); bad["fixed_controls_steered_prior"]="different"
    with pytest.raises(ValueError,match="physical data changed"): decompose_four_arms(FourArmResult(rates,bad,hashes))
