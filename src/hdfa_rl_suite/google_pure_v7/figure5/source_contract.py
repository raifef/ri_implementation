"""Machine-readable mapping from public Figure 5 statements to local choices."""
from __future__ import annotations
from .common import SOURCE_STATUSES, atomic_json, atomic_text, figure5_root, read_config

ARTICLE = "https://www.nature.com/articles/s41586-026-10759-2"
SUPPLEMENT = "https://media.springernature.com/original/springer-static/esm/art%3A10.1038%2Fs41586-026-10759-2/MediaObjects/41586_2026_10759_MOESM1_ESM.pdf"

def build_source_contract() -> dict:
    declared = read_config("source_contract.yaml")
    fields = [
      {"field":"5a.question","value":"When does aggregate sampled-candidate steering beat a fixed policy?","status":"EXPLICITLY_SPECIFIED","source":"main Fig. 5a; SI VI.A"},
      {"field":"5a.axes","value":{"x":"drift frequency (epochs^-1)","y":"entropy regularization","colour":"normalized stochastic-policy improvement"},"status":"EXPLICITLY_SPECIFIED","source":"main Fig. 5a; SI Fig. S8c"},
      {"field":"5a.normalization","value":"(N_stochastic-N_fixed)/(N_optimal-N_fixed)","status":"EXPLICITLY_SPECIFIED","source":"SI VI.A"},
      {"field":"5a.budget","value":{"epochs":1000,"candidates":50,"cycles_per_candidate":36000,"total_cycles":1800000000},"status":"EXPLICITLY_SPECIFIED","source":"SI VI.A"},
      {"field":"5a.frequency_anchor","value":1/150,"status":"EXPLICITLY_SPECIFIED","source":"main text and SI VI.A"},
      {"field":"5a.exact_frequency_grid","value":declared["panel_a_frequency_grid"],"status":"SYNTHETIC_REPRODUCTION_CHOICE","source":"public source does not enumerate the grid"},
      {"field":"5a.exact_entropy_grid","value":declared["panel_a_entropy_grid"],"status":"SYNTHETIC_REPRODUCTION_CHOICE","source":"the public source illustrates values but does not publish the Fig. 5a grid"},
      {"field":"5b.question","value":"Does sparse RL convergence degrade with code distance?","status":"EXPLICITLY_SPECIFIED","source":"main Fig. 5b-c; SI VI.B"},
      {"field":"5b.distances","value":[3,5,7,9,11,13,15],"status":"EXPLICITLY_SPECIFIED","source":"SI VI.B"},
      {"field":"5b.memory_cycles","value":10,"status":"EXPLICITLY_SPECIFIED","source":"SI VI.B"},
      {"field":"5b.control_count","value":"[(2d^2-1)+(4d^2-4d)]P","status":"EXPLICITLY_SPECIFIED","source":"SI Eq. 7"},
      {"field":"5b.d15_p30_controls","value":38670,"status":"DERIVED_FROM_EXPLICIT_SOURCE","source":"SI Eq. 7 and Table I"},
      {"field":"5b.irreducible_floor","value":"control-independent physical errors define Lambda*","status":"EXPLICITLY_SPECIFIED","source":"main Fig. 5b and SI VI.B"},
      {"field":"5b.exact_candidate_budget","value":None,"status":"NOT_PUBLICLY_SPECIFIED","source":"SI VI.B does not enumerate candidates/cycles/epochs for the scaling run"},
      {"field":"5b.proprietary_simulator","value":"unavailable","status":"NOT_PUBLICLY_SPECIFIED","source":"main Code availability"},
      {"field":"5c.axes","value":{"x":"1-Lambda/Lambda*","y":"1e2 d_t Lambda/Lambda*"},"status":"EXPLICITLY_SPECIFIED","source":"main Fig. 5c caption; SI Eq. 8"},
      {"field":"5c.parameters_per_gate","value":[1,10,30],"status":"EXPLICITLY_SPECIFIED","source":"main Fig. 5c"},
      {"field":"5c.fit","value":"origin-constrained local linear fit; gamma slope","status":"DERIVED_FROM_EXPLICIT_SOURCE","source":"SI Eq. 8"},
      {"field":"5c.derivative_estimator_and_fit_window","value":"finite difference with declared local window","status":"SYNTHETIC_REPRODUCTION_CHOICE","source":"public source states point estimates and linear fits but not numerical estimator details"},
      {"field":"shared.seed_cohorts","value":"repository-held independent development blocks","status":"SYNTHETIC_REPRODUCTION_CHOICE","source":"public simulation seeds are not specified"},
      {"field":"shared.synthetic_plant","value":"repository quadratic detector surrogate using resolved v7 pure controller","status":"SYNTHETIC_REPRODUCTION_CHOICE","source":"Google simulator/code and full hyperparameters are proprietary"},
    ]
    invalid=sorted({row["status"] for row in fields}-set(SOURCE_STATUSES))
    if invalid: raise RuntimeError(f"invalid source status: {invalid}")
    result={"schema_version":"google-pure-v7-figure5-source-contract.v1","article_doi":"10.1038/s41586-026-10759-2",
            "article_url":ARTICLE,"supplement_url":SUPPLEMENT,"source_data_doi":"10.5281/zenodo.17566521",
            "code_availability":"PROPRIETARY_NOT_PUBLIC","literal_reproduction_possible":False,"fields":fields}
    root=figure5_root()/"source_contract"; atomic_json(root/"figure5_source_contract.json",result)
    lines=["# Figure 5 Source Contract","","> Paper-anchored synthetic reproduction; not Google's proprietary simulator.",""]
    lines += [f"- **{r['field']}** — `{r['status']}` — {r['source']}" for r in fields]
    atomic_text(root/"figure5_source_contract.md","\n".join(lines)+"\n"); return result
