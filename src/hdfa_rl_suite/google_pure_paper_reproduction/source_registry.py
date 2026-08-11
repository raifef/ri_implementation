"""Primary-source registry and figure/table source contract."""
from __future__ import annotations

from typing import Any

from .storage import atomic_json, atomic_text, initialise_layout

ARTICLE = "https://www.nature.com/articles/s41586-026-10759-2"
SUPPLEMENT = "https://media.springernature.com/original/springer-static/esm/art%3A10.1038%2Fs41586-026-10759-2/MediaObjects/41586_2026_10759_MOESM1_ESM.pdf"
DATA_CONCEPT_DOI = "10.5281/zenodo.17566521"
DATA_RELEASE_DOI = "10.5281/zenodo.18896801"


def source_contract() -> dict[str, Any]:
    sources = {
        "article": {"url": ARTICLE, "doi": "10.1038/s41586-026-10759-2", "type": "peer_reviewed_article"},
        "supplement": {"url": SUPPLEMENT, "type": "publisher_supplementary_information"},
        "public_data_concept": {"doi": DATA_CONCEPT_DOI, "type": "zenodo_concept_record"},
        "public_data_release": {
            "doi": DATA_RELEASE_DOI, "record_id": "18896801", "version": "2.0.0",
            "archive": "google_reinforcement_learning_qec.zip", "bytes": 7_786_791_716,
            "sha256": "39563ad104bcbec2e36907373b25d176cf7f2a2e3852d8390623223dadf96e76",
        },
    }
    figures = {
        "5a": {
            "source": ["article:Fig.5a", "supplement:VI.A", "supplement:VIII"],
            "public": {"distance": 3, "epochs": 1000, "candidates_per_epoch": 50, "cycles_per_candidate": 36000,
                       "candidate_cycles": 1_800_000_000, "critical_frequency_epochs_inverse": 1/150,
                       "normalization": "(C_fixed-C_candidates)/(C_fixed-C_optimal)"},
            "not_public": ["simulator implementation", "full sweep coordinates", "all random seeds", "exact plotting interpolation"],
            "reproduction_type": "paper_anchored_independent_synthetic",
        },
        "5b": {
            "source": ["article:Fig.5b", "supplement:VI.B", "supplement:Table I"],
            "public": {"distances": [3,5,7,9,11,13,15], "parameters_per_gate": [1,10,30],
                       "distance_15_p30_controls": 38670, "threshold_physical_error": 0.00179},
            "not_public": ["proprietary simulator", "exact per-epoch stochastic samples"],
            "reproduction_type": "paper_anchored_independent_sparse_surrogate",
        },
        "5c": {
            "source": ["article:Fig.5c", "supplement:VI.B"],
            "public": {"x_axis": "1-Lambda/Lambda*", "y_axis": "1e2 d_t Lambda/Lambda*",
                       "parameters_per_gate": [1,10,30], "claim": "approximately distance-independent local convergence"},
            "not_public": ["numeric fit coefficients", "raw source curves"],
            "reproduction_type": "paper_anchored_independent_synthetic",
        },
    }
    families = {
        "natural_drift": {"source": ["article:Fig.3a/4", "supplement:III"], "anchor": "about 4 dB low-frequency LER suppression", "released_epoch_trace": False},
        "randomized_recovery": {"source": ["supplement:IV"], "anchor": "roughly 1000 epochs", "released_training_trace": False},
        "step_response": {"source": ["article:Fig.4b"], "anchor": "about 130 epochs", "released_training_trace": False},
        "public_endpoints": {"source": ["article:Fig.3", "public_data_release"], "released_static_memory_shots": True},
    }
    result = {
        "schema_version": "google-paper-source-contract.v1", "paper": "Reinforcement learning control of quantum error correction",
        "published": "2026-07-08", "sources": sources, "figures": figures, "experiment_families": families,
        "code_availability": "Google custom code is proprietary; SI VIII provides the mathematical algorithm description.",
        "anti_conflation": {
            "public_data_is_hardware_output_not_simulator_source": True,
            "synthetic_panels_are_not_exact_google_code_reproductions": True,
            "control_only_2_4x_never_merged_with_control_plus_decoder_3_5x": True,
            "plot_similarity_is_not_quantitative_reproduction": True,
        },
    }
    return result


def build_source_contract() -> dict[str, Any]:
    result = source_contract(); root = initialise_layout() / "source_contract"
    atomic_json(root / "source_contract.json", result)
    lines = ["# Public source contract", "", "> The simulation code is proprietary. Figure 5 outputs in this repository are independent, paper-anchored synthetic reproductions.", "", "## Primary sources", ""]
    lines += [f"- `{key}`: {value.get('url', value.get('doi'))}" for key, value in result["sources"].items()]
    lines += ["", "## Anti-conflation gates", ""] + [f"- {key}: **{value}**" for key, value in result["anti_conflation"].items()]
    atomic_text(root / "source_contract.md", "\n".join(lines) + "\n")
    return result

