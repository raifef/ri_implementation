"""Figure 5b/5c raw-data lineage and derivative validation."""
from __future__ import annotations

from typing import Any

import numpy as np

from .contracts import NONFINAL_FIELDS, V12_SCHEMA
from .io import ARTIFACT_ROOT, ROOT, atomic_json, atomic_text, canonical_hash, read_json

FAMILY_SLUG = {
    "FIGURE5B_SPARSE_SCALING": "fig5b",
    "FIGURE5C_CONVERGENCE_LAW": "fig5c",
}


def _load_merged(family: str) -> tuple[dict[str, Any], dict[str, Any]]:
    stem = family.lower()
    protocol = read_json(ROOT / f"artifacts/google_pure_paper_reproduction/experiment_protocols/{stem}_validation.json")
    merged_path = (ROOT / "artifacts/google_pure_paper_reproduction/synthetic_reproduction" /
                   FAMILY_SLUG[family] / protocol["protocol_hash"][:16] / "merged.json")
    merged = read_json(merged_path)
    if merged["protocol_hash"] != protocol["protocol_hash"] or not merged.get("complete"):
        raise RuntimeError(f"{family} merged lineage is stale or incomplete")
    return protocol, merged


def audit_figure5b_lineage() -> dict[str, Any]:
    protocol, merged = _load_merged("FIGURE5B_SPARSE_SCALING")
    table, condition_rows = [], []
    for row in merged["rows"]:
        trajectory = row["trajectory"]
        fields = ("epoch", "physical_error", "logical_learned", "logical_candidate",
                  "logical_fixed", "logical_oracle", "logical_floor")
        lengths = {key: len(trajectory[key]) for key in fields}
        aligned = len(set(lengths.values())) == 1
        if not aligned:
            raise RuntimeError(f"Figure5b trajectory fields are misaligned: {lengths}")
        physical = np.asarray(trajectory["physical_error"], dtype=float)
        logical = np.asarray(trajectory["logical_learned"], dtype=float)
        floor = float(row["logical_floor"])
        initial_excess = max(float(logical[0] - floor), np.finfo(float).tiny)
        progress = float((logical[0] - logical[-1]) / initial_excess)
        identity = {"distance": row["distance"], "parameters_per_gate": row["parameters_per_gate"],
                    "seed": row["seed"], "plant_hash": row["plant_instance_hash"],
                    "graph_hash": row["graph_instance_hash"]}
        condition_rows.append({**identity, "epochs": len(logical), "fields_aligned": aligned,
                               "physical_span": float(np.ptp(physical)), "logical_span": float(np.ptp(logical)),
                               "floor_normalized_progress": progress,
                               "visibly_evolving": progress >= .05})
        for index, epoch in enumerate(trajectory["epoch"]):
            state_id = canonical_hash({**identity, "epoch": epoch, "lambda": trajectory["lambda"][index],
                                       "physical_error": trajectory["physical_error"][index],
                                       "logical_error": trajectory["logical_learned"][index]})
            table.append({"distance": row["distance"], "parameters_per_gate": row["parameters_per_gate"],
                          "seed": row["seed"], "epoch": int(epoch),
                          "policy_state_id": state_id,
                          "policy_state_id_kind": "DERIVED_AUDIT_ID_FROM_RETAINED_EPOCH_STATE",
                          "physical_error": float(trajectory["physical_error"][index]),
                          "logical_error_learned": float(trajectory["logical_learned"][index]),
                          "logical_error_candidate": float(trajectory["logical_candidate"][index]),
                          "logical_error_fixed": float(trajectory["logical_fixed"][index]),
                          "logical_error_oracle": float(trajectory["logical_oracle"][index]),
                          "irreducible_floor": float(trajectory["logical_floor"][index])})
    all_vary = all(row["physical_span"] > 0 and row["logical_span"] > 0 for row in condition_rows)
    visible = all(row["visibly_evolving"] for row in condition_rows)
    result = {"schema_version": V12_SCHEMA, "protocol_hash": protocol["protocol_hash"],
              "merged_row_count": len(merged["rows"]), "trajectory_table_row_count": len(table),
              "condition_audit": condition_rows, "trajectory_table": table,
              "raw_quantities_vary": all_vary, "visibly_evolving_gate": visible,
              "lineage_verdict": "ACQUISITION_DIRECTIONAL_SCALE_ATTENUATION" if all_vary and not visible else (
                  "TRAJECTORY_VISIBLE" if visible else "MERGE_OR_PLOTTING_LINEAGE_FAILURE"),
              "original_per_epoch_checkpoint_ids_retained": False,
              "classification": "INVALID_DIAGNOSTIC" if not visible else "PARTIAL",
              **NONFINAL_FIELDS}
    atomic_json(ARTIFACT_ROOT / "lineage/figure5b_lineage.json", result)
    lines = ["# Figure 5b lineage audit", "",
             f"Verdict: **{result['lineage_verdict']}**", "",
             f"The {len(condition_rows)} raw trajectories contain varying physical and logical quantities, so merge/plot lineage is intact. "
             "Their floor-normalized progress does not meet the frozen 5% visibility gate; acquisition dynamics are the failure.", "",
             "Original per-epoch checkpoint identifiers were not retained. V12 emits explicit derived audit IDs and does not relabel them as original checkpoints."]
    atomic_text(ARTIFACT_ROOT / "lineage/figure5b_lineage.md", "\n".join(lines))
    _plot_figure5b(merged["rows"])
    return result


def convergence_derivative(lambdas: np.ndarray, lambda_star: float, *, fit_min: float = 1e-4,
                           fit_max: float = .7) -> dict[str, Any]:
    values = np.asarray(lambdas, dtype=float)
    if values.ndim != 1 or len(values) < 2 or not np.all(np.isfinite(values)) or lambda_star <= 0:
        raise ValueError("finite one-dimensional Lambda trajectory and positive Lambda* required")
    ratio = values / float(lambda_star)
    x = 1.0 - ratio[:-1]
    y = 100.0 * np.diff(ratio)
    mask = (x > fit_min) & (x < fit_max) & np.isfinite(x) & np.isfinite(y)
    count = int(np.count_nonzero(mask))
    slope = None
    r_squared = None
    if count >= 2 and float(np.dot(x[mask], x[mask])) > 0:
        slope = float(np.dot(x[mask], y[mask]) / np.dot(x[mask], x[mask]))
        prediction = slope * x[mask]
        denominator = float(np.sum((y[mask] - np.mean(y[mask])) ** 2))
        r_squared = 1.0 - float(np.sum((y[mask] - prediction) ** 2)) / denominator if denominator else 1.0
    return {"x_distance": x.tolist(), "normalized_speed": y.tolist(),
            "fit_mask": mask.astype(int).tolist(), "fit_point_count": count,
            "gamma_times_100": slope, "r_squared": r_squared,
            "identifiable": slope is not None and r_squared is not None}


def audit_figure5c_lineage() -> dict[str, Any]:
    protocol, merged = _load_merged("FIGURE5C_CONVERGENCE_LAW")
    rows = []
    for row in merged["rows"]:
        derivative = convergence_derivative(np.asarray(row["trajectory"]["lambda"]), float(row["lambda_star"]))
        rows.append({"distance": row["distance"], "parameters_per_gate": row["parameters_per_gate"],
                     "seed": row["seed"], "stored_gamma_times_100": row["gamma_times_100"],
                     "stored_r_squared": row["convergence_fit_r_squared"], **derivative})
    identifiable = sum(row["identifiable"] for row in rows)
    nonzero = sum(bool(np.any(np.abs(row["normalized_speed"]) > 0)) for row in rows)
    result = {"schema_version": V12_SCHEMA, "protocol_hash": protocol["protocol_hash"],
              "condition_count": len(rows), "nonzero_derivative_condition_count": nonzero,
              "identifiable_fit_condition_count": identifiable, "conditions": rows,
              "zero_fallback_forbidden": True,
              "lineage_verdict": "PREREGISTERED_LOCAL_WINDOW_NOT_REACHED_DURING_ACQUISITION" if nonzero and not identifiable else (
                  "DERIVATIVE_IDENTIFIABLE" if identifiable == len(rows) else "DERIVATIVE_LINEAGE_FAILURE"),
              "classification": "INVALID_DIAGNOSTIC" if identifiable != len(rows) else "PARTIAL",
              **NONFINAL_FIELDS}
    atomic_json(ARTIFACT_ROOT / "lineage/figure5c_lineage.json", result)
    atomic_text(ARTIFACT_ROOT / "lineage/figure5c_lineage.md",
                "# Figure 5c lineage audit\n\n"
                f"Verdict: **{result['lineage_verdict']}**.\n\n"
                f"All {nonzero}/{len(rows)} trajectories have nonzero finite differences, but only {identifiable}/{len(rows)} enter the preregistered local-fit window. "
                "The old zero-valued fallback was therefore a presentation error; V12 records the fit as unidentifiable without weakening the gate.")
    _plot_figure5c(rows)
    return result


def validate_figure5c_derivative() -> dict[str, Any]:
    gamma = .017
    ratio = 1.0 - .55 * np.exp(-gamma * np.arange(160))
    derived = convergence_derivative(ratio, 1.0, fit_min=1e-4, fit_max=.7)
    expected = 100.0 * (1.0 - np.exp(-gamma))
    observed = derived["gamma_times_100"]
    error = None if observed is None else abs(float(observed) - expected)
    result = {"schema_version": V12_SCHEMA, "fixture": "Lambda/Lambda*=1-0.55*exp(-gamma*t)",
              "gamma": gamma, "expected_discrete_gamma_times_100": expected,
              "observed_gamma_times_100": observed, "absolute_error": error,
              "fit_point_count": derived["fit_point_count"],
              "pass": observed is not None and error is not None and error < 1e-10 and derived["r_squared"] > .999999,
              **NONFINAL_FIELDS}
    atomic_json(ARTIFACT_ROOT / "lineage/figure5c_derivative_fixture.json", result)
    if not result["pass"]:
        raise RuntimeError("Figure5c synthetic derivative fixture failed")
    return result


def _plot_figure5b(rows: list[dict[str, Any]]) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FormatStrFormatter, MaxNLocator, NullLocator

    parameter_counts = sorted({int(row["parameters_per_gate"]) for row in rows})
    figure, axes = plt.subplots(1, len(parameter_counts), figsize=(14, 4.5), constrained_layout=True)
    colour_handle = None
    for axis, parameter_count in zip(np.atleast_1d(axes), parameter_counts):
        for row in [item for item in rows if int(item["parameters_per_gate"]) == parameter_count]:
            trajectory = row["trajectory"]
            epoch = np.asarray(trajectory["epoch"])
            colour_handle = axis.scatter(trajectory["physical_error"], trajectory["logical_learned"],
                                         c=epoch, cmap="viridis", s=5, alpha=.45)
            axis.plot(trajectory["physical_error"], trajectory["logical_learned"],
                      linewidth=.45, alpha=.28)
            axis.hlines(row["logical_floor"], min(trajectory["physical_error"]),
                        max(trajectory["physical_error"]), color="black", linewidth=.45, alpha=.3)
        axis.set(xscale="log", yscale="log", xlabel="Physical error rate",
                 ylabel="Logical error rate", title=f"{parameter_count} parameters/gate")
        axis.xaxis.set_major_locator(MaxNLocator(4))
        axis.xaxis.set_major_formatter(FormatStrFormatter("%.4g"))
        axis.xaxis.set_minor_locator(NullLocator())
        axis.grid(alpha=.18)
    if colour_handle is not None:
        figure.colorbar(colour_handle, ax=np.atleast_1d(axes).tolist(), label="Epoch", shrink=.8)
    figure.suptitle("Figure 5b retained physical/logical trajectories (zoomed diagnostic, non-final)")
    path = ARTIFACT_ROOT / "lineage/figure5b_physical_logical_trajectory.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=180)
    plt.close(figure)


def _plot_figure5c(rows: list[dict[str, Any]]) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    parameter_counts = sorted({int(row["parameters_per_gate"]) for row in rows})
    figure, axes = plt.subplots(1, len(parameter_counts), figsize=(14, 4.5), constrained_layout=True)
    for axis, parameter_count in zip(np.atleast_1d(axes), parameter_counts):
        selected = [row for row in rows if int(row["parameters_per_gate"]) == parameter_count]
        for row in selected:
            axis.plot(row["x_distance"], row["normalized_speed"], marker=".", markersize=2,
                      linewidth=.45, alpha=.35)
        axis.axhline(0, color="black", linewidth=.7)
        axis.axvspan(1e-4, .7, color="#009E73", alpha=.08, label="Preregistered fit window")
        axis.set(xlabel="1 - Lambda/Lambda*", ylabel="100 ΔLambda/Lambda*",
                 title=f"{parameter_count} parameters/gate")
        axis.grid(alpha=.18)
    axes[0].legend(fontsize=7)
    figure.suptitle("Figure 5c nonzero retained finite differences (fits unidentifiable)")
    path = ARTIFACT_ROOT / "lineage/figure5c_derivative_trajectories.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=180)
    plt.close(figure)
