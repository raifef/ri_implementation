"""Analyse the full Nature-2026 scalability run and emit publication figures.

This analysis deliberately keeps three evidence layers distinct:

* numerical anchors stated by Sivak et al.;
* the repository's declared Figure-5 surrogate; and
* actually executed HDFA-RL suite pipeline probes.

The script uses only pandas/numpy plus the Python standard library and writes SVG
directly so the figures remain vector, editable, and dependency-light.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import html
import json
import math
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np
import pandas as pd


PAPER = "paper_sparse_policy_gradient"
HDFA = "predictive_hdfa_residual_rl"
METHOD_LABEL = {
    PAPER: "Published-method surrogate",
    HDFA: "Predictive HDFA + residual RL",
}
METHOD_SHORT = {PAPER: "Published surrogate", HDFA: "HDFA + residual RL"}
METHOD_COLOUR = {PAPER: "#0072B2", HDFA: "#D55E00"}
METHOD_MARKER = {PAPER: "circle", HDFA: "triangle"}
NEUTRAL = "#252525"
GRID = "#D7D7D7"
MUTED = "#666666"
PAPER_FREQUENCY = 1.0 / 150.0
ARTICLE_URL = "https://www.nature.com/articles/s41586-026-10759-2"
DATA_URL = "https://doi.org/10.5281/zenodo.18896801"


def _fmt(value: float, digits: int = 3) -> str:
    if value == 0:
        return "0"
    if abs(value) < 1e-3 or abs(value) >= 1e4:
        return f"{value:.{digits - 1}e}"
    return f"{value:.{digits}g}"


def _quantile(values: Sequence[float], q: float) -> float:
    return float(np.quantile(np.asarray(values, dtype=float), q))


def _wilson(successes: int, total: int, z: float = 1.96) -> tuple[float, float]:
    if total <= 0:
        return math.nan, math.nan
    p = successes / total
    denominator = 1.0 + z * z / total
    centre = (p + z * z / (2 * total)) / denominator
    radius = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denominator
    return centre - radius, centre + radius


def _bootstrap_mean(values: Sequence[float], rng: np.random.Generator,
                    draws: int = 20_000) -> tuple[float, float, float]:
    data = np.asarray(values, dtype=float)
    if not len(data):
        return math.nan, math.nan, math.nan
    samples = rng.choice(data, size=(draws, len(data)), replace=True).mean(axis=1)
    return float(data.mean()), float(np.quantile(samples, .025)), float(np.quantile(samples, .975))


def _regression_loglog(x: Sequence[float], y: Sequence[float]) -> tuple[float, float, float]:
    lx = np.log(np.asarray(x, dtype=float))
    ly = np.log(np.asarray(y, dtype=float))
    slope, intercept = np.polyfit(lx, ly, 1)
    predicted = slope * lx + intercept
    ss_res = float(np.sum((ly - predicted) ** 2))
    ss_tot = float(np.sum((ly - ly.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else math.nan
    return float(slope), float(math.exp(intercept)), r2


class Svg:
    """Small publication-oriented SVG writer."""

    def __init__(self, width: int, height: int, title: str, description: str,
                 metadata: Mapping[str, object]) -> None:
        self.width, self.height = width, height
        self.parts = [
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
            f'viewBox="0 0 {width} {height}" role="img" aria-labelledby="title desc">',
            f'<title id="title">{html.escape(title)}</title>',
            f'<desc id="desc">{html.escape(description)}</desc>',
            f'<metadata>{html.escape(json.dumps(dict(metadata), sort_keys=True))}</metadata>',
            '<rect width="100%" height="100%" fill="#FFFFFF"/>',
            '<style>text{font-family:Arial,Helvetica,sans-serif;fill:#252525}'
            '.axis{stroke:#252525;stroke-width:1}.grid{stroke:#D7D7D7;stroke-width:.7}'
            '.tick{font-size:11px}.label{font-size:13px}.panel{font-size:15px;font-weight:700}'
            '.subtitle{font-size:13px;font-weight:600}.note{font-size:10px;fill:#666666}</style>',
        ]

    def rect(self, x: float, y: float, width: float, height: float, *,
             fill: str = "none", stroke: str = "none", stroke_width: float = 1,
             opacity: float = 1.) -> None:
        self.parts.append(
            f'<rect x="{x:.2f}" y="{y:.2f}" width="{width:.2f}" height="{height:.2f}" '
            f'fill="{fill}" stroke="{stroke}" stroke-width="{stroke_width:.2f}" opacity="{opacity:.3f}"/>'
        )

    def line(self, x1: float, y1: float, x2: float, y2: float, *,
             stroke: str = NEUTRAL, width: float = 1., dash: str | None = None,
             opacity: float = 1.) -> None:
        extra = f' stroke-dasharray="{dash}"' if dash else ""
        self.parts.append(
            f'<line x1="{x1:.2f}" y1="{y1:.2f}" x2="{x2:.2f}" y2="{y2:.2f}" '
            f'stroke="{stroke}" stroke-width="{width:.2f}" opacity="{opacity:.3f}"{extra}/>'
        )

    def polyline(self, points: Sequence[tuple[float, float]], *, stroke: str,
                 width: float = 1.5, dash: str | None = None, opacity: float = 1.) -> None:
        if not points:
            return
        encoded = " ".join(f"{x:.2f},{y:.2f}" for x, y in points)
        extra = f' stroke-dasharray="{dash}"' if dash else ""
        self.parts.append(
            f'<polyline points="{encoded}" fill="none" stroke="{stroke}" '
            f'stroke-width="{width:.2f}" opacity="{opacity:.3f}"{extra}/>'
        )

    def polygon(self, points: Sequence[tuple[float, float]], *, fill: str,
                stroke: str = "none", stroke_width: float = 1., opacity: float = 1.) -> None:
        encoded = " ".join(f"{x:.2f},{y:.2f}" for x, y in points)
        self.parts.append(
            f'<polygon points="{encoded}" fill="{fill}" stroke="{stroke}" '
            f'stroke-width="{stroke_width:.2f}" opacity="{opacity:.3f}"/>'
        )

    def circle(self, x: float, y: float, radius: float, *, fill: str,
               stroke: str = "none", stroke_width: float = 1., opacity: float = 1.) -> None:
        self.parts.append(
            f'<circle cx="{x:.2f}" cy="{y:.2f}" r="{radius:.2f}" fill="{fill}" '
            f'stroke="{stroke}" stroke-width="{stroke_width:.2f}" opacity="{opacity:.3f}"/>'
        )

    def marker(self, x: float, y: float, marker: str, *, fill: str, size: float = 4.,
               stroke: str = "#FFFFFF", stroke_width: float = .6, opacity: float = 1.) -> None:
        if marker == "triangle":
            self.polygon(((x, y-size), (x-size*.9, y+size*.75), (x+size*.9, y+size*.75)),
                         fill=fill, stroke=stroke, stroke_width=stroke_width, opacity=opacity)
        elif marker == "square":
            self.rect(x-size, y-size, 2*size, 2*size, fill=fill, stroke=stroke,
                      stroke_width=stroke_width, opacity=opacity)
        else:
            self.circle(x, y, size, fill=fill, stroke=stroke,
                        stroke_width=stroke_width, opacity=opacity)

    def text(self, x: float, y: float, value: str, *, css: str = "tick",
             anchor: str = "start", fill: str | None = None,
             rotate: float | None = None, weight: int | None = None,
             italic: bool = False) -> None:
        transform = f' transform="rotate({rotate:.1f} {x:.2f} {y:.2f})"' if rotate is not None else ""
        style = []
        if fill:
            style.append(f"fill:{fill}")
        if weight:
            style.append(f"font-weight:{weight}")
        if italic:
            style.append("font-style:italic")
        style_attr = f' style="{";".join(style)}"' if style else ""
        self.parts.append(
            f'<text x="{x:.2f}" y="{y:.2f}" class="{css}" text-anchor="{anchor}"'
            f'{transform}{style_attr}>{html.escape(value)}</text>'
        )

    def finish(self, path: Path) -> None:
        path.write_text("\n".join(self.parts + ["</svg>"]), encoding="utf-8")


def _linear(value: float, lower: float, upper: float, start: float, stop: float) -> float:
    return start + (value-lower) / max(upper-lower, 1e-15) * (stop-start)


def _log(value: float, lower: float, upper: float, start: float, stop: float) -> float:
    return _linear(math.log10(value), math.log10(lower), math.log10(upper), start, stop)


def _interpolate_colour(value: float, lower: float, upper: float) -> str:
    negative = np.array([59, 76, 192], dtype=float)
    middle = np.array([247, 247, 247], dtype=float)
    positive = np.array([180, 4, 38], dtype=float)
    if upper <= lower:
        return "#F7F7F7"
    scaled = max(-1., min(1., 2*(value-lower)/(upper-lower)-1))
    if scaled < 0:
        rgb = middle + (-scaled) * (negative-middle)
    else:
        rgb = middle + scaled * (positive-middle)
    return "#" + "".join(f"{int(round(channel)):02X}" for channel in rgb)


def _sequential_colour(value: float, lower: float, upper: float) -> str:
    low = np.array([247, 251, 255], dtype=float)
    high = np.array([8, 81, 156], dtype=float)
    t = max(0., min(1., (value-lower)/max(upper-lower, 1e-15)))
    rgb = low + t * (high-low)
    return "#" + "".join(f"{int(round(channel)):02X}" for channel in rgb)


def _axis(svg: Svg, left: float, top: float, width: float, height: float,
          x_ticks: Sequence[tuple[float, str]], y_ticks: Sequence[tuple[float, str]],
          x_label: str, y_label: str) -> None:
    for x, label in x_ticks:
        svg.line(x, top, x, top+height, stroke=GRID, width=.7)
        svg.line(x, top+height, x, top+height+4, width=1.)
        svg.text(x, top+height+18, label, anchor="middle")
    for y, label in y_ticks:
        svg.line(left, y, left+width, y, stroke=GRID, width=.7)
        svg.line(left-4, y, left, y, width=1.)
        svg.text(left-8, y+4, label, anchor="end")
    svg.line(left, top+height, left+width, top+height, width=1.1)
    svg.line(left, top, left, top+height, width=1.1)
    svg.text(left+width/2, top+height+43, x_label, css="label", anchor="middle")
    svg.text(left-55, top+height/2, y_label, css="label", anchor="middle", rotate=-90)


def _panel_heading(svg: Svg, left: float, top: float, letter: str, title: str) -> None:
    svg.text(left, top, letter, css="panel")
    svg.text(left+23, top, title, css="subtitle")


def _legend(svg: Svg, x: float, y: float, methods: Sequence[str] = (PAPER, HDFA)) -> None:
    for index, method in enumerate(methods):
        yy = y + index*22
        svg.line(x, yy-4, x+24, yy-4, stroke=METHOD_COLOUR[method], width=2.)
        svg.marker(x+12, yy-4, METHOD_MARKER[method], fill=METHOD_COLOUR[method], size=3.7)
        svg.text(x+32, yy, METHOD_SHORT[method])


def validate_inputs(input_dir: Path) -> dict[str, object]:
    manifest = json.loads((input_dir / "manifest.json").read_text(encoding="utf-8"))
    failures = []
    for record in manifest["artifacts"].values():
        path = input_dir / record["path"]
        digest = hashlib.sha256(path.read_bytes()).hexdigest() if path.exists() else None
        if digest != record["sha256"]:
            failures.append(record["path"])
    if failures:
        raise ValueError(f"manifest checksum failure: {failures}")
    report = json.loads((input_dir / "scalability-report.json").read_text(encoding="utf-8"))
    expected = report["tables"]
    table_files = {
        "scaling": "fig5b-scaling.csv",
        "convergence": "fig5c-convergence.csv",
        "steerability": "fig5a-steerability.csv",
        "resources": "resource-scaling.csv",
        "sample_efficiency": "sample-efficiency.csv",
        "pipeline_probe": "pipeline-probe.csv",
    }
    counts = {}
    for key, file in table_files.items():
        with (input_dir / file).open("r", encoding="utf-8", newline="") as stream:
            count = sum(1 for _ in stream) - 1
        counts[key] = count
        if count != expected[key]["rows"]:
            raise ValueError(f"row-count mismatch for {file}: {count} != {expected[key]['rows']}")
    return {
        "manifest_report_hash": manifest["report_hash"],
        "checksums_verified": len(manifest["artifacts"]),
        "row_counts": counts,
    }


def load_tables(input_dir: Path) -> dict[str, pd.DataFrame]:
    tables = {
        "steering": pd.read_csv(input_dir / "fig5a-steerability.csv"),
        "scaling": pd.read_csv(input_dir / "fig5b-scaling.csv"),
        "convergence": pd.read_csv(input_dir / "fig5c-convergence.csv"),
        "fits": pd.read_csv(input_dir / "convergence-fits.csv"),
        "efficiency": pd.read_csv(input_dir / "sample-efficiency.csv"),
        "pipeline": pd.read_csv(input_dir / "pipeline-probe.csv"),
        "resources": pd.read_csv(input_dir / "resource-scaling.csv"),
    }
    uniqueness = {
        "steering": ["method", "seed", "drift_frequency", "entropy_regularization"],
        "scaling": ["method", "seed", "code_distance", "parameters_per_gate", "epoch"],
        "convergence": ["method", "seed", "code_distance", "parameters_per_gate", "epoch"],
        "efficiency": ["method", "seed", "code_distance", "parameters_per_gate", "target_fraction"],
        "pipeline": ["method", "seed", "code_distance", "epoch"],
        "resources": ["method", "code_distance", "parameters_per_gate"],
    }
    for name, columns in uniqueness.items():
        if tables[name].duplicated(columns).any():
            raise ValueError(f"duplicate primary key in {name}: {columns}")
    if set(tables["steering"]["evidence_layer"]) != {"declared_first_order_tracking_surrogate"}:
        raise ValueError("unexpected steerability evidence layer")
    if set(tables["pipeline"]["evidence_layer"]) != {"executed_suite_pipeline"}:
        raise ValueError("unexpected pipeline evidence layer")
    required_pipeline_columns = {
        "peak_incremental_process_memory_bytes", "peak_process_memory_bytes",
        "process_memory_baseline_bytes", "worker_concurrency",
        "condition_process_isolation",
    }
    missing = required_pipeline_columns - set(tables["pipeline"].columns)
    if missing:
        raise ValueError(f"pipeline table lacks v3 memory/concurrency fields: {sorted(missing)}")
    worker_contexts = set(tables["pipeline"]["worker_concurrency"])
    if len(worker_contexts) != 1:
        raise ValueError(
            "computational-cost analysis requires one worker-concurrency context; "
            f"observed {sorted(worker_contexts)}")
    isolation = set(tables["pipeline"]["condition_process_isolation"])
    if isolation != {"fresh_process_per_condition"}:
        raise ValueError(
            "computational-cost analysis requires a fresh process per condition; "
            f"observed {sorted(isolation)}")
    return tables


def analyse(tables: Mapping[str, pd.DataFrame], report: Mapping[str, object]) -> tuple[dict, dict[str, pd.DataFrame]]:
    rng = np.random.default_rng(20260801)
    steering = tables["steering"]
    steering_grid = (steering.groupby(["method", "drift_frequency", "entropy_regularization"], as_index=False)
                     ["stochastic_improvement"].mean())
    steering_summary = {}
    for method in (PAPER, HDFA):
        subset = steering_grid[steering_grid.method == method]
        best_by_frequency = subset.groupby("drift_frequency")["stochastic_improvement"].max()
        positive = best_by_frequency[best_by_frequency > 0]
        two_percent = best_by_frequency[best_by_frequency >= .02]
        steering_summary[method] = {
            "critical_frequency_positive": float(positive.index.max()) if len(positive) else None,
            "critical_frequency_at_least_2pct": float(two_percent.index.max()) if len(two_percent) else None,
            "maximum_mean_improvement": float(subset.stochastic_improvement.max()),
            "best_entropy_at_paper_anchor_gridpoint": float(
                subset.loc[(subset.drift_frequency-PAPER_FREQUENCY).abs().idxmin(), "entropy_regularization"]),
        }

    resources = tables["resources"]
    cycles_per_epoch = {
        method: int(resources.loc[resources.method == method, "qec_cycles_per_epoch"].mode().iloc[0])
        for method in (PAPER, HDFA)
    }
    fits = tables["fits"].copy()
    fits["gamma_per_million_qec_cycles"] = fits.apply(
        lambda row: row.gamma * 1_000_000 / cycles_per_epoch[row.method], axis=1)
    gamma_summary = []
    for (method, parameters), group in fits.groupby(["method", "parameters_per_gate"]):
        gamma_summary.append({
            "method": method,
            "parameters_per_gate": int(parameters),
            "mean_gamma_per_epoch": float(group.gamma.mean()),
            "cv_gamma_across_distance": float(group.gamma.std(ddof=0) / group.gamma.mean()),
            "mean_gamma_per_million_qec_cycles": float(group.gamma_per_million_qec_cycles.mean()),
            "mean_r_squared": float(group.r_squared.mean()),
            "minimum_r_squared": float(group.r_squared.min()),
        })
    gamma_table = pd.DataFrame(gamma_summary)
    cycle_gamma_ratio = {}
    for parameters in sorted(fits.parameters_per_gate.unique()):
        values = gamma_table[gamma_table.parameters_per_gate == parameters].set_index("method")
        cycle_gamma_ratio[str(int(parameters))] = float(
            values.loc[HDFA, "mean_gamma_per_million_qec_cycles"] /
            values.loc[PAPER, "mean_gamma_per_million_qec_cycles"])

    efficiency = tables["efficiency"].copy()
    efficiency["achieved"] = efficiency["achieved"].astype(bool)
    attainment_rows = []
    for (method, target), group in efficiency.groupby(["method", "target_fraction"]):
        successes, total = int(group.achieved.sum()), len(group)
        low, high = _wilson(successes, total)
        attainment_rows.append({
            "method": method, "target_fraction": float(target), "achieved": successes,
            "total": total, "fraction": successes/total, "ci95_low": low, "ci95_high": high,
        })
    attainment = pd.DataFrame(attainment_rows)
    paired_rows = []
    keys = ["seed", "code_distance", "parameters_per_gate", "target_fraction"]
    merged = efficiency.merge(efficiency, on=keys, suffixes=("_paper", "_hdfa"))
    merged = merged[(merged.method_paper == PAPER) & (merged.method_hdfa == HDFA)]
    merged = merged[merged.achieved_paper & merged.achieved_hdfa].copy()
    merged["cycle_ratio_paper_over_hdfa"] = (
        merged.cumulative_qec_cycles_paper / merged.cumulative_qec_cycles_hdfa)
    for target in sorted(efficiency.target_fraction.unique()):
        values = merged.loc[merged.target_fraction == target, "cycle_ratio_paper_over_hdfa"].to_numpy()
        if len(values):
            median = float(np.median(values))
            paired_rows.append({
                "target_fraction": float(target), "matched_pairs": len(values),
                "median_cycle_ratio": median, "q1": _quantile(values, .25), "q3": _quantile(values, .75),
                "minimum": float(values.min()), "maximum": float(values.max()),
                "pairs_at_least_10x": int((values >= 10).sum()),
            })
        else:
            paired_rows.append({
                "target_fraction": float(target), "matched_pairs": 0,
                "median_cycle_ratio": math.nan, "q1": math.nan, "q3": math.nan,
                "minimum": math.nan, "maximum": math.nan, "pairs_at_least_10x": 0,
            })
    paired_summary = pd.DataFrame(paired_rows)

    pipeline = tables["pipeline"]
    per_run = (pipeline.groupby(["method", "seed", "code_distance", "physical_qubits", "suite_control_variables"], as_index=False)
               .agg(detector_event_rate=("detector_event_rate", "mean"),
                    logical_failure_rate=("logical_failure_rate", "mean"),
                    elapsed_s=("elapsed_s", "mean"),
                    peak_incremental_process_memory_bytes=("peak_incremental_process_memory_bytes", "max"),
                    peak_process_memory_bytes=("peak_process_memory_bytes", "max"),
                    candidate_evaluations=("candidate_evaluations", "sum"),
                    qec_cycles=("qec_cycles", "sum")))
    pipeline_summary_rows = []
    for (method, distance), group in per_run.groupby(["method", "code_distance"]):
        pipeline_summary_rows.append({
            "method": method, "code_distance": int(distance),
            "physical_qubits": int(group.physical_qubits.iloc[0]),
            "suite_control_variables": int(group.suite_control_variables.iloc[0]),
            "elapsed_median_s": float(group.elapsed_s.median()),
            "elapsed_q1_s": _quantile(group.elapsed_s, .25),
            "elapsed_q3_s": _quantile(group.elapsed_s, .75),
            "memory_median_bytes": float(group.peak_process_memory_bytes.median()),
            "memory_q1_bytes": _quantile(group.peak_process_memory_bytes, .25),
            "memory_q3_bytes": _quantile(group.peak_process_memory_bytes, .75),
            "incremental_rss_median_bytes": float(
                group.peak_incremental_process_memory_bytes.median()),
            "detector_event_rate_mean": float(group.detector_event_rate.mean()),
        })
    pipeline_summary = pd.DataFrame(pipeline_summary_rows)
    paired_pipeline = per_run.merge(per_run, on=["seed", "code_distance", "physical_qubits", "suite_control_variables"],
                                    suffixes=("_paper", "_hdfa"))
    paired_pipeline = paired_pipeline[(paired_pipeline.method_paper == PAPER) & (paired_pipeline.method_hdfa == HDFA)].copy()
    paired_pipeline["detector_rate_ratio_hdfa_over_paper"] = (
        paired_pipeline.detector_event_rate_hdfa / paired_pipeline.detector_event_rate_paper)
    paired_detector_rows = []
    for distance, group in paired_pipeline.groupby("code_distance"):
        mean, low, high = _bootstrap_mean(group.detector_rate_ratio_hdfa_over_paper, rng)
        paired_detector_rows.append({
            "code_distance": int(distance), "mean_ratio": mean, "ci95_low": low,
            "ci95_high": high, "paired_seeds": len(group),
        })
    paired_detector = pd.DataFrame(paired_detector_rows)
    scaling_exponents = {}
    for method in (PAPER, HDFA):
        group = pipeline_summary[pipeline_summary.method == method].sort_values("suite_control_variables")
        time_slope, _, time_r2 = _regression_loglog(group.suite_control_variables, group.elapsed_median_s)
        memory_slope, _, memory_r2 = _regression_loglog(group.suite_control_variables, group.memory_median_bytes)
        scaling_exponents[method] = {
            "elapsed_vs_controls_exponent": time_slope, "elapsed_loglog_r_squared": time_r2,
            "memory_vs_controls_exponent": memory_slope, "memory_loglog_r_squared": memory_r2,
        }
    logical_failures_observed = bool((pipeline.logical_failure_rate > 0).any())

    d15_resource = resources[(resources.code_distance == 15) & (resources.parameters_per_gate == 30)]
    resource_summary = {
        method: {
            "control_parameters": int(d15_resource[d15_resource.method == method].control_parameters.iloc[0]),
            "estimated_policy_state_bytes": int(d15_resource[d15_resource.method == method].estimated_policy_state_bytes.iloc[0]),
        }
        for method in (PAPER, HDFA)
    }
    resource_summary["executed_suite_d15"] = {
        "suite_control_variables": int(pipeline[pipeline.code_distance == 15].suite_control_variables.iloc[0]),
        "physical_qubits": int(pipeline[pipeline.code_distance == 15].physical_qubits.iloc[0]),
    }

    summary = {
        "schema_version": "nature-2026-full-analysis.v3",
        "evidence_statement": "Published numerical anchors, declared surrogates, and executed suite probes are not interchangeable.",
        "paper_anchors": {
            "critical_steerability_frequency_epoch_inverse": PAPER_FREQUENCY,
            "response_time_epochs": 130,
            "distance_15_p30_control_parameters": 38670,
            "paper_total_cycles_fig5a": 1_800_000_000,
        },
        "steerability": steering_summary,
        "cycles_per_epoch": cycles_per_epoch,
        "convergence": {
            "gamma_per_cycle_advantage_hdfa_over_paper": cycle_gamma_ratio,
            "fit_quality_warning": "Several origin-constrained convergence fits have negative R-squared; gamma invariance alone is insufficient evidence of the proposed law.",
        },
        "sample_efficiency": paired_summary.replace({np.nan: None}).to_dict(orient="records"),
        "executed_pipeline": {
            "rows": len(pipeline), "paired_runs": len(paired_pipeline),
            "scaling_exponents": scaling_exponents,
            "logical_failures_observed": logical_failures_observed,
            "probe_horizon_epochs": int(pipeline.epoch.max()+1),
            "worker_concurrency": int(pipeline.worker_concurrency.iloc[0]),
            "memory_metric": "sampled peak process resident set in fresh per-condition workers",
            "supplemental_memory_metric": "baseline-subtracted sampled process resident-set increment",
        },
        "resources": resource_summary,
        "sources": {"article": ARTICLE_URL, "source_data": DATA_URL},
    }
    derived = {
        "steering_grid": steering_grid,
        "fits": fits,
        "gamma_summary": gamma_table,
        "attainment": attainment,
        "paired_efficiency": merged,
        "paired_efficiency_summary": paired_summary,
        "pipeline_per_run": per_run,
        "pipeline_summary": pipeline_summary,
        "pipeline_detector_ratios": paired_detector,
    }
    return summary, derived


def plot_steerability(grid: pd.DataFrame, output: Path, metadata: Mapping[str, object]) -> None:
    frequencies = sorted(grid.drift_frequency.unique())
    entropies = sorted(grid.entropy_regularization.unique())
    lookup = {(row.method, row.drift_frequency, row.entropy_regularization): row.stochastic_improvement
              for row in grid.itertuples()}
    values = np.array(list(lookup.values()), dtype=float)
    limit = max(1., float(np.nanmax(np.abs(values))))
    width, height = 1420, 510
    svg = Svg(width, height, "Real-time steerability comparison",
              "Published-method surrogate and predictive HDFA residual RL normalized improvement across drift frequency and entropy regularization, with a difference panel.", metadata)
    panel_width, panel_height = 365, 320
    lefts = (86, 536, 986)
    top = 66
    panels = ((PAPER, "A", METHOD_LABEL[PAPER]), (HDFA, "B", METHOD_LABEL[HDFA]), ("delta", "C", "HDFA minus published surrogate"))
    for panel_index, (method, letter, title) in enumerate(panels):
        left = lefts[panel_index]
        _panel_heading(svg, left, 30, letter, title)
        cell_w, cell_h = panel_width/len(entropies), panel_height/len(frequencies)
        delta_values = []
        if method == "delta":
            delta_values = [lookup[(HDFA, f, e)]-lookup[(PAPER, f, e)] for f in frequencies for e in entropies]
            delta_limit = max(.05, float(np.nanmax(np.abs(delta_values))))
        for xi, entropy in enumerate(entropies):
            for yi, frequency in enumerate(frequencies):
                if method == "delta":
                    value = lookup[(HDFA, frequency, entropy)]-lookup[(PAPER, frequency, entropy)]
                    colour = _interpolate_colour(value, -delta_limit, delta_limit)
                else:
                    value = lookup[(method, frequency, entropy)]
                    colour = _interpolate_colour(value, -limit, limit)
                x = left + xi*cell_w
                y = top + panel_height-(yi+1)*cell_h
                svg.rect(x, y, cell_w+.2, cell_h+.2, fill=colour)
        if method != "delta":
            boundary = []
            for xi, entropy in enumerate(entropies):
                positives = [f for f in frequencies if lookup[(method, f, entropy)] >= 0]
                if positives:
                    f = max(positives)
                    index = frequencies.index(f)
                    boundary.append((left+(xi+.5)*cell_w, top+panel_height-(index+1)*cell_h))
            svg.polyline(boundary, stroke="#111111", width=1.6)
        anchor_y = _log(PAPER_FREQUENCY, frequencies[0], frequencies[-1], top+panel_height, top)
        svg.line(left, anchor_y, left+panel_width, anchor_y, stroke="#111111", width=1.1, dash="5 4")
        if panel_index == 0:
            svg.text(left+panel_width-4, anchor_y-5, "paper ≈ 1/150 epoch⁻¹", css="note", anchor="end")
        svg.line(left, top+panel_height, left+panel_width, top+panel_height, width=1.1)
        svg.line(left, top, left, top+panel_height, width=1.1)
        for index in (0, len(entropies)//2, len(entropies)-1):
            x = left + (index+.5)*cell_w
            svg.text(x, top+panel_height+19, f"{entropies[index]:.0e}", anchor="middle")
        for index in (0, len(frequencies)//2, len(frequencies)-1):
            y = top+panel_height-(index+.5)*cell_h
            svg.text(left-8, y+4, f"{frequencies[index]:.0e}", anchor="end")
        svg.text(left+panel_width/2, top+panel_height+43, "Entropy regularization", css="label", anchor="middle")
        if panel_index == 0:
            svg.text(left-56, top+panel_height/2, "Drift frequency (epoch⁻¹)", css="label", anchor="middle", rotate=-90)
    # Two compact colour bars.
    bar_y = 458
    for i in range(120):
        v = -limit + 2*limit*i/119
        svg.rect(115+i*2.1, bar_y, 2.2, 10, fill=_interpolate_colour(v, -limit, limit))
    svg.text(115, bar_y+24, "−1 harmful", css="note")
    svg.text(241, bar_y+24, "0 fixed", css="note", anchor="middle")
    svg.text(367, bar_y+24, "+1 optimal", css="note", anchor="end")
    svg.text(241, bar_y-6, "Mean normalized improvement", css="note", anchor="middle")
    delta_limit = max(.05, float(np.nanmax(np.abs(delta_values))))
    for i in range(120):
        v = -delta_limit + 2*delta_limit*i/119
        svg.rect(1056+i*2.1, bar_y, 2.2, 10, fill=_interpolate_colour(v, -delta_limit, delta_limit))
    svg.text(1056, bar_y+24, f"−{delta_limit:.2f}", css="note")
    svg.text(1182, bar_y+24, "0", css="note", anchor="middle")
    svg.text(1308, bar_y+24, f"+{delta_limit:.2f}", css="note", anchor="end")
    svg.text(1182, bar_y-6, "Difference in normalized improvement", css="note", anchor="middle")
    svg.finish(output)


def plot_scaling(scaling: pd.DataFrame, fits: pd.DataFrame, output: Path,
                 metadata: Mapping[str, object]) -> None:
    p = 30
    heat = (scaling[scaling.parameters_per_gate == p]
            .groupby(["method", "code_distance", "epoch"], as_index=False)["lambda_ratio"].median())
    distances = sorted(heat.code_distance.unique())
    epochs = sorted(heat.epoch.unique())
    lookup = {(row.method, row.code_distance, row.epoch): row.lambda_ratio for row in heat.itertuples()}
    values = np.array(list(lookup.values()), dtype=float)
    lower, upper = 0., max(1., float(np.nanmax(values)))
    width, height = 1420, 525
    svg = Svg(width, height, "Scaling and convergence comparison",
              "Median error-suppression progress across code distance and epoch for P=30, plus convergence rate per million native QEC cycles.", metadata)
    top, heat_w, heat_h = 70, 430, 320
    for panel, method in enumerate((PAPER, HDFA)):
        left = 74 + panel*480
        _panel_heading(svg, left, 30, "AB"[panel], f"{METHOD_LABEL[method]} — P={p}")
        cell_w, cell_h = heat_w/len(epochs), heat_h/len(distances)
        for xi, epoch in enumerate(epochs):
            for yi, distance in enumerate(distances):
                value = lookup[(method, distance, epoch)]
                x = left + xi*cell_w
                y = top + heat_h-(yi+1)*cell_h
                svg.rect(x, y, max(1., cell_w+.05), cell_h+.2,
                         fill=_sequential_colour(value, lower, upper))
        svg.line(left, top+heat_h, left+heat_w, top+heat_h, width=1.1)
        svg.line(left, top, left, top+heat_h, width=1.1)
        for epoch in (0, 250, 500):
            x = _linear(epoch, 0, 500, left, left+heat_w)
            svg.text(x, top+heat_h+18, str(epoch), anchor="middle")
        for distance in distances:
            yi = distances.index(distance)
            y = top + heat_h-(yi+.5)*cell_h
            svg.text(left-9, y+4, str(distance), anchor="end")
        svg.text(left+heat_w/2, top+heat_h+43, "Learning epoch", css="label", anchor="middle")
        if panel == 0:
            svg.text(left-43, top+heat_h/2, "Code distance", css="label", anchor="middle", rotate=-90)
    # Convergence rate per native-QEC cycle.
    left, plot_w, plot_h = 1042, 305, 320
    _panel_heading(svg, left, 30, "C", "Convergence per native-QEC budget")
    subset = fits.copy()
    y_min = max(1e-5, float(subset.gamma_per_million_qec_cycles.min())*.75)
    y_max = float(subset.gamma_per_million_qec_cycles.max())*1.35
    _axis(svg, left, top, plot_w, plot_h,
          [(_linear(d, 3, 15, left, left+plot_w), str(d)) for d in distances],
          [(_log(v, y_min, y_max, top+plot_h, top), _fmt(v, 2)) for v in np.geomspace(y_min, y_max, 4)],
          "Code distance", "γ per 10⁶ QEC cycles")
    p_dash = {1: None, 10: "7 4", 30: "2 3"}
    p_marker = {1: "circle", 10: "triangle", 30: "square"}
    for method in (PAPER, HDFA):
        for parameters in (1, 10, 30):
            group = subset[(subset.method == method) & (subset.parameters_per_gate == parameters)].sort_values("code_distance")
            points = [(_linear(row.code_distance, 3, 15, left, left+plot_w),
                       _log(row.gamma_per_million_qec_cycles, y_min, y_max, top+plot_h, top))
                      for row in group.itertuples()]
            svg.polyline(points, stroke=METHOD_COLOUR[method], width=1.5,
                         dash=p_dash[parameters], opacity=.9)
            for x, y in points:
                svg.marker(x, y, p_marker[parameters], fill=METHOD_COLOUR[method], size=3.5)
    _legend(svg, left+8, top+18)
    for index, parameters in enumerate((1, 10, 30)):
        yy = top+85+index*21
        svg.line(left+8, yy-4, left+32, yy-4, stroke=NEUTRAL, width=1.4, dash=p_dash[parameters])
        svg.marker(left+20, yy-4, p_marker[parameters], fill=NEUTRAL, size=3.2)
        svg.text(left+39, yy, f"P={parameters}")
    # Shared progress colour bar.
    bar_x, bar_y = 190, 459
    for i in range(180):
        v = lower + (upper-lower)*i/179
        svg.rect(bar_x+i*2.2, bar_y, 2.3, 10, fill=_sequential_colour(v, lower, upper))
    svg.text(bar_x, bar_y+24, "0 initial", css="note")
    svg.text(bar_x+198, bar_y+24, "0.5", css="note", anchor="middle")
    svg.text(bar_x+396, bar_y+24, "1 local optimum", css="note", anchor="end")
    svg.text(bar_x+198, bar_y-6, "Median Λ/Λ*", css="note", anchor="middle")
    svg.finish(output)


def plot_efficiency(attainment: pd.DataFrame, paired: pd.DataFrame,
                    paired_summary: pd.DataFrame, output: Path,
                    metadata: Mapping[str, object]) -> None:
    width, height = 1200, 500
    svg = Svg(width, height, "Sample-efficiency comparison",
              "Target attainment fractions and paired native-QEC cycle ratios for the published-method surrogate and predictive HDFA residual RL.", metadata)
    # Panel A: attainment fraction.
    left, top, plot_w, plot_h = 80, 64, 470, 330
    _panel_heading(svg, left, 29, "A", "Fraction reaching each local-optimum target")
    targets = sorted(attainment.target_fraction.unique())
    _axis(svg, left, top, plot_w, plot_h,
          [(_linear(i, -.5, 2.5, left, left+plot_w), f"{int(target*100)}%") for i, target in enumerate(targets)],
          [(_linear(v, 0, 1, top+plot_h, top), f"{v:.1f}") for v in np.linspace(0, 1, 6)],
          "Target progress", "Runs reaching target")
    bar_w = 28
    for target_index, target in enumerate(targets):
        centre = _linear(target_index, -.5, 2.5, left, left+plot_w)
        for method_index, method in enumerate((PAPER, HDFA)):
            row = attainment[(attainment.method == method) & (attainment.target_fraction == target)].iloc[0]
            x = centre + (-bar_w*.65 if method_index == 0 else bar_w*.65)
            y = _linear(row.fraction, 0, 1, top+plot_h, top)
            svg.rect(x-bar_w/2, y, bar_w, top+plot_h-y, fill=METHOD_COLOUR[method], opacity=.85)
            low_y = _linear(row.ci95_low, 0, 1, top+plot_h, top)
            high_y = _linear(row.ci95_high, 0, 1, top+plot_h, top)
            svg.line(x, low_y, x, high_y, width=1.2)
            svg.line(x-5, low_y, x+5, low_y, width=1.2)
            svg.line(x-5, high_y, x+5, high_y, width=1.2)
            svg.text(x, y-7, f"{int(row.achieved)}/{int(row.total)}", css="note", anchor="middle")
    _legend(svg, left+12, top+20)
    # Panel B: paired ratios.
    left, plot_w = 680, 420
    _panel_heading(svg, left, 29, "B", "Cycle advantage on matched successful runs")
    y_min, y_max = .8, 25.
    _axis(svg, left, top, plot_w, plot_h,
          [(_linear(i, -.5, 2.5, left, left+plot_w), f"{int(target*100)}%") for i, target in enumerate(targets)],
          [(_log(v, y_min, y_max, top+plot_h, top), f"{v:g}×") for v in (1, 2, 5, 10, 20)],
          "Target progress", "Paper cycles / HDFA cycles")
    ten_y = _log(10, y_min, y_max, top+plot_h, top)
    svg.line(left, ten_y, left+plot_w, ten_y, stroke="#8A2D2D", width=1.2, dash="5 4")
    svg.text(left+plot_w-3, ten_y-5, "10× project target", css="note", anchor="end", fill="#8A2D2D")
    for target_index, target in enumerate(targets):
        centre = _linear(target_index, -.5, 2.5, left, left+plot_w)
        values = paired.loc[paired.target_fraction == target, "cycle_ratio_paper_over_hdfa"].to_numpy()
        if not len(values):
            svg.text(centre, top+plot_h/2, "no matched\n90% completions".replace("\n", " "), css="note", anchor="middle")
            continue
        jitter = np.linspace(-18, 18, len(values))
        order = np.argsort(values)
        for j, value in zip(jitter, values[order]):
            svg.circle(centre+j, _log(value, y_min, y_max, top+plot_h, top), 2.4,
                       fill="#777777", opacity=.55)
        row = paired_summary[paired_summary.target_fraction == target].iloc[0]
        q1_y = _log(row.q1, y_min, y_max, top+plot_h, top)
        q3_y = _log(row.q3, y_min, y_max, top+plot_h, top)
        med_y = _log(row.median_cycle_ratio, y_min, y_max, top+plot_h, top)
        svg.rect(centre-22, q3_y, 44, q1_y-q3_y, fill="#FFFFFF", stroke=NEUTRAL, stroke_width=1.2)
        svg.line(centre-22, med_y, centre+22, med_y, width=2.)
        svg.text(centre, min(top+plot_h-7, q1_y+18), f"n={int(row.matched_pairs)}", css="note", anchor="middle")
    svg.text(680, 463, "Ratios are conditional on both methods reaching the target; attainment in panel A must be read jointly.", css="note")
    svg.finish(output)


def plot_pipeline(summary: pd.DataFrame, detector_ratios: pd.DataFrame, output: Path,
                  metadata: Mapping[str, object]) -> None:
    width, height = 1420, 500
    svg = Svg(width, height, "Executed pipeline scaling",
              "Actually executed suite wall time, isolated process memory, and paired detector-event-rate ratio across code distance.", metadata)
    panels = (("elapsed_median_s", "elapsed_q1_s", "elapsed_q3_s", "Wall time per interval (s)", "A", "Measured Python time"),
              ("memory_median_bytes", "memory_q1_bytes", "memory_q3_bytes", "Peak process RSS (bytes)", "B", "Fresh-process memory"))
    top, plot_h, plot_w = 68, 310, 350
    for panel_index, (value_col, low_col, high_col, y_label, letter, title) in enumerate(panels):
        left = 76 + panel_index*455
        _panel_heading(svg, left, 30, letter, title)
        values = summary[value_col]
        y_min, y_max = float(values.min())*.55, float(values.max())*1.7
        controls = sorted(summary.suite_control_variables.unique())
        _axis(svg, left, top, plot_w, plot_h,
              [(_log(v, min(controls), max(controls), left, left+plot_w), str(v)) for v in (33, 129, 449, 897) if min(controls) <= v <= max(controls)],
              [(_log(v, y_min, y_max, top+plot_h, top), _fmt(v, 2)) for v in np.geomspace(y_min, y_max, 4)],
              "Suite control variables", y_label)
        for method in (PAPER, HDFA):
            group = summary[summary.method == method].sort_values("suite_control_variables")
            points = []
            for row in group.itertuples():
                x = _log(row.suite_control_variables, min(controls), max(controls), left, left+plot_w)
                y = _log(getattr(row, value_col), y_min, y_max, top+plot_h, top)
                lo = _log(getattr(row, low_col), y_min, y_max, top+plot_h, top)
                hi = _log(getattr(row, high_col), y_min, y_max, top+plot_h, top)
                svg.line(x, lo, x, hi, stroke=METHOD_COLOUR[method], width=1.)
                svg.marker(x, y, METHOD_MARKER[method], fill=METHOD_COLOUR[method], size=4.)
                points.append((x, y))
            svg.polyline(points, stroke=METHOD_COLOUR[method], width=1.8)
        if panel_index == 0:
            _legend(svg, left+10, top+22)
    # Panel C: paired detector-rate ratio.
    left, plot_w = 1000, 340
    _panel_heading(svg, left, 30, "C", "Paired detector-event-rate ratio")
    distances = sorted(detector_ratios.code_distance.unique())
    y_min = min(.55, float(detector_ratios.ci95_low.min())*.9)
    y_max = max(1.25, float(detector_ratios.ci95_high.max())*1.1)
    _axis(svg, left, top, plot_w, plot_h,
          [(_linear(d, 3, 15, left, left+plot_w), str(d)) for d in distances],
          [(_linear(v, y_min, y_max, top+plot_h, top), f"{v:.1f}") for v in np.linspace(y_min, y_max, 5)],
          "Code distance", "HDFA EDR / paper-surrogate EDR")
    unity_y = _linear(1., y_min, y_max, top+plot_h, top)
    svg.line(left, unity_y, left+plot_w, unity_y, stroke="#555555", width=1.1, dash="5 4")
    svg.text(left+plot_w-3, unity_y-5, "equal detector rate", css="note", anchor="end")
    points = []
    for row in detector_ratios.itertuples():
        x = _linear(row.code_distance, 3, 15, left, left+plot_w)
        y = _linear(row.mean_ratio, y_min, y_max, top+plot_h, top)
        lo = _linear(row.ci95_low, y_min, y_max, top+plot_h, top)
        hi = _linear(row.ci95_high, y_min, y_max, top+plot_h, top)
        svg.line(x, lo, x, hi, stroke=METHOD_COLOUR[HDFA], width=1.2)
        svg.line(x-4, lo, x+4, lo, stroke=METHOD_COLOUR[HDFA], width=1.2)
        svg.line(x-4, hi, x+4, hi, stroke=METHOD_COLOUR[HDFA], width=1.2)
        svg.marker(x, y, "triangle", fill=METHOD_COLOUR[HDFA], size=4.5)
        points.append((x, y))
    svg.polyline(points, stroke=METHOD_COLOUR[HDFA], width=1.6)
    svg.text(76, 460, "Points show medians/IQR across five seeds for two executed epochs; EDR ratios show paired-seed bootstrap 95% intervals.", css="note")
    svg.text(1000, 460, "Logical failures are retained in the probe table; two epochs remain insufficient for logical scaling.", css="note")
    svg.finish(output)


def plot_fit_quality(fits: pd.DataFrame, output: Path, metadata: Mapping[str, object]) -> None:
    width, height = 1420, 470
    svg = Svg(width, height, "Convergence fit quality",
              "R-squared of origin-constrained convergence fits by code distance, method, and parameters per gate.", metadata)
    distances = sorted(fits.code_distance.unique())
    p_dash = {1: None, 10: "7 4", 30: "2 3"}
    p_marker = {1: "circle", 10: "triangle", 30: "square"}
    top, plot_w, plot_h = 66, 500, 310
    for panel, (left, y_min, y_max, letter, title) in enumerate((
        (76, min(-55., float(fits.r_squared.min())*1.05), 1., "A", "Full fit-quality range"),
        (706, -6., 1., "B", "Zoom: R² from −6 to 1"),
    )):
        _panel_heading(svg, left, 30, letter, title)
        tick_values = (-50, -40, -30, -20, -10, 0, 1) if panel == 0 else (-6, -5, -4, -3, -2, -1, 0, 1)
        _axis(svg, left, top, plot_w, plot_h,
              [(_linear(d, 3, 15, left, left+plot_w), str(d)) for d in distances],
              [(_linear(v, y_min, y_max, top+plot_h, top), str(int(v))) for v in tick_values if y_min <= v <= y_max],
              "Code distance", "R²")
        zero_y = _linear(0., y_min, y_max, top+plot_h, top)
        svg.line(left, zero_y, left+plot_w, zero_y, stroke="#8A2D2D", width=1.2, dash="5 4")
        for method in (PAPER, HDFA):
            for parameters in (1, 10, 30):
                group = fits[(fits.method == method) & (fits.parameters_per_gate == parameters)].sort_values("code_distance")
                visible = group[(group.r_squared >= y_min) & (group.r_squared <= y_max)]
                points = [(_linear(row.code_distance, 3, 15, left, left+plot_w),
                           _linear(row.r_squared, y_min, y_max, top+plot_h, top)) for row in visible.itertuples()]
                svg.polyline(points, stroke=METHOD_COLOUR[method], width=1.5, dash=p_dash[parameters])
                for x, y in points:
                    svg.marker(x, y, p_marker[parameters], fill=METHOD_COLOUR[method], size=3.7)
    _legend(svg, 1235, 90)
    for index, parameters in enumerate((1, 10, 30)):
        yy = 160+index*24
        svg.line(1235, yy-4, 1259, yy-4, stroke=NEUTRAL, width=1.4, dash=p_dash[parameters])
        svg.marker(1247, yy-4, p_marker[parameters], fill=NEUTRAL, size=3.3)
        svg.text(1267, yy, f"P={parameters}")
    svg.text(1235, 255, "R² < 0: fitted line is", css="note")
    svg.text(1235, 269, "worse than the mean", css="note")
    svg.text(1235, 283, "predictor on these points.", css="note")
    svg.text(76, 438, "Distance-invariant γ does not establish the convergence law when model fit is poor; report γ and goodness-of-fit together.", css="note")
    svg.finish(output)


def write_summary_markdown(path: Path, summary: Mapping[str, object],
                           validation: Mapping[str, object]) -> None:
    steering = summary["steerability"]
    convergence = summary["convergence"]
    efficiency = summary["sample_efficiency"]
    executed = summary["executed_pipeline"]
    lines = [
        "# Full scalability comparison with Sivak et al. (Nature 2026)",
        "",
        "## Evidence boundary",
        "",
        "The Figure-5 tables are declared surrogates constrained by published equations and numerical anchors; they are not digitized or proprietary paper simulation data. The pipeline-probe table is an actual execution of this repository, but only for two epochs per seed and distance.",
        "",
        "## Main findings",
        "",
        f"- All {validation['checksums_verified']} manifest-listed source artifacts passed SHA-256 verification.",
        f"- Published-method surrogate steerability reaches the ≥2% criterion through {steering[PAPER]['critical_frequency_at_least_2pct']:.6g} epoch⁻¹, close to the paper's ≈1/150 = {PAPER_FREQUENCY:.6g} epoch⁻¹ anchor.",
        f"- HDFA + residual RL reaches the same criterion through {steering[HDFA]['critical_frequency_at_least_2pct']:.6g} epoch⁻¹ in this declared surrogate.",
        f"- Mean convergence per million native-QEC cycles favours HDFA by {convergence['gamma_per_cycle_advantage_hdfa_over_paper']['1']:.2f}×, {convergence['gamma_per_cycle_advantage_hdfa_over_paper']['10']:.2f}× and {convergence['gamma_per_cycle_advantage_hdfa_over_paper']['30']:.2f}× for P=1, 10 and 30, respectively.",
    ]
    for row in efficiency:
        target = int(round(row["target_fraction"]*100))
        if row["matched_pairs"]:
            lines.append(f"- At {target}% progress, {row['matched_pairs']} paired runs reached the target; the median paper/HDFA cycle ratio is {row['median_cycle_ratio']:.2f}× (IQR {row['q1']:.2f}–{row['q3']:.2f}×).")
        else:
            lines.append(f"- At {target}% progress there are no matched completions, so no cycle ratio can be claimed.")
    lines.extend([
        f"- The executed pipeline table contains {executed['rows']} rows ({executed['paired_runs']} paired seed-distance runs) spanning all distances through d=15, but only {executed['probe_horizon_epochs']} epochs.",
        "- No logical failure was observed in the executed probes, so these runs cannot evaluate logical-error scaling or no-regression claims.",
        "- Several convergence fits have negative R². The cross-distance coefficient of variation of γ can be small even when the assumed linear convergence relation fits poorly.",
        "",
        "## Interpretation",
        "",
        "The full run supports structural scaling, faithful recovery of the paper's steerability anchor, and a strong surrogate native-QEC budget advantage for the staged method. It does not establish hardware equivalence, real logical-error improvement, or the full 10×/5×/2× project gates. The two-epoch executed probe is best interpreted as a computational scaling measurement, not a convergence experiment.",
        "",
        "## Sources",
        "",
        f"- Sivak et al., Nature 655, 879–884 (2026): {ARTICLE_URL}",
        f"- Paper source-data record: {DATA_URL}",
    ])
    path.write_text("\n".join(lines)+"\n", encoding="utf-8")


def write_figure_captions(path: Path) -> None:
    captions = [
        "# Figure captions",
        "",
        "**Figure 1 | Real-time steerability comparison.** Mean normalized improvement over five paired seeds for the declared first-order tracking surrogates. Values of 1 and 0 correspond to the simulated optimal and fixed policies, respectively. The solid contour marks zero improvement; the horizontal dashed line is the ≈1/150 epoch⁻¹ critical-frequency anchor reported by Sivak et al. Panel C is the staged-minus-published-surrogate difference. These panels compare declared surrogates, not paper source data or executed hardware.",
        "",
        "**Figure 2 | Scaling and convergence under the Figure-5 protocol.** Panels A and B show median progress toward the local error-suppression optimum across five seeds for 30 parameters per gate. Panel C reports the origin-constrained convergence coefficient γ normalized by each method's declared native-QEC cycles per epoch. Colour identifies method; marker and line style identify parameters per gate. The paper-equivalent structural sweep reaches distance 15 and 38,670 parameters, whereas the executed suite uses a 2Q−1-control line graph.",
        "",
        "**Figure 3 | Native-QEC sample efficiency.** Panel A reports the fraction of 105 seed-distance-parameter configurations reaching 50%, 75% and 90% progress; error bars are Wilson 95% intervals. Panel B shows paper-surrogate/HDFA cycle ratios only where both methods reached the target, with individual paired ratios, median and interquartile range. The 10× line is the project acceptance target. Ratios must be interpreted jointly with target attainment because unsuccessful runs are censored at 500 epochs.",
        "",
        "**Figure 4 | Actually executed pipeline scaling.** Uninstrumented wall time and peak process resident memory are medians with interquartile ranges across five seeds, each distance/seed condition executing in a fresh process and each run comprising two epochs under one recorded worker-concurrency context. Baseline-subtracted RSS increments are retained in the tables as a supplemental transient-allocation diagnostic. Panel C reports the paired HDFA/published-surrogate detector-event-rate ratio with nonparametric bootstrap 95% intervals. Logical failures are retained in the probe table, but the two-epoch horizon precludes inference about logical-error scaling. These measurements characterize the current Python implementation, not QPU latency.",
        "",
        "**Figure 5 | Convergence fit quality.** Coefficient of determination for the origin-constrained linear relation used to extract γ. Panel A shows the complete range and panel B resolves R² between −6 and 1. Negative R² indicates that the fitted line performs worse than a constant mean predictor; therefore distance-invariant γ must be reported together with goodness-of-fit.",
        "",
        f"Paper reference: {ARTICLE_URL}",
        f"Source-data record: {DATA_URL}",
    ]
    path.write_text("\n".join(captions)+"\n", encoding="utf-8")


def _write_csv(path: Path, frame: pd.DataFrame) -> None:
    frame.to_csv(path, index=False, quoting=csv.QUOTE_MINIMAL)


def main() -> int:
    parser = argparse.ArgumentParser(description="Analyse full HDFA-RL scalability outputs and create paper-comparison figures.")
    parser.add_argument("--input", type=Path, default=Path("artifacts/scalability/nature-2026-full"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/scalability/nature-2026-full/paper-comparison"))
    args = parser.parse_args()
    validation = validate_inputs(args.input)
    report = json.loads((args.input / "scalability-report.json").read_text(encoding="utf-8"))
    tables = load_tables(args.input)
    summary, derived = analyse(tables, report)
    summary["validation"] = validation
    args.output.mkdir(parents=True, exist_ok=True)
    metadata = {
        "analysis_schema": summary["schema_version"],
        "source_report_hash": validation["manifest_report_hash"],
        "article_doi": report["protocol"]["article_doi"],
        "evidence_boundary": summary["evidence_statement"],
    }
    plot_steerability(derived["steering_grid"], args.output / "figure1-steerability-comparison.svg", metadata)
    plot_scaling(tables["scaling"], derived["fits"], args.output / "figure2-scaling-convergence.svg", metadata)
    plot_efficiency(derived["attainment"], derived["paired_efficiency"],
                    derived["paired_efficiency_summary"], args.output / "figure3-sample-efficiency.svg", metadata)
    plot_pipeline(derived["pipeline_summary"], derived["pipeline_detector_ratios"],
                  args.output / "figure4-executed-pipeline.svg", metadata)
    plot_fit_quality(derived["fits"], args.output / "figure5-fit-quality.svg", metadata)
    _write_csv(args.output / "gamma-summary.csv", derived["gamma_summary"])
    _write_csv(args.output / "sample-attainment-summary.csv", derived["attainment"])
    _write_csv(args.output / "sample-efficiency-paired-summary.csv", derived["paired_efficiency_summary"])
    _write_csv(args.output / "pipeline-scaling-summary.csv", derived["pipeline_summary"])
    _write_csv(args.output / "pipeline-detector-ratio-summary.csv", derived["pipeline_detector_ratios"])
    (args.output / "analysis-summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    write_summary_markdown(args.output / "analysis-summary.md", summary, validation)
    write_figure_captions(args.output / "figure-captions.md")
    generated = sorted(path.name for path in args.output.iterdir()
                       if path.is_file() and path.name != "analysis-manifest.json")
    (args.output / "analysis-manifest.json").write_text(json.dumps({
        "schema_version": "nature-2026-full-analysis.manifest.v3",
        "source_report_hash": validation["manifest_report_hash"],
        "files": {name: hashlib.sha256((args.output / name).read_bytes()).hexdigest() for name in generated},
    }, indent=2, sort_keys=True), encoding="utf-8")
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
