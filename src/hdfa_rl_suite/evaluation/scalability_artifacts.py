"""Machine-readable tables and dependency-free SVG figures for scalability reports."""
from __future__ import annotations

from dataclasses import asdict
import csv
import hashlib
import html
import json
import math
from pathlib import Path
from typing import Iterable, Mapping, Sequence

from .scalability import (
    HDFA_METHOD,
    PAPER_METHOD,
    ConvergenceFit,
    ConvergencePoint,
    PipelineProbeFailure,
    PipelineProbePoint,
    ResourcePoint,
    ScalabilityReport,
    ScalingPoint,
    SteerabilityPoint,
)


_METHOD_LABELS = {
    PAPER_METHOD: "Published-method reimplementation",
    HDFA_METHOD: "Predictive HDFA + residual RL",
}
_DISTANCE_COLOURS = ("#d66a4a", "#e4a64d", "#b8a85d", "#70a47d", "#218b85", "#167287", "#123d54")


class _Svg:
    def __init__(self, width: int, height: int, title: str, description: str) -> None:
        self.width, self.height = width, height
        self.parts = [
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" role="img">',
            f"<title>{html.escape(title)}</title><desc>{html.escape(description)}</desc>",
            '<rect width="100%" height="100%" fill="#ffffff"/>',
        ]

    def line(self, x1: float, y1: float, x2: float, y2: float, *, stroke: str = "#252525",
             width: float = 1., dash: str | None = None) -> None:
        extra = f' stroke-dasharray="{dash}"' if dash else ""
        self.parts.append(f'<line x1="{x1:.2f}" y1="{y1:.2f}" x2="{x2:.2f}" y2="{y2:.2f}" '
                          f'stroke="{stroke}" stroke-width="{width}"{extra}/>' )

    def rect(self, x: float, y: float, width: float, height: float, *, fill: str = "none",
             stroke: str = "none", stroke_width: float = 1.) -> None:
        self.parts.append(f'<rect x="{x:.2f}" y="{y:.2f}" width="{width:.2f}" height="{height:.2f}" '
                          f'fill="{fill}" stroke="{stroke}" stroke-width="{stroke_width}"/>')

    def circle(self, x: float, y: float, radius: float, *, fill: str, stroke: str = "none") -> None:
        self.parts.append(f'<circle cx="{x:.2f}" cy="{y:.2f}" r="{radius:.2f}" fill="{fill}" stroke="{stroke}"/>')

    def polyline(self, points: Sequence[tuple[float, float]], *, stroke: str, width: float = 1.5) -> None:
        if points:
            encoded = " ".join(f"{x:.2f},{y:.2f}" for x, y in points)
            self.parts.append(f'<polyline points="{encoded}" fill="none" stroke="{stroke}" stroke-width="{width}"/>')

    def text(self, x: float, y: float, value: str, *, size: int = 12, anchor: str = "start",
             fill: str = "#252525", weight: str = "400", rotate: float | None = None) -> None:
        transform = f' transform="rotate({rotate:.1f} {x:.2f} {y:.2f})"' if rotate is not None else ""
        self.parts.append(f'<text x="{x:.2f}" y="{y:.2f}" font-family="Arial,sans-serif" font-size="{size}" '
                          f'font-weight="{weight}" text-anchor="{anchor}" fill="{fill}"{transform}>'
                          f'{html.escape(value)}</text>')

    def finish(self) -> str:
        return "\n".join(self.parts + ["</svg>"])


def _write_csv(path: Path, rows: Iterable[object]) -> None:
    payload = [asdict(row) for row in rows]
    if not payload:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=tuple(payload[0]))
        writer.writeheader()
        writer.writerows(payload)


def _scale(value: float, lower: float, upper: float, start: float, stop: float, *, log: bool = False) -> float:
    if log:
        value, lower, upper = math.log10(value), math.log10(lower), math.log10(upper)
    fraction = (value-lower) / max(upper-lower, 1e-15)
    return start + fraction * (stop-start)


def _diverging(value: float) -> str:
    value = max(-1., min(1., value))
    if value >= 0:
        t = value
        return f"rgb({round(238-190*t)},{round(232-100*t)},{round(211-75*t)})"
    t = -value
    return f"rgb({round(238-35*t)},{round(232-155*t)},{round(211-79*t)})"


def _epoch_colour(value: float) -> str:
    value = max(0., min(1., value))
    return f"rgb({round(16+232*value)},{round(61+99*value)},{round(84-45*value)})"


def _aggregate_steering(rows: Sequence[SteerabilityPoint]) -> Mapping[tuple[str, float, float], float]:
    grouped: dict[tuple[str, float, float], list[float]] = {}
    for row in rows:
        grouped.setdefault((row.method, row.drift_frequency, row.entropy_regularization), []).append(row.stochastic_improvement)
    return {key: sum(values)/len(values) for key, values in grouped.items()}


def _plot_steerability(report: ScalabilityReport, path: Path) -> None:
    svg = _Svg(1120, 520, "Real-time steerability phase diagram",
               "Mean normalized improvement versus drift frequency and entropy regularization for both methods.")
    values = _aggregate_steering(report.steerability)
    frequencies = sorted(report.config.steering_frequencies)
    entropies = sorted(report.config.entropy_regularizations)
    for panel, method in enumerate((PAPER_METHOD, HDFA_METHOD)):
        left, top, width, height = 80 + panel*535, 70, 430, 350
        svg.text(left, 38, _METHOD_LABELS[method], size=17, weight="500")
        cell_w, cell_h = width/len(entropies), height/len(frequencies)
        for xi, entropy in enumerate(entropies):
            for yi, frequency in enumerate(frequencies):
                value = values[(method, frequency, entropy)]
                x, y = left + xi*cell_w, top + height-(yi+1)*cell_h
                svg.rect(x, y, cell_w+.3, cell_h+.3, fill=_diverging(value),
                         stroke="#222222" if value >= 0 else "none", stroke_width=.45)
        threshold_y = _scale(report.protocol.empirical_steerability_frequency,
                             frequencies[0], frequencies[-1], top+height, top, log=True)
        svg.line(left, threshold_y, left+width, threshold_y, stroke="#111111", width=1.2, dash="6 4")
        svg.text(left+width-4, threshold_y-5, "paper ≈ 1/150 epoch⁻¹", size=11, anchor="end")
        svg.line(left, top+height, left+width, top+height)
        svg.line(left, top, left, top+height)
        for index in (0, len(entropies)//2, len(entropies)-1):
            x = left + (index+.5)*cell_w
            svg.text(x, top+height+22, f"{entropies[index]:.0e}", anchor="middle", size=11)
        for index in (0, len(frequencies)//2, len(frequencies)-1):
            y = top + height-(index+.5)*cell_h
            svg.text(left-9, y+4, f"{frequencies[index]:.0e}", anchor="end", size=11)
        svg.text(left+width/2, top+height+48, "Entropy regularization", anchor="middle", size=13)
        if panel == 0:
            svg.text(left-58, top+height/2, "Drift frequency (epoch⁻¹)", anchor="middle", size=13, rotate=-90)
    legend_x = 1015
    for index in range(100):
        value = 1-2*index/99
        svg.rect(legend_x, 110+index*2.6, 20, 2.7, fill=_diverging(value))
    svg.text(1040, 119, "+1 optimal", size=11)
    svg.text(1040, 245, "0 fixed", size=11)
    svg.text(1040, 372, "−1 harmful", size=11)
    svg.text(1000, 88, "Normalized improvement", size=12)
    path.write_text(svg.finish(), encoding="utf-8")


def _mean_scaling(rows: Sequence[ScalingPoint], parameters: int) -> Mapping[tuple[str, int, int], tuple[float, float]]:
    grouped: dict[tuple[str, int, int], list[tuple[float, float]]] = {}
    for row in rows:
        if row.parameters_per_gate == parameters:
            grouped.setdefault((row.method, row.code_distance, row.epoch), []).append(
                (row.physical_error_rate, row.logical_error_rate))
    return {key: (sum(x for x, _ in values)/len(values), sum(y for _, y in values)/len(values))
            for key, values in grouped.items()}


def _plot_scaling(report: ScalabilityReport, path: Path) -> None:
    parameters = max(report.config.parameters_per_gate)
    aggregated = _mean_scaling(report.scaling, parameters)
    rows = [(physical, logical) for (method, _, _), (physical, logical) in aggregated.items()
            if method in (PAPER_METHOD, HDFA_METHOD)]
    x_min, x_max = min(x for x, _ in rows)*.9, max(x for x, _ in rows)*1.1
    y_min, y_max = max(1e-12, min(y for _, y in rows)*.5), min(.6, max(y for _, y in rows)*1.5)
    svg = _Svg(1120, 540, "Logical-error scaling during calibration",
               "Logical error rate versus physical error rate, coloured by epoch, at the largest parameters-per-gate setting.")
    for panel, method in enumerate((PAPER_METHOD, HDFA_METHOD)):
        left, top, width, height = 85+panel*535, 65, 430, 375
        svg.text(left, 35, f"{_METHOD_LABELS[method]} — P={parameters}", size=17, weight="500")
        for distance_index, distance in enumerate(report.config.distances):
            data = [(epoch, *aggregated[(method, distance, epoch)]) for epoch in range(report.config.epochs+1)]
            points = [(_scale(x, x_min, x_max, left, left+width, log=True),
                       _scale(y, y_min, y_max, top+height, top, log=True)) for _, x, y in data]
            svg.polyline(points, stroke=_DISTANCE_COLOURS[distance_index % len(_DISTANCE_COLOURS)], width=1.2)
            stride = max(1, report.config.epochs//18)
            for epoch, x, y in data[::stride]:
                svg.circle(_scale(x, x_min, x_max, left, left+width, log=True),
                           _scale(y, y_min, y_max, top+height, top, log=True), 2.8,
                           fill=_epoch_colour(epoch/max(1, report.config.epochs)))
            x, y = points[0]
            svg.text(x+5, y-3, f"d={distance}", size=10,
                     fill=_DISTANCE_COLOURS[distance_index % len(_DISTANCE_COLOURS)])
        threshold_x = _scale(report.config.physical_error_threshold, x_min, x_max, left, left+width, log=True)
        svg.line(threshold_x, top, threshold_x, top+height, stroke="#ba3030", dash="5 4")
        svg.text(threshold_x-4, top+15, "1.79×10⁻³", size=10, anchor="end", fill="#9a2020")
        svg.line(left, top+height, left+width, top+height)
        svg.line(left, top, left, top+height)
        for value in (x_min, report.config.physical_error_threshold, x_max):
            svg.text(_scale(value, x_min, x_max, left, left+width, log=True), top+height+22,
                     f"{value:.1e}", anchor="middle", size=10)
        for value in (y_min, math.sqrt(y_min*y_max), y_max):
            svg.text(left-8, _scale(value, y_min, y_max, top+height, top, log=True)+4,
                     f"{value:.0e}", anchor="end", size=10)
        svg.text(left+width/2, top+height+48, "Mean physical error rate", anchor="middle", size=13)
        if panel == 0:
            svg.text(left-62, top+height/2, "Logical error rate", anchor="middle", size=13, rotate=-90)
    svg.text(930, 487, "Epoch: 0", size=11)
    for index in range(70):
        svg.rect(985+index, 475, 1.2, 13, fill=_epoch_colour(index/69))
    svg.text(1075, 487, str(report.config.epochs), size=11)
    path.write_text(svg.finish(), encoding="utf-8")


def _plot_convergence(report: ScalabilityReport, path: Path) -> None:
    parameters_values = tuple(sorted(report.config.parameters_per_gate))
    panel_width, panel_height = 320, 245
    width, height = 75 + panel_width*len(parameters_values), 90 + panel_height*2
    svg = _Svg(width, height, "Convergence-rate scaling",
               "Normalized speed against distance to local optimum, faceted by method and parameters per gate.")
    fits = {(fit.method, fit.code_distance, fit.parameters_per_gate): fit for fit in report.fits}
    for row_index, method in enumerate((PAPER_METHOD, HDFA_METHOD)):
        for column, parameters in enumerate(parameters_values):
            left, top = 62+column*panel_width, 55+row_index*panel_height
            plot_w, plot_h = panel_width-42, panel_height-65
            first_seed = report.config.seeds[0]
            epoch_stride = max(1, report.config.epochs//80)
            selected = [row for row in report.convergence if row.method == method
                        and row.parameters_per_gate == parameters and row.seed == first_seed
                        and row.epoch % epoch_stride == 0]
            x_max = max((row.distance_to_local_optimum for row in selected), default=1.)
            y_max = max((100*row.normalized_speed for row in selected), default=1.)
            for row in selected:
                distance_index = report.config.distances.index(row.code_distance)
                svg.circle(_scale(row.distance_to_local_optimum, 0., x_max, left, left+plot_w),
                           _scale(100*row.normalized_speed, 0., y_max, top+plot_h, top), 1.7,
                           fill=_DISTANCE_COLOURS[distance_index % len(_DISTANCE_COLOURS)])
            for distance_index, distance in enumerate(report.config.distances):
                fit = fits.get((method, distance, parameters))
                if fit and math.isfinite(fit.gamma):
                    svg.line(left, top+plot_h,
                             left+plot_w, _scale(100*fit.gamma*x_max, 0., y_max, top+plot_h, top),
                             stroke=_DISTANCE_COLOURS[distance_index % len(_DISTANCE_COLOURS)], width=1.)
            svg.line(left, top+plot_h, left+plot_w, top+plot_h)
            svg.line(left, top, left, top+plot_h)
            svg.text(left+plot_w/2, top+plot_h+34, "Distance to local optimum", anchor="middle", size=11)
            svg.text(left-42, top+plot_h/2, "Speed ×100", anchor="middle", size=11, rotate=-90)
            svg.text(left+4, top+17, f"P={parameters}", size=13, weight="500")
            if column == 0:
                svg.text(left, top-16, _METHOD_LABELS[method], size=14, weight="500")
            svg.text(left, top+plot_h+16, "0", size=9, anchor="middle")
            svg.text(left+plot_w, top+plot_h+16, f"{x_max:.2f}", size=9, anchor="middle")
            svg.text(left-5, top+plot_h+3, "0", size=9, anchor="end")
            svg.text(left-5, top+4, f"{y_max:.2f}", size=9, anchor="end")
    path.write_text(svg.finish(), encoding="utf-8")


def _plot_resources(report: ScalabilityReport, path: Path) -> None:
    selected = [row for row in report.resources if row.parameters_per_gate == max(report.config.parameters_per_gate)]
    x_min, x_max = min(row.control_parameters for row in selected), max(row.control_parameters for row in selected)
    y_min = min(row.estimated_policy_state_bytes for row in selected)
    y_max = max(row.estimated_policy_state_bytes for row in selected)
    svg = _Svg(760, 500, "Controller resource scaling",
               "Estimated sparse controller state versus paper-equivalent control parameter count.")
    left, top, width, height = 95, 55, 560, 350
    for method, colour in ((PAPER_METHOD, "#8a3e3e"), (HDFA_METHOD, "#156b80")):
        rows = sorted((row for row in selected if row.method == method), key=lambda item: item.control_parameters)
        points = [(_scale(row.control_parameters, x_min, x_max, left, left+width, log=True),
                   _scale(row.estimated_policy_state_bytes, y_min, y_max, top+height, top, log=True)) for row in rows]
        svg.polyline(points, stroke=colour, width=2.)
        for row, (x, y) in zip(rows, points):
            svg.circle(x, y, 4., fill=colour)
            svg.text(x+5, y-5, f"d={row.code_distance}", size=10, fill=colour)
    svg.line(left, top+height, left+width, top+height)
    svg.line(left, top, left, top+height)
    svg.text(left+width/2, top+height+44, "Control parameters (log scale)", anchor="middle", size=13)
    svg.text(left-68, top+height/2, "Estimated sparse state bytes (log)", anchor="middle", size=13, rotate=-90)
    svg.line(455, 440, 490, 440, stroke="#8a3e3e", width=2.)
    svg.text(498, 444, _METHOD_LABELS[PAPER_METHOD], size=11)
    svg.line(455, 462, 490, 462, stroke="#156b80", width=2.)
    svg.text(498, 466, _METHOD_LABELS[HDFA_METHOD], size=11)
    path.write_text(svg.finish(), encoding="utf-8")


def _plot_pipeline(rows: Sequence[PipelineProbePoint], path: Path) -> None:
    svg = _Svg(760, 500, "Executed suite throughput probe",
               "Measured Python wall time per interval versus suite control variables.")
    left, top, width, height = 90, 55, 565, 350
    x_min, x_max = min(row.suite_control_variables for row in rows), max(row.suite_control_variables for row in rows)
    y_min = max(1e-6, min(row.elapsed_s for row in rows)); y_max = max(row.elapsed_s for row in rows)
    for method, colour in ((PAPER_METHOD, "#8a3e3e"), (HDFA_METHOD, "#156b80")):
        grouped: dict[int, list[float]] = {}
        for row in rows:
            if row.method == method:
                grouped.setdefault(row.suite_control_variables, []).append(row.elapsed_s)
        points = [(_scale(control, x_min, x_max, left, left+width, log=x_min != x_max),
                   _scale(sum(values)/len(values), y_min, y_max, top+height, top, log=y_min != y_max))
                  for control, values in sorted(grouped.items())]
        svg.polyline(points, stroke=colour, width=2.)
        for x, y in points:
            svg.circle(x, y, 4., fill=colour)
    svg.line(left, top+height, left+width, top+height); svg.line(left, top, left, top+height)
    svg.text(left+width/2, top+height+42, "Suite control variables", anchor="middle", size=13)
    svg.text(left-62, top+height/2, "Wall time per interval (s)", anchor="middle", size=13, rotate=-90)
    path.write_text(svg.finish(), encoding="utf-8")


def write_scalability_artifacts(report: ScalabilityReport, output_directory: Path) -> Mapping[str, str]:
    """Write a report, tidy tables, matched plots and checksum manifest."""
    output_directory.mkdir(parents=True, exist_ok=True)
    paths = {
        "report": output_directory / "scalability-report.json",
        "scaling": output_directory / "fig5b-scaling.csv",
        "convergence": output_directory / "fig5c-convergence.csv",
        "fits": output_directory / "convergence-fits.csv",
        "steerability": output_directory / "fig5a-steerability.csv",
        "resources": output_directory / "resource-scaling.csv",
        "efficiency": output_directory / "sample-efficiency.csv",
        "pipeline": output_directory / "pipeline-probe.csv",
        "pipeline_failures": output_directory / "pipeline-failures.csv",
        "fig5a": output_directory / "fig5a-steerability.svg",
        "fig5b": output_directory / "fig5b-scaling.svg",
        "fig5c": output_directory / "fig5c-convergence.svg",
        "resource_plot": output_directory / "resource-scaling.svg",
    }
    _write_csv(paths["scaling"], report.scaling)
    _write_csv(paths["convergence"], report.convergence)
    _write_csv(paths["fits"], report.fits)
    _write_csv(paths["steerability"], report.steerability)
    _write_csv(paths["resources"], report.resources)
    _write_csv(paths["efficiency"], report.sample_efficiency)
    _write_csv(paths["pipeline"], report.pipeline_probe)
    _write_csv(paths["pipeline_failures"], report.pipeline_failures)
    _plot_steerability(report, paths["fig5a"])
    _plot_scaling(report, paths["fig5b"])
    _plot_convergence(report, paths["fig5c"])
    _plot_resources(report, paths["resource_plot"])
    if report.pipeline_probe:
        paths["pipeline_plot"] = output_directory / "pipeline-probe.svg"
        _plot_pipeline(report.pipeline_probe, paths["pipeline_plot"])
    report_summary = {
        "schema_version": report.schema_version,
        "report_hash": report.report_hash,
        "protocol": asdict(report.protocol),
        "config": asdict(report.config),
        "fits": [asdict(row) for row in report.fits],
        "gates": [asdict(row) for row in report.gates],
        "limitations": report.limitations,
        "environment": dict(report.environment),
        "tables": {
            "scaling": {"path": paths["scaling"].name, "rows": len(report.scaling)},
            "convergence": {"path": paths["convergence"].name, "rows": len(report.convergence)},
            "steerability": {"path": paths["steerability"].name, "rows": len(report.steerability)},
            "resources": {"path": paths["resources"].name, "rows": len(report.resources)},
            "sample_efficiency": {"path": paths["efficiency"].name, "rows": len(report.sample_efficiency)},
            "pipeline_probe": {"path": paths["pipeline"].name, "rows": len(report.pipeline_probe)},
            "pipeline_failures": {"path": paths["pipeline_failures"].name,
                                  "rows": len(report.pipeline_failures)},
            "condition_checkpoints": {
                "path": "checkpoints",
                "rows": len(tuple((output_directory / "checkpoints").glob("*.json")))
                if (output_directory / "checkpoints").exists() else 0,
            },
        },
    }
    paths["report"].write_text(json.dumps(report_summary, indent=2, sort_keys=True), encoding="utf-8")
    manifest = {
        "schema_version": "evaluation.scalability.manifest.v3",
        "report_hash": report.report_hash,
        "article_doi": report.protocol.article_doi,
        "source_data_version_doi": report.protocol.source_data_version_doi,
        "artifacts": {},
    }
    for name, path in paths.items():
        if path.exists():
            content = path.read_bytes()
            manifest["artifacts"][name] = {
                "path": path.name, "bytes": len(content), "sha256": hashlib.sha256(content).hexdigest(),
            }
    checkpoint_directory = output_directory / "checkpoints"
    if checkpoint_directory.exists():
        for checkpoint in sorted(checkpoint_directory.glob("*.json")):
            content = checkpoint.read_bytes()
            manifest["artifacts"][f"checkpoint:{checkpoint.stem}"] = {
                "path": checkpoint.relative_to(output_directory).as_posix(),
                "bytes": len(content),
                "sha256": hashlib.sha256(content).hexdigest(),
            }
    manifest_path = output_directory / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    paths["manifest"] = manifest_path
    return {name: str(path) for name, path in paths.items() if path.exists()}
