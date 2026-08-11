"""Create censoring-aware, paper-style figures from the completed v5 comparison.

The output deliberately separates (1) immutable claims from Sivak et al.,
(2) declared Figure-5 surrogates, and (3) executed HDFA-RL suite evidence.
No cross-layer numerical value is presented as a like-for-like hardware result.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from statistics import fmean, median
from typing import Iterable, Mapping, Sequence

import numpy as np
from PIL import Image, ImageDraw, ImageFont


ARM_ORDER = (
    "fixed",
    "periodic_recalibration",
    "full_control_detector_rl",
    "predictive_hdfa_no_residual",
    "predictive_hdfa_residual_rl",
    "oracle",
)
ARM_LABEL = {
    "fixed": "Fixed",
    "periodic_recalibration": "Periodic",
    "full_control_detector_rl": "Google-style RL",
    "predictive_hdfa_no_residual": "Predictive only",
    "predictive_hdfa_residual_rl": "HDFA + residual RL",
    "oracle": "Oracle",
}
ARM_SHORT = {
    "fixed": "Fixed",
    "periodic_recalibration": "Periodic",
    "full_control_detector_rl": "Full RL",
    "predictive_hdfa_no_residual": "Predictive",
    "predictive_hdfa_residual_rl": "HDFA+RL",
    "oracle": "Oracle",
}
ARM_COLOR = {
    "fixed": "#727A84",
    "periodic_recalibration": "#C28A20",
    "full_control_detector_rl": "#2864B7",
    "predictive_hdfa_no_residual": "#1B8A84",
    "predictive_hdfa_residual_rl": "#7B4AB5",
    "oracle": "#27855B",
}
PAPER_METHOD = "paper_sparse_policy_gradient"
HDFA_METHOD = "predictive_hdfa_residual_rl"
METHOD_LABEL = {PAPER_METHOD: "Google-style RL surrogate", HDFA_METHOD: "HDFA + residual RL surrogate"}
METHOD_COLOR = {PAPER_METHOD: "#2864B7", HDFA_METHOD: "#7B4AB5"}

NAVY = "#13243A"
INK = "#253140"
MUTED = "#66717E"
GRID = "#D9E0E8"
PALE = "#F4F7FA"
RED = "#A23A3A"
GREEN = "#267653"
GOLD = "#A57418"
WHITE = "#FFFFFF"


def _font(size: int, bold: bool = False, italic: bool = False) -> ImageFont.FreeTypeFont:
    if bold and italic:
        name = "arialbi.ttf"
    elif bold:
        name = "arialbd.ttf"
    elif italic:
        name = "ariali.ttf"
    else:
        name = "arial.ttf"
    return ImageFont.truetype(str(Path("C:/Windows/Fonts") / name), size=size)


def _text(draw: ImageDraw.ImageDraw, xy: tuple[float, float], value: str, size: int = 34,
          *, fill: str = INK, bold: bool = False, italic: bool = False,
          anchor: str = "la") -> None:
    draw.text(xy, value, font=_font(size, bold, italic), fill=fill, anchor=anchor)


def _wrapped(draw: ImageDraw.ImageDraw, xy: tuple[int, int], value: str, width: int,
             size: int = 31, *, fill: str = INK, bold: bool = False,
             line_gap: int = 8) -> int:
    words = value.split()
    lines: list[str] = []
    current = ""
    font = _font(size, bold)
    for word in words:
        candidate = f"{current} {word}".strip()
        if current and draw.textbbox((0, 0), candidate, font=font)[2] > width:
            lines.append(current)
            current = word
        else:
            current = candidate
    if current:
        lines.append(current)
    x, y = xy
    for line in lines:
        draw.text((x, y), line, font=font, fill=fill)
        y += size + line_gap
    return y


def _canvas(title: str, subtitle: str, width: int = 3200, height: int = 1800
            ) -> tuple[Image.Image, ImageDraw.ImageDraw]:
    image = Image.new("RGB", (width, height), WHITE)
    draw = ImageDraw.Draw(image)
    _text(draw, (110, 72), title, 58, fill=NAVY, bold=True)
    _text(draw, (112, 142), subtitle, 29, fill=MUTED)
    draw.line((110, 194, width - 110, 194), fill=GRID, width=3)
    return image, draw


def _panel(draw: ImageDraw.ImageDraw, box: tuple[int, int, int, int], letter: str,
           title: str) -> tuple[int, int, int, int]:
    left, top, right, bottom = box
    draw.rounded_rectangle(box, radius=18, fill=WHITE, outline=GRID, width=2)
    draw.rounded_rectangle((left + 22, top + 20, left + 74, top + 72), radius=9,
                           fill=NAVY)
    _text(draw, (left + 48, top + 46), letter, 29, fill=WHITE, bold=True, anchor="mm")
    _text(draw, (left + 92, top + 47), title, 35, fill=NAVY, bold=True, anchor="lm")
    return left + 76, top + 105, right - 38, bottom - 60


def _axis(draw: ImageDraw.ImageDraw, plot: tuple[int, int, int, int],
          *, x_ticks: Sequence[tuple[float, str]] = (),
          y_ticks: Sequence[tuple[float, str]] = (), x_label: str = "", y_label: str = "",
          y_grid: bool = True) -> None:
    left, top, right, bottom = plot
    if y_grid:
        for y, label in y_ticks:
            draw.line((left, y, right, y), fill=GRID, width=2)
            _text(draw, (left - 14, y), label, 24, fill=MUTED, anchor="rm")
    draw.line((left, top, left, bottom), fill=INK, width=3)
    draw.line((left, bottom, right, bottom), fill=INK, width=3)
    for x, label in x_ticks:
        draw.line((x, bottom, x, bottom + 8), fill=INK, width=2)
        _text(draw, (x, bottom + 18), label, 23, fill=MUTED, anchor="ma")
    if x_label:
        _text(draw, ((left + right) / 2, bottom + 56), x_label, 26, fill=MUTED, anchor="ma")
    if y_label:
        # Pillow's anchor support keeps rotated labels reliable.
        label = Image.new("RGBA", (bottom - top, 50), (255, 255, 255, 0))
        d = ImageDraw.Draw(label)
        d.text(((bottom - top) / 2, 25), y_label, font=_font(26), fill=MUTED,
               anchor="mm")
        label = label.rotate(90, expand=True)
        draw._image.paste(label, (left - 74, int((top + bottom - label.height) / 2)), label)


def _lin(value: float, low: float, high: float, start: float, stop: float) -> float:
    return start + (value - low) / max(1e-15, high - low) * (stop - start)


def _log(value: float, low: float, high: float, start: float, stop: float) -> float:
    return _lin(math.log10(value), math.log10(low), math.log10(high), start, stop)


def _save(image: Image.Image, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path, dpi=(300, 300), optimize=True)


def _percentile(values: Sequence[float], q: float) -> float:
    return float(np.quantile(np.asarray(values, dtype=float), q))


def _cluster_bootstrap_fraction(rows: Sequence[Mapping[str, object]], status: str,
                                seed: int = 20260801) -> tuple[float, float, float]:
    by_seed: dict[int, list[float]] = defaultdict(list)
    for row in rows:
        by_seed[int(row["seed"])].append(1.0 if row["completion_status"] == status else 0.0)
    clusters = np.asarray([fmean(values) for _, values in sorted(by_seed.items())])
    rng = np.random.default_rng(seed)
    samples = clusters[rng.integers(0, len(clusters), size=(20_000, len(clusters)))].mean(axis=1)
    return float(clusters.mean()), float(np.quantile(samples, .025)), float(np.quantile(samples, .975))


def _write_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _arm_summary(report: Mapping[str, object]) -> list[dict[str, object]]:
    metrics = report["metrics"]
    output: list[dict[str, object]] = []
    for arm in ARM_ORDER:
        rows = [row for row in metrics if row["arm"] == arm]
        complete, low, high = _cluster_bootstrap_fraction(rows, "completed")
        output.append({
            "arm": arm,
            "label": ARM_LABEL[arm],
            "runs": len(rows),
            "completed": sum(row["completion_status"] == "completed" for row in rows),
            "censored": sum(row["completion_status"] == "censored" for row in rows),
            "completion_fraction": complete,
            "completion_ci95_low": low,
            "completion_ci95_high": high,
            "lifecycle_violations": sum(int(row["lifecycle_violation_count"]) for row in rows),
            "mean_bootstrap_count": fmean(float(row["bootstrap_count"]) for row in rows),
            "mean_qec_cycles": fmean(float(row["qec_cycles"]) for row in rows),
            "mean_candidate_evaluations": fmean(float(row["candidate_evaluations"]) for row in rows),
            "mean_detector_event_rate": fmean(float(row["detector_event_rate"]) for row in rows),
            "mean_integrated_excess_detector_events": fmean(
                float(row["integrated_excess_detector_events"]) for row in rows),
            "mean_exploration_damage": fmean(float(row["exploration_damage"]) for row in rows),
            "mean_logical_failure_probability": fmean(
                float(row["logical_circuit_failure_probability"]) for row in rows),
            "mean_logical_error_per_round": fmean(float(row["logical_error_per_round"]) for row in rows),
        })
    return output


def _trajectory_summary(report: Mapping[str, object]) -> list[dict[str, object]]:
    output: list[dict[str, object]] = []
    trajectories = report["trajectories"]
    for arm in ("full_control_detector_rl", "predictive_hdfa_residual_rl"):
        rows = [row for row in trajectories if row["arm"] == arm]
        for interval in sorted({int(row["interval"]) for row in rows}):
            values = [float(row["detector_rate"]) for row in rows if int(row["interval"]) == interval]
            output.append({
                "arm": arm,
                "interval": interval,
                "at_risk": len(values),
                "median_detector_rate": median(values),
                "q1_detector_rate": _percentile(values, .25),
                "q3_detector_rate": _percentile(values, .75),
            })
    return output


def _figure_effectiveness(arm_summary: Sequence[Mapping[str, object]],
                          recovery: Sequence[Mapping[str, object]], output: Path) -> None:
    image, draw = _canvas(
        "Authoritative Stage 0-7 effectiveness study",
        "Five scenarios x five seeds; uncertainty clusters scenario replicates by independent seed",
    )
    a = _panel(draw, (80, 235, 1060, 1720), "A", "Completion without censoring")
    b = _panel(draw, (1110, 235, 2140, 1720), "B", "Observed recovery attainment")
    c = _panel(draw, (2190, 235, 3120, 1720), "C", "Lifecycle and recalibration burden")

    # Panel A
    left, top, right, bottom = a
    plot = (left + 5, top + 20, right - 5, bottom - 145)
    y_ticks = [(_lin(v, 0, 1, plot[3], plot[1]), f"{int(v*100)}") for v in (0, .25, .5, .75, 1)]
    _axis(draw, plot, y_ticks=y_ticks, y_label="Completed runs (%)")
    bar_w = (plot[2] - plot[0]) / len(ARM_ORDER) * .55
    for index, row in enumerate(arm_summary):
        x = _lin(index + .5, 0, len(ARM_ORDER), plot[0], plot[2])
        y = _lin(float(row["completion_fraction"]), 0, 1, plot[3], plot[1])
        lo = _lin(float(row["completion_ci95_low"]), 0, 1, plot[3], plot[1])
        hi = _lin(float(row["completion_ci95_high"]), 0, 1, plot[3], plot[1])
        draw.rounded_rectangle((x - bar_w/2, y, x + bar_w/2, plot[3]), radius=9,
                               fill=ARM_COLOR[str(row["arm"])])
        draw.line((x, lo, x, hi), fill=INK, width=3)
        draw.line((x - 10, lo, x + 10, lo), fill=INK, width=3)
        draw.line((x - 10, hi, x + 10, hi), fill=INK, width=3)
        _text(draw, (x, y - 15), f"{int(row['completed'])}/25", 26, bold=True, anchor="mb")
        _text(draw, (x, plot[3] + 25), ARM_SHORT[str(row["arm"])], 24,
              fill=MUTED, anchor="ma")
    _wrapped(draw, (plot[0], plot[3] + 84),
             "The primary HDFA + residual-RL arm completed 9/25 runs. Sixteen runs were censored after failed Stage-0 OOD re-entry; the Google-style full-control RL comparator completed 25/25.",
             plot[2] - plot[0], 25, fill=MUTED)

    # Panel B
    left, top, right, bottom = b
    plot = (left + 10, top + 20, right - 15, bottom - 150)
    y_ticks = [(_lin(v, 0, 1, plot[3], plot[1]), f"{int(v*100)}") for v in (0, .25, .5, .75, 1)]
    x_positions = {target: _lin(index, 0, 2, plot[0] + 80, plot[2] - 80)
                   for index, target in enumerate((.5, .75, .9))}
    _axis(draw, plot,
          x_ticks=[(x_positions[t], f"{int(t*100)}%") for t in (.5, .75, .9)],
          y_ticks=y_ticks, x_label="Recovery target", y_label="Runs reaching target (%)")
    for arm in ARM_ORDER:
        rows = sorted((row for row in recovery if row["arm"] == arm),
                      key=lambda row: row["target_fraction"])
        points = []
        for row in rows:
            x = x_positions[float(row["target_fraction"])]
            y = _lin(float(row["reached_fraction"]), 0, 1, plot[3], plot[1])
            lo = _lin(float(row["reached_fraction_ci95"][0]), 0, 1, plot[3], plot[1])
            hi = _lin(float(row["reached_fraction_ci95"][1]), 0, 1, plot[3], plot[1])
            draw.line((x, lo, x, hi), fill=ARM_COLOR[arm], width=2)
            draw.ellipse((x - 7, y - 7, x + 7, y + 7), fill=ARM_COLOR[arm], outline=WHITE, width=2)
            points.append((x, y))
        draw.line(points, fill=ARM_COLOR[arm], width=4)
    legend_y = plot[3] + 88
    for index, arm in enumerate(ARM_ORDER):
        x = plot[0] + (index % 3) * 285
        y = legend_y + (index // 3) * 42
        draw.line((x, y, x + 35, y), fill=ARM_COLOR[arm], width=5)
        _text(draw, (x + 45, y), ARM_SHORT[arm], 22, fill=MUTED, anchor="lm")

    # Panel C
    left, top, right, bottom = c
    plot = (left + 20, top + 20, right - 20, bottom - 160)
    max_v = max(float(row["mean_bootstrap_count"]) for row in arm_summary)
    y_ticks = [(_lin(v, 0, max(2.2, max_v * 1.25), plot[3], plot[1]), f"{v:g}")
               for v in (0, .5, 1, 1.5, 2)]
    _axis(draw, plot, y_ticks=y_ticks, y_label="Mean Stage-0 executions per run")
    for index, row in enumerate(arm_summary):
        x = _lin(index + .5, 0, len(ARM_ORDER), plot[0], plot[2])
        value = float(row["mean_bootstrap_count"])
        y = _lin(value, 0, max(2.2, max_v * 1.25), plot[3], plot[1])
        draw.rounded_rectangle((x - 42, y, x + 42, plot[3]), radius=8,
                               fill=ARM_COLOR[str(row["arm"])])
        _text(draw, (x, y - 13), f"{value:.1f}", 25, bold=True, anchor="mb")
        _text(draw, (x, plot[3] + 25), ARM_SHORT[str(row["arm"])], 23,
              fill=MUTED, anchor="ma")
    _text(draw, (plot[2] - 5, plot[1] + 22),
          "Lifecycle violation events: predictive 5; HDFA+RL 5", 21,
          fill=RED, bold=True, anchor="ra")
    _wrapped(draw, (plot[0], plot[3] + 84),
             "Re-entry is scientifically legitimate after OOD evidence, but repeated bootstrap failure prevented a complete paired estimand. Lifecycle safety worked by censoring rather than silently continuing.",
             plot[2] - plot[0], 25, fill=MUTED)
    _save(image, output)


def _figure_trajectories(trajectory: Sequence[Mapping[str, object]], output: Path) -> None:
    image, draw = _canvas(
        "Primary-arm detector trajectories and censoring risk set",
        "Observed trajectories only; shrinking HDFA risk sets make late-horizon marginal means non-comparable",
    )
    a = _panel(draw, (80, 235, 2100, 1720), "A", "Detector-event rate by control interval")
    b = _panel(draw, (2150, 235, 3120, 1720), "B", "Number of runs still observed")
    methods = ("full_control_detector_rl", "predictive_hdfa_residual_rl")

    left, top, right, bottom = a
    plot = (left + 30, top + 20, right - 25, bottom - 130)
    max_y = max(float(row["q3_detector_rate"]) for row in trajectory) * 1.12
    y_values = np.linspace(0, max(.16, max_y), 5)
    _axis(draw, plot,
          x_ticks=[(_lin(v, 0, 31, plot[0], plot[2]), str(v)) for v in (0, 8, 16, 24, 31)],
          y_ticks=[(_lin(float(v), 0, max(.16, max_y), plot[3], plot[1]), f"{v:.2f}") for v in y_values],
          x_label="Control interval", y_label="Detector-event rate")
    for arm in methods:
        rows = sorted((row for row in trajectory if row["arm"] == arm),
                      key=lambda row: int(row["interval"]))
        upper = [(_lin(int(row["interval"]), 0, 31, plot[0], plot[2]),
                  _lin(float(row["q3_detector_rate"]), 0, max(.16, max_y), plot[3], plot[1])) for row in rows]
        lower = [(_lin(int(row["interval"]), 0, 31, plot[0], plot[2]),
                  _lin(float(row["q1_detector_rate"]), 0, max(.16, max_y), plot[3], plot[1])) for row in reversed(rows)]
        fill = ARM_COLOR[arm] + "36"
        # PIL needs explicit RGBA for translucent ribbons.
        overlay = Image.new("RGBA", image.size, (255, 255, 255, 0))
        od = ImageDraw.Draw(overlay)
        od.polygon(upper + lower, fill=tuple(int(ARM_COLOR[arm][i:i+2], 16)
                                             for i in (1, 3, 5)) + (45,))
        image.paste(overlay, (0, 0), overlay)
        points = [(_lin(int(row["interval"]), 0, 31, plot[0], plot[2]),
                   _lin(float(row["median_detector_rate"]), 0, max(.16, max_y), plot[3], plot[1]))
                  for row in rows]
        draw.line(points, fill=ARM_COLOR[arm], width=7)
        for x, y in points[::4]:
            draw.ellipse((x - 5, y - 5, x + 5, y + 5), fill=ARM_COLOR[arm])
    legend_y = plot[1] + 26
    for index, arm in enumerate(methods):
        x = plot[0] + 35 + index * 480
        draw.line((x, legend_y, x + 55, legend_y), fill=ARM_COLOR[arm], width=7)
        _text(draw, (x + 70, legend_y), ARM_LABEL[arm], 26, fill=MUTED, anchor="lm")
    _wrapped(draw, (plot[0], plot[3] + 78),
             "Lines are interval-wise medians; ribbons are IQRs. These are descriptive observed-data summaries, not a matched treatment effect after HDFA censoring.",
             plot[2] - plot[0], 25, fill=MUTED)

    left, top, right, bottom = b
    plot = (left + 20, top + 20, right - 20, bottom - 160)
    _axis(draw, plot,
          x_ticks=[(_lin(v, 0, 31, plot[0], plot[2]), str(v)) for v in (0, 8, 16, 24, 31)],
          y_ticks=[(_lin(v, 0, 25, plot[3], plot[1]), str(v)) for v in (0, 5, 10, 15, 20, 25)],
          x_label="Control interval", y_label="Runs at risk")
    for arm in methods:
        rows = sorted((row for row in trajectory if row["arm"] == arm),
                      key=lambda row: int(row["interval"]))
        points = [(_lin(int(row["interval"]), 0, 31, plot[0], plot[2]),
                   _lin(int(row["at_risk"]), 0, 25, plot[3], plot[1])) for row in rows]
        draw.line(points, fill=ARM_COLOR[arm], width=7)
        for x, y in points[::4]:
            draw.ellipse((x - 6, y - 6, x + 6, y + 6), fill=ARM_COLOR[arm])
    _text(draw, (plot[0] + 25, plot[1] + 28), "25", 28, fill=ARM_COLOR[methods[0]], bold=True)
    final_hdfa = next(row for row in trajectory
                      if row["arm"] == methods[1] and int(row["interval"]) == 31)
    _text(draw, (plot[2] - 15,
                 _lin(int(final_hdfa["at_risk"]), 0, 25, plot[3], plot[1]) - 16),
          f"n={final_hdfa['at_risk']}", 27, fill=ARM_COLOR[methods[1]], bold=True, anchor="rb")
    _wrapped(draw, (plot[0], plot[3] + 80),
             "Full RL remains at n=25. HDFA + residual RL falls to n=9 by the final interval; late apparent rate suppression is therefore vulnerable to informative censoring.",
             plot[2] - plot[0], 25, fill=MUTED)
    _save(image, output)


def _figure_acceptance(report: Mapping[str, object], output: Path) -> None:
    image, draw = _canvas(
        "Logical evidence and predeclared acceptance gates",
        "Stim 1.16.0 + PyMatching 2.4.0 evaluation; hollow markers denote censored runs",
    )
    a = _panel(draw, (80, 235, 1700, 1720), "A", "Circuit-level logical failure probability")
    b = _panel(draw, (1750, 235, 3120, 1720), "B", "Architecture-wide gate outcomes")
    metrics = report["metrics"]

    left, top, right, bottom = a
    plot = (left + 35, top + 20, right - 20, bottom - 150)
    y_low, y_high = 7e-4, 7e-2
    y_ticks_values = (1e-3, 3e-3, 1e-2, 3e-2)
    _axis(draw, plot,
          y_ticks=[(_log(v, y_low, y_high, plot[3], plot[1]), f"{v:.0e}") for v in y_ticks_values],
          y_label="Logical failure probability", y_grid=True)
    rng = np.random.default_rng(20260801)
    for index, arm in enumerate(ARM_ORDER):
        rows = [row for row in metrics if row["arm"] == arm]
        centre = _lin(index + .5, 0, len(ARM_ORDER), plot[0], plot[2])
        for jitter, row in zip(rng.uniform(-42, 42, len(rows)), rows):
            value = float(row["logical_circuit_failure_probability"])
            y = _log(max(y_low, value), y_low, y_high, plot[3], plot[1])
            x = centre + float(jitter)
            if row["completion_status"] == "completed":
                draw.ellipse((x - 7, y - 7, x + 7, y + 7), fill=ARM_COLOR[arm],
                             outline=WHITE, width=1)
            else:
                draw.ellipse((x - 8, y - 8, x + 8, y + 8), fill=WHITE,
                             outline=ARM_COLOR[arm], width=3)
        med = median(float(row["logical_circuit_failure_probability"]) for row in rows)
        y = _log(med, y_low, y_high, plot[3], plot[1])
        draw.line((centre - 44, y, centre + 44, y), fill=INK, width=5)
        _text(draw, (centre, plot[3] + 24), ARM_SHORT[arm], 24, fill=MUTED, anchor="ma")
    _wrapped(draw, (plot[0], plot[3] + 82),
             "All points use the named circuit-level adapter. Because censored runs have shorter and outcome-dependent horizons, arm medians are descriptive and cannot establish logical-performance superiority.",
             plot[2] - plot[0], 25, fill=MUTED)

    left, top, right, bottom = b
    row_top = top + 30
    header = (left + 5, row_top, right - 5, row_top + 70)
    draw.rounded_rectangle(header, radius=10, fill=NAVY)
    columns = (left + 28, left + 595, left + 875, right - 175)
    for x, value in zip(columns, ("Gate", "Measured", "Required", "Status")):
        _text(draw, (x, row_top + 35), value, 25, fill=WHITE, bold=True, anchor="lm")
    display = {
        "sample_efficiency_to_observed_90pct_recovery": ("90% sample efficiency", "53.3x nominal*", ">=10x"),
        "integrated_excess_edr": ("Integrated excess EDR", "2.07x", ">=5x"),
        "exploration_damage": ("Exploration damage", "31.2x nominal*", ">=2x"),
        "one_interval_recurring_recovery": ("One-interval recovery", "35%", ">=90%"),
        "no_final_performance_loss": ("Final EDR difference", "-0.1386*", "upper CI <=0.005"),
    }
    y = row_top + 88
    for index, gate in enumerate(report["gates"]):
        label, measured, required = display[gate["gate_id"]]
        fill = PALE if index % 2 == 0 else WHITE
        draw.rectangle((left + 5, y, right - 5, y + 105), fill=fill)
        _text(draw, (columns[0], y + 52), label, 26, bold=True, anchor="lm")
        _text(draw, (columns[1], y + 52), measured, 25, anchor="lm")
        _text(draw, (columns[2], y + 52), required, 25, anchor="lm")
        draw.rounded_rectangle((columns[3] - 8, y + 21, columns[3] + 132, y + 80),
                               radius=18, fill="#F6DEDE")
        _text(draw, (columns[3] + 62, y + 51), "FAIL", 25, fill=RED, bold=True, anchor="mm")
        y += 105
    _wrapped(draw, (left + 15, y + 25),
             "*Nominal conditional values do not pass: lifecycle violations and incomplete staged runs invalidate the primary paired estimand. All five gates therefore failed under the predeclared authoritative protocol.",
             right - left - 50, 26, fill=RED, bold=True)
    _wrapped(draw, (left + 15, y + 175),
             "The report is authoritative (no missing data or design invalidity), accepted = false. Negative evidence is retained rather than converted into a success by extrapolation or complete-case selection.",
             right - left - 50, 25, fill=MUTED)
    _save(image, output)


def _figure_scalability(analysis: Mapping[str, object], attainment_rows: Sequence[Mapping[str, str]],
                        gamma_rows: Sequence[Mapping[str, str]], output: Path) -> None:
    image, draw = _canvas(
        "Comparison with the Google RL paper's scalability claims",
        "Published anchors, declared surrogates and executed software probes are shown as separate evidence layers",
    )
    boxes = (
        (80, 235, 1575, 955, "A", "Real-time steerability frequency"),
        (1625, 235, 3120, 955, "B", "Surrogate target attainment"),
        (80, 1005, 1575, 1720, "C", "Convergence-fit credibility"),
        (1625, 1005, 3120, 1720, "D", "Conditional native-QEC cycle ratio"),
    )
    panels = [_panel(draw, box[:4], box[4], box[5]) for box in boxes]

    # A: frequency anchors.
    left, top, right, bottom = panels[0]
    values = (
        ("Google paper anchor", 1/150, "#1D1D1D"),
        ("Google-style surrogate", float(analysis["steerability"][PAPER_METHOD]["critical_frequency_at_least_2pct"]), METHOD_COLOR[PAPER_METHOD]),
        ("HDFA surrogate", float(analysis["steerability"][HDFA_METHOD]["critical_frequency_at_least_2pct"]), METHOD_COLOR[HDFA_METHOD]),
    )
    x_low, x_high = .0045, .0115
    plot = (left + 255, top + 35, right - 60, bottom - 155)
    _axis(draw, plot,
          x_ticks=[(_lin(v, x_low, x_high, plot[0], plot[2]), f"{v:.3f}")
                   for v in (.005, .007, .009, .011)],
          x_label="Drift frequency (epoch^-1)", y_grid=False)
    for idx, (label, value, color) in enumerate(values):
        y = plot[1] + 80 + idx * 130
        x = _lin(value, x_low, x_high, plot[0], plot[2])
        _text(draw, (plot[0] - 25, y), label, 25, fill=MUTED, anchor="rm")
        draw.line((plot[0], y, x, y), fill=color, width=8)
        draw.ellipse((x - 12, y - 12, x + 12, y + 12), fill=color)
        _text(draw, (x + 20, y), f"{value:.5f}", 25, fill=color, bold=True, anchor="lm")
    _wrapped(draw, (left, bottom - 48), "The HDFA value is a surrogate contrast, not a hardware reproduction.", right-left, 23, fill=MUTED)

    # B: attainment fractions.
    left, top, right, bottom = panels[1]
    plot = (left + 35, top + 25, right - 30, bottom - 100)
    _axis(draw, plot,
          x_ticks=[(_lin(i, 0, 2, plot[0] + 100, plot[2] - 100), f"{t}%")
                   for i, t in enumerate((50, 75, 90))],
          y_ticks=[(_lin(v, 0, 1, plot[3], plot[1]), f"{int(v*100)}") for v in (0, .25, .5, .75, 1)],
          x_label="Local-optimum recovery target", y_label="Runs reaching target (%)")
    for method, offset in ((PAPER_METHOD, -25), (HDFA_METHOD, 25)):
        rows = sorted((row for row in attainment_rows if row["method"] == method),
                      key=lambda row: float(row["target_fraction"]))
        points = []
        for idx, row in enumerate(rows):
            x = _lin(idx, 0, 2, plot[0] + 100, plot[2] - 100) + offset
            frac = float(row["fraction"])
            y = _lin(frac, 0, 1, plot[3], plot[1])
            draw.ellipse((x - 10, y - 10, x + 10, y + 10), fill=METHOD_COLOR[method])
            points.append((x, y))
        draw.line(points, fill=METHOD_COLOR[method], width=5)
    for idx, method in enumerate((PAPER_METHOD, HDFA_METHOD)):
        x = plot[0] + 15 + idx * 430
        y = plot[1] + 15
        draw.line((x, y, x + 45, y), fill=METHOD_COLOR[method], width=6)
        _text(draw, (x + 57, y), METHOD_LABEL[method], 22, fill=MUTED, anchor="lm")

    # C: R2 credibility.
    left, top, right, bottom = panels[2]
    plot = (left + 55, top + 25, right - 30, bottom - 160)
    y_min, y_max = -55., 1.
    _axis(draw, plot,
          x_ticks=[(_lin(i, 0, 2, plot[0] + 120, plot[2] - 120), f"P={p}")
                   for i, p in enumerate((1, 10, 30))],
          y_ticks=[(_lin(v, y_min, y_max, plot[3], plot[1]), str(v)) for v in (-50, -40, -30, -20, -10, 0)],
          x_label="Control parameters per gate", y_label="Mean fit R-squared")
    threshold_y = _lin(.8, y_min, y_max, plot[3], plot[1])
    draw.line((plot[0], threshold_y, plot[2], threshold_y), fill=RED, width=3)
    _text(draw, (plot[2] - 5, threshold_y - 8), "credibility threshold 0.8", 21, fill=RED, anchor="rb")
    for method, offset in ((PAPER_METHOD, -30), (HDFA_METHOD, 30)):
        rows = sorted((row for row in gamma_rows if row["method"] == method),
                      key=lambda row: int(row["parameters_per_gate"]))
        for idx, row in enumerate(rows):
            x = _lin(idx, 0, 2, plot[0] + 120, plot[2] - 120) + offset
            value = float(row["mean_r_squared"])
            y = _lin(value, y_min, y_max, plot[3], plot[1])
            draw.ellipse((x - 12, y - 12, x + 12, y + 12), fill=METHOD_COLOR[method])
    _wrapped(draw, (left, bottom - 48), "No fitted gamma passes R-squared > 0.8; size invariance of a poorly fitted gamma is not credible scaling evidence.", right-left, 23, fill=RED)

    # D: conditional cycle ratios.
    left, top, right, bottom = panels[3]
    rows = analysis["sample_efficiency"]
    plot = (left + 70, top + 30, right - 45, bottom - 155)
    _axis(draw, plot,
          x_ticks=[(_lin(i, 0, 2, plot[0] + 110, plot[2] - 110), f"{int(float(row['target_fraction'])*100)}%")
                   for i, row in enumerate(rows)],
          y_ticks=[(_lin(v, 0, 22, plot[3], plot[1]), str(v)) for v in (0, 5, 10, 15, 20)],
          x_label="Target", y_label="Paper / HDFA native-QEC cycles")
    ten = _lin(10, 0, 22, plot[3], plot[1])
    draw.line((plot[0], ten, plot[2], ten), fill=GOLD, width=3)
    _text(draw, (plot[2] - 8, ten - 7), "10x project target", 22, fill=GOLD, anchor="rb")
    for idx, row in enumerate(rows):
        x = _lin(idx, 0, 2, plot[0] + 110, plot[2] - 110)
        value = row["median_cycle_ratio"]
        if value is None:
            draw.line((x - 15, plot[3] - 12, x + 15, plot[3] + 12), fill=RED, width=4)
            draw.line((x - 15, plot[3] + 12, x + 15, plot[3] - 12), fill=RED, width=4)
            continue
        value = float(value)
        y = _lin(value, 0, 22, plot[3], plot[1])
        q1 = _lin(float(row["q1"]), 0, 22, plot[3], plot[1])
        q3 = _lin(float(row["q3"]), 0, 22, plot[3], plot[1])
        draw.line((x, q1, x, q3), fill=METHOD_COLOR[HDFA_METHOD], width=8)
        draw.ellipse((x - 13, y - 13, x + 13, y + 13), fill=METHOD_COLOR[HDFA_METHOD])
        _text(draw, (x, y - 18), f"{value:.2f}x", 24, bold=True, anchor="mb")
    _save(image, output)


def _figure_compute(pipeline_rows: Sequence[Mapping[str, str]],
                    detector_rows: Sequence[Mapping[str, str]], output: Path) -> None:
    image, draw = _canvas(
        "Executed computational scaling to distance 15",
        "Fresh process per distance/seed condition; timings characterize this Python implementation, not QPU latency",
    )
    boxes = (
        (80, 235, 1575, 955, "A", "Wall time per control interval"),
        (1625, 235, 3120, 955, "B", "Peak process memory"),
        (80, 1005, 1575, 1720, "C", "HDFA / full-RL compute-time ratio"),
        (1625, 1005, 3120, 1720, "D", "Paired detector-rate ratio"),
    )
    panels = [_panel(draw, box[:4], box[4], box[5]) for box in boxes]
    by_method: dict[str, list[Mapping[str, str]]] = defaultdict(list)
    for row in pipeline_rows:
        by_method[row["method"]].append(row)
    controls = sorted({int(row["suite_control_variables"]) for row in pipeline_rows})

    def log_panel(panel: tuple[int, int, int, int], value_key: str,
                  y_low: float, y_high: float, y_label: str,
                  tick_values: Sequence[float], tick_labels: Sequence[str]) -> None:
        left, top, right, bottom = panel
        plot = (left + 105, top + 25, right - 35, bottom - 95)
        _axis(draw, plot,
              x_ticks=[(_log(v, min(controls), max(controls), plot[0], plot[2]), str(v))
                       for v in (33, 97, 193, 481, 897)],
              y_ticks=[(_log(v, y_low, y_high, plot[3], plot[1]), label)
                       for v, label in zip(tick_values, tick_labels)],
              x_label="Suite control variables", y_label=y_label)
        for method in (PAPER_METHOD, HDFA_METHOD):
            rows = sorted(by_method[method], key=lambda row: int(row["suite_control_variables"]))
            points = [(_log(int(row["suite_control_variables"]), min(controls), max(controls), plot[0], plot[2]),
                       _log(float(row[value_key]), y_low, y_high, plot[3], plot[1])) for row in rows]
            draw.line(points, fill=METHOD_COLOR[method], width=6)
            for x, y in points:
                draw.ellipse((x - 8, y - 8, x + 8, y + 8), fill=METHOD_COLOR[method])
        for idx, method in enumerate((PAPER_METHOD, HDFA_METHOD)):
            x = plot[0] + 15 + idx * 445
            y = plot[1] + 15
            draw.line((x, y, x + 45, y), fill=METHOD_COLOR[method], width=6)
            _text(draw, (x + 55, y), METHOD_LABEL[method], 21, fill=MUTED, anchor="lm")

    log_panel(panels[0], "elapsed_median_s", .003, 20., "Median seconds",
              (.003, .03, .3, 3., 20.), ("0.003", "0.03", "0.3", "3", "20"))
    log_panel(panels[1], "memory_median_bytes", 3e7, 1.2e9, "Peak RSS (bytes)",
              (3e7, 1e8, 3e8, 1e9), ("30 MB", "100 MB", "300 MB", "1 GB"))

    # C ratios by distance.
    left, top, right, bottom = panels[2]
    plot = (left + 55, top + 25, right - 35, bottom - 150)
    distances = sorted({int(row["code_distance"]) for row in pipeline_rows})
    keyed = {(row["method"], int(row["code_distance"])): row for row in pipeline_rows}
    ratios = [float(keyed[(HDFA_METHOD, d)]["elapsed_median_s"]) /
              float(keyed[(PAPER_METHOD, d)]["elapsed_median_s"]) for d in distances]
    _axis(draw, plot,
          x_ticks=[(_lin(d, 3, 15, plot[0], plot[2]), str(d)) for d in distances],
          y_ticks=[(_lin(v, 0, 125, plot[3], plot[1]), str(v)) for v in (0, 25, 50, 75, 100, 125)],
          x_label="Code distance", y_label="Wall-time ratio")
    points = [(_lin(d, 3, 15, plot[0], plot[2]), _lin(v, 0, 125, plot[3], plot[1]))
              for d, v in zip(distances, ratios)]
    draw.line(points, fill=METHOD_COLOR[HDFA_METHOD], width=6)
    for (x, y), value in zip(points, ratios):
        draw.ellipse((x - 9, y - 9, x + 9, y + 9), fill=METHOD_COLOR[HDFA_METHOD])
        _text(draw, (x, y - 16), f"{value:.0f}x", 22, fill=METHOD_COLOR[HDFA_METHOD], bold=True, anchor="mb")
    _wrapped(draw, (left, bottom - 48), "At d=15: 12.65 s vs 0.143 s per interval (88x); both paths scale approximately linearly in implemented controls.", right-left, 23, fill=MUTED)

    # D paired EDR ratios.
    left, top, right, bottom = panels[3]
    plot = (left + 55, top + 25, right - 35, bottom - 150)
    y_low, y_high = .55, 1.4
    _axis(draw, plot,
          x_ticks=[(_lin(d, 3, 15, plot[0], plot[2]), str(d)) for d in distances],
          y_ticks=[(_lin(v, y_low, y_high, plot[3], plot[1]), f"{v:.1f}") for v in (.6, .8, 1., 1.2, 1.4)],
          x_label="Code distance", y_label="HDFA / full-RL EDR")
    unity = _lin(1., y_low, y_high, plot[3], plot[1])
    draw.line((plot[0], unity, plot[2], unity), fill=INK, width=3)
    points = []
    for row in sorted(detector_rows, key=lambda row: int(row["code_distance"])):
        d = int(row["code_distance"])
        x = _lin(d, 3, 15, plot[0], plot[2])
        y = _lin(float(row["mean_ratio"]), y_low, y_high, plot[3], plot[1])
        lo = _lin(float(row["ci95_low"]), y_low, y_high, plot[3], plot[1])
        hi = _lin(float(row["ci95_high"]), y_low, y_high, plot[3], plot[1])
        draw.line((x, lo, x, hi), fill=METHOD_COLOR[HDFA_METHOD], width=4)
        draw.line((x - 8, lo, x + 8, lo), fill=METHOD_COLOR[HDFA_METHOD], width=3)
        draw.line((x - 8, hi, x + 8, hi), fill=METHOD_COLOR[HDFA_METHOD], width=3)
        draw.ellipse((x - 9, y - 9, x + 9, y + 9), fill=METHOD_COLOR[HDFA_METHOD])
        points.append((x, y))
    draw.line(points, fill=METHOD_COLOR[HDFA_METHOD], width=5)
    _wrapped(draw, (left, bottom - 48), "Only d=11 and d=13 exclude unity at 95%; the two-epoch probe is throughput evidence, not a convergence or hardware-performance experiment.", right-left, 23, fill=MUTED)
    _save(image, output)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--comparison-root", type=Path,
                        default=Path("artifacts/comparison/nature-2026-v5"))
    parser.add_argument("--output", type=Path,
                        default=Path("artifacts/comparison/nature-2026-v5/paper-report"))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    data_dir, figure_dir = args.output / "data", args.output / "figures"
    data_dir.mkdir(exist_ok=True)
    figure_dir.mkdir(exist_ok=True)

    effectiveness = json.loads((args.comparison_root / "authoritative-effectiveness.json").read_text(encoding="utf-8"))
    scalability_analysis = args.output / "scalability-analysis"
    analysis = json.loads((scalability_analysis / "analysis-summary.json").read_text(encoding="utf-8"))

    arm_summary = _arm_summary(effectiveness)
    trajectory = _trajectory_summary(effectiveness)
    _write_csv(data_dir / "effectiveness-arm-summary.csv", arm_summary)
    _write_csv(data_dir / "primary-trajectory-summary.csv", trajectory)
    _write_csv(data_dir / "recovery-summary.csv", effectiveness["recovery_summaries"])
    _write_csv(data_dir / "acceptance-gates.csv", effectiveness["gates"])

    def read_csv(path: Path) -> list[dict[str, str]]:
        with path.open(encoding="utf-8", newline="") as stream:
            return list(csv.DictReader(stream))

    attainment = read_csv(scalability_analysis / "sample-attainment-summary.csv")
    gamma = read_csv(scalability_analysis / "gamma-summary.csv")
    pipeline = read_csv(scalability_analysis / "pipeline-scaling-summary.csv")
    detector = read_csv(scalability_analysis / "pipeline-detector-ratio-summary.csv")

    _figure_effectiveness(arm_summary, effectiveness["recovery_summaries"],
                          figure_dir / "figure1-effectiveness-and-lifecycle.png")
    _figure_trajectories(trajectory, figure_dir / "figure2-primary-trajectories.png")
    _figure_acceptance(effectiveness, figure_dir / "figure3-logical-and-acceptance.png")
    _figure_scalability(analysis, attainment, gamma,
                        figure_dir / "figure4-google-paper-scalability-comparison.png")
    _figure_compute(pipeline, detector, figure_dir / "figure5-computational-scaling.png")

    censor_reasons = Counter(
        row["censoring_reason"] for row in effectiveness["metrics"]
        if row["arm"] == "predictive_hdfa_residual_rl" and row["censoring_reason"])
    summary = {
        "schema_version": "hdfa-rl-google-paper-report-analysis.v1",
        "effectiveness_report_hash": effectiveness["report_hash"],
        "scalability_report_hash": analysis["validation"]["manifest_report_hash"],
        "authoritative": effectiveness["authoritative"],
        "accepted": effectiveness["accepted"],
        "acceptance_failure_reasons": effectiveness["acceptance_failure_reasons"],
        "arm_summary": arm_summary,
        "staged_censor_reasons": dict(censor_reasons),
        "acceptance_gates": effectiveness["gates"],
        "recovery_summaries": effectiveness["recovery_summaries"],
        "scalability_analysis": analysis,
        "paper_anchors": {
            "fine_tuning_ler_reduction_percent": 20,
            "steering_ler_reduction_percent": 24,
            "steering_ler_stability_improvement": 2.4,
            "decoder_assisted_ler_reduction_percent": 31,
            "decoder_assisted_stability_improvement": 3.5,
            "response_time_epochs": 130,
            "critical_frequency_epoch_inverse": 1/150,
            "distance_15_controls_approximate": 40_000,
            "distance_7_surface_ler_per_cycle": 7.72e-4,
            "distance_5_colour_ler_per_cycle": 8.19e-3,
        },
        "claim_boundary": effectiveness["design_audit"]["claim_scope"],
        "provenance": effectiveness["provenance"],
    }
    (data_dir / "paper-comparison-summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
