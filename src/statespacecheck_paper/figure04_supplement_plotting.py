"""Plotting for the Figure-4 supplement: broadening, coverage, rate, and behavior.

Renders the analyses in :mod:`statespacecheck_paper.figure04_broadening` as one
compact multi-panel figure. Every panel reads precomputed results; no
diagnostic is recomputed here.
"""

from __future__ import annotations

from typing import Any, cast

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from numpy.typing import NDArray

from statespacecheck_paper.figure04_broadening import BroadeningResults, Figure4BroadeningConfig
from statespacecheck_paper.style import COLORS

MODEL_LABELS = {"continuous": "Continuous", "continuous_fragmented": "Continuous–Fragmented"}
MODEL_COLORS = {"continuous": COLORS["predictive"], "continuous_fragmented": COLORS["likelihood"]}
METRIC_LABELS = {"hpd_overlap": "HPD overlap", "predictive_pvalue": "Rank p-value"}


def _ecdf(values: NDArray[np.integer] | NDArray[np.floating]) -> tuple[NDArray[Any], NDArray[Any]]:
    x = np.sort(np.asarray(values, dtype=float))
    y = np.arange(1, x.size + 1) / x.size
    return x, y


def plot_region_size_ecdf(ax: Axes, broadening: BroadeningResults, coverage: float) -> None:
    """ECDFs of predictive HPD-region size at all events and at rescued events."""
    arrays = broadening.arrays_by_coverage[f"{coverage:g}"]
    rescued = arrays.flag_continuous & ~arrays.flag_continuous_fragmented
    for model, sizes in (
        ("continuous", arrays.size_continuous),
        ("continuous_fragmented", arrays.size_continuous_fragmented),
    ):
        x, y = _ecdf(sizes)
        ax.step(x, y, where="post", color=MODEL_COLORS[model], label=f"{MODEL_LABELS[model]}, all")
        if rescued.any():
            x, y = _ecdf(sizes[rescued])
            ax.step(
                x,
                y,
                where="post",
                color=MODEL_COLORS[model],
                linestyle="--",
                label=f"{MODEL_LABELS[model]}, rescued",
            )
    x, y = _ecdf(arrays.likelihood_size)
    ax.step(x, y, where="post", color=COLORS["threshold"], linestyle=":", label="Likelihood")
    ax.set_xlabel(f"{int(round(100 * coverage))}% HPD region size (position bins)")
    ax.set_ylabel("Cumulative fraction of spike events")
    ax.set_ylim(0, 1)
    ax.legend(fontsize=6, frameon=False, loc="lower right")


def plot_uniform_mixture_rescue(
    ax: Axes, broadening: BroadeningResults, config: Figure4BroadeningConfig
) -> None:
    """Fraction of original Continuous flags removed versus added uniform mass."""
    by_coverage = cast(dict[str, Any], broadening.summary["by_coverage"])
    styles = {"0.5": ":", "0.8": "--", "0.95": "-"}
    for key, record in by_coverage.items():
        weights = [w for w in config.uniform_weights if w < 1.0]
        removed = [
            record["uniform_mixture"][f"{w:g}"]["fraction_original_flags_removed"] for w in weights
        ]
        ax.plot(
            weights,
            removed,
            marker="o",
            markersize=3,
            color=COLORS["predictive"],
            linestyle=styles.get(key, "-"),
            label=f"{int(round(100 * float(record['coverage'])))}% HPD, uniform mixture",
        )
        rescue = record["rescue_fraction"]
        if rescue is not None:
            ax.axhline(
                rescue,
                color=COLORS["likelihood"],
                linestyle=styles.get(key, "-"),
                linewidth=0.8,
                label=f"{int(round(100 * float(record['coverage'])))}% HPD, Cont.–Frag. rescue",
            )
    ax.set_xlabel("Uniform weight $w$ added to the Continuous prediction")
    ax.set_ylabel("Fraction of original\nContinuous flags removed")
    ax.set_ylim(0, 1.02)
    ax.legend(fontsize=5.5, frameon=False, loc="lower right")


def plot_rate_vs_unit_flag_fraction(ax: Axes, rate_summary: dict[str, Any]) -> None:
    """Per-unit session rate against per-unit rank-p-value and HPD flag fractions."""
    per_unit = rate_summary["per_unit"]
    rates = np.asarray(per_unit["rate_hz"], dtype=float)
    active = np.asarray(per_unit["active"], dtype=bool)
    for model, marker in (("continuous", "o"), ("continuous_fragmented", "^")):
        for metric, filled in (("predictive_pvalue", True), ("hpd_overlap", False)):
            values = np.array(
                [np.nan if v is None else v for v in per_unit[f"{model}_{metric}_flag_fraction"]],
                dtype=float,
            )
            ax.scatter(
                rates[active],
                values[active],
                s=9,
                marker=marker,
                facecolors=MODEL_COLORS[model] if filled else "none",
                edgecolors=MODEL_COLORS[model],
                linewidths=0.6,
                alpha=0.8,
                label=f"{MODEL_LABELS[model]}, {METRIC_LABELS[metric]}",
            )
    ax.axvline(
        rate_summary["low_rate_threshold_hz"],
        color=COLORS["threshold"],
        linestyle=":",
        linewidth=0.8,
    )
    ax.set_xscale("log")
    ax.set_xlabel("Unit session rate (Hz)")
    ax.set_ylabel("Fraction of the unit's\nspike events flagged")
    ax.set_ylim(-0.02, 1.02)
    ax.legend(
        fontsize=5.5,
        frameon=False,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.28),
        ncol=2,
        columnspacing=0.8,
        handletextpad=0.3,
    )


def plot_rate_group_contributions(ax: Axes, rate_summary: dict[str, Any]) -> None:
    """Fractions of active units, events, and flags contributed by each rate group."""
    groups = rate_summary["rate_groups"]
    labels = [g["label_hz"] for g in groups]
    x = np.arange(len(groups))
    series = [
        ("Active units", [g["fraction_of_active_units"] for g in groups], COLORS["reference"]),
        ("Spike events", [g["fraction_of_events"] for g in groups], COLORS["posterior"]),
        (
            "Rank-p flags (Cont.)",
            [g["continuous_predictive_pvalue_fraction_of_flags"] or 0.0 for g in groups],
            COLORS["metric_combined"],
        ),
        (
            "HPD flags (Cont.)",
            [g["continuous_hpd_overlap_fraction_of_flags"] or 0.0 for g in groups],
            COLORS["hpd_overlap"],
        ),
    ]
    width = 0.2
    for k, (name, values, color) in enumerate(series):
        ax.bar(x + (k - 1.5) * width, values, width=width, color=color, label=name)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_xlabel("Unit session rate group (Hz)")
    ax.set_ylabel("Fraction of total")
    ax.set_ylim(0, 1)
    ax.legend(fontsize=5.5, frameon=False, loc="upper left")


def plot_behavior_strata(ax: Axes, rate_summary: dict[str, Any]) -> None:
    """Flag fractions during immobility versus movement, by model and metric."""
    behavior = rate_summary["behavior"]
    strata = ("immobile", "moving")
    combos = [
        (model, metric)
        for metric in ("hpd_overlap", "predictive_pvalue")
        for model in ("continuous", "continuous_fragmented")
    ]
    x = np.arange(len(combos))
    width = 0.38
    for k, stratum in enumerate(strata):
        values = [
            100.0 * behavior[stratum][f"{model}_{metric}_flag_fraction"] for model, metric in combos
        ]
        ax.bar(
            x + (k - 0.5) * width,
            values,
            width=width,
            color=COLORS["threshold"] if stratum == "immobile" else COLORS["reference"],
            label=f"{stratum} (n={behavior[stratum]['n_events']:,})",
        )
    ax.set_xticks(x)
    ax.set_xticklabels(
        [
            f"{METRIC_LABELS[m]}\n{'Cont.' if mo == 'continuous' else 'Cont.–Frag.'}"
            for mo, m in combos
        ],
        fontsize=6,
    )
    ax.set_ylabel("% of spike events flagged")
    ax.legend(
        fontsize=5.5,
        frameon=False,
        title=f"Head speed < {behavior['speed_cutoff_cm_s']:g} cm/s",
        title_fontsize=5.5,
    )


def compose_figure04_supplement(
    broadening: BroadeningResults,
    rate_summary: dict[str, Any],
    config: Figure4BroadeningConfig,
) -> Figure:
    """Assemble the five-panel Figure-4 supplement."""
    fig, axes = plt.subplot_mosaic(
        [["a", "b", "c"], ["d", "e", "e"]],
        figsize=(7.2, 4.6),
        dpi=450,
        constrained_layout=True,
    )
    plot_region_size_ecdf(axes["a"], broadening, 0.95)
    plot_uniform_mixture_rescue(axes["b"], broadening, config)
    plot_rate_vs_unit_flag_fraction(axes["c"], rate_summary)
    plot_rate_group_contributions(axes["d"], rate_summary)
    plot_behavior_strata(axes["e"], rate_summary)
    for key, ax in axes.items():
        ax.text(
            -0.18,
            1.04,
            key,
            transform=ax.transAxes,
            fontsize=9,
            fontweight="bold",
            ha="left",
            va="bottom",
        )
    return fig
