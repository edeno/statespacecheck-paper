"""Real hippocampal data diagnostics for state space model.

This script generates Figure 4, which shows per-cell diagnostic metrics
for decoder models on real neural recording data from hippocampus.
Panel (a) shows a longer context window with surrounding run periods,
panel (b) shows a zoomed-in detail view with the Continuous decoder, and
panel (c) shows the same detail view with the Continuous-Fragmented decoder.

Requires:
- non_local_detector package for decoder models
- Pre-exported neural recording data in data/real/
"""

from __future__ import annotations

import argparse
import dataclasses
import warnings
from pathlib import Path
from typing import Any, Literal

import joblib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.transforms import blended_transform_factory

from statespacecheck_paper.analysis import PerCellDiagnostics
from statespacecheck_paper.load_local_data import load_neural_recording_from_files
from statespacecheck_paper.paths import ANIMAL_DATE_EPOCH, DATA_PATH
from statespacecheck_paper.real_data_analysis import (
    compute_flag_confusion,
    compute_model_diagnostics,
    create_decoder_environment,
    extract_place_fields,
    fit_decoder_models,
    get_spike_counts,
)
from statespacecheck_paper.real_data_plotting import (
    plot_per_spike_metric_hexbin_row,
    plot_single_model_diagnostics,
    plot_track_graph_2d,
)
from statespacecheck_paper.style import save_figure, set_figure_defaults

# -----------------------------
# Configuration
# -----------------------------

__all__ = ["ANIMAL_DATE_EPOCH", "DATA_PATH"]


def _fig4_cache_path() -> Path:
    """Path for the cached Figure-4 decoder outputs (under data/intermediates).

    A single joblib bundle is used rather than netCDF because the decoder
    results carry a ``state_bins`` MultiIndex coordinate, which netCDF cannot
    serialize; joblib (pickle) preserves it exactly.
    """
    return DATA_PATH / "intermediates" / f"{ANIMAL_DATE_EPOCH}_fig4_cache.joblib"


# Time window for Figure 4a (context view)
# Shows ~20 seconds encompassing running and immobility at reward well
FIGURE_4A_CONTEXT_CENTER = 190000  # Time index for context center
FIGURE_4A_CONTEXT_HALF_WIDTH = 5000  # Half-width in time points (~20 seconds)

# Time window for Figure 4b/c (detail view)
# Centered on a period of clear diagnostic activity at reward well
FIGURE_4B_DETAIL_CENTER = 193069  # Time index with KL spike during immobility
FIGURE_4B_DETAIL_HALF_WIDTH = 500  # Half-width in time points (~2 seconds at 500 Hz)


def shift_diagnostic_event_times(
    diagnostics: PerCellDiagnostics,
    time_offset: float,
) -> PerCellDiagnostics:
    """Return diagnostics with event timestamps shifted by ``time_offset``.

    Returns the original instance unchanged when ``event_time`` is
    ``None`` (simulated data path) so callers don't need to branch.
    """
    if diagnostics.event_time is None:
        return diagnostics
    return dataclasses.replace(
        diagnostics,
        event_time=np.asarray(diagnostics.event_time, dtype=np.float64) - time_offset,
    )


def diagnostic_event_mean(diagnostics: PerCellDiagnostics, metric: str) -> float:
    """Return the per-spike mean for a diagnostic metric."""
    event_key = f"event_{metric}"
    if not hasattr(diagnostics, event_key):
        raise KeyError(f"Missing per-spike diagnostic array: {event_key}")
    return float(np.nanmean(getattr(diagnostics, event_key)))


def run_demo(*, use_cache: bool = True) -> None:
    """Run the full Figure 4 generation pipeline.

    Loads data, fits Continuous and ContFrag decoder models, computes
    diagnostics, and generates Figure 4 with context (a), Continuous
    detail (b), and ContFrag detail (c) panels.

    Parameters
    ----------
    use_cache : bool, default True
        When True and a complete cache of decoder outputs exists under
        ``data/intermediates``, load it and skip the expensive fit/decode
        step. When False (``--force-recompute``), always recompute and
        overwrite the cache. Fitting + decoding both models takes several
        minutes; figure-only edits (styling, thresholds) reuse the cache.
    """
    # Load data
    print("Loading data...")
    data = load_neural_recording_from_files(DATA_PATH, ANIMAL_DATE_EPOCH)
    print(f"  Loaded {len(data['spike_times'])} cells")

    # Data the figure needs regardless of cache state.
    position_info = data["position_info"]
    time = position_info.index.values
    position = position_info[["head_position_x", "head_position_y"]].values
    linear_position = position_info["linear_position"].values
    spike_times_list: list[Any] = list(data["spike_times"])

    # The expensive decoder outputs (fit + decode + diagnostics) are cached so
    # that figure-only changes can be previewed without re-running the models.
    cache_path = _fig4_cache_path()
    if use_cache and cache_path.exists():
        print("Loading cached decoder outputs (use --force-recompute to rebuild)...")
        bundle = joblib.load(cache_path)
        continuous_results = bundle["continuous_results"]
        contfrag_results = bundle["contfrag_results"]
        continuous_diagnostics = bundle["continuous_diagnostics"]
        contfrag_diagnostics = bundle["contfrag_diagnostics"]
        spike_counts = bundle["spike_counts"]
        place_field_peaks = bundle["place_field_peaks"]
    else:
        # Environment is only needed to fit the decoders.
        env = create_decoder_environment(
            track_graph=data["track_graph"],
            edge_order=data["linear_edge_order"],
            edge_spacing=data["linear_edge_spacing"],
        )

        print("Fitting models...")
        continuous_model, contfrag_model = fit_decoder_models(
            position=position,
            spike_times=spike_times_list,
            time=time,
            environment=env,
        )

        print(f"Decoding {len(time)} time points...")
        decode_outputs = ["filter", "predictive_posterior", "log_likelihood"]
        continuous_results = continuous_model.predict(
            spike_times=spike_times_list,
            time=time,
            return_outputs=decode_outputs,
        )
        contfrag_results = contfrag_model.predict(
            spike_times=spike_times_list,
            time=time,
            return_outputs=decode_outputs,
        )

        spike_counts = get_spike_counts(spike_times_list, time)

        print("Computing diagnostics...")
        continuous_diagnostics = compute_model_diagnostics(
            continuous_model, continuous_results, spike_counts, time, spike_times=spike_times_list
        )
        contfrag_diagnostics = compute_model_diagnostics(
            contfrag_model, contfrag_results, spike_counts, time, spike_times=spike_times_list
        )

        # Extract place fields for raster sorting (use continuous model).
        place_fields, position_bins = extract_place_fields(continuous_model)
        if np.any(np.all(np.isnan(place_fields), axis=1)):
            warnings.warn(
                "Some cells have all-NaN place fields; peak positions may be incorrect",
                stacklevel=2,
            )
        place_field_peaks = position_bins[np.nanargmax(place_fields, axis=1)]

        print("Caching decoder outputs to data/intermediates ...")
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {
                "continuous_results": continuous_results,
                "contfrag_results": contfrag_results,
                "continuous_diagnostics": continuous_diagnostics,
                "contfrag_diagnostics": contfrag_diagnostics,
                "spike_counts": spike_counts,
                "place_field_peaks": place_field_peaks,
            },
            cache_path,
        )

    # Print summary
    print("\n=== Diagnostic Summary (all time points) ===")
    for name, diag in [("Continuous", continuous_diagnostics), ("ContFrag", contfrag_diagnostics)]:
        print(f"\n{name}:")
        for metric in ["hpd_overlap", "kl_divergence", "spike_prob"]:
            print(f"  {metric}: {diagnostic_event_mean(diag, metric):.4f}")

    # Generate Figure 4
    print("\nGenerating Figure 4...")
    set_figure_defaults(context="paper")

    # Diagnostic thresholds. HPD overlap and the predictive p-value use fixed
    # cutoffs of 0.05. The KL divergence has no natural fixed cutoff, so we reuse
    # the threshold from the Figure 3 simulation: the 99th percentile of the KL
    # divergence over the matched-model baseline window (seed=1), which is ~4.52.
    diagnostic_thresholds = {
        "hpd_overlap": 0.05,
        "kl_divergence": 4.52,
        "spike_prob": 0.05,
    }

    # Per-spike flag agreement between the two decoders at these thresholds.
    # "Cont-only" is the rescue quadrant (flagged by Continuous but not by
    # Continuous-Fragmented); "rescue" is its fraction of all Continuous flags.
    metric_directions: dict[str, Literal["below", "above"]] = {
        "hpd_overlap": "below",
        "kl_divergence": "above",
        "spike_prob": "below",
    }
    print("\n=== Flag agreement: Continuous (A) vs Cont-Frag (B) ===")
    for metric, worse_when in metric_directions.items():
        conf = compute_flag_confusion(
            continuous_diagnostics,
            contfrag_diagnostics,
            metric,
            diagnostic_thresholds[metric],
            worse_when=worse_when,
        )
        print(
            f"  {metric}: n={conf.n:,} both={conf.both:,} cont-only={conf.a_only:,} "
            f"cf-only={conf.b_only:,} neither={conf.neither:,} "
            f"rescue={100 * conf.rescue_rate:.1f}%"
        )

    # Define time slices
    context_slice = slice(
        FIGURE_4A_CONTEXT_CENTER - FIGURE_4A_CONTEXT_HALF_WIDTH,
        FIGURE_4A_CONTEXT_CENTER + FIGURE_4A_CONTEXT_HALF_WIDTH,
    )
    detail_slice = slice(
        FIGURE_4B_DETAIL_CENTER - FIGURE_4B_DETAIL_HALF_WIDTH,
        FIGURE_4B_DETAIL_CENTER + FIGURE_4B_DETAIL_HALF_WIDTH,
    )

    # Convert time to relative seconds from start of context window
    time_arr = np.asarray(time, dtype=np.float64)
    time_offset = time_arr[context_slice.start]
    time_relative = time_arr - time_offset

    # Shift xarray time coordinates to relative seconds
    continuous_results = continuous_results.assign_coords(
        time=continuous_results.coords["time"].values - time_offset
    )
    contfrag_results = contfrag_results.assign_coords(
        time=contfrag_results.coords["time"].values - time_offset
    )

    # Shift spike times to relative seconds
    spike_times_relative: list[Any] = [st - time_offset for st in spike_times_list]
    continuous_diagnostics_relative = shift_diagnostic_event_times(
        continuous_diagnostics,
        time_offset,
    )
    contfrag_diagnostics_relative = shift_diagnostic_event_times(
        contfrag_diagnostics,
        time_offset,
    )

    # Two-row figure: decoding panels on top, track+hexbin row on bottom.
    fig = plt.figure(figsize=(10.0, 9.5), dpi=450, constrained_layout=True)
    subfigs_rows = fig.subfigures(2, 1, height_ratios=[6.5, 2.6], hspace=0.0)

    # Top row: three columns (a) context, (b) Continuous detail, (c) ContFrag detail
    subfigs = subfigs_rows[0].subfigures(1, 3, width_ratios=[3, 2, 2], wspace=0.03)

    # Shared plotting kwargs for detail panels
    detail_kwargs: dict[str, Any] = dict(
        spike_times=spike_times_relative,
        spike_counts=spike_counts,
        place_field_peaks=place_field_peaks,
        time_slice_ind=detail_slice,
        thresholds=diagnostic_thresholds,
        track_graph=data["track_graph"],
        edge_order=data["linear_edge_order"],
        edge_spacing=data["linear_edge_spacing"],
    )

    # Panel (a): context view (wider time window, Continuous model)
    _, axes_a = plot_single_model_diagnostics(
        time_relative,
        linear_position,
        continuous_results,
        continuous_diagnostics_relative,
        spike_times=spike_times_relative,
        spike_counts=spike_counts,
        place_field_peaks=place_field_peaks,
        time_slice_ind=context_slice,
        model_name="",
        thresholds=diagnostic_thresholds,
        track_graph=data["track_graph"],
        edge_order=data["linear_edge_order"],
        edge_spacing=data["linear_edge_spacing"],
        fig=subfigs[0],
    )

    # Highlight the detail window region in panel (a)
    detail_start = time_relative[detail_slice.start]
    detail_end = time_relative[detail_slice.stop - 1]
    for ax in axes_a:
        ax.axvspan(detail_start, detail_end, alpha=0.15, color="gray", zorder=0)

    # Panel (b): Continuous detail view
    _, axes_b = plot_single_model_diagnostics(
        time_relative,
        linear_position,
        continuous_results,
        continuous_diagnostics_relative,
        model_name="Continuous",
        fig=subfigs[1],
        **detail_kwargs,
    )

    # Panel (c): ContFrag detail view
    _, axes_c = plot_single_model_diagnostics(
        time_relative,
        linear_position,
        contfrag_results,
        contfrag_diagnostics_relative,
        model_name="Cont-Frag",
        fig=subfigs[2],
        **detail_kwargs,
    )

    # Match y-axis limits between detail panels for direct comparison
    for i in range(6):
        ylim_b = axes_b[i].get_ylim()
        ylim_c = axes_c[i].get_ylim()
        shared_ylim = (min(ylim_b[0], ylim_c[0]), max(ylim_b[1], ylim_c[1]))
        axes_b[i].set_ylim(shared_ylim)
        axes_c[i].set_ylim(shared_ylim)

    # Annotate behavioral states on context panel (top of predictive row)
    behavioral_periods = [
        (0.5, "Run"),
        (4.5, "Immobile"),
        (9.5, "Run"),
        (15.0, "Immobile"),
    ]
    ax_top = axes_a[0]
    for t_center, label in behavioral_periods:
        ax_top.text(
            t_center,
            1.05,
            label,
            transform=blended_transform_factory(ax_top.transData, ax_top.transAxes),
            fontsize=6,
            ha="center",
            va="bottom",
            fontstyle="italic",
        )

    # Panel labels - place in axes coordinates on the predictive row of each
    # top-row column.
    for axes, label in [(axes_a, "a"), (axes_b, "b"), (axes_c, "c")]:
        axes[0].text(
            -0.05,
            1.15,
            label,
            fontsize=8,
            fontweight="bold",
            transform=axes[0].transAxes,
            va="top",
            ha="right",
        )

    # Bottom row: 2D track layout (d) and whole-session metric hexbins (e)
    subfigs_bot = subfigs_rows[1].subfigures(1, 2, width_ratios=[2.5, 7], wspace=0.05)
    ax_track = subfigs_bot[0].subplots()
    # Reward wells sit at the arm tips, i.e. the degree-1 (leaf) nodes of the
    # track graph. Mark them so the 2D layout connects to the linearized axis
    # used in panels (a)-(c).
    track_graph = data["track_graph"]
    reward_well_nodes = [n for n in track_graph.nodes if track_graph.degree(n) == 1]
    plot_track_graph_2d(
        track_graph=track_graph,
        position_info=position_info,
        ax=ax_track,
        edge_order=data["linear_edge_order"],
        reward_well_nodes=reward_well_nodes,
        scalebar_length=20,
        scalebar_label="20 cm",
    )
    ax_track.text(
        -0.05,
        1.05,
        "d",
        fontsize=8,
        fontweight="bold",
        transform=ax_track.transAxes,
        va="top",
        ha="right",
    )

    axes_hexbin = subfigs_bot[1].subplots(1, 3)
    plot_per_spike_metric_hexbin_row(
        continuous_diagnostics,
        contfrag_diagnostics,
        axes_hexbin,
        model_a_name="Continuous",
        model_b_name="Cont-Frag",
        thresholds=diagnostic_thresholds,
    )
    axes_hexbin[0].text(
        -0.18,
        1.10,
        "e",
        fontsize=8,
        fontweight="bold",
        transform=axes_hexbin[0].transAxes,
        va="top",
        ha="right",
    )

    save_figure("manuscript/figures/main/figure04", close=True)
    print("Saved manuscript/figures/main/figure04.{pdf,png}")
    print("\nFigure 4 complete!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate Figure 4.")
    parser.add_argument(
        "--force-recompute",
        action="store_true",
        help=(
            "Re-fit and re-decode both models instead of loading the cached "
            "decoder outputs under data/intermediates (overwrites the cache)."
        ),
    )
    args = parser.parse_args()
    run_demo(use_cache=not args.force_recompute)
