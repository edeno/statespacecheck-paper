"""Real hippocampal data diagnostics for state space model.

This script generates Figure 4, which shows per-cell diagnostic metrics
for decoder models on real neural recording data from hippocampus.
Panel (a) shows a detail view with the Continuous decoder and panel (b)
shows the same detail view with the Continuous-Fragmented decoder.

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


# Time window for Figure 4a/b (detail view)
# Centered on a period of clear diagnostic activity at reward well
DETAIL_CENTER = 193069  # Time index with KL spike during immobility
DETAIL_HALF_WIDTH = 500  # Half-width in time points (~2 seconds at 500 Hz)


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
    diagnostics, and generates Figure 4 with Continuous detail (a) and
    ContFrag detail (b) panels.

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
    # cutoffs of 0.05. The KL divergence has no natural fixed cutoff, so it is
    # shown without a threshold line or a flagged-region callout.
    diagnostic_thresholds = {
        "hpd_overlap": 0.05,
        "spike_prob": 0.05,
    }

    # Per-spike flag agreement between the two decoders at these thresholds.
    # "Cont-only" is the rescue quadrant (flagged by Continuous but not by
    # Continuous-Fragmented); "rescue" is its fraction of all Continuous flags.
    metric_directions: dict[str, Literal["below", "above"]] = {
        "hpd_overlap": "below",
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

    # Define the detail-window time slice shared by both decoder panels.
    detail_slice = slice(
        DETAIL_CENTER - DETAIL_HALF_WIDTH,
        DETAIL_CENTER + DETAIL_HALF_WIDTH,
    )

    # Convert time to relative seconds from start of the detail window
    time_arr = np.asarray(time, dtype=np.float64)
    time_offset = time_arr[detail_slice.start]
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

    # Two-row figure: (a)/(b) detail zooms with a track inset on top, and
    # (c) whole-session metric hexbins on the bottom.
    fig = plt.figure(figsize=(7.2, 6.1), dpi=450, constrained_layout=True)
    subfigs_rows = fig.subfigures(2, 1, height_ratios=[5.0, 2.6], hspace=0.02)

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

    # Top row: (a) Continuous and (b) ContFrag detail zooms, side by side,
    # with a small unlettered track inset on the right for spatial context.
    subfigs_top = subfigs_rows[0].subfigures(
        1,
        5,
        width_ratios=[0.055, 1.0, 1.07, 0.42, 0.04],
        wspace=0.005,
    )

    # Panel (a): Continuous detail view
    _, axes_a = plot_single_model_diagnostics(
        time_relative,
        linear_position,
        continuous_results,
        continuous_diagnostics_relative,
        model_name="Continuous Model",
        fig=subfigs_top[1],
        **detail_kwargs,
    )
    axes_a[3].set_ylabel("HPD\noverlap", fontsize=7, labelpad=7)

    # Panel (b): ContFrag detail view
    _, axes_b = plot_single_model_diagnostics(
        time_relative,
        linear_position,
        contfrag_results,
        contfrag_diagnostics_relative,
        model_name="Cont.-Frag. Model",
        fig=subfigs_top[2],
        **detail_kwargs,
    )

    # Match y-axis limits between detail panels for direct comparison
    for i in range(6):
        ylim_a = axes_a[i].get_ylim()
        ylim_b = axes_b[i].get_ylim()
        shared_ylim = (min(ylim_a[0], ylim_b[0]), max(ylim_a[1], ylim_b[1]))
        axes_a[i].set_ylim(shared_ylim)
        axes_b[i].set_ylim(shared_ylim)

    # Panel (b) repeats the row scales from panel (a), so keep only the
    # model-specific data and title on the right stack.
    for ax in axes_b:
        ax.set_ylabel("")
        ax.tick_params(axis="y", left=False, labelleft=False)
    for text in axes_b[0].texts:
        if text.get_text() == "Animal Position":
            text.set_visible(False)

    # Keep threshold / worse-fit row annotations only on panel (b), where they
    # read as shared labels for both model stacks.
    for ax in axes_a[3:]:
        for text in ax.texts:
            if text.get_text() == "Threshold" or "Worse fit" in text.get_text():
                text.set_visible(False)

    # Panel labels - place in axes coordinates on the predictive row of each.
    panel_label_x = {"a": -0.115, "b": -0.05}
    for axes, label in [(axes_a, "a"), (axes_b, "b")]:
        axes[0].text(
            panel_label_x[label],
            1.24,
            label,
            fontsize=8,
            fontweight="bold",
            transform=axes[0].transAxes,
            va="top",
            ha="right",
        )

    # Unlettered track inset beside panels (a) and (b). Use the same row
    # rhythm as the detail stacks and place it beside the likelihood row.
    track_gs = subfigs_top[3].add_gridspec(
        6,
        3,
        height_ratios=[2, 2, 1.5, 1, 1, 1],
        width_ratios=[0.01, 0.68, 0.31],
    )
    ax_track = subfigs_top[3].add_subplot(track_gs[1, 1])
    # Reward wells sit at the arm tips, i.e. the degree-1 (leaf) nodes of the
    # track graph. Mark them so the 2D layout connects to the linearized axis
    # used in panels (a)-(b).
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
    ax_track.set_anchor("W")
    # ``add_scalebar`` appends the scale bar as the final line. Move the bar
    # and label together so the label clears the nearby reward-well marker.
    scale_bar_shift = 22.0
    scale_bar_drop = 5.0
    scale_bar_line = ax_track.lines[-1]
    scale_bar_line.set_xdata(np.asarray(scale_bar_line.get_xdata()) + scale_bar_shift)
    scale_bar_line.set_ydata(np.asarray(scale_bar_line.get_ydata()) - scale_bar_drop)
    scale_bar_line.set_linewidth(2.0)
    for text in ax_track.texts:
        if text.get_text() == "20 cm":
            x_pos, y_pos = text.get_position()
            text.set_position((x_pos + scale_bar_shift + 10, y_pos - 4 - scale_bar_drop))
            text.set_fontsize(8.5)
            text.set_clip_on(False)

    # Align the track diagram itself with the shared right-side diagnostic
    # annotations: the diagram's left edge should begin where the annotation
    # text ends.
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    annotation_bboxes = [
        text.get_window_extent(renderer)
        for ax in axes_b[3:]
        for text in ax.texts
        if text.get_visible()
        and (text.get_text() == "Threshold" or "Worse fit" in text.get_text())
    ]
    track_bboxes = []
    for artist in [*ax_track.lines, *ax_track.collections, *ax_track.texts]:
        if not artist.get_visible():
            continue
        bbox = artist.get_window_extent(renderer)
        if np.isfinite([bbox.x0, bbox.x1, bbox.y0, bbox.y1]).all():
            track_bboxes.append(bbox)
    if annotation_bboxes and track_bboxes:
        annotation_right = max(bbox.x1 for bbox in annotation_bboxes)
        track_left = min(bbox.x0 for bbox in track_bboxes)
        pos = ax_track.get_position()
        ax_track.set_in_layout(False)
        # The trajectory line's vector bbox extends slightly farther left than
        # the visually salient rendered diagram, so add a small pixel-level
        # correction measured on the exported PNG.
        visual_edge_correction_px = 7.0
        track_shift = (
            annotation_right - track_left + visual_edge_correction_px
        ) / ax_track.figure.bbox.width
        ax_track.set_position(
            [pos.x0 + track_shift, pos.y0, pos.width, pos.height]
        )
    track_size_scale = 1.10
    pos = ax_track.get_position()
    ax_track.set_position(
        [
            pos.x0,
            pos.y0 - pos.height * (track_size_scale - 1) / 2,
            pos.width * track_size_scale,
            pos.height * track_size_scale,
        ]
    )

    # Bottom row: whole-session metric hexbins.
    subfigs_bot = subfigs_rows[1].subfigures(1, 3, width_ratios=[0.16, 7, 0.16], wspace=0.015)
    axes_hexbin = subfigs_bot[1].subplots(1, 3, gridspec_kw={"wspace": -0.02})
    axes_before_hexbin = tuple(fig.axes)
    plot_per_spike_metric_hexbin_row(
        continuous_diagnostics,
        contfrag_diagnostics,
        axes_hexbin,
        model_a_name="Continuous",
        model_b_name="Cont-Frag",
        thresholds=diagnostic_thresholds,
        colorbar_pad=0.006,
    )
    for ax, anchor in zip(axes_hexbin, ("E", "C", "W"), strict=True):
        ax.set_anchor(anchor)
    hexbin_colorbar_axes = [ax for ax in fig.axes if ax not in axes_before_hexbin]
    fig.canvas.draw()
    hexbin_positions = [ax.get_position() for ax in axes_hexbin]
    panel_width = min(pos.width for pos in hexbin_positions)
    panel_height = min(pos.height for pos in hexbin_positions)
    panel_gap = min(
        hexbin_positions[1].x0 - hexbin_positions[0].x1,
        hexbin_positions[2].x0 - hexbin_positions[1].x1,
    )
    panel_gap = max(panel_gap, 0.0)
    panel_left = hexbin_positions[0].x0
    panel_bottom = hexbin_positions[0].y0
    for panel_idx, ax in enumerate(axes_hexbin):
        ax.set_position(
            [
                panel_left + panel_idx * (panel_width + panel_gap),
                panel_bottom,
                panel_width,
                panel_height,
            ]
        )
    if hexbin_colorbar_axes:
        colorbar_ax = hexbin_colorbar_axes[-1]
        colorbar_pos = colorbar_ax.get_position()
        colorbar_gap = max(panel_gap * 0.5, 0.006)
        colorbar_ax.set_position(
            [
                axes_hexbin[-1].get_position().x1 + colorbar_gap,
                colorbar_pos.y0,
                colorbar_pos.width,
                colorbar_pos.height,
            ]
        )
    fig.set_constrained_layout(False)
    if hexbin_colorbar_axes:
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        colorbar_label_bbox = hexbin_colorbar_axes[-1].yaxis.label.get_window_extent(renderer)
        track_bboxes = []
        for artist in [*ax_track.lines, *ax_track.collections, *ax_track.texts]:
            if not artist.get_visible():
                continue
            bbox = artist.get_window_extent(renderer)
            if np.isfinite([bbox.x0, bbox.x1, bbox.y0, bbox.y1]).all():
                track_bboxes.append(bbox)
        if track_bboxes:
            target_right = colorbar_label_bbox.x1
            track_right = max(bbox.x1 for bbox in track_bboxes)
            pos = ax_track.get_position()
            track_shift = (target_right - track_right) / ax_track.figure.bbox.width
            ax_track.set_position(
                [pos.x0 + track_shift, pos.y0, pos.width, pos.height]
            )
    axes_hexbin[0].text(
        -0.18,
        1.10,
        "c",
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
