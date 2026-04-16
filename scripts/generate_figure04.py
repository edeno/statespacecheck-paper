"""Real hippocampal data diagnostics for state space model.

This script generates Figure 4, which shows per-cell diagnostic metrics
for the Continuous decoder model on real neural recording data from
hippocampus. Panel (a) shows a longer context window with surrounding
run periods, and panel (b) shows a zoomed-in detail view.

Requires:
- non_local_detector package for decoder models
- Pre-exported neural recording data in data/real/
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.transforms import blended_transform_factory

from statespacecheck_paper.load_local_data import load_neural_recording_from_files
from statespacecheck_paper.real_data_analysis import (
    compute_model_diagnostics,
    create_decoder_environment,
    extract_place_fields,
    fit_decoder_models,
    get_spike_counts,
)
from statespacecheck_paper.real_data_plotting import (
    plot_single_model_diagnostics,
)
from statespacecheck_paper.style import save_figure, set_figure_defaults

# Check for optional decoder dependencies
try:
    import non_local_detector  # noqa: F401

    DECODERS_AVAILABLE = True
except ImportError:
    DECODERS_AVAILABLE = False
    warnings.warn(
        "non_local_detector not available. Figure 4 requires fitted decoder models. "
        "Install with: pip install non_local_detector",
        stacklevel=2,
    )


# -----------------------------
# Configuration
# -----------------------------

DATA_PATH = Path(__file__).parent.parent / "data"
ANIMAL_DATE_EPOCH = "j1620210710_02_r1"

# Time window for Figure 4b (detail view)
FIGURE_4B_WINDOW_CENTER = 177301  # Time index around which to show detail
FIGURE_4B_WINDOW_HALF_WIDTH = 50  # Half-width in time points

# Time window for Figure 4a (context view showing surrounding run periods)
FIGURE_4A_CONTEXT_HALF_WIDTH = 500  # Half-width in time points


def run_demo() -> None:
    """Run the full Figure 4 generation pipeline.

    Loads data, fits the Continuous decoder model, computes diagnostics,
    and generates Figure 4 with context (a) and detail (b) panels.
    """
    # Load data
    print("Loading data...")
    data = load_neural_recording_from_files(DATA_PATH, ANIMAL_DATE_EPOCH)
    print(f"  Loaded {len(data['spike_times'])} cells")

    # Create environment
    env = create_decoder_environment(
        track_graph=data["track_graph"],
        edge_order=data["linear_edge_order"],
        edge_spacing=data["linear_edge_spacing"],
    )

    # Prepare training data
    position_info = data["position_info"]
    time = position_info.index.values
    position = position_info[["head_position_x", "head_position_y"]].values
    linear_position = position_info["linear_position"].values
    spike_times_list: list[Any] = list(data["spike_times"])

    # Fit continuous model only
    print("Fitting model...")
    continuous_model, _ = fit_decoder_models(
        position=position,
        spike_times=spike_times_list,
        time=time,
        environment=env,
    )

    # Decode all time points
    print(f"Decoding {len(time)} time points...")
    continuous_results = continuous_model.predict(
        spike_times=spike_times_list,
        time=time,
        return_outputs=["filter", "predictive_posterior", "log_likelihood"],
    )

    # Get spike counts for all time
    spike_counts = get_spike_counts(spike_times_list, time)

    # Compute diagnostics
    print("Computing diagnostics...")
    continuous_diagnostics = compute_model_diagnostics(
        continuous_model, continuous_results, spike_counts, time
    )

    # Extract place fields to get peak positions for sorting raster
    place_fields, position_bins = extract_place_fields(continuous_model)
    if np.any(np.all(np.isnan(place_fields), axis=1)):
        warnings.warn(
            "Some cells have all-NaN place fields; peak positions may be incorrect",
            stacklevel=2,
        )
    place_field_peaks = position_bins[np.nanargmax(place_fields, axis=1)]

    # Print summary
    print("\n=== Diagnostic Summary (all time points) ===")
    for metric in ["hpd_overlap", "kl_divergence", "spike_prob"]:
        cont_mean = np.nanmean(continuous_diagnostics[metric])
        print(f"{metric}: {cont_mean:.4f}")

    # Generate Figure 4
    print("\nGenerating Figure 4...")
    set_figure_defaults(context="paper")

    # Define time slices
    context_slice = slice(
        FIGURE_4B_WINDOW_CENTER - FIGURE_4A_CONTEXT_HALF_WIDTH,
        FIGURE_4B_WINDOW_CENTER + FIGURE_4A_CONTEXT_HALF_WIDTH,
    )
    detail_slice = slice(
        FIGURE_4B_WINDOW_CENTER - FIGURE_4B_WINDOW_HALF_WIDTH,
        FIGURE_4B_WINDOW_CENTER + FIGURE_4B_WINDOW_HALF_WIDTH,
    )

    # Two side-by-side panels: (a) context, (b) detail
    fig = plt.figure(figsize=(7.0, 7.0), dpi=450, constrained_layout=True)
    subfigs = fig.subfigures(1, 2, width_ratios=[3, 2], wspace=0.03)

    # Panel (a): context view (wider time window)
    _, axes_a = plot_single_model_diagnostics(
        time,
        linear_position,
        continuous_results,
        continuous_diagnostics,
        spike_times=spike_times_list,
        spike_counts=spike_counts,
        place_field_peaks=place_field_peaks,
        time_slice_ind=context_slice,
        model_name="",
        track_graph=data["track_graph"],
        edge_order=data["linear_edge_order"],
        edge_spacing=data["linear_edge_spacing"],
        fig=subfigs[0],
    )

    # Highlight the detail window region in panel (a)
    time_arr = np.asarray(time)
    detail_start = time_arr[detail_slice.start]
    detail_end = time_arr[detail_slice.stop - 1]
    for ax in axes_a:
        ax.axvspan(detail_start, detail_end, alpha=0.15, color="gray", zorder=0)

    # Panel (b): detail view (zoomed-in)
    _, axes_b = plot_single_model_diagnostics(
        time,
        linear_position,
        continuous_results,
        continuous_diagnostics,
        spike_times=spike_times_list,
        spike_counts=spike_counts,
        place_field_peaks=place_field_peaks,
        time_slice_ind=detail_slice,
        model_name="",
        track_graph=data["track_graph"],
        edge_order=data["linear_edge_order"],
        edge_spacing=data["linear_edge_spacing"],
        fig=subfigs[1],
    )

    # Panel labels
    label_x = 0.01
    trans_a = blended_transform_factory(fig.transFigure, axes_a[0].transAxes)
    axes_a[0].text(
        label_x,
        1.15,
        "a",
        fontsize=8,
        fontweight="bold",
        transform=trans_a,
        va="top",
        ha="left",
    )
    trans_b = blended_transform_factory(fig.transFigure, axes_b[0].transAxes)
    axes_b[0].text(
        0.45,
        1.15,
        "b",
        fontsize=8,
        fontweight="bold",
        transform=trans_b,
        va="top",
        ha="left",
    )

    save_figure("figures/main/figure04", close=True)
    print("Saved figures/main/figure04.{pdf,png}")
    print("\nFigure 4 complete!")


if __name__ == "__main__":
    run_demo()
