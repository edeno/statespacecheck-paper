"""Real hippocampal data model comparison for state space model diagnostics.

This script generates Figure 4, which compares per-cell diagnostic metrics
between Continuous and Continuous-Fragmented decoder models on real neural
recording data from hippocampus.

Requires:
- non_local_detector package for decoder models
- Pre-exported neural recording data in data/real/
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any

import numpy as np

from statespacecheck_paper.load_local_data import load_neural_recording_from_files
from statespacecheck_paper.real_data_analysis import (
    compute_model_diagnostics,
    create_decoder_environment,
    fit_decoder_models,
    get_spike_counts,
)
from statespacecheck_paper.real_data_plotting import (
    plot_diagnostic_summary_comparison,
    plot_model_comparison_with_posterior,
)
from statespacecheck_paper.style import save_figure, set_figure_defaults

# Check for optional decoder dependencies
try:
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

# Time parameters for development
TEST_WINDOW_START = 5_000  # Start index for decoding window
TEST_WINDOW_END = 15_000  # End index for decoding window


def run_demo() -> None:
    """Run the full Figure 4 generation pipeline.

    Loads data, fits models, computes diagnostics, and generates figures.
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

    # Fit models
    print("Fitting models...")
    continuous_model, contfrag_model = fit_decoder_models(
        position=position,
        spike_times=spike_times_list,
        time=time,
        environment=env,
    )

    # Run decoding on test window
    test_time = time[TEST_WINDOW_START:TEST_WINDOW_END]
    test_linear_position = linear_position[TEST_WINDOW_START:TEST_WINDOW_END]

    print(f"Decoding {len(test_time)} time points...")

    continuous_results = continuous_model.predict(
        spike_times=spike_times_list,
        time=test_time,
        return_outputs="predictive_posterior",
    )
    contfrag_results = contfrag_model.predict(
        spike_times=spike_times_list,
        time=test_time,
        return_outputs="predictive_posterior",
    )

    # Get spike counts
    spike_counts = get_spike_counts(spike_times_list, test_time)

    # Compute diagnostics
    print("Computing diagnostics...")
    continuous_diagnostics = compute_model_diagnostics(
        continuous_model, continuous_results, spike_counts, test_time
    )
    contfrag_diagnostics = compute_model_diagnostics(
        contfrag_model, contfrag_results, spike_counts, test_time
    )

    # Print summary
    print("\n=== Diagnostic Summary ===")
    for metric in ["hpd_overlap", "kl_divergence", "spike_prob"]:
        cont_mean = np.nanmean(continuous_diagnostics[metric])
        frag_mean = np.nanmean(contfrag_diagnostics[metric])
        print(f"{metric}:")
        print(f"  Continuous: {cont_mean:.4f}")
        print(f"  ContFrag:   {frag_mean:.4f}")

    # Generate figures
    print("\nGenerating Figure 4...")
    set_figure_defaults()

    # Figure 4a: Model comparison with posterior and diagnostics
    fig, axes = plot_model_comparison_with_posterior(
        test_time,
        test_linear_position,
        continuous_results,
        contfrag_results,
        continuous_diagnostics,
        contfrag_diagnostics,
        model_a_name="Continuous",
        model_b_name="Continuous-Fragmented",
    )
    save_figure("figures/main/figure04a", close=True)
    print("Saved figures/main/figure04a.{pdf,png}")

    # Figure 4b: Summary bar chart
    fig, axes = plot_diagnostic_summary_comparison(
        continuous_diagnostics,
        contfrag_diagnostics,
        model_a_name="Continuous",
        model_b_name="Continuous-Fragmented",
    )
    save_figure("figures/main/figure04b", close=True)
    print("Saved figures/main/figure04b.{pdf,png}")

    print("\nFigure 4 complete!")


if __name__ == "__main__":
    run_demo()
