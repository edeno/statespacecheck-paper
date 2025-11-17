"""Generate candidate figures for Figure 3: Real data analysis.

This script runs decoder models on real hippocampal data and generates
candidate figures showing model checking diagnostics. All figures are saved
to figures/candidates/figure03/ for review and selection.

The analysis compares two decoder models:
- Continuous: Standard position decoder
- Continuous-Fragmented: Decoder with fragmented state

Candidate figures generated:
1. HPD overlap distributions (histograms)
2. Model comparison scatter plots
3. Lowest overlap examples for each model
4. Highest difference examples
5. Sustained poor overlap regions
6. HPD overlap vs covariates (speed, position, MUA)
7. HPD overlap mechanics at specific timepoints
8. Ahead/behind distance analysis
9. Fragmented state detection

Run with:
    uv run python scripts/generate_figure03_candidates.py
"""

from __future__ import annotations

import logging

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from non_local_detector import Environment
from non_local_detector.analysis import get_ahead_behind_distance, get_trajectory_data
from non_local_detector.likelihoods.common import get_spikecount_per_time_bin
from non_local_detector.models import (
    ContFragSortedSpikesClassifier,
    SortedSpikesDecoder,
)
from scipy.ndimage import label
from statespacecheck import hpd_overlap
from track_linearization import get_linearized_position

from statespacecheck_paper.load_local_data import load_neural_recording_from_files
from statespacecheck_paper.real_data_analysis import (
    find_sustained_low_overlap,
    get_multiunit_population_firing_rate,
)
from statespacecheck_paper.real_data_plotting import (
    plot_hpd_overlap_at_time,
    plot_model_checking,
    plot_posterior_consistency_vs_array,
    plot_posterior_consistency_vs_covariate,
    plot_single_model_checking,
)
from statespacecheck_paper.style import save_figure, set_figure_defaults

FORMAT = "%(asctime)s %(message)s"
logging.basicConfig(level="INFO", format=FORMAT, datefmt="%d-%b-%y %H:%M:%S")


def load_and_prepare_data() -> tuple:
    """Load neural data and prepare for analysis.

    Returns
    -------
    time : pd.Index
        Time index.
    position : np.ndarray
        Linearized position.
    position_2d : np.ndarray
        2D position coordinates.
    position_info : pd.DataFrame
        Full position information.
    spike_times : list[np.ndarray]
        Spike times for each unit.
    track_graph : nx.Graph
        Track graph.
    edge_order : list[tuple]
        Edge order for linearization.
    edge_spacing : float
        Edge spacing.
    """
    logging.info("Loading data...")
    data = load_neural_recording_from_files(
        data_path="data/",
        animal_date_epoch="j1620210710_02_r1",
    )

    position_info = data["position_info"]
    spike_times = data["spike_times"]
    track_graph = data["track_graph"]
    edge_order = data["linear_edge_order"]
    edge_spacing = 1.5  # Override default

    time = position_info.index
    position_2d = position_info[["head_position_x", "head_position_y"]].to_numpy()
    position = get_linearized_position(
        position_2d,
        track_graph,
        edge_order,
        edge_spacing,
    ).linear_position.values

    return (
        time,
        position,
        position_2d,
        position_info,
        spike_times,
        track_graph,
        edge_order,
        edge_spacing,
    )


def fit_and_predict_models(
    spike_times: list,
    position_2d: np.ndarray,
    time: pd.Index,
    track_graph,
    edge_order: list,
    edge_spacing: float,
) -> tuple:
    """Fit decoder models and generate predictions.

    Parameters
    ----------
    spike_times : list[np.ndarray]
        Spike times.
    position_2d : np.ndarray
        2D position.
    time : pd.Index
        Time index.
    track_graph : nx.Graph
        Track graph.
    edge_order : list
        Edge order.
    edge_spacing : float
        Edge spacing.

    Returns
    -------
    cont_model : SortedSpikesDecoder
        Continuous model.
    cont_frag_model : ContFragSortedSpikesClassifier
        Continuous-fragmented model.
    cont_results : xr.Dataset
        Continuous model results.
    cont_frag_results : xr.Dataset
        Continuous-fragmented model results.
    cont_hpd_overlap : np.ndarray
        HPD overlap for continuous model.
    cont_frag_hpd_overlap : np.ndarray
        HPD overlap for continuous-fragmented model.
    """
    logging.info("Creating environment...")
    env = Environment(
        track_graph=track_graph,
        edge_order=edge_order,
        edge_spacing=edge_spacing,
    )

    logging.info("Fitting continuous model...")
    cont_model = SortedSpikesDecoder(environments=env).fit(
        spike_times=spike_times,
        position=position_2d,
        position_time=time,
    )

    logging.info("Fitting continuous-fragmented model...")
    cont_frag_model = ContFragSortedSpikesClassifier(environments=env).fit(
        spike_times=spike_times,
        position=position_2d,
        position_time=np.asarray(time),
    )

    logging.info("Running continuous model prediction...")
    cont_results = cont_model.predict(
        spike_times=spike_times,
        time=time,
        position_time=time,
        position=position_2d,
        return_outputs=["log_likelihood", "predictive_posterior"],
    )

    logging.info("Running continuous-fragmented model prediction...")
    cont_frag_results = cont_frag_model.predict(
        spike_times=spike_times,
        time=time,
        position_time=time,
        position=position_2d,
        return_outputs=["log_likelihood", "predictive_posterior"],
    )

    logging.info("Computing HPD overlap...")
    cont_hpd_overlap = hpd_overlap(
        state_dist=cont_results.predictive_posterior.dropna("state_bins").to_numpy(),
        likelihood=np.exp(cont_results.log_likelihood.dropna("state_bins").to_numpy()),
    )
    cont_frag_hpd_overlap = hpd_overlap(
        state_dist=cont_frag_results.predictive_posterior.dropna("state_bins").to_numpy(),
        likelihood=np.exp(cont_frag_results.log_likelihood.dropna("state_bins").to_numpy()),
    )

    return (
        cont_model,
        cont_frag_model,
        cont_results,
        cont_frag_results,
        cont_hpd_overlap,
        cont_frag_hpd_overlap,
    )


def generate_overlap_distribution_figures(
    cont_hpd_overlap: np.ndarray,
    cont_frag_hpd_overlap: np.ndarray,
) -> None:
    """Generate HPD overlap distribution comparison figures.

    Parameters
    ----------
    cont_hpd_overlap : np.ndarray
        Continuous model HPD overlap.
    cont_frag_hpd_overlap : np.ndarray
        Continuous-fragmented model HPD overlap.
    """
    logging.info("Generating overlap distribution figures...")

    # Figure: Distribution comparison
    plt.figure(figsize=(12, 4))
    bins = np.linspace(0, 1, 70)

    ax1 = plt.subplot(1, 2, 1)
    ax1.hist(
        cont_hpd_overlap,
        bins=bins,
        alpha=0.7,
        label="Continuous",
        color="tab:blue",
        density=True,
        edgecolor="none",
    )
    ax1.hist(
        cont_frag_hpd_overlap,
        bins=bins,
        alpha=0.7,
        label="ContFrag",
        color="tab:orange",
        density=True,
        edgecolor="none",
    )
    ax1.set_xlabel("HPD Overlap", fontsize=12)
    ax1.set_ylabel("Density", fontsize=12)
    ax1.set_title("Distribution of HPD Overlap", fontsize=14)
    ax1.legend(frameon=False)
    ax1.grid(axis="y", linestyle=":", alpha=0.4)

    ax2 = plt.subplot(1, 2, 2)
    diff_bins = np.linspace(-1, 1, 70)
    ax2.hist(
        cont_frag_hpd_overlap - cont_hpd_overlap,
        bins=diff_bins,
        density=True,
        color="tab:green",
        alpha=0.7,
        edgecolor="none",
    )
    ax2.axvline(0, color="red", linestyle="--", label="Zero Difference")
    ax2.set_xlabel("ContFrag - Cont HPD Overlap", fontsize=12)
    ax2.set_ylabel("Density", fontsize=12)
    ax2.set_title("Difference in HPD Overlap", fontsize=14)
    ax2.legend(frameon=False)
    ax2.grid(axis="y", linestyle=":", alpha=0.4)

    plt.tight_layout()
    save_figure("figures/candidates/figure03/hpd_overlap_distributions")
    plt.close()

    # Figure: Difference histogram
    plt.figure(figsize=(6, 4))
    plt.hist(cont_frag_hpd_overlap - cont_hpd_overlap, bins=70, density=True)
    plt.xlabel("Difference in HPD overlap")
    plt.ylabel("Density")
    plt.title("Posterior Consistency: ContFrag - Cont")
    plt.axvline(0, color="red", linestyle="--", label="Zero Difference")
    plt.legend()
    save_figure("figures/candidates/figure03/hpd_overlap_difference_histogram")
    plt.close()

    # Figure: Scatter comparison
    plt.figure(figsize=(6, 6))
    plt.scatter(
        cont_hpd_overlap,
        cont_frag_hpd_overlap,
        alpha=0.3,
        s=5,
        color="tab:blue",
        label="Posterior Consistency",
    )
    plt.plot([0, 1], [0, 1], color="red", linestyle="--", label="y = x")
    plt.xlabel("Continuous HPD Overlap", fontsize=12)
    plt.ylabel("ContFrag HPD Overlap", fontsize=12)
    plt.title("Posterior Consistency: Continuous vs ContFrag", fontsize=14)
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.grid(True, linestyle=":", alpha=0.7)
    plt.legend()
    plt.tight_layout()
    save_figure("figures/candidates/figure03/model_comparison_scatter")
    plt.close()


def generate_lowest_overlap_examples(
    cont_hpd_overlap: np.ndarray,
    cont_frag_hpd_overlap: np.ndarray,
    time: pd.Index,
    position: np.ndarray,
    spike_times: list,
    cont_model,
    cont_frag_model,
    cont_results,
    cont_frag_results,
) -> None:
    """Generate figures showing lowest HPD overlap examples.

    Parameters
    ----------
    cont_hpd_overlap : np.ndarray
        Continuous model HPD overlap.
    cont_frag_hpd_overlap : np.ndarray
        Continuous-fragmented model HPD overlap.
    time : pd.Index
        Time index.
    position : np.ndarray
        Position.
    spike_times : list
        Spike times.
    cont_model : Model
        Continuous model.
    cont_frag_model : Model
        Continuous-fragmented model.
    cont_results : xr.Dataset
        Continuous results.
    cont_frag_results : xr.Dataset
        Continuous-fragmented results.
    """
    logging.info("Generating lowest overlap examples...")

    low_cont_inds = np.argsort(cont_hpd_overlap)[:3]
    low_cont_frag_inds = np.argsort(cont_frag_hpd_overlap)[:3]

    # Continuous model examples
    for i, ind in enumerate(low_cont_inds):
        logging.info(
            f"Continuous model, example {i + 1}, time index: {ind}, "
            f"HPD overlap: {cont_hpd_overlap[ind]:.3f}"
        )
        fig, _ = plot_single_model_checking(
            slice(ind - 500, ind + 500),
            time,
            position,
            spike_times,
            cont_model,
            cont_results,
            cont_hpd_overlap,
            overlap_threshold=0.2,
            model_label="Continuous",
            color="tab:blue",
        )
        save_figure(f"figures/candidates/figure03/lowest_overlap_cont_{i + 1:02d}")
        plt.close(fig)

    # Continuous-fragmented model examples
    for i, ind in enumerate(low_cont_frag_inds):
        logging.info(
            f"ContFrag model, example {i + 1}, time index: {ind}, "
            f"HPD overlap: {cont_frag_hpd_overlap[ind]:.3f}"
        )
        fig, _ = plot_single_model_checking(
            slice(ind - 500, ind + 500),
            time,
            position,
            spike_times,
            cont_frag_model,
            cont_frag_results,
            cont_frag_hpd_overlap,
            overlap_threshold=0.2,
            model_label="ContFrag",
            color="tab:orange",
        )
        save_figure(f"figures/candidates/figure03/lowest_overlap_contfrag_{i + 1:02d}")
        plt.close(fig)


def generate_highest_difference_examples(
    cont_hpd_overlap: np.ndarray,
    cont_frag_hpd_overlap: np.ndarray,
    time: pd.Index,
    position: np.ndarray,
    spike_times: list,
    cont_model,
    cont_frag_model,
    cont_results,
    cont_frag_results,
) -> None:
    """Generate figures showing highest difference examples.

    Parameters
    ----------
    cont_hpd_overlap : np.ndarray
        Continuous model HPD overlap.
    cont_frag_hpd_overlap : np.ndarray
        Continuous-fragmented model HPD overlap.
    time : pd.Index
        Time index.
    position : np.ndarray
        Position.
    spike_times : list
        Spike times.
    cont_model : Model
        Continuous model.
    cont_frag_model : Model
        Continuous-fragmented model.
    cont_results : xr.Dataset
        Continuous results.
    cont_frag_results : xr.Dataset
        Continuous-fragmented results.
    """
    logging.info("Generating highest difference examples...")

    diff = cont_frag_hpd_overlap - cont_hpd_overlap
    n_examples = 3

    high_pos_inds = np.argsort(diff)[-n_examples:]
    high_neg_inds = np.argsort(diff)[:n_examples]

    # Highest positive differences
    for i, ind in enumerate(high_pos_inds):
        logging.info(
            f"High positive diff example {i + 1}, time index: {ind}, diff: {diff[ind]:.3f}"
        )
        # Continuous model
        fig, _ = plot_single_model_checking(
            slice(ind - 500, ind + 500),
            time,
            position,
            spike_times,
            cont_model,
            cont_results,
            cont_hpd_overlap,
            overlap_threshold=0.2,
            model_label="Continuous",
            color="tab:blue",
        )
        save_figure(f"figures/candidates/figure03/high_pos_diff_{i + 1:02d}_cont")
        plt.close(fig)

        # Continuous-fragmented model
        fig, _ = plot_single_model_checking(
            slice(ind - 500, ind + 500),
            time,
            position,
            spike_times,
            cont_frag_model,
            cont_frag_results,
            cont_frag_hpd_overlap,
            overlap_threshold=0.2,
            model_label="ContFrag",
            color="tab:orange",
        )
        save_figure(f"figures/candidates/figure03/high_pos_diff_{i + 1:02d}_contfrag")
        plt.close(fig)

    # Highest negative differences
    for i, ind in enumerate(high_neg_inds):
        logging.info(
            f"High negative diff example {i + 1}, time index: {ind}, diff: {diff[ind]:.3f}"
        )
        # Continuous model
        fig, _ = plot_single_model_checking(
            slice(ind - 500, ind + 500),
            time,
            position,
            spike_times,
            cont_model,
            cont_results,
            cont_hpd_overlap,
            overlap_threshold=0.2,
            model_label="Continuous",
            color="tab:blue",
        )
        save_figure(f"figures/candidates/figure03/high_neg_diff_{i + 1:02d}_cont")
        plt.close(fig)

        # Continuous-fragmented model
        fig, _ = plot_single_model_checking(
            slice(ind - 500, ind + 500),
            time,
            position,
            spike_times,
            cont_frag_model,
            cont_frag_results,
            cont_frag_hpd_overlap,
            overlap_threshold=0.2,
            model_label="ContFrag",
            color="tab:orange",
        )
        save_figure(f"figures/candidates/figure03/high_neg_diff_{i + 1:02d}_contfrag")
        plt.close(fig)


def generate_sustained_region_figures(
    cont_hpd_overlap: np.ndarray,
    cont_frag_hpd_overlap: np.ndarray,
    time: pd.Index,
    position: np.ndarray,
    spike_times: list,
    cont_model,
    cont_frag_model,
    cont_results,
    cont_frag_results,
) -> None:
    """Generate figures for sustained poor overlap regions.

    Parameters
    ----------
    cont_hpd_overlap : np.ndarray
        Continuous model HPD overlap.
    cont_frag_hpd_overlap : np.ndarray
        Continuous-fragmented model HPD overlap.
    time : pd.Index
        Time index.
    position : np.ndarray
        Position.
    spike_times : list
        Spike times.
    cont_model : Model
        Continuous model.
    cont_frag_model : Model
        Continuous-fragmented model.
    cont_results : xr.Dataset
        Continuous results.
    cont_frag_results : xr.Dataset
        Continuous-fragmented results.
    """
    logging.info("Finding sustained poor overlap regions...")

    cont_regions = find_sustained_low_overlap(
        cont_hpd_overlap, threshold=0.5, min_duration=0.010, sampling_frequency=500
    )
    cont_frag_regions = find_sustained_low_overlap(
        cont_frag_hpd_overlap,
        threshold=0.5,
        min_duration=0.010,
        sampling_frequency=500,
    )

    logging.info(f"Continuous model: {len(cont_regions)} sustained regions")
    logging.info(f"ContFrag model: {len(cont_frag_regions)} sustained regions")

    n_regions = 10  # Number of regions to plot

    for idx, regions in enumerate([cont_regions, cont_frag_regions]):
        model = cont_model if idx == 0 else cont_frag_model
        results = cont_results if idx == 0 else cont_frag_results
        hpd_overlap = cont_hpd_overlap if idx == 0 else cont_frag_hpd_overlap
        model_label = "Continuous" if idx == 0 else "ContFrag"
        color = "tab:blue" if idx == 0 else "tab:orange"
        other_model = cont_frag_model if idx == 0 else cont_model
        other_results = cont_frag_results if idx == 0 else cont_results
        other_hpd_overlap = cont_frag_hpd_overlap if idx == 0 else cont_hpd_overlap
        other_label = "ContFrag" if idx == 0 else "Continuous"
        other_color = "tab:orange" if idx == 0 else "tab:blue"

        for n, (start, end) in enumerate(regions[:n_regions]):
            start_ext = max(0, start - 500)
            end_ext = min(len(time), end + 500)
            logging.info(
                f"Plotting {model_label} sustained region {n + 1}: {start_ext} to {end_ext}"
            )

            # Primary model
            fig, _ = plot_single_model_checking(
                slice(start_ext, end_ext),
                time,
                position,
                spike_times,
                model,
                results,
                hpd_overlap,
                overlap_threshold=0.5,
                model_label=model_label,
                color=color,
            )
            save_figure(
                f"figures/candidates/figure03/sustained_{model_label.lower()}_{n + 1:02d}_primary"
            )
            plt.close(fig)

            # Comparison model
            fig, _ = plot_single_model_checking(
                slice(start_ext, end_ext),
                time,
                position,
                spike_times,
                other_model,
                other_results,
                other_hpd_overlap,
                overlap_threshold=0.5,
                model_label=other_label,
                color=other_color,
            )
            filename = f"sustained_{model_label.lower()}_{n + 1:02d}_comparison"
            save_figure(f"figures/candidates/figure03/{filename}")
            plt.close(fig)


def generate_covariate_figures(
    cont_hpd_overlap: np.ndarray,
    cont_frag_hpd_overlap: np.ndarray,
    position_info: pd.DataFrame,
    spike_times: list,
    time: pd.Index,
) -> None:
    """Generate HPD overlap vs covariate figures.

    Parameters
    ----------
    cont_hpd_overlap : np.ndarray
        Continuous model HPD overlap.
    cont_frag_hpd_overlap : np.ndarray
        Continuous-fragmented model HPD overlap.
    position_info : pd.DataFrame
        Position information.
    spike_times : list
        Spike times.
    time : pd.Index
        Time index.
    """
    logging.info("Generating covariate comparison figures...")

    # Speed
    fig, _ = plot_posterior_consistency_vs_covariate(
        covariate="head_speed",
        hpd_overlap=cont_hpd_overlap,
        position_info=position_info,
        suptitle="Cont Model Posterior Consistency vs Speed",
    )
    save_figure("figures/candidates/figure03/overlap_vs_speed_cont")
    plt.close(fig)

    fig, _ = plot_posterior_consistency_vs_covariate(
        covariate="head_speed",
        hpd_overlap=cont_frag_hpd_overlap,
        position_info=position_info,
        suptitle="ContFrag Model Posterior Consistency vs Speed",
    )
    save_figure("figures/candidates/figure03/overlap_vs_speed_contfrag")
    plt.close(fig)

    # Linear position
    fig, _ = plot_posterior_consistency_vs_covariate(
        covariate="linear_position",
        hpd_overlap=cont_hpd_overlap,
        position_info=position_info,
        suptitle="Cont Model Posterior Consistency vs Position",
    )
    save_figure("figures/candidates/figure03/overlap_vs_position_cont")
    plt.close(fig)

    fig, _ = plot_posterior_consistency_vs_covariate(
        covariate="linear_position",
        hpd_overlap=cont_frag_hpd_overlap,
        position_info=position_info,
        suptitle="ContFrag Model Posterior Consistency vs Position",
    )
    save_figure("figures/candidates/figure03/overlap_vs_position_contfrag")
    plt.close(fig)

    # Multiunit population firing rate
    logging.info("Computing multiunit population firing rate...")
    spikecount = np.stack(
        [get_spikecount_per_time_bin(spike_times=st, time=time) for st in spike_times],
        axis=1,
    )

    multiunit_population_firing_rate = get_multiunit_population_firing_rate(
        multiunit=spikecount,
        sampling_frequency=1 / (time[1] - time[0]),
        smoothing_sigma=0.015,
    )

    fig, _ = plot_posterior_consistency_vs_array(
        x=multiunit_population_firing_rate,
        covariate_label="MUA",
        hpd_overlap=cont_hpd_overlap,
        suptitle="Cont Model Posterior Consistency vs MUA",
    )
    save_figure("figures/candidates/figure03/overlap_vs_mua_cont")
    plt.close(fig)

    fig, _ = plot_posterior_consistency_vs_array(
        x=multiunit_population_firing_rate,
        covariate_label="MUA",
        hpd_overlap=cont_frag_hpd_overlap,
        suptitle="ContFrag Model Posterior Consistency vs MUA",
    )
    save_figure("figures/candidates/figure03/overlap_vs_mua_contfrag")
    plt.close(fig)

    # MUA during immobility
    head_speed_interp = (
        pd.Series(position_info["head_speed"].values, index=position_info.index)
        .reindex(time)
        .interpolate()
        .values
    )
    immobility_mask = head_speed_interp < 4.0

    fig, _ = plot_posterior_consistency_vs_array(
        x=multiunit_population_firing_rate[immobility_mask],
        covariate_label="MUA (Immobility Only)",
        hpd_overlap=cont_hpd_overlap[immobility_mask],
        suptitle="Cont Model Posterior Consistency vs MUA (Immobility)",
    )
    save_figure("figures/candidates/figure03/overlap_vs_mua_immobile_cont")
    plt.close(fig)

    fig, _ = plot_posterior_consistency_vs_array(
        x=multiunit_population_firing_rate[immobility_mask],
        covariate_label="MUA (Immobility Only)",
        hpd_overlap=cont_frag_hpd_overlap[immobility_mask],
        suptitle="ContFrag Model Posterior Consistency vs MUA (Immobility)",
    )
    save_figure("figures/candidates/figure03/overlap_vs_mua_immobile_contfrag")
    plt.close(fig)

    # MUA during mobility
    mobility_mask = head_speed_interp >= 10.0

    fig, _ = plot_posterior_consistency_vs_array(
        x=multiunit_population_firing_rate[mobility_mask],
        covariate_label="MUA (Mobility Only)",
        hpd_overlap=cont_hpd_overlap[mobility_mask],
        suptitle="Cont Model Posterior Consistency vs MUA (Mobility)",
    )
    save_figure("figures/candidates/figure03/overlap_vs_mua_mobile_cont")
    plt.close(fig)

    fig, _ = plot_posterior_consistency_vs_array(
        x=multiunit_population_firing_rate[mobility_mask],
        covariate_label="MUA (Mobility Only)",
        hpd_overlap=cont_frag_hpd_overlap[mobility_mask],
        suptitle="ContFrag Model Posterior Consistency vs MUA (Mobility)",
    )
    save_figure("figures/candidates/figure03/overlap_vs_mua_mobile_contfrag")
    plt.close(fig)


def generate_mechanics_figures(
    cont_results,
    spike_times: list,
    time: pd.Index,
    cont_hpd_overlap: np.ndarray,
    track_graph,
    edge_order: list,
    edge_spacing: float,
) -> None:
    """Generate HPD overlap mechanics figures at specific timepoints.

    Parameters
    ----------
    cont_results : xr.Dataset
        Continuous model results.
    spike_times : list
        Spike times.
    time : pd.Index
        Time index.
    cont_hpd_overlap : np.ndarray
        Continuous model HPD overlap.
    track_graph : nx.Graph
        Track graph.
    edge_order : list
        Edge order.
    edge_spacing : float
        Edge spacing.
    """
    logging.info("Generating HPD overlap mechanics figures...")

    spike_counts = np.stack(
        [get_spikecount_per_time_bin(st, time=cont_results.time.to_numpy()) for st in spike_times],
        axis=1,
    )

    timepoints = [41, 42, 44, 709133, 195314]

    for t in timepoints:
        logging.info(f"Generating mechanics figure for timepoint {t}")
        fig, _ = plot_hpd_overlap_at_time(
            cont_results,
            spike_counts,
            cont_hpd_overlap,
            t,
            track_graph,
            edge_order,
            edge_spacing,
        )
        save_figure(f"figures/candidates/figure03/hpd_mechanics_t{t:06d}")
        plt.close(fig)


def generate_ahead_behind_figures(
    cont_results,
    cont_frag_results,
    cont_hpd_overlap: np.ndarray,
    cont_frag_hpd_overlap: np.ndarray,
    position_info: pd.DataFrame,
    track_graph,
    cont_model,
    cont_frag_model,
) -> None:
    """Generate ahead/behind distance analysis figures.

    Parameters
    ----------
    cont_results : xr.Dataset
        Continuous model results.
    cont_frag_results : xr.Dataset
        Continuous-fragmented model results.
    cont_hpd_overlap : np.ndarray
        Continuous model HPD overlap.
    cont_frag_hpd_overlap : np.ndarray
        Continuous-fragmented model HPD overlap.
    position_info : pd.DataFrame
        Position information.
    track_graph : nx.Graph
        Track graph.
    cont_model : Model
        Continuous model.
    cont_frag_model : Model
        Continuous-fragmented model.
    """
    logging.info("Computing ahead/behind distance for continuous model...")

    (
        actual_projected_position,
        actual_edges,
        actual_orientation,
        mental_position_2d,
        mental_position_edges,
    ) = get_trajectory_data(
        posterior=cont_results.predictive_posterior.unstack("state_bins").squeeze(),
        track_graph=track_graph,
        decoder=cont_model,
        actual_projected_position=position_info[["projected_x_position", "projected_y_position"]],
        track_segment_id=position_info["track_segment_id"],
        actual_orientation=position_info["head_orientation"],
    )

    cont_dist = get_ahead_behind_distance(
        track_graph=track_graph,
        actual_projected_position=actual_projected_position,
        actual_edges=actual_edges,
        actual_orientation=actual_orientation,
        mental_position_2d=mental_position_2d,
        mental_position_edges=mental_position_edges,
    )

    fig, ax = plt.subplots(figsize=(6, 3))
    ax.scatter(
        cont_dist,
        cont_hpd_overlap,
        s=2,
        alpha=0.15,
        color="tab:blue",
        edgecolor="none",
    )
    ax.set_xlabel("Ahead/Behind Distance", fontsize=11)
    ax.set_ylabel("HPD Overlap", fontsize=11)
    ax.set_title("Posterior Consistency vs Ahead/Behind (Cont)", fontsize=13, pad=8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.yaxis.grid(True, linestyle=":", alpha=0.4)
    ax.xaxis.grid(False)
    ax.set_xlim((-250, 250))
    ax.axvline(0, color="grey", linestyle="--", label="Zero Distance")
    plt.tight_layout()
    save_figure("figures/candidates/figure03/overlap_vs_ahead_behind_cont")
    plt.close(fig)

    logging.info("Computing ahead/behind distance for cont-frag model...")

    (
        actual_projected_position,
        actual_edges,
        actual_orientation,
        mental_position_2d,
        mental_position_edges,
    ) = get_trajectory_data(
        posterior=cont_frag_results.predictive_posterior.unstack("state_bins").sum("state"),
        track_graph=track_graph,
        decoder=cont_frag_model,
        actual_projected_position=position_info[["projected_x_position", "projected_y_position"]],
        track_segment_id=position_info["track_segment_id"],
        actual_orientation=position_info["head_orientation"],
    )

    cont_frag_dist = get_ahead_behind_distance(
        track_graph=track_graph,
        actual_projected_position=actual_projected_position,
        actual_edges=actual_edges,
        actual_orientation=actual_orientation,
        mental_position_2d=mental_position_2d,
        mental_position_edges=mental_position_edges,
    )

    fig, ax = plt.subplots(figsize=(6, 3))
    ax.scatter(
        cont_frag_dist,
        cont_frag_hpd_overlap,
        s=2,
        alpha=0.15,
        color="tab:orange",
        edgecolor="none",
    )
    ax.set_xlabel("Ahead/Behind Distance", fontsize=11)
    ax.set_ylabel("HPD Overlap", fontsize=11)
    ax.set_title("Posterior Consistency vs Ahead/Behind (ContFrag)", fontsize=13, pad=8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.yaxis.grid(True, linestyle=":", alpha=0.4)
    ax.xaxis.grid(False)
    ax.set_xlim((-250, 250))
    ax.axvline(0, color="grey", linestyle="--", label="Zero Distance")
    plt.tight_layout()
    save_figure("figures/candidates/figure03/overlap_vs_ahead_behind_contfrag")
    plt.close(fig)


def generate_fragmented_state_figures(
    cont_frag_results,
    time: pd.Index,
    position: np.ndarray,
    spike_times: list,
    cont_model,
    cont_frag_model,
    cont_results,
    cont_hpd_overlap: np.ndarray,
    cont_frag_hpd_overlap: np.ndarray,
) -> None:
    """Generate fragmented state detection figures.

    Parameters
    ----------
    cont_frag_results : xr.Dataset
        Continuous-fragmented model results.
    time : pd.Index
        Time index.
    position : np.ndarray
        Position.
    spike_times : list
        Spike times.
    cont_model : Model
        Continuous model.
    cont_frag_model : Model
        Continuous-fragmented model.
    cont_results : xr.Dataset
        Continuous results.
    cont_hpd_overlap : np.ndarray
        Continuous model HPD overlap.
    cont_frag_hpd_overlap : np.ndarray
        Continuous-fragmented model HPD overlap.
    """
    logging.info("Finding sustained fragmented state regions...")

    frag_mask = (
        cont_frag_results.acausal_state_probabilities.sel(states="Fragmented") > 0.8
    ).values

    labels_array, n_labels = label(frag_mask)
    min_samples = 5

    sustained_fragments = [
        (np.where(labels_array == i)[0][0], np.where(labels_array == i)[0][-1])
        for i in range(1, n_labels + 1)
        if np.sum(labels_array == i) >= min_samples
    ]

    logging.info(f"Found {len(sustained_fragments)} sustained fragmented regions")

    for i, (start, end) in enumerate(sustained_fragments[:10]):
        start_ext = max(0, start - 50)
        end_ext = min(len(time), end + 50)
        logging.info(f"Plotting fragmented region {i + 1}: {start_ext} to {end_ext}")

        fig, _ = plot_model_checking(
            slice(start_ext, end_ext),
            time,
            position,
            spike_times,
            cont_model,
            cont_frag_model,
            cont_results,
            cont_frag_results,
            cont_hpd_overlap,
            cont_frag_hpd_overlap,
            overlap_threshold=0.2,
        )
        save_figure(f"figures/candidates/figure03/fragmented_region_{i + 1:02d}")
        plt.close(fig)


def main() -> None:
    """Generate all candidate figures for Figure 3."""
    set_figure_defaults()

    # Load and prepare data
    (
        time,
        position,
        position_2d,
        position_info,
        spike_times,
        track_graph,
        edge_order,
        edge_spacing,
    ) = load_and_prepare_data()

    # Fit models and generate predictions
    (
        cont_model,
        cont_frag_model,
        cont_results,
        cont_frag_results,
        cont_hpd_overlap,
        cont_frag_hpd_overlap,
    ) = fit_and_predict_models(
        spike_times, position_2d, time, track_graph, edge_order, edge_spacing
    )

    # Generate all candidate figures
    generate_overlap_distribution_figures(cont_hpd_overlap, cont_frag_hpd_overlap)

    generate_lowest_overlap_examples(
        cont_hpd_overlap,
        cont_frag_hpd_overlap,
        time,
        position,
        spike_times,
        cont_model,
        cont_frag_model,
        cont_results,
        cont_frag_results,
    )

    generate_highest_difference_examples(
        cont_hpd_overlap,
        cont_frag_hpd_overlap,
        time,
        position,
        spike_times,
        cont_model,
        cont_frag_model,
        cont_results,
        cont_frag_results,
    )

    generate_sustained_region_figures(
        cont_hpd_overlap,
        cont_frag_hpd_overlap,
        time,
        position,
        spike_times,
        cont_model,
        cont_frag_model,
        cont_results,
        cont_frag_results,
    )

    generate_covariate_figures(
        cont_hpd_overlap, cont_frag_hpd_overlap, position_info, spike_times, time
    )

    generate_mechanics_figures(
        cont_results,
        spike_times,
        time,
        cont_hpd_overlap,
        track_graph,
        edge_order,
        edge_spacing,
    )

    generate_ahead_behind_figures(
        cont_results,
        cont_frag_results,
        cont_hpd_overlap,
        cont_frag_hpd_overlap,
        position_info,
        track_graph,
        cont_model,
        cont_frag_model,
    )

    generate_fragmented_state_figures(
        cont_frag_results,
        time,
        position,
        spike_times,
        cont_model,
        cont_frag_model,
        cont_results,
        cont_hpd_overlap,
        cont_frag_hpd_overlap,
    )

    logging.info("All candidate figures generated successfully!")


if __name__ == "__main__":
    main()
