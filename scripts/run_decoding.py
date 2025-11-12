from __future__ import annotations

import logging

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from non_local_detector import Environment
from non_local_detector.analysis import get_ahead_behind_distance, get_trajectory_data
from non_local_detector.likelihoods.common import get_spikecount_per_time_bin
from non_local_detector.model_checking.highest_posterior_density import (
    get_highest_posterior_threshold,
)
from non_local_detector.models import (
    ContFragSortedSpikesClassifier,
    SortedSpikesDecoder,
)
from scipy.ndimage import gaussian_filter1d, label
from statespacecheck import hpd_overlap
from track_linearization import get_linearized_position, plot_graph_as_1D

FORMAT = "%(asctime)s %(message)s"
logging.basicConfig(level="INFO", format=FORMAT, datefmt="%d-%b-%y %H:%M:%S")


def load_data():
    path = "/Users/edeno/Downloads/"
    animal_date_epoch = "j1620210710_02_r1"
    position_info = pd.read_pickle(path + f"{animal_date_epoch}_position_info.pkl")
    spike_times = joblib.load(path + f"{animal_date_epoch}_HPC_spike_times.pkl")
    track_graph = joblib.load(path + f"{animal_date_epoch}_track_graph.pkl")
    linear_edge_order = joblib.load(path + f"{animal_date_epoch}_linear_edge_order.pkl")
    linear_edge_spacing = joblib.load(
        path + f"{animal_date_epoch}_linear_edge_spacing.pkl"
    )

    return (
        position_info,
        spike_times,
        track_graph,
        linear_edge_order,
        linear_edge_spacing,
    )


position_info, spike_times, track_graph, edge_order, edge_spacing = load_data()
edge_spacing = 1.5
time = position_info.index
position_2d = position_info[["head_position_x", "head_position_y"]].to_numpy()
position = get_linearized_position(
    position_2d,
    track_graph,
    edge_order,
    edge_spacing,
).linear_position.values


env = Environment(
    track_graph=track_graph,
    edge_order=edge_order,
    edge_spacing=edge_spacing,
)


cont_model = SortedSpikesDecoder(
    environments=env,
).fit(
    spike_times=spike_times,
    position=position_2d,
    position_time=time,
)

cont_frag_model = ContFragSortedSpikesClassifier(
    environments=env,
).fit(
    spike_times=spike_times,
    position=position_2d,
    position_time=np.asarray(time),
)

cont_results = cont_model.predict(
    spike_times=spike_times,
    time=time,
    position_time=time,
    position=position_2d,
    return_outputs=["log_likelihood", "predictive_posterior"],
)

cont_frag_results = cont_frag_model.predict(
    spike_times=spike_times,
    time=time,
    position_time=time,
    position=position_2d,
    return_outputs=["log_likelihood", "predictive_posterior"],
)

print(cont_results)
cont_results.isel(time=0).predictive_posterior.unstack("state_bins").squeeze().plot()


cont_frag_results.isel(time=0).predictive_posterior.unstack("state_bins").sum(
    "state"
).plot(x="position")


cont_hpd_overlap = hpd_overlap(
    state_dist=cont_results.predictive_posterior.dropna("state_bins").to_numpy(),
    likelihood=np.exp(cont_results.log_likelihood.dropna("state_bins").to_numpy()),
)
cont_frag_hpd_overlap = hpd_overlap(
    state_dist=cont_frag_results.predictive_posterior.dropna("state_bins").to_numpy(),
    likelihood=np.exp(cont_frag_results.log_likelihood.dropna("state_bins").to_numpy()),
)


def plot_raster(spike_times, time_slice, ax=None, sort_order=None, **eventplot_kwargs):
    """Plot spike raster for a given time slice."""
    if ax is None:
        ax = plt.gca()
    time_slice_spike_times = [
        neuron_spike_times[
            (neuron_spike_times >= time_slice.start)
            & (neuron_spike_times < time_slice.stop)
        ]
        for neuron_spike_times in spike_times
    ]
    if sort_order is not None:
        time_slice_spike_times = [time_slice_spike_times[i] for i in sort_order]
    ax.eventplot(
        time_slice_spike_times,
        linelengths=0.5,
        colors="black",
        rasterized=True,
        **eventplot_kwargs,
    )
    ax.set_ylabel("Neuron")
    ax.set_xlabel("Time")


def plot_posterior(
    posterior,
    time,
    position,
    time_slice_ind,
    ax=None,
    title=None,
    scatter_kwargs=None,
    **plot_kwargs,
):
    """Plot posterior probability as an image with position trace."""
    if ax is None:
        ax = plt.gca()
    im = (
        posterior.isel(time=time_slice_ind)
        .unstack("state_bins")
        .sum("state")
        .plot(
            x="time",
            y="position",
            ax=ax,
            add_colorbar=False,
            robust=True,
            cmap="bone_r",
            rasterized=True,
            **plot_kwargs,
        )
    )
    if scatter_kwargs is None:
        scatter_kwargs = {
            "color": "magenta",
            "s": 1,
            "rasterized": True,
            "clip_on": False,
        }
    ax.scatter(
        time[time_slice_ind],
        position[time_slice_ind],
        **scatter_kwargs,
    )
    if title:
        ax.set_title(title)
    ax.set_ylabel("Position")
    return im


def plot_overlap_regions(
    ax, time, time_slice_ind, hpd_overlap, threshold=0.2, color="tab:blue", alpha=0.5
):
    """Highlight regions where HPD overlap is below threshold."""
    labels, n_labels = label(hpd_overlap[time_slice_ind] < threshold)
    for label_ in range(1, n_labels + 1):
        bad_overlap_ind = np.where(labels == label_)[0]
        ax.axvspan(
            time[time_slice_ind][bad_overlap_ind[0]],
            time[time_slice_ind][bad_overlap_ind[-1]],
            color=color,
            alpha=alpha,
        )


def plot_overlap_trace(
    time, time_slice_ind, overlaps, labels=None, ax=None, **plot_kwargs
):
    """Plot HPD overlap traces."""
    if ax is None:
        ax = plt.gca()
    if isinstance(overlaps, list | tuple):
        for overlap, label in zip(
            overlaps, labels or [None] * len(overlaps), strict=False
        ):
            ax.plot(
                time[time_slice_ind],
                overlap[time_slice_ind],
                label=label,
                **plot_kwargs,
            )
    else:
        ax.plot(
            time[time_slice_ind], overlaps[time_slice_ind], label=labels, **plot_kwargs
        )
    if labels:
        ax.legend()
    ax.set_ylabel("HPD Overlap")
    ax.set_xlabel("Time")


def plot_acausal_state_prob(
    results,
    time_slice_ind,
    ax=None,
    title=None,
    **plot_kwargs,
):
    """Plot acausal state probability as an image with position trace."""
    if ax is None:
        ax = plt.gca()
    im = results.acausal_state_probabilities.isel(time=time_slice_ind).plot(
        x="time",
        hue="states",
        ax=ax,
        rasterized=True,
        **plot_kwargs,
    )
    ax.set_ylim((0.0, 1.05))
    if title:
        ax.set_title(title)
    ax.set_ylabel("State Prob.")
    return im


# Example of composing a figure using the above:
def plot_model_checking(
    time_slice_ind,
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
):
    time_slice = slice(time[time_slice_ind.start], time[time_slice_ind.stop])
    sort_order = np.argsort(
        cont_model.environments[0]
        .place_bin_centers_[
            cont_model.encoding_model_[("", 0)]["place_fields"].argmax(axis=1)
        ]
        .squeeze()
    )
    fig, axes = plt.subplots(4, 1, figsize=(7, 8), sharex=True, constrained_layout=True)
    plot_posterior(
        cont_results.predictive_posterior,
        time,
        position,
        time_slice_ind,
        ax=axes[0],
        title="Continuous Decoder",
    )
    plot_overlap_regions(
        axes[0],
        time,
        time_slice_ind,
        cont_hpd_overlap,
        threshold=overlap_threshold,
        color="tab:blue",
    )
    plot_posterior(
        cont_frag_results.predictive_posterior,
        time,
        position,
        time_slice_ind,
        ax=axes[1],
        title="Continuous Fragmented Decoder",
    )
    plot_overlap_regions(
        axes[1],
        time,
        time_slice_ind,
        cont_frag_hpd_overlap,
        threshold=overlap_threshold,
        color="tab:orange",
    )
    plot_raster(spike_times, time_slice, ax=axes[2], sort_order=sort_order)
    plot_overlap_regions(
        axes[2],
        time,
        time_slice_ind,
        cont_hpd_overlap,
        threshold=overlap_threshold,
        color="tab:blue",
    )
    plot_overlap_regions(
        axes[2],
        time,
        time_slice_ind,
        cont_frag_hpd_overlap,
        threshold=overlap_threshold,
        color="tab:orange",
    )
    plot_overlap_trace(
        time,
        time_slice_ind,
        [cont_hpd_overlap, cont_frag_hpd_overlap],
        labels=["Continuous", "Continuous Fragmented"],
        ax=axes[3],
    )
    plt.show()


def plot_single_model_checking(
    time_slice_ind,
    time,
    position,
    spike_times,
    model,
    results,
    hpd_overlap,
    overlap_threshold=0.2,
    model_label="Model",
    color="tab:blue",
):
    # Keep 16:9 ratio but set height to 4 inches
    width = 16 / 9 * 4  # width = aspect_ratio * height
    fig, axes = plt.subplots(
        4, 1, figsize=(width, 4), sharex=True, constrained_layout=True
    )
    # Posterior
    plot_posterior(
        results.predictive_posterior,
        time,
        position,
        time_slice_ind,
        ax=axes[0],
        title=f"{model_label} Decoder",
    )
    plot_overlap_regions(
        axes[0],
        time,
        time_slice_ind,
        hpd_overlap,
        threshold=overlap_threshold,
        color=color,
    )
    axes[0].set_xlabel("")

    # Raster
    sort_order = np.argsort(
        model.environments[0]
        .place_bin_centers_[
            model.encoding_model_[("", 0)]["place_fields"].argmax(axis=1)
        ]
        .squeeze()
    )
    plot_raster(
        spike_times,
        slice(time[time_slice_ind.start], time[time_slice_ind.stop]),
        ax=axes[1],
        sort_order=sort_order,
    )
    plot_overlap_regions(
        axes[1],
        time,
        time_slice_ind,
        hpd_overlap,
        threshold=overlap_threshold,
        color=color,
    )
    axes[1].set_xlabel("")

    # HPD overlap trace
    plot_overlap_trace(
        time,
        time_slice_ind,
        hpd_overlap,
        labels=model_label,
        ax=axes[2],
        color=color,
    )
    axes[2].set_xlabel("")
    # State probability plot
    plot_acausal_state_prob(
        results,
        time_slice_ind,
        ax=axes[3],
        title="",
    )
    axes[3].set_title("")
    axes[3].set_xlabel("Time")
    plt.show()


low_cont_inds = np.argsort(cont_hpd_overlap)[:3]
low_cont_frag_inds = np.argsort(cont_frag_hpd_overlap)[:3]

# Show 3 lowest HPD overlap examples for Continuous model
for i, ind in enumerate(low_cont_inds):
    print(
        f"Continuous model, example {i+1}, time index: {ind}, "
        f"HPD overlap: {cont_hpd_overlap[ind]:.3f}"
    )
    plot_single_model_checking(
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

# Show 3 lowest HPD overlap examples for ContFrag model
for i, ind in enumerate(low_cont_frag_inds):
    print(
        f"ContFrag model, example {i+1}, time index: {ind},"
        f" HPD overlap: {cont_frag_hpd_overlap[ind]:.3f}"
    )
    plot_single_model_checking(
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


plt.figure(figsize=(12, 4))
bins = np.linspace(0, 1, 70)

# HPD overlap distributions
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

# Histogram of differences
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
plt.show()


# Find indices where the difference is highest in positive and negative directions
diff = cont_frag_hpd_overlap - cont_hpd_overlap

n_examples = 3  # number of examples to show for each direction

# Highest positive differences (ContFrag much higher than Cont)
high_pos_inds = np.argsort(diff)[-n_examples:]
# Highest negative differences (Cont much higher than ContFrag)
high_neg_inds = np.argsort(diff)[:n_examples]

for i, ind in enumerate(high_pos_inds):
    print(f"High positive diff example {i+1}, time index: {ind}, diff: {diff[ind]:.3f}")
    plot_single_model_checking(
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
    plot_single_model_checking(
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

for i, ind in enumerate(high_neg_inds):
    print(f"High negative diff example {i+1}, time index: {ind}, diff: {diff[ind]:.3f}")
    plot_single_model_checking(
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
    plot_single_model_checking(
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


def find_sustained_low_overlap(
    hpd_overlap,
    threshold=0.5,
    min_duration=0.010,
    smooth_sigma=1.0,
    sampling_frequency=1,
):
    """
    Find contiguous regions where a smoothed HPD overlap is below threshold
    for at least min_duration seconds.

    Parameters
    ----------
    hpd_overlap : np.ndarray
        Array of HPD overlap values.
    threshold : float
        Threshold for low overlap.
    min_duration : seconds
        Minimum duration (in seconds) for contiguous samples below threshold.
    smooth_sigma : float
        Standard deviation for Gaussian smoothing (in samples).

    Returns
    -------
    regions : list of (start_idx, end_idx)
        List of (start, end) indices for each sustained low-overlap region.
    """

    smoothed = gaussian_filter1d(hpd_overlap, sigma=smooth_sigma)
    low_mask = smoothed < threshold
    labels, n_labels = label(low_mask)
    regions = []

    # Convert min_duration from seconds to number of samples
    min_duration = int(min_duration * sampling_frequency)
    for label_id in range(1, n_labels + 1):
        inds = np.where(labels == label_id)[0]
        if len(inds) >= min_duration:
            start = inds[0]
            end = inds[-1]
            regions.append((start, end))
    return regions


# Find sustained poor overlap regions for both models
cont_regions = find_sustained_low_overlap(
    cont_hpd_overlap, threshold=0.5, min_duration=0.010, sampling_frequency=500
)
cont_frag_regions = find_sustained_low_overlap(
    cont_frag_hpd_overlap, threshold=0.5, min_duration=0.010, sampling_frequency=500
)

print("Continuous model total sustained poor overlap regions:", len(cont_regions))
print("ContFrag model total sustained poor overlap regions:", len(cont_frag_regions))

# Plot the first N sustained poor overlap regions for each model (if any)
N = 10  # Number of regions to plot

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

    for n, (start, end) in enumerate(regions[:N]):
        # extend the region by 500 samples before and after
        start_ext = max(0, start - 500)
        end_ext = min(len(time), end + 500)
        print(
            f"Plotting {model_label} model sustained poor overlap region"
            f" {n+1}: {start_ext} to {end_ext}"
        )
        plot_single_model_checking(
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
        plot_single_model_checking(
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


smoothed = gaussian_filter1d(cont_hpd_overlap, sigma=1.000)
threshold = 0.2
low_mask = smoothed < threshold
labels, n_labels = label(low_mask)

plt.figure(figsize=(10, 2))
plt.plot(time, cont_hpd_overlap, label="Original", alpha=0.5)
plt.plot(time, smoothed, label="Smoothed", alpha=0.8)

plt.xlim((time[0], time[10000]))


np.unique(labels)


def plot_posterior_consistency_vs_covariate(
    covariate,
    covariate_label=None,
    hpd_overlap=None,
    position_info=None,
    gridsize=100,
    alpha_scatter=0.15,
    scatter_size=2,
    cmap="Blues",
    bins="log",
    mincnt=1,
    figsize=(10, 2),
    suptitle=None,
):
    """
    Plot posterior consistency (HPD overlap) vs a covariate using scatter and hexbin plots.

    Parameters
    ----------
    covariate : str
        Column name in position_info to use as x-axis.
    covariate_label : str or None
        Label for x-axis. If None, uses covariate.
    hpd_overlap : np.ndarray or pd.Series
        Array of HPD overlap values.
    position_info : pd.DataFrame
        DataFrame containing covariate.
    gridsize : int
        Hexbin grid size.
    alpha_scatter : float
        Alpha for scatter plot.
    scatter_size : float
        Marker size for scatter plot.
    cmap : str
        Colormap for hexbin.
    bins : str
        Binning for hexbin.
    mincnt : int
        Minimum count for hexbin.
    figsize : tuple
        Figure size.
    suptitle : str or None
        Figure super-title.
    """
    if covariate_label is None:
        covariate_label = covariate
    if suptitle is None:
        suptitle = f"Posterior Consistency vs {covariate_label}"

    x = position_info[covariate]
    y = hpd_overlap

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize, constrained_layout=True)

    # Scatter plot
    ax1.scatter(
        x, y, s=scatter_size, alpha=alpha_scatter, color="tab:blue", edgecolor="none"
    )
    ax1.set_xlabel(covariate_label, fontsize=11)
    ax1.set_ylabel("HPD Overlap", fontsize=11)
    ax1.set_title("Scatter", fontsize=12)
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)
    ax1.yaxis.grid(True, linestyle=":", alpha=0.5)
    ax1.xaxis.grid(False)

    # Hexbin plot
    hb = ax2.hexbin(
        x, y, gridsize=gridsize, cmap=cmap, bins=bins, mincnt=mincnt, alpha=0.8
    )
    ax2.set_xlabel(covariate_label, fontsize=11)
    ax2.set_ylabel("HPD Overlap", fontsize=11)
    ax2.set_title("Density", fontsize=12)
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)
    ax2.yaxis.grid(True, linestyle=":", alpha=0.5)
    ax2.xaxis.grid(False)
    cb = plt.colorbar(hb, ax=ax2, label="log10(N)")
    cb.ax.tick_params(labelsize=9)

    fig.suptitle(suptitle, fontsize=14, y=1.05)


# Example usage:
plot_posterior_consistency_vs_covariate(
    covariate="head_speed",
    hpd_overlap=cont_hpd_overlap,
    position_info=position_info,
    suptitle="Cont Model Posterior Consistency",
)

plot_posterior_consistency_vs_covariate(
    covariate="head_speed",
    hpd_overlap=cont_frag_hpd_overlap,
    position_info=position_info,
    suptitle="Cont Frag Model Posterior Consistency",
)


plot_posterior_consistency_vs_covariate(
    covariate="linear_position",
    hpd_overlap=cont_hpd_overlap,
    position_info=position_info,
    suptitle="Cont Model Posterior Consistency",
)
plot_posterior_consistency_vs_covariate(
    covariate="linear_position",
    hpd_overlap=cont_frag_hpd_overlap,
    position_info=position_info,
    suptitle="Cont Frag Model Posterior Consistency",
)


def gaussian_smooth(data, sigma, sampling_frequency, axis=0, truncate=8):
    """1D convolution of the data with a Gaussian.

    The standard deviation of the gaussian is in the units of the sampling
    frequency. The function is just a wrapper around scipy's
    `gaussian_filter1d`, The support is truncated at 8 by default, instead
    of 4 in `gaussian_filter1d`

    Parameters
    ----------
    data : array_like
    sigma : float
    sampling_frequency : int
    axis : int, optional
    truncate : int, optional

    Returns
    -------
    smoothed_data : array_like

    """
    return gaussian_filter1d(
        data, sigma * sampling_frequency, truncate=truncate, axis=axis, mode="constant"
    )


def get_multiunit_population_firing_rate(
    multiunit, sampling_frequency, smoothing_sigma=0.015
):
    """Calculates the multiunit population firing rate.

    Parameters
    ----------
    multiunit : ndarray, shape (n_time, n_signals)
        Binary array of multiunit spike times.
    sampling_frequency : float
        Number of samples per second.
    smoothing_sigma : float or np.timedelta
        Amount to smooth the firing rate over time. The default is
        given assuming time is in units of seconds.


    Returns
    -------
    multiunit_population_firing_rate : ndarray, shape (n_time,)

    """
    return gaussian_smooth(
        multiunit.sum(axis=1) * sampling_frequency, smoothing_sigma, sampling_frequency
    )


spikecount = np.stack(
    [
        get_spikecount_per_time_bin(
            spike_times=st,
            time=time,
        )
        for st in spike_times
    ],
    axis=1,
)

multiunit_population_firing_rate = get_multiunit_population_firing_rate(
    multiunit=spikecount,
    sampling_frequency=1 / (time[1] - time[0]),
    smoothing_sigma=0.015,
)


def plot_posterior_consistency_vs_covariate2(
    x,
    covariate_label=None,
    hpd_overlap=None,
    gridsize=100,
    alpha_scatter=0.15,
    scatter_size=2,
    cmap="Blues",
    bins="log",
    mincnt=1,
    figsize=(10, 2),
    suptitle=None,
):
    y = hpd_overlap
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize, constrained_layout=True)

    # Scatter plot
    ax1.scatter(
        x, y, s=scatter_size, alpha=alpha_scatter, color="tab:blue", edgecolor="none"
    )
    ax1.set_xlabel(covariate_label, fontsize=11)
    ax1.set_ylabel("HPD Overlap", fontsize=11)
    ax1.set_title("Scatter", fontsize=12)
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)
    ax1.yaxis.grid(True, linestyle=":", alpha=0.5)
    ax1.xaxis.grid(False)

    # Hexbin plot
    hb = ax2.hexbin(
        x, y, gridsize=gridsize, cmap=cmap, bins=bins, mincnt=mincnt, alpha=0.8
    )
    ax2.set_xlabel(covariate_label, fontsize=11)
    ax2.set_ylabel("HPD Overlap", fontsize=11)
    ax2.set_title("Density", fontsize=12)
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)
    ax2.yaxis.grid(True, linestyle=":", alpha=0.5)
    ax2.xaxis.grid(False)
    cb = plt.colorbar(hb, ax=ax2, label="log10(N)")
    cb.ax.tick_params(labelsize=9)

    fig.suptitle(suptitle, fontsize=14, y=1.05)

    return fig, ax1, ax2


fig, ax1, ax2 = plot_posterior_consistency_vs_covariate2(
    x=multiunit_population_firing_rate,
    covariate_label="MUA",
    hpd_overlap=cont_hpd_overlap,
    suptitle="Cont Model Posterior Consistency vs MUA",
)

fig, ax1, ax2 = plot_posterior_consistency_vs_covariate2(
    x=multiunit_population_firing_rate,
    covariate_label="MUA",
    hpd_overlap=cont_frag_hpd_overlap,
    suptitle="ContFrag Model Posterior Consistency vs MUA",
)


head_speed_interp = (
    pd.Series(position_info["head_speed"].values, index=position_info.index)
    .reindex(time)
    .interpolate()
    .values
)
immobility_mask = head_speed_interp < 4.0

fig, ax1, ax2 = plot_posterior_consistency_vs_covariate2(
    x=multiunit_population_firing_rate[immobility_mask],
    covariate_label="MUA (Immobility Only)",
    hpd_overlap=cont_hpd_overlap[immobility_mask],
    suptitle="Cont Model Posterior Consistency vs MUA (Immobility Only)",
)
fig, ax1, ax2 = plot_posterior_consistency_vs_covariate2(
    x=multiunit_population_firing_rate[immobility_mask],
    covariate_label="MUA (Immobility Only)",
    hpd_overlap=cont_frag_hpd_overlap[immobility_mask],
    suptitle="Cont Frag Model Posterior Consistency vs MUA (Immobility Only)",
)


head_speed_interp = (
    pd.Series(position_info["head_speed"].values, index=position_info.index)
    .reindex(time)
    .interpolate()
    .values
)
mobility_mask = head_speed_interp >= 10.0

plot_posterior_consistency_vs_covariate2(
    x=multiunit_population_firing_rate[mobility_mask],
    covariate_label="MUA (Mobility Only)",
    hpd_overlap=cont_hpd_overlap[mobility_mask],
    suptitle="Cont Model Posterior Consistency vs MUA (Mobility Only)",
)
plot_posterior_consistency_vs_covariate2(
    x=multiunit_population_firing_rate[mobility_mask],
    covariate_label="MUA (Mobility Only)",
    hpd_overlap=cont_frag_hpd_overlap[mobility_mask],
    suptitle="Cont Frag Model Posterior Consistency vs MUA (Mobility Only)",
)


quantiles = np.quantile(time, [0, 0.33, 0.67, 1.0])


n_quantiles = 5
quantiles = np.arange(0, 1 + 1 / n_quantiles, 1 / n_quantiles)
time_ind_quantiles = np.quantile(np.arange(len(time), dtype=int), quantiles).astype(int)

plt.plot(
    np.array(
        [
            cont_hpd_overlap[start_ind:end_ind].mean()
            for start_ind, end_ind in zip(
                time_ind_quantiles[:-1], time_ind_quantiles[1:], strict=False
            )
        ]
    )
)
plt.plot(
    np.array(
        [
            cont_frag_hpd_overlap[start_ind:end_ind].mean()
            for start_ind, end_ind in zip(
                time_ind_quantiles[:-1], time_ind_quantiles[1:], strict=False
            )
        ]
    )
)
plt.legend(["Cont", "ContFrag"])
plt.xticks(np.arange(len(quantiles) - 1), [f"{q:.2f}" for q in quantiles[:-1]])
plt.xlabel("Quantile of Time")
plt.ylabel("Mean HPD Overlap")
plt.title("Mean HPD Overlap by Time Quantile")
plt.grid(axis="y", linestyle=":", alpha=0.5)
plt.show()


plt.hist(cont_frag_hpd_overlap - cont_hpd_overlap, bins=70, density=True)
plt.xlabel("Difference in HPD overlap")
plt.ylabel("Density")
plt.title("Posterior Consistency: ContFrag - Cont")
plt.axvline(0, color="red", linestyle="--", label="Zero Difference")
plt.legend()


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


time_slice_ind = slice(40, 50)
plot_model_checking(
    time_slice_ind,
    time,
    position,
    spike_times,
    cont_model,
    cont_frag_model,
    cont_results,
    cont_frag_results,
    cont_hpd_overlap,
    cont_frag_hpd_overlap,
    overlap_threshold=0.1,
)


spike_counts = np.stack(
    [
        get_spikecount_per_time_bin(st, time=cont_results.time.to_numpy())
        for st in spike_times
    ],
    axis=1,
)


def plot_hpd_overlap_at_time(
    results,
    spike_counts,
    overlap,
    t,
    coverage=0.95,
    figsize=(7, 5),
):
    """
    Plot posterior, likelihood, and their HPD intersection at a given time index.

    Parameters
    ----------
    posterior_at_t : np.ndarray
        Posterior values at time t.
    likelihood_at_t : np.ndarray
        Likelihood values at time t.
    position_bins : np.ndarray
        Position bins corresponding to posterior/likelihood.
    n_total_spikes : int
        Number of spikes at time t.
    t : int
        Time index.
    coverage : float
        HPD coverage (default 0.95).
    figsize : tuple
        Figure size.
    """
    posterior_at_t = results.predictive_posterior.dropna("state_bins").to_numpy()[t]
    likelihood_at_t = np.exp(results.log_likelihood.dropna("state_bins").to_numpy())[t]
    position_bins = results.predictive_posterior.dropna(
        "state_bins"
    ).position.to_numpy()

    posterior_threshold = get_highest_posterior_threshold(
        posterior_at_t[None], coverage=coverage
    )
    likelihood_threshold = get_highest_posterior_threshold(
        likelihood_at_t[None], coverage=coverage
    )
    isin_posterior_hpd = posterior_at_t >= posterior_threshold[:, None]
    isin_likelihood_hpd = likelihood_at_t >= likelihood_threshold[:, None]

    denom = np.min(
        np.stack(
            (isin_posterior_hpd.sum(axis=1), isin_likelihood_hpd.sum(axis=1)), axis=1
        ),
        axis=1,
    )
    # Avoid division by zero
    denom = np.clip(denom, 1, None)

    fig, axes = plt.subplots(
        3, 1, figsize=figsize, sharex=True, constrained_layout=True
    )
    axes[0].plot(position_bins, posterior_at_t, label="Posterior", color="tab:blue")
    axes[0].axhline(
        posterior_threshold,
        color="tab:blue",
        linestyle="--",
        label=f"{int(coverage*100)}% HPD Threshold",
    )
    axes[0].fill_between(
        position_bins,
        posterior_at_t,
        posterior_threshold,
        where=isin_posterior_hpd[0],
        color="tab:blue",
        alpha=0.3,
    )
    axes[0].set_title("Posterior")
    axes[0].legend()

    axes[1].plot(position_bins, likelihood_at_t, label="Likelihood", color="tab:orange")
    axes[1].fill_between(
        position_bins,
        likelihood_at_t,
        likelihood_threshold,
        where=isin_likelihood_hpd[0],
        color="tab:orange",
        alpha=0.3,
    )
    axes[1].axhline(
        likelihood_threshold,
        color="tab:orange",
        linestyle="--",
        label=f"{int(coverage*100)}% HPD Threshold",
    )
    axes[1].set_title("Likelihood")
    axes[1].legend()

    axes[2].plot(
        position_bins,
        isin_posterior_hpd[0] & isin_likelihood_hpd[0],
        label="HPD Intersection",
        color="tab:green",
    )
    axes[2].set_title("HPD Intersection")
    axes[2].set_xlabel("Position")
    axes[2].set_ylabel("Is Overlap")
    axes[2].legend()
    plot_graph_as_1D(track_graph, edge_order, edge_spacing, ax=axes[2])

    n_total_spikes = spike_counts[t].sum()
    n_hpd_likelihood = isin_likelihood_hpd[0].sum()
    n_hpd_posterior = isin_posterior_hpd[0].sum()
    denom = int(denom[0])
    n_overlap = np.sum(isin_posterior_hpd[0] & isin_likelihood_hpd[0])
    plt.suptitle(
        f"Index {t}, overlap = {overlap[t]:.3f}, n_spikes = {n_total_spikes},\n"
        f"n_hpd_likelihood = {n_hpd_likelihood}, n_hpd_posterior = {n_hpd_posterior}, \n"
        f"n_overlap = {n_overlap}, denom = {denom}",
        fontsize=12,
        y=1.12,
    )

    return fig, axes


plot_hpd_overlap_at_time(
    cont_results,
    spike_counts,
    cont_hpd_overlap,
    t=41,
)


plot_hpd_overlap_at_time(
    cont_results,
    spike_counts,
    cont_hpd_overlap,
    t=42,
)


plot_hpd_overlap_at_time(
    cont_results,
    spike_counts,
    cont_hpd_overlap,
    t=44,
)


plot_hpd_overlap_at_time(
    cont_results,
    spike_counts,
    cont_hpd_overlap,
    t=709133,
)


plot_hpd_overlap_at_time(
    cont_results,
    spike_counts,
    cont_hpd_overlap,
    t=195314,
)


time_slice_ind = slice(100_000, 140_000)
plot_model_checking(
    time_slice_ind,
    time,
    position,
    spike_times,
    cont_model,
    cont_frag_model,
    cont_results,
    cont_frag_results,
    cont_hpd_overlap,
    cont_frag_hpd_overlap,
    overlap_threshold=0.1,
)


time_slice_ind = slice(227_000, 232_000)
plot_model_checking(
    time_slice_ind,
    time,
    position,
    spike_times,
    cont_model,
    cont_frag_model,
    cont_results,
    cont_frag_results,
    cont_hpd_overlap,
    cont_frag_hpd_overlap,
    overlap_threshold=0.1,
)


time_slice_ind = slice(300_000, 350_000)
plot_model_checking(
    time_slice_ind,
    time,
    position,
    spike_times,
    cont_model,
    cont_frag_model,
    cont_results,
    cont_frag_results,
    cont_hpd_overlap,
    cont_frag_hpd_overlap,
    overlap_threshold=0.1,
)


time_slice_ind = slice(330_000, 340_000)
plot_model_checking(
    time_slice_ind,
    time,
    position,
    spike_times,
    cont_model,
    cont_frag_model,
    cont_results,
    cont_frag_results,
    cont_hpd_overlap,
    cont_frag_hpd_overlap,
    overlap_threshold=0.1,
)


plt.plot(
    cont_model.environments[0].place_bin_centers_,
    cont_model.encoding_model_[("", 0)]["place_fields"].T,
)
plt.xlabel("Position")


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
    actual_projected_position=position_info[
        ["projected_x_position", "projected_y_position"]
    ],
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
ax.set_title(
    "Posterior Consistency vs Ahead/Behind Distance (Cont)", fontsize=13, pad=8
)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.yaxis.grid(True, linestyle=":", alpha=0.4)
ax.xaxis.grid(False)
ax.set_xlim((-250, 250))
ax.axvline(0, color="grey", linestyle="--", label="Zero Distance")
plt.tight_layout()
plt.show()


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
    actual_projected_position=position_info[
        ["projected_x_position", "projected_y_position"]
    ],
    track_segment_id=position_info["track_segment_id"],
    actual_orientation=position_info["head_orientation"],
)

con_frag_dist = get_ahead_behind_distance(
    track_graph=track_graph,
    actual_projected_position=actual_projected_position,
    actual_edges=actual_edges,
    actual_orientation=actual_orientation,
    mental_position_2d=mental_position_2d,
    mental_position_edges=mental_position_edges,
)


fig, ax = plt.subplots(figsize=(6, 3))
ax.scatter(
    con_frag_dist,
    cont_frag_hpd_overlap,
    s=2,
    alpha=0.15,
    color="tab:orange",
    edgecolor="none",
)
ax.set_xlabel("Ahead/Behind Distance", fontsize=11)
ax.set_ylabel("HPD Overlap", fontsize=11)
ax.set_title(
    "Posterior Consistency vs Ahead/Behind Distance (ContFrag)", fontsize=13, pad=8
)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.yaxis.grid(True, linestyle=":", alpha=0.4)
ax.xaxis.grid(False)
ax.set_xlim((-250, 250))
ax.axvline(0, color="grey", linestyle="--", label="Zero Distance")
plt.tight_layout()
plt.show()


# Boolean mask where Fragmented state probability > 0.8
frag_mask = (
    cont_frag_results.acausal_state_probabilities.sel(states="Fragmented") > 0.8
).values

# Label contiguous regions
labels, n_labels = label(frag_mask)

# Find regions longer than 10 ms (5 samples at 500 Hz)
min_samples = 5
sustained_fragments = [
    (np.where(labels == i)[0][0], np.where(labels == i)[0][-1])
    for i in range(1, n_labels + 1)
    if np.sum(labels == i) >= min_samples
]

print("Sustained fragmented state regions (>10 ms):", sustained_fragments)
# Plot each sustained fragmented region with 100 ms (50 samples at 500 Hz) extension on both sides
for i, (start, end) in enumerate(sustained_fragments):
    start_ext = max(0, start - 50)
    end_ext = min(len(time), end + 50)
    print(f"Plotting sustained fragmented region {i+1}: {start_ext} to {end_ext}")
    plot_model_checking(
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


# Boolean mask where Fragmented state probability > 0.8
frag_mask = (
    cont_frag_results.acausal_state_probabilities.sel(states="Fragmented") > 0.8
).values

# Label contiguous regions
labels, n_labels = label(frag_mask)

# Find regions longer than 10 ms (5 samples at 500 Hz)
min_samples = 5
sustained_fragments = [
    (np.where(labels == i)[0][0], np.where(labels == i)[0][-1])
    for i in range(1, n_labels + 1)
    if np.sum(labels == i) >= min_samples
]

print("Sustained fragmented state regions (>10 ms):", sustained_fragments)
