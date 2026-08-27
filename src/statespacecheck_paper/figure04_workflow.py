"""Figure-4 workflow: load derived data, fit/decode (or load cache), summarize.

Assembles everything the Figure-4 render needs. Inputs are pre-exported derived
data (position/track + spike times), not a raw-data pipeline, so this is a
*workflow*: load the fresh track data, load a fingerprint-matching decode cache
or fit + decode both models and cache the result, compute the per-spike
diagnostics, and calculate/print the manuscript summary scalars. In-memory names
are spelled out (``continuous_fragmented_*``) even though the on-disk cache keys
stay ``contfrag_*``.
"""

from __future__ import annotations

import dataclasses
import warnings
from typing import Any, Literal, cast

import numpy as np
from numpy.typing import NDArray

from statespacecheck_paper.diagnostics import SpikeEventDiagnostics
from statespacecheck_paper.figure04_cache import (
    Figure4Paths,
    compute_figure04_cache_fingerprint,
    load_figure04_cache,
    save_figure04_cache,
)
from statespacecheck_paper.load_local_data import load_neural_recording_from_files
from statespacecheck_paper.real_data_analysis import (
    Figure4Config,
    compute_flag_confusion,
    compute_model_diagnostics,
    create_decoder_environment,
    extract_place_fields,
    extract_shared_position_place_fields,
    fit_decoder_models,
    get_spike_counts,
)


@dataclasses.dataclass(frozen=True)
class Figure4RenderData:
    """Everything the Figure-4 render needs: fresh track data + decode payload.

    The decode payload (results / diagnostics / spike counts / place fields) is
    the expensive, cacheable content. The position/track data is always loaded
    fresh from :class:`Figure4Paths` (it is cheap and never cached), then
    combined with the decode payload here so the render reads a single object.
    In-memory fields spell out ``continuous_fragmented_*``; the on-disk cache
    keys remain ``contfrag_*``.
    """

    # Position / track data (always loaded fresh; not cached)
    position_info: Any
    time: NDArray[np.float64]
    position: NDArray[np.float64]
    linear_position: NDArray[np.float64]
    spike_times_list: list[Any]
    track_graph: Any
    edge_order: Any
    edge_spacing: Any
    # Decode payload (cached or recomputed)
    continuous_results: Any
    continuous_fragmented_results: Any
    continuous_diagnostics: SpikeEventDiagnostics
    continuous_fragmented_diagnostics: SpikeEventDiagnostics
    spike_counts: NDArray[np.int64]
    place_field_peaks: NDArray[np.float64]
    diagnostic_place_fields: NDArray[np.float64]
    diagnostic_position_bins: NDArray[np.float64]


def compute_mean_spike_event_diagnostic(diagnostics: SpikeEventDiagnostics, metric: str) -> float:
    """Return the per-spike mean for a diagnostic metric."""
    event_key = f"event_{metric}"
    if not hasattr(diagnostics, event_key):
        raise KeyError(f"Missing per-spike diagnostic array: {event_key}")
    return float(np.nanmean(getattr(diagnostics, event_key)))


def _compute_figure04_decode_results(
    *,
    position: NDArray[np.float64],
    spike_times_list: list[Any],
    time: NDArray[np.float64],
    track_graph: Any,
    edge_order: Any,
    edge_spacing: Any,
) -> dict[str, Any]:
    """Fit both decoders, decode, and compute the cacheable decode payload.

    Returns exactly the keys stored in (and loaded from) the Figure-4 cache.
    """
    # Environment is only needed to fit the decoders.
    env = create_decoder_environment(
        track_graph=track_graph,
        edge_order=edge_order,
        edge_spacing=edge_spacing,
    )

    print("Fitting models...")
    continuous_model, continuous_fragmented_model = fit_decoder_models(
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
    continuous_fragmented_results = continuous_fragmented_model.predict(
        spike_times=spike_times_list,
        time=time,
        return_outputs=decode_outputs,
    )

    spike_counts = get_spike_counts(spike_times_list, time)

    print("Computing diagnostics...")
    continuous_diagnostics = compute_model_diagnostics(
        continuous_model, continuous_results, spike_counts, time, spike_times=spike_times_list
    )
    continuous_fragmented_diagnostics = compute_model_diagnostics(
        continuous_fragmented_model,
        continuous_fragmented_results,
        spike_counts,
        time,
        spike_times=spike_times_list,
    )

    # Extract place fields for raster sorting (use continuous model).
    place_fields, position_bins = extract_place_fields(continuous_model)
    if np.any(np.all(np.isnan(place_fields), axis=1)):
        warnings.warn(
            "Some cells have all-NaN place fields; peak positions may be incorrect",
            stacklevel=2,
        )
    place_field_peaks = position_bins[np.nanargmax(place_fields, axis=1)]

    # Shared interior place fields for the mean per-spike likelihood row.
    # The row is meant to be identical across decoders, so verify the two
    # models agree on both fields and grid before storing a single copy.
    diagnostic_place_fields, diagnostic_position_bins = extract_shared_position_place_fields(
        continuous_model
    )
    continuous_fragmented_place_fields, continuous_fragmented_position_bins = (
        extract_shared_position_place_fields(continuous_fragmented_model)
    )
    if not np.allclose(
        diagnostic_place_fields, continuous_fragmented_place_fields, equal_nan=True
    ) or not np.allclose(
        diagnostic_position_bins, continuous_fragmented_position_bins, equal_nan=True
    ):
        raise ValueError(
            "Continuous and Continuous--Fragmented place fields or position "
            "grids differ; the shared likelihood row would misrepresent one "
            "of the decoders."
        )

    return {
        "continuous_results": continuous_results,
        "contfrag_results": continuous_fragmented_results,
        "continuous_diagnostics": continuous_diagnostics,
        "contfrag_diagnostics": continuous_fragmented_diagnostics,
        "spike_counts": spike_counts,
        "place_field_peaks": place_field_peaks,
        "diagnostic_place_fields": diagnostic_place_fields,
        "diagnostic_position_bins": diagnostic_position_bins,
    }


def prepare_figure04_render_data(
    config: Figure4Config,
    paths: Figure4Paths,
    *,
    use_cache: bool = True,
) -> Figure4RenderData:
    """Assemble the Figure-4 render data: fresh track data + cached/computed decode.

    Reads only the injected ``config`` and ``paths`` (never the module-global
    ``DATA_PATH`` / ``ANIMAL_DATE_EPOCH``), so it is exercisable with synthetic
    inputs and a temporary cache directory. The cache is keyed on
    :func:`compute_figure04_cache_fingerprint`; a config / data / dependency
    change forces a recompute. The position/track data is always loaded fresh
    (it is cheap and never cached).

    Parameters
    ----------
    config : Figure4Config
        Decoder configuration; hashed into the cache fingerprint.
    paths : Figure4Paths
        Injected data-location identifiers.
    use_cache : bool, default True
        When True and a fingerprint-matching cache exists, load it instead of
        recomputing. When False, always recompute and overwrite the cache.
    """
    print("Loading data...")
    data = load_neural_recording_from_files(paths.data_path, paths.animal_date_epoch)
    print(f"  Loaded {len(data['spike_times'])} cells")

    position_info = data["position_info"]
    track_data: dict[str, Any] = dict(
        position_info=position_info,
        time=position_info.index.values,
        position=position_info[["head_position_x", "head_position_y"]].values,
        linear_position=position_info["linear_position"].values,
        spike_times_list=list(data["spike_times"]),
        track_graph=data["track_graph"],
        edge_order=data["linear_edge_order"],
        edge_spacing=data["linear_edge_spacing"],
    )

    expected_fingerprint = compute_figure04_cache_fingerprint(config, paths)
    payload: dict[str, object] | None = None
    if use_cache:
        print("Loading cached decoder outputs (use --force-recompute to rebuild)...")
        payload = load_figure04_cache(paths.cache_path, expected_fingerprint)
        if payload is None:
            print(
                "  No matching cache (absent, unreadable, or fingerprint mismatch "
                "from a config / data / non_local_detector change); recomputing."
            )

    if payload is None:
        payload = _compute_figure04_decode_results(
            position=track_data["position"],
            spike_times_list=track_data["spike_times_list"],
            time=track_data["time"],
            track_graph=track_data["track_graph"],
            edge_order=track_data["edge_order"],
            edge_spacing=track_data["edge_spacing"],
        )
        print("Caching decoder outputs to data/intermediates ...")
        save_figure04_cache(paths.cache_path, expected_fingerprint, payload)

    # Map the preserved serialized keys explicitly onto the fully spelled fields.
    # ``payload`` values cross the deliberately-untyped cache boundary; cast to
    # Any so the typed dataclass fields accept them (Phase 5 adds typed I/O).
    typed_payload = cast("dict[str, Any]", payload)
    return Figure4RenderData(
        **track_data,
        continuous_results=typed_payload["continuous_results"],
        continuous_fragmented_results=typed_payload["contfrag_results"],
        continuous_diagnostics=typed_payload["continuous_diagnostics"],
        continuous_fragmented_diagnostics=typed_payload["contfrag_diagnostics"],
        spike_counts=typed_payload["spike_counts"],
        place_field_peaks=typed_payload["place_field_peaks"],
        diagnostic_place_fields=typed_payload["diagnostic_place_fields"],
        diagnostic_position_bins=typed_payload["diagnostic_position_bins"],
    )


def print_figure04_summary(
    render_data: Figure4RenderData,
    thresholds: dict[str, float],
    metric_directions: dict[str, Literal["below", "above"]],
) -> None:
    """Print the whole-session diagnostic event-means and flag-agreement counts.

    These scalars can appear in the manuscript text, so their computation must
    stay identical to the cached decode; the render layer never alters them.
    """
    print("\n=== Diagnostic Summary (all time points) ===")
    for name, diag in [
        ("Continuous", render_data.continuous_diagnostics),
        ("ContFrag", render_data.continuous_fragmented_diagnostics),
    ]:
        print(f"\n{name}:")
        for metric in ["hpd_overlap", "kl_divergence", "predictive_pvalue"]:
            print(f"  {metric}: {compute_mean_spike_event_diagnostic(diag, metric):.4f}")

    # Per-spike flag agreement between the two decoders at these thresholds.
    # "Cont-only" is the rescue quadrant (flagged by Continuous but not by
    # Continuous-Fragmented); "rescue" is its fraction of all Continuous flags.
    print("\n=== Flag agreement: Continuous (A) vs Cont-Frag (B) ===")
    for metric, worse_when in metric_directions.items():
        conf = compute_flag_confusion(
            render_data.continuous_diagnostics,
            render_data.continuous_fragmented_diagnostics,
            metric,
            thresholds[metric],
            worse_when=worse_when,
        )
        print(
            f"  {metric}: n={conf.n:,} both={conf.both:,} cont-only={conf.a_only:,} "
            f"cf-only={conf.b_only:,} neither={conf.neither:,} "
            f"rescue={100 * conf.rescue_rate:.1f}%"
        )
