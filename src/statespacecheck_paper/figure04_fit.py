"""Figure-4 fit and decode: the computation behind the decode cache.

:func:`fit_and_decode` fits the Continuous and Continuous-Fragmented decoders on
a :class:`~statespacecheck_paper.figure04_input.NeuralRecordingData`, decodes
the same recording, and returns the decode payload (decoder outputs, spike
counts, and place fields) that :mod:`figure04_cache` stores. :func:`decode_time`
is the decode time grid it fits and decodes on.

This module, and every module it depends on, is hashed into the decode-cache
fingerprint (``figure04_cache._DECODE_SOURCE_FILES``), so any executable change
here refits both models. Orchestration, result containers, and progress
messages therefore live in :mod:`figure04_workflow`, outside that hash.
"""

from __future__ import annotations

import warnings

import numpy as np
from numpy.typing import NDArray

from statespacecheck_paper.figure04_decoder import (
    Figure4DecoderConfig,
    Figure4ExecutionConfig,
    Figure4PackageDefaults,
    create_decoder_environment,
    fit_decoder_models,
    get_spike_counts,
    validate_package_defaults,
)
from statespacecheck_paper.figure04_input import NeuralRecordingData
from statespacecheck_paper.figure04_place_fields import (
    DECODER_PREDICTIVE_VAR,
    extract_agreed_place_fields,
    extract_place_fields,
)


def decode_time(recording: NeuralRecordingData) -> NDArray[np.float64]:
    """Return the decode time grid: the recording's position timestamps (s).

    Parameters
    ----------
    recording : NeuralRecordingData

    Returns
    -------
    time : np.ndarray, shape (n_time,)
    """
    return np.asarray(recording.position_info.index.to_numpy(), dtype=np.float64)


def fit_and_decode(
    recording: NeuralRecordingData,
    *,
    decoder_config: Figure4DecoderConfig,
    execution_config: Figure4ExecutionConfig,
    package_defaults: Figure4PackageDefaults,
) -> dict[str, object]:
    """Fit both decoders on the full recording, decode it, and return the decode payload.

    Both models are fitted with every head-position sample and every spike in
    the recording (no training mask), then decode that same recording on
    :func:`decode_time`. Before decoding, :func:`validate_package_defaults`
    checks the installed ``non_local_detector`` version and the built models
    against ``package_defaults`` and raises on any drift.

    Parameters
    ----------
    recording : NeuralRecordingData
    decoder_config : Figure4DecoderConfig
        Injected decoder parameters.
    execution_config : Figure4ExecutionConfig
        Performance-only parameters.
    package_defaults : Figure4PackageDefaults
        Recorded ``non_local_detector`` defaults to check the models against.

    Returns
    -------
    dict
        Exactly the decode-cache keys: ``continuous_results`` and
        ``continuous_fragmented_results`` (``xr.Dataset`` with the predictive
        distribution, the log-likelihood, and the smoothed posterior),
        ``spike_counts`` ``(n_time, n_cells)``, ``place_field_peaks``
        ``(n_cells,)``, ``diagnostic_place_fields`` ``(n_cells, n_position_bins)``,
        and ``diagnostic_position_bins`` ``(n_position_bins,)``.
    """
    time = decode_time(recording)
    head_position = recording.position_info[["head_position_x", "head_position_y"]].to_numpy(
        dtype=np.float64
    )
    spike_times_list = list(recording.spike_times)  # non_local_detector wants a list

    # Environment is only needed to fit the decoders.
    env = create_decoder_environment(
        track_graph=recording.track_graph,
        edge_order=list(recording.linear_edge_order),
        edge_spacing=recording.linear_edge_spacing,
        place_bin_size=decoder_config.position_bin_size_cm,
    )

    continuous_model, continuous_fragmented_model = fit_decoder_models(
        position=head_position,
        spike_times=spike_times_list,
        time=time,
        environment=env,
        decoder_config=decoder_config,
        execution_config=execution_config,
    )

    # Runtime guard: the recorded non_local_detector package defaults shape
    # the decode but are not injected, so a dependency bump could silently change
    # them. Fail loudly here rather than produce a different published figure.
    validate_package_defaults(continuous_model, continuous_fragmented_model, package_defaults)

    # non_local_detector always returns the smoothed (acausal) posterior. The
    # diagnostics, figure, and site export read the prediction; the viewer cache
    # also reads the log-likelihood and the smoothed posterior. Nothing reads
    # the causal "filter" output, so it is not requested.
    decode_outputs = [DECODER_PREDICTIVE_VAR, "log_likelihood"]
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

    # Extract place fields for raster sorting (use continuous model).
    place_fields, position_bins = extract_place_fields(continuous_model)
    if np.any(np.all(np.isnan(place_fields), axis=1)):
        warnings.warn(
            "Some cells have all-NaN place fields; peak positions may be incorrect",
            stacklevel=2,
        )
    place_field_peaks = position_bins[np.nanargmax(place_fields, axis=1)]

    # Shared interior place fields for the mean per-spike likelihood row.
    diagnostic_place_fields, diagnostic_position_bins = extract_agreed_place_fields(
        continuous_model, continuous_fragmented_model
    )

    return {
        "continuous_results": continuous_results,
        "continuous_fragmented_results": continuous_fragmented_results,
        "spike_counts": spike_counts,
        "place_field_peaks": place_field_peaks,
        "diagnostic_place_fields": diagnostic_place_fields,
        "diagnostic_position_bins": diagnostic_position_bins,
    }
