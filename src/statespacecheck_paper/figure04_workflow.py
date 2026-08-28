"""Figure-4 workflow: load derived data, fit/decode (or load cache), summarize.

Assembles everything the Figure-4 render needs. Inputs are pre-exported derived
data (a :class:`~statespacecheck_paper.load_local_data.NeuralRecordingData`),
not a raw-data pipeline, so this is a *workflow*: load the fresh recording, load
a fingerprint-matching decode cache or fit + decode both models and cache the
result, compute the per-spike diagnostics, and calculate the manuscript
summary scalars.

The in-memory decode results are a typed :class:`Figure4DecodeResults` whose
fields spell out ``continuous_fragmented_*``; the on-disk cache keys stay
``contfrag_*`` and the mapping is confined to
:meth:`Figure4DecodeResults.from_cache_payload` / ``to_cache_payload``.
"""

from __future__ import annotations

import dataclasses
import warnings
from collections.abc import Mapping
from typing import Literal

import numpy as np
import xarray as xr
from numpy.typing import NDArray

from statespacecheck_paper.diagnostics import SpikeEventDiagnostics
from statespacecheck_paper.figure04_cache import (
    _FIGURE04_CACHE_PAYLOAD_KEYS,
    Figure4Paths,
    compute_figure04_cache_fingerprint,
    load_figure04_cache,
    save_figure04_cache,
)
from statespacecheck_paper.figure04_decoder import (
    Figure4Config,
    Figure4DecoderConfig,
    Figure4ExecutionConfig,
    Figure4Provenance,
    create_decoder_environment,
    fit_decoder_models,
    get_spike_counts,
    validate_provenance_defaults,
)
from statespacecheck_paper.figure04_diagnostics import (
    FlagConfusion,
    compute_flag_confusion,
    compute_model_diagnostics,
)
from statespacecheck_paper.figure04_place_fields import (
    extract_place_fields,
    extract_shared_position_place_fields,
)
from statespacecheck_paper.load_local_data import (
    NeuralRecordingData,
    load_neural_recording_from_files,
)


@dataclasses.dataclass(frozen=True)
class Figure4DecodeResults:
    """The expensive, cacheable Figure-4 decode outputs, typed and validated.

    The contained xarray datasets and diagnostic objects are treated as
    read-only by convention (the frozen wrapper does not deep-freeze them); the
    four arrays are copied to their dtypes and marked read-only at construction.
    In-memory field names spell out ``continuous_fragmented_*``; the serialized
    cache keys remain ``contfrag_*`` (see :meth:`from_cache_payload`).

    Parameters
    ----------
    continuous_results, continuous_fragmented_results : xr.Dataset
        Decoder outputs (filter / predictive / log-likelihood) per model.
    continuous_diagnostics, continuous_fragmented_diagnostics : SpikeEventDiagnostics
        Per-spike diagnostics per model.
    spike_counts : np.ndarray, shape (n_time, n_cells)
    place_field_peaks : np.ndarray, shape (n_cells,)
    diagnostic_place_fields : np.ndarray, shape (n_cells, n_position_bins)
    diagnostic_position_bins : np.ndarray, shape (n_position_bins,)
    """

    continuous_results: xr.Dataset
    continuous_fragmented_results: xr.Dataset
    continuous_diagnostics: SpikeEventDiagnostics
    continuous_fragmented_diagnostics: SpikeEventDiagnostics
    spike_counts: NDArray[np.int64]
    place_field_peaks: NDArray[np.float64]
    diagnostic_place_fields: NDArray[np.float64]
    diagnostic_position_bins: NDArray[np.float64]

    def __post_init__(self) -> None:
        # Unconditional copies (``np.array``, not ``np.asarray``): otherwise the
        # subsequent ``setflags(write=False)`` would freeze a caller-owned array.
        spike_counts = np.array(self.spike_counts, dtype=np.int64)
        place_field_peaks = np.array(self.place_field_peaks, dtype=np.float64)
        diagnostic_place_fields = np.array(self.diagnostic_place_fields, dtype=np.float64)
        diagnostic_position_bins = np.array(self.diagnostic_position_bins, dtype=np.float64)

        if spike_counts.ndim != 2:
            raise ValueError(f"spike_counts must be (n_time, n_cells); got {spike_counts.shape}")
        n_time, n_cells = spike_counts.shape
        if place_field_peaks.shape != (n_cells,):
            raise ValueError(
                f"place_field_peaks must be ({n_cells},); got {place_field_peaks.shape}"
            )
        if diagnostic_place_fields.ndim != 2 or diagnostic_place_fields.shape[0] != n_cells:
            raise ValueError(
                f"diagnostic_place_fields must be ({n_cells}, n_position_bins); "
                f"got {diagnostic_place_fields.shape}"
            )
        n_bins = diagnostic_place_fields.shape[1]
        if diagnostic_position_bins.shape != (n_bins,):
            raise ValueError(
                f"diagnostic_position_bins must be ({n_bins},); "
                f"got {diagnostic_position_bins.shape}"
            )

        # Tie the decode timelines together: both decoder result datasets and
        # both per-spike diagnostics must live on the same ``n_time`` grid as
        # ``spike_counts``, so an inconsistent bundle is rejected at construction
        # rather than silently misaligning Figure 4. Require the actual ``time``
        # coordinate (not just a matching length) -- ``compose_figure04`` reads
        # it, and comparing the two decoders' coordinates catches a bundle that
        # pairs decodes from different windows even when the lengths agree.
        result_time_coords: dict[str, NDArray[np.float64]] = {}
        for name, dataset in (
            ("continuous_results", self.continuous_results),
            ("continuous_fragmented_results", self.continuous_fragmented_results),
        ):
            if "time" not in dataset.coords:
                raise ValueError(
                    f"{name} must carry a 'time' coordinate; the decode timeline "
                    "cannot be verified without it."
                )
            dataset_time = np.asarray(dataset.coords["time"].values, dtype=np.float64)
            if dataset_time.shape != (n_time,):
                raise ValueError(
                    f"{name} has {dataset_time.shape[0]} time samples but spike_counts "
                    f"has {n_time}; the decode timelines must match."
                )
            result_time_coords[name] = dataset_time
        if not np.array_equal(
            result_time_coords["continuous_results"],
            result_time_coords["continuous_fragmented_results"],
        ):
            raise ValueError(
                "continuous_results and continuous_fragmented_results have different "
                "'time' coordinates; the two decoders must share one timeline."
            )
        for name, diagnostics in (
            ("continuous_diagnostics", self.continuous_diagnostics),
            ("continuous_fragmented_diagnostics", self.continuous_fragmented_diagnostics),
        ):
            time_ind = diagnostics.event_time_ind
            cell_ind = diagnostics.event_cell_ind
            if time_ind.size and (time_ind.min() < 0 or time_ind.max() >= n_time):
                raise ValueError(
                    f"{name}.event_time_ind falls outside [0, {n_time}); the "
                    "diagnostics do not match the spike_counts timeline."
                )
            if cell_ind.size and (cell_ind.min() < 0 or cell_ind.max() >= n_cells):
                raise ValueError(
                    f"{name}.event_cell_ind falls outside [0, {n_cells}); the "
                    "diagnostics do not match the spike_counts cell count."
                )

        for name, arr in (
            ("spike_counts", spike_counts),
            ("place_field_peaks", place_field_peaks),
            ("diagnostic_place_fields", diagnostic_place_fields),
            ("diagnostic_position_bins", diagnostic_position_bins),
        ):
            arr.setflags(write=False)
            object.__setattr__(self, name, arr)

    @classmethod
    def from_cache_payload(cls, payload: Mapping[str, object]) -> Figure4DecodeResults:
        """Build from the serialized cache payload (the ``contfrag_*`` keys)."""
        missing = [key for key in _FIGURE04_CACHE_PAYLOAD_KEYS if key not in payload]
        if missing:
            raise ValueError(f"decode payload missing keys: {missing}")
        return cls(
            continuous_results=_cast_dataset(payload["continuous_results"]),
            continuous_fragmented_results=_cast_dataset(payload["contfrag_results"]),
            continuous_diagnostics=_cast_diagnostics(payload["continuous_diagnostics"]),
            continuous_fragmented_diagnostics=_cast_diagnostics(payload["contfrag_diagnostics"]),
            spike_counts=np.asarray(payload["spike_counts"], dtype=np.int64),
            place_field_peaks=np.asarray(payload["place_field_peaks"], dtype=np.float64),
            diagnostic_place_fields=np.asarray(
                payload["diagnostic_place_fields"], dtype=np.float64
            ),
            diagnostic_position_bins=np.asarray(
                payload["diagnostic_position_bins"], dtype=np.float64
            ),
        )

    def to_cache_payload(self) -> dict[str, object]:
        """Return the serialized cache payload (mapping fields to ``contfrag_*``)."""
        return {
            "continuous_results": self.continuous_results,
            "contfrag_results": self.continuous_fragmented_results,
            "continuous_diagnostics": self.continuous_diagnostics,
            "contfrag_diagnostics": self.continuous_fragmented_diagnostics,
            "spike_counts": self.spike_counts,
            "place_field_peaks": self.place_field_peaks,
            "diagnostic_place_fields": self.diagnostic_place_fields,
            "diagnostic_position_bins": self.diagnostic_position_bins,
        }


def _cast_dataset(value: object) -> xr.Dataset:
    """Narrow a cache-payload value to ``xr.Dataset`` (fails clearly otherwise)."""
    if not isinstance(value, xr.Dataset):
        raise TypeError(f"expected xr.Dataset decode result; got {type(value).__name__}")
    return value


def _cast_diagnostics(value: object) -> SpikeEventDiagnostics:
    """Narrow a cache-payload value to ``SpikeEventDiagnostics`` (fails otherwise)."""
    if not isinstance(value, SpikeEventDiagnostics):
        raise TypeError(f"expected SpikeEventDiagnostics; got {type(value).__name__}")
    return value


@dataclasses.dataclass(frozen=True)
class Figure4RenderData:
    """Everything the Figure-4 render needs: the recording + typed decode results.

    The position/track data is always loaded fresh (:class:`NeuralRecordingData`,
    cheap, never cached); the decode results are the expensive cacheable content
    (:class:`Figure4DecodeResults`). The three per-time arrays derived from the
    recording are copied and marked read-only at construction, must share a
    single 1-D ``n_time``, and that ``n_time`` must match the decode timeline
    (``decode_results.spike_counts.shape[0]``) so a cached decode cannot pair
    with a differently sized fresh recording.
    """

    recording: NeuralRecordingData
    time: NDArray[np.float64]
    head_position: NDArray[np.float64]
    linear_position: NDArray[np.float64]
    decode_results: Figure4DecodeResults

    def __post_init__(self) -> None:
        # Unconditional copies (see Figure4DecodeResults) so freezing cannot
        # reach back into a caller-owned array.
        time = np.array(self.time, dtype=np.float64)
        head_position = np.array(self.head_position, dtype=np.float64)
        linear_position = np.array(self.linear_position, dtype=np.float64)
        if time.ndim != 1:
            raise ValueError(f"time must be 1-D (n_time,); got shape {time.shape}")
        n_time = time.shape[0]
        if head_position.shape != (n_time, 2):
            raise ValueError(f"head_position must be ({n_time}, 2); got {head_position.shape}")
        if linear_position.shape != (n_time,):
            raise ValueError(f"linear_position must be ({n_time},); got {linear_position.shape}")
        # The decode results must have been produced on this same recording
        # timeline; a cached decode of a different-length session would otherwise
        # silently pair with the freshly loaded position data.
        decode_n_time = self.decode_results.spike_counts.shape[0]
        if decode_n_time != n_time:
            raise ValueError(
                f"decode_results timeline ({decode_n_time}) does not match the "
                f"recording timeline ({n_time}); they must be the same session."
            )
        for name, arr in (
            ("time", time),
            ("head_position", head_position),
            ("linear_position", linear_position),
        ):
            arr.setflags(write=False)
            object.__setattr__(self, name, arr)


def compute_mean_spike_event_diagnostic(diagnostics: SpikeEventDiagnostics, metric: str) -> float:
    """Return the per-spike mean for a diagnostic metric."""
    event_key = f"event_{metric}"
    if not hasattr(diagnostics, event_key):
        raise KeyError(f"Missing per-spike diagnostic array: {event_key}")
    values = np.asarray(getattr(diagnostics, event_key), dtype=np.float64)
    if values.size == 0:
        raise ValueError(f"Cannot compute {event_key} mean: no spike events are present")
    if np.any(np.isnan(values)) or np.any(np.isneginf(values)):
        raise ValueError(f"Cannot compute {event_key} mean: undefined event value present")
    return float(np.mean(values))


@dataclasses.dataclass(frozen=True)
class Figure4DiagnosticMeans:
    """Whole-session mean of each per-spike diagnostic for one decoder."""

    hpd_overlap: float
    kl_divergence: float
    predictive_pvalue: float


@dataclasses.dataclass(frozen=True)
class Figure4Summary:
    """Manuscript-facing Figure-4 scalars, independent of output formatting."""

    continuous: Figure4DiagnosticMeans
    continuous_fragmented: Figure4DiagnosticMeans
    flag_confusions: tuple[FlagConfusion, ...]


def _compute_diagnostic_means(
    diagnostics: SpikeEventDiagnostics,
) -> Figure4DiagnosticMeans:
    """Compute the three manuscript diagnostic means for one decoder."""
    return Figure4DiagnosticMeans(
        hpd_overlap=compute_mean_spike_event_diagnostic(diagnostics, "hpd_overlap"),
        kl_divergence=compute_mean_spike_event_diagnostic(diagnostics, "kl_divergence"),
        predictive_pvalue=compute_mean_spike_event_diagnostic(diagnostics, "predictive_pvalue"),
    )


def _compute_figure04_decode_results(
    recording: NeuralRecordingData,
    *,
    time: NDArray[np.float64],
    head_position: NDArray[np.float64],
    decoder_config: Figure4DecoderConfig,
    execution_config: Figure4ExecutionConfig,
    provenance: Figure4Provenance,
) -> Figure4DecodeResults:
    """Fit both decoders, decode, and compute the cacheable decode results."""
    spike_times_list = list(recording.spike_times)  # non_local_detector wants a list

    # Environment is only needed to fit the decoders.
    env = create_decoder_environment(
        track_graph=recording.track_graph,
        edge_order=list(recording.linear_edge_order),
        edge_spacing=recording.linear_edge_spacing,
        place_bin_size=decoder_config.position_bin_size_cm,
    )

    print("Fitting models...")
    continuous_model, continuous_fragmented_model = fit_decoder_models(
        position=head_position,
        spike_times=spike_times_list,
        time=time,
        environment=env,
        decoder_config=decoder_config,
        execution_config=execution_config,
    )

    # Runtime guard: the non_local_detector defaults recorded as provenance shape
    # the decode but are not injected, so a dependency bump could silently change
    # them. Fail loudly here rather than produce a different published figure.
    validate_provenance_defaults(continuous_model, continuous_fragmented_model, provenance)

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

    return Figure4DecodeResults(
        continuous_results=continuous_results,
        continuous_fragmented_results=continuous_fragmented_results,
        continuous_diagnostics=continuous_diagnostics,
        continuous_fragmented_diagnostics=continuous_fragmented_diagnostics,
        spike_counts=spike_counts,
        place_field_peaks=place_field_peaks,
        diagnostic_place_fields=diagnostic_place_fields,
        diagnostic_position_bins=diagnostic_position_bins,
    )


def prepare_figure04_render_data(
    config: Figure4Config,
    paths: Figure4Paths,
    *,
    use_cache: bool = True,
) -> Figure4RenderData:
    """Assemble the Figure-4 render data: fresh recording + cached/computed decode.

    Reads only the injected ``config`` and ``paths`` (never the module-global
    ``DATA_PATH`` / ``ANIMAL_DATE_EPOCH``), so it is exercisable with synthetic
    inputs and a temporary cache directory. The cache is keyed on
    :func:`compute_figure04_cache_fingerprint`; a config / data / dependency
    change forces a recompute. The recording is always loaded fresh (it is cheap
    and never cached).

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
    recording = load_neural_recording_from_files(paths.data_path, paths.animal_date_epoch)
    print(f"  Loaded {len(recording.spike_times)} cells")

    position_info = recording.position_info
    time = np.asarray(position_info.index.to_numpy(), dtype=np.float64)
    head_position = position_info[["head_position_x", "head_position_y"]].to_numpy(dtype=np.float64)
    linear_position = position_info["linear_position"].to_numpy(dtype=np.float64)

    expected_fingerprint = compute_figure04_cache_fingerprint(config, paths)
    decode_results: Figure4DecodeResults | None = None
    if use_cache:
        print("Loading cached decoder outputs (use --force-recompute to rebuild)...")
        payload = load_figure04_cache(paths.cache_path, expected_fingerprint)
        if payload is None:
            print(
                "  No matching cache (absent, unreadable, or fingerprint mismatch "
                "from a config / data / non_local_detector change); recomputing."
            )
        else:
            decode_results = Figure4DecodeResults.from_cache_payload(payload)

    if decode_results is None:
        decode_results = _compute_figure04_decode_results(
            recording,
            time=time,
            head_position=head_position,
            decoder_config=config.decoder,
            execution_config=config.execution,
            provenance=config.provenance,
        )
        print("Caching decoder outputs to data/intermediates ...")
        save_figure04_cache(
            paths.cache_path, expected_fingerprint, decode_results.to_cache_payload()
        )

    return Figure4RenderData(
        recording=recording,
        time=time,
        head_position=head_position,
        linear_position=linear_position,
        decode_results=decode_results,
    )


def compute_figure04_summary(
    render_data: Figure4RenderData,
    thresholds: Mapping[str, float],
    metric_directions: Mapping[str, Literal["below", "above"]],
) -> Figure4Summary:
    """Compute whole-session event means and two-decoder flag agreement.

    These scalars can appear in the manuscript text, so their computation must
    stay identical to the cached decode; the render layer never alters them.
    """
    decode = render_data.decode_results
    missing_thresholds = set(metric_directions) - set(thresholds)
    if missing_thresholds:
        raise ValueError(
            f"metric_directions contains metrics without thresholds: {sorted(missing_thresholds)}"
        )

    flag_confusions = []
    for metric, worse_when in metric_directions.items():
        flag_confusions.append(
            compute_flag_confusion(
                decode.continuous_diagnostics,
                decode.continuous_fragmented_diagnostics,
                metric,
                thresholds[metric],
                worse_when=worse_when,
            )
        )

    return Figure4Summary(
        continuous=_compute_diagnostic_means(decode.continuous_diagnostics),
        continuous_fragmented=_compute_diagnostic_means(decode.continuous_fragmented_diagnostics),
        flag_confusions=tuple(flag_confusions),
    )


def format_figure04_summary(summary: Figure4Summary) -> str:
    """Format a computed Figure-4 summary for command-line output."""
    lines = ["=== Diagnostic Summary (all time points) ==="]
    for model_name, means in (
        ("Continuous", summary.continuous),
        ("ContFrag", summary.continuous_fragmented),
    ):
        lines.extend(
            [
                "",
                f"{model_name}:",
                f"  hpd_overlap: {means.hpd_overlap:.4f}",
                f"  kl_divergence: {means.kl_divergence:.4f}",
                f"  predictive_pvalue: {means.predictive_pvalue:.4f}",
            ]
        )

    # "cont-only" is the rescue quadrant: flagged by Continuous but not by
    # Continuous-Fragmented. Rescue rate is its fraction of Continuous flags.
    lines.extend(["", "=== Flag agreement: Continuous (A) vs Cont-Frag (B) ==="])
    for confusion in summary.flag_confusions:
        lines.append(
            f"  {confusion.metric}: n={confusion.n:,} both={confusion.both:,} "
            f"cont-only={confusion.a_only:,} cf-only={confusion.b_only:,} "
            f"neither={confusion.neither:,} rescue={100 * confusion.rescue_rate:.1f}%"
        )
    return "\n".join(lines)
