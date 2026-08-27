"""Extracting place fields and marginalized posteriors from fitted models.

Helpers that read the fitted ``non_local_detector`` decoder models and their
``predict`` outputs: per-observation-model place fields, the single shared
position-dependent observation likelihood used for cross-model diagnostics, and
the state-marginalized posterior over position.
"""

from __future__ import annotations

from typing import Any, Literal

import numpy as np
import pandas as pd
import xarray as xr
from numpy.typing import NDArray


def extract_place_fields(
    model: Any,
    environment_name: str = "",
    encoding_group: int = 0,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Extract place fields and position bins from fitted decoder model.

    Retrieves the place field firing rates and corresponding position bin centers
    from a fitted `SortedSpikesDecoder` or `ContFragSortedSpikesClassifier` model.

    Parameters
    ----------
    model : SortedSpikesDecoder or ContFragSortedSpikesClassifier
        Fitted decoder model from non_local_detector package.
    environment_name : str, default ""
        Name of the environment in the model. Default empty string for standard
        single-environment models.
    encoding_group : int, default 0
        Encoding group index. Default 0 for standard models.

    Returns
    -------
    place_fields : np.ndarray, shape (n_cells, n_bins)
        Firing rate at each position bin for each cell (in Hz or spikes/time).
    position_bins : np.ndarray, shape (n_bins,)
        Position bin centers.

    Examples
    --------
    >>> # Requires fitted model from non_local_detector
    >>> # place_fields, position_bins = extract_place_fields(model)
    >>> # place_fields.shape  # (n_cells, n_bins)
    >>> # position_bins.shape  # (n_bins,)
    """
    # Access place fields from encoding model
    # Key is tuple (environment_name, encoding_group)
    key = (environment_name, encoding_group)
    place_fields: NDArray[np.float64] = model.encoding_model_[key]["place_fields"]

    # Get position bin centers from environment
    # environments is a list; encoding_group corresponds to environment index
    position_bins: NDArray[np.float64] = model.environments[
        encoding_group
    ].place_bin_centers_.squeeze()

    return place_fields, position_bins


def extract_place_fields_concat(
    model: Any,
) -> tuple[NDArray[np.float64], NDArray[np.bool_]]:
    """Concatenate per-observation-model place fields + the interior mask.

    Returns the place fields aligned with the predictive posterior's
    full ``state_bins`` axis (i.e. before the interior-mask filter):
    one ``(n_cells, n_state_bins_full)`` array stacked across the
    model's observation models, plus the matching boolean
    ``is_track_interior_state_bins_`` mask. Callers that only need
    the interior bins do ``place_fields[:, interior_mask]``.

    Used by the interactive cache builder, which keeps both arrays so the
    viewer can reconstruct the non-interior NaN columns. Diagnostics that
    compare models with different numbers of discrete states should instead
    use :func:`extract_shared_position_place_fields`.
    """
    place_fields = np.concatenate(
        [
            extract_place_fields(
                model,
                environment_name=obs.environment_name,
                encoding_group=obs.encoding_group,
            )[0]
            for obs in model.observation_models
        ],
        axis=1,
    )
    interior_mask: NDArray[np.bool_] = np.asarray(model.is_track_interior_state_bins_, dtype=bool)
    return place_fields, interior_mask


def extract_shared_position_place_fields(
    model: Any,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Extract one shared observation likelihood over interior positions.

    Multi-state decoders can repeat the same position-dependent observation
    model once per discrete state. Those repeated columns belong in the joint
    decoder state space, but not in diagnostics intended to compare the neural
    evidence about position across models with different numbers of states.
    This helper verifies that all discrete states share the same place fields
    and position grid, then returns exactly one copy restricted to track
    interior bins.

    Raises
    ------
    ValueError
        If the model has no observation models, different observation
        likelihoods or position grids across states, or inconsistent track
        interior masks. In those cases there is no single shared positional
        likelihood to use after marginalizing the discrete state.
    """
    observation_models = list(model.observation_models)
    if not observation_models:
        raise ValueError("Decoder model has no observation models.")

    first = observation_models[0]
    place_fields, position_bins = extract_place_fields(
        model,
        environment_name=first.environment_name,
        encoding_group=first.encoding_group,
    )
    place_fields = np.asarray(place_fields, dtype=np.float64)
    position_bins = np.asarray(position_bins, dtype=np.float64).reshape(-1)

    for observation_model in observation_models[1:]:
        candidate_fields, candidate_bins = extract_place_fields(
            model,
            environment_name=observation_model.environment_name,
            encoding_group=observation_model.encoding_group,
        )
        candidate_fields = np.asarray(candidate_fields, dtype=np.float64)
        candidate_bins = np.asarray(candidate_bins, dtype=np.float64).reshape(-1)
        if candidate_fields.shape != place_fields.shape or not np.allclose(
            candidate_fields, place_fields, equal_nan=True
        ):
            raise ValueError(
                "Cannot compute position-marginal diagnostics because the "
                "observation likelihood differs across discrete states."
            )
        if candidate_bins.shape != position_bins.shape or not np.allclose(
            candidate_bins, position_bins, equal_nan=True
        ):
            raise ValueError(
                "Cannot compute position-marginal diagnostics because the "
                "position grid differs across discrete states."
            )

    n_positions = position_bins.size
    if place_fields.shape[1] != n_positions:
        raise ValueError(
            f"Place fields have {place_fields.shape[1]} bins but the position "
            f"grid has {n_positions}."
        )

    interior_mask = np.asarray(model.is_track_interior_state_bins_, dtype=bool).reshape(-1)
    if interior_mask.size % n_positions != 0:
        raise ValueError(
            f"Interior mask has {interior_mask.size} bins, which is not "
            f"divisible by the {n_positions}-bin position grid."
        )
    state_masks = interior_mask.reshape(-1, n_positions)
    if state_masks.shape[0] != len(observation_models):
        raise ValueError(
            f"Interior mask represents {state_masks.shape[0]} states but the "
            f"model has {len(observation_models)} observation models."
        )
    if not np.all(state_masks == state_masks[0]):
        raise ValueError(
            "Cannot compute position-marginal diagnostics because the track "
            "interior mask differs across discrete states."
        )

    position_mask = state_masks[0]
    return place_fields[:, position_mask], position_bins[position_mask]


def get_state_marginalized_posterior(
    results: xr.Dataset,
    posterior_type: Literal["predictive", "acausal"] = "predictive",
) -> NDArray[np.float64]:
    """Extract state-marginalized posterior from decoder results.

    For multi-state models (e.g., ContFragSortedSpikesClassifier), sums over
    states to get the marginal posterior over position. For single-state models,
    simply extracts the posterior. Also handles NaN state bins (e.g., track edges).

    Parameters
    ----------
    results : xr.Dataset
        Decoding results from model.predict() containing posterior distributions.
    posterior_type : {"predictive", "acausal"}, default "predictive"
        Type of posterior to extract:
        - "predictive": One-step-ahead prediction p(x_t | y_{1:t-1})
        - "acausal": Smoothed posterior p(x_t | y_{1:T})

    Returns
    -------
    posterior : np.ndarray, shape (n_time, n_bins)
        State-marginalized posterior summed over states, with NaN bins dropped.

    Raises
    ------
    ValueError
        If ``posterior_type`` is not ``"predictive"`` or ``"acausal"``,
        or if the ``state_bins`` MultiIndex on a multi-state model is
        malformed (e.g. duplicate ``(state, position)`` entries) and
        cannot be unstacked. Refusing here is intentional: a silent
        fallback would return a per-state slice labeled as the
        marginal posterior, producing a wrong figure.

    Examples
    --------
    >>> # Requires xarray Dataset from non_local_detector
    >>> # posterior = get_state_marginalized_posterior(results, "predictive")
    >>> # posterior.shape  # (n_time, n_bins)
    """
    # Select appropriate posterior
    if posterior_type == "predictive":
        posterior_da = results.predictive_posterior
    elif posterior_type == "acausal":
        posterior_da = results.acausal_posterior
    else:
        raise ValueError(
            f"Invalid posterior_type: {posterior_type}. Must be 'predictive' or 'acausal'."
        )

    # Drop NaN state bins (e.g., track interior only)
    posterior_da = posterior_da.dropna("state_bins")

    # Multi-state models encode (state, position) in state_bins as a
    # MultiIndex; single-state models use a plain Index. Branch on
    # the index type rather than catching a generic unstack failure,
    # which would silently treat a malformed multi-state model as
    # single-state and produce a per-state slice labeled as marginal.
    state_bins_index = posterior_da.indexes["state_bins"]
    if isinstance(state_bins_index, pd.MultiIndex):
        try:
            unstacked = posterior_da.unstack("state_bins")
        except (ValueError, KeyError) as e:
            raise ValueError(
                "Failed to unstack the state_bins MultiIndex on the "
                "decoder posterior; the index is malformed (likely "
                "duplicate (state, position) entries) and cannot be "
                f"marginalized. Underlying error: {e}"
            ) from e
        # ``skipna=False``: if the per-state interior masks differed, unstack
        # would back-fill missing (state, position) cells with NaN, and a
        # skipna sum would silently produce an asymmetric marginal that still
        # looks like a distribution. Callers pair this with
        # ``extract_shared_position_place_fields`` (which rejects state-varying
        # masks), but this keeps the marginal honest even without that guard.
        if "state" in unstacked.dims:
            marginalized = unstacked.sum("state", skipna=False)
        else:
            marginalized = unstacked
        posterior: NDArray[np.float64] = np.asarray(marginalized.values)
    else:
        # Single-state model: no states to sum over.
        posterior = np.asarray(posterior_da.values)

    return posterior
