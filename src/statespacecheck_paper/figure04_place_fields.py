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


def extract_agreed_place_fields(
    continuous_model: Any,
    continuous_fragmented_model: Any,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Extract the shared interior place fields both Figure-4 decoders agree on.

    The per-spike diagnostics use one position likelihood for both decoders,
    so each model's :func:`extract_shared_position_place_fields` must match the
    other's (fields and grid, NaN equal to NaN) before one copy is used.

    Returns
    -------
    place_fields : np.ndarray, shape (n_cells, n_interior_bins)
    position_bins : np.ndarray, shape (n_interior_bins,)

    Raises
    ------
    ValueError
        If the two models' place fields or position grids differ.
    """
    place_fields, position_bins = extract_shared_position_place_fields(continuous_model)
    other_fields, other_bins = extract_shared_position_place_fields(continuous_fragmented_model)
    if not np.allclose(place_fields, other_fields, equal_nan=True) or not np.allclose(
        position_bins, other_bins, equal_nan=True
    ):
        raise ValueError(
            "Continuous and Continuous--Fragmented place fields or position "
            "grids differ; the shared likelihood row would misrepresent one "
            "of the decoders."
        )
    return place_fields, position_bins


def marginalize_state_bins(distribution_da: xr.DataArray) -> xr.DataArray:
    """Sum a distribution over the decoder's discrete states.

    Multi-state models encode ``(state, position)`` in ``state_bins`` as a
    MultiIndex, which is unstacked and summed over ``state``; a single-state
    model's plain ``state_bins`` index is returned unchanged. Branching on the
    index type, rather than catching a generic unstack failure, keeps a
    malformed multi-state index from being treated as single-state and
    returning a per-state slice labeled as the marginal.

    Parameters
    ----------
    distribution_da : xr.DataArray, dims (time, state_bins)
        Distribution over the decoder's state bins.

    Returns
    -------
    xr.DataArray
        Dims ``(time, position)`` for a multi-state model, else unchanged.

    Raises
    ------
    ValueError
        If the ``state_bins`` MultiIndex is malformed (e.g. duplicate
        ``(state, position)`` entries) and cannot be unstacked.
    """
    if not isinstance(distribution_da.indexes["state_bins"], pd.MultiIndex):
        return distribution_da
    try:
        unstacked: xr.DataArray = distribution_da.unstack("state_bins")
    except (ValueError, KeyError, TypeError) as e:
        raise ValueError(
            "Failed to unstack the state_bins MultiIndex on the decoder "
            "distribution; the index is malformed (likely duplicate "
            f"(state, position) entries) and cannot be marginalized. Underlying error: {e}"
        ) from e
    # ``skipna=False``: if the per-state interior masks differed, unstack would
    # back-fill missing (state, position) cells with NaN, and a skipna sum would
    # silently produce an asymmetric marginal that still looks like a
    # distribution. ``get_state_marginalized_posterior``'s callers also pair this
    # with ``extract_shared_position_place_fields`` (which rejects state-varying
    # masks); the heatmap relies on this rule directly to draw such cells as NaN.
    if "state" in unstacked.dims:
        marginal: xr.DataArray = unstacked.sum("state", skipna=False)
        return marginal
    return unstacked


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

    posterior: NDArray[np.float64] = np.asarray(marginalize_state_bins(posterior_da).values)
    return posterior
