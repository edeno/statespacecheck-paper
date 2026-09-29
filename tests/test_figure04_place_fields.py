"""Tests for Figure-4 place-field and position-marginal distribution extraction."""

from __future__ import annotations

from typing import Literal
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from statespacecheck_paper.figure04_place_fields import (
    DECODER_PREDICTIVE_VAR,
    DECODER_SMOOTHED_VAR,
    extract_agreed_place_fields,
    extract_place_fields,
    extract_shared_position_place_fields,
    marginal_position_distribution,
    marginalize_state_bins,
)

from ._decoder_inputs import xarray_results

# ---------------------------------------------------------------------------
# extract_place_fields
# ---------------------------------------------------------------------------


def _mock_model(
    place_fields: np.ndarray,
    position_bins: np.ndarray,
    env_name: str = "",
    encoding_group: int = 0,
) -> MagicMock:
    """Mock with the small surface used by ``extract_place_fields``."""
    mock_model = MagicMock()
    mock_model.encoding_model_ = {(env_name, encoding_group): {"place_fields": place_fields}}
    mock_env = MagicMock()
    mock_env.place_bin_centers_ = position_bins.reshape(-1, 1)
    envs: list[MagicMock | None] = [None] * (encoding_group + 1)
    envs[encoding_group] = mock_env
    mock_model.environments = envs
    return mock_model


class TestExtractPlaceFields:
    def test_extracts_default_environment(self, rng: np.random.Generator) -> None:
        n_cells, n_bins = 10, 50
        place_fields = rng.random((n_cells, n_bins)) * 10
        position_bins = np.linspace(0, 100, n_bins)
        model = _mock_model(place_fields, position_bins)
        pf, bins = extract_place_fields(model)
        np.testing.assert_array_equal(pf, place_fields)
        np.testing.assert_array_equal(bins, position_bins)

    def test_extracts_named_environment_and_group(self, rng: np.random.Generator) -> None:
        n_cells, n_bins = 5, 30
        place_fields = rng.random((n_cells, n_bins)) * 10
        position_bins = np.linspace(0, 50, n_bins)
        model = _mock_model(place_fields, position_bins, env_name="env1", encoding_group=1)
        pf, bins = extract_place_fields(model, environment_name="env1", encoding_group=1)
        np.testing.assert_array_equal(pf, place_fields)
        np.testing.assert_array_equal(bins, position_bins)


# ---------------------------------------------------------------------------
# marginal_position_distribution
# ---------------------------------------------------------------------------


class TestMarginalPositionDistribution:
    @pytest.mark.parametrize(
        ("kind", "variable"),
        [("predictive", DECODER_PREDICTIVE_VAR), ("smoothed", DECODER_SMOOTHED_VAR)],
    )
    def test_single_state_passthrough(
        self,
        rng: np.random.Generator,
        kind: Literal["predictive", "smoothed"],
        variable: str,
    ) -> None:
        """Single-state model: no states to marginalize, output equals input."""
        n_time, n_bins = 100, 50
        posterior_data = rng.dirichlet(np.ones(n_bins), size=n_time)
        results = xarray_results(posterior_data, variable)
        result = marginal_position_distribution(results, kind)
        assert result.shape == (n_time, n_bins)
        np.testing.assert_allclose(result, posterior_data)

    def test_multi_state_sums_over_states(self, rng: np.random.Generator) -> None:
        """Multi-state model: result is the sum across states."""
        n_time, n_bins, n_states = 100, 50, 2
        posterior_per_state = rng.dirichlet(np.ones(n_bins), size=(n_time, n_states))
        posterior_per_state = posterior_per_state / posterior_per_state.sum(
            axis=(1, 2), keepdims=True
        )
        states = ["Continuous", "Fragmented"]
        positions = np.arange(n_bins, dtype=float)
        multi_index = pd.MultiIndex.from_product([states, positions], names=["state", "position"])
        results = xarray_results(
            posterior_per_state.reshape(n_time, -1),
            "predictive_posterior",
            state_bins=multi_index,
        )
        result = marginal_position_distribution(results, "predictive")
        np.testing.assert_allclose(result, posterior_per_state.sum(axis=1), rtol=1e-5)

    def test_unstack_failure_raises(self) -> None:
        """A malformed state_bins index that fails to unstack must
        raise — silently treating it as single-state and returning a
        per-state slice would render a wrong figure with no warning."""
        # Build a posterior with a state_bins coordinate that has
        # duplicate (state, position) entries, which xarray cannot
        # unstack into a (state, position) rectangular product.
        n_time = 20
        # Duplicated (state, position) tuples — unstack fails.
        broken_index = pd.MultiIndex.from_tuples(
            [("A", 0), ("A", 0), ("B", 0), ("B", 0)], names=["state", "position"]
        )
        results = xarray_results(
            np.random.default_rng(0).random((n_time, 4)),
            "predictive_posterior",
            state_bins=broken_index,
        )
        with pytest.raises(ValueError, match="Failed to unstack"):
            marginal_position_distribution(results, "predictive")


# ---------------------------------------------------------------------------
# extract_shared_position_place_fields
# ---------------------------------------------------------------------------


class TestExtractSharedPositionPlaceFields:
    def test_rejects_state_specific_observation_likelihoods(self) -> None:
        position_bins = np.array([0.0, 1.0, 2.0])
        model = MagicMock()
        model.observation_models = [
            MagicMock(environment_name="", encoding_group=0),
            MagicMock(environment_name="", encoding_group=1),
        ]
        model.encoding_model_ = {
            ("", 0): {"place_fields": np.ones((2, 3))},
            ("", 1): {"place_fields": np.full((2, 3), 2.0)},
        }
        environments = []
        for _ in range(2):
            environment = MagicMock()
            environment.place_bin_centers_ = position_bins[:, np.newaxis]
            environments.append(environment)
        model.environments = environments
        model.is_track_interior_state_bins_ = np.ones(6, dtype=bool)

        with pytest.raises(ValueError, match="likelihood differs"):
            extract_shared_position_place_fields(model)

    def test_rejects_state_dependent_interior_mask(self) -> None:
        # Same place fields and grid across states, but the track-interior
        # mask differs per state -> the shared position marginal is ill-defined
        # and must be rejected rather than silently reshaped into a wrong mask.
        position_bins = np.array([0.0, 1.0, 2.0])
        place_fields = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        model = MagicMock()
        model.observation_models = [
            MagicMock(environment_name="", encoding_group=0),
            MagicMock(environment_name="", encoding_group=0),
        ]
        model.encoding_model_ = {("", 0): {"place_fields": place_fields}}
        environment = MagicMock()
        environment.place_bin_centers_ = position_bins[:, np.newaxis]
        model.environments = [environment]
        # State 0 interior [T, F, T]; state 1 interior [T, T, F] -> differ.
        model.is_track_interior_state_bins_ = np.array([True, False, True, True, True, False])

        with pytest.raises(ValueError, match="interior mask differs"):
            extract_shared_position_place_fields(model)


def _single_state_model(place_fields: np.ndarray, position_bins: np.ndarray) -> MagicMock:
    """One-state model whose interior mask keeps every bin but the last."""
    model = _mock_model(place_fields, position_bins)
    model.observation_models = [MagicMock(environment_name="", encoding_group=0)]
    model.is_track_interior_state_bins_ = np.arange(position_bins.size) < position_bins.size - 1
    return model


class TestExtractAgreedPlaceFields:
    def test_returns_the_shared_interior_fields(self) -> None:
        position_bins = np.array([0.0, 1.0, 2.0])
        place_fields = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        fields, bins = extract_agreed_place_fields(
            _single_state_model(place_fields, position_bins),
            _single_state_model(place_fields.copy(), position_bins.copy()),
        )
        np.testing.assert_array_equal(fields, place_fields[:, :2])
        np.testing.assert_array_equal(bins, position_bins[:2])

    def test_rejects_models_whose_fields_differ(self) -> None:
        position_bins = np.array([0.0, 1.0, 2.0])
        place_fields = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        with pytest.raises(ValueError, match="place fields or position grids differ"):
            extract_agreed_place_fields(
                _single_state_model(place_fields, position_bins),
                _single_state_model(place_fields + 1.0, position_bins),
            )


def _state_bins_da(values: np.ndarray, index: pd.MultiIndex | np.ndarray) -> xr.DataArray:
    return xr.DataArray(
        values,
        dims=["time", "state_bins"],
        coords={"time": np.arange(values.shape[0]), "state_bins": index},
    )


class TestMarginalizeStateBins:
    def test_position_missing_from_one_state_stays_nan(self) -> None:
        # skipna=False: a position NaN in one state must not become that
        # state's partial sum labeled as the marginal.
        values = np.ones((2, 6))
        values[:, 0] = np.nan  # state A, position 0
        index = pd.MultiIndex.from_product(
            [["A", "B"], [0.0, 1.0, 2.0]], names=["state", "position"]
        )
        marginal = marginalize_state_bins(_state_bins_da(values, index))
        assert marginal.dims == ("time", "position")
        np.testing.assert_array_equal(marginal.values, [[np.nan, 2.0, 2.0]] * 2)

    def test_single_state_index_is_returned_unchanged(self) -> None:
        da = _state_bins_da(np.ones((2, 3)), np.arange(3))
        assert marginalize_state_bins(da) is da

    def test_index_without_a_state_level_is_only_unstacked(self) -> None:
        index = pd.MultiIndex.from_product([[0.0, 1.0, 2.0]], names=["position"])
        values = np.arange(6.0).reshape(2, 3)
        marginal = marginalize_state_bins(_state_bins_da(values, index))
        assert marginal.dims == ("time", "position")
        np.testing.assert_array_equal(marginal.values, values)


class TestExtractAgreedPlaceFieldsEdges:
    def test_rejects_models_whose_grids_differ(self) -> None:
        place_fields = np.array([[1.0, 2.0, 3.0]])
        position_bins = np.array([0.0, 1.0, 2.0])
        with pytest.raises(ValueError, match="place fields or position grids differ"):
            extract_agreed_place_fields(
                _single_state_model(place_fields, position_bins),
                _single_state_model(place_fields, position_bins + 1.0),
            )

    def test_matching_nan_fields_agree(self) -> None:
        place_fields = np.array([[np.nan, 2.0, 3.0]])
        position_bins = np.array([0.0, 1.0, 2.0])
        fields, _ = extract_agreed_place_fields(
            _single_state_model(place_fields, position_bins),
            _single_state_model(place_fields.copy(), position_bins),
        )
        np.testing.assert_array_equal(fields, [[np.nan, 2.0]])
