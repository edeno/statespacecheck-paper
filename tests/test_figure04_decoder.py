"""Tests for Figure-4 decoder configuration and construction."""

from __future__ import annotations

import dataclasses
from typing import Any

import numpy as np
import pytest

import statespacecheck_paper.figure04_decoder as figure04_decoder


class TestFigure4ConfigMatchesManuscript:
    """Drift guard: the decoder models the code actually builds must carry the
    exact parameters stated in the manuscript (``main.tex:294``).

    Rather than reconstruct the nested transition structure (which risks
    changing the decode), this fits nothing and asserts the *resolved*
    attributes of the models produced by
    :func:`figure04_decoder.build_decoder_models`. These attributes are set at
    construction time from ``non_local_detector`` class defaults plus the two
    explicit KDE parameters, so the check needs neither real data nor a fit; a
    dependency bump that silently changes any default fails here.
    """

    @staticmethod
    def _build_models() -> tuple[Any, Any]:
        pytest.importorskip("non_local_detector")
        from non_local_detector.environment import Environment

        # Exercise the real injection path: the workflow builds the Environment
        # with place_bin_size from the config and passes the decoder config into
        # build_decoder_models. The injected values equal the non_local_detector
        # defaults, so the resolved attributes checked below are unchanged --
        # which is exactly the equivalence this guard pins.
        config = figure04_decoder.Figure4Config()
        env = Environment(place_bin_size=config.decoder.position_bin_size_cm)
        return figure04_decoder.build_decoder_models(env, config.decoder, config.execution)

    def test_continuous_observation_and_transition(self) -> None:
        from non_local_detector.continuous_state_transitions import RandomWalk

        continuous_model, _ = self._build_models()
        config = figure04_decoder.Figure4Config()

        # main.tex:294 -- sorted-spikes KDE positional bandwidth sqrt(12.5).
        assert continuous_model.sorted_spikes_algorithm_params["position_std"] == pytest.approx(
            config.decoder.position_std
        )
        assert config.decoder.position_std == pytest.approx(float(np.sqrt(12.5)))

        # main.tex:294 -- zero-mean Gaussian random walk, movement_var = 6.0 cm^2.
        random_walk = continuous_model.continuous_transition_types[0][0]
        assert isinstance(random_walk, RandomWalk)
        assert random_walk.movement_var == pytest.approx(config.provenance.movement_var)
        assert config.provenance.movement_var == pytest.approx(6.0)

    def test_contfrag_discrete_dynamics(self) -> None:
        from non_local_detector.continuous_state_transitions import RandomWalk
        from non_local_detector.discrete_state_transitions import DiscreteStationaryDiagonal

        _, contfrag_model = self._build_models()
        config = figure04_decoder.Figure4Config()

        # main.tex:294 -- ContFrag Continuous-to-Continuous transition reuses the
        # same random walk (movement_var = 6.0).
        random_walk = contfrag_model.continuous_transition_types[0][0]
        assert isinstance(random_walk, RandomWalk)
        assert random_walk.movement_var == pytest.approx(config.provenance.movement_var)

        # main.tex:294 -- mode-transition matrix [[0.98, 0.02], [0.02, 0.98]],
        # i.e. a stationary diagonal (0.98, 0.98).
        discrete_transition_type = contfrag_model.discrete_transition_type
        assert isinstance(discrete_transition_type, DiscreteStationaryDiagonal)
        np.testing.assert_array_equal(
            np.asarray(discrete_transition_type.diagonal_values, dtype=float),
            np.asarray(config.provenance.contfrag_diagonal_values, dtype=float),
        )

        # main.tex:294 -- Continuous / Fragmented modes initialized at (0.5, 0.5).
        np.testing.assert_array_equal(
            np.asarray(contfrag_model.discrete_initial_conditions, dtype=float),
            np.asarray(config.provenance.contfrag_discrete_initial_conditions, dtype=float),
        )

    def test_unprinted_effective_defaults(self) -> None:
        """Concentration / regularization are not printed in the manuscript but
        shape the decode; pin them so a dependency bump fails loudly."""
        continuous_model, contfrag_model = self._build_models()
        config = figure04_decoder.Figure4Config()

        assert contfrag_model.discrete_transition_concentration == pytest.approx(
            config.provenance.discrete_transition_concentration
        )
        assert config.provenance.discrete_transition_concentration == pytest.approx(1.1)

        for model in (continuous_model, contfrag_model):
            assert model.discrete_transition_regularization == pytest.approx(
                config.provenance.discrete_transition_regularization
            )
        assert config.provenance.discrete_transition_regularization == pytest.approx(1e-10)

    def test_binning_values_the_code_uses(self) -> None:
        """Position bin size (from the Environment) and time bin size (from the
        sampling frequency) are the values the decode actually uses."""
        continuous_model, contfrag_model = self._build_models()
        config = figure04_decoder.Figure4Config()

        # main.tex:294 -- ~2 cm spatial bins.
        for model in (continuous_model, contfrag_model):
            assert model.environments[0].place_bin_size == pytest.approx(
                config.decoder.position_bin_size_cm
            )
        assert config.decoder.position_bin_size_cm == pytest.approx(2.0)

        # main.tex:294 -- 2 ms spike bins == 500 Hz sampling frequency.
        for model in (continuous_model, contfrag_model):
            assert model.sampling_frequency == pytest.approx(config.decoder.sampling_frequency_hz)
        assert config.decoder.sampling_frequency_hz == pytest.approx(500.0)
        assert config.decoder.time_bin_size_ms == pytest.approx(2.0)

    def test_config_records_manuscript_dependency_version(self) -> None:
        """The provenance string must match the manuscript-stated version."""
        config = figure04_decoder.Figure4Config()
        assert config.provenance.non_local_detector_version == "0.6.10.dev214+g956fdccaf"


class TestConfigValueValidation:
    @pytest.mark.parametrize(
        "kwargs",
        [
            {"position_std": 0.0},
            {"position_std": -1.0},
            {"position_std": float("nan")},
            {"position_bin_size_cm": 0.0},
            {"sampling_frequency_hz": -5.0},
            {"sampling_frequency_hz": float("inf")},
        ],
    )
    def test_decoder_config_rejects_nonpositive_or_nonfinite(self, kwargs: dict) -> None:
        with pytest.raises(ValueError, match="must be finite and positive"):
            figure04_decoder.Figure4DecoderConfig(**kwargs)

    @pytest.mark.parametrize("block_size", [0, -1])
    def test_execution_config_rejects_nonpositive_block_size(self, block_size: int) -> None:
        with pytest.raises(ValueError, match="block_size must be a positive integer"):
            figure04_decoder.Figure4ExecutionConfig(block_size=block_size)

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"movement_var": 0.0},
            {"discrete_transition_concentration": -1.0},
            {"discrete_transition_regularization": 0.0},
            {"contfrag_diagonal_values": (1.2, 0.98)},
            {"contfrag_discrete_initial_conditions": (-0.1, 1.1)},
            {"non_local_detector_version": ""},
        ],
    )
    def test_provenance_rejects_invalid(self, kwargs: dict) -> None:
        with pytest.raises(ValueError):
            figure04_decoder.Figure4Provenance(**kwargs)


class TestNonDefaultPropagation:
    """Non-default config values must flow through to the built models (guarding
    against the injection being accidentally hardcoded)."""

    @staticmethod
    def _models(decoder: Any = None, execution: Any = None) -> tuple[Any, Any]:
        pytest.importorskip("non_local_detector")
        from non_local_detector.environment import Environment

        decoder = decoder or figure04_decoder.Figure4DecoderConfig()
        env = Environment(place_bin_size=decoder.position_bin_size_cm)
        return figure04_decoder.build_decoder_models(env, decoder, execution)

    def test_position_std_propagates(self) -> None:
        cont, cf = self._models(figure04_decoder.Figure4DecoderConfig(position_std=5.0))
        assert cont.sorted_spikes_algorithm_params["position_std"] == pytest.approx(5.0)
        assert cf.sorted_spikes_algorithm_params["position_std"] == pytest.approx(5.0)

    def test_block_size_propagates(self) -> None:
        cont, _ = self._models(execution=figure04_decoder.Figure4ExecutionConfig(block_size=1234))
        assert cont.sorted_spikes_algorithm_params["block_size"] == 1234

    def test_sampling_frequency_propagates(self) -> None:
        cont, cf = self._models(figure04_decoder.Figure4DecoderConfig(sampling_frequency_hz=250.0))
        assert cont.sampling_frequency == pytest.approx(250.0)
        assert cf.sampling_frequency == pytest.approx(250.0)

    def test_position_bin_size_propagates_via_environment(self) -> None:
        pytest.importorskip("non_local_detector")
        env = figure04_decoder.create_decoder_environment(
            track_graph=None, edge_order=[], edge_spacing=0.0, place_bin_size=3.0
        )
        assert env.place_bin_size == pytest.approx(3.0)


class TestValidateProvenanceDefaults:
    @staticmethod
    def _models() -> tuple[Any, Any]:
        pytest.importorskip("non_local_detector")
        from non_local_detector.environment import Environment

        return figure04_decoder.build_decoder_models(Environment())

    def test_passes_for_recorded_defaults(self) -> None:
        cont, cf = self._models()
        # nld defaults still match Figure4Provenance -> no raise.
        figure04_decoder.validate_provenance_defaults(cont, cf)

    def test_raises_on_scalar_drift(self) -> None:
        cont, cf = self._models()
        drifted = dataclasses.replace(figure04_decoder.Figure4Provenance(), movement_var=999.0)
        with pytest.raises(ValueError, match="default drift"):
            figure04_decoder.validate_provenance_defaults(cont, cf, drifted)

    def test_raises_on_array_drift(self) -> None:
        cont, cf = self._models()
        drifted = dataclasses.replace(
            figure04_decoder.Figure4Provenance(), contfrag_diagonal_values=(0.5, 0.5)
        )
        with pytest.raises(ValueError, match="default drift"):
            figure04_decoder.validate_provenance_defaults(cont, cf, drifted)
