"""The Figure-4 model registry: one machine ID and one label pair per model."""

from __future__ import annotations

import pytest

from statespacecheck_paper.figure04_models import (
    CONTINUOUS,
    CONTINUOUS_FRAGMENTED,
    FIGURE04_MODEL_IDS,
    FIGURE04_MODELS,
    Figure4Model,
    figure04_model,
)


def test_registry_lists_both_models_in_figure_order() -> None:
    assert FIGURE04_MODELS == (CONTINUOUS, CONTINUOUS_FRAGMENTED)
    assert FIGURE04_MODEL_IDS == ("continuous", "continuous_fragmented")


def test_labels_match_the_manuscript_names() -> None:
    # The manuscript writes Continuous--Fragmented, an en dash in LaTeX.
    assert (CONTINUOUS.label, CONTINUOUS.short_label) == ("Continuous", "Cont.")
    assert (CONTINUOUS_FRAGMENTED.label, CONTINUOUS_FRAGMENTED.short_label) == (
        "Continuous\N{EN DASH}Fragmented",
        "Cont.\N{EN DASH}Frag.",
    )


@pytest.mark.parametrize("model", FIGURE04_MODELS)
def test_lookup_by_id_returns_the_model(model: Figure4Model) -> None:
    assert figure04_model(model.id) is model


def test_lookup_rejects_an_unknown_id() -> None:
    with pytest.raises(ValueError, match="Unknown Figure 4 model: 'not_a_model'"):
        figure04_model("not_a_model")
