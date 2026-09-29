"""Identity and display labels of the two Figure-4 decoders.

Each model has one machine ID, used in data keys, file names, and command-line
choices, and one pair of display labels, used wherever a figure, printout,
viewer, or web page names it. The full labels match the manuscript's
"Continuous" and "Continuous--Fragmented"; the short labels fit tight figure
text. This module imports no sibling module, and the decode-cache modules do
not import it, so editing a label never invalidates the decode.
"""

from __future__ import annotations

import dataclasses
from typing import Literal

Figure4ModelId = Literal["continuous", "continuous_fragmented"]


@dataclasses.dataclass(frozen=True)
class Figure4Model:
    """One Figure-4 decoder's machine ID and display labels.

    Attributes
    ----------
    id : {"continuous", "continuous_fragmented"}
        Machine ID for data keys, file names, and command-line choices.
    label : str
        Full display label, as in the manuscript.
    short_label : str
        Abbreviated display label for tight figure text.
    """

    id: Figure4ModelId
    label: str
    short_label: str


CONTINUOUS = Figure4Model(id="continuous", label="Continuous", short_label="Cont.")
CONTINUOUS_FRAGMENTED = Figure4Model(
    id="continuous_fragmented",
    label="Continuous\N{EN DASH}Fragmented",
    short_label="Cont.\N{EN DASH}Frag.",
)
# In the order the figure, summary, and viewer present them.
FIGURE4_MODELS: tuple[Figure4Model, ...] = (CONTINUOUS, CONTINUOUS_FRAGMENTED)
FIGURE4_MODEL_IDS: tuple[Figure4ModelId, ...] = tuple(model.id for model in FIGURE4_MODELS)


def figure4_model(model_id: str) -> Figure4Model:
    """Return the Figure-4 model with machine ID ``model_id``.

    Raises
    ------
    ValueError
        If ``model_id`` names no Figure-4 model.
    """
    for model in FIGURE4_MODELS:
        if model.id == model_id:
            return model
    raise ValueError(f"Unknown Figure 4 model: {model_id!r}; expected one of {FIGURE4_MODEL_IDS}")
