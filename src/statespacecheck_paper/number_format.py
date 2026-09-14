"""Presentation rounding shared by the manuscript prose and the figures.

The reporting policy (see ``reported_values``) says decoding errors print to
two significant figures and flag and rescue percentages to the nearest whole
percent. Figure 3b prints the same quantities in its heatmap and error row,
so both the macro emitter and the figure code call these two functions: the
figure and the text beside it cannot round the same number differently.

Imports no sibling module, so it is a leaf of the dependency graph.
"""

from __future__ import annotations

import math

# Significant figures for decoding errors and for the derived constants the
# manuscript hedges with "approximately". Percentages use whole percents.
SIGNIFICANT_FIGURES = 2


def whole_percent(value: float) -> str:
    """Render a percentage to the nearest whole percent.

    Parameters
    ----------
    value : float
        Percentage in ``[0, 100]``.

    Returns
    -------
    str
        The rounded percentage without a sign.

    Examples
    --------
    >>> whole_percent(36.77115625352582), whole_percent(0.6675931668463562)
    ('37', '1')
    """
    return f"{float(value):.0f}"


def significant(value: float, digits: int = SIGNIFICANT_FIGURES) -> str:
    """Render a value to a stated number of significant figures.

    Used for decoding errors and derived constants that the manuscript
    introduces with "approximately" or a tilde. For example, 42.169 renders
    as 42 and 199.47 as 200 at two significant figures. The format preserves
    trailing fractional zeros, as in 8.0, and uses plain decimal notation.

    Parameters
    ----------
    value : float
        Value to render; NumPy scalars are accepted and converted. Zero prints
        as ``"0"``; a decoding error of exactly zero is a valid summary.
    digits : int, default ``SIGNIFICANT_FIGURES``
        Significant figures to keep.

    Returns
    -------
    str
        The value in plain decimal notation.

    Examples
    --------
    >>> significant(199.47114020071638, 2)
    '200'
    >>> significant(0.19947114020071638, 2)
    '0.20'
    >>> significant(0.999, 2), significant(9.99, 2), significant(0.0, 2)
    ('1.0', '10', '0')
    """
    # The emitter passes Python floats and the figure code NumPy scalars, and
    # round() differs between them: Python rounds the exact binary value
    # (2.45 is stored just above 2.45, so it gives 2.5) while NumPy gives 2.4.
    # Normalize first so both rendering paths print the same digits.
    value = float(value)
    if value == 0.0:
        return "0"
    exponent = math.floor(math.log10(abs(value)))
    rounded = round(value, -(exponent - digits + 1))
    # Rounding can carry across a power of ten (0.999 -> 1.0, 9.99 -> 10);
    # the decimal count must follow the rounded value's exponent, or the
    # output shows one significant figure too many.
    exponent = math.floor(math.log10(abs(rounded)))
    return f"{rounded:.{max(0, digits - 1 - exponent)}f}"
