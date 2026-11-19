"""Exact decimal parsing and half-even arithmetic shared by money bridges.

Decimal is used only to read finite decimal text. All arithmetic uses Fraction;
it therefore has no dependency on the process's decimal context.
"""

import math
import re
from decimal import Decimal, InvalidOperation
from fractions import Fraction
from numbers import Integral, Real

_DECIMAL_TEXT = re.compile(r"[0-9]+(?:\.[0-9]+)?")


def _decimal_text(value: object, label: str, *, allow_zero: bool = False) -> Fraction:
    """A positive decimal written as JSON text; optionally admit exact zero."""
    if not isinstance(value, str):
        raise ValueError(
            f"Money normalization {label} must be decimal text, never a JSON "
            "number: a binary float cannot carry the declared decimal exactly."
        )
    if _DECIMAL_TEXT.fullmatch(value) is None:
        raise ValueError(f"Money normalization {label} must be plain decimal text.")
    try:
        result = Fraction(Decimal(value))
    except (InvalidOperation, ValueError) as error:
        raise ValueError(f"Money normalization {label} is not decimal text.") from error
    if result < 0 or (result == 0 and not allow_zero):
        restriction = "nonnegative" if allow_zero else "positive"
        raise ValueError(f"Money normalization {label} must be {restriction}.")
    return result


def _half_even(value: Fraction) -> int:
    """Round an exact rational to an integer, ties to even."""
    return round(value)


def _shortest_decimal(value: object) -> Fraction:
    """The shortest round-trip decimal of a float, read without arithmetic."""
    number = float(value)
    if not math.isfinite(number):
        raise ValueError("Money normalization refuses NaN and infinite amounts.")
    return Fraction(Decimal(repr(number)))


def _exact_value(value: object) -> Fraction:
    """Read integers exactly and floats with the legacy shortest-decimal rule.

    New exact-arithmetic callers use this helper without changing donor money
    normalization's established ``_shortest_decimal`` conversion. An integer
    never passes through float64. Fractions and finite Decimals keep their
    exact value; unsupported values and lossy float conversions refuse.
    """
    if isinstance(value, bool):
        raise ValueError("Exact money refuses boolean amounts.")
    if isinstance(value, Integral):
        return Fraction(int(value))
    if isinstance(value, Fraction):
        return value
    if isinstance(value, Decimal):
        if not value.is_finite():
            raise ValueError("Money normalization refuses NaN and infinite amounts.")
        return Fraction(value)
    if isinstance(value, Real):
        number = float(value)
        if not math.isfinite(number):
            raise ValueError("Money normalization refuses NaN and infinite amounts.")
        if number != value:
            raise ValueError("Exact money refuses a lossy float conversion.")
        return _shortest_decimal(number)
    raise ValueError("Exact money requires an integer, float, Fraction or Decimal.")
