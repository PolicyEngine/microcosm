"""Declared estimators for an open top band's tail shape.

``grouped_pareto_mle@1`` fits a Pareto alpha by maximum likelihood to grouped
counts: contiguous anchor bands ``(a_k, b_k]`` with counts ``n_k`` and the
open top above ``c`` with count ``n_T``. With ``L0 = a_1`` and
``S(x) = (x / L0) ** -alpha``, band ``k`` has probability
``p_k = S(a_k) - S(b_k)`` and the top ``p_T = S(c)``, so the log likelihood
is ``l(alpha) = sum n_k ln p_k + n_T ln p_T``.

Writing ``u_k = ln(a_k / L0)``, ``d_k = ln(b_k / a_k)`` and ``t = ln(c / L0)``,
``ln p_k = -alpha u_k + ln(1 - exp(-alpha d_k))`` and ``ln p_T = -alpha t``.
Each term is concave in alpha, because ``ln(1 - exp(-alpha d))`` has second
derivative ``-d**2 exp(alpha d) / (exp(alpha d) - 1)**2 < 0``. So

``l'(alpha) = sum n_k (d_k / expm1(alpha d_k) - u_k) - n_T t``

is strictly decreasing and has at most one root. The estimator bisects a
declared bracket for a declared number of iterations, rounds the midpoint
once to six decimals (half-even) and carries alpha as ``p / 10**6``. It
refuses, by name, a bracket on which ``l'`` does not change sign and any
alpha at or below one, where the top band's mean ``alpha c / (alpha - 1)``
is infinite. Under the r2c design's bracket ``[1 + 2**-20, 64]`` the
rounded alpha is at least 1.000001, so ``alpha_not_above_one`` cannot fire
there: heavy-tail data whose root lies at or below one is refused as
``bracket_without_sign_change`` instead, and the refusal names that reason.

The estimator reads no country data: the caller passes the anchors, their
counts and the declared bracket and iteration count.
"""

from __future__ import annotations

import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from decimal import Decimal
from fractions import Fraction

__all__ = [
    "ALPHA_DECIMALS",
    "GROUPED_PARETO_MLE_REF",
    "MAX_BISECTION_ITERATIONS",
    "ParetoAnchor",
    "ParetoFit",
    "TailEstimatorRefusalError",
    "grouped_pareto_mle",
    "grouped_pareto_score",
    "parse_grouped_pareto_declaration",
]

GROUPED_PARETO_MLE_REF = "grouped_pareto_mle@1"
#: Alpha is rounded once to this many decimals and carried as p / 10**6.
ALPHA_DECIMALS = 6
#: Float bisection stops narrowing after about 64 halvings; more is refused
#: so a declaration cannot claim precision the arithmetic cannot give.
MAX_BISECTION_ITERATIONS = 200

_DECIMAL_TEXT = re.compile(r"^(0|[1-9][0-9]*)(\.[0-9]+)?$")


class TailEstimatorRefusalError(ValueError):
    """A named refusal from a tail estimator; ``code`` names the reason."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(f"{GROUPED_PARETO_MLE_REF} refuses {code}: {message}")
        self.code = code


@dataclass(frozen=True)
class ParetoAnchor:
    """One anchor band: ``(lower, upper]`` or the open top when ``upper`` is None."""

    reference: str
    lower: Fraction
    upper: Fraction | None
    count: float


@dataclass(frozen=True)
class ParetoFit:
    """The fitted alpha and everything its receipt records."""

    numerator: int
    denominator: int
    anchors: tuple[ParetoAnchor, ...]
    bracket: tuple[Fraction, Fraction]
    iterations: int
    final_bracket: tuple[float, float]
    score_at_final_bracket: tuple[float, float]
    midpoint: float

    @property
    def alpha(self) -> Fraction:
        return Fraction(self.numerator, self.denominator)

    @property
    def alpha_float(self) -> float:
        """The binary64 value nearest ``p / 10**6``, as ``float(text)`` gives."""
        return self.numerator / self.denominator

    @property
    def top_lower(self) -> Fraction:
        return self.anchors[-1].lower

    @property
    def top_band_mean(self) -> Fraction:
        """``alpha c / (alpha - 1)``, exact."""
        alpha = self.alpha
        return alpha * self.top_lower / (alpha - 1)

    def receipt(self) -> dict[str, object]:
        """The deterministic receipt the design lists (r2c design section 2)."""

        def text(value: Fraction) -> str:
            return _fraction_text(value)

        return {
            "estimator": GROUPED_PARETO_MLE_REF,
            "estimator_id": GROUPED_PARETO_MLE_REF.split("@")[0],
            "estimator_version": int(GROUPED_PARETO_MLE_REF.split("@")[1]),
            "anchor": text(self.anchors[0].lower),
            "anchors": [
                {
                    "reference": anchor.reference,
                    "lower": text(anchor.lower),
                    "upper": None if anchor.upper is None else text(anchor.upper),
                    "count": anchor.count,
                }
                for anchor in self.anchors
            ],
            "bracket": [text(self.bracket[0]), text(self.bracket[1])],
            "iterations": self.iterations,
            "final_bracket": list(self.final_bracket),
            "score_at_final_bracket": list(self.score_at_final_bracket),
            "alpha": {"numerator": self.numerator, "denominator": self.denominator},
            "alpha_decimal": text(self.alpha),
            "top_band_lower": text(self.top_lower),
            "top_band_mean": {
                "numerator": self.top_band_mean.numerator,
                "denominator": self.top_band_mean.denominator,
            },
            "top_band_mean_decimal": str(
                _round_half_even(self.top_band_mean, ALPHA_DECIMALS)
            ),
        }


def _fraction_text(value: Fraction) -> str:
    """Exact decimal text when one exists, else ``p/q``."""
    denominator = value.denominator
    for prime in (2, 5):
        while denominator % prime == 0:
            denominator //= prime
    if denominator != 1:
        return f"{value.numerator}/{value.denominator}"
    decimal = Decimal(value.numerator) / Decimal(value.denominator)
    # Exact: the denominator divides a power of ten.
    text = format(decimal.normalize(), "f")
    return text


def _round_half_even(value: Fraction, places: int) -> Decimal:
    quantum = Decimal(1).scaleb(-places)
    scaled = value * 10**places
    floor = scaled.numerator // scaled.denominator
    remainder = scaled - floor
    half = Fraction(1, 2)
    if remainder > half or (remainder == half and floor % 2 == 1):
        floor += 1
    return (Decimal(floor) * quantum).quantize(quantum)


def _decimal_fraction(value: object, name: str) -> Fraction:
    if not isinstance(value, str) or not _DECIMAL_TEXT.match(value):
        raise ValueError(
            f"{GROUPED_PARETO_MLE_REF} {name} must be nonnegative decimal text."
        )
    return Fraction(value)


def _edge(value: object, name: str) -> Fraction:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{GROUPED_PARETO_MLE_REF} {name} must be a number.")
    if not math.isfinite(value) or value <= 0:
        raise ValueError(
            f"{GROUPED_PARETO_MLE_REF} {name} must be finite and positive."
        )
    return Fraction(value)


def parse_grouped_pareto_declaration(
    estimator: Mapping[str, object],
) -> tuple[tuple[str, ...], tuple[Fraction, Fraction], int]:
    """Validate ``{id, anchor_references, bracket, iterations}``.

    Returns the anchor references, the bracket as exact fractions and the
    iteration count. The bracket is two decimal strings, so a declaration
    states an exact rational rather than a binary float.
    """

    if not isinstance(estimator, Mapping) or set(estimator) != {
        "id",
        "anchor_references",
        "bracket",
        "iterations",
    }:
        raise ValueError(
            "A grouped Pareto estimator declares exactly id, anchor_references, "
            "bracket and iterations."
        )
    if estimator["id"] != GROUPED_PARETO_MLE_REF:
        raise ValueError(
            f"The estimator id must be {GROUPED_PARETO_MLE_REF!r}; "
            f"{estimator['id']!r} has no implementation."
        )
    references = estimator["anchor_references"]
    if (
        not isinstance(references, list)
        or len(references) < 2
        or any(not isinstance(item, str) or not item for item in references)
        or len(set(references)) != len(references)
    ):
        raise ValueError(
            f"{GROUPED_PARETO_MLE_REF} anchor_references must list at least one "
            "finite band and the open top, each once."
        )
    bracket = estimator["bracket"]
    if not isinstance(bracket, list) or len(bracket) != 2:
        raise ValueError(f"{GROUPED_PARETO_MLE_REF} bracket must be [low, high].")
    low = _decimal_fraction(bracket[0], "bracket low")
    high = _decimal_fraction(bracket[1], "bracket high")
    if not 0 < low < high:
        raise ValueError(
            f"{GROUPED_PARETO_MLE_REF} bracket must satisfy 0 < low < high."
        )
    iterations = estimator["iterations"]
    if (
        isinstance(iterations, bool)
        or not isinstance(iterations, int)
        or not 1 <= iterations <= MAX_BISECTION_ITERATIONS
    ):
        raise ValueError(
            f"{GROUPED_PARETO_MLE_REF} iterations must be an integer from 1 to "
            f"{MAX_BISECTION_ITERATIONS}."
        )
    return tuple(references), (low, high), iterations


def _anchors(
    bands: Sequence[tuple[str, object, object, object]],
) -> tuple[ParetoAnchor, ...]:
    if len(bands) < 2:
        raise ValueError(
            f"{GROUPED_PARETO_MLE_REF} needs at least one finite band and the open top."
        )
    anchors = []
    for index, (reference, lower, upper, count) in enumerate(bands):
        last = index == len(bands) - 1
        low = _edge(lower, "band lower")
        if last:
            if upper is not None:
                raise ValueError(
                    f"{GROUPED_PARETO_MLE_REF} fits an open top band; "
                    f"{reference!r} declares an upper bound."
                )
            high = None
        else:
            if upper is None:
                raise ValueError(
                    f"{GROUPED_PARETO_MLE_REF}: only the top anchor may be open."
                )
            high = _edge(upper, "band upper")
            if high <= low:
                raise ValueError(
                    f"{GROUPED_PARETO_MLE_REF} band {reference!r} must have "
                    "upper above lower."
                )
        if anchors and anchors[-1].upper != low:
            raise ValueError(
                f"{GROUPED_PARETO_MLE_REF} anchor bands must be contiguous and "
                "increasing."
            )
        if (
            isinstance(count, bool)
            or not isinstance(count, (int, float))
            or not math.isfinite(count)
            or count < 0
        ):
            raise ValueError(
                f"{GROUPED_PARETO_MLE_REF} counts must be finite and nonnegative."
            )
        anchors.append(ParetoAnchor(reference, low, high, float(count)))
    return tuple(anchors)


def _terms(
    anchors: Sequence[ParetoAnchor],
) -> tuple[list[tuple[float, float, float]], float, float]:
    """``(n_k, u_k, d_k)`` per finite band, then ``n_T`` and ``t``."""
    origin = anchors[0].lower
    finite = [
        (
            anchor.count,
            math.log(anchor.lower / origin),
            math.log(anchor.upper / anchor.lower),
        )
        for anchor in anchors[:-1]
    ]
    top = anchors[-1]
    return finite, top.count, math.log(top.lower / origin)


def grouped_pareto_score(anchors: Sequence[ParetoAnchor], alpha: float) -> float:
    """``l'(alpha)`` for the anchors, as the module docstring writes it."""
    finite, top_count, top_offset = _terms(anchors)
    parts = [
        count * (width / math.expm1(alpha * width) - offset)
        for count, offset, width in finite
    ]
    parts.append(-top_count * top_offset)
    return math.fsum(parts)


def grouped_pareto_mle(
    bands: Sequence[tuple[str, object, object, object]],
    *,
    bracket: tuple[Fraction, Fraction],
    iterations: int,
) -> ParetoFit:
    """Fit alpha to ``(reference, lower, upper, count)`` anchor rows.

    The rows are contiguous and increasing; the last is the open top
    (``upper`` is None). ``bracket`` and ``iterations`` come from the node's
    declaration.
    """

    anchors = _anchors(bands)
    low, high = (float(bracket[0]), float(bracket[1]))
    score_low = grouped_pareto_score(anchors, low)
    score_high = grouped_pareto_score(anchors, high)
    if not (score_low > 0 > score_high):
        raise TailEstimatorRefusalError(
            "bracket_without_sign_change",
            f"l'(alpha) is {score_low!r} at {_fraction_text(bracket[0])} and "
            f"{score_high!r} at {_fraction_text(bracket[1])}; the root is not "
            "inside the declared bracket.",
        )
    for _ in range(iterations):
        middle = (low + high) / 2
        score = grouped_pareto_score(anchors, middle)
        if score > 0:
            low, score_low = middle, score
        else:
            high, score_high = middle, score
    midpoint = (low + high) / 2
    rounded = _round_half_even(Fraction(midpoint), ALPHA_DECIMALS)
    denominator = 10**ALPHA_DECIMALS
    numerator = int(rounded.scaleb(ALPHA_DECIMALS))
    if numerator <= denominator:
        raise TailEstimatorRefusalError(
            "alpha_not_above_one",
            f"the fitted alpha {rounded} is at most one, so the open top band "
            "has no finite mean.",
        )
    return ParetoFit(
        numerator=numerator,
        denominator=denominator,
        anchors=anchors,
        bracket=bracket,
        iterations=iterations,
        final_bracket=(low, high),
        score_at_final_bracket=(score_low, score_high),
        midpoint=midpoint,
    )
