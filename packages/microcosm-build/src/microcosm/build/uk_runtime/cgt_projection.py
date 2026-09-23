"""Project the frame's capital gains against the frozen exempt amount (#970).

policyengine-uk carries ``capital_gains`` forward from the build period by
the OBR per-capita GDP growth path while the annual exempt amount stays at
its legislated nominal value, so a gainer at or just below the exempt amount
in the build period becomes a CGT taxpayer in a later projected year. The
calibration seam's ``cgt_projection_entrants`` gate counts those entrants
year by year and fences them against a published band count; this module
supplies the projection the gate needs, read from the installed engine at
January-first instants (the donor-uprating precedent) and reported in full
so the receipt is reproducible from the parameter tree alone.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from importlib import metadata
from types import MappingProxyType

__all__ = [
    "UK_CGT_EXEMPT_AMOUNT_PARAMETER",
    "UK_CGT_GAINS_GROWTH_PARAMETER",
    "UK_CGT_PROJECTION_ARTIFACT_KEY",
    "UK_CGT_PROJECTION_INSTANT_RULE",
    "UK_CGT_PROJECTION_PINS_ENGINE",
    "UKCGTProjection",
    "uk_cgt_projection",
    "uk_cgt_projection_from_pins",
]

#: The engine's uprating index for ``capital_gains`` (per-capita GDP growth).
UK_CGT_GAINS_GROWTH_PARAMETER = "gov.economic_assumptions.yoy_growth.obr.per_capita.gdp"
#: The legislated annual exempt amount, flat in nominal terms from 2024.
UK_CGT_EXEMPT_AMOUNT_PARAMETER = "gov.hmrc.cgt.annual_exempt_amount"
#: The evidence-context artifact key the seam supplies and the binding reads.
UK_CGT_PROJECTION_ARTIFACT_KEY = "cgt_projection"
#: Parameters are read at 1 January of each year, as the donor uprating does.
UK_CGT_PROJECTION_INSTANT_RULE = "january_first"
#: The engine label of a projection built from the manifest pins because no
#: engine is installed; the binding still drift-checks it against the pins.
UK_CGT_PROJECTION_PINS_ENGINE = "manifest_pins (policyengine-uk unavailable)"

ParameterReader = Callable[[str, int], float]


@dataclass(frozen=True)
class UKCGTProjection:
    """Growth factors and exempt amounts from the build period to a horizon.

    ``cumulative_gains_factor_by_year[y]`` multiplies build-period gains to
    reach year ``y``: the product of ``1 + g`` over the years after the base
    up to and including ``y``. ``exempt_amount_by_year`` covers the base year
    and every projected year. Keys are years as strings so the payload is
    JSON-stable.
    """

    base_year: int
    horizon_year: int
    growth_parameter: str
    exempt_amount_parameter: str
    yoy_growth_by_year: Mapping[str, float]
    cumulative_gains_factor_by_year: Mapping[str, float]
    exempt_amount_by_year: Mapping[str, float]
    engine: str
    instant_rule: str = UK_CGT_PROJECTION_INSTANT_RULE

    def __post_init__(self) -> None:
        if self.horizon_year <= self.base_year:
            raise ValueError("The CGT projection horizon must follow the base year.")
        years = [str(year) for year in range(self.base_year + 1, self.horizon_year + 1)]
        if list(self.yoy_growth_by_year) != years:
            raise ValueError("yoy_growth_by_year must cover every projected year.")
        if list(self.cumulative_gains_factor_by_year) != years:
            raise ValueError(
                "cumulative_gains_factor_by_year must cover every projected year."
            )
        if list(self.exempt_amount_by_year) != [str(self.base_year), *years]:
            raise ValueError(
                "exempt_amount_by_year must cover the base year and every "
                "projected year."
            )
        for mapping, label in (
            (self.yoy_growth_by_year, "growth"),
            (self.cumulative_gains_factor_by_year, "cumulative factor"),
            (self.exempt_amount_by_year, "exempt amount"),
        ):
            for year, value in mapping.items():
                if not isinstance(value, int | float) or not math.isfinite(
                    float(value)
                ):
                    raise ValueError(
                        f"CGT projection {label} for {year} is not finite."
                    )
        if any(float(value) <= -1.0 for value in self.yoy_growth_by_year.values()):
            raise ValueError("CGT projection growth must exceed -100 percent.")
        if any(float(value) <= 0.0 for value in self.exempt_amount_by_year.values()):
            raise ValueError("CGT projection exempt amounts must be positive.")
        object.__setattr__(
            self, "yoy_growth_by_year", MappingProxyType(dict(self.yoy_growth_by_year))
        )
        object.__setattr__(
            self,
            "cumulative_gains_factor_by_year",
            MappingProxyType(dict(self.cumulative_gains_factor_by_year)),
        )
        object.__setattr__(
            self,
            "exempt_amount_by_year",
            MappingProxyType(dict(self.exempt_amount_by_year)),
        )

    @property
    def projected_years(self) -> tuple[int, ...]:
        return tuple(range(self.base_year + 1, self.horizon_year + 1))

    def payload(self) -> dict[str, object]:
        """A JSON-stable description of the projection for receipts."""

        return {
            "base_year": self.base_year,
            "horizon_year": self.horizon_year,
            "growth_parameter": self.growth_parameter,
            "exempt_amount_parameter": self.exempt_amount_parameter,
            "instant_rule": self.instant_rule,
            "engine": self.engine,
            "yoy_growth_by_year": dict(self.yoy_growth_by_year),
            "cumulative_gains_factor_by_year": dict(
                self.cumulative_gains_factor_by_year
            ),
            "exempt_amount_by_year": dict(self.exempt_amount_by_year),
        }


def _installed_engine() -> str:
    try:
        return f"policyengine-uk=={metadata.version('policyengine-uk')}"
    except metadata.PackageNotFoundError:  # pragma: no cover - engine extra absent
        return "policyengine-uk (version unavailable)"


def uk_cgt_projection(
    base_year: int,
    horizon_year: int,
    *,
    growth_parameter: str = UK_CGT_GAINS_GROWTH_PARAMETER,
    exempt_amount_parameter: str = UK_CGT_EXEMPT_AMOUNT_PARAMETER,
    parameter_reader: ParameterReader | None = None,
    engine_label: str | None = None,
) -> UKCGTProjection:
    """Read the growth path and exempt amounts from the engine (or a reader).

    Without ``parameter_reader`` the installed policyengine-uk tree is read
    through :func:`microcosm.build.uk_runtime.donor_uprating.engine_parameter_reader`
    at 1 January of each year, which is how the donor uprating reads the same
    tree. A non-finite growth rate, a rate at or below -100 percent or a
    non-positive exempt amount fails closed.
    """

    base_year = int(base_year)
    horizon_year = int(horizon_year)
    if horizon_year <= base_year:
        raise ValueError(
            f"The CGT projection horizon {horizon_year} must follow the base "
            f"year {base_year}."
        )
    if parameter_reader is None:
        from microcosm.build.uk_runtime.donor_uprating import engine_parameter_reader

        reader = engine_parameter_reader()
        engine = _installed_engine()
    else:
        reader = parameter_reader
        engine = engine_label or "supplied_parameter_reader"
    growth: dict[str, float] = {}
    cumulative: dict[str, float] = {}
    exempt: dict[str, float] = {
        str(base_year): float(reader(exempt_amount_parameter, base_year))
    }
    factor = 1.0
    for year in range(base_year + 1, horizon_year + 1):
        rate = float(reader(growth_parameter, year))
        if not math.isfinite(rate) or rate <= -1.0:
            raise ValueError(
                f"CGT projection growth {growth_parameter!r} for {year} is "
                f"unusable: {rate!r}."
            )
        factor *= 1.0 + rate
        growth[str(year)] = rate
        cumulative[str(year)] = factor
        exempt[str(year)] = float(reader(exempt_amount_parameter, year))
    for year, amount in exempt.items():
        if not math.isfinite(amount) or amount <= 0.0:
            raise ValueError(
                f"CGT projection exempt amount {exempt_amount_parameter!r} for "
                f"{year} is unusable: {amount!r}."
            )
    return UKCGTProjection(
        base_year=base_year,
        horizon_year=horizon_year,
        growth_parameter=growth_parameter,
        exempt_amount_parameter=exempt_amount_parameter,
        yoy_growth_by_year=growth,
        cumulative_gains_factor_by_year=cumulative,
        exempt_amount_by_year=exempt,
        engine=engine,
    )


def uk_cgt_projection_from_pins(
    base_year: int,
    horizon_year: int,
    *,
    growth_by_year: Mapping[str, float],
    exempt_amount_by_year: Mapping[str, float],
    growth_parameter: str = UK_CGT_GAINS_GROWTH_PARAMETER,
    exempt_amount_parameter: str = UK_CGT_EXEMPT_AMOUNT_PARAMETER,
) -> UKCGTProjection:
    """The projection the manifest pins describe, for an engine-free build.

    The seam reads the installed engine when there is one and drift-checks it
    against the pins; without an engine (the secrets-free fast lane, a
    data-only build) the pins themselves are the reviewed statement of the
    engine's uprating, and the receipt names that source. A year the pins do
    not cover fails closed here rather than at the gate.
    """

    def read(path: str, year: int) -> float:
        table = (
            growth_by_year
            if path == growth_parameter
            else exempt_amount_by_year
            if path == exempt_amount_parameter
            else None
        )
        if table is None or str(year) not in table:
            raise ValueError(
                f"The CGT projection pins carry no {path!r} value for {year}."
            )
        return float(table[str(year)])

    return uk_cgt_projection(
        base_year,
        horizon_year,
        growth_parameter=growth_parameter,
        exempt_amount_parameter=exempt_amount_parameter,
        parameter_reader=read,
        engine_label=UK_CGT_PROJECTION_PINS_ENGINE,
    )
