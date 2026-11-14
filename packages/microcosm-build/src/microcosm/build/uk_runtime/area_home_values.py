"""The WAS chain's home values off each owner's constituency, G8 (microcosm#931).

Form F (C-15). After segment 2 of the WAS chain, each England and Wales owner
with a positive drawn main residence ``v2`` reads its assigned constituency's
knots (``high_value_home_value_knots.json``) at its weighted mid-rank within
its FRS region, divided by the factor ``regional_property_uprating`` computes
on ``v2``; that stage, declaring the kept ``v2``
(``main_residence_value_rank_basis``) its denominator, returns it to the knot
value to one rounding, and a home read at the knots' cap to the cap or below.
A region without a factor keeps the knot value, as that stage leaves it. Every other row keeps ``v2`` bit for bit, and the release
boundary drops the kept column. Ties break on household-id keys over a fixed
stream with seed source NONE (C-43). The evidence counts clone 0's distinct
true sources, rounded or banded, with no value, sum or factor (C-16, C-44);
the module's own refusals name columns and declared constants only. It reads
the areas the spine's area stages assign; the packaged WAS stage declares it,
and the regional property uprating the kept draw as its denominator
(:func:`area_home_values_declared`), in place of the retired
``regional_home_values`` stage (microcosm#931).
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from importlib.resources import files
from types import MappingProxyType
from typing import Any

import numpy as np
import pandas as pd

from microcosm.build.source_manifest import SourceStageSpec
from microcosm.build.uk_runtime.geographic_clones import (
    geographic_clone_count,
    ladder_clone_index_column,
)
from microcosm.build.uk_runtime.property_localisation import (
    RANK_BASIS_COLUMN,
    ValueKnots,
    _england_wales,
    load_value_knots,
    localise_main_residence_values,
    owner_mask,
    owner_uniforms,
)
from microcosm.build.uk_runtime.receipt_counts import (
    distinct_count,
    rounded_count,
    source_household_ids,
    support_band,
)
from microcosm.build.uk_runtime.regional_uprating import (
    UK_REGIONAL_LAND_VALUES_RESOURCE,
    regional_reference_factors,
)
from microcosm.build.uk_runtime.rowwise_geography import (
    AREA_TYPE_TO_CROSSWALK_COLUMN,
    FRS_REGION_TO_REGION_CODE,
)
from microcosm.build.uk_runtime.spi_support import support_channel_column
from microcosm.frame import Frame

__all__ = [
    "AREA_HOME_VALUES_EXEMPT_COLUMNS",
    "AREA_HOME_VALUES_INPUTS",
    "AREA_HOME_VALUES_OPERATION_KIND",
    "AREA_HOME_VALUES_STREAM",
    "FRS_REGION_TO_KNOTS_REGION",
    "KNOTS_ARTIFACT_ROLE",
    "KNOTS_LEVEL",
    "KNOTS_RESOURCE",
    "METHOD_VERSION",
    "REGIONAL_REFERENCE_ARTIFACT_ROLE",
    "UKAreaHomeValues",
    "UKAreaHomeValuesLocaliser",
    "UKAreaHomeValuesResources",
    "area_home_values_declared",
    "area_home_values_operation_parameters",
    "clone_zero_rows",
    "load_area_home_values_resources",
    "load_committed_area_home_values_resources",
    "localise_area_home_values",
    "support_clip_exemptions",
]

#: The operation the WAS stage declares to localise its home values.
AREA_HOME_VALUES_OPERATION_KIND = "localise_main_residence_value"
#: The keyed stream the rank's tie-breaks draw on: seed source NONE and a
#: fixed stream, as ruled for ``regional_home_values`` on 9 Nov (C-43). It is
#: this step's own; the stage's operation declares it, so its contract hash
#: covers it.
AREA_HOME_VALUES_STREAM = (
    "sha256-u53-v1",
    "uk-spine-area-home-values-v1",
    0,
    0,
)
METHOD_VERSION = "uk-spine-area-home-values-v1"
#: The two artifacts the WAS stage declares for this step (role, resource):
#: the constituency knots and the regional reference the factor reads.
KNOTS_ARTIFACT_ROLE = "value_knots"
KNOTS_RESOURCE = "high_value_home_value_knots.json"
REGIONAL_REFERENCE_ARTIFACT_ROLE = "regional_reference"
_ARTIFACTS = (
    (KNOTS_ARTIFACT_ROLE, KNOTS_RESOURCE),
    (REGIONAL_REFERENCE_ARTIFACT_ROLE, UK_REGIONAL_LAND_VALUES_RESOURCE),
)
#: The knots' level the values are read at.
KNOTS_LEVEL = "constituency"
#: The knots' region codes for the FRS regions they cover. Wales is the
#: country code the knots use, not the placeholder region code the area
#: assignment keeps for it (``W99999999``). The evidence counts by these
#: codes, which the home-value ranking gate declares.
FRS_REGION_TO_KNOTS_REGION: Mapping[str, str] = MappingProxyType(
    {
        **{
            region: code
            for region, code in FRS_REGION_TO_REGION_CODE.items()
            if code.startswith("E12")
        },
        "WALES": "W92000004",
    }
)
#: The WAS columns the support clip leaves to this step: a localised home can
#: pass the donor's largest main residence, and the property total with it
#: (C-18). The WAS stage's support clip declares them exempt.
AREA_HOME_VALUES_EXEMPT_COLUMNS = ("main_residence_value", "property_wealth")
_CONSTITUENCY_COLUMN = AREA_TYPE_TO_CROSSWALK_COLUMN["constituency"]
_CHANNEL_COLUMN = support_channel_column("household")
#: The household columns the step reads, besides ``source_household_id``,
#: the true source its evidence counts by, the support channel the factor's
#: mean leaves SPI rows out of, and the clone index the evidence counts clone
#: 0 by, where the frame carries one.
AREA_HOME_VALUES_INPUTS = (
    "household_id",
    "region",
    "tenure_type",
    _CONSTITUENCY_COLUMN,
)
_CLONE_INDEX_COLUMN = ladder_clone_index_column("household")
_WHAT = "The area home values"


def area_home_values_operation_parameters() -> dict[str, object]:
    """The reviewed operation the WAS stage must declare verbatim.

    The draw it reads and where, the rows it rewrites, the rank and its
    stream, the knots and their level, the value and the factor it is divided
    by, the region map its evidence counts by, the rank basis it keeps and the
    rows it leaves.
    """

    return {
        "value_column": "main_residence_value",
        "after_segment": 2,
        "rows": (
            "owner households (tenure owned outright or with a mortgage) with a "
            "positive drawn main_residence_value in the FRS regions of England "
            "and Wales"
        ),
        "rank": (
            "weighted mid-rank of the drawn main_residence_value among the owner "
            "households of the FRS region with a positive draw, both support "
            "channels, at the stage's household weights, ties broken by a draw "
            "keyed on the household id"
        ),
        "stream": list(AREA_HOME_VALUES_STREAM),
        "knots": KNOTS_RESOURCE,
        "knots_level": KNOTS_LEVEL,
        "constituency_column": _CONSTITUENCY_COLUMN,
        "regional_reference": UK_REGIONAL_LAND_VALUES_RESOURCE,
        "value": (
            "np.interp(rank, grid, the assigned constituency's knots) divided by "
            "the FRS region's factor, a home at the knots' cap taking the largest "
            "quotient the factor multiplies back to the cap or below"
        ),
        "factor": (
            "avg_house_price / unweighted_mean(drawn main_residence_value | "
            "region, FRS-base owners, >0), the regional property uprating's "
            "factor on the draw"
        ),
        "region_map": dict(sorted(FRS_REGION_TO_KNOTS_REGION.items())),
        "rank_basis_column": RANK_BASIS_COLUMN,
        "kept": (
            "Scotland, Northern Ireland, renters and owners without a positive "
            "draw keep the draw"
        ),
        "method": METHOD_VERSION,
    }


def support_clip_exemptions(stage: SourceStageSpec) -> frozenset[str]:
    """The columns the WAS stage's ``support_clip`` declares exempt.

    The declared ``exempt`` list, as ``etb_services.support_clip_exempt``
    reads its stage's: none, unless the stage declares the area home values,
    whose two columns :func:`area_home_values_declared` holds it to.
    """

    exempt: set[str] = set()
    for operation in stage.operations:
        if operation.kind == "support_clip":
            declared = operation.parameters.get("exempt")
            if declared:
                exempt |= {str(column) for column in declared}
    return frozenset(exempt)


def area_home_values_declared(stage: SourceStageSpec) -> bool:
    """Whether the WAS stage declares this step, held to its reviewed form.

    Declared, the stage carries :data:`AREA_HOME_VALUES_OPERATION_KIND` once,
    with :func:`area_home_values_operation_parameters` verbatim; the knots and
    the regional reference as artifacts, each with
    ``runtime_sha256_required``; ``main_residence_value_rank_basis`` among its
    outputs; and its support clip exempting exactly
    :data:`AREA_HOME_VALUES_EXEMPT_COLUMNS`. Undeclared, it carries neither
    the output nor a support clip exemption. Anything between is refused,
    since a half declaration would clip a localised home back to the donor's
    range or drop the rank basis the regional uprating divides by.
    """

    operations = [
        operation
        for operation in stage.operations
        if operation.kind == AREA_HOME_VALUES_OPERATION_KIND
    ]
    exempt = support_clip_exemptions(stage)
    if not operations:
        if RANK_BASIS_COLUMN in stage.outputs or exempt:
            raise ValueError(
                f"The WAS stage declares {RANK_BASIS_COLUMN!r} or a support clip "
                f"exemption without {AREA_HOME_VALUES_OPERATION_KIND!r}; both "
                "belong to the area home values (microcosm#931)."
            )
        return False
    if len(operations) != 1:
        raise ValueError(
            f"The WAS stage declares {AREA_HOME_VALUES_OPERATION_KIND!r} more "
            "than once."
        )
    if dict(operations[0].parameters) != area_home_values_operation_parameters():
        raise ValueError(
            f"The WAS stage's {AREA_HOME_VALUES_OPERATION_KIND!r} differs from the "
            "reviewed operation area_home_values states (microcosm#931)."
        )
    declared = sorted(
        (str(artifact.get("role")), str(artifact.get("resource")))
        for artifact in stage.artifacts
        if artifact.get("runtime_sha256_required") is True
        and (str(artifact.get("role")), str(artifact.get("resource"))) in _ARTIFACTS
    )
    if declared != sorted(_ARTIFACTS):
        raise ValueError(
            f"The area home values read the artifacts {list(_ARTIFACTS)} (role, "
            "resource), each once with runtime_sha256_required: true."
        )
    if RANK_BASIS_COLUMN not in stage.outputs:
        raise ValueError(
            f"The area home values keep the draw as {RANK_BASIS_COLUMN!r}, which "
            "the WAS stage must declare among its outputs."
        )
    if exempt != set(AREA_HOME_VALUES_EXEMPT_COLUMNS):
        raise ValueError(
            "The area home values need the WAS support clip to exempt exactly "
            f"{list(AREA_HOME_VALUES_EXEMPT_COLUMNS)}, which a localised home "
            "can lift past the donor's range (C-18)."
        )
    return True


@dataclass(frozen=True)
class UKAreaHomeValuesResources:
    """The constituency knots and the regional reference, parsed, with their
    bytes' SHA-256."""

    knots: ValueKnots = field(repr=False)
    regional_reference: Mapping[str, Any] = field(repr=False)
    knots_sha256: str
    regional_reference_sha256: str


def load_area_home_values_resources(
    *, knots: bytes, regional_reference: bytes
) -> UKAreaHomeValuesResources:
    """The two resources from their bytes: the knots through the property
    loader, the regional reference as the regional uprating reads it."""

    return UKAreaHomeValuesResources(
        knots=load_value_knots(knots),
        regional_reference=json.loads(regional_reference.decode("utf-8")),
        knots_sha256=hashlib.sha256(knots).hexdigest(),
        regional_reference_sha256=hashlib.sha256(regional_reference).hexdigest(),
    )


def load_committed_area_home_values_resources() -> UKAreaHomeValuesResources:
    """The committed knots and regional reference the stage declares."""

    package = files("microcosm.build.uk")
    return load_area_home_values_resources(
        knots=package.joinpath(KNOTS_RESOURCE).read_bytes(),
        regional_reference=package.joinpath(
            UK_REGIONAL_LAND_VALUES_RESOURCE
        ).read_bytes(),
    )


@dataclass(frozen=True)
class UKAreaHomeValues:
    """The step on one household table: every row's value and rank basis, the
    rows rewritten, and what the evidence counts among clone-0 rows. The
    source counts are exact and print only rounded or banded; nothing here
    prints in the repr but the clone count, the knots' basis year and the
    digests."""

    value: np.ndarray = field(repr=False)
    rank_basis: np.ndarray = field(repr=False)
    rewritten: np.ndarray = field(repr=False)
    clone_count: int
    sources_rewritten_by_region: Mapping[str, int] = field(repr=False)
    sources_at_cap: int = field(repr=False)
    ranking: str = field(repr=False)
    knots_basis_year: int
    knots_sha256: str
    regional_reference_sha256: str

    def evidence(self) -> dict[str, object]:
        """The step's evidence block (C-16): rounded and banded counts of
        clone-0 sources, digests and declared constants; no value, sum or
        factor."""

        return {
            "clone_count": int(self.clone_count),
            "counted_among": "clone_0",
            "sources_rewritten_by_region": {
                region: rounded_count(count)
                for region, count in sorted(self.sources_rewritten_by_region.items())
            },
            "sources_at_cap": support_band(self.sources_at_cap),
            "knots_basis_year": int(self.knots_basis_year),
            "knots_sha256": self.knots_sha256,
            "knots_level": KNOTS_LEVEL,
            "regional_reference_sha256": self.regional_reference_sha256,
            "ranking": {"rank_basis": self.ranking},
            "rank_basis_column": RANK_BASIS_COLUMN,
            "stream": list(AREA_HOME_VALUES_STREAM),
            "method": METHOD_VERSION,
        }


def _require_columns(household: pd.DataFrame) -> None:
    missing = [
        column
        for column in (*AREA_HOME_VALUES_INPUTS, "source_household_id")
        if column not in household.columns
    ]
    if missing:
        raise ValueError(
            f"{_WHAT} read household {missing}, which the earlier spine stages "
            "write; the frame lacks them (microcosm#931)."
        )


def clone_zero_rows(household: pd.DataFrame) -> np.ndarray:
    """Whether each household row is geographic clone 0's: its
    ``household_clone_index`` is 0, or the table carries no index (one
    clone). A count taken among these rows does not grow with the clone count
    (C-44); the WAS stage's donor-range cap counts by it too."""

    if _CLONE_INDEX_COLUMN not in household.columns:
        return np.ones(len(household), dtype=bool)
    return household[_CLONE_INDEX_COLUMN].to_numpy(dtype=np.int64) == 0


def localise_area_home_values(
    household: pd.DataFrame,
    weights: np.ndarray,
    value: np.ndarray,
    *,
    resources: UKAreaHomeValuesResources,
) -> UKAreaHomeValues:
    """Every household row's main residence, England and Wales owners'
    localised (G8, form F).

    ``value`` is the WAS chain's segment-2 draw in ``household``'s row order,
    ``weights`` the stage's household weights in that order. Each England and
    Wales owner with a positive draw reads its assigned constituency's knots
    at its weighted mid-rank within its FRS region, divided by its region's
    factor on the draw; a home read at the knots' cap takes the largest
    quotient the factor multiplies back to the cap or below. Every other row
    keeps its draw bit for bit. A row's
    value depends on its own key and its region's pooled ranks and factor,
    never on the table's row order.

    The evidence counts a source among clone 0's rows of positive weight: a
    row of zero weight carries no localised value into a weighted figure
    (``receipt_counts.distinct_count``, microcosm#1262).

    Refused, by name: a missing input column; weights or draws that do not
    align or are not finite; an England and Wales owner with a positive draw
    whose ``constituency_code`` is not in England and Wales, or another owner
    with a positive draw whose code is; a constituency without knots; a
    localised value that is not positive; and the refusals of the rank and the
    source reader.
    """

    _require_columns(household)
    drawn = np.asarray(value, dtype=float)
    weight = np.asarray(weights, dtype=float)
    if drawn.shape != (len(household),) or weight.shape != (len(household),):
        raise ValueError(
            f"{_WHAT} read one drawn value and one weight per household row."
        )
    if not np.isfinite(drawn).all():
        raise ValueError(f"{_WHAT} read finite drawn main residence values.")
    sources = source_household_ids(household, stage=_WHAT)
    region = household["region"].astype(str).to_numpy()
    tenure = household["tenure_type"].to_numpy()
    codes = np.asarray(household[_CONSTITUENCY_COLUMN].to_numpy(dtype=object)).astype(
        str
    )
    in_england_wales = np.isin(region, list(FRS_REGION_TO_KNOTS_REGION))
    owners = owner_mask(tenure) & (drawn > 0)
    if not np.array_equal(owners & in_england_wales, owners & _england_wales(codes)):
        raise ValueError(
            f"{_WHAT} read an owner's {_CONSTITUENCY_COLUMN!r} in its FRS region's "
            "nation: every owner with a drawn home value in England and Wales "
            "needs a constituency there, and no other owner may hold one; the "
            "area draw places each household in its own FRS region "
            "(microcosm#931)."
        )
    rank, receipt = owner_uniforms(
        spine_key=household["household_id"].astype(str).to_numpy(dtype=object),
        region=region,
        tenure_type=tenure,
        value=drawn,
        weight=weight,
        stream=AREA_HOME_VALUES_STREAM,
    )
    knot_value, rewritten = localise_main_residence_values(
        u=rank,
        constituency_code=codes,
        tenure_type=tenure,
        value=drawn,
        knots=resources.knots,
    )
    if np.any(knot_value[rewritten] <= 0):
        raise ValueError(
            f"{_WHAT} read a knot value that is not positive: the regional "
            "uprating's owners are the rows with a positive main residence, so "
            "every owner the step ranks must keep one (microcosm#931)."
        )
    factor_table = household.loc[
        :, [column for column in ("region", _CHANNEL_COLUMN) if column in household]
    ].copy()
    factor_table["main_residence_value"] = drawn
    factors = regional_reference_factors(factor_table, resources.regional_reference)
    localised = knot_value.copy()
    at_knots_cap = knot_value >= resources.knots.cap
    for name, factor in factors.items():
        rows = rewritten & (region == name)
        quotient = knot_value[rows] / factor
        # The regional uprating multiplies the quotient back by this factor.
        # A home read at the knots' cap takes the largest quotient whose
        # product does not pass the cap, so the uprating never returns it one
        # rounding above uk_support's price-paid bound (microcosm#931).
        capped = at_knots_cap[rows]
        over = capped & (quotient * factor > knot_value[rows])
        while over.any():
            quotient[over] = np.nextafter(quotient[over], 0.0)
            over = capped & (quotient * factor > knot_value[rows])
        localised[rows] = quotient
    if not np.isfinite(localised).all() or np.any(localised[rewritten] <= 0):
        raise ValueError(f"{_WHAT} must be finite and positive where they localise.")
    counted = clone_zero_rows(household)
    knots_region = np.asarray(
        [FRS_REGION_TO_KNOTS_REGION.get(name, "") for name in region], dtype=object
    )
    rewritten_by_region = {
        str(code): distinct_count(
            sources, rewritten & counted & (knots_region == code), weight
        )
        for code in sorted(set(knots_region[rewritten & counted]))
    }
    at_cap = distinct_count(
        sources, rewritten & counted & (knot_value >= resources.knots.cap), weight
    )
    return UKAreaHomeValues(
        value=localised,
        rank_basis=drawn.copy(),
        rewritten=rewritten,
        clone_count=geographic_clone_count(household, stage=_WHAT),
        sources_rewritten_by_region=MappingProxyType(rewritten_by_region),
        sources_at_cap=int(at_cap),
        ranking=str(receipt["rank_basis"]),
        knots_basis_year=int(resources.knots.basis_year),
        knots_sha256=resources.knots_sha256,
        regional_reference_sha256=resources.regional_reference_sha256,
    )


@dataclass(frozen=True)
class UKAreaHomeValuesLocaliser:
    """The step bound to one WAS stage frame, for ``impute_was_wealth``.

    Built from the frame the WAS stage runs on, before its chain draws:
    the household columns the step reads and the stage's household weights.
    Called with the segment-2 draw, it returns
    :func:`localise_area_home_values`' result.
    """

    household: pd.DataFrame = field(repr=False)
    weights: np.ndarray = field(repr=False)
    resources: UKAreaHomeValuesResources = field(repr=False)

    @classmethod
    def from_frame(
        cls, frame: Frame, *, resources: UKAreaHomeValuesResources
    ) -> UKAreaHomeValuesLocaliser:
        if not isinstance(frame, Frame):
            raise TypeError(f"{_WHAT} read the households of a Frame.")
        household = frame.table("household")
        _require_columns(household)
        columns = [
            column
            for column in (
                *AREA_HOME_VALUES_INPUTS,
                "source_household_id",
                _CHANNEL_COLUMN,
                _CLONE_INDEX_COLUMN,
            )
            if column in household.columns
        ]
        return cls(
            household=household.loc[:, columns].copy(),
            weights=np.asarray(frame.weights_for("household").values, dtype=float),
            resources=resources,
        )

    def __call__(self, value: np.ndarray) -> UKAreaHomeValues:
        return localise_area_home_values(
            self.household, self.weights, value, resources=self.resources
        )
