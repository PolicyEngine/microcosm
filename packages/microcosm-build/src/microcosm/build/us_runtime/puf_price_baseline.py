"""The developmental CPI-U price-restatement baseline for the 2015 PUF.

One exact positive ratio of two published CPI-U all-items annual averages —
313689/237017, 2015 to 2024 — applied to every known money field of the
thirteen-field source projection. It restates 2015-file purchasing power at
2024 prices. It is **not** income aging: it carries no category-specific
nominal growth, no return-mass transition, and no claim that any field's real
level changed. ``S006`` is declared with an explicit identity series so the
no-mass-transition choice is recorded rather than implied.

Three things this module is, and three it refuses
-------------------------------------------------
*It pins the resource.* The series is ``CUUR0000SA0``, U.S. city average, all
items, not seasonally adjusted, ``M13`` annual average, base 1982-84=100. The
levels are the publisher's own decimal text, copied verbatim, and the factor
table it builds carries the ``developmental_public_resource`` authority: the
retained bytes are checked against their named digest and their two records
against the declared level text. Retrieval location and time remain caller
metadata, not claims proved by hashing. The authority cannot become
release-eligible.

*It declares the recipe.* One ``nominal_price_restatement`` series for all
thirteen money fields, one source reference year, one target year, ``RECID``
as the identifier, ``S006`` as a design weight whose declared factor is
exactly one.

*It adapts the artifact to the money view.* The projection is an artifact and
the growth transform consumes a table, so
:func:`adapt_projection_to_money_view` joins the projection to the reviewed
status artifact on ``RECID``, removes only the disclosure-aggregate rows,
records what it removed, and hands back ``FLPDYR``, the demographic status and
the unaltered integer ``S006`` separately. The growth itself is
:func:`~microcosm.build.us_runtime.puf_growth.apply_puf_growth`; nothing here
multiplies an amount.

It refuses to interpret an aggregate token, to put the out-of-universe
sentinel anywhere near a numeric input, and to declare that this baseline
under- or over-states any 2024 amount. The public universes that would support
such a statement are not matched to this file.
"""

from __future__ import annotations

import hashlib
import json
import math
import struct
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType

import numpy as np
import pandas as pd

from .puf_growth import (
    DESIGN_WEIGHT_FIELD,
    PROVENANCE_RECID_COLUMN,
    PROVENANCE_SOURCE_AGI_COLUMN,
    SOURCE_AGI_FIELD,
    SOURCE_RECID_FIELD,
    CompiledPufGrowth,
    FactorAuthority,
    FieldRole,
    GrownPufTable,
    GrowthFactorTable,
    GrowthKind,
    GrowthRecipe,
    GrowthRule,
    PufGrowthRefusalError,
    SignBranch,
    compile_puf_growth,
)
from .puf_monetary_agi_projection import AGI_FIELD
from .puf_monetary_source import AGGREGATE_SENTINEL, PufMonetarySourceProjection
from .puf_source_agi import PUF_SOURCE_YEAR

__all__ = [
    "CPI_U_ANNUAL_AVERAGE_PERIOD",
    "CPI_U_INDEX_BASE",
    "CPI_U_SERIES_ID",
    "CPI_U_SERIES_TITLE",
    "DESIGN_WEIGHT_IDENTITY_SERIES",
    "PRICE_BASELINE_LEVELS",
    "PRICE_BASELINE_RATIO",
    "PRICE_BASELINE_RECIPE_ID",
    "PRICE_BASELINE_SERIES_NAME",
    "PRICE_BASELINE_SOURCE_YEAR",
    "PRICE_BASELINE_TABLE_ID",
    "PRICE_BASELINE_TARGET_YEAR",
    "PRICE_BASELINE_FACTOR_BITS",
    "PAIRWISE_OBSERVATIONS",
    "PufMoneyViewAdaptation",
    "adapt_projection_to_money_view",
    "compile_price_baseline",
    "price_baseline_factor_document",
    "price_baseline_invariants",
    "price_baseline_recipe",
    "require_price_baseline_contract",
    "verify_cpi_resource_bytes",
]

#: The primary BLS series this baseline is read from.
CPI_U_SERIES_ID = "CUUR0000SA0"
CPI_U_SERIES_TITLE = (
    "All items in U.S. city average, all urban consumers, not seasonally adjusted"
)
CPI_U_ANNUAL_AVERAGE_PERIOD = "M13"
CPI_U_INDEX_BASE = "1982-84=100"

#: The source file's representative year, and the target the restatement
#: states amounts in. The source year is the growth slice's own PUF source
#: year, not a second opinion about it.
PRICE_BASELINE_SOURCE_YEAR = PUF_SOURCE_YEAR
PRICE_BASELINE_TARGET_YEAR = 2024

#: The publisher's own decimal text for the two annual averages. These are
#: level strings, never floats: the factor table stores text and the ratio is
#: computed once, in one place, by the growth contract.
PRICE_BASELINE_LEVELS = MappingProxyType(
    {PRICE_BASELINE_SOURCE_YEAR: "237.017", PRICE_BASELINE_TARGET_YEAR: "313.689"}
)

#: The exact rational factor, as ``(numerator, denominator)``. Both published
#: levels carry exactly three decimals, so the ratio of the thousandths-scaled
#: integers is the exact ratio of the published levels.
PRICE_BASELINE_RATIO = (313689, 237017)

#: The one float64 the adopted factor is, as its little-endian bit pattern.
#: Derived from the exact rational here rather than written down, so it cannot
#: drift from :data:`PRICE_BASELINE_RATIO`; a test separately checks that the
#: division of the two published level *texts* lands on the same double.
#: :func:`require_price_baseline_contract` compares an incoming contract's
#: resolved bits against this, which is the check that makes every downstream
#: statement about "the factor" a statement about the contract in hand.
PRICE_BASELINE_FACTOR_BITS = struct.pack(
    "<d", PRICE_BASELINE_RATIO[0] / PRICE_BASELINE_RATIO[1]
).hex()

PRICE_BASELINE_SERIES_NAME = "bls.cpi_u.all_items.us_city_average.annual_average"

#: The design-weight series. Both levels are exactly one, so the declared
#: factor is exactly 1.0 and "no return-mass transition" is a recorded
#: declaration rather than an omission a later reader has to infer.
DESIGN_WEIGHT_IDENTITY_SERIES = "declared_identity.no_return_mass_transition"

PRICE_BASELINE_TABLE_ID = "us_puf_price_baseline_cpi_u_2015_2024_v1"
PRICE_BASELINE_RECIPE_ID = "us_puf_price_baseline_2015_to_2024_v1"

#: A generous ceiling on the retained BLS flat-file body. The genuine
#: ``cu.data.1.AllItems`` distribution is roughly 2.7 MB; this is not a
#: prediction of its future size, only a refusal of an input much larger than the retained source before splitting it into lines.
_CPI_RESOURCE_MAX_BYTES = 16 * 1024 * 1024

_CITATION = (
    f"BLS {CPI_U_SERIES_ID} ({CPI_U_SERIES_TITLE}), {CPI_U_ANNUAL_AVERAGE_PERIOD} "
    f"annual average, base {CPI_U_INDEX_BASE}"
)

#: Pairwise relations the retained booklets leave unresolved. They are
#: **observed**, never enforced: the violation count is measured before and
#: after growth and must be equal, which is a property of one positive factor.
#: A nonzero count is evidence about the edited public file, not a defect this
#: transform may repair.
PAIRWISE_OBSERVATIONS = (
    ("qualified_within_ordinary_dividends", "E00600", "E00650"),
    ("taxable_within_total_pensions", "E01500", "E01700"),
)

_STATUS_COLUMNS = (
    SOURCE_RECID_FIELD,
    "FLPDYR",
    DESIGN_WEIGHT_FIELD,
    "disclosure_aggregate",
    "demographic_status",
)


def _require(condition: bool, reason: str, field: str = "price_baseline") -> None:
    if not condition:
        raise PufGrowthRefusalError(reason, field)


def _module_sha256() -> str:
    """This module's own bytes, so the factor table names the code that built it."""
    return hashlib.sha256(Path(__file__).resolve().read_bytes()).hexdigest()


def _canonical(value: object) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False, ensure_ascii=True
    ).encode("ascii")


def _cpi_annual_average_records(
    resource_bytes: bytes, years: Sequence[int]
) -> dict[int, bytes]:
    """The CPI-U all-items ``M13`` level text for each of *years*, or fewer.

    BLS pads ``series_id`` and ``value`` with spaces for monospace column
    alignment; the character that actually delimits a field is the tab, so
    every field is stripped before it is compared or returned — the same
    convention this lane's own retrieval script used to extract and hash
    these two lines. A row whose first field is this series but which does
    not split into the publisher's five tab-delimited fields
    (``series_id``, ``year``, ``period``, ``value``, ``footnote_codes``) is
    refused rather than silently skipped, because a truncated or reshuffled
    row is exactly the failure mode this check exists to catch. Rows for a
    year not in *years* are ignored: this baseline does not require the file
    to be otherwise well-formed, only that its own two records are.
    """
    series = CPI_U_SERIES_ID.encode("ascii")
    period = CPI_U_ANNUAL_AVERAGE_PERIOD.encode("ascii")
    wanted = {str(year).encode("ascii") for year in years}
    found: dict[int, bytes] = {}
    for line in resource_bytes.split(b"\n"):
        if not line.strip():
            continue
        fields = line.split(b"\t")
        if fields[0].strip() != series:
            continue
        _require(len(fields) == 5, "PRICE_BASELINE_RESOURCE_LINE_SHAPE")
        _, year_field, period_field, value_field, _ = (
            field.strip() for field in fields
        )
        if period_field != period or year_field not in wanted:
            continue
        year = int(year_field)
        _require(
            year not in found, "PRICE_BASELINE_RESOURCE_DUPLICATE_RECORD", str(year)
        )
        found[year] = value_field
    return found


def verify_cpi_resource_bytes(
    resource_bytes: bytes, *, resource_sha256: str
) -> Mapping[str, object]:
    """Refuse unless *resource_bytes* is this baseline's named public resource.

    The whole-file digest is bound first — the caller's ``resource_sha256``
    is no longer a string nobody checks. Then the bytes themselves are parsed
    with the publisher's own tab-delimited, space-padded convention (see
    :func:`_cpi_annual_average_records`) and must contain **exactly** the two
    ``CUUR0000SA0`` ``M13`` records this baseline reads, at the two years
    :data:`PRICE_BASELINE_LEVELS` names, each with exactly
    the declared decimal text after removing BLS column padding. Different
    numeric spellings and near-equal floats are refused: source lexical
    identity is stronger than arithmetic agreement.

    Args:
        resource_bytes: The exact bytes retrieved from ``source_url``.
        resource_sha256: The digest the caller claims for those bytes.

    Returns:
        A receipt naming the series, period, verified digest and the two
        matched records.

    Raises:
        PufGrowthRefusalError: On a type, size, digest, shape, duplicate,
            missing-record or value mismatch.
    """
    _require(
        isinstance(resource_bytes, (bytes, bytearray)), "PRICE_BASELINE_RESOURCE_TYPE"
    )
    _require(len(resource_bytes) > 0, "PRICE_BASELINE_RESOURCE_EMPTY")
    _require(
        len(resource_bytes) <= _CPI_RESOURCE_MAX_BYTES,
        "PRICE_BASELINE_RESOURCE_OVERSIZE",
    )
    body = bytes(resource_bytes)
    actual_sha256 = hashlib.sha256(body).hexdigest()
    _require(actual_sha256 == resource_sha256, "PRICE_BASELINE_RESOURCE_DIGEST")

    years = tuple(PRICE_BASELINE_LEVELS)
    found = _cpi_annual_average_records(body, years)
    records: dict[str, object] = {}
    for year in years:
        _require(year in found, "PRICE_BASELINE_RESOURCE_RECORD_MISSING", str(year))
        _require(
            found[year] == PRICE_BASELINE_LEVELS[year].encode("ascii"),
            "PRICE_BASELINE_RESOURCE_RECORD_VALUE",
            str(year),
        )
        # Decode only after exact ASCII equality; malformed values must refuse
        # by the same domain reason rather than leaking a parser exception.
        level_text = found[year].decode("ascii")
        found_value = float(level_text)
        records[str(year)] = MappingProxyType(
            {"level_text": level_text, "level_float64": found_value}
        )
    return MappingProxyType(
        {
            "series_id": CPI_U_SERIES_ID,
            "period": CPI_U_ANNUAL_AVERAGE_PERIOD,
            "resource_sha256": actual_sha256,
            "resource_bytes": len(body),
            "records": MappingProxyType(records),
        }
    )


def price_baseline_factor_document(
    *, source_url: str, resource_sha256: str, retrieved_utc: str, resource_bytes: bytes
) -> bytes:
    """The developmental factor table, as the exact bytes it hashes to.

    Args:
        source_url: The publisher endpoint the levels were read from.
        resource_sha256: Digest of the exact resource bytes that were read.
        retrieved_utc: When they were read, as an ISO 8601 UTC instant.
        resource_bytes: The exact bytes retrieved from ``source_url``. Checked
            against ``resource_sha256`` and parsed for the two annual-average
            records this baseline names — see
            :func:`verify_cpi_resource_bytes`. Without this the two prior
            arguments were caller assertions nothing in this module verified.

    Returns:
        A canonical factor-table document carrying the price series and the
        identity design-weight series, with the
        ``developmental_public_resource`` authority.
    """
    verify_cpi_resource_bytes(resource_bytes, resource_sha256=resource_sha256)
    return _canonical(
        {
            "authority": "developmental_public_resource",
            "provenance": {
                "dependency_versions": {},
                "generated_by": (
                    "microcosm.build.us_runtime.puf_price_baseline."
                    "price_baseline_factor_document"
                ),
                "generator_code_sha256": _module_sha256(),
                "index_or_total": "index",
                "parameter_paths": [],
                "population_divisor": "none",
                "resource_sha256": resource_sha256,
                "retrieved_utc": retrieved_utc,
                "rounding": (
                    "none; the two published levels are copied verbatim as "
                    "decimal text and the factor is one float64 division of them"
                ),
                "source_url": source_url,
            },
            "schema_version": 1,
            "series": {
                DESIGN_WEIGHT_IDENTITY_SERIES: {
                    "kind": str(GrowthKind.DESIGN_WEIGHT_GROWTH),
                    "levels": {
                        str(PRICE_BASELINE_SOURCE_YEAR): "1",
                        str(PRICE_BASELINE_TARGET_YEAR): "1",
                    },
                },
                PRICE_BASELINE_SERIES_NAME: {
                    "kind": str(GrowthKind.NOMINAL_PRICE_RESTATEMENT),
                    "levels": {
                        str(year): level
                        for year, level in PRICE_BASELINE_LEVELS.items()
                    },
                },
            },
            "table_id": PRICE_BASELINE_TABLE_ID,
            "years": [PRICE_BASELINE_SOURCE_YEAR, PRICE_BASELINE_TARGET_YEAR],
        }
    )


def price_baseline_recipe(
    money_fields: Sequence[str], *, recipe_id: str = PRICE_BASELINE_RECIPE_ID
) -> GrowthRecipe:
    """One price series for every money field, plus the two structural rules.

    Args:
        money_fields: The projection's money fields. Must contain
            ``E00100``, because the growth contract requires it, and must not
            contain ``RECID`` or ``S006``, which are not money.
        recipe_id: Recipe identity, for a caller running more than one.

    Raises:
        PufGrowthRefusalError: If the field set is not a money field set.
    """
    _require(bool(money_fields), "PRICE_BASELINE_NO_MONEY_FIELD")
    fields = tuple(money_fields)
    _require(len(set(fields)) == len(fields), "PRICE_BASELINE_DUPLICATE_FIELD")
    _require(SOURCE_AGI_FIELD in fields, "PRICE_BASELINE_AGI_FIELD", SOURCE_AGI_FIELD)
    for reserved in (SOURCE_RECID_FIELD, DESIGN_WEIGHT_FIELD):
        _require(reserved not in fields, "PRICE_BASELINE_NON_MONEY_FIELD", reserved)
    rules = [
        GrowthRule(
            field,
            FieldRole.MONEY,
            PRICE_BASELINE_SOURCE_YEAR,
            PRICE_BASELINE_TARGET_YEAR,
            GrowthKind.NOMINAL_PRICE_RESTATEMENT,
            SignBranch.ANY,
            PRICE_BASELINE_SERIES_NAME,
            _CITATION,
        )
        for field in fields
    ]
    rules.append(
        GrowthRule(
            SOURCE_RECID_FIELD,
            FieldRole.IDENTIFIER,
            PRICE_BASELINE_SOURCE_YEAR,
            PRICE_BASELINE_SOURCE_YEAR,
            GrowthKind.NONE,
        )
    )
    rules.append(
        GrowthRule(
            DESIGN_WEIGHT_FIELD,
            FieldRole.DESIGN_WEIGHT,
            PRICE_BASELINE_SOURCE_YEAR,
            PRICE_BASELINE_TARGET_YEAR,
            GrowthKind.DESIGN_WEIGHT_GROWTH,
            SignBranch.ANY,
            DESIGN_WEIGHT_IDENTITY_SERIES,
            "declared identity: this baseline transitions no return mass",
        )
    )
    return GrowthRecipe(recipe_id, tuple(rules))


def require_price_baseline_contract(
    compiled: CompiledPufGrowth, *, resource_bytes: bytes
) -> None:
    """Refuse any contract that is not the adopted price baseline.

    :func:`adapt_projection_to_money_view` and
    :func:`price_baseline_invariants` both call this before they record
    anything, because both go on to state the source year, the exact rational
    factor, the ``RECID`` identity and "this baseline transitions no return
    mass" as facts about the run. A contract that named a different pair of
    years, mixed a second factor series, aged rather than restated, split a
    field across sign branches, or grew the design weight would leave every
    other check in this module passing while those recorded statements were
    false. The false receipt is the failure this refuses.

    What is checked is exactly what those receipts assert:

    * the ``developmental_public_resource`` authority, and no release
      eligibility;
    * every money rule at ``2015 -> 2024``, ``nominal_price_restatement``,
      ``SignBranch.ANY``, on the one named CPI series, resolving to the
      published level texts and to the exact float64 of ``313689/237017``;
    * exactly one identifier rule, on ``RECID``, at the baseline source year;
    * exactly one design-weight rule, on ``S006``, at the same two years, on
      the declared identity series, whose resolved factor is exactly ``1.0``
      and whose two level texts are the same text — not two texts whose
      float64 quotient happens to round to one;
    * no rule in any other role;
    * the exact resource bytes against the digest the contract names, and the
      two source records against their declared series, years, period and text.

    The *field set* is deliberately not fixed: a contract over a subset of the
    projection's money fields is still this baseline, and the adapter checks
    the subset relation against the artifact separately.

    Args:
        compiled: The contract a caller is about to run.
        resource_bytes: The retained BLS body; checked against the contract's
            own resource digest even if it came from the generic growth compiler.

    Raises:
        PufGrowthRefusalError: With a ``PRICE_BASELINE_*`` reason naming the
            first field that does not match.
    """
    _require(type(compiled) is CompiledPufGrowth, "COMPILED_TYPE")
    _require(
        compiled.factors.authority is FactorAuthority.DEVELOPMENTAL_PUBLIC_RESOURCE,
        "PRICE_BASELINE_AUTHORITY",
    )
    _require(compiled.release_eligible is False, "PRICE_BASELINE_RELEASE_ELIGIBLE")
    verify_cpi_resource_bytes(
        resource_bytes,
        resource_sha256=compiled.factors.provenance["resource_sha256"],
    )

    rules = compiled.recipe.rules
    money = tuple(rule for rule in rules if rule.role is FieldRole.MONEY)
    identifiers = tuple(rule for rule in rules if rule.role is FieldRole.IDENTIFIER)
    weights = tuple(rule for rule in rules if rule.role is FieldRole.DESIGN_WEIGHT)
    for rule in rules:
        _require(
            rule.role
            in (FieldRole.MONEY, FieldRole.IDENTIFIER, FieldRole.DESIGN_WEIGHT),
            "PRICE_BASELINE_ROLE",
            rule.field,
        )
    _require(bool(money), "PRICE_BASELINE_NO_MONEY_FIELD")
    _require(
        SOURCE_AGI_FIELD in {rule.field for rule in money},
        "PRICE_BASELINE_AGI_FIELD",
        SOURCE_AGI_FIELD,
    )
    source_level = PRICE_BASELINE_LEVELS[PRICE_BASELINE_SOURCE_YEAR]
    target_level = PRICE_BASELINE_LEVELS[PRICE_BASELINE_TARGET_YEAR]
    for rule in money:
        field = rule.field
        _require(
            rule.source_reference_year == PRICE_BASELINE_SOURCE_YEAR,
            "PRICE_BASELINE_SOURCE_YEAR",
            field,
        )
        _require(
            rule.target_year == PRICE_BASELINE_TARGET_YEAR,
            "PRICE_BASELINE_TARGET_YEAR",
            field,
        )
        _require(
            rule.growth_kind is GrowthKind.NOMINAL_PRICE_RESTATEMENT,
            "PRICE_BASELINE_GROWTH_KIND",
            field,
        )
        # ``SignBranch.ANY`` is the field's only rule, which is what lets one
        # positive factor be checked per row rather than per branch.
        _require(
            rule.sign_branch is SignBranch.ANY, "PRICE_BASELINE_SIGN_BRANCH", field
        )
        _require(
            rule.factor_series == PRICE_BASELINE_SERIES_NAME,
            "PRICE_BASELINE_SERIES",
            field,
        )
        resolved = compiled.factor_for(field, SignBranch.ANY)
        # The substantive claim first — this is the one float64 — and then the
        # two level texts it must have come from. The order matters because
        # more than one pair of texts divides to the same double, and this
        # baseline is the published pair, not merely a pair that agrees.
        _require(
            resolved.ratio_bits == PRICE_BASELINE_FACTOR_BITS,
            "PRICE_BASELINE_FACTOR",
            field,
        )
        _require(
            resolved.source_level == source_level, "PRICE_BASELINE_SOURCE_LEVEL", field
        )
        _require(
            resolved.target_level == target_level, "PRICE_BASELINE_TARGET_LEVEL", field
        )

    # ``GrowthRecipe`` already requires exactly one identifier field, and
    # ``GrowthRule`` already fixes a non-money, non-weight role to
    # ``GrowthKind.NONE`` with equal years. What is open here is which field
    # carries the role and which year it names.
    _require(
        len(identifiers) == 1 and identifiers[0].field == SOURCE_RECID_FIELD,
        "PRICE_BASELINE_IDENTIFIER",
    )
    identifier = identifiers[0]
    _require(
        identifier.source_reference_year == PRICE_BASELINE_SOURCE_YEAR,
        "PRICE_BASELINE_IDENTIFIER_YEAR",
        identifier.field,
    )

    # Likewise the design-weight role already fixes the kind and the sign
    # branch. What is open is the field, the series, and whether the declared
    # factor really is one.
    _require(
        len(weights) == 1 and weights[0].field == DESIGN_WEIGHT_FIELD,
        "PRICE_BASELINE_DESIGN_WEIGHT",
    )
    weight = weights[0]
    _require(
        weight.factor_series == DESIGN_WEIGHT_IDENTITY_SERIES,
        "PRICE_BASELINE_DESIGN_WEIGHT_SERIES",
        weight.field,
    )
    _require(
        weight.source_reference_year == PRICE_BASELINE_SOURCE_YEAR
        and weight.target_year == PRICE_BASELINE_TARGET_YEAR,
        "PRICE_BASELINE_DESIGN_WEIGHT_YEAR",
        weight.field,
    )
    resolved_weight = compiled.factor_for(weight.field, SignBranch.ANY)
    # The substantive claim first: the declared weight factor is exactly one.
    # Then the two level texts, which must be the same text and not merely two
    # texts whose float64 quotient rounds to one.
    _require(
        resolved_weight.value == 1.0,
        "PRICE_BASELINE_DESIGN_WEIGHT_FACTOR",
        weight.field,
    )
    _require(
        resolved_weight.source_level == resolved_weight.target_level,
        "PRICE_BASELINE_DESIGN_WEIGHT_LEVEL",
        weight.field,
    )


def compile_price_baseline(
    money_fields: Sequence[str],
    *,
    source_url: str,
    resource_sha256: str,
    retrieved_utc: str,
    resource_bytes: bytes,
    recipe_id: str = PRICE_BASELINE_RECIPE_ID,
) -> CompiledPufGrowth:
    """Compile the price-baseline contract; never release-eligible.

    ``resource_bytes`` is required and is checked against ``resource_sha256``
    by :func:`price_baseline_factor_document`; this function cannot hand back
    a contract pinned to resource metadata nobody verified.
    """
    recipe = price_baseline_recipe(money_fields, recipe_id=recipe_id)
    factors = GrowthFactorTable.from_bytes(
        price_baseline_factor_document(
            source_url=source_url,
            resource_sha256=resource_sha256,
            retrieved_utc=retrieved_utc,
            resource_bytes=resource_bytes,
        )
    )
    compiled = compile_puf_growth(recipe, factors)
    # The authority gate already guarantees this. Asserting it here means the
    # one function a caller reaches for cannot hand back an eligible contract
    # even if the gate is later widened by mistake.
    _require(compiled.release_eligible is False, "PRICE_BASELINE_RELEASE_ELIGIBLE")
    # The same gate the two entry points apply, so this function cannot hand
    # back a contract they would decline.
    require_price_baseline_contract(compiled, resource_bytes=resource_bytes)
    return compiled


# --------------------------------------------------------------------------
# The artifact-to-money-view adapter
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class PufMoneyViewAdaptation:
    """The money view the growth transform consumes, plus what stayed outside.

    Attributes:
        money_view: One row per amount-known return, carrying ``RECID`` and
            the declared money columns as int64. Never the design weight, and
            never a disclosure-aggregate row.
        recid: The kept rows' identifiers, in money-view order.
        flpdyr: The kept rows' raw observation year, carried from the status
            artifact and never restated.
        demographic_status: The kept rows' demographic match flag.
        design_weight: The kept rows' ``S006``, in the delivered integer
            hundredths, unaltered and outside the money view.
        excluded: The audit identity of the removed disclosure-aggregate rows.
        receipt: What was joined, kept and removed.
    """

    money_view: pd.DataFrame
    recid: np.ndarray
    flpdyr: np.ndarray
    demographic_status: np.ndarray
    design_weight: np.ndarray
    excluded: Mapping[str, object]
    receipt: Mapping[str, object]


def _status_arrays(status: object, rows: int | None = None) -> Mapping[str, np.ndarray]:
    """Every status column this adapter reads, present and the same length.

    The length check covers ``FLPDYR``, ``S006`` and the demographic status as
    well as the two join keys, so a short or long side column is a refusal
    rather than a silently misaligned slice later.
    """
    try:
        typed = status.typed
    except AttributeError as error:
        raise PufGrowthRefusalError("STATUS_ARTIFACT_TYPE") from error
    # Presence first for every column, then length for every column, so a
    # missing column is reported as missing rather than as a length mismatch.
    for name in _STATUS_COLUMNS:
        _require(name in typed, "STATUS_ARTIFACT_COLUMN", name)
    if rows is not None:
        for name in _STATUS_COLUMNS:
            _require(
                int(np.asarray(typed[name]).shape[0]) == rows,
                "STATUS_COLUMN_ROWS",
                name,
            )
    return typed


def adapt_projection_to_money_view(
    decoded: PufMonetarySourceProjection,
    status: object,
    compiled: CompiledPufGrowth,
    *,
    resource_bytes: bytes,
) -> PufMoneyViewAdaptation:
    """Join the projection to the status artifact and drop only the aggregates.

    The join is by position **and** proved by value: the two artifacts' whole
    ``RECID`` sequences and disclosure flags must be element-wise identical
    before anything is selected, so a row's amounts and its status cannot come
    from different returns.

    Exactly the rows outside the individual-return amount universe are
    removed, and their identity is recorded. Every removed row must hold the
    out-of-universe sentinel in every projected column, and no kept row may
    hold it, so the sentinel cannot reach a numeric input in either direction.

    The contract is checked against the adopted baseline before anything is
    recorded — see :func:`require_price_baseline_contract` — because the
    receipt below states the source year, the design-weight identity and the
    absence of a return-mass transition as facts about this run.

    Args:
        decoded: A decoded thirteen-field projection.
        status: The decoded ``puf_2015_raw_return_status`` artifact.
        compiled: The compiled price-baseline contract.
        resource_bytes: The retained BLS body authenticated by the entry gate.

    Returns:
        The money view and everything deliberately kept out of it.

    Raises:
        PufGrowthRefusalError: On any contract, join, universe or sentinel
            violation.
    """
    require_price_baseline_contract(compiled, resource_bytes=resource_bytes)
    _require(
        isinstance(decoded, PufMonetarySourceProjection), "PROJECTION_ARTIFACT_TYPE"
    )
    projection_recid = np.asarray(decoded.typed[SOURCE_RECID_FIELD], dtype="<i8")
    rows = int(projection_recid.shape[0])
    _require(rows > 0, "PROJECTION_NO_ROWS")
    typed = _status_arrays(status, rows)
    status_recid = np.asarray(typed[SOURCE_RECID_FIELD], dtype="<i8")
    _require(bool(np.array_equal(projection_recid, status_recid)), "STATUS_RECID_JOIN")
    _require(len(np.unique(projection_recid)) == rows, "DUPLICATE_RECID")
    projection_flags = np.asarray(decoded.typed["disclosure_aggregate"], dtype="<u1")
    status_flags = np.asarray(typed["disclosure_aggregate"], dtype="<u1")
    _require(bool(np.array_equal(projection_flags, status_flags)), "STATUS_FLAG_JOIN")

    known = np.asarray(decoded.amount_known, dtype=bool)
    _require(bool(np.array_equal(known, projection_flags == 0)), "AMOUNT_KNOWN_RULE")
    kept = int(np.count_nonzero(known))
    removed = rows - kept
    _require(kept > 0, "NO_AMOUNT_KNOWN_ROW")
    _require(
        removed == int(decoded.facts["row_class_rows"]["disclosure_aggregate"]),
        "EXCLUDED_ROW_COUNT",
    )

    money_fields = compiled.money_fields
    _require(set(money_fields) <= set(decoded.typed), "PROJECTION_MISSING_MONEY_FIELD")
    _require(AGI_FIELD in money_fields, "PRICE_BASELINE_AGI_FIELD", AGI_FIELD)
    columns: dict[str, np.ndarray] = {}
    for field in money_fields:
        values = np.asarray(decoded.typed[field], dtype="<i8")
        _require(int(values.shape[0]) == rows, "COLUMN_ROWS", field)
        # Both directions: no aggregate slot may be anything but the sentinel,
        # and no ordinary slot may be the sentinel.
        _require(
            bool((values[~known] == AGGREGATE_SENTINEL).all()),
            "AGGREGATE_SLOT_NOT_SENTINEL",
            field,
        )
        _require(
            not bool((values[known] == AGGREGATE_SENTINEL).any()),
            "SENTINEL_IN_ORDINARY_ROW",
            field,
        )
        columns[field] = values[known].copy()

    money_view = pd.DataFrame(index=pd.RangeIndex(kept))
    money_view[SOURCE_RECID_FIELD] = pd.Series(
        projection_recid[known].copy(), index=money_view.index, dtype="int64"
    )
    for field in money_fields:
        money_view[field] = pd.Series(
            columns[field], index=money_view.index, dtype="int64"
        )
    _require(
        set(money_view.columns) == set(compiled.money_view_fields),
        "MONEY_VIEW_COLUMNS",
    )
    _require(DESIGN_WEIGHT_FIELD not in money_view.columns, "DESIGN_WEIGHT_IN_VIEW")

    design_weight = np.asarray(typed[DESIGN_WEIGHT_FIELD], dtype="<i8")
    flpdyr = np.asarray(typed["FLPDYR"], dtype="<i2")
    demographic_status = np.asarray(typed["demographic_status"], dtype="<i1")

    excluded_tokens = {
        field: [decoded.lexical[field][position] for position in np.flatnonzero(~known)]
        for field in sorted(decoded.lexical)
    }
    excluded = MappingProxyType(
        {
            "rows": removed,
            "recid": [int(value) for value in projection_recid[~known]],
            "flpdyr": [int(value) for value in flpdyr[~known]],
            "design_weight_s006": [int(value) for value in design_weight[~known]],
            "demographic_status": [int(value) for value in demographic_status[~known]],
            "typed_slot": "aggregate_typed_sentinel_in_every_projected_column",
            "aggregate_typed_sentinel": AGGREGATE_SENTINEL,
            # A digest, not the tokens: the removal is auditable without any
            # receipt, report or log ever spelling an aggregate amount.
            "delivered_token_sha256": hashlib.sha256(
                _canonical(excluded_tokens)
            ).hexdigest(),
            "token_class_counts": {
                field: dict(classes["disclosure_aggregate"])
                for field, classes in decoded.facts["column_facts"].items()
            },
            "interpretation": "refused; the delivered token remains the authority",
        }
    )

    # Read off the contract rather than restated from module constants: the
    # gate above proved the contract declares these, so the receipt can carry
    # the contract's own answer and a reader can check the two agree.
    weight_rule = compiled.design_weight_rule
    weight_factor = compiled.factor_for(weight_rule.field, SignBranch.ANY)
    money_source_years = sorted(
        {
            rule.source_reference_year
            for rule in compiled.recipe.rules
            if rule.role is FieldRole.MONEY
        }
    )
    receipt = MappingProxyType(
        {
            "artifact_kind": "microcosm.us_puf_price_baseline_money_view",
            "contract_sha256": compiled.sha256,
            "factor_authority": str(compiled.factors.authority),
            "release_eligible": compiled.release_eligible,
            "projection_rows": rows,
            "money_view_rows": kept,
            "excluded_rows": removed,
            "amount_known_rule": decoded.facts["amount_known_rule"],
            "amount_known_rows": int(decoded.facts["amount_known_rows"]),
            "recid_join": "elementwise_identical_projection_and_status_sequences",
            "recid_unique": True,
            "money_view_fields": sorted(money_view.columns),
            # A contract over a subset of the projection's money fields is
            # still this baseline, but which columns it left behind is part of
            # what the run did and is named rather than inferred from a count.
            "projected_fields": sorted(decoded.facts["column_facts"]),
            "projected_fields_not_in_contract": sorted(
                set(decoded.facts["column_facts"]) - set(money_fields)
            ),
            "design_weight_field": DESIGN_WEIGHT_FIELD,
            "design_weight_in_money_view": False,
            "design_weight_units": "delivered_integer_hundredths_unaltered",
            "design_weight_series": weight_rule.factor_series,
            "design_weight_declared_factor_float64": weight_factor.value,
            "return_mass_transition": "none_declared_identity_series",
            "flpdyr_treatment": "carried_as_observation_metadata_never_restated",
            "source_reference_year": money_source_years[0],
            "target_year": compiled.target_year,
            "factor_ratio_bits": compiled.factor_for(
                money_fields[0], SignBranch.ANY
            ).ratio_bits,
            "entry_contract_gate": "require_price_baseline_contract",
        }
    )
    return PufMoneyViewAdaptation(
        money_view=money_view,
        recid=projection_recid[known].copy(),
        flpdyr=flpdyr[known].copy(),
        demographic_status=demographic_status[known].copy(),
        design_weight=design_weight[known].copy(),
        excluded=excluded,
        receipt=receipt,
    )


# --------------------------------------------------------------------------
# Invariants observed across the transform
# --------------------------------------------------------------------------


def _violations(frame, greater: str, lesser: str) -> int | None:
    if greater not in frame.columns or lesser not in frame.columns:
        return None
    return int((frame[lesser] > frame[greater]).sum())


def _bits(values: np.ndarray) -> np.ndarray:
    """A float64 array's raw bit patterns, so ``-0.0`` is not ``0.0``."""
    return np.ascontiguousarray(values, dtype="float64").view("<u8")


def price_baseline_invariants(
    adaptation: PufMoneyViewAdaptation,
    grown: GrownPufTable,
    compiled: CompiledPufGrowth,
    *,
    resource_bytes: bytes,
    design_weight_after: Sequence[float] | np.ndarray | None = None,
    relative_tolerance: float = 1e-12,
) -> Mapping[str, object]:
    """Measure what one positive factor must leave true, and refuse otherwise.

    Three things happen here, in this order, and the order is deliberate.

    *The summaries.* Per column: dtype, finiteness, the sign set, the zero set,
    no negative zero, and the exact integer sum of the input column times the
    factor against the grown ``math.fsum``. Then the pairwise relations the
    booklets leave open, counted before and after and required only to be
    **equal**. These are the numbers a reader wants in the receipt, and each
    names a specific symptom when it fires.

    *The exact row comparison.* Every kept row of every money column must be
    bit-for-bit the contract's own resolved float64 factor applied to that
    row's own delivered amount, and the two provenance columns must be
    bit-for-bit their sources. This is the statement the summaries cannot
    make: a coordinated rearrangement preserves every total, every sign, every
    zero and every pairwise count, and is caught only here. It runs last so
    that the narrower checks keep reporting the narrower diagnosis — and so
    that each of them stays independently falsifiable.

    *The design weight.* Compared, if a caller supplies the weights it carried
    forward, against the adaptation's own ``S006``. Not supplied, it is
    reported as a declaration and **not** as a measurement.

    Args:
        adaptation: The adaptation whose money view was grown.
        grown: The result of :func:`~microcosm.build.us_runtime.puf_growth.apply_puf_growth`
            on that money view, under this same contract.
        compiled: The contract, checked against the adopted baseline.
        resource_bytes: The retained BLS body authenticated by the entry gate.
        design_weight_after: The post-growth design weights, if the caller ran
            :func:`~microcosm.build.us_runtime.puf_growth.apply_puf_design_weight_growth`.
            Omitted, ``design_weight_unchanged`` is ``None``, not ``True``.
        relative_tolerance: Tolerance for the scaled-sum comparison only.

    Raises:
        PufGrowthRefusalError: On any contract, identity, row, sum, pairwise or
            weight violation.
    """
    require_price_baseline_contract(compiled, resource_bytes=resource_bytes)
    _require(type(grown) is GrownPufTable, "GROWN_TYPE")
    _require(type(adaptation) is PufMoneyViewAdaptation, "ADAPTATION_TYPE")
    # One contract, named the same way by all three objects. Without this the
    # measurements below would describe a contract the table was not grown
    # under, and every number in the receipt would be attributed to it.
    contract_sha256 = compiled.sha256
    _require(grown.contract_sha256 == contract_sha256, "GROWN_CONTRACT_MISMATCH")
    _require(
        grown.receipt["contract_sha256"] == contract_sha256,
        "GROWN_RECEIPT_CONTRACT_MISMATCH",
    )
    _require(
        adaptation.receipt["contract_sha256"] == contract_sha256,
        "ADAPTATION_CONTRACT_MISMATCH",
    )
    source = adaptation.money_view
    table = grown.table
    rows = len(source)
    _require(len(table) == rows, "GROWN_ROWS")
    _require(
        bool(
            np.array_equal(
                table[SOURCE_RECID_FIELD].to_numpy(),
                source[SOURCE_RECID_FIELD].to_numpy(),
            )
        ),
        "GROWN_RECID",
    )
    factor = compiled.factor_for(compiled.money_fields[0], SignBranch.ANY).value
    columns: dict[str, object] = {}
    for field in compiled.money_fields:
        resolved = compiled.factor_for(field, SignBranch.ANY)
        _require(
            resolved.ratio_bits
            == compiled.factor_for(compiled.money_fields[0], SignBranch.ANY).ratio_bits,
            "MIXED_FACTOR",
            field,
        )
        original = source[field].to_numpy()
        result = table[field].to_numpy()
        _require(result.dtype == np.dtype("float64"), "GROWN_DTYPE", field)
        _require(bool(np.isfinite(result).all()), "GROWN_NONFINITE", field)
        _require(
            bool((np.sign(result) == np.sign(original.astype("float64"))).all()),
            "SIGN_CHANGED",
            field,
        )
        zeros = result == 0.0
        _require(not bool(np.signbit(result[zeros]).any()), "NEGATIVE_ZERO", field)
        _require(bool(np.array_equal(zeros, original == 0)), "ZERO_SET_CHANGED", field)
        exact_sum = int(sum(int(value) for value in original.tolist()))
        gross_sum = int(sum(abs(int(value)) for value in original.tolist()))
        grown_sum = math.fsum(result.tolist())
        expected = exact_sum * factor
        # Scaled by the column's **gross** magnitude, not its net total. The
        # float error one positive factor can accumulate is bounded by the
        # magnitude that passed through it; a signed column whose positives
        # and negatives nearly cancel has a tiny net and the same gross, and
        # scaling by the net alone would refuse a correct result for it.
        net_scale = max(abs(expected), 1.0)
        scale = max(abs(gross_sum * factor), net_scale)
        _require(
            abs(grown_sum - expected) <= relative_tolerance * scale,
            "SCALED_SUM_DISAGREEMENT",
            field,
        )
        columns[field] = {
            "rows": rows,
            "exact_source_sum": exact_sum,
            "gross_source_sum_absolute": gross_sum,
            "grown_fsum": grown_sum,
            "expected_grown_sum": expected,
            "scale_basis": "gross_absolute_source_sum_times_factor",
            "relative_difference": (
                abs(grown_sum - expected) / scale if scale else 0.0
            ),
            "relative_difference_net_scaled": (
                abs(grown_sum - expected) / net_scale if net_scale else 0.0
            ),
            "negative": int((original < 0).sum()),
            "positive": int((original > 0).sum()),
            "zero": int((original == 0).sum()),
            "ratio_bits": resolved.ratio_bits,
            "grown_minimum": float(result.min()),
            "grown_maximum": float(result.max()),
        }
    pairwise = {}
    for name, greater, lesser in PAIRWISE_OBSERVATIONS:
        before = _violations(source, greater, lesser)
        after = _violations(table, greater, lesser)
        # A contract over a subset of the projection's fields can leave one
        # side of a relation out of the money view entirely. ``None == None``
        # would then pass, and the entry would read as an observation that was
        # never made, so the absence is named instead of being counted.
        _require((before is None) == (after is None), "PAIRWISE_COLUMN_ASYMMETRY", name)
        observable = before is not None
        if observable:
            _require(before == after, "PAIRWISE_VIOLATIONS_MOVED", name)
        pairwise[name] = {
            "greater": greater,
            "lesser": lesser,
            "violations_before": before,
            "violations_after": after,
            "status": (
                "observed_not_enforced"
                if observable
                else "not_observable_column_absent"
            ),
            "absent_columns": sorted(
                column for column in (greater, lesser) if column not in source.columns
            ),
        }

    # The exact row comparison. ``apply_puf_growth`` selects ``values != 0.0``
    # for a SignBranch.ANY rule and writes ``values[selected] * factor``, so
    # the expected array is reproduced here by the same two operations in the
    # same order and compared on raw bits. Nothing is recomputed from the
    # levels or the rational: the contract's own resolved float64 is the one
    # multiplier, which is what makes this a check of the transform rather
    # than a second opinion about the factor.
    for field in compiled.money_fields:
        resolved = compiled.factor_for(field, SignBranch.ANY)
        original = source[field].to_numpy()
        expected = original.astype("float64")
        selected = expected != 0.0
        with np.errstate(over="ignore", under="ignore"):
            expected[selected] = expected[selected] * resolved.value
        result = table[field].to_numpy()
        disagreeing = int(np.count_nonzero(_bits(result) != _bits(expected)))
        _require(disagreeing == 0, "ROW_GROWTH_DISAGREEMENT", field)

    # The two columns ``apply_puf_growth`` copies rather than grows. A silent
    # edit here would restate provenance while every money column still
    # matched.
    for column, origin in (
        (PROVENANCE_RECID_COLUMN, source[SOURCE_RECID_FIELD].to_numpy()),
        (
            PROVENANCE_SOURCE_AGI_COLUMN,
            source[SOURCE_AGI_FIELD].to_numpy().astype("float64"),
        ),
    ):
        _require(column in table.columns, "PROVENANCE_COLUMN_MISSING", column)
        carried = table[column].to_numpy()
        if column == PROVENANCE_SOURCE_AGI_COLUMN:
            _require(
                bool((_bits(carried) == _bits(origin)).all()),
                "PROVENANCE_DISAGREEMENT",
                column,
            )
        else:
            _require(
                bool(np.array_equal(carried, origin)),
                "PROVENANCE_DISAGREEMENT",
                column,
            )

    design_weight, design_weight_unchanged = _design_weight_observation(
        adaptation, compiled, design_weight_after
    )
    return MappingProxyType(
        {
            "rows": rows,
            "factor_ratio_bits": compiled.factor_for(
                compiled.money_fields[0], SignBranch.ANY
            ).ratio_bits,
            "factor_float64": factor,
            "exact_rational": {
                "numerator": PRICE_BASELINE_RATIO[0],
                "denominator": PRICE_BASELINE_RATIO[1],
            },
            "output_rounding": "none_unrounded_float64",
            "columns": columns,
            "pairwise_observations": pairwise,
            "row_exact_growth": {
                "rows": rows,
                "fields": list(compiled.money_fields),
                "comparison": (
                    "bitwise float64 equality of every kept row against the "
                    "contract's own resolved factor applied to that row's "
                    "delivered amount, zeros left untouched"
                ),
                "disagreeing_rows": 0,
                "provenance_columns_compared": list(
                    (PROVENANCE_RECID_COLUMN, PROVENANCE_SOURCE_AGI_COLUMN)
                ),
            },
            "contract_sha256": contract_sha256,
            "contract_named_by": {
                "compiled": contract_sha256,
                "grown_result": grown.contract_sha256,
                "grown_receipt": grown.receipt["contract_sha256"],
                "adaptation_receipt": adaptation.receipt["contract_sha256"],
            },
            "design_weight": design_weight,
            # ``None`` when no post-growth weight was supplied. This is never
            # ``True`` unless the arrays were actually compared.
            "design_weight_unchanged": design_weight_unchanged,
            "release_eligible": compiled.release_eligible,
        }
    )


def _design_weight_observation(
    adaptation: PufMoneyViewAdaptation,
    compiled: CompiledPufGrowth,
    design_weight_after: Sequence[float] | np.ndarray | None,
) -> tuple[dict[str, object], bool | None]:
    """What the design weight did, separating the declaration from a measurement.

    The declaration is the contract's: an identity series whose two levels are
    both one, so the declared factor is exactly ``1.0``. That is a statement
    about the recipe, and it is reported as one.

    The measurement needs a second array. ``adaptation.design_weight`` is the
    status artifact's own ``S006`` sliced to the kept rows — the before side,
    with its digest recorded so the comparison is auditable. The after side is
    whatever the caller carried forward. Without it there is no observation to
    report, and this returns ``None`` rather than manufacturing one.
    """
    rule = compiled.design_weight_rule
    declared = compiled.factor_for(rule.field, SignBranch.ANY)
    before = np.asarray(adaptation.design_weight)
    _require(before.ndim == 1, "DESIGN_WEIGHT_SHAPE", rule.field)
    record: dict[str, object] = {
        "field": rule.field,
        "rows": int(before.shape[0]),
        "declared_series": rule.factor_series,
        "declared_source_level": declared.source_level,
        "declared_target_level": declared.target_level,
        "declared_factor_float64": declared.value,
        "declared_return_mass_transition": "none",
        "before_source": (
            "adaptation.design_weight: the status artifact's own S006, in "
            "delivered integer hundredths, sliced to the kept rows"
        ),
        "before_sha256": hashlib.sha256(
            np.ascontiguousarray(before, dtype="<i8").tobytes()
        ).hexdigest(),
    }
    # Deliberately *not* multiplying the source weights by the declared factor
    # here and reporting that nothing moved. The gate has already required
    # that factor to be exactly 1.0, and ``x * 1.0 == x`` for every float64,
    # so such a check could not fail and its "measured" result would be an
    # arithmetic tautology dressed as evidence about this run's weights. The
    # only real observation needs the array the caller carried forward.
    before64 = before.astype("float64")
    if design_weight_after is None:
        record["observed"] = False
        record["basis"] = "declared_only_no_post_growth_weight_supplied"
        record["note"] = (
            "The contract declares an identity factor, so this baseline "
            "transitions no return mass by declaration. Whether the weights "
            "this caller carried forward are in fact unchanged is not "
            "measured here; pass design_weight_after to measure it."
        )
        return record, None
    after = np.asarray(design_weight_after)
    _require(after.ndim == 1, "DESIGN_WEIGHT_AFTER_SHAPE", rule.field)
    _require(
        int(after.shape[0]) == int(before.shape[0]),
        "DESIGN_WEIGHT_AFTER_ROWS",
        rule.field,
    )
    after64 = np.asarray(after, dtype="float64")
    _require(
        bool(np.isfinite(after64).all()), "DESIGN_WEIGHT_AFTER_NONFINITE", rule.field
    )
    changed = int(np.count_nonzero(_bits(after64) != _bits(before64)))
    _require(changed == 0, "DESIGN_WEIGHT_CHANGED", rule.field)
    record["observed"] = True
    record["basis"] = (
        "bitwise float64 equality of the supplied post-growth weights against "
        "the adaptation's own S006"
    )
    record["rows_changed"] = 0
    record["after_sha256"] = hashlib.sha256(
        np.ascontiguousarray(after64).tobytes()
    ).hexdigest()
    return record, True
