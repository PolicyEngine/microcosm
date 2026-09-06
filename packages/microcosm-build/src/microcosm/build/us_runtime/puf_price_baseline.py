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
levels really did come from a public publisher, and that authority can never
become release-eligible.

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
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType

import numpy as np
import pandas as pd

from .puf_growth import (
    DESIGN_WEIGHT_FIELD,
    SOURCE_AGI_FIELD,
    SOURCE_RECID_FIELD,
    CompiledPufGrowth,
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
    "PAIRWISE_OBSERVATIONS",
    "PufMoneyViewAdaptation",
    "adapt_projection_to_money_view",
    "compile_price_baseline",
    "price_baseline_factor_document",
    "price_baseline_invariants",
    "price_baseline_recipe",
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

PRICE_BASELINE_SERIES_NAME = "bls.cpi_u.all_items.us_city_average.annual_average"

#: The design-weight series. Both levels are exactly one, so the declared
#: factor is exactly 1.0 and "no return-mass transition" is a recorded
#: declaration rather than an omission a later reader has to infer.
DESIGN_WEIGHT_IDENTITY_SERIES = "declared_identity.no_return_mass_transition"

PRICE_BASELINE_TABLE_ID = "us_puf_price_baseline_cpi_u_2015_2024_v1"
PRICE_BASELINE_RECIPE_ID = "us_puf_price_baseline_2015_to_2024_v1"

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


def price_baseline_factor_document(
    *, source_url: str, resource_sha256: str, retrieved_utc: str
) -> bytes:
    """The developmental factor table, as the exact bytes it hashes to.

    Args:
        source_url: The publisher endpoint the levels were read from.
        resource_sha256: Digest of the exact resource bytes that were read.
        retrieved_utc: When they were read, as an ISO 8601 UTC instant.

    Returns:
        A canonical factor-table document carrying the price series and the
        identity design-weight series, with the
        ``developmental_public_resource`` authority.
    """
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


def compile_price_baseline(
    money_fields: Sequence[str],
    *,
    source_url: str,
    resource_sha256: str,
    retrieved_utc: str,
    recipe_id: str = PRICE_BASELINE_RECIPE_ID,
) -> CompiledPufGrowth:
    """Compile the price-baseline contract; never release-eligible."""
    recipe = price_baseline_recipe(money_fields, recipe_id=recipe_id)
    factors = GrowthFactorTable.from_bytes(
        price_baseline_factor_document(
            source_url=source_url,
            resource_sha256=resource_sha256,
            retrieved_utc=retrieved_utc,
        )
    )
    compiled = compile_puf_growth(recipe, factors)
    # The authority gate already guarantees this. Asserting it here means the
    # one function a caller reaches for cannot hand back an eligible contract
    # even if the gate is later widened by mistake.
    _require(compiled.release_eligible is False, "PRICE_BASELINE_RELEASE_ELIGIBLE")
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

    Args:
        decoded: A decoded thirteen-field projection.
        status: The decoded ``puf_2015_raw_return_status`` artifact.
        compiled: The compiled price-baseline contract.

    Returns:
        The money view and everything deliberately kept out of it.

    Raises:
        PufGrowthRefusalError: On any join, universe or sentinel violation.
    """
    _require(type(compiled) is CompiledPufGrowth, "COMPILED_TYPE")
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
            "design_weight_field": DESIGN_WEIGHT_FIELD,
            "design_weight_in_money_view": False,
            "design_weight_units": "delivered_integer_hundredths_unaltered",
            "return_mass_transition": "none_declared_identity_series",
            "flpdyr_treatment": "carried_as_observation_metadata_never_restated",
            "source_reference_year": PRICE_BASELINE_SOURCE_YEAR,
            "target_year": compiled.target_year,
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


def price_baseline_invariants(
    adaptation: PufMoneyViewAdaptation,
    grown: GrownPufTable,
    compiled: CompiledPufGrowth,
    *,
    relative_tolerance: float = 1e-12,
) -> Mapping[str, object]:
    """Measure what one positive factor must leave true, and refuse otherwise.

    Sums are checked against the *exact* integer sum of the input column
    divided out of the grown total, which is a statement about this transform's
    arithmetic and not about any external total. Pairwise relations are
    observed before and after and required only to be **equal**: a violation
    in the edited public file is evidence, not something to repair.
    """
    _require(type(compiled) is CompiledPufGrowth, "COMPILED_TYPE")
    _require(type(grown) is GrownPufTable, "GROWN_TYPE")
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
        grown_sum = math.fsum(result.tolist())
        expected = exact_sum * factor
        scale = max(abs(expected), 1.0)
        _require(
            abs(grown_sum - expected) <= relative_tolerance * scale,
            "SCALED_SUM_DISAGREEMENT",
            field,
        )
        columns[field] = {
            "rows": rows,
            "exact_source_sum": exact_sum,
            "grown_fsum": grown_sum,
            "expected_grown_sum": expected,
            "relative_difference": (
                abs(grown_sum - expected) / scale if scale else 0.0
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
        _require(before == after, "PAIRWISE_VIOLATIONS_MOVED", name)
        pairwise[name] = {
            "greater": greater,
            "lesser": lesser,
            "violations_before": before,
            "violations_after": after,
            "status": "observed_not_enforced",
        }
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
            "design_weight_unchanged": True,
            "release_eligible": compiled.release_eligible,
        }
    )
