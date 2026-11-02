"""Exact, declared normalization of donor money columns.

A country declares, as spec data, how each donor money concept reaches its
destination currency and money period: one exchange rate and one feed-derived
period factor per index family, or a named quantile map that is still owed.
This module validates that declaration and applies it with exact rationals.
It holds no country, currency, rate, factor, series or date.

For a binary64 donor value ``v``:

- ``x`` is the shortest round-trip decimal of ``v`` (``repr``), read exactly.
- The first rounding converts currency. With the rate as the exact rational
  ``num/den`` of its declared decimal, ``to_per_from`` gives
  ``m1 = half_even(x * num/den * 10**digits)`` and ``from_per_to`` gives
  ``m1 = half_even(x * 10**digits * den/num)``; a quoted rate is never
  inverted through a rounded constant.
- The second rounding applies the selected factor ``p / 10**k`` from its
  authenticated receipt: ``m2 = half_even(m1 * p / 10**k)``.
- The rewritten column is ``m2 / 10**digits`` and the integer companion
  ``<column><cents_suffix>`` is ``m2``.

Rounding twice (currency, then factor) can differ from rounding
``x * rate * factor`` once. Each rounding moves a value by at most half a
minor unit, so ``|m2 - half_even(exact * f)| <= floor(1 + f / 2)`` minor
units, which is one minor unit for every factor below 2.

Every normalized column maps through its concept's index family to one
declared factor, or to an explicit identity with a reason; nothing falls back
to a factor of one. Every factor ``money.normalize@1`` applies ages money to
the one declared ``target_period``: the kernel checks each selected factor
through ``money_factor_inputs``, while a direct caller of
``normalize_money_tables`` supplies its factors unchecked. Every money concept
that survives transport is declared exactly once: normalized, quantile mapped
or pending.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping
from datetime import date
from fractions import Fraction

import numpy as np
import pandas as pd

from microcosm.build.ledger_artifact import chronicle_feed_identity
from microcosm.build.ledger_targets import (
    ALLOWED_ASSERTION_POLICIES,
    selector_field_is_supported,
)
from microcosm.build.reference_factors import (
    factor_document_sha256,
    resolve_feed_operand,
)
from microcosm.frame.concepts import (
    CONCEPTS,
    IndexFamily,
    TransportRule,
    Unit,
    concept_for_column,
)
from microcosm.frame.exact_money import (
    _decimal_text as _decimal_text,
)
from microcosm.frame.exact_money import (
    _half_even as _half_even,
)
from microcosm.frame.exact_money import (
    _shortest_decimal as _shortest_decimal,
)

__all__ = [
    "MONEY_FACTOR_PURPOSE",
    "check_money_factor",
    "money_column_rosters",
    "money_concept_columns",
    "money_factor_inputs",
    "normalize_money_tables",
    "normalized_column_families",
    "resolve_money_rate",
    "selected_money_factors",
    "validate_money_normalization",
]

#: The only factor purpose a money normalization may apply.
MONEY_FACTOR_PURPOSE = "money_normalization_input"
_QUOTES = frozenset({"to_per_from", "from_per_to"})
_ROUNDING_MODES = frozenset({"half_even"})
_PERIOD_TYPES = frozenset({"calendar_year"})
_PERIOD_BASES = frozenset({"declared_convention", "authenticated"})
_RATE_FORMS = frozenset({"decimal_text", "feed_selector"})
_MAX_MONEY_DIGITS = 6
_INT64_MIN = -(2**63)
_INT64_MAX = 2**63 - 1
_FIELDS = frozenset(
    {
        "schema_version",
        "from_currency",
        "to_currency",
        "rate",
        "quote",
        "rounding",
        "donor_money_period",
        "target_period",
        "columns",
        "index_family_factors",
        "factor_profile",
        "cents_suffix",
        "quantile_mapped",
        "pending",
    }
)
_DESCRIPTIVE_FIELDS = frozenset({"description", "status", "notes"})
_CURRENCY = re.compile(r"[A-Z]{3}")
_SUFFIX = re.compile(r"_[a-z0-9_]+")
_ROSTERS = ("normalized", "quantile_mapped", "pending")


def _mapping(value: object, label: str) -> dict:
    if not isinstance(value, Mapping):
        raise ValueError(f"Money normalization {label} must be an object.")
    return dict(value)


def _closed(value: object, fields: frozenset, label: str) -> dict:
    row = _mapping(value, label)
    unknown = sorted(row.keys() - fields)
    if unknown:
        raise ValueError(f"Money normalization {label} has unknown fields {unknown}.")
    missing = sorted(fields - row.keys())
    if missing:
        raise ValueError(f"Money normalization {label} is missing fields {missing}.")
    return row


def _text(value: object, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Money normalization {label} must be nonempty text.")
    return value


def _rate(value: object) -> dict:
    row = _mapping(value, "rate")
    form = row.get("form")
    if form not in _RATE_FORMS:
        raise ValueError(
            f"Money normalization rate form must be one of {sorted(_RATE_FORMS)}."
        )
    if form == "decimal_text":
        row = _closed(row, frozenset({"form", "text", "citation"}), "rate")
        _decimal_text(row["text"], "rate text")
        citation = _mapping(row["citation"], "rate citation")
        if not {"publisher", "title", "url"} <= citation.keys() or any(
            not isinstance(item, str) or not item for item in citation.values()
        ):
            raise ValueError(
                "Money normalization rate citation needs publisher, title and url "
                "as nonempty text."
            )
        return row
    row = _closed(
        row,
        frozenset(
            {
                "form",
                "selector",
                "validation",
                "assertion_policy",
                "feed_identity",
                "expected_decimal",
            }
        ),
        "rate",
    )
    selector = _mapping(row["selector"], "rate selector")
    if not selector or any(not selector_field_is_supported(key) for key in selector):
        raise ValueError(
            "Money normalization rate selector must use the closed selector vocabulary."
        )
    _mapping(row["validation"], "rate validation")
    if row["assertion_policy"] not in ALLOWED_ASSERTION_POLICIES:
        raise ValueError("Unknown money normalization rate assertion policy.")
    row["feed_identity"] = chronicle_feed_identity(
        _mapping(row["feed_identity"], "rate feed identity")
    )
    _decimal_text(row["expected_decimal"], "rate expected_decimal")
    return row


def _period(value: object) -> dict:
    row = _closed(
        value,
        frozenset({"type", "value", "basis", "evidence"}),
        "donor_money_period",
    )
    if row["type"] not in _PERIOD_TYPES:
        raise ValueError(
            f"Money normalization donor period type must be one of {sorted(_PERIOD_TYPES)}."
        )
    if type(row["value"]) is not int or not 1 <= row["value"] <= 9999:
        raise ValueError("Money normalization donor period value must be a year.")
    if row["basis"] not in _PERIOD_BASES:
        raise ValueError(
            "Money normalization donor period basis must be one of "
            f"{sorted(_PERIOD_BASES)}."
        )
    evidence = row["evidence"]
    if not isinstance(evidence, list) or not evidence:
        raise ValueError(
            "Money normalization donor period evidence must cite its basis."
        )
    for citation in evidence:
        _text(citation, "donor period evidence")
    return row


def _period_coverage(period: Mapping) -> tuple[str, str]:
    year = period["value"]
    return date(year, 1, 1).isoformat(), date(year, 12, 31).isoformat()


def _money_concept(entity: object, column: object, label: str):
    if not isinstance(entity, str) or not isinstance(column, str) or not column:
        raise ValueError(f"Money normalization {label} must name entity columns.")
    item = concept_for_column(entity, column)
    if item is None or item.unit is not Unit.BASE_CURRENCY:
        raise ValueError(
            f"Money normalization {label} {entity}.{column} is not a money concept."
        )
    return item


def _columns(value: object) -> dict[str, list[str]]:
    columns = _mapping(value, "columns")
    if not columns:
        raise ValueError("Money normalization columns must name normalized columns.")
    result = {}
    for entity, names in columns.items():
        if (
            not isinstance(names, list)
            or not names
            or len(set(map(str, names))) != len(names)
        ):
            raise ValueError(
                "Money normalization columns must be nonempty lists of distinct names."
            )
        for name in names:
            item = _money_concept(entity, name, "column")
            if item.transport is TransportRule.DROP:
                # Quantile maps and pending rows may still name a dropped
                # concept; a normalizer would only fail later, at run time.
                raise ValueError(
                    f"Money normalization column {entity}.{name} is dropped in "
                    "transport, so it cannot be normalized."
                )
        result[entity] = list(names)
    return result


def _roster(value: object, label: str) -> dict[str, dict[str, dict]]:
    roster = _mapping(value, label)
    result = {}
    for entity, rows in roster.items():
        rows = _mapping(rows, f"{label} entity")
        if not rows:
            raise ValueError(f"Money normalization {label} lists an empty entity.")
        result[entity] = {}
        for column, row in rows.items():
            _money_concept(entity, column, label)
            row = _closed(row, frozenset({"awaits", "reason"}), f"{label} entry")
            awaits = row["awaits"]
            if (
                not isinstance(awaits, list)
                or not awaits
                or any(not isinstance(item, str) or not item for item in awaits)
                or len(set(awaits)) != len(awaits)
            ):
                raise ValueError(
                    f"Money normalization {label} entry must name the references "
                    "it awaits."
                )
            _text(row["reason"], f"{label} reason")
            result[entity][column] = row
    return result


def _profile(value: object) -> dict | None:
    if value is None:
        return None
    row = _closed(
        value, frozenset({"dimension", "profiles", "central_profile"}), "factor_profile"
    )
    _text(row["dimension"], "factor profile dimension")
    profiles = row["profiles"]
    if (
        not isinstance(profiles, list)
        or not profiles
        or any(not isinstance(name, str) or not name for name in profiles)
        or len(set(profiles)) != len(profiles)
    ):
        raise ValueError("Money normalization factor profiles must be distinct names.")
    if row["central_profile"] not in profiles:
        raise ValueError("Money normalization central profile must be a profile.")
    return row


def _family_entry(value: object, family: str, profile: dict | None) -> dict:
    row = _mapping(value, f"index family {family!r}")
    if "identity" in row:
        row = _closed(row, frozenset({"identity", "reason"}), f"identity {family!r}")
        if row["identity"] is not True:
            raise ValueError(
                f"Money normalization identity family {family!r} must declare "
                "identity true."
            )
        _text(row["reason"], f"identity reason for {family!r}")
        return row
    if profile is None:
        row = _closed(
            row, frozenset({"factor_id", "disclosure"}), f"index family {family!r}"
        )
        _text(row["factor_id"], f"factor id for {family!r}")
    else:
        row = _closed(
            row, frozenset({"factors", "disclosure"}), f"index family {family!r}"
        )
        factors = _mapping(row["factors"], f"factors for {family!r}")
        if set(factors) != set(profile["profiles"]):
            raise ValueError(
                f"Money normalization family {family!r} must name one factor for "
                "every declared profile."
            )
        for name in factors.values():
            _text(name, f"factor id for {family!r}")
    _text(row["disclosure"], f"disclosure for {family!r}")
    return row


def money_concept_columns() -> dict[tuple[str, str], object]:
    """Every money concept column, keyed ``(entity, column)``."""
    return {
        (item.entity, item.name): item
        for item in CONCEPTS
        if item.unit is Unit.BASE_CURRENCY
    }


def validate_money_normalization(document: object) -> dict:
    """Validate a complete declaration; every field is required.

    Descriptive ``description``, ``status`` and ``notes`` text may ride along
    and is returned unchanged; nothing reads it.
    """
    row = _mapping(document, "declaration")
    descriptive = {key: row.pop(key) for key in _DESCRIPTIVE_FIELDS if key in row}
    for key, value in descriptive.items():
        _text(value, key)
    row = _closed(row, _FIELDS, "declaration")
    if type(row["schema_version"]) is not int or row["schema_version"] != 1:
        raise ValueError("Money normalization requires schema_version 1.")
    for key in ("from_currency", "to_currency"):
        if not isinstance(row[key], str) or _CURRENCY.fullmatch(row[key]) is None:
            raise ValueError(f"Money normalization {key} must be an ISO 4217 code.")
    if row["from_currency"] == row["to_currency"]:
        raise ValueError("Money normalization converts between distinct currencies.")
    row["rate"] = _rate(row["rate"])
    if row["quote"] not in _QUOTES:
        raise ValueError(f"Money normalization quote must be one of {sorted(_QUOTES)}.")
    rounding = _closed(row["rounding"], frozenset({"digits", "mode"}), "rounding")
    if (
        type(rounding["digits"]) is not int
        or not 0 <= rounding["digits"] <= _MAX_MONEY_DIGITS
    ):
        raise ValueError(
            f"Money normalization rounding digits must be 0 to {_MAX_MONEY_DIGITS}."
        )
    if rounding["mode"] not in _ROUNDING_MODES:
        raise ValueError("Money normalization rounding mode must be half_even.")
    row["donor_money_period"] = _period(row["donor_money_period"])
    if type(row["target_period"]) is not int or not 1 <= row["target_period"] <= 9999:
        raise ValueError(
            "Money normalization target_period must be the integer period label "
            "its factors age money to."
        )
    row["columns"] = _columns(row["columns"])
    if (
        not isinstance(row["cents_suffix"], str)
        or _SUFFIX.fullmatch(row["cents_suffix"]) is None
    ):
        raise ValueError(
            "Money normalization cents_suffix must be lowercase text starting with _."
        )
    profile = _profile(row["factor_profile"])
    row["quantile_mapped"] = _roster(row["quantile_mapped"], "quantile_mapped")
    row["pending"] = _roster(row["pending"], "pending")
    seen: dict[tuple[str, str], str] = {}
    for roster, entity, column in (
        *(
            ("normalized", entity, column)
            for entity, names in row["columns"].items()
            for column in names
        ),
        *(
            (name, entity, column)
            for name in ("quantile_mapped", "pending")
            for entity, rows in row[name].items()
            for column in rows
        ),
    ):
        key = (entity, column)
        if key in seen:
            raise ValueError(
                f"Money column {entity}.{column} is declared both {seen[key]} and "
                f"{roster}; each column has exactly one producer route."
            )
        seen[key] = roster
        companion = f"{column}{row['cents_suffix']}"
        if concept_for_column(entity, companion) is not None:
            raise ValueError(
                f"Money normalization companion {entity}.{companion} collides with "
                "a concept column."
            )
    unclassified = sorted(
        f"{entity}.{column}"
        for (entity, column), item in money_concept_columns().items()
        if item.transport is not TransportRule.DROP and (entity, column) not in seen
    )
    if unclassified:
        raise ValueError(
            "Money normalization must declare every transported money concept as "
            f"normalized, quantile_mapped or pending; unclassified: {unclassified}."
        )
    families = _mapping(row["index_family_factors"], "index_family_factors")
    used = {
        money_concept_columns()[entity, column].monetary.index_family.value
        for entity, names in row["columns"].items()
        for column in names
    }
    known = {item.value for item in IndexFamily}
    if set(families) - known:
        raise ValueError(
            f"Money normalization names unknown index families {sorted(set(families) - known)}."
        )
    unmapped = sorted(used - set(families))
    if unmapped:
        raise ValueError(
            f"Money normalization index families {unmapped} have no declared factor "
            "or explicit identity; a family is never silently left at one."
        )
    unused = sorted(set(families) - used)
    if unused:
        raise ValueError(
            f"Money normalization declares index families {unused} that no "
            "normalized column uses."
        )
    row["index_family_factors"] = {
        family: _family_entry(entry, family, profile)
        for family, entry in families.items()
    }
    row["factor_profile"] = profile
    return {**row, **descriptive}


def money_column_rosters(declaration: Mapping) -> dict[tuple[str, str], tuple]:
    """``(entity, column)`` to ``(route, awaits)`` for every declared column."""
    result = {
        (entity, column): ("normalized", ())
        for entity, names in declaration["columns"].items()
        for column in names
    }
    for route in ("quantile_mapped", "pending"):
        for entity, rows in declaration[route].items():
            for column, row in rows.items():
                result[entity, column] = (route, tuple(row["awaits"]))
    return result


def normalized_column_families(declaration: Mapping) -> dict[tuple[str, str], str]:
    """Each normalized column's index family, read from the concept schema."""
    concepts = money_concept_columns()
    return {
        (entity, column): concepts[entity, column].monetary.index_family.value
        for entity, names in declaration["columns"].items()
        for column in names
    }


def selected_money_factors(
    declaration: Mapping, profile_selection: Mapping
) -> tuple[str | None, dict[str, str | None]]:
    """The selected profile and each family's factor id (``None``: identity).

    A selection that omits the declared dimension selects its central profile,
    as a scenario that leaves a knob at its central value does.
    """
    selection = _mapping(profile_selection, "profile selection")
    profile = declaration["factor_profile"]
    chosen = None
    if profile is not None:
        chosen = selection.get(profile["dimension"], profile["central_profile"])
        if chosen not in profile["profiles"]:
            raise ValueError(
                f"Unknown money normalization profile {chosen!r} for "
                f"{profile['dimension']!r}."
            )
    factors = {}
    for family, entry in declaration["index_family_factors"].items():
        if entry.get("identity") is True:
            factors[family] = None
        elif profile is None:
            factors[family] = entry["factor_id"]
        else:
            factors[family] = entry["factors"][chosen]
    return chosen, factors


def check_money_factor(entry: Mapping, declaration: Mapping) -> tuple[int, int]:
    """Return ``(p, k)`` for a factor ``p / 10**k`` fit for this declaration.

    The factor must carry the money-normalization purpose, its source
    schedule must cover exactly the declared donor money period, and it must
    age money to the declared ``target_period``, so every family reaches the
    same destination period.
    """
    factor_declaration = entry["declaration"]
    if factor_declaration["purpose"] != MONEY_FACTOR_PURPOSE:
        raise ValueError(
            f"Factor {factor_declaration['id']!r} has purpose "
            f"{factor_declaration['purpose']!r}; money normalization applies only "
            f"{MONEY_FACTOR_PURPOSE} factors."
        )
    target = factor_declaration["target_period"]
    if type(target) is not int or target != declaration["target_period"]:
        raise ValueError(
            f"Factor {factor_declaration['id']!r} ages money to {target!r}, not "
            f"the declared target_period {declaration['target_period']!r}."
        )
    schedule = factor_declaration["source_schedule"]
    coverage = (
        schedule[0]["validation"]["period_coverage"]["start_date"],
        schedule[-1]["validation"]["period_coverage"]["end_date"],
    )
    if coverage != _period_coverage(declaration["donor_money_period"]):
        raise ValueError(
            f"Factor {factor_declaration['id']!r} source schedule covers "
            f"{coverage[0]}..{coverage[1]}, not the declared donor money period."
        )
    digits = entry["receipt"]["rounding"]["digits"]
    rational = entry["operational_factor"]
    if (
        type(digits) is not int
        or rational["denominator"] != 10**digits
        or type(rational["numerator"]) is not int
        or rational["numerator"] <= 0
    ):
        raise ValueError(
            "A money factor must be a positive integer over 10**digits from its receipt."
        )
    return rational["numerator"], digits


def resolve_money_rate(
    declaration: Mapping,
    facts: Iterable | None = None,
    feed_identity: Mapping | None = None,
) -> tuple[Fraction, dict]:
    """The exact rate and its evidence; a feed rate is replayed from facts."""
    rate = declaration["rate"]
    if rate["form"] == "decimal_text":
        return _decimal_text(rate["text"], "rate text"), {
            "form": "decimal_text",
            "text": rate["text"],
            "citation": dict(rate["citation"]),
        }
    if facts is None or feed_identity is None:
        raise ValueError("A feed-selected rate needs the authenticated feed facts.")
    if rate["feed_identity"] != chronicle_feed_identity(feed_identity):
        raise ValueError(
            "Money rate feed identity mismatch; re-declaration is required."
        )
    value, evidence = resolve_feed_operand(
        {"selector": rate["selector"], "validation": rate["validation"]},
        facts,
        assertion_policy=rate["assertion_policy"],
    )
    if value != _decimal_text(rate["expected_decimal"], "rate expected_decimal"):
        raise ValueError(
            "Money rate replay differs from its declared expected decimal; "
            "re-declaration is required."
        )
    return value, {
        "form": "feed_selector",
        "text": rate["expected_decimal"],
        "operand": evidence,
    }


def _rational(value: Fraction) -> dict:
    return {"numerator": value.numerator, "denominator": value.denominator}


def _exact_decimal(value: Fraction) -> str:
    """Exact decimal text of a rational whose denominator divides 10**k."""
    denominator, places = value.denominator, 0
    while denominator % 10 == 0:
        denominator //= 10
        places += 1
    while denominator % 2 == 0 or denominator % 5 == 0:
        factor = 2 if denominator % 2 == 0 else 5
        denominator //= factor
        places += 1
    if denominator != 1:
        raise ValueError("A money sum is always a terminating decimal.")
    scaled = int(value * Fraction(10**places))
    sign = "-" if scaled < 0 else ""
    digits = str(abs(scaled)).rjust(places + 1, "0")
    return sign + (digits if not places else f"{digits[:-places]}.{digits[-places:]}")


def _minor_units(values: Iterable[int], digits: int) -> str:
    total = sum(values)
    sign = "-" if total < 0 else ""
    text = str(abs(total)).rjust(digits + 1, "0")
    return sign + (text if not digits else f"{text[:-digits]}.{text[-digits:]}")


def normalize_money_tables(
    tables: Mapping[str, pd.DataFrame],
    declaration: Mapping,
    *,
    rate: Fraction,
    rate_text: str,
    factors: Mapping[str, Mapping],
) -> tuple[dict[tuple[str, str], np.ndarray], list[dict]]:
    """Normalize every declared column once; return arrays and per-column receipts.

    ``factors`` maps each used family to ``{"identity": True, "reason"}`` or
    ``{"factor_id", "numerator", "digits", "declaration_sha256",
    "receipt_sha256"}``. A table already carrying a declared companion column
    was normalized before and is refused, as is a NaN, an infinite value, a
    negative amount in an unsigned concept, or a minor-unit total outside int64.
    """
    if not isinstance(rate, Fraction) or rate <= 0:
        raise ValueError("Money normalization needs a positive exact rate.")
    digits = declaration["rounding"]["digits"]
    scale = 10**digits
    exchange = rate if declaration["quote"] == "to_per_from" else Fraction(1) / rate
    minor_exchange = exchange * Fraction(scale)
    suffix = declaration["cents_suffix"]
    concepts = money_concept_columns()
    families = normalized_column_families(declaration)
    period = dict(declaration["donor_money_period"])
    arrays: dict[tuple[str, str], np.ndarray] = {}
    receipts = []
    for (entity, column), family in families.items():
        table = tables[entity]
        companion = f"{column}{suffix}"
        if companion in table:
            raise ValueError(
                f"{entity}.{column} already carries {companion}; money is normalized "
                "exactly once per lineage."
            )
        if column not in table:
            raise ValueError(f"Money column {entity}.{column} is absent.")
        factor = factors[family]
        identity = factor.get("identity") is True
        ratio = (
            Fraction(1)
            if identity
            else Fraction(factor["numerator"], 10 ** factor["digits"])
        )
        signed = concepts[entity, column].monetary.signed
        before = Fraction(0)
        first, second = [], []
        for value in table[column].tolist():
            amount = _shortest_decimal(value)
            if amount < 0 and not signed:
                raise ValueError(
                    f"Unsigned money column {entity}.{column} holds a negative amount."
                )
            before += amount
            converted = _half_even(amount * minor_exchange)
            aged = _half_even(Fraction(converted) * ratio)
            if any(
                not _INT64_MIN <= value <= _INT64_MAX for value in (converted, aged)
            ):
                raise ValueError(
                    f"Money column {entity}.{column} exceeds int64 minor units."
                )
            first.append(converted)
            second.append(aged)
        # Residuals are linear: sum(m1 - x*q) = sum(m1) - q*sum(x), exactly.
        fx_residual = Fraction(sum(first)) - before * minor_exchange
        factor_residual = Fraction(sum(second)) - Fraction(sum(first)) * ratio
        arrays[entity, column] = np.array(
            [float(Fraction(units, scale)) for units in second], dtype=np.float64
        )
        arrays[entity, companion] = np.array(second, dtype=np.int64)
        factor_receipt = (
            {"identity": True, "reason": factor["reason"]}
            if factor.get("identity") is True
            else {
                "factor_id": factor["factor_id"],
                "declaration_sha256": factor["declaration_sha256"],
                "receipt_sha256": factor["receipt_sha256"],
                "operational_factor": {
                    "numerator": factor["numerator"],
                    "denominator": 10 ** factor["digits"],
                },
            }
        )
        receipts.append(
            {
                "entity": entity,
                "column": column,
                "cents_column": companion,
                "index_family": family,
                "n": len(second),
                "sum_before": _exact_decimal(before),
                "sum_after_fx": _minor_units(first, digits),
                "sum_after": _minor_units(second, digits),
                "fx_rounding_residual_minor_units": _rational(fx_residual),
                "factor_rounding_residual_minor_units": _rational(factor_residual),
                "factor": factor_receipt,
                "rate_text": rate_text,
                "quote": declaration["quote"],
                "donor_money_period": period,
            }
        )
    return arrays, receipts


def money_factor_inputs(entry: Mapping, declaration: Mapping) -> dict:
    """The arithmetic inputs and identities of one checked, selected factor."""
    numerator, digits = check_money_factor(entry, declaration)
    return {
        "factor_id": entry["declaration"]["id"],
        "numerator": numerator,
        "digits": digits,
        "declaration_sha256": factor_document_sha256(entry["declaration"]),
        "receipt_sha256": entry["receipt"]["receipt_sha256"],
    }
