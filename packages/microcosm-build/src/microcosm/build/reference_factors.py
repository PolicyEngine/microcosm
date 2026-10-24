"""Closed, declaration-driven factor calculations with exact feed operands.

The operational rational is rounded once. Irrational roots remain certified
intervals in the evidence; they are never represented as exact rationals.
Composition checks declared evidence; compilation must replay it against the
facts returned by the authenticated artifact loader.
"""

from __future__ import annotations

import hashlib
from collections.abc import Iterable, Mapping
from datetime import date, timedelta
from decimal import Decimal, InvalidOperation
from fractions import Fraction

from microcosm.build.ledger_artifact import LexicalFloat, chronicle_feed_identity
from microcosm.build.ledger_targets import (
    ALLOWED_ASSERTION_POLICIES,
    fact_matches_selector,
    selector_field_is_supported,
)
from microcosm.graph.canonical import canonical_json

OPERATION = "projected_period_average_ratio"
OPERATION_VERSION = 1
# Exact powers in projected receipts grow with both schedule size and root
# precision. Keep ordinary schedules within the stock integer serialization
# budget, including when rounding needs a finer root enclosure.
_MAX_SUBPERIODS_PER_YEAR = 12
_MAX_ROUNDING_DIGITS = 18
_MAX_RECEIPT_ROOT_PLACES = 256
_DECLARATION_FIELDS = frozenset(
    {
        "schema_version",
        "id",
        "operation",
        "operation_version",
        "purpose",
        "feed_identity",
        "source_period",
        "target_period",
        "source_schedule",
        "target_schedule",
        "growth",
        "subperiods_per_year",
        "projection_policy",
        "rounding",
        "expected_operational_factor",
        "validity",
        "evidence",
        "decision_references",
    }
)
_VALIDATION_FIELDS = frozenset(
    {
        "unit",
        "source_sha256",
        "source_vintage",
        "entity_role",
        "provenance_class",
        "assertion",
        "period_coverage",
        "source_column_id",
    }
)


def factor_document_sha256(document: object) -> str:
    """Hash canonical declaration/evidence bytes without a registry identity."""
    return hashlib.sha256(canonical_json(document)).hexdigest()


def _mapping(value: object, label: str) -> dict:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be an object.")
    return dict(value)


def _closed(value: object, fields: set | frozenset, label: str) -> dict:
    result = _mapping(value, label)
    unknown = result.keys() - fields
    if unknown:
        raise ValueError(f"Unknown {label} fields: {sorted(unknown)}.")
    return result


def _closed_feed_identity(value: object, label: str) -> dict:
    identity = _mapping(value, label)
    normalized = chronicle_feed_identity(identity)
    unknown = identity.keys() - normalized.keys()
    if unknown:
        raise ValueError(f"Unknown {label} fields: {sorted(unknown)}.")
    return normalized


def _list(value: object, label: str) -> list:
    if not isinstance(value, (list, tuple)) or not value:
        raise ValueError(f"{label} must be a nonempty list.")
    return list(value)


def _decimal(value: object, label: str) -> Fraction:
    if isinstance(value, bool) or not isinstance(value, (str, int, Decimal)):
        raise ValueError(f"{label} requires a finite lexical decimal, never a float.")
    try:
        result = Decimal(value)
    except (InvalidOperation, ValueError) as exc:
        raise ValueError(f"{label} requires a finite decimal.") from exc
    if not result.is_finite():
        raise ValueError(f"{label} requires a finite decimal.")
    return Fraction(result)


def _rational(value: Fraction) -> dict:
    return {"numerator": value.numerator, "denominator": value.denominator}


def _read_rational(value: object, label: str) -> Fraction:
    row = _closed(value, {"numerator", "denominator"}, label)
    numerator, denominator = row.get("numerator"), row.get("denominator")
    if type(numerator) is not int or type(denominator) is not int or denominator <= 0:
        raise ValueError(
            f"{label} requires integer numerator and positive denominator."
        )
    return Fraction(numerator, denominator)


def _integer_root(value: int, degree: int) -> int:
    lower, upper = 0, 1 << ((value.bit_length() + degree - 1) // degree)
    while lower < upper:
        middle = (lower + upper + 1) // 2
        if middle**degree <= value:
            lower = middle
        else:
            upper = middle - 1
    return lower


def certified_nth_root_bounds(
    value: Fraction, degree: int, decimal_places: int
) -> tuple[Fraction, Fraction, bool]:
    """Enclose a positive root on an integer grid; mark only exact roots exact."""
    if value <= 0 or type(degree) is not int or degree <= 0:
        raise ValueError("Root radicand and degree must be positive.")
    if type(decimal_places) is not int or not 0 <= decimal_places <= 4096:
        raise ValueError("Root precision must be between 0 and 4096.")
    numerator_root = _integer_root(value.numerator, degree)
    denominator_root = _integer_root(value.denominator, degree)
    if (
        numerator_root**degree == value.numerator
        and denominator_root**degree == value.denominator
    ):
        exact = Fraction(numerator_root, denominator_root)
        return exact, exact, True
    scale = 10**decimal_places
    floor = _integer_root(value.numerator * scale**degree // value.denominator, degree)
    return Fraction(floor, scale), Fraction(floor + 1, scale), False


def _round_nearest(value: Fraction, digits: int) -> int:
    scaled = value * 10**digits
    quotient, remainder = divmod(scaled.numerator, scaled.denominator)
    twice = 2 * remainder
    return quotient + (
        twice > scaled.denominator
        or (twice == scaled.denominator and quotient % 2 == 1)
    )


def _declaration(declaration: object) -> dict:
    row = _closed(declaration, _DECLARATION_FIELDS, "factor declaration")
    missing = _DECLARATION_FIELDS - row.keys()
    if missing:
        raise ValueError(f"Factor declaration missing fields: {sorted(missing)}.")
    if type(row["schema_version"]) is not int or row["schema_version"] != 1:
        raise ValueError("Factor declaration requires schema_version 1.")
    if (
        row["operation"] != OPERATION
        or type(row["operation_version"]) is not int
        or row["operation_version"] != OPERATION_VERSION
    ):
        raise ValueError("Unknown factor operation or operation version.")
    if not isinstance(row["id"], str) or not row["id"]:
        raise ValueError("Factor id must be nonempty text.")
    if row["purpose"] not in {"profile_input", "diagnostic"}:
        raise ValueError("Unknown factor purpose.")
    for field in ("source_period", "target_period"):
        label = row[field]
        if type(label) not in {str, int} or isinstance(label, str) and not label:
            raise ValueError("Factor period labels require nonempty text or integers.")
    if row["projection_policy"] != "observed_preferred_latest_contiguous":
        raise ValueError("Unknown factor projection policy.")
    frequency = row["subperiods_per_year"]
    if type(frequency) is not int or not 1 <= frequency <= _MAX_SUBPERIODS_PER_YEAR:
        raise ValueError(
            "subperiods_per_year must be an integer from 1 to "
            f"{_MAX_SUBPERIODS_PER_YEAR} for serializable exact receipts."
        )
    rounding = _closed(row["rounding"], {"digits", "mode"}, "factor rounding")
    if (
        type(rounding.get("digits")) is not int
        or not 0 <= rounding["digits"] <= _MAX_ROUNDING_DIGITS
    ):
        raise ValueError(
            "Factor rounding digits must be an integer from 0 to "
            f"{_MAX_ROUNDING_DIGITS} for serializable exact receipts."
        )
    if rounding.get("mode") != "nearest_half_even":
        raise ValueError("Factor rounding must be nearest_half_even.")
    expected = _read_rational(row["expected_operational_factor"], "expected factor")
    if (
        expected <= 0
        or row["expected_operational_factor"]["denominator"] != 10 ** rounding["digits"]
    ):
        raise ValueError("Expected operational factor must be positive p / 10**digits.")
    validity = _closed(
        row["validity"],
        {"declared_on", "valid_from", "valid_until", "as_of"},
        "factor validity",
    )
    try:
        dates = {
            key: date.fromisoformat(validity[key])
            for key in ("declared_on", "valid_from", "valid_until", "as_of")
        }
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("Factor validity requires explicit ISO dates.") from exc
    if (
        not dates["declared_on"] <= dates["as_of"]
        or not dates["valid_from"] <= dates["as_of"] <= dates["valid_until"]
    ):
        raise ValueError(
            "Factor declaration is expired or not yet valid at its declared as_of."
        )
    if not dates["valid_from"] <= date.today() <= dates["valid_until"]:
        raise ValueError("Factor declaration is expired or not yet valid today.")
    for key in ("evidence", "decision_references"):
        if any(not isinstance(item, str) or not item for item in _list(row[key], key)):
            raise ValueError(f"{key} must contain nonempty text.")
    row["feed_identity"] = _closed_feed_identity(
        row["feed_identity"], "factor declaration feed identity"
    )
    return row


def _schedule(value: object, frequency: int, *, target: bool) -> list[dict]:
    rows = _list(value, "factor schedule")
    if len(rows) != frequency:
        raise ValueError("Factor schedule must contain every declared subperiod once.")
    result, previous_end, periods = [], None, set()
    for index, raw in enumerate(rows):
        fields = {"selector", "validation", "weight", "sequence"}
        if target:
            fields.add("projection_offset")
        row = _closed(raw, fields, "factor schedule slot")
        if type(row.get("sequence")) is not int or row["sequence"] != index:
            raise ValueError("Factor schedule is misordered or noncontiguous.")
        if target and (
            type(row.get("projection_offset")) is not int
            or row["projection_offset"] != index
        ):
            raise ValueError("Invalid projection offsets.")
        selector = _mapping(row.get("selector"), "factor selector")
        if not selector or any(
            not selector_field_is_supported(key) for key in selector
        ):
            raise ValueError(
                "Unknown or empty factor selector; use the closed selector vocabulary."
            )
        if "period_value" not in selector or isinstance(
            selector["period_value"], (list, tuple)
        ):
            raise ValueError("Each schedule slot must select one explicit period.")
        period = str(selector["period_value"])
        if period in periods:
            raise ValueError("Duplicate factor schedule period.")
        periods.add(period)
        validation = _validation(row.get("validation"))
        coverage = validation["period_coverage"]
        try:
            start, end = (
                date.fromisoformat(coverage["start_date"]),
                date.fromisoformat(coverage["end_date"]),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("Factor schedules require ISO period coverage.") from exc
        if start > end or (
            previous_end is not None and start != previous_end + timedelta(days=1)
        ):
            raise ValueError("Factor schedule coverage is misordered or noncontiguous.")
        previous_end = end
        if _decimal(row.get("weight"), "schedule weight") <= 0:
            raise ValueError("Averaging weights must be positive.")
        result.append(row)
    return result


def _validation(value: object) -> dict:
    result = _closed(value, _VALIDATION_FIELDS, "factor operand validation")
    required = _VALIDATION_FIELDS - {"source_column_id"}
    if required - result.keys():
        raise ValueError(
            "Factor operands require unit, source hash/vintage, role, assertion, provenance and coverage."
        )
    if result["assertion"] not in {"observation", "source_projection"}:
        raise ValueError("Unknown factor assertion.")
    coverage = _closed(
        result["period_coverage"],
        {"start_date", "end_date", "basis", "source_period_label", "notes"},
        "factor operand coverage",
    )
    if not {"start_date", "end_date"} <= coverage.keys():
        raise ValueError("Factor operand coverage requires both endpoints.")
    for key in required - {"period_coverage"}:
        if not isinstance(result[key], str) or not result[key]:
            raise ValueError("Factor operand validations require nonempty text.")
    source_hash = result["source_sha256"]
    if len(source_hash) != 64 or any(
        character not in "0123456789abcdef" for character in source_hash
    ):
        raise ValueError("Factor operand source hash must be a lowercase SHA-256.")
    return result


def _operand(
    slot: dict,
    facts: tuple,
    *,
    optional: bool = False,
    assertion_policy: str = "observed_only",
) -> tuple[Fraction, dict] | None:
    selector = slot["selector"]
    # Exact keys do not make duplicate same-period source vintages unambiguous.
    broad = {
        key: value
        for key, value in selector.items()
        if key
        not in {
            "aggregate_fact_key",
            "semantic_fact_key",
            "legacy_fact_key",
            "source_record_id",
            "record_set_id",
            "layout_record_set_id",
            "record_set_spec_id",
            "layout_record_set_spec_id",
        }
    }
    if (
        "period_value" in broad
        and sum(fact_matches_selector(fact, broad) for fact in facts) > 1
    ):
        raise ValueError("Duplicate factor period across source vintages or facts.")
    matches = [fact for fact in facts if fact_matches_selector(fact, selector)]
    if not matches and optional:
        return None
    if len(matches) != 1:
        raise ValueError(
            f"Factor selector must resolve exactly once; found {len(matches)}."
        )
    fact = matches[0]
    validation = _validation(slot["validation"])
    actual = {
        "unit": fact.get("observed_measure", {}).get("unit"),
        "source_sha256": fact.get("source", {}).get("source_sha256"),
        "source_vintage": fact.get("source", {}).get("vintage"),
        "entity_role": fact.get("entity", {}).get("role"),
        "provenance_class": fact.get("provenance_class"),
        "assertion": fact.get("assertion", "observation"),
        "period_coverage": fact.get("period_coverage", {}),
        "source_column_id": fact.get("layout", {}).get("source_column_id"),
    }
    for key, expected in validation.items():
        if key == "period_coverage":
            if any(actual[key].get(field) != item for field, item in expected.items()):
                raise ValueError("Factor operand period coverage mismatch.")
        elif actual[key] != expected:
            raise ValueError(f"Factor operand {key} mismatch.")
    # Reuse the existing target assertion-policy vocabulary after all match paths.
    if (
        actual["assertion"] == "source_projection"
        and assertion_policy != "allow_source_projection"
    ):
        raise ValueError("Source projections require allow_source_projection.")
    value = fact.get("value")
    if isinstance(value, bool):
        raise ValueError("Boolean factor operand is not numeric.")
    if isinstance(value, LexicalFloat):
        token = value.lexical_token
        recovered = str(float(value))
        if _decimal(token, "feed numeric token") != _decimal(
            recovered, "float decimal recovery"
        ):
            raise ValueError(
                "Float conversion does not reproduce the feed lexical numeric token."
            )
        proof = {
            "numeric_representation": "binary_float_with_lexical_token",
            "lexical_token": token,
            "recovered_decimal": recovered,
            "decimal_values_equal": True,
        }
    elif type(value) is int:
        token = str(value)
        proof = {
            "numeric_representation": "exact_integer",
            "lexical_token": token,
            "recovered_decimal": token,
            "decimal_values_equal": True,
        }
    elif isinstance(value, Decimal):
        token = str(value)
        proof = {
            "numeric_representation": "exact_decimal",
            "lexical_token": token,
            "recovered_decimal": token,
            "decimal_values_equal": True,
        }
    else:
        raise ValueError(
            "Factor operand lacks an authenticated lexical numeric token; "
            "load the artifact with lexical_floats=True."
        )
    numeric = _decimal(token, "feed numeric token")
    return numeric, {
        "selector": selector,
        "aggregate_fact_key": fact.get("aggregate_fact_key"),
        "decimal_operand": token,
        "numeric_token_proof": proof,
        "validation": validation,
        "period": fact.get("period"),
        "source": fact.get("source"),
        "unit": actual["unit"],
        "assertion": actual["assertion"],
        "provenance_class": actual["provenance_class"],
        "entity_role": actual["entity_role"],
        "period_coverage": actual["period_coverage"],
    }


def resolve_reference_factor(
    declaration: Mapping, facts: Iterable, feed_identity: Mapping
) -> tuple[dict, dict]:
    """Resolve one closed operation, returning its operational rational and proof."""
    row = _declaration(declaration)
    feed_identity = chronicle_feed_identity(
        _mapping(feed_identity, "factor feed identity")
    )
    if canonical_json(row["feed_identity"]) != canonical_json(feed_identity):
        raise ValueError("Factor feed identity mismatch; re-declaration is required.")
    fact_rows = tuple(facts)
    frequency = row["subperiods_per_year"]
    source = _schedule(row["source_schedule"], frequency, target=False)
    target = _schedule(row["target_schedule"], frequency, target=True)
    source_values, source_evidence = [], []
    for slot in source:
        if slot["validation"]["assertion"] != "observation":
            raise ValueError("Source schedule requires observations.")
        value, evidence = _operand(slot, fact_rows)
        if value <= 0:
            raise ValueError("Factor denominator operands must be positive.")
        source_values.append((value, _decimal(slot["weight"], "source weight")))
        evidence.update({"weight": slot["weight"], "sequence": slot["sequence"]})
        source_evidence.append(evidence)
    denominator = sum(value * weight for value, weight in source_values) / sum(
        weight for _, weight in source_values
    )
    growth = row["growth"]
    growth_evidence, effective_growth = None, Fraction(0)
    if growth is not None:
        growth = _closed(
            growth,
            {
                "selector",
                "validation",
                "unit_multiplier",
                "scenario_delta",
                "sensitivity_provenance",
                "assertion_policy",
            },
            "factor growth",
        )
        _validation(growth.get("validation"))
        selector = _mapping(growth.get("selector"), "growth selector")
        if not selector or any(
            not selector_field_is_supported(key) for key in selector
        ):
            raise ValueError("Unknown growth selector field.")
        policy = growth.get("assertion_policy")
        if policy not in ALLOWED_ASSERTION_POLICIES:
            raise ValueError("Unknown growth assertion policy.")
        growth_value, growth_evidence = _operand(
            growth, fact_rows, assertion_policy=policy
        )
        multiplier = _decimal(growth.get("unit_multiplier"), "growth unit multiplier")
        delta = _decimal(growth.get("scenario_delta"), "scenario delta")
        if multiplier <= 0:
            raise ValueError("Growth unit multiplier must be positive.")
        if any(
            not isinstance(item, str) or not item
            for item in _list(
                growth.get("sensitivity_provenance"), "sensitivity provenance"
            )
        ):
            raise ValueError("Missing sensitivity provenance.")
        effective_growth = (growth_value + delta) * multiplier
        growth_evidence.update(
            {
                "unit_multiplier": growth["unit_multiplier"],
                "scenario_delta": growth["scenario_delta"],
                "sensitivity_provenance": growth["sensitivity_provenance"],
                "effective_growth": _rational(effective_growth),
            }
        )
    if 1 + effective_growth <= 0:
        raise ValueError("Effective annual growth must have positive 1 + g.")
    observations, target_evidence, gap = [], [], False
    for slot in target:
        if slot["validation"]["assertion"] != "observation":
            raise ValueError("Target schedules prefer observations only.")
        resolved = _operand(slot, fact_rows, optional=slot["sequence"] != 0)
        if resolved is None:
            gap = True
            observations.append(None)
            target_evidence.append(
                {
                    "selector": slot["selector"],
                    "validation": slot["validation"],
                    "status": "projected",
                }
            )
        else:
            if gap:
                raise ValueError(
                    "Observed target subperiods must be contiguous from the anchor."
                )
            value, evidence = resolved
            if value <= 0:
                raise ValueError("Observed target anchors must be positive.")
            observations.append(value)
            target_evidence.append({**evidence, "status": "observed"})
    if gap and growth is None:
        raise ValueError("Projected target periods require a declared growth operand.")
    last_observed = max(
        index for index, value in enumerate(observations) if value is not None
    )
    anchor = observations[last_observed]
    target_weights = [_decimal(slot["weight"], "target weight") for slot in target]

    def numerator_at(root: Fraction) -> Fraction:
        return sum(
            (value if value is not None else anchor * root ** (index - last_observed))
            * target_weights[index]
            for index, value in enumerate(observations)
        ) / sum(target_weights)

    digits = row["rounding"]["digits"]
    precision = digits + 8
    while precision <= _MAX_RECEIPT_ROOT_PLACES:
        lower, upper, exact = certified_nth_root_bounds(
            1 + effective_growth, frequency, precision
        )
        numerator_lower, numerator_upper = numerator_at(lower), numerator_at(upper)
        factor_lower, factor_upper = (
            numerator_lower / denominator,
            numerator_upper / denominator,
        )
        rounded_lower, rounded_upper = (
            _round_nearest(factor_lower, digits),
            _round_nearest(factor_upper, digits),
        )
        if rounded_lower == rounded_upper:
            break
        precision *= 2
    else:
        raise ValueError(
            "Factor rounding cannot be certified within the supported exact receipt "
            "precision."
        )
    factor = {"numerator": rounded_lower, "denominator": 10**digits}
    if rounded_lower <= 0:
        raise ValueError("Operational factor must be positive.")
    if factor != row["expected_operational_factor"]:
        raise ValueError(
            f"Declared expected factor mismatch: computed {factor}; re-declaration is required."
        )
    for index, evidence in enumerate(target_evidence):
        evidence.update(
            {
                "sequence": index,
                "weight": target[index]["weight"],
                "projection_offset": 0
                if observations[index] is not None
                else index - last_observed,
            }
        )
        if observations[index] is None:
            evidence["projection_anchor"] = {
                "aggregate_fact_key": target_evidence[last_observed][
                    "aggregate_fact_key"
                ]
            }
    receipt = {
        "schema_version": 1,
        "operation": OPERATION,
        "operation_version": OPERATION_VERSION,
        "factor_id": row["id"],
        "declaration_sha256": factor_document_sha256(row),
        "feed_identity": dict(feed_identity),
        "source_period": row["source_period"],
        "target_period": row["target_period"],
        "source_schedule": source_evidence,
        "target_schedule": target_evidence,
        "subperiods_per_year": frequency,
        "projection_policy": row["projection_policy"],
        "denominator_mean": _rational(denominator),
        "growth": growth_evidence,
        "root_bounds": {
            "lower": _rational(lower),
            "upper": _rational(upper),
            "exact": exact,
            "decimal_places": precision,
            "degree": frequency,
            "radicand": _rational(1 + effective_growth),
        },
        "numerator_mean_bounds": {
            "lower": _rational(numerator_lower),
            "upper": _rational(numerator_upper),
        },
        "unrounded_factor_bounds": {
            "lower": _rational(factor_lower),
            "upper": _rational(factor_upper),
        },
        "rounding": row["rounding"],
        "operational_factor": factor,
        "validity": row["validity"],
        "evidence": row["evidence"],
        "decision_references": row["decision_references"],
    }
    receipt["receipt_sha256"] = factor_document_sha256(receipt)
    return factor, receipt


def validate_factor_context(
    context: Mapping, selected_ids: Iterable[str] | None = None
) -> dict[str, dict]:
    """Check selected declared evidence without claiming source authentication.

    Only selected declarations/evidence are validated or returned. Independent
    profile edits therefore cannot change a branch's identity or availability.
    """
    document = _closed(
        context,
        {"schema_version", "factors", "country", "feed_identity"},
        "factor context",
    )
    if (
        type(document.get("schema_version")) is not int
        or document["schema_version"] != 1
    ):
        raise ValueError("Factor context requires schema_version 1.")
    if "feed_identity" in document:
        document["feed_identity"] = _closed_feed_identity(
            document["feed_identity"], "factor context feed identity"
        )
    entries = _list(document.get("factors"), "factor context entries")
    by_id = {}
    for raw in entries:
        entry = _mapping(raw, "factor context entry")
        declaration = _mapping(entry.get("declaration"), "factor declaration")
        factor_id = declaration.get("id")
        if not isinstance(factor_id, str) or not factor_id or factor_id in by_id:
            raise ValueError("Factor ids must be unique nonempty text.")
        by_id[factor_id] = entry
    selected = list(by_id) if selected_ids is None else list(selected_ids)
    result = {}
    for factor_id in selected:
        if factor_id not in by_id:
            raise ValueError(f"Missing selected factor {factor_id!r}.")
        entry = _closed(
            by_id[factor_id],
            {"declaration", "operational_factor", "receipt"},
            "factor context entry",
        )
        declaration = _declaration(entry["declaration"])
        if (
            "feed_identity" in document
            and document["feed_identity"] != declaration["feed_identity"]
        ):
            raise ValueError(
                "Factor context feed identity is incompatible with the declaration."
            )
        factor = entry.get("operational_factor")
        _read_rational(factor, "declared operational factor")
        if factor != declaration["expected_operational_factor"]:
            raise ValueError(
                "Declared operational factor differs from its expectation."
            )
        receipt = _mapping(entry.get("receipt"), "factor receipt")
        body = {key: value for key, value in receipt.items() if key != "receipt_sha256"}
        if receipt.get("receipt_sha256") != factor_document_sha256(body):
            raise ValueError("Factor receipt hash mismatch.")
        if (
            type(receipt.get("schema_version")) is not int
            or receipt.get("schema_version") != 1
            or receipt.get("operation") != declaration["operation"]
            or type(receipt.get("operation_version")) is not int
            or receipt.get("operation_version") != declaration["operation_version"]
            or receipt.get("factor_id") != factor_id
            or receipt.get("declaration_sha256") != factor_document_sha256(declaration)
            or receipt.get("operational_factor") != factor
            or receipt.get("feed_identity") != declaration["feed_identity"]
        ):
            raise ValueError("Factor declaration and receipt are incompatible.")
        result[factor_id] = entry
    return result


def replay_reference_factor(
    declaration: Mapping,
    facts: Iterable,
    feed_identity: Mapping,
    operational_factor: Mapping,
    receipt: Mapping,
) -> None:
    """Authenticate evidence by recomputing it from the loaded feed facts."""
    factor, recomputed = resolve_reference_factor(declaration, facts, feed_identity)
    if factor != operational_factor or canonical_json(recomputed) != canonical_json(
        receipt
    ):
        raise ValueError("Factor replay differs from declared result or evidence.")
