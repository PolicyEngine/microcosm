"""Exact donor normalization, integer money bridges and annual transfers.

``money.normalize@1`` rewrites every column a spec's money-normalization
declaration lists, once, and adds each column's integer minor-unit companion.
Its parameters are the declaration (canonical JSON) and the factor context
(the country's declared factor document), each with its resource digest, plus
the scenario's profile selection. The node reads exactly one source, the
authenticated Chronicle feed: every selected factor is replayed against it
with :func:`~microcosm.build.reference_factors.replay_reference_factor`, as
target compilation does, so a self-consistent receipt alone never
authenticates a factor. A feed-selected exchange rate is replayed from the
same facts.

CREATE records the donor's model period in the frame metadata
(``donor_time_period``); the kernel refuses a frame whose recorded period is
absent, null or different from the declaration's. This checks consistency
with the declared money basis; it does not authenticate that basis.

``money.minor_units@1`` converts declared money values to authoritative
integer minor units. ``money.annual_transfers@1`` consumes those integers,
explicit receipt flags, annualization and exposure; it never calculates tax.
"""

from __future__ import annotations

from collections.abc import Mapping
from fractions import Fraction

import numpy as np
import pandas as pd

import microcosm.build.chronicle_epoch as chronicle_epoch_module
import microcosm.build.ledger_artifact as ledger_artifact_module
import microcosm.build.ledger_targets as ledger_targets_module
import microcosm.build.money_normalization as money_normalization_module
import microcosm.build.reference_factors as reference_factors_module
import microcosm.frame.concepts as concepts_module
import microcosm.frame.exact_money as exact_money_module
from microcosm.build.ledger_artifact import (
    chronicle_feed_identity,
    load_ledger_consumer_artifact,
    validate_chronicle_feed_pin,
)
from microcosm.build.money_normalization import (
    money_factor_inputs,
    normalize_money_tables,
    normalized_column_families,
    resolve_money_rate,
    selected_money_factors,
    validate_money_normalization,
)
from microcosm.build.reference_factors import (
    replay_reference_factor,
    validate_factor_context,
)
from microcosm.frame.exact_money import _decimal_text, _exact_value, _half_even
from microcosm.graph import (
    ROWS_ALL,
    Capabilities,
    Determinism,
    KernelBase,
    KernelContext,
    KernelRegistry,
    KernelResult,
    Numeric,
    SeedSource,
    source_hash,
)

from . import graph_inputs
from .graph_inputs import (
    canonical_document_param,
    require_params,
    sha256_param,
    single_source,
)

__all__ = [
    "MONEY_ANNUAL_TRANSFERS",
    "MONEY_ANNUAL_TRANSFERS_REF",
    "MONEY_MINOR_UNITS",
    "MONEY_MINOR_UNITS_REF",
    "MONEY_NORMALIZE",
    "MONEY_NORMALIZE_REF",
    "MoneyNormalizeKernel",
    "MoneyAnnualTransfersKernel",
    "MoneyMinorUnitsKernel",
    "register_money_kernels",
    "weekly_gross_cents_for_reporting",
]

MONEY_NORMALIZE_REF = "money.normalize@1"
MONEY_MINOR_UNITS_REF = "money.minor_units@1"
MONEY_ANNUAL_TRANSFERS_REF = "money.annual_transfers@1"
#: The frame-metadata key CREATE writes from the donor's ``_time_period``.
DONOR_TIME_PERIOD = "donor_time_period"


def _recorded_period(context: KernelContext, declaration) -> int:
    if DONOR_TIME_PERIOD not in context.frame_metadata:
        raise ValueError(
            f"{MONEY_NORMALIZE_REF} needs the frame's recorded {DONOR_TIME_PERIOD}; "
            "CREATE records it from the donor."
        )
    recorded = context.frame_metadata[DONOR_TIME_PERIOD]
    declared = declaration["donor_money_period"]
    if recorded is None:
        raise ValueError(
            f"The donor records no money period, but the declaration names "
            f"{declared['type']} {declared['value']}."
        )
    if type(recorded) is not int or recorded != declared["value"]:
        raise ValueError(
            f"The donor's recorded money period {recorded!r} differs from the "
            f"declared {declared['type']} {declared['value']}."
        )
    return recorded


def _authenticated_facts(context: KernelContext, entries) -> tuple[tuple, dict]:
    """Load the one facts source and authenticate it as target compilation does."""
    source = single_source(context, MONEY_NORMALIZE_REF)
    feed_pin = (
        canonical_document_param(context, MONEY_NORMALIZE_REF, "chronicle_feed_pin")
        if "chronicle_feed_pin" in context.params
        else None
    )
    artifact = load_ledger_consumer_artifact(
        context.sources[source], feed_pin=feed_pin, lexical_floats=True
    )
    identities = []
    for entry in entries:
        identity = chronicle_feed_identity(
            feed_pin if feed_pin is not None else entry["declaration"]["feed_identity"]
        )
        if feed_pin is None:
            validate_chronicle_feed_pin(artifact, identity)
            if (artifact.manifest or {}).get("source_commit") != identity[
                "source_commit"
            ]:
                raise ValueError(
                    "Factor source commit cannot be authenticated without a feed pin."
                )
        replay_reference_factor(
            entry["declaration"],
            artifact.facts,
            identity,
            entry["operational_factor"],
            entry["receipt"],
        )
        identities.append(identity)
    if feed_pin is not None:
        identity = chronicle_feed_identity(feed_pin)
    elif identities:
        identity = identities[0]
    else:
        identity = None
    return artifact.facts, {
        "identity": identity,
        "facts_sha256": artifact.facts_sha256,
        "manifest_sha256": artifact.manifest_sha256,
    }


class MoneyNormalizeKernel(KernelBase):
    """``money.normalize@1``: declared exact currency and period normalization."""

    ref = MONEY_NORMALIZE_REF
    capabilities = Capabilities(
        Determinism.DETERMINISTIC,
        numeric=Numeric.BITWISE,
        seed_source=SeedSource.NONE,
        dependencies=("numpy", "pandas"),
    )
    _required = frozenset(
        {
            "declaration",
            "declaration_sha256",
            "factor_context",
            "factor_context_sha256",
            "profile_selection",
        }
    )

    def implementation_hash(self) -> str:
        return source_hash(
            type(self),
            graph_inputs,
            money_normalization_module,
            reference_factors_module,
            ledger_artifact_module,
            ledger_targets_module,
            chronicle_epoch_module,
            concepts_module,
            exact_money_module,
            dependencies=self.capabilities.dependencies,
        )

    def run(self, context: KernelContext) -> KernelResult:
        require_params(
            context,
            self.ref,
            required=self._required,
            optional=frozenset({"chronicle_feed_pin"}),
        )
        if context.artifacts or context.node.artifact_outputs:
            raise ValueError(f"{self.ref} reads and writes no artifacts.")
        if any(item.rows != ROWS_ALL for item in context.node.inputs) or any(
            item.rows != ROWS_ALL for item in context.node.outputs
        ):
            raise ValueError(f"{self.ref} requires whole-row slices and outputs.")
        for name in ("declaration_sha256", "factor_context_sha256"):
            sha256_param(context, self.ref, name)
        declaration = validate_money_normalization(
            canonical_document_param(context, self.ref, "declaration")
        )
        period = _recorded_period(context, declaration)
        profile, chosen = selected_money_factors(
            declaration,
            canonical_document_param(context, self.ref, "profile_selection"),
        )
        families = normalized_column_families(declaration)
        suffix = declaration["cents_suffix"]
        expected_outputs = {
            (entity, column, "float64", True) for entity, column in families
        } | {
            (entity, f"{column}{suffix}", "int64", False) for entity, column in families
        }
        declared_outputs = {
            (item.entity, item.column, item.dtype, item.rewrite)
            for item in context.node.outputs
        }
        if declared_outputs != expected_outputs:
            raise ValueError(
                f"{self.ref} must rewrite exactly the declared money columns as "
                "float64 and add each int64 companion, and own nothing else."
            )
        sliced = {
            (item.entity, column)
            for item in context.node.inputs
            for column in item.columns
        }
        if sliced != set(families):
            raise ValueError(
                f"{self.ref} slices exactly the declared money columns it rewrites."
            )
        factor_ids = sorted({value for value in chosen.values() if value is not None})
        entries = validate_factor_context(
            canonical_document_param(context, self.ref, "factor_context"), factor_ids
        )
        inputs = {
            factor_id: money_factor_inputs(entries[factor_id], declaration)
            for factor_id in factor_ids
        }
        facts, feed = _authenticated_facts(
            context, [entries[factor_id] for factor_id in factor_ids]
        )
        rate, rate_evidence = resolve_money_rate(declaration, facts, feed["identity"])
        factors = {
            family: (
                {
                    "identity": True,
                    "reason": declaration["index_family_factors"][family]["reason"],
                }
                if factor_id is None
                else inputs[factor_id]
            )
            for family, factor_id in chosen.items()
        }
        arrays, columns = normalize_money_tables(
            context.tables,
            declaration,
            rate=rate,
            rate_text=rate_evidence["text"],
            factors=factors,
        )
        result = {}
        for (entity, column), values in arrays.items():
            ids = context.tables[entity][f"{entity}_id"]
            result[entity, column] = pd.Series(
                values,
                index=pd.Index(ids.to_numpy(copy=True), name=f"{entity}_id"),
                name=column,
                dtype=values.dtype,
            )
        return KernelResult(
            columns=result,
            receipt={
                "declaration_sha256": context.params["declaration_sha256"],
                "factor_context_sha256": context.params["factor_context_sha256"],
                "from_currency": declaration["from_currency"],
                "to_currency": declaration["to_currency"],
                "quote": declaration["quote"],
                "rounding": dict(declaration["rounding"]),
                "rate": rate_evidence,
                "donor_money_period": dict(declaration["donor_money_period"]),
                "recorded_donor_time_period": period,
                "profile": profile,
                "factors": {family: factors[family] for family in sorted(factors)},
                "feed": {
                    "facts_sha256": feed["facts_sha256"],
                    "manifest_sha256": feed["manifest_sha256"],
                },
                "columns": columns,
            },
        )


MONEY_NORMALIZE = MoneyNormalizeKernel()


def _closed_money_document(value, required, *, optional=frozenset(), label):
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be an object.")
    row = dict(value)
    if row.keys() - required - optional or required - row.keys():
        raise ValueError(f"{label} requires exactly {sorted(required)} fields.")
    return row


def _money_text(value, label):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be nonempty text.")
    return value


def _money_int64(value, label):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{label} must be integer minor units.")
    value = int(value)
    if not -(2**63) <= value <= 2**63 - 1:
        raise ValueError(f"{label} exceeds int64 minor units.")
    return value


def _money_weeks(value):
    if type(value) is not int or value <= 0:
        raise ValueError(
            "weeks_per_year must be an explicitly declared positive integer."
        )
    return value


def _money_context(context, ref, *, optional=frozenset()):
    require_params(context, ref, required=frozenset({"declaration"}), optional=optional)
    if context.artifacts or context.node.artifact_outputs or context.sources:
        raise ValueError(f"{ref} reads and writes no artifacts or sources.")
    if any(item.rows != ROWS_ALL for item in context.node.inputs) or any(
        item.rows != ROWS_ALL for item in context.node.outputs
    ):
        raise ValueError(f"{ref} requires whole-row slices and outputs.")
    for name in optional:
        if name in context.params:
            _money_text(context.params[name], name)
    return canonical_document_param(context, ref, "declaration")


def _money_columns(context, ref, entity, inputs, outputs):
    declared_inputs = {
        (item.entity, column) for item in context.node.inputs for column in item.columns
    }
    declared_outputs = {
        (item.entity, item.column, item.dtype, item.rewrite)
        for item in context.node.outputs
    }
    if declared_inputs != {(entity, column) for column in inputs}:
        raise ValueError(f"{ref} slices exactly its declared source and flag columns.")
    if declared_outputs != {(entity, column, "int64", False) for column in outputs}:
        raise ValueError(f"{ref} owns exactly its declared int64 minor-unit columns.")
    if entity not in context.tables:
        raise ValueError(f"{ref} entity {entity!r} is absent.")
    table = context.tables[entity]
    missing = inputs - set(table)
    if missing:
        raise ValueError(f"{ref} source columns are absent: {sorted(missing)}.")
    if set(outputs) & set(table):
        raise ValueError(f"{ref} refuses existing output columns; money has one owner.")
    return table


def _money_result_column(table, entity, column, values):
    return pd.Series(
        np.asarray(values, dtype=np.int64),
        index=pd.Index(table[f"{entity}_id"].to_numpy(copy=True), name=f"{entity}_id"),
        name=column,
    )


class MoneyMinorUnitsKernel(KernelBase):
    """Convert declared money columns to authoritative int64 minor units.

    A quantized engine bridge produces float64. ``rounding: exact`` reads a
    float's shortest round-trip decimal and preserves integer sources exactly
    through the shared exact-money helpers. It requires an integer multiple
    of the declared ``quantum``. Thus the integer
    multiple, rather than a binary64 multiplication, is authoritative. An
    unquantized before-tax payment may declare ``rounding: half_even``.

    An optional engine reference attributes derived outputs to a declared
    pre-calibration rules stage, and an optional period names that stage's
    period. Composition finds a stage's nodes by the prepared reference they
    carry, so a node with only a period can't produce a stage's derived output.
    These parameters never execute an engine. This kernel owns no weekly
    reporting value.
    """

    ref = MONEY_MINOR_UNITS_REF
    capabilities = Capabilities(
        Determinism.DETERMINISTIC,
        numeric=Numeric.BITWISE,
        seed_source=SeedSource.NONE,
        dependencies=("numpy", "pandas"),
    )

    def implementation_hash(self) -> str:
        return source_hash(
            type(self),
            graph_inputs,
            exact_money_module,
            dependencies=self.capabilities.dependencies,
        )

    def run(self, context: KernelContext) -> KernelResult:
        declaration = _closed_money_document(
            _money_context(
                context, self.ref, optional=frozenset({"engine_ref", "period"})
            ),
            frozenset({"entity", "columns"}),
            label=self.ref,
        )
        entity = _money_text(declaration["entity"], "entity")
        rows = declaration["columns"]
        if not isinstance(rows, list) or not rows:
            raise ValueError(f"{self.ref} columns must be a nonempty list.")
        entries, inputs, outputs = [], set(), set()
        for value in rows:
            row = _closed_money_document(
                value,
                frozenset({"source_column", "output_column", "quantum", "rounding"}),
                optional=frozenset({"source_tax_basis"}),
                label="minor-unit column",
            )
            source = _money_text(row["source_column"], "source_column")
            output = _money_text(row["output_column"], "output_column")
            quantum = _decimal_text(row["quantum"], "quantum")
            if row["rounding"] not in {"exact", "half_even"}:
                raise ValueError("Minor-unit rounding must be exact or half_even.")
            if "source_tax_basis" in row and row["source_tax_basis"] != "before_tax":
                raise ValueError("source_tax_basis must be before_tax when declared.")
            if source in inputs or output in outputs:
                raise ValueError(
                    "Minor-unit columns must have unique sources and outputs."
                )
            entries.append((row, source, output, quantum))
            inputs.add(source)
            outputs.add(output)
        table = _money_columns(context, self.ref, entity, inputs, outputs)
        result, receipts = {}, []
        for row, source, output, quantum in entries:
            units = []
            for value in table[source].tolist():
                exact = _exact_value(value) / quantum
                if row["rounding"] == "exact" and exact.denominator != 1:
                    raise ValueError(
                        f"{entity}.{source} is not an exact multiple of quantum {row['quantum']}."
                    )
                units.append(_money_int64(_half_even(exact), f"{entity}.{output}"))
            result[entity, output] = _money_result_column(table, entity, output, units)
            receipts.append({**row, "n": len(units), "sum_minor_units": sum(units)})
        return KernelResult(
            columns=result, receipt={"entity": entity, "columns": receipts}
        )


def weekly_gross_cents_for_reporting(annual_gross_cents, weeks_per_year):
    """Report annual cents / declared weeks, half-even; never a graph output.

    Annual transfers consume annual cents directly. This reporting-only helper
    must never feed annualization or become a persisted weekly gross column.
    """
    amount = _money_int64(annual_gross_cents, "annual_gross_cents")
    return _money_int64(
        _half_even(Fraction(amount, _money_weeks(weeks_per_year))),
        "reporting weekly gross cents",
    )


class MoneyAnnualTransfersKernel(KernelBase):
    """Form annual transfers from explicit flags, authoritative cents and exposure.

    Annual gross cents are used directly. Weekly before-tax cents multiply by
    the declared weeks per year. Both multiply by a binary receipt flag and
    exact exposure. There is no exposure default: its central value and
    convention must be present. Integer exposure needs no rounding;
    non-integer exposure rounds half-even exactly once, at the end. Every
    input and result is checked for int64 overflow and receipt sums use Python
    integers, which cannot overflow. This kernel performs no tax calculation.
    """

    ref = MONEY_ANNUAL_TRANSFERS_REF
    capabilities = Capabilities(
        Determinism.DETERMINISTIC,
        numeric=Numeric.BITWISE,
        seed_source=SeedSource.NONE,
        dependencies=("numpy", "pandas"),
    )

    def implementation_hash(self) -> str:
        return source_hash(
            type(self),
            graph_inputs,
            exact_money_module,
            dependencies=self.capabilities.dependencies,
        )

    def run(self, context: KernelContext) -> KernelResult:
        declaration = _closed_money_document(
            _money_context(context, self.ref),
            frozenset({"entity", "weeks_per_year", "exposure", "transfers"}),
            label=self.ref,
        )
        entity = _money_text(declaration["entity"], "entity")
        weeks = _money_weeks(declaration["weeks_per_year"])
        exposure = _closed_money_document(
            declaration["exposure"],
            frozenset({"central", "convention"}),
            label="exposure",
        )
        if exposure["convention"] != "full_year_stock_equivalent":
            raise ValueError("Exposure convention must be full_year_stock_equivalent.")
        central = exposure["central"]
        if type(central) is int:
            fraction = Fraction(central)
        elif isinstance(central, str):
            fraction = _decimal_text(central, "exposure central", allow_zero=True)
        else:
            raise ValueError(
                "Exposure central must be an integer or exact decimal text."
            )
        if fraction < 0:
            raise ValueError("Exposure central must be nonnegative.")
        rows = declaration["transfers"]
        if not isinstance(rows, list) or not rows:
            raise ValueError("Annual transfers must be a nonempty list.")
        entries, inputs, outputs = [], set(), set()
        for value in rows:
            row = _closed_money_document(
                value,
                frozenset(
                    {"flag_column", "source_column", "source_basis", "output_column"}
                ),
                optional=frozenset({"source_tax_basis"}),
                label="annual transfer",
            )
            for name in ("flag_column", "source_column", "output_column"):
                _money_text(row[name], name)
            basis = row["source_basis"]
            if basis not in {"annual_gross_cents", "weekly_before_tax_cents"}:
                raise ValueError(
                    "Annual transfers require annual gross or weekly before-tax cents."
                )
            if (
                basis == "weekly_before_tax_cents"
                and row.get("source_tax_basis") != "before_tax"
            ):
                raise ValueError(
                    "Weekly source_tax_basis must explicitly be before_tax."
                )
            if "source_tax_basis" in row and row["source_tax_basis"] != "before_tax":
                raise ValueError("source_tax_basis must be before_tax when declared.")
            if row["output_column"] in outputs:
                raise ValueError("Annual transfers require unique output columns.")
            inputs.update((row["flag_column"], row["source_column"]))
            outputs.add(row["output_column"])
            entries.append(row)
        table = _money_columns(context, self.ref, entity, inputs, outputs)
        result, receipts = {}, []
        for row in entries:
            output, values = row["output_column"], []
            factor = weeks if row["source_basis"] == "weekly_before_tax_cents" else 1
            for flag, amount in zip(
                table[row["flag_column"]].tolist(),
                table[row["source_column"]].tolist(),
                strict=True,
            ):
                if not isinstance(
                    flag, (bool, np.bool_, int, np.integer)
                ) or flag not in (0, 1):
                    raise ValueError(
                        f"{entity}.{row['flag_column']} must be a finite binary flag."
                    )
                amount = _money_int64(amount, f"{entity}.{row['source_column']}")
                if amount < 0:
                    raise ValueError(
                        "Annual gross transfer amounts must be nonnegative."
                    )
                exact = int(flag) * amount * factor * fraction
                value = exact.numerator if exact.denominator == 1 else _half_even(exact)
                values.append(_money_int64(value, f"{entity}.{output}"))
            result[entity, output] = _money_result_column(table, entity, output, values)
            receipts.append({**row, "n": len(values), "sum_minor_units": sum(values)})
        return KernelResult(
            columns=result,
            receipt={
                "entity": entity,
                "weeks_per_year": weeks,
                "exposure": exposure,
                "transfers": receipts,
            },
        )


MONEY_MINOR_UNITS = MoneyMinorUnitsKernel()
MONEY_ANNUAL_TRANSFERS = MoneyAnnualTransfersKernel()


def register_money_kernels(registry: KernelRegistry) -> KernelRegistry:
    """Register money kernels explicitly and idempotently."""
    registry.register(MONEY_NORMALIZE)
    registry.register(MONEY_MINOR_UNITS)
    registry.register(MONEY_ANNUAL_TRANSFERS)
    return registry
