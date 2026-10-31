"""Exact donor money normalization as a graph kernel.

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
"""

from __future__ import annotations

import pandas as pd

import microcosm.build.chronicle_epoch as chronicle_epoch_module
import microcosm.build.ledger_artifact as ledger_artifact_module
import microcosm.build.ledger_targets as ledger_targets_module
import microcosm.build.money_normalization as money_normalization_module
import microcosm.build.reference_factors as reference_factors_module
import microcosm.frame.concepts as concepts_module
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
    "MONEY_NORMALIZE",
    "MONEY_NORMALIZE_REF",
    "MoneyNormalizeKernel",
    "register_money_kernels",
]

MONEY_NORMALIZE_REF = "money.normalize@1"
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


def register_money_kernels(registry: KernelRegistry) -> KernelRegistry:
    """Register the money kernel explicitly and idempotently."""
    registry.register(MONEY_NORMALIZE)
    return registry
