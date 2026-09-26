"""Project authenticated survey geography into declared household cells.

This nonstructural node qualifies observed inputs only. Shared atomic operators
own location assignment; the country runner owns receiving-Population and warm
replay admission. Preparation bytes and descriptive receipts grant no authority.
"""

from __future__ import annotations

import json

import pandas as pd

from microcosm.frame import WeightKind
from microcosm.graph import (
    ArtifactInput,
    Capabilities,
    Determinism,
    KernelBase,
    KernelResult,
    Node,
    Numeric,
    Owned,
    Slice,
    source_hash,
)
from microcosm.graph.canonical import canonical_json

from . import graph_survey_population as source_graph

NODE = "survey_population.observed_geography"
REF = "us.survey_population.observed_geography@1"
PHASE = "us.survey_population.observed_geography.v1"
OUTPUT_COLUMNS = (
    "survey_geography_origin_key",
    "survey_observed_state",
    "survey_observed_puma",
)


def _require(condition, reason):
    if not condition:
        raise ValueError("SURVEY_GEOGRAPHY_PROJECTION_" + reason)


def _qualifier():
    # Declaration-only imports need not load retained source owners.
    from . import current_survey_geography

    return current_survey_geography


def current_survey_geography_node(
    *,
    preparation_sha256,
    projection_receipt_sha256,
    population,
    node_id=NODE,
):
    """Declare three new observed columns; digest arguments authenticate nothing."""
    _require(
        type(population) is str
        and 0 < len(population) <= 256
        and type(node_id) is str
        and 0 < len(node_id) <= 256
        and population != node_id,
        "NODE_NAMES",
    )
    return Node(
        id=node_id,
        kernel=REF,
        population=population,
        inputs=(Slice("household", source_graph._provenance_columns("household")),),
        outputs=tuple(Owned("household", name, "string") for name in OUTPUT_COLUMNS),
        params={
            "phase": PHASE,
            "preparation_sha256": source_graph._digest(preparation_sha256),
            "projection_receipt_sha256": source_graph._digest(
                projection_receipt_sha256
            ),
        },
        artifact_inputs=(
            ArtifactInput(
                "preparation",
                source_graph.CREATE_NODE,
                "preparation",
                source_graph.PREPARATION_TYPE,
            ),
        ),
        description="Project qualified observed state, PUMA and stable source draw keys.",
    )


def _check_context(context, *, payload, source_frame, receipt_sha256):
    expected = current_survey_geography_node(
        preparation_sha256=source_graph._sha(payload),
        projection_receipt_sha256=receipt_sha256,
        population=context.node.population,
        node_id=context.node.id,
    )
    _require(
        context.node == expected and dict(context.params) == dict(expected.params),
        "DECLARATION",
    )
    _require(
        not context.sources
        and set(context.artifacts) == {"preparation"}
        and set(context.tables) == {"household"},
        "CONTEXT_ROSTER",
    )
    source_graph._artifact(
        context, "preparation", source_graph.PREPARATION_TYPE, payload
    )
    table = context.tables["household"]
    columns = ("household_id", *source_graph._provenance_columns("household"))
    _require(tuple(table.columns) == columns, "CONTEXT_COLUMNS")
    try:
        pd.testing.assert_frame_equal(
            table,
            source_frame.table("household").loc[:, list(columns)],
            check_exact=True,
        )
    except (AssertionError, ValueError, TypeError):
        raise ValueError("SURVEY_GEOGRAPHY_PROJECTION_HOUSEHOLD_ORIGIN") from None
    _require(
        set(context.weights) == {"household"}
        and context.weights["household"].kind is WeightKind.IMPORTANCE
        and len(context.weights["household"].values) == len(table),
        "ALLOCATED_WEIGHTS_REQUIRED",
    )


def _check_projection(qualifier, projection, *, payload, receipt_sha256):
    _require(
        type(projection) is qualifier.CurrentSurveyGeographyValues
        and qualifier.COLUMNS == OUTPUT_COLUMNS
        and type(projection.receipt) is bytes
        and 0 < len(projection.receipt) <= qualifier.MAX_RECEIPT_BYTES
        and source_graph._sha(projection.receipt) == receipt_sha256,
        "PROJECTION_RECEIPT",
    )
    document = json.loads(projection.receipt)
    _require(
        document["protocol"] == qualifier.PROTOCOL
        and document["preparation_sha256"] == source_graph._sha(payload)
        and document["columns"] == list(OUTPUT_COLUMNS)
        and document["households"] == len(projection.household)
        and document["projection_sha256"]
        == qualifier._projection_digest(projection.household)
        and document["source_admission_issued"] is False
        and document["population_admission_issued"] is False
        and document["release_eligible"] is False,
        "QUALIFIED_PROJECTION",
    )
    return document


class CurrentSurveyGeographyKernel(KernelBase):
    """Retain the live preparation; independently qualify every executed output."""

    ref = REF
    capabilities = Capabilities(
        determinism=Determinism.DETERMINISTIC,
        numeric=Numeric.PLATFORM_BITWISE,
        dependencies=source_graph.STAGE_DEPENDENCIES,
    )

    def __init__(self, preparation):
        self._preparation = preparation

    def implementation_hash(self):
        qualifier = _qualifier()
        demographics = qualifier.demographics
        # Preserve the source stage's maintained dependency fence, and bind the
        # additional actual projection functions and their demographic owners.
        return source_graph._sha(
            canonical_json(
                {
                    "source_stage": source_graph._Kernel.implementation_hash(self),
                    "projection": source_hash(
                        type(self),
                        qualifier,
                        qualifier.qualify_current_survey_geography,
                        demographics,
                        demographics.qualify_current_asec_demographics,
                        demographics.demographic,
                        demographics.demographic.load_authenticated_asec_demographic_source,
                        demographics.demographic._snapshot,
                        demographics.household,
                        demographics.source_csv_builtin,
                        qualifier.dtype_for_token,
                        dependencies=self.capabilities.dependencies,
                    ),
                }
            )
        )

    def run(self, context):
        preparation = self._preparation
        owner = source_graph._source_owner()
        _require(
            type(preparation) is owner.AuthenticatedSurveyPopulationPreparation,
            "PREPARATION_TYPE",
        )
        entry = preparation._checked()
        payload, state = entry[1], entry[2]
        receipt_sha256 = source_graph._digest(
            context.params.get("projection_receipt_sha256")
        )
        _check_context(
            context,
            payload=payload,
            source_frame=state.frame,
            receipt_sha256=receipt_sha256,
        )
        qualifier = _qualifier()
        projection = qualifier.qualify_current_survey_geography(preparation)
        document = _check_projection(
            qualifier, projection, payload=payload, receipt_sha256=receipt_sha256
        )
        _require(
            projection.household.index.tolist()
            == context.tables["household"].household_id.tolist(),
            "PROJECTION_HOUSEHOLD_ORDER",
        )
        receipt = {
            "phase": PHASE,
            "preparation_sha256": source_graph._sha(payload),
            "projection_receipt_sha256": receipt_sha256,
            "source_projection": document,
            "population_admission_issued": False,
            "release_eligible": False,
        }
        receipt_bytes = canonical_json(receipt)
        result = KernelResult(
            columns={
                ("household", name): projection.household[name].copy(deep=True)
                for name in OUTPUT_COLUMNS
            },
            receipt=receipt,
        )
        # Finish source-owner I/O before rechecking mutable source views,
        # projected context and every detached returned column/receipt.
        final_entry = preparation._checked()
        _require(
            self._preparation is preparation
            and final_entry is entry
            and owner._ISSUED.get(id(preparation)) is entry
            and preparation.payload == payload,
            "FINAL_ISSUANCE",
        )
        owner._pure_final(state)
        _check_context(
            context,
            payload=payload,
            source_frame=state.frame,
            receipt_sha256=receipt_sha256,
        )
        _check_projection(
            qualifier, projection, payload=payload, receipt_sha256=receipt_sha256
        )
        _require(
            set(result.columns) == {("household", name) for name in OUTPUT_COLUMNS}
            and result.frame is result.keep is result.expand is result.weights is None
            and result.strata is None
            and not result.artifacts
            and canonical_json(result.receipt) == receipt_bytes,
            "FINAL_RESULT",
        )
        returned = pd.DataFrame(
            {name: result.columns[("household", name)] for name in OUTPUT_COLUMNS}
        )
        _require(
            qualifier._projection_digest(returned) == document["projection_sha256"],
            "FINAL_OUTPUT",
        )
        return result
