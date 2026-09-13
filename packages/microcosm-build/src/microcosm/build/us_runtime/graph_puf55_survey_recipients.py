"""Typed PUF55 route artifacts after current financial leaves.

Both nodes declare the original survey directory and atomic-support input,
which the retained financial/source owners recheck during qualification. The
source codecs and artifact edges do not replace those live authority checks.
These two nodes perform no fit, attachment or structural/weight change. A host
must retain the actual financial run and verify materialized typed artifacts
on cold execution and replay; decoded projection receipts grant no authority.
"""

from __future__ import annotations

import sys
from pathlib import Path

from microcosm.graph import (
    ArtifactInput,
    ArtifactOutput,
    ArtifactType,
    Capabilities,
    Determinism,
    KernelBase,
    KernelResult,
    Node,
    Numeric,
    source_hash,
)
from microcosm.graph.keys import opaque_artifact_key

from . import puf55_survey_recipients as values

financial = values.financial
source_graph = financial.survey
host = financial.financial.host
codec, model_input = financial.codec, values.model_input
PROJECTION_NODE = "survey_puf55.recipient_projection"
MATRIX_NODE = "survey_puf55.recipient_matrices"
PROJECTION_TYPE = ArtifactType("microcosm.us.puf55_survey_recipient_projection", 1)
_NAMES = {
    values.PROFILES[0].value: "matrix_nine",
    values.PROFILES[1].value: "matrix_eight",
}


def _edges(run):
    return (
        ArtifactInput(
            "preparation",
            source_graph.CREATE_NODE,
            "preparation",
            source_graph.PREPARATION_TYPE,
        ),
        ArtifactInput(
            "financial_projection",
            financial.financial.PROJECTION_NODE,
            "projection",
            financial.financial.PROJECTION_TYPE,
        ),
        ArtifactInput(
            "financial_matrix",
            financial.financial.PROJECTION_NODE,
            "matrix",
            model_input.RECIPIENT_MATRIX_TYPE,
        ),
        *financial.financial_tax_gate_edges(run),
    )


def _pins(run):
    entry = financial._run_entry(run)
    keys = dict(entry[2].keys)
    hashes = {(node, name): digest for node, name, digest in entry[2].artifact_hashes}
    return {
        edge.name: {
            "producer_key": keys[edge.producer],
            "artifact_key": opaque_artifact_key(keys[edge.producer], edge.artifact),
            "payload_sha256": hashes[edge.producer, edge.artifact],
        }
        for edge in _edges(run)
    }


def _check_values(qualified):
    values._require(
        type(qualified) is values.Puf55SurveyRecipients, "GRAPH_VALUES_TYPE"
    )
    values._result_stamp(qualified)
    run = qualified.financial_run
    entry = financial._run_entry(run)
    financial._pure_run(run, entry)
    document = codec.decode_json(qualified.receipt)
    values._require(
        document["protocol"] == values.PROTOCOL
        and document["financial_run_sha256"] == codec.sha(entry[1])
        and document["preparation_sha256"] == codec.sha(entry[2].preparation_entry[1])
        and document["person_projection_sha256"]
        == values._table_digest(qualified.person)
        and document["tax_unit_projection_sha256"]
        == values._table_digest(qualified.tax_unit)
        and tuple(row["profile"] for row in document["routes"])
        == tuple(name for name, _ in qualified.matrices)
        and tuple(row["matrix_sha256"] for row in document["routes"])
        == tuple(codec.sha(p) for _, p in qualified.matrices),
        "GRAPH_VALUES_BINDING",
    )
    names = tuple(name for name, _ in qualified.matrices)
    values._require(
        names and names == tuple(p.value for p in values.PROFILES if p.value in names),
        "GRAPH_ROUTE_ROSTER",
    )
    return entry, document


def puf55_survey_recipient_nodes(qualified):
    """Bind expected projections; executing kernels retain/requalify the live run."""
    entry, _ = _check_values(qualified)
    params = {
        "protocol": values.PROTOCOL,
        "financial_run_sha256": codec.sha(entry[1]),
        "preparation_sha256": codec.sha(entry[2].preparation_entry[1]),
        "projection_sha256": codec.sha(qualified.receipt),
        "host_edges": codec.encode_json(_pins(qualified.financial_run)).decode(),
        "route_matrices": codec.encode_json(
            [[name, codec.sha(p)] for name, p in qualified.matrices]
        ).decode(),
    }
    common = {
        "population": entry[2].financial_population.version,
        # The same source names already declared by the authenticated prefix:
        # survey_population_source and us_atomic_block_support. Full live-run
        # checks borrow both, including the recipient's ASEC literal reread.
        "sources": tuple(name for name, _ in entry[2].source_items),
        # Reading the seven financial-owned leaves creates the actual attach
        # dependency, even though neither node owns any receiving cells.
        "inputs": financial.financial._inputs(entry[2].financial_population.frame),
        "params": params,
    }
    projection = Node(
        PROJECTION_NODE,
        Puf55SurveyRecipientProjectionKernel.ref,
        **common,
        artifact_inputs=_edges(qualified.financial_run),
        artifact_outputs=(ArtifactOutput("projection", PROJECTION_TYPE),),
    )
    matrix = Node(
        MATRIX_NODE,
        Puf55SurveyRecipientMatrixKernel.ref,
        **common,
        artifact_inputs=(
            *_edges(qualified.financial_run),
            ArtifactInput("projection", PROJECTION_NODE, "projection", PROJECTION_TYPE),
        ),
        artifact_outputs=tuple(
            ArtifactOutput(_NAMES[name], model_input.RECIPIENT_MATRIX_TYPE)
            for name, _ in qualified.matrices
        ),
    )
    return projection, matrix


class _Kernel(KernelBase):
    capabilities = Capabilities(
        determinism=Determinism.DETERMINISTIC,
        numeric=Numeric.PLATFORM_BITWISE,
        dependencies=financial.financial._Kernel.capabilities.dependencies,
    )

    def __init__(self, financial_run):
        self._financial_run = financial_run

    def implementation_hash(self):
        # Bind the actual current-money graph owner closure plus each additional
        # numerical/source module. No receipt string stands in for source code.
        return codec.sha(
            codec.encode_json(
                {
                    "financial": source_hash(
                        financial, dependencies=self.capabilities.dependencies
                    ),
                    "source": source_graph._Kernel.implementation_hash(self),
                    "recipient": source_hash(
                        sys.modules[__name__],
                        values,
                        values.ss,
                        values.ss.reports,
                        values.ss.reports.basis_owner,
                        values.puf_source,
                        values.full,
                        values.support,
                        values.provenance,
                        model_input,
                        *host.survey_budget._modules(),
                        dependencies=self.capabilities.dependencies,
                    ),
                }
            )
        )

    def _context(self, context, qualified):
        bound = financial._run_entry(self._financial_run)
        # Resolve the retained literal paths before the existing value/owner
        # checks. The executor supplies canonical paths; support recipes retain
        # a string and may name an accepted alias in an ancestor directory.
        expected_sources = tuple(
            (name, str(Path(path).resolve(strict=True)))
            for name, path in bound[2].source_items
        )
        entry, _ = _check_values(qualified)
        expected = {node.id: node for node in puf55_survey_recipient_nodes(qualified)}
        values._require(
            context.node.id in expected
            and context.node == expected[context.node.id]
            and dict(context.params) == dict(context.node.params),
            "GRAPH_DECLARATION",
        )
        values._require(
            entry is bound
            and tuple(
                sorted((name, str(path)) for name, path in context.sources.items())
            )
            == expected_sources,
            "GRAPH_SOURCE_ROSTER",
        )
        values._require(
            set(context.artifacts) == {e.name for e in context.node.artifact_inputs},
            "GRAPH_ARTIFACT_ROSTER",
        )
        host._current_context_frame(context, entry[2].financial_population.frame)
        payloads = {
            "preparation": entry[2].preparation_entry[1],
            "financial_projection": entry[2].projection,
            "financial_matrix": entry[2].matrix,
        }
        if entry[2].rebase_property_taxes:
            payloads["property_tax_verification"] = entry[2].tax_verification
        for edge in _edges(qualified.financial_run):
            artifact = host.shared.artifact(context, edge.name, edge.type)
            values._require(
                {
                    "producer_key": artifact.producer_key,
                    "artifact_key": artifact.key,
                    "payload_sha256": codec.sha(artifact.payload),
                }
                == _pins(qualified.financial_run)[edge.name]
                and artifact.payload == payloads[edge.name],
                "GRAPH_HOST_ARTIFACT",
            )
        if context.node.id == MATRIX_NODE:
            artifact = host.shared.artifact(context, "projection", PROJECTION_TYPE)
            values._require(
                artifact.payload == qualified.receipt, "GRAPH_PROJECTION_ARTIFACT"
            )

    def run(self, context):
        run = self._financial_run
        qualified = values.qualify_puf55_survey_recipients(run)
        entry = financial._run_entry(run)
        self._context(context, qualified)
        outputs = (
            {"projection": qualified.receipt}
            if context.node.id == PROJECTION_NODE
            else {_NAMES[name]: payload for name, payload in qualified.matrices}
        )
        output_seal = tuple(sorted(outputs.items()))
        receipt = {
            "protocol": values.PROTOCOL,
            "projection_sha256": codec.sha(qualified.receipt),
            "artifact_sha256": {n: codec.sha(p) for n, p in outputs.items()},
            "population_changed": False,
            "release_eligible": False,
        }
        receipt_bytes = codec.encode_json(receipt)
        result = KernelResult(artifacts=dict(outputs), receipt=receipt)
        # Recheck the live source/store/support owners, then the retained run,
        # context and outputs. Retained-run/context checks also read the store;
        # this artifact-output kernel is not a source-free replay boundary.
        financial.check_atomic_survey_financial_run(run)
        financial._pure_run(run, entry)
        self._context(context, qualified)
        values._require(
            self._financial_run is run
            and all(
                type(name) is str and type(payload) is bytes
                for name, payload in result.artifacts.items()
            )
            and tuple(sorted(result.artifacts.items())) == output_seal
            and codec.encode_json(result.receipt) == receipt_bytes
            and not result.columns
            and result.frame is result.keep is result.expand is result.weights is None
            and result.strata is None,
            "GRAPH_FINAL_RESULT",
        )
        return result


class Puf55SurveyRecipientProjectionKernel(_Kernel):
    ref = "us.survey_puf55.recipient_projection@1"


class Puf55SurveyRecipientMatrixKernel(_Kernel):
    ref = "us.survey_puf55.recipient_matrices@1"


def verify_materialized_puf55_survey_recipients(financial_run, *, projection, matrices):
    """Check host-pinned artifact bytes on replay; this is not an artifact issuer.

    The caller must first verify actual producer keys/types/implementation hashes
    against its complete compiled graph and retained receiving Population.
    """
    expected = values.qualify_puf55_survey_recipients(financial_run)
    values._require(
        type(projection) is bytes
        and projection == expected.receipt
        and type(matrices) is tuple
        and matrices == expected.matrices,
        "MATERIALIZED_ARTIFACTS",
    )
    return expected
