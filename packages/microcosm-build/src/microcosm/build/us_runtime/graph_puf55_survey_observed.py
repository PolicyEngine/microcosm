"""Opt-in, source-qualified fixed-input artifacts for each explicit survey arm.

This fragment does not fit, apply, finalize, prune, or issue a population. The
host supplies real recipient projection/matrix nodes and retains the genuine
financial run. On replay it must verify the complete compiled producer keys and
requalify these values; detached receipts cannot grant source authority.
"""

from __future__ import annotations

import sys

from microcosm.graph import (
    ArtifactInput,
    ArtifactOutput,
    ArtifactType,
    KernelResult,
    Node,
    source_hash,
)
from microcosm.graph.keys import opaque_artifact_key

from . import graph_puf55_survey_recipients as parent
from . import puf55_survey_observed as values

codec = parent.codec
QUALIFICATION_TYPE = ArtifactType(
    "microcosm.us.puf55_survey_fixed_input_qualification", 1
)
SOURCE_BASIS_TYPE = ArtifactType(
    "microcosm.us.puf55_survey_fixed_input_source_basis", 1
)


def fixed_input_node_id(arm):
    values.recipients.recipient_protocol(arm)
    return "survey_puf55." + ("original" if arm == 0 else "clone") + "_fixed_inputs"


def observed_artifact_names(qualified, profile):
    values.check_fixed_input_binding(qualified)
    values.require(profile in dict(qualified.recipients.matrices), "GRAPH_PROFILE")
    prefix = parent._NAMES[profile] + "__"
    return {target: prefix + target for target in qualified.tax_unit_values}


def puf55_survey_fixed_input_nodes(qualified):
    """Declare actual source/matrix dependencies and explicit development rules."""
    document = values.check_fixed_input_binding(qualified)
    recipient = qualified.recipients
    entry, _ = parent._check_values(recipient)
    projection_id, matrix_id = parent.recipient_node_ids(recipient.arm)
    return (
        Node(
            fixed_input_node_id(recipient.arm),
            Puf55SurveyFixedInputKernel.ref,
            population=entry[2].financial_population.version,
            sources=tuple(name for name, _ in entry[2].source_items),
            inputs=parent.financial.financial._inputs(
                entry[2].financial_population.frame
            ),
            params={
                "protocol": values.PROTOCOL,
                "recipient_arm": recipient.arm,
                "financial_run_sha256": document["financial_run_sha256"],
                "qualification_sha256": codec.sha(qualified.receipt),
                "rule_metadata": codec.encode_json(document["rule_metadata"]).decode(),
                "host_edges": codec.encode_json(
                    parent._pins(recipient.financial_run)
                ).decode(),
            },
            artifact_inputs=(
                *parent._edges(recipient.financial_run),
                ArtifactInput(
                    "projection",
                    projection_id,
                    "projection",
                    parent.projection_type(recipient.arm),
                ),
                *(
                    ArtifactInput(
                        parent._NAMES[profile],
                        matrix_id,
                        parent._NAMES[profile],
                        parent.model_input.RECIPIENT_MATRIX_TYPE,
                    )
                    for profile, _ in recipient.matrices
                ),
            ),
            artifact_outputs=(
                ArtifactOutput("qualification", QUALIFICATION_TYPE),
                ArtifactOutput("source_basis", SOURCE_BASIS_TYPE),
                *(
                    ArtifactOutput(name, values.observed.OBSERVED_TARGET_TYPE)
                    for profile, _ in recipient.matrices
                    for name in observed_artifact_names(qualified, profile).values()
                ),
            ),
        ),
    )


def _payloads(qualified, matrices):
    """Pure detached codec; the caller supplies actual checked graph edges."""
    values.check_fixed_input_binding(qualified)
    values.require(
        set(matrices) == set(dict(qualified.recipients.matrices)), "GRAPH_MATRIX_ROSTER"
    )
    basis = qualified.source_basis
    output = {
        "qualification": qualified.receipt,
        # Retain original amounts, raw routing columns and unresolved source
        # statuses for inspection; exact fixed values use the typed binary codec.
        "source_basis": codec.encode_json(
            {
                "protocol": values.PROTOCOL,
                "source_qualification": codec.decode_json(qualified.source_evidence),
                "source_basis_sha256": None
                if basis is None
                else values.recipients._table_digest(basis),
                "source_basis_table": None
                if basis is None
                else basis.to_json(orient="table"),
                "source_taxability_or_component_split_resolved": False,
            }
        ),
    }
    for profile, (payload, producer_key) in matrices.items():
        encoded = values.observed_target_artifacts(
            qualified,
            profile=profile,
            matrix_payload=payload,
            matrix_producer_key=producer_key,
        )
        names = observed_artifact_names(qualified, profile)
        output.update({names[target]: data for target, data in encoded.items()})
    return output


class Puf55SurveyFixedInputKernel(parent._Kernel):
    ref = "us.survey_puf55.qualified_fixed_inputs@1"

    def __init__(self, financial_run, *, development_rules=()):
        super().__init__(financial_run)
        values.development_rule_metadata(development_rules)
        self._development_rules = development_rules

    def implementation_hash(self):
        return codec.sha(
            codec.encode_json(
                {
                    "recipient": super().implementation_hash(),
                    "fixed_inputs": source_hash(
                        sys.modules[__name__],
                        values,
                        values.routing,
                        values.leaves,
                        values.observed,
                        dependencies=self.capabilities.dependencies,
                    ),
                }
            )
        )

    def _matrices(self, context, qualified):
        self._context(
            context,
            qualified.recipients,
            expected_nodes=puf55_survey_fixed_input_nodes(qualified),
        )
        projection = parent.host.shared.artifact(
            context, "projection", parent.projection_type(qualified.recipients.arm)
        )
        values.require(
            projection.key
            == opaque_artifact_key(projection.producer_key, "projection"),
            "GRAPH_PROJECTION_KEY",
        )
        parent.host.shared.siblings(
            context, tuple(parent._NAMES[p] for p, _ in qualified.recipients.matrices)
        )
        matrices = {}
        for profile, expected in qualified.recipients.matrices:
            name = parent._NAMES[profile]
            artifact = parent.host.shared.artifact(
                context, name, parent.model_input.RECIPIENT_MATRIX_TYPE
            )
            values.require(
                artifact.payload == expected
                and artifact.key == opaque_artifact_key(artifact.producer_key, name),
                "GRAPH_MATRIX_EDGE",
            )
            matrices[profile] = artifact.payload, artifact.producer_key
        return matrices

    def run(self, context):
        run, rules = self._financial_run, self._development_rules
        values.require(
            context.node.id in (fixed_input_node_id(0), fixed_input_node_id(1)),
            "GRAPH_NODE",
        )
        arm = 0 if context.node.id == fixed_input_node_id(0) else 1
        qualified = values.qualify_puf55_survey_fixed_inputs(
            run, arm=arm, development_rules=rules
        )
        seal = values.fixed_input_stamp(qualified)
        outputs = _payloads(qualified, self._matrices(context, qualified))
        receipt = {
            "protocol": values.PROTOCOL,
            "recipient_arm": arm,
            "qualification_sha256": codec.sha(qualified.receipt),
            "artifact_sha256": {
                name: codec.sha(payload) for name, payload in outputs.items()
            },
            "population_changed": False,
            "release_eligible": False,
        }
        result = KernelResult(artifacts=dict(outputs), receipt=receipt)
        receipt_bytes = codec.encode_json(receipt)
        # Close the actual source/run checks after context/store/path I/O. This
        # intentionally requalifies source rules as well as the financial frame.
        fresh = values.qualify_puf55_survey_fixed_inputs(
            run, arm=arm, development_rules=rules
        )
        values.require(values.fixed_input_stamp(fresh) == seal, "GRAPH_SOURCE_CHANGED")
        expected = _payloads(fresh, self._matrices(context, fresh))
        values.require(
            self._financial_run is run
            and self._development_rules == rules
            and values.fixed_input_stamp(qualified) == seal
            and result.artifacts == outputs == expected
            and all(type(payload) is bytes for payload in result.artifacts.values())
            and codec.encode_json(result.receipt) == receipt_bytes
            and not result.columns
            and result.frame is result.keep is result.expand is result.weights is None
            and result.strata is None,
            "GRAPH_FINAL_RESULT",
        )
        return result


def verify_materialized_puf55_fixed_inputs(
    financial_run, *, artifacts, matrices, arm=0, development_rules=()
):
    """Requalify bytes after the host checks actual compiled producer identities.

    ``matrices`` maps profile to (actual payload, actual producer key). Neither
    that mapping nor a digest grants authority: the real financial run and the
    selected source values are requalified here, and compiled-key validation is
    still required of the host. This function issues no population or verdict.
    """
    qualified = values.qualify_puf55_survey_fixed_inputs(
        financial_run, arm=arm, development_rules=development_rules
    )
    values.require(
        type(artifacts) is dict and artifacts == _payloads(qualified, matrices),
        "MATERIALIZED_FIXED_INPUTS",
    )
    return qualified
