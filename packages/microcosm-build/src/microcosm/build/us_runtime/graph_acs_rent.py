"""Source-blind ACS rent matrix preparation, validation and person placement.

The caller supplies the truthful source-available population after its real
relationship and tenure operators. This module does not select donor origins
or merge a treated projection into a different population.
"""

import numpy as np
import pandas as pd

from microcosm.build.gates import GateReport, GateResult
from microcosm.fit import _graph_legacy_qrf as fit_codec
from microcosm.fit import graph_legacy_apply_matrix, model_input, qrf, qrf_target
from microcosm.fit.graph_legacy_qrf import (
    LegacyQRFTrainKernel,
    legacy_qrf_apply_matrix_nodes,
    legacy_qrf_train_nodes,
)
from microcosm.fit.qrf_target import LegacyQRFTrainingState
from microcosm.frame import US_SCHEMA, EntitySchema, Frame, Weights
from microcosm.graph import (
    ArtifactInput,
    ArtifactOutput,
    Capabilities,
    Determinism,
    KernelBase,
    KernelRegistry,
    KernelResult,
    Node,
    Numeric,
    Owned,
    Slice,
    StructuralDelta,
)
from microcosm.graph.kernel import KernelRole

from . import _graph_acs_rent_artifacts as codec
from . import graph_acs_rent_source as source
from . import graph_context, housing_inputs

RENT_PREFIX = "early_preparation.rent"
RENT_OUTPUT_POPULATION = f"{RENT_PREFIX}.boundary"
RENT_OUTPUT_CONTEXT = f"{RENT_PREFIX}.finalize"
_COMMON_INPUTS = {
    "frame_context",
    "encoding",
    "placement",
    "raw_draw",
    "apply_state",
    "training_state",
}
_SOCIAL = (
    "social_security_retirement",
    "social_security_disability",
    "social_security_survivors",
    "social_security_dependents",
)
_PENSION = (
    "taxable_private_pension_income",
    "tax_exempt_private_pension_income",
    "taxable_public_pension_income",
    "tax_exempt_public_pension_income",
)
_PERSON_REQUIRED = (
    "is_household_head",
    "age",
    "is_female",
    "employment_income_before_lsr",
    "self_employment_income_before_lsr",
)
_HOUSEHOLD_REQUIRED = ("state_fips", "tenure_type")


def _phase(context):
    if context.params != {"phase": codec.RENT_PHASE}:
        raise ValueError("ACS rent requires its exact pre-clone outer phase.")


def _context_view(context):
    bound = codec.artifact(
        context, "frame_context", graph_context.US_FRAME_CONTEXT_TYPE
    )
    document = graph_context._decode(bound.payload)
    graph_context._mass_records(document["mass_log"])
    if set(context.tables) != {"person", "household"}:
        raise ValueError("ACS rent requires actual declared person/household views.")
    for entity, table in context.tables.items():
        codec.strict_ids(table[f"{entity}_id"], label=entity)
        expected = document["entities"][entity]
        if any(
            expected[key] != value
            for key, value in graph_context._row_identity(table, entity).items()
        ) or set(table.columns) - set(expected["columns"]):
            raise ValueError(f"ACS rent {entity} context row/column identity differs.")
    if (
        document["weight_sources"].get("household")
        != context.weights["household"].kind.value
    ):
        raise ValueError("ACS rent context household weight authority differs.")
    person = context.tables["person"]
    if "pre_subsidy_rent" in person and person["pre_subsidy_rent"].notna().any():
        raise ValueError(
            "ACS rent graph requires fresh unavailable recipient rent; refusing overwrite."
        )
    return document, bound


def _same_declared_producer(context, left, right):
    edges = {edge.name: edge for edge in context.node.artifact_inputs}
    if (
        edges[left].producer != edges[right].producer
        or context.artifacts[left].producer_key != context.artifacts[right].producer_key
    ):
        raise ValueError(f"ACS rent {left}/{right} require the same actual producer.")


def _validation_bindings(context):
    return {
        name: codec.binding(context.artifacts[name]) for name in sorted(_COMMON_INPUTS)
    }


def _validate_application(context):
    document, frame_context = _context_view(context)
    encoding = codec.artifact(context, "encoding", codec.ACS_RENT_ENCODING_TYPE)
    descriptor = source.decode_donor_descriptor(encoding.payload)
    placement_value = codec.artifact(
        context, "placement", codec.ACS_RENT_PLACEMENT_TYPE
    )
    placement = codec.decode_placement(placement_value.payload)
    state_value = codec.artifact(
        context, "apply_state", graph_legacy_apply_matrix.MATRIX_APPLY_STATE_TYPE
    )
    state = graph_legacy_apply_matrix.decode_matrix_apply_state(state_value.payload)
    raw_value = codec.artifact(context, "raw_draw", fit_codec.RAW_TARGET_TYPE)
    training_value = codec.artifact(
        context, "training_state", fit_codec.TRAINING_STATE_TYPE
    )
    training, training_state = fit_codec.read_training(training_value.payload)
    _same_declared_producer(context, "raw_draw", "apply_state")
    header = placement.header
    if (
        state["matrix_sha256"] != header["matrix_sha256"]
        or state["matrix_producer_key"] != placement_value.producer_key
    ):
        raise ValueError(
            "ACS rent matrix state differs from actual placement producer/digest."
        )
    for prefix, value in (("frame_context", frame_context), ("encoding", encoding)):
        binding = codec.binding(value)
        if (
            header[f"{prefix}_sha256"] != binding["sha256"]
            or header[f"{prefix}_producer_key"] != binding["producer_key"]
        ):
            raise ValueError(f"ACS rent placement has stale {prefix} binding.")
    application = state["application"]
    chain = application["state"]
    if (
        LegacyQRFTrainingState.from_chain(qrf.QRFChainState.from_dict(chain))
        != training_state
    ):
        raise ValueError(
            "ACS rent application configuration differs from fitted training state."
        )
    if (
        chain["targets"] != ["rent"]
        or chain["completed_targets"] != ["rent"]
        or chain["predictors"] != descriptor["predictors"]
        or chain["entity"] != "household"
        or chain["weight_kind"] != "design"
        or application["seed"] != chain["model_config"]["seed"]
        or application["models"] != training["models"]
        or application["raw_targets"]
        != [{"target": "rent", "sha256": codec.sha(raw_value.payload)}]
        or codec.encode_json(chain["donor_index"])
        != codec.encode_json(descriptor["training_index"])
    ):
        raise ValueError(
            "ACS rent state/raw/fit contract differs from the donor or final rent target."
        )
    person, household = context.tables["person"], context.tables["household"]
    if codec.encode_json(
        qrf._index_identity(person.index).to_dict()
    ) != codec.encode_json(header["person_index"]):
        raise ValueError("ACS rent original person index changed before finalization.")
    for entity, table in (("person", person), ("household", household)):
        if codec.encode_json(
            graph_context._row_identity(table, entity)
        ) != codec.encode_json(header[f"{entity}_identity"]):
            raise ValueError(f"ACS rent current {entity} rows differ from placement.")
    ids, heads, positions = codec.head_association(person, household)
    if (
        not np.array_equal(ids, placement.household_ids)
        or not np.array_equal(heads, placement.head_person_ids)
        or not np.array_equal(positions, placement.head_positions)
    ):
        raise ValueError(
            "ACS rent current household/head association differs from placement."
        )
    index = pd.Index(placement.household_ids)
    raw = fit_codec.read_raw_target(raw_value.payload, target="rent", index=index)
    # Finalization needs only the exact index and checked positions; no feature
    # values are invented or read by the helper at this operation boundary.
    prepared = housing_inputs.AcsRentRecipientPreparation(
        pd.DataFrame(index=index), positions, person.index
    )
    values = housing_inputs.finalize_acs_rent(
        prepared, pd.Series(raw, index=index, name="rent")
    )
    return values, document


class _RentKernel(KernelBase):
    def implementation_hash(self):
        return codec.encoding_implementation_hash(
            type(self),
            codec,
            source,
            graph_context,
            housing_inputs,
            model_input,
            graph_legacy_apply_matrix,
            fit_codec,
            GateReport,
            GateResult,
            qrf,
            qrf_target,
            Frame,
            Weights,
            EntitySchema,
            dependencies=self.capabilities.dependencies,
        )


class USACSRentBoundaryKernel(_RentKernel):
    ref = "us.production.acs_rent_boundary@1"
    capabilities = Capabilities(
        Determinism.DETERMINISTIC,
        structural=StructuralDelta.FILTER,
        numeric=Numeric.PLATFORM_BITWISE,
        dependencies=("numpy", "pandas", "pyarrow"),
    )

    def run(self, context):
        _phase(context)
        if (
            context.node.kernel != self.ref
            or context.node.outputs
            or context.node.sources
            or set(context.artifacts) != {"frame_context"}
            or context.node.artifact_outputs
            or context.node.structural is not StructuralDelta.FILTER
            or set(context.tables) != {"person"}
        ):
            raise ValueError(
                "ACS rent boundary requires its checked population declaration."
            )
        bound = codec.artifact(
            context, "frame_context", graph_context.US_FRAME_CONTEXT_TYPE
        )
        document = graph_context._decode(bound.payload)
        graph_context._mass_records(document["mass_log"])
        person = context.tables["person"]
        codec.strict_ids(person["person_id"], label="person")
        identity = graph_context._row_identity(person, "person")
        if any(
            document["entities"]["person"][key] != value
            for key, value in identity.items()
        ):
            raise ValueError("ACS rent boundary person identity differs from context.")
        # Frame.select prunes unreferenced groups. An all-person stage boundary
        # must refuse that shape before structural execution can change rows.
        for group in US_SCHEMA.group_entities:
            membership = person[US_SCHEMA.membership_column(group)]
            codec.strict_ids(membership, label=f"{group} membership", unique=False)
            if membership.nunique() != document["entities"][group]["rows"]:
                raise ValueError(
                    f"ACS rent boundary would prune unreferenced {group} rows."
                )
        return KernelResult(
            keep=pd.Series(
                True, index=pd.Index(person["person_id"], name="person_id"), dtype=bool
            )
        )


class USACSRentRecipientKernel(_RentKernel):
    ref = "us.production.acs_rent_recipient@1"
    capabilities = Capabilities(
        Determinism.DETERMINISTIC,
        numeric=Numeric.PLATFORM_BITWISE,
        dependencies=("numpy", "pandas", "pyarrow"),
    )

    def run(self, context):
        _phase(context)
        if (
            context.node.kernel != self.ref
            or context.node.outputs
            or context.node.sources
            or set(context.artifacts) != {"frame_context", "encoding"}
            or context.node.artifact_outputs
            != (
                ArtifactOutput("matrix", model_input.RECIPIENT_MATRIX_TYPE),
                ArtifactOutput("placement", codec.ACS_RENT_PLACEMENT_TYPE),
            )
        ):
            raise ValueError(
                "ACS rent recipient requires its exact matrix/placement declaration."
            )
        _, frame_context = _context_view(context)
        encoding = codec.artifact(context, "encoding", codec.ACS_RENT_ENCODING_TYPE)
        descriptor = source.decode_donor_descriptor(encoding.payload)
        view = Frame(
            dict(context.tables),
            EntitySchema(group_entities=("household",)),
            {"household": context.weights["household"]},
        )
        string_runtime = codec.string_runtime_identity()
        prepared = housing_inputs.prepare_acs_rent_recipient(
            view, donor_context=descriptor["encoding_context"]
        )
        matrix = model_input.encode_recipient_matrix(
            prepared.predictors,
            entity="household",
            entity_ids=prepared.predictors.index.to_numpy(copy=False),
        )
        placement = codec.encode_placement(
            prepared,
            person=context.tables["person"],
            household=context.tables["household"],
            matrix_sha256=codec.sha(matrix),
            frame_context=frame_context,
            encoding=encoding,
        )
        codec.assert_string_runtime(string_runtime)
        return KernelResult(
            artifacts={"matrix": matrix, "placement": placement},
            receipt={
                "phase": codec.RENT_PHASE,
                "households": len(prepared.predictors),
                "matrix_sha256": codec.sha(matrix),
                "placement_sha256": codec.sha(placement),
                "string_runtime": string_runtime,
            },
        )


class USACSRentValidationKernel(_RentKernel):
    ref = "us.production.acs_rent_validation@1"
    capabilities = Capabilities(
        Determinism.DETERMINISTIC,
        role=KernelRole.GATE,
        numeric=Numeric.PLATFORM_BITWISE,
        dependencies=("numpy", "pandas", "pyarrow"),
    )

    def run(self, context):
        try:
            _phase(context)
            if (
                context.node.kernel != self.ref
                or context.node.outputs
                or context.node.sources
                or set(context.artifacts) != _COMMON_INPUTS
                or context.node.artifact_outputs
                != (ArtifactOutput("validation", codec.ACS_RENT_VALIDATION_TYPE),)
            ):
                raise ValueError("ACS rent validation declaration is incomplete.")
            values, _ = _validate_application(context)
            passed, failure = True, None
            bindings = _validation_bindings(context)
            details = {"person_rows": len(values)}
        except (ValueError, KeyError, TypeError) as error:
            # Let the graph's existing gate-exception contract record named
            # evidence and mark validation bytes unavailable. A manufactured
            # failed validation payload would still be available to consumers.
            raise ValueError(f"us_acs_rent_assignment_integrity: {error}") from error
        gate = GateResult(
            "us_acs_rent_assignment_integrity",
            passed,
            () if passed else (failure,),
            details,
        )
        payload = codec.encode_json(
            {
                "schema_version": 1,
                "passed": passed,
                "failure": failure,
                "bindings": bindings,
            }
        )
        return KernelResult(
            artifacts={"validation": payload},
            receipt={
                "phase": codec.RENT_PHASE,
                "scope": "rent_assignment_integrity",
                "outcome": "pass" if passed else "fail",
                "evidence": GateReport((gate,)).to_manifest(),
            },
        )


class USACSRentFinalizeKernel(_RentKernel):
    ref = "us.production.acs_rent_finalize@1"
    capabilities = Capabilities(
        Determinism.DETERMINISTIC,
        numeric=Numeric.PLATFORM_BITWISE,
        dependencies=("numpy", "pandas", "pyarrow"),
    )

    def run(self, context):
        _phase(context)
        if (
            context.node.kernel != self.ref
            or context.node.sources
            or set(context.artifacts) != _COMMON_INPUTS | {"validation"}
            or context.node.artifact_outputs
            != (ArtifactOutput("frame_context", graph_context.US_FRAME_CONTEXT_TYPE),)
            or len(context.node.outputs) != 1
            or (
                context.node.outputs[0].entity,
                context.node.outputs[0].column,
                context.node.outputs[0].dtype,
            )
            != ("person", "pre_subsidy_rent", "float64")
        ):
            raise ValueError(
                "ACS rent finalization requires its exact output and checked inputs."
            )
        validation = codec.artifact(
            context, "validation", codec.ACS_RENT_VALIDATION_TYPE
        )
        expected = {
            "schema_version": 1,
            "passed": True,
            "failure": None,
            "bindings": _validation_bindings(context),
        }
        if validation.payload != codec.encode_json(expected):
            raise ValueError(
                "ACS rent finalization requires a matching passed validation gate."
            )
        values, document = _validate_application(context)
        columns = document["entities"]["person"]["columns"]
        if "pre_subsidy_rent" not in columns:
            columns.append("pre_subsidy_rent")
        return KernelResult(
            columns={
                ("person", "pre_subsidy_rent"): pd.Series(
                    values,
                    index=pd.Index(
                        context.tables["person"]["person_id"], name="person_id"
                    ),
                )
            },
            artifacts={"frame_context": codec.encode_json(document)},
            receipt={
                "phase": codec.RENT_PHASE,
                "validation_sha256": codec.sha(validation.payload),
                "person_rows": len(values),
                "scope": "rent_placement_only",
            },
        )


def us_acs_rent_nodes(
    columns,
    *,
    base,
    context_producer,
    donor_descriptor,
    seed,
    n_estimators=housing_inputs._DEFAULT_N_ESTIMATORS,
):
    """Declare the rent operation on the caller's already prepared projection."""
    descriptor = source.decode_donor_descriptor(donor_descriptor)
    inventory = {(col.entity, col.column): col for col in columns}
    if len(inventory) != len(columns):
        raise ValueError("ACS rent input inventory repeats columns.")
    for entity, names in (
        ("person", _PERSON_REQUIRED),
        ("household", _HOUSEHOLD_REQUIRED),
    ):
        if any((entity, name) not in inventory for name in names):
            raise ValueError(f"ACS rent lacks required prepared {entity} inputs.")
    groups = [
        tuple(name for name in names if ("person", name) in inventory)
        for names in (_SOCIAL, _PENSION)
    ]
    if any(not group for group in groups):
        raise ValueError(
            "ACS rent requires declared social-security and pension components."
        )
    optional = (
        ("pre_subsidy_rent",) if ("person", "pre_subsidy_rent") in inventory else ()
    )
    predictor_slices = (
        Slice("person", (*_PERSON_REQUIRED, *groups[0], *groups[1], *optional)),
        Slice("household", _HOUSEHOLD_REQUIRED),
    )
    final_slices = (
        Slice("person", ("is_household_head", *optional)),
        Slice("household", ("tenure_type",)),
    )
    if optional and inventory[("person", "pre_subsidy_rent")].dtype != "float64":
        raise ValueError("ACS rent incumbent must be unavailable float64.")
    donor_id = f"{RENT_PREFIX}.donor"
    recipient_id = f"{RENT_PREFIX}.recipient"
    validation_id = f"{RENT_PREFIX}.validation"
    donor = Node(
        donor_id,
        source.USACSRentDonorKernel.ref,
        sources=(source.RENT_SOURCE_NAME,),
        structural=StructuralDelta.CREATE,
        params={
            "phase": codec.RENT_PHASE,
            "donor_descriptor": donor_descriptor.decode(),
        },
        outputs=tuple(
            Owned("household", name, "float64")
            for name in (*descriptor["predictors"], "rent")
        ),
        artifact_outputs=(ArtifactOutput("encoding", codec.ACS_RENT_ENCODING_TYPE),),
    )
    fits = legacy_qrf_train_nodes(
        f"{RENT_PREFIX}.fit",
        population=donor_id,
        entity="household",
        predictors=tuple(descriptor["predictors"]),
        targets=("rent",),
        seed=seed,
        n_estimators=n_estimators,
        phase=codec.RENT_PHASE,
    )
    shared_edges = (
        ArtifactInput(
            "frame_context",
            context_producer,
            "frame_context",
            graph_context.US_FRAME_CONTEXT_TYPE,
        ),
        ArtifactInput("encoding", donor_id, "encoding", codec.ACS_RENT_ENCODING_TYPE),
    )
    boundary = Node(
        RENT_OUTPUT_POPULATION,
        USACSRentBoundaryKernel.ref,
        base=base,
        structural=StructuralDelta.FILTER,
        inputs=(Slice("person", ("is_household_head",)),),
        params={"phase": codec.RENT_PHASE},
        artifact_inputs=(shared_edges[0],),
    )
    recipient = Node(
        recipient_id,
        USACSRentRecipientKernel.ref,
        population=base,
        inputs=predictor_slices,
        params={"phase": codec.RENT_PHASE},
        artifact_inputs=shared_edges,
        artifact_outputs=(
            ArtifactOutput("matrix", model_input.RECIPIENT_MATRIX_TYPE),
            ArtifactOutput("placement", codec.ACS_RENT_PLACEMENT_TYPE),
        ),
    )
    applies = legacy_qrf_apply_matrix_nodes(
        f"{RENT_PREFIX}.apply",
        population=RENT_OUTPUT_POPULATION,
        fit_nodes=fits,
        matrix_producer=recipient_id,
        seed=seed,
        phase=codec.RENT_PHASE,
    )
    common = (
        *shared_edges,
        ArtifactInput(
            "placement", recipient_id, "placement", codec.ACS_RENT_PLACEMENT_TYPE
        ),
        ArtifactInput(
            "raw_draw", applies[-1].id, "raw_draw", fit_codec.RAW_TARGET_TYPE
        ),
        ArtifactInput(
            "apply_state",
            applies[-1].id,
            "apply_state",
            graph_legacy_apply_matrix.MATRIX_APPLY_STATE_TYPE,
        ),
        ArtifactInput(
            "training_state",
            fits[-1].id,
            "training_state",
            fit_codec.TRAINING_STATE_TYPE,
        ),
    )
    validation = Node(
        validation_id,
        USACSRentValidationKernel.ref,
        population=RENT_OUTPUT_POPULATION,
        inputs=(
            Slice("person", ("is_household_head",)),
            Slice("household", ("tenure_type",)),
        ),
        params={"phase": codec.RENT_PHASE},
        artifact_inputs=common,
        artifact_outputs=(
            ArtifactOutput("validation", codec.ACS_RENT_VALIDATION_TYPE),
        ),
    )
    final = Node(
        RENT_OUTPUT_CONTEXT,
        USACSRentFinalizeKernel.ref,
        population=RENT_OUTPUT_POPULATION,
        inputs=final_slices,
        outputs=(
            Owned("person", "pre_subsidy_rent", "float64", rewrite=bool(optional)),
        ),
        params={"phase": codec.RENT_PHASE},
        artifact_inputs=(
            *common,
            ArtifactInput(
                "validation",
                validation_id,
                "validation",
                codec.ACS_RENT_VALIDATION_TYPE,
            ),
        ),
        artifact_outputs=(
            ArtifactOutput("frame_context", graph_context.US_FRAME_CONTEXT_TYPE),
        ),
    )
    return (donor, *fits, boundary, recipient, *applies, validation, final)


def register_us_acs_rent_kernels(registry: KernelRegistry, *, source_codecs=None):
    for kernel in (
        source.USACSRentDonorKernel(source_codecs),
        LegacyQRFTrainKernel(),
        graph_legacy_apply_matrix.LegacyQRFApplyMatrixKernel(),
        USACSRentBoundaryKernel(),
        USACSRentRecipientKernel(),
        USACSRentValidationKernel(),
        USACSRentFinalizeKernel(),
    ):
        registry.register(kernel)
