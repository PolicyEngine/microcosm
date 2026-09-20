"""Typed development placement fragment, awaiting checked-host integration.

These declarations and detached results are not an issuer or a runnable host.
The integrating owner must admit/recheck its genuine financial and arm-one
ancestors, receiving population, source-qualified fixed inputs, fitted producers
and final artifact/storage fences. No declaration or transport hash grants that
source authority. Every target's actual application remains an explicit edge.
"""

from __future__ import annotations

import numpy as np

from microcosm.graph import (
    ArtifactInput,
    ArtifactOutput,
    ArtifactType,
    KernelResult,
    Node,
    Owned,
    Slice,
    StructuralDelta,
)
from microcosm.graph import population as population_ops

from . import puf55_original_placement as values

application, attachment = values.application, values.attachment
fixed_graph, codec = application.fixed_graph, values.codec
KEEP_NODE = "survey_puf55.original.placement_receiving"
ATTACH_NODE = "survey_puf55.original.placement_attach"
KEEP_REF = "us.survey_puf55.original_placement_keep@1"
ATTACH_REF = "us.survey_puf55.original_placement_attach@1"
PLACEMENT_TYPE = ArtifactType("microcosm.us.puf55_original_placement", 1)
require = values.require


def _application_layout(qualified, routes, *, population, seeds):
    nodes = application.original_application_nodes(
        qualified, routes, population=population, **seeds
    )
    layout, offset = [], 0
    for route in routes:
        count = len(route.fits)
        layout.append((route, nodes[offset : offset + count]))
        offset += count
    return tuple(layout)


def _edges(qualified, layout):
    result = [
        ArtifactInput(
            "qualification",
            fixed_graph.fixed_input_node_id(0),
            "qualification",
            fixed_graph.QUALIFICATION_TYPE,
        ),
        ArtifactInput(
            "source_basis",
            fixed_graph.fixed_input_node_id(0),
            "source_basis",
            fixed_graph.SOURCE_BASIS_TYPE,
        ),
    ]
    for r, (route, applies) in enumerate(layout):
        result.append(
            ArtifactInput(
                f"r{r}_matrix",
                fixed_graph.parent.recipient_node_ids(0)[1],
                fixed_graph.parent._NAMES[route.profile.value],
                fixed_graph.parent.model_input.RECIPIENT_MATRIX_TYPE,
            )
        )
        for target, name in fixed_graph.observed_artifact_names(
            qualified, route.profile.value
        ).items():
            result.append(
                ArtifactInput(
                    f"r{r}_fixed_{target}",
                    fixed_graph.fixed_input_node_id(0),
                    name,
                    application.observed.OBSERVED_TARGET_TYPE,
                )
            )
        for t, (fit, apply) in enumerate(zip(route.fits, applies, strict=True)):
            for node in (fit, apply):
                for artifact in node.artifact_outputs:
                    result.append(
                        ArtifactInput(
                            f"r{r}_t{t}_{artifact.name}",
                            node.id,
                            artifact.name,
                            artifact.type,
                        )
                    )
    return tuple(result)


def original_placement_nodes(
    qualified, inputs, routes, *, after, clone_one_seed, original_application_seed
):
    """Declare keep-all after the actual terminal and eight-or-fewer rewrites.

    The full actual PUF55 roster is 52 person plus 3 tax-unit targets. No SCF
    mortgage fields, QBI-support output or tuition allocation is introduced here.
    """
    require(type(after) is ArtifactInput, "AFTER")
    require(after.producer not in (KEEP_NODE, ATTACH_NODE), "AFTER")
    values._axes(inputs, qualified)
    seeds = dict(
        clone_one_seed=clone_one_seed,
        original_application_seed=original_application_seed,
    )
    # Application reads the already-qualified original matrix; it must use the
    # pre-placement receiving version, not depend on its own output version.
    layout = _application_layout(
        qualified, routes, population=inputs.receiving.version, seeds=seeds
    )
    require(bool(layout), "ROUTES")
    profiles = tuple(route.profile for route, _ in layout)
    require(
        all(profile.targets == profiles[0].targets for profile in profiles),
        "ROUTE_TARGETS",
    )
    candidates = values.candidate_outputs(inputs, profiles[0])
    frame = inputs.receiving.frame
    slices = []
    for entity in frame.entities:
        columns = [frame.schema.entity_id_column(entity)]
        if entity == "person":
            columns.extend(
                frame.schema.membership_column(group)
                for group in frame.schema.group_entities
            )
            columns.extend(qualified.person_values)
        columns.extend(name for e, name, _ in candidates if e == entity)
        slices.append(Slice(entity, tuple(dict.fromkeys(columns))))
    params = {
        "protocol": values.PROTOCOL,
        "profiles": tuple(profile.value for profile in profiles),
        "policy": tuple(sorted(values.output_policy(profiles[0]).items())),
        **seeds,
    }
    keep = Node(
        KEEP_NODE,
        KEEP_REF,
        base=inputs.receiving.version,
        structural=StructuralDelta.FILTER,
        mass="conserve",
        inputs=tuple(slices),
        params=params,
        artifact_inputs=(
            ArtifactInput(
                "arm_one_finalization",
                attachment.ATTACH_NODE,
                "finalization",
                attachment.FINALIZATION_TYPE,
            ),
            ArtifactInput("terminal", after.producer, after.artifact, after.type),
        ),
        description="Preserve all current entities and weights after the checked arm-one and enrichment terminal; open an explicit version for conservative original-arm development placement.",
    )
    attach = Node(
        ATTACH_NODE,
        ATTACH_REF,
        population=KEEP_NODE,
        inputs=tuple(slices),
        outputs=tuple(
            Owned(entity, name, dtype, rewrite=True)
            for entity, name, dtype in candidates
        ),
        params=params,
        artifact_inputs=_edges(qualified, layout),
        artifact_outputs=(ArtifactOutput("placement", PLACEMENT_TYPE),),
        description="Preserve all existing/source-known values; place three modeled unit values and five modeled person values only on eligible complete singletons. Mixed-known units and other person allocation remain unresolved; no pruning or source-observation claim.",
    )
    return keep, attach


def keep_all_population(inputs, node):
    """Independently materialize the declared keep-all, retaining all group axes."""
    require(
        node.id == KEEP_NODE
        and node.kernel == KEEP_REF
        and node.base == inputs.receiving.version
        and node.structural is StructuralDelta.FILTER
        and node.mass == "conserve"
        and not node.outputs,
        "KEEP_DECLARATION",
    )
    stamp = values._stamp(inputs)
    frame = inputs.receiving.frame
    selected = frame.select(np.ones(frame.n("person"), dtype=bool))
    for entity in frame.entities:
        require(
            selected.table(entity).equals(frame.table(entity)), "KEEP_FRAME_CHANGED"
        )
    result = population_ops.patch(inputs.receiving, node, KernelResult(frame=selected))
    require(values._stamp(inputs) == stamp, "KEEP_INPUT_CHANGED")
    return result


def original_placement_result(
    node,
    qualified,
    inputs,
    routes,
    artifacts,
    *,
    after,
    clone_one_seed,
    original_application_seed,
):
    """Check every typed application through the strict merger, then place.

    Called by a future checked host with executor-supplied artifacts. This pure
    function cannot authenticate arbitrary supplied producer keys or ancestors.
    """
    seeds = dict(
        clone_one_seed=clone_one_seed,
        original_application_seed=original_application_seed,
    )
    expected = original_placement_nodes(
        qualified, inputs, routes, after=after, **seeds
    )[1]
    require(node == expected, "DECLARATION")
    stamp = values._stamp(inputs)
    fixed_stamp = values.values.fixed_input_stamp(qualified)
    require(set(artifacts) == {e.name for e in node.artifact_inputs}, "ARTIFACT_ROSTER")
    # Authenticate envelope/type/key consistency. Actual producer keys must be
    # bound by the host to these declared graph producers, not supplied claims.
    for edge in node.artifact_inputs:
        application._edge(artifacts[edge.name], edge.type, edge.artifact)

    def artifact_stamp():
        return tuple(
            (name, v.payload, v.type, v.key, v.producer_key, v.numerics)
            for name, v in sorted(artifacts.items())
        )

    artifact_seal = artifact_stamp()
    require(artifacts["qualification"].payload == qualified.receipt, "QUALIFICATION")
    expected_basis = fixed_graph._payloads(
        qualified,
        {
            route.profile.value: (
                artifacts[f"r{r}_matrix"].payload,
                artifacts[f"r{r}_matrix"].producer_key,
            )
            for r, route in enumerate(routes)
        },
    )["source_basis"]
    require(artifacts["source_basis"].payload == expected_basis, "SOURCE_BASIS")
    transport = []
    for r, route in enumerate(routes):
        transport.append(
            application.OriginalRouteArtifacts(
                route.profile,
                artifacts[f"r{r}_matrix"],
                tuple(
                    application.OriginalTargetArtifacts(
                        *(
                            artifacts[f"r{r}_t{t}_{name}"]
                            for name in (
                                "model",
                                "training_state",
                                "raw_draw",
                                "conditioning",
                                "apply_state",
                            )
                        )
                    )
                    for t in range(len(route.profile.targets))
                ),
                tuple(
                    (target, artifacts[f"r{r}_fixed_{target}"])
                    for target in fixed_graph.observed_artifact_names(
                        qualified, route.profile.value
                    )
                ),
            )
        )
    conditioning, receipt = application.merge_puf55_original_conditioning(
        qualified, tuple(transport), **seeds
    )
    columns, payload = values.placement_result(
        qualified, inputs, conditioning, receipt, profile=routes[0].profile
    )
    require(
        values._stamp(inputs) == stamp
        and values.values.fixed_input_stamp(qualified) == fixed_stamp
        and artifact_stamp() == artifact_seal,
        "RESULT_INPUT_CHANGED",
    )
    return KernelResult(
        columns=columns,
        artifacts={"placement": payload},
        receipt=codec.decode_json(payload),
    )
