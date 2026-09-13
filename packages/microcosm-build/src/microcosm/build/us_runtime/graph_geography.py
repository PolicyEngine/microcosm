"""Declared lookup import, household assignment, and geography validation.

The PUMA operator keeps native ACS PUMAs and draws a supported joint tract/CD
cell for each household; county derives from that tract. Reference import and
the assignment-integrity and joint-support gates are separate graph nodes.
No target values are fitted here.
"""

from __future__ import annotations

import hashlib
from collections.abc import Sequence

import numpy as np
import pandas as pd

from microcosm.build.gates import GateReport
from microcosm.graph import (
    ArtifactInput,
    ArtifactOutput,
    ArtifactType,
    Capabilities,
    Determinism,
    KernelBase,
    KernelContext,
    KernelRegistry,
    KernelResult,
    Node,
    Numeric,
    Owned,
    SeedSource,
    Slice,
    SourceRef,
    StructuralDelta,
)
from microcosm.graph.canonical import canonical_json
from microcosm.graph.codecs import load_source_bytes
from microcosm.graph.kernel import KernelRole
from microcosm.graph.population import dtype_for_token

from .congressional_district_geography import CONGRESSIONAL_DISTRICT_GEOID_COLUMN
from .congressional_district_vintage import (
    CURRENT_CONGRESSIONAL_DISTRICT_PREFIX,
    CURRENT_CONGRESSIONAL_DISTRICT_VINTAGE,
    decode_congressional_district_vintage_crosswalk,
    validate_packaged_congressional_district_vintage_crosswalk,
)
from .graph_context import (
    US_FRAME_CONTEXT_TYPE,
    _decode,
    _mass_records,
    _row_identity,
)
from .graph_implementation import (
    STAGE_DEPENDENCIES,
    implementation_hash,
    implementation_manifest,
)
from .puma_ladder import (
    US_PUMA_LADDER_COLUMNS,
    assign_us_puma_ladder,
    decode_us_puma_ladder,
    us_puma_ladder_assignment_summary,
    us_puma_ladder_gate,
    us_puma_ladder_joint_support_gate,
)

US_PUMA_LOOKUP_TYPE = ArtifactType("microcosm.us.puma_ladder_npz", 1)
US_CD_CROSSWALK_TYPE = ArtifactType("microcosm.us.cd_crosswalk_csv", 1)
GEOGRAPHY_PHASE = "assign_us_puma_ladder"
LOOKUP_SOURCES = (
    SourceRef(
        "us_puma_ladder_2020",
        "raw-bytes-v1",
        "Census population-weighted PUMA overlaps.",
    ),
    SourceRef(
        "us_cd_crosswalk_117_119",
        "raw-bytes-v1",
        "Pinned 117th-to-119th population crosswalk.",
    ),
)


def _lookups(ladder_bytes: bytes, crosswalk_bytes: bytes):
    ladder = decode_us_puma_ladder(ladder_bytes)
    crosswalk = decode_congressional_district_vintage_crosswalk(crosswalk_bytes)
    validate_packaged_congressional_district_vintage_crosswalk(crosswalk)
    expected = tuple(
        sorted(
            int(value.removeprefix(CURRENT_CONGRESSIONAL_DISTRICT_PREFIX))
            for value in crosswalk["target_geography_id"].unique()
        )
    )
    actual = tuple(sorted(int(value) for value in np.unique(ladder.cd_overlap_cd)))
    if actual != expected:
        raise ValueError(
            "PUMA ladder and national crosswalk district universes differ."
        )
    if (
        ladder.layer_vintages["congressional_district"]
        != CURRENT_CONGRESSIONAL_DISTRICT_VINTAGE
    ):
        raise ValueError("PUMA ladder has the wrong congressional district vintage.")
    return ladder, expected


class _USGeographyKernel(KernelBase):
    def implementation_hash(self) -> str:
        return implementation_hash("geography")


class USGeographyLookupKernel(_USGeographyKernel):
    ref = "us.production.geography_lookup@2"
    capabilities = Capabilities(
        determinism=Determinism.DETERMINISTIC,
        numeric=Numeric.PLATFORM_BITWISE,
        dependencies=STAGE_DEPENDENCIES["geography"],
    )

    def run(self, context: KernelContext) -> KernelResult:
        if context.params != {"phase": GEOGRAPHY_PHASE}:
            raise ValueError("Unsupported US geography lookup parameters.")
        ladder_bytes = load_source_bytes(
            "raw-bytes-v1", context.sources["us_puma_ladder_2020"]
        )
        crosswalk_bytes = load_source_bytes(
            "raw-bytes-v1", context.sources["us_cd_crosswalk_117_119"]
        )
        ladder, expected = _lookups(ladder_bytes, crosswalk_bytes)
        return KernelResult(
            artifacts={"ladder": ladder_bytes, "crosswalk": crosswalk_bytes},
            receipt={
                "phase": GEOGRAPHY_PHASE,
                "implementation": implementation_manifest("geography"),
                "district_vintage": CURRENT_CONGRESSIONAL_DISTRICT_VINTAGE,
                "district_count": len(expected),
                "puma_count": len(ladder),
                "ladder_sha256": hashlib.sha256(ladder_bytes).hexdigest(),
                "crosswalk_sha256": hashlib.sha256(crosswalk_bytes).hexdigest(),
            },
        )


class USGeographyBoundaryKernel(_USGeographyKernel):
    """Preserve all rows while opening the geography rewrite version."""

    ref = "us.production.geography_boundary@2"
    capabilities = Capabilities(
        determinism=Determinism.DETERMINISTIC,
        structural=StructuralDelta.FILTER,
        dependencies=STAGE_DEPENDENCIES["geography"],
    )

    def run(self, context: KernelContext) -> KernelResult:
        if context.params != {"phase": GEOGRAPHY_PHASE}:
            raise ValueError("US geography boundary requires its registered phase.")
        person = context.tables["person"]
        return KernelResult(
            receipt={"implementation": implementation_manifest("geography")},
            keep=pd.Series(
                True, index=pd.Index(person["person_id"], name="person_id"), dtype=bool
            ),
        )


class USGeographyAssignKernel(_USGeographyKernel):
    ref = "us.production.geography_assign@2"
    capabilities = Capabilities(
        determinism=Determinism.SEEDED,
        seed_source=SeedSource.PARAM,
        numeric=Numeric.PLATFORM_BITWISE,
        dependencies=STAGE_DEPENDENCIES["geography"],
    )

    def run(self, context: KernelContext) -> KernelResult:
        if context.params != {
            "phase": GEOGRAPHY_PHASE,
            "seed": 0,
            "assign_tract": False,
        }:
            raise ValueError(
                "US geography requires the declared seed and joint county/CD contract."
            )
        artifact = context.artifacts.get("frame_context")
        if artifact is None or artifact.type != US_FRAME_CONTEXT_TYPE:
            raise ValueError("US geography requires a typed frame context.")
        document = _decode(artifact.payload)
        _mass_records(document["mass_log"])
        household = context.tables["household"]
        declared = document["entities"]["household"]
        if any(
            declared[key] != value
            for key, value in _row_identity(household, "household").items()
        ):
            raise ValueError("US geography household identity does not match context.")
        if set(household.columns) - set(declared["columns"]):
            raise ValueError("US geography has undeclared household input columns.")
        if (
            document["weight_sources"].get("household")
            != context.weights["household"].kind.value
        ):
            raise ValueError(
                "US geography requires the matching household weight source."
            )
        ladder, expected = _lookups(
            context.artifacts["ladder"].payload, context.artifacts["crosswalk"].payload
        )
        table = assign_us_puma_ladder(
            household,
            ladder,
            seed=0,
            assign_tract=False,
            expected_congressional_district_vintage=CURRENT_CONGRESSIONAL_DISTRICT_VINTAGE,
        )
        columns = {}
        for owned in context.node.outputs:
            if (
                owned.entity != "household"
                or owned.column not in US_PUMA_LADDER_COLUMNS
            ):
                raise ValueError("US geography has an unsupported owned output.")
            original = table[owned.column]
            converted = original.astype(dtype_for_token(owned.dtype))
            if not original.isna().equals(converted.isna()) or not np.array_equal(
                original.to_numpy(), converted.to_numpy()
            ):
                raise ValueError(
                    "US geography storage conversion changed values or missingness."
                )
            table[owned.column] = converted
            columns[(owned.entity, owned.column)] = pd.Series(
                converted.array,
                index=pd.Index(table["household_id"], name="household_id"),
            )
        if set(column for _, column in columns) != set(US_PUMA_LADDER_COLUMNS):
            raise ValueError(
                "US geography must own exactly its three household outputs."
            )
        # This operator changes only these three cells on each household. The
        # graph boundary preserves every row, weight and membership. Carry the
        # authenticated descriptors for untouched entities without copying
        # their population tables into this household-only operator.
        for column in US_PUMA_LADDER_COLUMNS:
            if column not in declared["columns"]:
                declared["columns"].append(column)
        return KernelResult(
            columns=columns,
            artifacts={"frame_context": canonical_json(document)},
            receipt={
                "phase": GEOGRAPHY_PHASE,
                "implementation": implementation_manifest("geography"),
                "seed": 0,
                "assign_tract": False,
                "district_vintage": CURRENT_CONGRESSIONAL_DISTRICT_VINTAGE,
                "target_district_count": len(expected),
                "summary": us_puma_ladder_assignment_summary(
                    table,
                    ladder,
                    weight_values=context.weights["household"].values,
                ),
            },
        )


class USGeographyGateKernel(_USGeographyKernel):
    ref = "us.production.geography_gate@2"
    capabilities = Capabilities(
        determinism=Determinism.DETERMINISTIC,
        role=KernelRole.GATE,
        numeric=Numeric.PLATFORM_BITWISE,
        dependencies=STAGE_DEPENDENCIES["geography"],
    )

    def run(self, context: KernelContext) -> KernelResult:
        if context.params != {"phase": GEOGRAPHY_PHASE}:
            raise ValueError("US geography gate requires its registered phase.")
        gate = us_puma_ladder_gate(
            context.tables["household"],
            context.weights["household"].values,
            assign_tract=False,
        )
        ladder, _ = _lookups(
            context.artifacts["ladder"].payload, context.artifacts["crosswalk"].payload
        )
        joint_gate = us_puma_ladder_joint_support_gate(
            context.tables["household"],
            ladder,
            assign_tract=False,
            expected_congressional_district_vintage=CURRENT_CONGRESSIONAL_DISTRICT_VINTAGE,
        )
        return KernelResult(
            receipt={
                "outcome": "pass" if gate.passed and joint_gate.passed else "fail",
                "evidence": GateReport((gate, joint_gate)).to_manifest(),
                "scope": "geography_assignment_integrity",
            }
        )


def us_geography_nodes(
    columns: Sequence[Owned], *, base: str, context_producer: str
) -> tuple[Node, ...]:
    """Append a declared production geography stage after source assembly.

    This checks assignment integrity, not CD calibration or release readiness.
    """
    inventory = {(owned.entity, owned.column): owned for owned in columns}
    if len(inventory) != len(columns):
        raise ValueError("US geography input column declarations repeat coordinates.")
    if ("person", "age") not in inventory:
        raise ValueError("US geography requires the assembled person age column.")
    if ("household", "state_fips") not in inventory:
        raise ValueError("US geography requires household state_fips.")
    input_columns = ("state_fips",) + (
        ("puma",) if ("household", "puma") in inventory else ()
    )
    boundary = f"{GEOGRAPHY_PHASE}.boundary"
    lookup = f"{GEOGRAPHY_PHASE}.lookup"
    output_types = {
        "puma": "string",
        CONGRESSIONAL_DISTRICT_GEOID_COLUMN: "int64",
        "county_fips": "string",
    }
    outputs = []
    for column, dtype in output_types.items():
        incumbent = inventory.get(("household", column))
        if incumbent:
            if incumbent.dtype not in (
                {"int64", "Int64"} if dtype == "int64" else {"string"}
            ):
                raise ValueError(f"Unsupported incumbent geography dtype: {column}.")
            dtype = incumbent.dtype
        outputs.append(Owned("household", column, dtype, rewrite=incumbent is not None))
    return (
        Node(
            id=lookup,
            kernel=USGeographyLookupKernel.ref,
            population=base,
            sources=tuple(source.name for source in LOOKUP_SOURCES),
            params={"phase": GEOGRAPHY_PHASE},
            artifact_outputs=(
                ArtifactOutput("ladder", US_PUMA_LOOKUP_TYPE),
                ArtifactOutput("crosswalk", US_CD_CROSSWALK_TYPE),
            ),
        ),
        Node(
            id=boundary,
            kernel=USGeographyBoundaryKernel.ref,
            base=base,
            structural=StructuralDelta.FILTER,
            inputs=(Slice("person", ("age",)),),
            params={"phase": GEOGRAPHY_PHASE},
        ),
        Node(
            id=GEOGRAPHY_PHASE,
            kernel=USGeographyAssignKernel.ref,
            population=boundary,
            inputs=(Slice("household", input_columns),),
            outputs=tuple(outputs),
            params={"phase": GEOGRAPHY_PHASE, "seed": 0, "assign_tract": False},
            artifact_inputs=(
                ArtifactInput(
                    "frame_context",
                    context_producer,
                    "frame_context",
                    US_FRAME_CONTEXT_TYPE,
                ),
                ArtifactInput("ladder", lookup, "ladder", US_PUMA_LOOKUP_TYPE),
                ArtifactInput("crosswalk", lookup, "crosswalk", US_CD_CROSSWALK_TYPE),
            ),
            artifact_outputs=(ArtifactOutput("frame_context", US_FRAME_CONTEXT_TYPE),),
        ),
        Node(
            id=f"{GEOGRAPHY_PHASE}.gate",
            kernel=USGeographyGateKernel.ref,
            population=boundary,
            inputs=(Slice("household", ("state_fips", *US_PUMA_LADDER_COLUMNS)),),
            params={"phase": GEOGRAPHY_PHASE},
            artifact_inputs=(
                ArtifactInput("ladder", lookup, "ladder", US_PUMA_LOOKUP_TYPE),
                ArtifactInput("crosswalk", lookup, "crosswalk", US_CD_CROSSWALK_TYPE),
            ),
        ),
    )


def register_us_geography_kernels(registry: KernelRegistry) -> None:
    for kernel in (
        USGeographyLookupKernel(),
        USGeographyBoundaryKernel(),
        USGeographyAssignKernel(),
        USGeographyGateKernel(),
    ):
        registry.register(kernel)
