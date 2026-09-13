"""Real, declared US source construction for the population graph.

The ACS source is a directory containing exactly the two Census archives. Its
content key binds both archives without absolute paths entering the node key.
The ASEC source is the authenticated operator-free raw-stage checkpoint. A
candidate pool or an enriched legacy checkpoint cannot substitute for it. The
prepared current-money source is a directory of the reviewed restoration
inputs; its loader is declared here and executed only by its CREATE kernel.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import asdict, dataclass
from importlib import import_module
from pathlib import Path
from threading import RLock

import numpy as np
import pandas as pd

from microcosm.build.serialization_dtypes import canonicalize_frame_string_dtypes
from microcosm.frame import US_SCHEMA, EntitySchema, Frame
from microcosm.graph import (
    ArtifactInput,
    ArtifactOutput,
    ArtifactType,
    Capabilities,
    Determinism,
    Graph,
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
    WeightTransition,
)
from microcosm.graph.canonical import canonical_json
from microcosm.graph.codecs import SourceCodecRegistry, load_raw_bytes
from microcosm.graph.population import dtype_for_token, token_for_dtype

from .acs_housing_universe_source import ACS_HU_CODEC, load_graph_acs_housing_universe
from .acs_inputs import map_acs_native_inputs
from .acs_pums import AcsPumsSource, build_acs_pums_unit_frame
from .asec_checkpoint import load_asec_raw_stage_checkpoint_v4
from .asec_prepared_source import load_graph_asec_prepared
from .graph_context import (
    US_FRAME_CONTEXT_TYPE,
    _decode,
    _json_data,
    _normative_metadata,
    _row_identity,
    encode_us_frame_context,
)
from .graph_implementation import (
    STAGE_DEPENDENCIES,
    implementation_hash,
    implementation_manifest,
    validate_source_codecs,
)
from .operator_boundary import assert_operator_free_source_frame
from .support_provenance import support_channel_column

# Preserve every historical star export, including public imported bindings.
# Star import explicitly requests all three lazy operations. Ordinary source
# imports leave them unresolved.
__all__ = [
    "annotations",
    "Sequence",
    "asdict",
    "dataclass",
    "Path",
    "np",
    "pd",
    "canonicalize_frame_string_dtypes",
    "US_SCHEMA",
    "EntitySchema",
    "Frame",
    "ArtifactInput",
    "ArtifactOutput",
    "ArtifactType",
    "Capabilities",
    "Determinism",
    "Graph",
    "KernelBase",
    "KernelContext",
    "KernelRegistry",
    "KernelResult",
    "Node",
    "Numeric",
    "Owned",
    "SeedSource",
    "Slice",
    "SourceRef",
    "StructuralDelta",
    "WeightTransition",
    "canonical_json",
    "SourceCodecRegistry",
    "load_raw_bytes",
    "dtype_for_token",
    "token_for_dtype",
    "ACS_HU_CODEC",
    "load_graph_acs_housing_universe",
    "map_acs_native_inputs",
    "AcsPumsSource",
    "build_acs_pums_unit_frame",
    "load_asec_raw_stage_checkpoint_v4",
    "load_graph_asec_prepared",
    "US_FRAME_CONTEXT_TYPE",
    "encode_us_frame_context",
    "STAGE_DEPENDENCIES",
    "implementation_hash",
    "implementation_manifest",
    "validate_source_codecs",
    "assert_operator_free_source_frame",
    "assemble_stacked_spine",  # noqa: F822 - lazy __getattr__ export
    "harmonize_stacked_spine_weights",  # noqa: F822 - lazy __getattr__ export
    "prepare_stacked_spine",  # noqa: F822 - lazy __getattr__ export
    "support_channel_column",
    "ASEC_CODEC",
    "ACS_CODEC",
    "ASEC_PREPARED_CODEC",
    "ASSEMBLY_PHASE",
    "US_STACK_PREPARATION_TYPE",
    "US_SOURCE_DEPENDENCIES",
    "GraphAssemblyResult",
    "load_graph_asec",
    "load_graph_acs",
    "us_source_codecs",
    "assembly_from_sources",
    "frame_column_declarations",
    "USAssemblyCreateKernel",
    "USSpineHarmonizeKernel",
    "us_assembly_graph",
    "us_assembly_registry",
    "import_module",
    "RLock",
]

_STACKED_ALIASES = frozenset(
    {
        "assemble_stacked_spine",
        "prepare_stacked_spine",
        "harmonize_stacked_spine_weights",
    }
)
_STACKED_LOCK = RLock()


def _stacked_alias(name: str):
    if name in globals():
        return globals()[name]
    value = getattr(import_module("microcosm.build.us_runtime.stacked_spine"), name)
    with _STACKED_LOCK:
        return globals().setdefault(name, value)


def __getattr__(name: str):
    if name in _STACKED_ALIASES:
        return _stacked_alias(name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(set(globals()) | _STACKED_ALIASES)


ASEC_CODEC = "us-asec-raw-stage-v4"
ACS_CODEC = "us-acs-native-2024-v1"
ASEC_PREPARED_CODEC = "us-asec-prepared-current-money-v3"
ASSEMBLY_PHASE = "assemble_stacked_spine"
US_STACK_PREPARATION_TYPE = ArtifactType("microcosm.us.stacked_spine_preparation", 1)
US_SOURCE_DEPENDENCIES = STAGE_DEPENDENCIES["assembly_prepare"]


@dataclass(frozen=True)
class GraphAssemblyResult:
    frame: Frame
    receipt: object
    dtype_transitions: tuple[dict[str, str], ...]


def _canonical_assembly(frame: Frame, sources: tuple[Frame, ...]):
    """Represent source-declared integer/bool/string absence with nullable storage.

    Legacy union assembly fills absent non-float columns with object None.
    The graph has no arbitrary-object dtype. Promotion is determined by the
    actual source dtype, never by guessing from the assembled observations.
    Values and the missing mask must be identical after the explicit boundary.
    """
    transitions = []
    for entity in US_SCHEMA.entities:
        table = frame.table(entity)
        for column in table:
            original = table[column]
            if not pd.api.types.is_object_dtype(original.dtype):
                continue
            dtypes = [
                source.table(entity)[column].dtype
                for source in sources
                if column in source.table(entity)
            ]
            if not dtypes:
                continue  # graph-added provenance strings are handled below
            if all(pd.api.types.is_bool_dtype(dtype) for dtype in dtypes):
                target = "boolean"
            elif all(pd.api.types.is_integer_dtype(dtype) for dtype in dtypes):
                target = "Int64"
            elif all(isinstance(dtype, pd.StringDtype) for dtype in dtypes):
                target = "string"
            else:
                continue
            promoted = original.astype(dtype_for_token(target))
            if not original.isna().equals(promoted.isna()):
                raise ValueError(
                    f"US graph dtype promotion changed nulls: {entity}.{column}."
                )
            observed = original.notna()
            if not np.array_equal(
                original[observed].to_numpy(), promoted[observed].to_numpy()
            ):
                raise ValueError(
                    f"US graph dtype promotion changed values: {entity}.{column}."
                )
            table[column] = promoted
            transitions.append(
                {"entity": entity, "column": column, "from": "object", "to": target}
            )
    canonicalize_frame_string_dtypes(frame, boundary="US graph assembly", in_place=True)
    # Build checkpoint strings use NumPy NaN; the graph column codec requires
    # nullable pandas strings. This is an explicit physical storage boundary.
    for entity in US_SCHEMA.entities:
        table = frame.table(entity)
        for column in table:
            original = table[column]
            if isinstance(
                original.dtype, pd.StringDtype
            ) and original.dtype != dtype_for_token("string"):
                promoted = original.astype(dtype_for_token("string"))
                if not original.isna().equals(promoted.isna()):
                    raise ValueError(
                        f"US graph string promotion changed nulls: {entity}.{column}."
                    )
                observed = original.notna()
                if not np.array_equal(
                    original[observed].to_numpy(), promoted[observed].to_numpy()
                ):
                    raise ValueError(
                        f"US graph string promotion changed values: {entity}.{column}."
                    )
                table[column] = promoted
                transitions.append(
                    {
                        "entity": entity,
                        "column": column,
                        "from": "string[python,nan]",
                        "to": "string[python,pd.NA]",
                    }
                )
    return tuple(transitions)


def load_graph_asec(path: Path, *, store=None) -> Frame:
    return _decode_graph_asec(path, store=store)


def _decode_graph_asec(path: Path, *, store=None) -> Frame:
    del store
    frame, _binding = load_asec_raw_stage_checkpoint_v4(path)
    return frame


def load_graph_acs(path: Path, *, store=None) -> Frame:
    return _decode_graph_acs(path, store=store)


def _decode_graph_acs(path: Path, *, store=None) -> Frame:
    del store
    expected = {"csv_hus.zip", "csv_pus.zip"}
    if not path.is_dir() or {entry.name for entry in path.iterdir()} != expected:
        raise ValueError("US ACS graph source must contain exactly the two archives.")
    if any(not (path / name).is_file() for name in expected):
        raise ValueError("US ACS graph source archives must be regular files.")
    source = AcsPumsSource(path / "csv_hus.zip", path / "csv_pus.zip", vintage=2024)
    raw, _receipt = build_acs_pums_unit_frame(source)
    mapped = map_acs_native_inputs(raw)
    assert_operator_free_source_frame(
        mapped.frame, label="US graph ACS source", native_inputs=mapped.native_inputs
    )
    return mapped.frame


def us_source_codecs() -> SourceCodecRegistry:
    codecs = SourceCodecRegistry()
    codecs.register(ASEC_CODEC, load_graph_asec)
    codecs.register(ACS_CODEC, load_graph_acs)
    # The prepared current-money directory source. Its loader lives with the
    # preparation it names, so registering it here adds no import cycle and no
    # second preparation path.
    codecs.register(ASEC_PREPARED_CODEC, load_graph_asec_prepared)
    codecs.register_bytes("raw-bytes-v1", load_raw_bytes)
    codecs.register(ACS_HU_CODEC, load_graph_acs_housing_universe)
    validate_source_codecs(codecs)
    return codecs


def assembly_from_sources(
    asec_path: Path, acs_path: Path, *, sample_fraction: float, sample_seed: int
):
    """The direct-call parity oracle and the CREATE kernel share this operation."""
    asec = load_graph_asec(asec_path)
    acs = load_graph_acs(acs_path)
    result = _stacked_alias("assemble_stacked_spine")(
        asec, acs, sample_fraction=sample_fraction, sample_seed=sample_seed
    )
    transitions = _canonical_assembly(result.frame, (asec, acs))
    return GraphAssemblyResult(result.frame, result.receipt, transitions)


def frame_column_declarations(frame: Frame) -> tuple[Owned, ...]:
    """Inventory data cells; the graph owns IDs and membership implicitly."""
    if frame.schema != US_SCHEMA:
        raise ValueError("US graph columns require the US schema.")
    return tuple(
        Owned(entity, column, token_for_dtype(table[column].dtype))
        for entity in US_SCHEMA.entities
        for table in (frame.table(entity),)
        for column in table.columns
        if column != US_SCHEMA.entity_id_column(entity)
        and not (
            entity == US_SCHEMA.person_entity
            and column
            in {
                US_SCHEMA.membership_column(group) for group in US_SCHEMA.group_entities
            }
        )
    )


def _source_implementation_hash() -> str:
    """Compatibility spelling for the new preparation identity only."""
    # This runs before cache lookup, including resume=require. A renamed or
    # substituted public loader cannot claim the declared codec implementation.
    us_source_codecs()
    return implementation_hash("assembly_prepare")


class USAssemblyCreateKernel(KernelBase):
    ref = "us.production.spine_prepare@2"
    capabilities = Capabilities(
        determinism=Determinism.SEEDED,
        numeric=Numeric.PLATFORM_BITWISE,
        seed_source=SeedSource.PARAM,
        structural=StructuralDelta.CREATE,
        dependencies=US_SOURCE_DEPENDENCIES,
    )

    def implementation_hash(self) -> str:
        return _source_implementation_hash()

    def run(self, context: KernelContext) -> KernelResult:
        if set(context.params) != {"phase", "sample_fraction", "sample_seed"}:
            raise ValueError("US assembly parameters have an unsupported contract.")
        if context.params["phase"] != ASSEMBLY_PHASE:
            raise ValueError("US assembly has an incorrect outer phase.")
        asec = load_graph_asec(context.sources["asec_raw_stage"])
        acs = load_graph_acs(context.sources["acs_native"])
        result = _stacked_alias("prepare_stacked_spine")(
            asec,
            acs,
            sample_fraction=context.params["sample_fraction"],
            sample_seed=context.params["sample_seed"],
        )
        transitions = _canonical_assembly(result.frame, (asec, acs))
        if frame_column_declarations(result.frame) != context.node.outputs:
            raise ValueError(
                "US assembly source columns differ from the declared inventory."
            )
        return KernelResult(
            frame=result.frame,
            receipt={
                "phase": ASSEMBLY_PHASE,
                "implementation": implementation_manifest("assembly_prepare"),
                "preparation": result.receipt,
                "dtype_transitions": transitions,
            },
            artifacts={
                "frame_context": encode_us_frame_context(result.frame),
                "preparation": canonical_json(_json_data(result.receipt)),
            },
        )


class USSpineHarmonizeKernel(KernelBase):
    ref = "us.production.spine_harmonize@2"
    capabilities = Capabilities(
        determinism=Determinism.DETERMINISTIC,
        numeric=Numeric.PLATFORM_BITWISE,
        structural=StructuralDelta.REWEIGHT,
        dependencies=STAGE_DEPENDENCIES["assembly_harmonize"],
    )

    def implementation_hash(self) -> str:
        return implementation_hash("assembly_harmonize")

    def run(self, context: KernelContext) -> KernelResult:
        if context.params != {"phase": ASSEMBLY_PHASE}:
            raise ValueError(
                "US spine harmonization has an unsupported phase contract."
            )
        bound = context.artifacts["frame_context"]
        prepared = context.artifacts["preparation"]
        if (
            bound.type != US_FRAME_CONTEXT_TYPE
            or prepared.type != US_STACK_PREPARATION_TYPE
            or bound.producer_key != prepared.producer_key
        ):
            raise ValueError(
                "US spine preparation/context must share a typed producer."
            )
        document = _decode(bound.payload)
        if document["weight_sources"] != {"household": "design"}:
            raise ValueError(
                "US spine harmonization requires its explicit DESIGN anchor."
            )
        for entity in ("person", "household"):
            if any(
                document["entities"][entity][key] != value
                for key, value in _row_identity(context.tables[entity], entity).items()
            ):
                raise ValueError(f"US spine {entity} context identity differs.")
        # This is canonical, producer-owned JSON, not a self-authored declaration.
        import json

        preparation = json.loads(prepared.payload)
        if canonical_json(preparation) != prepared.payload:
            raise ValueError("US spine preparation must be canonical JSON.")
        if preparation["assembly_metadata"] != document["metadata"]:
            raise ValueError("US spine preparation and context metadata differ.")
        result = _stacked_alias("harmonize_stacked_spine_weights")(
            household=context.tables["household"],
            weights=context.weights["household"],
            preparation=preparation,
        )
        # The graph's declared mass is person mass by source stratum. Compute
        # it with the same Frame reducer from minimal declared membership views.
        tables = {
            "person": context.tables["person"][["person_id", "person_household_id"]],
            "household": context.tables["household"][["household_id"]],
        }
        schema = EntitySchema(group_entities=("household",))
        before = Frame(
            tables, schema, {"household": context.weights["household"]}, context.strata
        )
        after = Frame(tables, schema, {"household": result.weights}, context.strata)
        before_mass, after_mass = before.stratum_mass(), after.stratum_mass()
        document["metadata"] = _normative_metadata(result.metadata)
        document["mass_log"] = [
            _json_data(asdict(item)) for item in result.legacy_mass_log
        ]
        document["weight_sources"] = {"household": "importance"}
        return KernelResult(
            weights=result.weights,
            artifacts={"frame_context": canonical_json(document)},
            receipt={
                "phase": ASSEMBLY_PHASE,
                "implementation": implementation_manifest("assembly_harmonize"),
                "assembly": result.receipt,
                "mass": {
                    "policy": "declared",
                    "before": float(before_mass.sum()),
                    "after": float(after_mass.sum()),
                    "stratum_before": before_mass.to_dict(),
                    "stratum_after": after_mass.to_dict(),
                },
            },
        )


def us_assembly_graph(
    columns: Sequence[Owned], *, sample_fraction: float, sample_seed: int
) -> Graph:
    """Construct only the assembly development graph, never a release pool."""
    return Graph(
        country="us",
        sources=(
            SourceRef(
                "asec_raw_stage", ASEC_CODEC, "Operator-free ASEC raw-stage checkpoint."
            ),
            SourceRef(
                "acs_native", ACS_CODEC, "2024 ACS one-year national PUMS archives."
            ),
        ),
        nodes=(
            Node(
                id=f"{ASSEMBLY_PHASE}.prepare",
                kernel=USAssemblyCreateKernel.ref,
                outputs=tuple(columns),
                structural=StructuralDelta.CREATE,
                sources=("asec_raw_stage", "acs_native"),
                params={
                    "phase": ASSEMBLY_PHASE,
                    "sample_fraction": sample_fraction,
                    "sample_seed": sample_seed,
                },
                artifact_outputs=(
                    ArtifactOutput("frame_context", US_FRAME_CONTEXT_TYPE),
                    ArtifactOutput("preparation", US_STACK_PREPARATION_TYPE),
                ),
            ),
            Node(
                id=ASSEMBLY_PHASE,
                kernel=USSpineHarmonizeKernel.ref,
                base=f"{ASSEMBLY_PHASE}.prepare",
                structural=StructuralDelta.REWEIGHT,
                inputs=(
                    Slice("household", (support_channel_column("household"),)),
                    Slice("person", (support_channel_column("person"),)),
                ),
                weights=WeightTransition("household", "importance", mass="declared"),
                mass="declared",
                params={"phase": ASSEMBLY_PHASE},
                artifact_inputs=(
                    ArtifactInput(
                        "frame_context",
                        f"{ASSEMBLY_PHASE}.prepare",
                        "frame_context",
                        US_FRAME_CONTEXT_TYPE,
                    ),
                    ArtifactInput(
                        "preparation",
                        f"{ASSEMBLY_PHASE}.prepare",
                        "preparation",
                        US_STACK_PREPARATION_TYPE,
                    ),
                ),
                artifact_outputs=(
                    ArtifactOutput("frame_context", US_FRAME_CONTEXT_TYPE),
                ),
            ),
        ),
    )


def us_assembly_registry() -> KernelRegistry:
    kernels = KernelRegistry()
    kernels.register(USAssemblyCreateKernel())
    kernels.register(USSpineHarmonizeKernel())
    return kernels
