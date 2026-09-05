"""Authenticated external ACS rent donor preparation, independent of recipients."""

from functools import partial
from pathlib import Path

import numpy as np
import pandas as pd

from microcosm.fit import _graph_legacy_qrf as fit_codec
from microcosm.fit import qrf
from microcosm.fit.qrf import _index_identity, _IndexIdentity
from microcosm.frame import EntitySchema, Frame, WeightKind, Weights
from microcosm.graph import (
    ArtifactOutput,
    Capabilities,
    Determinism,
    KernelBase,
    KernelResult,
    Numeric,
    Owned,
    SourceRef,
    StructuralDelta,
    load_source,
    source_hash,
)
from microcosm.graph.codecs import SourceCodecRegistry

from . import _graph_acs_rent_artifacts as artifacts
from . import graph_context, housing_inputs

RENT_SOURCE_CODEC = "us-acs-rent-2022-v1"
RENT_SOURCE_NAME = "acs_rent_2022"
RENT_SOURCE = SourceRef(
    RENT_SOURCE_NAME, RENT_SOURCE_CODEC, "Pinned archived ACS 2022 rent donor."
)
_DESCRIPTOR_KEY = "acs_rent_donor_descriptor"
_DESCRIPTOR_FIELDS = {
    "schema_version",
    "source_sha256",
    "encoding_context",
    "predictors",
    "training_rows",
    "training_index",
    "training_sha256",
    "weights_sha256",
}


def _file_sha(path):
    import hashlib

    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def _table_sha(table: pd.DataFrame, predictors: tuple[str, ...]) -> str:
    # These are the only non-model fields on either real preparation table.
    # Verify the complete remaining inventory, rather than dynamically selecting
    # a subset and potentially concealing an unexpected source input.
    values = table.drop(columns=["household_id", "household_weight"], errors="ignore")
    columns = (*predictors, "rent")
    if tuple(values.columns) != columns or any(
        dtype != np.dtype("<f8") for dtype in values.dtypes
    ):
        raise ValueError(
            "ACS rent training values require the exact float64 model inventory."
        )
    return artifacts.sha(values.to_numpy(copy=False).tobytes(order="C"))


def decode_donor_descriptor(payload):
    if not isinstance(payload, bytes) or len(payload) > artifacts.HEADER_MAX_BYTES:
        raise ValueError("ACS rent donor descriptor exceeds its bound.")
    document = artifacts.decode_json(payload)
    if (
        set(document) != _DESCRIPTOR_FIELDS
        or type(document["schema_version"]) is not int
        or document["schema_version"] != 1
    ):
        raise ValueError("Unsupported ACS rent donor descriptor.")
    for key in ("source_sha256", "training_sha256", "weights_sha256"):
        if not artifacts.is_hash(document[key]):
            raise ValueError("Malformed ACS rent donor digest.")
    context = housing_inputs._validated_rent_encoding_context(
        document["encoding_context"]
    )
    if not isinstance(document["predictors"], list) or document["predictors"] != list(
        context["predictors"]
    ):
        raise ValueError("ACS rent donor predictors differ from encoding.")
    rows = document["training_rows"]
    index = _IndexIdentity.from_dict(document["training_index"])
    if (
        type(rows) is not int
        or not 0 < rows <= 10_000
        or index.length != rows
        or index.dtype != "int64"
        or index.class_name not in {"Index", "RangeIndex"}
    ):
        raise ValueError("ACS rent donor row/index contract differs.")
    return document


def _descriptor(prepared, source_sha256):
    context = prepared.context
    return {
        "schema_version": 1,
        "source_sha256": source_sha256,
        "encoding_context": {
            "schema": context["schema"],
            "predictors": list(context["predictors"]),
            "category_levels": {
                name: list(levels)
                for name, levels in context["category_levels"].items()
            },
        },
        "predictors": list(prepared.predictors),
        "training_rows": len(prepared.training),
        "training_index": _index_identity(prepared.training.index).to_dict(),
        "training_sha256": _table_sha(prepared.training, prepared.predictors),
        "weights_sha256": artifacts.sha(
            prepared.training["household_weight"].to_numpy(copy=False).tobytes()
        ),
    }


def load_graph_acs_rent_source(
    path, *, store=None, expected_sha256=housing_inputs.ACS_2022_RENT_ARTIFACT_SHA256
):
    """Load real archived H5 and construct its actual model-staging population.

    Explicit alternate pins support separately named synthetic fixture registries.
    The production registration below always uses the registered archive pin.
    """
    del store
    path = Path(path)
    if not artifacts.is_hash(expected_sha256) or not path.is_file():
        raise ValueError("ACS rent requires a pinned regular local H5 file.")
    donor = housing_inputs.load_acs_2022_rent_donor(
        path, expected_sha256=expected_sha256
    )
    prepared = housing_inputs.prepare_acs_rent_donor(donor)
    if _file_sha(path) != expected_sha256:
        raise ValueError("ACS rent source changed during preparation.")
    index = prepared.training.index
    if index.dtype != np.dtype("int64") or not index.is_unique:
        raise ValueError(
            "ACS rent prepared row labels require original unique int64 values."
        )
    ids = index.to_numpy(copy=True)
    table = prepared.training.loc[:, [*prepared.predictors, "rent"]].copy()
    table.insert(0, "household_id", ids)
    document = _descriptor(prepared, expected_sha256)
    decode_donor_descriptor(artifacts.encode_json(document))
    return Frame(
        {
            "household": table,
            "person": pd.DataFrame({"person_id": ids, "person_household_id": ids}),
        },
        EntitySchema(group_entities=("household",)),
        {
            "household": Weights(
                prepared.training["household_weight"].to_numpy(copy=True),
                WeightKind.DESIGN,
            )
        },
        metadata={_DESCRIPTOR_KEY: document},
    )


def prepare_acs_rent_source_descriptor(
    path, *, expected_sha256=housing_inputs.ACS_2022_RENT_ARTIFACT_SHA256
):
    """Perform local source preflight for explicit compile-time encoded columns."""
    frame = load_graph_acs_rent_source(path, expected_sha256=expected_sha256)
    return artifacts.encode_json(
        graph_context._json_data(frame.metadata[_DESCRIPTOR_KEY])
    )


def register_us_acs_rent_source_codec(registry):
    registry.register(RENT_SOURCE_CODEC, load_graph_acs_rent_source)


def _default_registry():
    registry = SourceCodecRegistry()
    register_us_acs_rent_source_codec(registry)
    return registry


class USACSRentDonorKernel(KernelBase):
    ref = "us.production.acs_rent_donor@1"
    capabilities = Capabilities(
        Determinism.DETERMINISTIC,
        structural=StructuralDelta.CREATE,
        numeric=Numeric.PLATFORM_BITWISE,
        dependencies=("numpy", "pandas", "h5py"),
    )

    def __init__(self, source_codecs=None):
        self.source_codecs = (
            _default_registry() if source_codecs is None else source_codecs
        )

    def implementation_hash(self):
        loader = self.source_codecs.get(RENT_SOURCE_CODEC)
        if isinstance(loader, partial):
            loader = loader.func
        return source_hash(
            type(self),
            loader,
            housing_inputs,
            artifacts,
            fit_codec,
            graph_context,
            qrf,
            Frame,
            Weights,
            EntitySchema,
            dependencies=self.capabilities.dependencies,
        )

    def run(self, context):
        if (
            context.node.kernel != self.ref
            or set(context.params) != {"phase", "donor_descriptor"}
            or context.params["phase"] != artifacts.RENT_PHASE
            or context.node.sources != (RENT_SOURCE_NAME,)
            or context.node.inputs
            or context.node.artifact_inputs
            or context.node.structural is not StructuralDelta.CREATE
        ):
            raise ValueError(
                "ACS rent donor requires its exact external CREATE declaration."
            )
        if not isinstance(context.params["donor_descriptor"], str):
            raise ValueError("ACS rent donor descriptor must be canonical JSON text.")
        payload = context.params["donor_descriptor"].encode()
        descriptor = decode_donor_descriptor(payload)
        expected_outputs = tuple(
            Owned("household", name, "float64")
            for name in (*descriptor["predictors"], "rent")
        )
        if (
            context.node.outputs != expected_outputs
            or context.node.artifact_outputs
            != (ArtifactOutput("encoding", artifacts.ACS_RENT_ENCODING_TYPE),)
        ):
            raise ValueError("ACS rent donor ownership differs from its descriptor.")
        frame = load_source(
            RENT_SOURCE_CODEC,
            context.sources[RENT_SOURCE_NAME],
            registry=self.source_codecs,
        )
        actual = frame.metadata.get(_DESCRIPTOR_KEY)
        if artifacts.encode_json(graph_context._json_data(actual)) != payload:
            raise ValueError(
                "ACS rent prepared descriptor differs from authenticated preflight."
            )
        table = frame.table("household")
        if (
            _table_sha(table, tuple(descriptor["predictors"]))
            != descriptor["training_sha256"]
            or artifacts.sha(frame.weights_for("household").values.tobytes())
            != descriptor["weights_sha256"]
            or artifacts.encode_json(_index_identity(table.index).to_dict())
            != artifacts.encode_json(descriptor["training_index"])
        ):
            raise ValueError(
                "ACS rent actual model rows/index/weights differ from descriptor."
            )
        if frame.weights_for(
            "household"
        ).kind is not WeightKind.DESIGN or not np.array_equal(
            table["household_id"].to_numpy(), table.index.to_numpy()
        ):
            raise ValueError(
                "ACS rent staging IDs/design weights differ from original sample rows."
            )
        return KernelResult(
            frame=frame,
            artifacts={"encoding": payload},
            receipt={
                "phase": artifacts.RENT_PHASE,
                "source_sha256": descriptor["source_sha256"],
                "training_rows": len(table),
                "encoding_sha256": artifacts.sha(payload),
                "row_ids": "prepared_sample_index_labels",
            },
        )
