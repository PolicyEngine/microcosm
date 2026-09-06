"""Real-graph consumer for one signed nominal December-2022 SIPP equity target.

Source-only donor evidence, a single complete-roster predictor, and the real
weighted sign-regime QRF kernels. Donor design weights are SIPP source weights,
never calibrated recipient weights. Nothing here is a US population estimate,
a year transfer, a PolicyEngine input or an adequacy claim.
"""

from __future__ import annotations

import json
import os
import shutil
import stat
from collections import Counter
from dataclasses import dataclass
from decimal import Decimal
from functools import partial
from pathlib import Path

import numpy as np
import pandas as pd

import microcosm.fit.graph_models as graph_models
from microcosm.frame import EntitySchema, Frame, WeightKind, Weights
from microcosm.graph import (
    ArtifactInput,
    ArtifactOutput,
    ArtifactType,
    Capabilities,
    Determinism,
    Graph,
    KernelBase,
    KernelContext,
    KernelResult,
    Node,
    Owned,
    Slice,
    SourceRef,
    StructuralDelta,
    load_source,
    source_hash,
)
from microcosm.graph.canonical import canonical_json, sha256_domain
from microcosm.graph.store import ContentStore

from . import sipp_december_readback as readback
from . import sipp_december_source as s
from . import sipp_equity_contract as contract_module
from . import sipp_measurements as m

__all__ = [
    "DONOR_CODEC",
    "DONOR_POPULATION",
    "DONOR_SUPPORT_TYPE",
    "DRAW_NODE",
    "EquityGraphError",
    "FIT_NODE",
    "RECIPIENT_CODEC",
    "RECIPIENT_POPULATION",
    "ROSTER_NODE",
    "VALIDATE_NODE",
    "SippDonorCreateKernel",
    "SippRecipientRosterKernel",
    "SippSupportValidationKernel",
    "build_sipp_equity_graph",
    "load_sipp_equity_donor_frame",
    "prepare_source_store",
    "register_codecs",
]

DONOR_CODEC = "sipp-december-equity-v1"
RECIPIENT_CODEC = "csv-tables"
DONOR_POPULATION = "sipp_december_equity_donors"
RECIPIENT_POPULATION = "sipp_equity_recipients"
ROSTER_NODE = RECIPIENT_POPULATION
VALIDATE_NODE = "sipp.equity.support"
FIT_NODE = "sipp.equity.fit"
DRAW_NODE = "sipp.equity.draw"
DONOR_SUPPORT_TYPE = ArtifactType("microcosm.build.sipp_equity_support", 1)
_SUPPORT_MAX_BYTES = 64 * 1024
_SCRATCH_DIRNAME = "readback-scratch"
_STORE_MEMBERS = ("objects", "tmp")


class EquityGraphError(ValueError):
    """Sanitized refusal at this consumer's source or predictor boundary."""


# --------------------------------------------------------------------------
# Prepared read-only source store
# --------------------------------------------------------------------------
def _copy_object(source_root: Path, dest_root: Path, key: str, payload: str) -> None:
    src = ContentStore.__new__(ContentStore)
    # object_path is a pure function of the root; build the two paths directly
    # rather than constructing a store over a directory we are still preparing.
    src_dir = source_root / "objects" / key[:2] / key
    dest_dir = dest_root / "objects" / key[:2] / key
    if not src_dir.is_dir() or src_dir.is_symlink():
        raise EquityGraphError("SOURCE_OBJECT_MISSING")
    names = sorted(entry.name for entry in os.scandir(src_dir))
    if names != sorted(("meta.json", payload)):
        raise EquityGraphError("SOURCE_OBJECT_LAYOUT")
    dest_dir.mkdir(parents=True)
    for name in names:
        origin = src_dir / name
        if origin.is_symlink() or not stat.S_ISREG(origin.lstat().st_mode):
            raise EquityGraphError("SOURCE_OBJECT_NOT_REGULAR")
        shutil.copyfile(origin, dest_dir / name)
    del src


def prepare_source_store(accepted_store_root, destination, *, contract=None) -> Path:
    """Copy exactly the two accepted objects into a fresh private source store.

    The destination gets ``objects/`` and an empty ``tmp/`` so a ContentStore can
    be constructed over it read-only. No original CSV and no unrelated store
    object is copied. The destination must not already exist.
    """
    contract = contract or contract_module.load_contract()
    pins = contract.source
    source_root = Path(accepted_store_root)
    dest = Path(destination)
    if dest.exists():
        raise EquityGraphError("SOURCE_STORE_EXISTS")
    (dest / "objects").mkdir(parents=True)
    (dest / "tmp").mkdir()
    _copy_object(source_root, dest, pins["receipt_key"], "payload.json")
    _copy_object(source_root, dest, pins["projection_key"], "payload.bin")
    return dest


def _check_closed_roster(root: Path, contract) -> None:
    """Refuse extra members, symlinks and unexpected file types at our boundary.

    The executor resolves a source path strictly before the codec sees it, so
    this checks the directory this codec actually controls, not the caller's
    original path.
    """
    pins = contract.source
    if sorted(entry.name for entry in os.scandir(root)) != sorted(_STORE_MEMBERS):
        raise EquityGraphError("SOURCE_STORE_MEMBERS")
    tmp = root / "tmp"
    if tmp.is_symlink() or not tmp.is_dir() or any(os.scandir(tmp)):
        raise EquityGraphError("SOURCE_STORE_TMP_NOT_EMPTY")
    expected = {pins["receipt_key"], pins["projection_key"]}
    found = set()
    for shard in os.scandir(root / "objects"):
        if shard.is_symlink() or not shard.is_dir():
            raise EquityGraphError("SOURCE_STORE_SHARD")
        for obj in os.scandir(shard.path):
            if obj.is_symlink() or not obj.is_dir():
                raise EquityGraphError("SOURCE_STORE_OBJECT")
            found.add(obj.name)
    if found != expected:
        raise EquityGraphError("SOURCE_STORE_ROSTER")


# --------------------------------------------------------------------------
# Target-specific admission over the complete reconstructed roster
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class _Admitted:
    key: tuple[str, int]
    members: tuple[tuple[str, int], ...]
    size: int
    weight: Decimal
    target: Decimal


def _target_measurement(household, field: str) -> m.Measurement:
    found = [v for v in household.measurements if v.concept == field]
    if len(found) != 1:
        raise EquityGraphError("TARGET_NOT_EXACTLY_ONE_MEASUREMENT")
    return found[0]


def admit_households(households, contract) -> tuple[tuple[_Admitted, ...], dict]:
    """Apply only the target-specific admission rule, keeping every exclusion.

    The rule is: known household-copy-agreed equity, known complete-member mean
    weight, and a strictly positive mean weight. Flag 9 is retained as
    component-derived with allocation unresolved; admission never relabels it.
    """
    field = contract.target["source_field"]
    admitted: list[_Admitted] = []
    excluded: list[dict] = []
    reasons: Counter = Counter()
    for household in households:
        measure = _target_measurement(household, field)
        weight = household.weight
        why: list[str] = []
        if measure.value is None or measure.status != "household_copy_agreement":
            why.append("target_unknown")
        elif weight.value is None or weight.status != "mean_member_weight":
            why.append("weight_unknown")
        elif weight.value <= 0:
            why.append("weight_not_positive")
        if why:
            reasons[why[0]] += 1
            excluded.append(
                {
                    "key": [household.key[0], int(household.key[1])],
                    "reason": why[0],
                    "measurement_reasons": list(measure.reasons),
                    "weight_reasons": list(weight.reasons),
                }
            )
            continue
        admitted.append(
            _Admitted(
                key=(household.key[0], int(household.key[1])),
                members=tuple(household.members),
                size=len(household.members),
                weight=weight.value,
                target=measure.value,
            )
        )
    admitted.sort(key=lambda a: a.key)
    excluded.sort(key=lambda e: (e["key"][0], e["key"][1]))
    return tuple(admitted), {
        "rule_id": contract.admission["rule_id"],
        "denominator_residences": len(households),
        "admitted_residences": len(admitted),
        "excluded_residences": len(excluded),
        "excluded_reasons_exclusive": dict(sorted(reasons.items())),
        "excluded": excluded,
    }


def coverage_report(readback_result, contract) -> dict:
    """Full source coverage over the complete roster, before any admission.

    Every denominator, known/unknown split, weight profile and flag/quality
    count comes from the producer's approved Decimal helpers over all members.
    No population weight is aggregated by this module.
    """
    households = readback_result.households
    field = contract.target["source_field"]
    index = s._COPIED.index(field)
    return {
        "roster": s._counts(households),
        "target": s._profile(h.measurements[index] for h in households),
        "diagnostics": {
            name: s._profile(h.measurements[i] for h in households)
            for i, name in enumerate(s._COPIED)
            if name != field
        },
        "bank": s._profile(readback_result.bank),
        "household_weights": s._weights(h.weight for h in households),
        "person_weights": s._weights(
            w for h in households for w in h.person_weights
        ),
        "flag_9_policy": contract.admission["flag_9_policy"],
        "vehicle_count_is_not_an_admission_requirement": True,
    }


def _checked_float(value: Decimal, code: str) -> float:
    """Convert exact source Decimal evidence to the Frame's float64 weight type."""
    converted = float(value)
    if not np.isfinite(converted) or converted <= 0.0:
        raise EquityGraphError(code)
    if Decimal(converted).is_nan() or Decimal(converted).is_infinite():
        raise EquityGraphError(code)
    return converted


def _donor_frame(admitted, coverage, admission, contract, readback_result) -> Frame:
    if not admitted:
        raise EquityGraphError("NO_ADMITTED_DONORS")
    target_name = contract.target["name"]
    household_ids = list(range(1, len(admitted) + 1))
    lineage_households = [
        {"household_id": i, "ssuid": a.key[0], "residence": a.key[1]}
        for i, a in zip(household_ids, admitted, strict=True)
    ]
    person_rows = []
    lineage_people = []
    person_id = 0
    for household_id, entry in zip(household_ids, admitted, strict=True):
        for ssuid, pnum in sorted(entry.members):
            person_id += 1
            person_rows.append((person_id, household_id))
            lineage_people.append(
                {"person_id": person_id, "ssuid": ssuid, "pnum": int(pnum)}
            )
    person = pd.DataFrame(
        person_rows, columns=["person_id", "person_household_id"]
    ).astype({"person_id": "int64", "person_household_id": "int64"})
    household = pd.DataFrame(
        {
            "household_id": np.array(household_ids, dtype=np.int64),
            contract.predictor["name"]: np.array(
                [a.size for a in admitted], dtype=np.int64
            ),
            target_name: np.array(
                [float(a.target) for a in admitted], dtype=np.float64
            ),
        }
    )
    weights = np.array(
        [_checked_float(a.weight, "WEIGHT_CONVERSION") for a in admitted],
        dtype=np.float64,
    )
    lineage = {
        "grain": contract.target["grain"],
        "households": lineage_households,
        "people": lineage_people,
    }
    metadata = {
        "consumer_contract_identity": contract.identity,
        "producer_binding": readback_result.producer_binding,
        "projection_key": readback_result.projection_key,
        "receipt_key": readback_result.receipt_key,
        "target": target_name,
        "target_period": contract.target["period"],
        "weight_kind": "design",
        "weight_basis": "sipp_mean_member_wpfinwgt_december_2022",
        "weight_is_calibrated": False,
        "exact_source_weights": [str(a.weight) for a in admitted],
        "exact_source_targets": [str(a.target) for a in admitted],
        "admission": admission,
        "coverage": coverage,
        "lineage_digest": sha256_domain(
            "sipp-december-equity-lineage/1", canonical_json(lineage)
        ),
        "lineage": lineage,
    }
    return Frame(
        {"person": person, "household": household},
        EntitySchema(person_entity="person", group_entities=("household",)),
        {"household": Weights(weights, WeightKind.DESIGN)},
        metadata=metadata,
    )


def load_sipp_equity_donor_frame(
    path, *, store: ContentStore | None = None, contract=None
) -> Frame:
    """Frame codec: prepared closed source store -> admitted donor Frame.

    ``contract`` is the explicit test-only construction route for invented pins;
    the production registration binds the packaged contract.
    """
    del store
    contract = contract or contract_module.load_contract()
    root = Path(path)
    _check_closed_roster(root, contract)
    scratch = root.parent / _SCRATCH_DIRNAME
    scratch.mkdir(exist_ok=True)
    result = readback.read_sipp_december_projection(
        ContentStore(root), scratch, contract=contract
    )
    coverage = coverage_report(result, contract)
    admitted, admission = admit_households(result.households, contract)
    return _donor_frame(admitted, coverage, admission, contract, result)


def register_codecs(registry, *, contract=None) -> None:
    """Register this consumer's Frame codec on a caller-owned registry."""
    if contract is None:
        registry.register(DONOR_CODEC, load_sipp_equity_donor_frame)
        return
    registry.register(
        DONOR_CODEC, partial(load_sipp_equity_donor_frame, contract=contract)
    )


# --------------------------------------------------------------------------
# Kernels
# --------------------------------------------------------------------------
def _encode_support(payload: dict) -> bytes:
    encoded = canonical_json(payload)
    if len(encoded) > _SUPPORT_MAX_BYTES:
        raise EquityGraphError("SUPPORT_ARTIFACT_TOO_LARGE")
    return encoded


class SippDonorCreateKernel(KernelBase):
    """CREATE the admitted donor population and its typed support artifact."""

    ref = "build.sipp.equity.donor@1"
    capabilities = Capabilities(
        Determinism.DETERMINISTIC, structural=StructuralDelta.CREATE
    )

    def implementation_hash(self):
        return source_hash(
            type(self),
            readback,
            contract_module,
            s,
            m,
            dependencies=self.capabilities.dependencies,
        )

    def __init__(self, *, contract=None, registry=None):
        # Both are the explicit test-only construction route; production uses
        # the packaged contract and the shipped registry.
        self._contract = contract
        self._registry = registry

    def run(self, context: KernelContext) -> KernelResult:
        node = context.node
        contract = self._contract or contract_module.load_contract()
        if node.artifact_outputs != (
            ArtifactOutput("donor_support", DONOR_SUPPORT_TYPE),
        ) or node.artifact_inputs:
            raise EquityGraphError("DONOR_NODE_ARTIFACTS")
        if set(context.params) != {"concept", "period"}:
            raise EquityGraphError("DONOR_NODE_PARAMS")
        contract.require_request(
            concept=context.params["concept"], period=context.params["period"]
        )
        (name,) = node.sources
        frame = (
            load_source(DONOR_CODEC, context.sources[name])
            if self._registry is None
            else load_source(
                DONOR_CODEC, context.sources[name], registry=self._registry
            )
        )
        table = frame.table("household")
        sizes = sorted({int(v) for v in table[contract.predictor["name"]]})
        metadata = frame.metadata
        support = {
            "supported_household_sizes": sizes,
            "consumer_contract_identity": contract.identity,
            "producer_binding": metadata["producer_binding"],
            "projection_key": metadata["projection_key"],
            "receipt_key": metadata["receipt_key"],
            "admission_rule_id": contract.admission["rule_id"],
            "admitted_residences": int(len(table)),
            "target": contract.target["name"],
            "target_period": contract.target["period"],
            "predictor": contract.predictor["name"],
        }
        return KernelResult(
            frame=frame,
            artifacts={"donor_support": _encode_support(support)},
            receipt={
                "admitted_residences": int(len(table)),
                "excluded_residences": int(
                    metadata["admission"]["excluded_residences"]
                ),
                "excluded_reasons_exclusive": dict(
                    metadata["admission"]["excluded_reasons_exclusive"]
                ),
                "supported_household_sizes": sizes,
                "lineage_digest": metadata["lineage_digest"],
                "consumer_contract_identity": contract.identity,
                "producer_binding": metadata["producer_binding"],
                "weight_kind": "design",
                "weight_is_calibrated": False,
            },
        )


class SippRecipientRosterKernel(KernelBase):
    """CREATE a caller-supplied recipient population and own its roster size."""

    ref = "build.sipp.equity.recipient@1"
    capabilities = Capabilities(
        Determinism.DETERMINISTIC, structural=StructuralDelta.CREATE
    )

    def implementation_hash(self):
        return source_hash(
            type(self), contract_module, dependencies=self.capabilities.dependencies
        )

    def __init__(self, *, contract=None):
        self._contract = contract

    def run(self, context: KernelContext) -> KernelResult:
        contract = self._contract or contract_module.load_contract()
        node = context.node
        if set(context.params) != {"codec"}:
            raise EquityGraphError("RECIPIENT_NODE_PARAMS")
        (name,) = node.sources
        frame = load_source(context.params["codec"], context.sources[name])
        roster = contract.predictor["roster_column"]
        household = frame.table("household")
        person = frame.table("person")
        if roster in household.columns:
            raise EquityGraphError("RECIPIENT_ROSTER_COLUMN_SUPPLIED")
        membership = frame.schema.membership_column("household")
        counts = person.groupby(membership).size()
        ids = household["household_id"].to_numpy()
        # The Frame already refused orphan or dangling membership; this refuses
        # a supplied size claim that disagrees with the roster it must equal.
        derived = counts.reindex(ids)
        if derived.isna().any() or (derived.to_numpy() <= 0).any():
            raise EquityGraphError("RECIPIENT_INCOMPLETE_MEMBERSHIP")
        sizes = derived.to_numpy().astype(np.int64)
        supplied = household.get(contract.predictor["name"])
        if supplied is not None and not np.array_equal(
            supplied.to_numpy().astype(np.int64), sizes
        ):
            raise EquityGraphError("RECIPIENT_SIZE_CONFLICT")
        household = household.assign(**{roster: sizes})
        rebuilt = Frame(
            {"person": person, "household": household},
            frame.schema,
            {"household": frame.weights_for("household")},
            metadata=dict(frame.metadata) | {"roster_column": roster},
        )
        return KernelResult(
            frame=rebuilt,
            receipt={
                "recipient_households": int(len(household)),
                "recipient_people": int(len(person)),
                "roster_column": roster,
                "roster_sizes": sorted({int(v) for v in sizes}),
            },
        )


class SippSupportValidationKernel(KernelBase):
    """Own the predictor DRAW reads, and only after every size is supported.

    This is the structural predecessor of DRAW: the compiler derives that edge
    from column ownership, so a refusal here aborts the run before any draw.
    """

    ref = "build.sipp.equity.support@1"
    capabilities = Capabilities(Determinism.DETERMINISTIC)

    def implementation_hash(self):
        return source_hash(
            type(self), contract_module, dependencies=self.capabilities.dependencies
        )

    def __init__(self, *, contract=None):
        self._contract = contract

    def run(self, context: KernelContext) -> KernelResult:
        contract = self._contract or contract_module.load_contract()
        node = context.node
        roster = contract.predictor["roster_column"]
        predictor = contract.predictor["name"]
        if (
            len(node.artifact_inputs) != 1
            or node.artifact_inputs[0].name != "donor_support"
            or node.artifact_inputs[0].type != DONOR_SUPPORT_TYPE
            or set(context.artifacts) != {"donor_support"}
        ):
            raise EquityGraphError("SUPPORT_NODE_ARTIFACTS")
        if node.inputs != (Slice("household", (roster,)),) or tuple(
            (o.entity, o.column, o.dtype) for o in node.outputs
        ) != (("household", predictor, "int64"),):
            raise EquityGraphError("SUPPORT_NODE_SHAPE")
        payload = context.artifacts["donor_support"].payload
        if len(payload) > _SUPPORT_MAX_BYTES:
            raise EquityGraphError("SUPPORT_ARTIFACT_TOO_LARGE")
        support = json.loads(payload)
        if (
            support.get("consumer_contract_identity") != contract.identity
            or support.get("target") != contract.target["name"]
            or support.get("target_period") != contract.target["period"]
            or support.get("predictor") != predictor
        ):
            raise EquityGraphError("SUPPORT_ARTIFACT_IDENTITY")
        supported = support.get("supported_household_sizes")
        if not isinstance(supported, list) or not supported:
            raise EquityGraphError("SUPPORT_ARTIFACT_SIZES")
        table = context.tables["household"]
        values = table[roster]
        if values.dtype != np.dtype("int64"):
            raise EquityGraphError("ROSTER_COLUMN_DTYPE")
        sizes = values.to_numpy()
        if (sizes < 1).any():
            raise EquityGraphError("ROSTER_SIZE_NOT_POSITIVE")
        allowed = set(int(v) for v in supported)
        unsupported = sorted({int(v) for v in sizes} - allowed)
        if unsupported:
            # No silent extrapolation, no fill, and no diagnostics-only draw.
            raise EquityGraphError(
                "UNSUPPORTED_RECIPIENT_HOUSEHOLD_SIZE:"
                + ",".join(str(v) for v in unsupported)
            )
        index = pd.Index(table["household_id"].to_numpy(copy=True), name="household_id")
        return KernelResult(
            columns={
                ("household", predictor): pd.Series(
                    sizes.astype(np.int64), index=index, name=predictor
                )
            },
            receipt={
                "checked_households": int(len(sizes)),
                "requested_sizes": sorted({int(v) for v in sizes}),
                "supported_household_sizes": sorted(allowed),
                "unsupported_sizes": [],
                "donor_admission_rule_id": support.get("admission_rule_id"),
            },
        )


# --------------------------------------------------------------------------
# Graph
# --------------------------------------------------------------------------
def build_sipp_equity_graph(
    *, concept: str, period: int, recipient_codec: str = RECIPIENT_CODEC, contract=None
) -> Graph:
    """Build the four-node graph, refusing a wrong concept or period first.

    The refusal happens before any source access or kernel execution, and the
    same contract check runs again inside the donor kernel so a hand-built node
    cannot bypass it.
    """
    contract = contract or contract_module.load_contract()
    contract.require_request(concept=concept, period=period)
    target = contract.target["name"]
    predictor = contract.predictor["name"]
    roster = contract.predictor["roster_column"]
    model = contract.model
    donor = Node(
        DONOR_POPULATION,
        SippDonorCreateKernel.ref,
        sources=(DONOR_POPULATION,),
        structural=StructuralDelta.CREATE,
        outputs=(
            Owned("household", predictor, "int64"),
            Owned("household", target, "float64"),
        ),
        artifact_outputs=(ArtifactOutput("donor_support", DONOR_SUPPORT_TYPE),),
        params={"concept": concept, "period": period},
        description="admitted December-2022 SIPP equity donors, source weights",
    )
    recipient = Node(
        ROSTER_NODE,
        SippRecipientRosterKernel.ref,
        sources=(RECIPIENT_POPULATION,),
        structural=StructuralDelta.CREATE,
        outputs=(Owned("household", roster, "int64"),),
        params={"codec": recipient_codec},
        description="caller recipient households; roster size derived and owned here",
    )
    validate = Node(
        VALIDATE_NODE,
        SippSupportValidationKernel.ref,
        population=RECIPIENT_POPULATION,
        inputs=(Slice("household", (roster,)),),
        outputs=(Owned("household", predictor, "int64"),),
        artifact_inputs=(
            ArtifactInput(
                "donor_support", DONOR_POPULATION, "donor_support", DONOR_SUPPORT_TYPE
            ),
        ),
        description="refuse unsupported recipient sizes, then own the predictor",
    )
    fit = Node(
        FIT_NODE,
        graph_models.QRFTrainKernel.ref,
        population=DONOR_POPULATION,
        inputs=(Slice("household", (predictor, target)),),
        artifact_outputs=(ArtifactOutput("model", graph_models.QRF_MODEL_TYPE),),
        params={
            "predictors": (predictor,),
            "targets": (target,),
            "n_estimators": model["n_estimators"],
            "seed": model["seed"],
            "zero_atol": model["zero_atol"],
            "max_samples_leaf": model["max_samples_leaf"],
        },
    )
    draw = Node(
        DRAW_NODE,
        graph_models.QRFApplyKernel.ref,
        population=RECIPIENT_POPULATION,
        inputs=(Slice("household", (predictor,)),),
        outputs=(Owned("household", target, "float64"),),
        artifact_inputs=(
            ArtifactInput("model", FIT_NODE, "model", graph_models.QRF_MODEL_TYPE),
        ),
        params={"random_stream": "sipp-equity-dec2022/1", "period": period},
    )
    return Graph(
        country="us",
        sources=(
            SourceRef(DONOR_POPULATION, DONOR_CODEC),
            SourceRef(RECIPIENT_POPULATION, recipient_codec),
        ),
        nodes=(donor, recipient, validate, fit, draw),
    )
