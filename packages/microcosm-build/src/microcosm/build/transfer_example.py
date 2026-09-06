"""A synthetic engineering example of one fitted model and two destinations.

Run with ``python -m microcosm.build.transfer_example --output <directory>``.
All inputs are generated, all populations are fictional, and consumption is a
synthetic quantity. No national population or tax-benefit result is certified.
The real QRF, content store, and Adam calibrator execute this example.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from microcosm.build.monetary_targets import (
    MonetaryBasis,
    prepare_monetary_measure,
)
from microcosm.calibrate.kernels import CALIBRATE_ADAM
from microcosm.fit.graph_models import (
    QRF_MODEL_TYPE,
    QRFApplyKernel,
    QRFTrainKernel,
)
from microcosm.frame import EntitySchema, Frame, WeightKind, Weights
from microcosm.graph import (
    ArtifactInput,
    ArtifactOutput,
    Capabilities,
    ContentStore,
    Determinism,
    Graph,
    KernelBase,
    KernelContext,
    KernelRegistry,
    KernelResult,
    Node,
    Numeric,
    Owned,
    RunManifest,
    Slice,
    SourceRef,
    StructuralDelta,
    WeightTransition,
    compile_graph,
    graph_to_json,
    load_source,
    run_graph,
    source_hash,
)

CONVERSION_FACTOR = 12 / 100
"""Recorded floating-point conversion, not an economic exchange rate."""

ANNUAL_BASIS = MonetaryBasis(
    currency="XXX",
    unit="base_currency",
    period="2024",
    temporal_basis="annual_flow",
    sector="synthetic_households",
    perimeter="fictional household consumption",
    valuation="synthetic nominal units",
)

_PREDICTORS = ("household_size", "dwelling")
_MONTHLY = "monthly_consumption_minor"
_ANNUAL = "annual_consumption"
_DESTINATIONS = ("alpha", "beta")
_SCOPE = "synthetic_engineering"


@dataclass(frozen=True)
class DestinationTargets:
    """Synthetic count margins; no consumption outcome enters calibration."""

    household_total: float
    size_total: float

    def __post_init__(self) -> None:
        values = (self.household_total, self.size_total)
        if any(
            isinstance(value, bool) or not np.isfinite(value) or value <= 0
            for value in values
        ):
            raise ValueError("Synthetic target totals must be positive finite numbers.")

    def rows(self) -> tuple[tuple, ...]:
        return (
            ("household_count", "households", None, self.household_total, None),
            ("household_size_total", "household_size", None, self.size_total, None),
        )


@dataclass(frozen=True)
class TransferInputs:
    """Separate donor, recipient, and held-out data boundaries."""

    donor: Frame
    recipients: Mapping[str, Frame]
    references: Mapping[str, Frame]


@dataclass(frozen=True)
class TransferResult:
    manifest: RunManifest
    report: dict


def default_targets() -> dict[str, DestinationTargets]:
    return {
        "alpha": DestinationTargets(1200.0, 3480.0),
        "beta": DestinationTargets(800.0, 1680.0),
    }


def _frame(table: pd.DataFrame, weights: np.ndarray, *, stratum: str) -> Frame:
    ids = table["household_id"].to_numpy(copy=True)
    # A linked person is a structural placeholder, not a claim that household
    # size is one. All estimates and calibration use household weights.
    person = pd.DataFrame({"person_id": ids, "person_household_id": ids})
    return Frame(
        {"person": person, "household": table},
        EntitySchema(group_entities=("household",)),
        {"household": Weights(weights, WeightKind.DESIGN)},
        pd.Series(stratum, index=person.index, name="stratum"),
    )


def _predictors(count: int, start: int) -> pd.DataFrame:
    index = np.arange(count, dtype=np.int64)
    return pd.DataFrame(
        {
            "household_id": index + start,
            "household_size": 1 + index % 4,
            "dwelling": (index // 4) % 2,
            "households": np.ones(count, dtype=np.float64),
        }
    )


def make_synthetic_inputs() -> TransferInputs:
    """Generate donors, predictor-only recipients, and independent references.

    The reference process is specified independently of model predictions and
    solver results: monthly consumption is 8000 + 5000*size + 1000*dwelling.
    Its known design weights tilt household size to each fictional population's
    declared mean. Donors additionally carry +/-400 residuals and unequal
    design weights. These are fixtures for integration behavior, not empirical
    evidence about transferring populations between real countries.
    """
    donor = _predictors(64, 1)
    donor[_MONTHLY] = (
        8000.0
        + 5000.0 * donor["household_size"]
        + 1000.0 * donor["dwelling"]
        + np.where((np.arange(len(donor)) // 8) % 2, -400.0, 400.0)
    )
    donor_frame = _frame(
        donor,
        1.0 + np.arange(len(donor)) % 3,
        stratum="synthetic_donor",
    )
    recipients: dict[str, Frame] = {}
    references: dict[str, Frame] = {}
    for index, (name, targets) in enumerate(default_targets().items(), start=1):
        table = _predictors(24, index * 1000)
        recipients[name] = _frame(
            table,
            np.full(len(table), targets.household_total / len(table)),
            stratum=f"synthetic_recipient_{name}",
        )
        reference = _predictors(96, index * 10000)
        reference[_ANNUAL] = (
            8000.0
            + 5000.0 * reference["household_size"]
            + 1000.0 * reference["dwelling"]
        ) * CONVERSION_FACTOR
        size = reference["household_size"].to_numpy(dtype=np.float64)
        target_mean = targets.size_total / targets.household_total
        design = (targets.household_total / len(size)) * (
            1 + (target_mean - size.mean()) * (size - size.mean()) / size.var()
        )
        references[name] = _frame(
            reference, design, stratum=f"synthetic_holdout_{name}"
        )
    return TransferInputs(donor_frame, recipients, references)


def _digest(payload: object) -> str:
    return hashlib.sha256(
        json.dumps(
            payload, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()
    ).hexdigest()


def _source_frame_path(store: ContentStore, frame: Frame) -> Path:
    """Persist the fixture with an identity covering all tables and weights."""
    frame.revalidate()
    tables = {
        **{name: frame.table(name) for name in frame.entities},
        **{name: frame.link(name) for name in frame.links},
    }
    payload = {
        "scope": _SCOPE,
        "schema": asdict(frame.schema),
        "tables": {
            name: table.to_dict(orient="tight") for name, table in tables.items()
        },
        "dtypes": {
            name: [str(dtype) for dtype in table.dtypes]
            for name, table in tables.items()
        },
        "weights": {
            name: {
                "kind": frame.weights_for(name).kind.value,
                "values": frame.weights_for(name).values.tolist(),
            }
            for name in frame.weighted_entities
        },
        "strata": frame.strata.to_frame().to_dict(orient="tight"),
        "strata_dtype": str(frame.strata.dtype),
        "mass_log": [asdict(record) for record in frame.mass_log],
    }
    return store.put_frame(_digest(payload), frame)


class _SourceKernel(KernelBase):
    ref = "example.transfer.source@1"
    capabilities = Capabilities(
        Determinism.DETERMINISTIC,
        structural=StructuralDelta.CREATE,
        dependencies=("numpy", "pandas"),
    )

    def implementation_hash(self) -> str:
        return source_hash(
            type(self), load_source, dependencies=self.capabilities.dependencies
        )

    def run(self, context: KernelContext) -> KernelResult:
        source = context.node.sources[0]
        frame = load_source("frame-store", context.sources[source])
        return KernelResult(frame=frame, receipt={"scope": _SCOPE, "source": source})


class _AnnualizeKernel(KernelBase):
    ref = "example.transfer.annualize@1"
    capabilities = Capabilities(
        Determinism.DETERMINISTIC,
        numeric=Numeric.PLATFORM_BITWISE,
        dependencies=("numpy", "pandas"),
    )

    def implementation_hash(self) -> str:
        return source_hash(
            type(self),
            prepare_monetary_measure,
            dependencies=self.capabilities.dependencies,
        )

    def run(self, context: KernelContext) -> KernelResult:
        basis = MonetaryBasis(**dict(context.params["basis"]))
        if basis != ANNUAL_BASIS:
            raise ValueError(
                "Destination basis does not match this synthetic annual measure."
            )
        table = context.tables["household"]
        raw = table[_MONTHLY].to_numpy(dtype=np.float64)
        source_convention = "monthly synthetic minor units"
        factor = float(context.params["factor"])
        prepared = prepare_monetary_measure(
            raw,
            record_ids=table["household_id"].to_numpy(),
            basis=basis,
            factor=factor,
            source_identity_sha256=_digest(
                {"source_convention": source_convention, "values": raw.tolist()}
            ),
            bridge_description="Synthetic unit conversion: monthly minor units to annual base units; no economic exchange rate.",
            bridge_source_sha256=_digest(
                {
                    "source_convention": source_convention,
                    "basis": asdict(basis),
                    "factor": factor,
                }
            ),
        )
        return KernelResult(
            columns={
                ("household", _ANNUAL): pd.Series(
                    prepared.values,
                    index=pd.Index(table["household_id"], name="household_id"),
                    dtype="float64",
                )
            },
            receipt={
                "scope": _SCOPE,
                "source_convention": source_convention,
                "factor": factor,
                "prepared": prepared.receipt(),
            },
        )


class _EvaluateKernel(KernelBase):
    ref = "example.transfer.evaluate@1"
    capabilities = Capabilities(
        Determinism.DETERMINISTIC,
        numeric=Numeric.PLATFORM_BITWISE,
        dependencies=("numpy", "pandas"),
    )

    def implementation_hash(self) -> str:
        return source_hash(
            type(self), load_source, dependencies=self.capabilities.dependencies
        )

    def run(self, context: KernelContext) -> KernelResult:
        table = context.tables["household"]
        weights = context.weights["household"].values
        reference = load_source("frame-store", context.sources[context.node.sources[0]])
        heldout = reference.table("household")
        reference_weights = reference.weights_for("household").values
        actual_mean = float(np.average(table[_ANNUAL], weights=weights))
        reference_mean = float(np.average(heldout[_ANNUAL], weights=reference_weights))
        relative_error = abs(actual_mean / reference_mean - 1)
        comparisons = []
        for size in sorted(set(heldout["household_size"])):
            candidate_mask = table["household_size"].to_numpy() == size
            reference_mask = heldout["household_size"].to_numpy() == size
            if not candidate_mask.any():
                comparisons.append({"household_size": int(size), "supported": False})
                continue
            candidate = float(
                np.average(
                    table.loc[candidate_mask, _ANNUAL], weights=weights[candidate_mask]
                )
            )
            expected = float(
                np.average(
                    heldout.loc[reference_mask, _ANNUAL],
                    weights=reference_weights[reference_mask],
                )
            )
            comparisons.append(
                {
                    "household_size": int(size),
                    "supported": True,
                    "candidate_mean": candidate,
                    "reference_mean": expected,
                    "relative_error": abs(candidate / expected - 1),
                }
            )
        residuals = []
        for name, measure, _filter, value, _se in context.params["targets"]:
            observed = float(np.dot(table[measure].to_numpy(dtype=np.float64), weights))
            residuals.append(
                {
                    "name": name,
                    "target": value,
                    "observed": observed,
                    "relative_error": abs(observed / value - 1),
                }
            )
        calibration_tolerance = float(context.params["calibration_tolerance"])
        heldout_tolerance = float(context.params["heldout_tolerance"])
        calibration_passed = all(
            row["relative_error"] <= calibration_tolerance for row in residuals
        )
        heldout_passed = relative_error <= heldout_tolerance and all(
            row["supported"] and row["relative_error"] <= heldout_tolerance
            for row in comparisons
        )
        receipt = {
            "scope": _SCOPE,
            "reference_source": context.node.sources[0],
            "candidate_households": len(table),
            "reference_households": len(heldout),
            "weight_kind": context.weights["household"].kind.value,
            "effective_sample_size": float(
                weights.sum() ** 2 / np.square(weights).sum()
            ),
            "max_weight_share": float(weights.max() / weights.sum()),
            "calibration_residuals": residuals,
            "calibration_passed": calibration_passed,
            "calibration_tolerance": calibration_tolerance,
            "heldout_passed": heldout_passed,
            "heldout_tolerance": heldout_tolerance,
            "heldout_relative_error": relative_error,
            "candidate_consumption_mean": actual_mean,
            "reference_consumption_mean": reference_mean,
            "consumption_by_household_size": comparisons,
            "acceptance_scope": "fixture expectation only; not scientific certification",
        }
        # Evidence belongs in the evaluation receipt, without mutating the
        # population whose independent predictions and weights it evaluates.
        return KernelResult(receipt=receipt)


def _source_node(node_id: str, source: str, *, donor: bool = False) -> Node:
    outputs = (
        Owned("household", "household_size", "int64"),
        Owned("household", "dwelling", "int64"),
        Owned("household", "households", "float64"),
    )
    if donor:
        outputs += (Owned("household", _MONTHLY, "float64"),)
    return Node(
        node_id,
        _SourceKernel.ref,
        outputs=outputs,
        sources=(source,),
        structural=StructuralDelta.CREATE,
    )


def transfer_graph(
    targets: Mapping[str, DestinationTargets], *, basis: MonetaryBasis = ANNUAL_BASIS
) -> Graph:
    """Declare model reuse and ensure held-out data enters evaluation only."""
    if basis != ANNUAL_BASIS:
        raise ValueError(
            "Destination basis must match the declared synthetic annual basis."
        )
    if set(targets) != set(_DESTINATIONS):
        raise ValueError(
            "The synthetic example requires alpha and beta destination targets."
        )
    sources = [SourceRef("donor", "frame-store")]
    nodes = [
        _source_node("donor.source", "donor", donor=True),
        Node(
            "donor.fit",
            QRFTrainKernel.ref,
            inputs=(Slice("household", (*_PREDICTORS, _MONTHLY)),),
            params={
                "predictors": _PREDICTORS,
                "targets": (_MONTHLY,),
                "n_estimators": 24,
                "seed": 314159,
            },
            population="donor.source",
            artifact_outputs=(ArtifactOutput("model", QRF_MODEL_TYPE),),
        ),
    ]
    for index, destination in enumerate(_DESTINATIONS):
        source = f"{destination}.recipient"
        reference = f"{destination}.reference"
        population = f"{destination}.source"
        calibrated = f"{destination}.calibrate"
        rows = targets[destination].rows()
        sources.extend(
            (SourceRef(source, "frame-store"), SourceRef(reference, "frame-store"))
        )
        nodes.extend(
            (
                _source_node(population, source),
                Node(
                    f"{destination}.apply",
                    QRFApplyKernel.ref,
                    inputs=(Slice("household", _PREDICTORS),),
                    outputs=(Owned("household", _MONTHLY, "float64"),),
                    params={
                        "random_stream": (
                            "sha256-u53-v1",
                            "synthetic-transfer",
                            index,
                            271828,
                        ),
                        "period": 2024,
                    },
                    population=population,
                    artifact_inputs=(
                        ArtifactInput("model", "donor.fit", "model", QRF_MODEL_TYPE),
                    ),
                ),
                Node(
                    f"{destination}.annualize",
                    _AnnualizeKernel.ref,
                    inputs=(Slice("household", (_MONTHLY,)),),
                    outputs=(Owned("household", _ANNUAL, "float64"),),
                    params={
                        "basis": tuple(asdict(basis).items()),
                        "factor": CONVERSION_FACTOR,
                    },
                    population=population,
                ),
                Node(
                    calibrated,
                    CALIBRATE_ADAM.ref,
                    inputs=(Slice("household", ("households", "household_size")),),
                    params={
                        "targets": rows,
                        "epochs": 350,
                        "learning_rate": 0.03,
                        "max_weight_ratio": 5.0,
                        "weight_anchor": "design",
                        "mass": "free",
                    },
                    structural=StructuralDelta.REWEIGHT,
                    base=population,
                    weights=WeightTransition("household", "calibrated", mass="free"),
                    mass="free",
                ),
                Node(
                    f"{destination}.evaluate",
                    _EvaluateKernel.ref,
                    inputs=(
                        Slice("household", ("households", "household_size", _ANNUAL)),
                    ),
                    params={
                        "targets": rows,
                        "calibration_tolerance": 0.02,
                        "heldout_tolerance": 0.15,
                    },
                    population=calibrated,
                    sources=(reference,),
                ),
            )
        )
    return Graph("synthetic-transfer", tuple(sources), tuple(nodes))


def _plain(value):
    if isinstance(value, Mapping):
        return {key: _plain(child) for key, child in value.items()}
    if isinstance(value, tuple | list):
        return [_plain(child) for child in value]
    return value


def run_transfer_example(
    output: str | Path,
    *,
    inputs: TransferInputs | None = None,
    targets: Mapping[str, DestinationTargets] | None = None,
    basis: MonetaryBasis = ANNUAL_BASIS,
) -> TransferResult:
    """Execute the real kernels and write only to the explicit output directory.

    Reusing this directory reuses verified content-store results. Changed
    fixture inputs are separately addressed; existing source objects survive.
    No network access, country engines, credentials, or publication are needed.
    """
    graph = transfer_graph(
        default_targets() if targets is None else targets, basis=basis
    )
    inputs = make_synthetic_inputs() if inputs is None else inputs
    if set(inputs.recipients) != set(_DESTINATIONS) or set(inputs.references) != set(
        _DESTINATIONS
    ):
        raise ValueError(
            "The synthetic example requires separate alpha/beta recipients and references."
        )
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    store = ContentStore(output / "store")
    sources = {"donor": _source_frame_path(store, inputs.donor)}
    for destination in _DESTINATIONS:
        sources[f"{destination}.recipient"] = _source_frame_path(
            store, inputs.recipients[destination]
        )
        sources[f"{destination}.reference"] = _source_frame_path(
            store, inputs.references[destination]
        )
    registry = KernelRegistry()
    for kernel in (
        _SourceKernel(),
        QRFTrainKernel(),
        QRFApplyKernel(),
        _AnnualizeKernel(),
        CALIBRATE_ADAM,
        _EvaluateKernel(),
    ):
        registry.register(kernel)
    manifest = run_graph(
        compile_graph(graph), sources=sources, store=store, kernels=registry
    )
    fit = manifest.node("donor.fit")
    artifact_key = fit.opaque_artifacts["model"]
    report = {
        "schema_version": 1,
        "scope": _SCOPE,
        "description": "Fictional household consumption transfer; no national or tax-benefit claims.",
        "model": {
            "node_key": fit.key,
            "artifact_key": artifact_key,
            "training_population_key": manifest.node("donor.source").key,
        },
        "destinations": {},
    }
    for destination in _DESTINATIONS:
        evaluation = _plain(manifest.node(f"{destination}.evaluate").receipt)
        initial = inputs.recipients[destination].weights_for("household").values
        final = (
            manifest.population(f"{destination}.calibrate")
            .weights_for("household")
            .values
        )
        report["destinations"][destination] = {
            **evaluation,
            "model_artifact_key": artifact_key,
            "application_key": manifest.node(f"{destination}.apply").key,
            "transformation_key": manifest.node(f"{destination}.annualize").key,
            "calibration_key": manifest.node(f"{destination}.calibrate").key,
            "evaluation_key": manifest.node(f"{destination}.evaluate").key,
            "mass_policy": "free",
            "mass_before": float(initial.sum()),
            "mass_after": float(final.sum()),
        }
    manifest.save(output / "run_manifest.json")
    (output / "graph.json").write_text(graph_to_json(graph), encoding="utf-8")
    (output / "report.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    return TransferResult(manifest, report)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Directory for synthetic inputs, cache, and reports.",
    )
    args = parser.parse_args(argv)
    run_transfer_example(args.output)
    print(args.output / "report.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
