#!/usr/bin/env python3
"""Regenerate deterministic H1 fixtures from the real wrapped kernels."""

from __future__ import annotations

import json
import os
import sys
from collections.abc import Mapping, Sequence
from importlib import metadata as importlib_metadata
from pathlib import Path

import numpy as np
import pandas as pd

from microcosm.calibrate import Target, TargetSet, calibrate
from microcosm.calibrate.kernels import CALIBRATE_ADAM
from microcosm.fit import fit as fit_qrf
from microcosm.fit.kernels import QRF_PARAM_KERNEL
from microcosm.frame import (
    EntitySchema,
    ExportContract,
    Frame,
    VariableMetadata,
    WeightKind,
    Weights,
)
from microcosm.frame.kernels import SimulateRulesKernel
from microcosm.graph import (
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
    graph_to_json,
)

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = (
    ROOT / "packages" / "microcosm-graph" / "tests" / "fixtures" / "parity" / "kernels"
)

FIT_SEED = 947
TARGET_PARAMS = (
    ("income", "income", None, 390.0, 7.25),
    ("eligible_income", "income", "eligible", 240.0, 3),
)
CALIBRATE_PARAMS = {
    "targets": TARGET_PARAMS,
    "max_weight_ratio": 2.0,
    "epochs": 24,
    "learning_rate": 0.03,
    "mass": "conserve",
}
SIMULATE_VARIABLES = ("net_earnings", "housing_allowance")
PARITY_SCHEMA = EntitySchema(group_entities=("household",))


class ParityCsvSource(KernelBase):
    """CREATE kernel for one ``inputs.csv`` parity case."""

    ref = "parity.source.csv@1"
    capabilities = Capabilities(
        determinism=Determinism.DETERMINISTIC,
        numeric=Numeric.BITWISE,
        seed_source=SeedSource.NONE,
        structural=StructuralDelta.CREATE,
    )

    def run(self, context: KernelContext) -> KernelResult:
        case = context.params.get("case")
        if case not in {"fit.qrf", "calibrate", "simulate"}:
            raise ValueError(f"Unknown parity fixture case {case!r}.")
        table = pd.read_csv(
            context.sources["fixture"] / "inputs.csv", float_precision="round_trip"
        )
        frame = _frame_for_case(str(case), table)
        return KernelResult(frame=frame, receipt={"case": case, "rows": len(table)})


class ParityRulesEngine:
    """Pure-Python rules engine shared by generation and the H1 executor."""

    def variable_metadata(self, name: str) -> VariableMetadata:
        metadata = {
            "net_earnings": VariableMetadata(
                name="net_earnings", entity="person", dtype="float", period="year"
            ),
            "housing_allowance": VariableMetadata(
                name="housing_allowance",
                entity="household",
                dtype="float",
                period="year",
            ),
        }
        try:
            return metadata[name]
        except KeyError as error:
            raise ValueError(f"Unknown parity variable {name!r}.") from error

    def variables(self) -> Sequence[str]:
        return ("earnings", "housing_cost")

    def entity_schema(self) -> EntitySchema:
        return PARITY_SCHEMA

    def materialize(
        self,
        bundle: Frame,
        variables: Sequence[str],
        period: int | str,
    ) -> Mapping[str, np.ndarray]:
        year = int(str(period)[:4])
        earnings = bundle.table("person")["earnings"].to_numpy(dtype=np.float64)
        housing_cost = bundle.table("household")["housing_cost"].to_numpy(
            dtype=np.float64
        )
        available = {
            "net_earnings": earnings * np.float64(0.8) + np.float64(year - 2025),
            "housing_allowance": np.maximum(
                np.float64(12_000.0) - housing_cost, np.float64(0.0)
            ),
        }
        return {variable: available[variable] for variable in variables}

    def export_contract(self) -> ExportContract:
        return ExportContract.empty()

    def write_dataset(
        self,
        bundle: Frame,
        path: str | Path,
        period: int | str,
    ) -> None:
        del bundle, path, period


def simulate_kernel() -> SimulateRulesKernel:
    """Return the canonical parity binding used by pins and tests."""

    return SimulateRulesKernel("parity-stub", ParityRulesEngine())


def parity_registry() -> KernelRegistry:
    """Registry required to execute every generated parity graph."""

    registry = KernelRegistry()
    registry.register(ParityCsvSource())
    registry.register(QRF_PARAM_KERNEL)
    registry.register(CALIBRATE_ADAM)
    registry.register(simulate_kernel())
    return registry


def _frame_for_case(case: str, table: pd.DataFrame) -> Frame:
    if case == "fit.qrf":
        person = table.drop(columns="weight").astype(
            {
                "person_id": "int64",
                "person_household_id": "int64",
                "age": "float64",
                "score": "float64",
                "observed_y": "float64",
                "is_donor": "bool",
                "is_recipient": "bool",
            }
        )
        household = pd.DataFrame(
            {"household_id": person["person_household_id"].to_numpy(dtype=np.int64)}
        )
        return Frame(
            {"person": person, "household": household},
            PARITY_SCHEMA,
            {
                "person": Weights(
                    table["weight"].to_numpy(dtype=np.float64), WeightKind.DESIGN
                )
            },
        )
    if case == "calibrate":
        household = table.drop(columns="weight").astype(
            {"household_id": "int64", "income": "float64", "eligible": "float64"}
        )
        person = pd.DataFrame(
            {
                "person_id": household["household_id"].to_numpy(dtype=np.int64),
                "person_household_id": household["household_id"].to_numpy(
                    dtype=np.int64
                ),
            }
        )
        return Frame(
            {"person": person, "household": household},
            PARITY_SCHEMA,
            {
                "household": Weights(
                    table["weight"].to_numpy(dtype=np.float64),
                    WeightKind.IMPORTANCE,
                )
            },
        )
    if case == "simulate":
        person = table[["person_id", "person_household_id", "earnings"]].astype(
            {
                "person_id": "int64",
                "person_household_id": "int64",
                "earnings": "float64",
            }
        )
        household = table[["household_id", "housing_cost"]].astype(
            {"household_id": "int64", "housing_cost": "float64"}
        )
        return Frame(
            {"person": person, "household": household},
            PARITY_SCHEMA,
            {
                "household": Weights(
                    table["weight"].to_numpy(dtype=np.float64), WeightKind.DESIGN
                )
            },
        )
    raise ValueError(f"Unknown parity fixture case {case!r}.")


def _source_node(case: str, outputs: tuple[Owned, ...]) -> Node:
    return Node(
        "source",
        ParityCsvSource.ref,
        outputs=outputs,
        params={"case": case},
        structural=StructuralDelta.CREATE,
        sources=("fixture",),
    )


def _fit_case() -> tuple[Graph, pd.DataFrame, pd.DataFrame, object, int]:
    count = 52
    index = np.arange(count, dtype=np.int64)
    donors = index < 40
    table = pd.DataFrame(
        {
            "person_id": index + 1,
            "person_household_id": index + 1,
            "age": 20.0 + (index % 31),
            "score": (index % 13) * 0.5 + index * 0.03125,
            "observed_y": np.where(donors, ((index % 17) + 1) * 0.25, np.nan),
            "is_donor": donors,
            "is_recipient": ~donors,
            "weight": 1.0 + (index % 5) * 0.125,
        }
    )
    source = _source_node(
        "fit.qrf",
        (
            Owned("person", "age", "float64"),
            Owned("person", "score", "float64"),
            Owned("person", "observed_y", "float64"),
            Owned("person", "is_donor", "bool"),
            Owned("person", "is_recipient", "bool"),
        ),
    )
    node = Node(
        "fit_qrf",
        QRF_PARAM_KERNEL.ref,
        inputs=(
            Slice(
                "person",
                ("age", "score", "observed_y", "is_donor", "is_recipient"),
            ),
        ),
        outputs=(Owned("person", "y", "float64", rows="is_recipient"),),
        params={
            "donor_target": "observed_y",
            "n_estimators": 9,
            "min_samples_leaf": 1,
            "zero_atol": 1e-6,
            "max_samples_leaf": 8,
            "seed": FIT_SEED,
        },
        population="source",
    )
    donor = table.loc[donors, ["age", "score", "observed_y"]].rename(
        columns={"observed_y": "y"}
    )
    recipient = table.loc[~donors, ["age", "score"]]
    fitted = fit_qrf(
        donor,
        ["age", "score"],
        ["y"],
        weights=table.loc[donors, "weight"].to_numpy(dtype=np.float64),
        n_estimators=9,
        zero_atol=1e-6,
        max_samples_leaf=8,
        seed=FIT_SEED,
    )
    direct = pd.DataFrame(
        {"person.y": fitted.predict(recipient)["y"].to_numpy(dtype=np.float64)}
    )
    return (
        Graph("parity", (SourceRef("fixture", "csv-tables"),), (source, node)),
        table,
        direct,
        QRF_PARAM_KERNEL,
        FIT_SEED,
    )


def _calibrate_case() -> tuple[Graph, pd.DataFrame, pd.DataFrame, object, int]:
    table = pd.DataFrame(
        {
            "household_id": np.arange(6, dtype=np.int64),
            "income": np.asarray([10.0, 25.0, 40.0, 70.0, 90.0, 120.0]),
            "eligible": np.asarray([1.0, 0.0, 1.0, 1.0, 0.0, 1.0]),
            "weight": np.asarray([1.0, 1.5, 0.75, 2.0, 1.25, 0.5]),
        }
    )
    source = _source_node(
        "calibrate",
        (
            Owned("household", "income", "float64"),
            Owned("household", "eligible", "float64"),
        ),
    )
    node = Node(
        "calibrate",
        CALIBRATE_ADAM.ref,
        inputs=(Slice("household", ("income", "eligible")),),
        params=CALIBRATE_PARAMS,
        population="source",
        weights=WeightTransition("household", "calibrated", mass="conserve"),
    )
    frame = _frame_for_case("calibrate", table)
    targets = TargetSet(
        Target(
            name=name,
            entity="household",
            measure=measure,
            filter=filter_column,
            value=value,
        )
        for name, measure, filter_column, value, _se in TARGET_PARAMS
    )
    result = calibrate(
        frame,
        targets,
        weight_entity="household",
        method="adam",
        max_weight_ratio=CALIBRATE_PARAMS["max_weight_ratio"],
        epochs=CALIBRATE_PARAMS["epochs"],
        learning_rate=CALIBRATE_PARAMS["learning_rate"],
        mass=CALIBRATE_PARAMS["mass"],
        seed=0,
    )
    direct = pd.DataFrame(
        {"household.weights": result.weights.astype(np.float64, copy=False)}
    )
    return (
        Graph("parity", (SourceRef("fixture", "csv-tables"),), (source, node)),
        table,
        direct,
        CALIBRATE_ADAM,
        0,
    )


def _simulate_case() -> tuple[Graph, pd.DataFrame, pd.DataFrame, object, None]:
    ids = np.arange(1, 5, dtype=np.int64)
    table = pd.DataFrame(
        {
            "person_id": ids,
            "person_household_id": ids,
            "household_id": ids,
            "earnings": np.asarray([50_000.0, 25_000.0, 80_000.0, 0.0]),
            "housing_cost": np.asarray([9_600.0, 15_000.0, 8_000.0, 12_000.0]),
            "weight": np.asarray([125.0, 275.0, 80.0, 150.0]),
        }
    )
    source = _source_node(
        "simulate",
        (
            Owned("person", "earnings", "float64"),
            Owned("household", "housing_cost", "float64"),
        ),
    )
    kernel = simulate_kernel()
    node = Node(
        "simulate",
        kernel.ref,
        inputs=(
            Slice("person", ("earnings",)),
            Slice("household", ("housing_cost",)),
        ),
        outputs=(
            Owned("person", "net_earnings", "float64"),
            Owned("household", "housing_allowance", "float64"),
        ),
        params={
            "engine_ref": "parity-stub",
            "variables": SIMULATE_VARIABLES,
            "period": 2025,
        },
        population="source",
    )
    frame = _frame_for_case("simulate", table)
    materialized = ParityRulesEngine().materialize(
        frame, SIMULATE_VARIABLES, period=2025
    )
    direct = pd.DataFrame(
        {
            "person.net_earnings": materialized["net_earnings"],
            "household.housing_allowance": materialized["housing_allowance"],
        }
    )
    return (
        Graph("parity", (SourceRef("fixture", "csv-tables"),), (source, node)),
        table,
        direct,
        kernel,
        None,
    )


def _pins(node_id: str, kernel: object, seed: int | None) -> dict[str, object]:
    capabilities = kernel.capabilities  # type: ignore[attr-defined]
    dependencies = {
        name: importlib_metadata.version(name)
        for name in sorted(capabilities.dependencies)
    }
    return {
        "node": node_id,
        "seed": seed,
        "kernel": kernel.ref,  # type: ignore[attr-defined]
        "implementation_hash": kernel.implementation_hash(),  # type: ignore[attr-defined]
        "dependencies": dependencies,
    }


def _write_case(
    name: str,
    graph: Graph,
    inputs: pd.DataFrame,
    direct: pd.DataFrame,
    kernel: object,
    seed: int | None,
) -> None:
    destination = FIXTURES / name
    destination.mkdir(parents=True, exist_ok=True)
    (destination / "graph.json").write_text(graph_to_json(graph), encoding="utf-8")
    inputs.to_csv(destination / "inputs.csv", index=False, lineterminator="\n")
    direct.to_csv(destination / "direct.csv", index=False, lineterminator="\n")
    (destination / "pins.json").write_text(
        json.dumps(
            _pins(graph.nodes[-1].id, kernel, seed),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n",
        encoding="utf-8",
    )


def generate() -> None:
    """Regenerate all H1 kernel fixtures deterministically."""

    os.environ["POPULACE_FIT_N_JOBS"] = "1"
    os.environ["POPULACE_FIT_PREDICT_WORKERS"] = "1"
    FIXTURES.mkdir(parents=True, exist_ok=True)
    (FIXTURES / "PRODUCED_BY.txt").write_text(
        "Lane D: tools/graph_parity_fixtures.py\n", encoding="utf-8"
    )
    (FIXTURES / ".gitignore").write_text("_store/\n", encoding="utf-8")
    for name, builder in (
        ("fit.qrf", _fit_case),
        ("calibrate", _calibrate_case),
        ("simulate", _simulate_case),
    ):
        graph, inputs, direct, kernel, seed = builder()
        _write_case(name, graph, inputs, direct, kernel, seed)


def main() -> int:
    generate()
    print(f"wrote deterministic H1 fixtures under {FIXTURES.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    # source_hash includes module names. Delegate to the canonical import so the
    # stub pin is identical when this file is executed and when a test imports it.
    sys.path.insert(0, str(ROOT))
    from tools.graph_parity_fixtures import main as canonical_main

    raise SystemExit(canonical_main())
