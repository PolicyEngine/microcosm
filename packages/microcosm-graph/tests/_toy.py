"""The toy country: fixtures, kernels, and graphs for the acceptance suite.

A synthetic two-entity population — 200 households, 502 persons, typed
household design weights, two strata — plus the pure numpy/pandas kernels the
charter properties in ``docs/graph-acceptance.md`` are asserted against. No
country package, no restricted data, no network.

Everything here is written against the frozen interfaces
(``microcosm.graph.decl`` and ``microcosm.graph.kernel``) and the public
``microcosm.frame`` API only. Nothing here imports or inspects the executor,
the store, or any private module of the shard: the acceptance suite is
black-box by construction (charter process rule 3).

Three contracts the frozen interfaces leave open are pinned here, because a
kernel needs them to be executable at all. They are listed in the lane report
under "API assumptions" so the runtime lane can meet them:

1. Only a ``CREATE`` kernel returns ``KernelResult.frame``; it is the one
   kernel that builds a population rather than reading one. A ``FILTER``
   kernel returns the surviving-person mask as ``KernelResult.keep`` and a
   ``REWEIGHT`` kernel returns ``KernelResult.weights``; the executor
   applies both to the base version. No other kernel ever holds a
   population (charter B2). Gate and release kernels say so through
   ``Capabilities.role``. (Both were adopted into the frozen interface on
   2026-09-01; see the charter's "Interface freeze" amendments.)
2. ``KernelContext.tables`` carries an id-only view of every entity the node
   *owns*, on top of the declared input slices, so a kernel can index the
   cells it is responsible for. The person view carries its membership
   columns with its id column.
3. A mass record is ``{"policy", "before", "after", "stratum_before",
   "stratum_after"}`` at person level, matching ``Frame.stratum_mass``.

The module is loaded by file path from every acceptance file, because the
workspace runs pytest with ``--import-mode=importlib`` and a test module
therefore cannot ``import _toy`` by name. That is the pattern
``packages/microcosm-build/tests/test_uk_release_assembler.py`` already uses
for its sibling fixture module.
"""

from __future__ import annotations

import hashlib
import json
import shutil
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from microcosm.frame import EntitySchema, Frame, WeightKind, Weights
from microcosm.graph import (
    Capabilities,
    CompiledGraph,
    ContentStore,
    Determinism,
    Graph,
    KernelBase,
    KernelContext,
    KernelRegistry,
    KernelResult,
    KernelRole,
    Node,
    Numeric,
    Owned,
    Ownership,
    SeedSource,
    Slice,
    SourceRef,
    StructuralDelta,
    Tolerance,
    WeightTransition,
    compile_graph,
    run_graph,
    source_hash,
)

__all__ = [
    "CREATE",
    "FIXTURES",
    "GATE_OUTCOMES",
    "POOL",
    "PUBLISH_DECISION",
    "SOURCE",
    "STRATA",
    "TOY_TOLERANCE",
    "ToyKernel",
    "ToyRun",
    "absent_node",
    "artifact_bytes",
    "calibrated_node",
    "calls_by_ref",
    "chained_graph",
    "copy_source",
    "count_node",
    "derive",
    "descendants",
    "draw",
    "drop_nodes",
    "entrant_expand_node",
    "entrant_person_node",
    "full_graph",
    "gate_node",
    "graph_source_files",
    "id_column",
    "impute",
    "patch_node",
    "read_toy_frame",
    "release_node",
    "replace_node",
    "run_toy",
    "select_node",
    "simulate_node",
    "small_graph",
    "surviving_design_anchor",
    "total_calls",
    "toy_registry",
    "toy_schema",
    "toy_sources",
]

#: The packaged toy-country tables.
FIXTURES = Path(__file__).parent / "fixtures" / "toy_country"

#: The two strata the fixture assigns, whole households at a time.
STRATA = ("rural", "urban")

#: The five gate outcomes charter F4 freezes.
GATE_OUTCOMES = frozenset(
    {"pass", "fail", "evidence_absent", "not_applicable", "unreached"}
)

#: The one source every toy graph reads, through the ``csv-tables`` codec.
SOURCE = SourceRef("survey", "csv-tables", description="the toy country's tables")

#: Cross-machine numeric movement declared by the C5 toy producer.
TOY_TOLERANCE = Tolerance(rtol=1e-6)


def id_column(entity: str) -> str:
    """The frame convention: ``person_id`` for persons, ``{entity}_id`` elsewhere."""
    return f"{entity}_id"


def toy_schema() -> EntitySchema:
    """Persons in households, and one release row every person belongs to."""
    return EntitySchema(group_entities=("household", "release"))


def copy_source(
    destination: Path, *, edit: Mapping[str, tuple[str, str]] | None = None
) -> Path:
    """Copy the fixture tables to ``destination``, optionally editing bytes.

    Args:
        destination: Directory to write. Created by the copy.
        edit: File name to an ``(old, new)`` pair replaced once in that file.
            Charter A6 uses it to change a source's bytes under one name.

    Returns:
        ``destination``.
    """
    shutil.copytree(FIXTURES, destination)
    for name, (old, new) in (edit or {}).items():
        path = destination / name
        path.write_text(path.read_text().replace(old, new, 1))
    return destination


def toy_sources(tmp_path: Path, *, name: str = "source") -> dict[str, Path]:
    """A writable copy of the fixture tables, ready to hand to ``run_graph``."""
    return {"survey": copy_source(tmp_path / name)}


def read_toy_frame(source: Path) -> Frame:
    """Build the toy :class:`Frame` from a directory of fixture tables."""
    schema = json.loads((source / "schema.json").read_text())
    dtypes = schema["dtypes"]
    tables = {}
    for entity, file_name in schema["tables"].items():
        table = pd.read_csv(source / file_name)
        tables[entity] = table.astype(
            {column: dtypes[column] for column in table.columns}
        )
    strata = tables["person"].pop(schema["strata_column"])
    spec = schema["weights"]
    weight_table = pd.read_csv(source / schema["weights_table"])
    return Frame(
        tables,
        toy_schema(),
        {
            spec["entity"]: Weights(
                values=weight_table[spec["column"]].to_numpy(dtype="float64"),
                kind=WeightKind(spec["kind"]),
            )
        },
        strata,
    )


# ----------------------------------------------------------------------
# Kernels
# ----------------------------------------------------------------------


class ToyKernel(KernelBase):
    """A registered toy computation that counts its own executions.

    ``variant`` stands in for a source edit: it enters
    :meth:`implementation_hash` without touching any other kernel's hash, so
    charter A5 can move exactly one kernel's code identity.
    """

    def __init__(
        self, ref: str, capabilities: Capabilities, *, variant: str = "base"
    ) -> None:
        self.ref = ref
        self.capabilities = capabilities
        self.variant = variant
        self.calls = 0

    def implementation_hash(self) -> str:
        base = source_hash(type(self), dependencies=self.capabilities.dependencies)
        return hashlib.sha256(f"{base}/{self.variant}".encode()).hexdigest()

    def run(self, context: KernelContext) -> KernelResult:
        self.calls += 1
        return self.compute(context)

    def compute(self, context: KernelContext) -> KernelResult:
        raise NotImplementedError


def _owned_ids(context: KernelContext, entity: str) -> pd.Index:
    """Entity ids of the rows the executor projected for ``entity``."""
    return pd.Index(context.tables[entity][id_column(entity)])


def _household_positions(context: KernelContext) -> np.ndarray:
    """Each person's position in the household table."""
    households = context.tables["household"][id_column("household")]
    return pd.Index(households).get_indexer(
        context.tables["person"]["person_household_id"]
    )


def _by_stratum(strata: np.ndarray, values: np.ndarray) -> dict[str, float]:
    totals = pd.Series(values).groupby(strata).sum()
    return {str(label): float(total) for label, total in totals.items()}


def _mass_record(
    context: KernelContext,
    before: np.ndarray,
    after: np.ndarray,
    policy: str,
    *,
    keep: np.ndarray | None = None,
) -> dict[str, object]:
    """Person-level mass before and after, in total and per stratum."""
    positions = _household_positions(context)
    strata = context.strata.to_numpy()
    person_before = before[positions]
    person_after = after[positions]
    if keep is not None:
        person_after = np.where(keep, person_after, 0.0)
    return {
        "policy": policy,
        "before": float(person_before.sum()),
        "after": float(person_after.sum()),
        "stratum_before": _by_stratum(strata, person_before),
        "stratum_after": _by_stratum(strata, person_after),
    }


class SourceCsv(ToyKernel):
    """CREATE: the ``csv-tables`` codec turned into a population version."""

    def compute(self, context: KernelContext) -> KernelResult:
        frame = read_toy_frame(context.sources["survey"])
        return KernelResult(
            frame=frame,
            receipt={"persons": frame.n("person"), "households": frame.n("household")},
        )


class DeriveAdd(ToyKernel):
    """Deterministic: the sum of the input columns, times ``scale``."""

    def compute(self, context: KernelContext) -> KernelResult:
        entity = str(context.params["entity"])
        table = context.tables[entity]
        columns = [str(name) for name in context.params["columns"]]
        total = sum(table[column].to_numpy(dtype="float64") for column in columns)
        return KernelResult(
            columns={
                (entity, str(context.params["target"])): pd.Series(
                    total * float(context.params["scale"]),
                    index=_owned_ids(context, entity),
                    dtype="float64",
                )
            },
            receipt={"rows": int(len(table)), "columns_seen": tuple(table.columns)},
        )


class DrawUniform(ToyKernel):
    """Seeded: one uniform draw per owned row, from ``context.rng`` alone."""

    def compute(self, context: KernelContext) -> KernelResult:
        entity = str(context.params["entity"])
        ids = _owned_ids(context, entity)
        return KernelResult(
            columns={
                (entity, str(context.params["target"])): pd.Series(
                    context.rng.uniform(size=len(ids)), index=ids, dtype="float64"
                )
            },
            receipt={"draws": int(len(ids))},
        )


class ImputeChain(ToyKernel):
    """Seeded: the predictor mean plus RNG noise. Chains through its inputs."""

    def compute(self, context: KernelContext) -> KernelResult:
        entity = str(context.params["entity"])
        table = context.tables[entity]
        columns = [str(name) for name in context.params["predictors"]]
        predictors = np.column_stack(
            [table[column].to_numpy(dtype="float64") for column in columns]
        )
        ids = _owned_ids(context, entity)
        noise = context.rng.normal(size=len(ids)) * float(context.params["noise"])
        return KernelResult(
            columns={
                (entity, str(context.params["target"])): pd.Series(
                    predictors.mean(axis=1) + noise, index=ids, dtype="float64"
                )
            },
            receipt={
                "predictors": tuple(columns),
                "columns_seen": tuple(table.columns),
                "entities_seen": tuple(sorted(context.tables)),
            },
        )


class SimulateStub(ToyKernel):
    """A stub rules engine: a flat rate on one column, deterministically."""

    def compute(self, context: KernelContext) -> KernelResult:
        entity = str(context.params["entity"])
        base = context.tables[entity][str(context.params["column"])]
        return KernelResult(
            columns={
                (entity, str(context.params["target"])): pd.Series(
                    base.to_numpy(dtype="float64") * float(context.params["rate"]),
                    index=_owned_ids(context, entity),
                    dtype="float64",
                )
            },
            receipt={"engine": "toy-stub", "rate": context.params["rate"]},
        )


class PatchColumn(ToyKernel):
    """Re-own one carried column at masked positions, in a new version."""

    def compute(self, context: KernelContext) -> KernelResult:
        entity = str(context.params["entity"])
        ids = _owned_ids(context, entity)
        dtype = str(context.params["dtype"])
        if dtype == "boolean":
            values = pd.array(
                np.full(len(ids), bool(context.params["value"])), dtype=dtype
            )
        else:
            values = np.full(len(ids), float(context.params["value"]), dtype="float64")
        return KernelResult(
            columns={
                (entity, str(context.params["target"])): pd.Series(
                    values, index=ids, dtype=dtype
                )
            },
            receipt={"patched": int(len(ids))},
        )


class AbsentColumn(ToyKernel):
    """Own a cell and assert its absence: the column is null throughout."""

    def compute(self, context: KernelContext) -> KernelResult:
        entity = str(context.params["entity"])
        ids = _owned_ids(context, entity)
        dtype = str(context.params["dtype"])
        return KernelResult(
            columns={
                (entity, str(context.params["target"])): pd.Series(
                    pd.array([None] * len(ids), dtype=dtype), index=ids, dtype=dtype
                )
            },
            receipt={"absent": int(len(ids))},
        )


class SelectRows(ToyKernel):
    """FILTER: the surviving-person mask, for the executor to apply."""

    def compute(self, context: KernelContext) -> KernelResult:
        mask = (
            context.tables["person"][str(context.params["mask"])]
            .to_numpy(dtype="bool", na_value=False)
            .astype(bool)
        )
        weights = context.weights["household"].values
        return KernelResult(
            keep=pd.Series(mask, index=_owned_ids(context, "person"), dtype="bool"),
            receipt={
                "kept": int(mask.sum()),
                "mass": _mass_record(
                    context, weights, weights, str(context.params["policy"]), keep=mask
                ),
            },
        )


class ReweightScale(ToyKernel):
    """A ``design -> importance`` transition: scale weights by ``factor``."""

    def compute(self, context: KernelContext) -> KernelResult:
        entity = str(context.params["entity"])
        before = context.weights[entity].values
        after = before * float(context.params["factor"])
        return KernelResult(
            weights=Weights(values=after, kind=WeightKind(context.params["to_kind"])),
            receipt={"mass": _mass_record(context, before, after, "free")},
        )


class ExpandEntrants(ToyKernel):
    """EXPAND: add one copied person and one materialized entrant household."""

    def compute(self, context: KernelContext) -> KernelResult:
        person = context.tables["person"]
        person_ids = pd.Index(person["person_id"], name="person_id")
        person_source_id = int(person_ids[0])
        person_copy_id = int(person_ids.max()) + 1

        household = context.tables["household"]
        household_ids = pd.Index(household["household_id"], name="household_id")
        household_entrant_id = int(household_ids.max()) + 1
        household_target_ids = household_ids.append(
            pd.Index([household_entrant_id], dtype="int64", name="household_id")
        )
        person_target_ids = person_ids.append(
            pd.Index([person_copy_id], dtype="int64", name="person_id")
        )

        household_size = pd.concat(
            [
                household["household_size"].reset_index(drop=True),
                pd.Series([1], dtype="int64"),
            ],
            ignore_index=True,
        )
        materialized_size = pd.Series(
            household_size.array, index=household_target_ids, dtype="int64"
        )
        if context.params.get("missing_entrant_column") == "household_size":
            materialized_size = materialized_size.drop(index=household_entrant_id)
        memberships = pd.concat(
            [
                person["person_household_id"].reset_index(drop=True),
                pd.Series([household_entrant_id], dtype="int64"),
            ],
            ignore_index=True,
        )
        materialized_memberships = pd.Series(
            memberships.array, index=person_target_ids, dtype="int64"
        )

        empty_releases = pd.Series(
            [],
            index=pd.Index([], dtype="int64", name="release_id"),
            dtype="int64",
        )
        household_weights = context.weights["household"]
        expanded_weights = np.append(
            household_weights.values, float(context.params["entrant_weight"])
        )
        return KernelResult(
            expand={
                "person": pd.Series(
                    [person_source_id],
                    index=pd.Index([person_copy_id], dtype="int64", name="person_id"),
                    dtype="int64",
                ),
                "household": pd.Series(
                    pd.array([pd.NA], dtype="Int64"),
                    index=pd.Index(
                        [household_entrant_id],
                        dtype="int64",
                        name="household_id",
                    ),
                ),
                "release": empty_releases,
            },
            columns={
                ("household", "household_size"): materialized_size,
                ("person", "person_household_id"): materialized_memberships,
            },
            weights=Weights(expanded_weights, kind=household_weights.kind),
        )


class ExpandEntrantPerson(ToyKernel):
    """EXPAND: admit one entrant person into an incumbent household.

    The entrant copies nothing: every person column is materialized from a
    template row, its memberships name incumbent groups, and its stratum
    arrives through ``KernelResult.strata`` (amendment 14). ``strata_mode``
    exercises the refusals: ``missing`` omits the field, ``unknown_id``
    labels an id the node never adds, ``labels_incumbent`` labels an
    incumbent person as well.
    """

    def compute(self, context: KernelContext) -> KernelResult:
        person = context.tables["person"]
        person_ids = pd.Index(person["person_id"], name="person_id")
        template = person.iloc[0]
        entrant_id = int(person_ids.max()) + 1
        target_ids = person_ids.append(
            pd.Index([entrant_id], dtype="int64", name="person_id")
        )

        def overlay(column: str, dtype: str, value: object) -> pd.Series:
            values = pd.concat(
                [person[column].reset_index(drop=True), pd.Series([value])],
                ignore_index=True,
            )
            return pd.Series(pd.array(values, dtype=dtype), index=target_ids)

        columns = {
            ("person", "age"): overlay("age", "int64", 30),
            ("person", "income"): overlay("income", "float64", 12_500.0),
            ("person", "is_adult"): overlay("is_adult", "boolean", True),
            ("person", "receives_x"): overlay("receives_x", "boolean", False),
            ("person", "person_household_id"): overlay(
                "person_household_id", "int64", int(template["person_household_id"])
            ),
            ("person", "person_release_id"): overlay(
                "person_release_id", "int64", int(template["person_release_id"])
            ),
        }
        mode = str(context.params.get("strata_mode", "ok"))
        labelled = {
            "ok": [entrant_id],
            "unknown_id": [entrant_id + 1],
            "labels_incumbent": [int(person_ids[0]), entrant_id],
        }
        strata = (
            None
            if mode == "missing"
            else pd.Series(
                ["urban"] * len(labelled[mode]),
                index=pd.Index(labelled[mode], dtype="int64", name="person_id"),
                dtype=object,
                name="stratum",
            )
        )
        empty = {
            entity: pd.Series(
                [],
                index=pd.Index([], dtype="int64", name=id_column(entity)),
                dtype="int64",
            )
            for entity in ("household", "release")
        }
        household_weights = context.weights["household"]
        return KernelResult(
            expand={
                "person": pd.Series(
                    pd.array([pd.NA], dtype="Int64"),
                    index=pd.Index([entrant_id], dtype="int64", name="person_id"),
                ),
                **empty,
            },
            columns=columns,
            weights=Weights(
                household_weights.values.copy(), kind=household_weights.kind
            ),
            strata=strata,
        )


class ClaimMaterializedExpand(ToyKernel):
    """Claim kernel-supplied EXPAND columns through the ownership surface."""

    def compute(self, context: KernelContext) -> KernelResult:
        columns: dict[tuple[str, str], pd.Series] = {}
        for item in context.params["claim_cells"]:
            entity, column, dtype = (str(value) for value in item)
            table = context.tables[entity]
            columns[(entity, column)] = pd.Series(
                table[column].array.copy(),
                index=pd.Index(table[id_column(entity)], name=id_column(entity)),
                dtype=dtype,
            )
        return KernelResult(columns=columns)


class CalibrateToy(ToyKernel):
    """An ``importance -> calibrated`` transition hitting one target exactly."""

    def compute(self, context: KernelContext) -> KernelResult:
        entity = str(context.params["target_entity"])
        column = str(context.params["target_column"])
        before = context.weights[entity].values
        values = context.tables[entity][column].to_numpy(dtype="float64")
        target = float(context.params["target_total"])
        after = before * (target / float(before @ values))
        receipt: dict[str, object] = {
            "target": {"entity": entity, "column": column, "total": target},
            "achieved": float(after @ values),
            "max_weight_ratio": context.params["max_weight_ratio"],
            "weight_anchor": context.params["weight_anchor"],
            "mass": _mass_record(context, before, after, "declared"),
            "consumes_se": self.capabilities.consumes_se,
        }
        if self.capabilities.consumes_se:
            receipt["se_seen"] = context.params.get("target_se")
        return KernelResult(
            weights=Weights(values=after, kind=WeightKind.CALIBRATED), receipt=receipt
        )


class GateThreshold(ToyKernel):
    """A gate: pass iff one column's mean sits inside a declared band."""

    def compute(self, context: KernelContext) -> KernelResult:
        entity = str(context.params["entity"])
        values = context.tables[entity][str(context.params["column"])]
        if not bool(context.params.get("applicable", True)):
            outcome, evidence = "not_applicable", {"reason": "declared inapplicable"}
        elif bool(values.isna().all()):
            outcome, evidence = "evidence_absent", {"reason": "every value is null"}
        else:
            observed = float(values.astype("float64").mean())
            low = float(context.params["low"])
            high = float(context.params["high"])
            outcome = "pass" if low <= observed <= high else "fail"
            evidence = {"observed": observed, "low": low, "high": high}
        return KernelResult(
            columns={
                ("release", "gate_verdict"): pd.Series(
                    [outcome], index=_owned_ids(context, "release"), dtype="string"
                )
            },
            receipt={"outcome": outcome, "evidence": evidence},
        )


class GateReportsTolerance(ToyKernel):
    """A gate that reports the input owner's declared numeric tolerance."""

    def compute(self, context: KernelContext) -> KernelResult:
        entity = str(context.params["entity"])
        column = str(context.params["column"])
        observed = float(context.tables[entity][column].astype("float64").mean())
        declared = context.tolerances[(entity, column)]
        tolerance = (
            None
            if declared is None
            else {
                "rtol": declared.rtol,
                "atol": declared.atol,
                "ulps": declared.ulps,
            }
        )
        verdict_column = str(context.params["verdict_column"])
        return KernelResult(
            columns={
                ("release", verdict_column): pd.Series(
                    ["pass"], index=_owned_ids(context, "release"), dtype="string"
                )
            },
            receipt={
                "outcome": "pass",
                "evidence": {"observed": observed, "tolerance": tolerance},
            },
        )


class ReleaseTier(ToyKernel):
    """Derive a release tier from the gate verdicts declared as its inputs."""

    def compute(self, context: KernelContext) -> KernelResult:
        table = context.tables["release"]
        verdicts = [
            str(value)
            for column in table.columns
            if column != id_column("release")
            for value in table[column]
        ]
        # The executor certifies when every ancestral gate passed or did not
        # apply; a kernel that disagreed with that derivation would be rejected.
        certified = bool(verdicts) and all(
            verdict in ("pass", "not_applicable") for verdict in verdicts
        )
        tier = "certified" if certified else "evidence"
        return KernelResult(
            columns={
                ("release", "tier"): pd.Series(
                    [tier], index=_owned_ids(context, "release"), dtype="string"
                )
            },
            receipt={
                "outcome": "pass" if certified else "fail",
                "tier": tier,
                "verdicts": tuple(verdicts),
            },
        )


# ----------------------------------------------------------------------
# Kernels that misbehave. Every one of them exists to be rejected.
# ----------------------------------------------------------------------


class WritesOutsideOwnership(ToyKernel):
    """Return values for every person, including rows outside the owned mask."""

    def compute(self, context: KernelContext) -> KernelResult:
        ids = pd.Index(np.arange(1, int(context.params["n_persons"]) + 1))
        return KernelResult(
            columns={
                ("person", str(context.params["target"])): pd.Series(
                    np.zeros(len(ids)), index=ids, dtype="float64"
                )
            }
        )


class ReturnsDenseBool(ToyKernel):
    """Return a dense numpy ``bool`` for a cell declared nullable ``boolean``."""

    def compute(self, context: KernelContext) -> KernelResult:
        ids = _owned_ids(context, "person")
        return KernelResult(
            columns={
                ("person", str(context.params["target"])): pd.Series(
                    np.ones(len(ids), dtype=bool), index=ids, dtype="bool"
                )
            }
        )


class MutatesItsInput(ToyKernel):
    """Write into a projected input view. A read-only view makes this raise."""

    def compute(self, context: KernelContext) -> KernelResult:
        values = context.tables["person"][str(context.params["column"])].to_numpy()
        values[0] = -1
        ids = _owned_ids(context, "person")
        return KernelResult(
            columns={
                ("person", str(context.params["target"])): pd.Series(
                    np.zeros(len(ids)), index=ids, dtype="float64"
                )
            }
        )


class WritesIntoAbsentCell(ToyKernel):
    """Write a real value into a cell the node declared ABSENT."""

    def compute(self, context: KernelContext) -> KernelResult:
        ids = _owned_ids(context, "person")
        return KernelResult(
            columns={
                ("person", str(context.params["target"])): pd.Series(
                    np.ones(len(ids)), index=ids, dtype="float64"
                )
            }
        )


class Raises(ToyKernel):
    """Fail, loudly, with a message the gate's evidence is expected to carry."""

    MESSAGE = "toy kernel exploded on purpose"

    def compute(self, context: KernelContext) -> KernelResult:
        raise RuntimeError(self.MESSAGE)


class GateRaises(ToyKernel):
    """A gate that fails by exception. Its role makes that a failed verdict."""

    MESSAGE = "toy gate exploded on purpose"

    def compute(self, context: KernelContext) -> KernelResult:
        raise RuntimeError(self.MESSAGE)


class CountsAndSucceeds(ToyKernel):
    """A do-nothing deterministic kernel whose only job is to be counted."""

    def compute(self, context: KernelContext) -> KernelResult:
        ids = _owned_ids(context, "person")
        return KernelResult(
            columns={
                ("person", str(context.params["target"])): pd.Series(
                    np.zeros(len(ids)), index=ids, dtype="float64"
                )
            },
            receipt={"counted": int(len(ids))},
        )


# ----------------------------------------------------------------------
# Registry
# ----------------------------------------------------------------------

_DETERMINISTIC = Capabilities(determinism=Determinism.DETERMINISTIC)
_SEEDED = Capabilities(
    determinism=Determinism.SEEDED,
    numeric=Numeric.BITWISE,
    seed_source=SeedSource.EXECUTOR,
)
_CREATE = Capabilities(
    determinism=Determinism.DETERMINISTIC, structural=StructuralDelta.CREATE
)
_FILTER = Capabilities(
    determinism=Determinism.DETERMINISTIC, structural=StructuralDelta.FILTER
)
_EXPAND = Capabilities(
    determinism=Determinism.DETERMINISTIC, structural=StructuralDelta.EXPAND
)
_REWEIGHT = Capabilities(
    determinism=Determinism.DETERMINISTIC, structural=StructuralDelta.REWEIGHT
)


def toy_registry(*, variants: Mapping[str, str] | None = None) -> KernelRegistry:
    """Every toy kernel, registered under the refs the toy graphs name.

    Args:
        variants: Kernel ref to a variant token. A ref listed here gets a
            different :meth:`ToyKernel.implementation_hash` and nothing else,
            which is how charter A5 moves exactly one kernel's code identity.
    """
    chosen = dict(variants or {})
    registry = KernelRegistry()
    kernels = (
        SourceCsv("source.csv@1", _CREATE),
        DeriveAdd("derive.add@1", _DETERMINISTIC),
        DeriveAdd(
            "derive.tolerant@1",
            Capabilities(
                determinism=Determinism.DETERMINISTIC,
                numeric=Numeric.TOLERANCE_BOUND,
                tolerance=TOY_TOLERANCE,
            ),
        ),
        DrawUniform("draw.uniform@1", _SEEDED),
        ImputeChain("impute.chain@1", _SEEDED),
        SimulateStub("simulate.stub@1", _DETERMINISTIC),
        PatchColumn("patch.column@1", _DETERMINISTIC),
        AbsentColumn("absent.column@1", _DETERMINISTIC),
        SelectRows("select.rows@1", _FILTER),
        ExpandEntrants("expand.entrants@1", _EXPAND),
        ExpandEntrantPerson("expand.entrant_person@1", _EXPAND),
        ClaimMaterializedExpand("claim.expand@1", _DETERMINISTIC),
        ReweightScale("reweight.scale@1", _REWEIGHT),
        CalibrateToy(
            "calibrate.toy@1",
            Capabilities(
                determinism=Determinism.DETERMINISTIC,
                structural=StructuralDelta.REWEIGHT,
                consumes_se=True,
            ),
        ),
        CalibrateToy("calibrate.blind@1", _REWEIGHT),
        GateThreshold(
            "gate.threshold@1",
            Capabilities(determinism=Determinism.DETERMINISTIC, role=KernelRole.GATE),
        ),
        GateReportsTolerance(
            "gate.tolerance@1",
            Capabilities(determinism=Determinism.DETERMINISTIC, role=KernelRole.GATE),
        ),
        ReleaseTier(
            "release.tier@1",
            Capabilities(
                determinism=Determinism.DETERMINISTIC, role=KernelRole.RELEASE
            ),
        ),
        CountsAndSucceeds("count.calls@1", _DETERMINISTIC),
        WritesOutsideOwnership("bad.outside@1", _DETERMINISTIC),
        ReturnsDenseBool("bad.dense_bool@1", _DETERMINISTIC),
        MutatesItsInput("bad.mutate@1", _DETERMINISTIC),
        WritesIntoAbsentCell("bad.absent@1", _DETERMINISTIC),
        Raises("bad.raise@1", _DETERMINISTIC),
        GateRaises(
            "bad.gate_raise@1",
            Capabilities(determinism=Determinism.DETERMINISTIC, role=KernelRole.GATE),
        ),
    )
    for kernel in kernels:
        kernel.variant = chosen.get(kernel.ref, kernel.variant)
        registry.register(kernel)
    return registry


# ----------------------------------------------------------------------
# Running a toy graph, and reading what came out
# ----------------------------------------------------------------------


@dataclass(frozen=True)
class ToyRun:
    """One executed toy graph and everything a property needs to inspect it.

    Attributes:
        manifest: What :func:`microcosm.graph.run_graph` returned.
        store: The content store the run wrote into.
        registry: The kernels it ran, still carrying their call counts.
        compiled: The compiled graph, for predecessors and owners.
        sources: The source paths the run read.
    """

    manifest: object
    store: object
    registry: KernelRegistry
    compiled: CompiledGraph
    sources: Mapping[str, Path]

    def keys(self) -> dict[str, str]:
        """Node id to node key, for every node in the graph."""
        return {
            node_id: self.manifest.nodes[node_id].key for node_id in self.compiled.order
        }

    def seeds(self) -> dict[str, int]:
        """Node id to RNG seed, for every node in the graph."""
        return {
            node_id: self.manifest.nodes[node_id].seed
            for node_id in self.compiled.order
        }

    def misses(self) -> set[str]:
        """Node ids the run executed rather than read from the store."""
        return {
            node_id
            for node_id in self.compiled.order
            if not self.manifest.nodes[node_id].hit
        }

    def hits(self) -> set[str]:
        """Node ids the run read from the store."""
        return set(self.compiled.order) - self.misses()

    def bytes_of(self, node_id: str) -> dict[tuple[str, str], tuple[bytes, bytes]]:
        """Every artifact of ``node_id`` as ``(values, null mask)`` bytes."""
        return {
            cell: artifact_bytes(self.store, key)
            for cell, key in self.manifest.nodes[node_id].artifacts.items()
        }

    def all_bytes(self) -> dict[str, dict[tuple[str, str], tuple[bytes, bytes]]]:
        """Every artifact of every node, keyed by node id."""
        return {node_id: self.bytes_of(node_id) for node_id in self.compiled.order}


def run_toy(
    graph: Graph,
    root: Path,
    *,
    sources: Mapping[str, Path] | None = None,
    registry: KernelRegistry | None = None,
    store: ContentStore | None = None,
    resume: str = "auto",
    decisions: Sequence[Mapping[str, object]] = (),
) -> ToyRun:
    """Compile ``graph`` and run it into a store under ``root``.

    Every default is the plain case: a fresh copy of the fixture tables, a
    fresh registry, a fresh store, ``resume="auto"``, no decisions. A property
    that is *about* one of those passes it explicitly instead.
    """
    compiled = compile_graph(graph)
    if sources is None:
        sources = {"survey": copy_source(root / "source")}
    if registry is None:
        registry = toy_registry()
    if store is None:
        store = ContentStore(root / "store")
    manifest = run_graph(
        compiled,
        sources=dict(sources),
        store=store,
        kernels=registry,
        resume=resume,
        decisions=tuple(decisions),
    )
    return ToyRun(
        manifest=manifest,
        store=store,
        registry=registry,
        compiled=compiled,
        sources=sources,
    )


def artifact_bytes(store: ContentStore, key: str) -> tuple[bytes, bytes]:
    """A stored column as ``(value bytes, null-mask bytes)``.

    Charter properties that say "byte-identical" compare this pair, so a
    nullable column's mask is compared as well as its values, and a float
    column's signed zeros and NaN payloads are compared as bits.
    """
    series = store.load_column(key)
    mask = series.isna().to_numpy(dtype="bool")
    values = series.to_numpy(dtype="object" if series.dtype == "string" else None)
    if series.dtype.name in {"boolean", "Int64"}:
        values = series.fillna(
            False if series.dtype.name == "boolean" else 0
        ).to_numpy()
    if values.dtype == object:
        values = np.array([str(value) for value in values], dtype="U")
    return values.tobytes(), mask.tobytes()


def descendants(compiled: CompiledGraph, *node_ids: str) -> set[str]:
    """The transitive descendants of ``node_ids``, from declared predecessors."""
    frontier = set(node_ids)
    found: set[str] = set()
    while frontier:
        found |= frontier
        frontier = {
            node_id
            for node_id in compiled.order
            if node_id not in found and found & set(compiled.predecessors[node_id])
        }
    return found - set(node_ids)


def calls_by_ref(registry: KernelRegistry) -> dict[str, int]:
    """Executions per kernel ref since the registry was built."""
    return {ref: kernel.calls for ref, kernel in registry.as_mapping().items()}


def total_calls(registry: KernelRegistry) -> int:
    """Total kernel executions since the registry was built."""
    return sum(calls_by_ref(registry).values())


# ----------------------------------------------------------------------
# Graphs
# ----------------------------------------------------------------------

CREATE = Node(
    "survey",
    "source.csv@1",
    sources=("survey",),
    structural=StructuralDelta.CREATE,
    outputs=(
        Owned("person", "age", "int64"),
        Owned("person", "income", "float64"),
        Owned("person", "is_adult", "boolean"),
        Owned("person", "receives_x", "boolean"),
        Owned("household", "household_size", "int64"),
    ),
    description="load the toy country",
)


def derive(
    node_id: str,
    columns: Sequence[str],
    target: str,
    *,
    scale: float = 1.0,
    population: str = "survey",
    entity: str = "person",
    description: str = "",
    citation: str = "",
) -> Node:
    """A deterministic node: ``target = scale * sum(columns)``."""
    return Node(
        node_id,
        "derive.add@1",
        inputs=(Slice(entity, tuple(columns)),),
        outputs=(Owned(entity, target, "float64"),),
        params={
            "entity": entity,
            "columns": tuple(columns),
            "target": target,
            "scale": scale,
        },
        population=population,
        description=description,
        citation=citation,
    )


def draw(node_id: str, target: str, *, population: str = "survey") -> Node:
    """A seeded node: one uniform draw per person, from the node key alone."""
    return Node(
        node_id,
        "draw.uniform@1",
        inputs=(Slice("person", ("age",)),),
        outputs=(Owned("person", target, "float64"),),
        params={"entity": "person", "target": target},
        population=population,
    )


def impute(
    node_id: str,
    predictors: Sequence[str],
    target: str,
    *,
    noise: float = 0.5,
    population: str = "survey",
) -> Node:
    """A seeded chained target: its predictors are exactly its declared inputs."""
    return Node(
        node_id,
        "impute.chain@1",
        inputs=(Slice("person", tuple(predictors)),),
        outputs=(Owned("person", target, "float64"),),
        params={
            "entity": "person",
            "predictors": tuple(predictors),
            "target": target,
            "noise": noise,
        },
        population=population,
    )


def patch_node(
    node_id: str,
    target: str,
    dtype: str,
    value: object,
    *,
    population: str,
    kernel: str = "patch.column@1",
    mask: str = "is_adult",
) -> Node:
    """Re-own a carried column at ``mask`` positions inside ``population``."""
    return Node(
        node_id,
        kernel,
        inputs=(Slice("person", (mask,), rows=mask),),
        outputs=(Owned("person", target, dtype, rows=mask),),
        params={"entity": "person", "target": target, "dtype": dtype, "value": value},
        population=population,
    )


def absent_node(
    node_id: str,
    target: str,
    *,
    population: str = "survey",
    kernel: str = "absent.column@1",
    dtype: str = "float64",
) -> Node:
    """A node that declares its owned cell ABSENT: the column is null."""
    return Node(
        node_id,
        kernel,
        inputs=(Slice("person", ("age",)),),
        outputs=(Owned("person", target, dtype, ownership=Ownership.ABSENT),),
        params={"entity": "person", "target": target, "dtype": dtype},
        population=population,
    )


def select_node(
    node_id: str = "adults",
    *,
    base: str = "survey",
    mask: str = "is_adult",
    policy: str = "free",
) -> Node:
    """A FILTER: keep the persons ``mask`` marks, under a declared mass policy."""
    return Node(
        node_id,
        "select.rows@1",
        structural=StructuralDelta.FILTER,
        base=base,
        inputs=(Slice("person", (mask,)), Slice("household", ("household_size",))),
        params={"mask": mask, "policy": policy},
        mass=policy,
    )


def entrant_expand_node(
    node_id: str = "scheduled_entries",
    *,
    entrants: bool = True,
    missing_entrant_column: str | None = None,
) -> tuple[Node, Node]:
    """An EXPAND plus the ownership claim for its materialized person fields."""
    overlays = (
        ("household", "household_size", "int64"),
        ("person", "person_household_id", "int64"),
    )
    claim_cells = (overlays[0],)
    expand = Node(
        node_id,
        "expand.entrants@1",
        structural=StructuralDelta.EXPAND,
        base="survey",
        inputs=(
            Slice("person", ("age",)),
            Slice("household", ("household_size",)),
        ),
        params={
            "expand_cells": overlays,
            "expand_weight_entity": "household",
            "expand_weight_kind": "design",
            "entrant_weight": 125.0,
            "missing_entrant_column": missing_entrant_column,
        },
        mass="free",
        entrants=entrants,
    )
    claim = Node(
        f"claim_{node_id}",
        "claim.expand@1",
        outputs=tuple(
            Owned(entity, column, dtype) for entity, column, dtype in claim_cells
        ),
        params={
            "claim_cells": claim_cells,
            "materialized_expand_outputs": tuple(
                f"{entity}.{column}" for entity, column, _ in claim_cells
            ),
        },
        population=node_id,
    )
    return expand, claim


def entrant_person_node(
    node_id: str = "immigrant_cohort",
    *,
    strata_mode: str = "ok",
) -> tuple[Node, Node]:
    """An EXPAND admitting one entrant person, plus the claim of its cells."""
    data_cells = (
        ("person", "age", "int64"),
        ("person", "income", "float64"),
        ("person", "is_adult", "boolean"),
        ("person", "receives_x", "boolean"),
    )
    overlays = (
        *data_cells,
        ("person", "person_household_id", "int64"),
        ("person", "person_release_id", "int64"),
    )
    expand = Node(
        node_id,
        "expand.entrant_person@1",
        structural=StructuralDelta.EXPAND,
        base="survey",
        inputs=(
            Slice("person", ("age", "income", "is_adult", "receives_x")),
            Slice("household", ("household_size",)),
        ),
        params={
            "expand_cells": overlays,
            "expand_weight_entity": "household",
            "expand_weight_kind": "design",
            "strata_mode": strata_mode,
        },
        mass="free",
        entrants=True,
    )
    claim = Node(
        f"claim_{node_id}",
        "claim.expand@1",
        outputs=tuple(
            Owned(entity, column, dtype) for entity, column, dtype in data_cells
        ),
        params={
            "claim_cells": data_cells,
            "materialized_expand_outputs": tuple(
                f"{entity}.{column}" for entity, column, _ in data_cells
            ),
        },
        population=node_id,
    )
    return expand, claim


POOL = Node(
    "pool",
    "reweight.scale@1",
    structural=StructuralDelta.REWEIGHT,
    base="survey",
    inputs=(Slice("person", ("age",)), Slice("household", ("household_size",))),
    params={"entity": "household", "factor": 2.0, "to_kind": "importance"},
    weights=WeightTransition("household", "importance", mass="free"),
    mass="free",
    description="design -> importance",
)


def reweight_node(
    node_id: str,
    *,
    base: str,
    to_kind: str,
    factor: float = 2.0,
    mass: str = "free",
) -> Node:
    """A weight-kind transition node, for the ordering tests of charter D1."""
    return Node(
        node_id,
        "reweight.scale@1",
        structural=StructuralDelta.REWEIGHT,
        base=base,
        inputs=(Slice("person", ("age",)), Slice("household", ("household_size",))),
        params={"entity": "household", "factor": factor, "to_kind": to_kind},
        weights=WeightTransition("household", to_kind, mass=mass),
        mass=mass,
    )


def calibrated_node(
    node_id: str = "calibrated",
    *,
    base: str = "pool",
    kernel: str = "calibrate.toy@1",
    target_total: float = 400_000.0,
    max_weight_ratio: float = 5.0,
    target_se: float = 2500.0,
) -> Node:
    """An ``importance -> calibrated`` node hitting one declared target sum."""
    return Node(
        node_id,
        kernel,
        structural=StructuralDelta.REWEIGHT,
        base=base,
        inputs=(Slice("person", ("age",)), Slice("household", ("household_size",))),
        params={
            "target_entity": "household",
            "target_column": "household_size",
            "target_total": target_total,
            "target_se": target_se,
            "max_weight_ratio": max_weight_ratio,
            "weight_anchor": "design",
        },
        weights=WeightTransition("household", "calibrated", mass="declared"),
        mass="declared",
        description="importance -> calibrated",
    )


def simulate_node(population: str = "calibrated") -> Node:
    """The stub rules engine: a flat rate on the chained target."""
    return Node(
        "sim",
        "simulate.stub@1",
        inputs=(Slice("person", ("target_b",)),),
        outputs=(Owned("person", "tax", "float64"),),
        params={"entity": "person", "column": "target_b", "target": "tax", "rate": 0.2},
        population=population,
    )


def gate_node(
    node_id: str = "gate_tax",
    *,
    population: str = "calibrated",
    entity: str = "person",
    column: str = "tax",
    low: float = -1e12,
    high: float = 1e12,
    kernel: str = "gate.threshold@1",
    applicable: bool = True,
) -> Node:
    """A gate node: a verdict column on the one-row ``release`` entity."""
    return Node(
        node_id,
        kernel,
        inputs=(Slice(entity, (column,)),),
        outputs=(Owned("release", "gate_verdict", "string"),),
        params={
            "gate": True,
            "entity": entity,
            "column": column,
            "low": low,
            "high": high,
            "applicable": applicable,
        },
        population=population,
        description="the toy country's gate",
    )


def release_node(
    *,
    population: str = "calibrated",
    requires_decisions: Sequence[str] = (),
) -> Node:
    """A release node whose tier is derived from its declared gate ancestry."""
    return Node(
        "release",
        "release.tier@1",
        inputs=(Slice("release", ("gate_verdict",)),),
        outputs=(Owned("release", "tier", "string"),),
        params={"release": True, "requires_decisions": tuple(requires_decisions)},
        population=population,
        description="the toy country's release",
    )


def small_graph(
    *, extra: Sequence[Node] = (), nodes: Sequence[Node] | None = None
) -> Graph:
    """CREATE plus one deterministic node, one draw, and one chained pair."""
    if nodes is None:
        nodes = (
            CREATE,
            derive("resources", ("age", "income"), "resources", scale=1.5),
            draw("draw_a", "noise_a"),
            impute("target_a", ("age", "income"), "target_a"),
            impute("target_b", ("age", "target_a"), "target_b"),
        )
    return Graph("toy", (SOURCE,), (*nodes, *extra))


def chained_graph(leaves: Sequence[str] = ()) -> Graph:
    """The small graph plus one independent leaf per name in ``leaves``.

    The ``0347a009`` replay removes leaves from this graph and asserts that no
    surviving node's key, seed, or output moves.
    """
    return small_graph(
        extra=tuple(draw(f"leaf_{name}", f"leaf_{name}_value") for name in leaves)
    )


def full_graph(
    *,
    gate_low: float = -1e12,
    gate_high: float = 1e12,
    requires_decisions: Sequence[str] = (),
) -> Graph:
    """Charter G3's end-to-end toy country.

    source -> two chained targets -> reweight -> calibrate -> simulate (stub
    engine) -> gate -> release.
    """
    return Graph(
        "toy",
        (SOURCE,),
        (
            CREATE,
            derive("resources", ("age", "income"), "resources", scale=1.5),
            impute("target_a", ("age", "income"), "target_a"),
            impute("target_b", ("age", "target_a"), "target_b"),
            POOL,
            calibrated_node(),
            simulate_node(),
            gate_node(low=gate_low, high=gate_high),
            release_node(requires_decisions=requires_decisions),
        ),
    )


def count_node(
    node_id: str,
    reads: Sequence[str],
    target: str,
    *,
    population: str = "survey",
) -> Node:
    """A node whose only purpose is to prove whether the executor ran it."""
    return Node(
        node_id,
        "count.calls@1",
        inputs=(Slice("person", tuple(reads)),),
        outputs=(Owned("person", target, "float64"),),
        params={"entity": "person", "target": target},
        population=population,
    )


def replace_node(graph: Graph, *nodes: Node) -> Graph:
    """``graph`` with each node in ``nodes`` swapped in by id."""
    swapped = {node.id: node for node in nodes}
    return Graph(
        graph.country,
        graph.sources,
        tuple(swapped.get(node.id, node) for node in graph.nodes),
    )


def drop_nodes(graph: Graph, *node_ids: str) -> Graph:
    """``graph`` without the named nodes. Used by the ``0347a009`` replay."""
    dropped = set(node_ids)
    return Graph(
        graph.country,
        graph.sources,
        tuple(node for node in graph.nodes if node.id not in dropped),
    )


def graph_source_files() -> tuple[Path, ...]:
    """Every source file of the ``microcosm-graph`` shard.

    Located through the public package's ``__file__``, not by reaching into a
    private module: charter G2 and C4 are static properties of the shard, and
    the charter states them as AST checks.
    """
    import microcosm.graph

    root = Path(microcosm.graph.__file__).resolve().parent
    return tuple(sorted(root.rglob("*.py")))


def surviving_design_anchor(
    mask: str = "is_adult", column: str = "household_size"
) -> float:
    """``design @ column`` over the households a ``mask`` filter would keep.

    Charter D3 needs a calibration target whose ratio against the *design*
    anchor breaks the declared cap while its ratio against the incoming
    importance weights does not. Deriving the number from the fixture keeps
    the test free of magic constants.
    """
    frame = read_toy_frame(FIXTURES)
    person = frame.person
    keep = person[mask].to_numpy(dtype="bool", na_value=False)
    households = frame.table("household")
    referenced = set(person.loc[keep, "person_household_id"])
    surviving = households[id_column("household")].isin(referenced).to_numpy()
    design = frame.weights_for("household").values[surviving]
    values = households[column].to_numpy(dtype="float64")[surviving]
    return float(design @ values)


#: A signed human decision, as charter F5 consumes one.
PUBLISH_DECISION = {
    "name": "publish",
    "owner": "maria",
    "signature": "toy-signature-0001",
}
