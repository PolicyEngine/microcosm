"""Charter group D: weights and mass.

The typed-weights datatype exists; the review found its conservation and
lineage promise does not hold. Kind transitions compare against explicit
weights only, so a calibrated vector can re-enter as design through
inheritance; per-stratum mass is never conserved; ``select()`` drops mass
without a log entry; ``max_weight_ratio`` means "R times the starting vector of
this call", so a selection-then-refit chain ships records at up to R squared
relative to design while every local assertion passes (#493).

Each of those is one property here, stated as the executor rejecting the
declaration rather than a verifier noticing afterwards.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from microcosm.frame import WeightKind
from microcosm.graph import (
    Graph,
    GraphError,
    Node,
    NodeRejectedError,
    Owned,
    Slice,
    StructuralDelta,
    WeightTransition,
    compile_graph,
)

if "_toy" not in sys.modules:
    _SPEC = importlib.util.spec_from_file_location(
        "_toy", Path(__file__).with_name("_toy.py")
    )
    sys.modules["_toy"] = importlib.util.module_from_spec(_SPEC)
    _SPEC.loader.exec_module(sys.modules["_toy"])
toy = sys.modules["_toy"]


def test_d1_weight_transitions_are_typed_nodes(tmp_path: Path) -> None:
    """``design -> importance -> calibrated`` and nothing else.

    A regression is rejected, and so is a transition declared on weights the
    entity does not explicitly carry — the inheritance hole that lets a
    calibrated household vector re-enter as design through its persons.
    """
    run = toy.run_toy(toy.full_graph(), tmp_path / "forward")
    kinds = {
        version: run.manifest.population(version).weights_for("household").kind
        for version in ("survey", "pool", "calibrated")
    }
    assert kinds == {
        "survey": WeightKind.DESIGN,
        "pool": WeightKind.IMPORTANCE,
        "calibrated": WeightKind.CALIBRATED,
    }

    from microcosm.graph import NodeRejectedError

    backwards = toy.small_graph(
        nodes=(
            toy.CREATE,
            toy.POOL,
            toy.calibrated_node(),
            toy.reweight_node("regress", base="calibrated", to_kind="importance"),
        )
    )
    with pytest.raises(NodeRejectedError, match="regress"):
        toy.run_toy(backwards, tmp_path / "backwards")

    inherited = Node(
        "person_reweight",
        "reweight.scale@1",
        structural=StructuralDelta.REWEIGHT,
        base="survey",
        inputs=(Slice("person", ("age",)), Slice("household", ("household_size",))),
        params={"entity": "person", "factor": 2.0, "to_kind": "importance"},
        weights=WeightTransition("person", "importance", mass="free"),
        mass="free",
    )
    with pytest.raises(NodeRejectedError, match="person_reweight"):
        toy.run_toy(
            toy.small_graph(nodes=(toy.CREATE, inherited)), tmp_path / "inherited"
        )


def test_d2_mass_ledger(tmp_path: Path) -> None:
    """Population-changing nodes record mass, and ``conserve`` means conserve.

    A ``select`` that drops persons loses mass; under ``conserve`` that fails
    the node, and under ``free`` it is recorded, before and after, in total and
    per stratum. Neither path lets mass leave silently.
    """
    free = toy.run_toy(
        toy.small_graph(nodes=(toy.CREATE, toy.POOL, toy.select_node(base="pool"))),
        tmp_path / "free",
    )
    for node_id in ("pool", "adults"):
        mass = free.manifest.nodes[node_id].receipt["mass"]
        assert set(mass) == {
            "policy",
            "before",
            "after",
            "stratum_before",
            "stratum_after",
        }
        assert set(mass["stratum_before"]) == set(toy.STRATA)
        assert set(mass["stratum_after"]) == set(toy.STRATA)
    dropped = free.manifest.nodes["adults"].receipt["mass"]
    assert dropped["after"] < dropped["before"]
    assert all(
        dropped["stratum_after"][label] < dropped["stratum_before"][label]
        for label in toy.STRATA
    )

    from microcosm.graph import NodeRejectedError

    conserving = toy.small_graph(
        nodes=(toy.CREATE, toy.POOL, toy.select_node(base="pool", policy="conserve"))
    )
    with pytest.raises(NodeRejectedError, match="adults"):
        toy.run_toy(conserving, tmp_path / "conserve")


def test_d3_cap_anchored_to_design(tmp_path: Path) -> None:
    """``max_weight_ratio`` is asserted against design across composed stages.

    The chain is the #493 shape: importance re-weighting, then a selection,
    then a refit. Its calibration target is chosen so the refit's ratio against
    its own starting vector is 1.5 — inside a declared cap of 2 — while the
    shipped record sits at 3 times design. A cap anchored to the incoming
    vector passes it; a cap anchored to design must not.
    """
    anchor = toy.surviving_design_anchor()
    chain = (toy.CREATE, toy.POOL, toy.select_node("adults", base="pool"))

    within = toy.small_graph(
        nodes=(
            *chain,
            toy.calibrated_node(
                base="adults", target_total=1.5 * anchor, max_weight_ratio=2.0
            ),
        )
    )
    run = toy.run_toy(within, tmp_path / "within")
    assert run.manifest.nodes["calibrated"].receipt["weight_anchor"] == "design"

    over = toy.small_graph(
        nodes=(
            *chain,
            toy.calibrated_node(
                base="adults", target_total=3.0 * anchor, max_weight_ratio=2.0
            ),
        )
    )
    from microcosm.graph import NodeRejectedError

    with pytest.raises(NodeRejectedError, match="calibrated"):
        toy.run_toy(over, tmp_path / "over")


def test_d4_filters_are_binary(tmp_path: Path) -> None:
    """A row mask is boolean and complete: not a float, not a column with nulls.

    ``target.py`` converts a filter with ``mask != 0``, so a NaN is included
    rather than refused. Here the dtype is known at compile time and refused
    there, and a nullable mask that actually carries nulls is refused by the
    executor.
    """
    numeric_mask = toy.patch_node(
        "wrong_mask", "flag", "float64", 1.0, population="survey", mask="income"
    )
    with pytest.raises(GraphError, match="income"):
        compile_graph(toy.small_graph(nodes=(toy.CREATE, numeric_mask)))

    nullable_mask = toy.patch_node(
        "null_mask", "flag", "float64", 1.0, population="survey", mask="receives_x"
    )
    from microcosm.graph import NodeRejectedError

    with pytest.raises(NodeRejectedError, match="null_mask"):
        toy.run_toy(
            toy.small_graph(nodes=(toy.CREATE, nullable_mask)), tmp_path / "nulls"
        )


def test_d5_uncertainty_travels(tmp_path: Path) -> None:
    """A declared standard error reaches the kernel, or the kernel says it does
    not.

    ``TargetSpec.se`` is validated today, documented as unused, and dropped by
    ``to_target()``. Here the declared ``se`` is in the calibration kernel's
    inputs and echoed in its receipt; a kernel that ignores it carries
    ``consumes_se=False`` in the capability record the manifest keeps, so the
    omission is visible rather than silent.
    """
    consuming = toy.run_toy(toy.full_graph(), tmp_path / "consuming")
    receipt = consuming.manifest.nodes["calibrated"].receipt
    assert receipt["se_seen"] == 2500.0
    assert receipt["capabilities"]["consumes_se"] is True

    blind = toy.small_graph(
        nodes=(
            toy.CREATE,
            toy.POOL,
            toy.calibrated_node(kernel="calibrate.blind@1"),
        )
    )
    ignored = toy.run_toy(blind, tmp_path / "blind").manifest.nodes["calibrated"]
    assert ignored.receipt["capabilities"]["consumes_se"] is False
    assert "se_seen" not in ignored.receipt
    assert toy.calibrated_node(kernel="calibrate.blind@1").params["target_se"] == 2500.0


@pytest.mark.xfail(strict=True, reason="charter D6: partitioned mass pending")
def test_d6_mass_is_partitioned(tmp_path: Path) -> None:
    """Mass is conserved and receipted inside every partition value.

    The receipt retains D2's flat totals and stratum maps and adds
    ``mass["partition"]`` with ``entity``, ``column``, and nested
    ``stratum_before``/``stratum_after`` maps keyed first by the JSON-string
    partition value and then by stratum. This test fixes that public shape for
    the implementation lane.
    """
    missing = Graph(
        "toy",
        (toy.SOURCE,),
        (toy.CREATE,),
        mass_partition=("person", "period"),
    )
    with pytest.raises(GraphError, match=r"person\.period.*survey"):
        compile_graph(missing)

    source_path = toy.copy_source(tmp_path / "period-source")
    person_path = source_path / "person.csv"
    person = pd.read_csv(person_path)
    is_adult = person["is_adult"].astype("boolean").fillna(False).to_numpy(bool)
    person["period"] = np.where(
        ~is_adult | person["person_id"].mod(2).eq(0), 2024, 2025
    ).astype("int64")
    person.to_csv(person_path, index=False)

    schema_path = source_path / "schema.json"
    schema = json.loads(schema_path.read_text())
    schema["data_columns"]["person"].append("period")
    schema["dtypes"]["period"] = "int64"
    schema_path.write_text(json.dumps(schema, indent=2, sort_keys=True) + "\n")

    create = replace(
        toy.CREATE,
        outputs=(*toy.CREATE.outputs, Owned("person", "period", "int64")),
    )
    sources = {"survey": source_path}
    conserving = Graph(
        "toy",
        (toy.SOURCE,),
        (
            create,
            toy.select_node("period_conserve", base="survey", policy="conserve"),
        ),
        mass_partition=("person", "period"),
    )
    with pytest.raises(NodeRejectedError, match="period_conserve") as rejected:
        toy.run_toy(conserving, tmp_path / "conserve", sources=sources)
    assert "2024" in str(rejected.value)

    free = Graph(
        "toy",
        (toy.SOURCE,),
        (create, toy.select_node("period_free", base="survey", policy="free")),
        mass_partition=("person", "period"),
    )
    run = toy.run_toy(free, tmp_path / "free", sources=sources)
    mass = run.manifest.nodes["period_free"].receipt["mass"]
    partition = mass["partition"]
    assert set(partition) == {
        "entity",
        "column",
        "stratum_before",
        "stratum_after",
    }
    assert partition["entity"] == "person"
    assert partition["column"] == "period"

    weights = pd.read_csv(source_path / "weights.csv").set_index("household_id")[
        "design_weight"
    ]
    weighted = person.assign(
        _mass=person["person_household_id"].map(weights).astype("float64")
    )

    def expected(frame: pd.DataFrame) -> dict[str, dict[str, float]]:
        grouped = frame.groupby(["period", "stratum"], observed=True)["_mass"].sum()
        return {
            str(period): {
                str(stratum): float(grouped.loc[(period, stratum)])
                for stratum in grouped.loc[period].index
            }
            for period in grouped.index.get_level_values("period").unique()
        }

    before = expected(weighted)
    after = expected(weighted.loc[is_adult])
    for key, wanted in (("stratum_before", before), ("stratum_after", after)):
        actual = partition[key]
        assert set(actual) == set(wanted) == {"2024", "2025"}
        for period, strata in wanted.items():
            assert actual[period] == pytest.approx(strata)

    before_total = sum(sum(strata.values()) for strata in before.values())
    after_total = sum(sum(strata.values()) for strata in after.values())
    assert mass["before"] == pytest.approx(before_total)
    assert mass["after"] == pytest.approx(after_total)
    assert partition["stratum_after"]["2025"] == pytest.approx(
        partition["stratum_before"]["2025"]
    )
    assert sum(partition["stratum_after"]["2024"].values()) < sum(
        partition["stratum_before"]["2024"].values()
    )


def test_the_toy_country_declares_the_weight_lineage_the_charter_names() -> None:
    """The toy graph really does compose the three kinds, in order.

    Green from the first commit: it guards the fixtures the four xfail
    properties above are written against, so a change to ``_toy.py`` that
    quietly drops a transition fails loudly instead of weakening D1–D3.
    """
    compiled = compile_graph(toy.full_graph())
    transitions = [
        compiled.graph.node(node_id).weights
        for node_id in compiled.order
        if compiled.graph.node(node_id).weights is not None
    ]
    assert [t.to_kind for t in transitions] == ["importance", "calibrated"]
    assert [t.entity for t in transitions] == ["household", "household"]
    assert [t.mass for t in transitions] == ["free", "declared"]
    assert toy.CREATE.outputs[0] == Owned("person", "age", "int64")
