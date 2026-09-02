"""Charter group B: ownership and mutation.

The executor, not a convention and not an after-the-fact verifier, is what
keeps a kernel inside its declared cells. These are the runtime halves; the
compile-time halves of B1 (two owners, an unowned column) are already in
``test_graph_decl.py`` and are not repeated.

B3 is the WIC guard made structural. The 8/30 breach was a dense ``bool``
written over a nullable ``boolean`` incumbent, and the doctrine is that
non-donor cells stay byte-identical. Here a node in a later population version
re-owns a carried column at masked positions, and everything it does not own
has to come through untouched — bits and null mask alike.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from microcosm.graph import (
    GraphError,
    KernelContext,
    Node,
    NodeRejectedError,
    Owned,
    Ownership,
    Slice,
    StructuralDelta,
)

if "_toy" not in sys.modules:
    _SPEC = importlib.util.spec_from_file_location(
        "_toy", Path(__file__).with_name("_toy.py")
    )
    sys.modules["_toy"] = importlib.util.module_from_spec(_SPEC)
    _SPEC.loader.exec_module(sys.modules["_toy"])
toy = sys.modules["_toy"]

#: Columns a frame carries as structure rather than as owned data.
STRUCTURAL_COLUMNS = frozenset(
    {
        "person_id",
        "person_household_id",
        "person_release_id",
        "household_id",
        "release_id",
    }
)


def _bits(series: pd.Series) -> tuple[bytes, bytes]:
    """``(value bytes, null-mask bytes)`` — what "byte-identical" means here."""
    mask = series.isna().to_numpy(dtype="bool")
    if series.dtype.name == "boolean":
        values = series.fillna(False).to_numpy(dtype="bool")
    else:
        values = series.to_numpy()
    return values.tobytes(), mask.tobytes()


def _adult_mask(person: pd.DataFrame) -> np.ndarray:
    return person["is_adult"].to_numpy(dtype="bool", na_value=False)


def _owners_along_base_chain(
    compiled: object, version: str, entity: str, column: str
) -> list[str]:
    """Every node owning ``entity.column`` along ``version``'s base chain."""
    found: list[str] = []
    seen = version
    while True:
        owner = compiled.owners.get((seen, entity, column))
        if owner is not None:
            found.append(owner)
        node = compiled.graph.node(seen)
        if node.base is None:
            return found
        seen = node.base


def test_b1_ownership_is_total_and_exclusive_in_the_population(
    tmp_path: Path,
) -> None:
    """Every non-structural column of a run's population has exactly one owner.

    Total: no column appears that no node declared. Exclusive: no column is
    claimed twice in one version, and a re-owned column in a later version
    resolves to the nearest owner rather than to two owners at once.
    """
    run = toy.run_toy(toy.full_graph(), tmp_path / "run")
    for version in ("survey", "pool", "calibrated"):
        frame = run.manifest.population(version)
        for entity in frame.entities:
            for column in frame.table(entity).columns:
                if column in STRUCTURAL_COLUMNS:
                    continue
                owners = _owners_along_base_chain(run.compiled, version, entity, column)
                assert owners, f"{version}/{entity}.{column} is owned by nobody"
                assert len(set(owners)) == len(owners)
                assert (entity, column) in run.manifest.nodes[owners[0]].artifacts


def test_b2_executor_enforces_ownership(tmp_path: Path) -> None:
    """Cells outside the declared owned positions are rejected, and a kernel
    never holds the population that would let it write them itself."""
    assert set(KernelContext.__dataclass_fields__) == {
        "node",
        "tables",
        "weights",
        "strata",
        "params",
        "rng",
        "sources",
        "tolerances",  # amendment 13: declared tolerances of the inputs' owners
    }

    graph = toy.small_graph(
        nodes=(
            toy.CREATE,
            toy.POOL,
            toy.patch_node(
                "overreach",
                "reach",
                "float64",
                0.0,
                population="pool",
                kernel="bad.outside@1",
            ),
        )
    )
    from microcosm.graph import NodeRejectedError

    with pytest.raises(NodeRejectedError, match="overreach"):
        toy.run_toy(graph, tmp_path / "bad")

    honest = toy.replace_node(
        graph, toy.patch_node("overreach", "reach", "float64", 7.0, population="pool")
    )
    person = toy.run_toy(honest, tmp_path / "good").manifest.population("pool").person
    adult = _adult_mask(person)
    assert (person.loc[adult, "reach"] == 7.0).all()
    assert person.loc[~adult, "reach"].isna().all()


def test_b3_storage_preserving_patch(tmp_path: Path) -> None:
    """A masked patch keeps the incumbent dtype and every non-owned bit.

    The nullable ``boolean`` stays nullable ``boolean`` — the WIC breach — and
    the float column's non-owned positions, negative zeros included, survive
    as bytes.
    """
    graph = toy.small_graph(
        nodes=(
            toy.CREATE,
            toy.POOL,
            toy.patch_node(
                "patch_flag", "receives_x", "boolean", True, population="pool"
            ),
            toy.patch_node("patch_money", "income", "float64", 12.5, population="pool"),
        )
    )
    run = toy.run_toy(graph, tmp_path / "run")
    before = run.manifest.population("survey").person
    after = run.manifest.population("pool").person
    adult = _adult_mask(before)

    assert (
        after["receives_x"].dtype.name == before["receives_x"].dtype.name == "boolean"
    )
    assert after["income"].dtype == before["income"].dtype == np.dtype("float64")

    patched = after.loc[adult, "receives_x"]
    assert patched.notna().all()
    assert patched.fillna(False).to_numpy(dtype="bool").all()
    assert _bits(after.loc[~adult, "receives_x"]) == _bits(
        before.loc[~adult, "receives_x"]
    )
    assert before.loc[~adult, "receives_x"].isna().any(), "the fixture needs nulls"

    assert (after.loc[adult, "income"] == 12.5).all()
    kept_before = before.loc[~adult, "income"].to_numpy()
    kept_after = after.loc[~adult, "income"].to_numpy()
    assert kept_after.tobytes() == kept_before.tobytes()
    assert np.signbit(kept_before).any(), "the fixture needs negative zeros"
    assert np.array_equal(np.signbit(kept_after), np.signbit(kept_before))


def test_b4_inputs_are_immutable(tmp_path: Path) -> None:
    """An in-place write into a projected view raises, and the node fails.

    Nothing the kernel touched reaches the store: a later clean run over the
    same source reproduces the fixture's ``age`` column byte for byte.
    """
    vandal = Node(
        "vandal",
        "bad.mutate@1",
        inputs=(Slice("person", ("age",)),),
        outputs=(Owned("person", "vandalised", "float64"),),
        params={"column": "age", "target": "vandalised"},
        population="survey",
    )
    sources = toy.toy_sources(tmp_path)
    from microcosm.graph import NodeRejectedError

    with pytest.raises(NodeRejectedError, match="vandal"):
        toy.run_toy(
            toy.small_graph(nodes=(toy.CREATE, vandal)),
            tmp_path / "bad",
            sources=sources,
        )

    clean = toy.run_toy(toy.small_graph(), tmp_path / "clean", sources=sources)
    fixture = toy.read_toy_frame(sources["survey"])
    assert (
        clean.manifest.population("survey").person["age"].to_numpy().tobytes()
        == fixture.person["age"].to_numpy().tobytes()
    )


def test_b5_null_means_absence(tmp_path: Path) -> None:
    """An ABSENT cell is null everywhere, and writing a value into one fails."""
    honest = toy.small_graph(
        nodes=(toy.CREATE, toy.absent_node("no_data", "unobserved"))
    )
    run = toy.run_toy(honest, tmp_path / "honest")
    assert run.manifest.population("survey").person["unobserved"].isna().all()
    assert run.compiled.graph.node("no_data").outputs[0].ownership is Ownership.ABSENT
    assert ("person", "unobserved") in run.manifest.nodes["no_data"].artifacts

    liar = toy.small_graph(
        nodes=(
            toy.CREATE,
            toy.absent_node("no_data", "unobserved", kernel="bad.absent@1"),
        )
    )
    from microcosm.graph import NodeRejectedError

    with pytest.raises(NodeRejectedError, match="no_data"):
        toy.run_toy(liar, tmp_path / "liar")


def test_b6_entrants_are_declared(tmp_path: Path) -> None:
    """Null lineage is an explicit, complete, and receipted entrant contract.

    The backwards-compatible lineage shape keeps ``(new, source)`` pairs under
    ``expand`` and uses a null source for entrants. The test admits a household
    entrant because the frozen result interface has no separate output for a
    new person's stratum. Its complete carried data surface is materialized by
    the EXPAND kernel and passed through ``materialized_expand_outputs``.
    """
    with pytest.raises(GraphError, match="conserved_entrants"):
        Node(
            "conserved_entrants",
            "expand.entrants@1",
            structural=StructuralDelta.EXPAND,
            base="survey",
            entrants=True,
            mass="conserve",
        )

    expand, claim = toy.entrant_expand_node()
    run = toy.run_toy(
        toy.small_graph(nodes=(toy.CREATE, expand, claim)), tmp_path / "declared"
    )
    before = run.manifest.population("survey")
    after = run.manifest.population(expand.id)
    person_copy_id = int(before.person["person_id"].max()) + 1
    household_entrant_id = int(before.household["household_id"].max()) + 1

    assert len(after.person) == len(before.person) + 1
    assert len(after.household) == len(before.household) + 1
    copied = after.person.set_index("person_id").loc[person_copy_id]
    source = before.person.set_index("person_id").loc[1]
    assert copied["person_household_id"] == household_entrant_id
    pd.testing.assert_series_equal(
        copied.drop(labels="person_household_id"),
        source.drop(labels="person_household_id"),
        check_names=False,
    )
    entrant = after.household.set_index("household_id").loc[household_entrant_id]
    assert entrant["household_size"] == 1
    assert after.household["household_size"].dtype == np.dtype("int64")
    assert after.weights_for("household").values[-1] == 125.0
    lineage = run.manifest.nodes[expand.id].receipt
    assert lineage["expand"]["person"] == ((person_copy_id, 1),)
    assert lineage["expand"]["household"] == ((household_entrant_id, None),)

    undeclared, undeclared_claim = toy.entrant_expand_node(
        "undeclared_entrants", entrants=False
    )
    with pytest.raises(NodeRejectedError, match="undeclared_entrants"):
        toy.run_toy(
            toy.small_graph(nodes=(toy.CREATE, undeclared, undeclared_claim)),
            tmp_path / "undeclared",
        )

    incomplete, incomplete_claim = toy.entrant_expand_node(
        "incomplete_entrant", missing_entrant_column="household_size"
    )
    with pytest.raises(NodeRejectedError, match="incomplete_entrant") as error:
        toy.run_toy(
            toy.small_graph(nodes=(toy.CREATE, incomplete, incomplete_claim)),
            tmp_path / "incomplete",
        )
    assert "household_size" in str(error.value)


@pytest.mark.xfail(strict=True, reason="charter B7: entrant person strata pending")
def test_b7_entrant_persons_carry_their_stratum(tmp_path: Path) -> None:
    """An entrant person's stratum arrives through ``KernelResult.strata``.

    Immigrant cohorts are persons, so an EXPAND admitting entrants must be
    able to add a person that copies nobody: every column materialized,
    memberships naming incumbent groups, and its stratum declared by id.
    The ledger counts the entrant from the node that admits it; a missing
    label, a label for an id the node never adds, or a label for an
    incumbent person rejects the node by name.
    """
    expand, claim = toy.entrant_person_node()
    run = toy.run_toy(
        toy.small_graph(nodes=(toy.CREATE, expand, claim)), tmp_path / "cohort"
    )
    before = run.manifest.population("survey")
    after = run.manifest.population(expand.id)
    entrant_id = int(before.person["person_id"].max()) + 1

    assert len(after.person) == len(before.person) + 1
    assert len(after.household) == len(before.household)
    entrant = after.person.set_index("person_id").loc[entrant_id]
    assert entrant["age"] == 30 and entrant["income"] == 12_500.0
    assert (
        entrant["person_household_id"] == before.person["person_household_id"].iloc[0]
    )
    assert after.strata.iloc[-1] == "urban"
    assert after.strata.iloc[: len(before.person)].tolist() == before.strata.tolist()

    receipt = run.manifest.nodes[expand.id].receipt
    assert receipt["expand"]["person"] == ((entrant_id, None),)
    mass = receipt["mass"]
    assert mass["after"] > mass["before"]
    assert mass["stratum_after"]["urban"] > mass["stratum_before"]["urban"]
    assert mass["stratum_after"]["rural"] == mass["stratum_before"]["rural"]

    for mode in ("missing", "unknown_id", "labels_incumbent"):
        bad, bad_claim = toy.entrant_person_node(f"cohort_{mode}", strata_mode=mode)
        with pytest.raises(NodeRejectedError, match=f"cohort_{mode}"):
            toy.run_toy(
                toy.small_graph(nodes=(toy.CREATE, bad, bad_claim)),
                tmp_path / mode,
            )
