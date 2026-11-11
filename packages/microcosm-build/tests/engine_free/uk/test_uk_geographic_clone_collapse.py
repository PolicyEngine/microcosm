"""The FRS-trained fits fold geographic clones, and only them (microcosm#931)."""

from __future__ import annotations

import ast
import inspect
import json
from pathlib import Path

import numpy as np
import pandas as pd

from microcosm.build.uk_runtime import graph_kernels
from microcosm.build.uk_runtime.graph import _STAGE_CONSUMES, uk_spine_graph
from microcosm.build.uk_runtime.household_lineage import (
    UK_COPY_LINEAGE,
    collapse_geographic_clones,
    fold_geographic_clones,
    geographic_fold_receipt,
)
from microcosm.build.uk_runtime.national_frame import (
    UK_NATIONAL_SCHEMA,
    uk_national_frame,
)
from microcosm.build.uk_runtime.rowwise_dataset import ladder_clone_index_column
from test_support.microcosm_build.uk_clone_first_frames import clone_first_frame
from test_support.microcosm_build.uk_geographic_clones import with_geographic_clones


def _copies_frame():
    """One survey household and its SPI, high-value and CGT copies."""

    household = pd.DataFrame(
        {
            "household_id": np.arange(1, 5, dtype="int64"),
            "source_household_id": np.ones(4, dtype="int64"),
            "household_support_channel": ["frs", "spi", "frs", "frs"],
            "household_support_clone_index": np.asarray([0, 1, 0, 0], dtype="int64"),
            "household_is_high_value_support_copy": [False, False, True, False],
            "high_value_support_copy_index": np.asarray([0, 0, 1, 0], dtype="int64"),
            "household_is_cgt_support_copy": False,
            "cgt_support_copy_index": np.zeros(4, dtype="int64"),
            "household_is_capital_gains_clone": [False, False, False, True],
            "household_is_cgt_residential_clone": False,
            "cgt_residential_clone_index": np.zeros(4, dtype="int64"),
        }
    )
    person = pd.DataFrame(
        {
            "person_id": np.arange(11, 15, dtype="int64"),
            "person_household_id": household["household_id"],
            "person_benunit_id": np.arange(21, 25, dtype="int64"),
        }
    )
    benunit = pd.DataFrame({"benunit_id": np.arange(21, 25, dtype="int64")})
    return uk_national_frame(
        person=person,
        benunit=benunit,
        household=household,
        household_weights=np.asarray([2.0, 4.0, 6.0, 8.0]),
        time_period="2024",
    )


def _tables(frame):
    return {name: frame.table(name) for name in ("person", "benunit", "household")}


def _agreeing(n: int) -> pd.DataFrame:
    """Inputs every clone shares: the fit reads the same values on each."""

    return pd.DataFrame({"value": np.zeros(n)})


def test_only_geographic_clones_fold_into_their_survey_record() -> None:
    frame = _copies_frame()
    weights = frame.weights_for("household").values
    rows = np.ones(4, dtype=bool)
    # Without clones every copy is a row of its own, value for value.
    kept, summed = collapse_geographic_clones(
        _tables(frame), "household", rows, weights, inputs=_agreeing(4)
    )
    assert kept.tolist() == rows.tolist()
    assert summed.tobytes() == np.asarray(weights).tobytes()
    pool = with_geographic_clones(frame)
    pool_weights = pool.weights_for("household").values
    for entity in ("household", "person", "benunit"):
        n = pool.n(entity)
        kept, summed = collapse_geographic_clones(
            _tables(pool),
            entity,
            np.ones(n, dtype=bool),
            np.full(n, 0.5),
            inputs=_agreeing(n),
        )
        clone = pool.table(entity)[ladder_clone_index_column(entity)].to_numpy()
        # Each copy keeps its lowest-id clone, clone 0, which carries both
        # clones' weight.
        assert kept.tolist() == (clone == 0).tolist()
        assert summed[kept].tolist() == [1.0] * 4
    kept, summed = collapse_geographic_clones(
        _tables(pool),
        "household",
        np.ones(8, dtype=bool),
        pool_weights,
        inputs=_agreeing(8),
    )
    assert summed[kept].tolist() == list(weights)


def test_clones_that_differ_on_what_the_fit_reads_stay_rows_of_their_own() -> None:
    """A record's clones fold only where they agree on the fit's inputs: a
    clone that differs (a draw keyed on its own id, say) keeps its own row at
    its own weight, so nothing it carries is dropped (microcosm#931)."""

    pool = with_geographic_clones(_copies_frame())
    clone = pool.table("household")[ladder_clone_index_column("household")]
    values = np.zeros(8)
    differs = int(np.flatnonzero(clone.to_numpy() == 1)[0])
    values[differs] = 1.0
    kept, summed = collapse_geographic_clones(
        _tables(pool),
        "household",
        np.ones(8, dtype=bool),
        np.full(8, 0.5),
        inputs=pd.DataFrame({"value": values}),
    )
    # Three records fold to one row each; the fourth keeps both clones.
    assert int(kept.sum()) == 5
    assert kept[differs]
    assert sorted(summed[kept].tolist()) == [0.5, 0.5, 1.0, 1.0, 1.0]
    assert summed[differs] == 0.5
    assert np.isclose(summed[kept].sum(), 4.0)


def _reversed_tables(frame) -> dict[str, pd.DataFrame]:
    """Every table's rows in reverse order: each clone 1 before its clone 0."""

    return {
        name: table.iloc[::-1].reset_index(drop=True)
        for name, table in _tables(frame).items()
    }


def test_clones_in_reverse_row_order_keep_clone_zero() -> None:
    """A record trains on its lowest-id clone, clone 0, wherever its row
    sits: the UK entry orders the fold by the entity's ids, so a frame whose
    clone 1 rows come first folds as the clone-major one does
    (microcosm#931)."""

    tables = _reversed_tables(with_geographic_clones(_copies_frame()))
    for entity in ("household", "person", "benunit"):
        table = tables[entity]
        n = len(table)
        clone = table[ladder_clone_index_column(entity)].to_numpy()
        assert clone[0] == 1
        rows, weights = np.ones(n, dtype=bool), np.full(n, 0.5)
        kept, summed = collapse_geographic_clones(
            tables, entity, rows, weights, inputs=_agreeing(n)
        )
        assert kept.tolist() == (clone == 0).tolist(), entity
        assert summed[kept].tolist() == [1.0] * 4
        fold = fold_geographic_clones(
            tables, entity, rows, weights, inputs=_agreeing(n)
        )
        assert fold.kept.tobytes() == kept.tobytes()
        assert fold.weights.tobytes() == summed.tobytes()
        # Every clone points at its own record's clone 0, the lowest id the
        # record's clones carry.
        ids = table[UK_NATIONAL_SCHEMA.entity_id_column(entity)].to_numpy()
        assert (clone[fold.representative] == 0).all()
        assert (fold.key_group[fold.representative] == fold.key_group).all()
        for group in np.unique(fold.key_group):
            members = fold.key_group == group
            assert set(ids[fold.representative[members]]) == {ids[members].min()}
        assert not fold.varied.any()


def test_a_variant_clone_trains_on_its_own_row_in_reverse_row_order() -> None:
    """The record whose clone 1 differs keeps both rows, each its own
    variant's representative, and only its rows vary; the other records
    fold onto clone 0."""

    tables = _reversed_tables(with_geographic_clones(_copies_frame()))
    clone = tables["household"][ladder_clone_index_column("household")].to_numpy()
    values = np.zeros(8)
    differs = int(np.flatnonzero(clone == 1)[0])
    values[differs] = 1.0
    fold = fold_geographic_clones(
        tables,
        "household",
        np.ones(8, dtype=bool),
        np.full(8, 0.5),
        inputs=pd.DataFrame({"value": values}),
    )
    record = fold.key_group == fold.key_group[differs]
    assert int(record.sum()) == 2
    assert fold.kept[record].all()
    assert fold.representative[record].tolist() == np.flatnonzero(record).tolist()
    assert fold.varied.tolist() == record.tolist()
    assert fold.kept[~record].tolist() == (clone[~record] == 0).tolist()
    assert fold.weights[fold.kept & ~record].tolist() == [1.0] * 3


def _survey_frame(n: int):
    """``n`` survey households of one person in one benefit unit, no copies."""

    ids = np.arange(1, n + 1, dtype="int64")
    return uk_national_frame(
        person=pd.DataFrame(
            {
                "person_id": ids * 10,
                "person_household_id": ids,
                "person_benunit_id": ids * 10,
            }
        ),
        benunit=pd.DataFrame({"benunit_id": ids * 10}),
        household=pd.DataFrame({"household_id": ids, "source_household_id": ids}),
        household_weights=np.full(n, 3.0),
        time_period="2024",
    )


def _fold_receipt(frame, *, varied: int = 0, weightless: tuple[int, ...] = ()):
    """The receipt of a fit over every person of ``frame``.

    The clone-1 rows of the survey households whose source id is at most
    ``varied`` differ from their other clones on what the fit reads, and the
    survey households in ``weightless`` carry no weight.
    """

    tables = _tables(frame)
    person, household = tables["person"], tables["household"]
    by_household = household.set_index("household_id")
    members = person["person_household_id"]
    column = ladder_clone_index_column("household")
    clone = (
        members.map(by_household[column]).to_numpy()
        if column in by_household
        else np.zeros(len(person), dtype=np.int64)
    )
    source = members.map(by_household["source_household_id"]).to_numpy()
    household_weights = pd.Series(
        frame.weights_for("household").values, index=by_household.index
    )
    weights = np.where(
        np.isin(source, weightless),
        0.0,
        members.map(household_weights).to_numpy(dtype=float),
    )
    fold = fold_geographic_clones(
        tables,
        "person",
        np.ones(len(person), dtype=bool),
        weights,
        inputs=pd.DataFrame({"value": ((clone == 1) & (source <= varied)) * 1.0}),
    )
    return geographic_fold_receipt(tables, "person", fold, stage="The test fit")


def test_the_fold_receipt_is_absent_at_one_clone_and_rounded_at_three() -> None:
    """A fit's evidence records its geographic fold only on more than one
    clone, so a one-clone frame's receipts keep their bytes. Its counts are
    true source households of positive weight, rounded as every receipt count
    is (microcosm#931)."""

    frame = _survey_frame(25)
    assert _fold_receipt(frame) is None
    assert _fold_receipt(clone_first_frame(frame, 1)) is None
    three = clone_first_frame(frame, 3)
    # 25 sources read 30; none varies, so the share reads as its count does.
    assert _fold_receipt(three) == {
        "clone_count": 3,
        "records": 30,
        "varied_records": 0,
        "varied_share": 0,
    }
    # A source of no weight carries nothing: 24 sources read 20.
    assert _fold_receipt(three, weightless=(25,))["records"] == 20
    # A household's copies are records of their own in the fit, but its
    # counts are of sources: one survey household reads "<3" however many
    # of its copies vary.
    copies = clone_first_frame(_copies_frame(), 3)
    assert _fold_receipt(copies, varied=1) == {
        "clone_count": 3,
        "records": "<3",
        "varied_records": "<3",
        "varied_share": "<3",
    }


def test_one_and_two_varied_sources_give_the_same_receipt_bytes() -> None:
    """Under three varied source households the count and the share both
    read "<3", so frames whose records vary in one source and in two give
    the same bytes, as frames varying in three and in nine do ("3-9"). The
    share is a floored hundredth only from ten varied sources up
    (microcosm#931)."""

    three = clone_first_frame(_survey_frame(25), 3)

    def receipt_bytes(varied: int) -> bytes:
        receipt = _fold_receipt(three, varied=varied)
        return json.dumps(receipt, sort_keys=True).encode()

    assert receipt_bytes(1) == receipt_bytes(2)
    assert _fold_receipt(three, varied=1) == {
        "clone_count": 3,
        "records": 30,
        "varied_records": "<3",
        "varied_share": "<3",
    }
    assert receipt_bytes(3) == receipt_bytes(9)
    assert _fold_receipt(three, varied=9)["varied_share"] == "3-9"
    # Ten of 25 sources vary, the fewest a share is given at: the count reads
    # 10 and the share 0.4.
    assert _fold_receipt(three, varied=10) == {
        "clone_count": 3,
        "records": 30,
        "varied_records": 10,
        "varied_share": 0.4,
    }
    # Twelve of 25 sources vary: the count reads 10 and the share 0.48.
    assert _fold_receipt(three, varied=12) == {
        "clone_count": 3,
        "records": 30,
        "varied_records": 10,
        "varied_share": 0.48,
    }


def test_the_share_is_withheld_where_the_sources_that_do_not_vary_are_few() -> None:
    """The share is given only where the selected sources that do not vary
    are none or ten or more as well, so it never places them in a band: a
    frame where every source but one varies gives the bytes of one where
    every source but two does, and every source but three those of every
    source but nine. Where every source varies the share is 1
    (microcosm#931)."""

    three = clone_first_frame(_survey_frame(25), 3)

    def receipt_bytes(varied: int) -> bytes:
        receipt = _fold_receipt(three, varied=varied)
        return json.dumps(receipt, sort_keys=True).encode()

    assert receipt_bytes(24) == receipt_bytes(23)
    assert receipt_bytes(22) == receipt_bytes(16)
    # The share falls back to the varied count, rounded as every count is.
    assert _fold_receipt(three, varied=24) == {
        "clone_count": 3,
        "records": 30,
        "varied_records": 20,
        "varied_share": 20,
    }
    # Ten sources do not vary: the share is given again.
    assert _fold_receipt(three, varied=15)["varied_share"] == 0.6
    assert _fold_receipt(three, varied=25) == {
        "clone_count": 3,
        "records": 30,
        "varied_records": 30,
        "varied_share": 1.0,
    }


#: The stages whose fits fold clones, through either UK entry, and the module
#: each one's call is in: the SPI income stage runs its forests from
#: spi_income.
_COLLAPSING_STAGES = {
    "hmrc_spi_income_spine": "spi_income",
    "spi_housing_shell": "spi_housing_shell",
    "uc_reporter_redraw": "uc_reporter_redraw",
    "uc_capital_coherence": "uc_capital_coherence",
    "salary_sacrifice": "salary_sacrifice",
    "spi_employment_shell": "spi_employment_shell",
}
#: The UK entries a fit folds its clones through.
_FOLD_ENTRIES = frozenset({"collapse_geographic_clones", "fold_geographic_clones"})


def _calls_fold(path: Path) -> bool:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    return any(
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id in _FOLD_ENTRIES
        for node in ast.walk(tree)
    )


def test_every_collapsing_stage_records_its_fold() -> None:
    """Every module whose fit folds geographic clones also builds the fold's
    receipt (``geographic_fold_receipt``), which its stage's evidence gives
    on a spine of more than one clone; so a new fold site cannot leave its
    fold out of its evidence (microcosm#931)."""

    runtime = Path(inspect.getfile(graph_kernels)).parent
    for module in sorted(set(_COLLAPSING_STAGES.values())):
        tree = ast.parse((runtime / f"{module}.py").read_text(encoding="utf-8"))
        assert any(
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "geographic_fold_receipt"
            for node in ast.walk(tree)
        ), module


def test_every_collapsing_stage_sees_the_lineage_live_at_it() -> None:
    """A finite surface prunes what it does not declare, and a step whose
    columns are pruned reads as one that has not run: its copies would fold
    into their sources. So every stage that collapses clones declares every
    lineage column live where it runs; in the geography-first order that
    includes the geographic clone index (microcosm#931)."""

    runtime = Path(inspect.getfile(graph_kernels)).parent
    # The module that defines the two entries calls one from the other.
    entries = Path(inspect.getfile(fold_geographic_clones)).resolve()
    callers = {
        path.stem
        for path in runtime.glob("*.py")
        if path.resolve() != entries and _calls_fold(path)
    }
    assert callers == set(_COLLAPSING_STAGES.values())
    for stage, module in _COLLAPSING_STAGES.items():
        names = {
            graph_kernels._stage_module(stage).__name__,
            *(m.__name__ for m in graph_kernels._STAGE_HELPER_MODULES.get(stage, ())),
        }
        assert f"microcosm.build.uk_runtime.{module}" in names, stage
    lineage = {
        ("household", column)
        for column in (UK_COPY_LINEAGE.source, *UK_COPY_LINEAGE.copy_columns)
    }
    live: set[tuple[str, str]] = set()
    checked = []
    for node in uk_spine_graph().nodes:
        if node.id in _COLLAPSING_STAGES and _STAGE_CONSUMES.get(node.id) is not None:
            declared = {
                (slice_.entity, column)
                for slice_ in node.inputs
                for column in slice_.columns
            }
            missing = sorted(live & lineage - declared)
            assert not missing, (node.id, missing)
            checked.append(node.id)
        live |= {(cell.entity, cell.column) for cell in node.outputs}
    assert sorted(checked) == [
        "spi_employment_shell",
        "spi_housing_shell",
        "uc_capital_coherence",
    ]
