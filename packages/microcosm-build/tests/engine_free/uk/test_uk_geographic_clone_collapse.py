"""The FRS-trained fits fold geographic clones, and only them (microcosm#931)."""

from __future__ import annotations

import ast
import inspect
from pathlib import Path

import numpy as np
import pandas as pd

from microcosm.build.uk_runtime import graph_kernels
from microcosm.build.uk_runtime.graph import _STAGE_CONSUMES, uk_spine_graph
from microcosm.build.uk_runtime.household_lineage import (
    UK_COPY_LINEAGE,
    collapse_geographic_clones,
)
from microcosm.build.uk_runtime.national_frame import uk_national_frame
from microcosm.build.uk_runtime.rowwise_dataset import ladder_clone_index_column
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


def test_only_geographic_clones_fold_into_their_survey_record() -> None:
    frame = _copies_frame()
    weights = frame.weights_for("household").values
    rows = np.ones(4, dtype=bool)
    # Without clones every copy is a row of its own, value for value.
    kept, summed = collapse_geographic_clones(
        _tables(frame), "household", rows, weights
    )
    assert kept.tolist() == rows.tolist()
    assert summed.tobytes() == np.asarray(weights).tobytes()
    pool = with_geographic_clones(frame)
    pool_weights = pool.weights_for("household").values
    for entity in ("household", "person", "benunit"):
        n = pool.n(entity)
        kept, summed = collapse_geographic_clones(
            _tables(pool), entity, np.ones(n, dtype=bool), np.full(n, 0.5)
        )
        clone = pool.table(entity)[ladder_clone_index_column(entity)].to_numpy()
        # Each copy keeps its first clone, which carries both clones' weight.
        assert kept.tolist() == (clone == 0).tolist()
        assert summed[kept].tolist() == [1.0] * 4
    kept, summed = collapse_geographic_clones(
        _tables(pool), "household", np.ones(8, dtype=bool), pool_weights
    )
    assert summed[kept].tolist() == list(weights)


#: The stages whose fits collapse clones, and the module each one's call is
#: in: the SPI income stage runs its forests from spi_income.
_COLLAPSING_STAGES = {
    "hmrc_spi_income_spine": "spi_income",
    "spi_housing_shell": "spi_housing_shell",
    "uc_reporter_redraw": "uc_reporter_redraw",
    "uc_capital_coherence": "uc_capital_coherence",
    "salary_sacrifice": "salary_sacrifice",
}


def _calls_collapse(path: Path) -> bool:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    return any(
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "collapse_geographic_clones"
        for node in ast.walk(tree)
    )


def test_every_collapsing_stage_sees_the_lineage_live_at_it() -> None:
    """A finite surface prunes what it does not declare, and a step whose
    columns are pruned reads as one that has not run: its copies would fold
    into their sources. So every stage that collapses clones declares every
    lineage column live where it runs; in the geography-first order that
    includes the geographic clone index (microcosm#931)."""

    runtime = Path(inspect.getfile(graph_kernels)).parent
    callers = {path.stem for path in runtime.glob("*.py") if _calls_collapse(path)}
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
    assert sorted(checked) == ["spi_housing_shell", "uc_capital_coherence"]
