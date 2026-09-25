"""The US release tool's stored-input gate and its register (microcosm#1026).

``microcosm.data.stored_inputs`` owns the rule and the reviewed register of
stored columns that are deliberately not policyengine-us variables. The data
shard depends on no other Microcosm shard, so it spells the register's names
itself; this file binds every entry to the producer definition it names. It
also pins the release tool's use of the rule: the batched pre-export gate over
the export frame's modeled stored tables, that model against the real writer
(``requires_us``), and the post-write check that the written H5 earns the
gate's verdict.
"""

from __future__ import annotations

import ast
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from microcosm.build.outer_stage_runtime import _POOLED_SOURCE_PROVENANCE_COLUMNS
from microcosm.build.us_runtime import puf_capital_gains_tail as puf_tail
from microcosm.build.us_runtime.support_provenance import (
    support_channel_column,
    support_clone_index_column,
    support_source_id_column,
)
from microcosm.data.stored_inputs import (
    _US_ENTITIES,
    US_STORED_NON_VARIABLE_COLUMNS,
    CertifiedEngine,
    h5_stored_tables,
)
from microcosm.frame import Frame, WeightKind, Weights
from microcosm.frame.units import (
    _TAX_UNIT_ROLE_COLUMN,
    TAX_UNIT_FILING_STATUS_COLUMN,
    US_SCHEMA,
)


def _load_builder_module():
    root = Path(__file__).resolve().parents[3]
    path = root / "tools" / "build_us_fiscal_refresh_release.py"
    spec = importlib.util.spec_from_file_location(
        "build_us_fiscal_refresh_release", path
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def builder():
    return _load_builder_module()


def _engine(*variables: str) -> CertifiedEngine:
    return CertifiedEngine(
        label="policyengine-us 2.2.1", variables=frozenset(variables)
    )


_STRUCTURAL = (
    "person_id",
    "person_household_id",
    "person_tax_unit_id",
    "person_spm_unit_id",
    "person_family_id",
    "person_marital_unit_id",
    "household_id",
    "tax_unit_id",
    "spm_unit_id",
    "family_id",
    "marital_unit_id",
    "household_weight",
    "age",
    "takes_up_wic_if_eligible",
)


def _frame(**person_columns) -> Frame:
    """Two households, one person each, every US entity populated."""

    ids = np.asarray([1, 2], dtype=np.int64)
    person = pd.DataFrame(
        {
            "person_id": ids,
            "person_household_id": ids,
            "person_tax_unit_id": ids * 10,
            "person_spm_unit_id": ids * 100,
            "person_family_id": ids * 1_000,
            "person_marital_unit_id": ids * 10_000,
            "age": np.asarray([40, 70], dtype=np.int64),
            **person_columns,
        }
    )
    return Frame(
        {
            "person": person,
            "household": pd.DataFrame({"household_id": ids}),
            "tax_unit": pd.DataFrame({"tax_unit_id": ids * 10}),
            "spm_unit": pd.DataFrame({"spm_unit_id": ids * 100}),
            "family": pd.DataFrame({"family_id": ids * 1_000}),
            "marital_unit": pd.DataFrame({"marital_unit_id": ids * 10_000}),
        },
        US_SCHEMA,
        {"household": Weights(np.asarray([1_500.0, 2_500.0]), WeightKind.CALIBRATED)},
    )


# ---------------------------------------------------------------------------
# The register is bound to the producers it names
# ---------------------------------------------------------------------------


def test_the_register_spells_the_us_entities_in_schema_order():
    assert _US_ENTITIES == tuple(US_SCHEMA.entities)


def test_every_register_entry_is_a_named_producer_column():
    """The register is exactly the support provenance columns of every US
    entity, the pooled-source provenance columns, microunit's two tax-unit
    construction columns and the PUF capital-gains tail provenance columns."""

    support = {
        column(entity)
        for entity in US_SCHEMA.entities
        for column in (
            support_source_id_column,
            support_channel_column,
            support_clone_index_column,
        )
    }
    tail = {
        puf_tail.PUF_CAPITAL_GAINS_TAIL_APPLIED_COLUMN,
        puf_tail.PUF_CAPITAL_GAINS_TAIL_DONOR_SOURCE_ID_COLUMN,
        puf_tail.PUF_CAPITAL_GAINS_TAIL_DONOR_SYNTHETIC_COLUMN,
        puf_tail.PUF_CAPITAL_GAINS_TAIL_DONOR_FILING_STATUS_COLUMN,
        puf_tail.PUF_CAPITAL_GAINS_TAIL_DONOR_AGI_BAND_COLUMN,
        puf_tail.PUF_CAPITAL_GAINS_TAIL_TRANSFER_WEIGHT_COLUMN,
    }
    construction = {_TAX_UNIT_ROLE_COLUMN, TAX_UNIT_FILING_STATUS_COLUMN}

    assert set(US_STORED_NON_VARIABLE_COLUMNS) == (
        support | set(_POOLED_SOURCE_PROVENANCE_COLUMNS) | construction | tail
    )
    assert len(US_STORED_NON_VARIABLE_COLUMNS) == 18 + 4 + 2 + 6


def test_every_register_reason_names_where_its_column_comes_from():
    expected_owner = {
        **{
            column(entity): "microcosm.build.us_runtime.support_provenance"
            for entity in US_SCHEMA.entities
            for column in (
                support_source_id_column,
                support_channel_column,
                support_clone_index_column,
            )
        },
        **{
            column: "_POOLED_SOURCE_PROVENANCE_COLUMNS"
            for column in _POOLED_SOURCE_PROVENANCE_COLUMNS
        },
        _TAX_UNIT_ROLE_COLUMN: "microcosm.frame.units._TAX_UNIT_ROLE_COLUMN",
        TAX_UNIT_FILING_STATUS_COLUMN: (
            "microcosm.frame.units.TAX_UNIT_FILING_STATUS_COLUMN"
        ),
    }
    for column, owner in expected_owner.items():
        assert owner in US_STORED_NON_VARIABLE_COLUMNS[column], column
    for column in US_STORED_NON_VARIABLE_COLUMNS:
        if column.startswith("puf_capital_gains_tail_"):
            assert (
                "microcosm.build.us_runtime.puf_capital_gains_tail"
                in US_STORED_NON_VARIABLE_COLUMNS[column]
            )


# ---------------------------------------------------------------------------
# The release tool's gate
# ---------------------------------------------------------------------------


def test_the_modeled_stored_tables_add_only_the_household_weight(builder):
    stored = builder._export_stored_tables(_frame())

    assert stored["household"] == ("household_id", "household_weight")
    assert stored["person"][-1] == "age"
    assert set(stored) == set(US_SCHEMA.entities)
    assert all(
        stored[entity] == tuple(_frame().table(entity).columns)
        for entity in US_SCHEMA.entities
        if entity != "household"
    )


def test_the_frame_hdf_boundary_fallback_is_read_from_metadata(tmp_path):
    """``put_frame_table`` falls back to pandas' fixed format for a nullable
    Boolean with missing values; the check reads that layout's columns too."""

    pytest.importorskip("tables")
    from microcosm.frame.materialize import put_frame_table

    path = tmp_path / "populace_us_2024.h5"
    spm_unit = pd.DataFrame(
        {
            "spm_unit_id": [1, 2],
            "stale_flag": pd.array([True, None], dtype="boolean"),
        }
    )
    person = pd.DataFrame({"person_id": [1, 2]}, index=pd.Index([5, 6], name="x"))
    with pd.HDFStore(path, mode="w") as store:
        put_frame_table(
            store, "spm_unit", spm_unit, preferred_format="table", data_columns=True
        )
        put_frame_table(
            store, "person", person, preferred_format="table", data_columns=True
        )

    assert h5_stored_tables(path) == {
        "person": ("person_id",),
        "spm_unit": ("spm_unit_id", "stale_flag"),
    }


def test_the_modeled_stored_tables_skip_an_empty_table_like_the_writer(builder):
    tables = {
        "person": pd.DataFrame({"person_id": [1]}),
        "household": pd.DataFrame({"household_id": [1], "household_weight": [2.0]}),
        "family": pd.DataFrame({"family_id": pd.Series([], dtype="int64")}),
    }
    frame = SimpleNamespace(entities=tuple(tables), table=tables.__getitem__)

    assert builder._export_stored_tables(frame) == {
        "person": ("person_id",),
        "household": ("household_id", "household_weight"),
    }


def test_the_gate_refuses_would_claim_wic_by_name(builder, monkeypatch):
    engine = _engine(*_STRUCTURAL)
    monkeypatch.setattr(builder, "installed_us_engine", lambda: engine)

    failures, details = builder._stored_input_gate_failures(
        _frame(would_claim_wic=[True, False], person_source_id=[7, 8]),
        stage="export frame",
    )

    assert len(failures) == 1
    assert failures[0].startswith(
        "Stored model inputs failed (export frame): stored column "
        "'would_claim_wic' (person table)"
    )
    assert details["evaluated"] is True
    assert details["refused"] == ["would_claim_wic"]
    assert details["registered_non_variables"] == ["person_source_id"]
    assert details["engine"] == "policyengine-us 2.2.1"
    assert len(details["register_sha256"]) == 64


def test_the_gate_passes_the_live_name(builder, monkeypatch):
    monkeypatch.setattr(builder, "installed_us_engine", lambda: _engine(*_STRUCTURAL))

    failures, details = builder._stored_input_gate_failures(
        _frame(takes_up_wic_if_eligible=[True, False]), stage="export frame"
    )

    assert failures == []
    assert details["refused"] == []


def test_the_gate_grades_the_materialized_household_weight(builder, monkeypatch):
    """The gate grades what the writer stores, not only the frame's columns:
    an engine without ``household_weight`` refuses the column the writer adds."""

    engine = _engine(*(name for name in _STRUCTURAL if name != "household_weight"))
    monkeypatch.setattr(builder, "installed_us_engine", lambda: engine)

    failures, _ = builder._stored_input_gate_failures(_frame(), stage="export frame")

    assert [line.split("'")[1] for line in failures] == ["household_weight"]


def test_the_gate_fails_closed_without_an_engine(builder, monkeypatch):
    def missing():
        raise ImportError("No module named 'policyengine_us'")

    monkeypatch.setattr(builder, "installed_us_engine", missing)

    failures, details = builder._stored_input_gate_failures(
        _frame(), stage="export frame"
    )

    assert len(failures) == 1
    assert "cannot be imported" in failures[0]
    assert details["evaluated"] is False


def test_the_post_write_check_compares_refused_sets(builder, monkeypatch, tmp_path):
    engine = _engine(*_STRUCTURAL)
    monkeypatch.setattr(builder, "installed_us_engine", lambda: engine)
    written = {"person": ("person_id", "would_claim_wic")}
    monkeypatch.setattr(builder, "h5_stored_tables", lambda path: written)
    path = tmp_path / "populace_us_2024.h5"

    assert (
        builder._written_stored_input_verdict_mismatch(
            path, {"evaluated": True, "refused": ["would_claim_wic"]}
        )
        is None
    )
    mismatch = builder._written_stored_input_verdict_mismatch(
        path, {"evaluated": True, "refused": []}
    )
    assert mismatch is not None
    assert "would_claim_wic" in mismatch
    assert mismatch.startswith("Stored-input gate premise failed")
    # A premise failure is not a gate verdict: evidence mode cannot convert it
    # (test_evidence_mode_conversion_is_pinned_structurally keys on this text).
    assert "Release gates failed" not in mismatch
    assert (
        builder._written_stored_input_verdict_mismatch(
            path, {"evaluated": False, "error": "no engine"}
        )
        is None
    )


def test_main_runs_the_gate_in_the_batch_and_checks_the_written_h5(builder):
    """In ``_main``, the gate's failures join the batched pre-export raise
    before the H5 is written, and the written H5 is checked before any
    post-export stage scores it."""

    source = Path(builder.__file__).read_text()
    tree = ast.parse(source)
    main = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == "_main"
    )
    body = ast.get_source_segment(source, main)
    assert body is not None

    gate = body.index("_stored_input_gate_failures(")
    joined = body.index("terminal_gate_failures.extend(stored_input_failures_)")
    batched_raise = body.index('"Release gates failed: " + "; ".join(terminal_gate')
    write = body.index("release_engine.write_dataset(export_frame, dataset_path")
    premise = body.index("_written_stored_input_verdict_mismatch(")
    scorer = body.index("_open_post_export_scorer(")

    assert body.count("_stored_input_gate_failures(") == 1
    assert body.count("_written_stored_input_verdict_mismatch(") == 1
    assert gate < joined < batched_raise < write < premise < scorer


# ---------------------------------------------------------------------------
# Against the installed engine and the real writer
# ---------------------------------------------------------------------------


@pytest.mark.requires_us
def test_the_modeled_stored_tables_are_what_the_writer_stores(builder, tmp_path):
    """Differential: the gate's model of the export against the H5 the real
    writer produces from the same frame, register columns and a stale input
    included; the gate and the written bytes reach the same verdict."""

    from microcosm.frame.adapters.policyengine_us import PolicyEngineUSEngine

    frame = _frame(
        would_claim_wic=[True, False],
        person_source_id=[11, 12],
        person_support_channel=["asec", "asec"],
        source_year=[2024, 2024],
        tax_unit_role_input=["HEAD", "HEAD"],
    )
    path = tmp_path / "populace_us_2024.h5"
    PolicyEngineUSEngine().write_dataset(frame, path, period=builder.PERIOD)

    written = h5_stored_tables(path)
    modeled = builder._export_stored_tables(frame)
    assert {table: set(columns) for table, columns in written.items()} == {
        table: set(columns) for table, columns in modeled.items()
    }

    failures, details = builder._stored_input_gate_failures(frame, stage="export frame")
    assert details["refused"] == ["would_claim_wic"]
    assert len(failures) == 1
    assert builder._written_stored_input_verdict_mismatch(path, details) is None


@pytest.mark.requires_us
def test_the_writer_refuses_the_engine_names_outside_the_convention(builder, tmp_path):
    """policyengine-us's only variables outside the lowercase convention are
    state-code formulas; the writer refuses to store a formula-owned column,
    so an uppercase stored column can never be one of them."""

    from microcosm.frame.adapters.policyengine_us import PolicyEngineUSEngine

    frame = _frame()
    frame = Frame(
        {
            **{entity: frame.table(entity) for entity in frame.entities},
            "household": frame.table("household").assign(CA=[True, False]),
        },
        US_SCHEMA,
        {"household": frame.weights_for("household")},
    )

    with pytest.raises(
        ValueError, match="formula-owned column\\(s\\) present: \\['CA'\\]"
    ):
        PolicyEngineUSEngine().write_dataset(
            frame, tmp_path / "populace_us_2024.h5", period=builder.PERIOD
        )
