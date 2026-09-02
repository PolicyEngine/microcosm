#!/usr/bin/env python3
"""Regenerate the hermetic charter-H3 US post-transfer parity fixture."""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import tempfile
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from types import ModuleType

import numpy as np
import pandas as pd

import microcosm.build.us_runtime.puf_capital_gains_tail as tail_module
from microcosm.build.us_runtime.graph import us_post_transfer_graph, us_registry
from microcosm.build.us_runtime.multispine_pool import (
    derive_multispine_pool_inputs,
    materialize_multispine_agreement_outputs,
    seed_multispine_pool_inputs,
)
from microcosm.build.us_runtime.puf_support import (
    bind_puf_clone_attachment_tail_descendant,
    clone_us_frame_for_puf_support,
    validate_puf_clone_attachment,
)
from microcosm.build.us_runtime.qbi_inputs import (
    US_QBI_BOOLEAN_OUTPUT_COLUMNS,
    US_QBI_OUTPUT_COLUMNS,
)
from microcosm.build.us_runtime.stacked_spine import (
    assemble_stacked_spine,
    prepare_stacked_tail_derivation,
    validate_stacked_spine_frame,
)
from microcosm.build.us_runtime.take_up_contract import load_take_up_contract
from microcosm.frame import US_SCHEMA, Frame, MassChangeRecord
from microcosm.frame.adapters.policyengine_us import PolicyEngineUSEngine
from microcosm.graph import (
    ContentStore,
    compile_graph,
    graph_to_json,
    load_source,
    run_graph,
)
from microcosm.graph.population import dtype_for_token, token_for_dtype

_REPOSITORY = Path(__file__).resolve().parents[1]
_DEFAULT_OUTPUT = (
    _REPOSITORY
    / "packages"
    / "microcosm-graph"
    / "tests"
    / "fixtures"
    / "parity"
    / "us_post_transfer"
)
_POOL_TOOL_TEST = (
    _REPOSITORY
    / "packages"
    / "microcosm-build"
    / "tests"
    / "test_us_multispine_pool_tool.py"
)
_SAMPLE_SEED = 578
_SOURCE_HOUSEHOLDS = 100

_NORMALIZED_STRING_COLUMNS: Mapping[str, tuple[str, ...]] = {
    entity: (f"{entity}_support_channel",) for entity in US_SCHEMA.entities
}
_STRATA_SOURCE_COORDINATE = "person.__stratum__"
_EXPECTED_COLUMNS = (
    "person.schedule_d_capital_gain_distributions",
    "tax_unit.takes_up_eitc",
    "person.ssi",
)


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _json_ready(value: object) -> object:
    if isinstance(value, Mapping):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, np.ndarray):
        return [_json_ready(item) for item in value.tolist()]
    if isinstance(value, np.generic):
        return _json_ready(value.item())
    if isinstance(value, Sequence) and not isinstance(value, str | bytes | bytearray):
        return [_json_ready(item) for item in value]
    if value is None or isinstance(value, bool | int | float | str):
        return value
    raise TypeError(
        "Fixture context contains a non-JSON value: "
        f"{type(value).__module__}.{type(value).__qualname__}."
    )


def _csv_codec_float(value: float) -> str:
    """Encode a float so the generic CSV codec decodes its exact bits."""

    mantissa, separator, exponent = format(value, ".17e").partition("e")
    if not separator:
        raise ValueError(f"Cannot encode non-finite fixture float {value!r}.")
    split = mantissa.index(".") + 2
    return f"{mantissa[:split]}_{mantissa[split:]}e{exponent}"


def _write_root_csv(
    path: Path,
    table: pd.DataFrame,
    dtypes: Mapping[str, str],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(
        path,
        index=False,
        lineterminator="\n",
        float_format=_csv_codec_float,
    )
    decoded = pd.read_csv(path).astype(dict(dtypes))
    for column in table.select_dtypes(include=["floating"]).columns:
        expected = table[column].to_numpy(copy=False)
        actual = decoded[column].to_numpy(dtype=expected.dtype, copy=False)
        if expected.tobytes() != actual.tobytes():
            raise RuntimeError(
                f"Root CSV codec changed binary values in {path.name}:{column}."
            )


def _pool_tool_test_helpers() -> ModuleType:
    name = "graph_us_post_transfer_pool_tool_test_helpers"
    spec = importlib.util.spec_from_file_location(name, _POOL_TOOL_TEST)
    if spec is None or spec.loader is None:
        raise RuntimeError(
            f"Cannot load pool-tool fixture helpers from {_POOL_TOOL_TEST}."
        )
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _replace_frame(
    frame: Frame,
    tables: Mapping[str, pd.DataFrame],
    *,
    strata: pd.Series | None = None,
) -> Frame:
    return Frame(
        dict(tables),
        frame.schema,
        {entity: frame.weights_for(entity) for entity in frame.weighted_entities},
        frame.strata if strata is None else strata,
        mass_log=frame.mass_log,
        metadata=frame.metadata,
    )


def _checkpoint_source_frame(
    builder,
    *,
    measured_offset: float,
) -> Frame:
    frame = builder(
        count=_SOURCE_HOUSEHOLDS,
        measured_offset=measured_offset,
    )
    tables = {entity: frame.table(entity).copy(deep=True) for entity in frame.entities}
    # Both origins are ordinary occupied-housing rows. Supplying this on the
    # ASEC arm before assembly avoids a mixed numeric/object transport dtype.
    tables["household"]["TYPEHUGQ"] = pd.Series(
        1,
        index=tables["household"].index,
        dtype="int64",
    )
    return _replace_frame(frame, tables)


def _post_transfer_prerequisites(frame: Frame) -> Frame:
    tables = {entity: frame.table(entity).copy(deep=True) for entity in frame.entities}
    person = tables["person"]
    rows = len(person)
    if rows != 2:
        raise RuntimeError(f"Expected two sampled spine rows, found {rows}.")
    person["age"] = person["A_AGE"].astype("float64")
    person["SEMP"] = np.asarray([10.0, 20.0], dtype=np.float64)
    person["long_term_capital_gains_before_response"] = np.asarray(
        [100.0, 200.0], dtype=np.float64
    )
    person["non_sch_d_capital_gains"] = np.asarray([5.0, 10.0], dtype=np.float64)
    person["schedule_d_capital_gain_distributions"] = np.asarray(
        [7.0, np.nan], dtype=np.float64
    )
    person["self_employment_income_before_lsr"] = np.asarray(
        [10.0, 20.0], dtype=np.float64
    )
    person["non_qualified_dividend_income"] = np.asarray([1.0, 2.0], dtype=np.float64)
    for column in US_QBI_OUTPUT_COLUMNS:
        person[column] = (
            np.zeros(rows, dtype=np.bool_)
            if column in US_QBI_BOOLEAN_OUTPUT_COLUMNS
            else np.zeros(rows, dtype=np.float64)
        )
    person.loc[person.index[0], "business_is_sstb"] = True
    person.loc[person.index[0], "sstb_self_employment_income_before_lsr"] = 5.0
    person.loc[person.index[0], "qualified_bdc_income"] = 2.0
    person.loc[person.index[0], "qualified_reit_and_ptp_income"] = 3.0
    person.loc[person.index[0], "w2_wages_from_qualified_business"] = 100.0
    person.loc[person.index[0], "unadjusted_basis_qualified_property"] = 200.0

    # EITC is deliberately absent so seed owns a genuinely produced flag.
    # Every other contract cell is a dense incumbent and therefore exercises
    # amendment-8 rewrite projection and byte preservation.
    for program in load_take_up_contract().programs:
        if program.variable == "takes_up_eitc":
            continue
        table = tables[program.entity]
        table[program.variable] = np.resize(
            np.asarray([True, False], dtype=np.bool_),
            len(table),
        )
    return _replace_frame(frame, tables)


def _attach_tail_descendant(frame: Frame) -> Frame:
    cloned = clone_us_frame_for_puf_support(frame)
    attachment = validate_puf_clone_attachment(
        cloned,
        boundary="H3 fixture clone attachment",
        expected_fraction=1.0,
        expected_seed=0,
    )
    tax_unit = cloned.table("tax_unit")
    clone_one = tax_unit["tax_unit_support_clone_index"].eq(1)
    tax_unit_row = tax_unit.loc[clone_one].iloc[0]
    person = cloned.table("person")
    person_row = person.loc[
        person["person_tax_unit_id"].eq(int(tax_unit_row["tax_unit_id"]))
    ].iloc[0]
    household_id = int(person_row["person_household_id"])
    household = cloned.table("household")
    household_position = int(
        household.index[household["household_id"].eq(household_id)][0]
    )
    assignment: dict[str, object] = {
        "recipient_household_id": household_id,
        "recipient_tax_unit_id": int(tax_unit_row["tax_unit_id"]),
        "donor_source_id": 999,
        "filing_status_code": 1,
        tail_module._TAIL_SYNTHETIC_COLUMN: False,
        tail_module._TAIL_AGI_BAND_INDEX_COLUMN: 0,
        "assigned_weight": float(
            cloned.weights_for("household").values[household_position] / 2.0
        ),
    }
    for column, value in zip(
        tail_module.PUF_CAPITAL_GAINS_TAIL_PERSON_COLUMNS,
        (1.0, 500.0, 2.0, 3.0),
        strict=True,
    ):
        assignment[column] = value
    for column in tail_module.PUF_CAPITAL_GAINS_TAIL_TAX_UNIT_COLUMNS:
        assignment[column] = 4.0
    tailed, _receipt = tail_module._clone_and_transfer(
        cloned,
        pd.DataFrame([assignment]),
    )
    tail_household = tailed.table("household")
    source_id = int(
        tail_household.loc[
            tail_household["household_support_clone_index"].eq(2),
            "household_source_id",
        ].iloc[0]
    )
    return bind_puf_clone_attachment_tail_descendant(
        tailed,
        attachment_receipt=attachment,
        tail_manifest={
            "stage": "h3_fixture_tail",
            "manifest_sha256": "0" * 64,
            "records": [{"recipient_household_source_id": source_id}],
        },
    )


def _normalize_legacy_strings(frame: Frame) -> Frame:
    tables = {entity: frame.table(entity).copy(deep=True) for entity in frame.entities}
    audited = {
        (entity, column)
        for entity, columns in _NORMALIZED_STRING_COLUMNS.items()
        for column in columns
    }
    observed = {
        (entity, column)
        for entity, table in tables.items()
        for column in table.columns
        if table[column].dtype == object
        and not table[column].dropna().empty
        and table[column].dropna().map(lambda value: isinstance(value, str)).all()
    }
    if observed != audited:
        raise RuntimeError(
            "Legacy object-string audit drifted: "
            f"expected={sorted(audited)}, observed={sorted(observed)}."
        )
    if (
        frame.strata.dtype != object
        or not frame.strata.map(lambda value: isinstance(value, str)).all()
    ):
        raise RuntimeError("Legacy strata is no longer an object-string Series.")
    string_dtype = pd.StringDtype(storage="python")
    for entity, columns in _NORMALIZED_STRING_COLUMNS.items():
        for column in columns:
            tables[entity][column] = tables[entity][column].astype(string_dtype)
    return _replace_frame(frame, tables, strata=frame.strata.astype(string_dtype))


def _synthetic_stacked_pool() -> Frame:
    helpers = _pool_tool_test_helpers()
    builder = helpers._many_household_source_frame
    with pd.option_context("future.infer_string", False):
        asec = _checkpoint_source_frame(builder, measured_offset=0.0)
        acs = _checkpoint_source_frame(builder, measured_offset=1_000.0)
        assembled = assemble_stacked_spine(
            asec,
            acs,
            sample_fraction=0.01,
            sample_seed=_SAMPLE_SEED,
        ).frame
        prepared = _post_transfer_prerequisites(assembled)
        tailed = _attach_tail_descendant(prepared)
        normalized = _normalize_legacy_strings(tailed)
    validate_stacked_spine_frame(normalized, boundary="H3 fixture serialized root")
    return normalized


def _frame_document(frame: Frame) -> dict[str, object]:
    return {
        "schema_version": 1,
        "columns": {
            entity: list(frame.table(entity).columns) for entity in frame.entities
        },
        "metadata": _json_ready(frame.metadata),
        "mass_log": [
            {
                "entity": record.entity,
                "old_total": record.old_total,
                "new_total": record.new_total,
                "declared_factor": record.declared_factor,
                "reason": record.reason,
            }
            for record in frame.mass_log
        ],
    }


def _source_dtype(token: str) -> str:
    return "string[python]" if token == "string" else token


def _write_root_source(root: Path, frame: Frame) -> None:
    graph = us_post_transfer_graph()
    declared = {
        (owned.entity, owned.column): owned.dtype
        for owned in graph.node("create_stacked_pool").outputs
    }
    structural = {
        (entity, frame.schema.entity_id_column(entity)) for entity in frame.entities
    }
    structural.update(
        (frame.schema.person_entity, frame.schema.membership_column(group))
        for group in frame.schema.group_entities
    )
    observed = {
        (entity, column)
        for entity in frame.entities
        for column in frame.table(entity).columns
        if (entity, column) not in structural
    }
    if observed != set(declared):
        raise RuntimeError(
            "H3 source cells differ from CREATE ownership: "
            f"missing={sorted(set(declared) - observed)}, "
            f"extra={sorted(observed - set(declared))}."
        )

    dtypes: dict[str, dict[str, str]] = {}
    for entity in frame.entities:
        table = frame.table(entity).copy(deep=True)
        entity_dtypes: dict[str, str] = {}
        for column in table.columns:
            token = declared.get((entity, column), "int64")
            if token_for_dtype(table[column].dtype) != token:
                raise RuntimeError(
                    f"H3 source dtype for {entity}.{column} is "
                    f"{table[column].dtype}, not {token}."
                )
            entity_dtypes[column] = _source_dtype(token)
        if entity == frame.schema.person_entity:
            table["__stratum__"] = frame.strata.array
            entity_dtypes["__stratum__"] = "string[python]"
        dtypes[entity] = entity_dtypes
        _write_root_csv(root / f"{entity}.csv", table, entity_dtypes)

    _write_json(
        root / "schema.json",
        {
            "schema": {
                "person_entity": frame.schema.person_entity,
                "group_entities": list(frame.schema.group_entities),
                "links": [],
            },
            "dtypes": dtypes,
            "strata_column": "__stratum__",
            "weights": {
                "household": {
                    "kind": frame.weights_for("household").kind.value,
                }
            },
        },
    )
    _write_json(
        root / "weights.json",
        {
            "weights": {
                "household": {"values": frame.weights_for("household").values.tolist()}
            }
        },
    )
    _write_json(root / "frame.json", _frame_document(frame))


def _reload_root_source(root: Path) -> Frame:
    loaded = load_source("csv-tables", root)
    graph = us_post_transfer_graph()
    tables = {
        entity: loaded.table(entity).copy(deep=True) for entity in loaded.entities
    }
    for owned in graph.node("create_stacked_pool").outputs:
        tables[owned.entity][owned.column] = tables[owned.entity][owned.column].astype(
            dtype_for_token(owned.dtype)
        )
    document = json.loads((root / "frame.json").read_text(encoding="utf-8"))
    mass_log = tuple(
        MassChangeRecord(
            entity=record["entity"],
            old_total=record["old_total"],
            new_total=record["new_total"],
            declared_factor=record["declared_factor"],
            reason=record["reason"],
        )
        for record in document["mass_log"]
    )
    frame = Frame(
        tables,
        US_SCHEMA,
        {"household": loaded.weights_for("household")},
        loaded.strata.copy(deep=True),
        mass_log=mass_log,
        metadata=document["metadata"],
    )
    validate_stacked_spine_frame(frame, boundary="H3 fixture codec reload")
    return frame


def _assert_root_round_trip(expected: Frame, actual: Frame) -> None:
    for entity in expected.entities:
        pd.testing.assert_frame_equal(
            expected.table(entity),
            actual.table(entity),
            check_column_type=False,
        )
    pd.testing.assert_series_equal(
        expected.strata,
        actual.strata,
        check_names=False,
    )
    np.testing.assert_array_equal(
        expected.weights_for("household").values,
        actual.weights_for("household").values,
    )
    if expected.mass_log != actual.mass_log:
        raise RuntimeError("H3 source codec changed the Frame mass log.")
    if _json_ready(expected.metadata) != _json_ready(actual.metadata):
        raise RuntimeError("H3 source codec changed the Frame metadata.")


def _run_oracle(frame: Frame, engine: object) -> tuple[Frame, dict[str, object]]:
    source_schedule = frame.table("person")["schedule_d_capital_gain_distributions"]
    prepared, prepare_receipt = prepare_stacked_tail_derivation(frame)
    if int(prepare_receipt["previously_observed_rows"]) < 1:
        raise RuntimeError("The H3 fixture did not exercise prepare's rewrite.")
    clone_two = frame.table("person")["person_support_clone_index"].eq(2)
    retained = ~clone_two & source_schedule.notna()
    if (
        source_schedule.loc[retained].to_numpy().tobytes()
        != prepared.table("person")
        .loc[retained, "schedule_d_capital_gain_distributions"]
        .to_numpy()
        .tobytes()
    ):
        raise RuntimeError("Prepare changed a retained Schedule-D incumbent.")

    derived = derive_multispine_pool_inputs(prepared)
    prepared_schedule = prepared.table("person")[
        "schedule_d_capital_gain_distributions"
    ]
    observed = prepared_schedule.notna()
    if (
        prepared_schedule.loc[observed].to_numpy().tobytes()
        != derived.frame.table("person")
        .loc[observed, "schedule_d_capital_gain_distributions"]
        .to_numpy()
        .tobytes()
    ):
        raise RuntimeError("Derive changed a non-null Schedule-D incumbent.")

    before_seed = {
        (program.entity, program.variable): derived.frame.table(program.entity)[
            program.variable
        ].copy(deep=True)
        for program in load_take_up_contract().programs
        if program.variable != "takes_up_eitc"
    }
    seeded = seed_multispine_pool_inputs(derived.frame, engine=engine)
    for coordinate, before in before_seed.items():
        entity, column = coordinate
        after = seeded.frame.table(entity)[column]
        if before.dtype != after.dtype or (
            before.to_numpy().tobytes() != after.to_numpy().tobytes()
        ):
            raise RuntimeError(f"Seed changed incumbent {entity}.{column} bytes.")
    materialized = materialize_multispine_agreement_outputs(
        seeded.frame,
        engine=engine,
    )
    return materialized.frame, {
        "prepare": prepare_receipt,
        "derive": derived.receipt,
        "seed": seeded.receipt,
        "materialize": materialized.receipt,
    }


def _expected_table(frame: Frame) -> pd.DataFrame:
    row_counts = {entity: len(frame.table(entity)) for entity in frame.entities}
    if len(set(row_counts.values())) != 1:
        raise RuntimeError(
            "H3 expected.csv requires its selected entity grains to align: "
            f"{row_counts}."
        )
    columns: dict[str, np.ndarray] = {}
    for label in _EXPECTED_COLUMNS:
        entity, column = label.split(".", 1)
        columns[label] = frame.table(entity)[column].to_numpy(copy=True)
    return pd.DataFrame(columns)


def _write_expected(path: Path, expected: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    expected.to_csv(
        path,
        index=False,
        lineterminator="\n",
        float_format=lambda value: repr(float(value)),
    )
    decoded = pd.read_csv(path, float_precision="round_trip")
    for column in expected:
        if decoded[column].dtype != expected[column].dtype or (
            decoded[column].to_numpy().tobytes()
            != expected[column].to_numpy().tobytes()
        ):
            raise RuntimeError(f"expected.csv changed bytes for {column}.")


def _normalization_markdown() -> str:
    lines = [
        "# US post-transfer parity string normalization",
        "",
        "The pool-tool synthetic helper and unchanged stacked assembly retain the",
        "following textual cells as pandas `object` under legacy string inference.",
        "Interface amendment 10 requires graph strings to use",
        '`StringDtype(storage="python")`. The generator casts exactly this audited',
        "surface before serializing the common source used by both the oracle and",
        "graph. No values, row order, column order, weights, mass records, receipts,",
        "or metadata change.",
        "",
    ]
    for entity, columns in _NORMALIZED_STRING_COLUMNS.items():
        lines.append(f"- `{entity}`: " + ", ".join(f"`{name}`" for name in columns))
    lines.extend((f"- source strata: `{_STRATA_SOURCE_COORDINATE}`", ""))
    return "\n".join(lines)


def _verify_graph(output: Path, expected: pd.DataFrame, engine: object) -> None:
    graph = us_post_transfer_graph()
    compiled = compile_graph(graph)
    with tempfile.TemporaryDirectory(prefix="microcosm-h3-") as temporary:
        manifest = run_graph(
            compiled,
            sources={"stacked": output / "sources"},
            store=ContentStore(Path(temporary) / "store"),
            kernels=us_registry(engine=engine),
            resume="forbid",
            decisions=(),
        )
    final = manifest.population(compiled.versions[compiled.order[-1]])
    for label in expected:
        entity, column = label.split(".", 1)
        actual = final.table(entity)[column]
        if actual.dtype != expected[label].dtype or (
            actual.to_numpy().tobytes() != expected[label].to_numpy().tobytes()
        ):
            raise RuntimeError(f"Graph parity failed for {label}.")


def generate(output: Path) -> None:
    """Write all deterministic H3 artifacts beneath `output`."""

    output.mkdir(parents=True, exist_ok=True)
    sources = output / "sources"
    sources.mkdir(parents=True, exist_ok=True)

    root = _synthetic_stacked_pool()
    _write_root_source(sources, root)
    reloaded = _reload_root_source(sources)
    _assert_root_round_trip(root, reloaded)

    engine = PolicyEngineUSEngine()
    oracle, _receipts = _run_oracle(reloaded, engine)
    expected = _expected_table(oracle)
    _write_expected(output / "expected.csv", expected)

    graph = us_post_transfer_graph()
    (output / "us_post_transfer.json").write_text(
        graph_to_json(graph) + "\n",
        encoding="utf-8",
    )
    (output / "NORMALIZATION.md").write_text(
        _normalization_markdown(),
        encoding="utf-8",
    )
    (output / "PRODUCED_BY.txt").write_text(
        "tools/graph_us_post_transfer_fixture.py; unchanged prepare/derive/seed/"
        "materialize oracle over the pool-tool synthetic stacked source.\n",
        encoding="utf-8",
    )
    _verify_graph(output, expected, engine)


def _parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=_DEFAULT_OUTPUT)
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> int:
    args = _parse_args(argv)
    output = args.output.resolve()
    generate(output)
    size = sum(path.stat().st_size for path in output.rglob("*") if path.is_file())
    print(f"Wrote H3 fixture to {output} ({size} bytes).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
