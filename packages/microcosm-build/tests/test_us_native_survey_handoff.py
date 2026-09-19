"""Invented storage/coverage checks; never issue a substitute native owner."""

import json
from types import SimpleNamespace

import numpy as np
import pytest
from test_us_common_frame_export_contract import _parent

from microcosm.build.us_runtime import native_survey_handoff as handoff
from microcosm.frame.rules import ExportContract


def test_coverage_names_missing_leaves_and_excludes_prior_year_income():
    frame = _parent()
    coverage = handoff.native_survey_input_inventory(frame)
    by_name = {(row["entity"], row["variable"]): row for row in coverage}
    assert by_name["person", "weekly_hours_worked_before_lsr"]["status"] == "missing"
    assert (
        by_name["person", "self_employment_income_last_year"]["status"]
        == "excluded_native_scope"
    )
    assert (
        by_name["person", "previous_year_income_available"]["status"]
        == "excluded_native_scope"
    )
    assert frame.n("person") == 6
    assert "employment_income_last_year" not in frame.person


def test_coverage_does_not_infer_signal_or_applicability_from_presence():
    frame = _parent()
    frame.person["weekly_hours_worked_before_lsr"] = [
        0.0,
        10.0,
        np.nan,
        30.0,
        20.0,
        0.0,
    ]
    row = next(
        r
        for r in handoff.native_survey_input_inventory(frame)
        if r["variable"] == "weekly_hours_worked_before_lsr"
    )
    assert row["status"] == "contains_unknowns"
    assert row["missing_values"] == 1
    assert row["source_signal_verified"] is False
    assert row["applicability_verified"] is False


def test_engine_contract_keeps_unsupported_projection_explicit():
    frame = _parent()
    contract = ExportContract(
        required=("absent_engine_input",),
        forbidden=("money",),
        optional=(),
        formula_owned_excluded=("hours",),
        closed=True,
    )
    inventory = handoff.engine_export_inventory(frame, contract)
    assert inventory["missing_required"] == ["absent_engine_input"]
    assert inventory["forbidden_present"] == ["money"]
    assert inventory["formula_owned_present"] == ["hours"]
    assert "origin" in inventory["unexpected_columns"]
    assert frame.person.money.tolist() == [-0.0, 2.0, 3.0, 4.0, 5.0, 6.0]


def test_fake_owner_and_descriptive_checked_view_cannot_export(tmp_path):
    for fake in (
        SimpleNamespace(population=SimpleNamespace(frame=_parent())),
        object(),
    ):
        with pytest.raises(ValueError, match="UNISSUED_RUN"):
            handoff.write_native_survey_development_checkpoint(
                fake, tmp_path / "refused"
            )
        assert not (tmp_path / "refused").exists()


def test_invented_checkpoint_readback_preserves_full_frame(tmp_path):
    frame = _parent()
    report = {
        "protocol": handoff.PROTOCOL,
        "release_eligible": False,
        "invented_storage_test": True,
    }
    root = tmp_path / "checkpoint"
    # Test the storage primitive, not the public native-owner exporter.
    handoff._write_checkpoint(frame, report, root)
    loaded = handoff.load_native_survey_development_checkpoint(root)
    handoff.same_replayed_frame(frame, loaded.frame)
    assert loaded.owner_live_verified is False
    assert loaded.report == report
    assert loaded.frame.schema == frame.schema
    assert (
        loaded.frame.weights_for("household").kind
        == frame.weights_for("household").kind
    )
    assert loaded.frame.metadata == frame.metadata
    assert (
        loaded.frame.person.reported.isna().tolist()
        == frame.person.reported.isna().tolist()
    )
    assert np.signbit(loaded.frame.person.money.iloc[0])
    assert loaded.frame.table("household").assigned_block.tolist() == [
        "001",
        "002",
        "003",
    ]


def test_checkpoint_does_not_overwrite_existing_destination(tmp_path):
    with pytest.raises(FileExistsError):
        handoff._write_checkpoint(_parent(), {}, tmp_path)


@pytest.mark.parametrize("target", ["population.h5", "handoff.json"])
def test_checkpoint_rejects_changed_payload_or_report(tmp_path, target):
    root = tmp_path / "checkpoint"
    handoff._write_checkpoint(
        _parent(), {"protocol": handoff.PROTOCOL, "release_eligible": False}, root
    )
    path = root / target
    if target.endswith("h5"):
        with path.open("ab") as output:
            output.write(b"changed")
    else:
        report = json.loads(path.read_text())
        report["report"]["release_eligible"] = True
        path.write_text(json.dumps(report))
    with pytest.raises(ValueError, match="CHECKPOINT"):
        handoff.load_native_survey_development_checkpoint(root)


def test_maintained_builder_entry_refuses_unissued_owner_before_output(tmp_path):
    from test_us_fiscal_refresh_builder import _load_builder_module

    builder = _load_builder_module()
    with pytest.raises(ValueError, match="UNISSUED_RUN"):
        builder.prepare_native_survey_development_input(object(), tmp_path / "refused")
    assert not (tmp_path / "refused").exists()


@pytest.mark.parametrize("mutated", ["frame", "report", "final_digest", "final_frame"])
def test_final_owner_io_cannot_mutate_already_checked_readback(tmp_path, mutated):
    # These are descriptive invented values for the pure comparison helper.
    # No public handoff accepts them as a substitute for an issued native run.
    original = _parent()
    report = {"owner_receipt_sha256": "invented-receipt", "release_eligible": False}
    root = tmp_path / "readback"
    handoff._write_checkpoint(original, report, root)
    loaded = handoff.load_native_survey_development_checkpoint(root)
    borrowed = handoff.NativeSurveyDevelopmentInput(original, report)
    final = SimpleNamespace(
        digest="invented-receipt", population=SimpleNamespace(frame=original)
    )
    handoff._validate_readback_after_owner_io(
        borrowed, loaded.frame, loaded.report, final
    )
    # Model a mutation during the final foreign I/O, after a prior comparison.
    if mutated == "frame":
        loaded.frame.person.loc[0, "money"] = 1234.0
    elif mutated == "report":
        loaded.report["release_eligible"] = True
    elif mutated == "final_digest":
        final.digest = "changed"
    else:
        final.population.frame = loaded.frame
    with pytest.raises(
        ValueError, match="NATIVE_HANDOFF_CHECKPOINT_OWNER|SURVEY_POPULATION_REPLAY"
    ):
        handoff._validate_readback_after_owner_io(
            borrowed, loaded.frame, loaded.report, final
        )


def _projection_parent():
    """Six real entity tables with invented provenance and nullable source roles."""
    import pandas as pd

    from microcosm.build.spm_input_contract import ROLE_INPUT, UNIVERSE_INPUT
    from microcosm.frame import Frame, MassChangeRecord
    from microcosm.frame.units import US_SCHEMA

    original = _parent()
    tables = {entity: original.table(entity).copy() for entity in original.entities}
    for entity, ids in (
        ("spm_unit", [10, 10, 20, 20, 30, 30]),
        ("family", [11, 11, 21, 21, 31, 31]),
        ("marital_unit", [12, 12, 22, 22, 32, 32]),
    ):
        tables["person"][f"person_{entity}_id"] = np.array(ids, dtype=np.int64)
        tables[entity] = pd.DataFrame({f"{entity}_id": np.unique(ids)})
    tables["person"]["is_household_head"] = tables["person"].pop("reported")
    tables["person"]["is_household_head"].array._data[1] = True
    tables["person"]["hours"].array._data[1] = 87
    tables["person"][ROLE_INPUT] = np.array([False, False, True, False, False, False])
    tables["person"]["engine_only"] = np.arange(6, dtype=float)
    tables["person"].loc[3, "money"] = np.array(
        [0x7FF8000000001234], dtype=np.uint64
    ).view(np.float64)[0]
    tables["spm_unit"][UNIVERSE_INPUT] = pd.Series(
        ["INCLUDED", "OUTSIDE", "UNRESOLVED"], dtype="string"
    )
    return Frame(
        {entity: tables[entity] for entity in US_SCHEMA.entities},
        US_SCHEMA,
        {"household": original.weights_for("household")},
        original.strata,
        metadata={
            "construction": "invented-native-projection",
            "nested": {"basis": ("retained", 2024)},
        },
        mass_log=(
            MassChangeRecord("household", 6.0, 6.0, 1.0, "invented unchanged mass"),
        ),
    )


def _projection_declaration(frame):
    from microcosm.build.spm_input_contract import UNIVERSE_INPUT, UNIVERSE_STATUSES

    selected = {
        "person": (
            "money",
            "hours",
            "is_household_head",
            "is_spm_independent_minor_role",
        ),
        "household": ("county",),
        "tax_unit": ("deduction",),
        "spm_unit": (UNIVERSE_INPUT,),
        "family": (),
        "marital_unit": (),
    }
    columns = []
    for entity in frame.entities:
        names = [frame.schema.entity_id_column(entity)]
        if entity == "person":
            names += [
                frame.schema.membership_column(group)
                for group in frame.schema.group_entities
            ]
        names += list(selected[entity])
        columns.append(
            (
                entity,
                tuple((name, str(frame.table(entity)[name].dtype)) for name in names),
            )
        )
    return handoff.NativeSurveyEngineProjectionSpec(
        period=2024,
        consumer_identity="invented caller declaration; no consumer qualification",
        columns=tuple(columns),
        export_contract=ExportContract(
            required=(
                "household_weight",
                "money",
                "hours",
                "is_household_head",
                "is_spm_independent_minor_role",
                UNIVERSE_INPUT,
            ),
            optional=("county", "deduction"),
            forbidden=("source_person_id",),
            formula_owned_excluded=("engine_only",),
            closed=True,
        ),
        enum_domains=(("spm_unit", UNIVERSE_INPUT, tuple(sorted(UNIVERSE_STATUSES))),),
        consumer_id_dtype="int64",
        spm_settings=(("geography_kind", "county"),),
    )


def test_projection_preserves_six_entities_context_and_exact_nullable_backing():
    from microcosm.graph.population import storage_equal

    source = _projection_parent()
    result = handoff._project_native_survey_frame(
        source, _projection_declaration(source)
    )
    assert result.source_frame is source and result.frame is not source
    assert result.frame.schema == source.schema and len(result.frame.entities) == 6
    for entity in source.entities:
        assert result.frame.table(entity).index.identical(source.table(entity).index)
        for column in result.frame.table(entity):
            assert storage_equal(
                source.table(entity)[column], result.frame.table(entity)[column]
            )
    assert result.frame.metadata == source.metadata
    assert result.frame.mass_log == source.mass_log
    assert storage_equal(source.strata, result.frame.strata)
    assert (
        result.frame.weights_for("household").kind
        is source.weights_for("household").kind
    )
    assert (
        result.frame.weights_for("household").values.tobytes()
        == source.weights_for("household").values.tobytes()
    )
    assert result.frame.weights_for("household") is not source.weights_for("household")
    assert result.frame.person.is_household_head.array._data[1]
    assert result.frame.person.hours.array._data[1] == 87
    assert np.signbit(result.frame.person.money.iloc[0])
    result.frame.person.loc[0, "money"] = 123.0
    assert source.person.money.iloc[0] == 0.0 and np.signbit(
        source.person.money.iloc[0]
    )


def test_projection_readiness_remains_unqualified_and_aggregate_only():
    source = _projection_parent()
    result = handoff._project_native_survey_frame(
        source, _projection_declaration(source)
    )
    report = result.report
    for flag in (
        "simulation_ready",
        "release_eligible",
        "consumer_qualified",
        "domain_qualified",
        "formula_ownership_qualified",
        "source_signal_qualified",
        "source_applicability_qualified",
    ):
        assert report[flag] is False
    assert report["spm_status_counts"] == {"INCLUDED": 1, "OUTSIDE": 1, "UNRESOLVED": 1}
    assert report["headship_unknown_count"] == 2
    excluded = {(row["entity"], row["column"]) for row in report["excluded_columns"]}
    assert ("person", "source_person_id") in excluded
    assert ("person", "engine_only") in excluded
    assert ("household", "origin") in excluded
    assert "values" not in {key for row in report["columns"] for key in row}
    assert report["declaration_kind"] == "caller_supplied_unqualified"
    json.dumps(report, allow_nan=False)


@pytest.mark.parametrize(
    "kind,value,consumer,accepted",
    (
        ("int64", 2**53 + 1, "int64", True),
        ("uint64", 2**63 + 1, "uint64", True),
        ("uint64", 2**63 + 1, "int64", False),
        ("int64", 2**32, "int32", False),
        ("int64", -1, "uint64", False),
    ),
)
def test_projection_id_casts_are_only_losslessness_checks(
    kind, value, consumer, accepted
):
    from dataclasses import replace

    source = _projection_parent()
    source.person["person_id"] = np.arange(6, dtype=kind) + np.array(value, dtype=kind)
    spec = replace(_projection_declaration(source), consumer_id_dtype=consumer)
    if accepted:
        result = handoff._project_native_survey_frame(source, spec)
        assert result.frame.person.person_id.dtype == source.person.person_id.dtype
        assert (
            result.frame.person.person_id.to_numpy().tobytes()
            == source.person.person_id.to_numpy().tobytes()
        )
    else:
        with pytest.raises(ValueError, match="ID_CAST_RANGE"):
            handoff._project_native_survey_frame(source, spec)


@pytest.mark.parametrize(
    "change,code",
    (
        ("missing", "MISSING_COLUMN"),
        ("wrong_grain", "COLUMN_ENTITY"),
        ("dtype", "COLUMN_DTYPE"),
        ("formula", "EXCLUDED_INPUT"),
        ("prior_year", "EXCLUDED_INPUT"),
        ("extra", "UNDECLARED_INPUT"),
        ("open_contract", "CLOSED_CONTRACT"),
        ("missing_entity", "ENTITY_ROSTER"),
        ("missing_membership", "STRUCTURAL_COLUMNS"),
        ("bad_status", "DECLARED_DOMAIN"),
    ),
)
def test_projection_refusals_are_codes_without_row_examples(change, code):
    from dataclasses import replace

    source = _projection_parent()
    spec = _projection_declaration(source)
    columns = [(entity, list(names)) for entity, names in spec.columns]
    if change == "missing":
        source.person.drop(columns="money", inplace=True)
    elif change == "wrong_grain":
        source.person.drop(columns="money", inplace=True)
        source.table("household")["money"] = np.arange(3, dtype=float)
    elif change == "dtype":
        source.person["money"] = source.person.money.astype("Float64")
    elif change in ("formula", "prior_year", "extra"):
        name = {
            "formula": "engine_only",
            "prior_year": "self_employment_income_last_year",
            "extra": "unreviewed",
        }[change]
        source.person[name] = np.arange(6, dtype=float)
        columns[0][1].append((name, "float64"))
    elif change == "open_contract":
        spec = replace(
            spec, export_contract=replace(spec.export_contract, closed=False)
        )
    elif change == "missing_entity":
        columns.pop()
    elif change == "missing_membership":
        columns[0] = (
            columns[0][0],
            [(n, d) for n, d in columns[0][1] if n != "person_family_id"],
        )
    else:
        source.table("spm_unit").loc[0, "spm_unit_spm_universe_status"] = "UNKNOWN"
    spec = replace(spec, columns=tuple((e, tuple(c)) for e, c in columns))
    with pytest.raises(ValueError, match="NATIVE_ENGINE_PROJECTION_" + code) as caught:
        handoff._project_native_survey_frame(source, spec)
    assert str(caught.value) == "NATIVE_ENGINE_PROJECTION_" + code


def test_projection_public_entry_refuses_frames_checkpoints_and_forged_runs(tmp_path):
    source = _projection_parent()
    spec = _projection_declaration(source)
    root = tmp_path / "descriptive"
    handoff._write_checkpoint(source, {"protocol": handoff.PROTOCOL}, root)
    checkpoint = handoff.load_native_survey_development_checkpoint(root)
    forged = handoff.native.SurveyEnrichmentRun(
        None, None, None, None, None, None, (), b"invented"
    )
    for unsupported in (source, checkpoint, checkpoint.report, forged, object()):
        with pytest.raises(ValueError, match="UNISSUED_RUN"):
            handoff.prepare_native_survey_engine_input(unsupported, declaration=spec)


@pytest.mark.parametrize(
    "mutated",
    (
        "value",
        "null_backing",
        "null_mask",
        "weight",
        "weight_dtype",
        "codec",
        "strata",
        "metadata",
        "mass",
        "mass_type",
        "axis",
        "report",
        "declaration",
        "digest",
        "population",
        "source",
    ),
)
def test_projection_final_owner_io_cannot_mutate_previously_checked_result(mutated):
    from dataclasses import replace

    from microcosm.frame import MassChangeRecord, Weights

    source = _projection_parent()
    spec = _projection_declaration(source)
    result = handoff._project_native_survey_frame(source, spec)
    # Pure comparison values only; neither this test nor the helper issues an owner.
    population = SimpleNamespace(frame=source)
    final = SimpleNamespace(digest="invented-checked-owner", population=population)
    expected = (
        final.digest,
        population,
        handoff._projection_stamp(result.frame),
        handoff._json(result.report),
        handoff._projection_spec_bytes(spec),
    )
    handoff._validate_engine_projection_after_owner_io(
        result, spec, final, expected=expected
    )
    if mutated == "value":
        result.frame.person.loc[0, "money"] = 999.0
    elif mutated == "null_backing":
        result.frame.person.hours.array._data[1] += 1
    elif mutated == "null_mask":
        result.frame.person.hours.array._mask[1] = False
    elif mutated == "weight":
        result.frame._weights["household"] = Weights(
            np.array([1.0, 2.0, 4.0]), result.frame.weights_for("household").kind
        )
    elif mutated == "weight_dtype":
        values = result.frame.weights_for("household").values.view(np.uint64)
        object.__setattr__(result.frame.weights_for("household"), "values", values)
    elif mutated == "codec":
        result.frame._metadata = {"private_example": object()}
    elif mutated == "strata":
        result.frame.strata.iloc[0] = "changed"
    elif mutated == "metadata":
        result.frame._metadata = {"changed": True}
    elif mutated == "mass":
        result.frame._mass_log = (
            MassChangeRecord("household", 6.0, 7.0, None, "changed"),
        )
    elif mutated == "mass_type":

        class OtherMass(MassChangeRecord):
            pass

        result.frame._mass_log = (
            OtherMass("household", 6.0, 6.0, 1.0, "invented unchanged mass"),
        )
    elif mutated == "axis":
        result.frame.person.index = result.frame.person.index.rename("changed")
    elif mutated == "report":
        result.report["simulation_ready"] = True
    elif mutated == "declaration":
        spec = replace(spec, period=2025)
    elif mutated == "digest":
        final.digest = "changed"
    elif mutated == "population":
        final.population = SimpleNamespace(frame=source)
    else:
        source.person.loc[0, "money"] = 987.0
    with pytest.raises(ValueError, match="NATIVE_ENGINE_PROJECTION_"):
        handoff._validate_engine_projection_after_owner_io(
            result, spec, final, expected=expected
        )


def test_projection_no_consumer_id_dtype_never_casts_or_claims_readiness():
    from dataclasses import replace

    source = _projection_parent()
    source.person["person_id"] = np.arange(6, dtype=np.uint64) + np.uint64(2**63 + 1)
    result = handoff._project_native_survey_frame(
        source, replace(_projection_declaration(source), consumer_id_dtype=None)
    )
    assert result.frame.person.person_id.dtype == np.dtype("uint64")
    assert result.report["consumer_id_dtype"] is None
    assert result.report["consumer_qualified"] is False
    assert result.report["id_cast_performed"] is False


def test_projection_source_structure_refusal_has_no_private_row_examples():
    source = _projection_parent()
    spec = _projection_declaration(source)
    source.person.loc[0, "person_household_id"] = 987654321
    with pytest.raises(
        ValueError, match="^NATIVE_ENGINE_PROJECTION_STRUCTURE$"
    ) as caught:
        handoff._project_native_survey_frame(source, spec)
    assert "987654321" not in str(caught.value)


@pytest.mark.parametrize("name", handoff.US_PRIOR_YEAR_INCOME_OUTPUT_COLUMNS)
def test_projection_excludes_complete_shared_prior_year_family(name):
    from dataclasses import replace

    source = _projection_parent()
    source.person[name] = np.arange(6, dtype=float)
    spec = _projection_declaration(source)
    columns = list(spec.columns)
    columns[0] = (columns[0][0], (*columns[0][1], (name, "float64")))
    spec = replace(
        spec,
        columns=tuple(columns),
        export_contract=replace(
            spec.export_contract, optional=(*spec.export_contract.optional, name)
        ),
    )
    with pytest.raises(ValueError, match="^NATIVE_ENGINE_PROJECTION_EXCLUDED_INPUT$"):
        handoff._project_native_survey_frame(source, spec)


def test_projection_missing_required_is_not_default_filled():
    from dataclasses import replace

    source = _projection_parent()
    spec = _projection_declaration(source)
    spec = replace(
        spec,
        export_contract=replace(
            spec.export_contract,
            required=(*spec.export_contract.required, "absent_primitive"),
        ),
    )
    with pytest.raises(ValueError, match="^NATIVE_ENGINE_PROJECTION_MISSING_REQUIRED$"):
        handoff._project_native_survey_frame(source, spec)
    assert "absent_primitive" not in source.person


def test_projection_contract_required_weight_is_satisfied_by_typed_weights_only():
    source = _projection_parent()
    spec = _projection_declaration(source)
    assert "household_weight" in spec.export_contract.required
    result = handoff._project_native_survey_frame(source, spec)
    assert "household_weight" not in result.frame.table("household")
    assert result.report["typed_weight_inputs"] == [
        {
            "entity": "household",
            "name": "household_weight",
            "kind": "importance",
            "rows": 3,
            "materialized_as_source_column": False,
        }
    ]


@pytest.mark.parametrize(
    "excluded_as", ("forbidden", "formula_owned_excluded", "undeclared")
)
def test_projection_contract_cannot_hide_retained_typed_weight(excluded_as):
    from dataclasses import replace

    source = _projection_parent()
    spec = _projection_declaration(source)
    contract = spec.export_contract
    changes = {
        "required": tuple(
            name for name in contract.required if name != "household_weight"
        )
    }
    if excluded_as != "undeclared":
        changes[excluded_as] = (*getattr(contract, excluded_as), "household_weight")
    spec = replace(spec, export_contract=replace(contract, **changes))
    code = (
        "UNDECLARED_TYPED_WEIGHT" if excluded_as == "undeclared" else "EXCLUDED_INPUT"
    )
    with pytest.raises(ValueError, match="^NATIVE_ENGINE_PROJECTION_" + code + "$"):
        handoff._project_native_survey_frame(source, spec)
