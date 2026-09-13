"""Invented checkpoint-byte tests; private test pins are NOT Census evidence."""

import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from microcosm.build.frame_checkpoint import write_frame_checkpoint
from microcosm.build.serialization_dtypes import canonicalize_frame_string_dtypes
from microcosm.build.us_runtime import asec_current_money as money
from microcosm.build.us_runtime import asec_current_money_source as source
from microcosm.build.us_runtime import asec_household_observations as household
from microcosm.frame import US_SCHEMA, Frame, WeightKind, Weights


@pytest.mark.parametrize(
    "series",
    [
        pd.Series(
            ["HEAD", "DEPENDENT", "HEAD", "", "é", "e\u0301", "\x00", "😀"] * 100,
            dtype=object,
        ),
        pd.Series(["2024", None, "2024", ""] * 100, dtype="string[python]"),
        pd.Series(["2024", None, "2024", ""] * 100, dtype="string[pyarrow]"),
        pd.Series([str(i) for i in range(1500)] * 2, dtype=object),
        pd.Series(["a" * 300000, "b" * 300000, "a" * 300000], dtype=object),
        pd.Series([None, pd.NA, pd.NaT, np.nan], dtype=object),
        pd.Series([b"a", b"", b"a", None], dtype=object),
        pd.Series([True, False, None, np.bool_(True)], dtype=object),
        pd.Series([0, 2**80, -(2**80), None], dtype=object),
        pd.Series([0.0, -0.0, np.inf, -np.inf, float("nan"), None], dtype=object),
    ],
)
def test_source_scalar_digest_preserves_original_exact_byte_stream(series):
    # Independent pre-optimization scalar stream, including dtype metadata.
    expected = hashlib.sha256()
    expected.update(
        source._json(
            source.checkpoint._series_spec(series, label="authenticated source")
        )
    )
    for value in series.to_numpy(dtype=object, copy=False):
        encoded = source.checkpoint._encode_object_scalar(value)
        expected.update(len(encoded).to_bytes(8, "little"))
        expected.update(encoded)
    actual = hashlib.sha256()
    source._series_digest(actual, series)
    assert actual.digest() == expected.digest()


def test_repeated_source_strings_do_not_hide_later_value_or_order_mutations():
    series = pd.Series(["2024", "2024", "2025"], dtype="string[python]")

    def digest():
        value = hashlib.sha256()
        source._series_digest(value, series)
        return value.digest()

    before = digest()
    series.iloc[0] = "2025"
    assert digest() != before
    series.iloc[0] = "2024"
    assert digest() == before
    series.iloc[:] = ["2025", "2024", "2024"]
    assert digest() != before


def _original_fixture_module():
    spec = importlib.util.spec_from_file_location(
        "current_money_v4_fixture",
        Path(__file__).with_name("test_us_asec_checkpoint.py"),
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def invented_checkpoints(tmp_path, monkeypatch):
    """Exercise the real loaders with test-only substitutions of private file pins."""
    pytest.importorskip("microunit")
    fixture = _original_fixture_module()
    person = pd.DataFrame(
        {
            "person_id": np.arange(101, 107, dtype=np.int64),
            **{
                f"person_{g}_id": np.repeat(np.arange(1, 4, dtype=np.int64), 2)
                for g in US_SCHEMA.group_entities
            },
            "source_year": np.repeat([2022, 2023, 2024], 2),
            "source_household_id": np.repeat([7, 7, 7], 2),
            "PERIDNUM": [str(i).zfill(22) for i in range(1, 7)],
            "PH_SEQ": np.repeat([1, 2, 3], 2),
            "A_LINENO": [1, 2] * 3,
            "A_AGE": [55, 14] * 3,
            "A_MARITL": [7, 7] * 3,
            "A_SPOUSE": [0] * 6,
            "PEPAR1": [0, 0] * 3,
            "PEPAR2": [0] * 6,
            "A_EXPRRP": [1, 10] * 3,
            "A_ENRLW": [2] * 6,
            "A_FTPT": [0] * 6,
            "A_HSCOL": [0] * 6,
            **{
                c: [2] * 6
                for c in (
                    "PEDISDRS",
                    "PEDISEAR",
                    "PEDISEYE",
                    "PEDISOUT",
                    "PEDISPHY",
                    "PEDISREM",
                )
            },
            **{f: [0.0] * 6 for f in money.FIELDS if f != "HTOTVAL"},
            "LKWEEKS": [-1] * 6,
            "PAW_TYP": [0] * 6,
            "tax_unit_role_input": pd.array(["HEAD", "DEPENDENT"] * 3, dtype="string"),
            "is_related_to_head_or_spouse": [True] * 6,
            "unrelated_raw_observation": np.arange(6, dtype=np.int64),
        },
        index=pd.Index([8, 8, 3, 9, 9, 1], name="original_person_row"),
    )
    person["WSAL_VAL"] = [60000.0, 4800.0] * 3
    person["SEMP_VAL"] = [0.0, -1.0, 0.0, -9999.0, 0.0, 0.0]
    person["PTOTVAL"] = person.WSAL_VAL.to_numpy()
    person["ANN_VAL"] = -1.0
    for col in fixture.checkpoint_module.ASEC_REPORTED_COVERAGE_RAW_COLUMNS:
        person[col] = np.ones(6, dtype=np.int64)
    tables = {
        "person": person,
        **{
            g: pd.DataFrame({f"{g}_id": np.arange(1, 4, dtype=np.int64)})
            for g in US_SCHEMA.group_entities
        },
    }
    tables["household"]["state_fips"] = np.array([6, 36, 4], dtype=np.int64)
    tables["household"]["H_TENURE"] = np.array([2, 2, 2], dtype=np.int64)
    tables["tax_unit"]["filing_status_input"] = pd.array(
        ["HEAD_OF_HOUSEHOLD"] * 3, dtype="string"
    )
    frame = Frame(
        tables,
        US_SCHEMA,
        {"household": Weights(np.array([1.0, 2.0, 3.0]), WeightKind.DESIGN)},
        pd.Series(
            ["2022"] * 2 + ["2023"] * 2 + ["2024"] * 2,
            index=person.index,
            dtype="string",
        ),
    )
    frame = canonicalize_frame_string_dtypes(
        frame, boundary="invented current-money fixture"
    )
    metadata = fixture._v4_binding(frame)
    metadata["source_receipt"]["target_year"] = 2024
    cohort_sources = []
    for year in (2022, 2023, 2024):
        state = {2022: 6, 2023: 36, 2024: 4}[year]
        raw = pd.DataFrame(
            {
                "H_SEQ": [7],
                "HTOTVAL": [64800],
                "H_LIVQRT": [1],
                "HRHTYPE": [1],
                "H_HHTYPE": [1],
                "H_TENURE": [2],
                "GESTFIPS": [state],
            },
            dtype=np.int64,
        )
        path = tmp_path / f"invented-cohort-{year}.h5"
        raw.to_hdf(path, key="household", format="fixed")
        cohort_sources.append(
            household.AsecHouseholdObservationSource(
                year, path, money._sha(path.read_bytes())
            )
        )
    metadata["source_receipt"]["sources"] = [
        {"year": x.year, "path": str(x.path), "sha256": x.sha256}
        for x in cohort_sources
    ]
    registry = fixture.checkpoint_module.ASEC_EDUCATION_ASSISTANCE_ARCHIVES
    for column, mapping in metadata["raw_source_mappings"].items():
        mapping["source_pins"] = [
            {
                "income_year": y,
                "locator": registry[y].zip_url,
                "member": registry[y].member,
                "member_sha256": registry[y].member_sha256,
                "sha256": registry[y].zip_sha256,
            }
            for y in (2022, 2023, 2024)
        ]
        if column in fixture.checkpoint_module.ASEC_REPORTED_COVERAGE_RAW_COLUMNS:
            mapping["audit"] = {
                str(y): {
                    "rows": registry[y].rows,
                    "yes_rows": 1,
                    "no_rows": registry[y].rows - 1,
                    "weighted_yes_share": 0.25,
                }
                for y in (2022, 2023, 2024)
            }
    parent = tmp_path / "invented-original-v4.h5"
    write_frame_checkpoint(parent, frame, metadata=metadata)
    parent_sha = money._sha(parent.read_bytes())
    augmented = household.with_asec_household_observations(
        frame,
        checkpoint_metadata=metadata,
        checkpoint_sha256=parent_sha,
        sources=cohort_sources,
    )
    attachment = tmp_path / "invented-household-attachment.h5"
    write_frame_checkpoint(
        attachment,
        augmented.frame,
        metadata={
            "schema_version": 1,
            "artifact_kind": "microcosm.asec_household_observations_source",
            "parent_checkpoint_sha256": parent_sha,
            "household_observations": augmented.receipt,
        },
    )
    # Explicit private monkeypatch is confined to these invented-fixture tests.
    # The production public loader accepts neither an override nor caller pins.
    monkeypatch.setattr(
        source,
        "_SOURCE_PINS",
        (
            parent_sha,
            money._sha(attachment.read_bytes()),
            tuple((x.year, x.sha256) for x in cohort_sources),
        ),
    )
    return parent, attachment, frame, augmented.frame


def test_real_loaders_bind_invented_bytes_scope_and_all_cohort_evidence(
    tmp_path, monkeypatch
):
    parent, attachment, original, augmented = invented_checkpoints(
        tmp_path, monkeypatch
    )
    loaded = source.load_authenticated_current_money_source(parent, attachment)
    assert type(loaded.source) is money.AuthenticatedAsecSource
    authority = json.loads(loaded.source.identity)
    assert [x["survey_year"] for x in authority["sidecars"]] == [2023, 2024, 2025]
    assert tuple(loaded.scope.person_ids) == tuple(original.person.person_id)
    ready = loaded.ready()
    assert type(ready) is money.ReadyCurrentMoney
    assert (
        json.loads(ready.header)["source_authentication"] == "checkpoint_bytes_verified"
    )
    assert ready.field("ANN_VAL").amounts.tolist() == [0.0] * 6
    for entity in augmented.entities:
        pd.testing.assert_frame_equal(
            loaded.frame.table(entity), augmented.table(entity)
        )


def test_public_loader_has_no_caller_hash_or_coordinate_override(tmp_path):
    with pytest.raises(TypeError):
        source.load_authenticated_current_money_source(
            tmp_path / "x", tmp_path / "y", parent_sha256="a" * 64
        )


@pytest.mark.parametrize("which", ["parent", "attachment"])
def test_changed_checkpoint_bytes_refuse_before_authority(tmp_path, monkeypatch, which):
    parent, attachment, *_ = invented_checkpoints(tmp_path, monkeypatch)
    path = parent if which == "parent" else attachment
    with path.open("ab") as handle:
        handle.write(b"changed")
    with pytest.raises(money.MoneyRefusalError, match="SOURCE_BYTES"):
        source.load_authenticated_current_money_source(parent, attachment)


@pytest.mark.parametrize(
    "mutation", ["amount", "scope", "other_group", "metadata", "weights"]
)
def test_post_load_mutation_cannot_inherit_source_authority(
    tmp_path, monkeypatch, mutation
):
    parent, attachment, *_ = invented_checkpoints(tmp_path, monkeypatch)
    loaded = source.load_authenticated_current_money_source(parent, attachment)
    if mutation == "amount":
        loaded.frame.person.iloc[0, loaded.frame.person.columns.get_loc("WSAL_VAL")] = (
            11.0
        )
    elif mutation == "scope":
        loaded.frame.person.iloc[
            0, loaded.frame.person.columns.get_loc("source_year")
        ] = 2024
    elif mutation == "other_group":
        loaded.frame.table("family")["surprise"] = 1
    elif mutation == "metadata":
        with pytest.raises(TypeError):
            loaded.frame.metadata["changed"] = True
        loaded.validate()
        return
    else:
        loaded.frame.weights_for("household").values.setflags(write=True)
        loaded.frame.weights_for("household").values[0] = 99.0
    with pytest.raises(money.MoneyRefusalError, match="SOURCE_CHANGED"):
        loaded.ready()


@pytest.mark.parametrize(
    "change,reason",
    [
        ("parent", "ATTACHMENT_DOCUMENT"),
        ("output", "ATTACHMENT_OUTPUT_DIGEST"),
        ("cohort", "ATTACHMENT_SOURCE_COVERAGE"),
        ("counts", "ATTACHMENT_SOURCE_COVERAGE"),
        ("signed_zero", "PARENT_PROJECTION_CHANGED"),
    ],
)
def test_forged_attachment_metadata_or_parent_projection_refuses(
    tmp_path, monkeypatch, change, reason
):
    parent, attachment, *_ = invented_checkpoints(tmp_path, monkeypatch)
    loaded = source.checkpoint.load_frame_checkpoint(attachment)
    receipt = loaded.metadata["household_observations"]
    if change == "parent":
        loaded.metadata["parent_checkpoint_sha256"] = "0" * 64
    elif change == "output":
        receipt["outputs"]["asec_HTOTVAL"]["sha256"] = "0" * 64
    elif change == "cohort":
        receipt["sources"][0]["sha256"] = "0" * 64
    elif change == "counts":
        receipt["sources"][0]["joined_rows"] += 1
    else:
        loaded.frame.person["CAP_VAL"] = -0.0
    write_frame_checkpoint(attachment, loaded.frame, metadata=loaded.metadata)
    monkeypatch.setattr(
        source,
        "_SOURCE_PINS",
        (
            source._SOURCE_PINS[0],
            money._sha(attachment.read_bytes()),
            source._SOURCE_PINS[2],
        ),
    )
    with pytest.raises(money.MoneyRefusalError, match=reason):
        source.load_authenticated_current_money_source(parent, attachment)


def test_missing_one_ed_val_cohort_cannot_be_hidden_by_valid_other_mappings(
    tmp_path, monkeypatch
):
    parent, attachment, *_ = invented_checkpoints(tmp_path, monkeypatch)
    raw = source.checkpoint.load_frame_checkpoint(parent)
    raw.metadata["raw_source_mappings"]["ED_VAL"]["source_pins"].pop()
    write_frame_checkpoint(parent, raw.frame, metadata=raw.metadata)
    pin = money._sha(parent.read_bytes())
    attached = source.checkpoint.load_frame_checkpoint(attachment)
    attached.metadata["parent_checkpoint_sha256"] = pin
    attached.metadata["household_observations"]["input_checkpoint_sha256"] = pin
    write_frame_checkpoint(attachment, attached.frame, metadata=attached.metadata)
    monkeypatch.setattr(
        source,
        "_SOURCE_PINS",
        (pin, money._sha(attachment.read_bytes()), source._SOURCE_PINS[2]),
    )
    with pytest.raises(money.MoneyRefusalError, match="ED_VAL_SOURCE_COVERAGE"):
        source.load_authenticated_current_money_source(parent, attachment)


def test_verified_scope_does_not_authorize_changed_projected_amounts_or_indices(
    tmp_path, monkeypatch
):
    parent, attachment, *_ = invented_checkpoints(tmp_path, monkeypatch)
    loaded = source.load_authenticated_current_money_source(parent, attachment)
    views = loaded.views()
    views.person.iloc[0, views.person.columns.get_loc("WSAL_VAL")] = 1.0
    with pytest.raises(money.MoneyRefusalError, match="AUTHENTICATED_INPUT_MISMATCH"):
        money.classify_asec_money(views, loaded.spec, loaded.scope)
    views = loaded.views()
    views.person.index = pd.RangeIndex(len(views.person))
    with pytest.raises(money.MoneyRefusalError, match="AUTHENTICATED_INPUT_MISMATCH"):
        money.classify_asec_money(views, loaded.spec, loaded.scope)


def test_authenticated_codec_roundtrip_and_expected_header_refusal(
    tmp_path, monkeypatch
):
    from microcosm.build.us_runtime._asec_current_money_codec import (
        decode_current_money,
        encode_current_money,
    )

    parent, attachment, *_ = invented_checkpoints(tmp_path, monkeypatch)
    loaded = source.load_authenticated_current_money_source(parent, attachment)
    ready = loaded.ready()
    payload = encode_current_money(ready)
    replay = decode_current_money(payload, ready.bindings, expected=ready)
    assert (
        type(replay) is money.ReadyCurrentMoney
        and encode_current_money(replay) == payload
    )
    header = json.loads(ready.header)
    header["input_sha256"] = "0" * 64
    with pytest.raises(money.MoneyRefusalError, match="AUTHENTICATED_HEADER_BINDING"):
        money.MoneyBindings(loaded.spec, money._json(header))


def test_private_verified_copy_binds_the_decoded_bytes_when_original_path_changes(
    tmp_path, monkeypatch
):
    parent, attachment, *_ = invented_checkpoints(tmp_path, monkeypatch)
    original = source.asec_checkpoint.load_asec_raw_stage_checkpoint_v4

    def replace_original_then_read_verified_copy(path):
        assert Path(path) != parent
        parent.write_bytes(b"replacement at the original operational path")
        return original(path)

    monkeypatch.setattr(
        source.asec_checkpoint,
        "load_asec_raw_stage_checkpoint_v4",
        replace_original_then_read_verified_copy,
    )
    loaded = source.load_authenticated_current_money_source(parent, attachment)
    assert loaded.frame.person.WSAL_VAL.iloc[0] == 60000.0
    assert loaded.ready().field("WSAL_VAL").amounts[0] == 60000.0 * (174.4 / 163.6)
    with pytest.raises(money.MoneyRefusalError, match="SOURCE_BYTES_MISMATCH"):
        source.load_authenticated_current_money_source(parent, attachment)


def test_live_source_operator_projection_is_bound_to_verification(
    tmp_path, monkeypatch
):
    parent, attachment, *_ = invented_checkpoints(tmp_path, monkeypatch)
    loaded = source.load_authenticated_current_money_source(parent, attachment)
    projection = dict(source.operator_boundary.PRE_ASSEMBLY_OPERATOR_OUTPUT_FAMILIES)
    projection["invented_new_owner"] = {
        "person": frozenset({"unrelated_raw_observation"})
    }
    monkeypatch.setattr(
        source.operator_boundary, "PRE_ASSEMBLY_OPERATOR_OUTPUT_FAMILIES", projection
    )
    with pytest.raises(money.MoneyRefusalError, match="SOURCE_IMPLEMENTATION_CHANGED"):
        loaded.ready()
