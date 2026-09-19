"""A real retained source draw on tiny invented Census/parent fixture bytes."""

import hashlib
import json
import shutil
import sys
from dataclasses import asdict, replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from microcosm.build.us_runtime import current_asec_immigration_assignment as owner
from microcosm.frame import WeightKind


def source_arguments(tmp_path, monkeypatch):
    import test_us_current_asec_immigration_donor as fixture

    from microcosm.build.us_runtime import asec_person_income_source as restoration

    full, partial, unselected = fixture.source_arguments(tmp_path, monkeypatch)
    donor = owner.donor_owner
    root = full["source_dir"] / "asec"
    pins, paths = [], {}
    for year, member, archive, *_ in donor.literals.original.asec._MEMBER_PINS:
        path = root / member
        raw = pd.read_csv(path, dtype=str, keep_default_na=False)
        if year == 2024:
            adult, teen = raw.A_AGE.eq("55"), raw.A_AGE.eq("15")
            assert adult.sum() == teen.sum() == 1
            raw.loc[adult | teen, "PRCITSHP"] = "5"
            raw.loc[adult, ["PENATVTY", "PEINUSYR", "A_LFSR"]] = ["303", "25", "1"]
            raw.loc[teen, ["PENATVTY", "PEINUSYR"]] = ["164", "28"]
            raw.to_csv(path, index=False)
        payload = path.read_bytes()
        pins.append(
            (
                year,
                member,
                archive,
                hashlib.sha256(payload).hexdigest(),
                len(raw),
                len(payload),
            )
        )
        paths[year] = path
        shutil.copyfile(path, partial["source_dir"] / "asec" / member)
    for module in (
        donor.literals.original.asec,
        restoration,
        donor.demographics.demographic,
    ):
        monkeypatch.setattr(module, "_MEMBER_PINS", tuple(pins))
    output = tmp_path / "immigration-assignment-source"
    restoration.restore_asec_person_income_source(
        root / "parent.h5",
        root / "household-attachment.h5",
        member_paths=paths,
        output_dir=output,
    )
    for args in (full, partial):
        shutil.copyfile(
            output / restoration.CHECKPOINT_FILENAME,
            args["source_dir"] / "asec" / "person-income-attachment.h5",
        )
    return full, partial, unselected


@pytest.fixture(scope="module")
def actual(tmp_path_factory):
    with pytest.MonkeyPatch.context() as patch:
        full_args, partial_args, unselected = source_arguments(
            tmp_path_factory.mktemp("asec-immigration-assignment"), patch
        )
        full = owner.source.prepare_authenticated_survey_population(**full_args)
        partial = owner.source.prepare_authenticated_survey_population(**partial_args)
        left = owner.donor_owner.borrow_full_asec_immigration_donor(full)
        right = owner.donor_owner.borrow_full_asec_immigration_donor(partial)
        originals = [
            (donor, owner.source._frame_identity(donor.frame))
            for donor in (left, right)
        ]
        calls, draws = [], []
        previous = sys.getprofile()

        def observe(frame, event, arg):
            # Read only invented fixture call inputs, without replacing the live
            # rule function or bypassing the owner's implementation fences.
            if event != "call":
                return
            if frame.f_code is owner.source_runtime.run_source_stage.__code__:
                calls.append(dict(frame.f_locals))
            elif frame.f_code is owner.rules._stable_person_draws.__code__:
                draws.append(
                    frame.f_locals["person"].loc[:, list(owner._LINEAGE)].copy()
                )

        try:
            sys.setprofile(observe)
            assigned = owner.assign_full_asec_immigration(left, seed=42)
        finally:
            sys.setprofile(previous)
        smaller = owner.assign_full_asec_immigration(right, seed=42)
        yield SimpleNamespace(
            full=full,
            partial=partial,
            left=left,
            right=right,
            assigned=assigned,
            smaller=smaller,
            unselected=unselected,
            calls=calls,
            draws=draws,
        )
        for donor, seal in originals:
            donor.validate()
            assert owner.source._frame_identity(donor.frame) == seal


def test_real_full_assignment_survives_selection_with_original_pairs(actual):
    left, right = actual.assigned, actual.smaller
    left.validate()
    right.validate()
    assert (
        actual.unselected
        not in actual.partial._checked()[2].native[1].frame.person.person_id.tolist()
    )
    assert actual.unselected in right.pairs.index
    assert (
        right.pairs.loc[actual.unselected, "immigration_status_str"]
        == "PAROLED_ONE_YEAR"
    )
    assert (
        right.pairs.loc[actual.unselected, "ssn_card_type"] == "NON_CITIZEN_VALID_EAD"
    )
    assert sorted(right.frame.person.immigration_status_str.tolist()) == [
        "CITIZEN",
        "CITIZEN",
        "PAROLED_ONE_YEAR",
        "UNDOCUMENTED",
    ]
    assert owner.source._frame_identity(left.frame) == owner.source._frame_identity(
        right.frame
    )
    pd.testing.assert_frame_equal(left.pairs, right.pairs)
    pd.testing.assert_frame_equal(
        right.pairs.loc[:, owner.donor_owner.literals.original.ASEC_KEYS],
        actual.right.raw.loc[:, owner.donor_owner.literals.original.ASEC_KEYS],
    )
    assert right.pairs.income_year.eq(2024).all()
    assert right.pairs.index.equals(actual.right.raw.index)


def test_exactly_one_call_uses_original_design_weights_and_observation_year(actual):
    assert len(actual.calls) == 1
    call = actual.calls[0]
    assert call["config"].seed == 42
    assert call["config"].target_year == 2025
    handler = call["operation_handlers"]["derive_immigration_status"]
    assert handler.func is owner.rules.derive_us_immigration_status_from_manifest
    assert handler.keywords["observation_year_column"] == owner.OBSERVATION_YEAR_COLUMN
    person = call["tables"]["person"]
    assert person[owner.OBSERVATION_YEAR_COLUMN].eq(2025).all()
    assert actual.left.frame.weights_for("household").kind is WeightKind.DESIGN
    assert actual.left.frame.weights_for("household").values.tolist() == [
        2552.12,
        100.0,
    ]
    np.testing.assert_array_equal(
        person.person_weight,
        actual.left.frame.resolve_weights("person").values,
    )
    source_stage = asdict(call["stage"])
    assert (
        json.loads(owner.source._encode(source_stage))
        == json.loads(actual.assigned.receipt)["control_stage"]
    )
    assert actual.draws
    expected = pd.DataFrame(
        {
            "source_year": 2024,
            "source_household_id": actual.left.frame.person.PH_SEQ.map(int),
            "source_person_id": actual.left.frame.person.PERIDNUM,
        }
    )
    for keys in actual.draws:
        pd.testing.assert_frame_equal(keys, expected)
    assert not {"WSAL_VAL", "SEMP_VAL"} & set(person)
    assert "person_weight" not in actual.assigned.frame.person


def test_every_nonowned_cell_and_typed_frame_property_survives(actual):
    source, assigned = actual.right.frame, actual.smaller.frame
    for entity in source.entities:
        table = assigned.table(entity)
        if entity == "person":
            table = table.drop(columns=list(owner.rules.US_IMMIGRATION_OUTPUT_COLUMNS))
        pd.testing.assert_frame_equal(table, source.table(entity))
    assert assigned.weighted_entities == source.weighted_entities
    for entity in source.weighted_entities:
        assert assigned.weights_for(entity) is source.weights_for(entity)
    pd.testing.assert_series_equal(assigned.strata, source.strata)
    assert assigned.metadata == source.metadata
    assert assigned.mass_log == source.mass_log
    assert (
        tuple(owner.rules.US_IMMIGRATION_REQUIRED_SOURCE_COLUMNS)
        == owner.donor_owner.literals.ASEC_VALUE_COLUMNS
    )


def test_captured_stage_matches_maintained_frame_operator(actual):
    person = actual.calls[0]["tables"]["person"].drop(columns=["person_weight"])
    expected = owner.rules.with_us_immigration_inputs(
        owner._with_person(actual.left.frame, person),
        seed=42,
        time_period=2025,
        person_weight_scale=1.0,
        observation_year_column=owner.OBSERVATION_YEAR_COLUMN,
    )
    columns = list(owner.rules.US_IMMIGRATION_OUTPUT_COLUMNS)
    pd.testing.assert_frame_equal(
        actual.assigned.frame.person.loc[:, columns], expected.person.loc[:, columns]
    )


def test_receipt_binds_source_rules_and_unqualified_mixed_date_assumptions(actual):
    assigned = actual.smaller
    receipt = json.loads(assigned.receipt)
    assert receipt["purpose"] == "development_imputation"
    assert receipt["status_assignment_performed"]
    for flag in (
        "observed_legal_status",
        "national_stock_alignment_qualified",
        "composition_gate_performed",
        "prior_income_columns_consumed",
        "source_admission_issued",
    ):
        assert receipt[flag] is False
    assert receipt["income_year"] == 2024 and receipt["observation_year"] == 2025
    assert receipt["person_weight_scale"] == 1.0
    assert receipt["weight_source"] == "original_HSUP_WGT/100"
    assert receipt["person_weight_authority"] == "none"
    assert receipt["readset"] == list(owner.donor_owner.literals.ASEC_VALUE_COLUMNS)
    assert receipt["donor_receipt_sha256"] == owner.source._sha(actual.right.receipt)
    assert receipt["control_stage"] == json.loads(owner._stage())
    assert receipt["control_stage_sha256"] == owner.source._sha(owner._stage())
    assert receipt["implementation"] == json.loads(owner._implementation())
    assert (
        receipt["control_reference_bases"]["humanitarian_status_stocks.tps"]
        == "as of 2025-03-31"
    )
    assert (
        "not qualified" in receipt["control_reference_bases"]["undocumented_students"]
    )


@pytest.mark.parametrize("seed", [True, -1, 2**64, 4.5, "42", None])
def test_invalid_seed_refuses(actual, seed):
    with pytest.raises(ValueError, match="ASSIGNMENT_SEED"):
        owner.assign_full_asec_immigration(actual.right, seed=seed)


def test_detached_inputs_and_copies_cannot_gain_authority(actual):
    for value in (actual.right.frame, actual.right.receipt, actual.partial):
        with pytest.raises(ValueError, match="DONOR_TYPE"):
            owner.assign_full_asec_immigration(value, seed=42)
    with pytest.raises(ValueError, match="ISSUED_OWNER_REQUIRED"):
        owner.assign_full_asec_immigration(replace(actual.right), seed=42)
    with pytest.raises(ValueError, match="ISSUED_OWNER_REQUIRED"):
        replace(actual.smaller).validate()


@pytest.mark.parametrize(
    "columns",
    [
        ("ssn_card_type",),
        ("immigration_status_str",),
        owner.rules.US_IMMIGRATION_OUTPUT_COLUMNS,
    ],
)
def test_preexisting_partial_or_complete_outputs_never_passthrough(actual, columns):
    people = actual.right.frame.person
    try:
        for name in columns:
            people[name] = "unowned"
        with pytest.raises(ValueError, match="PREEXISTING_OUTPUT"):
            owner.assign_full_asec_immigration(actual.right, seed=42)
    finally:
        people.drop(columns=list(columns), inplace=True)
    actual.right.validate()


@pytest.mark.parametrize(
    "surface,column",
    [
        ("pairs", "ssn_card_type"),
        ("pairs", "PERIDNUM"),
        ("frame", "immigration_status_str"),
        ("frame", "A_AGE"),
        ("donor_raw", "A_LFSR"),
        ("donor_frame", "A_AGE"),
    ],
)
def test_retained_input_output_and_keys_are_not_mutable_authority(
    actual, surface, column
):
    assigned = actual.smaller
    table = {
        "pairs": assigned.pairs,
        "frame": assigned.frame.person,
        "donor_raw": actual.right.raw,
        "donor_frame": actual.right.frame.person,
    }[surface]
    index = table.index[0]
    previous = table.at[index, column]
    try:
        table.at[index, column] = 99 if column == "A_AGE" else "changed"
        with pytest.raises(ValueError):
            assigned.validate()
    finally:
        table.at[index, column] = previous
    assigned.validate()


@pytest.mark.parametrize("field", ["frame", "pairs", "receipt"])
def test_owner_identity_and_projection_identity_are_retained(actual, field):
    assigned = actual.smaller
    previous = getattr(assigned, field)
    replacement = (
        owner._with_person(previous, previous.person.copy(deep=True))
        if field == "frame"
        else previous.copy(deep=True)
        if field == "pairs"
        else b"{}"
    )
    try:
        object.__setattr__(assigned, field, replacement)
        with pytest.raises(ValueError, match="ASSIGNMENT_PROJECTION_CHANGED"):
            assigned.validate()
    finally:
        object.__setattr__(assigned, field, previous)
    assigned.validate()


@pytest.mark.parametrize(
    "surface", ["table_attributes", "weight_kind", "weight_storage"]
)
def test_decorated_frame_and_typed_weights_remain_sealed(actual, surface):
    assigned = actual.smaller
    weights = assigned.frame.weights_for("household")
    before_kind = weights.kind
    before_writeable = weights.values.flags.writeable
    before_attrs = dict(assigned.frame.person.attrs)
    try:
        if surface == "table_attributes":
            assigned.frame.person.attrs["changed"] = True
        elif surface == "weight_kind":
            object.__setattr__(weights, "kind", WeightKind.IMPORTANCE)
        else:
            weights.values.setflags(write=not before_writeable)
        with pytest.raises(ValueError):
            assigned.validate()
    finally:
        assigned.frame.person.attrs.clear()
        assigned.frame.person.attrs.update(before_attrs)
        object.__setattr__(weights, "kind", before_kind)
        weights.values.setflags(write=before_writeable)
    assigned.validate()


def test_issuance_refuses_donor_mutation_after_rule_draw(actual, monkeypatch):
    reader = Path.read_text
    raw = actual.right.raw
    index, before = raw.index[0], raw.A_LFSR.iloc[0]
    reads = []

    def late(path, *args, **kwargs):
        result = reader(path, *args, **kwargs)
        if path.name == "source_stages.json":
            reads.append(True)
            # The second read is the issuance validator, after the actual draw.
            if len(reads) == 2:
                raw.at[index, "A_LFSR"] = "changed"
        return result

    try:
        with monkeypatch.context() as patch:
            patch.setattr(Path, "read_text", late)
            with pytest.raises(ValueError):
                owner.assign_full_asec_immigration(actual.right, seed=42)
        assert len(reads) == 2
    finally:
        raw.at[index, "A_LFSR"] = before
    actual.right.validate()


def test_new_runtime_keeps_provenance_guards():
    import test_us_spine_blindness as guard

    path = Path(owner.__file__)
    assert path.name in guard._US_LAUNCH_GRAPH_RUNTIME_MODULES
    assert path.name not in guard._SOURCE_SPINE_PROVENANCE_OWNERS
    assert not guard._non_owner_source_spine_accesses(path.name, path.read_text())


@pytest.mark.parametrize("changed_read", [1, 2])
def test_transient_manifest_change_cannot_misrepresent_consumed_controls(
    actual, monkeypatch, changed_read
):
    reader = Path.read_text
    reads = []

    def transient(path, *args, **kwargs):
        text = reader(path, *args, **kwargs)
        if path.name == "source_stages.json":
            reads.append(True)
            if len(reads) == changed_read:
                # This control changes the invented recipient's parole draw.
                return text.replace('"target": 158000', '"target": 0')
        return text

    with monkeypatch.context() as patch:
        patch.setattr(Path, "read_text", transient)
        with pytest.raises(ValueError, match="CONTROL_STAGE_CHANGED"):
            owner.assign_full_asec_immigration(actual.right, seed=42)
    assert len(reads) == 2
    actual.right.validate()


def test_final_manifest_io_cannot_invalidate_retained_source_files(
    tmp_path, monkeypatch
):
    full_args, _, _ = source_arguments(tmp_path, monkeypatch)
    prepared = owner.source.prepare_authenticated_survey_population(**full_args)
    donor = owner.donor_owner.borrow_full_asec_immigration_donor(prepared)
    assigned = owner.assign_full_asec_immigration(donor, seed=42)
    source_path = full_args["source_dir"] / "asec" / "pppub25.csv"
    original = source_path.read_bytes()
    reader = Path.read_text
    changed = []

    def late(path, *args, **kwargs):
        text = reader(path, *args, **kwargs)
        if path.name == "source_stages.json" and not changed:
            changed.append(True)
            source_path.write_bytes(original + b"\n")
        return text

    with monkeypatch.context() as patch:
        patch.setattr(Path, "read_text", late)
        with pytest.raises(ValueError, match="SOURCE_FILES_CHANGED"):
            assigned.validate()
    assert changed


def test_final_manifest_io_cannot_invalidate_retained_donor_code(tmp_path, monkeypatch):
    # Point implementation-byte validation at an identical scratch copy before
    # issuance, so the mutation never edits a maintained implementation file.
    copied_code = tmp_path / "retained-donor-implementation.py"
    copied_code.write_bytes(Path(owner.donor_owner.__file__).read_bytes())
    monkeypatch.setattr(owner.donor_owner, "__file__", str(copied_code))
    full_args, _, _ = source_arguments(tmp_path, monkeypatch)
    prepared = owner.source.prepare_authenticated_survey_population(**full_args)
    donor = owner.donor_owner.borrow_full_asec_immigration_donor(prepared)
    assigned = owner.assign_full_asec_immigration(donor, seed=42)
    original = copied_code.read_bytes()
    reader = Path.read_text
    changed = []

    def late(path, *args, **kwargs):
        text = reader(path, *args, **kwargs)
        if path.name == "source_stages.json" and not changed:
            changed.append(True)
            copied_code.write_bytes(original + b"\n# changed retained code\n")
        return text

    with monkeypatch.context() as patch:
        patch.setattr(Path, "read_text", late)
        with pytest.raises(ValueError, match="IMPLEMENTATION_FILES_CHANGED"):
            assigned.validate()
    assert changed


@pytest.mark.parametrize("surface", ["function", "defaults", "constant"])
def test_changed_live_rule_implementation_refuses(actual, monkeypatch, surface):
    with monkeypatch.context() as patch:
        if surface == "function":
            patch.setattr(
                owner.rules,
                "derive_us_immigration_status_from_manifest",
                lambda *a, **kw: None,
            )
        elif surface == "defaults":
            patch.setattr(
                owner.rules.derive_us_immigration_status_from_manifest,
                "__kwdefaults__",
                {},
            )
        else:
            patch.setitem(owner.rules._PAROLE_ORIGIN_CODES, "ukraine", (303,))
        with pytest.raises(ValueError, match="IMPLEMENTATION_CHANGED"):
            actual.smaller.validate()
        with pytest.raises(ValueError, match="IMPLEMENTATION_CHANGED"):
            owner.assign_full_asec_immigration(actual.right, seed=42)
    actual.smaller.validate()


@pytest.mark.parametrize(
    "surface", ["pair", "donor", "function", "code_bytes", "stage"]
)
def test_final_foreign_io_and_dependency_changes_refuse(actual, monkeypatch, surface):
    assigned = actual.smaller
    index = assigned.pairs.index[0]
    pair_before = assigned.pairs.at[index, "ssn_card_type"]
    raw_before = actual.right.raw.at[index, "A_LFSR"]
    reader = Path.read_bytes
    changed = []

    def late(path):
        result = reader(path)
        if not changed and str(path) == owner.__file__:
            changed.append(True)
            if surface == "pair":
                assigned.pairs.at[index, "ssn_card_type"] = "changed"
            elif surface == "donor":
                actual.right.raw.at[index, "A_LFSR"] = "changed"
            elif surface == "function":
                monkeypatch.setattr(
                    owner.rules, "_stable_person_draws", lambda *a, **k: None
                )
            elif surface == "code_bytes":
                return result + b"\n# changed implementation\n"
        return result

    try:
        with monkeypatch.context() as patch:
            patch.setattr(Path, "read_bytes", late)
            if surface == "stage":
                # Stage loader reads text, after final implementation-byte I/O.
                text_reader = Path.read_text

                def changed_stage(path, *args, **kwargs):
                    text = text_reader(path, *args, **kwargs)
                    if path.name == "source_stages.json":
                        return text.replace('"target": 9700000', '"target": 9700001')
                    return text

                patch.setattr(Path, "read_text", changed_stage)
            with pytest.raises(ValueError):
                assigned.validate()
        assert changed
    finally:
        assigned.pairs.at[index, "ssn_card_type"] = pair_before
        actual.right.raw.at[index, "A_LFSR"] = raw_before
    if surface != "function":
        assigned.validate()
