"""Invented original-source composition; no PUF, QRF, native data or engine."""

import copy
import hashlib
import json
import shutil
from dataclasses import fields, is_dataclass, replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from microcosm.build.us_runtime import current_property_income_sources as owner
from microcosm.build.us_runtime import graph_survey_population as graph
from microcosm.fit import model_input
from microcosm.frame import Frame, WeightKind, Weights


def _detached(value):
    # Avoid Python 3.14 copy's __slotnames__ class-cache mutation on core
    # slotted types. Real constructors preserve the same physical contents.
    if isinstance(value, Frame):
        return Frame(
            {name: value.table(name).copy(deep=True) for name in value.entities},
            value.schema,
            {
                name: _detached(value.weights_for(name))
                for name in value.weighted_entities
            },
            value.strata.copy(deep=True),
            mass_log=value.mass_log,
            metadata=value.metadata,
        )
    if isinstance(value, Weights):
        return Weights(value.values.copy(), value.kind)
    if isinstance(value, pd.DataFrame):
        return value.copy(deep=True)
    if is_dataclass(value):
        return type(value)(
            **{
                field.name: _detached(getattr(value, field.name))
                for field in fields(value)
            }
        )
    if type(value) is tuple:
        return tuple(_detached(item) for item in value)
    if type(value) is list:
        return [_detached(item) for item in value]
    if type(value) is dict:
        return {key: _detached(item) for key, item in value.items()}
    return copy.deepcopy(value)


def source_arguments(tmp_path, monkeypatch):
    import test_us_survey_population_preparation as fixture
    from test_us_asec_coverage_authentication import _changed_parent
    from test_us_current_asec_income_routing import routing_arguments

    from microcosm.build.frame_checkpoint import load_frame_checkpoint
    from microcosm.build.us_runtime import asec_person_income_source as restoration

    original = fixture._person

    def person(*args, **kwargs):
        row = original(*args, **kwargs)
        row.update(FINTP="0", FRETP="0", INTP="100", RETP="0")
        if row["SERIALNO"] == "2024HU0000001" and int(row["SPORDER"]) == 1:
            row.update(INTP="-800", ADJINC="1000133")
        elif row["SERIALNO"] == "2024HU0000002":
            row["INTP"] = "bad"
        elif row["SERIALNO"] == "2024GQ0000001":
            row.update(AGEP="14", INTP="", RETP="", WAGP="", SEMP="")
        return row

    monkeypatch.setattr(fixture, "_person", person)
    arguments = routing_arguments(tmp_path, monkeypatch)
    folder = arguments["source_dir"] / "asec"
    parent_path, attachment = folder / "parent.h5", folder / "household-attachment.h5"
    people = load_frame_checkpoint(parent_path).frame.person
    ids = people.person_id.to_numpy()
    literals = {}
    for pid, age in zip(ids, people.A_AGE, strict=True):
        literals[int(pid)] = {
            **{
                name: "0"
                for name in (*owner.interest.READ_COLUMNS, *owner.dividend.READ_COLUMNS)
            },
            "A_AGE": str(int(age)),
            "INT_YN": "2" if age >= 15 else "0",
            "RINT_YN": "2" if age >= 15 else "0",
            "DIV_YN": "2" if age >= 15 else "0",
            "SUR_YN": "2" if age >= 15 else "0",
        }
    literals[105].update(
        INT_VAL="140",
        TRDINT_VAL="100",
        INT_YN="1",
        RINT_YN="1",
        RINT_SC1="4",
        RINT_VAL1="40",
        DIV_VAL="140",
        DIV_YN="1",
    )
    literals[107].update(
        INT_VAL="500",
        TRDINT_VAL="500",
        INT_YN="1",
        DIV_VAL="500",
        DIV_YN="1",
        SUR_YN="1",
        SUR_SC2="8",
    )
    changes = {}
    for name in ("INT_VAL", "DIV_VAL"):
        values = people[name].to_numpy(copy=True)
        for pid in (105, 106, 107, 108):
            values[ids == pid] = int(literals[pid][name])
        changes[name] = values
    _changed_parent(parent_path, attachment, monkeypatch, changes)
    updated = load_frame_checkpoint(parent_path).frame.person.set_index("PERIDNUM")
    paths, pins = {}, []
    for year, member, archive, *_ in owner.routing.coverage._MEMBER_PINS:
        path = folder / f"pppub{year - 1999}.csv"
        raw = pd.read_csv(path, dtype=str, keep_default_na=False)
        for name in set(
            (*owner.interest.READ_COLUMNS, *owner.dividend.READ_COLUMNS)
        ) - set(owner.routing.COORDINATE_COLUMNS):
            raw[name] = [
                literals[int(updated.loc[key, "person_id"])][name]
                for key in raw.PERIDNUM
            ]
        # Historical parent values remain their own current-money observations.
        for name in ("INT_VAL", "DIV_VAL"):
            raw[name] = [str(int(updated.loc[key, name])) for key in raw.PERIDNUM]
        raw.iloc[::-1].to_csv(path, index=False)
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
    for module in (owner.routing.coverage, restoration):
        monkeypatch.setattr(module, "_MEMBER_PINS", tuple(pins))
    output = tmp_path / "property-restored-money"
    restoration.restore_asec_person_income_source(
        parent_path, attachment, member_paths=paths, output_dir=output
    )
    shutil.copyfile(
        output / restoration.CHECKPOINT_FILENAME, folder / "person-income-attachment.h5"
    )
    return {**arguments, "store_root": tmp_path / "property-survey-store"}


@pytest.fixture(scope="module")
def actual(tmp_path_factory):
    patch = pytest.MonkeyPatch()
    root = tmp_path_factory.mktemp("property-source-composition")
    try:
        arguments = source_arguments(root, patch)
        live = graph.run_authenticated_survey_population(
            **arguments, clones=True, return_values=True
        )
        frame_seal = owner._frame_seal(live.preparation._checked()[2].frame)
        result = owner.qualify_current_property_income_sources(
            live.preparation, live.allocated_population, live.clone_population
        )
        assert owner._frame_seal(live.preparation._checked()[2].frame) == frame_seal
        (root / "composition-metadata.json").write_text(
            json.dumps(
                {
                    "scope": "invented source-composition fixture; no model or native data",
                    "survey_stage_order": list(live.compiled.order),
                    "source_people": result.source_frame.n("person"),
                    "source_households": result.source_frame.n("household"),
                    "allocated_people": live.allocated_population.frame.n("person"),
                    "clone_people": live.clone_population.frame.n("person"),
                    "eligible_donors": len(result.donor_columns),
                    "eligible_recipients": len(result.recipient_columns),
                },
                indent=2,
            )
            + "\n"
        )
        yield live, result, arguments
    finally:
        patch.undo()


def test_actual_source_composition_preserves_design_and_signed_anchor(actual):
    live, result, _ = actual
    assert len(result.origins) == 9
    assert len(result.donor_columns) == 1
    assert len(result.recipient_columns) == 3
    assert result.donor_frame.weights_for("household").kind is WeightKind.DESIGN
    assert (
        live.allocated_population.frame.weights_for("household").kind
        is WeightKind.IMPORTANCE
    )
    row = result.donor_columns.iloc[0]
    assert row.property_ordinary_interest == 100
    assert row.property_retirement_interest == 40
    assert row.property_dividends == 140
    assert row.property_broad_receipts == -400
    assert row.property_reported_total == -120
    selected = result.recipient_columns.property_reported_total
    assert (selected < 0).any()
    amount = np.float64(-800) * (np.float64(1000133) / 1000000)
    anchor = result.recipient_diagnostics
    exact_id = anchor.index[
        anchor.SERIALNO.eq("2024HU0000001") & anchor.SPORDER.eq("1")
    ][0]
    assert selected.loc[exact_id].view("uint64") == amount.view("uint64")
    pd.testing.assert_frame_equal(
        model_input.decode_recipient_matrix(result.recipient_matrix).features,
        result.recipient_columns,
    )
    assert result.recipient_diagnostics.excluded_under15.sum() == 1
    assert result.recipient_diagnostics.excluded_unknown_anchor.sum() == 2
    assert result.evidence["source_admission_issued"] is False
    assert not result.evidence["model_fitted"]
    assert not result.evidence["clone_attachment_performed"]
    assert json.loads(result.origin_document)["persons"]["rows"]


def _values(result):
    return (
        result.shared_predictors,
        result.acs_anchor_values,
        result.asec_interest_values,
        result.asec_routing_values,
        result.asec_dividend_values,
    )


@pytest.mark.parametrize(
    "target",
    [
        "asec_person",
        "asec_native",
        "acs_person",
        "acs_native",
        "donor_features",
        "recipient_matrix",
        "origin_document",
    ],
)
def test_source_composition_refuses_axis_mismatch(actual, target):
    _, result, _ = actual
    values = list(_detached(_values(result)))
    origin = result.origin_document
    if target == "asec_person":
        values[2].person.index = values[2].person.index[::-1]
    elif target == "asec_native":
        values[3].person["native_person_id"] = values[
            3
        ].person.native_person_id.to_numpy()[::-1]
    elif target == "acs_person":
        values[1].anchors.index = values[1].anchors.index[::-1]
    elif target == "acs_native":
        values[1].anchors["native_person_id"] = values[
            1
        ].anchors.native_person_id.to_numpy()[::-1]
    elif target == "donor_features":
        values[0].donor_columns.index = values[0].donor_columns.index[::-1]
    elif target == "recipient_matrix":
        matrix = model_input.decode_recipient_matrix(values[0].matrix)
        features = matrix.features.iloc[::-1]
        values[0] = replace(
            values[0],
            matrix=model_input.encode_recipient_matrix(
                features, entity="person", entity_ids=features.index.to_numpy()
            ),
        )
    else:
        document = json.loads(origin)
        document["persons"]["rows"].reverse()
        origin = owner.shared.codec.encode_json(document)
    with pytest.raises(ValueError, match="PROPERTY_INCOME_SOURCES"):
        owner._compose(tuple(values), origin)


@pytest.mark.parametrize(
    "target",
    [
        "basis",
        "nullable_storage",
        "donor_columns",
        "recipient_columns",
        "origin_document",
        "source_frame",
        "projection",
    ],
)
def test_complete_description_seal_detects_every_returned_surface(actual, target):
    _, result, _ = actual
    before = owner.property_income_sources_seal(result)
    changed = _detached(result)
    if target == "basis":
        changed.donor_basis.person.iloc[
            0, changed.donor_basis.person.columns.get_loc("property_reported_total")
        ] += 1
    elif target == "nullable_storage":
        array = changed.acs_anchor_values.anchors.property_income_amount.array
        position = np.flatnonzero(array._mask)[0]
        array._data[position] = 321.0
    elif target == "donor_columns":
        changed.donor_columns.iloc[0, 0] += 1
    elif target == "recipient_columns":
        changed.recipient_columns.iloc[0, 0] += 1
    elif target == "origin_document":
        changed = replace(changed, origin_document=changed.origin_document + b" ")
    elif target == "source_frame":
        changed.source_frame.person.iloc[
            0, changed.source_frame.person.columns.get_loc("age")
        ] += 1
    else:
        changed = replace(changed, projection=changed.projection + b" ")
    assert owner.property_income_sources_seal(changed) != before
    assert owner.property_income_sources_seal(result) == before


def test_no_eligible_recipient_is_explicit_without_empty_matrix_or_zero_fill(actual):
    _, result, _ = actual
    values = list(_detached(_values(result)))
    anchors = values[1].anchors
    anchors["property_income_known"] = False
    anchors["property_income_amount"] = pd.array([None] * len(anchors), dtype="Float64")
    anchors["property_income_status"] = "missing_source_amount"
    rebuilt = owner._compose(tuple(values), result.origin_document)
    assert rebuilt.recipient_frame is None and rebuilt.recipient_matrix is None
    assert len(rebuilt.recipient_columns) == 0
    assert len(rebuilt.recipient_diagnostics) == 5


@pytest.mark.parametrize("target", ["columns", "geography_config"])
def test_final_io_cannot_change_already_composed_values(actual, monkeypatch, target):
    live, result, _ = actual
    # The first test already ran every real source owner. Here the same detached
    # values isolate the final real preparation I/O fence; no new issuer or
    # source admission is mocked. Any unchecked descriptive replay still has to
    # survive the real original-owner check and final full-value seal.
    values = list(_detached(_values(result)))
    config = None
    if target == "geography_config":
        config = owner.shared.host.survey_budget.geography.AtomicSurveyReconstruction(
            "/invented/support.json",
            "a" * 64,
            (
                ("district", "invented-district"),
                ("population", "invented-population"),
                ("puma", "invented-puma"),
            ),
            1,
        )
        # This is only the option-lifetime fault injection. No support file is
        # opened or admitted, and no geography reconstruction is claimed.
        values[0] = replace(values[0], geography_config_payload=config.to_bytes())
    for module, name, value in (
        (owner.shared, "qualify_current_survey_predictors", values[0]),
        (owner.acs, "qualify_current_acs_income_anchors", values[1]),
        (owner.interest, "qualify_current_asec_interest", values[2]),
        (owner.routing, "qualify_current_asec_income_routing", values[3]),
        (owner.dividend, "qualify_current_asec_dividend", values[4]),
    ):
        monkeypatch.setattr(module, name, lambda *a, _value=value, **kw: _value)
    original = owner._compose
    completed = []

    def compose(*args):
        value = original(*args)
        completed.append(value)
        return value

    monkeypatch.setattr(owner, "_compose", compose)
    open_path = Path.open
    mutated = []

    def open_during_final_io(path, *args, **kwargs):
        stream = open_path(path, *args, **kwargs)
        if completed and not mutated:
            if target == "columns":
                completed[0].donor_columns.iloc[0, 0] += 1
            else:
                object.__setattr__(config, "seed", 2)
            mutated.append(str(path))
        return stream

    # Keep the issuer method itself untouched: replacing it is rightly refused
    # by its own live-code seal before the intended last-I/O fence is reached.
    monkeypatch.setattr(Path, "open", open_during_final_io)
    reason = (
        "FINAL_DERIVED_VALUES_CHANGED"
        if target == "columns"
        else "FINAL_GEOGRAPHY_CONFIG_CHANGED"
    )
    with pytest.raises(ValueError, match=reason):
        owner.qualify_current_property_income_sources(
            live.preparation,
            live.allocated_population,
            live.clone_population,
            geography_config=config,
        )

    assert len(mutated) == 1


def test_unissued_plain_object_refuses_without_source_io():
    with pytest.raises(ValueError, match="PREPARATION_TYPE"):
        owner.qualify_current_property_income_sources(object(), None, None)


def test_constructor_weight_copy_preserves_source_live_identity():
    before = owner.shared.source._live()
    copied = _detached(Weights(np.array([1.0]), WeightKind.DESIGN))
    assert copied.values.tolist() == [1.0]
    assert owner.shared.source._live() == before


def test_unissued_preparation_copy_refuses_before_composition(actual):
    live, _, _ = actual
    before = owner.shared.source._live()
    # Copy only the descriptive payload into an unissued identity. Avoid the
    # unrelated stdlib copy class-cache side effect in this ownership control.
    copied = object.__new__(type(live.preparation))
    object.__setattr__(copied, "payload", live.preparation.payload)
    with pytest.raises(ValueError):
        owner.qualify_current_property_income_sources(
            copied,
            live.allocated_population,
            live.clone_population,
        )
    assert owner.shared.source._live() == before
