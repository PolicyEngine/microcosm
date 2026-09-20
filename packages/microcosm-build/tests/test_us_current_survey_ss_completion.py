"""Real invented source custody and pure SS report completion contracts."""

import sys

import numpy as np
import pandas as pd
import pytest
from ss_report_source_fixture import ss_report_source_arguments

from microcosm.build.us_runtime import current_survey_ss_completion as values


@pytest.fixture(scope="module")
def qualified_source(tmp_path_factory):
    root = tmp_path_factory.mktemp("ss-report-source")
    with pytest.MonkeyPatch.context() as patch:
        arguments = ss_report_source_arguments(root, patch)
        preparation = values.source.prepare_authenticated_survey_population(**arguments)
        entry = preparation._checked()
        before = values.source._frame_identity(entry[2].frame)
        qualified = values.qualify_current_survey_ss_model_inputs(preparation)
        yield preparation, qualified, arguments, before


def test_genuine_full_basis_includes_omitted_resolved_design_donors(qualified_source):
    preparation, q, arguments, before = qualified_source
    full = q.full_source
    assert tuple(full.asec_basis.index) == tuple(range(105, 113))
    assert q.evidence["resolved_positive_donors"] == 4
    assert q.evidence["full_original_asec_rows"] == 8
    selected = q.originals.loc[q.originals.source.eq("asec"), "native_person_id"]
    assert set(selected) == {107, 108, 111, 112}
    labels = q.donor_columns[values.LABEL]
    assert labels.loc[105] == "social_security_retirement"
    assert labels.loc[109] == "social_security_disability"
    assert labels.loc[110] == "social_security_dependents"
    assert labels.loc[111] == "social_security_survivors"
    assert {105, 109, 110}.isdisjoint(selected)
    donor_weights = full.asec_frame.resolve_weights("person")
    assert donor_weights.kind.value == "design"
    by_id = pd.Series(donor_weights.values, index=full.asec_basis.index)
    assert by_id.loc[105] == by_id.loc[109] == by_id.loc[110] == 2552.12
    assert by_id.loc[111] == 100
    assert q.evidence["class_design_weight_mass"] == {
        "social_security_retirement": 2552.12,
        "social_security_disability": 2552.12,
        "social_security_dependents": 2552.12,
        "social_security_survivors": 100.0,
    }
    # Qualifying raw household coverage agrees with all eight current persons.
    raw = pd.read_csv(arguments["source_dir"] / "asec/hhpub25.csv")
    assert dict(zip(raw.H_SEQ, raw.H_NUMPER, strict=True)) == {7: 4, 8: 4}
    assert full.asec_frame.person.groupby("person_household_id").size().tolist() == [
        4,
        4,
    ]
    person = q.originals
    native = person.set_index("native_person_id")
    asec = native.loc[selected.to_numpy()]
    assert asec.loc[107, list(values.basis.COMPONENTS)].isna().all()
    assert not asec.loc[107, "allowed_social_security_retirement"]
    assert asec.loc[108, list(values.basis.COMPONENTS)].isna().all()
    assert np.isnan(asec.loc[108, "social_security_source_total"])
    assert (asec.loc[112, list(values.basis.COMPONENTS)] == 0).all()
    assert asec.loc[112, "social_security_source_total"] == 0
    _, needed = values._report_selection(person)
    assert q.recipient_features.index.equals(person.index[needed])
    recipient = person.loc[q.recipient_features.index]
    assert set(recipient.source) == {"asec", "acs"}
    assert recipient.loc[recipient.source.eq("asec"), "native_person_id"].tolist() == [
        107
    ]
    assert tuple(q.recipient_features.columns) == values.FEATURES
    assert np.isfinite(q.recipient_features.to_numpy()).all()
    assert q.matrix is not None
    assert not q.evidence["individual_beneficiary_assignment_claim"]
    assert q.evidence["scientific_qualification"] == "pending"
    assert not q.evidence["source_admission_issued"]
    assert not q.evidence["release_eligible"]
    selected_again = values.observed.qualify_current_social_security(preparation)
    pd.testing.assert_frame_equal(full.selected.person, selected_again.person)
    assert values.source._frame_identity(preparation._checked()[2].frame) == before


@pytest.mark.parametrize("change", ["feature", "money", "reason", "allocation"])
def test_live_configuration_rebinding_refuses_before_source_work(
    qualified_source, monkeypatch, change
):
    preparation, _, _, _ = qualified_source
    if change == "feature":
        monkeypatch.setattr(values, "FEATURES", values.FEATURES[::-1])
    elif change == "money":
        monkeypatch.setattr(values, "MONEY_FIELDS", values.MONEY_FIELDS[::-1])
    elif change == "reason":
        monkeypatch.setitem(values.basis.REASON_COMPONENTS, 7, (0,))
    else:
        monkeypatch.setattr(
            values.observed, "READ_COLUMNS", values.observed.READ_COLUMNS[::-1]
        )
    with pytest.raises(ValueError, match="IMPLEMENTATION_CHANGED"):
        values.qualify_current_survey_ss_model_inputs(preparation)


def test_final_source_callback_cannot_mutate_returned_predictor_values(
    qualified_source,
):
    preparation, _, _, _ = qualified_source
    previous = sys.getprofile()
    changed = []

    def mutate(frame, event, result):
        if previous is not None:
            previous(frame, event, result)
        caller = frame.f_back
        if (
            event == "return"
            and frame.f_code is values.observed._file_sha.__code__
            and caller is not None
            and caller.f_code is values.qualify_current_survey_ss_model_inputs.__code__
            and "result" in caller.f_locals
            and not changed
        ):
            caller.f_locals["result"].recipient_features.iloc[0, 0] += 1
            changed.append(True)

    try:
        sys.setprofile(mutate)
        with pytest.raises(ValueError, match="FINAL_VALUES"):
            values.qualify_current_survey_ss_model_inputs(preparation)
    finally:
        sys.setprofile(previous)
    assert changed == [True]


def test_literal_source_edit_revokes_model_input_qualification(qualified_source):
    preparation, _, arguments, _ = qualified_source
    path = arguments["source_dir"] / "asec/pppub25.csv"
    original = path.read_bytes()
    raw = pd.read_csv(path, dtype=str, keep_default_na=False)
    row = raw.PERIDNUM.eq(str(9).zfill(22))
    assert row.sum() == 1 and raw.loc[row, "RESNSS1"].tolist() == ["2"]
    raw.loc[row, "RESNSS1"] = "1"
    try:
        raw.to_csv(path, index=False)
        with pytest.raises(ValueError):
            values.qualify_current_survey_ss_model_inputs(preparation)
    finally:
        path.write_bytes(original)


def test_detached_data_cannot_establish_model_source_authority():
    with pytest.raises(ValueError, match="PREPARATION_TYPE"):
        values.qualify_current_survey_ss_model_inputs({"source_authenticated": True})


def reports():
    """Numeric contract only; this table makes no source qualification claim."""
    table = pd.DataFrame(
        {
            "social_security_source_total": [1200.0, 1200.0, 0.0, np.nan, 1200.0],
            "source_reporting_universe": [True, True, True, False, True],
        },
        index=pd.Index([1, 2, 3, 4, 5], dtype="int64", name="person_id"),
    )
    components = np.array(
        [[1200, 0, 0, 0], [np.nan] * 4, [0] * 4, [np.nan] * 4, [np.nan] * 4]
    )
    allowed = np.array(
        [[True] * 4, [False, True, True, True], [True] * 4, [False] * 4, [True] * 4]
    )
    for i, column in enumerate(values.basis.COMPONENTS):
        table[column] = components[:, i]
        table["allowed_" + column] = allowed[:, i]
    return table


def scores():
    return pd.DataFrame(
        [[0.25, 0.25, 0.25, 0.25], [0.1, 0.2, 0.3, 0.4]],
        index=pd.Index([2, 5], dtype="int64", name="person_id"),
        columns=values.basis.COMPONENTS,
        dtype="float64",
    )


def test_completion_preserves_source_bits_and_uses_reason_conditional_mean():
    table, probabilities = reports(), scores()
    before = values._table(table)
    out = values.complete_reports(table, probabilities)
    np.testing.assert_array_equal(out.loc[2], [0, 400, 400, 400])
    np.testing.assert_array_equal(out.loc[5], [120, 240, 360, 480])
    assert out.loc[[1, 2, 3, 5]].sum(axis=1).tolist() == [1200, 1200, 0, 1200]
    assert values._bits(out.loc[[1, 3, 4]].to_numpy()) == values._bits(
        table.loc[[1, 3, 4], list(values.basis.COMPONENTS)].to_numpy()
    )
    assert values._table(table) == before
    assert probabilities.loc[2].tolist() == [0.25] * 4


@pytest.mark.parametrize(
    "defect",
    [
        "axis",
        "classes",
        "dtype",
        "nan",
        "negative",
        "sum",
        "no_allowed_mass",
        "partial_source",
    ],
)
def test_invalid_probability_or_source_contract_refuses(defect):
    table, probability = reports(), scores()
    if defect == "axis":
        probability = probability.iloc[::-1]
    elif defect == "classes":
        probability = probability.loc[:, list(values.basis.COMPONENTS[::-1])]
    elif defect == "dtype":
        probability = probability.astype("float32")
    elif defect == "nan":
        probability.iloc[0, 0] = np.nan
    elif defect == "negative":
        probability.iloc[0, 0] = -0.1
    elif defect == "sum":
        probability.iloc[0, 0] = 0
    elif defect == "no_allowed_mass":
        probability.loc[2] = [1, 0, 0, 0]
    else:
        table.loc[2, values.basis.COMPONENTS[0]] = 1
    with pytest.raises(ValueError):
        values.complete_reports(table, probability)


@pytest.mark.parametrize("method", ["__init__", "__setattr__"])
def test_qualified_projection_callable_rebinding_refuses_before_invocation(
    monkeypatch, method
):
    called = []

    def replacement(*args, **kwargs):
        called.append(True)
        raise AssertionError("rebound projection constructor reached")

    monkeypatch.setattr(values.QualifiedSSModelInputs, method, replacement)
    with pytest.raises(ValueError, match="IMPLEMENTATION_CHANGED"):
        values.qualify_current_survey_ss_model_inputs(object())
    assert not called
