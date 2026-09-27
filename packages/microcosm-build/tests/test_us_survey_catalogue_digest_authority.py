"""Real invented preparation owners retain final catalogue/provider checks."""

import sys
from types import SimpleNamespace

import pytest
from test_us_survey_population_preparation import fixture

from microcosm.build.us_runtime import survey_population_preparation as owner


def _must_not_execute(*_args, **_kwargs):
    raise AssertionError("Changed JSON provider executed before owner refusal")


def test_changed_catalogue_json_providers_refuse_before_use(tmp_path, monkeypatch):
    result = owner.prepare_authenticated_survey_population(
        **fixture(tmp_path, monkeypatch)
    )
    encoder = owner.json.JSONEncoder
    module = owner.json.encoder
    mutations = [
        (owner.json, "JSONEncoder", type("ChangedJSONEncoder", (encoder,), {})),
        (owner.json, "encoder", SimpleNamespace(**vars(module))),
        (module, "c_make_encoder", _must_not_execute),
        (module, "encode_basestring", _must_not_execute),
    ]
    for name in ("__init__", "encode", "iterencode"):
        function = getattr(encoder, name)
        mutations.extend(
            [
                (encoder, name, _must_not_execute),
                (function, "__code__", _must_not_execute.__code__),
                (function, "__defaults__", ("changed-catalogue-default",)),
                (function, "__kwdefaults__", {"changed_catalogue_default": True}),
            ]
        )
    assert len(mutations) == 16
    assert owner._live() == owner._LIVE
    observed = 0
    for target, name, changed in mutations:
        before = getattr(target, name)
        with monkeypatch.context() as change:
            change.setattr(target, name, changed)
            assert getattr(target, name) is changed
            assert owner._live() != owner._LIVE
            with pytest.raises(
                owner.SurveyPopulationPreparationError,
                match="^FINAL_AUTHORITY_CHANGED$",
            ):
                result.checked_view()
            observed += 1
        assert getattr(target, name) is before
        assert owner._live() == owner._LIVE
    assert observed == 16
    result.checked_view()


@pytest.mark.parametrize("field", ["records", "vacancies", "c_make_encoder"])
def test_late_catalogue_or_provider_change_refuses_current_borrow(
    tmp_path, monkeypatch, field
):
    result = owner.prepare_authenticated_survey_population(
        **fixture(tmp_path, monkeypatch)
    )
    state = owner._ISSUED[id(result)][2]
    catalogue = owner.acs_catalogue._lookup(state.catalogues[0])
    if field == "c_make_encoder":
        target = owner.json.encoder
        original = target.c_make_encoder
        assert original is not None
        replacement = _must_not_execute
        reason = "FINAL_AUTHORITY_CHANGED"
    else:
        target = catalogue
        original = getattr(catalogue, field)
        assert original
        records = list(original)
        record = list(records[-1])
        record[3] += "0"
        records[-1] = tuple(record)
        replacement = tuple(records)
        reason = "NESTED_EVIDENCE_CHANGED"
    fired = []

    def observe(frame, event, _argument):
        if (
            event == "return"
            and frame.f_code is owner._producer.__code__
            and frame.f_back is not None
            and frame.f_back.f_code is owner._validate.__code__
            and not fired
        ):
            fired.append(True)
            if field == "c_make_encoder":
                setattr(target, field, replacement)
            else:
                object.__setattr__(target, field, replacement)

    previous = sys.getprofile()
    try:
        sys.setprofile(observe)
        with pytest.raises(
            owner.SurveyPopulationPreparationError, match="^" + reason + "$"
        ):
            result.checked_view()
        assert fired == [True]
    finally:
        sys.setprofile(previous)
        if field == "c_make_encoder":
            setattr(target, field, original)
        else:
            object.__setattr__(target, field, original)
    assert getattr(target, field) is original
    result.checked_view()
