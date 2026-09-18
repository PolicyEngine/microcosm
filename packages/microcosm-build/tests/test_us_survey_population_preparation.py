"""Actual complete source catalogues and native issuers over invented bytes."""

import copy
import json
import os
import shutil
import sys
from fractions import Fraction
from types import SimpleNamespace

import pandas as pd
import pytest
from test_us_acs_person_coverage_authentication import (
    Member,
    _csv,
    _household,
    _person,
    build_fixture,
)
from test_us_asec_2024_native_population import _fixture as asec_fixture

from microcosm.build.us_runtime import survey_population_preparation as owner
from microcosm.frame import WeightKind


def fixture(
    tmp_path,
    monkeypatch,
    fraction=Fraction(1),
    seed=41,
    *,
    zero=True,
    zero_key=None,
    missing_asec_money=None,
    current_predictor_money=None,
    acs_ssp_values=None,
):
    asec_root = tmp_path / "asec-original"
    asec_root.mkdir()
    paths = asec_fixture(
        asec_root,
        monkeypatch,
        extra_household=True,
        missing_money=missing_asec_money,
        current_predictor_money=current_predictor_money,
        tokens=("2", "1", "2", "1", "2", "1"),
        household_rows=None
        if zero and zero_key is None
        else [
            [
                "00007",
                "1",
                "0" if zero_key == "00007" else "000255212",
                "6",
                "1",
                "2",
            ],
            [
                "00008",
                "1",
                "0" if zero_key == "00008" else "000010000",
                "6",
                "1",
                "2",
            ],
        ],
    )
    monkeypatch.setattr(
        owner.acs_native.housing.shutil,
        "disk_usage",
        lambda _: SimpleNamespace(free=64 * 1024**3),
    )
    households = [
        _household("2024HU0000001", NP=2),
        _household("2024HU0000002", NP=1),
        _household("2024GQ0000001", NP=1, TYPEHUGQ=2, WGTP=0, TEN=""),
        _household("2024GQ0000002", NP=1, TYPEHUGQ=3, WGTP=0, TEN=""),
        _household("2024HU0000003", NP=0, TEN=""),
    ]
    people = [
        _person("2024HU0000001", 1, 20, AGEP=30, MIL="1", ESR="4"),
        _person("2024HU0000001", 2, 25, AGEP=15, MIL="", ESR=""),
        _person("2024HU0000002", 1, 20, AGEP=80, MIL="4", ESR="6"),
        _person("2024GQ0000001", 1, 37, AGEP=40, MIL="4", ESR="6", PWGTP=77),
        _person("2024GQ0000002", 1, 38, AGEP=50, MIL="4", ESR="6", PWGTP=78),
    ]
    if acs_ssp_values is not None:
        seen = set()
        for person in people:
            key = (person["SERIALNO"], person["SPORDER"])
            if key in acs_ssp_values:
                person["SSP"] = acs_ssp_values[key]
                seen.add(key)
        assert seen == set(acs_ssp_values)
    acs = build_fixture(
        tmp_path / "acs-original",
        monkeypatch,
        household_members=(Member("psam_husa.csv", _csv(households)),),
        person_members=(Member("psam_pusa.csv", _csv(people)),),
    )
    source = tmp_path / "combined-source"
    source.mkdir()
    (source / "acs").mkdir()
    (source / "asec").mkdir()
    for name in ("csv_hus.zip", "csv_pus.zip"):
        shutil.copyfile(acs.source_dir / name, source / "acs" / name)
    for argument, name in (
        ("parent_path", "parent.h5"),
        ("household_attachment_path", "household-attachment.h5"),
        ("person_income_attachment_path", "person-income-attachment.h5"),
        ("household_member_path", "hhpub25.csv"),
    ):
        shutil.copyfile(paths[argument], source / "asec" / name)
    for year, path in paths["person_member_paths"].items():
        shutil.copyfile(path, source / "asec" / f"pppub{year - 1999}.csv")
    request = {
        "protocol": owner.REQUEST_PROTOCOL,
        "declaration": owner.domains.DECLARATION,
        "fraction": [fraction.numerator, fraction.denominator],
        "seed": seed,
    }
    (source / "selection-request.json").write_bytes(
        json.dumps(request, sort_keys=True, separators=(",", ":")).encode()
    )
    snapshots = tmp_path / "captures"
    snapshots.mkdir()
    return dict(
        source_dir=source, snapshot_root=snapshots, fraction=fraction, seed=seed
    )


def test_real_sources_prepare_once_without_allocation(tmp_path, monkeypatch):
    arguments = fixture(tmp_path, monkeypatch)
    result = owner.prepare_authenticated_survey_population(**arguments)
    view = result.checked_view()
    frame = view.frame
    assert frame.weights_for("household").kind is WeightKind.DESIGN
    assert frame.n("household") == 6 and frame.n("person") == 9
    assert not view.receipt["release_eligible"]
    assert len(view.receipt["origins"]["households"]) == 6
    assert any(
        row["original_anchor"] == [0, 1]
        for row in view.receipt["origins"]["households"]
    )
    assert view.receipt["catalogues"]["acs"]["counts"]["vacancies"] == 1
    assert view.selection_plan.source_authenticated is False
    assert view.receipt["protocol"] == "microcosm.us.survey-population-preparation.v2"
    assert owner.REQUEST_PROTOCOL == "microcosm.us.survey-population-request.v1"
    assert owner.SOURCE_CODEC == "us-survey-population-source-v1"
    state = owner._ISSUED[id(result)][2]
    native_acs, native_asec = state.source_frames
    known_acs_ages = {
        ("2024HU0000001", "1"): 30,
        ("2024HU0000001", "2"): 15,
        ("2024HU0000002", "1"): 80,
        ("2024GQ0000001", "1"): 40,
        ("2024GQ0000002", "1"): 50,
    }
    acs_households = native_acs.table("household").set_index("household_id")
    acs_serials = native_acs.person["person_household_id"].map(
        acs_households["SERIALNO"]
    )
    assert not acs_serials.isna().any()
    acs_keys = list(
        zip(
            acs_serials.map(str),
            native_acs.person["SPORDER"].map(str),
            strict=True,
        )
    )
    assert len(acs_keys) == len(known_acs_ages) and set(acs_keys) == set(known_acs_ages)
    expected_acs = [known_acs_ages[key] for key in acs_keys]
    known_asec_ages = {
        str(5).zfill(22): 55,
        str(6).zfill(22): 14,
        str(7).zfill(22): 55,
        str(8).zfill(22): 14,
    }
    asec_keys = native_asec.person["PERIDNUM"].map(str).tolist()
    assert len(asec_keys) == len(known_asec_ages) and set(asec_keys) == set(
        known_asec_ages
    )
    expected_asec = [known_asec_ages[key] for key in asec_keys]
    assert native_acs.person["AGEP"].map(str).tolist() == list(map(str, expected_acs))
    assert native_acs.person["A_AGE"].tolist() == expected_acs
    assert native_asec.person["A_AGE"].tolist() == expected_asec
    assert frame.person["A_AGE"].tolist() == expected_acs + expected_asec
    assert frame.person["age"].tolist() == expected_acs + expected_asec
    assert not frame.person["age"].isna().any()
    assert "age" not in native_asec.person  # The original ASEC owner stays raw.
    evidence = view.receipt["origins"]["observed_age_normalization"]
    assert evidence["rule"] == owner.observed_age.rule_document()
    assert evidence["sources"]["acs"]["common_age_preexisting"] is True
    assert evidence["sources"]["asec"]["common_age_preexisting"] is False
    for index, channel in enumerate(("acs", "asec")):
        assert (
            evidence["sources"][channel]["native_frame_sha256"]
            == state.source_frame_seals[index]
            == view.receipt["native"][channel]["frame_sha256"]
        )

    # Copy comparisons use the real issued source Frame, with no new source read.
    normalized = owner._normalized_source_copy(native_asec)
    owner._verify_normalized_copy(native_asec, normalized)
    normalized.person["A_AGE"] = normalized.person["A_AGE"] + 1
    assert normalized.person["A_AGE"].iloc[0] == 56
    assert native_asec.person["A_AGE"].iloc[0] == 55
    with pytest.raises(
        owner.SurveyPopulationPreparationError, match="NORMALIZED_SOURCE_STORAGE"
    ):
        owner._verify_normalized_copy(native_asec, normalized)
    normalized = owner._normalized_source_copy(native_asec)
    normalized.person["age"] = normalized.person["age"] + 1
    assert normalized.person["age"].iloc[0] == 56
    with pytest.raises(
        owner.SurveyPopulationPreparationError, match="NORMALIZED_AGE_IDENTITY"
    ):
        owner._verify_normalized_copy(native_asec, normalized)
    for change in ("column_name", "column_dtype", "strata_name"):
        normalized = owner._normalized_source_copy(native_asec)
        if change == "column_name":
            normalized.person.columns = normalized.person.columns.rename("changed")
        elif change == "column_dtype":
            dtype = "string" if normalized.person.columns.dtype == object else object
            normalized.person.columns = pd.Index(normalized.person.columns, dtype=dtype)
        else:
            normalized.strata.name = "changed"
        with pytest.raises(owner.SurveyPopulationPreparationError, match="NORMALIZED_"):
            owner._verify_normalized_copy(native_asec, normalized)

    # A temporary rule/callable change cannot borrow this issued preparation.
    assert owner.observed_age in owner._modules()
    assert owner.graph_population in owner._modules()
    for module, name, changed in (
        (owner.observed_age, "RULE", "changed"),
        (owner.observed_age, "AGE_CONVENTION", "changed"),
        (owner.observed_age, "MAX_ROWS", 1),
        (owner.observed_age, "MAX_EXACT_FLOAT64_INTEGER", 1),
        (owner.observed_age, "normalize_observed_age", lambda *args: None),
        (owner.graph_population, "_storage_parts", lambda *args: None),
    ):
        with monkeypatch.context() as local:
            local.setattr(module, name, changed)
            with pytest.raises(
                owner.SurveyPopulationPreparationError, match="AUTHORITY_CHANGED"
            ):
                owner.verify_survey_population_preparation(result)
    owner.verify_materialized_survey_population(result, frame)


def test_single_selection_uses_complete_n_and_exact_fraction(tmp_path, monkeypatch):
    arguments = fixture(tmp_path, monkeypatch, fraction=Fraction(2, 3), zero=False)
    calls = []

    def trace(frame, event, arg):
        if (
            event == "call"
            and frame.f_code
            is owner.survey_domain_sample.select_domain_households.__code__
        ):
            calls.append(True)

    previous = sys.getprofile()
    sys.setprofile(trace)
    try:
        result = owner.prepare_authenticated_survey_population(**arguments)
        view = result.checked_view()
        owner.verify_survey_population_preparation(result)
    finally:
        sys.setprofile(previous)
    assert len(calls) == 1
    cell = next(
        c
        for c in view.selection_plan.cells
        if c.source is owner.domains.Source.ASEC
        and c.domain is owner.domains.Domain.SHARED_HOUSING
    )
    assert (
        cell.eligible_households,
        cell.selected_households,
        cell.inclusion_probability,
    ) == (2, 1, Fraction(1, 2))
    assert view.frame.n("household") == 5
    assert view.receipt["native"]["asec"]["households"] == 1
    assert view.receipt["native"]["asec"]["persons"] == 2
    asec_people = [
        r for r in view.receipt["origins"]["persons"]["rows"] if r[1] == "asec"
    ]
    assert len(asec_people) == 2 and len({r[-1] for r in asec_people}) == 1
    asec = next(
        r for r in view.receipt["origins"]["households"] if r["source"] == "asec"
    )
    assert asec["raw_native_id"] in {"00007", "00008"}
    assert all(r["source_year"] == 2024 for r in view.receipt["origins"]["households"])


def test_replay_reconstructs_sources_and_rejects_changed_candidate(
    tmp_path, monkeypatch
):
    arguments = fixture(tmp_path, monkeypatch)
    first = owner.prepare_authenticated_survey_population(**arguments)
    payload = first.to_bytes()
    assert (
        json.loads(payload)["protocol"]
        == "microcosm.us.survey-population-preparation.v2"
    )
    replay = owner.prepare_authenticated_survey_population(
        **arguments, candidate=payload
    )
    assert replay is not first and replay.to_bytes() == payload
    document = json.loads(payload)
    document["selection"]["cells"][0]["share"] = [999, 1]
    calls = []

    def trace(frame, event, arg):
        if (
            event == "call"
            and frame.f_code
            is owner.asec_native.load_authenticated_asec_2024_native_population.__code__
        ):
            calls.append(True)

    previous = sys.getprofile()
    sys.setprofile(trace)
    try:
        with pytest.raises(
            owner.SurveyPopulationPreparationError, match="CANDIDATE_MISMATCH"
        ):
            owner.prepare_authenticated_survey_population(
                **arguments, candidate=owner._encode(document)
            )
    finally:
        sys.setprofile(previous)
    assert calls == [True]


def test_selected_zero_support_refuses_without_redraw_or_native_issuance(
    tmp_path, monkeypatch
):
    # Direct the invented source weights before pinning. The fixed seed is not
    # searched or retried; this fixture-design draw is outside source issuance.
    fraction, seed = Fraction(2, 3), 41
    domain = owner.domains.Domain.SHARED_HOUSING.value
    control = owner.survey_domain_sample.select_domain_households(
        row_ids=("00007", "00008"),
        source_channels=("asec", "asec"),
        domain_keys=(domain, domain),
        cells=((domain, "asec"),),
        fraction=fraction,
        seed=seed,
    )[0]
    assert control.eligible_households == 2 and len(control.positions) == 1
    selected_key = ("00007", "00008")[control.positions[0]]
    arguments = fixture(tmp_path, monkeypatch, fraction, seed, zero_key=selected_key)
    calls = {"sampler": 0, "acs_native": 0, "asec_native": 0}
    codes = {
        owner.survey_domain_sample.select_domain_households.__code__: "sampler",
        owner.acs_native.issue_acs_native_coverage.__code__: "acs_native",
        owner.asec_native.load_authenticated_asec_2024_native_population.__code__: "asec_native",
    }
    refusals = []

    def profile(frame, event, arg):
        if event == "call" and frame.f_code in codes:
            calls[codes[frame.f_code]] += 1

    def trace(frame, event, arg):
        if event == "exception" and frame.f_code is owner.selection._require.__code__:
            refusals.append(str(arg[1]))
        return trace

    old_profile, old_trace = sys.getprofile(), sys.gettrace()
    sys.setprofile(profile)
    sys.settrace(trace)
    try:
        with pytest.raises(owner.SurveyPopulationPreparationError):
            owner.prepare_authenticated_survey_population(**arguments)
    finally:
        sys.setprofile(old_profile)
        sys.settrace(old_trace)
    assert "SELECTED_CELL_HAS_NO_POSITIVE_ANCHOR" in refusals
    assert calls == {"sampler": 1, "acs_native": 0, "asec_native": 0}


@pytest.mark.parametrize("kind", ["wrapped_code", "closure", "default", "kwdefault"])
def test_callable_configuration_drift_refuses_before_producer_io(monkeypatch, kind):
    if kind == "wrapped_code":
        monkeypatch.setattr(
            owner._regular_reader.__wrapped__, "__code__", (lambda p, n: None).__code__
        )
    elif kind == "closure":
        monkeypatch.setattr(
            owner._regular_reader.__closure__[0], "cell_contents", lambda p, n: None
        )
    elif kind == "default":
        monkeypatch.setattr(owner._check_scalars, "__defaults__", (999,))
    else:
        monkeypatch.setitem(
            owner.prepare_authenticated_survey_population.__kwdefaults__,
            "candidate",
            b"substitution",
        )
    calls = []

    def profile(frame, event, arg):
        if event == "call" and frame.f_code is owner._code_bytes.__code__:
            calls.append(True)

    previous = sys.getprofile()
    sys.setprofile(profile)
    try:
        with pytest.raises(
            owner.SurveyPopulationPreparationError, match="PRODUCER_CHANGED"
        ):
            owner._producer()
    finally:
        sys.setprofile(previous)
    assert calls == []


def test_defensive_view_forgery_and_materialized_copy(tmp_path, monkeypatch):
    result = owner.prepare_authenticated_survey_population(
        **fixture(tmp_path, monkeypatch)
    )
    view = result.checked_view()
    view.receipt["release_eligible"] = True
    assert result.checked_view().receipt["release_eligible"] is False
    for forged in (copy.copy(result), view, view.receipt, view.payload):
        with pytest.raises(owner.SurveyPopulationPreparationError):
            owner.verify_survey_population_preparation(forged)
    materialized = owner._copy_source(view.frame)
    materialized.weights_for("household").values.setflags(write=True)
    owner.verify_materialized_survey_population(result, materialized)
    materialized.person.iloc[0, materialized.person.columns.get_loc("age")] += 1
    with pytest.raises(
        owner.SurveyPopulationPreparationError, match="MATERIALIZED_FRAME_CHANGED"
    ):
        owner.verify_materialized_survey_population(result, materialized)
    object.__setattr__(result, "payload", result.payload + b" ")
    for borrow in (
        lambda: result.frame,
        lambda: result.context,
        lambda: result.selection_plan,
        lambda: result.receipt,
        result.to_bytes,
        result.checked_view,
        lambda: owner.verify_survey_population_preparation(result),
        lambda: owner.verify_materialized_survey_population(result, view.frame),
    ):
        with pytest.raises(owner.SurveyPopulationPreparationError):
            borrow()


@pytest.mark.parametrize(
    "kind",
    ["frame", "plan", "payload", "nested_acs", "nested_asec", "source", "source_extra"],
)
def test_late_mutation_at_final_producer_return_refuses_current_borrow(
    tmp_path, monkeypatch, kind
):
    arguments = fixture(tmp_path, monkeypatch)
    result = owner.prepare_authenticated_survey_population(**arguments)
    state = owner._ISSUED[id(result)][2]
    fired = []

    def trace(frame, event, arg):
        if (
            event == "return"
            and frame.f_code is owner._producer.__code__
            and frame.f_back.f_code is owner._validate.__code__
            and not fired
        ):
            fired.append(True)
            if kind == "source_extra":
                (arguments["source_dir"] / "asec" / "extra.txt").write_text("invented")
            elif kind == "frame":
                state.frame.person.iloc[
                    0, state.frame.person.columns.get_loc("age")
                ] += 1
            elif kind == "plan":
                object.__setattr__(state.plan.selected[0], "share", Fraction(99))
            elif kind == "payload":
                object.__setattr__(result, "payload", result.payload + b" ")
            elif kind == "nested_acs":
                literal = owner.acs_native._owned(state.native[0]).literal
                object.__setattr__(literal, "payload", literal.payload + b" ")
            elif kind == "nested_asec":
                nested = owner.asec_native._ISSUED[id(state.native[1])][2]
                object.__setattr__(
                    nested.coverage, "_body", nested.coverage._body + b"x"
                )
            else:
                path = arguments["source_dir"] / "selection-request.json"
                path.write_bytes(path.read_bytes() + b" ")

    previous = sys.getprofile()
    sys.setprofile(trace)
    try:
        with pytest.raises(owner.SurveyPopulationPreparationError):
            result.checked_view()
    finally:
        sys.setprofile(previous)
    assert fired == [True]


@pytest.mark.parametrize("change", ["extra", "symlink", "missing"])
def test_exact_source_roster_refuses_before_issuers(tmp_path, monkeypatch, change):
    arguments = fixture(tmp_path, monkeypatch)
    source = arguments["source_dir"]
    if change == "extra":
        (source / "extra.json").write_bytes(b"{}")
    elif change == "missing":
        (source / "asec/hhpub25.csv").unlink()
    else:
        path = source / "asec/hhpub25.csv"
        target = source.parent / "member.csv"
        path.rename(target)
        path.symlink_to(target)
    calls = []

    def trace(frame, event, arg):
        if (
            event == "call"
            and frame.f_code is owner.acs_catalogue.issue_acs_source_catalogue.__code__
        ):
            calls.append(True)

    previous = sys.getprofile()
    sys.setprofile(trace)
    try:
        with pytest.raises(owner.SurveyPopulationPreparationError):
            owner.prepare_authenticated_survey_population(**arguments)
    finally:
        sys.setprofile(previous)
    assert not calls


def test_request_changed_between_decode_and_roster_binding_refuses(
    tmp_path, monkeypatch
):
    arguments = fixture(tmp_path, monkeypatch)
    fired = []

    def profile(frame, event, arg):
        if (
            event == "return"
            and frame.f_code is owner._request.__code__
            and frame.f_back.f_code
            is owner.prepare_authenticated_survey_population.__code__
            and not fired
        ):
            fired.append(True)
            path = arguments["source_dir"] / "selection-request.json"
            document = json.loads(path.read_bytes())
            document["seed"] += 1
            path.write_bytes(owner._encode(document))

    previous = sys.getprofile()
    sys.setprofile(profile)
    try:
        with pytest.raises(owner.SurveyPopulationPreparationError):
            owner.prepare_authenticated_survey_population(**arguments)
    finally:
        sys.setprofile(previous)
    assert fired == [True]


@pytest.mark.parametrize(
    "change", ["duplicate", "whitespace", "unreduced", "bool_seed", "extra", "oversize"]
)
def test_request_is_canonical_and_bounded(tmp_path, change):
    request = {
        "protocol": owner.REQUEST_PROTOCOL,
        "declaration": owner.domains.DECLARATION,
        "fraction": [1, 2],
        "seed": 41,
    }
    if change == "unreduced":
        request["fraction"] = [2, 4]
    elif change == "bool_seed":
        request["seed"] = True
    elif change == "extra":
        request["extra"] = 0
    payload = json.dumps(request, sort_keys=True, separators=(",", ":")).encode()
    if change == "duplicate":
        payload = payload.replace(b'"seed":41', b'"seed":41,"seed":41')
    elif change == "whitespace":
        payload += b"\n"
    elif change == "oversize":
        payload = b"x" * (owner.MAX_REQUEST_BYTES + 1)
    (tmp_path / "selection-request.json").write_bytes(payload)
    with pytest.raises(owner.SurveyPopulationPreparationError):
        owner.read_survey_population_request(tmp_path)


@pytest.mark.parametrize("change", ["fifo", "symlink"])
def test_regular_reader_refuses_replacement_without_blocking(
    tmp_path, monkeypatch, change
):
    path = tmp_path / "input.txt"
    path.write_bytes(b"invented")
    target = tmp_path / "other.txt"
    target.write_bytes(b"different")
    actual_open = os.open
    fired = []

    def replace_at_open(value, flags, *args, **kwargs):
        if value == path and not fired:
            fired.append(True)
            path.unlink()
            if change == "fifo":
                os.mkfifo(path)
            else:
                path.symlink_to(target)
        return actual_open(value, flags, *args, **kwargs)

    monkeypatch.setattr(os, "open", replace_at_open)
    with pytest.raises((owner.SurveyPopulationPreparationError, OSError)):
        with owner._regular_reader(path, 100) as (stream, _info):
            pytest.fail("Replacement must be refused before reading")
    assert fired == [True]


def test_transport_bound_precedes_append():
    rows = []
    budget = [owner.MAX_PAYLOAD_BYTES - 1]
    with pytest.raises(owner.SurveyPopulationPreparationError, match="ORIGIN_LIMIT"):
        owner._bounded_append(rows, [1, "acs", 2], budget)
    assert rows == []
    with pytest.raises(owner.SurveyPopulationPreparationError, match="PAYLOAD_LIMIT"):
        owner._encode({"rows": ["invented"] * 100}, 30)
    with pytest.raises(owner.SurveyPopulationPreparationError, match="SCALAR_LIMIT"):
        owner._encode({"nested": ["x" * (owner._MAX_SCALAR_BYTES + 1)]})


def test_semantic_identity_includes_column_and_strata_indexes(tmp_path, monkeypatch):
    result = owner.prepare_authenticated_survey_population(
        **fixture(tmp_path, monkeypatch)
    )
    view = result.checked_view()
    candidate = owner._copy_source(view.frame)
    expected = owner._frame_identity(candidate)
    candidate.person.columns.name = "different"
    assert owner._frame_identity(candidate) != expected
    candidate.person.columns.name = None
    candidate.strata.index.name = "different"
    assert owner._frame_identity(candidate) != expected
