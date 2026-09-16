"""Invented numerical partitions and real original-source financial ownership."""

import hashlib
import shutil
import sys
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from test_us_acs_person_coverage_authentication import (
    Member,
    _csv,
    _household,
    _person,
    build_fixture,
)
from test_us_graph_atomic_survey_population import _support_payload
from test_us_puf55_survey_ss_measurement import _values
from test_us_survey_social_security import _social_security_arguments

from microcosm.build.us_runtime import current_asec_demographics as demographics
from microcosm.build.us_runtime import native_household_origin as origin
from microcosm.build.us_runtime import puf55_survey_recipients as values
from microcosm.frame import US_SCHEMA, Frame, WeightKind, Weights


def _frames():
    """Plain invented Frames for private arithmetic, never source/run issuers."""
    person, units, report = _values()
    person["age"] = 55
    person["prior_wages_not_a_predictor"] = 99_999_999.0
    person["unrelated_social_security_cell"] = np.resize([-0.0, np.nan], len(person))
    for i, name in enumerate(values.financial.values.OUTPUTS):
        person[name] = (person.person_id + i).to_numpy(dtype=np.float64)
    for group in US_SCHEMA.group_entities:
        person["person_" + group + "_id"] = person.person_tax_unit_id
    tables = {
        "person": person,
        "tax_unit": units,
        **{
            g: pd.DataFrame({g + "_id": units.tax_unit_id.to_numpy()})
            for g in US_SCHEMA.group_entities
            if g != "tax_unit"
        },
    }
    for entity, table in tables.items():
        table[values.provenance.spine_source_id_column(entity)] = table[entity + "_id"]
        table[values.provenance.support_channel_column(entity)] = pd.array(
            ["asec"] * len(table), dtype="string"
        )
    cloned = {}
    for entity, table in tables.items():
        arms = []
        for arm in (0, 1):
            copy = table.copy(deep=True)
            copy[values.provenance.support_source_id_column(entity)] = copy[
                entity + "_id"
            ]
            copy[values.provenance.support_clone_index_column(entity)] = np.full(
                len(copy), arm, dtype=np.int64
            )
            copy[entity + "_id"] += 1000 * arm
            if entity == "person":
                for group in US_SCHEMA.group_entities:
                    copy["person_" + group + "_id"] += 1000 * arm
            arms.append(copy)
        cloned[entity] = pd.concat(arms, ignore_index=True)

    def frame(tables):
        return Frame(
            tables,
            US_SCHEMA,
            {
                "household": Weights(
                    np.ones(len(tables["household"])), WeightKind.IMPORTANCE
                )
            },
            pd.Series(["invented"] * len(tables["person"]), dtype="string"),
        )

    return frame(tables), frame(cloned), report


def _project(native, receiving, report):
    measured, _ = values.ss._measure(native.person, native.table("tax_unit"), report)
    return values._project(native, receiving, report, measured)


def test_disjoint_routes_cover_all_clone_one_and_preserve_both_arms():
    native, receiving, report = _frames()
    before = {e: receiving.table(e).copy(deep=True) for e in receiving.entities}
    source_report = report.copy(deep=True)
    # These are complete current financial leaves on an ASEC-only invented
    # surface. Raw ACS WAGP/SEMP columns are neither needed nor manufactured.
    assert not {"WAGP", "SEMP"} & set(receiving.person)
    person, units, matrices, money_evidence = _project(native, receiving, report)
    assert money_evidence["aggregation"] == "sum_all_modeled_tax_unit_members"
    assert money_evidence["raw_acs_universe_requalified_here"] is False
    assert money_evidence["source_admission_issued"] is False
    assert tuple(name for name, _ in matrices) == tuple(
        p.value for p in values.PROFILES
    )
    nine, eight = [
        values.model_input.decode_recipient_matrix(p).features for _, p in matrices
    ]
    assert nine.index.tolist() == [1010, 1020, 1030, 1040]
    assert eight.index.tolist() == [1050]
    assert set(nine.index).isdisjoint(eight.index)
    assert set(nine.index) | set(eight.index) == {1010, 1020, 1030, 1040, 1050}
    assert tuple(nine) == values.PROFILES[0].predictors
    assert tuple(eight) == values.PROFILES[1].predictors
    assert [len(p.targets) for p in values.PROFILES] == [55, 55]
    np.testing.assert_array_equal(nine[values.ss.TOTAL], [100, 500, 400, 0])
    assert np.isnan(units.loc[1050, values.ss.TOTAL])
    # Money deliberately includes dependent earnings; SS deliberately does not.
    assert nine.loc[1010, "puf_predictor_employment_income"] == 1 + 2
    assert nine.loc[1020, "puf_predictor_employment_income"] == 3 + 4 + 5
    assert nine.loc[1030, "puf_2015_filing_status_code"] == 2
    assert nine.loc[1030, "puf_2015_capped_return_size"] == 3
    for arm in (0, 1):
        observed = person.loc[report.index + 1000 * arm].copy()
        observed.index = report.index
        pd.testing.assert_frame_equal(observed, report, check_exact=True)
    for entity, table in before.items():
        pd.testing.assert_frame_equal(receiving.table(entity), table, check_exact=True)
    pd.testing.assert_frame_equal(report, source_report, check_exact=True)


@pytest.mark.parametrize("known", (False, True))
def test_empty_route_omitted_without_empty_matrix_or_unknown_coercion(known):
    native, receiving, report = _frames()
    included = native.person.tax_unit_role_input.isin(("HEAD", "SPOUSE")).to_numpy()
    report.loc[included, "social_security_source_total"] = 0.0 if known else np.nan
    report.loc[included, "source_reporting_universe"] = known
    _, units, matrices, _ = _project(native, receiving, report)
    assert tuple(name for name, _ in matrices) == (
        values.PROFILES[0 if known else 1].value,
    )
    matrix = values.model_input.decode_recipient_matrix(matrices[0][1]).features
    assert len(matrix) == 5 and len(matrix.columns) == (9 if known else 8)
    assert units[values.ss.KNOWN].eq(known).all()
    assert (
        units[values.ss.TOTAL].eq(0).all()
        if known
        else units[values.ss.TOTAL].isna().all()
    )


def test_source_and_receiving_order_are_joined_by_identity():
    native, receiving, report = _frames()
    expected = _project(native, receiving, report)
    for frame in (native, receiving):
        for entity in frame.entities:
            table = frame.table(entity)
            table.iloc[:] = table.iloc[::-1].to_numpy()
    observed = _project(native, receiving, report.iloc[::-1])
    for got, want in zip(observed[:2], expected[:2], strict=True):
        pd.testing.assert_frame_equal(
            got.sort_index(), want.sort_index(), check_exact=True
        )
    for (_, got), (_, want) in zip(observed[2], expected[2], strict=True):
        pd.testing.assert_frame_equal(
            values.model_input.decode_recipient_matrix(got).features.sort_index(),
            values.model_input.decode_recipient_matrix(want).features.sort_index(),
            check_exact=True,
        )


@pytest.mark.parametrize(
    "defect",
    ("role", "membership", "clone", "source", "money_missing", "carried_alias"),
)
def test_projection_refuses_invalid_roles_origin_or_money_before_routing(defect):
    native, receiving, report = _frames()
    person = receiving.person
    if defect == "role":
        person.loc[person.person_id.eq(1001), "tax_unit_role_input"] = "DEPENDENT"
    elif defect == "membership":
        person.loc[person.person_id.eq(1001), "person_tax_unit_id"] = 1020
    elif defect == "clone":
        person.loc[
            person.person_id.eq(1001),
            values.provenance.support_clone_index_column("person"),
        ] = 0
    elif defect == "source":
        person.loc[
            person.person_id.eq(1001),
            values.provenance.spine_source_id_column("person"),
        ] = 999
    elif defect == "money_missing":
        person.loc[person.person_id.eq(1002), "employment_income_before_lsr"] = np.nan
    else:
        person["employment_income"] = 999.0
    with pytest.raises(
        ValueError, match="PUF55_SURVEY_RECIPIENTS_|missing values before coercion"
    ):
        _project(native, receiving, report)


@pytest.mark.parametrize("defect", ("missing_dividend_component", "nonfinite_income"))
def test_current_financial_money_refuses_missing_components_and_nonfinite_values(
    defect,
):
    native, receiving, report = _frames()
    column, value = (
        ("qualified_dividend_income", np.nan)
        if defect == "missing_dividend_component"
        else ("self_employment_income_before_lsr", np.inf)
    )
    receiving.person.loc[receiving.person.person_id.eq(1002), column] = value
    with pytest.raises(ValueError, match="missing values before coercion|nonfinite"):
        _project(native, receiving, report)


def _recipient_source_arguments(root, patch):
    """Extend original SS fixture members before any preparation is issued."""
    args = _social_security_arguments(root, patch, ambiguous=True)
    # One genuine source record is outside the age15 question universe. The
    # real native unit constructor, not this test, assigns its modeled role.
    # Its sole-person GQ tax unit exercises the actual eight-feature route.
    households = [
        _household("2024HU0000001", NP=2),
        _household("2024HU0000002", NP=1),
        _household("2024GQ0000001", NP=1, TYPEHUGQ=2, WGTP=0, TEN=""),
        _household("2024GQ0000002", NP=1, TYPEHUGQ=3, WGTP=0, TEN=""),
        _household("2024HU0000003", NP=0, TEN=""),
    ]
    people = [
        _person("2024HU0000001", 1, 20, AGEP=30, MIL="1", ESR="4", SSP=1200),
        _person("2024HU0000001", 2, 25, AGEP=15, MIL="", ESR=""),
        _person("2024HU0000002", 1, 20, AGEP=80, MIL="4", ESR="6"),
        _person("2024GQ0000001", 1, 37, AGEP=40, MIL="4", ESR="6", PWGTP=77),
        _person(
            "2024GQ0000002",
            1,
            38,
            AGEP=14,
            MAR=5,
            MIL="",
            ESR="",
            PWGTP=78,
            WAGP=None,
            SEMP=None,
            SSP=None,
            SSIP=None,
            RETP=None,
            INTP=None,
        ),
    ]
    acs = build_fixture(
        root / "recipient-acs-original",
        patch,
        household_members=(Member("psam_husa.csv", _csv(households)),),
        person_members=(Member("psam_pusa.csv", _csv(people)),),
    )
    for name in ("csv_hus.zip", "csv_pus.zip"):
        shutil.copyfile(acs.source_dir / name, args["source_dir"] / "acs" / name)
    asec = args["source_dir"] / "asec"
    # Use the maintained literal-source owner directly, not a ready() substitute.
    from microcosm.build.us_runtime import asec_person_income_source as restoration

    members, pins = {}, []
    for year, member, archive, *_ in values.ss.reports.coverage._MEMBER_PINS:
        path = asec / f"pppub{year - 1999}.csv"
        table = pd.read_csv(path, dtype=str, keep_default_na=False)
        lines = table.A_LINENO.map(int)
        table["A_SEX"], table["AXSEX"] = (
            lines.map({1: "1", 2: "2"}),
            lines.map({1: "0", 2: "4"}),
        )
        table["A_EXPRRP"], table["P_SEQ"] = lines.map({1: "1", 2: "5"}), table.A_LINENO
        table.iloc[::-1].to_csv(path, index=False)
        payload = path.read_bytes()
        pins.append(
            (
                year,
                member,
                archive,
                hashlib.sha256(payload).hexdigest(),
                len(table),
                len(payload),
            )
        )
        members[year] = path
    for module in (values.ss.reports.coverage, restoration, demographics.demographic):
        patch.setattr(module, "_MEMBER_PINS", tuple(pins))
    output = root / "recipient-restored-money"
    restoration.restore_asec_person_income_source(
        asec / "parent.h5",
        asec / "household-attachment.h5",
        member_paths=members,
        output_dir=output,
    )
    shutil.copyfile(
        output / restoration.CHECKPOINT_FILENAME, asec / "person-income-attachment.h5"
    )
    path = asec / "hhpub25.csv"
    table = pd.read_csv(path, dtype=str, keep_default_na=False)
    table["GESTFIPS"] = table.H_SEQ.map({"00007": "06", "00008": "36"})
    table.iloc[::-1].to_csv(path, index=False)
    payload = path.read_bytes()
    patch.setattr(
        origin,
        "_ASEC_MEMBER_PINS",
        tuple(
            replace(
                pin,
                member_sha256=hashlib.sha256(payload).hexdigest(),
                size_bytes=len(payload),
            )
            for pin in origin._ASEC_MEMBER_PINS
        ),
    )
    return args


@pytest.fixture(scope="module")
def recipient_financial_run(tmp_path_factory):
    """One real financial execution shared across bounded source/graph checks."""
    root = tmp_path_factory.mktemp("puf55-recipient-financial")
    with pytest.MonkeyPatch.context() as patch:
        args = _recipient_source_arguments(root, patch)
        payload, source_ids = _support_payload()
        path = root / "invented-support.npz"
        path.write_bytes(payload)
        config = values.financial.reconstruction.AtomicSurveyReconstruction(
            support_path=str(path),
            support_sha256=hashlib.sha256(payload).hexdigest(),
            source_ids=tuple(sorted(source_ids.items())),
            seed=17,
        )
        run = values.financial.run_atomic_survey_financial(
            **args,
            geography_config=config,
            store_root=root / "store",
            demographic_conditioning=True,
            n_estimators=2,
            return_values=True,
        )
        qualified = values.qualify_puf55_survey_recipients(run)
        yield SimpleNamespace(run=run, qualified=qualified)
        values.financial.check_atomic_survey_financial_run(run)


def test_actual_issued_financial_run_retains_source_reports_and_current_money(
    recipient_financial_run,
):
    case = recipient_financial_run
    run, got = case.run, case.qualified
    assert got.financial_run is run
    evidence = values.financial.codec.decode_json(got.receipt)
    assert evidence["financial_run_sha256"] == values.financial.codec.sha(
        values.financial._run_entry(run)[1]
    )
    assert evidence["current_money_tax_units_reconstructed_here"] is False
    assert evidence["population_changed"] is False
    assert evidence["release_eligible"] is False
    assert tuple(name for name, _ in got.matrices) == tuple(
        p.value for p in values.PROFILES
    )
    fallback = values.model_input.decode_recipient_matrix(got.matrices[1][1]).features
    assert len(fallback) == 1 and len(fallback.columns) == 8
    assert got.tax_unit.loc[fallback.index, values.ss.TOTAL].isna().all()
    assert not got.tax_unit.loc[fallback.index, values.ss.KNOWN].any()
    person = run.financial_population.frame.person
    # Compare both receiving arms against the actual original source issuers,
    # not a second invocation of recipient projection arithmetic.
    preparation = values.financial._run_entry(run)[2].prefix.preparation
    report = values.ss.reports.qualify_current_social_security(preparation)
    measurement = values.ss.qualify_puf55_survey_ss_measurement(preparation)
    assert measurement.preparation is preparation
    for entity, observed, issued in (
        ("person", got.person, report.person),
        (
            "tax_unit",
            got.tax_unit.loc[:, list(values.ss.COLUMNS)],
            measurement.tax_unit,
        ),
    ):
        table = run.financial_population.frame.table(entity)
        identity = entity + "_id"
        source_key = values.provenance.support_source_id_column(entity)
        clone_index = values.provenance.support_clone_index_column(entity)
        for arm in (0, 1):
            rows = table.loc[table[clone_index].eq(arm)]
            source_ids = pd.Index(rows[source_key].to_numpy(copy=True), name=identity)
            assert source_ids.is_unique and set(source_ids) == set(issued.index)
            expected = issued.loc[source_ids].copy(deep=True)
            actual = observed.loc[rows[identity]].copy(deep=True)
            actual.index = source_ids
            pd.testing.assert_frame_equal(actual, expected, check_exact=True)
    for _, rows in person.groupby(values.provenance.support_source_id_column("person")):
        assert len(rows) == 2
        outputs = got.person.loc[rows.person_id]
        left, right = outputs.iloc[0].copy(), outputs.iloc[1].copy()
        left.name = right.name
        pd.testing.assert_series_equal(left, right, check_exact=True)
    assert any(got.person[c].isna().any() for c in values.full.SURVEY_SS_COMPONENTS)
    assert not any(
        "prior" in p for profile in values.PROFILES for p in profile.predictors
    )
    ids = {
        int(i)
        for _, payload in got.matrices
        for i in values.model_input.decode_recipient_matrix(payload).features.index
    }
    units = run.financial_population.frame.table("tax_unit")
    assert ids == set(
        units.loc[
            units[values.provenance.support_clone_index_column("tax_unit")].eq(1),
            "tax_unit_id",
        ]
    )
    values.financial._pure_run(run, values.financial._run_entry(run))


@pytest.mark.parametrize("kind", ("receipt", "copied_run"))
def test_descriptive_values_cannot_issue_financial_authority(
    recipient_financial_run, kind
):
    case = recipient_financial_run
    fake = (
        values.financial.codec.decode_json(case.qualified.receipt)
        if kind == "receipt"
        else replace(case.run)
    )
    with pytest.raises(ValueError, match="UNISSUED_FINANCIAL_RUN"):
        values.qualify_puf55_survey_recipients(fake)


@pytest.mark.parametrize("field", ("allowed", "component", "basis"))
def test_every_source_projection_return_is_checked_against_original_literals(
    recipient_financial_run,
    field,
):
    fired = []

    def profile(frame, event, arg):
        caller = frame.f_back
        if (
            event == "return"
            and frame.f_code
            is values.ss.reports.qualify_current_social_security.__code__
            and caller is not None
            and caller.f_code is values.qualify_puf55_survey_recipients.__code__
        ):
            fired.append(True)
            if field == "allowed":
                column = "allowed_" + values.full.SURVEY_SS_COMPONENTS[0]
                arg.person.loc[arg.person.index[0], column] = not bool(
                    arg.person.iloc[0][column]
                )
            elif field == "component":
                arg.person.loc[
                    arg.person.index[0], values.full.SURVEY_SS_COMPONENTS[0]
                ] = 999.0
            else:
                arg.person.loc[arg.person.index[0], "basis_origin"] = (
                    "invented_changed_basis"
                )

    previous = sys.getprofile()
    sys.setprofile(profile)
    try:
        with pytest.raises(ValueError, match="COMPLETE_SOURCE_REPORT"):
            values.qualify_puf55_survey_recipients(recipient_financial_run.run)
    finally:
        sys.setprofile(previous)
    assert fired == [True]


@pytest.mark.parametrize(
    "target", ("person", "matrix", "columns_name", "nan_bits", "mutable_matrix")
)
def test_final_run_check_cannot_mutate_detached_recipient_outputs(
    recipient_financial_run, target
):
    fired = []

    def profile(frame, event, arg):
        caller = frame.f_back
        if (
            event == "return"
            and frame.f_code
            is values.financial.check_atomic_survey_financial_run.__code__
            and caller is not None
            and caller.f_code is values.qualify_puf55_survey_recipients.__code__
            and "result" in caller.f_locals
            and not fired
        ):
            fired.append(True)
            result = caller.f_locals["result"]
            if target == "person":
                result.person.iloc[
                    0, result.person.columns.get_loc("social_security_source_total")
                ] += 1
            elif target == "matrix":
                object.__setattr__(result, "matrices", ())
            elif target == "columns_name":
                result.person.columns.name = "changed"
            elif target == "nan_bits":
                column = values.full.SURVEY_SS_COMPONENTS[0]
                position = np.flatnonzero(result.person[column].isna().to_numpy())[0]
                changed = result.person[column].to_numpy(copy=True)
                assert changed.dtype == np.dtype("float64")
                original_bits = changed.view(np.uint64).copy()
                changed.view(np.uint64)[position] ^= np.uint64(1)
                result.person[column] = changed
                observed = result.person[column].to_numpy(copy=True)
                assert observed.dtype == np.dtype("float64")
                assert np.isnan(observed[position])
                assert (
                    observed.view(np.uint64)[position] ^ original_bits[position]
                ) == np.uint64(1)
                np.testing.assert_array_equal(
                    observed.view(np.uint64), changed.view(np.uint64)
                )
            else:
                name, payload = result.matrices[0]
                object.__setattr__(
                    result,
                    "matrices",
                    ((name, bytearray(payload)), *result.matrices[1:]),
                )

    previous = sys.getprofile()
    sys.setprofile(profile)
    try:
        with pytest.raises(
            ValueError, match="PUF55_SURVEY_RECIPIENTS_(FINAL_PROJECTION|RESULT_TYPE)"
        ):
            values.qualify_puf55_survey_recipients(recipient_financial_run.run)
    finally:
        sys.setprofile(previous)
    assert fired == [True]
