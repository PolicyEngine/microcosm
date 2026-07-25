"""US child-disability stage tests (populace #453)."""

from __future__ import annotations

import importlib.util

import numpy as np
import pandas as pd
import pytest

import populace.build.us_runtime.child_disability as child_disability_module
import populace.build.us_runtime.full_sipp_donor as full_sipp_donor_module
import populace.build.us_runtime.sipp_financial_assets as sipp_financial_assets_module
import populace.build.us_runtime.sipp_head_start as sipp_head_start_module
import populace.build.us_runtime.sipp_vehicles as sipp_vehicles_module
import populace.build.us_runtime.ssi_disability_criteria as ssi_criteria_module
import populace.build.us_runtime.voluntary_filing as voluntary_filing_module
from populace.build.us_runtime import (
    CHILD_DISABILITY_SIPP_USERS_GUIDE_URL,
    SIPP_CHILD_DISABILITY_SOURCE_COLUMNS,
    SIPP_SSI_DISABILITY_DIFFICULTY_PREDICTORS,
    SIPP_SSI_DISABILITY_MODEL_PREDICTORS,
    US_CHILD_DISABILITY_AGE_0_FALLBACK_RATE,
    US_CHILD_DISABILITY_AGE_0_SHARE_BAND,
    US_CHILD_DISABILITY_AGE_1_4_SHARE_BAND,
    US_CHILD_DISABILITY_AGE_1_4_TARGET_RATE,
    US_CHILD_DISABILITY_AGE_5_14_SHARE_BAND,
    US_CHILD_DISABILITY_AGE_5_14_TARGET_RATE,
    US_CHILD_DISABILITY_STAGE_NAME,
    US_STAGE_NAMES,
    clone_us_frame_for_puf_support,
    load_sipp_2023_child_disability_donor,
    resolve_sipp_2023_child_disability_donor,
    us_child_disability_signal_gate,
    us_child_disability_stage_spec,
    us_child_disability_summary,
    with_us_child_disability_inputs,
    with_us_ssi_disability_criteria,
    with_us_ssi_take_up,
)
from populace.frame import US_SCHEMA, Frame, WeightKind, Weights

TIME_PERIOD = 2024
_policyengine_us_installed = importlib.util.find_spec("policyengine_us") is not None
requires_us = pytest.mark.skipif(
    not _policyengine_us_installed,
    reason="requires the policyengine-us [us] extra",
)


@pytest.fixture
def sipp_child_disability_donor() -> pd.DataFrame:
    """Small SIPP-shaped donor with both reviewed calibration bands."""

    row = np.arange(2_800)
    age = 1 + (row % 14)
    within_age = row // 14
    return pd.DataFrame(
        {
            "age": age,
            "is_female": (row % 2) == 0,
            "household_income_proxy": (row % 30) * 5_000.0,
            "is_disabled": np.where(
                age <= 4,
                (within_age % 20) == 0,
                (within_age % 8) == 0,
            ),
            "sipp_weight": 1.0,
        }
    )


def _frame(
    *,
    n_age_0_4: int = 4_000,
    n_age_5_14: int = 4_000,
    adult_ages: tuple[int, ...] = (15, 17, 40, 70),
) -> Frame:
    ages = np.concatenate(
        [
            np.arange(n_age_0_4) % 5,
            5 + np.arange(n_age_5_14) % 10,
            np.asarray(adult_ages, dtype=np.int64),
        ]
    )
    n = len(ages)
    ids = np.arange(1, n + 1, dtype=np.int64)
    households = ids.copy()
    person = pd.DataFrame(
        {
            "person_id": ids,
            "person_source_id": [f"source-{value}" for value in ids],
            "person_household_id": households,
            "person_tax_unit_id": households + 10_000,
            "person_spm_unit_id": households + 20_000,
            "person_family_id": households + 30_000,
            "person_marital_unit_id": households + 40_000,
            "age": ages,
            "is_female": (ids % 2) == 0,
            "employment_income_before_lsr": (ids % 40) * 2_500.0,
            "is_disabled": False,
            "adult_payload": [f"row-{value}" for value in ids],
        }
    )
    # Adult values deliberately vary; the child stage must preserve them.
    adult_mask = person["age"] >= 15
    person.loc[adult_mask, "is_disabled"] = [True, False, True, False]
    tables = {
        "person": person,
        "household": pd.DataFrame({"household_id": households}),
        "tax_unit": pd.DataFrame({"tax_unit_id": households + 10_000}),
        "spm_unit": pd.DataFrame({"spm_unit_id": households + 20_000}),
        "family": pd.DataFrame({"family_id": households + 30_000}),
        "marital_unit": pd.DataFrame({"marital_unit_id": households + 40_000}),
    }
    return Frame(
        tables,
        US_SCHEMA,
        {
            "household": Weights(
                # Nonuniform weights make row-share implementations fail the
                # weighted calibration and summary assertions below.
                values=np.where(ids % 2 == 0, 3.0, 1.0).astype(np.float64),
                kind=WeightKind.DESIGN,
            )
        },
    )


def _run(frame: Frame, donor: pd.DataFrame, *, seed: int = 453) -> Frame:
    return with_us_child_disability_inputs(
        frame,
        seed=seed,
        time_period=TIME_PERIOD,
        sipp_donor=donor,
    )


def _replace_person(frame: Frame, person: pd.DataFrame) -> Frame:
    tables = {entity: frame.table(entity).copy() for entity in frame.entities}
    tables["person"] = person
    return Frame(
        tables,
        frame.schema,
        {entity: frame.weights_for(entity) for entity in frame.weighted_entities},
        frame.strata,
        mass_log=frame.mass_log,
    )


def test_manifest_declares_child_disability_stage() -> None:
    spec = us_child_disability_stage_spec()
    assert spec.stage == US_CHILD_DISABILITY_STAGE_NAME
    assert spec.grain == "person"
    assert spec.outputs == ("is_disabled",)
    assert [operation.kind for operation in spec.operations] == [
        "read_table",
        "fit_weighted_logistic",
        "assign_binary_from_rate",
    ]
    fit = dict(spec.operations[1].parameters)
    assert fit["target_source_items"] == [
        "EHEARING",
        "ESEEING",
        "ECOGNIT",
        "EAMBULAT",
        "ESELFCARE",
        "EDDELAY",
        "EPLAYDIF",
        "ESCHOOLWK",
    ]
    assert fit["age_domain"] == [1, 14]
    assert fit["calibration_age_bands"] == {
        "age_1_4": [1, 4],
        "age_5_14": [5, 14],
    }
    assert fit["pinned_weighted_rates"] == {
        "age_1_4": US_CHILD_DISABILITY_AGE_1_4_TARGET_RATE,
        "age_5_14": US_CHILD_DISABILITY_AGE_5_14_TARGET_RATE,
    }
    age_0 = dict(spec.operations[2].parameters)
    assert age_0["age_domain"] == [0, 0]
    assert age_0["source_age_domain"] == [1, 4]
    assert age_0["rate"] == US_CHILD_DISABILITY_AGE_0_FALLBACK_RATE
    guide = [
        artifact
        for artifact in spec.artifacts
        if artifact.get("kind") == "survey_users_guide"
    ]
    assert guide == [
        {
            "kind": "survey_users_guide",
            "format": "pdf",
            "vintage": "2023",
            "locator": CHILD_DISABILITY_SIPP_USERS_GUIDE_URL,
            "section": "4.3.4 Disability",
            "measure": "RDIS_ALT",
        }
    ]
    assert US_STAGE_NAMES.index(US_CHILD_DISABILITY_STAGE_NAME) == (
        US_STAGE_NAMES.index("eligibility_inputs") + 1
    )
    assert US_STAGE_NAMES.index(US_CHILD_DISABILITY_STAGE_NAME) < (
        US_STAGE_NAMES.index("scf_wealth")
    )
    assert US_STAGE_NAMES.index("scf_wealth") < US_STAGE_NAMES.index(
        "ssi_disability_criteria"
    )


def test_sipp_resolver_uses_only_a_verified_local_file_then_falls_through(
    tmp_path, monkeypatch
) -> None:
    explicit = tmp_path / "explicit.csv"
    local = tmp_path / "local.csv"
    fetched = tmp_path / "fetched.csv"
    explicit.write_bytes(b"good")
    local.write_bytes(b"good")
    fetch_calls: list[dict[str, object]] = []

    monkeypatch.setattr(
        child_disability_module,
        "SIPP_2023_CHILD_DISABILITY_LOCAL_PATH",
        local,
    )
    monkeypatch.setattr(
        child_disability_module,
        "SIPP_2023_CHILD_DISABILITY_DONOR_SIZE_BYTES",
        4,
    )
    monkeypatch.setattr(
        child_disability_module,
        "SIPP_2023_CHILD_DISABILITY_DONOR_SHA256",
        "valid-sha",
    )
    monkeypatch.setattr(
        child_disability_module,
        "_sha256_file",
        lambda path: "valid-sha" if path.read_bytes() == b"good" else "stale-sha",
    )

    def fake_fetch(*, local_path, expected_sha256, expected_size_bytes):
        fetch_calls.append(
            {
                "local_path": local_path,
                "expected_sha256": expected_sha256,
                "expected_size_bytes": expected_size_bytes,
            }
        )
        if (
            local_path.is_file()
            and local_path.stat().st_size == expected_size_bytes
            and child_disability_module._sha256_file(local_path) == expected_sha256
        ):
            return local_path
        return fetched

    monkeypatch.setattr(
        child_disability_module,
        "fetch_sipp_2023_financial_asset_donor",
        fake_fetch,
    )

    assert resolve_sipp_2023_child_disability_donor(explicit) == explicit
    assert resolve_sipp_2023_child_disability_donor() == local
    assert fetch_calls == [
        {
            "local_path": local,
            "expected_sha256": "valid-sha",
            "expected_size_bytes": 4,
        }
    ]

    # An explicit user path is fail-fast and may never fall through.
    explicit.write_bytes(b"evil")
    with pytest.raises(ValueError, match="SHA-256 verification"):
        resolve_sipp_2023_child_disability_donor(explicit)
    assert len(fetch_calls) == 1

    # A same-size, wrong-SHA developer file must not block the shared pin.
    local.write_bytes(b"evil")
    assert resolve_sipp_2023_child_disability_donor() == fetched
    assert len(fetch_calls) == 2

    # Nor may a stale byte length short-circuit the pinned fetch chain.
    local.write_bytes(b"x")
    assert resolve_sipp_2023_child_disability_donor() == fetched
    assert len(fetch_calls) == 3


def test_explicit_sipp_resolver_fails_fast_when_file_mutates_during_hash(
    tmp_path, monkeypatch
) -> None:
    explicit = tmp_path / "explicit.csv"
    explicit.write_bytes(b"good")
    expected_sha256 = child_disability_module.hashlib.sha256(b"good").hexdigest()
    full_sipp_donor_module.clear_full_sipp_sha256_cache()
    real_hash = full_sipp_donor_module._hash_file_contents

    monkeypatch.setattr(
        child_disability_module,
        "SIPP_2023_CHILD_DISABILITY_DONOR_SIZE_BYTES",
        4,
    )
    monkeypatch.setattr(
        child_disability_module,
        "SIPP_2023_CHILD_DISABILITY_DONOR_SHA256",
        expected_sha256,
    )

    def mutate_after_hash(source_path, *, chunk_size):
        digest = real_hash(source_path, chunk_size=chunk_size)
        source_path.write_bytes(b"changed-size")
        return digest

    monkeypatch.setattr(
        full_sipp_donor_module,
        "_hash_file_contents",
        mutate_after_hash,
    )

    with pytest.raises(
        full_sipp_donor_module.FullSIPPDonorMutationError,
        match="changed during SHA-256 verification",
    ):
        resolve_sipp_2023_child_disability_donor(explicit)
    full_sipp_donor_module.clear_full_sipp_sha256_cache()


def test_full_sipp_sha256_is_shared_across_all_stage_loaders_and_rechecks_mutation(
    tmp_path, monkeypatch
) -> None:
    path = tmp_path / "pu2023.csv"
    path.write_bytes(b"good")
    full_sipp_donor_module.clear_full_sipp_sha256_cache()
    scans: list[bytes] = []
    real_hash = full_sipp_donor_module._hash_file_contents

    def counting_hash(source_path, *, chunk_size):
        scans.append(source_path.read_bytes())
        return real_hash(source_path, chunk_size=chunk_size)

    monkeypatch.setattr(
        full_sipp_donor_module,
        "_hash_file_contents",
        counting_hash,
    )
    wrappers = (
        child_disability_module._sha256_file,
        sipp_financial_assets_module._sha256_file,
        ssi_criteria_module._sha256_file,
        sipp_head_start_module._sha256_file,
        sipp_vehicles_module._sha256_file,
        voluntary_filing_module._sha256_file,
    )

    digests = [wrapper(path) for wrapper in wrappers]

    assert len(set(digests)) == 1
    assert scans == [b"good"]

    # Same-size replacement bytes change the filesystem fingerprint and
    # therefore cannot reuse the attestation cached for the old file.
    path.write_bytes(b"evil")
    assert wrappers[0](path) != digests[0]
    assert scans == [b"good", b"evil"]


def test_small_sipp_file_builds_household_income_proxy(tmp_path) -> None:
    raw = pd.DataFrame(
        [
            {
                "SSUID": "h1",
                "ERESIDENCEID": 1,
                "PNUM": 1,
                "MONTHCODE": 12,
                "WPFINWGT": 2.0,
                "TAGE": 35,
                "ESEX": 2,
                "TPTOTINC": 1_000,
                "RSSI_MNYN": 2,
                "THHLDSTATUS": 1,
                "RDIS_ALT": 2,
                "ECOGNIT": 2,
                "EHEARING": 2,
                "ESEEING": 2,
                "EAMBULAT": 2,
                "ESELFCARE": 2,
            },
            {
                "SSUID": "h1",
                "ERESIDENCEID": 1,
                "PNUM": 2,
                "MONTHCODE": 12,
                "WPFINWGT": 2.0,
                "TAGE": 3,
                "ESEX": 1,
                "TPTOTINC": np.nan,
                "RSSI_MNYN": 2,
                "THHLDSTATUS": 1,
                "RDIS_ALT": 1,
                "ECOGNIT": 2,
                "EHEARING": 2,
                "ESEEING": 2,
                "EAMBULAT": 2,
                "ESELFCARE": 2,
                "EDDELAY": 1,
            },
            {
                "SSUID": "h2",
                "ERESIDENCEID": 1,
                "PNUM": 1,
                "MONTHCODE": 12,
                "WPFINWGT": 1.0,
                "TAGE": 8,
                "ESEX": 2,
                "TPTOTINC": 500,
                "RSSI_MNYN": 1,
                "THHLDSTATUS": 1,
                "RDIS_ALT": 1,
                "ECOGNIT": 2,
                "EHEARING": 2,
                "ESEEING": 2,
                "EAMBULAT": 2,
                "ESELFCARE": 2,
                "EPLAYDIF": 1,
            },
            {
                "SSUID": "h4",
                "ERESIDENCEID": 1,
                "PNUM": 1,
                "MONTHCODE": 12,
                "WPFINWGT": 1.0,
                "TAGE": 12,
                "ESEX": 2,
                "TPTOTINC": 500,
                "RSSI_MNYN": 2,
                "THHLDSTATUS": 1,
                "RDIS_ALT": 2,
                "ECOGNIT": 2,
                "EHEARING": 2,
                "ESEEING": 2,
                "EAMBULAT": 2,
                "ESELFCARE": 2,
                "EPLAYDIF": 2,
                "ESCHOOLWK": 2,
            },
            {
                "SSUID": "h2",
                "ERESIDENCEID": 1,
                "PNUM": 1,
                "MONTHCODE": 11,
                "WPFINWGT": 1.0,
                "TAGE": 8,
                "ESEX": 2,
                "TPTOTINC": 99_999,
                "RSSI_MNYN": 1,
                "THHLDSTATUS": 1,
                "RDIS_ALT": 1,
                "ECOGNIT": 1,
                "EHEARING": 1,
                "ESEEING": 1,
                "EAMBULAT": 1,
                "ESELFCARE": 1,
                "EPLAYDIF": 1,
            },
            {
                "SSUID": "h3",
                "ERESIDENCEID": 1,
                "PNUM": 1,
                "MONTHCODE": 12,
                "WPFINWGT": 0.0,
                "TAGE": 9,
                "ESEX": 1,
                "TPTOTINC": np.nan,
                "RSSI_MNYN": 2,
                "THHLDSTATUS": 1,
                "RDIS_ALT": 1,
                "ECOGNIT": 2,
                "EHEARING": 2,
                "ESEEING": 2,
                "EAMBULAT": 2,
                "ESELFCARE": 2,
                "ESCHOOLWK": 1,
            },
        ],
        columns=SIPP_CHILD_DISABILITY_SOURCE_COLUMNS,
    )
    path = tmp_path / "pu2023.csv"
    raw.to_csv(path, sep="|", index=False)

    donor = load_sipp_2023_child_disability_donor(
        path,
        expected_sha256=None,
        expected_size_bytes=None,
        chunksize=2,
    )

    assert donor["age"].tolist() == [3.0, 8.0, 12.0]
    # The first two positives come only from the child-specific questions.
    assert donor["is_disabled"].tolist() == [True, True, False]
    assert donor.attrs["source_audit"]["age_1_4_rows"] == 1
    assert donor.attrs["source_audit"]["age_1_4_positive_rows"] == 1
    assert donor.attrs["source_audit"]["age_5_14_rows"] == 3
    assert donor.attrs["source_audit"]["age_5_14_monthly_ssi_rows"] == 1
    assert donor.attrs["source_audit"]["rdis_alt_mismatch_rows"] == 0
    # The adult in h1 supplies the proxy.  The child-only h2 residence remains
    # zero even when the child reports TPTOTINC, preventing SSI label leakage;
    # the non-December row must not enter either residence aggregate.
    assert donor["household_income_proxy"].tolist() == [12_000.0, 0.0, 0.0]


def test_child_disability_is_deterministic_under_seed(
    sipp_child_disability_donor: pd.DataFrame,
) -> None:
    frame = _frame()
    first = _run(frame, sipp_child_disability_donor, seed=17)
    second = _run(frame, sipp_child_disability_donor, seed=17)

    np.testing.assert_array_equal(
        first.table("person")["is_disabled"],
        second.table("person")["is_disabled"],
    )


def test_classifier_preserves_a_strong_sex_gradient() -> None:
    row = np.arange(4_000)
    within_sex = row // 2
    female = (row % 2) == 0
    later_donor = pd.DataFrame(
        {
            "age": 10.0,
            "is_female": female,
            "household_income_proxy": 50_000.0,
            # 20% for girls versus 2% for boys, with an 11% marginal rate.
            "is_disabled": np.where(
                female,
                (within_sex % 5) == 0,
                (within_sex % 50) == 0,
            ),
            "sipp_weight": 1.0,
        }
    )
    early_row = np.arange(800)
    early_donor = pd.DataFrame(
        {
            "age": 1 + (early_row % 4),
            "is_female": (early_row % 2) == 0,
            "household_income_proxy": 50_000.0,
            "is_disabled": (early_row // 4) % 20 == 0,
            "sipp_weight": 1.0,
        }
    )
    donor = pd.concat([early_donor, later_donor], ignore_index=True)
    result = _run(_frame(n_age_5_14=8_000), donor, seed=96)
    children = result.table("person").loc[lambda table: table["age"].between(5, 14)]
    assigned_share = children.groupby("is_female")["is_disabled"].mean()

    assert assigned_share[True] > assigned_share[False] + 0.08


def test_rows_age_15_and_over_keep_all_values(
    sipp_child_disability_donor: pd.DataFrame,
) -> None:
    frame = _frame()
    before = frame.table("person").copy(deep=True)
    result = _run(frame, sipp_child_disability_donor)
    adult = before["age"] >= 15

    pd.testing.assert_frame_equal(
        result.table("person").loc[adult],
        before.loc[adult],
        check_exact=True,
    )
    assert us_child_disability_signal_gate(result, input_frame=frame).passed


def test_noncanonical_adult_boolean_storage_is_not_rewritten(
    sipp_child_disability_donor: pd.DataFrame,
) -> None:
    frame = _frame()
    person = frame.table("person").copy()
    adult_positions = np.flatnonzero(person["age"].to_numpy() >= 15)
    poisoned_storage = person["is_disabled"].to_numpy(dtype=np.uint8)
    poisoned_storage[adult_positions[0]] = 0x02
    person["is_disabled"] = poisoned_storage.view(np.bool_)
    frame = _replace_person(frame, person)
    before = (
        frame.table("person")["is_disabled"]
        .to_numpy(copy=False)
        .view(np.uint8)[adult_positions]
    ).copy()
    assert before[0] == 0x02

    result = _run(frame, sipp_child_disability_donor)

    after = (
        result.table("person")["is_disabled"]
        .to_numpy(copy=False)
        .view(np.uint8)[adult_positions]
    )
    np.testing.assert_array_equal(after, before)


def test_gate_measures_weighted_child_share_before_and_after(
    sipp_child_disability_donor: pd.DataFrame,
) -> None:
    frame = _frame()
    result = _run(frame, sipp_child_disability_donor)
    gate = us_child_disability_signal_gate(result, input_frame=frame)
    changes = gate.details["weighted_child_is_disabled_share_change"]
    weights = result.resolve_weights("person").values

    for key, lower, upper in (
        ("age_0", 0, 0),
        ("age_1_4", 1, 4),
        ("age_5_14", 5, 14),
    ):
        mask = result.table("person")["age"].between(lower, upper).to_numpy()
        before = float(
            np.average(
                frame.table("person").loc[mask, "is_disabled"],
                weights=weights[mask],
            )
        )
        after = float(
            np.average(
                result.table("person").loc[mask, "is_disabled"],
                weights=weights[mask],
            )
        )
        assert changes[key]["before"] == pytest.approx(before)
        assert changes[key]["after"] == pytest.approx(after)
        assert changes[key]["absolute_change"] == pytest.approx(after - before)


def test_age_5_14_share_lands_inside_sipp_gate_band(
    sipp_child_disability_donor: pd.DataFrame,
) -> None:
    frame = _frame()
    result = _run(frame, sipp_child_disability_donor)
    summary = us_child_disability_summary(result)
    low, high = US_CHILD_DISABILITY_AGE_5_14_SHARE_BAND
    person = result.table("person")
    band = person["age"].between(5, 14)
    weights = result.resolve_weights("person").values
    expected_weighted_share = np.average(
        person.loc[band, "is_disabled"], weights=weights[band]
    )

    assert low <= summary["age_5_14_disabled_share"] <= high
    assert summary["age_5_14_disabled_share"] == pytest.approx(expected_weighted_share)
    gate = us_child_disability_signal_gate(result, input_frame=frame)
    assert gate.passed, gate.failures


def test_age_1_4_share_lands_inside_eddelay_informed_gate_band(
    sipp_child_disability_donor: pd.DataFrame,
) -> None:
    result = _run(_frame(), sipp_child_disability_donor)
    share = us_child_disability_summary(result)["age_1_4_disabled_share"]
    low, high = US_CHILD_DISABILITY_AGE_1_4_SHARE_BAND

    assert low <= share <= high
    assert share == pytest.approx(US_CHILD_DISABILITY_AGE_1_4_TARGET_RATE, abs=0.015)
    assert US_CHILD_DISABILITY_AGE_1_4_TARGET_RATE == 0.050645332184


def test_age_0_share_uses_only_adjacent_observed_band_fallback(
    sipp_child_disability_donor: pd.DataFrame,
) -> None:
    result = _run(_frame(), sipp_child_disability_donor)
    share = us_child_disability_summary(result)["age_0_disabled_share"]
    low, high = US_CHILD_DISABILITY_AGE_0_SHARE_BAND

    assert low <= share <= high
    assert share == pytest.approx(US_CHILD_DISABILITY_AGE_0_FALLBACK_RATE, abs=0.015)
    assert (
        US_CHILD_DISABILITY_AGE_0_FALLBACK_RATE
        == US_CHILD_DISABILITY_AGE_1_4_TARGET_RATE
    )
    assert US_CHILD_DISABILITY_AGE_5_14_TARGET_RATE == 0.124518076756


def test_existing_child_true_is_never_cleared(
    sipp_child_disability_donor: pd.DataFrame,
) -> None:
    frame = _frame(n_age_0_4=20, n_age_5_14=20)
    person = frame.table("person").copy()
    child = person["age"] < 15
    existing_index = person.index[child][::3]
    person.loc[existing_index, "is_disabled"] = True
    seeded = _replace_person(frame, person)

    result = _run(seeded, sipp_child_disability_donor)

    assert result.table("person").loc[existing_index, "is_disabled"].all()


def test_signal_gate_fails_on_poisoned_child_surface(
    sipp_child_disability_donor: pd.DataFrame,
) -> None:
    frame = _frame()
    result = _run(frame, sipp_child_disability_donor)
    poisoned_person = result.table("person").copy()
    poisoned_person.loc[poisoned_person["age"].between(5, 14), "is_disabled"] = False
    poisoned = _replace_person(result, poisoned_person)

    gate = us_child_disability_signal_gate(poisoned, input_frame=frame)

    assert not gate.passed
    assert any("5-14" in failure for failure in gate.failures)


def test_signal_gate_fails_when_an_adult_row_changes(
    sipp_child_disability_donor: pd.DataFrame,
) -> None:
    frame = _frame()
    result = _run(frame, sipp_child_disability_donor)
    poisoned_person = result.table("person").copy()
    adult_index = poisoned_person.index[poisoned_person["age"] >= 15][0]
    poisoned_person.loc[adult_index, "adult_payload"] = "poisoned"
    poisoned = _replace_person(result, poisoned_person)

    gate = us_child_disability_signal_gate(poisoned, input_frame=frame)

    assert not gate.passed
    assert "age 15+: output rows differ from the stage input." in gate.failures


@requires_us
def test_actual_child_disability_criteria_and_take_up_pipeline_controls_ssi(
    sipp_child_disability_donor: pd.DataFrame,
) -> None:
    """Exercise the production stages and simulate the exact resulting frame."""

    from policyengine_us import Microsimulation
    from policyengine_us.data import USSingleYearDataset

    household_ids = np.arange(1, 121, dtype=np.int64)
    person_ids = np.arange(1, 2 * len(household_ids) + 1, dtype=np.int64)
    person_household_ids = np.repeat(household_ids, 2)
    is_child = person_ids % 2 == 0
    ages = np.where(
        is_child,
        1 + ((person_household_ids - 1) % 14),
        35,
    )
    zeros = np.zeros(len(person_ids), dtype=np.float64)
    person = pd.DataFrame(
        {
            "person_id": person_ids,
            "person_household_id": person_household_ids,
            "person_tax_unit_id": person_household_ids + 1_000,
            "person_spm_unit_id": person_household_ids + 2_000,
            "person_family_id": person_household_ids + 3_000,
            "person_marital_unit_id": person_ids + 4_000,
            "age": ages,
            "is_female": person_ids % 4 < 2,
            "is_tax_unit_dependent": is_child,
            "is_disabled": False,
            "A_MARITL": np.where(is_child, 7, 5),
            "employment_income_before_lsr": zeros,
            "taxable_interest_income": zeros,
            "tax_exempt_interest_income": zeros,
            "qualified_dividend_income": zeros,
            "non_qualified_dividend_income": zeros,
            "rental_income": zeros,
            "bank_account_assets": zeros,
            "stock_assets": zeros,
            "bond_assets": zeros,
            "PEDISDRS": np.full(len(person_ids), 2),
            "PEDISEAR": np.full(len(person_ids), 2),
            "PEDISEYE": np.full(len(person_ids), 2),
            "PEDISOUT": np.full(len(person_ids), 2),
            "PEDISPHY": np.full(len(person_ids), 2),
            "PEDISREM": np.full(len(person_ids), 2),
            "social_security_disability": zeros,
            "disability_benefits": zeros,
            "SSI_VAL": zeros,
            "own_children_in_household": np.where(is_child, 0, 1),
        }
    )
    base = Frame(
        {
            "person": person,
            "household": pd.DataFrame(
                {
                    "household_id": household_ids,
                    "state_code": "CA",
                }
            ),
            "tax_unit": pd.DataFrame(
                {
                    "tax_unit_id": household_ids + 1_000,
                    "filing_status": "HEAD_OF_HOUSEHOLD",
                }
            ),
            "spm_unit": pd.DataFrame({"spm_unit_id": household_ids + 2_000}),
            "family": pd.DataFrame({"family_id": household_ids + 3_000}),
            "marital_unit": pd.DataFrame({"marital_unit_id": person_ids + 4_000}),
        },
        US_SCHEMA,
        {
            "household": Weights(
                np.ones(len(household_ids), dtype=np.float64),
                WeightKind.DESIGN,
            )
        },
    )
    expanded = clone_us_frame_for_puf_support(base)
    disability_frame = _run(expanded, sipp_child_disability_donor, seed=453)

    rng = np.random.default_rng(747)
    criteria_donor = pd.DataFrame(index=np.arange(120))
    criteria_donor["age"] = np.arange(120, dtype=np.float64)
    criteria_donor["is_female"] = rng.integers(0, 2, 120)
    criteria_donor["is_married"] = rng.integers(0, 2, 120)
    criteria_donor["employment_income"] = rng.gamma(2.0, 5_000.0, 120)
    criteria_donor["interest_income"] = rng.gamma(1.0, 100.0, 120)
    criteria_donor["dividend_income"] = rng.gamma(1.0, 100.0, 120)
    criteria_donor["rental_income"] = rng.normal(0.0, 500.0, 120)
    criteria_donor["bank_account_assets"] = rng.gamma(1.0, 1_000.0, 120)
    criteria_donor["stock_assets"] = rng.gamma(1.0, 1_000.0, 120)
    criteria_donor["bond_assets"] = rng.gamma(1.0, 200.0, 120)
    criteria_donor["count_under_18"] = rng.integers(0, 5, 120)
    for predictor in SIPP_SSI_DISABILITY_DIFFICULTY_PREDICTORS:
        criteria_donor[predictor] = rng.integers(0, 2, 120)
    criteria_donor["social_security_disability"] = rng.integers(0, 2, 120) * 6_000.0
    criteria_donor["has_disability_income"] = rng.integers(0, 2, 120)
    criteria_donor["meets_ssi_disability_criteria"] = np.arange(120) % 5 == 0
    criteria_donor["household_weight"] = np.arange(1, 121, dtype=np.float64)
    criteria_donor = criteria_donor.loc[
        :,
        [
            *SIPP_SSI_DISABILITY_MODEL_PREDICTORS,
            "meets_ssi_disability_criteria",
            "household_weight",
        ],
    ]
    criteria_frame = with_us_ssi_disability_criteria(
        disability_frame,
        seed=453,
        time_period=TIME_PERIOD,
        sipp_donor=criteria_donor,
    )

    def simulation(frame: Frame) -> Microsimulation:
        tables = {entity: frame.table(entity).copy() for entity in frame.entities}
        tables["household"]["household_weight"] = frame.weights_for("household").values
        dataset = USSingleYearDataset(
            person=tables["person"],
            household=tables["household"],
            tax_unit=tables["tax_unit"],
            spm_unit=tables["spm_unit"],
            family=tables["family"],
            marital_unit=tables["marital_unit"],
            time_period=TIME_PERIOD,
        )
        return Microsimulation(dataset=dataset)

    criteria_simulation = simulation(criteria_frame)
    uncapped_ssi = np.asarray(
        criteria_simulation.calculate(
            "uncapped_ssi",
            period="2024-12",
            map_to="person",
        ),
        dtype=np.float64,
    )

    criteria_person = criteria_frame.table("person")
    person_weights = np.asarray(
        criteria_frame.resolve_weights("person").values,
        dtype=np.float64,
    )
    targets: dict[str, float] = {}
    for key, lower, upper in (
        ("under_18", 0, 17),
        ("18_64", 18, 64),
        ("65_plus", 65, 200),
    ):
        band = criteria_person["age"].between(lower, upper).to_numpy()
        capacity = float(person_weights[band & (uncapped_ssi > 0.0)].sum())
        targets[key] = capacity / 2.0 if capacity > 0.0 else 1.0
    final_frame, take_up_diagnostics = with_us_ssi_take_up(
        criteria_frame,
        uncapped_ssi=uncapped_ssi,
        seed=453,
        targets=targets,
    )
    assert take_up_diagnostics["bernoulli_law_violation_count"] == 0

    final_person = final_frame.table("person")
    child = final_person["age"].lt(15).to_numpy()
    selected = (
        child
        & final_person["is_disabled"].to_numpy(dtype=bool)
        & final_person["meets_ssi_disability_criteria"].to_numpy(dtype=bool)
        & final_person["takes_up_ssi_if_eligible"].to_numpy(dtype=bool)
        & (uncapped_ssi > 0.0)
    )
    rejected_by_take_up = (
        child
        & final_person["meets_ssi_disability_criteria"].to_numpy(dtype=bool)
        & ~final_person["takes_up_ssi_if_eligible"].to_numpy(dtype=bool)
        & (uncapped_ssi > 0.0)
    )
    unselected = (
        child
        & ~final_person["is_disabled"].to_numpy(dtype=bool)
        & ~final_person["meets_ssi_disability_criteria"].to_numpy(dtype=bool)
    )
    assert selected.any()
    assert rejected_by_take_up.any()
    assert unselected.any()

    final_simulation = simulation(final_frame)
    ssi = np.asarray(
        final_simulation.calculate("ssi", period="2024-12", map_to="person"),
        dtype=np.float64,
    )
    parent_deeming = np.asarray(
        final_simulation.calculate(
            "ssi_unearned_income_deemed_from_ineligible_parent",
            period="2024-12",
            map_to="person",
        ),
        dtype=np.float64,
    )
    selected_row = int(np.flatnonzero(selected)[0])
    rejected_row = int(np.flatnonzero(rejected_by_take_up)[0])
    unselected_row = int(np.flatnonzero(unselected)[0])

    assert parent_deeming[selected_row] == 0.0
    assert ssi[selected_row] > 0.0
    assert parent_deeming[rejected_row] == 0.0
    assert uncapped_ssi[rejected_row] > 0.0
    assert ssi[rejected_row] == 0.0
    assert uncapped_ssi[unselected_row] == 0.0
    assert ssi[unselected_row] == 0.0
