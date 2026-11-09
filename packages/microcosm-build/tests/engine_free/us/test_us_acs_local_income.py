"""The separate ASEC-channel income transfer for the ACS local lane
(microcosm#1022): its plan, its OIP and aligned-RETP predictor extensions
(microcosm#1056 review), receipt and gate, and a real QRF fit through the
multispine seam."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import pytest

from microcosm.build.us_runtime import acs_local_income, acs_multispine
from microcosm.build.us_runtime.acs_local_income import (
    ACS_LOCAL_ALIGNED_RETIREMENT_DONOR_COMPONENTS,
    ACS_LOCAL_ALIGNED_RETIREMENT_FEATURE,
    ACS_LOCAL_CHILD_SUPPORT_FAMILY,
    ACS_LOCAL_INCOME_DONOR_CHANNEL,
    ACS_LOCAL_INCOME_PREDICTOR_EXTENSIONS,
    ACS_LOCAL_INCOME_SHARED_RETIREMENT_COMPONENTS,
    ACS_LOCAL_INCOME_TRANSFER_COLUMNS,
    ACS_LOCAL_INCOME_TRANSFER_FAMILIES,
    ACS_LOCAL_INCOME_TRANSFER_GATE_NAME,
    ACS_LOCAL_INCOME_TRANSFER_ISSUE,
    ACS_LOCAL_INCOME_TRANSFER_METHOD,
    ACS_LOCAL_OTHER_INCOME_COLUMN,
    ACS_LOCAL_OTHER_INCOME_DONOR_COMPONENTS,
    ACS_LOCAL_OTHER_INCOME_FEATURE,
    ACS_LOCAL_RETIREMENT_FAMILY,
    ACS_LOCAL_SHARED_RETIREMENT_FEATURE,
    ACS_LOCAL_WORK_DISABILITY_FAMILY,
    acs_local_income_predictor_extension_receipt,
    acs_local_income_transfer_signal_gate,
    acs_local_income_transfer_target_families,
    map_acs_local_other_income,
    record_acs_local_income_transfer,
    require_acs_local_income_donor,
    without_acs_local_other_income,
)
from microcosm.build.us_runtime.acs_pums import ACS_2024_1YR_SPINE, AcsPumsSource
from microcosm.build.us_runtime.acs_transfer import (
    ACS_DONOR_CHANNEL_AUTO,
    ASEC_PUF_DONOR_SPINE,
    acs_transfer_execution_contract_identity,
    declared_acs_transfer_target_families,
    required_acs_transfer_inputs,
)
from microcosm.build.us_runtime.base_pool import spine_column
from microcosm.build.us_runtime.support_provenance import (
    support_channel_column,
    support_clone_index_column,
    support_source_id_column,
)
from microcosm.frame import US_SCHEMA, Frame, WeightKind, Weights

TAG = spine_column("person")


def _pooled(
    *,
    spines: tuple[str, ...] = (
        ASEC_PUF_DONOR_SPINE,
        ASEC_PUF_DONOR_SPINE,
        ACS_2024_1YR_SPINE,
        ACS_2024_1YR_SPINE,
    ),
    amounts: dict[str, list[float]] | None = None,
) -> Frame:
    """Two donor and two ACS persons, one household each."""

    n = len(spines)
    person = pd.DataFrame(
        {
            "person_id": np.arange(1, n + 1),
            "person_household_id": np.arange(1, n + 1),
            "person_tax_unit_id": np.arange(1, n + 1),
            "person_spm_unit_id": np.arange(1, n + 1),
            "person_family_id": np.arange(1, n + 1),
            "person_marital_unit_id": np.arange(1, n + 1),
            TAG: list(spines),
        }
    )
    for column in ACS_LOCAL_INCOME_TRANSFER_COLUMNS:
        person[column] = [0.0, 1_200.0, 0.0, 900.0][:n]
    for column in _EXTENSION_ONLY_COMPONENTS:
        person[column] = [0.0, 300.0, 50.0, 0.0][:n]
    for column, values in (amounts or {}).items():
        person[column] = values
    ids = np.arange(1, n + 1)
    return Frame(
        {
            "person": person,
            "household": pd.DataFrame({"household_id": ids}),
            "tax_unit": pd.DataFrame({"tax_unit_id": ids}),
            "spm_unit": pd.DataFrame({"spm_unit_id": ids}),
            "family": pd.DataFrame({"family_id": ids}),
            "marital_unit": pd.DataFrame({"marital_unit_id": ids}),
        },
        US_SCHEMA,
        {"household": Weights(np.full(n, 100.0), WeightKind.DESIGN)},
    )


_OIP, _ALIGNED = ACS_LOCAL_INCOME_PREDICTOR_EXTENSIONS
#: Donor columns the two extensions read that are not transfer targets.
_EXTENSION_ONLY_COMPONENTS = tuple(
    column
    for extension in ACS_LOCAL_INCOME_PREDICTOR_EXTENSIONS
    for column in extension.donor_components
    if column not in ACS_LOCAL_INCOME_TRANSFER_COLUMNS
)
_FAMILY_OF = {
    column: family
    for family, targets in ACS_LOCAL_INCOME_TRANSFER_FAMILIES.items()
    for column in targets
}
_COVERAGE = {
    "column": ACS_LOCAL_OTHER_INCOME_COLUMN,
    "persons": 2,
    "observed_rows": 2,
    "blank_rows": 0,
    "positive_rows": 1,
    "weighted_positive_share": 0.5,
}


def _family_predictors(family: str) -> list[str]:
    """The predictors a family is fit with: shared ones plus its extensions."""

    predictors = ["age", "is_female", "__acs_transfer_state_fips"]
    predictors.append(ACS_LOCAL_SHARED_RETIREMENT_FEATURE)
    for extension in ACS_LOCAL_INCOME_PREDICTOR_EXTENSIONS:
        if family not in extension.families:
            continue
        if extension.replaces is None:
            predictors.append(extension.feature)
        else:
            predictors = [
                extension.feature if name == extension.replaces else name
                for name in predictors
            ]
    return predictors


def _donor_summary() -> dict[str, Any]:
    return {
        "channel": ACS_LOCAL_INCOME_DONOR_CHANNEL,
        "predictor_extensions": {
            extension.feature: {"weighted_recipient_share": 0.1}
            for extension in ACS_LOCAL_INCOME_PREDICTOR_EXTENSIONS
        },
    }


def _receipt(
    predictors: dict[str, list[str]] | None = None, **overrides: Any
) -> dict[str, Any]:
    receipt = record_acs_local_income_transfer(
        _donor_summary(),
        [
            {
                "column": column,
                "donor_channel": ACS_LOCAL_INCOME_DONOR_CHANNEL,
                "imputed_recipient_rows": 2,
                "unmodeled_recipient_rows": 0,
                "predictors": (predictors or {}).get(
                    column, _family_predictors(_FAMILY_OF[column])
                ),
            }
            for column in ACS_LOCAL_INCOME_TRANSFER_COLUMNS
        ],
        acs_persons=2,
        other_income=_COVERAGE,
    )
    receipt.update(overrides)
    return receipt


def test_target_families_are_person_only_and_disjoint_from_the_shared_plan():
    families = acs_local_income_transfer_target_families()
    assert set(families) == {"person"}
    assert families["person"] == dict(ACS_LOCAL_INCOME_TRANSFER_FAMILIES)
    columns = {c for targets in families["person"].values() for c in targets}
    assert columns == set(ACS_LOCAL_INCOME_TRANSFER_COLUMNS)
    assert not columns & required_acs_transfer_inputs()
    assert not columns & set(ACS_LOCAL_INCOME_SHARED_RETIREMENT_COMPONENTS)


def test_the_shared_declared_plan_does_not_carry_the_local_income_columns():
    shared = {
        column
        for families in declared_acs_transfer_target_families().values()
        for targets in families.values()
        for column in targets
    }
    assert not shared & set(ACS_LOCAL_INCOME_TRANSFER_COLUMNS)


def test_target_families_refuse_overlap_with_the_shared_plan(monkeypatch):
    monkeypatch.setattr(
        acs_local_income,
        "required_acs_transfer_inputs",
        lambda: frozenset({"workers_compensation"}),
    )
    with pytest.raises(ValueError, match="double counting"):
        acs_local_income_transfer_target_families()


def test_receipt_counts_each_column_from_the_income_pass_provenance():
    receipt = _receipt()
    assert receipt["issue"] == ACS_LOCAL_INCOME_TRANSFER_ISSUE
    assert receipt["method"] == ACS_LOCAL_INCOME_TRANSFER_METHOD
    assert receipt["donor_channel"] == ACS_LOCAL_INCOME_DONOR_CHANNEL
    assert receipt["acs_persons"] == 2
    assert receipt["not_transferred_shared_plan_components"] == list(
        ACS_LOCAL_INCOME_SHARED_RETIREMENT_COMPONENTS
    )
    assert receipt["predictor_extensions"] == (
        acs_local_income_predictor_extension_receipt()
    )
    assert receipt["acs_other_income"] == _COVERAGE
    for family in ACS_LOCAL_INCOME_TRANSFER_FAMILIES:
        assert receipt["family_predictors"][family] == sorted(
            _family_predictors(family)
        )
    for column in ACS_LOCAL_INCOME_TRANSFER_COLUMNS:
        entry = receipt["columns"][column]
        assert entry["imputed_rows"] == 2
        assert entry["unmodeled_rows"] == 0
        assert entry["donor_channel"] == [ACS_LOCAL_INCOME_DONOR_CHANNEL]
        assert entry["predictors"] == sorted(_family_predictors(_FAMILY_OF[column]))


def test_donor_without_an_asec_role_is_refused():
    with pytest.raises(ValueError, match="ASEC observation role"):
        require_acs_local_income_donor(_pooled())


def test_donor_summary_requires_complete_non_negative_amounts(monkeypatch):
    donor = _pooled(spines=(ASEC_PUF_DONOR_SPINE,) * 4)
    monkeypatch.setattr(
        acs_local_income,
        "resolve_acs_donor_channel",
        lambda frame, channel: (frame, channel),
    )
    summary = require_acs_local_income_donor(donor)
    assert summary["channel"] == ACS_LOCAL_INCOME_DONOR_CHANNEL
    assert summary["person_rows"] == 4
    assert summary["columns"]["workers_compensation"]["positive_rows"] == 2
    # Each extension's donor analog: components and weighted share positive.
    other_income = summary["predictor_extensions"][ACS_LOCAL_OTHER_INCOME_FEATURE]
    assert other_income["donor_components"] == list(
        ACS_LOCAL_OTHER_INCOME_DONOR_COMPONENTS
    )
    assert other_income["positive_rows"] == 3
    assert other_income["weighted_recipient_share"] == pytest.approx(0.75)
    aligned = summary["predictor_extensions"][ACS_LOCAL_ALIGNED_RETIREMENT_FEATURE]
    assert aligned["donor_components"] == list(
        ACS_LOCAL_ALIGNED_RETIREMENT_DONOR_COMPONENTS
    )
    assert aligned["positive_rows"] == 3
    for column in ("veterans_benefits", "taxable_private_pension_income"):
        lacking = _pooled(spines=(ASEC_PUF_DONOR_SPINE,) * 4)
        lacking = Frame(
            {
                **{e: lacking.table(e) for e in lacking.entities if e != "person"},
                "person": lacking.table("person").drop(columns=[column]),
            },
            lacking.schema,
            {"household": lacking.weights_for("household")},
        )
        with pytest.raises(ValueError, match=rf"lacks \['{column}'\], component"):
            require_acs_local_income_donor(lacking)
    negative_component = _pooled(
        spines=(ASEC_PUF_DONOR_SPINE,) * 4,
        amounts={"alimony_income": [0.0, -3.0, 0.0, 0.0]},
    )
    with pytest.raises(ValueError, match="alimony_income has 0 missing and 1 negative"):
        require_acs_local_income_donor(negative_component)
    negative = _pooled(
        spines=(ASEC_PUF_DONOR_SPINE,) * 4,
        amounts={"child_support_received": [0.0, -5.0, 0.0, 1.0]},
    )
    with pytest.raises(ValueError, match="negative"):
        require_acs_local_income_donor(negative)
    absent = _pooled(spines=(ASEC_PUF_DONOR_SPINE,) * 4)
    absent = Frame(
        {
            **{e: absent.table(e) for e in absent.entities if e != "person"},
            "person": absent.table("person").drop(columns=["keogh_distributions"]),
        },
        absent.schema,
        {"household": absent.weights_for("household")},
    )
    with pytest.raises(ValueError, match="lacks"):
        require_acs_local_income_donor(absent)


def test_gate_passes_on_a_complete_transfer_with_a_clean_receipt():
    gate = acs_local_income_transfer_signal_gate(_pooled(), receipt=_receipt())
    assert gate.passed, gate.failures
    assert gate.name == ACS_LOCAL_INCOME_TRANSFER_GATE_NAME
    acs = gate.details["per_spine"][ACS_2024_1YR_SPINE]
    assert acs["rows"] == 2


@pytest.mark.parametrize(
    ("amounts", "receipt_overrides", "expected"),
    [
        (
            {"child_support_received": [0.0, 10.0, np.nan, 5.0]},
            {},
            "missing",
        ),
        ({"workers_compensation": [0.0, 10.0, -1.0, 5.0]}, {}, "negative"),
        (
            {"disability_benefits": [0.0, 10.0, 0.0, 0.0]},
            {},
            "all zero although the donor",
        ),
        ({}, {"donor_channel": "puf"}, "donor channel"),
        ({}, {"acs_persons": 5}, "ACS person"),
        (
            {},
            {"method": "separate_local_qrf_pass_missing_cells_only"},
            "predates the reviewed OIP and aligned-RETP predictors",
        ),
        ({}, {"predictor_extensions": None}, "not the reviewed OIP"),
        ({}, {"acs_other_income": {"blank_rows": 0}}, "no ACS OIP coverage"),
        (
            {},
            {"donor": {"channel": ACS_LOCAL_INCOME_DONOR_CHANNEL}},
            f"no donor coverage for {ACS_LOCAL_OTHER_INCOME_FEATURE}",
        ),
    ],
    ids=[
        "missing-acs-cell",
        "negative",
        "acs-all-zero",
        "wrong-channel",
        "count",
        "stale-method",
        "no-extensions",
        "no-oip-coverage",
        "no-donor-coverage",
    ],
)
def test_gate_fails(amounts, receipt_overrides, expected):
    gate = acs_local_income_transfer_signal_gate(
        _pooled(amounts=amounts), receipt=_receipt(**receipt_overrides)
    )
    assert not gate.passed
    assert expected in " ".join(gate.failures)


def _shared_predictors() -> list[str]:
    return ["age", "is_female", ACS_LOCAL_SHARED_RETIREMENT_FEATURE]


@pytest.mark.parametrize(
    ("column", "predictors", "expected"),
    [
        (
            "child_support_received",
            _shared_predictors(),
            f"was fit without {ACS_LOCAL_OTHER_INCOME_FEATURE}",
        ),
        (
            "workers_compensation",
            [*_family_predictors(ACS_LOCAL_WORK_DISABILITY_FAMILY), _OIP.feature],
            "which is reviewed for ['acs_local_child_support'] only",
        ),
        (
            "keogh_distributions",
            [*_family_predictors(ACS_LOCAL_RETIREMENT_FAMILY), _OIP.feature],
            "which is reviewed for ['acs_local_child_support'] only",
        ),
        (
            "disability_benefits",
            _shared_predictors(),
            f"was fit without {ACS_LOCAL_ALIGNED_RETIREMENT_FEATURE}",
        ),
        (
            "taxable_401k_distributions",
            [
                *_family_predictors(ACS_LOCAL_RETIREMENT_FAMILY),
                ACS_LOCAL_SHARED_RETIREMENT_FEATURE,
            ],
            f"still uses the shared {ACS_LOCAL_SHARED_RETIREMENT_FEATURE}",
        ),
        (
            "child_support_expense",
            [
                *_family_predictors(ACS_LOCAL_CHILD_SUPPORT_FAMILY),
                ACS_LOCAL_ALIGNED_RETIREMENT_FEATURE,
            ],
            "reviewed for ['acs_local_work_disability_income', "
            "'acs_local_retirement_distributions'] only",
        ),
    ],
    ids=[
        "child-support-without-oip",
        "oip-on-workers-comp",
        "oip-on-retirement",
        "disability-without-aligned-retp",
        "retirement-still-shared-retp",
        "aligned-retp-on-child-support",
    ],
)
def test_gate_refuses_a_predictor_extension_off_its_reviewed_families(
    column, predictors, expected
):
    gate = acs_local_income_transfer_signal_gate(
        _pooled(), receipt=_receipt(predictors={column: predictors})
    )
    assert not gate.passed
    assert expected in " ".join(gate.failures)


def test_gate_refuses_a_receipt_from_before_the_predictor_extensions():
    """A pre-change staging receipt: old method id, no extensions or coverage."""

    stale = _receipt(
        predictors={
            column: ["age", "is_female", ACS_LOCAL_SHARED_RETIREMENT_FEATURE]
            for column in ACS_LOCAL_INCOME_TRANSFER_COLUMNS
        },
        method="separate_local_qrf_pass_missing_cells_only",
    )
    for key in ("predictor_extensions", "family_predictors", "acs_other_income"):
        stale.pop(key)
    stale["donor"] = {"channel": ACS_LOCAL_INCOME_DONOR_CHANNEL}
    gate = acs_local_income_transfer_signal_gate(_pooled(), receipt=stale)
    assert not gate.passed
    failures = " ".join(gate.failures)
    for expected in (
        "predates the reviewed OIP",
        "not the reviewed OIP and aligned-RETP definitions",
        "no ACS OIP coverage",
        "no donor coverage",
        f"was fit without {ACS_LOCAL_OTHER_INCOME_FEATURE}",
        f"was fit without {ACS_LOCAL_ALIGNED_RETIREMENT_FEATURE}",
    ):
        assert expected in failures, expected


def test_gate_fails_without_a_receipt_or_with_unmodeled_rows():
    missing = acs_local_income_transfer_signal_gate(_pooled(), receipt=None)
    assert not missing.passed
    assert "No acs_local_income_transfer staging receipt" in " ".join(missing.failures)
    receipt = _receipt()
    receipt["columns"]["keogh_distributions"]["unmodeled_rows"] = 1
    unmodeled = acs_local_income_transfer_signal_gate(_pooled(), receipt=receipt)
    assert not unmodeled.passed
    assert "keogh_distributions left 1" in " ".join(unmodeled.failures)


def test_gate_fails_on_missing_tags_or_an_unknown_spine():
    frame = _pooled()
    person = frame.table("person").copy()
    person[TAG] = person[TAG].where(person.index > 0)
    untagged = Frame(
        {
            **{e: frame.table(e) for e in frame.entities if e != "person"},
            "person": person,
        },
        frame.schema,
        {"household": frame.weights_for("household")},
    )
    assert "Missing person origin tags" in " ".join(
        acs_local_income_transfer_signal_gate(untagged, receipt=_receipt()).failures
    )
    unknown = _pooled(
        spines=(
            ASEC_PUF_DONOR_SPINE,
            ASEC_PUF_DONOR_SPINE,
            ACS_2024_1YR_SPINE,
            "other",
        )
    )
    gate = acs_local_income_transfer_signal_gate(unknown, receipt=_receipt())
    assert "unsupported spine" in " ".join(gate.failures)


# ---------------------------------------------------------------------------
# The OIP and aligned-RETP predictor extensions (microcosm#1056 review)
# ---------------------------------------------------------------------------


def test_oip_extends_only_the_child_support_family():
    assert _OIP.feature == ACS_LOCAL_OTHER_INCOME_FEATURE
    assert _OIP.families == (ACS_LOCAL_CHILD_SUPPORT_FAMILY,)
    assert _OIP.replaces is None
    assert _OIP.recipient_source == ACS_LOCAL_OTHER_INCOME_COLUMN
    # The family holds child support paid too (neutral in the holdout).
    assert ACS_LOCAL_INCOME_TRANSFER_FAMILIES[ACS_LOCAL_CHILD_SUPPORT_FAMILY] == (
        "child_support_received",
        "child_support_expense",
    )
    # Workers' compensation awaits review; retirement was neutral.
    assert ACS_LOCAL_WORK_DISABILITY_FAMILY not in _OIP.families
    assert ACS_LOCAL_RETIREMENT_FAMILY not in _OIP.families
    # The buildable ASEC analog: UC, WC, VA, child support and OI_VAL leaves.
    assert _OIP.donor_components == (
        "unemployment_compensation",
        "workers_compensation",
        "veterans_benefits",
        "child_support_received",
        "alimony_income",
        "strike_benefits",
        "miscellaneous_income",
    )
    (definition, _aligned) = acs_local_income_predictor_extension_receipt()
    assert definition["not_carried_by_donor"] == {
        "FIN_VAL": "financial assistance from people outside the household"
    }
    assert set(definition["withheld_from"]) == {
        ACS_LOCAL_WORK_DISABILITY_FAMILY,
        ACS_LOCAL_RETIREMENT_FAMILY,
    }


def test_aligned_retp_stands_in_for_the_shared_analog_in_two_families_only():
    shared = acs_transfer_execution_contract_identity()
    shared_analog = shared["donor_combined_components"][
        ACS_LOCAL_SHARED_RETIREMENT_FEATURE
    ]
    assert _ALIGNED.replaces == ACS_LOCAL_SHARED_RETIREMENT_FEATURE
    assert (
        _ALIGNED.recipient_source
        == (shared["recipient_combined_sources"][ACS_LOCAL_SHARED_RETIREMENT_FEATURE])
    )
    assert _ALIGNED.families == (
        ACS_LOCAL_WORK_DISABILITY_FAMILY,
        ACS_LOCAL_RETIREMENT_FAMILY,
    )
    # The shared analog, plus the five distributions and disability benefits.
    assert list(ACS_LOCAL_ALIGNED_RETIREMENT_DONOR_COMPONENTS[:3]) == shared_analog
    assert ACS_LOCAL_ALIGNED_RETIREMENT_DONOR_COMPONENTS[3:] == (
        *ACS_LOCAL_INCOME_TRANSFER_FAMILIES[ACS_LOCAL_RETIREMENT_FAMILY],
        "disability_benefits",
    )
    (_oip, definition) = acs_local_income_predictor_extension_receipt()
    assert set(definition["not_carried_by_donor"]) == {"SUR_VAL", "DST code 7"}


def test_the_shared_contract_is_unchanged_and_the_local_one_binds_the_extensions():
    shared = acs_transfer_execution_contract_identity()
    assert "person_predictor_extensions" not in shared
    # The shared retirement predictor keeps its pension/regular-IRA analog.
    assert shared["donor_combined_components"][
        ACS_LOCAL_SHARED_RETIREMENT_FEATURE
    ] == list(ACS_LOCAL_INCOME_SHARED_RETIREMENT_COMPONENTS)
    assert ACS_LOCAL_OTHER_INCOME_FEATURE not in shared["person_optional_predictors"]
    local = acs_transfer_execution_contract_identity(
        person_predictor_extensions=ACS_LOCAL_INCOME_PREDICTOR_EXTENSIONS
    )
    assert local["sha256"] != shared["sha256"]
    assert local["person_predictor_extensions"] == [
        extension.identity() for extension in ACS_LOCAL_INCOME_PREDICTOR_EXTENSIONS
    ]
    assert {
        key: value
        for key, value in local.items()
        if key
        not in {
            "person_predictor_extensions",
            "sha256",
        }
    } == {key: value for key, value in shared.items() if key != "sha256"}


def _acs_rows(**person_columns: Any) -> Frame:
    """Four ACS persons: OIP 0, 1,000, blank (age 10) and 250."""

    columns = {
        "AGEP": [30, 45, 10, 70],
        "OIP": [0.0, 1_000.0, np.nan, 250.0],
        **person_columns,
    }
    return _raw_acs(n=4, **columns)


def test_oip_maps_to_adjusted_dollars_and_keeps_under_15_blanks_missing():
    frame = _acs_rows()
    result = map_acs_local_other_income(frame)
    person = result.frame.table("person")
    values = person[ACS_LOCAL_OTHER_INCOME_COLUMN].to_numpy(dtype=np.float64)
    np.testing.assert_allclose(values[[0, 1, 3]], [0.0, 1_100.0, 275.0])
    assert np.isnan(values[2])
    # The raw OIP stays as loaded; nothing else changes.
    pd.testing.assert_series_equal(person["OIP"], frame.table("person")["OIP"])
    assert ACS_LOCAL_OTHER_INCOME_COLUMN not in frame.table("person")
    coverage = result.coverage
    assert coverage["transformation"].startswith("OIP * ADJINC / 1_000_000")
    assert coverage["persons"] == 4
    assert coverage["observed_rows"] == 3
    assert coverage["blank_rows"] == 1
    assert coverage["positive_rows"] == 2
    dropped = without_acs_local_other_income(result.frame)
    assert list(dropped.table("person").columns) == list(frame.table("person").columns)
    with pytest.raises(ValueError, match="carries no 'acs_other_income'"):
        without_acs_local_other_income(frame)


@pytest.mark.parametrize(
    ("columns", "expected"),
    [
        ({"OIP": [0.0, 1_000.0, np.nan, np.nan]}, "contradicts its universe"),
        ({"OIP": [0.0, 1_000.0, 5.0, 250.0]}, "contradicts its universe"),
        ({"OIP": [0.0, -1.0, np.nan, 250.0]}, "non-negative"),
        ({"OIP": ["0", "x", None, "250"]}, "blank or a finite"),
        ({"acs_other_income": 1.0}, "refuses to overwrite"),
    ],
    ids=["blank-at-15-plus", "value-under-15", "negative", "non-numeric", "exists"],
)
def test_oip_mapping_refuses_invalid_sources(columns, expected):
    with pytest.raises(ValueError, match=expected):
        map_acs_local_other_income(_acs_rows(**columns))


def test_oip_mapping_requires_the_oip_column():
    frame = _acs_rows()
    person = frame.table("person").drop(columns=["OIP"])
    frame = Frame(
        {
            **{e: frame.table(e) for e in frame.entities if e != "person"},
            "person": person,
        },
        frame.schema,
        {"household": frame.weights_for("household")},
    )
    with pytest.raises(ValueError, match=r"requires person column\(s\) \['OIP'\]"):
        map_acs_local_other_income(frame)


# ---------------------------------------------------------------------------
# A real QRF fit through the multispine seam (review of microcosm#1056)
# ---------------------------------------------------------------------------

#: A shared-plan retirement leaf: the shared transfer, not this pass, owns it.
_SHARED_LEAF = "taxable_ira_distributions"
#: The PUF clone role's amounts sit far outside the ASEC role's support, so a
#: fit on the wrong role cannot go unnoticed.
_PUF_ROLE_AMOUNT = 5_000_000.0


def _support_donor(n_asec: int = 40) -> Frame:
    """An ASEC-by-PUF donor: an ASEC observation role and its PUF clones.

    Every person is a household. The ASEC role carries measured income (half
    of each leaf positive); each PUF clone repeats its source person's
    demographics with out-of-support amounts. Only the PUF role carries the
    shared-plan leaf.
    """

    rng = np.random.default_rng(1056)
    n = 2 * n_asec
    ids = np.arange(1, n + 1)
    roles = np.asarray(["asec"] * n_asec + ["puf_tax_detail"] * n_asec, dtype=object)
    person = pd.DataFrame(
        {
            "person_id": ids,
            "age": np.tile(rng.integers(18, 85, n_asec), 2).astype(float),
            "is_female": np.tile(rng.integers(0, 2, n_asec).astype(bool), 2),
        }
    )
    for entity in US_SCHEMA.entities:
        if entity != "person":
            person[f"person_{entity}_id"] = ids
    for offset, column in enumerate(ACS_LOCAL_INCOME_TRANSFER_COLUMNS):
        asec = np.where(
            rng.random(n_asec) < 0.5, rng.integers(100, 20_000, n_asec), 0
        ).astype(float)
        puf = np.where(asec > 0, _PUF_ROLE_AMOUNT + offset, 0.0)
        person[column] = np.concatenate([asec, puf])
    person[_SHARED_LEAF] = np.concatenate(
        [np.zeros(n_asec), np.where(rng.random(n_asec) < 0.5, 3_000.0, 0.0)]
    )
    # The extensions' other donor components (the shared pension analog and
    # the OIP leaves), on both roles.
    for column in _EXTENSION_ONLY_COMPONENTS:
        if column in person:
            continue
        values = np.where(rng.random(n_asec) < 0.3, rng.integers(100, 9_000, n_asec), 0)
        person[column] = np.tile(values.astype(float), 2)
    tables = {"person": person}
    for entity in US_SCHEMA.entities:
        if entity != "person":
            tables[entity] = pd.DataFrame({f"{entity}_id": ids})
    tables["household"]["state_fips"] = np.resize([6, 36, 48], n)
    for entity, table in tables.items():
        table[support_channel_column(entity)] = roles
        table[support_source_id_column(entity)] = np.tile(np.arange(1, n_asec + 1), 2)
        table[support_clone_index_column(entity)] = (roles == "puf_tax_detail").astype(
            int
        )
    return Frame(
        tables,
        US_SCHEMA,
        {"household": Weights(np.full(n, 100.0), WeightKind.DESIGN)},
        pd.Series(ASEC_PUF_DONOR_SPINE, index=person.index, dtype=object),
    )


def _raw_acs(n: int = 24, *, children: int = 0, **person_columns: Any) -> Frame:
    """A raw ACS PUMS spine: two-person households, one PUMA.

    Everyone is an adult except the last ``children`` persons, aged 10 with
    the Census blanks for OIP and RETP.
    """

    rng = np.random.default_rng(2024)
    households = 500 + np.arange(n) // 2
    person = pd.DataFrame(
        {
            "person_id": np.arange(500, 500 + n),
            "person_household_id": households,
            "person_tax_unit_id": households,
            "person_spm_unit_id": households,
            "person_family_id": households,
            "person_marital_unit_id": households,
            "AGEP": rng.integers(18, 85, n),
            "SEX": rng.integers(1, 3, n),
            "RELSHIPP": np.resize([20, 21], n),
            "ADJINC": 1_100_000,
            "WAGP": rng.integers(0, 60_000, n).astype(float),
            "SEMP": 0.0,
            "SSP": 0.0,
            "SSIP": 0.0,
            "RETP": np.where(rng.random(n) < 0.3, 12_000.0, 0.0),
            "INTP": 0.0,
            "OIP": np.where(rng.random(n) < 0.3, 4_000.0, 0.0),
            **person_columns,
        }
    )
    if children:
        person.loc[n - children :, ["AGEP"]] = 10
        person.loc[n - children :, ["OIP", "RETP"]] = np.nan
    unique = np.unique(households)
    return Frame(
        {
            "person": person,
            "household": pd.DataFrame(
                {
                    "household_id": unique,
                    "state_fips": np.resize([6, 36, 48], len(unique)),
                    "puma": "0600100",
                    "ADJHSG": 1_000_000,
                    "TEN": 3,
                    "RNTP": 1_000.0,
                    "GRNTP": 1_200.0,
                    "TAXAMT": np.nan,
                }
            ),
            "tax_unit": pd.DataFrame({"tax_unit_id": unique}),
            "spm_unit": pd.DataFrame({"spm_unit_id": unique}),
            "family": pd.DataFrame({"family_id": unique}),
            "marital_unit": pd.DataFrame({"marital_unit_id": unique}),
        },
        US_SCHEMA,
        {"household": Weights(np.full(len(unique), 50.0), WeightKind.DESIGN)},
        pd.Series(ACS_2024_1YR_SPINE, index=person.index, dtype=object),
    )


def _multispine(
    monkeypatch,
    tmp_path,
    base: Frame,
    raw: Frame,
    calls: list,
    *,
    income_transfer: bool = True,
):
    """Run the local lane's multispine seam with the real QRF.

    Only the ACS loader is replaced (by ``raw``); every
    ``transfer_acs_inputs`` call is recorded and delegated to the real one.
    """

    real_transfer = acs_multispine.transfer_acs_inputs

    def recorded(recipient, donor, **kwargs):
        calls.append(kwargs)
        return real_transfer(recipient, donor, **kwargs)

    monkeypatch.setattr(
        acs_multispine, "build_acs_pums_unit_frame", lambda *_a, **_k: (raw, {})
    )
    monkeypatch.setattr(acs_multispine, "transfer_acs_inputs", recorded)
    return acs_multispine.build_optional_acs_multispine(
        base,
        AcsPumsSource(tmp_path / "csv_hus.zip", tmp_path / "csv_pus.zip"),
        target_families={"person": {"shared_retirement": (_SHARED_LEAF,)}},
        income_transfer=income_transfer,
        seed=5,
        n_estimators=5,
    )


def test_a_real_fit_fills_missing_acs_income_from_the_asec_role_only(
    monkeypatch, tmp_path
):
    base = _support_donor()
    donor_before = base.table("person").copy()
    stored = np.full(24, np.nan)
    stored[[0, 5]] = [777.0, 0.0]
    raw = _raw_acs(children=2, child_support_received=stored)
    calls: list = []

    result = _multispine(monkeypatch, tmp_path, base, raw, calls)

    person = result.frame.table("person")
    acs = person.loc[person[TAG].eq(ACS_2024_1YR_SPINE)].reset_index(drop=True)
    donor = person.loc[person[TAG].eq(ASEC_PUF_DONOR_SPINE)].reset_index(drop=True)
    asec_role = donor_before[support_channel_column("person")].eq("asec")
    assert len(acs) == 24
    # Every missing ACS cell of all nine leaves is filled finite and >= 0,
    # inside the ASEC role's measured support, never the PUF clones' values.
    for column in ACS_LOCAL_INCOME_TRANSFER_COLUMNS:
        values = acs[column].to_numpy(dtype=np.float64)
        assert np.isfinite(values).all(), column
        assert (values >= 0).all(), column
        assert values.max() <= donor_before.loc[asec_role, column].max(), column
        assert (acs[column] > 0).any(), column
    # Stored ACS values survive the null-only merge, a stored 0 included.
    assert acs.loc[0, "child_support_received"] == 777.0
    assert acs.loc[5, "child_support_received"] == 0.0
    # Donor rows, and the donor frame itself, are unchanged.
    for column in (*ACS_LOCAL_INCOME_TRANSFER_COLUMNS, _SHARED_LEAF):
        np.testing.assert_array_equal(
            donor[column].to_numpy(), donor_before[column].to_numpy()
        )
    pd.testing.assert_frame_equal(base.table("person"), donor_before)

    # Two real fits: this pass on the ASEC role, then the shared plan on the
    # PUF role; neither transfers the other's leaves.
    assert [call["donor_channel"] for call in calls] == [
        ACS_LOCAL_INCOME_DONOR_CHANNEL,
        ACS_DONOR_CHANNEL_AUTO,
    ]
    assert calls[0]["target_families"] == acs_local_income_transfer_target_families()
    # The extensions go to the local call only; the shared call passes none.
    assert calls[0]["person_predictor_extensions"] == (
        ACS_LOCAL_INCOME_PREDICTOR_EXTENSIONS
    )
    assert "person_predictor_extensions" not in calls[1]
    # The mapped OIP was a predictor source only: it never reaches the pool.
    assert ACS_LOCAL_OTHER_INCOME_COLUMN not in person
    imputed = result.provenance["imputed_inputs"]
    local = {item["column"]: item for item in imputed if item["column"] != _SHARED_LEAF}
    assert set(local) == set(ACS_LOCAL_INCOME_TRANSFER_COLUMNS)
    assert len(imputed) == len(ACS_LOCAL_INCOME_TRANSFER_COLUMNS) + 1
    for item in local.values():
        assert item["donor_channel"] == ACS_LOCAL_INCOME_DONOR_CHANNEL
    # OIP only for the child support family; the aligned RETP analog, in place
    # of the shared one, only for the work/disability and retirement families.
    for column, item in local.items():
        used = set(item["predictors"])
        child_support = _FAMILY_OF[column] == ACS_LOCAL_CHILD_SUPPORT_FAMILY
        assert (ACS_LOCAL_OTHER_INCOME_FEATURE in used) is child_support, column
        assert (ACS_LOCAL_ALIGNED_RETIREMENT_FEATURE in used) is not child_support
        assert (ACS_LOCAL_SHARED_RETIREMENT_FEATURE in used) is child_support
        # Adults are fit with the extension; the two children (OIP and RETP
        # blank) in a pattern without it.
        patterns = {
            tuple(pattern["observed_optional_predictors"]): pattern["recipient_rows"]
            for pattern in item["patterns"]
        }
        extension = _OIP if child_support else _ALIGNED
        assert (
            sum(
                rows
                for observed, rows in patterns.items()
                if extension.feature in observed
            )
            == 22
        )
        assert patterns[()] == 2
    (shared,) = [item for item in imputed if item["column"] == _SHARED_LEAF]
    assert shared["donor_channel"] == "puf_tax_detail"
    assert result.provenance["fit_configuration"]["income_donor_channel"] == (
        ACS_LOCAL_INCOME_DONOR_CHANNEL
    )

    # The receipt comes from the fit and the gate accepts it.
    receipt = result.provenance["acs_local_income_transfer"]
    assert receipt["issue"] == ACS_LOCAL_INCOME_TRANSFER_ISSUE
    assert receipt["method"] == ACS_LOCAL_INCOME_TRANSFER_METHOD
    assert receipt["donor_channel"] == ACS_LOCAL_INCOME_DONOR_CHANNEL
    assert receipt["acs_persons"] == 24
    assert receipt["donor"]["person_rows"] == 40
    assert receipt["predictor_extensions"] == (
        acs_local_income_predictor_extension_receipt()
    )
    assert receipt["acs_other_income"]["observed_rows"] == 22
    assert receipt["acs_other_income"]["blank_rows"] == 2
    for extension in ACS_LOCAL_INCOME_PREDICTOR_EXTENSIONS:
        coverage = receipt["donor"]["predictor_extensions"][extension.feature]
        assert 0 < coverage["weighted_recipient_share"] <= 1
        assert (
            coverage["weighted_recipient_share_age_15_plus"]
            == (coverage["weighted_recipient_share"])
        )
    assert (
        ACS_LOCAL_OTHER_INCOME_FEATURE
        in (receipt["family_predictors"][ACS_LOCAL_CHILD_SUPPORT_FAMILY])
    )
    for column in ACS_LOCAL_INCOME_TRANSFER_COLUMNS:
        entry = receipt["columns"][column]
        assert entry["imputed_rows"] == (
            22 if column == "child_support_received" else 24
        )
        assert entry["unmodeled_rows"] == 0
        assert entry["donor_channel"] == [ACS_LOCAL_INCOME_DONOR_CHANNEL]
        assert entry["predictors"]
    gate = acs_local_income_transfer_signal_gate(result.frame, receipt=receipt)
    assert gate.passed, gate.failures


def test_a_real_fit_refuses_a_plan_that_overlaps_the_shared_plan(monkeypatch, tmp_path):
    shared_leaves = set(ACS_LOCAL_INCOME_SHARED_RETIREMENT_COMPONENTS)
    assert shared_leaves <= required_acs_transfer_inputs()
    overlapping = {
        **ACS_LOCAL_INCOME_TRANSFER_FAMILIES,
        "acs_local_retirement_distributions": (
            *ACS_LOCAL_INCOME_TRANSFER_FAMILIES["acs_local_retirement_distributions"],
            _SHARED_LEAF,
        ),
    }
    monkeypatch.setattr(
        acs_local_income, "ACS_LOCAL_INCOME_TRANSFER_FAMILIES", overlapping
    )
    monkeypatch.setattr(
        acs_local_income,
        "ACS_LOCAL_INCOME_TRANSFER_COLUMNS",
        (*ACS_LOCAL_INCOME_TRANSFER_COLUMNS, _SHARED_LEAF),
    )
    calls: list = []
    with pytest.raises(ValueError, match="double counting"):
        _multispine(monkeypatch, tmp_path, _support_donor(), _raw_acs(), calls)
    # Refused before any fit: no transfer ran, so no leaf was written twice.
    assert calls == []


def test_the_local_extensions_leave_the_shared_transfer_draws_unchanged(
    monkeypatch, tmp_path
):
    """The shared plan's draws, seeds and predictors are byte-identical with
    and without the local pass and its predictor extensions."""

    base = _support_donor()
    raw = _raw_acs(children=2)
    with_calls: list = []
    without_calls: list = []
    with_pass = _multispine(monkeypatch, tmp_path, base, raw, with_calls)
    # Snapshot: the second run wraps (and so also records into) this recorder.
    with_calls = list(with_calls)
    without_pass = _multispine(
        monkeypatch, tmp_path, base, raw, without_calls, income_transfer=False
    )

    assert len(with_calls) == 2 and len(without_calls) == 1
    assert with_calls[1] == without_calls[0]

    def shared(result):
        person = result.frame.table("person")
        acs = person.loc[person[TAG].eq(ACS_2024_1YR_SPINE)].reset_index(drop=True)
        (item,) = [
            item
            for item in result.provenance["imputed_inputs"]
            if item["column"] == _SHARED_LEAF
        ]
        return acs[_SHARED_LEAF].to_numpy(dtype=np.float64), item

    with_draws, with_item = shared(with_pass)
    without_draws, without_item = shared(without_pass)
    np.testing.assert_array_equal(with_draws, without_draws)
    assert with_item == without_item
    assert ACS_LOCAL_OTHER_INCOME_FEATURE not in with_item["predictors"]
    assert ACS_LOCAL_ALIGNED_RETIREMENT_FEATURE not in with_item["predictors"]
    assert ACS_LOCAL_SHARED_RETIREMENT_FEATURE in with_item["predictors"]
