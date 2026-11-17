"""The CPS-ORG wage imputation and the ACS input transfer fit one target at a time.

Each caller fits a target, draws it for the recipients and frees its forests
and sign classifier before it fits the next target. These tests pin that the
draws stay byte-identical to what the callers produced when they fitted every
target first and predicted afterwards, and that no earlier target's fitted
objects are alive when the next target's fit starts.

The frozen digests were recorded by running origin/main 3b75cf151's
``impute_us_org_wages`` and ``transfer_acs_inputs`` (which fitted every target,
then predicted) on these fixtures, with and without single-threaded native
libraries and QRF thread pools; both gave the same bytes.
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path

import pandas as pd
import pytest

import microcosm.build.us_runtime.acs_transfer as acs_transfer_module
from microcosm.build.us_runtime.acs_transfer import (
    AcsPersonPredictorExtension,
    AcsTransferResult,
    transfer_acs_inputs,
)
from microcosm.build.us_runtime.acs_transfer_bank import AcsTransferTargetBankStore
from microcosm.build.us_runtime.org_wages import ORG_QRF_TARGETS, impute_us_org_wages
from test_support.microcosm_build.qrf_chain_doubles import (
    FitThenPredictQRF,
    table_bytes_digest,
    tracked_target_fits,
)
from test_support.microcosm_build.qrf_per_target import (
    ACS_FIXTURE_FAMILIES,
    ACS_FIXTURE_N_ESTIMATORS,
    ACS_FIXTURE_REGIME_EVIDENCE,
    ACS_FIXTURE_SEED,
    ORG_FIXTURE_N_ESTIMATORS,
    ORG_FIXTURE_SEED,
    acs_transfer_fixture,
    org_wage_fixture,
)

# ---------------------------------------------------------------------------
# CPS-ORG wage imputation
# ---------------------------------------------------------------------------

_ORG_ORIGIN_MAIN_DIGESTS = {
    "carried": "e6cc97f0c8b7f14b052697c6f9f1ca5fa22fd6901a2d58fc7152b99b79f20727",
    "wages": "cea79a74a73a82d531ef8e449933689a29a2a5dd6766e2477bbb9e74a3309d3c",
}


def _impute_org_fixture() -> tuple[pd.DataFrame, pd.DataFrame]:
    frame, donor = org_wage_fixture()
    return impute_us_org_wages(
        frame,
        donor,
        seed=ORG_FIXTURE_SEED,
        n_estimators=ORG_FIXTURE_N_ESTIMATORS,
    )


def test_org_wage_imputation_reproduces_origin_main_bytes() -> None:
    carried, wages = _impute_org_fixture()

    assert {
        "carried": table_bytes_digest(carried),
        "wages": table_bytes_digest(wages),
    } == _ORG_ORIGIN_MAIN_DIGESTS
    # The fixture draws both is_paid_hourly classes and positive wages.
    active = wages["hourly_wage"].to_numpy() > 0
    assert active.any() and not active.all()
    assert wages.loc[active, "is_paid_hourly"].nunique() == 2


def test_org_wage_imputation_matches_fitting_every_target_first(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    carried, wages = _impute_org_fixture()
    monkeypatch.setattr("microcosm.fit.QRF", FitThenPredictQRF)
    reference_carried, reference_wages = _impute_org_fixture()

    for actual, expected in ((carried, reference_carried), (wages, reference_wages)):
        pd.testing.assert_frame_equal(actual, expected, check_exact=True)
        assert table_bytes_digest(actual) == table_bytes_digest(expected)


def test_org_wage_imputation_frees_each_target_before_fitting_the_next(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with tracked_target_fits(monkeypatch) as lifetimes:
        _impute_org_fixture()
        assert lifetimes.alive() == []

    assert lifetimes.violations == []
    assert lifetimes.fitted_targets == list(ORG_QRF_TARGETS)


# ---------------------------------------------------------------------------
# ACS input transfer
# ---------------------------------------------------------------------------

_INTEREST_EXTENSION = AcsPersonPredictorExtension(
    feature="__acs_transfer_fixture_interest_total",
    donor_components=("taxable_interest_income",),
    recipient_source="taxable_interest_income",
    families=("fixture_income",),
)

#: Transfer options per run: the shared plan, an opt-in person predictor
#: (as the local-area tool's income pass uses), and families split into
#: batches of two targets (the tool passes its own batch size).
_ACS_RUN_OPTIONS: dict[str, dict[str, object]] = {
    "shared": {},
    "extension": {"person_predictor_extensions": (_INTEREST_EXTENSION,)},
    "two_targets_per_fit": {"max_targets_per_fit": 2},
}

_ACS_ORIGIN_MAIN_DIGESTS = {
    "shared": {
        "person": "0cb0c10244a9dabeb32e5990d22a83ba03577047dbabd5554754269de72fb648",
        "household": "afeb86cd88e684a8ade35d755d0600623c0fef67a4941c2de347ef7d3a99fd9c",
        "tax_unit": "61592c687abd6d9ce359654b7ea6c57c9c37214c702195d4816e72106c59ecb1",
    },
    "extension": {
        "person": "2f80f645419a7d0a11abc82dc3a10ffec808e9eac4a5be9683dfb1603917ab70",
        "household": "afeb86cd88e684a8ade35d755d0600623c0fef67a4941c2de347ef7d3a99fd9c",
        "tax_unit": "61592c687abd6d9ce359654b7ea6c57c9c37214c702195d4816e72106c59ecb1",
    },
    "two_targets_per_fit": {
        "person": "ba46f5eb133e772d147c7600ddb5e8effe83f98cdf484bf0167b621debf3b33c",
        "household": "afeb86cd88e684a8ade35d755d0600623c0fef67a4941c2de347ef7d3a99fd9c",
        "tax_unit": "61592c687abd6d9ce359654b7ea6c57c9c37214c702195d4816e72106c59ecb1",
    },
}


def _transfer_acs_fixture(
    run: str,
    *,
    target_bank: AcsTransferTargetBankStore | None = None,
) -> AcsTransferResult:
    recipient, donor = acs_transfer_fixture()
    return transfer_acs_inputs(
        recipient,
        donor,
        target_families=ACS_FIXTURE_FAMILIES,
        seed=ACS_FIXTURE_SEED,
        n_estimators=ACS_FIXTURE_N_ESTIMATORS,
        regime_evidence_targets=ACS_FIXTURE_REGIME_EVIDENCE,
        target_bank=target_bank,
        **_ACS_RUN_OPTIONS[run],
    )


def _bank_store(root: Path) -> AcsTransferTargetBankStore:
    return AcsTransferTargetBankStore(
        root,
        identity={
            "artifact_kind": "fixture_acs_transfer_target_bank_identity",
            "fixture_version": 1,
            "name": "qrf_target_release",
        },
    )


def _table_digests(result: AcsTransferResult) -> dict[str, str]:
    return {
        entity: table_bytes_digest(result.frame.table(entity))
        for entity in result.frame.entities
    }


@pytest.mark.parametrize("run", sorted(_ACS_RUN_OPTIONS))
def test_acs_transfer_reproduces_origin_main_bytes(run: str) -> None:
    result = _transfer_acs_fixture(run)

    assert _table_digests(result) == _ACS_ORIGIN_MAIN_DIGESTS[run]
    person = result.frame.table("person")
    for target in ("rental_income", "short_term_capital_gains"):
        values = person[target].to_numpy()
        assert (values < 0).any() and (values == 0).any() and (values > 0).any()
    assert all(len(record.patterns) > 1 for record in result.imputed_inputs)


def test_banked_acs_transfer_reproduces_origin_main_bytes(tmp_path: Path) -> None:
    cold = _transfer_acs_fixture("shared", target_bank=_bank_store(tmp_path / "bank"))
    resumed = _transfer_acs_fixture(
        "shared", target_bank=_bank_store(tmp_path / "bank")
    )

    assert _table_digests(cold) == _ACS_ORIGIN_MAIN_DIGESTS["shared"]
    assert _table_digests(resumed) == _ACS_ORIGIN_MAIN_DIGESTS["shared"]


@pytest.mark.parametrize("run", sorted(_ACS_RUN_OPTIONS))
def test_acs_transfer_matches_fitting_every_target_first(
    run: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    result = _transfer_acs_fixture(run)
    monkeypatch.setattr(acs_transfer_module, "QRF", FitThenPredictQRF)
    reference = _transfer_acs_fixture(run)

    assert result.frame.entities == reference.frame.entities
    for entity in reference.frame.entities:
        pd.testing.assert_frame_equal(
            result.frame.table(entity),
            reference.frame.table(entity),
            check_exact=True,
        )
    assert _table_digests(result) == _table_digests(reference)
    assert result.imputed_inputs == reference.imputed_inputs
    assert result.fit_records == reference.fit_records


@pytest.mark.parametrize("banked", [False, True], ids=["in_memory", "banked"])
def test_acs_transfer_frees_each_target_before_fitting_the_next(
    banked: bool,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    target_bank = _bank_store(tmp_path / "bank") if banked else None
    with tracked_target_fits(monkeypatch) as lifetimes:
        result = _transfer_acs_fixture("shared", target_bank=target_bank)
        assert lifetimes.alive() == []

    assert lifetimes.violations == []
    # Every target is fitted once per availability pattern of its family.
    assert Counter(lifetimes.fitted_targets) == {
        record.column: len(record.patterns) for record in result.imputed_inputs
    }
