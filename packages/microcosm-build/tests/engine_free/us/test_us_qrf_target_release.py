"""The CPS-ORG wage imputation and the ACS input transfer fit one target at a time.

Each caller fits a target, draws it for the recipients and frees its forests
and sign classifier before it fits the next target. These tests pin that the
draws stay byte-identical to what the callers produced when they fitted every
target first and predicted afterwards, and that no earlier target's fitted
objects are alive when the next target's fit starts.

Two kinds of comparison:

- In this process, against ``FitThenPredictQRF``: the real model run the old
  way (one ``fit`` over every target, then one ``predict``) on the callers'
  own inputs. These run on every platform.
- Against SHA-256 digests of every output table, recorded by running
  origin/main 3b75cf151's ``impute_us_org_wages`` and ``transfer_acs_inputs``
  (which fitted every target, then predicted) on these fixtures. QRF draws are
  bitwise reproducible only within one platform (see
  ``docs/graph-qrf-cross-platform.md``), so the digests are kept per operating
  system and machine: macOS arm64 natively, and Linux x86_64 (CI's platform)
  under Docker emulation on Apple silicon. Python 3.13 and 3.14 gave the same
  bytes on each, as did single-threaded native libraries and QRF thread
  pools. They are checked only where the platform and the versions of the
  libraries that compute the draws match the recording
  (``_RECORDED_DEPENDENCIES``); elsewhere these tests skip.

A change that is meant to move these outputs re-records the digests by
running the code it starts from on each platform, not the changed code.
"""

from __future__ import annotations

import dataclasses
import platform
from collections import Counter
from importlib.metadata import version
from pathlib import Path

import numpy as np
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
from microcosm.fit import RegimeGatedQRF
from microcosm.frame import Frame
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
# origin/main's output digests, by platform
# ---------------------------------------------------------------------------

_CARRIED = "e6cc97f0c8b7f14b052697c6f9f1ca5fa22fd6901a2d58fc7152b99b79f20727"
_HOUSEHOLD = "afeb86cd88e684a8ade35d755d0600623c0fef67a4941c2de347ef7d3a99fd9c"

_ORIGIN_MAIN_DIGESTS: dict[tuple[str, str], dict[str, dict]] = {
    ("Darwin", "arm64"): {
        "org": {
            "carried": _CARRIED,
            "wages": "cea79a74a73a82d531ef8e449933689a29a2a5dd6766e2477bbb9e74a3309d3c",
        },
        "acs": {
            "shared": {
                "person": "0cb0c10244a9dabeb32e5990d22a83ba03577047dbabd5554754269de72fb648",
                "household": _HOUSEHOLD,
                "tax_unit": "61592c687abd6d9ce359654b7ea6c57c9c37214c702195d4816e72106c59ecb1",
            },
            "extension": {
                "person": "2f80f645419a7d0a11abc82dc3a10ffec808e9eac4a5be9683dfb1603917ab70",
                "household": _HOUSEHOLD,
                "tax_unit": "61592c687abd6d9ce359654b7ea6c57c9c37214c702195d4816e72106c59ecb1",
            },
            "two_targets_per_fit": {
                "person": "ba46f5eb133e772d147c7600ddb5e8effe83f98cdf484bf0167b621debf3b33c",
                "household": _HOUSEHOLD,
                "tax_unit": "61592c687abd6d9ce359654b7ea6c57c9c37214c702195d4816e72106c59ecb1",
            },
        },
    },
    ("Linux", "x86_64"): {
        "org": {
            "carried": _CARRIED,
            "wages": "76731096ec682835990da06233f08b712475ceb78ccf29d198396cc147097d60",
        },
        "acs": {
            "shared": {
                "person": "62d53e4838cc36bb24946a5bfdb75bd42107a5e74561c590d9cac0ae273e87d2",
                "household": _HOUSEHOLD,
                "tax_unit": "6c98d6272405549d61daf548764c7a459b5fe0a644acc6a9c640c469c74cf294",
            },
            "extension": {
                "person": "18c385b4f77c2c895e63cfd8df40a01ff0ed624283003b13e7f6fd78265e6a94",
                "household": _HOUSEHOLD,
                "tax_unit": "6c98d6272405549d61daf548764c7a459b5fe0a644acc6a9c640c469c74cf294",
            },
            "two_targets_per_fit": {
                "person": "cb91f8676be420d48610ea2b6c0726408d0aa427553f1ad2ef56627c0a515b16",
                "household": _HOUSEHOLD,
                "tax_unit": "6c98d6272405549d61daf548764c7a459b5fe0a644acc6a9c640c469c74cf294",
            },
        },
    },
}


#: Versions of the libraries that compute the draws, on both recordings.
_RECORDED_DEPENDENCIES = {
    "numpy": "2.4.6",
    "pandas": "3.0.3",
    "scipy": "1.17.1",
    "scikit-learn": "1.8.0",
    "quantile-forest": "1.4.2",
}


def _origin_main_digests() -> dict[str, dict]:
    """This platform's recorded digests, or a skip where they do not apply."""

    here = (platform.system(), platform.machine())
    recorded = _ORIGIN_MAIN_DIGESTS.get(here)
    if recorded is None:
        pytest.skip(
            f"origin/main's digests were recorded on {sorted(_ORIGIN_MAIN_DIGESTS)}, "
            f"not {here}; the in-process comparisons cover this platform."
        )
    installed = {name: version(name) for name in _RECORDED_DEPENDENCIES}
    if installed != _RECORDED_DEPENDENCIES:
        pytest.skip(
            f"origin/main's digests were recorded with {_RECORDED_DEPENDENCIES}, "
            f"not {installed}; the in-process comparisons cover these versions."
        )
    return recorded


# ---------------------------------------------------------------------------
# CPS-ORG wage imputation
# ---------------------------------------------------------------------------


def _impute_org_fixture() -> tuple[pd.DataFrame, pd.DataFrame]:
    frame, donor = org_wage_fixture()
    return impute_us_org_wages(
        frame,
        donor,
        seed=ORG_FIXTURE_SEED,
        n_estimators=ORG_FIXTURE_N_ESTIMATORS,
    )


def test_org_wage_imputation_reproduces_origin_main_bytes() -> None:
    expected = _origin_main_digests()["org"]
    carried, wages = _impute_org_fixture()

    assert {
        "carried": table_bytes_digest(carried),
        "wages": table_bytes_digest(wages),
    } == expected
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
    expected = _origin_main_digests()["acs"][run]
    result = _transfer_acs_fixture(run)

    assert _table_digests(result) == expected
    person = result.frame.table("person")
    for target in ("rental_income", "short_term_capital_gains"):
        values = person[target].to_numpy()
        assert (values < 0).any() and (values == 0).any() and (values > 0).any()
    assert all(len(record.patterns) > 1 for record in result.imputed_inputs)


def test_banked_acs_transfer_reproduces_origin_main_bytes(tmp_path: Path) -> None:
    expected = _origin_main_digests()["acs"]["shared"]
    cold = _transfer_acs_fixture("shared", target_bank=_bank_store(tmp_path / "bank"))
    resumed = _transfer_acs_fixture(
        "shared", target_bank=_bank_store(tmp_path / "bank")
    )

    assert _table_digests(cold) == expected
    assert _table_digests(resumed) == expected


def _assert_same_transfer(
    result: AcsTransferResult, reference: AcsTransferResult
) -> None:
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


@pytest.mark.parametrize("run", sorted(_ACS_RUN_OPTIONS))
def test_acs_transfer_matches_fitting_every_target_first(
    run: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    result = _transfer_acs_fixture(run)
    monkeypatch.setattr(acs_transfer_module, "QRF", FitThenPredictQRF)
    reference = _transfer_acs_fixture(run)

    _assert_same_transfer(result, reference)


def test_banked_acs_transfer_matches_fitting_every_target_first(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cold = _transfer_acs_fixture("shared", target_bank=_bank_store(tmp_path / "bank"))
    resumed = _transfer_acs_fixture(
        "shared", target_bank=_bank_store(tmp_path / "bank")
    )
    monkeypatch.setattr(acs_transfer_module, "QRF", FitThenPredictQRF)
    reference = _transfer_acs_fixture("shared")

    _assert_same_transfer(cold, reference)
    _assert_same_transfer(resumed, reference)


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


class _MisreportedRegimeQRF(RegimeGatedQRF):
    """The real QRF, but every chain step reports a regime its data never has."""

    def fit_draw_next(self, *args: object, **kwargs: object) -> object:
        step = super().fit_draw_next(*args, **kwargs)
        return dataclasses.replace(step, regime="misreported")


@pytest.mark.parametrize("banked", [False, True], ids=["in_memory", "banked"])
def test_acs_transfer_refuses_a_target_drawn_under_another_regime(
    banked: bool,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(acs_transfer_module, "QRF", _MisreportedRegimeQRF)
    target_bank = _bank_store(tmp_path / "bank") if banked else None

    with pytest.raises(RuntimeError, match="reported regime 'misreported'"):
        _transfer_acs_fixture("shared", target_bank=target_bank)


def _first_target_part(step: object, part: str) -> object:
    regressor = step.fitted._target_models["first"].positive.model
    if part == "leaf array":
        return regressor.forest_.y_train_leaves.base
    return regressor.estimators_[0]


@pytest.mark.parametrize("part", ["leaf array", "tree"])
def test_lifetime_check_notices_a_kept_part_of_an_earlier_forest(
    part: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rng = np.random.default_rng(3)
    donor = pd.DataFrame(
        {
            "x": rng.normal(size=200),
            "first": rng.lognormal(size=200),
            "second": rng.lognormal(size=200),
        }
    )
    recipients = donor.loc[:, ["x"]]
    model = RegimeGatedQRF(n_estimators=2, seed=4)
    kept: list[object] = []
    with tracked_target_fits(monkeypatch) as lifetimes:
        state = model.start_chain(donor, ["x"], ["first", "second"], weights="none")
        step = model.fit_draw_next(
            donor,
            recipients,
            pd.DataFrame(index=donor.index),
            state=state,
            weights="none",
        )
        kept.append(_first_target_part(step, part))
        priors = pd.DataFrame({"first": step.raw_draw}, index=donor.index)
        state = step.state
        del step
        with pytest.raises(AssertionError, match=f"'first', 'positive {part}'"):
            model.fit_draw_next(donor, recipients, priors, state=state, weights="none")

    assert len(lifetimes.violations) == 1


def _with_joint_immigration_codec(donor: Frame) -> Frame:
    """The donor with the two immigration columns one joint model target encodes."""

    tables = {name: donor.table(name).copy() for name in donor.entities}
    person = tables["person"]
    non_citizen = np.random.default_rng(13).random(len(person)) < 0.3
    person["ssn_card_type"] = np.where(non_citizen, "NONE", "CITIZEN")
    person["immigration_status_str"] = np.where(non_citizen, "UNDOCUMENTED", "CITIZEN")
    return Frame(
        tables,
        donor.schema,
        {name: donor.weights_for(name) for name in donor.weighted_entities},
        donor.strata,
    )


@pytest.mark.parametrize("banked", [False, True], ids=["in_memory", "banked"])
def test_joint_codec_transfer_matches_fitting_every_target_first(
    banked: bool,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Two exported columns drawn from one model target, as before."""

    families = {
        "person": {
            "fixture_immigration": ("ssn_card_type", "immigration_status_str"),
            **ACS_FIXTURE_FAMILIES["person"],
        }
    }

    def run(target_bank: AcsTransferTargetBankStore | None) -> AcsTransferResult:
        recipient, donor = acs_transfer_fixture()
        return transfer_acs_inputs(
            recipient,
            _with_joint_immigration_codec(donor),
            target_families=families,
            seed=ACS_FIXTURE_SEED,
            n_estimators=ACS_FIXTURE_N_ESTIMATORS,
            target_bank=target_bank,
        )

    result = run(_bank_store(tmp_path / "bank") if banked else None)
    monkeypatch.setattr(acs_transfer_module, "QRF", FitThenPredictQRF)
    reference = run(None)

    _assert_same_transfer(result, reference)
    person = result.frame.table("person")
    assert set(person["immigration_status_str"]) == {"CITIZEN", "UNDOCUMENTED"}


class _FitPredictOnlyQRF:
    """A QRF stand-in with ``fit`` and ``predict`` but no chain methods."""

    def __init__(self, **_options: object) -> None:
        pass

    def fit(self, *_args: object, **_kwargs: object) -> object:
        raise AssertionError("the transfer must not fall back to fit")


@pytest.mark.parametrize("banked", [False, True], ids=["in_memory", "banked"])
def test_acs_transfer_refuses_a_qrf_without_chain_methods(
    banked: bool,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(acs_transfer_module, "QRF", _FitPredictOnlyQRF)
    target_bank = _bank_store(tmp_path / "bank") if banked else None

    with pytest.raises(TypeError, match="start_chain and fit_draw_next"):
        _transfer_acs_fixture("shared", target_bank=target_bank)
