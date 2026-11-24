"""The ACS transfer conditions later targets on raw draws, not exported values.

The transfer fits one target at a time: each later target in a family is drawn
on the recipients' raw float64 draws of the earlier ones
(``RegimeGatedQRF.fit_draw_next``). Two kinds of exported value differ from
those raw draws, and neither may reach a later target:

- ``first_home_mortgage_origination_year`` is ``discrete_numeric``: the
  transfer snaps its draws to the donor support when it decodes them, after
  every family is fitted;
- the self-employment components are set to zero after the transfer where
  the recipient's own total is kept (``keep_measured_self_employment_totals``).

The fixture (:mod:`test_support.microcosm_build.qrf_raw_priors`) puts the
year ahead of ``health_savings_account_ald``, and the components ahead of
``farm_rent_income`` and ``estate_income_would_be_qualified``. These tests pin
that:

- the outputs are byte-identical to what e89769879c (main before #1667, which
  fitted every target and then predicted) produced on this fixture, where
  those digests were recorded;
- they equal ``FitThenPredictQRF``'s outputs, the real model run that old
  way, on every platform;
- every prior the chain feeds forward is byte-equal to the earlier raw draw,
  in chain order, and the year is snapped and the components zeroed only on
  export;
- the fixture would catch the change on the running platform: a chain fed
  the snapped year or the zeroed components draws different later targets.

QRF draws are bitwise reproducible only within one platform (see
``docs/graph-qrf-cross-platform.md``), so the digests are kept per operating
system and machine, and are checked only where the platform and the versions
of the libraries that compute the draws match the recording
(``_RECORDED_DEPENDENCIES``); elsewhere those two tests skip. Only macOS arm64
is recorded.

A change that is meant to move these outputs records new digests from its own
code, says why the outputs move, and renames what here says the digests come
from e89769879c (this text, ``_PRE_1667_DIGESTS``, ``_pre_1667_digests`` and
the ``*_reproduces_pre_1667_bytes`` tests). Any other change must leave them
passing.
"""

from __future__ import annotations

import platform
from importlib.metadata import version
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import microcosm.build.us_runtime.acs_transfer as acs_transfer_module
from microcosm.build.us_runtime.acs_transfer import (
    ACS_SELF_EMPLOYMENT_COMPONENT_INPUTS,
    AcsTransferResult,
    transfer_acs_inputs,
)
from microcosm.build.us_runtime.acs_transfer_bank import AcsTransferTargetBankStore
from test_support.microcosm_build.qrf_chain_doubles import (
    FitThenPredictQRF,
    table_bytes_digest,
)
from test_support.microcosm_build.qrf_raw_priors import (
    ESTATE_FLAG,
    FARM_RENT,
    HSA,
    MORTGAGE_BALANCE,
    ORIGINATION_YEAR,
    RAW_PRIOR_FIXTURE_FAMILIES,
    RAW_PRIOR_FIXTURE_N_ESTIMATORS,
    RAW_PRIOR_FIXTURE_SEED,
    SELF_EMPLOYMENT_COMPONENTS,
    ChainStep,
    acs_raw_prior_fixture,
    origination_year_support,
    prior_recording_qrf,
    snapped_prior_qrf,
    zeroed_prior_qrf,
)

# ---------------------------------------------------------------------------
# e89769879c's output digests, by platform
# ---------------------------------------------------------------------------

#: ``table_bytes_digest`` of each output table, from e89769879c's
#: ``transfer_acs_inputs`` on this fixture, in memory and banked alike.
_PRE_1667_DIGESTS: dict[tuple[str, str], dict[str, str]] = {
    ("Darwin", "arm64"): {
        "person": "b395e4936e76c677c6100cc0391d4ea1008717e614a4fb7cbffcf6ca93ea265c",
        "household": "548c8ba27511a28ea1b2a9717a296730aa4dc677f2010ad423e928d2d5172499",
        "tax_unit": "3434b12b368aaa7221793b95e8d66a056a9e1565b4a152570a08013ea3dc50f3",
    },
}

#: Versions of the libraries that compute the draws, on the recording.
_RECORDED_DEPENDENCIES = {
    "numpy": "2.4.6",
    "pandas": "3.0.3",
    "scipy": "1.17.1",
    "scikit-learn": "1.8.0",
    "quantile-forest": "1.4.2",
}


def _pre_1667_digests() -> dict[str, str]:
    """This platform's recorded digests, or a skip where they do not apply."""

    here = (platform.system(), platform.machine())
    recorded = _PRE_1667_DIGESTS.get(here)
    if recorded is None:
        pytest.skip(
            f"e89769879c's digests were recorded on {sorted(_PRE_1667_DIGESTS)}, "
            f"not {here}; the in-process comparisons cover this platform."
        )
    installed = {name: version(name) for name in _RECORDED_DEPENDENCIES}
    if installed != _RECORDED_DEPENDENCIES:
        pytest.skip(
            f"e89769879c's digests were recorded with {_RECORDED_DEPENDENCIES}, "
            f"not {installed}; the in-process comparisons cover these versions."
        )
    return recorded


# ---------------------------------------------------------------------------
# Runs
# ---------------------------------------------------------------------------


def _transfer(
    *, target_bank: AcsTransferTargetBankStore | None = None
) -> AcsTransferResult:
    recipient, donor = acs_raw_prior_fixture()
    return transfer_acs_inputs(
        recipient,
        donor,
        target_families=RAW_PRIOR_FIXTURE_FAMILIES,
        seed=RAW_PRIOR_FIXTURE_SEED,
        n_estimators=RAW_PRIOR_FIXTURE_N_ESTIMATORS,
        target_bank=target_bank,
    )


def _bank_store(root: Path) -> AcsTransferTargetBankStore:
    return AcsTransferTargetBankStore(
        root,
        identity={
            "artifact_kind": "fixture_acs_transfer_target_bank_identity",
            "fixture_version": 1,
            "name": "acs_transfer_raw_priors",
        },
    )


def _table_digests(result: AcsTransferResult) -> dict[str, str]:
    return {
        entity: table_bytes_digest(result.frame.table(entity))
        for entity in result.frame.entities
    }


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


@pytest.fixture(scope="module")
def production() -> AcsTransferResult:
    return _transfer()


@pytest.fixture(scope="module")
def fit_first() -> AcsTransferResult:
    """The transfer with every target fitted before any is predicted."""

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(acs_transfer_module, "QRF", FitThenPredictQRF)
        return _transfer()


@pytest.fixture(scope="module")
def banked(
    tmp_path_factory: pytest.TempPathFactory,
) -> tuple[AcsTransferResult, AcsTransferResult]:
    """A cold banked run, then a run resumed from the bank it wrote."""

    root = tmp_path_factory.mktemp("acs_raw_prior_bank") / "bank"
    cold = _transfer(target_bank=_bank_store(root))
    resumed = _transfer(target_bank=_bank_store(root))
    return cold, resumed


# ---------------------------------------------------------------------------
# Identity with main before #1667
# ---------------------------------------------------------------------------


def test_raw_prior_transfer_reproduces_pre_1667_bytes(
    production: AcsTransferResult,
) -> None:
    assert _table_digests(production) == _pre_1667_digests()


def test_banked_raw_prior_transfer_reproduces_pre_1667_bytes(
    banked: tuple[AcsTransferResult, AcsTransferResult],
) -> None:
    expected = _pre_1667_digests()
    cold, resumed = banked

    assert _table_digests(cold) == expected
    assert _table_digests(resumed) == expected


def test_raw_prior_transfer_matches_fitting_every_target_first(
    production: AcsTransferResult,
    fit_first: AcsTransferResult,
) -> None:
    # FitThenPredictQRF ignores the priors the caller passes: its one predict
    # call chains its own raw draws, so it is the reference for what the
    # chain must feed forward.
    _assert_same_transfer(production, fit_first)


def test_banked_raw_prior_transfer_matches_fitting_every_target_first(
    banked: tuple[AcsTransferResult, AcsTransferResult],
    fit_first: AcsTransferResult,
) -> None:
    cold, resumed = banked

    _assert_same_transfer(cold, fit_first)
    _assert_same_transfer(resumed, fit_first)


# ---------------------------------------------------------------------------
# What the chain feeds forward, and what it exports
# ---------------------------------------------------------------------------


def test_transfer_snaps_the_year_and_zeroes_the_components_on_export(
    production: AcsTransferResult,
) -> None:
    # The fixture reaches both cases: the year is discrete by name, and both
    # components are ones the transfer zeroes.
    assert ORIGINATION_YEAR in acs_transfer_module._DISCRETE_NUMERIC_TARGETS
    assert set(SELF_EMPLOYMENT_COMPONENTS) <= set(ACS_SELF_EMPLOYMENT_COMPONENT_INPUTS)
    _recipient, donor = acs_raw_prior_fixture()
    support = origination_year_support(donor)

    year = production.frame.table("tax_unit")[ORIGINATION_YEAR]
    assert year.dtype == np.dtype(np.int64)
    assert np.isin(year.to_numpy(dtype=np.float64), support).all()
    person = production.frame.table("person")
    reconciliation = {
        record.column: record.reconciliation
        for record in production.imputed_inputs
        if record.entity == "person" and record.reconciliation is not None
    }
    for component in SELF_EMPLOYMENT_COMPONENTS:
        assert (person[component].to_numpy() == 0.0).all()
        assert reconciliation[component]["cells_kept_at_zero"] == len(person)
        assert reconciliation[component]["nonzero_draws_discarded"] > 0
    assert all(len(record.patterns) > 1 for record in production.imputed_inputs)


#: Each fixture family's targets, in chain order.
_FAMILY_TARGETS: dict[str, tuple[str, ...]] = {
    family: targets
    for families in RAW_PRIOR_FIXTURE_FAMILIES.values()
    for family, targets in families.items()
}


def _chains(steps: list[ChainStep]) -> dict[str, list[list[ChainStep]]]:
    """Each family's availability patterns, each pattern's steps in call order."""

    family_of = {
        target: family
        for family, targets in _FAMILY_TARGETS.items()
        for target in targets
    }
    patterns: dict[tuple[str, tuple], list[ChainStep]] = {}
    for step in steps:
        key = (family_of[step.target], tuple(step.index))
        patterns.setdefault(key, []).append(step)
    chains: dict[str, list[list[ChainStep]]] = {}
    for (family, _rows), chain in patterns.items():
        chains.setdefault(family, []).append(chain)
    return chains


def _assert_chain_feeds_raw_draws(
    chain: list[ChainStep], targets: tuple[str, ...]
) -> None:
    """One pattern drew ``targets`` in order, each on the earlier raw draws."""

    assert [step.target for step in chain] == list(targets)
    for position, step in enumerate(chain):
        assert list(step.priors.columns) == list(targets[:position])
        assert step.priors.index.equals(step.index)
        for earlier, prior in enumerate(targets[:position]):
            fed = step.priors[prior]
            assert fed.dtype == np.dtype(np.float64)
            assert fed.to_numpy().tobytes() == chain[earlier].raw_draw.tobytes(), (
                f"{step.target!r} was drawn on a {prior!r} prior that is not "
                "the raw draw"
            )


@pytest.mark.parametrize("bank", [False, True], ids=["in_memory", "banked"])
def test_later_targets_condition_on_raw_draws(
    bank: bool,
    production: AcsTransferResult,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    steps: list[ChainStep] = []
    monkeypatch.setattr(acs_transfer_module, "QRF", prior_recording_qrf(steps))
    target_bank = _bank_store(tmp_path / "bank") if bank else None
    recorded = _transfer(target_bank=target_bank)

    patterns = {
        record.column: len(record.patterns) for record in production.imputed_inputs
    }
    chains = _chains(steps)
    assert sorted(chains) == sorted(_FAMILY_TARGETS)
    for family, targets in _FAMILY_TARGETS.items():
        assert len(chains[family]) == patterns[targets[0]] > 1
        for chain in chains[family]:
            _assert_chain_feeds_raw_draws(chain, targets)

    # The fed-forward values are not the exported ones: some year priors are
    # off the donor support the export snaps to, and the component priors are
    # nonzero where the export is all zero.
    _recipient, donor = acs_raw_prior_fixture()
    support = origination_year_support(donor)
    year_priors = np.concatenate(
        [
            step.priors[ORIGINATION_YEAR].to_numpy()
            for step in steps
            if step.target == HSA
        ]
    )
    assert (~np.isin(year_priors, support)).any()
    for component in SELF_EMPLOYMENT_COMPONENTS:
        component_priors = np.concatenate(
            [
                step.priors[component].to_numpy()
                for step in steps
                if component in step.priors.columns
            ]
        )
        assert (component_priors != 0.0).any(), component

    # The spy records without moving a draw.
    assert _table_digests(recorded) == _table_digests(production)


def _rows_differing(left: pd.Series, right: pd.Series) -> int:
    return int((left.to_numpy() != right.to_numpy()).sum())


def test_fixture_discriminates_snapped_and_zeroed_priors(
    production: AcsTransferResult,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _recipient, donor = acs_raw_prior_fixture()
    tax_unit = production.frame.table("tax_unit")
    person = production.frame.table("person")

    monkeypatch.setattr(
        acs_transfer_module,
        "QRF",
        snapped_prior_qrf(ORIGINATION_YEAR, origination_year_support(donor)),
    )
    snapped = _transfer()
    monkeypatch.setattr(
        acs_transfer_module, "QRF", zeroed_prior_qrf(SELF_EMPLOYMENT_COMPONENTS)
    )
    zeroed = _transfer()

    # Fed the snapped year, the chain draws a different HSA deduction; the
    # year, the earlier target and the other family do not move.
    snapped_tax_unit = snapped.frame.table("tax_unit")
    changed = _rows_differing(snapped_tax_unit[HSA], tax_unit[HSA])
    assert changed > 0, "a snapped year prior moved no HSA draw"
    for column in (MORTGAGE_BALANCE, ORIGINATION_YEAR):
        pd.testing.assert_series_equal(
            snapped_tax_unit[column], tax_unit[column], check_exact=True
        )
    pd.testing.assert_frame_equal(
        snapped.frame.table("person"), person, check_exact=True
    )

    # Fed the zeroed components, the chain draws different farm rents and
    # estate flags; the exported components and the other family do not move.
    zeroed_person = zeroed.frame.table("person")
    for column in (FARM_RENT, ESTATE_FLAG):
        changed = _rows_differing(zeroed_person[column], person[column])
        assert changed > 0, f"zeroed component priors moved no {column} draw"
    for column in SELF_EMPLOYMENT_COMPONENTS:
        pd.testing.assert_series_equal(
            zeroed_person[column], person[column], check_exact=True
        )
    pd.testing.assert_frame_equal(
        zeroed.frame.table("tax_unit"), tax_unit, check_exact=True
    )
