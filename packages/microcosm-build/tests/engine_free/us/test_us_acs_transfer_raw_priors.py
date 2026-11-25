"""The ACS transfer conditions later targets on raw draws, not exported values.

The transfer fits one target at a time: each later target in a family is drawn
on the recipients' raw float64 draws of the earlier ones
(``RegimeGatedQRF.fit_draw_next``). Some exported values differ from those raw
draws, and none may reach a later target. The fixture reaches two of them:

- ``first_home_mortgage_origination_year`` is ``discrete_numeric``: the
  transfer snaps its draws to the donor support when it decodes them, which
  happens inside the outer family loop, after the year's own family is fitted;
- the self-employment components are set to zero after the transfer where
  the recipient's own total is kept (``keep_measured_self_employment_totals``).

The whole production batches these tests run reach two more, on
supplemental donor columns: the passive partnership/S-corp subset, set to zero
where the row's line 32 is zero or its tax unit's line 32 is zero
(``reconcile_acs_passive_partnership_s_corp``), and, on the ACS local lane,
``partnership_income``, kept at zero where the row's ACS ``SEMP`` already
counts it (``semp_covers_partnership_income``).
They reach one more on the recipient side: where a row already observes an
earlier target, the export keeps the observed value (the transfer fills only
null cells), while the later targets are drawn on the row's raw draw. ASEC
clone-0 rows hold the source-written
``self_employed_pension_contributions_desired`` before post-PUF batch 3 and
lack its later targets, so that batch runs with the pension observed on some
rows. In no other post-PUF batch does a target that a producer role writes
precede a later target that the same role does not produce. That inventory
concerns what ``transfer_acs_inputs`` itself exports; stacked calibration,
such as the child-support step, runs after that call.

No other decode or export rule changes a value ahead of a later target in the
plans these tests read: boolean raw draws are exactly 0 or 1, which the
boolean decode keeps; the immigration pair, ``weeks_unemployed``, the
second-home year and the adult-care expense never precede a later target;
and no integer-typed target other than the year does.

The fixture (:mod:`test_support.microcosm_build.qrf_raw_priors`) puts the
year ahead of ``health_savings_account_ald``, and the components ahead of
``farm_rent_income`` and ``estate_income_would_be_qualified``, in the order
production's declared ACS plan draws them. These tests pin that:

- the outputs are byte-identical to what e89769879c (main before #1667)
  produced on this fixture, in memory and banked. At e89769879c the in-memory
  path fitted every target and then predicted, while the banked path already
  drew target by target;
- they equal ``FitThenPredictQRF``'s outputs, the real model run the old
  in-memory way, on every platform;
- every prior the chain feeds forward is byte-equal to the earlier raw draw,
  in chain order. That holds too when a banked transfer is interrupted right
  after the checkpoint of any fixture target that has a later target in its
  family (both components, ``farm_rent_income``,
  ``first_home_mortgage_balance`` and the year) and resumed: the retry loads
  that target and the ones before it from their checkpoints and draws the
  later targets fresh. It holds as well when a bank keeps one target's
  checkpoint between targets of its family that the retry rebuilds (the year
  between the balance and the HSA deduction, and the partnership earnings
  between the other person targets);
- the year is snapped and the components zeroed only on export. Each
  exported year is the donor support point nearest its raw draw, and the
  decoder sends an exact midpoint between two support points to the lower one;
- production's plans put the fixture's targets in the fixture's order. The
  plans are the batches ``transfer_acs_inputs`` itself plans for the declared
  ACS plan, each stacked gap-fill direction and each stacked post-PUF group.
  Where the stacked callers pass regime-evidence targets, the plan is read
  again at the transfer's first fit, after the widening, and must not have
  changed. Each batch that holds two or more fixture targets then runs whole
  under its production name, in memory and banked, with supplemental donor
  columns for its other targets: every availability pattern draws the batch
  in the plan's order, each target on the earlier raw draws, and the
  patterns together cover every recipient row. Where the batch draws the
  passive subset, some rows draw a zero line 32 under a nonzero passive
  draw, and some draw a nonzero line 32 that the other person in their tax
  unit offsets; the export zeroes the passive subset on both. In post-PUF
  batch 3 the rows that observe the pension are still drawn, some draw a
  pension other than the observed one, and the export keeps the observed
  value there. A retry of that batch from its bank, with the pension loaded
  from its checkpoint and every later target rebuilt, draws them on the
  checkpoint's raw pension draw and reproduces the cold run's outputs;
- the ACS local lane's run of the declared batch that holds
  ``partnership_income`` draws every later target on the raw partnership
  draws, in memory and banked, while the export zeroes some of them;
- the fixture would catch the change on the running platform: a chain fed
  the snapped year, the zeroed components or the zeroed partnership earnings
  alone draws different later targets.

QRF draws are bitwise reproducible only within one platform (see
``docs/graph-qrf-cross-platform.md``), so e89769879c's digests are kept per
operating system and machine. The two ``*_reproduces_pre_1667_bytes`` tests
run only where both the platform and the installed versions of the libraries
that compute the draws (``_RECORDED_DEPENDENCIES``) match the recording, and
skip elsewhere. Only macOS arm64 is recorded. Every other test runs on every
platform.

A change that is meant to move these outputs records new digests from its own
code, says why the outputs move, and renames what here says the digests come
from e89769879c (this text, ``_PRE_1667_DIGESTS``, ``_pre_1667_digests`` and
the ``*_reproduces_pre_1667_bytes`` tests). Any other change must leave them
passing.
"""

from __future__ import annotations

import inspect
import platform
import shutil
from collections.abc import Mapping, Sequence
from importlib.metadata import version
from pathlib import Path
from typing import NoReturn

import numpy as np
import pandas as pd
import pytest

import microcosm.build.us_runtime.acs_transfer as acs_transfer_module
from microcosm.build.us_runtime.acs_transfer import (
    ACS_SELF_EMPLOYMENT_COMPONENT_INPUTS,
    DEFAULT_ACS_TRANSFER_MAX_TARGETS_PER_FIT,
    AcsTransferResult,
    AcsTransferTargetCheckpoint,
    declared_acs_transfer_target_families,
    transfer_acs_inputs,
)
from microcosm.build.us_runtime.acs_transfer_bank import AcsTransferTargetBankStore
from microcosm.build.us_runtime.stacked_spine import (
    _stacked_post_transfer_calibration_specs,
    stacked_gap_fill_plan,
)
from microcosm.build.us_runtime.us_late_producer_registry import (
    CANONICAL_US_LATE_TRANSFER_GROUPS,
)
from microcosm.frame import US_SCHEMA, Frame, WeightKind, Weights
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

#: The component the estate flag follows in the fixture.
_PARTNERSHIP = "partnership_self_employment_net_earnings"


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
    *,
    target_bank: AcsTransferTargetBankStore | None = None,
    target_families: Mapping[
        str, Mapping[str, tuple[str, ...]]
    ] = RAW_PRIOR_FIXTURE_FAMILIES,
    frames: tuple[Frame, Frame] | None = None,
    **options: bool,
) -> AcsTransferResult:
    recipient, donor = acs_raw_prior_fixture() if frames is None else frames
    return transfer_acs_inputs(
        recipient,
        donor,
        target_families=target_families,
        seed=RAW_PRIOR_FIXTURE_SEED,
        n_estimators=RAW_PRIOR_FIXTURE_N_ESTIMATORS,
        target_bank=target_bank,
        **options,
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


def _recorded_transfer(
    target_bank: AcsTransferTargetBankStore | None = None,
) -> tuple[list[ChainStep], AcsTransferResult]:
    steps: list[ChainStep] = []
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(acs_transfer_module, "QRF", prior_recording_qrf(steps))
        result = _transfer(target_bank=target_bank)
    return steps, result


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


@pytest.fixture(scope="module")
def recorded_in_memory() -> tuple[list[ChainStep], AcsTransferResult]:
    """Every ``fit_draw_next`` call of an in-memory run, and its result."""

    return _recorded_transfer()


@pytest.fixture(scope="module")
def recorded_banked(
    tmp_path_factory: pytest.TempPathFactory,
) -> tuple[list[ChainStep], AcsTransferResult, AcsTransferTargetBankStore]:
    """Every ``fit_draw_next`` call of a cold banked run, its result and bank."""

    bank = _bank_store(tmp_path_factory.mktemp("acs_raw_prior_recorded") / "bank")
    steps, result = _recorded_transfer(bank)
    return steps, result, bank


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


def _nearest_support_point(value: float, support: np.ndarray) -> int:
    """The support point nearest ``value``, the lower one on a tie.

    This is written apart from the transfer's decoder but copies its tie rule,
    which the export test pins on the decoder: it takes the upper of the two
    neighbouring points only when it is strictly nearer.
    """

    return int(min(support.tolist(), key=lambda point: (abs(point - value), point)))


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

    # The decoder sends a draw halfway between two support points to the
    # lower one, as the independent reference below does. The fixture draws
    # no exact midpoint, so this pins the tie rule on the decoder itself.
    midpoints = (support[:-1] + support[1:]) / 2
    lower = support[:-1].astype(np.int64).tolist()
    decoded = acs_transfer_module._prediction_values(
        midpoints,
        encoding=acs_transfer_module._TargetEncoding(
            model_values=pd.Series(support),
            kind="discrete_numeric",
            support=support,
            model_target=ORIGINATION_YEAR,
        ),
        entity="tax_unit",
        target=ORIGINATION_YEAR,
    )
    assert decoded.dtype == np.dtype(np.int64)
    assert decoded.tolist() == lower, (
        "the decoder no longer sends a year halfway between two donor support "
        "points to the lower one"
    )
    assert acs_transfer_module._snap_to_support(midpoints, support).tolist() == lower
    assert [_nearest_support_point(value, support) for value in midpoints] == lower


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
    chain: list[ChainStep], targets: tuple[str, ...], *, run: str = ""
) -> None:
    """One pattern drew ``targets`` in order, each on the earlier raw draws.

    ``run`` names the run in failure messages.
    """

    assert [step.target for step in chain] == list(targets), run
    for position, step in enumerate(chain):
        assert list(step.priors.columns) == list(targets[:position]), run
        assert step.priors.index.equals(step.index), run
        for earlier, prior in enumerate(targets[:position]):
            fed = step.priors[prior]
            assert fed.dtype == np.dtype(np.float64), run
            assert fed.to_numpy().tobytes() == chain[earlier].raw_draw.tobytes(), (
                f"{run}{step.target!r} was drawn on a {prior!r} prior that is "
                "not the raw draw"
            )


def _step_key(step: ChainStep) -> tuple[str, tuple]:
    """A step's target and pattern rows, which name it within one run."""

    return step.target, tuple(step.index)


def _assert_exported_years_snap_raw_draws(
    result: AcsTransferResult, steps: list[ChainStep], support: np.ndarray
) -> None:
    """Each exported year is the support point nearest its own raw draw."""

    exported = result.frame.table("tax_unit")[ORIGINATION_YEAR]
    year_steps = [step for step in steps if step.target == ORIGINATION_YEAR]
    drawn_rows = np.concatenate([step.index.to_numpy() for step in year_steps])
    assert sorted(drawn_rows.tolist()) == sorted(exported.index.tolist())
    for step in year_steps:
        expected = [_nearest_support_point(value, support) for value in step.raw_draw]
        assert exported.loc[step.index].tolist() == expected, (
            "an exported year is not the donor support point nearest its raw draw"
        )


@pytest.mark.parametrize("path", ["in_memory", "banked"])
def test_later_targets_condition_on_raw_draws(
    path: str,
    production: AcsTransferResult,
    request: pytest.FixtureRequest,
) -> None:
    steps, recorded = request.getfixturevalue(f"recorded_{path}")[:2]

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

    # The export snaps each recorded raw year draw on its own.
    _assert_exported_years_snap_raw_draws(recorded, steps, support)

    # The spy records without moving a draw.
    assert _table_digests(recorded) == _table_digests(production)


# ---------------------------------------------------------------------------
# A banked transfer resumed from a checkpoint
# ---------------------------------------------------------------------------


def _bank_order(bank: AcsTransferTargetBankStore) -> tuple[str, ...]:
    """The model targets ``bank`` was asked for, by target index."""

    targets = bank.receipt()["targets"]
    return tuple(
        targets[str(index)]["descriptor"]["model_target"]
        for index in range(len(targets))
    )


def _interrupt_after_writing(
    bank: AcsTransferTargetBankStore,
    model_target: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Stop the transfer once ``bank`` has durably written ``model_target``."""

    durable_write = bank.write_target

    def write_then_interrupt(checkpoint: AcsTransferTargetCheckpoint) -> None:
        durable_write(checkpoint)
        if checkpoint.model_target == model_target:
            raise RuntimeError(f"fixture interrupt after durable {model_target}")

    monkeypatch.setattr(bank, "write_target", write_then_interrupt)


def _assert_resume_reuses_raw_draws(
    resumed: AcsTransferResult,
    resumed_bank: AcsTransferTargetBankStore,
    steps: list[ChainStep],
    recorded_banked: tuple[
        list[ChainStep], AcsTransferResult, AcsTransferTargetBankStore
    ],
    *,
    loaded: str,
    rebuilt: Sequence[str],
) -> None:
    """A banked retry rebuilt exactly ``rebuilt`` and loaded every other target.

    ``steps`` are the retry's ``fit_draw_next`` calls. Its priors, raw draws,
    checkpoints and outputs must equal the uninterrupted run's, and the
    loaded target ``loaded`` must reach a rebuilt target as a prior.
    """

    reference_steps, uninterrupted, reference_bank = recorded_banked
    order = _bank_order(reference_bank)
    receipt = resumed_bank.receipt()["targets"]
    reference_receipt = reference_bank.receipt()["targets"]
    assert [
        (receipt[str(index)]["source"], receipt[str(index)]["load_status"])
        for index in range(len(order))
    ] == [
        ("rebuilt", "missing") if target in rebuilt else ("checkpoint", "resumed")
        for target in order
    ]
    # Each rebuilt target was fitted fresh for every pattern; no loaded one was.
    reference = {_step_key(step): step for step in reference_steps}
    assert sorted(_step_key(step) for step in steps) == sorted(
        key for key in reference if key[0] in rebuilt
    )

    # Every prior fed to a fresh fit is the uninterrupted run's raw draw.
    for step in steps:
        rows = tuple(step.index)
        expected = reference[_step_key(step)]
        assert list(step.priors.columns) == list(expected.priors.columns)
        for prior in step.priors.columns:
            fed = step.priors[prior]
            assert fed.dtype == np.dtype(np.float64)
            assert (
                fed.to_numpy().tobytes() == reference[(prior, rows)].raw_draw.tobytes()
            ), (
                f"on resume, {step.target!r} was drawn on a {prior!r} prior that "
                "is not the uninterrupted run's raw draw"
            )
        assert step.raw_draw.tobytes() == expected.raw_draw.tobytes(), step.target

    # Every target, loaded or rebuilt, holds the uninterrupted run's raw draw
    # and chain states.
    for index in range(len(order)):
        for digest in ("raw_draw_sha256", "content_metadata_sha256"):
            assert (
                receipt[str(index)][digest] == reference_receipt[str(index)][digest]
            ), (
                order[index],
                digest,
            )

    # The loaded prior reached a fresh fit. Where the export changes it (the
    # year and the components), it did so with values the export changes.
    fed_loaded = [
        step.priors[loaded].to_numpy() for step in steps if loaded in step.priors
    ]
    assert fed_loaded, f"no target drawn on resume was fed {loaded!r}"
    if loaded == ORIGINATION_YEAR:
        _recipient, donor = acs_raw_prior_fixture()
        support = origination_year_support(donor)
        assert (~np.isin(np.concatenate(fed_loaded), support)).any()
    elif loaded in SELF_EMPLOYMENT_COMPONENTS:
        assert (np.concatenate(fed_loaded) != 0.0).any()

    _assert_same_transfer(resumed, uninterrupted)


#: Every checkpoint after which a resumed transfer feeds a loaded prior to a
#: fresh fit: each fixture target that has a later target in its family.
_RESUME_POINTS: tuple[str, ...] = tuple(
    target for targets in _FAMILY_TARGETS.values() for target in targets[:-1]
)


@pytest.mark.parametrize("loaded", _RESUME_POINTS)
def test_resumed_bank_draws_later_targets_on_loaded_raw_draws(
    loaded: str,
    recorded_banked: tuple[
        list[ChainStep], AcsTransferResult, AcsTransferTargetBankStore
    ],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # A retry interrupted after ``loaded``'s checkpoint loads ``loaded`` from
    # that checkpoint into the priors and fits the later targets fresh. A cold
    # run draws ``loaded`` itself, and a run resumed from a complete bank fits
    # nothing, so only such a retry feeds a loaded prior to a new fit.
    order = _bank_order(recorded_banked[2])
    resumed_from = order.index(loaded) + 1

    root = tmp_path / "bank"
    interrupted_bank = _bank_store(root)
    _interrupt_after_writing(interrupted_bank, loaded, monkeypatch)
    with pytest.raises(RuntimeError, match=f"fixture interrupt after durable {loaded}"):
        _transfer(target_bank=interrupted_bank)
    for index, target in enumerate(order):
        written = interrupted_bank.target_path(index, target).is_file()
        assert written == (index < resumed_from), target

    steps: list[ChainStep] = []
    monkeypatch.setattr(acs_transfer_module, "QRF", prior_recording_qrf(steps))
    resumed_bank = _bank_store(root)
    resumed = _transfer(target_bank=resumed_bank)

    # The resume loaded ``loaded`` and every target before it from their
    # checkpoints, and fitted every later target fresh.
    _assert_resume_reuses_raw_draws(
        resumed,
        resumed_bank,
        steps,
        recorded_banked,
        loaded=loaded,
        rebuilt=order[resumed_from:],
    )


#: Fixture targets with targets of their own family on both sides: a bank
#: that keeps one of these checkpoints and drops the rest of its family's
#: makes a retry rebuild targets on both sides of a loaded one.
_MIXED_BANK_KEPT: tuple[str, ...] = (ORIGINATION_YEAR, _PARTNERSHIP)


@pytest.mark.parametrize("kept", _MIXED_BANK_KEPT)
def test_mixed_bank_draws_rebuilt_targets_on_a_loaded_raw_draw(
    kept: str,
    recorded_banked: tuple[
        list[ChainStep], AcsTransferResult, AcsTransferTargetBankStore
    ],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # The bank loads each target on its own, so it can hold a later target's
    # checkpoint while an earlier one is missing (as the mixed-hole test in
    # test_us_acs_transfer.py shows). A retry then rebuilds the earlier
    # targets of ``kept``'s family, loads ``kept`` from its checkpoint into
    # the priors, and draws the later targets fresh on it.
    reference_bank = recorded_banked[2]
    order = _bank_order(reference_bank)
    family = next(targets for targets in _FAMILY_TARGETS.values() if kept in targets)
    position = family.index(kept)
    assert 0 < position < len(family) - 1, family
    rebuilt = tuple(target for target in family if target != kept)

    root = tmp_path / "bank"
    shutil.copytree(reference_bank.root, root)
    copied = _bank_store(root)
    for index, target in enumerate(order):
        if target in rebuilt:
            copied.target_path(index, target).unlink()

    steps: list[ChainStep] = []
    monkeypatch.setattr(acs_transfer_module, "QRF", prior_recording_qrf(steps))
    resumed_bank = _bank_store(root)
    resumed = _transfer(target_bank=resumed_bank)

    # Within ``kept``'s family the receipt reads rebuilt, checkpoint, rebuilt.
    receipt = resumed_bank.receipt()["targets"]
    assert [receipt[str(order.index(target))]["source"] for target in family] == (
        ["rebuilt"] * position
        + ["checkpoint"]
        + ["rebuilt"] * (len(family) - position - 1)
    )
    _assert_resume_reuses_raw_draws(
        resumed,
        resumed_bank,
        steps,
        recorded_banked,
        loaded=kept,
        rebuilt=rebuilt,
    )


# ---------------------------------------------------------------------------
# Production's order
# ---------------------------------------------------------------------------


#: One planned batch: its entity, its (production) family name and its targets.
_Batch = tuple[str, str, tuple[str, ...]]


class _PlanRecordedError(Exception):
    """Stops a transfer once it has planned its batches."""


def _plan_recipient() -> Frame:
    """One person on production's US entities, lacking every transfer target."""

    person = pd.DataFrame(
        {"person_id": [1]}
        | {
            US_SCHEMA.membership_column(group): [1]
            for group in US_SCHEMA.group_entities
        }
    )
    return Frame(
        {US_SCHEMA.person_entity: person}
        | {
            group: pd.DataFrame({US_SCHEMA.id_column(group): [1]})
            for group in US_SCHEMA.group_entities
        },
        US_SCHEMA,
        {"household": Weights(np.ones(1), WeightKind.DESIGN)},
        pd.Series("acs_2024_1yr", index=person.index, dtype=object),
    )


def _active_plan(
    target_families: Mapping[str, Mapping[str, Sequence[str]]],
    regime_evidence_targets: tuple[tuple[str, str], ...] = (),
) -> list[_Batch]:
    """The batches ``transfer_acs_inputs`` plans for ``target_families``.

    The transfer itself plans: it normalizes the families in the recipient's
    entity order, splits them into bounded batches and keeps the targets the
    recipient lacks. The recipient lacks every target, as an ACS recipient
    lacks the PUF targets. Without ``regime_evidence_targets`` the transfer
    stops at that activation filter, before any fit. With them, as the stacked
    callers pass them, it goes on through the widening that restores a
    batch's planned targets and stops at its first fit, where the plan its
    loop fits must equal the filtered plan.
    """

    activation_filter = acs_transfer_module._missing_target_families
    filtered: list[tuple[_Batch, ...]] = []
    widened: list[tuple[_Batch, ...]] = []

    def snapshot(plan: Sequence[Sequence[object]]) -> tuple[_Batch, ...]:
        return tuple(
            (str(entity), str(family), tuple(targets))
            for entity, family, targets in plan
        )

    def record_plan(families: Sequence[_Batch], *, recipient: Frame) -> list[_Batch]:
        planned = activation_filter(families, recipient=recipient)
        # A snapshot: production goes on with the filter's own list, so an
        # edit the widening makes to it in place cannot move this baseline.
        filtered.append(snapshot(planned))
        if not regime_evidence_targets:
            raise _PlanRecordedError
        return planned

    def record_first_fit(
        *_args: object, targets: tuple[str, ...], **_kwargs: object
    ) -> NoReturn:
        # The transfer's loop fits the batches of its local ``active`` plan.
        # This reads production's local variable by name, so renaming it
        # raises a KeyError here: the check fails loudly rather than passing.
        active = snapshot(inspect.currentframe().f_back.f_locals["active"])
        assert active[0][2] == targets
        widened.append(active)
        raise _PlanRecordedError

    recipient = _plan_recipient()
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(acs_transfer_module, "_missing_target_families", record_plan)
        patch.setattr(acs_transfer_module, "_fit_family_patterns", record_first_fit)
        with pytest.raises(_PlanRecordedError):
            transfer_acs_inputs(
                recipient,
                recipient,
                target_families=target_families,
                max_targets_per_fit=DEFAULT_ACS_TRANSFER_MAX_TARGETS_PER_FIT,
                regime_evidence_targets=regime_evidence_targets,
            )
    (active,) = filtered
    if regime_evidence_targets:
        assert widened == [active], (
            "the regime-evidence widening changed the planned batches: the "
            f"filter planned {[batch[1] for batch in active]}, the first fit's "
            f"loop holds {[batch[1] for batch in widened[0]]}, and these differ "
            f"from the filter's: {[batch for batch in widened[0] if batch not in active]}"
        )
    return list(active)


def _regime_evidence(
    target_families: Mapping[str, Mapping[str, Sequence[str]]], *, stage: str
) -> tuple[tuple[str, str], ...]:
    """The regime-evidence targets the stacked callers pass with these families."""

    return tuple(
        (spec.entity, spec.target)
        for spec in _stacked_post_transfer_calibration_specs(
            target_families, stage=stage
        )
    )


def _production_batches() -> dict[str, list[_Batch]]:
    """Each production plan's batches, as the transfer plans them, in order.

    The staging builder passes the declared ACS plan. Stacked gap fill passes
    each direction's families (``stacked_gap_fill_plan``), and canonical
    post-PUF execution passes each late-transfer group's families, each with
    the regime-evidence targets of its calibration specs.
    """

    return {
        "declared ACS plan": _active_plan(declared_acs_transfer_target_families()),
        **{
            f"stacked gap-fill direction {direction.name!r}": _active_plan(
                direction.target_families,
                _regime_evidence(direction.target_families, stage="early_gap_fill"),
            )
            for direction in stacked_gap_fill_plan()
        },
        **{
            f"stacked post-PUF group {group.name!r}": _active_plan(
                group.target_families,
                _regime_evidence(group.target_families, stage="late_transfer"),
            )
            for group in CANONICAL_US_LATE_TRANSFER_GROUPS
        },
    }


#: Seed of the supplemental donor columns.
_SUPPLEMENT_SEED = 1723
#: The line-32 components, and the passive subset of line 32 that the export
#: zeroes where they sum to zero for the person or for the person's tax unit.
_LINE_32 = ("partnership_income", "s_corp_income")
_PASSIVE = "passive_partnership_s_corp_income"
#: The recipient's tax-unit link, which the export's line-32 rule sums over.
_TAX_UNIT = "person_tax_unit_id"
#: Seed of the supplemental line-32 columns, and each one's values and shares.
#: Each sign regime holds one value, so a person's line 32 can be zero and two
#: people's can cancel exactly, and two signs keep each sign gate binary.
_LINE_32_SEED = 20261125
_LINE_32_SUPPORT: dict[str, tuple[tuple[float, ...], tuple[float, ...]]] = {
    "partnership_income": ((0.0, 100.0), (0.6, 0.4)),
    "s_corp_income": ((-100.0, 100.0), (0.5, 0.5)),
}
#: A target some recipients already observe ahead of later targets in its
#: batch. Canonical post-PUF execution requires ASEC rows to hold the
#: source-written pension before batch 3, while only PUF clones produce the
#: batch's later targets (``stacked_spine``: the post-PUF producer surfaces,
#: ``_post_puf_producer_mask`` and the activation check), so ASEC clone-0
#: rows enter that batch observing the pension and lacking the rest.
_OBSERVED_PENSION = "self_employed_pension_contributions_desired"
#: Seed of the observed pension values, and the share of rows left unobserved.
_OBSERVED_PENSION_SEED = 17233
_UNOBSERVED_PENSION_SHARE = 0.5


def _with_batch_columns(donor: Frame, entity: str, targets: Sequence[str]) -> Frame:
    """``donor`` with a column for each of ``targets`` that it lacks.

    The fixture has donor columns for its own targets only. Each supplemental
    column is a positive amount from one seeded stream, so most of these
    targets have one sign regime and their fits stay cheap. The two line-32
    columns instead come from a second seeded stream: ``partnership_income``
    is 0 or 100 and ``s_corp_income`` is -100 or 100. Each of their sign
    regimes holds one value, which its forest draws exactly, so recipients
    draw a zero line 32 under a nonzero passive subset, either on their own
    or with the other person in their tax unit offsetting them, and both
    branches of the export's passive reconciliation fire. The fixture's
    columns are unchanged.
    """

    table = donor.table(entity)
    missing = [target for target in targets if target not in table.columns]
    amounts = np.round(
        np.random.default_rng(_SUPPLEMENT_SEED).lognormal(
            7.0, 1.0, (len(missing), len(table))
        ),
        2,
    )
    line_32 = np.random.default_rng(_LINE_32_SEED)
    for column, target in zip(amounts, missing, strict=True):
        if target in _LINE_32_SUPPORT:
            values, shares = _LINE_32_SUPPORT[target]
            column[:] = line_32.choice(values, len(table), p=shares)
    return donor.with_columns(entity, **dict(zip(missing, amounts, strict=True)))


def _with_observed_pension(recipient: Frame, entity: str) -> tuple[Frame, pd.Series]:
    """``recipient`` observing the pension on a seeded half of its rows.

    The observed values are positive amounts, and the other rows are null.
    Returns the recipient and its pension column.
    """

    rows = len(recipient.table(entity))
    rng = np.random.default_rng(_OBSERVED_PENSION_SEED)
    pension = np.round(rng.lognormal(7.0, 1.0, rows), 2)
    pension[rng.random(rows) < _UNOBSERVED_PENSION_SHARE] = np.nan
    observing = recipient.with_columns(entity, **{_OBSERVED_PENSION: pension})
    return observing, observing.table(entity)[_OBSERVED_PENSION]


def _patterns(steps: list[ChainStep]) -> dict[tuple, list[ChainStep]]:
    """Each availability pattern's steps, in call order, in one run."""

    patterns: dict[tuple, list[ChainStep]] = {}
    for step in steps:
        patterns.setdefault(tuple(step.index), []).append(step)
    return patterns


def _run_whole_batch(
    batch: _Batch,
    frames: tuple[Frame, Frame],
    *,
    bank: AcsTransferTargetBankStore | None,
    run: str,
    monkeypatch: pytest.MonkeyPatch,
    **options: bool,
) -> tuple[dict[tuple, list[ChainStep]], AcsTransferResult]:
    """Run one production batch whole, under its name, and check its chains.

    Every availability pattern must draw the batch in the plan's order, each
    target on the earlier raw draws, and the patterns together must draw every
    recipient row once. Each target's provenance must count the recipient
    cells the transfer filled: every row of a target the recipient lacks, and
    the null rows of one it observes in part. ``run`` names the run in failure
    messages.
    """

    entity, family, targets = batch
    steps: list[ChainStep] = []
    monkeypatch.setattr(acs_transfer_module, "QRF", prior_recording_qrf(steps))
    result = _transfer(
        target_bank=bank,
        target_families={entity: {family: targets}},
        frames=frames,
        **options,
    )
    patterns = _patterns(steps)
    orders = {tuple(step.target for step in chain) for chain in patterns.values()}
    assert orders == {targets}, (
        f"{run}draws {sorted(orders)}; the plan orders it {list(targets)}"
    )
    for chain in patterns.values():
        _assert_chain_feeds_raw_draws(chain, targets, run=run)
    rows = frames[0].table(entity).index
    drawn_rows = [row for chain in patterns.values() for row in chain[0].index]
    assert sorted(drawn_rows) == sorted(rows), (
        f"{run}the patterns drew {len(drawn_rows)} rows, not each of the "
        f"{len(rows)} eligible rows once"
    )
    recipient = frames[0].table(entity)
    assert {
        (record.column, record.imputed_recipient_rows)
        for record in result.imputed_inputs
    } == {
        (
            target,
            int(recipient[target].isna().sum())
            if target in recipient.columns
            else len(rows),
        )
        for target in targets
    }, run
    return patterns, result


def _raw_draws(
    patterns: dict[tuple, list[ChainStep]], target: str, index: pd.Index
) -> pd.Series:
    """``target``'s raw draw on every row of ``index``, from each pattern."""

    raw = pd.Series(np.nan, index=index)
    for chain in patterns.values():
        (step,) = [candidate for candidate in chain if candidate.target == target]
        raw.loc[step.index] = step.raw_draw
    return raw


def _assert_resume_after_the_pension(
    batch: _Batch,
    frames: tuple[Frame, Frame],
    cold: tuple[
        dict[tuple, list[ChainStep]], AcsTransferResult, AcsTransferTargetBankStore
    ],
    observed: pd.Series,
    *,
    root: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Resume ``batch`` from a copy of its cold bank that keeps the pension only.

    The retry loads the pension from its checkpoint, on a recipient that
    observes it on some rows, and rebuilds every later target. Each must be
    drawn on the checkpoint's raw pension draw, not the observed value, and
    the outputs must equal the cold run's.
    """

    entity, family, targets = batch
    patterns, cold_result, cold_bank = cold
    order = _bank_order(cold_bank)
    rebuilt = order[order.index(_OBSERVED_PENSION) + 1 :]
    assert rebuilt, order
    shutil.copytree(cold_bank.root, root)
    copied = _bank_store(root)
    for index, target in enumerate(order):
        if target in rebuilt:
            copied.target_path(index, target).unlink()

    steps: list[ChainStep] = []
    monkeypatch.setattr(acs_transfer_module, "QRF", prior_recording_qrf(steps))
    resumed_bank = _bank_store(root)
    resumed = _transfer(
        target_bank=resumed_bank,
        target_families={entity: {family: targets}},
        frames=frames,
    )
    _assert_resume_reuses_raw_draws(
        resumed,
        resumed_bank,
        steps,
        (
            [step for chain in patterns.values() for step in chain],
            cold_result,
            cold_bank,
        ),
        loaded=_OBSERVED_PENSION,
        rebuilt=rebuilt,
    )
    # On the rows that observe the pension, the first rebuilt target was fed
    # the checkpoint's raw draw, and on some of them that is not the observed
    # value, so a retry that fed the observed value would fail above.
    fed = pd.concat(
        [step.priors[_OBSERVED_PENSION] for step in steps if step.target == rebuilt[0]]
    )
    held = observed.loc[fed.index].notna()
    assert (fed[held] != observed.loc[fed.index][held]).any(), (
        "on resume, every observed pension equals the raw draw fed forward, so "
        "the check cannot tell which one reached the later targets"
    )


def test_production_draws_the_fixture_targets_in_the_fixture_order(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # The fixture's cases exist only where production chains the year ahead
    # of the HSA deduction and the components ahead of the farm rent and the
    # estate flag. Each batch is one chain; separate batches never condition
    # on each other.
    plans = _production_batches()
    runtime_batches: set[tuple[str, str, tuple[str, ...], tuple[str, ...]]] = set()
    for plan, batches in plans.items():
        for entity, batch, targets in batches:
            for chain in RAW_PRIOR_FIXTURE_FAMILIES.get(entity, {}).values():
                shared = [target for target in chain if target in targets]
                if not shared:
                    continue
                drawn = sorted(shared, key=targets.index)
                assert drawn == shared, (
                    f"the {plan} plans {drawn} in that order in {entity} batch "
                    f"{batch!r}; the fixture chains {list(chain)}"
                )
                # The transfer's model-target order for the whole batch is
                # the batch's order.
                assert acs_transfer_module._model_target_names(targets) == targets
                if len(shared) > 1:
                    runtime_batches.add((entity, batch, targets, tuple(shared)))

    # Stacked gap fill passes regime-evidence targets, so its plans were read
    # at the transfer's first fit, after the widening.
    assert any(
        _regime_evidence(direction.target_families, stage="early_gap_fill")
        for direction in stacked_gap_fill_plan()
    )

    # The transfer draws each such batch whole at runtime, under its
    # production name, in memory and banked. The donor gets a supplemental
    # column for each batch target the fixture lacks; the recipient is the
    # fixture's, which has every required predictor and lacks every target
    # except the pension, observed on some rows where a batch draws it ahead
    # of later targets. Every recipient row lacks a later target, so every
    # row is eligible.
    assert runtime_batches
    observed_runs: list[str] = []
    for number, (entity, batch, targets, shared) in enumerate(sorted(runtime_batches)):
        recipient, donor = acs_raw_prior_fixture()
        observed: pd.Series | None = None
        if _OBSERVED_PENSION in targets[:-1]:
            recipient, observed = _with_observed_pension(recipient, entity)
        frames = (recipient, _with_batch_columns(donor, entity, targets))
        for path, bank in (
            ("in memory", None),
            ("banked", _bank_store(tmp_path / f"order_{number}")),
        ):
            run = f"the transfer, {path}, of {entity} batch {batch!r}: "
            patterns, result = _run_whole_batch(
                (entity, batch, targets),
                frames,
                bank=bank,
                run=run,
                monkeypatch=monkeypatch,
            )
            if bank is not None:
                cold = (patterns, result, bank)
            # The fixture's targets reach the draw boundary in the fixture's
            # order.
            for chain in patterns.values():
                fixture_order = [step.target for step in chain if step.target in shared]
                assert fixture_order == list(shared), run
            if observed is not None:
                # The rows that observe the pension were drawn with the rest,
                # and the chain check above fed the later targets their raw
                # pension draw; the export keeps the observed value there,
                # and on some rows that value is not the raw draw.
                pension = result.frame.table(entity)[_OBSERVED_PENSION]
                raw = _raw_draws(patterns, _OBSERVED_PENSION, pension.index)
                held = observed.notna()
                assert held.any() and raw[held].notna().all(), run
                assert (pension[held] == observed[held]).all(), (
                    f"{run}the export did not keep the observed pension"
                )
                assert (raw[held] != observed[held]).any(), (
                    f"{run}every observed pension equals its raw draw, so the "
                    "chain check cannot tell which one reached the later targets"
                )
                observed_runs.append(run)
            if _PASSIVE not in targets:
                continue
            # The export zeroes the passive subset where the row drew a zero
            # line 32, while the chain fed the later targets the raw passive
            # draw there, so the chain check above reached that export.
            exported = result.frame.table(entity)
            line_32 = sum(
                _raw_draws(patterns, column, exported.index) for column in _LINE_32
            )
            passive = _raw_draws(patterns, _PASSIVE, exported.index)
            cleared = (line_32 == 0.0) & (passive != 0.0)
            assert cleared.any(), (
                f"{run}no row drew a zero line 32 under a nonzero passive "
                "subset, so the passive reconciliation never fired"
            )
            assert (exported.loc[cleared, _PASSIVE] == 0.0).all(), run
            # The export also zeroes it where the row's own line 32 is nonzero
            # but its tax unit's sums to zero, the other person offsetting it,
            # and the chain check reached that branch too.
            unit_line_32 = line_32.groupby(exported[_TAX_UNIT]).transform("sum")
            offset = (line_32 != 0.0) & (unit_line_32 == 0.0) & (passive != 0.0)
            assert offset.any(), (
                f"{run}no row drew a nonzero line 32 that its tax unit offsets, "
                "under a nonzero passive subset, so the tax-unit branch of the "
                "passive reconciliation never fired"
            )
            assert (exported.loc[offset, _PASSIVE] == 0.0).all(), run

        if observed is not None:
            # A retry from the bank loads the pension from its checkpoint,
            # beside the observed values, and rebuilds the later targets.
            _assert_resume_after_the_pension(
                (entity, batch, targets),
                frames,
                cold,
                observed,
                root=tmp_path / f"resumed_{number}",
                monkeypatch=monkeypatch,
            )
            observed_runs.append(f"the resumed transfer of {entity} batch {batch!r}")

    # Post-PUF batch 3 draws the pension ahead of later targets, so the
    # observed case ran, in memory, banked and resumed from the bank.
    assert observed_runs, (
        f"no whole batch draws {_OBSERVED_PENSION!r} ahead of a later target; "
        "revisit the observed case"
    )

    # The declared ACS plan chains each fixture family whole in one batch.
    declared = plans["declared ACS plan"]
    for entity, families in RAW_PRIOR_FIXTURE_FAMILIES.items():
        for chain in families.values():
            holding = [
                batch
                for batch_entity, batch, targets in declared
                if batch_entity == entity and set(chain) <= set(targets)
            ]
            assert len(holding) == 1, (
                f"the declared ACS plan no longer chains {list(chain)} in one "
                f"{entity} batch; revisit the fixture"
            )


#: The leaf the ACS local lane keeps at zero where ACS ``SEMP`` counts it.
_SEMP_PARTNERSHIP = "partnership_income"


def test_acs_local_lane_draws_later_targets_on_raw_partnership_draws(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # The ACS local lane runs the declared plan with
    # semp_covers_partnership_income=True: after the transfer it keeps a
    # partnership_income draw at zero where the row's or its tax unit's ACS
    # SEMP already counts it. In the declared person batch that holds it,
    # partnership_income precedes later targets (s_corp_income, the passive
    # subset, the partnership earnings and the estate flag), which must be
    # drawn on the raw partnership draws.
    ((entity, batch, targets),) = [
        (entity, batch, targets)
        for entity, batch, targets in _production_batches()["declared ACS plan"]
        if entity == "person" and _SEMP_PARTNERSHIP in targets
    ]
    assert targets.index(_SEMP_PARTNERSHIP) < len(targets) - 1, targets
    recipient, donor = acs_raw_prior_fixture()
    person = recipient.table(entity)
    # ACS SEMP maps to self_employment_income_before_lsr, blank where blank.
    # Each household is one tax unit: its head heads it, and the other person
    # is the spouse from age 18 and a dependent below.
    recipient = recipient.with_columns(
        entity,
        SEMP=person["self_employment_income_before_lsr"].to_numpy(dtype=np.float64),
        tax_unit_role_input=np.where(
            person["is_household_head"],
            "HEAD",
            np.where(person["age"] >= 18, "SPOUSE", "DEPENDENT"),
        ).astype(object),
    )
    frames = (recipient, _with_batch_columns(donor, entity, targets))
    for path, bank in (
        ("in memory", None),
        ("banked", _bank_store(tmp_path / "semp_lane")),
    ):
        run = f"the ACS local lane's transfer, {path}, of {entity} batch {batch!r}: "
        patterns, result = _run_whole_batch(
            (entity, batch, targets),
            frames,
            bank=bank,
            run=run,
            monkeypatch=monkeypatch,
            semp_covers_partnership_income=True,
        )
        # The export zeroed some nonzero partnership draws and changed no
        # other, while the chain fed the later targets the raw draws there.
        exported = result.frame.table(entity)[_SEMP_PARTNERSHIP]
        raw = _raw_draws(patterns, _SEMP_PARTNERSHIP, exported.index)
        changed = exported.to_numpy() != raw.to_numpy()
        (receipt,) = [
            record.reconciliation
            for record in result.imputed_inputs
            if record.column == _SEMP_PARTNERSHIP
        ]
        assert (exported.to_numpy()[changed] == 0.0).all(), run
        assert changed.sum() == receipt["nonzero_draws_discarded"] > 0, (
            f"{run}the export zeroed {int(changed.sum())} partnership draws; "
            f"the receipt counts {receipt['nonzero_draws_discarded']}"
        )


# ---------------------------------------------------------------------------
# The fixture can tell
# ---------------------------------------------------------------------------


def _rows_differing(left: pd.Series, right: pd.Series) -> int:
    return int((left.to_numpy() != right.to_numpy()).sum())


def test_fixture_discriminates_snapped_and_zeroed_priors(
    production: AcsTransferResult,
    recorded_in_memory: tuple[list[ChainStep], AcsTransferResult],
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
    partnership_steps: list[ChainStep] = []
    monkeypatch.setattr(
        acs_transfer_module,
        "QRF",
        prior_recording_qrf(partnership_steps, base=zeroed_prior_qrf((_PARTNERSHIP,))),
    )
    partnership_zeroed = _transfer()

    snapped_tax_unit = snapped.frame.table("tax_unit")
    zeroed_person = zeroed.frame.table("person")
    partnership_person = partnership_zeroed.frame.table("person")
    affected = {
        f"{HSA} rows, snapped year": _rows_differing(
            snapped_tax_unit[HSA], tax_unit[HSA]
        ),
        f"{FARM_RENT} rows, zeroed components": _rows_differing(
            zeroed_person[FARM_RENT], person[FARM_RENT]
        ),
        f"{ESTATE_FLAG} rows, zeroed components": _rows_differing(
            zeroed_person[ESTATE_FLAG], person[ESTATE_FLAG]
        ),
        f"{ESTATE_FLAG} rows, zeroed {_PARTNERSHIP}": _rows_differing(
            partnership_person[ESTATE_FLAG], person[ESTATE_FLAG]
        ),
    }

    # Fed the snapped year, the chain draws a different HSA deduction; the
    # year, the earlier target and the other family do not move.
    assert affected[f"{HSA} rows, snapped year"] > 0, (
        f"a snapped year prior moved no HSA draw; affected rows: {affected}"
    )
    for column in (MORTGAGE_BALANCE, ORIGINATION_YEAR):
        pd.testing.assert_series_equal(
            snapped_tax_unit[column], tax_unit[column], check_exact=True
        )
    pd.testing.assert_frame_equal(
        snapped.frame.table("person"), person, check_exact=True
    )

    # Fed the zeroed components, the chain draws different farm rents and
    # estate flags; the exported components and the other family do not move.
    for column in (FARM_RENT, ESTATE_FLAG):
        assert affected[f"{column} rows, zeroed components"] > 0, (
            f"zeroed component priors moved no {column} draw; affected rows: {affected}"
        )
    for column in SELF_EMPLOYMENT_COMPONENTS:
        pd.testing.assert_series_equal(
            zeroed_person[column], person[column], check_exact=True
        )
    pd.testing.assert_frame_equal(
        zeroed.frame.table("tax_unit"), tax_unit, check_exact=True
    )

    # Fed zeroed partnership earnings alone, the chain draws different estate
    # flags, while every raw draw up to the partnership earnings is unchanged:
    # the partnership prior moves the estate flag on its own.
    assert affected[f"{ESTATE_FLAG} rows, zeroed {_PARTNERSHIP}"] > 0, (
        f"a zeroed {_PARTNERSHIP} prior moved no estate flag; affected rows: {affected}"
    )
    reference_steps, _recorded = recorded_in_memory
    reference = {_step_key(step): step for step in reference_steps}
    assert sorted(reference) == sorted(_step_key(step) for step in partnership_steps)
    family = _FAMILY_TARGETS["fixture_self_employment"]
    unchanged = family[: family.index(_PARTNERSHIP) + 1]
    for step in partnership_steps:
        if step.target in unchanged:
            assert (
                step.raw_draw.tobytes() == reference[_step_key(step)].raw_draw.tobytes()
            ), step.target
    pd.testing.assert_frame_equal(
        partnership_zeroed.frame.table("tax_unit"), tax_unit, check_exact=True
    )
