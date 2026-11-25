"""The ACS transfer conditions later targets on raw draws, not exported values.

The transfer fits one target at a time: each later target in a family is drawn
on the recipients' raw float64 draws of the earlier ones
(``RegimeGatedQRF.fit_draw_next``). Two kinds of exported value differ from
those raw draws, and neither may reach a later target:

- ``first_home_mortgage_origination_year`` is ``discrete_numeric``: the
  transfer snaps its draws to the donor support when it decodes them, which
  happens inside the outer family loop, after the year's own family is fitted;
- the self-employment components are set to zero after the transfer where
  the recipient's own total is kept (``keep_measured_self_employment_totals``).

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
  later targets fresh;
- the year is snapped and the components zeroed only on export. Each
  exported year is the donor support point nearest its raw draw, and the
  decoder sends an exact midpoint between two support points to the lower one;
- production's plans put the fixture's targets in the fixture's order. For
  each production batch that holds two or more of them, a transfer of those
  targets run under the batch's name draws them in that order, in memory and
  banked;
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

import platform
from collections.abc import Mapping
from importlib.metadata import version
from pathlib import Path

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
from microcosm.build.us_runtime.multispine_pool import (
    pool_pre_clone_gap_fill_target_families,
)
from microcosm.build.us_runtime.us_late_producer_registry import (
    CANONICAL_US_LATE_TRANSFER_GROUPS,
)
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
) -> AcsTransferResult:
    recipient, donor = acs_raw_prior_fixture()
    return transfer_acs_inputs(
        recipient,
        donor,
        target_families=target_families,
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
    reference_steps, uninterrupted, reference_bank = recorded_banked
    order = _bank_order(reference_bank)
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
    # checkpoints, and fitted every later target fresh for every pattern.
    receipt = resumed_bank.receipt()["targets"]
    reference_receipt = reference_bank.receipt()["targets"]
    assert [
        (receipt[str(index)]["source"], receipt[str(index)]["load_status"])
        for index in range(len(order))
    ] == [("checkpoint", "resumed")] * resumed_from + [("rebuilt", "missing")] * (
        len(order) - resumed_from
    )
    reference = {_step_key(step): step for step in reference_steps}
    assert sorted(_step_key(step) for step in steps) == sorted(
        key for key in reference if key[0] in order[resumed_from:]
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

    # Every target, loaded or rebuilt, holds the uninterrupted run's raw draw.
    for index in range(len(order)):
        assert (
            receipt[str(index)]["raw_draw_sha256"]
            == reference_receipt[str(index)]["raw_draw_sha256"]
        ), order[index]

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


# ---------------------------------------------------------------------------
# Production's order
# ---------------------------------------------------------------------------


def _production_batches() -> dict[str, list[tuple[str, str, tuple[str, ...]]]]:
    """Each production plan's chained batches, in the plan's order."""

    def split(
        families: dict[str, dict[str, tuple[str, ...]]],
    ) -> list[tuple[str, str, tuple[str, ...]]]:
        return acs_transfer_module._split_large_target_families(
            [
                (entity, family, tuple(targets))
                for entity, entity_families in families.items()
                for family, targets in sorted(entity_families.items())
            ],
            max_targets_per_fit=DEFAULT_ACS_TRANSFER_MAX_TARGETS_PER_FIT,
        )

    return {
        "declared ACS plan": split(declared_acs_transfer_target_families()),
        "stacked gap fill": split(pool_pre_clone_gap_fill_target_families()),
        "stacked post-PUF transfer": [
            (group.entity, group.family, tuple(group.targets))
            for group in CANONICAL_US_LATE_TRANSFER_GROUPS
        ],
    }


def _pattern_orders(steps: list[ChainStep]) -> set[tuple[str, ...]]:
    """The targets each availability pattern drew, in call order, in one run."""

    orders: dict[tuple, list[str]] = {}
    for step in steps:
        orders.setdefault(tuple(step.index), []).append(step.target)
    return {tuple(order) for order in orders.values()}


def test_production_draws_the_fixture_targets_in_the_fixture_order(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # The fixture's cases exist only where production chains the year ahead
    # of the HSA deduction and the components ahead of the farm rent and the
    # estate flag. Each batch is one chain; separate batches never condition
    # on each other.
    plans = _production_batches()
    runtime_batches: set[tuple[str, str, tuple[str, ...]]] = set()
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
                    runtime_batches.add((entity, batch, tuple(shared)))

    # The transfer draws each such batch in that order at runtime. The order
    # matters only where a batch holds two or more fixture targets; the
    # fixture has donor columns for those targets only, so each run holds
    # the batch's fixture targets, under the batch's production name.
    assert runtime_batches
    for number, (entity, batch, shared) in enumerate(sorted(runtime_batches)):
        for path, bank in (
            ("in memory", None),
            ("banked", _bank_store(tmp_path / f"order_{number}")),
        ):
            steps: list[ChainStep] = []
            monkeypatch.setattr(acs_transfer_module, "QRF", prior_recording_qrf(steps))
            _transfer(target_bank=bank, target_families={entity: {batch: shared}})
            orders = _pattern_orders(steps)
            assert orders == {shared}, (
                f"the transfer, {path}, draws {entity} batch {batch!r} as "
                f"{sorted(orders)}; the plan orders it {list(shared)}"
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
