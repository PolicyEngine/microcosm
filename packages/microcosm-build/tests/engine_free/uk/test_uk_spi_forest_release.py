"""The UK SPI income imputation frees each fitted model once it is finished with it.

``impute_uk_spi_income_support`` fits the stage-1 forests on the SPI tape, one
set of reference forests per cross-fitting fold on the FRS rows, then the
stage-2 forests. These tests pin that the stage-1 and reference forests are
dead when stage 2 starts, that each reference fold's forests are dead before
the next fold is fitted, that each stage-2 target's forests are dead before the
next target is fitted, and that the person table and every receipt stay
byte-identical.

Two kinds of comparison:

- In this process, against ``FitEveryTargetFirstQRF``: the real model with the
  lifetimes and order the stage used to have. The inputs the stage passes it
  and the receipts it reports are checked against expressions written out here
  from what the stage computed before. These run on every platform.
- Against SHA-256 digests recorded by running origin/main 402c3587d's
  ``impute_uk_spi_income_support`` on the fixture. QRF draws are bitwise
  reproducible only within one platform (``docs/graph-qrf-cross-platform.md``),
  so the digests are kept per operating system and machine: macOS arm64
  natively, and Linux x86_64 (CI's platform) under Docker emulation on Apple
  silicon. They are checked only where the platform and the versions of the
  libraries that compute the draws match the recording; elsewhere those tests
  skip.

A change that is meant to move these outputs records new digests from its own
code on each platform with
``python -m test_support.microcosm_build.uk_spi_forest_release``, says why the
outputs move, and renames what here says the digests come from origin/main
402c3587d. Any other change must leave them passing.
"""

from __future__ import annotations

import platform
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass
from importlib.metadata import version

import numpy as np
import pandas as pd
import pytest

from microcosm.build.gates import FitWeightRecord
from microcosm.build.stochastic_assignment import (
    integer_record_keys,
    stable_identity_uniforms,
)
from microcosm.build.uk_runtime.qrf_gates import (
    UK_DEFAULT_GATE_CALIBRATION,
    gate_level_reports,
    gate_level_settings_receipt,
    gate_levels,
    gate_outcomes,
    pooled_gate_level_reports,
    pooled_gate_levels,
    pooled_gate_outcomes,
)
from microcosm.build.uk_runtime.spi_income import (
    FRS_ONLY_FIT_NAME,
    SPI_DONOR_FIT_NAME,
    SPI_MINIMUM_RECIPIENT_AGE,
    SPI_STAGE2_REVIEWED_ABSENT_OUTPUTS,
    UKSPIIncomeImputationResult,
)
from microcosm.build.uk_runtime.spi_rank_transport import (
    SPI_TRANSPORT_FIT_NAME,
    SPI_TRANSPORT_SEED_OFFSET,
    analogue_targets,
)
from microcosm.build.uk_runtime.spi_support import (
    BASE_FRS_SUPPORT_CHANNEL,
    FRS_ONLY_SPI_FILL_INCOME_PREDICTOR_COLUMNS,
    FRS_ONLY_SPI_FILL_PERSON_COLUMNS,
    SPI_INCOME_QRF_OUTPUT_COLUMNS,
    SPI_SYNTHETIC_SUPPORT_CHANNEL,
    support_channel_column,
)
from microcosm.fit import RegimeGatedQRF
from test_support.microcosm_build.uk_spi_forest_release import (
    REFERENCE_FIT,
    SPI_FOREST_ALL_SCENARIOS,
    SPI_FOREST_SCENARIOS,
    SPI_FOREST_SEED,
    SPI_FOREST_TREES,
    STAGE1_FIT,
    STAGE2_FIT,
    FitEveryTargetFirstQRF,
    RecordedStage2Start,
    RecordedStage2Step,
    SPIForestLifetimes,
    fitted_target_parts,
    receipt_digest,
    run_spi_forest_scenario,
    spi_forest_support,
    spi_frozen_digests,
    spi_result_digests,
    tracked_spi_forest_fits,
)

# ---------------------------------------------------------------------------
# origin/main's digests, by platform
# ---------------------------------------------------------------------------

_ORIGIN_MAIN_DIGESTS: dict[tuple[str, str], dict[str, dict[str, str]]] = {
    ("Darwin", "arm64"): {
        "in_sample": {
            "person": "280c69c2248b6741e6b95cf26bd2c9eeb457131283bf1906fd327401340b9c13",
            "receipts": "1250371e2c3c3fe425c9497dbeedc09fd6046ffcf532d7f05e7f9a3e950d2280",
        },
        "cross_fitted_levels": {
            "person": "e938490d187e48bfc2b5f676598d58f898d6d97c71c5f439303a2ca8e64e9760",
            "receipts": "aca65550fc5502afb03d78a784018f62dce6d41aa2ed0e5b676e576630caaf3b",
        },
    },
    ("Linux", "x86_64"): {
        "in_sample": {
            "person": "5f582f65cb26446b52c5efa83abedcdcd294744b5dc0b059d9706279a1e15d39",
            "receipts": "a6e3f464f10b7c037b908d75a39274f94e253df612b8a99eaa11bf6746cb547e",
        },
        "cross_fitted_levels": {
            "person": "db4b50143da3dc3927f6de08d6efe51f57dfa9ad2e87c8c65b47d07abfb448be",
            "receipts": "d1103a63326a13d88981e9188b48162b62c7ad75cea2be459ef93a837bf6777c",
        },
    },
}

#: The stage-2 uniform salts as released, written out so that a change to the
#: stage's constants, which would redraw every row, fails here.
_STAGE2_FIT_NAME_AS_RELEASED = "uk_frs_only_spi_fill"
_STAGE2_SALT_NAMES_AS_RELEASED = {
    "individual_savings_account_interest_income": "tax_free_savings_income"
}

#: Versions of the libraries that compute the draws, on both recordings.
_RECORDED_DEPENDENCIES = {
    "numpy": "2.4.6",
    "pandas": "3.0.3",
    "scipy": "1.17.1",
    "scikit-learn": "1.8.0",
    "quantile-forest": "1.4.2",
}


def _origin_main_digests(scenario: str) -> dict[str, str]:
    """This platform's recorded digests, or a skip where they do not apply."""

    here = (platform.system(), platform.machine())
    recorded = _ORIGIN_MAIN_DIGESTS.get(here)
    if recorded is None:
        pytest.skip(
            f"origin/main's digests were recorded on {sorted(_ORIGIN_MAIN_DIGESTS)}, "
            f"not {here}; the same-run checks still run here."
        )
    installed = {name: version(name) for name in _RECORDED_DEPENDENCIES}
    if installed != _RECORDED_DEPENDENCIES:
        pytest.skip(
            f"origin/main's digests were recorded with {_RECORDED_DEPENDENCIES}, "
            f"not {installed}; the same-run checks still run here."
        )
    return recorded[scenario]


# ---------------------------------------------------------------------------
# One run of each scenario, under the lifetime check
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _TrackedRun:
    result: UKSPIIncomeImputationResult
    lifetimes: SPIForestLifetimes


@pytest.fixture(scope="module")
def tracked_run(tmp_path_factory) -> Callable[[str], _TrackedRun]:
    """Each scenario's run with real forests, made once for the tests that read it.

    Lifetime violations are recorded, not raised, so a kept model fails the
    lifetime test alone and the byte checks still report on the run's output.
    """

    runs: dict[str, _TrackedRun] = {}

    def run(scenario: str) -> _TrackedRun:
        if scenario not in runs:
            directory = tmp_path_factory.mktemp(scenario)
            with (
                pytest.MonkeyPatch.context() as monkeypatch,
                tracked_spi_forest_fits(
                    monkeypatch, stop_at_first_violation=False
                ) as lifetimes,
            ):
                result = run_spi_forest_scenario(scenario, directory, monkeypatch)
            runs[scenario] = _TrackedRun(result, lifetimes)
        return runs[scenario]

    return run


@dataclass(frozen=True)
class _ReferenceRun:
    result: UKSPIIncomeImputationResult
    kept_fits: list[object]
    chain_starts: list[RecordedStage2Start]
    steps: list[RecordedStage2Step]


@pytest.fixture(scope="module")
def reference_run(tmp_path_factory) -> Callable[[str], _ReferenceRun]:
    """Each scenario's run with ``FitEveryTargetFirstQRF``, made once."""

    runs: dict[str, _ReferenceRun] = {}

    def run(scenario: str) -> _ReferenceRun:
        if scenario not in runs:
            directory = tmp_path_factory.mktemp(f"reference_{scenario}")
            FitEveryTargetFirstQRF.forget_recorded()
            with pytest.MonkeyPatch.context() as monkeypatch:
                result = run_spi_forest_scenario(
                    scenario,
                    directory,
                    monkeypatch,
                    qrf_class=FitEveryTargetFirstQRF,
                )
            runs[scenario] = _ReferenceRun(
                result,
                list(FitEveryTargetFirstQRF.kept_fits),
                list(FitEveryTargetFirstQRF.chain_starts),
                list(FitEveryTargetFirstQRF.steps),
            )
            FitEveryTargetFirstQRF.forget_recorded()
        return runs[scenario]

    return run


# ---------------------------------------------------------------------------
# Output and receipts
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("scenario", SPI_FOREST_SCENARIOS)
def test_output_and_receipts_reproduce_origin_main_bytes(scenario, tracked_run) -> None:
    expected = _origin_main_digests(scenario)

    assert spi_frozen_digests(tracked_run(scenario).result) == expected


@pytest.mark.parametrize("scenario", SPI_FOREST_SCENARIOS)
def test_output_and_receipts_match_fitting_every_target_first(
    scenario, tracked_run, reference_run
) -> None:
    """Freeing models and drawing per target moves no value and no receipt."""

    reference = reference_run(scenario).result
    result = tracked_run(scenario).result

    pd.testing.assert_frame_equal(result.person, reference.person, check_exact=True)
    assert spi_result_digests(result) == spi_result_digests(reference)


@pytest.mark.parametrize("scenario", SPI_FOREST_ALL_SCENARIOS)
def test_receipts_equal_what_the_kept_models_report(scenario, reference_run) -> None:
    """The receipts, built as the stage built them from the models it held."""

    run = reference_run(scenario)
    options = SPI_FOREST_ALL_SCENARIOS[scenario]
    stage1, *reference_folds, stage2 = run.kept_fits
    stage1_gate = options.get("stage1_gate_calibration", UK_DEFAULT_GATE_CALIBRATION)
    stage2_gate = options.get("stage2_gate_calibration", UK_DEFAULT_GATE_CALIBRATION)
    level_columns = options.get("stage1_gate_level_columns", ())
    expected_outcomes = {
        **(
            {}
            if stage1_gate == UK_DEFAULT_GATE_CALIBRATION
            else {
                **gate_outcomes(SPI_DONOR_FIT_NAME, stage1),
                **pooled_gate_outcomes(SPI_TRANSPORT_FIT_NAME, reference_folds),
            }
        ),
        **(
            {}
            if stage2_gate == UK_DEFAULT_GATE_CALIBRATION
            else gate_outcomes(FRS_ONLY_FIT_NAME, stage2)
        ),
    }
    expected_levels = (
        {
            "declared": list(level_columns),
            "settings": gate_level_settings_receipt(),
            "offset_levels": {
                **gate_levels(SPI_DONOR_FIT_NAME, stage1),
                **pooled_gate_levels(SPI_TRANSPORT_FIT_NAME, reference_folds),
            },
            "level_fits": {
                **gate_level_reports(SPI_DONOR_FIT_NAME, stage1),
                **pooled_gate_level_reports(SPI_TRANSPORT_FIT_NAME, reference_folds),
            },
        }
        if level_columns
        else None
    )
    expected_weight_records = (
        FitWeightRecord(SPI_DONOR_FIT_NAME, stage1.weight_kind),
        FitWeightRecord(SPI_TRANSPORT_FIT_NAME, reference_folds[0].weight_kind),
        FitWeightRecord(FRS_ONLY_FIT_NAME, stage2.weight_kind),
    )

    assert len(reference_folds) == len(run.result.stage1_transport["fold_receipts"])
    assert run.result.gate_outcomes == expected_outcomes
    assert receipt_digest(run.result.gate_outcomes) == receipt_digest(expected_outcomes)
    assert receipt_digest(run.result.gate_levels) == receipt_digest(expected_levels)
    assert receipt_digest(run.result.fit_weight_records) == receipt_digest(
        expected_weight_records
    )


@pytest.mark.parametrize("scenario", SPI_FOREST_ALL_SCENARIOS)
def test_stage2_chain_gets_the_settings_and_inputs_of_the_whole_fit_and_draw(
    scenario, reference_run
) -> None:
    """Model settings, targets, rows, predictors, uniforms, groups and prior draws.

    The expected values are written out from what the stage passed to one fit
    over every target and one ``predict_from_uniforms``. The income predictors
    of the drawn rows hold stage-1 draws, which vary by platform, so only the
    byte tests cover their values.
    """

    run = reference_run(scenario)
    options = SPI_FOREST_ALL_SCENARIOS[scenario]
    support = spi_forest_support()
    person = support.person
    channel = person[support_channel_column("person")]
    base_rows = channel == BASE_FRS_SUPPORT_CHANNEL
    drawn_rows = (channel == SPI_SYNTHETIC_SUPPORT_CHANNEL) & person["age"].ge(
        SPI_MINIMUM_RECIPIENT_AGE
    )
    drawn_people = person.loc[drawn_rows]
    drawn_regions = drawn_people["person_household_id"].map(
        support.household.set_index("household_id")["region"]
    )
    genders = sorted(person["gender"].unique())
    regions = sorted(support.household["region"].unique())
    expected_targets = tuple(
        column
        for column in FRS_ONLY_SPI_FILL_PERSON_COLUMNS
        if column not in SPI_STAGE2_REVIEWED_ABSENT_OUTPUTS
    )
    expected_predictors = tuple(
        sorted(
            [
                "age",
                *FRS_ONLY_SPI_FILL_INCOME_PREDICTOR_COLUMNS,
                *(f"gender_{gender}" for gender in genders),
                *(f"region_{region}" for region in regions),
                *(
                    ["state_pension_receipt"]
                    if options.get("condition_on_state_pension_receipt")
                    else []
                ),
            ]
        )
    )
    expected_groups = person.loc[base_rows, "person_household_id"].to_numpy()
    record_keys = integer_record_keys(
        drawn_people["person_id"].to_numpy(), label="SPI stage-2 person_id"
    )
    stage2_seed = SPI_FOREST_SEED + 1

    assert len(run.chain_starts) == 1
    start = run.chain_starts[0]
    assert start.seed == stage2_seed
    assert start.n_estimators == SPI_FOREST_TREES
    assert start.gate_calibration == options.get(
        "stage2_gate_calibration", UK_DEFAULT_GATE_CALIBRATION
    )
    assert start.predictors == expected_predictors
    assert start.targets == expected_targets
    np.testing.assert_array_equal(start.groups, expected_groups)
    assert tuple(step.target for step in run.steps) == expected_targets
    for position, step in enumerate(run.steps):
        salt_name = _STAGE2_SALT_NAMES_AS_RELEASED.get(step.target, step.target)
        recipients = step.recipient_predictors
        assert recipients.index.equals(drawn_people.index)
        assert tuple(recipients.columns) == expected_predictors
        np.testing.assert_array_equal(recipients["age"], drawn_people["age"])
        for gender in genders:
            np.testing.assert_array_equal(
                recipients[f"gender_{gender}"], drawn_people["gender"].eq(gender)
            )
        for region in regions:
            np.testing.assert_array_equal(
                recipients[f"region_{region}"], drawn_regions.eq(region)
            )
        np.testing.assert_array_equal(step.groups, expected_groups)
        np.testing.assert_array_equal(
            step.quantiles,
            stable_identity_uniforms(
                record_keys,
                seed=stage2_seed,
                salt=f"{_STAGE2_FIT_NAME_AS_RELEASED}:{salt_name}",
            ),
        )
        np.testing.assert_array_equal(
            step.sign_uniforms,
            stable_identity_uniforms(
                record_keys,
                seed=stage2_seed,
                salt=f"{_STAGE2_FIT_NAME_AS_RELEASED}:{salt_name}:sign",
            ),
        )
        assert list(step.prior_draws.columns) == list(expected_targets[:position])
        for earlier in run.steps[:position]:
            assert (
                step.prior_draws[earlier.target].to_numpy().tobytes()
                == earlier.raw_draw.tobytes()
            )


def test_the_cross_fitted_scenario_fills_every_receipt_read_from_a_model(
    tracked_run,
) -> None:
    """The byte checks above would prove little over empty receipts."""

    result = tracked_run("cross_fitted_levels").result

    every_fit = {SPI_DONOR_FIT_NAME, SPI_TRANSPORT_FIT_NAME, FRS_ONLY_FIT_NAME}
    assert {key.split(":", 1)[0] for key in result.gate_outcomes} == every_fit
    stage1_fits = {SPI_DONOR_FIT_NAME, SPI_TRANSPORT_FIT_NAME}
    for receipt in ("offset_levels", "level_fits"):
        keys = result.gate_levels[receipt]
        assert {key.split(":", 1)[0] for key in keys} == stage1_fits
    assert result.pension_receipt_bridge is not None


# ---------------------------------------------------------------------------
# Lifetimes
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("scenario", SPI_FOREST_SCENARIOS)
def test_each_fit_starts_after_the_models_it_replaces_are_dead(
    scenario, tracked_run
) -> None:
    run = tracked_run(scenario)

    assert run.lifetimes.violations == []
    fits = Counter(fit for fit, _, _ in run.lifetimes.fitted)
    folds = len(run.result.stage1_transport["fold_receipts"])
    assert folds > 1
    assert fits[STAGE1_FIT] == len(SPI_INCOME_QRF_OUTPUT_COLUMNS)
    assert fits[REFERENCE_FIT] == folds * len(analogue_targets())
    assert fits[STAGE2_FIT] > 1


@pytest.mark.parametrize("scenario", SPI_FOREST_SCENARIOS)
def test_no_fitted_model_outlives_the_stage(scenario, tracked_run) -> None:
    lifetimes = tracked_run(scenario).lifetimes

    for fit in (STAGE1_FIT, REFERENCE_FIT, STAGE2_FIT):
        assert lifetimes.alive(fit) == []


def _fitted_forest_target(gate_calibration: str) -> object:
    """One real fitted target with a sign classifier and forests of both signs."""

    generator = np.random.default_rng(1)
    predictor = generator.random(300)
    target = np.select(
        [predictor < 1 / 3, predictor > 2 / 3],
        [-1.0 - predictor, 1.0 + predictor],
        default=0.0,
    )
    table = pd.DataFrame({"predictor": predictor, "target": target})
    fitted = RegimeGatedQRF(
        n_estimators=2, seed=SPI_FOREST_SEED, gate_calibration=gate_calibration
    ).fit(table, ["predictor"], ["target"], weights="none")
    return fitted._target_models["target"]


_IN_SAMPLE_GATE_PARTS = (
    "target model",
    "sign classifier",
    "sign classifier bins",
    "sign classifier tree",
    "sign classifier tree nodes",
    *(
        f"{sign} {part}"
        for sign in ("positive", "negative")
        for part in (
            "forest",
            "regressor",
            "tree",
            "compiled tree",
            "leaf sorter",
            "leaf array",
        )
    ),
)
_CROSS_FITTED_GATE_PARTS = (
    "fold sign classifier",
    "fold sign classifier bins",
    "fold sign classifier tree",
    "fold sign classifier tree nodes",
)


@pytest.mark.parametrize(
    ("gate_calibration", "kept_kind"),
    [
        *(("in_sample", kind) for kind in _IN_SAMPLE_GATE_PARTS),
        *(("cross_fitted", kind) for kind in _CROSS_FITTED_GATE_PARTS),
    ],
)
def test_the_lifetime_check_notices_one_kept_part(gate_calibration, kept_kind) -> None:
    lifetimes = SPIForestLifetimes(seed=SPI_FOREST_SEED)
    model = _fitted_forest_target(gate_calibration)
    lifetimes.record(STAGE1_FIT, 0, "target", model)
    kept = next(part for kind, part in fitted_target_parts(model) if kind == kept_kind)
    del model

    with pytest.raises(AssertionError, match=kept_kind):
        lifetimes.check_before_fit(STAGE2_FIT, 0, "next_target")

    assert lifetimes.violations
    del kept
    lifetimes.check_before_fit(STAGE2_FIT, 0, "next_target")


@pytest.mark.parametrize(
    ("kept_seed_offset", "kept_fit", "failing_fit"),
    [
        (0, STAGE1_FIT, STAGE2_FIT),
        (SPI_TRANSPORT_SEED_OFFSET, REFERENCE_FIT, REFERENCE_FIT),
    ],
    ids=["stage_1_kept", "reference_kept"],
)
def test_the_lifetime_check_notices_a_model_the_stage_keeps(
    kept_seed_offset, kept_fit, failing_fit, tmp_path, monkeypatch
) -> None:
    """The check run on the stage itself fails when one fit is held on to."""

    kept_fits: list[object] = []

    class KeepsOneFit(RegimeGatedQRF):
        def fit(self, *args, **kwargs):
            fitted = super().fit(*args, **kwargs)
            if self.seed == SPI_FOREST_SEED + kept_seed_offset:
                kept_fits.append(fitted)
            return fitted

    with (
        tracked_spi_forest_fits(monkeypatch) as lifetimes,
        pytest.raises(AssertionError, match=f"the {failing_fit} fit of"),
    ):
        run_spi_forest_scenario(
            "in_sample", tmp_path, monkeypatch, qrf_class=KeepsOneFit
        )

    assert kept_fits
    assert f"('{kept_fit}'," in lifetimes.violations[0]
