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
  lifetimes and order the stage used to have. It runs on every platform.
- Against SHA-256 digests recorded by running origin/main 402c3587d's
  ``impute_uk_spi_income_support`` on the fixture. QRF draws are bitwise
  reproducible only within one platform (``docs/graph-qrf-cross-platform.md``),
  so the digests are kept per operating system and machine: macOS arm64
  natively, and Linux x86_64 (CI's platform) under Docker emulation on Apple
  silicon. They are checked only where the platform and the versions of the
  libraries that compute the draws match the recording; elsewhere those tests
  skip.

A change that is meant to move these outputs records new digests from its own
code on each platform, says why the outputs move, and renames what here says
the digests come from origin/main 402c3587d. Any other change must leave them
passing.
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

from microcosm.build.uk_runtime.spi_income import (
    FRS_ONLY_FIT_NAME,
    SPI_DONOR_FIT_NAME,
    UKSPIIncomeImputationResult,
)
from microcosm.build.uk_runtime.spi_rank_transport import (
    SPI_TRANSPORT_FIT_NAME,
    SPI_TRANSPORT_SEED_OFFSET,
    analogue_targets,
)
from microcosm.build.uk_runtime.spi_support import SPI_INCOME_QRF_OUTPUT_COLUMNS
from microcosm.fit import RegimeGatedQRF
from test_support.microcosm_build.uk_spi_forest_release import (
    REFERENCE_FIT,
    SPI_FOREST_SCENARIOS,
    SPI_FOREST_SEED,
    STAGE1_FIT,
    STAGE2_FIT,
    FitEveryTargetFirstQRF,
    SPIForestLifetimes,
    fitted_target_parts,
    run_spi_forest_scenario,
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
            f"not {here}; the in-process comparison covers this platform."
        )
    installed = {name: version(name) for name in _RECORDED_DEPENDENCIES}
    if installed != _RECORDED_DEPENDENCIES:
        pytest.skip(
            f"origin/main's digests were recorded with {_RECORDED_DEPENDENCIES}, "
            f"not {installed}; the in-process comparison covers these versions."
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


# ---------------------------------------------------------------------------
# Output and receipts
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("scenario", SPI_FOREST_SCENARIOS)
def test_output_and_receipts_reproduce_origin_main_bytes(scenario, tracked_run) -> None:
    expected = _origin_main_digests(scenario)

    assert spi_frozen_digests(tracked_run(scenario).result) == expected


@pytest.mark.parametrize("scenario", SPI_FOREST_SCENARIOS)
def test_output_and_receipts_match_fitting_every_target_first(
    scenario, tracked_run, tmp_path, monkeypatch
) -> None:
    """Freeing models and drawing per target moves no value and no receipt."""

    FitEveryTargetFirstQRF.kept_fits.clear()
    reference = run_spi_forest_scenario(
        scenario, tmp_path, monkeypatch, qrf_class=FitEveryTargetFirstQRF
    )
    assert FitEveryTargetFirstQRF.kept_fits
    FitEveryTargetFirstQRF.kept_fits.clear()

    result = tracked_run(scenario).result
    pd.testing.assert_frame_equal(result.person, reference.person, check_exact=True)
    assert spi_result_digests(result) == spi_result_digests(reference)


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


def _fitted_forest_target() -> object:
    """One real fitted target with a sign classifier and a positive forest."""

    generator = np.random.default_rng(1)
    predictor = generator.random(200)
    table = pd.DataFrame(
        {
            "predictor": predictor,
            "target": np.where(predictor > 0.5, 1.0 + predictor, 0.0),
        }
    )
    fitted = RegimeGatedQRF(n_estimators=2, seed=SPI_FOREST_SEED).fit(
        table, ["predictor"], ["target"], weights="none"
    )
    return fitted._target_models["target"]


@pytest.mark.parametrize(
    "kept_kind",
    [
        "target model",
        "sign classifier",
        "sign classifier tree",
        "positive forest",
        "positive regressor",
        "positive tree",
        "positive leaf array",
    ],
)
def test_the_lifetime_check_notices_one_kept_part(kept_kind) -> None:
    lifetimes = SPIForestLifetimes(seed=SPI_FOREST_SEED)
    model = _fitted_forest_target()
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
