"""UK AI-exposure imputation: donor contract, draws, weights, planted signal.

The LFS/APS donor and the FRS major-group merge are UKDS-licensed and never
committed, so every test builds small synthetic frames: persons carry an
observed ``soc_major_group`` (assigned independently of the demographics, so
demographics alone cannot recover it) and an exposure score with a known
structure — a per-major-group base level plus a within-group education signal
plus noise. The behavioral contracts asserted are the ones the build relies
on: fit+impute runs end to end, draws stay inside the donor's observed
support, the fit honors the frame's typed weights, the planted signals
survive imputation directionally, conditioning on the observed major group
beats the blind fallback, and the zero-model baseline reproduces the
crosswalk's employment-weighted group means exactly.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from populace.build.plan import Stage
from populace.build.uk_runtime.exposure_imputation import (
    DEFAULT_EXPOSURE_COLUMN,
    SOC_MAJOR_GROUP_COLUMN,
    UK_EXPOSURE_IMPUTATION_STAGE_NAME,
    attach_exposure,
    exposure_from_major_group,
    exposure_imputation_stage,
    fit_exposure_imputer,
    impute_exposure,
)
from populace.frame import EntitySchema, Frame, WeightKind, Weights

#: One-person-per-household schema, so person-level weights are unambiguous
#: and every covariate lives on the person entity.
SCHEMA = EntitySchema(group_entities=("household",))

#: Predictors carried by the synthetic donor and target frames: the observed
#: major group first (the primary path), then a subset of the production
#: DEFAULT_PREDICTORS — the contract under test is the fit machinery, not the
#: LFS/FRS harmonization.
PREDICTORS = [
    SOC_MAJOR_GROUP_COLUMN,
    "age",
    "gender",
    "highest_education",
    "employment_income",
    "hours",
]

#: The documented blind fallback: demographics only, no observed occupation.
BLIND_PREDICTORS = [p for p in PREDICTORS if p != SOC_MAJOR_GROUP_COLUMN]

#: FRS adult.tab SOC2020 coding: major group as thousands.
MAJOR_GROUPS = (np.arange(1, 10) * 1000).astype("int64")

#: Planted per-major-group base exposure. Deliberately non-monotone in the
#: code so a model interpolating the group code numerically could not fake it.
GROUP_BASE = dict(
    zip(
        MAJOR_GROUPS.tolist(),
        [0.85, 0.75, 0.65, 0.55, 0.45, 0.35, 0.30, 0.20, 0.15],
        strict=True,
    )
)

#: Within-group education slope and noise scale of the planted exposure.
EDUCATION_SLOPE = 0.05
NOISE_SCALE = 0.03

DONOR_N = 400
TARGET_N = 250


def _person_frame(columns: dict[str, np.ndarray], weights: np.ndarray) -> Frame:
    """Assemble a one-person-per-household frame with person design weights."""
    n = len(weights)
    person = pd.DataFrame(
        {
            "person_id": np.arange(n, dtype="int64"),
            "person_household_id": np.arange(n, dtype="int64"),
            **columns,
        }
    )
    household = pd.DataFrame({"household_id": np.arange(n, dtype="int64")})
    return Frame(
        {"person": person, "household": household},
        SCHEMA,
        {"person": Weights(values=weights, kind=WeightKind.DESIGN)},
    )


def _covariates(n: int, rng: np.random.Generator) -> dict[str, np.ndarray]:
    """Draw the shared covariates, with the major group independent of them."""
    education = rng.integers(0, 4, size=n).astype("int64")
    income = np.exp(rng.normal(9.8, 0.5, size=n) + 0.25 * education)
    return {
        SOC_MAJOR_GROUP_COLUMN: rng.choice(MAJOR_GROUPS, size=n),
        "age": rng.integers(18, 65, size=n).astype("int64"),
        "gender": rng.integers(0, 2, size=n).astype("int64"),
        "highest_education": education,
        "employment_income": income,
        "hours": rng.uniform(10.0, 45.0, size=n),
    }


def _planted_exposure(
    columns: dict[str, np.ndarray], rng: np.random.Generator
) -> np.ndarray:
    """Exposure = major-group base + within-group education signal + noise."""
    base = np.vectorize(GROUP_BASE.get)(columns[SOC_MAJOR_GROUP_COLUMN])
    signal = EDUCATION_SLOPE * columns["highest_education"]
    noise = rng.normal(0.0, NOISE_SCALE, size=len(signal))
    return np.clip(base + signal + noise, 0.0, 1.0)


def _synthetic_population(
    n: int, seed: int, *, with_exposure: bool
) -> tuple[Frame, np.ndarray]:
    """Build a frame and its true planted exposure (attached only if asked)."""
    rng = np.random.default_rng(seed)
    columns = _covariates(n, rng)
    exposure = _planted_exposure(columns, rng)
    if with_exposure:
        columns = {**columns, DEFAULT_EXPOSURE_COLUMN: exposure}
    return _person_frame(columns, np.full(n, 5.0)), exposure


@pytest.fixture
def donor_frame() -> tuple[Frame, np.ndarray]:
    """A synthetic LFS-like donor: covariates, major group, exposure."""
    return _synthetic_population(DONOR_N, 1, with_exposure=True)


@pytest.fixture
def target_frame() -> tuple[Frame, np.ndarray]:
    """A synthetic FRS-like target: covariates and major group, no exposure.

    The true (never-attached) exposure is returned alongside so error
    comparisons can score the imputation against the planted ground truth.
    """
    return _synthetic_population(TARGET_N, 2, with_exposure=False)


# ----------------------------------------------------------------------------
# Fit + impute end to end, and draws stay in the donor's observed support
# ----------------------------------------------------------------------------


def test_fit_and_impute_runs_and_draws_stay_in_donor_range(
    donor_frame, target_frame
) -> None:
    """One draw per target person, every draw inside the donor's support.

    The QRF draws by interpolating observed conditional quantiles, so no draw
    can leave the donor's overall [min, max] — a value outside it would mean
    the model extrapolated an exposure no donor row exhibits.
    """
    donor, exposure = donor_frame
    target, _ = target_frame
    fitted = fit_exposure_imputer(donor, predictors=PREDICTORS, n_estimators=30, seed=0)
    draws = impute_exposure(fitted, target)

    assert isinstance(draws, pd.Series)
    assert draws.name == DEFAULT_EXPOSURE_COLUMN
    assert len(draws) == TARGET_N
    values = draws.to_numpy()
    assert np.isfinite(values).all()
    assert values.min() >= exposure.min()
    assert values.max() <= exposure.max()


def test_higher_education_persons_draw_higher_exposure(
    donor_frame, target_frame
) -> None:
    """The planted within-group education signal survives directionally.

    Major groups are assigned independently of education, so across many
    persons the group bases average out and the education slope must show:
    target persons in the top education band draw a higher mean exposure
    than those in the bottom band.
    """
    donor, _ = donor_frame
    target, _ = target_frame
    fitted = fit_exposure_imputer(donor, predictors=PREDICTORS, n_estimators=30, seed=0)
    draws = impute_exposure(fitted, target).to_numpy()
    education = target.person["highest_education"].to_numpy()

    low = draws[education == 0].mean()
    high = draws[education == 3].mean()
    assert high > low


# ----------------------------------------------------------------------------
# Primary vs blind path: the observed major group must earn its place
# ----------------------------------------------------------------------------


def test_blind_fallback_warns_and_runs(donor_frame, target_frame) -> None:
    """Omitting soc_major_group is allowed but loudly flagged."""
    donor, _ = donor_frame
    target, _ = target_frame
    with pytest.warns(UserWarning, match="blind fallback"):
        fitted = fit_exposure_imputer(
            donor, predictors=BLIND_PREDICTORS, n_estimators=20, seed=0
        )
    draws = impute_exposure(fitted, target)
    assert len(draws) == TARGET_N


def test_major_group_conditioning_beats_the_blind_path(
    donor_frame, target_frame
) -> None:
    """Conditioning on the observed major group reduces imputation error.

    The major group carries most of the planted variance and is independent
    of the demographics, so the blind path structurally cannot recover it:
    its draws mix the group bases. Scored against the planted ground truth,
    the within-group refinement's mean absolute error must be decisively
    smaller than the blind fallback's.
    """
    donor, _ = donor_frame
    target, truth = target_frame

    primary = fit_exposure_imputer(
        donor, predictors=PREDICTORS, n_estimators=30, seed=0
    )
    with pytest.warns(UserWarning, match="blind fallback"):
        blind = fit_exposure_imputer(
            donor, predictors=BLIND_PREDICTORS, n_estimators=30, seed=0
        )

    primary_mae = np.abs(impute_exposure(primary, target).to_numpy() - truth).mean()
    blind_mae = np.abs(impute_exposure(blind, target).to_numpy() - truth).mean()

    assert primary_mae < blind_mae - 0.05


# ----------------------------------------------------------------------------
# Weights: the fit reads the donor frame's typed weights and they move draws
# ----------------------------------------------------------------------------


def test_donor_weights_shift_the_imputed_distribution(target_frame) -> None:
    """A weighted fit reproduces the weighted, not unweighted, conditional.

    The donor mixes two exposure clusters with identical covariate
    distributions: a low cluster at weight 1 and a high cluster at weight 20.
    Per the populace-fit contract a Frame fit defaults to the typed design
    weights, so the weighted draws' mean must land near the high cluster,
    well above the unweighted (``weights="none"``) draws' mean.
    """
    target, _ = target_frame
    rng = np.random.default_rng(3)
    n = DONOR_N
    columns = _covariates(n, rng)
    low_cluster = rng.uniform(0.1, 0.2, size=n)
    high_cluster = rng.uniform(0.8, 0.9, size=n)
    is_high = np.arange(n) % 2 == 0
    exposure = np.where(is_high, high_cluster, low_cluster)
    weights = np.where(is_high, 20.0, 1.0)
    donor = _person_frame({**columns, DEFAULT_EXPOSURE_COLUMN: exposure}, weights)

    weighted = fit_exposure_imputer(
        donor, predictors=PREDICTORS, n_estimators=30, seed=0
    )
    unweighted = fit_exposure_imputer(
        donor, predictors=PREDICTORS, weights="none", n_estimators=30, seed=0
    )
    weighted_mean = impute_exposure(weighted, target).mean()
    unweighted_mean = impute_exposure(unweighted, target).mean()

    # Weighted population mean: (20*0.85 + 1*0.15) / 21 ~= 0.817;
    # unweighted: (0.85 + 0.15) / 2 = 0.5.
    assert weighted_mean > unweighted_mean + 0.15
    assert weighted.weight_kind == "design"
    assert unweighted.weight_kind == "none"


# ----------------------------------------------------------------------------
# Zero-model baseline: employment-weighted group means from the crosswalk
# ----------------------------------------------------------------------------


def test_exposure_from_major_group_reproduces_weighted_means() -> None:
    """The baseline is the crosswalk's employment-weighted mean per group."""
    crosswalk = pd.DataFrame(
        {
            SOC_MAJOR_GROUP_COLUMN: [1000, 1000, 2000, 2000],
            DEFAULT_EXPOSURE_COLUMN: [0.9, 0.5, 0.3, 0.1],
            "employment": [3.0, 1.0, 1.0, 1.0],
        }
    )
    codes = pd.Series([1000, 2000, 1000], name="whatever")
    out = exposure_from_major_group(codes, crosswalk)

    # group 1000: (3*0.9 + 1*0.5) / 4 = 0.8; group 2000: (0.3 + 0.1) / 2 = 0.2.
    assert out.tolist() == pytest.approx([0.8, 0.2, 0.8])
    assert out.name == DEFAULT_EXPOSURE_COLUMN
    assert out.index.equals(codes.index)


def test_exposure_from_major_group_refuses_unknown_code() -> None:
    """A person code absent from the crosswalk fails loudly, not NaN."""
    crosswalk = pd.DataFrame(
        {
            SOC_MAJOR_GROUP_COLUMN: [1000],
            DEFAULT_EXPOSURE_COLUMN: [0.5],
            "employment": [1.0],
        }
    )
    with pytest.raises(ValueError, match="absent from the crosswalk"):
        exposure_from_major_group(np.array([1000, 9000]), crosswalk)
    with pytest.raises(ValueError, match="missing column"):
        exposure_from_major_group(
            np.array([1000]), crosswalk.drop(columns=["employment"])
        )


# ----------------------------------------------------------------------------
# attach_exposure: immutability, pass-through, and explicit failures
# ----------------------------------------------------------------------------


def test_attach_exposure_returns_new_frame_with_column(
    donor_frame, target_frame
) -> None:
    """The column lands on the person table; everything else passes through."""
    donor, _ = donor_frame
    target, _ = target_frame
    fitted = fit_exposure_imputer(donor, predictors=PREDICTORS, n_estimators=20, seed=0)
    draws = impute_exposure(fitted, target)
    attached = attach_exposure(target, draws)

    assert DEFAULT_EXPOSURE_COLUMN in attached.person.columns
    assert attached.column_entity(DEFAULT_EXPOSURE_COLUMN) == "person"
    np.testing.assert_array_equal(
        attached.person[DEFAULT_EXPOSURE_COLUMN].to_numpy(), draws.to_numpy()
    )
    # The input frame is untouched (frames are immutable).
    assert DEFAULT_EXPOSURE_COLUMN not in target.person.columns
    # Weights and strata pass through.
    np.testing.assert_array_equal(
        attached.weights_for("person").values,
        target.weights_for("person").values,
    )
    assert attached.weights_for("person").kind is WeightKind.DESIGN
    assert attached.strata.tolist() == target.strata.tolist()


def test_attach_exposure_refuses_existing_column(donor_frame) -> None:
    """Attaching onto a frame that already carries the column is refused."""
    donor, exposure = donor_frame
    with pytest.raises(ValueError, match="already exists"):
        attach_exposure(donor, exposure)


def test_attach_exposure_refuses_misaligned_or_nonfinite_values(
    target_frame,
) -> None:
    """Length and finiteness violations fail loudly, naming the problem."""
    target, _ = target_frame
    with pytest.raises(ValueError, match="align positionally"):
        attach_exposure(target, np.zeros(TARGET_N - 1))
    bad = np.full(TARGET_N, 0.5)
    bad[0] = np.nan
    with pytest.raises(ValueError, match="non-finite"):
        attach_exposure(target, bad)


# ----------------------------------------------------------------------------
# Donor contract: missing columns fail with the actionable contract message
# ----------------------------------------------------------------------------


def test_fit_refuses_donor_without_exposure_column(target_frame) -> None:
    """A donor missing the exposure score names the crosswalk contract."""
    target, _ = target_frame
    with pytest.raises(ValueError, match="SOC->exposure"):
        fit_exposure_imputer(target, predictors=PREDICTORS)


def test_fit_refuses_donor_missing_predictors(donor_frame) -> None:
    """A donor missing a shared covariate names it and the contract."""
    donor, _ = donor_frame
    with pytest.raises(ValueError, match=r"\['sic_industry_division'\]"):
        fit_exposure_imputer(donor, predictors=[*PREDICTORS, "sic_industry_division"])


# ----------------------------------------------------------------------------
# Stage declaration: the plan-facing wrapper draws and attaches in one step
# ----------------------------------------------------------------------------


def test_exposure_imputation_stage_produces_the_column(
    donor_frame, target_frame
) -> None:
    """The stage transform attaches the exposure draws to the frame."""
    donor, _ = donor_frame
    target, _ = target_frame
    fitted = fit_exposure_imputer(donor, predictors=PREDICTORS, n_estimators=20, seed=0)
    stage = exposure_imputation_stage(fitted)

    assert isinstance(stage, Stage)
    assert stage.name == UK_EXPOSURE_IMPUTATION_STAGE_NAME
    assert stage.produces == (DEFAULT_EXPOSURE_COLUMN,)
    # The primary path consumes the observed major group, so the plan
    # executor refuses to run before the FRS adult.tab merge lands it.
    assert stage.consumes == tuple(PREDICTORS)
    assert SOC_MAJOR_GROUP_COLUMN in stage.consumes
    assert stage.donor is not None

    out = stage.transform(target)
    assert DEFAULT_EXPOSURE_COLUMN in out.person.columns
