"""UK AI-exposure imputation: LFS/APS-trained exposure scores onto FRS persons.

The FRS *does* observe occupation — the UKDA ``adult.tab`` files carry SOC
2020 at **major-group** level (1-digit; stored as thousands, ``1000``-``9000``)
— but not the 4-digit unit group that task-level AI-exposure measures are
published at. So exposure arrives the same way SPI incomes do
(:mod:`populace.build.uk_runtime.spi_support`): a donor survey that observes
the needed detail trains a conditional model, and the model draws values for
the target frame's persons from shared covariates.

**Design decision — refine within the true major group; impute the numeric
score, never the occupation code.** Two facts drive the design:

1. The stack's canonical imputer (:class:`populace.fit.RegimeGatedQRF`) is
   numeric-only by construction: every target is coerced to ``float64`` at
   fit time, regimes are detected from sign support, and draws are *linear
   interpolations* of quantile-forest predictions. A SOC code fed through
   that pipeline would be treated as a continuous quantity — a draw could
   land between two codes, inventing an occupation that does not exist. So
   the imputed quantity is the real-valued exposure score, for which the
   QRF's weighted conditional draws are exactly the right object.
2. The target person's 1-digit major group is *observed*, not modeled: it is
   merged from the raw FRS ``adult.tab`` (``SOC2020``, linked via
   SERNUM/BENUNIT/PERSON) in a separate data-prep step upstream of this
   module. The imputer's job is therefore **not** to guess occupation from
   demographics — it is to refine exposure *within* the known major group,
   predicting where in the group's exposure distribution this person sits
   from education, income, industry, age, and hours. ``soc_major_group`` is
   accordingly the first (most important) entry of
   :data:`DEFAULT_PREDICTORS`. As a numeric predictor the code's arbitrary
   scale is harmless — forests split on it, they never interpolate it.

Two companion paths bracket the model:

- **Blind fallback** (documented, warned): a frame whose persons lack
  ``soc_major_group`` can still be imputed from demographics alone by
  passing predictors without it; :func:`fit_exposure_imputer` emits a
  :class:`UserWarning` so a build log shows the degradation explicitly.
- **Zero-model baseline**: :func:`exposure_from_major_group` assigns every
  person the employment-weighted mean exposure of their major group straight
  from the crosswalk — no model at all. This is the transparent lower bound
  reported alongside the QRF refinement; the gap between the two is the
  within-group refinement, and their sensitivity is a robustness check.

DONOR CONTRACT
==============
The donor passed to :func:`fit_exposure_imputer` must be an LFS/APS-derived
:class:`~populace.frame.Frame` (or DataFrame with explicit weights) whose
person table already carries the exposure score, produced by joining each
person's fine (4-digit) SOC 2020 code against the SOC->exposure crosswalk
(the ``populace.build.uk_runtime.ai_exposure`` module, developed on its own
branch; it is referenced lazily here so the two branches stay independent).

The concrete donor this stage is built for is the **UKDS EUL Five-Quarter
Longitudinal LFS** panels (seven panels, Apr 2022 - Dec 2024; tab-delimited
``*.tab``, pooled n ~ 16.5k persons, ~10k with a wave-1 4-digit SOC across
311 unit groups). The donor-prep step maps its raw variables onto this
module's harmonized names: 4-digit SOC from ``SOC20M1`` (waves 2-5
``SOC20M2..SOC20M5`` for panel dedup checks), ``age`` from ``AGE1``,
``gender`` from sex, ``highest_education`` from ``HIQUL22D`` (detail in
``HIQUAL22``), ``sic_industry_division`` from ``INDS07``, earnings from
``GRSSWK``/``USGRS99``, ``hours``/FT-PT from the hours block, ``region``
from ``URESMC``, and the person weight from ``LGWT22``/``LGWT24``. The
interface stays donor-agnostic — any survey satisfying the column contract
below fits — but four caveats of *this* donor belong in every build that
uses it:

1. **Sparse 4-digit cells**: at n ~ 16.5k many unit groups are thin. Attach
   exposure at the 3-digit *minor* group by default, using the 4-digit score
   only where the unit-group cell count meets a threshold, so no exposure
   value rides on a handful of respondents.
2. **Panel overlap**: adjacent five-quarter panels share respondents;
   dedupe on ``PERSID`` before fitting, or overlapping persons are silently
   double-weighted.
3. **Vintage**: the 2023-24 panels are cleanest — ONS revised a 2021-22
   occupation-miscoding problem, so earlier panels' SOC codes carry known
   error.
4. **Licensing**: the panels are UKDS End User Licence materials — the
   project registration must cover this use, and the microdata must never
   be committed (see below).

Whatever the source survey, the donor must carry:

- ``ai_exposure`` (or the ``target`` you name): the numeric exposure score,
  finite for every row — persons without a valid SOC must be dropped or
  resolved *before* fitting, never NaN-filled;
- ``soc_major_group``: the 1-digit SOC 2020 major group, derived from the
  donor's fine SOC. Any consistent numeric coding works (``1``-``9`` or the
  FRS's ``1000``-``9000``), but donor and target frames must use the *same*
  coding — the model conditions on the value, it does not normalize it;
- every remaining predictor in :data:`DEFAULT_PREDICTORS` (or the
  ``predictors`` you name), harmonized to the target frame's coding of the
  same concepts: ``age`` (years), ``gender``, ``highest_education``
  (qualification band, ordered int), ``employment_income`` and
  ``self_employment_income`` (annual GBP, same uprating vintage as the
  target frame), ``hours`` (usual weekly hours), ``sic_industry_division``
  (SIC 2007 division), ``region``;
- design weights for the persons (the LFS/APS person weight), stored as the
  frame's typed weights so the fit is weighted by construction.

The LFS/APS and FRS microdata are UKDS End User Licence materials: donor
frames are built locally from licensed files and are **never committed** to
this repository. Tests exercise the contract with synthetic donors only.
"""

from __future__ import annotations

import warnings
from collections.abc import Sequence

import numpy as np
import pandas as pd

from populace.build.plan import DonorSpec, Stage
from populace.fit import DESIGN_WEIGHTS, FittedModel, WeightSpec, fit
from populace.frame import Frame

__all__ = [
    "DEFAULT_EXPOSURE_COLUMN",
    "DEFAULT_PREDICTORS",
    "LFS_APS_EXPOSURE_DONOR",
    "SOC_MAJOR_GROUP_COLUMN",
    "UK_EXPOSURE_IMPUTATION_STAGE_NAME",
    "attach_exposure",
    "exposure_from_major_group",
    "exposure_imputation_stage",
    "fit_exposure_imputer",
    "impute_exposure",
]

#: The person-table column the imputed exposure score lands on.
DEFAULT_EXPOSURE_COLUMN = "ai_exposure"

#: The observed 1-digit SOC 2020 major group on the person table. On the FRS
#: side it is merged from the raw UKDA ``adult.tab`` (``SOC2020``, values
#: ``1000``-``9000``, linked via SERNUM/BENUNIT/PERSON) in a separate
#: data-prep step; on the LFS/APS donor side it is derived from the fine SOC.
SOC_MAJOR_GROUP_COLUMN = "soc_major_group"

#: Covariates shared by the LFS/APS donor and the FRS target frame, in the
#: target frame's coding (see the module docstring's DONOR CONTRACT).
#: ``soc_major_group`` leads because it is the observed, most informative
#: predictor: the model refines exposure *within* the known major group.
DEFAULT_PREDICTORS = (
    SOC_MAJOR_GROUP_COLUMN,
    "age",
    "gender",
    "highest_education",
    "employment_income",
    "self_employment_income",
    "hours",
    "sic_industry_division",
    "region",
)

#: Stage name for the exposure-imputation step of a UK build plan.
UK_EXPOSURE_IMPUTATION_STAGE_NAME = "ai_exposure_imputation"

#: The donor declaration a UK build plan records for this stage. The vintage
#: note is generic on purpose: the concrete quarter/wave is a property of the
#: locally held UKDS files, recorded by the build that loads them.
LFS_APS_EXPOSURE_DONOR = DonorSpec(
    survey="ONS Labour Force Survey, Five-Quarter Longitudinal (EUL)",
    source="UK Data Service (End User Licence), five-quarter longitudinal "
    "LFS panels, Apr 2022 - Dec 2024; SOC 2020 exposure scores joined per "
    "populace.build.uk_runtime.ai_exposure",
    notes="Donor microdata are UKDS-licensed and never committed; the donor "
    "frame is assembled locally with SOC->exposure already joined (see the "
    "DONOR CONTRACT caveats: sparse 4-digit cells, PERSID dedup across "
    "overlapping panels, 2023-24 panels preferred post ONS occupation "
    "recoding). The target frame's soc_major_group comes from the raw FRS "
    "adult.tab (SOC2020, major-group level), merged upstream of this stage.",
)


def _crosswalk_hint() -> str:
    """Name where the donor's exposure column comes from, import-safely.

    The SOC->exposure crosswalk module (``ai_exposure``) is developed on its
    own branch; importing it lazily — and only to sharpen an error message —
    keeps this module usable before that branch lands.
    """
    try:  # pragma: no cover - exercised only once ai_exposure lands
        from populace.build.uk_runtime import ai_exposure  # noqa: F401

        return (
            "join it with populace.build.uk_runtime.ai_exposure "
            "(the SOC->exposure crosswalk) before fitting"
        )
    except ImportError:
        return (
            "join it via the SOC->exposure crosswalk "
            "(populace.build.uk_runtime.ai_exposure, once that module lands) "
            "before fitting"
        )


def _require_donor_columns(
    donor: Frame | pd.DataFrame, predictors: Sequence[str], target: str
) -> None:
    """Refuse a donor that does not satisfy the documented contract.

    :mod:`populace.fit` would also reject missing columns, but its message
    cannot know *why* the column should exist; this check names the donor
    contract and the crosswalk that produces the exposure column, so the
    failure is actionable.

    Raises:
        ValueError: Naming the missing columns and the fix.
    """
    if isinstance(donor, Frame):
        columns = {
            column
            for entity in donor.entities
            for column in donor.table(entity).columns
        }
    else:
        columns = set(donor.columns)
    if target not in columns:
        raise ValueError(
            f"Donor is missing the exposure target column {target!r}. The "
            "donor contract (see populace.build.uk_runtime.exposure_imputation) "
            "requires an LFS/APS-derived frame with the SOC-level exposure "
            f"score already attached; {_crosswalk_hint()}."
        )
    missing = sorted(set(predictors) - columns)
    if missing:
        raise ValueError(
            f"Donor is missing predictor column(s) {missing}. The donor "
            "contract requires the shared LFS/APS-FRS covariates "
            f"{list(predictors)}, harmonized to the target frame's coding "
            "(see populace.build.uk_runtime.exposure_imputation)."
        )


def fit_exposure_imputer(
    donor_frame: Frame | pd.DataFrame,
    predictors: Sequence[str] = DEFAULT_PREDICTORS,
    target: str = DEFAULT_EXPOSURE_COLUMN,
    *,
    weights: WeightSpec = DESIGN_WEIGHTS,
    **model_kwargs,
) -> FittedModel:
    """Fit the exposure imputer on an LFS/APS donor frame.

    A thin, contract-checking front door over :func:`populace.fit.fit`: the
    single target is the numeric exposure score (see the module docstring for
    why the score, not the SOC code, is the imputed quantity), and the fit is
    weight-aware by construction — a Frame donor defaults to its typed design
    weights (the LFS/APS person weight), a DataFrame donor must state its
    weights explicitly, and ``weights="none"`` is the only unweighted path.

    The primary path conditions on the person's observed
    :data:`SOC_MAJOR_GROUP_COLUMN` (refinement within the known major group).
    Omitting it from ``predictors`` selects the documented *blind* fallback —
    exposure from demographics alone — which still runs but emits a
    :class:`UserWarning`, so a build log records the degradation.

    Args:
        donor_frame: The LFS/APS-derived donor satisfying the DONOR CONTRACT
            in the module docstring.
        predictors: Conditioning covariates shared with the target frame.
            Defaults to :data:`DEFAULT_PREDICTORS`.
        target: The exposure column to learn. Defaults to
            :data:`DEFAULT_EXPOSURE_COLUMN`.
        weights: The fit's weight spec, per the
            :class:`populace.fit.ConditionalModel` contract.
        **model_kwargs: Forwarded to the canonical model (e.g.
            ``n_estimators``, ``seed``).

    Returns:
        A fitted :class:`populace.fit.FittedModel` whose single target is
        ``target``.

    Raises:
        ValueError: If the donor is missing the target or a predictor (the
            message names the donor contract), or on any
            :func:`populace.fit.fit` contract violation (non-finite target,
            unresolvable weights, ...).

    Warns:
        UserWarning: When ``predictors`` omit :data:`SOC_MAJOR_GROUP_COLUMN`
            (the blind fallback path).
    """
    predictors = list(predictors)
    if SOC_MAJOR_GROUP_COLUMN not in predictors:
        warnings.warn(
            f"Fitting the exposure imputer without {SOC_MAJOR_GROUP_COLUMN!r}: "
            "this is the blind fallback (exposure from demographics alone). "
            "The FRS observes the 1-digit SOC 2020 major group in adult.tab; "
            "merge it onto the person table upstream and include "
            f"{SOC_MAJOR_GROUP_COLUMN!r} in predictors for the primary "
            "within-group refinement path.",
            UserWarning,
            stacklevel=2,
        )
    _require_donor_columns(donor_frame, predictors, target)
    return fit(donor_frame, predictors, [target], weights=weights, **model_kwargs)


def impute_exposure(fitted: FittedModel, frame: Frame | pd.DataFrame) -> pd.Series:
    """Draw one exposure score per person of ``frame`` from the fitted model.

    Args:
        fitted: The model from :func:`fit_exposure_imputer`.
        frame: The target :class:`~populace.frame.Frame` (drawing from its
            person table) or a person-level DataFrame carrying the predictor
            columns.

    Returns:
        A :class:`pandas.Series` of exposure draws named after the fitted
        target, index-aligned to the person rows.

    Raises:
        ValueError: If ``fitted`` imputes anything but a single target (it
            did not come from :func:`fit_exposure_imputer`), or a predictor
            column is missing from ``frame``.
    """
    drawn = fitted.predict(frame)
    if len(drawn.columns) != 1:
        raise ValueError(
            "impute_exposure expects a model fitted on the single exposure "
            f"target, but this model imputes {list(drawn.columns)}; fit it "
            "with fit_exposure_imputer."
        )
    return drawn[drawn.columns[0]]


def exposure_from_major_group(
    soc_major_group: pd.Series | np.ndarray | Sequence[float],
    crosswalk: pd.DataFrame,
    *,
    group_column: str = SOC_MAJOR_GROUP_COLUMN,
    exposure_column: str = DEFAULT_EXPOSURE_COLUMN,
    employment_column: str = "employment",
) -> pd.Series:
    """Zero-imputation baseline: the major group's employment-weighted mean.

    No model at all: every person receives the employment-weighted mean
    exposure of their observed 1-digit major group, computed straight from
    the SOC->exposure crosswalk. This is the transparent lower bound reported
    alongside the QRF refinement (:func:`fit_exposure_imputer` /
    :func:`impute_exposure`); the difference between the two is exactly the
    within-group refinement, and their sensitivity is a robustness check.

    Args:
        soc_major_group: One major-group code per person (any consistent
            numeric coding — ``1``-``9`` or the FRS's ``1000``-``9000`` — as
            long as it matches ``crosswalk[group_column]`` exactly; codes are
            matched, never normalized).
        crosswalk: The SOC->exposure crosswalk at any SOC grain, one row per
            occupation, carrying ``group_column`` (the occupation's major
            group), ``exposure_column`` (its score), and
            ``employment_column`` (its employment count/weight, the
            aggregation weight).
        group_column: The crosswalk's major-group column. Defaults to
            :data:`SOC_MAJOR_GROUP_COLUMN`.
        exposure_column: The crosswalk's score column. Defaults to
            :data:`DEFAULT_EXPOSURE_COLUMN`.
        employment_column: The crosswalk's employment-weight column.

    Returns:
        A :class:`pandas.Series` named ``exposure_column``, one baseline
        score per person, index-aligned to a Series input (positional for
        array input).

    Raises:
        ValueError: If the crosswalk is missing a column, carries non-finite
            or negative employment, a group's total employment is zero, or a
            person's code is absent from the crosswalk. Messages name the
            culprits.
    """
    missing = sorted(
        {group_column, exposure_column, employment_column} - set(crosswalk.columns)
    )
    if missing:
        raise ValueError(f"crosswalk is missing column(s): {missing}.")

    employment = crosswalk[employment_column].to_numpy(dtype=np.float64)
    if not np.isfinite(employment).all() or (employment < 0).any():
        raise ValueError(
            f"crosswalk.{employment_column} must be finite and non-negative."
        )
    exposure = crosswalk[exposure_column].to_numpy(dtype=np.float64)
    if not np.isfinite(exposure).all():
        raise ValueError(f"crosswalk.{exposure_column} must be finite.")

    grouped = pd.DataFrame(
        {
            "group": crosswalk[group_column].to_numpy(),
            "mass": employment * exposure,
            "employment": employment,
        }
    ).groupby("group", sort=True)
    totals = grouped[["mass", "employment"]].sum()
    zero_groups = totals.index[totals["employment"] <= 0.0].tolist()
    if zero_groups:
        raise ValueError(
            f"Major group(s) {zero_groups} have zero total employment in the "
            "crosswalk; an employment-weighted mean is undefined there."
        )
    means = totals["mass"] / totals["employment"]

    if isinstance(soc_major_group, pd.Series):
        codes = soc_major_group
    else:
        codes = pd.Series(np.asarray(soc_major_group))
    unmatched = sorted(set(codes.unique()) - set(means.index))
    if unmatched:
        raise ValueError(
            f"soc_major_group code(s) {unmatched[:5]} are absent from the "
            f"crosswalk's {group_column!r} values {means.index.tolist()}; "
            "donor and target must use one consistent major-group coding "
            "(codes are matched, never normalized)."
        )
    out = codes.map(means)
    out.name = exposure_column
    return out


def attach_exposure(
    frame: Frame,
    values: pd.Series | np.ndarray | Sequence[float],
    column: str = DEFAULT_EXPOSURE_COLUMN,
) -> Frame:
    """Return a new frame with the exposure scores on the person table.

    Frames are immutable, so attachment reassembles: the person table gains
    ``column`` and every other table, the typed weights, the strata, and the
    mass log pass through untouched. Values are read **positionally** (a
    Series' index is ignored), matching how :func:`impute_exposure` returns
    draws aligned to the person rows.

    Args:
        frame: The target frame.
        values: One finite exposure score per person row, positionally
            aligned to the person table.
        column: The column name to attach under. Defaults to
            :data:`DEFAULT_EXPOSURE_COLUMN`.

    Returns:
        A new validated :class:`~populace.frame.Frame`.

    Raises:
        ValueError: If ``column`` already exists on any entity table (the
            stage must have one canonical producer, and silently overwriting
            scores would hide a double run), or ``values`` is not 1-D, has
            the wrong length, or contains non-finite entries.
    """
    try:
        owner = frame.column_entity(column)
    except ValueError:
        owner = None
    if owner is not None:
        raise ValueError(
            f"Column {column!r} already exists on the {owner!r} table; "
            "attach_exposure refuses to overwrite it. The exposure stage "
            "should run exactly once — pass a different column name to "
            "attach a second draw deliberately."
        )

    if isinstance(values, pd.Series):
        array = values.to_numpy(dtype=np.float64)
    else:
        array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1:
        raise ValueError(f"values must be 1-D, got shape {array.shape}.")
    person_entity = frame.schema.person_entity
    n_persons = frame.n(person_entity)
    if len(array) != n_persons:
        raise ValueError(
            f"values has {len(array)} entries but the {person_entity!r} table "
            f"has {n_persons} row(s); exposure draws must align positionally."
        )
    non_finite = int((~np.isfinite(array)).sum())
    if non_finite:
        raise ValueError(
            f"values contains {non_finite} non-finite entr(ies); exposure "
            "scores must be finite (the donor contract forbids NaN exposure)."
        )

    person = frame.person.copy()
    person[column] = array
    tables: dict[str, pd.DataFrame] = {person_entity: person}
    for entity in frame.schema.group_entities:
        tables[entity] = frame.table(entity)
    for link in frame.links:
        tables[link] = frame.link(link)
    weights = {entity: frame.weights_for(entity) for entity in frame.weighted_entities}
    return Frame(
        tables,
        frame.schema,
        weights,
        frame.strata,
        mass_log=frame.mass_log,
    )


def exposure_imputation_stage(
    fitted: FittedModel,
    *,
    column: str = DEFAULT_EXPOSURE_COLUMN,
    donor: DonorSpec = LFS_APS_EXPOSURE_DONOR,
) -> Stage:
    """Declare the exposure-imputation step of a UK build plan.

    The donor fit happens *before* plan assembly (the LFS/APS donor is a
    licensed local artifact, not a frame column, so the plan's
    consumes/produces bookkeeping cannot see it); the stage closes over the
    fitted model and, when run, draws for the frame's persons and attaches
    the scores. The stage consumes the model's predictors — on the primary
    path that includes :data:`SOC_MAJOR_GROUP_COLUMN`, so the plan executor
    refuses to run before the FRS adult.tab merge has put it on the frame.
    Any failure inside — missing predictors, a double run — aborts the
    build, per the plan executor's loudness rules.

    Args:
        fitted: The model from :func:`fit_exposure_imputer`.
        column: The produced person column. Defaults to
            :data:`DEFAULT_EXPOSURE_COLUMN`.
        donor: The donor declaration recorded on the stage. Defaults to
            :data:`LFS_APS_EXPOSURE_DONOR`.

    Returns:
        A :class:`populace.build.plan.Stage` consuming the model's predictors
        and producing ``column``.
    """

    def transform(frame: Frame) -> Frame:
        return attach_exposure(frame, impute_exposure(fitted, frame), column=column)

    return Stage(
        name=UK_EXPOSURE_IMPUTATION_STAGE_NAME,
        transform=transform,
        produces=(column,),
        consumes=tuple(fitted.predictors),
        donor=donor,
    )
