"""A small real-forest fixture for the UK SPI income imputation's forest lifetimes.

``impute_uk_spi_income_support`` fits three kinds of model: the stage-1 forests
on the SPI tape, one set of reference forests per cross-fitting fold on the FRS
rows, and the stage-2 forests. This module provides:

- :func:`spi_forest_support` and :func:`write_spi_forest_donor`, synthetic
  inputs on which every one of those models is a real ``RegimeGatedQRF``;
- :func:`run_spi_forest_scenario`, one run of the imputation on them;
- :func:`spi_result_digests`, SHA-256 digests of the person table and of every
  receipt in the result;
- :func:`tracked_spi_forest_fits`, which holds weak references to each fitted
  target and checks, when a later fit starts, that the models it replaces are
  dead.
"""

from __future__ import annotations

import dataclasses
import gc
import hashlib
import json
import weakref
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass, field, replace
from pathlib import Path
from types import SimpleNamespace
from typing import ClassVar

import numpy as np
import pandas as pd
import pytest

from microcosm.build.uk_runtime import frs_disability, spi_income
from microcosm.build.uk_runtime.frs_hmrc_source import FRS_HMRC_RETAINED_LEAF_COLUMNS
from microcosm.build.uk_runtime.spi_income import (
    SPI_DONOR_FILENAME,
    SPI_STAGE2_REVIEWED_ABSENT_OUTPUTS,
    UKSPIIncomeImputationResult,
    impute_uk_spi_income_support,
)
from microcosm.build.uk_runtime.spi_rank_transport import (
    SPI_TRANSPORT_SEED_OFFSET,
    analogue_targets,
)
from microcosm.build.uk_runtime.spi_support import (
    FRS_ONLY_SPI_FILL_PERSON_COLUMNS,
    SPI_INCOME_IMPUTATION_COLUMNS,
    UKSPISupportResult,
    create_uk_spi_support_tables,
    replace_uk_spi_support_tables,
)
from microcosm.fit import RegimeGatedQRF
from microcosm.fit.qrf import _draw_target_from_uniforms
from test_support.microcosm_build.uk_spi_income import _write_donor

SPI_FOREST_SEED = 9
SPI_FOREST_TREES = 3
_HOUSEHOLDS = 200
_HOUSEHOLDS_COPIED_TO_SPI = 40
_DONOR_ROWS = 400
_DONOR_AGE_BANDS = (2, 3, 4, 5)
_SUPPORT_SEED = 11
_DONOR_RESAMPLE_SEED = 3
_DONOR_VALUE_SEED = 5
_STATE_PENSION_AGE = 66
#: Donor columns that are codes or weights; every other column is an amount.
_DONOR_NON_AMOUNT_COLUMNS = frozenset(
    {"SEX", "FACT", "GORCODE", "AGERANGE", "MAINSRCE", "SEINC_NUM", "SPA"}
)

#: The runs the tests compare. ``in_sample`` is the stage's default gate;
#: ``cross_fitted_levels`` turns on every receipt read from the fitted models.
SPI_FOREST_SCENARIOS: dict[str, dict[str, object]] = {
    "in_sample": {},
    "cross_fitted_levels": {
        "stage1_gate_calibration": "cross_fitted",
        "stage2_gate_calibration": "cross_fitted",
        "stage1_gate_level_columns": ("spi_age_band", "has_pay", "main_source"),
        "condition_on_state_pension_receipt": True,
    },
}


def _zero_inflated_amounts(generator: np.random.Generator, position: int) -> np.ndarray:
    """One person column: zero for some people, a rounded positive amount else."""

    carrier_share = 0.45 + 0.1 * (position % 3 == 0)
    carriers = generator.random(_HOUSEHOLDS) < carrier_share
    amounts = generator.lognormal(6.0 + position % 4, 1.0, _HOUSEHOLDS)
    return np.where(carriers, np.round(amounts, 2), 0.0)


def spi_forest_support() -> UKSPISupportResult:
    """FRS people in one-person households, half of them copied to the SPI channel."""

    generator = np.random.default_rng(_SUPPORT_SEED)
    household_ids = np.arange(1, _HOUSEHOLDS + 1, dtype="int64")
    household = pd.DataFrame(
        {
            "household_id": household_ids,
            "household_weight": np.round(generator.uniform(5, 50, _HOUSEHOLDS), 3),
            "region": generator.choice(["LONDON", "WALES", "SCOTLAND"], _HOUSEHOLDS),
            "clone_index": 0,
            "household_is_capital_gains_clone": generator.random(_HOUSEHOLDS) < 0.3,
        }
    )
    columns: dict[str, object] = {
        "person_id": np.arange(1001, 1001 + _HOUSEHOLDS, dtype="int64"),
        "person_household_id": household_ids,
        "person_benunit_id": np.arange(2001, 2001 + _HOUSEHOLDS, dtype="int64"),
        "age": generator.integers(20, 85, _HOUSEHOLDS),
        "gender": generator.choice(["MALE", "FEMALE"], _HOUSEHOLDS),
        "is_uc_claimant": np.ones(_HOUSEHOLDS, dtype=bool),
        "employment_status": generator.choice(
            ["FT_EMPLOYED", "PT_EMPLOYED", "RETIRED"], _HOUSEHOLDS
        ),
        "miscellaneous_income": np.zeros(_HOUSEHOLDS, dtype=float),
    }
    amount_columns = (
        *SPI_INCOME_IMPUTATION_COLUMNS,
        *FRS_HMRC_RETAINED_LEAF_COLUMNS,
        *FRS_ONLY_SPI_FILL_PERSON_COLUMNS,
    )
    for position, column in enumerate(amount_columns, start=1):
        amounts = _zero_inflated_amounts(generator, position)
        if column not in SPI_STAGE2_REVIEWED_ABSENT_OUTPUTS:
            columns[column] = amounts
    owners = generator.random(_HOUSEHOLDS) < 0.3
    trades = owners & (generator.random(_HOUSEHOLDS) < 0.5)
    lets_property = owners & (generator.random(_HOUSEHOLDS) < 0.5)
    for column, values in {
        "stands_as_sole_owner_or_partner_of_company": owners,
        "is_engaged_in_owned_company_trade": owners,
        "owned_company_is_main_employment": owners,
        "owned_company_carries_on_trade": trades,
        "owned_company_carries_on_property_business": lets_property,
        "owned_company_income_share": np.where(owners, 0.5, 0.0),
    }.items():
        if column in columns:
            columns[column] = values
    columns["individual_savings_account_interest_income"] = _zero_inflated_amounts(
        generator, 2
    )
    person = pd.DataFrame(columns)
    benunit = pd.DataFrame({"benunit_id": person["person_benunit_id"].to_numpy()})
    dead_channel = create_uk_spi_support_tables(
        person=person,
        benunit=benunit,
        household=household,
        selected_household_ids=tuple(household_ids[:_HOUSEHOLDS_COPIED_TO_SPI]),
        source_year=2023,
    )
    support = replace_uk_spi_support_tables(
        person=dead_channel.person,
        benunit=dead_channel.benunit,
        household=dead_channel.household,
        seed=7,
        source_year=2023,
    )
    return replace(support, household=support.household.drop(columns="clone_index"))


def write_spi_forest_donor(directory: Path) -> Path:
    """Write the tape: the four-record test donor resampled and rescaled.

    Every amount of a record moves by that record's factor, so the tape's
    leaf reconciliation and its main-source codes keep holding. Age bands are
    dealt independently of the amounts, so each band holds every sign class
    and the stage-1 gates have levels to offset.
    """

    path = directory / SPI_DONOR_FILENAME
    _write_donor(path)
    records = pd.read_csv(path, sep="\t")
    donor = records.sample(
        n=_DONOR_ROWS, replace=True, random_state=_DONOR_RESAMPLE_SEED
    ).reset_index(drop=True)
    amounts = [
        column for column in donor.columns if column not in _DONOR_NON_AMOUNT_COLUMNS
    ]
    generator = np.random.default_rng(_DONOR_VALUE_SEED)
    factors = generator.uniform(0.5, 3.0, len(donor))
    donor[amounts] = donor[amounts].mul(factors, axis=0)
    donor["AGERANGE"] = generator.choice(_DONOR_AGE_BANDS, len(donor))
    donor.to_csv(path, sep="\t", index=False)
    return path


def use_fixture_disability_rates(monkeypatch: pytest.MonkeyPatch) -> None:
    """Give the disability refresh fixed rates, so no country engine is needed."""

    category_rates = frs_disability.UKDWPDisabilityCategoryRates(
        aa_lower=10,
        aa_higher=20,
        dla_sc_lower=10,
        dla_sc_middle=20,
        dla_sc_higher=30,
        dla_m_lower=10,
        dla_m_higher=20,
        pip_m_standard=10,
        pip_m_enhanced=20,
        pip_dl_standard=10,
        pip_dl_enhanced=20,
        instant="2024-01-01",
        source="fixture",
    )
    flag_rates = frs_disability.UKDWPDisabilityFlagRates(
        aa_higher=20,
        dla_sc_higher=30,
        pip_dl_enhanced=20,
        instant="2024-01-01",
        source="fixture",
    )
    monkeypatch.setattr(
        frs_disability,
        "uk_dwp_disability_category_rates",
        lambda year: category_rates,
    )
    monkeypatch.setattr(
        frs_disability, "uk_dwp_disability_flag_rates", lambda year: flag_rates
    )


def run_spi_forest_scenario(
    scenario: str,
    directory: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    qrf_class: type[RegimeGatedQRF] = RegimeGatedQRF,
) -> UKSPIIncomeImputationResult:
    """Run the imputation on the fixture with real forests."""

    use_fixture_disability_rates(monkeypatch)
    monkeypatch.setattr(spi_income, "QRF", qrf_class)
    monkeypatch.setattr(spi_income, "_verify_spi_donor_identity", lambda path: None)
    return impute_uk_spi_income_support(
        spi_forest_support(),
        write_spi_forest_donor(directory),
        seed=SPI_FOREST_SEED,
        n_estimators=SPI_FOREST_TREES,
        donor_sample_size=None,
        state_pension_age=_STATE_PENSION_AGE,
        allow_in_sample_reference=True,
        **SPI_FOREST_SCENARIOS[scenario],
    )


class FitEveryTargetFirstQRF(RegimeGatedQRF):
    """The real QRF with the lifetimes and order the stage used to have.

    Every fitted model stays referenced in :attr:`kept_fits`. ``start_chain``
    fits all targets at once, as ``fit`` does, and each ``fit_draw_next`` only
    draws its target from that fit, through the function
    ``predict_from_uniforms`` draws each target with. A run with this class
    computes what the stage computed when it fitted stage 2 whole and held
    every model until it returned.
    """

    kept_fits: ClassVar[list[object]] = []

    def fit(self, *args: object, **kwargs: object) -> object:
        fitted = super().fit(*args, **kwargs)
        self.kept_fits.append(fitted)
        return fitted

    def start_chain(
        self,
        frame_or_df: object,
        predictors: list[str],
        targets: list[str],
        *,
        weights: object,
        groups: object = None,
    ) -> object:
        self._every_target_fit = self.fit(
            frame_or_df, predictors, targets, weights=weights, groups=groups
        )
        return super().start_chain(
            frame_or_df, predictors, targets, weights=weights, groups=groups
        )

    def fit_draw_next(
        self,
        frame_or_df: object,
        recipient_predictors: pd.DataFrame,
        raw_prior_draws: pd.DataFrame,
        *,
        state: object,
        weights: object,
        quantiles: np.ndarray,
        sign_uniforms: np.ndarray,
        groups: object = None,
    ) -> SimpleNamespace:
        target = state.next_target
        fitted = self._every_target_fit
        features = recipient_predictors.loc[:, list(state.predictors)].copy()
        for prior in state.completed_targets:
            features[prior] = raw_prior_draws[prior].to_numpy(dtype=np.float64)
        target_outcome = fitted.gate_outcomes()[target]
        return SimpleNamespace(
            target=target,
            raw_draw=_draw_target_from_uniforms(
                features, fitted._target_models[target], quantiles, sign_uniforms
            ),
            state=replace(state, completed_targets=(*state.completed_targets, target)),
            regime=fitted.regimes()[target],
            weight_kind=fitted.weight_kind,
            fitted=SimpleNamespace(gate_outcomes=lambda: {target: target_outcome}),
        )


# ---------------------------------------------------------------------------
# Exact digests of a result
# ---------------------------------------------------------------------------


def _update_with_values(digest, label: object, values: pd.Index | pd.Series) -> None:
    digest.update(repr(label).encode("utf-8"))
    digest.update(str(values.dtype).encode("utf-8"))
    if isinstance(values.dtype, np.dtype) and values.dtype.kind in "biuf":
        digest.update(np.ascontiguousarray(values.to_numpy()).tobytes())
    else:
        digest.update(repr(list(values)).encode("utf-8"))


def table_digest(table: pd.DataFrame) -> str:
    """SHA-256 of a table's index and columns: names, dtypes and exact values.

    Boolean and numeric columns contribute their raw bytes, so a float that
    moves in its last bit, or a ``-0.0`` for a ``0.0``, changes the digest.
    Every other column contributes the ``repr`` of its values.
    """

    digest = hashlib.sha256()
    _update_with_values(digest, ("index", tuple(table.index.names)), table.index)
    for column in table.columns:
        _update_with_values(digest, column, table[column])
    return digest.hexdigest()


def _exact_receipt_value(value: object) -> object:
    """A JSON form of a receipt value that keeps types, order and every float bit."""

    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return {
            "dataclass": type(value).__name__,
            "fields": [
                [item.name, _exact_receipt_value(getattr(value, item.name))]
                for item in dataclasses.fields(value)
            ],
        }
    if isinstance(value, Mapping):
        return {
            "mapping": [
                [_exact_receipt_value(key), _exact_receipt_value(item)]
                for key, item in value.items()
            ]
        }
    if isinstance(value, list | tuple):
        return {
            type(value).__name__: [_exact_receipt_value(item) for item in value],
        }
    if isinstance(value, np.ndarray):
        return {
            "array": str(value.dtype),
            "shape": list(value.shape),
            "bytes": np.ascontiguousarray(value).tobytes().hex(),
        }
    if isinstance(value, np.generic):
        return {"numpy": str(value.dtype), "bytes": value.tobytes().hex()}
    if isinstance(value, float):
        return {"float": value.hex()}
    if value is None or isinstance(value, bool | int | str):
        return value
    return {"repr": type(value).__name__, "value": repr(value)}


def receipt_digest(value: object) -> str:
    """SHA-256 of one receipt, sensitive to key order, types and float bits."""

    text = json.dumps(_exact_receipt_value(value), separators=(",", ":"))
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def spi_result_digests(result: UKSPIIncomeImputationResult) -> dict[str, str]:
    """A digest of the person table and of every other field of the result.

    ``donor_path`` is left out: it names the test's temporary directory.
    """

    digests = {"person": table_digest(result.person)}
    for item in dataclasses.fields(result):
        if item.name not in {"person", "donor_path"}:
            digests[item.name] = receipt_digest(getattr(result, item.name))
    return digests


def spi_frozen_digests(result: UKSPIIncomeImputationResult) -> dict[str, str]:
    """The person table's digest and one digest over every receipt's."""

    digests = spi_result_digests(result)
    person = digests.pop("person")
    text = json.dumps(digests, separators=(",", ":"))
    return {
        "person": person,
        "receipts": hashlib.sha256(text.encode("utf-8")).hexdigest(),
    }


# ---------------------------------------------------------------------------
# Lifetimes of the fitted forests
# ---------------------------------------------------------------------------

STAGE1_FIT = "stage 1"
REFERENCE_FIT = "reference"
STAGE2_FIT = "stage 2"


def _classifier_parts(
    label: str, classifier: object
) -> list[tuple[str, object | None]]:
    """A gradient-boosted sign classifier, its binning, its trees and their nodes."""

    parts: list[tuple[str, object | None]] = [
        (label, classifier),
        (f"{label} bins", getattr(classifier, "_bin_mapper", None)),
    ]
    for boosting_round in getattr(classifier, "_predictors", None) or ():
        for tree in boosting_round:
            parts.append((f"{label} tree", tree))
            parts.append((f"{label} tree nodes", getattr(tree, "nodes", None)))
    return parts


def fitted_target_parts(model: object) -> list[tuple[str, object]]:
    """The weakly referenceable objects one fitted target owns, with their kinds.

    A cross-fitted sign gate contributes each fold's classifier. Each forest
    contributes its regressor, its trees and the numpy array of every leaf's
    training-row positions (the large, zero-padded allocation).
    """

    gate = getattr(model, "gate", None)
    parts: list[tuple[str, object | None]] = [
        ("target model", model),
        *_classifier_parts("sign classifier", gate),
    ]
    for fold_classifier in getattr(gate, "folds", None) or ():
        parts.extend(_classifier_parts("fold sign classifier", fold_classifier))
    for sign in ("positive", "negative"):
        forest = getattr(model, sign, None)
        regressor = getattr(forest, "model", None)
        parts.append((f"{sign} forest", forest))
        parts.append((f"{sign} regressor", regressor))
        parts.extend(
            (f"{sign} tree", tree)
            for tree in getattr(regressor, "estimators_", None) or ()
        )
        leaves = getattr(getattr(regressor, "forest_", None), "y_train_leaves", None)
        parts.append((f"{sign} leaf array", getattr(leaves, "base", None)))
    return [(kind, part) for kind, part in parts if part is not None]


@dataclass
class SPIForestLifetimes:
    """Weak references to what each ``RegimeGatedQRF._fit_target`` call built."""

    seed: int
    stop_at_first_violation: bool = True
    fitted: list[tuple[str, int, str]] = field(default_factory=list)
    violations: list[str] = field(default_factory=list)
    _references: list[tuple[str, int, str, str, weakref.ref]] = field(
        default_factory=list
    )
    _reference_fold: int = -1

    def fit_of(self, model_seed: int) -> str:
        """Which of the stage's models a fit belongs to, from its seed."""

        return {
            self.seed: STAGE1_FIT,
            self.seed + SPI_TRANSPORT_SEED_OFFSET: REFERENCE_FIT,
            self.seed + 1: STAGE2_FIT,
        }[model_seed]

    def fold_of(self, fit: str, target: str) -> int:
        """The cross-fitting fold of a reference fit; zero for the other fits."""

        if fit != REFERENCE_FIT:
            return 0
        if target == analogue_targets()[0]:
            self._reference_fold += 1
        return self._reference_fold

    def record(self, fit: str, fold: int, target: str, model: object) -> None:
        self.fitted.append((fit, fold, target))
        for kind, part in fitted_target_parts(model):
            self._references.append((fit, fold, target, kind, weakref.ref(part)))

    def alive(self, fit: str, *, before_fold: int | None = None) -> list[tuple]:
        """What is still alive of ``fit``, optionally of its earlier folds only."""

        return [
            (fit, fold, target, kind)
            for recorded, fold, target, kind, reference in self._references
            if recorded == fit
            and reference() is not None
            and (before_fold is None or fold < before_fold)
        ]

    def require_dead(self, alive: list[tuple], *, before: str) -> None:
        if alive:
            message = (
                f"{before} started while {len(alive)} objects were alive, "
                f"among them {alive[:5]}"
            )
            self.violations.append(message)
            if self.stop_at_first_violation:
                raise AssertionError(message)

    def check_before_fit(self, fit: str, fold: int, target: str) -> None:
        """Require the models this fit replaces to be dead.

        A stage-2 fit needs the stage-1 and reference models and every earlier
        stage-2 target dead. A reference fit needs the earlier folds dead; the
        stage-1 model stays, because it draws after the last fold.
        """

        before = f"the {fit} fit of {target!r}"
        if fit == STAGE2_FIT:
            self.require_dead(
                [
                    *self.alive(STAGE1_FIT),
                    *self.alive(REFERENCE_FIT),
                    *self.alive(STAGE2_FIT),
                ],
                before=before,
            )
        elif fit == REFERENCE_FIT:
            self.require_dead(
                self.alive(REFERENCE_FIT, before_fold=fold), before=before
            )


@contextmanager
def tracked_spi_forest_fits(
    monkeypatch: pytest.MonkeyPatch,
    *,
    seed: int = SPI_FOREST_SEED,
    stop_at_first_violation: bool = True,
) -> Iterator[SPIForestLifetimes]:
    """Check, as each target's fit starts, that the models it replaces are dead.

    The garbage collector is off inside the block, so an object counts as freed
    only when nothing references it, which is when its memory goes back; a
    reference cycle would keep it alive and fail the check.

    scikit-learn's compiled ``Tree`` node storage and quantile-forest's
    ``QuantileForest`` cannot be weakly referenced. Each belongs to a tracked
    object and is freed with it, unless something references it directly,
    which this check cannot see.

    A violation raises at once, unless ``stop_at_first_violation`` is false:
    then the run goes on and :attr:`SPIForestLifetimes.violations` lists them.
    """

    lifetimes = SPIForestLifetimes(
        seed=seed, stop_at_first_violation=stop_at_first_violation
    )
    original = RegimeGatedQRF._fit_target

    def fit_target(self: RegimeGatedQRF, *, target: str, **kwargs: object) -> object:
        fit = lifetimes.fit_of(self.seed)
        fold = lifetimes.fold_of(fit, target)
        lifetimes.check_before_fit(fit, fold, target)
        model = original(self, target=target, **kwargs)
        lifetimes.record(fit, fold, target, model)
        return model

    monkeypatch.setattr(RegimeGatedQRF, "_fit_target", fit_target)
    gc.collect()
    collector_was_enabled = gc.isenabled()
    gc.disable()
    try:
        yield lifetimes
    finally:
        if collector_was_enabled:
            gc.enable()
        monkeypatch.setattr(RegimeGatedQRF, "_fit_target", original)


__all__ = [name for name in globals() if not name.startswith("__")]
