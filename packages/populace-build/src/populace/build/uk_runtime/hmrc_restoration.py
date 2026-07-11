"""End-to-end HMRC/SPI restoration stage for the UK national build."""

from __future__ import annotations

import hashlib
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from populace.build.uk_runtime.hmrc_calibration import (
    DEFAULT_HMRC_CALIBRATION_EPOCHS,
    DEFAULT_HMRC_CALIBRATION_LEARNING_RATE,
    DEFAULT_HMRC_MAX_ABS_RELATIVE_ERROR,
    DEFAULT_HMRC_MAX_WEIGHT_RATIO,
    UKHMRCIncomeCalibration,
    UKHMRCTargetMaterialization,
    calibrate_uk_hmrc_income,
    materialize_uk_hmrc_calibration_frame,
)
from populace.build.uk_runtime.hmrc_income import (
    HMRCIncomeTargetSet,
    materialize_hmrc_spi_income_band_targets,
)
from populace.build.uk_runtime.national_build import (
    UKNationalDataset,
    validate_uk_national_dataset,
)
from populace.build.uk_runtime.release_input_coverage import (
    DEFAULT_MINIMUM_NONDEFAULT_MASS_SHARE,
)
from populace.build.uk_runtime.spi_income import (
    DEFAULT_SPI_DONOR_SAMPLE_SIZE,
    UKSPIIncomeImputationResult,
    impute_uk_spi_income_support,
)
from populace.build.uk_runtime.spi_support import (
    DEFAULT_SPI_PRIOR_MASS_SHARE,
    UKSPISupportResult,
    replace_uk_spi_support_tables,
)
from populace.frame import WeightKind

__all__ = [
    "CERTIFIED_UK_CANDIDATE_FILENAME",
    "CERTIFIED_UK_CANDIDATE_REVISION",
    "CERTIFIED_UK_CANDIDATE_SHA256",
    "CERTIFIED_UK_CANDIDATE_SIZE_BYTES",
    "HMRC_DISTRIBUTIONAL_INPUTS",
    "UKCertifiedCandidateIdentity",
    "UKHMRCIncomeRestorationResult",
    "UKHMRCIncomeStageTransform",
    "restore_uk_hmrc_income_family",
    "verify_certified_uk_candidate",
]

CERTIFIED_UK_CANDIDATE_FILENAME = "populace_uk_2023.h5"
CERTIFIED_UK_CANDIDATE_REVISION = (
    "populace-uk-2023-dd68c73-4aa4b14-20260619T023711Z"
)
CERTIFIED_UK_CANDIDATE_SHA256 = (
    "f17306ccb2aad7ff0130be3589b560afb2e2a12a943570911cd0c77f07934833"
)
CERTIFIED_UK_CANDIDATE_SIZE_BYTES = 1_315_880_118
HMRC_DISTRIBUTIONAL_INPUTS = (
    "gift_aid",
    "charitable_investment_gifts",
)


@dataclass(frozen=True)
class UKCertifiedCandidateIdentity:
    """Verified identity of the only accepted HMRC restoration base."""

    path: Path
    filename: str
    revision: str
    sha256: str
    size_bytes: int


@dataclass(frozen=True)
class UKHMRCIncomeRestorationResult:
    """Restored/calibrated national dataset plus complete stage evidence."""

    dataset: UKNationalDataset
    support: UKSPISupportResult
    imputation: UKSPIIncomeImputationResult
    source_targets: HMRCIncomeTargetSet
    materialization: UKHMRCTargetMaterialization
    calibration: UKHMRCIncomeCalibration
    distributional_mass_shares: dict[str, float]

    def evidence(self) -> dict[str, object]:
        """Return JSON-safe release provenance for the national driver."""

        calibration_result = self.calibration.result
        return {
            "stage": "hmrc_spi_income",
            "source_vintages": {
                "spi_donor": "2022-23",
                "hmrc_surface": self.source_targets.source.source_vintage,
                "mapped_build_period": self.source_targets.source.build_period,
            },
            "sources": {
                "spi_donor": {
                    "path": str(self.imputation.donor_path),
                    "sha256": self.imputation.donor_sha256,
                    "rows_used": self.imputation.donor_rows,
                },
                "hmrc_surface": {
                    "path": str(self.source_targets.source.local_path),
                    "sha256": self.source_targets.source.sha256,
                    "publication_url": self.source_targets.source.publication_url,
                    "ods_url": self.source_targets.source.ods_url,
                    "tables": list(self.source_targets.source.table_names),
                },
            },
            "spi_prior": {
                "replaced_households": self.support.replaced_spi_households,
                "mass_share": self.support.spi_prior_mass_share,
                "weight_kind": self.support.household_weight_kind.value,
                "mass_change_reason": self.support.mass_log[-1].reason,
            },
            "qrf_fits": [
                {
                    "fit_name": record.fit_name,
                    "weight_kind": record.weight_kind,
                }
                for record in self.imputation.fit_weight_records
            ],
            "reviewed_absent_stage2_outputs": dict(
                self.imputation.reviewed_absent_stage2_outputs
            ),
            "targets": {
                "count": len(self.source_targets.targets),
                "registry_version": self.calibration.registry.version,
                "taxpayer_rows": self.materialization.taxpayer_rows,
                "minimum_positive_support_rows": (
                    self.materialization.minimum_positive_support_rows
                ),
            },
            "calibration": {
                "input_weight_kind": "importance",
                "output_weight_kind": "calibrated",
                "mass": "conserve",
                "initial_total": float(calibration_result.initial_weights.sum()),
                "final_total": float(calibration_result.weights.sum()),
                "initial_loss": calibration_result.initial_loss,
                "final_loss": calibration_result.final_loss,
                "worst_target": self.calibration.worst_target,
                "maximum_abs_relative_error": (
                    self.calibration.maximum_abs_relative_error
                ),
                "options": dict(calibration_result.options),
            },
            "effective_mass_coverage": {
                "minimum_nondefault_mass_share": (
                    DEFAULT_MINIMUM_NONDEFAULT_MASS_SHARE
                ),
                "columns": dict(self.distributional_mass_shares),
            },
        }


@dataclass
class UKHMRCIncomeStageTransform:
    """Callable national-stage adapter retaining the last run's evidence."""

    spi_tab_path: Path
    hmrc_ods_path: Path
    seed: int = 42
    qrf_estimators: int = 100
    donor_sample_size: int | None = DEFAULT_SPI_DONOR_SAMPLE_SIZE
    spi_prior_mass_share: float = DEFAULT_SPI_PRIOR_MASS_SHARE
    calibration_epochs: int = DEFAULT_HMRC_CALIBRATION_EPOCHS
    calibration_learning_rate: float = DEFAULT_HMRC_CALIBRATION_LEARNING_RATE
    max_weight_ratio: float = DEFAULT_HMRC_MAX_WEIGHT_RATIO
    maximum_abs_relative_error: float = DEFAULT_HMRC_MAX_ABS_RELATIVE_ERROR
    simulation_factory: Callable[[Any], Any] | None = None
    last_result: UKHMRCIncomeRestorationResult | None = field(
        default=None,
        init=False,
    )

    def __call__(self, dataset: UKNationalDataset) -> UKNationalDataset:
        self.last_result = restore_uk_hmrc_income_family(
            dataset,
            spi_tab_path=self.spi_tab_path,
            hmrc_ods_path=self.hmrc_ods_path,
            seed=self.seed,
            qrf_estimators=self.qrf_estimators,
            donor_sample_size=self.donor_sample_size,
            spi_prior_mass_share=self.spi_prior_mass_share,
            calibration_epochs=self.calibration_epochs,
            calibration_learning_rate=self.calibration_learning_rate,
            max_weight_ratio=self.max_weight_ratio,
            maximum_abs_relative_error=self.maximum_abs_relative_error,
            simulation_factory=self.simulation_factory,
        )
        return self.last_result.dataset


def verify_certified_uk_candidate(path: str | Path) -> UKCertifiedCandidateIdentity:
    """Hash/size gate the certified Populace UK candidate before stages run."""

    candidate = Path(path).expanduser().resolve()
    if not candidate.is_file():
        raise FileNotFoundError(f"Certified Populace UK candidate not found: {candidate}.")
    size = candidate.stat().st_size
    if size != CERTIFIED_UK_CANDIDATE_SIZE_BYTES:
        raise ValueError(
            f"{candidate}: expected certified candidate size "
            f"{CERTIFIED_UK_CANDIDATE_SIZE_BYTES}, got {size}."
        )
    digest = _sha256(candidate)
    if digest != CERTIFIED_UK_CANDIDATE_SHA256:
        raise ValueError(
            f"{candidate}: sha256 {digest} does not match certified candidate "
            f"{CERTIFIED_UK_CANDIDATE_SHA256}."
        )
    return UKCertifiedCandidateIdentity(
        path=candidate,
        filename=CERTIFIED_UK_CANDIDATE_FILENAME,
        revision=CERTIFIED_UK_CANDIDATE_REVISION,
        sha256=digest,
        size_bytes=size,
    )


def restore_uk_hmrc_income_family(
    dataset: UKNationalDataset,
    *,
    spi_tab_path: str | Path,
    hmrc_ods_path: str | Path,
    seed: int = 42,
    qrf_estimators: int = 100,
    donor_sample_size: int | None = DEFAULT_SPI_DONOR_SAMPLE_SIZE,
    spi_prior_mass_share: float = DEFAULT_SPI_PRIOR_MASS_SHARE,
    calibration_epochs: int = DEFAULT_HMRC_CALIBRATION_EPOCHS,
    calibration_learning_rate: float = DEFAULT_HMRC_CALIBRATION_LEARNING_RATE,
    max_weight_ratio: float = DEFAULT_HMRC_MAX_WEIGHT_RATIO,
    maximum_abs_relative_error: float = DEFAULT_HMRC_MAX_ABS_RELATIVE_ERROR,
    simulation_factory: Callable[[Any], Any] | None = None,
) -> UKHMRCIncomeRestorationResult:
    """Replace dead SPI rows, run both QRFs, and fit all HMRC targets."""

    validate_uk_national_dataset(dataset)
    if not np.isclose(
        spi_prior_mass_share,
        DEFAULT_SPI_PRIOR_MASS_SHARE,
        rtol=0.0,
        atol=0.0,
    ):
        raise ValueError(
            "The HMRC source manifest reviews exactly a 50% SPI prior mass "
            "share; update/review the manifest before changing it."
        )
    source_targets = materialize_hmrc_spi_income_band_targets(
        hmrc_ods_path,
        build_period=dataset.time_period,
    )
    support = replace_uk_spi_support_tables(
        person=dataset.person,
        benunit=dataset.benunit,
        household=dataset.household,
        seed=seed,
        source_year=int(dataset.time_period),
        spi_prior_mass_share=spi_prior_mass_share,
        input_weight_kind=dataset.household_weight_kind,
        mass_log=dataset.mass_log,
    )
    imputation = impute_uk_spi_income_support(
        support,
        spi_tab_path,
        seed=seed,
        n_estimators=qrf_estimators,
        donor_sample_size=donor_sample_size,
        build_period=dataset.time_period,
    )
    imputed_dataset = dataset.with_tables(
        person=imputation.person,
        benunit=support.benunit,
        household=support.household,
        household_weight_kind=WeightKind.IMPORTANCE,
        mass_log=support.mass_log,
    )
    validate_uk_national_dataset(imputed_dataset)
    materialization = materialize_uk_hmrc_calibration_frame(
        imputed_dataset,
        source_targets,
        simulation_factory=simulation_factory,
    )
    calibration = calibrate_uk_hmrc_income(
        materialization,
        epochs=calibration_epochs,
        learning_rate=calibration_learning_rate,
        max_weight_ratio=max_weight_ratio,
        maximum_abs_relative_error=maximum_abs_relative_error,
        seed=seed,
    )
    calibrated_ids = calibration.result.frame.table("household")["household_id"]
    if not np.array_equal(
        calibrated_ids.to_numpy(),
        imputed_dataset.household["household_id"].to_numpy(),
    ):
        raise RuntimeError("HMRC calibrated weights lost household ID alignment.")
    household = imputed_dataset.household.copy()
    household["household_weight"] = calibration.result.weights
    calibrated_dataset = imputed_dataset.with_tables(
        household=household,
        household_weight_kind=WeightKind.CALIBRATED,
        mass_log=calibration.result.frame.mass_log,
    )
    validate_uk_national_dataset(calibrated_dataset)
    distributional_mass_shares = _distributional_mass_shares(calibrated_dataset)
    insufficient = {
        name: share
        for name, share in distributional_mass_shares.items()
        if share < DEFAULT_MINIMUM_NONDEFAULT_MASS_SHARE
    }
    if insufficient:
        raise RuntimeError(
            "Rebuilt SPI channel did not restore required effective-mass "
            f"coverage: {insufficient}."
        )
    return UKHMRCIncomeRestorationResult(
        dataset=calibrated_dataset,
        support=support,
        imputation=imputation,
        source_targets=source_targets,
        materialization=materialization,
        calibration=calibration,
        distributional_mass_shares=distributional_mass_shares,
    )


def _distributional_mass_shares(dataset: UKNationalDataset) -> dict[str, float]:
    person = dataset.person
    household_weights = dataset.household.set_index("household_id")[
        "household_weight"
    ]
    mapped = person["person_household_id"].map(household_weights)
    if mapped.isna().any() or not mapped.gt(0.0).all():
        raise RuntimeError(
            "Cannot audit HMRC distributional inputs without positive person mass."
        )
    weights = mapped.to_numpy(dtype=float)
    total = float(weights.sum())
    shares: dict[str, float] = {}
    for column in HMRC_DISTRIBUTIONAL_INPUTS:
        if column not in person:
            raise RuntimeError(f"HMRC stage omitted distributional input {column!r}.")
        values = pd.to_numeric(person[column], errors="coerce").to_numpy(
            dtype=float,
            na_value=np.nan,
        )
        if not np.isfinite(values).all():
            raise RuntimeError(f"HMRC distributional input {column!r} is non-finite.")
        shares[column] = float(weights[values != 0.0].sum()) / total
    return shares


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
