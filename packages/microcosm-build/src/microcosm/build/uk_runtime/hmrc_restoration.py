"""Guarded real-donor HMRC replay for the UK national build.

The current FRS instrument cannot materialize the complete HMRC Total Income
measure used to assign the 13 published bands.  The national stage therefore
does the source-faithful work that remains admissible -- retain the adjudicated
FRS leaves, rebuild one positive-mass SPI channel, run both weighted QRFs, and
derive the SPI accounting identities exactly -- but it does not calibrate to
non-comparable band facts.  All 208 published facts are carried into an
aggregate replay report with the reviewed source fences.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from microcosm.build.gates import FitWeightRecord
from microcosm.build.uk_runtime.content_identity import uk_frame_content_identity
from microcosm.build.uk_runtime.frs_hmrc_leaves import (
    UKFRSHMRCRetainedLeavesStageTransform,
)
from microcosm.build.uk_runtime.hmrc_income import (
    HMRCIncomeTargetSet,
    materialize_hmrc_spi_income_band_targets,
    verify_hmrc_spi_collated_ods,
)
from microcosm.build.uk_runtime.hmrc_replay import (
    HMRCReplayReport,
    build_conservative_hmrc_replay_report,
)
from microcosm.build.uk_runtime.hmrc_source_contract import (
    HMRC_DISTRIBUTIONAL_INPUTS,
    assert_uk_hmrc_income_source_contract_current,
)
from microcosm.build.uk_runtime.national_frame import (
    UKStagingProvenance,
    _uk_source_file_fingerprint,
    _UKSourceFileFingerprint,
    uk_household_weight_kind,
    uk_national_frame,
    uk_time_period,
    validate_uk_national_frame,
)
from microcosm.build.uk_runtime.release_identity import UK_RELEASE_TIER_FRS
from microcosm.build.uk_runtime.release_input_coverage import (
    DEFAULT_MINIMUM_NONDEFAULT_MASS_SHARE,
)
from microcosm.build.uk_runtime.spi_income import (
    DEFAULT_SPI_DONOR_SAMPLE_SIZE,
    SPI_SOURCE_TI_FORMULA,
    UKSPIIncomeImputationResult,
    assert_frs_hmrc_auxiliary_crosswalk_available,
    impute_uk_spi_income_support,
    verify_spi_donor_identity,
)
from microcosm.build.uk_runtime.spi_support import (
    DEFAULT_SPI_PRIOR_MASS_SHARE,
    SPI_HMRC_TOTAL_EARNED_INCOME_COLUMN,
    SPI_HMRC_TOTAL_INVESTMENT_INCOME_COLUMN,
    SPI_SYNTHETIC_SUPPORT_CHANNEL,
    UKSPISupportResult,
    replace_uk_spi_support_tables,
    support_channel_column,
)
from microcosm.frame import Frame, WeightKind

__all__ = [
    "CERTIFIED_UK_CANDIDATE_FILENAME",
    "CERTIFIED_UK_CANDIDATE_REVISION",
    "CERTIFIED_UK_CANDIDATE_SHA256",
    "CERTIFIED_UK_CANDIDATE_SIZE_BYTES",
    "CERTIFIED_UK_CANDIDATE_TIER",
    "HMRC_DISTRIBUTIONAL_INPUTS",
    "UKCertifiedCandidateIdentity",
    "UKHMRCIncomeRestorationResult",
    "UKHMRCIncomeStageTransform",
    "assert_uk_hmrc_income_source_contract_current",
    "restore_uk_hmrc_income_family",
    "verify_certified_uk_candidate",
]

CERTIFIED_UK_CANDIDATE_FILENAME = "populace_uk_2023.h5"
CERTIFIED_UK_CANDIDATE_REVISION = "populace-uk-2023-dd68c73-4aa4b14-20260619T023711Z"
CERTIFIED_UK_CANDIDATE_TIER = UK_RELEASE_TIER_FRS
CERTIFIED_UK_CANDIDATE_SHA256 = (
    "f17306ccb2aad7ff0130be3589b560afb2e2a12a943570911cd0c77f07934833"
)
CERTIFIED_UK_CANDIDATE_SIZE_BYTES = 1_315_880_118
_CERTIFIED_CANDIDATE_VERIFICATION_TOKEN = object()


@dataclass(frozen=True)
class UKCertifiedCandidateIdentity:
    """Verified identity of the only accepted HMRC replay base."""

    path: Path
    filename: str
    tier: str
    revision: str
    sha256: str
    size_bytes: int
    _verification_token: object | None = field(
        default=None,
        init=False,
        repr=False,
        compare=False,
    )
    _source_file_fingerprint: _UKSourceFileFingerprint | None = field(
        default=None,
        init=False,
        repr=False,
        compare=False,
    )


@dataclass(frozen=True)
class UKHMRCIncomeRestorationResult:
    """Importance-weight replay frame plus aggregate-only evidence."""

    frame: Frame
    support: UKSPISupportResult
    imputation: UKSPIIncomeImputationResult
    source_targets: HMRCIncomeTargetSet
    replay_report: HMRCReplayReport
    distributional_mass_shares: Mapping[str, float]
    post_draw_identity_rows: int

    def evidence(self) -> dict[str, object]:
        """Return JSON-safe aggregate evidence for the national driver."""

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
                    "size_bytes": self.imputation.donor_size_bytes,
                    "rows_used": self.imputation.donor_rows,
                },
                "hmrc_surface": {
                    "path": str(self.source_targets.source.local_path),
                    "sha256": self.source_targets.source.sha256,
                    "size_bytes": self.source_targets.source.size_bytes,
                    "mime_type": self.source_targets.source.mime_type,
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
            "post_draw_identity": {
                "formula": SPI_SOURCE_TI_FORMULA,
                "rows_checked": self.post_draw_identity_rows,
                "exact": True,
            },
            "targets": {
                "count": len(self.source_targets.targets),
                "classification": dict(self.replay_report.summary),
            },
            "calibration": {
                "performed": False,
                "reason": (
                    "Complete FRS Total Income band assignment is unavailable; "
                    "the 208 facts are reviewed exclusions rather than biased "
                    "calibration constraints."
                ),
                "output_weight_kind": uk_household_weight_kind(self.frame).value,
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
    """Callable national-stage adapter retaining the last replay evidence."""

    spi_tab_path: Path
    hmrc_ods_path: Path
    certified_candidate: UKCertifiedCandidateIdentity
    retained_leaves_transform: UKFRSHMRCRetainedLeavesStageTransform | None = None
    seed: int = 42
    qrf_estimators: int = 100
    donor_sample_size: int | None = DEFAULT_SPI_DONOR_SAMPLE_SIZE
    spi_prior_mass_share: float = DEFAULT_SPI_PRIOR_MASS_SHARE
    last_result: UKHMRCIncomeRestorationResult | None = field(
        default=None,
        init=False,
    )
    staging_provenance: UKStagingProvenance | None = field(
        default=None,
        init=False,
    )
    bound_input_identity: str | None = field(
        default=None,
        init=False,
        repr=False,
    )

    @property
    def fit_weight_records(self) -> tuple[FitWeightRecord, ...]:
        """Return immutable fit-weight evidence from the most recent run."""

        if self.last_result is None:
            return ()
        return tuple(self.last_result.imputation.fit_weight_records)

    def checkpoint_metadata(self) -> dict[str, object]:
        """JSON-safe evidence the stage checkpoint carries for a resume.

        Everything a later process consumes without re-running the stage:
        the fit-weight records the terminal weights audit reads, and the
        aggregate family evidence and replay-report payload the national
        driver writes as stage sidecars (the adversarial-review blocker: a
        resumed run must still produce byte-identical evidence reports).
        """

        if self.last_result is None:
            raise RuntimeError(
                "checkpoint metadata requires a completed SPI restoration run."
            )
        return {
            "fit_weight_records": [
                {"fit_name": record.fit_name, "weight_kind": record.weight_kind}
                for record in self.last_result.imputation.fit_weight_records
            ],
            "evidence": self.last_result.evidence(),
            "replay_payload": self.last_result.replay_report.to_payload(),
            "output_content_identity": uk_frame_content_identity(
                self.last_result.frame
            ),
        }

    def resume_from_checkpoint(
        self,
        metadata: Mapping[str, object],
        frame: Frame,
    ) -> None:
        """Rehydrate a completed run's evidence surface from its record.

        ``frame`` is the stage's checkpointed output; the recorded output
        identity must match its content — the same drift check the
        retained-leaves stage runs, and this is the terminal stage, whose
        frame feeds the batched gates and the staging writer. The runtime's
        checkpoint sha covers the bytes on disk; this check covers the
        record/frame pairing.
        """

        payload = metadata.get("fit_weight_records")
        evidence = metadata.get("evidence")
        replay_payload = metadata.get("replay_payload")
        output_identity = metadata.get("output_content_identity")
        if (
            not isinstance(payload, list)
            or not all(isinstance(entry, Mapping) for entry in payload)
            or not isinstance(evidence, Mapping)
            or not isinstance(replay_payload, Mapping)
            or not isinstance(output_identity, str)
            or not output_identity
        ):
            raise RuntimeError(
                "SPI restoration resume requires the checkpoint record to "
                "carry the run's fit-weight records, family evidence, "
                "replay payload, and output content identity; a record "
                "without them cannot feed the weights audit, the driver's "
                "stage reports, or the drift check."
            )
        if uk_frame_content_identity(frame) != output_identity:
            raise RuntimeError(
                "SPI restoration checkpoint content does not match its "
                "recorded output identity; refusing to resume from a "
                "drifted record."
            )
        records = tuple(
            FitWeightRecord(
                fit_name=str(entry["fit_name"]),
                weight_kind=str(entry["weight_kind"]),
            )
            for entry in payload
        )
        self.last_result = _ResumedHMRCRestoration(
            frame=frame,
            imputation=_ResumedImputationEvidence(fit_weight_records=records),
            evidence_payload=dict(evidence),
            replay_payload=dict(replay_payload),
        )
        # The rehydration consumes the single-use binding: the stage will
        # not run, so a binding must not outlive the resume either.
        self.staging_provenance = None
        self.bound_input_identity = None

    def bind_staging_provenance(
        self,
        provenance: UKStagingProvenance,
        frame: Frame,
    ) -> None:
        """Receive the load provenance and the loaded frame from the driver.

        Provenance travels beside the frame, never inside it, so the driver
        hands both to the one stage whose fence binds the loaded bytes to the
        verified certified candidate. Binding records the loaded frame's
        content identity — derived here, inside the attesting code — so the
        fence can require that the pipeline it sits in started from a frame
        whose full content matches what the driver loaded, a guarantee that
        survives a process boundary where object identity cannot.
        """

        if not isinstance(provenance, UKStagingProvenance):
            raise TypeError("staging provenance must be UKStagingProvenance.")
        if not isinstance(frame, Frame):
            raise TypeError("bound frame must be a microcosm Frame.")
        self.staging_provenance = provenance
        self.bound_input_identity = uk_frame_content_identity(frame)

    def __call__(self, frame: Frame) -> Frame:
        # Single-use: a binding never outlives the run that consumes it, so
        # a stale binding from an earlier build can never fence a later one.
        staging_provenance = self.staging_provenance
        bound_input_identity = self.bound_input_identity
        self.staging_provenance = None
        self.bound_input_identity = None
        retained = (
            None
            if self.retained_leaves_transform is None
            else self.retained_leaves_transform.last_result
        )
        if retained is None:
            raise RuntimeError(
                "HMRC replay requires the raw-FRS retained-leaves stage to run "
                "immediately before the SPI stage."
            )
        # Descent fence A, content-addressed: the frame this stage received
        # must carry the exact content the retained-leaves stage produced.
        # Same-object is the free fast path (identity proves content); a
        # rehydrated (checkpointed) frame passes by content; a substituted
        # or tampered frame does not. The content check — and the
        # absence-is-refusal branch inside _retained_content_identity — is
        # the cross-process path's guarantee; the fast path shares the
        # same-process in-place-mutation exposure every consumer of
        # Frame.table has, mitigated by validate_uk_national_frame's
        # revalidation at each seam.
        if retained.frame is not frame and _retained_content_identity(
            retained, "output_content_identity"
        ) != uk_frame_content_identity(frame):
            raise RuntimeError(
                "HMRC replay raw-FRS evidence is not bound to the frame "
                "received from the immediately preceding retained-leaves stage."
            )
        # Descent fence B: the frame the retained-leaves stage consumed must
        # carry the exact content of the frame the driver loaded and bound.
        if (
            bound_input_identity is not None
            and _retained_content_identity(retained, "input_content_identity")
            != bound_input_identity
        ):
            raise RuntimeError(
                "HMRC replay pipeline did not start from the frame the "
                "driver loaded and bound; the certified-candidate fence "
                "refuses a substituted input."
            )
        self.last_result = restore_uk_hmrc_income_family(
            frame,
            spi_tab_path=self.spi_tab_path,
            hmrc_ods_path=self.hmrc_ods_path,
            certified_candidate=self.certified_candidate,
            staging_provenance=staging_provenance,
            frs_source_evidence=retained.evidence(),
            seed=self.seed,
            qrf_estimators=self.qrf_estimators,
            donor_sample_size=self.donor_sample_size,
            spi_prior_mass_share=self.spi_prior_mass_share,
        )
        return self.last_result.frame


@dataclass(frozen=True)
class _ResumedImputationEvidence:
    """The audit slice of an SPI imputation, rehydrated from a checkpoint."""

    fit_weight_records: tuple[FitWeightRecord, ...]


@dataclass(frozen=True)
class _ResumedHMRCRestoration:
    """A completed SPI restoration rehydrated from its checkpoint record.

    Exposes the full surface a national build consumes downstream: the
    fit-weight audit records, the aggregate family evidence, and the replay
    payload the driver writes verbatim (its content came from the real
    report's ``to_payload()`` at completion time).
    """

    frame: Frame
    imputation: _ResumedImputationEvidence
    evidence_payload: dict[str, object]
    replay_payload: dict[str, object]

    def evidence(self) -> dict[str, object]:
        return dict(self.evidence_payload)


def _retained_content_identity(retained: object, attribute: str) -> str:
    """Read a content identity off a retained-leaves result, failing closed.

    A retained result without content identities cannot prove descent, so
    the fence refuses it instead of silently downgrading to a weaker check
    (the microcosm#617 lesson: absence is a refusal, never a default).
    """

    identity = getattr(retained, attribute, None)
    if not isinstance(identity, str) or not identity:
        raise RuntimeError(
            "HMRC replay retained-leaves evidence carries no "
            f"{attribute}; the descent fence refuses a result that cannot "
            "prove which frames its run consumed and produced."
        )
    return identity


def verify_certified_uk_candidate(path: str | Path) -> UKCertifiedCandidateIdentity:
    """Hash/size gate the certified Microcosm UK candidate before stages run."""

    candidate = Path(path).expanduser().resolve()
    if not candidate.is_file():
        raise FileNotFoundError(
            f"Certified Microcosm UK candidate not found: {candidate}."
        )
    fingerprint_before = _uk_source_file_fingerprint(candidate)
    size = fingerprint_before.size_bytes
    if size != CERTIFIED_UK_CANDIDATE_SIZE_BYTES:
        raise ValueError(
            f"{candidate}: expected certified candidate size "
            f"{CERTIFIED_UK_CANDIDATE_SIZE_BYTES}, got {size}."
        )
    digest = _sha256(candidate)
    fingerprint_after = _uk_source_file_fingerprint(candidate)
    if fingerprint_after != fingerprint_before:
        raise RuntimeError(
            "Certified Microcosm UK candidate changed while its SHA-256 was "
            "being verified."
        )
    if digest != CERTIFIED_UK_CANDIDATE_SHA256:
        raise ValueError(
            f"{candidate}: sha256 {digest} does not match certified candidate "
            f"{CERTIFIED_UK_CANDIDATE_SHA256}."
        )
    identity = UKCertifiedCandidateIdentity(
        path=candidate,
        filename=CERTIFIED_UK_CANDIDATE_FILENAME,
        tier=CERTIFIED_UK_CANDIDATE_TIER,
        revision=CERTIFIED_UK_CANDIDATE_REVISION,
        sha256=digest,
        size_bytes=size,
    )
    object.__setattr__(
        identity,
        "_verification_token",
        _CERTIFIED_CANDIDATE_VERIFICATION_TOKEN,
    )
    object.__setattr__(identity, "_source_file_fingerprint", fingerprint_after)
    return identity


def restore_uk_hmrc_income_family(
    frame: Frame,
    *,
    spi_tab_path: str | Path,
    hmrc_ods_path: str | Path,
    certified_candidate: UKCertifiedCandidateIdentity,
    staging_provenance: UKStagingProvenance | None = None,
    frs_source_evidence: Mapping[str, object],
    seed: int = 42,
    qrf_estimators: int = 100,
    donor_sample_size: int | None = DEFAULT_SPI_DONOR_SAMPLE_SIZE,
    spi_prior_mass_share: float = DEFAULT_SPI_PRIOR_MASS_SHARE,
) -> UKHMRCIncomeRestorationResult:
    """Run the admissible real-donor replay without biased calibration."""

    assert_uk_hmrc_income_source_contract_current()
    _validate_certified_candidate_identity(certified_candidate)
    _assert_reviewed_release_parameters(
        donor_sample_size=donor_sample_size,
        spi_prior_mass_share=spi_prior_mass_share,
    )
    if not isinstance(frs_source_evidence, Mapping) or not frs_source_evidence:
        raise ValueError("HMRC replay requires non-empty raw-FRS source evidence.")
    validate_uk_national_frame(frame)
    _assert_provenance_matches_certified_candidate(
        staging_provenance, certified_candidate
    )
    time_period = uk_time_period(frame)

    # The licensed donor and official ODS are one reviewed source pair. Bind
    # both identities before parsing either source or rebuilding support.
    verified_donor = verify_spi_donor_identity(spi_tab_path)
    verified_ods = verify_hmrc_spi_collated_ods(hmrc_ods_path)
    assert_frs_hmrc_auxiliary_crosswalk_available(frame.table("person"))
    source_targets = materialize_hmrc_spi_income_band_targets(
        verified_ods,
        build_period=time_period,
    )

    support = replace_uk_spi_support_tables(
        person=frame.table("person"),
        benunit=frame.table("benunit"),
        household=frame.table("household"),
        seed=seed,
        source_year=int(time_period),
        spi_prior_mass_share=spi_prior_mass_share,
        input_weight_kind=uk_household_weight_kind(frame),
        mass_log=frame.mass_log,
    )
    imputation = impute_uk_spi_income_support(
        support,
        spi_tab_path,
        seed=seed,
        n_estimators=qrf_estimators,
        donor_sample_size=donor_sample_size,
        build_period=time_period,
        verified_donor=verified_donor,
    )
    # The SPI replacement reshapes tables AND advances the weight kind, so the
    # stage hard-constructs its output frame with the support seam's reviewed
    # mass log (conservation-checked, factor 1.0).
    replay_frame = uk_national_frame(
        person=imputation.person,
        benunit=support.benunit,
        household=support.household,
        time_period=time_period,
        weight_kind=WeightKind.IMPORTANCE,
        mass_log=support.mass_log,
    )
    validate_uk_national_frame(replay_frame)
    identity_rows = _assert_post_draw_identity(replay_frame)
    distributional_mass_shares = _distributional_mass_shares(replay_frame)
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

    source_evidence = {
        "certified_candidate": {
            "filename": certified_candidate.filename,
            "tier": certified_candidate.tier,
            "revision": certified_candidate.revision,
            "sha256": certified_candidate.sha256,
            "size_bytes": certified_candidate.size_bytes,
        },
        "raw_frs_retained_leaves": _aggregate_frs_source_evidence(frs_source_evidence),
        "spi_donor": {
            "release": "2022-23",
            "sha256": imputation.donor_sha256,
            "size_bytes": imputation.donor_size_bytes,
            "rows_used": imputation.donor_rows,
        },
        "hmrc_surface": {
            "vintage": source_targets.source.source_vintage,
            "mapped_build_period": source_targets.source.build_period,
            "sha256": source_targets.source.sha256,
            "size_bytes": source_targets.source.size_bytes,
            "mime_type": source_targets.source.mime_type,
            "tables": list(source_targets.source.table_names),
        },
    }
    build_evidence = {
        "stage": "hmrc_spi_income",
        "output_weight_kind": uk_household_weight_kind(replay_frame).value,
        "calibration_performed": False,
        "spi_prior_mass_share": support.spi_prior_mass_share,
        "replaced_spi_households": support.replaced_spi_households,
        "mass_change_reason": support.mass_log[-1].reason,
    }
    qrf_evidence = {
        "fits": {
            record.fit_name: {"weight_kind": record.weight_kind}
            for record in imputation.fit_weight_records
        },
        "donor_rows": imputation.donor_rows,
        "stage2_training_rows": imputation.stage2_training_rows,
        "spi_prediction_rows": imputation.spi_prediction_rows,
        "post_draw_identity": {
            "formula": SPI_SOURCE_TI_FORMULA,
            "rows_checked": identity_rows,
            "exact": True,
        },
    }
    effective_mass_evidence = {
        "minimum_nondefault_mass_share": DEFAULT_MINIMUM_NONDEFAULT_MASS_SHARE,
        "denominator": "all_person_effective_mass",
        "required_support_channel": SPI_SYNTHETIC_SUPPORT_CHANNEL,
        "columns": distributional_mass_shares,
    }
    report = build_conservative_hmrc_replay_report(
        source_targets,
        source_evidence=source_evidence,
        build_evidence=build_evidence,
        qrf_evidence=qrf_evidence,
        effective_mass_evidence=effective_mass_evidence,
    )
    return UKHMRCIncomeRestorationResult(
        frame=replay_frame,
        support=support,
        imputation=imputation,
        source_targets=source_targets,
        replay_report=report,
        distributional_mass_shares=distributional_mass_shares,
        post_draw_identity_rows=identity_rows,
    )


def _aggregate_frs_source_evidence(
    evidence: Mapping[str, object],
) -> dict[str, object]:
    """Drop machine-local paths while retaining aggregate source identities."""

    result = dict(evidence)
    raw_sources = result.get("sources")
    if isinstance(raw_sources, Mapping):
        result["sources"] = {
            str(name): {
                str(key): value for key, value in dict(source).items() if key != "path"
            }
            for name, source in raw_sources.items()
            if isinstance(source, Mapping)
        }
    return result


def _assert_post_draw_identity(frame: Frame) -> int:
    """Require deterministic TEI + TII = TI on every rebuilt SPI draw."""

    person = frame.table("person")
    channel = support_channel_column("person")
    required = (
        channel,
        SPI_HMRC_TOTAL_EARNED_INCOME_COLUMN,
        SPI_HMRC_TOTAL_INVESTMENT_INCOME_COLUMN,
        "hmrc_spi_assessable_income",
    )
    missing = sorted(set(required) - set(person.columns))
    if missing:
        raise RuntimeError(
            f"HMRC replay omitted post-draw identity column(s): {missing}."
        )
    spi = person[channel].eq(SPI_SYNTHETIC_SUPPORT_CHANNEL).to_numpy(dtype=bool)
    if not spi.any():
        raise RuntimeError("HMRC replay contains no rebuilt SPI person draws.")
    numeric = person.loc[
        spi,
        [
            SPI_HMRC_TOTAL_EARNED_INCOME_COLUMN,
            SPI_HMRC_TOTAL_INVESTMENT_INCOME_COLUMN,
            "hmrc_spi_assessable_income",
        ],
    ].apply(pd.to_numeric, errors="coerce")
    values = numeric.to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise RuntimeError("HMRC post-draw identity contains non-finite values.")
    if not np.array_equal(
        numeric["hmrc_spi_assessable_income"].to_numpy(dtype=float),
        numeric[SPI_HMRC_TOTAL_EARNED_INCOME_COLUMN].to_numpy(dtype=float)
        + numeric[SPI_HMRC_TOTAL_INVESTMENT_INCOME_COLUMN].to_numpy(dtype=float),
    ):
        raise RuntimeError("HMRC TI must equal deterministic TEI + TII exactly.")
    return int(spi.sum())


def _distributional_mass_shares(frame: Frame) -> dict[str, float]:
    """Audit charitable signal on strictly positive rebuilt-SPI mass."""

    person = frame.table("person")
    person_channel = support_channel_column("person")
    if person_channel not in person:
        raise RuntimeError(
            "Cannot audit HMRC distributional inputs without person support "
            "channel provenance."
        )
    spi_people = (
        person[person_channel].eq(SPI_SYNTHETIC_SUPPORT_CHANNEL).to_numpy(dtype=bool)
    )
    if not spi_people.any():
        raise RuntimeError("Rebuilt HMRC family contains no SPI support people.")
    household_weights = frame.table("household").set_index("household_id")[
        "household_weight"
    ]
    mapped = pd.to_numeric(
        person["person_household_id"].map(household_weights),
        errors="coerce",
    ).to_numpy(dtype=float, na_value=np.nan)
    if not np.isfinite(mapped).all() or (mapped < 0.0).any():
        raise RuntimeError(
            "Cannot audit HMRC distributional inputs without finite, "
            "non-negative person mass."
        )
    positive = mapped > 0.0
    total = float(mapped[positive].sum())
    if total <= 0.0:
        raise RuntimeError("HMRC distributional audit has no positive person mass.")
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
        shares[column] = (
            float(mapped[positive & spi_people & (values != 0.0)].sum()) / total
        )
    return shares


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_certified_candidate_identity(
    identity: UKCertifiedCandidateIdentity,
) -> None:
    if not isinstance(identity, UKCertifiedCandidateIdentity):
        raise TypeError("HMRC replay requires a verified UKCertifiedCandidateIdentity.")
    if identity._verification_token is not _CERTIFIED_CANDIDATE_VERIFICATION_TOKEN:
        raise ValueError(
            "HMRC replay candidate identity must come from "
            "verify_certified_uk_candidate; matching metadata fields alone are "
            "not verified source evidence."
        )
    if identity._source_file_fingerprint is None:
        raise ValueError(
            "HMRC replay candidate identity lacks verified source-file provenance."
        )
    expected = (
        CERTIFIED_UK_CANDIDATE_FILENAME,
        CERTIFIED_UK_CANDIDATE_TIER,
        CERTIFIED_UK_CANDIDATE_REVISION,
        CERTIFIED_UK_CANDIDATE_SHA256,
        CERTIFIED_UK_CANDIDATE_SIZE_BYTES,
    )
    actual = (
        identity.filename,
        identity.tier,
        identity.revision,
        identity.sha256,
        identity.size_bytes,
    )
    if actual != expected:
        raise ValueError(
            "HMRC replay base identity does not match the certified Microcosm UK "
            "candidate contract."
        )


def _assert_provenance_matches_certified_candidate(
    provenance: UKStagingProvenance | None,
    identity: UKCertifiedCandidateIdentity,
) -> None:
    """Bind the loaded bytes to the H5 verified once by the driver.

    The provenance record is produced by ``load_uk_national_frame`` beside the
    frame; the national build driver hands it to this stage through
    ``bind_staging_provenance``, so an unbound run — a frame that did not come
    through the loader — fails closed here.
    """

    if provenance is None:
        raise ValueError(
            "HMRC replay requires the staging provenance of a UK national "
            "frame loaded from the verified certified-candidate H5."
        )
    if provenance.source_h5 != identity.path:
        raise ValueError(
            "HMRC replay frame source does not match the verified certified "
            f"candidate: loaded {provenance.source_h5}, verified {identity.path}."
        )
    if provenance.fingerprint != identity._source_file_fingerprint:
        raise ValueError(
            "HMRC replay candidate H5 changed after SHA-256 verification; the "
            "loaded bytes are not the certified bytes."
        )


def _assert_reviewed_release_parameters(
    *,
    donor_sample_size: int | None,
    spi_prior_mass_share: float,
) -> None:
    reviewed = {
        "donor_sample_size": (donor_sample_size, DEFAULT_SPI_DONOR_SAMPLE_SIZE),
        "spi_prior_mass_share": (
            spi_prior_mass_share,
            DEFAULT_SPI_PRIOR_MASS_SHARE,
        ),
    }
    drifted = {
        name: {"actual": actual, "reviewed": expected}
        for name, (actual, expected) in reviewed.items()
        if actual != expected
    }
    if drifted:
        raise ValueError(
            "HMRC release parameters disagree with the reviewed source "
            f"manifest: {drifted}. Update the manifest and runtime together."
        )
