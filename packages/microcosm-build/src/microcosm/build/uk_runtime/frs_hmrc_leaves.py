"""Retain adjudicated raw-FRS leaves for the UK HMRC income surface.

The certified UK candidate was built from the 2023-24 FRS, but it does not
retain the raw constituents needed to compare its income measure with the
published HMRC tables.  This stage reopens only ``adult.tab`` and
``benefits.tab`` and carries the source-faithful, adjudicated constituents
through the candidate's SPI, capital-gains, and geography-clone descendants.

The two partial concepts deliberately keep subset names.  They must never be
mistaken for the full SPI ``OSSBEN`` or ``SRP`` concepts.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from microcosm.build.uk_runtime.content_identity import uk_frame_content_identity
from microcosm.build.uk_runtime.national_frame import (
    uk_household_weight_kind,
    uk_national_frame,
    uk_time_period,
    validate_uk_national_frame,
)
from microcosm.build.uk_runtime.spi_support import (
    HOUSEHOLD_IS_SPI_SYNTHETIC_COLUMN,
    SPI_HMRC_INCAPACITY_BENEFIT_INCOME_COLUMN,
    SPI_HMRC_PAY_COLUMN,
    SPI_HMRC_UNEMPLOYMENT_BENEFIT_INCOME_COLUMN,
)
from microcosm.frame import Frame

__all__ = [
    "FRS_HMRC_INCPBEN_COLUMN",
    "FRS_HMRC_OSSBEN_IDENTIFIABLE_SUBSET_COLUMN",
    "FRS_HMRC_PAY_COLUMN",
    "FRS_HMRC_RETAINED_LEAF_COLUMNS",
    "FRS_HMRC_RETAINED_LEAF_SOURCE_EVIDENCE",
    "FRS_HMRC_SRP_REGULAR_CODE5_COLUMN",
    "FRS_HMRC_UBISJA_COLUMN",
    "FRS_HMRC_RETAINED_LEAVES_STAGE_NAME",
    "UKFRSHMRCRetainedLeavesResult",
    "UKFRSHMRCRetainedLeavesStageTransform",
    "UKFRSRawTableIdentity",
    "retain_uk_frs_hmrc_leaves",
]

FRS_SOURCE_VINTAGE = "2023-24"
FRS_SOURCE_BUILD_PERIOD = "2023"
FRS_WEEKS_IN_YEAR = 365.25 / 7
FRS_HMRC_RETAINED_LEAVES_STAGE_NAME = "frs_hmrc_retained_leaves"

# Full concepts use the normalized columns already consumed by the SPI/HMRC
# stage.  Partial concepts are fenced under their adjudicated subset names.
FRS_HMRC_PAY_COLUMN = SPI_HMRC_PAY_COLUMN
FRS_HMRC_UBISJA_COLUMN = SPI_HMRC_UNEMPLOYMENT_BENEFIT_INCOME_COLUMN
FRS_HMRC_INCPBEN_COLUMN = SPI_HMRC_INCAPACITY_BENEFIT_INCOME_COLUMN
FRS_HMRC_OSSBEN_IDENTIFIABLE_SUBSET_COLUMN = "ossben_identifiable_subset"
FRS_HMRC_SRP_REGULAR_CODE5_COLUMN = "srp_regular_code5"
FRS_HMRC_RETAINED_LEAF_COLUMNS = (
    FRS_HMRC_PAY_COLUMN,
    FRS_HMRC_UBISJA_COLUMN,
    FRS_HMRC_INCPBEN_COLUMN,
    FRS_HMRC_OSSBEN_IDENTIFIABLE_SUBSET_COLUMN,
    FRS_HMRC_SRP_REGULAR_CODE5_COLUMN,
)

FRS_HMRC_RETAINED_LEAF_SOURCE_EVIDENCE: dict[str, dict[str, object]] = {
    FRS_HMRC_PAY_COLUMN: {
        "spi_concept": "PAY",
        "scope": "full",
        "raw_sources": ["ADULT.INEARNS"],
        "formula": "max(0, ADULT.INEARNS) * (365.25 / 7)",
    },
    FRS_HMRC_UBISJA_COLUMN: {
        "spi_concept": "UBISJA",
        "scope": "full",
        "raw_sources": [
            "BENEFITS.BENEFIT=14:BENAMT",
            "BENEFITS.BENEFIT=19:BENAMT",
        ],
        "formula": "sum(BENAMT where BENEFIT in {14, 19}) * (365.25 / 7)",
    },
    FRS_HMRC_INCPBEN_COLUMN: {
        "spi_concept": "INCPBEN",
        "scope": "full",
        "raw_sources": ["BENEFITS.BENEFIT=17:BENAMT"],
        "formula": "sum(BENAMT where BENEFIT == 17) * (365.25 / 7)",
    },
    FRS_HMRC_OSSBEN_IDENTIFIABLE_SUBSET_COLUMN: {
        "spi_concept": "OSSBEN",
        "scope": "identifiable_subset",
        "raw_sources": [
            "BENEFITS.BENEFIT=13:BENAMT",
            "BENEFITS.BENEFIT=16,VAR2 in {1,3}:BENAMT",
        ],
        "formula": (
            "sum(BENAMT where BENEFIT == 13 or "
            "(BENEFIT == 16 and VAR2 in {1, 3})) * (365.25 / 7)"
        ),
    },
    FRS_HMRC_SRP_REGULAR_CODE5_COLUMN: {
        "spi_concept": "SRP",
        "scope": "regular_code5_subset",
        "raw_sources": ["BENEFITS.BENEFIT=5:BENAMT"],
        "formula": "sum(BENAMT where BENEFIT == 5) * (365.25 / 7)",
    },
}

_ADULT_REQUIRED_COLUMNS = ("sernum", "person", "inearns")
_BENEFITS_REQUIRED_COLUMNS = (
    "sernum",
    "person",
    "benefit",
    "benamt",
    "var2",
)
_CAPITAL_GAINS_FLAG = "household_is_capital_gains_clone"


@dataclass(frozen=True)
class UKFRSRawTableIdentity:
    """Stable identity and extraction surface for one raw FRS table."""

    path: Path
    filename: str
    source_vintage: str
    sha256: str
    size_bytes: int
    rows: int
    extracted_columns: tuple[str, ...]

    def evidence(self) -> dict[str, object]:
        """Return JSON-safe source evidence."""

        return {
            "path": str(self.path),
            "filename": self.filename,
            "source_vintage": self.source_vintage,
            "sha256": self.sha256,
            "size_bytes": self.size_bytes,
            "rows": self.rows,
            "extracted_columns": list(self.extracted_columns),
        }


@dataclass(frozen=True)
class UKFRSHMRCRetainedLeavesResult:
    """National frame plus raw-source and lineage evidence.

    ``input_content_identity`` and ``output_content_identity`` are the
    content identities (:func:`uk_frame_content_identity`) of the frame this
    stage consumed and the frame it produced, derived inside the attesting
    run. The SPI stage's descent fence compares against them, so the
    guarantee survives a process boundary: a checkpoint-rehydrated frame is
    content-identical to the one that was checkpointed, while a substituted
    or tampered frame is not.
    """

    frame: Frame
    adult_source: UKFRSRawTableIdentity
    benefits_source: UKFRSRawTableIdentity
    clone_id_multiplier: int
    spi_person_id_offset: int
    capital_gains_person_id_offset: int
    raw_source_people: int
    candidate_people: int
    source_signal_rows: dict[str, int]
    structural_zero_columns: tuple[str, ...]
    input_content_identity: str
    output_content_identity: str
    #: Raw-survey people outside the candidate base. Zero on a full-scale
    #: build (the completeness fence raises otherwise); on a #627 rung
    #: sample it receipts how much of the raw surface the rung dropped.
    source_people_outside_candidate: int = 0

    def evidence(self) -> dict[str, object]:
        """Return aggregate, JSON-safe evidence for a national build driver."""

        return {
            "stage": FRS_HMRC_RETAINED_LEAVES_STAGE_NAME,
            "source_vintage": FRS_SOURCE_VINTAGE,
            "mapped_build_period": uk_time_period(self.frame),
            "sources": {
                "adult": self.adult_source.evidence(),
                "benefits": self.benefits_source.evidence(),
            },
            "annualization": {
                "days_per_year": 365.25,
                "days_per_week": 7,
                "weeks_per_year": FRS_WEEKS_IN_YEAR,
            },
            "lineage": {
                "clone_id_multiplier": self.clone_id_multiplier,
                "spi_person_id_offset": self.spi_person_id_offset,
                "capital_gains_person_id_offset": (self.capital_gains_person_id_offset),
                "raw_source_people": self.raw_source_people,
                "candidate_people": self.candidate_people,
                "source_people_outside_candidate": (
                    self.source_people_outside_candidate
                ),
            },
            "retained_leaves": {
                column: {
                    **FRS_HMRC_RETAINED_LEAF_SOURCE_EVIDENCE[column],
                    "source_signal_rows": self.source_signal_rows[column],
                    "structural_zero": column in self.structural_zero_columns,
                }
                for column in FRS_HMRC_RETAINED_LEAF_COLUMNS
            },
        }


@dataclass
class UKFRSHMRCRetainedLeavesStageTransform:
    """Callable national-stage adapter retaining the last run's evidence."""

    adult_tab_path: Path
    benefits_tab_path: Path
    #: Declared #627 rung build: relaxes the raw-surface completeness fence
    #: into a receipted count. Never set on a release build.
    sampled_rung: bool = False
    last_result: UKFRSHMRCRetainedLeavesResult | None = field(
        default=None,
        init=False,
    )

    @classmethod
    def from_raw_frs_directory(
        cls,
        raw_frs_directory: str | Path,
        *,
        sampled_rung: bool = False,
    ) -> UKFRSHMRCRetainedLeavesStageTransform:
        """Resolve the two permitted tables from a CLI-supplied directory."""

        directory = Path(raw_frs_directory).expanduser()
        return cls(
            adult_tab_path=directory / "adult.tab",
            benefits_tab_path=directory / "benefits.tab",
            sampled_rung=sampled_rung,
        )

    def __call__(self, frame: Frame) -> Frame:
        # The result records the content identities of the frame this stage
        # consumed and produced, so the SPI stage's fence can assert descent
        # from the frame the driver loaded and bound — including across a
        # process boundary, where object identity cannot travel.
        self.last_result = retain_uk_frs_hmrc_leaves(
            frame,
            adult_tab_path=self.adult_tab_path,
            benefits_tab_path=self.benefits_tab_path,
            sampled_rung=self.sampled_rung,
        )
        return self.last_result.frame

    def checkpoint_metadata(self) -> dict[str, object]:
        """JSON-safe evidence the stage checkpoint carries for a resume.

        The SPI stage consumes the retained-leaves evidence and the descent
        identities; persisting them on the completed stage's run-context
        record is what lets a later process resume past this stage without
        re-running it.
        """

        if self.last_result is None:
            raise RuntimeError(
                "checkpoint metadata requires a completed retained-leaves run."
            )
        return {
            "evidence": self.last_result.evidence(),
            "input_content_identity": self.last_result.input_content_identity,
            "output_content_identity": self.last_result.output_content_identity,
        }

    def resume_from_checkpoint(
        self,
        metadata: Mapping[str, object],
        frame: Frame,
    ) -> None:
        """Rehydrate a completed run's evidence from its checkpoint record.

        ``frame`` is the stage's checkpointed output; the rehydrated result
        exposes exactly the surface the SPI stage's descent fence reads. The
        recorded output identity must match the loaded frame's content — a
        mismatch means the record and the checkpoint have drifted apart, and
        the resume fails closed.
        """

        evidence = metadata.get("evidence")
        input_identity = metadata.get("input_content_identity")
        output_identity = metadata.get("output_content_identity")
        if (
            not isinstance(evidence, Mapping)
            or not isinstance(input_identity, str)
            or not isinstance(output_identity, str)
        ):
            raise RuntimeError(
                "retained-leaves resume requires the checkpoint record to "
                "carry the run's evidence and content identities; a record "
                "without them cannot prove descent."
            )
        if uk_frame_content_identity(frame) != output_identity:
            raise RuntimeError(
                "retained-leaves checkpoint content does not match its "
                "recorded output identity; refusing to resume from a "
                "drifted record."
            )
        self.last_result = _ResumedRetainedLeaves(
            frame=frame,
            evidence_payload=dict(evidence),
            input_content_identity=input_identity,
            output_content_identity=output_identity,
        )


@dataclass(frozen=True)
class _ResumedRetainedLeaves:
    """A completed retained-leaves run rehydrated from its checkpoint.

    Carries exactly the surface the SPI stage consumes: the output frame,
    the JSON-safe evidence, and the descent content identities.
    """

    frame: Frame
    evidence_payload: dict[str, object]
    input_content_identity: str
    output_content_identity: str

    def evidence(self) -> dict[str, object]:
        return dict(self.evidence_payload)


@dataclass(frozen=True)
class _FileFingerprint:
    device: int
    inode: int
    size_bytes: int
    modified_ns: int
    changed_ns: int


@dataclass(frozen=True)
class _CandidateLineage:
    source_person_ids: np.ndarray
    clone_id_multiplier: int
    spi_person_id_offset: int
    capital_gains_person_id_offset: int
    canonical_raw_person_ids: frozenset[int]


def retain_uk_frs_hmrc_leaves(
    frame: Frame,
    *,
    adult_tab_path: str | Path,
    benefits_tab_path: str | Path,
    sampled_rung: bool = False,
) -> UKFRSHMRCRetainedLeavesResult:
    """Read two raw FRS tables and retain the adjudicated HMRC constituents.

    ``sampled_rung`` declares a #627 scale-ladder build: the candidate base
    deliberately carries only a sampled subset of source families, so the
    completeness fence (every raw-survey person present in the base) cannot
    hold. The raw surface is restricted to surviving canonicals and the
    dropped count is receipted instead — never silently. Full-scale builds
    keep the strict fence.
    """

    validate_uk_national_frame(frame)
    input_content_identity = uk_frame_content_identity(frame)
    time_period = uk_time_period(frame)
    if time_period not in {FRS_SOURCE_BUILD_PERIOD, FRS_SOURCE_VINTAGE}:
        raise ValueError(
            f"Raw FRS {FRS_SOURCE_VINTAGE} leaves may only map to build period "
            f"{FRS_SOURCE_BUILD_PERIOD!r}; got {time_period!r}."
        )

    adult, adult_source = _read_raw_frs_table(
        adult_tab_path,
        expected_filename="adult.tab",
        required_columns=_ADULT_REQUIRED_COLUMNS,
    )
    benefits, benefits_source = _read_raw_frs_table(
        benefits_tab_path,
        expected_filename="benefits.tab",
        required_columns=_BENEFITS_REQUIRED_COLUMNS,
    )
    source_leaves = _materialize_source_leaves(adult, benefits)
    lineage = _resolve_candidate_lineage(frame)
    unknown_source_ids = sorted(
        set(source_leaves.index) - lineage.canonical_raw_person_ids
    )
    if unknown_source_ids and not sampled_rung:
        raise ValueError(
            "Raw FRS retained leaves contain person identity value(s) absent "
            f"from the certified candidate base: {unknown_source_ids[:5]}."
        )
    source_people_outside_candidate = len(unknown_source_ids)
    # Signal-row evidence stays a fact about the SOURCE at every rung:
    # structural_zero must never be asserted from a sampled-away surface
    # (adversarial-review finding). The rung also cannot distinguish a
    # compact genuinely missing raw people from sampling loss — that check
    # remains the full-scale fence's, which stays strict.
    full_source_leaves = source_leaves
    if unknown_source_ids:
        # A rung sample deliberately drops most source families; restrict the
        # raw surface to the surviving canonicals and receipt the count.
        source_leaves = source_leaves.loc[
            source_leaves.index.isin(list(lineage.canonical_raw_person_ids))
        ]

    person = frame.table("person").copy()
    aligned = source_leaves.reindex(lineage.source_person_ids, fill_value=0.0)
    if aligned.isna().any().any():  # pragma: no cover - defensive
        raise RuntimeError("Raw FRS retained-leaf alignment produced missing values.")
    values = aligned.to_numpy(dtype=float)
    if not np.isfinite(values).all() or (values < 0.0).any():
        raise RuntimeError(
            "Raw FRS retained-leaf alignment produced non-finite or negative values."
        )
    for column in FRS_HMRC_RETAINED_LEAF_COLUMNS:
        person[column] = aligned[column].to_numpy(dtype=float)

    # Person-only replacement: mass is untouched, so the kind and mass log
    # carry through unchanged; Frame construction re-runs linkage validation.
    result_frame = uk_national_frame(
        person=person,
        benunit=frame.table("benunit"),
        household=frame.table("household"),
        time_period=time_period,
        weight_kind=uk_household_weight_kind(frame),
        mass_log=frame.mass_log,
    )
    validate_uk_national_frame(result_frame)
    _validate_retained_leaf_propagation(
        result_frame.table("person"),
        source_person_ids=lineage.source_person_ids,
        source_leaves=source_leaves,
    )
    source_signal_rows = {
        column: int((full_source_leaves[column] > 0.0).sum())
        for column in FRS_HMRC_RETAINED_LEAF_COLUMNS
    }
    structural_zero_columns = tuple(
        column
        for column in FRS_HMRC_RETAINED_LEAF_COLUMNS
        if source_signal_rows[column] == 0
    )
    return UKFRSHMRCRetainedLeavesResult(
        frame=result_frame,
        adult_source=adult_source,
        benefits_source=benefits_source,
        clone_id_multiplier=lineage.clone_id_multiplier,
        spi_person_id_offset=lineage.spi_person_id_offset,
        capital_gains_person_id_offset=lineage.capital_gains_person_id_offset,
        raw_source_people=len(source_leaves),
        candidate_people=len(person),
        source_people_outside_candidate=source_people_outside_candidate,
        source_signal_rows=source_signal_rows,
        structural_zero_columns=structural_zero_columns,
        input_content_identity=input_content_identity,
        output_content_identity=uk_frame_content_identity(result_frame),
    )


def _read_raw_frs_table(
    path: str | Path,
    *,
    expected_filename: str,
    required_columns: tuple[str, ...],
) -> tuple[pd.DataFrame, UKFRSRawTableIdentity]:
    source_path = Path(path).expanduser().resolve()
    if source_path.name.lower() != expected_filename:
        raise ValueError(
            f"Expected raw FRS table {expected_filename!r}, got {source_path.name!r}."
        )
    if not source_path.is_file():
        raise FileNotFoundError(f"Raw FRS table not found: {source_path}.")
    before = _file_fingerprint(source_path)
    digest = _sha256(source_path)
    after_hash = _file_fingerprint(source_path)
    if after_hash != before:
        raise RuntimeError(f"Raw FRS table changed while hashing: {source_path}.")
    required = set(required_columns)
    frame = pd.read_csv(
        source_path,
        sep="\t",
        usecols=lambda column: str(column).strip().lower() in required,
    )
    after_read = _file_fingerprint(source_path)
    if after_read != before:
        raise RuntimeError(f"Raw FRS table changed while reading: {source_path}.")
    frame.columns = frame.columns.astype(str).str.strip().str.lower()
    if frame.columns.duplicated().any():
        duplicates = frame.columns[frame.columns.duplicated()].tolist()
        raise ValueError(
            f"Raw FRS {expected_filename} has duplicate normalized columns: "
            f"{duplicates}."
        )
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(
            f"Raw FRS {expected_filename} is missing required column(s): {missing}."
        )
    frame = frame.loc[:, list(required_columns)]
    identity = UKFRSRawTableIdentity(
        path=source_path,
        filename=expected_filename,
        source_vintage=FRS_SOURCE_VINTAGE,
        sha256=digest,
        size_bytes=before.size_bytes,
        rows=len(frame),
        extracted_columns=required_columns,
    )
    return frame, identity


def _materialize_source_leaves(
    adult: pd.DataFrame,
    benefits: pd.DataFrame,
) -> pd.DataFrame:
    adult_ids = _raw_source_person_ids(adult, label="ADULT")
    if pd.Index(adult_ids).duplicated().any():
        duplicates = pd.Index(adult_ids)[pd.Index(adult_ids).duplicated()].unique()
        raise ValueError(
            "Raw FRS ADULT person identities must be unique; duplicate "
            f"value(s): {duplicates[:5].tolist()}."
        )
    earnings = _finite_numeric(adult["inearns"], label="ADULT.INEARNS")
    pay = np.maximum(earnings, 0.0) * FRS_WEEKS_IN_YEAR
    adult_leaf = pd.DataFrame(
        {FRS_HMRC_PAY_COLUMN: pay},
        index=pd.Index(adult_ids, name="source_person_id"),
    )

    benefit_ids = _raw_source_person_ids(benefits, label="BENEFITS")
    benefit_codes = _strict_integer_values(
        benefits["benefit"],
        label="BENEFITS.BENEFIT",
        minimum=0,
    )
    relevant = np.isin(benefit_codes, (5, 13, 14, 16, 17, 19))
    amounts = np.zeros(len(benefits), dtype=float)
    if relevant.any():
        relevant_amounts = _finite_numeric(
            benefits.loc[relevant, "benamt"],
            label="relevant BENEFITS.BENAMT",
        )
        if (relevant_amounts < 0.0).any():
            raise ValueError("Relevant BENEFITS.BENAMT values must be non-negative.")
        amounts[relevant] = relevant_amounts

    code16 = benefit_codes == 16
    contribution_based_esa = np.zeros(len(benefits), dtype=bool)
    if code16.any():
        var2 = _strict_integer_values(
            benefits.loc[code16, "var2"],
            label="BENEFITS.VAR2 for BENEFIT=16",
        )
        contribution_based_esa[code16] = np.isin(var2, (1, 3))

    benefit_leaf = pd.DataFrame(
        {
            FRS_HMRC_UBISJA_COLUMN: amounts * np.isin(benefit_codes, (14, 19)),
            FRS_HMRC_INCPBEN_COLUMN: amounts * (benefit_codes == 17),
            FRS_HMRC_OSSBEN_IDENTIFIABLE_SUBSET_COLUMN: amounts
            * ((benefit_codes == 13) | contribution_based_esa),
            FRS_HMRC_SRP_REGULAR_CODE5_COLUMN: amounts * (benefit_codes == 5),
        },
        index=pd.Index(benefit_ids, name="source_person_id"),
    )
    benefit_leaf = benefit_leaf.groupby(level=0, sort=False).sum()
    benefit_leaf *= FRS_WEEKS_IN_YEAR

    source_ids = adult_leaf.index.union(benefit_leaf.index, sort=False)
    result = pd.DataFrame(
        0.0,
        index=source_ids,
        columns=FRS_HMRC_RETAINED_LEAF_COLUMNS,
    )
    result.loc[adult_leaf.index, FRS_HMRC_PAY_COLUMN] = adult_leaf[FRS_HMRC_PAY_COLUMN]
    for column in benefit_leaf.columns:
        result.loc[benefit_leaf.index, column] = benefit_leaf[column]
    numeric = result.to_numpy(dtype=float)
    if not np.isfinite(numeric).all() or (numeric < 0.0).any():
        raise RuntimeError("Raw FRS source-leaf materialization is invalid.")
    return result


def _resolve_candidate_lineage(frame: Frame) -> _CandidateLineage:
    person = frame.table("person")
    household = frame.table("household")
    _require_columns(
        person,
        ("person_id", "person_household_id"),
        label="candidate person",
    )
    _require_columns(
        household,
        (
            "household_id",
            "clone_index",
            HOUSEHOLD_IS_SPI_SYNTHETIC_COLUMN,
            _CAPITAL_GAINS_FLAG,
        ),
        label="candidate household",
    )
    person_ids = _strict_integer_values(
        person["person_id"], label="candidate person_id", minimum=1
    )
    person_household_ids = _strict_integer_values(
        person["person_household_id"],
        label="candidate person_household_id",
        minimum=1,
    )
    household_ids = _strict_integer_values(
        household["household_id"], label="candidate household_id", minimum=1
    )
    clone_index = _strict_integer_values(
        household["clone_index"], label="candidate clone_index", minimum=0
    )
    spi = _strict_bool_values(
        household[HOUSEHOLD_IS_SPI_SYNTHETIC_COLUMN],
        label=HOUSEHOLD_IS_SPI_SYNTHETIC_COLUMN,
    )
    capital_gains = _strict_bool_values(
        household[_CAPITAL_GAINS_FLAG], label=_CAPITAL_GAINS_FLAG
    )
    household_metadata = pd.DataFrame(
        {
            "household_id": household_ids,
            "clone_index": clone_index,
            "spi": spi,
            "capital_gains": capital_gains,
        }
    ).set_index("household_id")
    mapped = household_metadata.reindex(person_household_ids)
    if mapped.isna().any().any():
        raise ValueError(
            "Candidate person_household_id cannot map every person to lineage metadata."
        )
    person_clone_index = mapped["clone_index"].to_numpy(dtype=np.int64)
    person_spi = mapped["spi"].to_numpy(dtype=bool)
    person_capital_gains = mapped["capital_gains"].to_numpy(dtype=bool)

    canonical_households = clone_index == 0
    canonical_people = person_clone_index == 0
    if not canonical_households.any() or not canonical_people.any():
        raise ValueError("Candidate lineage requires clone_index=0 rows.")
    canonical_max = max(
        int(household_ids[canonical_households].max()),
        int(person_ids[canonical_people].max()),
    )
    clone_multiplier = 10 ** max(1, len(str(canonical_max)))
    clone_reversed_household_ids = household_ids - clone_index * clone_multiplier
    clone_reversed_person_ids = person_ids - person_clone_index * clone_multiplier
    clone_reversed_person_households = (
        person_household_ids - person_clone_index * clone_multiplier
    )
    if (
        (clone_reversed_household_ids <= 0).any()
        or (clone_reversed_person_ids <= 0).any()
        or (clone_reversed_person_households <= 0).any()
    ):
        raise ValueError("Candidate clone reversal produced non-positive IDs.")

    canonical_household_metadata = pd.DataFrame(
        {
            "clone_household_id": household_ids[canonical_households],
            "spi": spi[canonical_households],
            "capital_gains": capital_gains[canonical_households],
        }
    ).set_index("clone_household_id")
    expected_household_metadata = canonical_household_metadata.reindex(
        clone_reversed_household_ids
    )
    if expected_household_metadata.isna().any().any():
        raise ValueError(
            "Candidate geography-clone household IDs do not reverse to the "
            "clone_index=0 surface."
        )
    if not np.array_equal(
        expected_household_metadata["spi"].to_numpy(dtype=bool), spi
    ) or not np.array_equal(
        expected_household_metadata["capital_gains"].to_numpy(dtype=bool),
        capital_gains,
    ):
        raise ValueError(
            "Candidate geography clones disagree with canonical household flags."
        )

    canonical_person = pd.DataFrame(
        {
            "clone_person_id": person_ids[canonical_people],
            "clone_household_id": person_household_ids[canonical_people],
            "spi": person_spi[canonical_people],
            "capital_gains": person_capital_gains[canonical_people],
        }
    ).set_index("clone_person_id")
    expected_people = canonical_person.reindex(clone_reversed_person_ids)
    if expected_people.isna().any().any():
        raise ValueError(
            "Candidate geography-clone person IDs do not reverse to the "
            "clone_index=0 surface."
        )
    if not np.array_equal(
        expected_people["clone_household_id"].to_numpy(dtype=np.int64),
        clone_reversed_person_households,
    ):
        raise ValueError(
            "Candidate geography-clone person/household memberships are inconsistent."
        )
    descriptors = pd.DataFrame(
        {
            "clone_person_id": clone_reversed_person_ids,
            "clone_index": person_clone_index,
        }
    )
    if descriptors.duplicated().any():
        raise ValueError(
            "Candidate geography-clone person lineage contains duplicate descendants."
        )

    canonical_person_ids = person_ids[canonical_people]
    canonical_person_spi = person_spi[canonical_people]
    canonical_person_capital_gains = person_capital_gains[canonical_people]
    raw_person = ~canonical_person_spi & ~canonical_person_capital_gains
    pre_capital_gains = ~canonical_person_capital_gains
    if not raw_person.any():
        raise ValueError("Candidate lineage has no canonical raw FRS people.")
    spi_offset = int(canonical_person_ids[raw_person].max()) + 1
    capital_gains_offset = int(canonical_person_ids[pre_capital_gains].max()) + 1
    source_person_ids = (
        clone_reversed_person_ids
        - person_spi.astype(np.int64) * spi_offset
        - person_capital_gains.astype(np.int64) * capital_gains_offset
    )
    canonical_raw_ids = frozenset(
        int(value) for value in canonical_person_ids[raw_person]
    )
    if (source_person_ids <= 0).any() or not set(source_person_ids).issubset(
        canonical_raw_ids
    ):
        bad = sorted(set(source_person_ids) - canonical_raw_ids)
        raise ValueError(
            "Candidate SPI/capital-gains person IDs do not reverse to the raw "
            f"FRS surface: {bad[:5]}."
        )

    raw_household = ~spi[canonical_households] & ~capital_gains[canonical_households]
    pre_capital_household = ~capital_gains[canonical_households]
    canonical_household_ids = household_ids[canonical_households]
    if not raw_household.any():
        raise ValueError("Candidate lineage has no canonical raw FRS households.")
    spi_household_offset = int(canonical_household_ids[raw_household].max()) + 1
    capital_household_offset = (
        int(canonical_household_ids[pre_capital_household].max()) + 1
    )
    source_household_ids = (
        clone_reversed_person_households
        - person_spi.astype(np.int64) * spi_household_offset
        - person_capital_gains.astype(np.int64) * capital_household_offset
    )
    if not np.array_equal(source_person_ids // 1000, source_household_ids):
        raise ValueError(
            "Candidate reversed person IDs disagree with reversed household IDs."
        )
    lineage_descriptors = pd.DataFrame(
        {
            "source_person_id": source_person_ids,
            "clone_index": person_clone_index,
            "spi": person_spi,
            "capital_gains": person_capital_gains,
        }
    )
    if lineage_descriptors.duplicated().any():
        raise ValueError(
            "Candidate person lineage contains duplicate stack identities."
        )
    return _CandidateLineage(
        source_person_ids=source_person_ids,
        clone_id_multiplier=clone_multiplier,
        spi_person_id_offset=spi_offset,
        capital_gains_person_id_offset=capital_gains_offset,
        canonical_raw_person_ids=canonical_raw_ids,
    )


def _validate_retained_leaf_propagation(
    person: pd.DataFrame,
    *,
    source_person_ids: np.ndarray,
    source_leaves: pd.DataFrame,
) -> None:
    expected = source_leaves.reindex(source_person_ids, fill_value=0.0)
    actual = person.loc[:, list(FRS_HMRC_RETAINED_LEAF_COLUMNS)].apply(
        pd.to_numeric, errors="coerce"
    )
    actual_values = actual.to_numpy(dtype=float)
    if not np.isfinite(actual_values).all() or (actual_values < 0.0).any():
        raise RuntimeError("Retained FRS HMRC leaves must be finite and non-negative.")
    if not np.array_equal(actual_values, expected.to_numpy(dtype=float)):
        raise RuntimeError("Retained FRS HMRC leaves lost source-person alignment.")


def _raw_source_person_ids(frame: pd.DataFrame, *, label: str) -> np.ndarray:
    households = _strict_integer_values(
        frame["sernum"], label=f"{label}.SERNUM", minimum=1
    )
    people = _strict_integer_values(frame["person"], label=f"{label}.PERSON", minimum=1)
    if (people >= 1000).any():
        raise ValueError(f"{label}.PERSON must be less than 1000.")
    maximum_household = (np.iinfo(np.int64).max - people) // 1000
    if (households > maximum_household).any():
        raise ValueError(f"{label} source person identity exceeds int64 range.")
    return households * 1000 + people


def _strict_integer_values(
    values: pd.Series,
    *,
    label: str,
    minimum: int | None = None,
) -> np.ndarray:
    numeric = pd.to_numeric(values, errors="coerce").to_numpy(
        dtype=float, na_value=np.nan
    )
    if not np.isfinite(numeric).all():
        raise ValueError(f"{label} must contain finite numeric values.")
    if not np.equal(numeric, np.floor(numeric)).all():
        raise ValueError(f"{label} must contain integer values.")
    if (np.abs(numeric) > np.iinfo(np.int64).max).any():
        raise ValueError(f"{label} exceeds int64 range.")
    result = numeric.astype(np.int64)
    if minimum is not None and (result < minimum).any():
        raise ValueError(f"{label} must be at least {minimum}.")
    return result


def _strict_bool_values(values: pd.Series, *, label: str) -> np.ndarray:
    if pd.api.types.is_bool_dtype(values.dtype):
        if values.isna().any():
            raise ValueError(f"{label} must not contain missing values.")
        return values.to_numpy(dtype=bool)
    numeric = _strict_integer_values(values, label=label)
    if not np.isin(numeric, (0, 1)).all():
        raise ValueError(f"{label} must contain only boolean or 0/1 values.")
    return numeric.astype(bool)


def _finite_numeric(values: pd.Series, *, label: str) -> np.ndarray:
    numeric = pd.to_numeric(values, errors="coerce").to_numpy(
        dtype=float, na_value=np.nan
    )
    if not np.isfinite(numeric).all():
        raise ValueError(f"{label} must contain finite numeric values.")
    return numeric


def _require_columns(
    frame: pd.DataFrame,
    columns: tuple[str, ...],
    *,
    label: str,
) -> None:
    missing = sorted(set(columns) - set(frame.columns))
    if missing:
        raise ValueError(f"{label} is missing required column(s): {missing}.")


def _file_fingerprint(path: Path) -> _FileFingerprint:
    stat = path.stat()
    return _FileFingerprint(
        device=stat.st_dev,
        inode=stat.st_ino,
        size_bytes=stat.st_size,
        modified_ns=stat.st_mtime_ns,
        changed_ns=stat.st_ctime_ns,
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
