"""Native ACS disability and weeks-worked inputs for the ACS local lane.

The ACS local lane imputed ``is_disabled`` and ``is_blind`` onto ACS rows from
ASEC donors, although the ACS asks the same six disability-difficulty items
the ASEC does, and it never loaded weeks worked at all (microcosm#1021). SNAP's
work-requirement tests read both: the disability exemption, and weeks worked
to average usual hours over the year (policyengine-us#9660).

This fresh-build stage runs on the ACS-only frame right after the shared
native mapping and before the ASEC transfer, so the transfer, which only fills
missing cells, leaves these columns alone. The ASEC rows keep the donor
release's own values; they are never touched here.

- **is_disabled** follows the ASEC definition of the eligibility-inputs stage
  (:mod:`~microcosm.build.us_runtime.eligibility_inputs`): any of the six
  difficulty items ``DEAR``/``DEYE``/``DOUT``/``DPHY``/``DREM``/``DDRS`` is 1,
  or the person reports SSI (``SSIP`` > 0) and is under 65, since SSI requires
  recipients under 65 to be disabled or blind. The ACS items are the ASEC's
  ``PEDIS*`` items (:data:`ACS_DIFFICULTY_TO_CPS`).
- **is_blind** is ``DEYE`` == 1 ("blind or serious difficulty seeing even
  when wearing glasses"), as the ASEC reads ``PEDISEYE``.
- **Universe.** Each item is asked from its minimum age
  (:data:`ACS_DIFFICULTY_MIN_AGE`: hearing and vision at all ages, cognitive,
  ambulatory and self-care from 5, independent living from 15). Below it the
  Census code is blank, which reads as "no difficulty reported", so a child
  under 5 can be disabled only through hearing, vision or SSI. In-universe
  codes must be 1 or 2, and a code below the minimum age is refused.
- **weeks_worked** is ``WKWN`` under the ``WKHP`` universe rules
  (:func:`~microcosm.build.us_runtime.acs_inputs.acs_weeks_worked_values`),
  and must be blank exactly where ``WKHP`` is. It is carried on the staging
  ACS spine only. Under the pinned policyengine-us, ``weeks_worked`` has a
  ``formula_2025``, so the consumer export's ``project_input_only`` holds it
  back as formula-owned. Exporting it needs policyengine-us#9660 (part A1
  deletes that formula) and a pin/ABI bump; until then staging records it
  and this module's gate checks it, and nothing downstream reads it.

Not exported (see the receipt's ``not_exported``):

- ``is_veteran`` is a formula (``veterans_benefits > 0``) for every year in
  policyengine-us, so an ACS-only input would need an engine input path and
  would leave ASEC rows on the formula. ACS ``MIL`` (loaded as ``ACS_MIL`` by
  microcosm#1020) is 2 for a veteran ("on active duty in the past, but not
  now"); 1 (now on active duty) and 3 (Reserves or National Guard training
  only) are not, and it is blank under 17.
- ``is_incapable_of_self_care`` stays on the ASEC transfer, fitted jointly
  with the care-expense inputs.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from microcosm.build.gates import GateResult
from microcosm.build.us_runtime.acs_inputs import (
    _nullable_source_codes,
    acs_weeks_worked_values,
    map_acs_weeks_worked,
)
from microcosm.build.us_runtime.acs_pums import (
    ACS_2024_1YR_SPINE,
    ACS_SOURCE_COLUMN_RENAMES,
)
from microcosm.build.us_runtime.acs_release_predictors import (
    ACS_DIFFICULTY_MIN_AGE,
    ACS_DIFFICULTY_TO_CPS,
)
from microcosm.build.us_runtime.acs_transfer import ASEC_PUF_DONOR_SPINE
from microcosm.build.us_runtime.base_pool import spine_column
from microcosm.build.us_runtime.eligibility_inputs import _SSI_DISABILITY_AGE_LIMIT
from microcosm.frame import Frame
from microcosm.frame.units import US_SCHEMA

__all__ = [
    "ACS_DISABILITY_ITEMS",
    "ACS_LOCAL_DISABILITY_COLUMNS",
    "ACS_LOCAL_WORK_DISABILITY_COLUMNS",
    "ACS_LOCAL_WORK_DISABILITY_GATE_NAME",
    "ACS_LOCAL_WORK_DISABILITY_ISSUE",
    "ACS_NATIVE_PROVENANCE",
    "WEEKS_WORKED_COLUMN",
    "WEEKS_WORKED_EXPORT_BLOCKER",
    "AcsLocalWorkDisabilityResult",
    "acs_local_work_disability_signal_gate",
    "map_acs_local_work_disability_inputs",
    "record_acs_local_work_disability_transfer",
]

ACS_LOCAL_WORK_DISABILITY_ISSUE = "microcosm#1021"
ACS_LOCAL_WORK_DISABILITY_GATE_NAME = "acs_local_work_disability_signal"
#: The provenance ``acs_inputs`` stamps on every native ACS mapping.
ACS_NATIVE_PROVENANCE = "acs_2024_1yr_native"
#: The six ACS difficulty items, named as the release predictor crosswalk
#: names them (DDRS, DEAR, DEYE, DOUT, DPHY, DREM).
ACS_DISABILITY_ITEMS: tuple[str, ...] = tuple(ACS_DIFFICULTY_TO_CPS)
#: The engine inputs this stage writes natively on every ACS row and exports.
ACS_LOCAL_DISABILITY_COLUMNS: tuple[str, ...] = ("is_disabled", "is_blind")
WEEKS_WORKED_COLUMN = "weeks_worked"
#: Everything this stage writes; ``weeks_worked`` stays in staging only.
ACS_LOCAL_WORK_DISABILITY_COLUMNS: tuple[str, ...] = (
    *ACS_LOCAL_DISABILITY_COLUMNS,
    WEEKS_WORKED_COLUMN,
)
#: What must land before ``weeks_worked`` can reach the consumer export.
WEEKS_WORKED_EXPORT_BLOCKER = "policyengine-us#9660"

_AGE = "age"
_USUAL_HOURS = "weekly_hours_worked_before_lsr"
_VISION_ITEM = "DEYE"
_SSI_AMOUNT = "SSIP"
_WEEKS_SOURCE_COLUMNS: tuple[str, ...] = ("WKWN", "WKHP", "WKL", "AGEP")
#: Census's full-year threshold: 50-52 weeks is "worked full year".
_FULL_YEAR_WEEKS = 50
_AGE_BANDS: tuple[tuple[str, float, float], ...] = (
    ("under_18", 0.0, 18.0),
    ("18_to_64", 18.0, 65.0),
    ("65_and_over", 65.0, math.inf),
)
_IS_DISABLED_DEFINITION = (
    "any of DEAR/DEYE/DOUT/DPHY/DREM/DDRS == 1, or SSIP > 0 and AGEP < "
    f"{_SSI_DISABILITY_AGE_LIMIT} (the ASEC eligibility-inputs definition with "
    "the same six items and SSI alignment)"
)
_IS_BLIND_DEFINITION = "DEYE == 1 (the ASEC reads PEDISEYE == 1)"
_ACS_MIL = ACS_SOURCE_COLUMN_RENAMES["MIL"]
_NOT_EXPORTED: Mapping[str, Mapping[str, object]] = {
    "is_veteran": {
        "status": "not_exported",
        "source_column": _ACS_MIL,
        "coding": {
            "veteran": [2],
            "not_veteran": [1, 3, 4],
            "blank": "under 17: not a veteran",
        },
        "reason": (
            "policyengine-us computes is_veteran from veterans_benefits > 0 for "
            "every year, so exporting an ACS-only value needs an engine input "
            "path first (policyengine-us#9658 military exception, "
            "policyengine-us#9662 pre-HR1 veteran exemption); ASEC rows would "
            "otherwise stay on the formula."
        ),
    },
    "is_incapable_of_self_care": {
        "status": "asec_transfer",
        "reason": (
            "Fitted jointly with the care-expense inputs on the ASEC transfer; "
            "DDRS/DREM/DPHY/DOUT are not a settled definition of incapacity "
            "for self-care."
        ),
    },
}


@dataclass(frozen=True)
class AcsLocalWorkDisabilityResult:
    """The mapped ACS frame, its native-input receipts and the stage receipt."""

    frame: Frame
    native_inputs: Mapping[str, Mapping[str, Any]]
    receipt: Mapping[str, Any]


@dataclass(frozen=True)
class _NativeDisability:
    is_disabled: np.ndarray
    is_blind: np.ndarray
    difficulty: np.ndarray
    ssi_increment: np.ndarray
    items: Mapping[str, Mapping[str, int]]


def _native_disability(person: pd.DataFrame) -> _NativeDisability:
    """The ACS rows' ``is_disabled``/``is_blind`` from their native items.

    Raises:
        ValueError: If a source column is absent, an age is missing or
            negative, or an item code contradicts its minimum question age.
    """

    missing = [
        column
        for column in (*ACS_DISABILITY_ITEMS, "AGEP", _SSI_AMOUNT)
        if column not in person
    ]
    if missing:
        raise ValueError(
            f"ACS native disability requires person column(s) {missing}; the "
            "pinned ACS PUMS source carries all six difficulty items."
        )
    age = pd.to_numeric(person["AGEP"], errors="coerce")
    if age.isna().any() or age.lt(0).any():
        raise ValueError("ACS native disability requires a nonnegative AGEP.")
    yes: dict[str, np.ndarray] = {}
    items: dict[str, Mapping[str, int]] = {}
    for item in ACS_DISABILITY_ITEMS:
        codes = _nullable_source_codes(person[item], minimum=1, maximum=2)
        minimum_age = ACS_DIFFICULTY_MIN_AGE[item]
        in_universe = age.ge(minimum_age)
        invalid = (in_universe & codes.isna()) | (~in_universe & codes.notna())
        if invalid.any():
            raise ValueError(
                f"ACS {item} contradicts its minimum question age {minimum_age}: "
                f"{int(invalid.sum())} row(s) are blank in universe or coded "
                "below it."
            )
        yes[item] = codes.eq(1).to_numpy(dtype=bool)
        items[item] = {
            "minimum_question_age": int(minimum_age),
            "yes_rows": int(yes[item].sum()),
            "out_of_universe_rows": int((~in_universe).sum()),
        }
    difficulty = np.column_stack([yes[item] for item in ACS_DISABILITY_ITEMS]).any(
        axis=1
    )
    ssi = pd.to_numeric(person[_SSI_AMOUNT], errors="coerce").fillna(0.0).to_numpy()
    ssi_rule = (ssi > 0) & (age.to_numpy(dtype=np.float64) < _SSI_DISABILITY_AGE_LIMIT)
    return _NativeDisability(
        is_disabled=difficulty | ssi_rule,
        is_blind=yes[_VISION_ITEM],
        difficulty=difficulty,
        ssi_increment=ssi_rule & ~difficulty,
        items=items,
    )


def map_acs_local_work_disability_inputs(frame: Frame) -> AcsLocalWorkDisabilityResult:
    """Write native ``is_disabled``, ``is_blind`` and ``weeks_worked`` on ACS rows.

    Runs on the ACS-only frame after
    :func:`~microcosm.build.us_runtime.acs_inputs.map_acs_native_inputs` and
    before the ASEC transfer. Every ACS person gets a disability value, so the
    transfer (which fills only missing cells) skips both targets.

    Returns:
        The mapped frame, native-input receipts in the ``acs_inputs`` format
        (which the staging coverage check reads), and the stage receipt. The
        receipt's ``imputed_rows`` are filled in after the transfer by
        :func:`record_acs_local_work_disability_transfer`.

    Raises:
        ValueError: If the frame is not US-schema, a source column is
            missing, a code contradicts its universe, ``WKWN`` and ``WKHP``
            are not blank together, or an output column already exists.
    """

    if frame.schema != US_SCHEMA:
        raise ValueError("ACS local work/disability inputs require the US schema.")
    if "WKWN" not in frame.table("person"):
        raise ValueError(
            "ACS local lane requires WKWN (weeks worked in the past 12 months); "
            "the pinned ACS PUMS source carries it."
        )
    weeks = map_acs_weeks_worked(frame)
    tables = {entity: weeks.frame.table(entity) for entity in weeks.frame.entities}
    person = tables["person"]
    native: dict[str, Mapping[str, Any]] = dict(weeks.native_inputs)
    disability = _native_disability(person)
    for column, values, source_columns, definition, extra in (
        (
            "is_disabled",
            disability.is_disabled,
            (*ACS_DISABILITY_ITEMS, _SSI_AMOUNT, "AGEP"),
            _IS_DISABLED_DEFINITION,
            {
                "difficulty_rows": int(disability.difficulty.sum()),
                "ssi_increment_rows": int(disability.ssi_increment.sum()),
            },
        ),
        (
            "is_blind",
            disability.is_blind,
            (_VISION_ITEM, "AGEP"),
            _IS_BLIND_DEFINITION,
            {},
        ),
    ):
        if column in person:
            raise ValueError(
                f"ACS native mapping refuses to overwrite existing column {column!r}."
            )
        person[column] = values
        native[column] = {
            "entity": "person",
            "source_columns": list(source_columns),
            "transformation": definition,
            "provenance": ACS_NATIVE_PROVENANCE,
            "observed_rows": len(values),
            "missing_rows": 0,
            "true_rows": int(values.sum()),
            **extra,
        }
    mapped = Frame(
        tables,
        frame.schema,
        {entity: frame.weights_for(entity) for entity in frame.weighted_entities},
        frame.strata,
        mass_log=frame.mass_log,
        metadata=frame.metadata,
    )
    weeks_receipt = native[WEEKS_WORKED_COLUMN]
    receipt: dict[str, Any] = {
        "issue": ACS_LOCAL_WORK_DISABILITY_ISSUE,
        "spine": ACS_2024_1YR_SPINE,
        "acs_persons": len(person),
        "is_disabled": {
            "source": ACS_NATIVE_PROVENANCE,
            "definition": _IS_DISABLED_DEFINITION,
            "native_rows": len(person),
            "true_rows": native["is_disabled"]["true_rows"],
            "difficulty_rows": native["is_disabled"]["difficulty_rows"],
            "ssi_increment_rows": native["is_disabled"]["ssi_increment_rows"],
        },
        "is_blind": {
            "source": ACS_NATIVE_PROVENANCE,
            "definition": _IS_BLIND_DEFINITION,
            "native_rows": len(person),
            "true_rows": native["is_blind"]["true_rows"],
        },
        "difficulty_items": {
            item: dict(disability.items[item]) for item in ACS_DISABILITY_ITEMS
        },
        WEEKS_WORKED_COLUMN: {
            "source": ACS_NATIVE_PROVENANCE,
            **{
                key: weeks_receipt[key]
                for key in (
                    "observed_rows",
                    "missing_rows",
                    "source_value_rows",
                    "structural_zero_rows",
                    "source_universe_unavailable_rows",
                    "source_unresolved_rows",
                    "hours_paired_rows",
                )
            },
            "exported": False,
            "export_blocker": WEEKS_WORKED_EXPORT_BLOCKER,
            "export_note": (
                "Carried on the staging ACS spine only: the pinned "
                "policyengine-us gives weeks_worked a formula_2025, so the "
                "consumer export holds it back as formula-owned until "
                f"{WEEKS_WORKED_EXPORT_BLOCKER} deletes that formula and the "
                "pin and ABI lock are bumped. Under-16 rows stay missing."
            ),
        },
        "not_exported": {key: dict(value) for key, value in _NOT_EXPORTED.items()},
    }
    return AcsLocalWorkDisabilityResult(mapped, native, receipt)


def record_acs_local_work_disability_transfer(
    receipt: Mapping[str, Any], imputed_inputs: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    """Return the receipt with how many ACS cells the transfer imputed.

    ``imputed_inputs`` is the JSON-ready imputed-input provenance of every
    transfer that ran after :func:`map_acs_local_work_disability_inputs`.
    Native-complete columns leave no missing cell to fill, so both counts
    should be zero; the gate refuses any other value.
    """

    updated: dict[str, Any] = {key: value for key, value in receipt.items()}
    for column in ACS_LOCAL_DISABILITY_COLUMNS:
        entries = [
            item
            for item in imputed_inputs
            if isinstance(item, Mapping) and item.get("column") == column
        ]
        updated[column] = {
            **dict(receipt[column]),
            "transfer_entries": len(entries),
            "imputed_rows": int(
                sum(int(item.get("imputed_recipient_rows") or 0) for item in entries)
            ),
        }
    return updated


def _ages(rows: pd.DataFrame) -> np.ndarray:
    source = rows[_AGE] if _AGE in rows else rows.get("AGEP")
    if source is None:
        return np.full(len(rows), np.nan)
    return pd.to_numeric(source, errors="coerce").to_numpy(dtype=np.float64)


def _weighted_share(weights: np.ndarray, mask: np.ndarray) -> float:
    total = float(weights.sum())
    return float(weights[mask].sum()) / total if total > 0 else 0.0


def _shares_by_age_band(
    ages: np.ndarray, weights: np.ndarray, masks: Mapping[str, np.ndarray]
) -> dict[str, dict[str, float]]:
    shares: dict[str, dict[str, float]] = {}
    for band, low, high in _AGE_BANDS:
        in_band = (ages >= low) & (ages < high)
        shares[band] = {
            "person_weight": float(weights[in_band].sum()),
            **{
                name: _weighted_share(weights[in_band], mask[in_band])
                for name, mask in masks.items()
            },
        }
    return shares


def _flag_values(rows: pd.DataFrame, column: str) -> tuple[np.ndarray, np.ndarray]:
    values = rows[column]
    present = values.notna().to_numpy(dtype=bool)
    flags = np.zeros(len(rows), dtype=bool)
    flags[present] = values[present].astype(bool).to_numpy(dtype=bool)
    return flags, present


def _flag_failures(
    spine: str, rows: pd.DataFrame, columns: dict[str, object]
) -> list[str]:
    """Missing, incomplete, out-of-domain or constant disability flags."""

    failures: list[str] = []
    for column in ACS_LOCAL_DISABILITY_COLUMNS:
        if column not in rows:
            failures.append(
                f"{spine}: missing {column}; the engine default False removes "
                "every disability exemption."
            )
            continue
        values = rows[column]
        present = values.notna().to_numpy(dtype=bool)
        observed = values[present]
        columns[column] = {"missing_rows": int((~present).sum())}
        if not present.all():
            failures.append(
                f"{spine}: {column} has {int((~present).sum())} missing row(s); "
                "the reviewed-null fill would make them False."
            )
        outside = observed[~observed.isin([0, 1]).to_numpy(dtype=bool)]
        if len(outside):
            failures.append(
                f"{spine}: {column} has non-boolean value(s) "
                f"{sorted(map(str, outside.unique()))}."
            )
            continue
        flags = observed.astype(bool).to_numpy(dtype=bool)
        unique = int(np.unique(flags).size)
        columns[column].update({"unique_values": unique, "true_rows": int(flags.sum())})
        if unique < 2:
            only = bool(flags[0]) if flags.size else None
            failures.append(
                f"{spine}: {column} is constant {only!r}; a spine with no "
                "variation carries no disability signal."
            )
    return failures


def _native_source_failures(
    spine: str, rows: pd.DataFrame, weights: np.ndarray, entry: dict[str, object]
) -> list[str]:
    """Recompute the ACS flags from their items and grade the stored values."""

    try:
        native = _native_disability(rows)
    except ValueError as exc:
        return [f"{spine}: native disability cannot be verified: {exc}"]
    failures: list[str] = []
    mismatches: dict[str, int] = {}
    for column, expected in (
        ("is_disabled", native.is_disabled),
        ("is_blind", native.is_blind),
    ):
        if column not in rows:
            continue
        stored, present = _flag_values(rows, column)
        mismatches[column] = int((present & (stored != expected)).sum())
        if mismatches[column]:
            failures.append(
                f"{spine}: {mismatches[column]} person(s) carry a {column} that "
                "differs from their native ACS items; the ACS value must be "
                "measured, not imputed."
            )
    ages = _ages(rows)
    entry["native_mismatch_rows"] = mismatches
    entry["difficulty_items"] = {
        item: dict(native.items[item]) for item in ACS_DISABILITY_ITEMS
    }
    # Informational: the difficulty-only share is ACS DIS (the published
    # disability recode) and the SSI increment is this stage's addition.
    entry["shares_by_age_band"] = _shares_by_age_band(
        ages,
        weights,
        {
            "difficulty_share": native.difficulty,
            "ssi_increment_share": native.ssi_increment,
            "is_disabled_share": native.is_disabled,
            "is_blind_share": native.is_blind,
        },
    )
    entry["ssi_increment"] = {
        "rows": int(native.ssi_increment.sum()),
        "weighted_share": _weighted_share(weights, native.ssi_increment),
    }
    return failures


def _weeks_failures(
    spine: str, rows: pd.DataFrame, entry: dict[str, object]
) -> list[str]:
    """Native, paired and non-constant ``weeks_worked`` on the ACS spine."""

    if WEEKS_WORKED_COLUMN not in rows:
        return [
            f"{spine}: missing {WEEKS_WORKED_COLUMN}; staging must carry native "
            f"WKWN ({ACS_LOCAL_WORK_DISABILITY_ISSUE})."
        ]
    missing = [column for column in _WEEKS_SOURCE_COLUMNS if column not in rows]
    if missing:
        return [
            f"{spine}: {WEEKS_WORKED_COLUMN} cannot be verified; missing source "
            f"column(s) {missing}."
        ]
    try:
        expected, counts = acs_weeks_worked_values(rows)
    except ValueError as exc:
        return [f"{spine}: {WEEKS_WORKED_COLUMN} cannot be verified: {exc}"]
    stored = pd.to_numeric(rows[WEEKS_WORKED_COLUMN], errors="coerce").to_numpy(
        dtype=np.float64
    )
    present = ~np.isnan(stored)
    same = (stored == expected) | (np.isnan(stored) & np.isnan(expected))
    failures: list[str] = []
    mismatches = int((~same).sum())
    if mismatches:
        failures.append(
            f"{spine}: {mismatches} person(s) carry a {WEEKS_WORKED_COLUMN} that "
            "differs from their native WKWN."
        )
    observed = stored[present]
    if np.unique(observed).size < 2:
        failures.append(f"{spine}: {WEEKS_WORKED_COLUMN} is constant or empty.")
    if ((observed < 0) | (observed > 52)).any():
        failures.append(f"{spine}: {WEEKS_WORKED_COLUMN} has value(s) outside [0, 52].")
    unpaired = 0
    if _USUAL_HOURS in rows:
        hours = pd.to_numeric(rows[_USUAL_HOURS], errors="coerce").to_numpy(
            dtype=np.float64
        )
        unpaired = int(
            (present & (np.isnan(hours) | ((stored > 0) != (hours > 0)))).sum()
        )
        if unpaired:
            failures.append(
                f"{spine}: {unpaired} person(s) have weeks worked without usual "
                "hours, or usual hours without weeks worked."
            )
    else:
        failures.append(
            f"{spine}: {_USUAL_HOURS} is missing; weeks worked cannot be paired."
        )
    workers = present & (stored > 0)
    entry[WEEKS_WORKED_COLUMN] = {
        "observed_rows": int(present.sum()),
        "missing_rows": int((~present).sum()),
        "native_mismatch_rows": mismatches,
        "hours_unpaired_rows": unpaired,
        "worker_rows": int(workers.sum()),
        "part_year_worker_rows": int((workers & (stored < _FULL_YEAR_WEEKS)).sum()),
        **counts,
    }
    return failures


def _receipt_failures(
    receipt: object, acs_rows: int, details: dict[str, object]
) -> list[str]:
    """The staging receipt must show native, unimputed ACS disability values."""

    if not isinstance(receipt, Mapping) or (
        receipt.get("issue") != ACS_LOCAL_WORK_DISABILITY_ISSUE
    ):
        details["receipt"] = {"present": False}
        return [
            "No acs_local_work_disability staging receipt "
            f"({ACS_LOCAL_WORK_DISABILITY_ISSUE}); the ACS disability values "
            "cannot be shown to be native. Re-run staging with the current builder."
        ]
    failures: list[str] = []
    summary: dict[str, object] = {"present": True}
    if receipt.get("acs_persons") != acs_rows:
        failures.append(
            f"receipt: records {receipt.get('acs_persons')!r} ACS person(s) but "
            f"the frame has {acs_rows}."
        )
    for column in ACS_LOCAL_DISABILITY_COLUMNS:
        entry = receipt.get(column)
        if not isinstance(entry, Mapping):
            failures.append(f"receipt: no {column} entry.")
            continue
        summary[column] = {
            "source": entry.get("source"),
            "imputed_rows": entry.get("imputed_rows"),
        }
        if entry.get("source") != ACS_NATIVE_PROVENANCE:
            failures.append(
                f"receipt: {column} source is {entry.get('source')!r}, not "
                f"{ACS_NATIVE_PROVENANCE!r}."
            )
        imputed = entry.get("imputed_rows")
        if type(imputed) is not int:
            failures.append(f"receipt: {column} records no transfer imputation count.")
        elif imputed:
            failures.append(
                f"receipt: the ASEC transfer imputed {imputed} ACS {column} cell(s)."
            )
    details["receipt"] = summary
    return failures


def acs_local_work_disability_signal_gate(
    frame: Frame,
    *,
    receipt: Mapping[str, Any] | None,
    require_weeks_worked: bool = True,
) -> GateResult:
    """Require complete, native disability inputs and staged weeks worked.

    For each spine, fails when ``is_disabled`` or ``is_blind`` is missing, has
    missing cells, is not boolean, or is constant. On the ACS spine it
    recomputes both flags from the native items and fails on any
    disagreement, and it fails unless ``receipt`` (the staging receipt) shows
    a native source with zero transfer-imputed cells. With
    ``require_weeks_worked`` (staging), the ACS spine must also carry
    ``weeks_worked`` equal to its native ``WKWN`` mapping, non-constant, and
    paired with usual hours. The finalize run passes ``False``: the consumer
    export holds ``weeks_worked`` back until ``WEEKS_WORKED_EXPORT_BLOCKER``.

    Details report, informationally, the ACS difficulty-only share (ACS DIS),
    the SSI increment and the ``is_disabled``/``is_blind`` shares by age band
    (under 18, 18-64, 65 and over), and each spine's ``is_disabled`` share by
    band.
    """

    person = frame.table("person")
    tag = spine_column("person")
    failures: list[str] = []
    by_spine: dict[str, object] = {}
    details: dict[str, object] = {"per_spine": by_spine}
    if tag not in person or person[tag].isna().any():
        failures.append(f"Missing person origin tags: {tag}.")
        return GateResult(
            name=ACS_LOCAL_WORK_DISABILITY_GATE_NAME,
            passed=False,
            failures=tuple(failures),
            details=details,
        )
    weights = np.asarray(frame.resolve_weights("person").values, dtype=np.float64)
    acs_rows = 0
    for spine in (ASEC_PUF_DONOR_SPINE, ACS_2024_1YR_SPINE):
        selected = person[tag].eq(spine).to_numpy(dtype=bool)
        graded = spine == ACS_2024_1YR_SPINE
        columns: dict[str, object] = {}
        entry: dict[str, object] = {
            "rows": int(selected.sum()),
            "graded": graded,
            "columns": columns,
        }
        by_spine[spine] = entry
        if graded:
            acs_rows = int(selected.sum())
        if not selected.any():
            failures.append(f"{spine}: no person rows.")
            continue
        rows = person.loc[selected]
        spine_failures = _flag_failures(spine, rows, columns)
        failures += spine_failures
        if "is_disabled" in rows and not spine_failures:
            flags, _ = _flag_values(rows, "is_disabled")
            entry["is_disabled_share_by_age_band"] = _shares_by_age_band(
                _ages(rows), weights[selected], {"is_disabled_share": flags}
            )
        if graded:
            failures += _native_source_failures(spine, rows, weights[selected], entry)
            if require_weeks_worked:
                failures += _weeks_failures(spine, rows, entry)
        elif WEEKS_WORKED_COLUMN in rows:
            # The pool contract projects the donor's weeks_worked away, so the
            # donor spine is reported, not graded.
            entry[WEEKS_WORKED_COLUMN] = {
                "observed_rows": int(rows[WEEKS_WORKED_COLUMN].notna().sum())
            }
    if not require_weeks_worked:
        details[WEEKS_WORKED_COLUMN] = {
            "evaluated": False,
            "reason": (
                "The consumer export holds weeks_worked back as formula-owned "
                f"under the pinned policyengine-us until "
                f"{WEEKS_WORKED_EXPORT_BLOCKER}; staging graded it."
            ),
        }
    if set(person[tag].unique()) - {ASEC_PUF_DONOR_SPINE, ACS_2024_1YR_SPINE}:
        failures.append("Work/disability origin tags contain an unsupported spine.")
    failures += _receipt_failures(receipt, acs_rows, details)
    return GateResult(
        name=ACS_LOCAL_WORK_DISABILITY_GATE_NAME,
        passed=not failures,
        failures=tuple(failures),
        details=details,
    )
