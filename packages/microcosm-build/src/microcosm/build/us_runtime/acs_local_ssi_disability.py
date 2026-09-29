"""SSI disability criteria for the ACS rows of the retained ACS local lane.

``meets_ssi_disability_criteria`` is the SSI-specific disability input the
donor release imputes from the December SIPP
(:mod:`~microcosm.build.us_runtime.ssi_disability_criteria`). The ACS local
lane never ran that stage on ACS rows: every ACS person reached the engine pass
with the column missing and the reviewed-null fill wrote the engine default,
``False`` (microcosm#1022). No one under 65 on an ACS row could then receive
SSI unless blind, and SSI feeds SNAP's unearned income, its elderly-or-disabled
definition and its categorical eligibility.

This fresh-build stage runs the same archived model on the ACS rows and fills
only missing cells: donor-spine values and any stored ACS value are kept
unchanged.

- **Model.** The archived SIPP QRF, unchanged: the same pinned full SIPP 2023
  file (:func:`load_acs_local_ssi_disability_donor`), observation and
  financial screens, nineteen predictors, weighted replacement sample and
  fixed forest seed, so the fitted forest is the donor stage's forest. The
  same post-prediction screen keeps a positive only for a person with a
  measured difficulty, Social Security disability or other disability income.
- **Predictors** come from a CPS-named view of each ACS person record
  (:data:`ACS_SSI_DISABILITY_SOURCE_COLUMNS`):

  - the six ACS difficulty items ``DDRS``/``DEAR``/``DEYE``/``DOUT``/``DPHY``/
    ``DREM`` are the ASEC ``PEDIS*`` items
    (:data:`~microcosm.build.us_runtime.acs_release_predictors.ACS_DIFFICULTY_TO_CPS`).
    Each is asked from its minimum age
    (:data:`~microcosm.build.us_runtime.acs_release_predictors.ACS_DIFFICULTY_MIN_AGE`:
    hearing and vision at all ages, cognitive, ambulatory and self-care from
    5, independent living from 15); below it the Census blank reads as "no
    difficulty", as the native ``is_disabled`` mapping reads it
    (microcosm#1021). A code that contradicts its universe is refused.
  - age, sex and wages are native (``age``, ``is_female``,
    ``employment_income_before_lsr``); the ``WAGP`` blank below 15 reads as no
    earnings, and a blank at 15 or over is refused.
  - marriage is ``A_MARITL`` 1 or 2, married with the spouse in the household,
    the SIPP ``EMS == 1`` definition the model was trained on.
  - interest, dividend and rental income, bank/stock/bond assets and Social
    Security disability come from the shared ASEC transfer (interest and
    dividends as their measured component pairs, as on the donor), and
    ``disability_benefits`` from the separate income pass (microcosm#1022),
    which must run first. A blank in any of them is refused, never read as 0.
- **Anchor.** An ACS person under 65 with ``ssi_reported`` (native ``SSIP``)
  > 0 meets the criteria: SSI requires recipients under 65 to be disabled or
  blind. It is the archived model's ASEC reporter anchor read from the
  harmonized ACS amount; ``SSIP`` is blank below 15, which reads as none.
- **Draws.** The fitted forest draws from caller uniforms: two seeded blake2b
  streams keyed on ``acs_2024_1yr:SERIALNO:SPORDER``, so a person's value
  depends only on the build seed and their own record, never on row order,
  the number of rows filled or the donor spine.

SSI take-up is not seeded here: the release tool's materialize assigns it on
the ACS rows against these criteria, after an engine pre-pass
(:mod:`~microcosm.build.us_runtime.acs_local_ssi_medicaid_take_up`,
microcosm#1022).
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from microcosm.build.gates import GateResult
from microcosm.build.us_runtime.acs_inputs import _nullable_source_codes
from microcosm.build.us_runtime.acs_pums import ACS_2024_1YR_SPINE
from microcosm.build.us_runtime.acs_release_predictors import (
    ACS_DIFFICULTY_MIN_AGE,
    ACS_DIFFICULTY_TO_CPS,
)
from microcosm.build.us_runtime.acs_transfer import ASEC_PUF_DONOR_SPINE
from microcosm.build.us_runtime.base_pool import spine_column
from microcosm.build.us_runtime.ssi_disability_criteria import (
    _ASEC_DIFFICULTY_SOURCE_COLUMNS,
    SIPP_2023_SSI_DISABILITY_DONOR_REVISION,
    SIPP_2023_SSI_DISABILITY_DONOR_SHA256,
    SIPP_2023_SSI_DISABILITY_DONOR_SIZE_BYTES,
    SIPP_2023_SSI_DISABILITY_DONOR_URL,
    SIPP_SSI_DISABILITY_FIT_PARAMETERS,
    SIPP_SSI_DISABILITY_MODEL_PREDICTORS,
    SSI_DISABILITY_ARCHIVED_SIPP_URL,
    US_SSI_DISABILITY_CRITERIA_OUTPUT_COLUMNS,
    US_SSI_DISABILITY_CRITERIA_STAGE_NAME,
    _archived_disability_signal,
    _coerce_boolean_predictions,
    _fit_ssi_disability_model,
    _person_table_ssi_disability_predictors,
    _reported_ssi_anchor,
    _ssi_disability_training_sample,
    load_sipp_2023_ssi_disability_donor,
    us_ssi_disability_criteria_stage_spec,
)
from microcosm.frame import Frame
from microcosm.frame.units import US_SCHEMA

__all__ = [
    "ACS_LOCAL_SSI_DISABILITY_COLUMN",
    "ACS_LOCAL_SSI_DISABILITY_GATE_NAME",
    "ACS_LOCAL_SSI_DISABILITY_ISSUE",
    "ACS_LOCAL_SSI_DISABILITY_METHOD",
    "ACS_LOCAL_SSI_DISABILITY_REVIEW_BAND",
    "ACS_LOCAL_SSI_DISABILITY_WORKING_AGES",
    "ACS_SSI_DISABILITY_SOURCE_COLUMNS",
    "acs_local_ssi_disability_signal_gate",
    "load_acs_local_ssi_disability_donor",
    "require_acs_local_ssi_disability_donor",
    "with_acs_local_ssi_disability_criteria",
]

ACS_LOCAL_SSI_DISABILITY_ISSUE = "microcosm#1022"
ACS_LOCAL_SSI_DISABILITY_GATE_NAME = "acs_local_ssi_disability_signal"
ACS_LOCAL_SSI_DISABILITY_COLUMN = US_SSI_DISABILITY_CRITERIA_OUTPUT_COLUMNS[0]
ACS_LOCAL_SSI_DISABILITY_METHOD = "archived_sipp_qrf_on_acs_rows_missing_cells_only"
#: Ages ``[18, 65)``: SSI's disability path is the only one open to them.
ACS_LOCAL_SSI_DISABILITY_WORKING_AGES = (18, 65)
#: Informational ACS/donor ratio band for the weighted 18-64 true share.
#: Outside it is reported for review, never a failure: the ACS universe
#: includes group-quarters residents while the ASEC's is the civilian
#: noninstitutional population, and the two spines differ in age and state mix.
ACS_LOCAL_SSI_DISABILITY_REVIEW_BAND = (0.5, 2.0)

_OUTPUT = ACS_LOCAL_SSI_DISABILITY_COLUMN
_AGE = "age"
_WAGES = "employment_income_before_lsr"
_REPORTED_SSI = "ssi_reported"
_MARITAL_STATUS = "A_MARITL"
#: CPS ``A_MARITL`` codes for married with the spouse present (civilian or
#: Armed Forces); the ACS loader sets 1 only when ``RELSHIPP`` pairs a spouse.
_MARRIED_SPOUSE_PRESENT = (1, 2)
#: ``WAGP`` is asked from age 15; below it the blank is the source universe.
_WAGES_MINIMUM_AGE = 15
#: Transferred person columns the view reads as measured amounts.
_TRANSFERRED_COLUMNS: tuple[str, ...] = (
    "taxable_interest_income",
    "tax_exempt_interest_income",
    "qualified_dividend_income",
    "non_qualified_dividend_income",
    "rental_income",
    "bank_account_assets",
    "stock_assets",
    "bond_assets",
    "social_security_disability",
    "disability_benefits",
)
#: The ACS person columns :func:`with_acs_local_ssi_disability_criteria` reads.
ACS_SSI_DISABILITY_SOURCE_COLUMNS: tuple[str, ...] = (
    "person_household_id",
    _AGE,
    "is_female",
    _MARITAL_STATUS,
    _WAGES,
    *_TRANSFERRED_COLUMNS,
    _REPORTED_SSI,
    *ACS_DIFFICULTY_TO_CPS,
    "SPORDER",
)
_PREDICTOR_BY_CPS_ITEM: Mapping[str, str] = {
    source: predictor for predictor, source in _ASEC_DIFFICULTY_SOURCE_COLUMNS.items()
}
_DRAW_KEY_FORMAT = f"{ACS_2024_1YR_SPINE}:SERIALNO:SPORDER"
_DRAW_SALT = "acs_local_ssi_disability_criteria"
_DRAW_STREAMS = ("quantile", "sign")
_MAPPING_DEFINITIONS: Mapping[str, str] = {
    "difficulty_items": (
        "ACS DDRS/DEAR/DEYE/DOUT/DPHY/DREM == 1 is the ASEC PEDIS* item == 1; "
        "a blank below the item's minimum question age reads as no difficulty"
    ),
    "age": "native ACS AGEP",
    "is_female": "native ACS SEX == 2",
    "is_married": "A_MARITL in (1, 2): married, spouse in the household (SIPP EMS == 1)",
    "employment_income": (
        "native adjusted ACS WAGP (employment_income_before_lsr); the blank "
        "below age 15 reads as 0"
    ),
    "interest_dividend_rental_income": (
        "the shared ASEC transfer's taxable/tax-exempt interest, qualified/"
        "non-qualified dividend and rental leaves"
    ),
    "assets": "the shared ASEC transfer's bank, stock and bond assets",
    "social_security_disability": "the shared ASEC transfer",
    "has_disability_income": (
        "disability_benefits > 0, from the separate local income pass"
    ),
    "reporter_anchor": "ssi_reported (adjusted ACS SSIP) > 0 and age < 65",
}


def load_acs_local_ssi_disability_donor(
    path: str | Path, *, time_period: int
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Load the pinned SIPP training frame and the identity the receipt pins.

    The byte length and sha-256 are verified against the donor stage's pin,
    so the returned identity describes the bytes actually read.
    """

    source = Path(path)
    donor = load_sipp_2023_ssi_disability_donor(
        source,
        expected_sha256=SIPP_2023_SSI_DISABILITY_DONOR_SHA256,
        expected_size_bytes=SIPP_2023_SSI_DISABILITY_DONOR_SIZE_BYTES,
        time_period=int(time_period),
    )
    return donor, {
        "path": str(source.resolve()),
        "url": SIPP_2023_SSI_DISABILITY_DONOR_URL,
        "revision": SIPP_2023_SSI_DISABILITY_DONOR_REVISION,
        "sha256": SIPP_2023_SSI_DISABILITY_DONOR_SHA256,
        "size_bytes": SIPP_2023_SSI_DISABILITY_DONOR_SIZE_BYTES,
        "time_period": int(time_period),
    }


def _boolean_cells(values: pd.Series) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``(flags, present, invalid)`` of a nullable boolean column."""

    present = values.notna().to_numpy(dtype=bool)
    valid = present & values.isin([0, 1]).to_numpy(dtype=bool)
    flags = np.zeros(len(values), dtype=bool)
    flags[valid] = values[valid].astype(bool).to_numpy(dtype=bool)
    return flags, present, present & ~valid


def _share(weights: np.ndarray, flags: np.ndarray) -> float:
    total = float(weights.sum())
    return float(weights[flags].sum()) / total if total > 0 else 0.0


def require_acs_local_ssi_disability_donor(base: Frame) -> dict[str, Any]:
    """Refuse a donor release that does not carry the criteria on every person.

    The stage keeps donor values unchanged, so a donor without a complete
    boolean column would reach the engine default ``False`` on the donor
    spine too, and the ACS/donor comparison would have no reference.
    """

    person = base.table("person")
    if _OUTPUT not in person:
        raise ValueError(
            f"The donor release lacks person column {_OUTPUT!r}; its SIPP "
            "SSI disability-criteria stage must have run "
            f"({ACS_LOCAL_SSI_DISABILITY_ISSUE})."
        )
    flags, present, invalid = _boolean_cells(person[_OUTPUT])
    if not present.all() or invalid.any():
        raise ValueError(
            f"The donor release's {_OUTPUT} has {int((~present).sum())} missing "
            f"and {int(invalid.sum())} non-boolean row(s); it must be complete."
        )
    weights = np.asarray(base.resolve_weights("person").values, dtype=np.float64)
    return {
        "person_rows": int(len(person)),
        "true_rows": int(flags.sum()),
        "weighted_true_share": _share(weights, flags),
    }


def _acs_ssi_disability_view(
    rows: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """The CPS-named person view the archived predictors read, and its receipt.

    Raises:
        ValueError: If a source column is absent, an age is missing or
            negative, a difficulty code contradicts its minimum question age,
            a transferred amount or an in-universe wage is blank, or sex or
            marital status is not coded.
    """

    missing = [
        column
        for column in ACS_SSI_DISABILITY_SOURCE_COLUMNS
        if column != "SPORDER" and column not in rows
    ]
    if missing:
        raise ValueError(
            f"ACS SSI disability criteria require person column(s) {missing} on "
            "ACS rows; the transferred ones come from the shared transfer and "
            "the local income pass."
        )
    age = pd.to_numeric(rows[_AGE], errors="coerce")
    if age.isna().any() or age.lt(0).any():
        raise ValueError("ACS SSI disability criteria require a nonnegative age.")
    view = pd.DataFrame(index=rows.index)
    view["person_household_id"] = rows["person_household_id"].to_numpy()
    view[_AGE] = age.to_numpy(dtype=np.float64)

    female = rows["is_female"]
    if female.isna().any() or not female.isin([0, 1]).all():
        raise ValueError("ACS SSI disability criteria require a boolean is_female.")
    view["is_female"] = female.astype(bool).to_numpy(dtype=np.float64)
    marital = pd.to_numeric(rows[_MARITAL_STATUS], errors="coerce")
    if marital.isna().any():
        raise ValueError(
            f"ACS SSI disability criteria require {_MARITAL_STATUS} on every row."
        )
    view["is_married"] = marital.isin(_MARRIED_SPOUSE_PRESENT).to_numpy(
        dtype=np.float64
    )

    wages = pd.to_numeric(rows[_WAGES], errors="coerce")
    in_wage_universe = age.ge(_WAGES_MINIMUM_AGE)
    blank_in_universe = wages.isna() & in_wage_universe
    if blank_in_universe.any():
        raise ValueError(
            f"{int(blank_in_universe.sum())} ACS person(s) aged "
            f"{_WAGES_MINIMUM_AGE}+ have a blank {_WAGES}."
        )
    view[_WAGES] = wages.fillna(0.0).to_numpy(dtype=np.float64)
    for column in _TRANSFERRED_COLUMNS:
        values = pd.to_numeric(rows[column], errors="coerce")
        blank = int(values.isna().sum())
        if blank:
            raise ValueError(
                f"{blank} ACS person(s) have a blank transferred {column}; the "
                "transfer must fill every ACS row before this stage runs."
            )
        view[column] = values.to_numpy(dtype=np.float64)

    items: dict[str, dict[str, Any]] = {}
    for item, target in ACS_DIFFICULTY_TO_CPS.items():
        codes = _nullable_source_codes(rows[item], minimum=1, maximum=2)
        minimum_age = ACS_DIFFICULTY_MIN_AGE[item]
        in_universe = age.ge(minimum_age)
        invalid = (in_universe & codes.isna()) | (~in_universe & codes.notna())
        if invalid.any():
            raise ValueError(
                f"ACS {item} contradicts its minimum question age {minimum_age}: "
                f"{int(invalid.sum())} row(s) are blank in universe or coded "
                "below it."
            )
        yes = codes.eq(1).to_numpy(dtype=bool)
        # 2 is the ASEC "no"; a blank below the question age reads as no.
        view[target] = np.where(yes, 1, 2).astype(np.int16)
        items[item] = {
            "cps_item": target,
            "predictor": _PREDICTOR_BY_CPS_ITEM[target],
            "minimum_question_age": int(minimum_age),
            "yes_rows": int(yes.sum()),
            "out_of_universe_rows": int((~in_universe).sum()),
        }
    view[_REPORTED_SSI] = pd.to_numeric(rows[_REPORTED_SSI], errors="coerce").to_numpy(
        dtype=np.float64
    )
    return view, {
        "definitions": dict(_MAPPING_DEFINITIONS),
        "difficulty_items": items,
        "wage_universe_zero_rows": int((wages.isna() & ~in_wage_universe).sum()),
    }


def _acs_draw_keys(household: pd.DataFrame, rows: pd.DataFrame) -> pd.Series:
    """``acs_2024_1yr:SERIALNO:SPORDER`` per ACS person; must be unique."""

    if "SERIALNO" not in household or "SPORDER" not in rows:
        raise ValueError(
            "ACS SSI disability draws are keyed on household SERIALNO and person "
            "SPORDER; the frame lacks one of them."
        )
    serial = rows["person_household_id"].map(
        household.set_index("household_id")["SERIALNO"]
    )
    order = pd.to_numeric(rows["SPORDER"], errors="coerce")
    if serial.isna().any() or order.isna().any():
        raise ValueError(
            "Every ACS person needs its household SERIALNO and its SPORDER to key "
            "the SSI disability draws."
        )
    keys = (
        f"{ACS_2024_1YR_SPINE}:"
        + serial.astype(str)
        + ":"
        + order.astype(np.int64).astype(str)
    ).reset_index(drop=True)
    if keys.duplicated().any():
        raise ValueError(
            f"{int(keys.duplicated().sum())} ACS person draw key(s) repeat; "
            f"{_DRAW_KEY_FORMAT} must identify one person."
        )
    return keys


def _keyed_uniforms(keys: Sequence[str], *, seed: int, stream: str) -> np.ndarray:
    """Seeded blake2b uniforms in ``[0, 1)``, one per key."""

    denominator = float(2**64)
    return np.asarray(
        [
            int.from_bytes(
                hashlib.blake2b(
                    f"{seed}:{_DRAW_SALT}:{stream}:{key}".encode(),
                    digest_size=8,
                ).digest(),
                byteorder="big",
                signed=False,
            )
            / denominator
            for key in keys
        ],
        dtype=np.float64,
    )


def _assignment_sha256(person_ids: np.ndarray, values: np.ndarray) -> str:
    """Digest of the ACS persons' final criteria, for reproducibility."""

    selected = pd.DataFrame(
        {"person_id": np.asarray(person_ids), "value": np.asarray(values, dtype=bool)}
    )
    hashed = pd.util.hash_pandas_object(selected, index=False).to_numpy()
    return hashlib.sha256(hashed.tobytes()).hexdigest()


def _json_ready(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def with_acs_local_ssi_disability_criteria(
    frame: Frame,
    *,
    sipp_donor: pd.DataFrame,
    seed: int,
    donor_identity: Mapping[str, Any] | None = None,
    n_estimators: int | None = None,
) -> tuple[Frame, dict[str, Any]]:
    """Fill missing ACS-row ``meets_ssi_disability_criteria`` from the SIPP model.

    Args:
        frame: The local lane's multispine US frame, after the shared transfer
            and the local income pass. Its person table must carry complete
            origin tags, the criteria column (complete on donor rows) and
            :data:`ACS_SSI_DISABILITY_SOURCE_COLUMNS` on ACS rows; the
            household table carries ``SERIALNO``.
        sipp_donor: The SIPP training frame from
            :func:`load_acs_local_ssi_disability_donor`.
        seed: The build seed for the keyed draws.
        donor_identity: The verified SIPP identity the receipt pins.
        n_estimators: Forest size; defaults to the archived 100.

    Returns:
        The frame with filled cells (the same object if none were missing)
        and a JSON-ready receipt with a digest of every ACS person's value.

    Raises:
        ValueError: If the frame is not US-schema, origin tags, the criteria
            column or a source column are missing, a stored value is not
            boolean, or an ACS predictor contradicts its universe.
    """

    if frame.schema != US_SCHEMA:
        raise ValueError("ACS SSI disability criteria require the US schema.")
    # The reused model must still be the packaged, archived contract.
    us_ssi_disability_criteria_stage_spec()
    forest_size = int(
        SIPP_SSI_DISABILITY_FIT_PARAMETERS["n_estimators"]
        if n_estimators is None
        else n_estimators
    )
    person = frame.table("person")
    tag = spine_column("person")
    if tag not in person or person[tag].isna().any():
        raise ValueError(
            f"ACS SSI disability criteria require complete origin tags: {tag}."
        )
    unknown = set(person[tag].unique()) - {ASEC_PUF_DONOR_SPINE, ACS_2024_1YR_SPINE}
    if unknown:
        raise ValueError("ACS SSI disability origin tags contain an unknown spine.")
    acs = person[tag].eq(ACS_2024_1YR_SPINE).to_numpy(dtype=bool)
    if not acs.any():
        raise ValueError("ACS SSI disability criteria found no ACS person rows.")
    if _OUTPUT not in person:
        raise ValueError(
            f"ACS SSI disability criteria require person column {_OUTPUT!r}; the "
            "donor release carries it, and without it the donor spine would "
            "reach the engine default too."
        )
    _, present, invalid = _boolean_cells(person[_OUTPUT])
    if invalid.any():
        raise ValueError(
            f"{int(invalid.sum())} stored {_OUTPUT} value(s) are not boolean."
        )
    missing = acs & ~present

    rows = person.loc[acs]
    view, mapping = _acs_ssi_disability_view(rows)
    receiver = _person_table_ssi_disability_predictors(view)
    acs_age = receiver[_AGE].to_numpy(dtype=np.float64)
    anchor = (_reported_ssi_anchor(view, age=acs_age) > 0.0) & (acs_age < 65.0)
    signal = _archived_disability_signal(receiver)
    fill = missing[acs]
    model_positive = np.zeros(len(rows), dtype=bool)
    training_rows = 0
    if fill.any():
        keys = _acs_draw_keys(frame.table("household"), rows)
        positions = np.flatnonzero(fill)
        subset_keys = keys.iloc[positions].tolist()
        training = _ssi_disability_training_sample(sipp_donor)
        training_rows = int(len(training))
        fitted = _fit_ssi_disability_model(training, n_estimators=forest_size)
        prediction = fitted.predict_from_uniforms(
            receiver.iloc[positions],
            quantiles={
                _OUTPUT: _keyed_uniforms(subset_keys, seed=int(seed), stream="quantile")
            },
            sign_uniforms={
                _OUTPUT: _keyed_uniforms(subset_keys, seed=int(seed), stream="sign")
            },
        )
        if _OUTPUT not in prediction:
            raise ValueError(f"SIPP SSI disability QRF prediction missing {_OUTPUT!r}.")
        model_positive[positions] = _coerce_boolean_predictions(
            prediction[_OUTPUT].to_numpy()
        )
    derived = (model_positive & signal) | anchor

    result = frame
    if fill.any():
        updated = person.copy(deep=False)
        filled = person[_OUTPUT].astype(object).copy()
        filled.iloc[np.flatnonzero(missing)] = derived[fill]
        updated[_OUTPUT] = filled.astype(bool) if filled.notna().all() else filled
        result = Frame(
            {
                entity: updated if entity == "person" else frame.table(entity)
                for entity in frame.entities
            },
            frame.schema,
            {entity: frame.weights_for(entity) for entity in frame.weighted_entities},
            frame.strata,
            mass_log=frame.mass_log,
            metadata=frame.metadata,
        )

    final_flags, final_present, _ = _boolean_cells(result.table("person")[_OUTPUT])
    weights = np.asarray(frame.resolve_weights("person").values, dtype=np.float64)
    low, high = ACS_LOCAL_SSI_DISABILITY_WORKING_AGES
    working = (acs_age >= low) & (acs_age < high)
    acs_flags = final_flags[acs]
    receipt: dict[str, Any] = {
        "issue": ACS_LOCAL_SSI_DISABILITY_ISSUE,
        "column": _OUTPUT,
        "method": ACS_LOCAL_SSI_DISABILITY_METHOD,
        "spine": ACS_2024_1YR_SPINE,
        "seed": int(seed),
        "draw_key": _DRAW_KEY_FORMAT,
        "draw_streams": list(_DRAW_STREAMS),
        "model": {
            "stage": US_SSI_DISABILITY_CRITERIA_STAGE_NAME,
            "archived_source": SSI_DISABILITY_ARCHIVED_SIPP_URL,
            "predictors": list(SIPP_SSI_DISABILITY_MODEL_PREDICTORS),
            "n_estimators": forest_size,
            "model_seed": SIPP_SSI_DISABILITY_FIT_PARAMETERS["model_seed"],
            "training_sample_seed": SIPP_SSI_DISABILITY_FIT_PARAMETERS[
                "training_sample_seed"
            ],
            "max_train_samples": SIPP_SSI_DISABILITY_FIT_PARAMETERS[
                "max_train_samples"
            ],
            "fitted": bool(fill.any()),
            "training_rows": training_rows,
        },
        "sipp_donor": {
            **dict(donor_identity or {}),
            "source_audit": _json_ready(sipp_donor.attrs.get("source_audit")),
        },
        "acs_persons": int(acs.sum()),
        "filled_rows": int(missing.sum()),
        "preserved_acs_rows": int((acs & present).sum()),
        "unfilled_acs_rows": int((acs & ~final_present).sum()),
        "predictor_mapping": mapping,
        "outcome": {
            "model_positive_rows": int((model_positive & fill).sum()),
            "screened_out_rows": int((model_positive & ~signal & fill).sum()),
            "reporter_anchor_rows": int((anchor & fill).sum()),
            "filled_true_rows": int((derived & fill).sum()),
            "acs_true_rows": int(acs_flags.sum()),
            "acs_weighted_true_share": _share(weights[acs], acs_flags),
            "acs_weighted_true_share_18_64": _share(
                weights[acs][working], acs_flags[working]
            ),
        },
        "assigned_sha256": _assignment_sha256(
            rows["person_id"].to_numpy()
            if "person_id" in rows
            else np.arange(len(rows)),
            acs_flags,
        ),
    }
    return result, receipt


def _receipt_failures(
    receipt: object, acs_rows: int, details: dict[str, object]
) -> list[str]:
    """The staging receipt must show the SIPP-pinned pass with no gaps."""

    if (
        not isinstance(receipt, Mapping)
        or receipt.get("issue") != ACS_LOCAL_SSI_DISABILITY_ISSUE
        or receipt.get("column") != _OUTPUT
    ):
        details["receipt"] = {"present": False}
        return [
            "No acs_local_ssi_disability staging receipt "
            f"({ACS_LOCAL_SSI_DISABILITY_ISSUE}); the ACS rows' {_OUTPUT} cannot "
            "be shown to be imputed. Re-run staging with the current builder."
        ]
    failures: list[str] = []
    if receipt.get("method") != ACS_LOCAL_SSI_DISABILITY_METHOD:
        failures.append(f"receipt: method {receipt.get('method')!r} is not the pass.")
    if receipt.get("acs_persons") != acs_rows:
        failures.append(
            f"receipt: records {receipt.get('acs_persons')!r} ACS person(s) but "
            f"the frame has {acs_rows}."
        )
    if type(receipt.get("filled_rows")) is not int:
        failures.append("receipt: no filled-row count.")
    if receipt.get("unfilled_acs_rows") != 0:
        failures.append(
            f"receipt: left {receipt.get('unfilled_acs_rows')!r} ACS row(s) unfilled."
        )
    donor = receipt.get("sipp_donor")
    if (
        not isinstance(donor, Mapping)
        or donor.get("sha256") != SIPP_2023_SSI_DISABILITY_DONOR_SHA256
    ):
        failures.append(
            "receipt: the SIPP donor is not the pinned full SIPP 2023 file "
            f"(sha256 {SIPP_2023_SSI_DISABILITY_DONOR_SHA256})."
        )
    details["receipt"] = {"present": True, "failures": len(failures)}
    return failures


def acs_local_ssi_disability_signal_gate(
    frame: Frame, *, receipt: Mapping[str, Any] | None
) -> GateResult:
    """Require a complete, non-constant criteria surface on the ACS rows.

    Fails when the column is absent, has a missing or non-boolean cell on
    either spine (the reviewed-null fill would make it ``False``), is constant
    or has a zero weighted true share among ACS persons aged 18-64, or unless
    ``receipt`` shows the SIPP-pinned pass with no unfilled ACS row. Details
    report, per spine, the weighted true share overall and at 18-64; the ACS/
    donor 18-64 ratio against :data:`ACS_LOCAL_SSI_DISABILITY_REVIEW_BAND`; and
    the ACS persons under 65 who report SSI but do not meet the criteria
    (informational only).
    """

    person = frame.table("person")
    tag = spine_column("person")
    low, high = ACS_LOCAL_SSI_DISABILITY_WORKING_AGES
    failures: list[str] = []
    per_spine: dict[str, Any] = {}
    details: dict[str, object] = {
        "column": _OUTPUT,
        "per_spine": per_spine,
        "working_ages": [low, high - 1],
        "review_band": list(ACS_LOCAL_SSI_DISABILITY_REVIEW_BAND),
    }
    if tag not in person or person[tag].isna().any():
        failures.append(f"Missing person origin tags: {tag}.")
        return GateResult(
            name=ACS_LOCAL_SSI_DISABILITY_GATE_NAME,
            passed=False,
            failures=tuple(failures),
            details=details,
        )
    if set(person[tag].unique()) - {ASEC_PUF_DONOR_SPINE, ACS_2024_1YR_SPINE}:
        failures.append("SSI disability origin tags contain an unsupported spine.")
    acs = person[tag].eq(ACS_2024_1YR_SPINE).to_numpy(dtype=bool)
    acs_rows = int(acs.sum())
    if _OUTPUT not in person or _AGE not in person:
        absent = [column for column in (_OUTPUT, _AGE) if column not in person]
        failures.append(
            f"Missing person column(s) {absent}; the engine default False fails "
            "every ACS person under 65 who is not blind."
        )
        failures += _receipt_failures(receipt, acs_rows, details)
        return GateResult(
            name=ACS_LOCAL_SSI_DISABILITY_GATE_NAME,
            passed=False,
            failures=tuple(failures),
            details=details,
        )
    weights = np.asarray(frame.resolve_weights("person").values, dtype=np.float64)
    age = pd.to_numeric(person[_AGE], errors="coerce").to_numpy(dtype=np.float64)
    working = (age >= low) & (age < high)
    flags, present, invalid = _boolean_cells(person[_OUTPUT])
    for spine in (ASEC_PUF_DONOR_SPINE, ACS_2024_1YR_SPINE):
        selected = person[tag].eq(spine).to_numpy(dtype=bool)
        entry: dict[str, Any] = {"rows": int(selected.sum())}
        per_spine[spine] = entry
        if not selected.any():
            failures.append(f"{spine}: no person rows.")
            continue
        entry["missing_rows"] = int((selected & ~present).sum())
        entry["invalid_rows"] = int((selected & invalid).sum())
        if entry["missing_rows"]:
            failures.append(
                f"{spine}: {_OUTPUT} has {entry['missing_rows']} missing row(s); "
                "the reviewed-null fill would make them False."
            )
        if entry["invalid_rows"]:
            failures.append(
                f"{spine}: {_OUTPUT} has {entry['invalid_rows']} non-boolean row(s)."
            )
        valid = selected & present & ~invalid
        band = valid & working
        entry.update(
            {
                "true_rows": int((valid & flags).sum()),
                "weighted_true_share": _share(weights[valid], flags[valid]),
                "working_age_rows": int(band.sum()),
                "working_age_unique_values": int(np.unique(flags[band]).size),
                "weighted_true_share_18_64": _share(weights[band], flags[band]),
            }
        )
    donor_entry = per_spine[ASEC_PUF_DONOR_SPINE]
    acs_entry = per_spine[ACS_2024_1YR_SPINE]
    if "working_age_rows" in acs_entry:
        if not acs_entry["working_age_rows"]:
            failures.append(f"{ACS_2024_1YR_SPINE}: no valid persons aged 18-64.")
        else:
            if acs_entry["working_age_unique_values"] < 2:
                failures.append(
                    f"{ACS_2024_1YR_SPINE}: {_OUTPUT} is constant among persons "
                    "aged 18-64; the surface carries no disability signal."
                )
            if not acs_entry["weighted_true_share_18_64"] > 0:
                failures.append(
                    f"{ACS_2024_1YR_SPINE}: no weighted person aged 18-64 meets "
                    f"{_OUTPUT}; no one under 65 could receive SSI unless blind."
                )
    donor_share = donor_entry.get("weighted_true_share_18_64")
    acs_share = acs_entry.get("weighted_true_share_18_64")
    ratio = (
        acs_share / donor_share
        if donor_share is not None and acs_share is not None and donor_share > 0
        else None
    )
    review_low, review_high = ACS_LOCAL_SSI_DISABILITY_REVIEW_BAND
    details["comparison"] = {
        "weighted_true_share_18_64_ratio": ratio,
        "within_review_band": ratio is not None and review_low <= ratio <= review_high,
    }
    reporters: dict[str, Any] = {"available": _REPORTED_SSI in person}
    if _REPORTED_SSI in person:
        reported = pd.to_numeric(person[_REPORTED_SSI], errors="coerce").to_numpy(
            dtype=np.float64
        )
        anchored = acs & (np.nan_to_num(reported, nan=0.0) > 0) & (age < 65)
        unmet = anchored & present & ~invalid & ~flags
        reporters.update(
            {
                "under_65_reporters": int(anchored.sum()),
                "without_criteria": int(unmet.sum()),
                "weighted_without_criteria": float(weights[unmet].sum()),
            }
        )
    details["acs_under_65_ssi_reporters"] = reporters
    failures += _receipt_failures(receipt, acs_rows, details)
    return GateResult(
        name=ACS_LOCAL_SSI_DISABILITY_GATE_NAME,
        passed=not failures,
        failures=tuple(failures),
        details=details,
    )
