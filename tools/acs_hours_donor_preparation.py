#!/usr/bin/env python3
"""Prepare age-15 hours donors/recipients from checked numeric projections.

Selection only: no H5/object decoding, imputation, weights or country calls.
The ASEC role relies on published Build P lineage, not a fresh raw-ASEC join.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

PERSON_FIELDS = (
    "person_id",
    "person_household_id",
    "person_source_id",
    "source_year",
    "source_household_id",
    "source_row_id",
    "person_support_clone_index",
    "A_AGE",
    "HRSWK",
    "WKSWORK",
    "A_HRS1",
    "SPORDER",
    "AGEP",
    "WAGP",
    "SEMP",
)
HOUSEHOLD_FIELDS = ("household_id", "household_support_clone_index")
ACS_KEYS = (
    "person_id",
    "person_household_id",
    "source_year",
    "source_household_id",
    "source_row_id",
    "SPORDER",
)
ACS_HOURS = ("WKHP", "WKL", "FWKHP")
YEAR_POLICIES = ("pooled_2022_2024", "source_2024_only")


def _require(condition: Any, message: str) -> None:
    if not np.all(condition):
        raise ValueError(message)


def _numeric_table(array: np.ndarray, required: Sequence[str]) -> None:
    _require(isinstance(array, np.ndarray), "Expected a numeric structured array")
    _require(
        array.ndim == 1 and array.dtype.names is not None, "Expected a numeric table"
    )
    _require(set(required) <= set(array.dtype.names), "Missing required numeric fields")
    for name in array.dtype.names:
        dtype = array.dtype.fields[name][0]
        _require(
            dtype.kind in "iuf" and dtype.subdtype is None,
            "Only scalar numeric fields allowed",
        )
        _require(not np.isinf(array[name]).any(), "Infinite numeric input: " + name)


def _integers(
    values: np.ndarray, name: str, low: int = 0, high: int = 2**53 - 1
) -> np.ndarray:
    _require(np.isfinite(values), "Missing integer field: " + name)
    _require((values >= low) & (values <= high), "Integer domain: " + name)
    _require(values == np.floor(values), "Nonintegral field: " + name)
    return values.astype(np.int64, copy=False)


def _unique(values: np.ndarray, name: str) -> None:
    _require(len(np.unique(values)) == len(values), "Duplicate " + name)


def _unique_columns(columns: Sequence[np.ndarray], name: str) -> None:
    _unique(np.rec.fromarrays(columns), name)


def _append_fields(
    person: np.ndarray, rows: np.ndarray, extra: Mapping[str, np.ndarray]
) -> np.ndarray:
    _require(
        not set(extra) & set(person.dtype.names), "Output field already exists in input"
    )
    dtype = np.dtype(
        person.dtype.descr + [(name, values.dtype) for name, values in extra.items()]
    )
    output = np.empty(len(rows), dtype=dtype)
    for name in person.dtype.names:
        output[name] = person[name][rows]
    for name, values in extra.items():
        output[name] = values
    return output


def prepare(
    person: np.ndarray,
    household: np.ndarray,
    acs: np.ndarray,
    *,
    recovery_receipt: Mapping[str, Any],
    donor_year_policy: str,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Validate full numeric tables and select age-15 records without recoding.

    This pure core trusts the caller's accepted full-recovery receipt and the
    numeric projection's provenance. ``prepare_files`` additionally checks all
    declared file digests. Extra numeric person columns are preserved verbatim;
    callers can retain potential predictors without committing to a model.
    """
    _require(donor_year_policy in YEAR_POLICIES, "Unknown donor year policy")
    parent_digest = recovery_receipt.get("parent_sha256")
    _require(
        isinstance(parent_digest, str)
        and len(parent_digest) == 64
        and set(parent_digest) <= set("0123456789abcdef"),
        "Invalid parent SHA256",
    )
    _numeric_table(person, PERSON_FIELDS)
    _numeric_table(household, HOUSEHOLD_FIELDS)
    _numeric_table(acs, ("parent_person_row", *ACS_KEYS, *ACS_HOURS))
    _require(
        len(person) > 0 and len(household) > 0 and len(acs) > 0,
        "Empty preparation input",
    )
    _require(
        recovery_receipt.get("protocol") == "acs-numeric-source-recovery-v1"
        and recovery_receipt.get("mode") == "full"
        and recovery_receipt.get("full_source_bijection") is True,
        "Accepted full ACS recovery required; pilots cannot define the donor complement",
    )
    _require(
        recovery_receipt.get("selected_people") == len(acs),
        "ACS count differs from receipt",
    )
    _require(
        recovery_receipt.get("parent_rows_checked_for_unique_current_ids")
        == {"person": len(person), "household": len(household)},
        "Complete parent projection required",
    )
    ids = _integers(person["person_id"], "person_id")
    hh_ids = _integers(household["household_id"], "household_id")
    _unique(ids, "person_id")
    _unique(hh_ids, "household_id")
    membership = _integers(person["person_household_id"], "person_household_id")
    source_year = _integers(person["source_year"], "source_year", 2022, 2024)
    source_row = _integers(person["source_row_id"], "source_row_id")
    source_hh = _integers(person["source_household_id"], "source_household_id")
    _integers(person["person_source_id"], "person_source_id")
    clone = _integers(
        person["person_support_clone_index"], "person_support_clone_index", 0, 1
    )
    hh_clone = _integers(
        household["household_support_clone_index"],
        "household_support_clone_index",
        0,
        1,
    )

    hh_order = np.argsort(hh_ids)
    positions = np.searchsorted(hh_ids[hh_order], membership)
    _require(positions < len(hh_ids), "Orphan person household")
    hh_index = hh_order[positions]
    _require(hh_ids[hh_index] == membership, "Orphan person household")
    _require(hh_clone[hh_index] == clone, "Person/household clone disagreement")
    used, first = np.unique(hh_index, return_index=True)
    _require(len(used) == len(household), "Household without members")
    _require(
        source_year == source_year[first][hh_index],
        "Household source year disagreement",
    )
    _require(
        source_hh == source_hh[first][hh_index],
        "Household source identity disagreement",
    )

    rows = _integers(
        acs["parent_person_row"], "ACS parent_person_row", 0, len(person) - 1
    )
    _require(np.diff(rows) > 0, "ACS rows must be unique and in parent order")
    is_acs = np.zeros(len(person), dtype=bool)
    is_acs[rows] = True
    for name in ACS_KEYS:
        source_values = _integers(acs[name], "ACS " + name)
        _require(source_values == person[name][rows], "ACS identity mismatch: " + name)
    _require(
        is_acs == np.isfinite(person["SPORDER"]), "ACS/SPORDER coverage disagreement"
    )
    _require(is_acs == is_acs[first][hh_index], "Mixed ACS/donor household")
    _require(source_year[rows] == 2024, "ACS source year must be 2024")
    _require(clone[rows] == 0, "ACS clone must be zero")
    _require(
        person["person_source_id"][rows] == source_row[rows],
        "ACS source ID disagreement",
    )
    _require(acs["SPORDER"] >= 1, "Invalid ACS SPORDER")
    _require(
        np.sort(source_row[rows]) == np.arange(len(acs)),
        "Incomplete ACS source-row coverage",
    )
    _unique_columns(
        [source_year[rows], source_hh[rows], acs["SPORDER"]], "ACS composite key"
    )
    _require(
        len(np.unique(hh_index[rows])) == recovery_receipt.get("selected_households"),
        "ACS household count differs from receipt",
    )
    _unique_columns(
        [is_acs[first], source_year[first], source_hh[first], hh_clone],
        "household source identity",
    )

    non_acs = ~is_acs
    _require(np.isnan(person["SPORDER"][non_acs]), "Unexpected donor SPORDER")
    _unique_columns(
        [source_year[non_acs], source_row[non_acs], clone[non_acs]],
        "donor source-year/row/clone",
    )
    _integers(person["A_AGE"][non_acs], "ASEC A_AGE", 0, 85)
    _require(
        (person["A_AGE"][non_acs] <= 80) | (person["A_AGE"][non_acs] == 85),
        "Invalid ASEC topcoded age",
    )
    acs_age = _integers(person["AGEP"][rows], "ACS AGEP", 0, 99)
    _require(person["A_AGE"][rows] == acs_age, "ACS source age disagreement")
    # CPS ASEC 2022/2023/2024 dictionaries: retain the -1 A_HRS1 NIU code.
    for field, low, high in (("HRSWK", 0, 99), ("WKSWORK", 0, 52), ("A_HRS1", -1, 99)):
        _integers(person[field][non_acs], "ASEC " + field, low, high)
    _require(
        (person["HRSWK"][non_acs] == 0) == (person["WKSWORK"][non_acs] == 0),
        "Incoherent annual hours/weeks zero status",
    )
    for field, low, high in (("WKHP", 1, 99), ("WKL", 1, 3), ("FWKHP", 0, 1)):
        values = acs[field]
        _integers(values[np.isfinite(values)], "ACS " + field, low, high)

    baseline = non_acs & (clone == 0)
    donor_mask = baseline & (person["A_AGE"] == 15)
    if donor_year_policy == "source_2024_only":
        donor_mask &= source_year == 2024
    donor_rows = np.flatnonzero(donor_mask)
    recipient_positions = np.flatnonzero(acs_age == 15)
    recipient_rows = rows[recipient_positions]
    _require(
        len(donor_rows) > 0 and len(recipient_rows) > 0,
        "Empty selected donor or recipient population",
    )
    donors = _append_fields(person, donor_rows, {"parent_person_row": donor_rows})
    recipients = _append_fields(
        person,
        recipient_rows,
        {
            "parent_person_row": recipient_rows,
            **{name: acs[name][recipient_positions] for name in ACS_HOURS},
        },
    )
    receipt = {
        "protocol": "acs-hours-donor-preparation-v1",
        "donor_role_basis": "published_build_p_lineage_non_acs_clone_zero",
        "fresh_raw_asec_join": False,
        "donor_year_policy": donor_year_policy,
        "donor_age": 15,
        "recipient_age": 15,
        "parent_sha256": recovery_receipt["parent_sha256"],
        "acs_linkage_basis": recovery_receipt.get("linkage_basis"),
        "original_staging_revision": recovery_receipt.get("original_staging_revision"),
        "puma_compared": recovery_receipt.get("puma_compared"),
        "parent_people": len(person),
        "parent_households": len(household),
        "acs_people_excluded_from_donors": len(acs),
        "puf_people_excluded_from_donors": int(np.sum(non_acs & (clone == 1))),
        "baseline_asec_people": int(baseline.sum()),
        "baseline_asec_age_min": int(person["A_AGE"][baseline].min()),
        "baseline_asec_age_max": int(person["A_AGE"][baseline].max()),
        "donor_people": len(donors),
        "recipient_people": len(recipients),
        "donors_by_source_year": {
            str(year): int(np.sum(donors["source_year"] == year))
            for year in (2022, 2023, 2024)
        },
        "donor_source_domains": {
            name: {"min": int(donors[name].min()), "max": int(donors[name].max())}
            for name in ("HRSWK", "WKSWORK", "A_HRS1")
        },
        "recipient_raw_missing": {
            name: int(np.isnan(recipients[name]).sum())
            for name in (*ACS_HOURS, "WAGP", "SEMP")
        },
        "person_projection_columns_preserved": list(person.dtype.names),
        "missingness_policy": "Preserve raw NaN/NIU and earnings; no filling or recoding",
        "imputation_performed": False,
        "weights_changed": False,
        "h5_changed": False,
        "country_model_called": False,
    }
    return donors, recipients, receipt


def sha256_file(path: Path) -> str:
    """Hash ordinary input/output bytes without deserializing them."""
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _read_receipt(path: Path, expected: str) -> dict[str, Any]:
    _require(path.stat().st_size <= 256 * 1024, "Oversized input receipt")
    raw = path.read_bytes()
    _require(hashlib.sha256(raw).hexdigest() == expected, "Receipt digest mismatch")
    return json.loads(raw)


def prepare_files(
    person_path: Path,
    household_path: Path,
    acs_path: Path,
    recovery_path: Path,
    projection_path: Path,
    output_dir: Path,
    *,
    expected_recovery_sha256: str,
    expected_projection_sha256: str,
    donor_year_policy: str,
) -> dict[str, Any]:
    """Check accepted receipt pins and write new local numeric sidecars.

    The projection receipt is the upstream extractor's declaration of
    ``parent_sha256``, ``person_sha256`` and ``household_sha256``. This tool
    validates those bindings; it does not independently reopen the parent H5.
    ``np.load(..., allow_pickle=False)`` is mandatory for all consumers too.
    """
    _require(not output_dir.exists(), "Output directory already exists")
    recovery = _read_receipt(recovery_path, expected_recovery_sha256)
    projection = _read_receipt(projection_path, expected_projection_sha256)
    _require(
        projection["parent_sha256"] == recovery["parent_sha256"],
        "Parent identity mismatch",
    )
    pins = (
        (person_path, projection["person_sha256"]),
        (household_path, projection["household_sha256"]),
        (acs_path, recovery["sidecar_sha256"]),
    )
    for path, expected in pins:
        _require(sha256_file(path) == expected, "Numeric input digest mismatch")
    _require(
        acs_path.stat().st_size == recovery["sidecar_bytes"],
        "ACS sidecar size mismatch",
    )
    arrays = [np.load(path, mmap_mode="r", allow_pickle=False) for path, _ in pins]
    donors, recipients, receipt = prepare(
        *arrays, recovery_receipt=recovery, donor_year_policy=donor_year_policy
    )
    for path, expected in pins:
        _require(
            sha256_file(path) == expected, "Numeric input changed during preparation"
        )
    receipt.update(
        {
            "recovery_receipt_sha256": expected_recovery_sha256,
            "projection_receipt_sha256": expected_projection_sha256,
            "numeric_projection_provenance": "trusted_upstream_extractor_declaration",
            "input_sha256": {
                "person": pins[0][1],
                "household": pins[1][1],
                "acs": pins[2][1],
            },
            "tool_sha256": sha256_file(Path(__file__)),
        }
    )
    with tempfile.TemporaryDirectory(
        prefix=".hours-preparation-", dir=output_dir.parent
    ) as tmp:
        temp = Path(tmp)
        receipt["outputs"] = {}
        for name, array in (
            ("asec-age15-donors.npy", donors),
            ("acs-age15-recipients.npy", recipients),
        ):
            path = temp / name
            np.save(path, array, allow_pickle=False)
            receipt["outputs"][name] = {
                "sha256": sha256_file(path),
                "bytes": path.stat().st_size,
            }
        (temp / "PREPARATION.json").write_text(
            json.dumps(receipt, sort_keys=True, indent=2, allow_nan=False) + "\n"
        )
        output_dir.mkdir()
        try:
            for name in (*receipt["outputs"], "PREPARATION.json"):
                os.rename(temp / name, output_dir / name)
        except BaseException:
            shutil.rmtree(output_dir)
            raise
    return receipt


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("person", "household", "acs", "recovery", "projection"):
        parser.add_argument(f"--{name}-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expected-recovery-sha256", required=True)
    parser.add_argument("--expected-projection-sha256", required=True)
    parser.add_argument("--donor-year-policy", choices=YEAR_POLICIES, required=True)
    receipt = prepare_files(**vars(parser.parse_args(argv)))
    print(
        json.dumps(
            {
                key: receipt[key]
                for key in ("donor_year_policy", "donor_people", "recipient_people")
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
