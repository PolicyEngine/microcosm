"""Restore an observed person total from closed, authenticated Census members.

P (v4) and H (household attachment) remain immutable. T appends one nominal
observation. A candidate T digest is a locator identity, never source authority:
verification reconstructs all of T from authenticated parents and CSV members.
"""

from __future__ import annotations

import csv
import hashlib
import os
import tempfile
from collections.abc import Mapping
from dataclasses import dataclass
from importlib import resources
from pathlib import Path

import numpy as np
import pandas as pd

from microcosm.build import frame_checkpoint as checkpoint
from microcosm.frame import Frame, Weights

from . import asec_current_money as money_schema
from . import asec_current_money_source as legacy
from .asec_current_money import MoneyRefusalError, _json, _require, _sha
from .education_assistance_source import ASEC_EDUCATION_ASSISTANCE_ARCHIVES

CHECKPOINT_FILENAME = "asec_person_income_source.checkpoint.h5"
ARTIFACT_KIND = "microcosm.asec_person_income_observation_attachment"
OBSERVED_COLUMN = "asec_PTOTVAL"
_READ_COLUMNS = ("PERIDNUM", "PH_SEQ", "A_LINENO", "A_AGE", "PTOTVAL")
# Closed code-owned source pins. No public caller pin authority exists.
_MEMBER_PINS = tuple(
    (year, p.member, p.zip_sha256, p.member_sha256, p.rows, p.member_size_bytes)
    for year, p in sorted(ASEC_EDUCATION_ASSISTANCE_ARCHIVES.items())
)
_ATTACHMENT_MAX_BYTES = 2_000_000_000
_ENCODING_CONTRACT = "independently_reconstructed_canonical_checkpoint_bytes_v1"


def _implementation() -> str:
    return _sha(
        _json(
            {
                "schema": 1,
                "legacy_verification": legacy._verification_identity(),
                "schema_module": _sha(
                    resources.files(__package__)
                    .joinpath("asec_current_money.py")
                    .read_bytes()
                ),
                "source_kind": money_schema.RESTORED_SOURCE_KIND,
                "field_columns": dict(money_schema.RESTORED_FIELD_COLUMNS),
                "field_zero_policy": dict(money_schema.RESTORED_FIELD_ZERO_POLICY),
                "reader_producer": _sha(
                    resources.files(__package__)
                    .joinpath("asec_person_income_source.py")
                    .read_bytes()
                ),
                "member_pins": _MEMBER_PINS,
                "columns": _READ_COLUMNS,
                "output": OBSERVED_COLUMN,
                "origin": "authenticated_census_csv_encoded_zero_v1",
                "encoding_contract": _ENCODING_CONTRACT,
            }
        )
    )


def _paths(member_paths):
    _require(
        isinstance(member_paths, Mapping)
        and set(member_paths) == {2022, 2023, 2024}
        and all(type(y) is int for y in member_paths)
        and all(isinstance(p, (str, Path)) for p in member_paths.values()),
        "MEMBER_PATHS",
    )


def _read_member(path: Path, *, rows: int, size: int) -> pd.DataFrame:
    _require(path.stat().st_size == size, "MEMBER_SIZE")
    with path.open("r", encoding="utf-8", newline="") as handle:
        header = next(csv.reader(handle), [])
    _require(
        len(header) == len(set(header)) and set(_READ_COLUMNS) <= set(header),
        "MEMBER_HEADER",
    )
    # Preserve source tokens until validation. In particular, never fillna and
    # never let permissive numeric coercion turn absent observations into zeros.
    table = pd.read_csv(
        path, usecols=list(_READ_COLUMNS), dtype="string", na_filter=False
    )
    _require(len(table) == rows, "MEMBER_ROWS")
    _require(
        bool(table.PERIDNUM.str.fullmatch(r"[0-9]{22}").all()), "MEMBER_PERSON_KEY"
    )
    for column in _READ_COLUMNS[1:]:
        _require(
            bool(table[column].str.fullmatch(r"-?[0-9]+").all()),
            "MEMBER_INTEGER" if column == "PTOTVAL" else f"MEMBER_INTEGER_{column}",
            column,
        )
        table[column] = table[column].astype("int64")
    _require(
        bool(table.PTOTVAL.between(-99999, 99999999).all()),
        "MEMBER_AMOUNT_DOMAIN",
        "PTOTVAL",
    )
    _require(
        not table.PERIDNUM.duplicated().any()
        and not table.duplicated(["PH_SEQ", "A_LINENO"]).any(),
        "MEMBER_DUPLICATE_KEY",
    )
    return table


def _owned_frame(parent: Frame) -> Frame:
    strata = parent.strata.copy(deep=True)
    strata.index = legacy._owned_index(strata.index)
    result = Frame(
        {
            **{e: parent.table(e) for e in parent.entities},
            **{n: parent.link(n) for n in parent.links},
        },
        parent.schema,
        {
            e: Weights(parent.weights_for(e).values, parent.weights_for(e).kind)
            for e in parent.weighted_entities
        },
        strata,
        mass_log=parent.mass_log,
        metadata=parent.metadata,
    )
    legacy._detach_frame_axes(result)
    return result


def _reconstruct(source, member_paths: Mapping[int, str | Path]):
    _paths(member_paths)
    source.validate()
    before = _implementation()
    parent = source.frame
    person = parent.person
    _require(OBSERVED_COLUMN not in person, "RESTORATION_ALREADY_ATTACHED")
    _require(
        all(
            c in person for c in ("source_household_id", "A_LINENO", "A_AGE", "PTOTVAL")
        ),
        "RESTORATION_SOURCE_COLUMNS",
    )
    for c in ("source_household_id", "A_LINENO", "A_AGE"):
        _require(
            person[c].dtype == np.dtype("int64") and not person[c].isna().any(),
            "RESTORATION_COORDINATE_DTYPE",
        )
    years = np.asarray(source.scope.person_years, dtype=np.int64)
    keys = np.asarray(source.scope.person_native_keys)
    output = np.empty(len(person), dtype=np.int64)
    joins = []
    with tempfile.TemporaryDirectory(
        prefix="microcosm-person-income-members-"
    ) as directory:
        for year, member, archive_pin, pin, rows, size in _MEMBER_PINS:
            staged = Path(directory) / member
            legacy._stage_verified(member_paths[year], pin, staged)
            raw = _read_member(staged, rows=rows, size=size)
            staged.unlink()
            positions = np.flatnonzero(years == year)
            _require(len(positions) == rows, "RESTORATION_COHORT_ROWS")
            recipient_keys = pd.Index(keys[positions])
            _require(not recipient_keys.has_duplicates, "RESTORATION_PERSON_KEY")
            indices = pd.Index(raw.PERIDNUM).get_indexer(recipient_keys)
            _require(
                bool((indices >= 0).all()) and len(np.unique(indices)) == rows,
                "RESTORATION_KEY_COVERAGE",
            )
            joined = raw.iloc[indices]
            for source_col, raw_col in (
                ("source_household_id", "PH_SEQ"),
                ("A_LINENO", "A_LINENO"),
                ("A_AGE", "A_AGE"),
            ):
                _require(
                    np.array_equal(
                        person[source_col].iloc[positions].to_numpy(),
                        joined[raw_col].to_numpy(),
                    ),
                    "RESTORATION_NATIVE_KEY_OR_AGE",
                )
            incumbent = person.PTOTVAL.iloc[positions]
            observed = joined.PTOTVAL.to_numpy(dtype=np.int64)
            known = ~incumbent.isna().to_numpy()
            _require(
                np.array_equal(
                    incumbent.to_numpy(dtype=np.float64, na_value=np.nan)[known],
                    observed[known],
                ),
                "RESTORATION_INCUMBENT_CONFLICT",
                "PTOTVAL",
            )
            # The full current cohort has an incumbent and must be checked, not
            # silently treated as another missing-column restoration.
            if year == 2024:
                _require(
                    bool(known.all()), "RESTORATION_INCUMBENT_INCOMPLETE", "PTOTVAL"
                )
            output[positions] = observed
            joins.append(
                {
                    "income_year": year,
                    "survey_year": year + 1,
                    "member": member,
                    "archive_sha256": archive_pin,
                    "member_sha256": pin,
                    "source_rows": rows,
                    "joined_rows": len(positions),
                    "unreferenced_source_rows": 0,
                    "incumbent_compared_rows": int(known.sum()),
                    "incumbent_conflicts": 0,
                    "native_key_or_age_conflicts": 0,
                }
            )
    result = _owned_frame(parent)
    result.person[OBSERVED_COLUMN] = output
    source.validate()
    _require(_implementation() == before, "RESTORATION_IMPLEMENTATION_CHANGED")
    evidence = source.source.evidence
    from .asec_current_money import _parse

    parent_evidence = _parse(evidence)
    receipt = {
        "schema_version": 1,
        "artifact_kind": ARTIFACT_KIND,
        "original_v4_sha256": parent_evidence["parent_sha256"],
        "household_attachment_sha256": parent_evidence["attachment_sha256"],
        "parent_frame_sha256": legacy._frame_signature(parent),
        "output_frame_sha256": legacy._frame_signature(result),
        "reader": "authenticated_census_csv_exact_integer_no_fill_v1",
        "join_keys": ["source_year", "PERIDNUM"],
        "crosscheck_columns": ["source_household_id", "A_LINENO", "A_AGE"],
        "sources": joins,
        "output_column": OBSERVED_COLUMN,
        "logical_field": "PTOTVAL",
        "nominal_basis": "income_year_us_dollars",
        "dtype": "int64",
        "observation_sha256": _sha(output.astype("<i8", copy=False).tobytes()),
        "zero_origin_policy": "authenticated_census_csv_encoded_zero_v1",
        "zero_claim": "encoded_source_value_not_respondent_answer",
        "implementation_sha256": before,
        "encoding_contract": _ENCODING_CONTRACT,
    }
    return result, receipt


def _document(receipt):
    return {
        "schema_version": 1,
        "artifact_kind": ARTIFACT_KIND,
        "original_v4_sha256": receipt["original_v4_sha256"],
        "parent_checkpoint_sha256": receipt["household_attachment_sha256"],
        "person_income_observations": receipt,
    }


def _candidate_snapshot(path, destination, *, expected_size):
    """Candidate hash is observed only; exact reconstruction supplies authority."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as src, destination.open("xb") as dst:
        before = os.fstat(src.fileno())
        _require(
            0 < before.st_size == expected_size <= _ATTACHMENT_MAX_BYTES,
            "RESTORATION_SIZE",
        )
        count = 0
        while chunk := src.read(1024 * 1024):
            count += len(chunk)
            _require(count <= expected_size, "RESTORATION_SIZE")
            digest.update(chunk)
            dst.write(chunk)
        after = os.fstat(src.fileno())
        _require(count == expected_size, "RESTORATION_SIZE")
        _require(
            (
                before.st_dev,
                before.st_ino,
                before.st_size,
                before.st_mtime_ns,
                before.st_ctime_ns,
            )
            == (
                after.st_dev,
                after.st_ino,
                after.st_size,
                after.st_mtime_ns,
                after.st_ctime_ns,
            ),
            "RESTORATION_BYTES_CHANGED",
        )
    return digest.hexdigest()


@dataclass(frozen=True)
class _VerifiedRestoration:
    frame: Frame
    receipt: dict
    attachment_sha256: str
    parent_source: legacy.AuthenticatedCurrentMoneySource


def _verify(
    parent_path, household_attachment_path, person_income_attachment_path, member_paths
):
    _paths(member_paths)
    source = legacy.load_authenticated_current_money_source(
        parent_path, household_attachment_path
    )
    expected, receipt = _reconstruct(source, member_paths)
    with tempfile.TemporaryDirectory(
        prefix="microcosm-person-income-candidate-"
    ) as directory:
        canonical = Path(directory) / "expected.checkpoint.h5"
        checkpoint.write_frame_checkpoint(
            canonical, expected, metadata=_document(receipt)
        )
        with canonical.open("rb") as stream:
            expected_digest = hashlib.file_digest(stream, "sha256").hexdigest()
        snapshot = Path(directory) / CHECKPOINT_FILENAME
        digest = _candidate_snapshot(
            person_income_attachment_path,
            snapshot,
            expected_size=canonical.stat().st_size,
        )
        # Never decode candidate HDF. Its shapes, links, attributes, storage and
        # datatypes cannot control a read before independent byte authority.
        _require(digest == expected_digest, "RESTORATION_CANONICAL_BYTES")
    source.validate()
    _require(
        _implementation() == receipt["implementation_sha256"],
        "RESTORATION_IMPLEMENTATION_CHANGED",
    )
    # Return reconstructed owned content; the checked candidate never supplies
    # mutable arrays to source issuance.
    return _VerifiedRestoration(expected, receipt, digest, source)


def verify_asec_person_income_source(
    parent_path: str | Path,
    household_attachment_path: str | Path,
    person_income_attachment_path: str | Path,
    *,
    member_paths: Mapping[int, str | Path],
) -> _VerifiedRestoration:
    """Authenticate and reconstruct T fully; this function does not issue money."""
    try:
        return _verify(
            parent_path,
            household_attachment_path,
            person_income_attachment_path,
            member_paths,
        )
    except MoneyRefusalError:
        raise
    except (
        OSError,
        ValueError,
        TypeError,
        KeyError,
        OverflowError,
        UnicodeError,
        csv.Error,
        StopIteration,
    ):
        raise MoneyRefusalError("RESTORATION_SOURCE_CONTRACT") from None


def restore_asec_person_income_source(
    parent_path: str | Path,
    household_attachment_path: str | Path,
    *,
    member_paths: Mapping[int, str | Path],
    output_dir: str | Path,
) -> dict[str, object]:
    """Write a new local T and receipt from closed sources; never replace parents."""
    destination = Path(output_dir)
    if destination.exists():
        raise FileExistsError(destination)
    try:
        _paths(member_paths)
        source = legacy.load_authenticated_current_money_source(
            parent_path, household_attachment_path
        )
        frame, receipt = _reconstruct(source, member_paths)
        destination.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(
            prefix=".asec-person-income-", dir=destination.parent
        ) as directory:
            staging = Path(directory)
            output = staging / CHECKPOINT_FILENAME
            checkpoint.write_frame_checkpoint(
                output, frame, metadata=_document(receipt)
            )
            verified = _verify(
                parent_path, household_attachment_path, output, member_paths
            )
            report = {
                **receipt,
                "output_sha256": verified.attachment_sha256,
                "output_file": CHECKPOINT_FILENAME,
            }
            (staging / "restoration.receipt.json").write_bytes(_json(report) + b"\n")
            # Exclusive final directory creation avoids replacing even a raced
            # empty directory. No historical artifact is opened for writing.
            destination.mkdir()
            for path in staging.iterdir():
                os.link(path, destination / path.name)
        return report
    except FileExistsError:
        raise
    except MoneyRefusalError:
        raise
    except (
        OSError,
        ValueError,
        TypeError,
        KeyError,
        OverflowError,
        UnicodeError,
        csv.Error,
        StopIteration,
    ):
        raise MoneyRefusalError("RESTORATION_SOURCE_CONTRACT") from None


def load_authenticated_restored_current_money_source(
    parent_path: str | Path,
    household_attachment_path: str | Path,
    person_income_attachment_path: str | Path,
    *,
    member_paths: Mapping[int, str | Path],
) -> legacy.AuthenticatedCurrentMoneySource:
    """Issue restored authority only after closed-source complete reconstruction."""
    from .asec_current_money import (
        _SOURCE_TOKEN,
        RESTORED_FIELD_COLUMNS,
        RESTORED_FIELD_ZERO_POLICY,
        RESTORED_SOURCE_KIND,
        AuthenticatedAsecSource,
        _parse,
        compile_asec_current_money_spec,
    )
    from .asec_current_money_resources import load_current_money_resources

    checked = verify_asec_person_income_source(
        parent_path,
        household_attachment_path,
        person_income_attachment_path,
        member_paths=member_paths,
    )
    frame = checked.frame
    scope = legacy._scope(frame)
    views = legacy._views(frame, scope, restored=True)
    scope_sha, input_sha = legacy._input_binding(views)
    evidence = _parse(checked.parent_source.source.identity)
    evidence.update(
        schema_version=2,
        source_kind=RESTORED_SOURCE_KIND,
        person_income_attachment_sha256=checked.attachment_sha256,
        field_source_columns=dict(RESTORED_FIELD_COLUMNS),
        field_zero_origin_policy=dict(RESTORED_FIELD_ZERO_POLICY),
        scope_sha256=scope_sha,
        input_sha256=input_sha,
        frame_sha256=legacy._frame_signature(frame),
        verification_sha256=checked.receipt["implementation_sha256"],
    )
    checked.parent_source.validate()
    _require(
        _implementation() == evidence["verification_sha256"],
        "RESTORATION_IMPLEMENTATION_CHANGED",
    )
    authority = AuthenticatedAsecSource(_json(evidence), _token=_SOURCE_TOKEN)
    spec = compile_asec_current_money_spec(load_current_money_resources(), authority)
    result = legacy.AuthenticatedCurrentMoneySource(
        frame, scope, authority, spec, _token=legacy._LOAD_TOKEN
    )
    result.validate()
    return result
