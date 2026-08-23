#!/usr/bin/env python3
"""Reproduce the frozen f001 retirement source, model, and data audit.

This command never builds a pool.  It verifies every frozen artifact digest
before decoding it, reads only the required deterministic-checkpoint columns
and numeric fixed-HDF blocks, and emits one canonical JSON proof.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import struct
import tempfile
from collections.abc import Mapping
from pathlib import Path

import h5py
import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT = SCRIPT_DIR / "f001_audit.json"
BASELINE_DIR = Path(
    "/Users/maxghenis/PolicyEngine/_buildo-runtime/out/battery-verify/baseline1pct"
)
PKG3_DIR = Path("/Users/maxghenis/PolicyEngine/_buildo-runtime/out/battery-verify/pkg3")
ASEC_RAW_STAGE = Path(
    "/Users/maxghenis/PolicyEngine/_buildo-runtime/out/591-pawtyp-pool/"
    "asec-producer-checkpoints/asec_raw_stage.checkpoint.h5"
)
STACKED_CHECKPOINT = (
    BASELINE_DIR
    / "pool.checkpoints/stacked"
    / "2e45c4d60f66b4321bc00ffa22816470bf162c59fd91956514832f97e066ed3c"
)
TRANSFER_BANK = (
    STACKED_CHECKPOINT
    / "acs-transfer"
    / "091dc2effbe638687de621f3ed4312f738489a58923bf1a7172ac5da8e3c6eb7"
)
DEFAULT_ADJUDICATION = Path(
    "/Users/maxghenis/PolicyEngine/_worktrees/microcosm-arm-split"
    "/experiments/battery_burndown/adjudication.json"
)

ZERO_ATOL = 1e-6
BATTERY_MIN_EFFECTIVE_SUPPORT = 5
QUANTILES = (0.1, 0.25, 0.5, 0.75, 0.9)
EXPECTED_DONOR_INDEX = {
    "class_name": "Index",
    "dtype": "int64",
    "length": 4311,
    "names_repr": "(None,)",
    "sha256": "58b9edf0e1779a0e47534e0b385f5332e45b1f607ab93147cf8916c9a06a6a00",
}
EXPECTED_CANONICAL_SELECTED_GATE_RECORDS_SHA256 = (
    "9f350b4efa9fb229c7e8bdc775f19e9e4d1a586199bb17361a6352cab1ead5ca"
)

PRIMARY_SHA256 = {
    "asec_raw_stage": "51e9fafcd6f16140018fa90c7afbeb6d79008bfc8c122e437d23a399b30553fe",
    "baseline_pool": "8ed64a03fdad77f7f1d3f9eea8e800a5f35b88bb7bb2a28c7d7b10b0632037f0",
    "baseline_gates": "1d6059868680f872fe04d452a536bcc3c215bafabb4c50d7740a469fe6a8b56a",
    "pkg3_pool": "2892596de1148711dc74da777b08e5fffec138ebc42253e34b32baf9d73886b9",
    "pkg3_gates": "3ace0af0fd9e2ed6cb37cb110280f0c5cade182118c62737635c7ad177050ac3",
    "assembled_checkpoint": (
        "754ceb74fc737a41b588003912ce58cdfe29a7f3c114c9aba26cd2e6ae9956bd"
    ),
    "transferred_checkpoint": (
        "5ff70151a9ea9c9707e794995bb739abcd76ddc1401aea37299da41314a91d68"
    ),
    "adjudication": "1c1597c8080cf4e0db0079c2a3e1bc345001525bc3ec6c296aa7575b47136c32",
}


class AuditError(RuntimeError):
    """Raised when frozen evidence does not reproduce exactly."""


class LegSpec:
    """Small immutable description of one retirement battery leg."""

    def __init__(
        self,
        *,
        target: str,
        family: str,
        stage: str,
        target_index: int,
        total_targets: int,
        checkpoint_relative: str,
        checkpoint_sha256: str,
        raw_draw_sha256: str,
        source_columns: tuple[str, ...],
        equation: str,
        classification: str,
        code_citations: tuple[str, ...],
    ) -> None:
        self.target = target
        self.family = family
        self.stage = stage
        self.target_index = target_index
        self.total_targets = total_targets
        self.checkpoint_relative = checkpoint_relative
        self.checkpoint_sha256 = checkpoint_sha256
        self.raw_draw_sha256 = raw_draw_sha256
        self.source_columns = source_columns
        self.equation = equation
        self.classification = classification
        self.code_citations = code_citations

    @property
    def gate_label(self) -> str:
        family = (
            "puf_tax_itemization"
            if self.family.startswith("puf_tax_itemization")
            else self.family
        )
        return f"person/{family}/{self.target}[clone_0]"


SLOTS = ("1", "2", "1_YNG", "2_YNG")
PENSION_SOURCES = ("PNSN_VAL", "ANN_VAL", "PEN_SC1", "PEN_SC2")
SS_SOURCES = ("SS_VAL", "RESNSS1", "RESNSS2", "A_AGE")
SLOT_SOURCES = tuple(
    column for suffix in SLOTS for column in (f"DST_SC{suffix}", f"DST_VAL{suffix}")
)

LEGS = (
    LegSpec(
        target="tax_exempt_private_pension_income",
        family="model_required_numeric",
        stage="early_asec_survey_to_acs",
        target_index=18,
        total_targets=47,
        checkpoint_relative=(
            "asec_survey_to_acs/targets/018__tax_exempt_private_pension_income.h5"
        ),
        checkpoint_sha256=(
            "bfaf519f1a7ba32b378a61a49a6f23ac621e98c8c49fae874902f44b5d0d7a82"
        ),
        raw_draw_sha256=(
            "bee9b66005382ec505380bd52ba621ed38c89b8b25c84a2dc642394c79bfcdec"
        ),
        source_columns=PENSION_SOURCES,
        equation="(PNSN_VAL + ANN_VAL) * (1 - 0.590), missing sources := 0",
        classification="concept_mismatch",
        code_citations=(
            "packages/microcosm-build/src/microcosm/build/us_runtime/cps_carried.py:41-43",
            "packages/microcosm-build/src/microcosm/build/us_runtime/cps_carried.py:188-200",
            "packages/microcosm-build/src/microcosm/build/us_runtime/cps_carried.py:345-362",
            "packages/microcosm-build/src/microcosm/build/us_runtime/acs_transfer.py:148-164,186-212,1891-1966",
        ),
    ),
    LegSpec(
        target="taxable_private_pension_income",
        family="puf_tax_itemization__batch_1",
        stage="early_asec_survey_to_acs",
        target_index=26,
        total_targets=47,
        checkpoint_relative=(
            "asec_survey_to_acs/targets/026__taxable_private_pension_income.h5"
        ),
        checkpoint_sha256=(
            "62f37d15cba851779e5f066c5248c3f4b0f0e3390646e5922c840c0ad38e6b93"
        ),
        raw_draw_sha256=(
            "07efb34b153a025fa375fd10956c4217959b005efd28b96ac65e035314d7d7d0"
        ),
        source_columns=PENSION_SOURCES,
        equation="(PNSN_VAL + ANN_VAL) * 0.590, missing sources := 0",
        classification="concept_mismatch",
        code_citations=(
            "packages/microcosm-build/src/microcosm/build/us_runtime/cps_carried.py:41-43",
            "packages/microcosm-build/src/microcosm/build/us_runtime/cps_carried.py:188-200",
            "packages/microcosm-build/src/microcosm/build/us_runtime/cps_carried.py:345-362",
            "packages/microcosm-build/src/microcosm/build/us_runtime/acs_transfer.py:148-164,186-212,1891-1966",
        ),
    ),
    LegSpec(
        target="taxable_ira_distributions",
        family="puf_tax_itemization__batch_1",
        stage="early_asec_survey_to_acs",
        target_index=27,
        total_targets=47,
        checkpoint_relative=(
            "asec_survey_to_acs/targets/027__taxable_ira_distributions.h5"
        ),
        checkpoint_sha256=(
            "eb8e38ecc77504b3301329f3619f02d6fcd42e6dfc8c56aee61759a4d99ae3e4"
        ),
        raw_draw_sha256=(
            "edf2d13aaa6b8525af22cf03ce3e22776b4543047d55b2660096f72514af4df4"
        ),
        source_columns=SLOT_SOURCES,
        equation="sum_s(1[DST_SC_s == 4] * DST_VAL_s), s in {1,2,1_YNG,2_YNG}",
        classification="dense_rung_refit_required",
        code_citations=(
            "packages/microcosm-build/src/microcosm/build/us_runtime/cps_carried.py:490-500",
            "packages/microcosm-build/src/microcosm/build/us_runtime/retirement_distributions.py:97-112,210-333",
            "packages/microcosm-build/src/microcosm/build/us_runtime/multispine_pool.py:582-586,1936-1942,1959-1988,2095-2164",
            "packages/microcosm-build/src/microcosm/build/us_runtime/acs_transfer.py:186-212,1891-1966",
        ),
    ),
    LegSpec(
        target="social_security_retirement",
        family="puf_tax_itemization__batch_1",
        stage="early_asec_survey_to_acs",
        target_index=28,
        total_targets=47,
        checkpoint_relative=(
            "asec_survey_to_acs/targets/028__social_security_retirement.h5"
        ),
        checkpoint_sha256=(
            "f13ede064316e51c1977a15088cc6e9a11bafe50cd69df0698315099ae1e39cc"
        ),
        raw_draw_sha256=(
            "9a8565dd071ff94bfd0d8d856ee1de06722d50d64e28e8d091b5073481ece32b"
        ),
        source_columns=SS_SOURCES,
        equation="SS_VAL assigned by precedence retirement > disability > survivors > dependents; unclassified uses age 62",
        classification="concept_mismatch",
        code_citations=(
            "packages/microcosm-build/src/microcosm/build/us_runtime/cps_carried.py:355-410",
            "packages/microcosm-build/src/microcosm/build/us_runtime/stacked_spine.py:7581-7698,7808-7820",
            "packages/microcosm-build/src/microcosm/build/us_runtime/acs_transfer.py:148-164,186-212,1891-1966",
        ),
    ),
    LegSpec(
        target="social_security_disability",
        family="puf_tax_itemization__batch_2",
        stage="early_asec_survey_to_acs",
        target_index=29,
        total_targets=47,
        checkpoint_relative=(
            "asec_survey_to_acs/targets/029__social_security_disability.h5"
        ),
        checkpoint_sha256=(
            "e4841f0ba3c986d69469edc27e50065db4e3ad76a4abde9b7494d048a2c4e6b4"
        ),
        raw_draw_sha256=(
            "ec8dc95c6e6905b939880a9d894fe1bf140d9b4ed247df1bb01d8f742bfa955e"
        ),
        source_columns=SS_SOURCES,
        equation="SS_VAL assigned by precedence retirement > disability > survivors > dependents; unclassified uses age 62",
        classification="concept_mismatch",
        code_citations=(
            "packages/microcosm-build/src/microcosm/build/us_runtime/cps_carried.py:355-410",
            "packages/microcosm-build/src/microcosm/build/us_runtime/stacked_spine.py:7581-7698,7808-7820",
            "packages/microcosm-build/src/microcosm/build/us_runtime/acs_transfer.py:148-164,186-212,1891-1966",
        ),
    ),
    LegSpec(
        target="social_security_dependents",
        family="puf_tax_itemization__batch_2",
        stage="early_asec_survey_to_acs",
        target_index=30,
        total_targets=47,
        checkpoint_relative=(
            "asec_survey_to_acs/targets/030__social_security_dependents.h5"
        ),
        checkpoint_sha256=(
            "c9b2f2870694ddc33f53979596d1d31dda4715a02481b3214fd470babebc25ea"
        ),
        raw_draw_sha256=(
            "27570eb8d8dfc95b6dfa715dbc392cb807f024227dbf6568f0f2191d1bcb11a9"
        ),
        source_columns=SS_SOURCES,
        equation="SS_VAL assigned by precedence retirement > disability > survivors > dependents; unclassified uses age 62",
        classification="concept_mismatch",
        code_citations=(
            "packages/microcosm-build/src/microcosm/build/us_runtime/cps_carried.py:355-410",
            "packages/microcosm-build/src/microcosm/build/us_runtime/stacked_spine.py:7581-7698,7808-7820",
            "packages/microcosm-build/src/microcosm/build/us_runtime/acs_transfer.py:148-164,186-212,1891-1966",
        ),
    ),
    LegSpec(
        target="social_security_survivors",
        family="puf_tax_itemization__batch_2",
        stage="early_asec_survey_to_acs",
        target_index=31,
        total_targets=47,
        checkpoint_relative=(
            "asec_survey_to_acs/targets/031__social_security_survivors.h5"
        ),
        checkpoint_sha256=(
            "d7e58ed45d1a7aa1aecaae34188b884f8c3715af8e47863109df2d36655e6a5b"
        ),
        raw_draw_sha256=(
            "04850fd4563274d0d9842fbfc7b795eac2b0b4333ef84e910387ebd4718e56ca"
        ),
        source_columns=SS_SOURCES,
        equation="SS_VAL assigned by precedence retirement > disability > survivors > dependents; unclassified uses age 62",
        classification="concept_mismatch",
        code_citations=(
            "packages/microcosm-build/src/microcosm/build/us_runtime/cps_carried.py:355-410",
            "packages/microcosm-build/src/microcosm/build/us_runtime/stacked_spine.py:7581-7698,7808-7820",
            "packages/microcosm-build/src/microcosm/build/us_runtime/acs_transfer.py:148-164,186-212,1891-1966",
        ),
    ),
    LegSpec(
        target="keogh_distributions",
        family="source_operator_retirement_distributions",
        stage="late_producer_dag",
        target_index=0,
        total_targets=5,
        checkpoint_relative=(
            "late_producer_dag/person/source_operator_retirement_distributions/"
            "targets/000__keogh_distributions.h5"
        ),
        checkpoint_sha256=(
            "ccf2710a82c7078a0a6bfdbe39c17e7e39f57cc57c9b53573575b39854585708"
        ),
        raw_draw_sha256=(
            "74a5ba958dad1499e160092240730191c5241594db19816db6c0a857987edd61"
        ),
        source_columns=SLOT_SOURCES,
        equation="sum_s(1[DST_SC_s == 5] * DST_VAL_s), s in {1,2,1_YNG,2_YNG}",
        classification="dense_rung_refit_required",
        code_citations=(
            "packages/microcosm-build/src/microcosm/build/us_runtime/retirement_distributions.py:97-112,210-269,336-473",
            "packages/microcosm-build/src/microcosm/build/us_runtime/stacked_spine.py:8340-8483,8518-8538,8631-8729",
            "packages/microcosm-build/src/microcosm/build/us_runtime/acs_transfer.py:186-212,1247-1380,1411-1648",
        ),
    ),
    LegSpec(
        target="taxable_401k_distributions",
        family="source_operator_retirement_distributions",
        stage="late_producer_dag",
        target_index=2,
        total_targets=5,
        checkpoint_relative=(
            "late_producer_dag/person/source_operator_retirement_distributions/"
            "targets/002__taxable_401k_distributions.h5"
        ),
        checkpoint_sha256=(
            "dcaea8920d9fbcfeecd46b2c6576d46e90625aad1a9b54781cd91f193333db86"
        ),
        raw_draw_sha256=(
            "f69ce5b357300d056f38348045a021959bec919e6a746f92bf2af9df2ca15758"
        ),
        source_columns=SLOT_SOURCES,
        equation="sum_s(1[DST_SC_s == 1] * DST_VAL_s), s in {1,2,1_YNG,2_YNG}",
        classification="dense_rung_refit_required",
        code_citations=(
            "packages/microcosm-build/src/microcosm/build/us_runtime/retirement_distributions.py:97-112,210-269,336-473",
            "packages/microcosm-build/src/microcosm/build/us_runtime/stacked_spine.py:8340-8483,8518-8538,8631-8729",
            "packages/microcosm-build/src/microcosm/build/us_runtime/acs_transfer.py:186-212,1247-1380,1411-1648",
        ),
    ),
    LegSpec(
        target="taxable_403b_distributions",
        family="source_operator_retirement_distributions",
        stage="late_producer_dag",
        target_index=3,
        total_targets=5,
        checkpoint_relative=(
            "late_producer_dag/person/source_operator_retirement_distributions/"
            "targets/003__taxable_403b_distributions.h5"
        ),
        checkpoint_sha256=(
            "f43088f26bbb24032d754f11931c58b875cd216436fbd3d43faa23737f52fce2"
        ),
        raw_draw_sha256=(
            "2712e4e36f42407bc67b511e3c73f119d7da5cf7fa81b71d22149b9089cc304f"
        ),
        source_columns=SLOT_SOURCES,
        equation="sum_s(1[DST_SC_s == 2] * DST_VAL_s), s in {1,2,1_YNG,2_YNG}",
        classification="dense_rung_refit_required",
        code_citations=(
            "packages/microcosm-build/src/microcosm/build/us_runtime/retirement_distributions.py:97-112,210-269,336-473",
            "packages/microcosm-build/src/microcosm/build/us_runtime/stacked_spine.py:8340-8483,8518-8538,8631-8729",
            "packages/microcosm-build/src/microcosm/build/us_runtime/acs_transfer.py:186-212,1247-1380,1411-1648",
        ),
    ),
    LegSpec(
        target="taxable_sep_distributions",
        family="source_operator_retirement_distributions",
        stage="late_producer_dag",
        target_index=4,
        total_targets=5,
        checkpoint_relative=(
            "late_producer_dag/person/source_operator_retirement_distributions/"
            "targets/004__taxable_sep_distributions.h5"
        ),
        checkpoint_sha256=(
            "5827b9ff27091276b1ff4102f446f4dffc420e574164a95f3f855621f9e7e4b2"
        ),
        raw_draw_sha256=(
            "28f63d4cab4cf68c3ef323d70f149df23b47442744a097abba281f4c484a1e0c"
        ),
        source_columns=SLOT_SOURCES,
        equation="sum_s(1[DST_SC_s == 6] * DST_VAL_s), s in {1,2,1_YNG,2_YNG}",
        classification="dense_rung_refit_required",
        code_citations=(
            "packages/microcosm-build/src/microcosm/build/us_runtime/retirement_distributions.py:97-112,210-269,336-473",
            "packages/microcosm-build/src/microcosm/build/us_runtime/stacked_spine.py:8340-8483,8518-8538,8631-8729",
            "packages/microcosm-build/src/microcosm/build/us_runtime/acs_transfer.py:186-212,1247-1380,1411-1648",
        ),
    ),
)

EXPECTED_SOURCE_POSITIVE = {
    "tax_exempt_private_pension_income": 261,
    "taxable_private_pension_income": 261,
    "taxable_ira_distributions": 59,
    "social_security_retirement": 650,
    "social_security_disability": 64,
    "social_security_dependents": 14,
    "social_security_survivors": 12,
    "keogh_distributions": 0,
    "taxable_401k_distributions": 86,
    "taxable_403b_distributions": 6,
    "taxable_sep_distributions": 4,
}
EXPECTED_LATE_DONOR_POSITIVE = {
    "keogh_distributions": 0,
    "taxable_401k_distributions": 62,
    "taxable_403b_distributions": 1,
    "taxable_sep_distributions": 2,
}
EXPECTED_PATTERNS = {
    "early_asec_survey_to_acs": (
        "pattern_00_677f6490",
        "pattern_01_5874881e",
        "pattern_02_7c3bceda",
        "pattern_03_04f75638",
    ),
    "late_producer_dag": (
        "pattern_00_36785ebf",
        "pattern_01_c6777728",
        "pattern_02_5e7dd311",
        "pattern_03_76a0101a",
    ),
}


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise AuditError(message)


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _verify_file(path: Path, expected: str, *, label: str) -> dict[str, object]:
    _require(path.is_file(), f"{label} is not a regular file: {path}")
    observed = _file_sha256(path)
    _require(
        observed == expected,
        f"{label} SHA-256 changed: expected {expected}, got {observed}: {path}",
    )
    return {
        "path": str(path.resolve()),
        "sha256": observed,
        "size_bytes": path.stat().st_size,
    }


def _canonical_json(value: object) -> str:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )


def _mapping_sha256(value: Mapping[str, object]) -> str:
    return hashlib.sha256(_canonical_json(value).encode()).hexdigest()


def _json_bytes(value: object) -> bytes:
    return (
        json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    ).encode()


def _decode_object_values(
    offsets: np.ndarray, payload: bytes, *, label: str
) -> np.ndarray:
    _require(
        offsets.ndim == 1 and offsets.dtype == np.dtype("int64") and len(offsets) >= 1,
        f"{label}: malformed object offsets",
    )
    _require(
        offsets[0] == 0
        and offsets[-1] == len(payload)
        and bool((np.diff(offsets) >= 1).all()),
        f"{label}: invalid object boundaries",
    )
    values = np.empty(len(offsets) - 1, dtype=object)
    for index, (start, stop) in enumerate(zip(offsets[:-1], offsets[1:], strict=True)):
        chunk = payload[int(start) : int(stop)]
        tag, body = chunk[0], chunk[1:]
        try:
            if tag in (0, 1, 2) and not body:
                value: object = None
            elif tag == 3 and not body:
                value = False
            elif tag == 4 and not body:
                value = True
            elif tag == 5:
                value = int(body.decode("ascii"))
            elif tag == 6 and len(body) == 8:
                value = struct.unpack("<d", body)[0]
            elif tag == 7:
                value = body.decode("utf-8")
            elif tag == 8:
                value = body
            else:
                raise ValueError
        except (UnicodeDecodeError, ValueError) as error:
            raise AuditError(
                f"{label}: invalid object scalar at row {index}"
            ) from error
        values[index] = value
    return values


def _read_checkpoint_column(
    columns_group: h5py.Group,
    column_index: int,
    spec: Mapping[str, object],
    *,
    row_count: int,
    label: str,
) -> np.ndarray:
    group = columns_group[f"c{column_index:05d}"]
    encoding = spec.get("encoding")
    if encoding == "numpy":
        _require(set(group) == {"values"}, f"{label}: unexpected numpy datasets")
        values = np.asarray(group["values"])
    elif encoding == "object_scalars_v1":
        _require(
            set(group) == {"offsets", "payload"},
            f"{label}: unexpected object datasets",
        )
        offsets = np.asarray(group["offsets"])
        raw_payload = np.asarray(group["payload"])
        _require(
            raw_payload.ndim == 1 and raw_payload.dtype == np.dtype("uint8"),
            f"{label}: object payload is not one-dimensional uint8",
        )
        values = _decode_object_values(offsets, raw_payload.tobytes(), label=label)
    else:
        raise AuditError(f"{label}: unsupported selected-column encoding {encoding!r}")
    _require(values.ndim == 1, f"{label}: selected column is not one-dimensional")
    _require(len(values) == row_count, f"{label}: selected column length changed")
    return values


def _checkpoint_table(
    root: h5py.Group,
    metadata: Mapping[str, object],
    *,
    table_name: str,
    selected: set[str],
) -> dict[str, np.ndarray]:
    raw_tables = metadata.get("tables")
    _require(isinstance(raw_tables, list), "checkpoint tables metadata is not a list")
    matches = [
        (index, table)
        for index, table in enumerate(raw_tables)
        if isinstance(table, dict) and table.get("name") == table_name
    ]
    _require(len(matches) == 1, f"checkpoint table {table_name!r} is not unique")
    table_index, table = matches[0]
    index_spec = table.get("index")
    _require(isinstance(index_spec, dict), f"{table_name}: missing index metadata")
    _require(index_spec.get("kind") == "range", f"{table_name}: index is not range")
    _require(index_spec.get("start") == 0, f"{table_name}: range does not start at 0")
    _require(index_spec.get("step") == 1, f"{table_name}: range step is not 1")
    row_count = index_spec.get("stop")
    _require(
        isinstance(row_count, int) and not isinstance(row_count, bool),
        f"{table_name}: invalid row count",
    )
    specs = table.get("columns")
    _require(isinstance(specs, list), f"{table_name}: columns metadata is not a list")
    columns_group = root[f"tables/t{table_index:05d}/columns"]
    found: dict[str, np.ndarray] = {}
    for column_index, raw_spec in enumerate(specs):
        _require(isinstance(raw_spec, dict), f"{table_name}: invalid column spec")
        name = raw_spec.get("name")
        if name not in selected:
            continue
        _require(isinstance(name, str), f"{table_name}: invalid selected column name")
        _require(name not in found, f"{table_name}.{name}: duplicate column")
        found[name] = _read_checkpoint_column(
            columns_group,
            column_index,
            raw_spec,
            row_count=row_count,
            label=f"{table_name}.{name}",
        )
    missing = sorted(selected - set(found))
    _require(not missing, f"{table_name}: selected columns absent: {missing}")
    return found


def _read_checkpoint_person_columns(
    path: Path, *, selected: set[str]
) -> tuple[dict[str, np.ndarray], set[str], int]:
    """Read a bounded person projection and return its full column inventory."""

    with h5py.File(path, mode="r") as h5:
        _require(
            set(h5) == {"_populace_frame_checkpoint"},
            f"checkpoint root changed: {path}",
        )
        root = h5["_populace_frame_checkpoint"]
        _require(
            {"metadata_json", "tables"} <= set(root),
            f"checkpoint datasets changed: {path}",
        )
        metadata_raw = np.asarray(root["metadata_json"])
        _require(
            metadata_raw.ndim == 1 and metadata_raw.dtype == np.dtype("uint8"),
            f"checkpoint metadata encoding changed: {path}",
        )
        metadata = json.loads(metadata_raw.tobytes().decode())
        _require(isinstance(metadata, dict), f"checkpoint metadata changed: {path}")
        schema_version = metadata.get("schema_version")
        _require(
            metadata.get("artifact_kind") == "populace_frame_checkpoint"
            and schema_version in {2, 3},
            f"checkpoint format changed: {path}",
        )
        raw_tables = metadata.get("tables")
        _require(isinstance(raw_tables, list), f"checkpoint tables changed: {path}")
        matches = [
            item
            for item in raw_tables
            if isinstance(item, dict) and item.get("name") == "person"
        ]
        _require(len(matches) == 1, f"checkpoint person table changed: {path}")
        raw_columns = matches[0].get("columns")
        _require(
            isinstance(raw_columns, list), f"checkpoint person columns changed: {path}"
        )
        inventory = {
            str(item["name"])
            for item in raw_columns
            if isinstance(item, dict) and isinstance(item.get("name"), str)
        }
        _require(
            len(inventory) == len(raw_columns),
            f"checkpoint person column inventory changed: {path}",
        )
        projected = _checkpoint_table(
            root, metadata, table_name="person", selected=selected
        )
    return projected, inventory, int(schema_version)


def _read_checkpoint_surface(path: Path) -> tuple[dict[str, np.ndarray], np.ndarray]:
    target_names = {leg.target for leg in LEGS}
    person_columns = {
        "person_id",
        "person_household_id",
        "person_source_id",
        "person_support_channel",
        "person_support_clone_index",
        "ADJINC",
        "RETP",
        "SSP",
        "acs_retirement_income",
        "acs_social_security_income",
        *PENSION_SOURCES,
        *SS_SOURCES,
        *SLOT_SOURCES,
        *target_names,
    }
    with h5py.File(path, mode="r") as h5:
        _require(
            set(h5) == {"_populace_frame_checkpoint"},
            "transferred checkpoint root changed",
        )
        root = h5["_populace_frame_checkpoint"]
        _require(
            set(root) == {"metadata_json", "strata", "tables", "weights"},
            "transferred checkpoint datasets changed",
        )
        metadata_raw = np.asarray(root["metadata_json"])
        _require(
            metadata_raw.ndim == 1 and metadata_raw.dtype == np.dtype("uint8"),
            "transferred checkpoint metadata is not one-dimensional uint8",
        )
        metadata = json.loads(metadata_raw.tobytes().decode())
        _require(isinstance(metadata, dict), "checkpoint metadata is not an object")
        _require(
            metadata.get("artifact_kind") == "populace_frame_checkpoint"
            and metadata.get("schema_version") == 3,
            "transferred checkpoint format changed",
        )
        _require(
            metadata.get("household_weight_kind") == "importance",
            "transferred checkpoint weights are not importance weights",
        )
        person = _checkpoint_table(
            root, metadata, table_name="person", selected=person_columns
        )
        household = _checkpoint_table(
            root, metadata, table_name="household", selected={"household_id"}
        )
        weights_meta = metadata.get("weights")
        _require(
            weights_meta == [{"entity": "household", "kind": "importance"}],
            "transferred checkpoint weight metadata changed",
        )
        weights_group = root["weights"]
        _require(set(weights_group) == {"w00000"}, "checkpoint weight datasets changed")
        household_weight = np.asarray(weights_group["w00000"], dtype=np.float64)

    household_ids = np.asarray(household["household_id"], dtype=np.int64)
    person_household_ids = np.asarray(person["person_household_id"], dtype=np.int64)
    _require(
        len(household_ids) == len(household_weight),
        "household weights and IDs have different lengths",
    )
    _require(
        bool((household_ids[1:] > household_ids[:-1]).all()),
        "household IDs are not strictly increasing",
    )
    positions = np.searchsorted(household_ids, person_household_ids)
    _require(
        bool((positions < len(household_ids)).all())
        and np.array_equal(household_ids[positions], person_household_ids),
        "person household membership does not resolve exactly",
    )
    return person, household_weight[positions]


def _decode_names(dataset: h5py.Dataset, *, label: str) -> list[str]:
    values = np.asarray(dataset)
    _require(values.ndim == 1, f"{label}: names dataset is not one-dimensional")
    names: list[str] = []
    for value in values:
        _require(isinstance(value, np.bytes_), f"{label}: non-byte column name")
        names.append(bytes(value).decode("utf-8"))
    return names


def _read_fixed_numeric_columns(
    path: Path, *, entity: str, selected: set[str]
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    with h5py.File(path, mode="r") as h5:
        _require(entity in h5, f"fixed HDF has no {entity!r} group")
        group = h5[entity]
        pandas_type = group.attrs.get("pandas_type")
        _require(
            isinstance(pandas_type, np.bytes_) and bytes(pandas_type) == b"frame",
            f"fixed HDF {entity} is not a pandas frame",
        )
        row_index = np.asarray(group["axis1"])
        _require(row_index.ndim == 1, f"fixed HDF {entity} row index changed")
        nblocks = int(group.attrs["nblocks"])
        found: dict[str, np.ndarray] = {}
        for block_index in range(nblocks):
            items = _decode_names(
                group[f"block{block_index}_items"],
                label=f"{entity}.block{block_index}_items",
            )
            wanted = [
                (position, name)
                for position, name in enumerate(items)
                if name in selected
            ]
            if not wanted:
                continue
            dataset = group[f"block{block_index}_values"]
            _require(
                dataset.ndim == 2
                and dataset.shape == (len(row_index), len(items))
                and np.issubdtype(dataset.dtype, np.number),
                f"fixed HDF selected {entity} block {block_index} is not numeric",
            )
            block = np.asarray(dataset)
            for position, name in wanted:
                _require(name not in found, f"fixed HDF duplicate column {name!r}")
                found[name] = np.ascontiguousarray(block[:, position])
        missing = sorted(selected - set(found))
        _require(not missing, f"fixed HDF selected columns absent: {missing}")
    return row_index, found


def _coerce_numeric(values: np.ndarray, *, missing: float) -> np.ndarray:
    if np.issubdtype(values.dtype, np.number):
        result = values.astype(np.float64, copy=True)
        result[np.isnan(result)] = missing
        return result
    result = np.empty(len(values), dtype=np.float64)
    for index, value in enumerate(values):
        if value is None:
            result[index] = missing
            continue
        try:
            converted = float(value)
        except (TypeError, ValueError):
            converted = missing
        result[index] = missing if math.isnan(converted) else converted
    return result


def _source_numeric(values: np.ndarray) -> np.ndarray:
    return _coerce_numeric(values, missing=0.0)


def _array_sha256(values: np.ndarray) -> str:
    normalized = np.ascontiguousarray(values, dtype="<f8")
    return hashlib.sha256(normalized.tobytes()).hexdigest()


def _bitwise_float_equal(left: np.ndarray, right: np.ndarray) -> bool:
    if left.shape != right.shape:
        return False
    left_bits = np.ascontiguousarray(left, dtype="<f8").view("<u8")
    right_bits = np.ascontiguousarray(right, dtype="<f8").view("<u8")
    return bool(np.array_equal(left_bits, right_bits))


def _sign_counts(values: np.ndarray) -> dict[str, int]:
    _require(bool(np.isfinite(values).all()), "sign counts require finite values")
    return {
        "negative": int((values < -ZERO_ATOL).sum()),
        "positive": int((values > ZERO_ATOL).sum()),
        "zero": int((np.abs(values) <= ZERO_ATOL).sum()),
    }


def _detect_regime(values: np.ndarray) -> str:
    counts = _sign_counts(values)
    has_negative = counts["negative"] > 0
    has_positive = counts["positive"] > 0
    has_zero = counts["zero"] > 0
    if has_negative and has_positive and has_zero:
        return "three_sign"
    if has_negative and has_positive:
        return "sign_only"
    if has_positive and has_zero:
        return "zero_inflated_positive"
    if has_negative and has_zero:
        return "zero_inflated_negative"
    if has_positive:
        return "positive_only"
    if has_negative:
        return "negative_only"
    return "degenerate_zero"


def _amount_stats(values: np.ndarray) -> dict[str, object]:
    counts = _sign_counts(values)
    positive = values[values > ZERO_ATOL]
    negative = values[values < -ZERO_ATOL]
    return {
        "row_count": len(values),
        "sign_counts": counts,
        "sum": float(values.sum()),
        "positive_amount": {
            "max": float(positive.max()) if len(positive) else None,
            "min": float(positive.min()) if len(positive) else None,
            "sum": float(positive.sum()),
        },
        "negative_amount": {
            "max_absolute": float(np.abs(negative).max()) if len(negative) else None,
            "sum": float(negative.sum()),
        },
        "values_sha256": _array_sha256(values),
    }


def _weighted_quantiles(values: np.ndarray, weights: np.ndarray) -> list[float]:
    _require(len(values) > 0, "weighted quantiles require nonempty support")
    _require(bool((weights > 0).all()), "weighted quantiles require positive weights")
    order = np.argsort(values, kind="stable")
    sorted_values = values[order]
    cumulative = np.cumsum(weights[order])
    cumulative /= cumulative[-1]
    positions = np.minimum(
        np.searchsorted(cumulative, np.asarray(QUANTILES), side="left"),
        len(sorted_values) - 1,
    )
    return [float(value) for value in sorted_values[positions]]


def _origin_stats(
    values: np.ndarray, weights: np.ndarray, mask: np.ndarray
) -> dict[str, object]:
    selected = values[mask]
    selected_weights = weights[mask]
    _require(bool(np.isfinite(selected).all()), "terminal values are not finite")
    _require(bool((selected_weights > 0).all()), "terminal weights are not positive")
    total_weight = float(selected_weights.sum())
    positive = selected > ZERO_ATOL
    negative = selected < -ZERO_ATOL
    return {
        **_amount_stats(selected),
        "weight_sum": total_weight,
        "weighted_incidence": {
            "negative": float(selected_weights[negative].sum() / total_weight),
            "positive": float(selected_weights[positive].sum() / total_weight),
        },
        "positive_weighted_quantiles": (
            _weighted_quantiles(selected[positive], selected_weights[positive])
            if positive.any()
            else None
        ),
    }


def _quantile_envelope_distance(left: list[float], right: list[float]) -> float:
    left_values = np.asarray(left, dtype=np.float64)
    right_values = np.asarray(right, dtype=np.float64)
    denominator = np.abs(left_values) + np.abs(right_values)
    distances = np.divide(
        2.0 * np.abs(left_values - right_values),
        denominator,
        out=np.zeros_like(denominator),
        where=denominator > 0,
    )
    return float(distances.max())


def _assert_close(observed: float, expected: float, *, label: str) -> None:
    _require(
        math.isclose(observed, expected, rel_tol=1e-12, abs_tol=1e-12),
        f"{label} changed: recomputed {observed}, gate {expected}",
    )


def _read_target_checkpoint(path: Path, spec: LegSpec) -> dict[str, object]:
    with h5py.File(path, mode="r") as h5:
        _require(
            set(h5) == {"metadata_json", "raw_draw_bits"},
            f"{spec.target}: target-bank datasets changed",
        )
        metadata_dataset = h5["metadata_json"]
        raw_dataset = h5["raw_draw_bits"]
        _require(
            metadata_dataset.ndim == 1 and metadata_dataset.dtype == np.dtype("uint8"),
            f"{spec.target}: metadata_json encoding changed",
        )
        _require(
            raw_dataset.ndim == 1 and raw_dataset.dtype == np.dtype("<u8"),
            f"{spec.target}: raw_draw_bits encoding changed",
        )
        metadata = json.loads(np.asarray(metadata_dataset).tobytes().decode())
        raw_bits = np.asarray(raw_dataset, dtype="<u8")

    _require(isinstance(metadata, dict), f"{spec.target}: metadata is not an object")
    expected_keys = {
        "artifact_kind",
        "content_metadata_sha256",
        "identity",
        "identity_sha256",
        "materializer_version",
        "pattern_steps",
        "raw_draw_sha256",
        "recipient_rows",
        "schema_version",
        "target",
    }
    _require(set(metadata) == expected_keys, f"{spec.target}: metadata keys changed")
    _require(
        metadata["artifact_kind"]
        == "populace_us_multispine_acs_transfer_target_checkpoint"
        and metadata["schema_version"] == 1
        and metadata["materializer_version"] == 2,
        f"{spec.target}: target-bank binding changed",
    )
    identity = metadata["identity"]
    _require(isinstance(identity, dict), f"{spec.target}: identity is not an object")
    _require(
        _mapping_sha256(identity) == metadata["identity_sha256"],
        f"{spec.target}: identity digest mismatch",
    )
    content = {
        key: value
        for key, value in metadata.items()
        if key != "content_metadata_sha256"
    }
    _require(
        _mapping_sha256(content) == metadata["content_metadata_sha256"],
        f"{spec.target}: content metadata digest mismatch",
    )
    observed_raw_sha256 = hashlib.sha256(raw_bits.tobytes()).hexdigest()
    _require(
        observed_raw_sha256 == metadata["raw_draw_sha256"] == spec.raw_draw_sha256,
        f"{spec.target}: raw draw digest changed",
    )
    _require(
        len(raw_bits) == metadata["recipient_rows"],
        f"{spec.target}: raw draw row binding changed",
    )
    raw_draw = raw_bits.view("<f8")
    _require(
        not bool(np.isinf(raw_draw).any()), f"{spec.target}: raw draws contain infinity"
    )

    target = metadata["target"]
    _require(isinstance(target, dict), f"{spec.target}: descriptor is not an object")
    _require(
        target.get("entity") == "person"
        and target.get("family") == spec.family
        and target.get("model_target") == spec.target
        and target.get("exported_targets") == [spec.target]
        and target.get("target_index") == spec.target_index
        and target.get("total_targets") == spec.total_targets,
        f"{spec.target}: descriptor changed",
    )

    raw_steps = metadata["pattern_steps"]
    _require(
        isinstance(raw_steps, list), f"{spec.target}: pattern steps are not a list"
    )
    patterns: list[dict[str, object]] = []
    observed_pattern_names: list[str] = []
    for raw_step in raw_steps:
        _require(isinstance(raw_step, dict), f"{spec.target}: invalid pattern step")
        _require(
            set(raw_step) == {"pattern", "state_after", "state_before_sha256"},
            f"{spec.target}: pattern-step keys changed",
        )
        pattern = raw_step["pattern"]
        state = raw_step["state_after"]
        _require(
            isinstance(pattern, str) and isinstance(state, dict),
            f"{spec.target}: invalid pattern state",
        )
        observed_pattern_names.append(pattern)
        _require(
            state.get("donor_index") == EXPECTED_DONOR_INDEX,
            f"{spec.target}/{pattern}: donor index changed",
        )
        _require(
            state.get("weight_kind") == "importance",
            f"{spec.target}/{pattern}: weight kind changed",
        )
        model_config = state.get("model_config")
        _require(
            isinstance(model_config, dict)
            and model_config.get("n_estimators") == 100
            and model_config.get("zero_atol") == ZERO_ATOL
            and model_config.get("fit_n_jobs") == -1
            and model_config.get("max_samples_leaf") is None
            and model_config.get("max_samples_leaf_kind") == "none",
            f"{spec.target}/{pattern}: model configuration changed",
        )
        completed = state.get("completed_targets")
        _require(
            isinstance(completed, list) and completed and completed[-1] == spec.target,
            f"{spec.target}/{pattern}: completed-target prefix changed",
        )
        recipient_index = state.get("recipient_index")
        _require(
            isinstance(recipient_index, dict)
            and isinstance(recipient_index.get("length"), int),
            f"{spec.target}/{pattern}: recipient identity changed",
        )
        patterns.append(
            {
                "model_config": model_config,
                "pattern": pattern,
                "predictors": state.get("predictors"),
                "recipient_index": recipient_index,
                "state_before_sha256": raw_step["state_before_sha256"],
                "weight_kind": state["weight_kind"],
                "weight_sha256": state.get("weight_sha256"),
            }
        )
    _require(
        tuple(observed_pattern_names) == EXPECTED_PATTERNS[spec.stage],
        f"{spec.target}: pattern order or membership changed",
    )
    modeled_rows = sum(pattern["recipient_index"]["length"] for pattern in patterns)
    _require(
        int(np.isfinite(raw_draw).sum()) == modeled_rows
        and int(np.isnan(raw_draw).sum()) == len(raw_draw) - modeled_rows,
        f"{spec.target}: raw-draw modeled/structural-NaN partition changed",
    )
    return {
        "checkpoint_sha256": spec.checkpoint_sha256,
        "content_metadata_sha256": metadata["content_metadata_sha256"],
        "identity_sha256": metadata["identity_sha256"],
        "patterns": patterns,
        "raw_draw_finite_modeled_rows": modeled_rows,
        "raw_draw_structural_nan_rows": len(raw_draw) - modeled_rows,
        "raw_draw_rows": len(raw_bits),
        "raw_draw_sha256": observed_raw_sha256,
        "target_descriptor": target,
    }


def _derive_sources(
    person: Mapping[str, np.ndarray], asec_clone0: np.ndarray
) -> tuple[dict[str, np.ndarray], dict[str, object]]:
    source: dict[str, np.ndarray] = {}
    pnsn = _source_numeric(person["PNSN_VAL"])[asec_clone0]
    annuity = _source_numeric(person["ANN_VAL"])[asec_clone0]
    pension = pnsn + annuity
    source["taxable_private_pension_income"] = pension * 0.590
    source["tax_exempt_private_pension_income"] = pension * (1 - 0.590)

    pension_code_1 = _coerce_numeric(person["PEN_SC1"][asec_clone0], missing=np.nan)
    pension_code_2 = _coerce_numeric(person["PEN_SC2"][asec_clone0], missing=np.nan)
    for slot, codes in (("PEN_SC1", pension_code_1), ("PEN_SC2", pension_code_2)):
        _require(
            bool(
                np.isfinite(codes).all()
                and (codes == np.floor(codes)).all()
                and np.isin(codes, np.arange(9)).all()
            ),
            f"ASEC {slot} violates the strict 0..8 integer source contract",
        )
    pension_code_1 = pension_code_1.astype(np.int64)
    pension_code_2 = pension_code_2.astype(np.int64)
    private_source = np.isin(pension_code_1, (1, 2)) | np.isin(pension_code_2, (1, 2))
    government_source = np.isin(pension_code_1, (3, 4, 5, 6)) | np.isin(
        pension_code_2, (3, 4, 5, 6)
    )
    unknown_source = np.isin(pension_code_1, (7, 8)) | np.isin(pension_code_2, (7, 8))
    positive_pnsn = pnsn > ZERO_ATOL
    mixed_private_government = positive_pnsn & private_source & government_source
    private_only = positive_pnsn & private_source & ~government_source & ~unknown_source
    government_only = (
        positive_pnsn & government_source & ~private_source & ~unknown_source
    )
    unresolved_pnsn = positive_pnsn & ~(
        private_only | government_only | mixed_private_government
    )
    _require(
        np.array_equal(
            private_only | government_only | mixed_private_government | unresolved_pnsn,
            positive_pnsn,
        ),
        "pension source categories do not partition positive PNSN_VAL",
    )

    def pension_category(mask: np.ndarray) -> dict[str, object]:
        return {
            "ANN_VAL_positive_rows": int((mask & (annuity > ZERO_ATOL)).sum()),
            "ANN_VAL_sum": float(annuity[mask].sum()),
            "PNSN_VAL_sum": float(pnsn[mask].sum()),
            "combined_sum": float(pension[mask].sum()),
            "rows": int(mask.sum()),
        }

    pension_source_categories = {
        "government_only_codes_3_to_6": pension_category(government_only),
        "mixed_private_and_government": pension_category(mixed_private_government),
        "private_only_codes_1_or_2": pension_category(private_only),
        "unresolved_code_7_8_or_no_recognized_code": pension_category(unresolved_pnsn),
    }
    _require(
        pension_source_categories["private_only_codes_1_or_2"]["rows"] == 140
        and pension_source_categories["government_only_codes_3_to_6"]["rows"] == 83
        and pension_source_categories["mixed_private_and_government"]["rows"] == 4
        and pension_source_categories["unresolved_code_7_8_or_no_recognized_code"][
            "rows"
        ]
        == 7,
        "frozen pension source-category carrier counts changed",
    )
    _assert_close(
        pension_source_categories["private_only_codes_1_or_2"]["PNSN_VAL_sum"],
        2_535_409.0,
        label="private-only PNSN_VAL",
    )
    _assert_close(
        pension_source_categories["government_only_codes_3_to_6"]["PNSN_VAL_sum"],
        3_361_227.0,
        label="government-only PNSN_VAL",
    )
    _assert_close(
        pension_source_categories["mixed_private_and_government"]["PNSN_VAL_sum"],
        90_214.0,
        label="mixed private/government PNSN_VAL",
    )
    _assert_close(
        pension_source_categories["unresolved_code_7_8_or_no_recognized_code"][
            "PNSN_VAL_sum"
        ],
        113_568.0,
        label="unresolved-source PNSN_VAL",
    )
    pension_declared_absence = (
        (annuity > ZERO_ATOL) | mixed_private_government | unresolved_pnsn
    )

    slot_codes: dict[str, np.ndarray] = {}
    slot_amounts: dict[str, np.ndarray] = {}
    for suffix in SLOTS:
        code = _coerce_numeric(person[f"DST_SC{suffix}"][asec_clone0], missing=np.nan)
        amount = _coerce_numeric(
            person[f"DST_VAL{suffix}"][asec_clone0], missing=np.nan
        )
        _require(
            bool(
                np.isfinite(code).all()
                and (code == np.floor(code)).all()
                and np.isin(code, np.arange(8)).all()
            ),
            f"ASEC DST_SC{suffix} violates the strict 0..7 integer source contract",
        )
        _require(
            bool(np.isfinite(amount).all() and (amount >= 0).all()),
            f"ASEC DST_VAL{suffix} violates the finite nonnegative source contract",
        )
        code_int = code.astype(np.int64)
        _require(
            not bool(((code_int == 0) & (amount != 0)).any()),
            f"ASEC DST slot {suffix} has an amount on NIU code 0",
        )
        slot_codes[suffix] = code_int
        slot_amounts[suffix] = amount

    account_targets = {
        1: "taxable_401k_distributions",
        2: "taxable_403b_distributions",
        4: "taxable_ira_distributions",
        5: "keogh_distributions",
        6: "taxable_sep_distributions",
    }
    for account_code, target in account_targets.items():
        values = np.zeros(int(asec_clone0.sum()), dtype=np.float64)
        for suffix in SLOTS:
            values += np.where(
                slot_codes[suffix] == account_code, slot_amounts[suffix], 0.0
            )
        source[target] = values

    social_security = _source_numeric(person["SS_VAL"])[asec_clone0]
    reason_1 = _source_numeric(person["RESNSS1"])[asec_clone0].astype(np.int64)
    reason_2 = _source_numeric(person["RESNSS2"])[asec_clone0].astype(np.int64)
    age = _source_numeric(person["A_AGE"])[asec_clone0].astype(np.int64)
    is_retirement = (reason_1 == 1) | (reason_2 == 1)
    is_disability = (reason_1 == 2) | (reason_2 == 2)
    is_survivor = np.isin(reason_1, (3, 5)) | np.isin(reason_2, (3, 5))
    is_dependent = np.isin(reason_1, (4, 6, 7)) | np.isin(reason_2, (4, 6, 7))
    unclassified = (
        (social_security > 0)
        & ~is_retirement
        & ~is_disability
        & ~is_survivor
        & ~is_dependent
    )
    source["social_security_retirement"] = np.where(
        is_retirement | (unclassified & (age >= 62)), social_security, 0.0
    )
    source["social_security_disability"] = np.where(
        (is_disability & ~is_retirement) | (unclassified & (age < 62)),
        social_security,
        0.0,
    )
    source["social_security_survivors"] = np.where(
        is_survivor & ~is_retirement & ~is_disability, social_security, 0.0
    )
    source["social_security_dependents"] = np.where(
        is_dependent & ~is_retirement & ~is_disability & ~is_survivor,
        social_security,
        0.0,
    )
    ss_total = sum(
        source[target]
        for target in (
            "social_security_retirement",
            "social_security_disability",
            "social_security_dependents",
            "social_security_survivors",
        )
    )
    _require(
        _bitwise_float_equal(ss_total, social_security),
        "current Social Security precedence does not conserve SS_VAL",
    )

    reason_pairs: dict[str, int] = {}
    positive_ss = social_security > ZERO_ATOL
    for first, second in zip(reason_1[positive_ss], reason_2[positive_ss], strict=True):
        key = f"({int(first)},{int(second)})"
        reason_pairs[key] = reason_pairs.get(key, 0) + 1

    recognized = {
        1: frozenset({"R"}),
        2: frozenset({"D"}),
        3: frozenset({"S"}),
        4: frozenset({"P"}),
        5: frozenset({"S"}),
        6: frozenset({"P"}),
    }
    cardinality = np.fromiter(
        (
            len(
                recognized.get(int(first), frozenset())
                | recognized.get(int(second), frozenset())
            )
            for first, second in zip(reason_1, reason_2, strict=True)
        ),
        dtype=np.int64,
        count=len(reason_1),
    )
    code_7_or_8 = np.isin(reason_1, (7, 8)) | np.isin(reason_2, (7, 8))
    ambiguity = positive_ss & (code_7_or_8 | (cardinality != 1))
    code_7_or_8_ambiguity = positive_ss & code_7_or_8
    recognized_multi_category_ambiguity = (
        positive_ss & (cardinality != 1) & ~code_7_or_8
    )
    _require(int(ambiguity.sum()) == 35, "declared-absence ambiguity count changed")
    _assert_close(
        float(social_security[code_7_or_8_ambiguity].sum()),
        495_214.0,
        label="Social Security reason-7/8 ambiguity amount",
    )
    _assert_close(
        float(social_security[recognized_multi_category_ambiguity].sum()),
        249_520.0,
        label="Social Security multi-category ambiguity amount",
    )

    evidence = {
        "distribution_slots": {
            suffix: {
                "account_code_carriers": {
                    str(code): int((slot_codes[suffix] == code).sum())
                    for code in range(1, 7)
                },
                "positive_amount_rows": int((slot_amounts[suffix] > ZERO_ATOL).sum()),
            }
            for suffix in SLOTS
        },
        "pension_sources": {
            "ANN_VAL_positive": int((annuity > ZERO_ATOL).sum()),
            "ANN_VAL_positive_sum": float(annuity[annuity > ZERO_ATOL].sum()),
            "PNSN_VAL_and_ANN_VAL_positive": int(
                ((pnsn > ZERO_ATOL) & (annuity > ZERO_ATOL)).sum()
            ),
            "PNSN_VAL_positive": int((pnsn > ZERO_ATOL).sum()),
            "PNSN_VAL_positive_sum": float(pnsn[pnsn > ZERO_ATOL].sum()),
            "current_derivation": (
                "private_taxable=0.590*(PNSN_VAL+ANN_VAL); "
                "private_tax_exempt=0.410*(PNSN_VAL+ANN_VAL); "
                "PEN_SC1/PEN_SC2 are ignored"
            ),
            "combined": _amount_stats(pension),
            "deterministic_government_to_private_misclassification": {
                "current_private_tax_exempt_amount": float(
                    pnsn[government_only].sum() * (1 - 0.590)
                ),
                "current_private_taxable_amount": float(
                    pnsn[government_only].sum() * 0.590
                ),
                "PNSN_VAL_sum": float(pnsn[government_only].sum()),
                "rows": int(government_only.sum()),
            },
            "owner_declared_absence_proposal": {
                "ambiguous_rows": int(pension_declared_absence.sum()),
                "ambiguous_combined_amount": float(
                    pension[pension_declared_absence].sum()
                ),
                "equation": (
                    "P=nz(PNSN_VAL), N=nz(ANN_VAL), "
                    "R={nonzero PEN_SC1,PEN_SC2}, "
                    "Rpriv=R intersect {1,2}, Rpub=R intersect {3,4,5,6}; "
                    "A=1[N>0 or (P>0 and (R intersects {7,8} or "
                    "Rpriv and Rpub or |Rpriv union Rpub|=0))]; "
                    "if P=N=0 all four pension leaves=0; if A=0 and "
                    "R subset {1,2}, private taxable/exempt=(.590P,.410P) "
                    "and public=(0,0); if A=0 and R subset {3,4,5,6}, "
                    "public taxable/exempt=(.590P,.410P) and private=(0,0); "
                    "if A=1 all four leaves=NA (declared absent)"
                ),
                "owner_action": "adjudication_required; no exclusion implemented",
            },
            "persisted_source_limitation": (
                "PEN_VAL1/PEN_VAL2 source-specific amounts are not present in the "
                "frozen raw, assembled, or transferred checkpoints"
            ),
            "source_categories": pension_source_categories,
            "source_code_pairs_on_positive_PNSN_VAL": dict(
                sorted(
                    {
                        f"({first},{second})": int(
                            (
                                positive_pnsn
                                & (pension_code_1 == first)
                                & (pension_code_2 == second)
                            ).sum()
                        )
                        for first, second in zip(
                            pension_code_1[positive_pnsn],
                            pension_code_2[positive_pnsn],
                            strict=True,
                        )
                    }.items()
                )
            ),
        },
        "social_security_declared_absence_proposal": {
            "ambiguous_positive_amount": float(social_security[ambiguity].sum()),
            "ambiguous_positive_rows": int(ambiguity.sum()),
            "recognized_cardinality_not_one_amount": float(
                social_security[recognized_multi_category_ambiguity].sum()
            ),
            "recognized_cardinality_not_one_rows": int(
                recognized_multi_category_ambiguity.sum()
            ),
            "reason_7_or_8_amount": float(social_security[code_7_or_8_ambiguity].sum()),
            "reason_7_or_8_rows": int((positive_ss & code_7_or_8).sum()),
            "equation": (
                "C(0)=empty,C(1)={R},C(2)={D},C(3)=C(5)={S},"
                "C(4)=C(6)={P}; U=C(RESNSS1) union C(RESNSS2); "
                "A=1[SS_VAL>0 and (RESNSS1 or RESNSS2 in {7,8} or |U|!=1)]; "
                "Y_k=0 if SS_VAL=0; Y_k=SS_VAL*1[U={k}] if A=0; "
                "Y_k=NA (declared absent) if A=1"
            ),
            "owner_action": "adjudication_required; no exclusion implemented",
            "positive_reason_pairs": dict(sorted(reason_pairs.items())),
        },
    }
    return source, evidence


def _native_disagreement(
    person: Mapping[str, np.ndarray], acs_clone0: np.ndarray
) -> dict[str, object]:
    adjustment = _coerce_numeric(person["ADJINC"][acs_clone0], missing=np.nan)
    retp = _coerce_numeric(person["RETP"][acs_clone0], missing=np.nan)
    ssp = _coerce_numeric(person["SSP"][acs_clone0], missing=np.nan)
    native_retirement = _coerce_numeric(
        person["acs_retirement_income"][acs_clone0], missing=np.nan
    )
    native_ss = _coerce_numeric(
        person["acs_social_security_income"][acs_clone0], missing=np.nan
    )
    adjusted_retirement = retp * (adjustment / 1_000_000)
    adjusted_ss = ssp * (adjustment / 1_000_000)
    _require(
        _bitwise_float_equal(adjusted_retirement, native_retirement),
        "ACS RETP adjustment does not bit-match acs_retirement_income",
    )
    _require(
        _bitwise_float_equal(adjusted_ss, native_ss),
        "ACS SSP adjustment does not bit-match acs_social_security_income",
    )

    taxable_pension = np.asarray(
        person["taxable_private_pension_income"][acs_clone0], dtype=np.float64
    )
    exempt_pension = np.asarray(
        person["tax_exempt_private_pension_income"][acs_clone0], dtype=np.float64
    )
    pension_total = taxable_pension + exempt_pension
    taxable_ira = np.asarray(
        person["taxable_ira_distributions"][acs_clone0], dtype=np.float64
    )
    retirement_bundle = pension_total + taxable_ira
    ss_components = sum(
        np.asarray(person[target][acs_clone0], dtype=np.float64)
        for target in (
            "social_security_retirement",
            "social_security_disability",
            "social_security_dependents",
            "social_security_survivors",
        )
    )

    def comparison(native: np.ndarray, modeled: np.ndarray) -> dict[str, object]:
        observed = np.isfinite(native)
        native_positive = observed & (native > ZERO_ATOL)
        modeled_positive = modeled > ZERO_ATOL
        return {
            "modeled": _amount_stats(modeled),
            "modeled_only_positive_rows": int(
                (modeled_positive & ~native_positive).sum()
            ),
            "native": {
                "missing_rows": int((~observed).sum()),
                "observed_rows": int(observed.sum()),
                "positive_rows": int(native_positive.sum()),
                "positive_sum": float(native[native_positive].sum()),
            },
            "native_only_positive_rows": int(
                (native_positive & ~modeled_positive).sum()
            ),
            "observed_amount_mismatch_rows": int(
                (
                    observed & ~np.isclose(native, modeled, rtol=0.0, atol=ZERO_ATOL)
                ).sum()
            ),
        }

    retirement = comparison(native_retirement, retirement_bundle)
    retirement["modeled_private_pension_total"] = _amount_stats(pension_total)
    retirement["modeled_taxable_ira"] = _amount_stats(taxable_ira)
    social_security = comparison(native_ss, ss_components)
    _require(
        retirement["native"]["observed_rows"] == 28999
        and retirement["native"]["positive_rows"] == 4974
        and retirement["native_only_positive_rows"] == 10
        and retirement["modeled_only_positive_rows"] == 2,
        "ACS retirement native-vs-modeled carrier reconciliation changed: "
        f"{retirement}",
    )
    _require(
        social_security["native"]["observed_rows"] == 28999
        and social_security["native"]["positive_rows"] == 7546
        and social_security["native_only_positive_rows"] == 316
        and social_security["modeled_only_positive_rows"] == 1,
        "ACS Social Security native-vs-modeled carrier reconciliation changed: "
        f"{social_security}",
    )
    pension_positive = pension_total > ZERO_ATOL
    split_mismatch = pension_positive & (
        ~np.isclose(taxable_pension, pension_total * 0.590, rtol=0.0, atol=ZERO_ATOL)
        | ~np.isclose(
            exempt_pension,
            pension_total * (1 - 0.590),
            rtol=0.0,
            atol=ZERO_ATOL,
        )
    )
    retirement["modeled_59_41_split_mismatch_rows"] = int(split_mismatch.sum())
    return {
        "acs_source_mapping": {
            "retirement": "RETP * ADJINC / 1_000_000",
            "social_security": "SSP * ADJINC / 1_000_000",
            "code_citations": [
                "packages/microcosm-build/src/microcosm/build/us_runtime/acs_inputs.py:133",
                "packages/microcosm-build/src/microcosm/build/us_runtime/acs_inputs.py:177",
                "packages/microcosm-build/src/microcosm/build/us_runtime/acs_inputs.py:307",
            ],
        },
        "pension_and_ira_bundle_vs_native_RETP": retirement,
        "social_security_components_vs_native_SSP": social_security,
        "unobserved_leaf_labels": {
            "ACS_RETP": [
                "tax_exempt_private_pension_income",
                "taxable_private_pension_income",
                "taxable_ira_distributions",
                "keogh_distributions",
                "taxable_401k_distributions",
                "taxable_403b_distributions",
                "taxable_sep_distributions",
            ],
            "ACS_SSP": [
                "social_security_retirement",
                "social_security_disability",
                "social_security_dependents",
                "social_security_survivors",
            ],
            "interpretation": (
                "ACS supplies only combined RETP and SSP amounts; the named target "
                "leaves are modeled labels, not separately observed ACS columns"
            ),
        },
    }


def _gate_records(gates: Mapping[str, object]) -> dict[str, object]:
    _require(
        gates.get("agreement_gate") == gates.get("terminal_gates"), "gate copies differ"
    )
    terminal = gates.get("terminal_gates")
    _require(isinstance(terminal, dict), "terminal_gates is not an object")
    gate_map = terminal.get("gates")
    _require(isinstance(gate_map, dict), "terminal gate map is not an object")
    battery = gate_map.get("us_by_origin_battery")
    _require(isinstance(battery, dict), "by-origin battery gate is absent")
    details = battery.get("details")
    _require(isinstance(details, dict), "by-origin battery details are absent")
    comparisons = details.get("comparisons")
    _require(isinstance(comparisons, dict), "battery comparisons are absent")
    selected: dict[str, object] = {}
    for spec in LEGS:
        _require(spec.gate_label in comparisons, f"missing gate {spec.gate_label}")
        selected[spec.gate_label] = comparisons[spec.gate_label]
    return selected


def _terminal_proof(
    person: Mapping[str, np.ndarray],
    person_weights: np.ndarray,
    asec_clone0: np.ndarray,
    acs_clone0: np.ndarray,
    gate_records: Mapping[str, object],
) -> dict[str, object]:
    result: dict[str, object] = {}
    for spec in LEGS:
        values = np.asarray(person[spec.target], dtype=np.float64)
        asec = _origin_stats(values, person_weights, asec_clone0)
        acs = _origin_stats(values, person_weights, acs_clone0)
        record = gate_records[spec.gate_label]
        _require(
            isinstance(record, dict), f"{spec.target}: gate record is not an object"
        )
        legs = record.get("legs")
        _require(isinstance(legs, dict), f"{spec.target}: gate legs are absent")
        positive_gate = legs.get("positive")
        negative_gate = legs.get("negative")
        _require(
            isinstance(positive_gate, dict) and isinstance(negative_gate, dict),
            f"{spec.target}: sign gate records are absent",
        )
        for origin_name, stats in (("asec", asec), ("acs", acs)):
            sign_counts = stats["sign_counts"]
            incidence = stats["weighted_incidence"]
            _require(
                positive_gate["nonzero_rows"][origin_name] == sign_counts["positive"]
                and negative_gate["nonzero_rows"][origin_name]
                == sign_counts["negative"],
                f"{spec.target}: terminal carrier counts do not match gate",
            )
            _assert_close(
                incidence["positive"],
                positive_gate[f"{origin_name}_incidence"],
                label=f"{spec.target}/{origin_name}/positive incidence",
            )
            _assert_close(
                incidence["negative"],
                negative_gate[f"{origin_name}_incidence"],
                label=f"{spec.target}/{origin_name}/negative incidence",
            )

        ratio: float | None = None
        qed: float | None = None
        asec_incidence = asec["weighted_incidence"]["positive"]
        acs_incidence = acs["weighted_incidence"]["positive"]
        if asec_incidence > 0:
            ratio = float(acs_incidence / asec_incidence)
            _assert_close(
                ratio,
                positive_gate["incidence_ratio_acs_over_asec"],
                label=f"{spec.target}/incidence ratio",
            )
        elif acs_incidence > 0:
            _require(
                positive_gate.get("incidence_ratio_acs_over_asec") == "inf",
                f"{spec.target}: one-sided carrier gate changed",
            )
        if (
            asec["sign_counts"]["positive"] >= BATTERY_MIN_EFFECTIVE_SUPPORT
            and acs["sign_counts"]["positive"] >= BATTERY_MIN_EFFECTIVE_SUPPORT
        ):
            qed = _quantile_envelope_distance(
                asec["positive_weighted_quantiles"],
                acs["positive_weighted_quantiles"],
            )
            _assert_close(
                qed,
                positive_gate["quantile_envelope_distance"],
                label=f"{spec.target}/quantile envelope",
            )
        result[spec.target] = {
            "acs": acs,
            "asec": asec,
            "gate_label": spec.gate_label,
            "incidence_ratio_acs_over_asec": ratio,
            "positive_quantile_envelope_distance": qed,
            "selected_gate_record": record,
        }
    return result


def _adjudication_proof(adjudication: Mapping[str, object]) -> dict[str, object]:
    rows = adjudication.get("adjudications")
    _require(isinstance(rows, list), "adjudication rows are not a list")
    selected = [
        row
        for row in rows
        if isinstance(row, dict)
        and row.get("action_group") == "retirement_model_and_data"
    ]
    _require(len(selected) == 16, "retirement adjudication is not 16 checks")
    leg_ids = sorted({str(row["leg_id"]) for row in selected})
    _require(len(leg_ids) == 11, "retirement adjudication is not 11 physical legs")
    _require(
        all(row.get("classification") == "BLOCKER" for row in selected),
        "retirement adjudication contains a non-BLOCKER classification",
    )
    compact = [
        {
            "baseline": row["baseline"],
            "best_arm_movement": row["best_arm_movement"],
            "check_id": row["check_id"],
            "classification": row["classification"],
            "criterion": row["criterion"],
            "leg_id": row["leg_id"],
            "ordinal": row["ordinal"],
        }
        for row in sorted(selected, key=lambda row: row["ordinal"])
    ]
    return {
        "action_group": "retirement_model_and_data",
        "failed_checks": len(compact),
        "physical_leg_ids": leg_ids,
        "physical_legs": len(leg_ids),
        "rows": compact,
        "source_base_commit": adjudication.get("base_commit"),
    }


def _build_report(args: argparse.Namespace) -> dict[str, object]:
    primary_paths = {
        "asec_raw_stage": args.asec_raw_stage,
        "baseline_pool": args.baseline_pool,
        "baseline_gates": args.baseline_gates,
        "pkg3_pool": args.pkg3_pool,
        "pkg3_gates": args.pkg3_gates,
        "assembled_checkpoint": args.assembled_checkpoint,
        "transferred_checkpoint": args.transferred_checkpoint,
        "adjudication": args.adjudication,
    }
    artifacts = {
        label: _verify_file(path, PRIMARY_SHA256[label], label=label)
        for label, path in primary_paths.items()
    }
    target_paths = {
        spec.target: args.transfer_bank / spec.checkpoint_relative for spec in LEGS
    }
    artifacts["target_checkpoints"] = {
        spec.target: _verify_file(
            target_paths[spec.target],
            spec.checkpoint_sha256,
            label=f"target_checkpoint:{spec.target}",
        )
        for spec in LEGS
    }

    # No artifact is decoded before all frozen byte identities above pass.
    person, person_weights = _read_checkpoint_surface(args.transferred_checkpoint)
    channels = np.asarray(person["person_support_channel"], dtype=object)
    _require(
        all(isinstance(value, str) for value in channels),
        "person support channel contains a non-string",
    )
    clones = np.asarray(person["person_support_clone_index"], dtype=np.int64)
    positive_weight = person_weights > 0
    asec_clone0 = (channels == "asec") & (clones == 0) & positive_weight
    acs_clone0 = (channels == "acs") & (clones == 0) & positive_weight
    asec_clone1 = (channels == "asec") & (clones == 1) & positive_weight
    _require(
        (int(asec_clone0.sum()), int(acs_clone0.sum()), int(asec_clone1.sum()))
        == (4311, 34293, 4311),
        "frozen support-role row counts changed",
    )

    retirement_source_columns = {
        *PENSION_SOURCES,
        *SS_SOURCES,
        *SLOT_SOURCES,
    }
    raw_person, raw_inventory, raw_schema_version = _read_checkpoint_person_columns(
        args.asec_raw_stage,
        selected={"person_id", *retirement_source_columns},
    )
    assembled_person, assembled_inventory, assembled_schema_version = (
        _read_checkpoint_person_columns(
            args.assembled_checkpoint,
            selected={
                "person_source_id",
                "person_support_channel",
                "person_support_clone_index",
                *retirement_source_columns,
            },
        )
    )
    _, transferred_inventory, transferred_schema_version = (
        _read_checkpoint_person_columns(args.transferred_checkpoint, selected=set())
    )
    for label, inventory in (
        ("raw", raw_inventory),
        ("assembled", assembled_inventory),
        ("transferred", transferred_inventory),
    ):
        missing = sorted(retirement_source_columns - inventory)
        _require(not missing, f"{label} checkpoint lost retirement sources: {missing}")
        _require(
            {"PEN_VAL1", "PEN_VAL2"}.isdisjoint(inventory),
            f"{label} checkpoint unexpectedly contains source-specific pension amounts",
        )

    assembled_channels = np.asarray(
        assembled_person["person_support_channel"], dtype=object
    )
    assembled_clones = np.asarray(
        assembled_person["person_support_clone_index"], dtype=np.int64
    )
    assembled_asec_clone0 = (assembled_channels == "asec") & (assembled_clones == 0)
    _require(
        int(assembled_asec_clone0.sum()) == 4311,
        "assembled ASEC clone-0 row count changed",
    )
    _require(
        np.array_equal(
            np.asarray(assembled_person["person_source_id"])[assembled_asec_clone0],
            np.asarray(person["person_source_id"])[asec_clone0],
        ),
        "assembled/transferred ASEC clone-0 source-person order changed",
    )

    raw_person_ids = np.asarray(raw_person["person_id"], dtype=np.int64)
    assembled_source_ids = np.asarray(
        assembled_person["person_source_id"], dtype=np.int64
    )[assembled_asec_clone0]
    _require(
        bool((raw_person_ids[1:] > raw_person_ids[:-1]).all()),
        "raw ASEC person IDs are not strictly increasing",
    )
    _require(
        len(np.unique(assembled_source_ids)) == len(assembled_source_ids),
        "assembled ASEC clone-0 source-person IDs are not unique",
    )
    raw_positions = np.searchsorted(raw_person_ids, assembled_source_ids)
    _require(
        bool((raw_positions < len(raw_person_ids)).all())
        and np.array_equal(raw_person_ids[raw_positions], assembled_source_ids),
        "assembled ASEC clone-0 source-person IDs do not resolve in raw ASEC",
    )
    for column in sorted(retirement_source_columns):
        raw_values = _coerce_numeric(raw_person[column][raw_positions], missing=np.nan)
        assembled_values = _coerce_numeric(
            assembled_person[column][assembled_asec_clone0], missing=np.nan
        )
        transferred_values = _coerce_numeric(
            person[column][asec_clone0], missing=np.nan
        )
        _require(
            np.array_equal(raw_values, assembled_values, equal_nan=True),
            f"raw/assembled ASEC clone-0 source changed: {column}",
        )
        _require(
            np.array_equal(assembled_values, transferred_values, equal_nan=True),
            f"assembled/transferred ASEC clone-0 source changed: {column}",
        )

    source_outputs, source_detail = _derive_sources(person, asec_clone0)
    target_bank = {
        spec.target: _read_target_checkpoint(target_paths[spec.target], spec)
        for spec in LEGS
    }

    gate_objects: dict[str, Mapping[str, object]] = {}
    for label, path in (
        ("baseline", args.baseline_gates),
        ("pkg3", args.pkg3_gates),
    ):
        parsed = json.loads(path.read_text(encoding="utf-8"))
        _require(isinstance(parsed, dict), f"{label} gates are not an object")
        gate_objects[label] = parsed
    baseline_records = _gate_records(gate_objects["baseline"])
    pkg3_records = _gate_records(gate_objects["pkg3"])
    _require(baseline_records == pkg3_records, "baseline/pkg3 retirement gates differ")
    selected_gate_sha256 = _mapping_sha256(baseline_records)
    _require(
        selected_gate_sha256 == EXPECTED_CANONICAL_SELECTED_GATE_RECORDS_SHA256,
        "selected retirement gate digest changed",
    )

    terminal = _terminal_proof(
        person,
        person_weights,
        asec_clone0,
        acs_clone0,
        baseline_records,
    )

    selected_pool_columns = {
        "person_id",
        "person_household_id",
        "person_support_clone_index",
        *(spec.target for spec in LEGS),
    }
    baseline_index, baseline_pool = _read_fixed_numeric_columns(
        args.baseline_pool, entity="person", selected=selected_pool_columns
    )
    pkg3_index, pkg3_pool = _read_fixed_numeric_columns(
        args.pkg3_pool, entity="person", selected=selected_pool_columns
    )
    _require(np.array_equal(baseline_index, pkg3_index), "pool row indexes differ")
    _require(
        np.array_equal(baseline_index, np.arange(len(baseline_index))),
        "pool row index is not the frozen 0-based order",
    )
    pool_column_sha256: dict[str, str] = {}
    for column in sorted(selected_pool_columns):
        left = baseline_pool[column]
        right = pkg3_pool[column]
        _require(
            left.dtype == right.dtype and np.array_equal(left, right),
            f"baseline/pkg3 pool column differs: {column}",
        )
        checkpoint_values = np.asarray(person[column])
        _require(
            left.dtype == checkpoint_values.dtype
            and np.array_equal(left, checkpoint_values),
            f"transferred/final pool column differs: {column}",
        )
        digest = hashlib.sha256(np.ascontiguousarray(left).tobytes()).hexdigest()
        pool_column_sha256[column] = digest

    source_reconciliation: dict[str, object] = {}
    donor_support: dict[str, object] = {}
    legs: dict[str, object] = {}
    for spec in LEGS:
        target_source = source_outputs[spec.target]
        clone0_values = np.asarray(person[spec.target][asec_clone0], dtype=np.float64)
        bitwise_equal = _bitwise_float_equal(target_source, clone0_values)
        _require(bitwise_equal, f"{spec.target}: source does not bit-match clone 0")
        source_counts = _sign_counts(target_source)
        _require(
            source_counts["positive"] == EXPECTED_SOURCE_POSITIVE[spec.target]
            and source_counts["negative"] == 0,
            f"{spec.target}: source sign support changed",
        )
        source_reconciliation[spec.target] = {
            "bitwise_equal": bitwise_equal,
            "clone0_target": _amount_stats(clone0_values),
            "max_absolute_error": float(np.max(np.abs(target_source - clone0_values))),
            "source_derived": _amount_stats(target_source),
        }

        donor_mask = asec_clone0 if spec.stage.startswith("early") else asec_clone1
        donor_clone = 0 if spec.stage.startswith("early") else 1
        donor_values = np.asarray(person[spec.target][donor_mask], dtype=np.float64)
        donor_counts = _sign_counts(donor_values)
        if spec.stage.startswith("early"):
            _require(
                donor_counts == source_counts,
                f"{spec.target}: early donor support differs from source clone 0",
            )
        else:
            _require(
                donor_counts["positive"] == EXPECTED_LATE_DONOR_POSITIVE[spec.target]
                and donor_counts["negative"] == 0,
                f"{spec.target}: late donor sign support changed",
            )
        regime = _detect_regime(donor_values)
        expected_regime = (
            "degenerate_zero"
            if spec.target == "keogh_distributions"
            else "zero_inflated_positive"
        )
        _require(regime == expected_regime, f"{spec.target}: donor regime changed")
        donor_support[spec.target] = {
            "donor_channel": "asec",
            "donor_clone_index": donor_clone,
            "donor_index": EXPECTED_DONOR_INDEX,
            "patterns": {
                pattern["pattern"]: {
                    "regime": regime,
                    "sign_counts": donor_counts,
                    "zero_atol": ZERO_ATOL,
                }
                for pattern in target_bank[spec.target]["patterns"]
            },
            "regime": regime,
            "sign_counts": donor_counts,
            "zero_atol": ZERO_ATOL,
        }
        if spec.target.startswith("social_security_"):
            disagreement_entry = (
                "ASEC partitions SS_VAL by RESNSS1/RESNSS2 precedence, while ACS "
                "supplies only combined SSP; the four component labels are unobserved"
            )
        elif spec.stage == "late_producer_dag":
            disagreement_entry = (
                "ACS supplies only combined RETP, and the late fit uses "
                "ASEC-origin clone-1 output from the internal CPS-trained PUF-role "
                "QRF instead of the exact ASEC clone-0 account label"
            )
        elif "pension" in spec.target:
            disagreement_entry = (
                "ASEC turns one PNSN_VAL-plus-ANN_VAL total into this 59/41 leaf, "
                "while ACS supplies only combined RETP and the pension leaves are "
                "predicted in independent transfer chains"
            )
        else:
            disagreement_entry = (
                "ASEC supplies the exact code-4 account label, while ACS has no "
                "IRA leaf and supplies only combined RETP as an optional predictor"
            )
        legs[spec.target] = {
            "classification": spec.classification,
            "code_citations": list(spec.code_citations),
            "equation": spec.equation,
            "family": spec.family,
            "source_columns": list(spec.source_columns),
            "stage": spec.stage,
            "where_asec_vs_acs_disagreement_enters": disagreement_entry,
        }

    adjudication = json.loads(args.adjudication.read_text(encoding="utf-8"))
    _require(isinstance(adjudication, dict), "adjudication is not an object")
    classifications = {
        "concept_mismatch": sorted(
            spec.target for spec in LEGS if spec.classification == "concept_mismatch"
        ),
        "dense_rung_refit_required": sorted(
            spec.target
            for spec in LEGS
            if spec.classification == "dense_rung_refit_required"
        ),
        "derivation_defect": sorted(
            spec.target for spec in LEGS if spec.classification == "derivation_defect"
        ),
        "owner_exclusions_added": [],
    }
    _require(
        len(classifications["concept_mismatch"]) == 6
        and len(classifications["dense_rung_refit_required"]) == 5
        and classifications["derivation_defect"] == []
        and classifications["owner_exclusions_added"] == []
        and set(classifications["concept_mismatch"])
        | set(classifications["dense_rung_refit_required"])
        == {spec.target for spec in LEGS},
        "retirement classification partition changed",
    )
    return {
        "acs_native_disagreement": _native_disagreement(person, acs_clone0),
        "adjudication": _adjudication_proof(adjudication),
        "artifacts": artifacts,
        "baseline_vs_pkg3": {
            "pool_row_count": len(baseline_index),
            "selected_gate_records_equal": True,
            "selected_gate_records_sha256": selected_gate_sha256,
            "selected_pool_columns_bitwise_equal": True,
            "selected_pool_columns_sha256": pool_column_sha256,
            "transferred_to_final_columns_bitwise_equal": True,
        },
        "classifications": classifications,
        "donor_support": donor_support,
        "legs": legs,
        "schema_version": 1,
        "scope": {
            "battery_min_effective_support": BATTERY_MIN_EFFECTIVE_SUPPORT,
            "builds_run": 0,
            "frozen_sample_fraction": 0.01,
            "frozen_sample_seed": 578,
            "model_seed": 0,
            "negative_sign_carriers_total": 0,
            "retirement_failed_checks_at_f025": 16,
            "retirement_physical_legs": 11,
            "stacked_checkpoint_directory_identity": (
                "2e45c4d60f66b4321bc00ffa22816470bf162c59fd91956514832f97e066ed3c"
            ),
            "stacked_checkpoint_directory_path": str(
                args.transferred_checkpoint.resolve().parent
            ),
        },
        "source_schema_and_carriage": {
            "raw_to_assembled_asec_clone0": {
                "join": "raw.person_id == assembled.person_source_id",
                "person_rows": int(asec_clone0.sum()),
                "source_columns": sorted(retirement_source_columns),
                "source_person_ids_unique_and_resolved": True,
                "values_equal_with_missing_preserved": True,
            },
            "assembled_to_transferred_asec_clone0": {
                "person_rows": int(asec_clone0.sum()),
                "source_columns": sorted(retirement_source_columns),
                "values_equal_with_missing_preserved": True,
            },
            "checkpoint_schema_versions": {
                "asec_raw_stage": raw_schema_version,
                "assembled": assembled_schema_version,
                "transferred": transferred_schema_version,
            },
            "pension_source_specific_amount_columns": {
                "absent_columns": ["PEN_VAL1", "PEN_VAL2"],
                "absent_from": [
                    "asec_raw_stage",
                    "assembled",
                    "transferred",
                ],
            },
        },
        "source_detail": source_detail,
        "source_to_clone0_reconciliation": source_reconciliation,
        "target_bank": target_bank,
        "terminal_f001": terminal,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--asec-raw-stage", type=Path, default=ASEC_RAW_STAGE)
    parser.add_argument("--baseline-pool", type=Path, default=BASELINE_DIR / "pool.h5")
    parser.add_argument(
        "--baseline-gates", type=Path, default=BASELINE_DIR / "pool.gates.json"
    )
    parser.add_argument("--pkg3-pool", type=Path, default=PKG3_DIR / "pool.h5")
    parser.add_argument("--pkg3-gates", type=Path, default=PKG3_DIR / "pool.gates.json")
    parser.add_argument(
        "--assembled-checkpoint",
        type=Path,
        default=STACKED_CHECKPOINT / "assembled.checkpoint.h5",
    )
    parser.add_argument(
        "--transferred-checkpoint",
        type=Path,
        default=STACKED_CHECKPOINT / "transferred.checkpoint.h5",
    )
    parser.add_argument("--transfer-bank", type=Path, default=TRANSFER_BANK)
    parser.add_argument("--adjudication", type=Path, default=DEFAULT_ADJUDICATION)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--check",
        action="store_true",
        help="Compare regenerated canonical bytes with --output instead of writing.",
    )
    return parser


def _atomic_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            os.fchmod(stream.fileno(), 0o644)
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def main() -> None:
    args = _parser().parse_args()
    report = _build_report(args)
    payload = _json_bytes(report)
    if args.check:
        _require(args.output.is_file(), f"audit output does not exist: {args.output}")
        observed = args.output.read_bytes()
        _require(observed == payload, f"audit output is stale: {args.output}")
        print(f"retirement frozen-artifact audit is current: {args.output}")
        return
    _atomic_write(args.output, payload)
    print(f"wrote retirement frozen-artifact audit: {args.output}")


if __name__ == "__main__":
    main()
