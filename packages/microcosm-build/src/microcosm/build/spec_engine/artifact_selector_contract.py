"""Single-authored wire contract for execution-ABI artifact selectors.

The compiler seals this document and the collector refuses any other one.  It
is deliberately data, rather than prose next to either caller, so a selector
change cannot silently leave the execution ABI describing the old behavior.
"""

from __future__ import annotations

import copy
import re
from datetime import datetime

from .canonical import sha256_json

ARTIFACT_LOCATOR_GRAMMAR = (
    "closed-runtime-output-plan-checkpoint-receipt-and-selector-contract-v4"
)
H5_TIME_PERIOD_KEY = "/_time_period"
H5_ARTIFACT_METADATA_KEY = "/_populace_staging_metadata"
H5_OPERATIONAL_METADATA_FIELDS = ("publication_run_id",)
RELEASE_ID_BRANDS = {"constants": "populace", "bundle": "microcosm"}
RELEASE_ID_TIMESTAMP_FORMAT = "%Y%m%dT%H%M%SZ"
RELEASE_ID_NONCE_PATTERN = "[0-9a-f]{8}"
_RELEASE_ID_PATTERN = re.compile(
    rf"^(?P<brand>[a-z][a-z0-9]*)-(?P<semantic>.+)-"
    rf"(?P<timestamp>[0-9]{{8}}T[0-9]{{6}}Z)-"
    rf"(?P<nonce>{RELEASE_ID_NONCE_PATTERN})$"
)


def _h5_header_contract() -> dict[str, object]:
    return {
        "record": "tag_H_then_u64be_length_and_microcosm_canonical_json_v1",
        "time_period_key": H5_TIME_PERIOD_KEY,
        "time_period_shape": "series_of_exactly_one_integer",
        "artifact_metadata_key": H5_ARTIFACT_METADATA_KEY,
        "artifact_metadata_shape": "series_of_exactly_one_strict_json_object",
        "excluded_metadata_fields": list(H5_OPERATIONAL_METADATA_FIELDS),
        "object_fields": ["artifact_metadata", "time_period"],
    }


_SELECTOR_ROWS = (
    {
        "id": "selector:canonical_json_bytes_v1",
        "input": "compiler_json_value",
        "encoding": "microcosm_canonical_json_v1",
    },
    {
        "id": "selector:directory_tree_bytes_v1",
        "input": "directory",
        "domain": "microcosm.selector.directory-tree-bytes.v1\0",
        "ordering": "utf8_posix_relative_path_bytes_lexicographic",
        "record": "tag_D_or_F_then_u64be_path_length_and_path_then_for_F_u64be_byte_length_and_literal_bytes",
        "directories": "explicit_records_including_empty_directories",
        "links_and_special_files": "refuse",
    },
    {
        "id": "selector:file_bytes_v1",
        "input": "regular_file",
        "encoding": "literal_complete_file_bytes",
        "links_and_special_files": "refuse",
    },
    {
        "id": "selector:h5_all_entity_tables_and_columns_v1",
        "input": "nullable_pool_h5",
        "domain": "microcosm.selector.h5-all-entity-tables-and-columns.v1\0",
        "header": _h5_header_contract(),
        "tables": "utf8_hdf_key_bytes_lexicographic_excluding_exact_reserved_header_keys",
        "columns": "stored_column_order_excluding_exact_entity_weight_column",
        "table_record": "tag_T_then_framed_hdf_key_then_index_then_u64be_column_count_then_column_records",
    },
    {
        "id": "selector:h5_all_weight_vectors_v1",
        "input": "nullable_pool_h5",
        "domain": "microcosm.selector.h5-all-weight-vectors.v1\0",
        "header": _h5_header_contract(),
        "tables": "utf8_hdf_key_bytes_lexicographic_excluding_exact_reserved_header_keys",
        "columns": "exact_entity_weight_column_only",
        "table_record": "tag_T_then_framed_hdf_key_then_index_then_u64be_column_count_then_column_records",
    },
    {
        "id": "selector:publication_normative_vector_v1",
        "input": "strict_json_object",
        "encoding": "apply_exact_compiler_receipt_rules_then_microcosm_canonical_json_v1",
    },
    {
        "id": "selector:terminal_gate_normative_rows_v1",
        "input": "strict_json_object",
        "encoding": "apply_exact_compiler_receipt_rules_then_microcosm_canonical_json_v1",
    },
)

_CONTRACT = {
    "schema_version": 2,
    "framing_endianness": "big",
    "length_width_bytes": 8,
    "release_id_normalization": {
        "brands_by_authority_mode": dict(RELEASE_ID_BRANDS),
        "semantic_middle": "preserve_exactly",
        "terminal_timestamp": RELEASE_ID_TIMESTAMP_FORMAT,
        "terminal_nonce_regex": RELEASE_ID_NONCE_PATTERN,
        "normal_form": "semantic_middle_only",
    },
    "artifact_binding": {
        "raw_identity": "resolved_path_sha256_size_bytes_v1",
        "embedded_identity_protocols": [
            "h5_artifact_metadata_v1",
            "json_root_publication_identity_v1",
        ],
        "authentication_order": "before_receipt_rule_normalization",
    },
    "h5_logical_codec": {
        "column_record": "tag_C_then_framed_utf8_name_then_framed_canonical_dtype_descriptor_then_u64be_value_count_then_values",
        "index_record": "tag_I_then_framed_canonical_descriptor_then_u64be_value_count_then_values",
        "index_descriptor": "kind_dtype_and_name_or_multi_index_dtypes_and_names",
        "categorical_dtype": "ordered_categories_count_and_sha256_over_concatenated_scalar_records",
        "scalar_tags": {
            "null": "N",
            "bool_false": "B0",
            "bool_true": "B1",
            "integer": "Z_then_framed_ascii_decimal",
            "float64": "R_then_ieee754_binary64_big_endian",
            "utf8_string": "S_then_framed_bytes",
            "bytes": "Y_then_framed_bytes",
            "timestamp": "P_then_framed_utf8_timezone_then_i64be_nanoseconds",
            "timedelta": "L_then_i64be_nanoseconds",
            "tuple": "Q_then_u64be_count_then_recursive_values",
        },
    },
    "selectors": list(_SELECTOR_ROWS),
}

ARTIFACT_SELECTOR_CONTRACT_SHA256 = sha256_json(_CONTRACT)


def artifact_selector_contract_wire() -> dict[str, object]:
    """Return a fresh JSON wire copy of the selector semantics."""

    wire = copy.deepcopy(_CONTRACT)
    wire["sha256"] = ARTIFACT_SELECTOR_CONTRACT_SHA256
    return wire


def normalize_release_id(value: object, *, authority_mode: str) -> str:
    """Validate one release id and return its preserved semantic middle.

    A release id is generation-specific only at its leading brand and its
    terminal operational timestamp/nonce pair.  Everything between those
    fields is behavior-bearing and therefore survives normalization exactly.
    """

    expected_brand = RELEASE_ID_BRANDS.get(authority_mode)
    if expected_brand is None:
        raise ValueError("authority_mode must be 'constants' or 'bundle'")
    if not isinstance(value, str):
        raise ValueError("release id must be a string")
    match = _RELEASE_ID_PATTERN.fullmatch(value)
    if match is None or match.group("brand") != expected_brand:
        raise ValueError(
            f"release id must use the {expected_brand!r} brand and sealed grammar"
        )
    semantic = match.group("semantic")
    if any(not segment for segment in semantic.split("-")):
        raise ValueError("release id semantic middle contains an empty segment")
    try:
        datetime.strptime(match.group("timestamp"), RELEASE_ID_TIMESTAMP_FORMAT)
    except ValueError as error:
        raise ValueError("release id has an invalid UTC timestamp") from error
    return semantic


__all__ = [
    "ARTIFACT_LOCATOR_GRAMMAR",
    "ARTIFACT_SELECTOR_CONTRACT_SHA256",
    "H5_ARTIFACT_METADATA_KEY",
    "H5_OPERATIONAL_METADATA_FIELDS",
    "H5_TIME_PERIOD_KEY",
    "RELEASE_ID_BRANDS",
    "RELEASE_ID_NONCE_PATTERN",
    "RELEASE_ID_TIMESTAMP_FORMAT",
    "artifact_selector_contract_wire",
    "normalize_release_id",
]
