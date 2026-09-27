"""Convert supplied UK ladder arrays for shared atomic geography operators.

This module does not acquire sources, admit native data, calibrate, or assign a
location. Source identities and mapping classifications are explicit caller
claims; publisher authentication belongs to the host. Initial spine/support
clones must exist before the resulting declaration is executed.
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Mapping
from numbers import Integral

import numpy as np

from microcosm.build.atomic_geography import (
    RELATIONS,
    decode_atomic_support,
    encode_atomic_support,
    validate_assignment_spec,
)
from microcosm.build.uk_runtime.rowwise_geography import FRS_REGION_TO_REGION_CODE
from microcosm.graph.canonical import canonical_json

SYSTEMS = (
    "uk_ew_output_area_2021",
    "uk_scotland_output_area_2022",
    "uk_ni_data_zone_2021",
)
SOURCES = {system: system + "_support" for system in SYSTEMS}
IDENTITY_COLUMN = "geography_household_key"
_INPUT_COLUMNS = (
    "oa_code",
    "population",
    "households",
    "constituency_code",
    "region_code",
    "lsoa_code",
    "msoa_code",
    "local_authority_code",
    "ward_code",
    "itl3_code",
)
_COUNT_COLUMNS = frozenset({"population", "households"})
_CODE = re.compile(r"[EWSN][0-9]{8}")
_WARD = re.compile(r"[EWSN][0-9]{2}[0-9RS][0-9]{5}")
_ITL = re.compile(r"TL[C-N][0-9A-Z]{2}")
_LIMIT = 2**53


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError("UK atomic support: " + message)


def _text(value: object) -> bool:
    return type(value) is str and bool(value) and value.strip() == value


def _profile(system: str) -> tuple[str, str, tuple[str, ...], str]:
    _require(system in SYSTEMS, "unknown area system")
    if system == SYSTEMS[0]:
        regions = tuple(
            name
            for name in FRS_REGION_TO_REGION_CODE
            if name not in {"SCOTLAND", "NORTHERN_IRELAND"}
        )
        return "output_area", "2021_census", regions, "EW"
    if system == SYSTEMS[1]:
        return "output_area", "2022_census", ("SCOTLAND",), "S"
    return "data_zone", "2021_census", ("NORTHERN_IRELAND",), "N"


def _metadata(descriptions: Mapping, *, vintage: str, system: str) -> dict:
    _require(type(descriptions) is dict, "column metadata must be a plain mapping")
    _require(set(descriptions) == set(_INPUT_COLUMNS), "exact column metadata required")
    result = {}
    for column in _INPUT_COLUMNS:
        item = descriptions[column]
        _require(type(item) is dict, "plain column description required")
        expected = (
            {"kind", "source", "basis"}
            if column in _COUNT_COLUMNS
            else {"kind", "source", "vintage", "relation"}
        )
        _require(set(item) == expected, "column description keys differ")
        _require(
            all(_text(value) for value in item.values()), "nonempty metadata required"
        )
        if column in _COUNT_COLUMNS:
            _require(item["kind"] == "weight", "count metadata must describe weight")
        else:
            _require(
                item["kind"] == "code" and item["relation"] in RELATIONS,
                "code mapping classification",
            )
        result[column] = dict(item)
    _require(
        result["oa_code"]["vintage"] == vintage
        and result["oa_code"]["relation"] == "exact",
        "atomic identity/vintage differs",
    )
    if system == SYSTEMS[2]:
        _require(
            result["constituency_code"]["relation"] != "inferred_modal",
            "NI postcode-modal mapping is not admitted by this adapter",
        )
    return result


def _counts(array: np.ndarray, *, positive: bool) -> np.ndarray:
    _require(
        array.dtype.kind in "iu" or array.dtype == np.dtype("float64"), "count dtype"
    )
    if array.dtype.kind == "f":
        _require(np.isfinite(array).all(), "non-finite count")
        _require(np.equal(array, np.floor(array)).all(), "non-integral count")
    _require(
        np.all(array > 0 if positive else array >= 0) and np.all(array <= _LIMIT),
        "count range",
    )
    integers = [int(value) for value in array]
    _require(0 < sum(integers) <= _LIMIT, "count total outside exact sampling range")
    converted = np.asarray(integers, dtype=np.int64)
    _require(
        np.array_equal(converted.astype(array.dtype), array),
        "count conversion is not exact",
    )
    return converted


def assemble_uk_atomic_area_support(
    *,
    system: str,
    arrays: Mapping[str, np.ndarray],
    column_metadata: Mapping,
) -> bytes:
    """Normalize one complete nation system from supplied legacy ladder columns.

    Rows are sorted by atomic code, never dropped. Counts remain exact; old
    float64 count arrays are accepted only when integral and exactly representable.
    Each declared FRS region must have positive household sampling mass. The
    caller supplies source/vintage/relation evidence; syntax is not authority.
    NI legacy ward_code means district electoral area, and lsoa_code aliases DZ.
    """
    level, vintage, regions, prefixes = _profile(system)
    descriptions = _metadata(column_metadata, vintage=vintage, system=system)
    _require(
        type(arrays) is dict and set(arrays) == set(_INPUT_COLUMNS),
        "exact arrays required",
    )
    values = {}
    for column in _INPUT_COLUMNS:
        array = arrays[column]
        _require(
            type(array) is np.ndarray and array.ndim == 1 and len(array) > 0,
            "nonempty one-dimensional arrays required",
        )
        if column in _COUNT_COLUMNS:
            values[column] = _counts(array, positive=column == "population")
            continue
        _require(array.dtype.kind == "U", "codes must be Unicode arrays")
        _require(
            np.all(np.char.str_len(array) > 0)
            and np.all(np.char.strip(array) == array),
            "blank or padded code",
        )
        pattern = (
            _ITL if column == "itl3_code" else _WARD if column == "ward_code" else _CODE
        )
        _require(
            all(pattern.fullmatch(str(value)) for value in array), "invalid code syntax"
        )
        if column != "itl3_code":
            _require(
                all(str(value)[0] in prefixes for value in array), "foreign nation code"
            )
        values[column] = array.copy()
    count = len(values["oa_code"])
    _require(
        all(len(array) == count for array in values.values()), "array lengths differ"
    )
    _require(len(np.unique(values["oa_code"])) == count, "duplicate atomic code")
    nations = [str(value)[0] for value in values["oa_code"]]
    for column in _INPUT_COLUMNS:
        if column not in {*_COUNT_COLUMNS, "itl3_code"}:
            _require(
                [str(value)[0] for value in values[column]] == nations,
                "mapping crosses a nation boundary",
            )
    if system == SYSTEMS[2]:
        _require(
            np.array_equal(values["lsoa_code"], values["oa_code"]),
            "NI legacy lsoa_code must alias its atomic Data Zone",
        )
    region_names = {FRS_REGION_TO_REGION_CODE[name]: name for name in regions}
    _require(
        set(values["region_code"].tolist()) == set(region_names),
        "incomplete or foreign FRS region coverage",
    )
    for code in region_names:
        _require(
            sum(int(v) for v in values["households"][values["region_code"] == code])
            > 0,
            "region has no household sampling mass",
        )
    order = np.argsort(values["oa_code"], kind="stable")
    values = {column: array[order] for column, array in values.items()}
    normalized = {"area": values["oa_code"]}
    normalized.update({k: v for k, v in values.items() if k != "oa_code"})
    columns = {"area": descriptions["oa_code"]}
    columns.update({k: v for k, v in descriptions.items() if k != "oa_code"})
    normalized["frs_region"] = np.asarray(
        [region_names[code] for code in values["region_code"]], dtype="U"
    )
    region_rule = canonical_json(
        {
            "mapping": dict(FRS_REGION_TO_REGION_CODE),
            "input": descriptions["region_code"],
        }
    )
    columns["frs_region"] = {
        "kind": "code",
        "source": "sha256:" + hashlib.sha256(region_rule).hexdigest(),
        "vintage": "frs_region_enum_v1",
        "relation": "exact",
    }
    for width in (3, 4):
        column = "itl1_code" if width == 3 else "itl2_code"
        normalized[column] = np.asarray([code[:width] for code in values["itl3_code"]])
        columns[column] = {
            **descriptions["itl3_code"],
            "source": descriptions["itl3_code"]["source"] + f"#prefix-{width}",
            "relation": "exact",
        }
    return encode_atomic_support(
        {
            "version": 1,
            "system": system,
            "level": level,
            "code_system": "uk_gss",
            "vintage": vintage,
            "columns": columns,
        },
        normalized,
    )


def _layers(system: str) -> tuple[tuple[str, str], ...]:
    shared = tuple(
        (column, column)
        for column in _INPUT_COLUMNS
        if column not in {"oa_code", *_COUNT_COLUMNS}
    )
    aliases = (("area", "oa_code"),)
    if system == SYSTEMS[0]:
        native = (("area", "output_area_code"),)
    elif system == SYSTEMS[1]:
        native = (
            ("area", "output_area_code"),
            ("lsoa_code", "data_zone_code"),
            ("msoa_code", "intermediate_zone_code"),
        )
    else:
        native = (
            ("area", "data_zone_code"),
            ("msoa_code", "super_data_zone_code"),
            ("ward_code", "district_electoral_area_code"),
        )
    return (
        *shared,
        *aliases,
        *native,
        ("itl1_code", "itl1_code"),
        ("itl2_code", "itl2_code"),
    )


def uk_atomic_assignment_definition(
    supports: Mapping[str, bytes],
    *,
    seed: int,
) -> dict:
    """Declare shared assignment after the complete initial spine/support roster.

    Source names are fixed by system, not file paths. The host must bind admitted
    bytes to those SourceRefs and authenticate the supplied full-spine identity.
    This function neither creates a graph barrier nor admits a native artifact.
    """
    _require(
        type(supports) is dict and set(supports) == set(SYSTEMS),
        "three systems required",
    )
    _require(
        isinstance(seed, Integral)
        and not isinstance(seed, (bool, np.bool_))
        and 0 <= seed < 2**32,
        "seed must be a bounded integer",
    )
    systems = []
    for system in SYSTEMS:
        _require(type(supports[system]) is bytes, "immutable support bytes required")
        support = decode_atomic_support(supports[system])
        level, vintage, regions, _ = _profile(system)
        _require(
            all(
                support.metadata[k] == value
                for k, value in (
                    ("system", system),
                    ("level", level),
                    ("code_system", "uk_gss"),
                    ("vintage", vintage),
                )
            ),
            "support system identity differs",
        )
        expected_columns = {
            "area",
            *(_INPUT_COLUMNS[1:]),
            "frs_region",
            "itl1_code",
            "itl2_code",
        }
        _require(set(support.arrays) == expected_columns, "support columns differ")
        # The declaration accepts this adapter's normalized form. This closes
        # forged derived aliases/region normalization without claiming source
        # authority from a self-consistent payload.
        restored_arrays = {
            column: support.arrays["area" if column == "oa_code" else column]
            for column in _INPUT_COLUMNS
        }
        restored_metadata = {
            column: dict(
                support.metadata["columns"]["area" if column == "oa_code" else column]
            )
            for column in _INPUT_COLUMNS
        }
        _require(
            assemble_uk_atomic_area_support(
                system=system, arrays=restored_arrays, column_metadata=restored_metadata
            )
            == supports[system],
            "support is not the canonical UK adapter output",
        )
        systems.append(
            {
                "id": system,
                "level": level,
                "code_system": "uk_gss",
                "vintage": vintage,
                "source": SOURCES[system],
                "selector": {"region": list(regions)},
                "constraints": [
                    {"input": "region", "support": "frs_region", "required": True}
                ],
                "observed_area": None,
                "stages": [
                    {"level": "constituency_code", "weight": "households"},
                    {"level": "area", "weight": "population"},
                ],
                "layers": [
                    {
                        "input": column,
                        "output": output,
                        **{
                            k: support.metadata["columns"][column][k]
                            for k in ("source", "vintage", "relation")
                        },
                    }
                    for column, output in _layers(system)
                ],
            }
        )
    return validate_assignment_spec(
        {
            "version": 1,
            "identity": [IDENTITY_COLUMN],
            "stream": ["sha256-u53-v1", "uk-post-clone-atomic-area-v1", 0, int(seed)],
            "outputs": {
                "area": "atomic_area_code",
                "system": "atomic_area_system",
                "basis": "atomic_area_basis",
            },
            "systems": systems,
        }
    )
