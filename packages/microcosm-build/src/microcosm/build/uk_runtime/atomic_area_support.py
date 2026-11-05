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
from microcosm.build.atomic_support_conditioning import extend_atomic_support
from microcosm.build.uk_runtime.rowwise_geography import FRS_REGION_TO_REGION_CODE
from microcosm.graph.canonical import canonical_json

SYSTEMS = (
    "uk_ew_output_area_2021",
    "uk_scotland_output_area_2022",
    "uk_ni_data_zone_2021",
)
SOURCES = {system: system + "_support" for system in SYSTEMS}
IDENTITY_COLUMN = "geography_household_key"
#: Assignment outputs the shared operators write on the household table.
ASSIGNMENT_COLUMNS = ("atomic_area_code", "atomic_area_system", "atomic_area_basis")
#: Nation-native aliases of derived layers; NA outside their own nation and
#: dropped at the single-year export boundary.
UK_NATIVE_ALIAS_COLUMNS = (
    "output_area_code",
    "data_zone_code",
    "intermediate_zone_code",
    "super_data_zone_code",
    "district_electoral_area_code",
)
#: The committed provenance of the three support artifacts: per-column source
#: and vintage of every derived layer, with the artifacts' digests.
PROVENANCE_RESOURCE = "uk_atomic_area_supports.provenance.json"
#: The conditioned supports (microcosm#931): each system's support plus the
#: area's IMD decile and, where council tax is banded, its households by band.
#: Only the geography-first move binds them; today's build binds SOURCES.
CONDITIONED_SOURCES = {system: system + "_conditioned_support" for system in SYSTEMS}
CONDITIONED_PROVENANCE_RESOURCE = "uk_atomic_area_conditioned_supports.provenance.json"
#: The FRS records the deprivation tenth of each address within its own nation.
IMD_DECILE_COLUMN = "imd_decile"
IMD_DECILES = tuple(str(decile) for decile in range(1, 11))
#: Households by council-tax band, times BAND_WEIGHT_SCALE: bands A to I in
#: England and Wales (only Wales has band I), A to H in Scotland. Northern
#: Ireland has domestic rates, so its support carries no band weights.
BAND_WEIGHT_SCALE = 10**6
BAND_WEIGHT_COLUMNS = {
    SYSTEMS[0]: tuple(f"households_band_{band}" for band in "abcdefghi"),
    SYSTEMS[1]: tuple(f"households_band_{band}" for band in "abcdefgh"),
    SYSTEMS[2]: (),
}


def uk_area_code_frames() -> dict[str, dict[str, str]]:
    """The code frames behind the exported area codes, per support system.

    microcosm#1114: consumers check their own area lists against the frames a
    release carries. The frames are the per-column vintages the three support
    artifacts declare (``2024_pcon`` for Westminster constituencies of July
    2024; ``2023_april_lad``, ``2019_council_area`` and ``2014_lgd`` for the
    April 2023 local-authority code set; ``2024_rgn`` for the English regions
    with the FRS sentinel codes for the other nations), read from the
    committed provenance rather than restated here. Keyed by the exported
    column name, then by support system.
    """

    import json
    from importlib.resources import files

    from .geography_ladder import UK_EXPORT_AREA_CODE_COLUMNS

    payload = json.loads(
        files("microcosm.build.uk").joinpath(PROVENANCE_RESOURCE).read_text()
    )
    supports = payload["supports"]
    _require(set(supports) == set(SYSTEMS), "provenance covers the three systems")
    frames: dict[str, dict[str, str]] = {}
    for ladder_column, export_column in UK_EXPORT_AREA_CODE_COLUMNS.items():
        frames[export_column] = {}
        for system in SYSTEMS:
            vintage = supports[system]["column_metadata"][ladder_column]["vintage"]
            _require(_text(vintage), f"{system} {ladder_column} vintage is text")
            frames[export_column][system] = str(vintage)
    return frames


def without_uk_native_alias_columns(household):
    """The household table without the nation-native alias columns.

    The aliases are NA outside their own nation by design, so no single-year
    dataset may carry them: not the release export and not the scratch
    dataset a measure engine loads (policyengine-uk refuses any NaN column).
    Returns the same object when the table carries none of them.
    """

    aliases = [c for c in UK_NATIVE_ALIAS_COLUMNS if c in household.columns]
    if not aliases:
        return household
    return household.drop(columns=aliases)


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
#: Every column of an unconditioned support, as the adapter normalizes it.
_SUPPORT_COLUMNS = frozenset(
    {"area", *_INPUT_COLUMNS[1:], "frs_region", "itl1_code", "itl2_code"}
)
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


def _canonical_support(system: str, support) -> bytes:
    """Re-assemble a decoded support's adapter columns into canonical bytes."""
    arrays = {
        column: support.arrays["area" if column == "oa_code" else column]
        for column in _INPUT_COLUMNS
    }
    metadata = {
        column: dict(
            support.metadata["columns"]["area" if column == "oa_code" else column]
        )
        for column in _INPUT_COLUMNS
    }
    return assemble_uk_atomic_area_support(
        system=system, arrays=arrays, column_metadata=metadata
    )


def assemble_uk_conditioned_area_support(
    *,
    base: bytes,
    imd_decile: np.ndarray,
    band_weights: Mapping[str, np.ndarray],
    column_metadata: Mapping,
) -> bytes:
    """Add each area's IMD decile and households by band to a canonical support.

    ``base`` must be this adapter's canonical output for its system, the
    support today's build binds. ``imd_decile`` gives each area, in support
    order, its decile code within the nation. ``band_weights`` gives each of
    the system's band columns (:data:`BAND_WEIGHT_COLUMNS`) the area's
    households in that band times :data:`BAND_WEIGHT_SCALE`; an area's band
    weights must sum to its households times the scale, up to half a unit
    per band of rounding. English areas carry no band I. The caller states
    every new column's source, vintage or basis, as for any support column.
    """
    _require(type(base) is bytes, "immutable support bytes required")
    support = decode_atomic_support(base)
    system = support.metadata["system"]
    _require(system in SYSTEMS, "unknown area system")
    _require(set(support.arrays) == _SUPPORT_COLUMNS, "base is already conditioned")
    _require(
        _canonical_support(system, support) == base,
        "base is not the canonical UK adapter output",
    )
    bands = BAND_WEIGHT_COLUMNS[system]
    _require(
        type(column_metadata) is dict
        and set(column_metadata) == {IMD_DECILE_COLUMN, *bands},
        "exact conditioning metadata required",
    )
    for column, item in column_metadata.items():
        code = column == IMD_DECILE_COLUMN
        _require(
            type(item) is dict
            and set(item)
            == (
                {"kind", "source", "vintage", "relation"}
                if code
                else {"kind", "source", "basis"}
            )
            and all(_text(value) for value in item.values())
            and item["kind"] == ("code" if code else "weight")
            and (not code or item["relation"] in RELATIONS),
            "conditioning column description",
        )
    areas = len(support.arrays["area"])
    _require(
        type(imd_decile) is np.ndarray
        and imd_decile.dtype.kind == "U"
        and imd_decile.shape == (areas,)
        and set(imd_decile.tolist()) <= set(IMD_DECILES),
        "one decile code, 1 to 10, per area",
    )
    _require(
        type(band_weights) is dict and set(band_weights) == set(bands),
        "exact band weights required",
    )
    weights = {}
    for band in bands:
        array = band_weights[band]
        _require(
            type(array) is np.ndarray
            and array.dtype.kind in "iu"
            and array.shape == (areas,)
            and bool(np.all(array >= 0)),
            "nonnegative integer band weights, one per area",
        )
        weights[band] = array.astype(np.int64)
    if bands:
        households = [int(value) for value in support.arrays["households"]]
        for row, count in enumerate(households):
            split = sum(int(weights[band][row]) for band in bands)
            _require(
                2 * abs(split - count * BAND_WEIGHT_SCALE) <= len(bands),
                "band weights must split the area's households",
            )
    if system == SYSTEMS[0]:
        english = np.char.startswith(np.asarray(support.arrays["area"]), "E")
        _require(
            not np.any(weights["households_band_i"][english]),
            "band I exists only in Wales",
        )
    return extend_atomic_support(
        base,
        arrays={IMD_DECILE_COLUMN: imd_decile.copy(), **weights},
        columns={column: dict(item) for column, item in column_metadata.items()},
    )


def uk_conditioned_support_base(payload: bytes) -> bytes:
    """The canonical support a conditioned support extends.

    Refuses a payload that is not exactly what
    :func:`assemble_uk_conditioned_area_support` makes of that base, so a
    caller can hold a conditioned support to the registered support it adds to.
    """
    _require(type(payload) is bytes, "immutable support bytes required")
    support = decode_atomic_support(payload)
    system = support.metadata["system"]
    _require(system in SYSTEMS, "unknown area system")
    extra = (IMD_DECILE_COLUMN, *BAND_WEIGHT_COLUMNS[system])
    _require(
        set(support.arrays) == _SUPPORT_COLUMNS | set(extra),
        "conditioned support columns differ",
    )
    base = _canonical_support(system, support)
    rebuilt = assemble_uk_conditioned_area_support(
        base=base,
        imd_decile=np.asarray(support.arrays[IMD_DECILE_COLUMN]).copy(),
        band_weights={
            band: np.asarray(support.arrays[band]).copy()
            for band in BAND_WEIGHT_COLUMNS[system]
        },
        column_metadata={
            column: dict(support.metadata["columns"][column]) for column in extra
        },
    )
    _require(rebuilt == payload, "support is not the canonical conditioned output")
    return base


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
    The sampling law is single-stage: one draw of the atomic area by census
    household count within the household's FRS region.
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
        _require(set(support.arrays) == _SUPPORT_COLUMNS, "support columns differ")
        # The declaration accepts this adapter's normalized form. This closes
        # forged derived aliases/region normalization without claiming source
        # authority from a self-consistent payload.
        _require(
            _canonical_support(system, support) == supports[system],
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
                # Single-stage law (2026-09-15 ruling): draw the atomic area by
                # census household count within the household's FRS region.
                "stages": [{"level": "area", "weight": "households"}],
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
