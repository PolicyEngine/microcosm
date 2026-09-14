"""Country-neutral atomic geography support, assignment and functional lookups.

Publisher adapters supply one row per atomic area. Country declarations choose
constraints, sampling stages and output names. Observed geography is never
overwritten. This module does not acquire sources or certify their authority.
"""

from __future__ import annotations

import bisect
import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from io import BytesIO
from types import MappingProxyType
from zipfile import ZIP_DEFLATED, ZipFile, ZipInfo

import numpy as np
import pandas as pd

from microcosm.graph.canonical import canonical_json
from microcosm.graph.randomness import keyed_uniform

RELATIONS = frozenset({"exact", "best_fit", "official_tabulation", "inferred_modal"})
_U53 = 2**53


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError("Atomic geography: " + message)


def _text(value: object) -> bool:
    return isinstance(value, str) and bool(value) and value.strip() == value


def _unique_json(pairs):
    result = {}
    for key, value in pairs:
        _require(key not in result, "duplicate metadata key")
        result[key] = value
    return result


@dataclass(frozen=True)
class AtomicSupport:
    """Validated, immutable arrays decoded from a pinned support artifact."""

    metadata: Mapping
    arrays: Mapping[str, np.ndarray]
    sha256: str


def encode_atomic_support(metadata: Mapping, arrays: Mapping[str, np.ndarray]) -> bytes:
    """Create deterministic NPZ bytes; validate through the same decoder as readers."""
    _require("metadata_json" not in arrays, "reserved metadata array")
    output = BytesIO()
    members = {**arrays, "metadata_json": np.asarray(canonical_json(metadata).decode())}
    with ZipFile(output, "w", compression=ZIP_DEFLATED) as archive:
        for name, array in sorted(members.items()):
            _require(_text(name) and name.isidentifier(), "invalid array name")
            member = BytesIO()
            np.lib.format.write_array(member, np.asarray(array), allow_pickle=False)
            info = ZipInfo(name + ".npy", date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = ZIP_DEFLATED
            archive.writestr(info, member.getvalue())
    payload = output.getvalue()
    decode_atomic_support(payload)
    return payload


def decode_atomic_support(payload: bytes) -> AtomicSupport:
    """Refuse ambiguous mappings, object arrays, untyped columns and invalid weights."""
    _require(type(payload) is bytes, "support requires immutable bytes")
    with np.load(BytesIO(payload), allow_pickle=False) as archive:
        _require(len(set(archive.files)) == len(archive.files), "duplicate array")
        meta_array = archive["metadata_json"]
        _require(
            meta_array.shape == () and meta_array.dtype.kind == "U", "metadata encoding"
        )
        metadata = json.loads(str(meta_array.item()), object_pairs_hook=_unique_json)
        _require(
            isinstance(metadata, dict)
            and set(metadata)
            == {"version", "system", "level", "code_system", "vintage", "columns"}
            and type(metadata["version"]) is int
            and metadata["version"] == 1,
            "metadata schema",
        )
        _require(
            all(
                _text(metadata[k])
                for k in ("system", "level", "code_system", "vintage")
            ),
            "area identity",
        )
        columns = metadata["columns"]
        _require(
            isinstance(columns, dict) and "area" in columns and bool(columns),
            "column metadata",
        )
        _require(set(archive.files) == {"metadata_json", *columns}, "undeclared array")
        arrays = {}
        for name, description in columns.items():
            _require(
                _text(name) and name.isidentifier() and name != "metadata_json",
                "column name",
            )
            _require(isinstance(description, dict), "column description")
            array = archive[name]
            _require(
                array.ndim == 1 and len(array) > 0, "nonempty one-dimensional support"
            )
            kind = description.get("kind")
            if kind == "code":
                _require(
                    set(description) == {"kind", "source", "vintage", "relation"},
                    "code metadata",
                )
                _require(
                    array.dtype.kind == "U" and np.all(np.char.str_len(array) > 0),
                    "nonempty string codes",
                )
                _require(np.all(np.char.strip(array) == array), "whitespace in codes")
                _require(
                    _text(description["vintage"])
                    and description["relation"] in RELATIONS,
                    "mapping relation or vintage",
                )
            else:
                _require(
                    kind == "weight"
                    and set(description) == {"kind", "source", "basis"},
                    "weight metadata",
                )
                _require(
                    array.dtype.kind in "iu" and np.all(array >= 0),
                    "nonnegative integer weights",
                )
                _require(_text(description["basis"]), "sampling basis")
                # Python accumulation cannot wrap before the bound is checked.
                total = sum(int(value) for value in array)
                _require(0 < total <= _U53, "weight total outside exact sampling range")
                array = array.astype(np.int64)
            _require(_text(description["source"]), "source identity")
            # Bytes-backed arrays cannot be made writable by changing flags.
            arrays[name] = np.frombuffer(array.tobytes(), dtype=array.dtype)
        area = arrays["area"]
        _require(columns["area"]["kind"] == "code", "atomic area must be a code")
        _require(
            columns["area"]["vintage"] == metadata["vintage"], "atomic vintage mismatch"
        )
        _require(
            columns["area"]["relation"] == "exact", "atomic identity must be exact"
        )
        _require(
            all(len(a) == len(area) for a in arrays.values()), "array lengths differ"
        )
        _require(len(np.unique(area)) == len(area), "atomic codes must be unique")
        _require(
            any(c["kind"] == "weight" for c in columns.values()),
            "missing sampling weights",
        )
    frozen = {
        **metadata,
        "columns": MappingProxyType(
            {k: MappingProxyType(v) for k, v in columns.items()}
        ),
    }
    return AtomicSupport(
        MappingProxyType(frozen),
        MappingProxyType(arrays),
        hashlib.sha256(payload).hexdigest(),
    )


def validate_assignment_spec(spec: Mapping) -> dict:
    """Normalize a graph parameter mapping without accepting undeclared options.

    All geography code columns use strings, preserving leading zeroes. A later
    engine adapter may declare a separate storage conversion when needed.
    """
    value = json.loads(canonical_json(spec))
    _require(
        set(value) == {"version", "identity", "stream", "outputs", "systems"}
        and value["version"] == 1
        and type(value["version"]) is int,
        "assignment schema",
    )
    identity, outputs, systems = value["identity"], value["outputs"], value["systems"]
    _require(
        isinstance(identity, list)
        and bool(identity)
        and all(_text(c) for c in identity)
        and len(set(identity)) == len(identity),
        "stable identity columns",
    )
    _require(
        isinstance(outputs, dict)
        and set(outputs) == {"area", "system", "basis"}
        and all(_text(c) for c in outputs.values())
        and len(set(outputs.values())) == 3,
        "assignment outputs",
    )
    keyed_uniform(stream=tuple(value["stream"]), keys=[])
    _require(isinstance(systems, list) and bool(systems), "area systems")
    ids, layers = set(), {}
    for system in systems:
        _require(
            isinstance(system, dict)
            and set(system)
            == {
                "id",
                "level",
                "code_system",
                "vintage",
                "source",
                "selector",
                "constraints",
                "observed_area",
                "stages",
                "layers",
            },
            "system schema",
        )
        _require(
            all(
                _text(system[k])
                for k in ("id", "level", "code_system", "vintage", "source")
            ),
            "system identity",
        )
        _require(system["id"] not in ids, "duplicate system")
        ids.add(system["id"])
        selector = system["selector"]
        _require(
            isinstance(selector, dict)
            and all(
                _text(c)
                and isinstance(v, list)
                and bool(v)
                and all(_text(x) for x in v)
                and len(set(v)) == len(v)
                for c, v in selector.items()
            ),
            "system selector",
        )
        _require(
            system["observed_area"] is None or _text(system["observed_area"]),
            "observed atomic column",
        )
        constraints = system["constraints"]
        _require(isinstance(constraints, list), "constraints")
        seen = set()
        for c in constraints:
            _require(
                isinstance(c, dict)
                and set(c) == {"input", "support", "required"}
                and _text(c["input"])
                and _text(c["support"])
                and type(c["required"]) is bool,
                "constraint schema",
            )
            _require(
                c["support"] not in seen and c["support"] != "area",
                "duplicate or atomic constraint",
            )
            seen.add(c["support"])
        stages = system["stages"]
        _require(isinstance(stages, list) and bool(stages), "sampling stages")
        _require(
            all(
                isinstance(s, dict)
                and set(s) == {"level", "weight"}
                and _text(s["level"])
                and _text(s["weight"])
                for s in stages
            ),
            "stage schema",
        )
        _require(
            stages[-1]["level"] == "area"
            and len({s["level"] for s in stages}) == len(stages),
            "stages must end at the unique atomic level",
        )
        _require(isinstance(system["layers"], list), "layers")
        local_layers = set()
        for layer in system["layers"]:
            _require(
                isinstance(layer, dict)
                and set(layer) == {"input", "output", "vintage", "relation", "source"}
                and all(_text(x) for x in layer.values())
                and layer["relation"] in RELATIONS,
                "layer schema",
            )
            name = layer["output"]
            _require(
                name not in outputs.values() and name not in local_layers,
                "duplicate layer output",
            )
            local_layers.add(name)
            layers.setdefault(name, []).append(layer)
    required = set(identity)
    for system in systems:
        required.update(system["selector"])
        required.update(c["input"] for c in system["constraints"])
        if system["observed_area"]:
            required.add(system["observed_area"])
    _require(
        not required.intersection({*outputs.values(), *layers}),
        "observed or identity columns would be overwritten",
    )
    return value


def _bind(spec: Mapping, supports: Mapping[str, AtomicSupport]):
    spec = validate_assignment_spec(spec)
    _require(
        set(supports) == {s["id"] for s in spec["systems"]}, "support systems differ"
    )
    for system in spec["systems"]:
        support = supports[system["id"]]
        _require(
            all(
                support.metadata[k] == system[v]
                for k, v in (
                    ("system", "id"),
                    ("level", "level"),
                    ("code_system", "code_system"),
                    ("vintage", "vintage"),
                )
            ),
            "support identity differs from declaration",
        )
        columns = support.metadata["columns"]
        for c in system["constraints"]:
            _require(
                c["support"] in columns and columns[c["support"]]["kind"] == "code",
                "unknown constraint",
            )
        for s in system["stages"]:
            _require(
                s["level"] in columns
                and columns[s["level"]]["kind"] == "code"
                and s["weight"] in columns
                and columns[s["weight"]]["kind"] == "weight",
                "unknown stage column",
            )
        for layer in system["layers"]:
            description = columns.get(layer["input"], {})
            _require(
                description.get("kind") == "code"
                and all(
                    description[k] == layer[k]
                    for k in ("source", "vintage", "relation")
                ),
                "layer source, relation or vintage differs",
            )
    return spec


def _route(households: pd.DataFrame, spec: Mapping) -> dict[str, np.ndarray]:
    masks, claimed = {}, np.zeros(len(households), dtype=np.int64)
    for system in spec["systems"]:
        mask = np.ones(len(households), dtype=bool)
        for name, accepted in system["selector"].items():
            mask &= households[name].isin(accepted).to_numpy()
        masks[system["id"]] = np.flatnonzero(mask)
        claimed += mask
    _require(np.all(claimed == 1), "every household must match exactly one area system")
    return masks


def _constraints(row, system) -> tuple:
    values = []
    for c in system["constraints"]:
        value = row[c["input"]]
        if pd.isna(value):
            _require(not c["required"], "missing required observed geography")
        else:
            _require(_text(value), "observed codes require nonempty strings")
            values.append((c["support"], value))
    return tuple(sorted(values))


class _SupportIndex:
    """Index each observed-column pattern once, rather than scanning per household."""

    def __init__(self, support):
        self.support = support
        self.groupings = {}
        self.area_index = pd.Index(support.arrays["area"])
        self.draw_cells = {}

    def rows(self, constraints):
        columns = tuple(k for k, _ in constraints)
        if columns not in self.groupings:
            if columns:
                table = pd.DataFrame({c: self.support.arrays[c] for c in columns})
                # Always group by a list: normalize pandas' single-column key below.
                groups = table.groupby(list(columns), sort=False, observed=True).indices
                self.groupings[columns] = {
                    (k,) if len(columns) == 1 and not isinstance(k, tuple) else k: v
                    for k, v in groups.items()
                }
            else:
                self.groupings[columns] = {(): np.arange(len(self.area_index))}
        key = tuple(v for _, v in constraints)
        _require(
            key in self.groupings[columns],
            "observed constraints have no common support",
        )
        return self.groupings[columns][key]

    def pick(self, constraints, stage, u):
        key = (constraints, stage["level"], stage["weight"])
        if key not in self.draw_cells:
            rows = self.rows(constraints)
            levels, inverse = np.unique(
                self.support.arrays[stage["level"]][rows], return_inverse=True
            )
            weights = np.zeros(len(levels), dtype=np.int64)
            np.add.at(weights, inverse, self.support.arrays[stage["weight"]][rows])
            cumulative = np.cumsum(weights).tolist()
            _require(cumulative[-1] > 0, "constraint cell has zero sampling mass")
            self.draw_cells[key] = levels, cumulative
        levels, cumulative = self.draw_cells[key]
        # Integer inverse CDF: no floating-point rounding across a cell boundary.
        threshold = (int(u * _U53) * cumulative[-1]) // _U53
        return str(levels[bisect.bisect_right(cumulative, threshold)])


def assign_atomic(
    households: pd.DataFrame, spec: Mapping, supports: Mapping[str, AtomicSupport]
) -> pd.DataFrame:
    """Select atomic areas without changing row order, ids or observed geography."""
    spec = _bind(spec, supports)
    _require(
        not set(spec["outputs"].values()).intersection(households.columns),
        "assignment output already exists",
    )
    _require(
        not households[spec["identity"]].isna().any().any()
        and not households.duplicated(spec["identity"]).any(),
        "null or duplicate draw identity",
    )
    routing = _route(households, spec)
    result = pd.DataFrame(index=households.index)
    for name in spec["outputs"].values():
        result[name] = pd.array([pd.NA] * len(households), dtype="string")
    definition = hashlib.sha256(canonical_json(spec)).hexdigest()
    for system in spec["systems"]:
        positions = routing[system["id"]]
        support = supports[system["id"]]
        index = _SupportIndex(support)
        identities = list(
            households.iloc[positions][spec["identity"]].itertuples(
                index=False, name=None
            )
        )
        draws = [
            keyed_uniform(
                stream=tuple(spec["stream"]),
                keys=[
                    (
                        *identity,
                        system["id"],
                        stage["level"],
                        definition,
                        support.sha256,
                    )
                    for identity in identities
                ],
            )
            for stage in system["stages"]
        ]
        for local, position in enumerate(positions):
            row = households.iloc[position]
            constraints = _constraints(row, system)
            area = row[system["observed_area"]] if system["observed_area"] else None
            observed = area is not None and not pd.isna(area)
            if observed:
                _require(_text(area), "invalid observed atomic code")
                match = index.area_index.get_indexer([area])[0]
                _require(
                    match >= 0
                    and all(
                        support.arrays[c][match] == value for c, value in constraints
                    ),
                    "observed atomic code is unsupported or conflicts",
                )
            else:
                for stage, uniforms in zip(system["stages"], draws, strict=True):
                    area = index.pick(constraints, stage, uniforms[local])
                    constraints = tuple(
                        sorted({**dict(constraints), stage["level"]: area}.items())
                    )
            for key, value in (
                ("area", area),
                ("system", system["id"]),
                ("basis", "observed" if observed else "assigned"),
            ):
                result.iloc[position, result.columns.get_loc(spec["outputs"][key])] = (
                    value
                )
    return result


def derive_geography(
    households: pd.DataFrame, spec: Mapping, supports: Mapping[str, AtomicSupport]
) -> pd.DataFrame:
    """Functional area lookups, including explicitly identified non-nesting conventions."""
    spec = _bind(spec, supports)
    routing = _route(households, spec)
    result = pd.DataFrame(index=households.index)
    for name in sorted(
        {layer["output"] for s in spec["systems"] for layer in s["layers"]}
    ):
        result[name] = pd.array([pd.NA] * len(households), dtype="string")
    for system in spec["systems"]:
        positions = routing[system["id"]]
        rows = households.iloc[positions]
        support = supports[system["id"]]
        assigned_system = rows[spec["outputs"]["system"]]
        _require(
            assigned_system.notna().all() and assigned_system.eq(system["id"]).all(),
            "assigned system disagrees with routing",
        )
        locations = pd.Index(support.arrays["area"]).get_indexer(
            rows[spec["outputs"]["area"]]
        )
        _require(np.all(locations >= 0), "missing atomic mapping")
        for layer in system["layers"]:
            result.iloc[positions, result.columns.get_loc(layer["output"])] = (
                support.arrays[layer["input"]][locations]
            )
    return result


def validate_geography(
    households: pd.DataFrame, spec: Mapping, supports: Mapping[str, AtomicSupport]
) -> dict:
    """Recheck constraints and functional mappings on a full or pruned population.

    Draws are not repeated. Initial support clones receive their own assignment
    after expansion; subsequent views retain it. Structural lineage verification
    remains the executor's responsibility. This gate is not a population-quality
    certificate.
    """
    spec = _bind(spec, supports)
    expected = derive_geography(households, spec, supports)
    for name in expected:
        _require(
            households[name].astype("string").equals(expected[name]),
            "derived geography differs from atomic mapping",
        )
    routing = _route(households, spec)
    for system in spec["systems"]:
        support = supports[system["id"]]
        rows = households.iloc[routing[system["id"]]]
        positions = pd.Index(support.arrays["area"]).get_indexer(
            rows[spec["outputs"]["area"]]
        )
        for (_, row), position in zip(rows.iterrows(), positions, strict=True):
            _require(
                all(
                    support.arrays[c][position] == v
                    for c, v in _constraints(row, system)
                ),
                "assigned area violates observed geography",
            )
            original = row[system["observed_area"]] if system["observed_area"] else None
            observed = original is not None and not pd.isna(original)
            _require(
                row[spec["outputs"]["basis"]]
                == ("observed" if observed else "assigned"),
                "assignment basis differs",
            )
            if observed:
                _require(
                    row[spec["outputs"]["area"]] == original,
                    "observed atomic area changed",
                )
    return {
        "outcome": "pass",
        "scope": "atomic_geography_mapping_integrity",
        "households": len(households),
        "support_sha256": {k: v.sha256 for k, v in sorted(supports.items())},
        "definition_sha256": hashlib.sha256(canonical_json(spec)).hexdigest(),
    }
