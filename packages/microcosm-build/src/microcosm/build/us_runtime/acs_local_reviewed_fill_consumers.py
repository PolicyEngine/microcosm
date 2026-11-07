"""Means-tested consumers of the ACS local lane's reviewed default fills
(microcosm#1022, part 6).

The release tool's engine pass (``fill_reviewed_nulls`` in
``tools/build_us_acs_local_release.py``) writes the policyengine-us default
into every missing cell of an engine input the staging register lists. Being
registered only means the fill is allowed. The packaged register
``acs_local_reviewed_fill_consumers.yaml`` adds what the fill does: for each
column the lane still default-fills, the means-tested programs that read it in
the pinned engine and, for each program, a note saying the default is harmless
for it (and why) or biases it in a known direction.

- :func:`reviewed_fill_register_failures` validates a register document without
  the engine: known entities and programs, one entry per column, every
  declared consumer covered by exactly one note, a direction on every
  ``known_bias`` note and none on a ``harmless`` one, no entry for a
  column the release tool never default-fills (``NEVER_DEFAULT_FILLED``), and
  the program couplings of :data:`REVIEWED_FILL_COUPLINGS`: a program that
  reads another program's result cannot be harmless where that program is a
  known bias (microcosm#1071 review).
- :func:`acs_local_reviewed_fill_consumer_gate` is the release gate. It reads
  the fill manifests the engine passes wrote and fails when a fill the release
  applied has no entry, when an entry's fill value or spines differ from the
  manifest, when the register itself fails validation, or when the
  policyengine-us the release runs is not the version the register was
  reviewed against (``reviewed_against``). An entry whose
  column the release did not fill is reported, not failed: which cells are
  missing depends on the data (a capped smoke may hold no group-quarters
  household), and an unused entry cannot hide an unreviewed fill.
- :func:`means_tested_consumers` is the dependency walk the register's
  ``consumers`` lists come from. It takes any index with
  ``consumer_receipts`` (the engine-tier test passes
  ``PolicyEngineUSVariableMetadataIndex``) and follows consumers from the
  column until nothing new is reached, recording the programs whose roots it
  meets. A column that is itself a root reaches its own program.

The module reads no origin tag: the manifests carry per-spine counts, and the
register's ``spines`` lists are compared with them as plain labels.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections import deque
from collections.abc import Collection, Iterable, Mapping
from dataclasses import dataclass
from importlib.resources import files
from typing import Protocol

from microcosm.build.gates import GateResult

__all__ = [
    "ACS_LOCAL_REVIEWED_FILL_CONSUMERS_ISSUE",
    "ACS_LOCAL_REVIEWED_FILL_CONSUMER_GATE_NAME",
    "ACS_LOCAL_REVIEWED_FILL_CONSUMER_REGISTER",
    "REVIEWED_FILL_BIAS_DIRECTIONS",
    "REVIEWED_FILL_COUPLINGS",
    "REVIEWED_FILL_VERDICTS",
    "ConsumerIndex",
    "acs_local_reviewed_fill_consumer_gate",
    "load_reviewed_fill_consumer_register",
    "means_tested_consumers",
    "reviewed_fill_register_failures",
    "reviewed_fill_register_sha256",
    "reviewed_fill_register_summary",
]

ACS_LOCAL_REVIEWED_FILL_CONSUMERS_ISSUE = "microcosm#1022"
ACS_LOCAL_REVIEWED_FILL_CONSUMER_GATE_NAME = "acs_local_reviewed_fill_consumers"
ACS_LOCAL_REVIEWED_FILL_CONSUMER_REGISTER = "acs_local_reviewed_fill_consumers.yaml"

#: ``harmless``: the default does not change the program's result on the
#: filled rows, or changes it only negligibly, and the note says why.
#: ``known_bias``: the default moves the program's result for an identifiable
#: group, in the note's ``direction``.
REVIEWED_FILL_VERDICTS = ("harmless", "known_bias")
#: Relative to the true value: more eligibility or benefit, less, or both.
REVIEWED_FILL_BIAS_DIRECTIONS = ("overstates", "understates", "mixed")
#: Program couplings every entry must honour (microcosm#1071 review), as
#: ``(upstream, downstream, bars)``. The downstream program reads the upstream
#: program's result, so where an entry marks the upstream ``known_bias`` and
#: declares the downstream a consumer, the downstream note must be
#: ``known_bias`` too. When ``bars`` is set, upstream eligibility bars the
#: downstream program, so the downstream direction must be the opposite of
#: the upstream one, or ``mixed``.
#:
#: - Medicaid eligibility bars the premium tax credit (``pays_aca_premium``):
#:   overstated Medicaid eligibility understates the PTC.
#: - CHIP eligibility bars it too. Its direction is not constrained: a CHIP
#:   bias that comes from Medicaid (which bars CHIP) leaves the PTC barred by
#:   Medicaid instead of freeing it.
#: - SSI and TANF cash are SNAP unearned income, and TANF receipt confers SNAP
#:   categorical eligibility, so either can move SNAP both ways.
#:
#: The engine-tier test checks each coupling is an edge of the engine's
#: dependency graph and that every column reaching it declares both programs.
REVIEWED_FILL_COUPLINGS: tuple[tuple[str, str, bool], ...] = (
    ("medicaid", "aca_ptc", True),
    ("chip", "aca_ptc", False),
    ("ssi", "snap", False),
    ("tanf", "snap", False),
)
_OPPOSITE_DIRECTIONS = {
    "overstates": ("understates", "mixed"),
    "understates": ("overstates", "mixed"),
    "mixed": ("mixed",),
}

_SCHEMA_VERSION = 1
_ENTITIES = ("person", "household", "tax_unit", "spm_unit", "family", "marital_unit")
_TOP_LEVEL_KEYS = frozenset(
    {"schema_version", "issue", "reviewed_against", "programs", "entries"}
)
_PROGRAM_KEYS = frozenset({"label", "roots"})
_ENTRY_KEYS = frozenset(
    {
        "entity",
        "column",
        "fill_value",
        "spines",
        "rows",
        "summary",
        "consumers",
        "notes",
    }
)
_NOTE_REQUIRED_KEYS = frozenset({"programs", "verdict", "note"})
_NOTE_KEYS = _NOTE_REQUIRED_KEYS | {"direction", "via"}
_NAME = re.compile(r"[a-z][a-z0-9_]*")


class ConsumerIndex(Protocol):
    """What the dependency walk needs from an engine index."""

    def consumer_receipts(self, name: str) -> Iterable[object]: ...


def load_reviewed_fill_consumer_register(path=None) -> object:
    """The register document, parsed with the spec engine's strict YAML 1.2
    subset (duplicate keys and tags are refused). ``path`` defaults to the
    packaged register."""

    from microcosm.build.spec_engine.yaml12 import load_yaml12

    if path is None:
        resource = files("microcosm.build.us_runtime").joinpath(
            ACS_LOCAL_REVIEWED_FILL_CONSUMER_REGISTER
        )
        return load_yaml12(
            resource.read_text(encoding="utf-8"),
            source=ACS_LOCAL_REVIEWED_FILL_CONSUMER_REGISTER,
        )
    with open(path, encoding="utf-8") as handle:
        return load_yaml12(handle.read(), source=str(path))


def reviewed_fill_register_sha256(document: object) -> str:
    """SHA-256 of the register as canonical JSON (sorted keys, no whitespace),
    so a receipt binds the notes the gate consulted, not their formatting."""

    payload = json.dumps(
        document, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _is_name_list(value: object) -> bool:
    return (
        isinstance(value, list)
        and bool(value)
        and all(isinstance(item, str) and _NAME.fullmatch(item) for item in value)
        and len(set(value)) == len(value)
    )


def _is_text(value: object) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _program_failures(programs: object) -> list[str]:
    if not isinstance(programs, Mapping) or not programs:
        return ["programs must be a non-empty mapping of program -> label and roots."]
    failures: list[str] = []
    owner: dict[str, str] = {}
    for program, spec in programs.items():
        if not isinstance(program, str) or not _NAME.fullmatch(program):
            failures.append(f"program key {program!r} is not a lowercase name.")
            continue
        if not isinstance(spec, Mapping) or set(spec) != _PROGRAM_KEYS:
            failures.append(f"program {program!r} must carry exactly label and roots.")
            continue
        if not _is_text(spec["label"]):
            failures.append(f"program {program!r} has no label.")
        if not _is_name_list(spec["roots"]):
            failures.append(
                f"program {program!r} roots must be a non-empty list of distinct "
                "engine variable names."
            )
            continue
        for root in spec["roots"]:
            if root in owner:
                failures.append(
                    f"root {root!r} belongs to both {owner[root]!r} and {program!r}."
                )
            owner[root] = program
    return failures


def _note_failures(label: str, notes: object, consumers: list[str], known: set[str]):
    if not isinstance(notes, list):
        return [f"{label}: notes must be a list."]
    failures: list[str] = []
    covered: dict[str, int] = {}
    for index, note in enumerate(notes):
        where = f"{label} note {index}"
        if not isinstance(note, Mapping):
            failures.append(f"{where} must be a mapping.")
            continue
        missing = sorted(_NOTE_REQUIRED_KEYS - set(note))
        extra = sorted(set(note) - _NOTE_KEYS)
        if missing or extra:
            failures.append(
                f"{where} is missing {missing} or carries unknown keys {extra}."
            )
            continue
        programs = note["programs"]
        if not _is_name_list(programs):
            failures.append(f"{where} programs must be a non-empty list of names.")
            continue
        for program in programs:
            if program not in known:
                failures.append(f"{where} names unknown program {program!r}.")
            elif program not in consumers:
                failures.append(
                    f"{where} has a note for {program!r}, which is not a declared "
                    "consumer of the column."
                )
            covered[program] = covered.get(program, 0) + 1
        verdict = note["verdict"]
        if verdict not in REVIEWED_FILL_VERDICTS:
            failures.append(
                f"{where} verdict {verdict!r} is not one of {REVIEWED_FILL_VERDICTS}."
            )
        direction = note.get("direction")
        if verdict == "known_bias" and direction not in REVIEWED_FILL_BIAS_DIRECTIONS:
            failures.append(
                f"{where} is a known_bias note without a direction in "
                f"{REVIEWED_FILL_BIAS_DIRECTIONS}."
            )
        if verdict == "harmless" and "direction" in note:
            failures.append(f"{where} is harmless but carries a direction.")
        if "via" in note:
            via = note["via"]
            if not _is_name_list(via):
                failures.append(f"{where} via must be a non-empty list of names.")
            elif any(item not in consumers or item in programs for item in via):
                failures.append(
                    f"{where} via {via} must name other declared consumers."
                )
        if not _is_text(note["note"]):
            failures.append(f"{where} has no note text.")
    for program in consumers:
        count = covered.get(program, 0)
        if count == 0:
            failures.append(
                f"{label}: declared consumer {program!r} has no harmless or "
                "known_bias note."
            )
        elif count > 1:
            failures.append(f"{label}: consumer {program!r} has {count} notes.")
    return failures


def _coupling_failures(label: str, notes: list, consumers: list[str]) -> list[str]:
    """Where an entry breaks :data:`REVIEWED_FILL_COUPLINGS`.

    Called only on notes :func:`_note_failures` accepted.
    """

    verdicts = {
        program: (note["verdict"], note.get("direction"))
        for note in notes
        for program in note["programs"]
    }
    failures: list[str] = []
    for upstream, downstream, bars in REVIEWED_FILL_COUPLINGS:
        upstream_verdict, upstream_direction = verdicts.get(upstream, (None, None))
        if upstream_verdict != "known_bias" or downstream not in consumers:
            continue
        verdict, direction = verdicts[downstream]
        if verdict != "known_bias":
            failures.append(
                f"{label}: {upstream} is a known bias and {downstream} reads its "
                f"result, so {downstream} cannot be harmless (microcosm#1071 "
                "review)."
            )
        elif bars and direction not in _OPPOSITE_DIRECTIONS[upstream_direction]:
            failures.append(
                f"{label}: {upstream} eligibility bars {downstream}, so "
                f"{downstream} must move against {upstream} ({upstream} "
                f"{upstream_direction}; {downstream} must be one of "
                f"{_OPPOSITE_DIRECTIONS[upstream_direction]}, not {direction!r}) "
                "(microcosm#1071 review)."
            )
    return failures


def reviewed_fill_register_failures(
    document: object,
    *,
    never_default_filled: Collection[tuple[str, str]] = (),
) -> list[str]:
    """Why ``document`` is not a sound register; empty if it is.

    ``never_default_filled`` is the release tool's ``NEVER_DEFAULT_FILLED``:
    the fill of such a column is refused before it happens, so an entry for it
    describes a fill that cannot occur and is stale by construction.
    """

    if not isinstance(document, Mapping):
        return ["the register must be a mapping."]
    failures: list[str] = []
    missing = sorted(_TOP_LEVEL_KEYS - set(document))
    extra = sorted(set(document) - _TOP_LEVEL_KEYS)
    if missing or extra:
        failures.append(f"the register is missing {missing} or carries {extra}.")
        return failures
    if document["schema_version"] != _SCHEMA_VERSION:
        failures.append(f"schema_version must be {_SCHEMA_VERSION}.")
    if document["issue"] != ACS_LOCAL_REVIEWED_FILL_CONSUMERS_ISSUE:
        failures.append(f"issue must be {ACS_LOCAL_REVIEWED_FILL_CONSUMERS_ISSUE!r}.")
    reviewed_against = document["reviewed_against"]
    if not isinstance(reviewed_against, Mapping) or not _is_text(
        reviewed_against.get("policyengine_us")
    ):
        failures.append("reviewed_against.policyengine_us must name the engine.")
    failures += _program_failures(document["programs"])
    known = (
        set(document["programs"])
        if isinstance(document["programs"], Mapping)
        else set()
    )
    entries = document["entries"]
    if not isinstance(entries, list):
        return failures + ["entries must be a list."]
    never = {column for _entity, column in never_default_filled}
    seen: set[str] = set()
    for index, entry in enumerate(entries):
        if not isinstance(entry, Mapping) or set(entry) != _ENTRY_KEYS:
            failures.append(f"entry {index} must carry exactly {sorted(_ENTRY_KEYS)}.")
            continue
        column = entry["column"]
        label = f"{entry['entity']}.{column}"
        if entry["entity"] not in _ENTITIES:
            failures.append(f"{label}: unknown entity {entry['entity']!r}.")
        if not isinstance(column, str) or not _NAME.fullmatch(column):
            failures.append(f"entry {index}: column {column!r} is not a variable name.")
            continue
        if column in seen:
            failures.append(f"{label}: more than one entry for the column.")
        seen.add(column)
        if column in never:
            failures.append(
                f"{label}: the release tool never default-fills this column "
                "(NEVER_DEFAULT_FILLED), so the entry is stale; remove it."
            )
        if not isinstance(entry["fill_value"], str):
            failures.append(f"{label}: fill_value must be the manifest's repr string.")
        if not _is_name_list(entry["spines"]):
            failures.append(f"{label}: spines must be a non-empty list of names.")
        for key in ("rows", "summary"):
            if not _is_text(entry[key]):
                failures.append(f"{label}: {key} must be non-empty text.")
        consumers = entry["consumers"]
        if not isinstance(consumers, list) or any(
            not isinstance(item, str) for item in consumers
        ):
            failures.append(f"{label}: consumers must be a list of program names.")
            continue
        if len(set(consumers)) != len(consumers):
            failures.append(f"{label}: consumers repeat a program.")
        unknown = sorted(set(consumers) - known)
        if unknown:
            failures.append(f"{label}: unknown consumer program(s) {unknown}.")
        note_failures = _note_failures(label, entry["notes"], consumers, known)
        failures += note_failures
        if not note_failures:
            failures += _coupling_failures(label, entry["notes"], consumers)
    return failures


@dataclass(frozen=True)
class _AppliedFill:
    entity: str
    column: str
    fill_value: str
    spines: frozenset[str]
    filled_rows: Mapping[str, int]


def _applied_fills(manifests: Mapping[str, object]):
    """The union of the fills the manifests record, and why any is unusable."""

    failures: list[str] = []
    applied: dict[tuple[str, str], _AppliedFill] = {}
    if not manifests:
        return applied, ["no reviewed-null fill manifest was supplied."]
    for name, manifest in manifests.items():
        fills = manifest.get("fills") if isinstance(manifest, Mapping) else None
        if not isinstance(fills, list):
            failures.append(
                f"{name} records no fills list, so the fills the engine pass "
                "applied cannot be established; re-run the stage that writes it."
            )
            continue
        if manifest.get("unregistered_violations"):
            failures.append(f"{name} records unregistered null violations.")
        for fill in fills:
            spines = (
                fill.get("missing_rows_by_spine") if isinstance(fill, Mapping) else None
            )
            if (
                not isinstance(fill, Mapping)
                or not isinstance(fill.get("entity"), str)
                or not isinstance(fill.get("column"), str)
                or not isinstance(fill.get("fill_value"), str)
                or not isinstance(spines, Mapping)
                or type(fill.get("filled_rows")) is not int
            ):
                failures.append(f"{name} holds a malformed fill record: {fill!r}.")
                continue
            key = (fill["entity"], fill["column"])
            previous = applied.get(key)
            if previous is not None and previous.fill_value != fill["fill_value"]:
                failures.append(
                    f"{key[0]}.{key[1]}: the manifests disagree on the fill value "
                    f"({previous.fill_value} vs {fill['fill_value']})."
                )
            applied[key] = _AppliedFill(
                entity=key[0],
                column=key[1],
                fill_value=fill["fill_value"],
                spines=(previous.spines if previous else frozenset())
                | frozenset(str(spine) for spine in spines),
                filled_rows={
                    **(dict(previous.filled_rows) if previous else {}),
                    name: fill["filled_rows"],
                },
            )
    return applied, failures


def acs_local_reviewed_fill_consumer_gate(
    manifests: Mapping[str, object],
    *,
    document: object | None = None,
    never_default_filled: Collection[tuple[str, str]] = (),
    installed_engine_version: str | None = None,
) -> GateResult:
    """Every reviewed default fill the release applied carries its notes.

    ``manifests`` maps a fill manifest's name to its parsed payload (the
    ``fill_reviewed_nulls`` manifest: a ``fills`` list of entity, column,
    fill value, filled rows and missing rows by spine). The gate fails when a
    manifest cannot say what was filled, when an applied fill has no register
    entry or its fill value or spines differ from the entry, and when the
    register fails :func:`reviewed_fill_register_failures`. Entries the
    release did not use are reported in the details, not failed.

    ``installed_engine_version`` is the policyengine-us version the release
    runs (the installed one, which packaging records as
    ``built_with_model_package``). The notes hold only for the engine they
    were reviewed against, so the gate fails when it is missing or differs
    from ``reviewed_against.policyengine_us`` (microcosm#1071 review): a
    release on a later engine needs the engine-tier consumer walk re-run and
    the register re-reviewed, even where that test is skipped.
    """

    if document is None:
        document = load_reviewed_fill_consumer_register()
    failures = [
        f"register: {failure}"
        for failure in reviewed_fill_register_failures(
            document, never_default_filled=never_default_filled
        )
    ]
    applied, manifest_failures = _applied_fills(manifests)
    failures += manifest_failures
    details: dict[str, object] = {
        "issue": ACS_LOCAL_REVIEWED_FILL_CONSUMERS_ISSUE,
        "register": ACS_LOCAL_REVIEWED_FILL_CONSUMER_REGISTER,
        "register_sha256": reviewed_fill_register_sha256(document),
        "manifests": {
            name: len(manifest["fills"])
            if isinstance(manifest, Mapping) and isinstance(manifest.get("fills"), list)
            else None
            for name, manifest in manifests.items()
        },
        "applied_fills": len(applied),
    }
    if installed_engine_version is not None:
        details["installed_policyengine_us"] = installed_engine_version
    if failures and any(failure.startswith("register: ") for failure in failures):
        return GateResult(
            name=ACS_LOCAL_REVIEWED_FILL_CONSUMER_GATE_NAME,
            passed=False,
            failures=tuple(failures),
            details=details,
        )

    entries = {entry["column"]: entry for entry in document["entries"]}
    details["reviewed_against"] = dict(document["reviewed_against"])
    details["register_entries"] = len(entries)
    reviewed_engine = document["reviewed_against"]["policyengine_us"]
    details["engine_matches_review"] = installed_engine_version == reviewed_engine
    if installed_engine_version is None:
        failures.append(
            "the release reported no policyengine-us version, so its engine "
            f"cannot be matched to the register's review ({reviewed_engine})."
        )
    elif installed_engine_version != reviewed_engine:
        failures.append(
            f"the release runs policyengine-us {installed_engine_version} but "
            f"the register was reviewed against {reviewed_engine}: re-run the "
            "engine-tier consumer walk (test_us_acs_local_reviewed_fill_consumers"
            ".py) on the new engine, review every note whose program's rules "
            "changed, and update reviewed_against."
        )
    fills_detail: dict[str, object] = {}
    known_bias: dict[str, dict[str, str]] = {}
    for (entity, column), fill in sorted(applied.items()):
        label = f"{entity}.{column}"
        entry = entries.get(column)
        if entry is None or entry["entity"] != entity:
            failures.append(
                f"{label}: a reviewed default fill ({fill.fill_value}) the release "
                "applied has no entry in "
                f"{ACS_LOCAL_REVIEWED_FILL_CONSUMER_REGISTER}; add one with its "
                "means-tested consumers and a note for each."
            )
            continue
        if entry["fill_value"] != fill.fill_value:
            failures.append(
                f"{label}: the release filled {fill.fill_value} but the entry "
                f"reviewed {entry['fill_value']}."
            )
        unexpected = sorted(fill.spines - set(entry["spines"]))
        if unexpected:
            failures.append(
                f"{label}: filled on spine(s) {unexpected} the entry's notes do "
                f"not cover (reviewed for {entry['spines']})."
            )
        verdicts = {
            program: (note["verdict"], note.get("direction"))
            for note in entry["notes"]
            for program in note["programs"]
        }
        biased = {
            program: direction
            for program, (verdict, direction) in sorted(verdicts.items())
            if verdict == "known_bias"
        }
        if biased:
            known_bias[label] = biased
        fills_detail[label] = {
            "fill_value": fill.fill_value,
            "filled_rows": dict(fill.filled_rows),
            "spines": sorted(fill.spines),
            "harmless": sorted(
                program
                for program, (verdict, _) in verdicts.items()
                if verdict == "harmless"
            ),
            "known_bias": biased,
        }
    used = {column for _entity, column in applied}
    unused = sorted(
        f"{entry['entity']}.{entry['column']}"
        for entry in document["entries"]
        if entry["column"] not in used
    )
    details["fills"] = fills_detail
    details["known_bias"] = known_bias
    details["unused_entries"] = unused
    if unused:
        details["warnings"] = [
            f"{len(unused)} register entr{'y' if len(unused) == 1 else 'ies'} "
            "describe a fill this release did not apply (reported, not failed)."
        ]
    return GateResult(
        name=ACS_LOCAL_REVIEWED_FILL_CONSUMER_GATE_NAME,
        passed=not failures,
        failures=tuple(failures),
        details=details,
    )


def reviewed_fill_register_summary(document: object) -> dict[str, object]:
    """The register's identity and known biases, for manifests and limitations."""

    entries = document["entries"]
    return {
        "issue": ACS_LOCAL_REVIEWED_FILL_CONSUMERS_ISSUE,
        "register": ACS_LOCAL_REVIEWED_FILL_CONSUMER_REGISTER,
        "sha256": reviewed_fill_register_sha256(document),
        "reviewed_against": dict(document["reviewed_against"]),
        "entries": len(entries),
        "known_bias": {
            f"{entry['entity']}.{entry['column']}": sorted(
                program
                for note in entry["notes"]
                if note["verdict"] == "known_bias"
                for program in note["programs"]
            )
            for entry in entries
            if any(note["verdict"] == "known_bias" for note in entry["notes"])
        },
    }


def means_tested_consumers(
    column: str,
    index: ConsumerIndex,
    programs: Mapping[str, Iterable[str]],
) -> dict[str, tuple[str, ...]]:
    """Each program whose roots ``column`` reaches, with one shortest path.

    Walks the index's consumer receipts breadth-first from ``column`` until
    nothing new is reached. ``programs`` maps a program to its root variables;
    a column that is itself a root reaches its own program. A variable the
    index does not know has no consumers; ``column`` itself must be known.
    """

    root_of: dict[str, str] = {}
    for program, roots in programs.items():
        for root in roots:
            root_of[root] = program

    def consumers(name: str) -> list[str]:
        try:
            receipts = index.consumer_receipts(name)
        except ValueError:
            if name == column:
                raise
            return []
        return sorted({receipt.consumer for receipt in receipts})

    parent: dict[str, str | None] = {column: None}
    queue = deque([column])
    while queue:
        node = queue.popleft()
        for consumer in consumers(node):
            if consumer not in parent:
                parent[consumer] = node
                queue.append(consumer)

    reached: dict[str, tuple[str, ...]] = {}
    for variable in parent:
        program = root_of.get(variable)
        if program is None:
            continue
        path: list[str] = []
        node: str | None = variable
        while node is not None:
            path.append(node)
            node = parent[node]
        candidate = tuple(reversed(path))
        if program not in reached or len(candidate) < len(reached[program]):
            reached[program] = candidate
    return dict(sorted(reached.items()))
