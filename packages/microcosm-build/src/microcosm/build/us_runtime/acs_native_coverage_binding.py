"""Closed real ACS preparation plus literal coverage issuance, without admission.

Private process ownership is required in addition to immutable evidence bytes.
This is a native population successor, not a decoder or a graph attachment.
"""

from __future__ import annotations

import ast
import builtins
import importlib.util
import json
import sys
import tempfile
from _thread import RLock
from dataclasses import InitVar, dataclass
from pathlib import Path
from types import BuiltinFunctionType, CodeType, FunctionType
from weakref import WeakKeyDictionary

from microcosm.frame import Frame

from . import acs_housing_universe_source as housing
from . import acs_person_coverage_authentication as coverage
from . import graph_implementation as implementation

PROTOCOL = "microcosm.acs-native-coverage-binding.v2"
MAX_ARCHIVE_BYTES = 8 * 1024**3  # combined compressed bytes, before capture
MAX_EXPANDED_BYTES = 16 * 1024**3  # combined, before opening any member
MAX_SOURCE_ROWS = 6_000_000  # per role, before full-source construction
MAX_EVIDENCE_BYTES = 2 * 1024**2
# Exact accepted direct AGEP -> A_AGE and AGEP -> age implementation, and owners.
# A new transform/owner version requires explicit review of this successor.
_ACCEPTED = {
    "acs_pums.py": "6ecf79f0dfb0c0bc0ad0af6be5fa65bd8c2d1e1968009402e4c8dee347de70dc",
    "acs_inputs.py": "aa4a8aeaba63dfef2f3e04fb89de59766deb088ed7f4d290aeba0425739916da",
    "acs_housing_universe_source.py": "beb46a4a05a13580a868be423809a77946441dcafc3a0f57160561157e93e9a3",
    "acs_person_coverage_authentication.py": "475aa795c8a5b49a0dd3405a0877866dddd03012a1f2b9743447fab1fe85bcff",
}
_TOKEN = object()
_ISSUED = WeakKeyDictionary()

# Only bytecode compilation is reused across producers. Fresh source reads,
# AST checks, local code indexes and loaded-function checks remain mandatory.
_COMPILE_CACHE_MAX_ENTRIES = 128
_COMPILE_CACHE_MAX_SOURCE_BYTES = 16 * 1024**2
_COMPILE_CACHE_MAX_ENTRY_BYTES = 1024**2
_COMPILE_CACHE_COMPILER = compile
_COMPILE_CACHE_DEFAULT_OPTIMIZE = sys.flags.optimize
_COMPILE_CACHE = {}
_COMPILE_CACHE_LOCK = RLock()


class ACSNativeCoverageBindingError(ValueError):
    """Static refusal without source values, paths or exception chains."""


def _require(condition, code):
    if not condition:
        raise ACSNativeCoverageBindingError(code)


def _clear_compile_cache():
    """Clear only compiled-source outputs, for process-local test isolation."""
    with _COMPILE_CACHE_LOCK:
        _COMPILE_CACHE.clear()


def _compile_source(
    source, filename, mode="exec", *, flags=0, dont_inherit=True, optimize=-1
):
    """Reuse bounded immutable bytecode; never reuse loaded-function validity.

    Non-original compilers and context-dependent or non-exact inputs bypass the
    cache. Compiler warning/audit events occur on misses; AST parsing still runs
    on every _live_code call. This shares the trusted-process scope of that check.
    """
    compiler = compile
    eligible = (
        compiler is _COMPILE_CACHE_COMPILER
        and type(compiler) is BuiltinFunctionType
        and compiler.__module__ == "builtins"
        and compiler.__name__ == "compile"
        and compiler.__self__ is builtins
        and type(source) is bytes
        and type(filename) is str
        and type(mode) is str
        and mode in ("exec", "eval", "single")
        and type(flags) is int
        and flags >= 0
        and dont_inherit is True
        and type(optimize) is int
        and optimize in (-1, 0, 1, 2)
        and len(source) <= _COMPILE_CACHE_MAX_ENTRY_BYTES
        and len(source) <= _COMPILE_CACHE_MAX_SOURCE_BYTES
        and _COMPILE_CACHE_MAX_ENTRIES > 0
    )
    key = None
    if eligible:
        effective_optimize = (
            _COMPILE_CACHE_DEFAULT_OPTIMIZE if optimize == -1 else optimize
        )
        key = (
            source,
            filename,
            mode,
            flags,
            dont_inherit,
            effective_optimize,
            id(compiler),
        )
        with _COMPILE_CACHE_LOCK:
            entry = _COMPILE_CACHE.get(key)
            if (
                type(entry) is tuple
                and len(entry) == 3
                and type(entry[0]) is tuple
                and entry[0] == key
                and entry[1] is compiler
                and type(entry[2]) is CodeType
            ):
                return entry[2]
            _COMPILE_CACHE.pop(key, None)

    # Compile outside the lock: instrumentation/audit hooks may reenter, and a
    # concurrent duplicate miss is harmless. Never retain a failed compilation.
    code = compiler(
        source,
        filename,
        mode,
        flags=flags,
        dont_inherit=dont_inherit,
        optimize=optimize,
    )
    if key is not None and type(code) is CodeType:
        with _COMPILE_CACHE_LOCK:
            # A nested/concurrent call may already have filled this key. FIFO
            # eviction bounds retained source keys; it is not a hard RSS cap.
            if key not in _COMPILE_CACHE:
                while _COMPILE_CACHE and (
                    len(_COMPILE_CACHE) >= _COMPILE_CACHE_MAX_ENTRIES
                    or sum(len(item[0]) for item in _COMPILE_CACHE) + len(source)
                    > _COMPILE_CACHE_MAX_SOURCE_BYTES
                ):
                    del _COMPILE_CACHE[next(iter(_COMPILE_CACHE))]
                _COMPILE_CACHE[key] = (key, compiler, code)
    return code


def _live_code(module, compiled):
    """Check loaded Python implementations against their current source bytes.

    Covers module functions, imported function aliases and source class methods;
    generated dataclass methods have no source code object. This is a drift
    check in a trusted process, not a sandbox against arbitrary Python execution.
    """
    path = getattr(module, "__file__", None)
    if path is None:
        return
    tree = ast.parse(Path(path).read_bytes())
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            value = getattr(module, node.name, None)
            _require(
                value is not None and value.__module__ == module.__name__,
                "LOADED_PRODUCER",
            )
        elif isinstance(node, ast.ImportFrom) and node.module != "__future__":
            origin_name = (
                importlib.util.resolve_name(
                    "." * node.level + (node.module or ""), module.__package__
                )
                if node.level
                else node.module
            )
            origin = sys.modules.get(origin_name)
            for alias in node.names:
                if alias.name != "*" and origin is not None:
                    _require(
                        getattr(module, alias.asname or alias.name, None)
                        is getattr(origin, alias.name, None),
                        "LOADED_PRODUCER",
                    )

    def check(function):
        if not isinstance(function, FunctionType):
            return
        origin = sys.modules.get(function.__globals__.get("__name__"))
        path = getattr(origin, "__file__", None)
        _require(path is not None, "LOADED_PRODUCER")
        if function.__code__.co_filename == "<string>":
            return  # dataclass-generated methods
        if path not in compiled:
            codes = {}

            def visit(code):
                codes[code.co_qualname] = code
                for value in code.co_consts:
                    if isinstance(value, CodeType):
                        visit(value)

            visit(
                _compile_source(
                    Path(path).read_bytes(), path, "exec", dont_inherit=True
                )
            )
            compiled[path] = codes
        _require(
            function.__globals__ is vars(origin)
            and function.__code__ == compiled[path].get(function.__code__.co_qualname),
            "LOADED_PRODUCER",
        )

        for cell in function.__closure__ or ():
            if isinstance(cell.cell_contents, FunctionType):
                check(cell.cell_contents)

    for value in vars(module).values():
        if isinstance(value, FunctionType):
            check(value)
        elif isinstance(value, type) and value.__module__ == module.__name__:
            for method in vars(value).values():
                if isinstance(method, (classmethod, staticmethod)):
                    method = method.__func__
                if isinstance(method, property):
                    check(method.fget)
                    check(method.fset)
                else:
                    check(method)


def _producer():
    _require(PROTOCOL == "microcosm.acs-native-coverage-binding.v2", "NATIVE_PROTOCOL")
    for name, expected in _ACCEPTED.items():
        _require(
            coverage._sha(Path(__file__).with_name(name).read_bytes()) == expected,
            "UNREVIEWED_PREPARATION",
        )
    manifest = implementation.implementation_manifest(housing.ACS_HU_STAGE)
    # Reuse the reviewed complete preparation closure, without changing its
    # graph inventory or historical hashes. Include source-only coverage's own
    # closure, C _csv provider (libpython where builtin), and this successor.
    names = {
        "microcosm.build.us_runtime.acs_person_coverage_authentication",
        "microcosm.build.us_runtime.acs_person_coverage_columns",
        __name__,
    }
    for item in manifest["modules"]:
        package, relative = item.split("/", 1)
        name = package + "." + relative.removesuffix(".py").replace("/", ".")
        names.add(name.removesuffix(".__init__"))
    compiled = {}
    for name in sorted(names):
        module = sys.modules.get(name)
        if module is not None:
            _live_code(module, compiled)
    return {
        "preparation": manifest,
        "coverage": coverage._producer(),
        "issuer_sha256": coverage._sha(Path(__file__).read_bytes()),
        "limits": [
            MAX_ARCHIVE_BYTES,
            MAX_EXPANDED_BYTES,
            MAX_SOURCE_ROWS,
            MAX_EVIDENCE_BYTES,
            sys.modules[housing.AcsPumsSource.__module__].MAX_EXACT_HOUSEHOLDS,
            sys.modules[housing.AcsPumsSource.__module__].MAX_EXACT_PERSON_ROWS,
        ],
        "accepted_age_transform": "literal_numeric_identity_AGEP_to_AGEP_A_AGE_age",
    }


def _archives(pins):
    _require(
        len(pins) == 2 and {p[0] for p in pins} == {"household", "person"},
        "SOURCE_ROLES",
    )
    return [
        {"role": role, "filename": name, "sha256": digest, "bytes": size}
        for role, name, digest, size in pins
    ]


def _members_equal(left, right):
    # Owners intentionally have distinct parser profiles and inventory shapes.
    # Compare common captured byte/header/row identities, never parser claims.
    fields = ("name", "bytes", "sha256", "compressed_bytes", "crc32", "rows")
    for role in ("household", "person"):
        a, b = left[role], right[role]
        _require(len(a) == len(b), "SOURCE_MEMBER_IDENTITY")
        for x, y in zip(a, b, strict=True):
            _require(all(x[k] == y[k] for k in fields), "SOURCE_MEMBER_IDENTITY")
            if y["applicable"]:
                _require(
                    x["header_sha256"] == y["header_sha256"], "SOURCE_MEMBER_IDENTITY"
                )


def _preflight(paths, serialnos=None):
    # Directory/expanded-byte gates precede member opening. Streaming lexical
    # gates then precede preparation's full projection and pandas construction.
    expanded = 0
    for role, path in paths.items():
        with coverage.zipfile.ZipFile(path) as archive:
            members, _prefix = coverage._members(archive, role)
            expanded += sum(member.file_size for member in members)
            _require(expanded <= MAX_EXPANDED_BYTES, "EXPANDED_BUDGET")
    inventories = {}
    if serialnos is None:
        # Preserve the whole-source route and its pre-construction row ceiling.
        for role, path in paths.items():
            inventories[role], _empty = coverage._inventory(path, role, frozenset())
            _require(
                sum(m["rows"] for m in inventories[role]) <= MAX_SOURCE_ROWS,
                "SOURCE_ROW_BUDGET",
            )
            if role == "person":
                _require(
                    sum(m["rows"] for m in inventories[role])
                    <= coverage.literal.MAX_SELECTED_ROWS,
                    "NATIVE_ROW_BUDGET",
                )
        return inventories, None
    selected = frozenset(serialnos)
    inventories["household"], households = coverage._inventory(
        paths["household"], "household", selected
    )
    _require(
        sum(m["rows"] for m in inventories["household"]) <= MAX_SOURCE_ROWS,
        "SOURCE_ROW_BUDGET",
    )
    _require(set(households) == selected, "SELECTION_UNKNOWN")
    _require(all(0 <= n <= 20 for n in households.values()), "SELECTED_NP")
    expected_rows = sum(households.values())
    _require(expected_rows > 0, "EMPTY_SELECTED_POPULATION")
    _require(expected_rows <= coverage.literal.MAX_SELECTED_ROWS, "NATIVE_ROW_BUDGET")
    # _inventory bounds the actual selected row/body accumulation, independently
    # of reported NP. Every source member is still streamed and hashed.
    inventories["person"], roster = coverage._inventory(
        paths["person"], "person", selected
    )
    _require(
        sum(m["rows"] for m in inventories["person"]) <= MAX_SOURCE_ROWS,
        "SOURCE_ROW_BUDGET",
    )
    counts = dict.fromkeys(selected, 0)
    for serial, _line in roster:
        counts[serial] += 1
    _require(counts == households, "SELECTED_ROSTER")
    _require(len(roster) == expected_rows, "SELECTED_ROSTER")
    return inventories, (households, roster)


@dataclass(frozen=True, slots=True)
class _Owned:
    frame: Frame
    payload: bytes
    prepared: housing.PreparedACSHousingPopulation
    literal: coverage.AuthenticatedACSPersonCoverage
    snapshots: tuple
    pins: tuple


@dataclass(frozen=True, slots=True, weakref_slot=True, eq=False)
class AuthenticatedACSNativeCoverage:
    """Process-owned immutable evidence; the live Frame is checked on every borrow."""

    payload: bytes
    _token: InitVar[object] = None

    def __post_init__(self, _token):
        _require(_token is _TOKEN, "ISSUANCE_CONSTRUCTOR")

    @property
    def receipt(self):
        return json.loads(_owned(self).payload)

    @property
    def frame(self):
        verify_acs_native_coverage(self)
        return _owned(self).frame

    @property
    def coverage_table(self):
        return _owned(self).literal.table

    @property
    def source_coverage(self):
        """The unchanged predecessor still has an ungranted native_binding."""
        return _owned(self).literal


def _owned(issuance):
    _require(type(issuance) is AuthenticatedACSNativeCoverage, "ISSUANCE_TYPE")
    owned = _ISSUED.get(issuance)
    _require(
        owned is not None
        and type(issuance.payload) is bytes
        and issuance.payload == owned.payload,
        "ISSUANCE_NOT_OWNED",
    )
    return owned


def _verify_sources(snapshots, pins):
    for paths in snapshots:
        for role, _name, digest, size in pins:
            path = paths[role]
            housing._path_components(path)
            _require(housing._persisted_sha(path, size) == digest, "SOURCE_CHANGED")
    _require(housing._pins() == pins, "SOURCE_AUTHORITY_CHANGED")


def _frame_sha256(frame):
    # Supplement the unchanged owner's typed-cell identity with pandas axis
    # descriptors/attrs and every entity's resolved original DESIGN weights.
    details = {"storage": housing.frame_content_sha256(frame), "entities": {}}
    for entity in frame.entities:
        table = frame.table(entity)
        weights = frame.resolve_weights(entity)
        details["entities"][entity] = {
            "index_type": type(table.index).__name__,
            "index_dtype": repr(table.index.dtype),
            "column_axis_name": table.columns.name,
            "column_axis_dtype": repr(table.columns.dtype),
            "dtypes": [repr(dtype) for dtype in table.dtypes],
            "attrs": table.attrs,
            "allows_duplicate_labels": table.flags.allows_duplicate_labels,
            "weight_kind": weights.kind.value,
            "weight_dtype": str(weights.values.dtype),
            "weight_shape": list(weights.values.shape),
            "weight_sha256": coverage._sha(weights.values.tobytes()),
        }
    details["strata_attrs"] = frame.strata.attrs
    details["strata_index_type"] = type(frame.strata.index).__name__
    details["strata_allows_duplicate_labels"] = (
        frame.strata.flags.allows_duplicate_labels
    )
    _require(not frame._link_tables, "UNEXPECTED_LINK_TABLES")
    return coverage._sha(coverage._json(details, MAX_EVIDENCE_BYTES))


def _verify_prepared_frame(prepared):
    recorded = json.loads(prepared.receipt_json)["frame_sha256"]
    _require(
        housing.frame_content_sha256(prepared.frame) == recorded,
        "PREPARATION_FRAME_CHANGED",
    )
    return recorded


def _verify_frame(owned):
    receipt = json.loads(owned.payload)
    _require(
        housing.frame_content_sha256(owned.frame) == receipt["frame_sha256"]
        and _frame_sha256(owned.frame) == receipt["exact_frame_sha256"],
        "FRAME_CHANGED",
    )
    _require(
        _verify_prepared_frame(owned.prepared) == receipt["frame_sha256"],
        "PREPARATION_FRAME_CHANGED",
    )


def _verify(owned, producer):
    receipt = json.loads(owned.payload)
    frame, prepared, literal = owned.frame, owned.prepared, owned.literal
    _require(receipt["protocol"] == PROTOCOL, "NATIVE_PROTOCOL")
    _require(receipt["producer"] == producer, "PRODUCER_CHANGED")
    _require(
        prepared.frame is frame
        and coverage._sha(prepared.receipt_json)
        == receipt["preparation_receipt_sha256"]
        and coverage._sha(prepared.source.receipt_json)
        == receipt["housing_receipt_sha256"]
        and coverage._sha(prepared.source.projection_json)
        == receipt["projection_sha256"]
        and coverage._sha(literal.payload) == receipt["coverage_payload_sha256"],
        "EVIDENCE_CHANGED",
    )
    _verify_frame(owned)
    housing.verify_acs_frame_projection(
        frame, json.loads(prepared.source.projection_json)
    )
    consistency = coverage.verify_acs_coverage_native_consistency(
        literal, frame
    ).receipt
    _require(
        consistency["original_age"]["relation"] == "numeric_identity"
        and consistency["original_age"]["matching_rows"] == frame.n("person"),
        "ORIGINAL_AGE_IDENTITY",
    )
    _verify_sources(owned.snapshots, owned.pins)
    _verify_frame(owned)


def verify_acs_native_coverage(issuance, frame=None):
    """Authenticate this issuance and exact live Frame, sources and producer.

    A separately built equal Frame is not this issuance. Any borrowed mutable
    Frame must be verified again at consumption; retaining a borrow bypasses no
    checks here but Python cannot intercept arbitrary downstream table reads.
    Callers must hold exclusive access to mutable tables during verification
    and consumption. Final checks detect changes during archive/producer work;
    they do not make pandas tables an atomic concurrent snapshot.
    """
    try:
        owned = _owned(issuance)
        _require(frame is None or frame is owned.frame, "FRAME_NOT_ISSUED")
        producer = _producer()
        _verify(owned, producer)
        _require(_producer() == producer, "PRODUCER_CHANGED")
        _verify_frame(owned)
        return issuance
    except ACSNativeCoverageBindingError:
        raise
    except Exception:
        raise ACSNativeCoverageBindingError("NATIVE_VERIFICATION_REFUSED") from None


def issue_acs_native_coverage(
    source_dir, *, snapshot_root, serialnos=None, candidate_path=None
):
    """Run real closed owners with optional exact engineering household selection.

    The complete selected roster is bounded before native construction. Full
    source validation and lexical projection remain. Candidate bytes, if given,
    are compared only after independent real reconstruction and never decoded.
    """
    try:
        serialnos = housing.AcsPumsSource.snapshot_serialnos(serialnos)
        # Snapshot path-like inputs once too, before producer/capture work.
        source_dir = Path(source_dir).absolute()
        snapshot_root = Path(snapshot_root).absolute()
        candidate_path = (
            None if candidate_path is None else Path(candidate_path).absolute()
        )
        coverage._json(
            serialnos, min(MAX_EVIDENCE_BYTES, housing.ACS_HU_RECEIPT_MAX_BYTES)
        )
        pins = housing._pins()
        archives = _archives(pins)
        _require(sum(p[3] for p in pins) <= MAX_ARCHIVE_BYTES, "ARCHIVE_BUDGET")
        producer = _producer()
        with housing._capture(source_dir, snapshot_root) as (private, paths, captured):
            _require(captured == pins, "SOURCE_AUTHORITY_CHANGED")
            inventory, selected_roster = _preflight(paths, serialnos)
            # These roots are outside the archive-only captured source directory.
            roots = [
                Path(tempfile.mkdtemp(prefix=prefix, dir=snapshot_root))
                for prefix in ("acs-native-preparation-", "acs-native-coverage-")
            ]
            prepared = housing.prepare_acs_housing_population(
                private, snapshot_root=roots[0], serialnos=serialnos
            )
            _verify_prepared_frame(prepared)
            _require(
                prepared.receipt["format"] == "microcosm.acs_housing_preparation.v2",
                "PREPARATION_PROTOCOL",
            )
            _require(
                prepared.receipt["requested_serialnos"]
                == (None if serialnos is None else list(serialnos)),
                "PREPARATION_SELECTION",
            )
            keys, _roster_sha = coverage._native_roster(prepared.frame)
            if selected_roster is not None:
                households, roster = selected_roster
                _require(
                    {s: n for s, n in households.items() if n > 0}
                    == keys.groupby("SERIALNO").size().to_dict()
                    and set(roster)
                    == set(zip(keys.SERIALNO, keys.SPORDER, strict=True)),
                    "SELECTED_NATIVE_ROSTER",
                )
            literal = coverage.load_authenticated_acs_person_coverage(
                private, snapshot_root=roots[1], frame=prepared.frame
            )
            hreceipt, creceipt = (
                json.loads(prepared.source.receipt_json),
                literal.receipt,
            )
            _require(
                hreceipt["archives"] == creceipt["archives"] == archives,
                "SOURCE_ARCHIVE_IDENTITY",
            )
            _members_equal(hreceipt["members"], creceipt["members"])
            _require(creceipt["members"] == inventory, "SOURCE_MEMBER_IDENTITY")
            relation = literal.native_binding.receipt["original_age"]
            _require(
                relation["relation"] == "numeric_identity", "ORIGINAL_AGE_IDENTITY"
            )
            snapshots = [
                {
                    role: Path(source_dir).absolute() / name
                    for role, name, _d, _s in pins
                },
                paths,
            ]
            for root in roots:
                children = tuple(root.iterdir())
                _require(len(children) == 1 and children[0].is_dir(), "CAPTURE_ROSTER")
                snapshots.append(
                    {role: children[0] / name for role, name, _d, _s in pins}
                )
            receipt = {
                "protocol": PROTOCOL,
                "producer": producer,
                "archives": archives,
                "vintage": 2024,
                "selection": {
                    "kind": "all" if serialnos is None else "engineering_exact_keys",
                    "requested_serialnos": serialnos,
                    "complete_selected_roster": True,
                    "person_rows": prepared.frame.n("person"),
                    "raw_person_keys_sha256": coverage._sha(
                        coverage._json(
                            sorted(
                                zip(keys.SERIALNO, map(int, keys.SPORDER), strict=True)
                            ),
                            coverage.MAX_BODY_BYTES,
                        )
                    ),
                    "vacant_serialnos": None
                    if selected_roster is None
                    else sorted(s for s, n in selected_roster[0].items() if n == 0),
                    "vacancy_status": "source_record_without_population_rows",
                    "before_person_accumulation_and_unit_assignment": serialnos
                    is not None,
                    "full_source_inclusion_probability": None,
                    "representative_sample": False,
                },
                "original_literal_authentication": {
                    "source_authenticated": True,
                    "fields": ["AGEP", "MIL", "ESR"],
                    "raw_literals_preserved": True,
                    "predecessor_native_binding_authenticated": False,
                },
                "native_preparation_issuance": {
                    "population_binding_authenticated": True,
                    "producer_executed_by_issuer": True,
                    "complete_original_nonvacant_roster": serialnos is None,
                    "complete_selected_original_roster": True,
                    "source_aliases_and_vintage_checked": True,
                    "age_relationship": "literal_AGEP_to_native_AGEP_A_AGE_to_model_age_identity",
                    "matching_age_rows": relation["matching_rows"],
                    "native_roster_sha256": creceipt["native_consistency"][
                        "native_roster_sha256"
                    ],
                },
                "design_anchors": {
                    "authenticated": True,
                    "kind": "design",
                    "source": "WGTP; single-person PWGTP for WGTP=0 GQ placeholders",
                    "renormalized": False,
                    "original_weight_literals_projection_sha256": coverage._sha(
                        prepared.source.projection_json
                    ),
                    "weight_sha256": coverage._sha(
                        prepared.frame.weights_for("household").values.tobytes()
                    ),
                },
                "assumptions": {
                    "coverage_status": "literal_source_fields_only",
                    "cross_survey_coverage_equivalence_established": False,
                    "domain_assignment_authenticated": False,
                    "period_harmonized": False,
                    "release_eligible": False,
                    "graph_attachment": "requires_reviewed_typed_successor",
                },
                "capture": {
                    "owners_share_physical_snapshot": False,
                    "owners_separately_capture_same_pins": True,
                    "issuer_preflight_capture_count": 1,
                    "owner_capture_count": 2,
                    "full_source_construction_peak_remains": True,
                    "selected_only_memory": False,
                },
                "frame_sha256": housing.frame_content_sha256(prepared.frame),
                "exact_frame_sha256": _frame_sha256(prepared.frame),
                "preparation_receipt_sha256": coverage._sha(prepared.receipt_json),
                "housing_receipt_sha256": coverage._sha(prepared.source.receipt_json),
                "projection_sha256": coverage._sha(prepared.source.projection_json),
                "coverage_payload_sha256": coverage._sha(literal.payload),
            }
            payload = coverage._json(receipt, MAX_EVIDENCE_BYTES)
            owned = _Owned(
                prepared.frame, payload, prepared, literal, tuple(snapshots), pins
            )
            if candidate_path is not None:
                target = private / "candidate.native-coverage"
                digest = housing._copy(
                    Path(candidate_path), target, len(payload), exact_size=len(payload)
                )
                _require(
                    digest == coverage._sha(payload)
                    and housing._persisted_sha(target, len(payload)) == digest,
                    "CANDIDATE_MISMATCH",
                )
            _verify(owned, producer)
            _require(_producer() == producer, "PRODUCER_CHANGED")
            _verify_frame(owned)
        # Capture's exit authority check must complete before minting the token.
        _verify_frame(owned)
        result = AuthenticatedACSNativeCoverage(payload, _token=_TOKEN)
        _ISSUED[result] = owned
        return result
    except ACSNativeCoverageBindingError:
        raise
    except Exception:
        raise ACSNativeCoverageBindingError("NATIVE_ISSUANCE_REFUSED") from None
