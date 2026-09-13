"""One source-backed canonical PUF55 donor CREATE, shared by both routes.

All ordinary returns reach one nine-predictor model Frame. An eight-predictor
training Slice can omit the transported SS total without constructing another
donor cohort. Technical model persons are not source persons. The emitted
receipts describe construction; the eventual host must verify actual source
keys, producer implementations and typed ancestry before trusting any model.
"""

from __future__ import annotations

import hashlib
import json
import sys
from collections.abc import Mapping
from dataclasses import fields, is_dataclass
from functools import partial
from pathlib import Path
from types import FunctionType

import numpy as np

from microcosm.graph import (
    ArtifactOutput,
    ArtifactType,
    Capabilities,
    Determinism,
    KernelBase,
    KernelResult,
    Node,
    Numeric,
    Owned,
    SeedSource,
    StructuralDelta,
    codecs,
    source_hash,
)
from microcosm.graph import canonical as graph_canonical
from microcosm.graph.canonical import canonical_json
from microcosm.graph.population import Population

from . import full_puf_enrichment as full
from . import graph_full_puf_enrichment as physical
from . import puf55_canonical_donor as projection
from . import puf55_route_finalization as numerical
from . import puf59_canonical as canonical
from . import puf59_canonical_artifact as envelope
from . import puf_full_source as source
from . import puf_full_source_graph as source_graph
from . import puf_interest_components as interest
from . import puf_raw_source as raw

CANONICAL_DONOR_NODE = "survey_puf55.canonical_donor"
CANONICAL_DONOR_TYPE = ArtifactType("microcosm.us.puf59_canonical_return", 2)
DONOR_PROJECTION_TYPE = ArtifactType("microcosm.us.puf55_donor_projection", 1)
PHASE = "us.survey_puf55.canonical_donor.v1"
MAX_PROJECTION_BYTES = 128 * 1024


def _require(condition, reason):
    if not condition:
        raise ValueError("PUF55_CANONICAL_SOURCE_" + reason)


def _sha(payload):
    return hashlib.sha256(payload).hexdigest()


def _pin(pin):
    _require(type(pin) is raw.SourcePin, "PIN_TYPE")
    values = tuple(getattr(pin, field.name) for field in fields(raw.SourcePin))
    _require(
        all(type(value) is str for value in values[:3])
        and type(pin.bytes) is int
        and all(type(value) is str for value in values[4:7])
        and type(pin.delivered_header) is tuple
        and all(type(value) is str for value in pin.delivered_header)
        and type(pin.data_records) is int,
        "PIN_FIELDS",
    )
    return values


def _definition(definition):
    """Pure check against reconstructed literals, never caller-made authority."""
    _require(type(definition) is raw.PufRawSourceDefinition, "DEFINITION_TYPE")
    payload, route, digest = definition.canonical, definition.route, definition.sha256
    _require(
        type(payload) is bytes
        and type(route) is str
        and route in ("packaged", "test_fixture")
        and type(digest) is str
        and _sha(payload) == digest
        and canonical_json(raw._thawed(definition.document)) == payload,
        "DEFINITION_BYTES",
    )
    rebuilt = raw._definition_from_document(json.loads(payload), route=route)
    actual = (
        route,
        payload,
        digest,
        _pin(definition.main),
        _pin(definition.demographic),
    )
    _require(
        actual
        == (
            rebuilt.route,
            rebuilt.canonical,
            rebuilt.sha256,
            _pin(rebuilt.main),
            _pin(rebuilt.demographic),
        ),
        "DEFINITION_PINS",
    )
    return actual


def _fresh_definition(fixture_definition):
    packaged_bytes = raw._packaged_bytes()
    _require(type(packaged_bytes) is bytes, "PACKAGED_BYTES")
    packaged = raw._definition_from_document(
        json.loads(packaged_bytes), route="packaged"
    )
    _definition(packaged)
    if fixture_definition is None:
        return packaged, _sha(packaged_bytes)
    _definition(fixture_definition)
    document = json.loads(fixture_definition.canonical)
    _require(
        fixture_definition.route == "test_fixture"
        and document.get("authority") == "invented_fixture_nonauthority"
        and not {fixture_definition.main.sha256, fixture_definition.demographic.sha256}
        & {packaged.main.sha256, packaged.demographic.sha256},
        "FIXTURE_AUTHORITY",
    )
    # Reproduce the maintained fixture contract with the freshly read packaged
    # pins, without consulting or accepting its process-global _PACKAGED cache.
    return raw._definition_from_document(document, route="test_fixture"), _sha(
        packaged_bytes
    )


def _recipe(seed, growth_scheme):
    _require(type(seed) is int and 0 <= seed < 2**64, "SEED")
    _require(
        type(growth_scheme) is str and growth_scheme in ("family_observed", "cpi_only"),
        "GROWTH_SCHEME",
    )
    qbi, growth = canonical.qbi, canonical.growth
    parameters = qbi._json(qbi.model_parameters()).encode()
    _require(_sha(parameters) == qbi.PARAMETERS_SHA256, "QBI_PARAMETERS")
    _require(
        _sha(growth._RECIPE_JSON.encode()) == growth.RECIPE_SHA256
        and canonical_json(growth.growth_recipe()) == canonical_json(growth._RECIPE),
        "GROWTH_RECIPE",
    )
    runtime_bands = interest.puf_e19200_agi_bands_runtime_identity()
    return {
        "phase": PHASE,
        "seed": seed,
        "growth_scheme": growth_scheme,
        "source_statistical_year": 2015,
        "money_year": 2024,
        "qbi_parameters_sha256": qbi.PARAMETERS_SHA256,
        "growth_recipe_sha256": growth.RECIPE_SHA256,
        "interest_asset_sha256": canonical.INTEREST_ASSET_SHA256,
        "interest_band_facts_sha256": canonical.INTEREST_BAND_FACTS_SHA256,
        "interest_runtime_sha256": runtime_bands["sha256"],
        "profile": full.PUF55_SURVEY_SS.value,
        "ordinary_cohort": "complete; zero design weights retained",
    }


def _params(definition, seed, growth_scheme, packaged_sha256):
    return {
        **_recipe(seed, growth_scheme),
        "definition": definition.params_text,
        "definition_sha256": definition.sha256,
        "packaged_definition_bytes_sha256": packaged_sha256,
        "source_pins": canonical_json(
            (_pin(definition.main), _pin(definition.demographic))
        ).decode("ascii"),
    }


def _node(definition, params):
    profile = full.require_puf_output_profile(full.PUF55_SURVEY_SS)
    _require(len(profile.predictors) == 9 and len(profile.targets) == 55, "PROFILE")
    return Node(
        CANONICAL_DONOR_NODE,
        CanonicalPuf55DonorKernel.ref,
        structural=StructuralDelta.CREATE,
        sources=(definition.main.source_name, definition.demographic.source_name),
        outputs=tuple(
            Owned("tax_unit", name, "float64")
            for name in (*profile.predictors, *profile.targets)
        ),
        params=params,
        artifact_outputs=(
            ArtifactOutput("full_return_source", source_graph.FULL_RETURN_SOURCE_TYPE),
            ArtifactOutput("canonical_donor", CANONICAL_DONOR_TYPE),
            ArtifactOutput("donor_projection", DONOR_PROJECTION_TYPE),
        ),
        description="Construct one complete source-pinned modeled PUF55 donor; nine/eight training Slices share its cohort.",
    )


def canonical_puf55_donor_node(
    *, seed=578, growth_scheme="family_observed", fixture_definition=None
):
    definition, packaged_sha = _fresh_definition(fixture_definition)
    return _node(definition, _params(definition, seed, growth_scheme, packaged_sha))


def _modules():
    # Include actual consumed codecs, numerical source/recipe owners and Frame
    # construction/storage helpers. No survey issuer or country engine is used.
    modules = (
        sys.modules[__name__],
        raw,
        source,
        source_graph,
        canonical,
        canonical.qbi,
        canonical.growth,
        envelope,
        projection,
        interest,
        full,
        full.support,
        numerical,
        full.codec,
        full.qrf_target,
        physical,
        physical.operator_column_contracts,
        physical.population_ops,
        physical.store_ops,
        codecs,
        graph_canonical,
        sys.modules[full.Frame.__module__],
        sys.modules[full.support.EntitySchema.__module__],
        sys.modules[full.support.Weights.__module__],
    )
    return tuple(dict.fromkeys(modules))


def _marker(value, depth=0):
    """Snapshot borrowed code/configuration as an immutable object graph.

    Functions can close over a namespace that contains the same function (for
    example generated class annotation machinery). Record an object's local
    reference before descending, so a back edge is explicit rather than an
    unbounded expansion. Each new call captures all reachable contents again;
    the memo is neither a source cache nor an exemption for mutable objects.
    """
    seen = {}

    def visit(item, level):
        kind = type(item)
        if kind in (type(None), bool, int, str, bytes, float):
            _require(level <= 16, "LIVE_DEPTH")
            return kind, item.hex() if kind is float else item
        previous = seen.get(id(item))
        if previous is not None:
            _require(previous[0] is item, "LIVE_REFERENCE")
            return "reference", previous[1]
        _require(level <= 16, "LIVE_DEPTH")
        reference = len(seen)
        # Keep the object alive while traversing borrowed mappings/dataclasses,
        # preventing an identity from being recycled during this snapshot.
        seen[id(item)] = (item, reference)
        if kind in (tuple, list):
            contents = tuple(visit(child, level + 1) for child in item)
        elif kind in (set, frozenset):
            contents = frozenset(visit(child, level + 1) for child in item)
        elif isinstance(item, Mapping):
            contents = tuple(
                (visit(k, level + 1), visit(v, level + 1)) for k, v in item.items()
            )
        elif kind is np.ndarray and not item.dtype.hasobject:
            contents = item.dtype.str, item.shape, item.tobytes()
        elif is_dataclass(item) and not isinstance(item, type):
            contents = tuple(
                (field.name, visit(getattr(item, field.name), level + 1))
                for field in fields(item)
            )
        elif kind is FunctionType:
            # Keep identity as well as immutable code/default/closure contents;
            # comparing retained mutable functions to themselves is not a seal.
            # Marshal bytes also encode interpreter reference-sharing state;
            # retaining a returned literal can change them without a code edit.
            # Code objects and their constant graph are immutable. Keep the
            # object alive, retain its identity, and snapshot public fields;
            # mutable function defaults/closures still use visit below.
            code_object = item.__code__
            code = (
                code_object,
                id(code_object),
                code_object.co_argcount,
                code_object.co_posonlyargcount,
                code_object.co_kwonlyargcount,
                code_object.co_nlocals,
                code_object.co_stacksize,
                code_object.co_flags,
                code_object.co_code,
                code_object.co_consts,
                code_object.co_names,
                code_object.co_varnames,
                code_object.co_freevars,
                code_object.co_cellvars,
                code_object.co_filename,
                code_object.co_name,
                code_object.co_qualname,
                code_object.co_firstlineno,
                code_object.co_linetable,
                code_object.co_exceptiontable,
            )
            defaults = visit(item.__defaults__, level + 1)
            kwdefaults = visit(item.__kwdefaults__, level + 1)
            closure = []
            for cell in item.__closure__ or ():
                try:
                    cell_value = cell.cell_contents
                except ValueError:
                    closure.append(("empty_cell",))
                else:
                    closure.append(("cell", visit(cell_value, level + 1)))
            contents = (
                item,
                code,
                defaults,
                kwdefaults,
                tuple(closure),
            )
        else:
            contents = id(item)
        return "object", reference, kind, contents

    return visit(value, depth)


def _live():
    result = []
    for module in _modules():
        result.append(("module", module, module.__name__, module.__file__))
        for name, value in vars(module).items():
            if name.startswith("__") or (module is raw and name == "_PACKAGED"):
                continue
            if isinstance(value, type) and value.__module__ == module.__name__:
                members = []
                for member, function in vars(value).items():
                    if isinstance(function, (staticmethod, classmethod)):
                        function = function.__func__
                    if isinstance(function, property):
                        function = function.fget
                    if type(function) is FunctionType:
                        members.append((member, _marker(function)))
                    elif not member.startswith("__"):
                        members.append((member, _marker(function)))
                result.append((module.__name__, name, value, tuple(members)))
            else:
                result.append((module.__name__, name, _marker(value)))
    result.append(
        (
            "rng",
            tuple(
                _marker(value)
                for value in (
                    np.random.default_rng,
                    np.random.Generator,
                    np.random.PCG64,
                    np.random.SeedSequence,
                )
            ),
        )
    )
    return tuple(result)


def _frame_seal(frame):
    return physical._population_stamp(
        Population.from_frame(frame, CANONICAL_DONOR_NODE)
    )


class CanonicalPuf55DonorKernel(KernelBase):
    ref = "us.survey_puf55.canonical_donor@1"
    capabilities = Capabilities(
        Determinism.SEEDED,
        numeric=Numeric.PLATFORM_BITWISE,
        seed_source=SeedSource.PARAM,
        structural=StructuralDelta.CREATE,
        dependencies=("numpy", "pandas"),
    )

    def __init__(
        self, *, seed=578, growth_scheme="family_observed", fixture_definition=None
    ):
        definition, packaged_sha = _fresh_definition(fixture_definition)
        self._definition = definition
        self.source_codecs = raw.puf_raw_source_codecs(definition)
        self.source_refs = raw.us_puf_raw_source_refs(definition)
        params = canonical_json(_params(definition, seed, growth_scheme, packaged_sha))
        # Retain primitive pins and immutable baselines before any future borrow.
        self._state = (
            definition,
            _definition(definition),
            params,
            packaged_sha,
            seed,
            growth_scheme,
            self.implementation_hash(),
            _live(),
        )
        self._check_state(self._state)

    def implementation_hash(self):
        # The executor calls this on required replay too. Resource reads cannot
        # live only in run(), which is deliberately skipped on a cache hit.
        state = getattr(self, "_state", None)
        if state is not None:
            self._check_state(state)
        packaged = raw._packaged_bytes()
        asset = interest.puf_e19200_interest_components_asset_identity()
        code = source_hash(*_modules(), dependencies=self.capabilities.dependencies)
        if state is not None:
            self._check_state(state)
        _require(
            type(packaged) is bytes
            and asset["asset_sha256"] == canonical.INTEREST_ASSET_SHA256
            and asset["agi_bands"]
            == interest.puf_e19200_agi_bands_runtime_identity()["agi_bands"],
            "IMPLEMENTATION_RESOURCES",
        )
        return _sha(
            canonical_json(
                {
                    "code": code,
                    "packaged_definition_bytes_sha256": _sha(packaged),
                    "interest_asset": asset,
                    "csv_acceptance": raw.csv_acceptance_profile(),
                }
            )
        )

    def _check_state(self, state):
        definition, definition_seal, params, packaged_sha, seed, scheme, _, live = state
        _require(
            self._state is state
            and self._definition is definition
            and _definition(definition) == definition_seal
            and canonical_json(_params(definition, seed, scheme, packaged_sha))
            == params
            and self.source_refs == raw.us_puf_raw_source_refs(definition)
            and _live() == live,
            "LIVE_STATE_CHANGED",
        )
        expected = (definition.main, definition.demographic)
        _require(
            type(self.source_codecs) is codecs.SourceCodecRegistry
            and not self.source_codecs.names(),
            "CODEC_MODE",
        )
        _require(
            set(self.source_codecs.bytes_names()) == {p.codec for p in expected},
            "CODEC_ROSTER",
        )
        for pin in expected:
            loader = self.source_codecs.get(pin.codec)
            _require(
                type(loader) is partial
                and loader.func is raw.load_pinned_puf_bytes
                and loader.args == ()
                and set(loader.keywords) == {"pin"}
                and loader.keywords["pin"] is pin
                and _pin(loader.keywords["pin"]) == _pin(pin),
                "CODEC_BINDING",
            )

    def _context(self, context, state, expected_paths=None):
        definition, _, params, *_ = state
        _require(
            context.node == _node(definition, json.loads(params))
            and canonical_json(dict(context.params)) == params
            and not context.tables
            and not context.weights
            and not context.artifacts
            and len(context.strata) == 0
            and set(context.sources)
            == {definition.main.source_name, definition.demographic.source_name},
            "DECLARATION_OR_CONTEXT",
        )
        paths = tuple(
            (p.source_name, context.sources[p.source_name])
            for p in (definition.main, definition.demographic)
        )
        _require(
            all(type(path) is type(Path()) and path.is_absolute() for _, path in paths),
            "SOURCE_PATH",
        )
        _require(
            expected_paths is None or paths == expected_paths, "SOURCE_PATH_CHANGED"
        )
        return paths

    def _read_sources(self, paths, state):
        definition = state[0]
        self._check_state(state)
        return tuple(
            codecs.load_source_bytes(pin.codec, path, registry=self.source_codecs)
            for pin, (_, path) in zip(
                (definition.main, definition.demographic), paths, strict=True
            )
        )

    def _read_resources(self, state):
        packaged_bytes = raw._packaged_bytes()
        asset = interest.puf_e19200_interest_components_asset_identity()
        implementation = self.implementation_hash()
        # All file/metadata reads precede these pure comparisons.
        self._check_state(state)
        _require(
            type(packaged_bytes) is bytes
            and _sha(packaged_bytes) == state[3]
            and implementation == state[6],
            "SOURCE_CODE_OR_RESOURCE_CHANGED",
        )
        _require(
            asset["asset_sha256"] == canonical.INTEREST_ASSET_SHA256
            and asset["agi_bands"]
            == interest.puf_e19200_agi_bands_runtime_identity()["agi_bands"],
            "INTEREST_RESOURCE",
        )

    def run(self, context):
        state = self._state
        self._check_state(state)
        paths = self._context(context, state)
        self._read_resources(state)
        buffers = self._read_sources(paths, state)
        definition, _, params, _, seed, scheme, *_ = state
        decoded = source.decode_full_puf_source(*buffers, definition)
        raw_payload = source.encode_full_puf_source(decoded)
        constructed = canonical.construct_canonical_puf59(
            decoded,
            interest_bands=interest.US_PUF_E19200_AGI_BANDS,
            interest_asset_sha256=canonical.INTEREST_ASSET_SHA256,
            seed=seed,
            growth_scheme=scheme,
        )
        canonical_payload = envelope.encode_canonical_puf59(
            constructed, expected_growth_scheme=scheme
        )
        donor, evidence = projection.canonical_puf55_donor_from_artifact(
            canonical_payload,
            expected_artifact_sha256=_sha(canonical_payload),
            expected_growth_scheme=scheme,
            profile=full.PUF55_SURVEY_SS,
        )
        profile = full.require_puf_output_profile(full.PUF55_SURVEY_SS)
        selected = full._validated_model_donor(donor, profile=profile)
        frame = numerical._model_donor_frame(selected)
        ids = decoded.status["RECID"][decoded.ordinary]
        _require(
            np.array_equal(donor.index.to_numpy(), ids)
            and np.array_equal(frame.table("tax_unit").index.to_numpy(), ids)
            and frame.weights_for("tax_unit").values.tobytes()
            == (
                decoded.status["S006"][decoded.ordinary].astype(np.float64) / 100
            ).tobytes()
            and all(dtype == np.dtype("float64") for dtype in selected.dtypes),
            "ORDINARY_COHORT_OR_CONVERSION",
        )
        frame_sha = _frame_seal(frame)
        receipt = {
            **json.loads(params),
            "protocol": PHASE,
            "source_route": definition.route,
            "source_rows": len(decoded.status["RECID"]),
            "ordinary_rows": len(ids),
            "zero_weight_rows": int((frame.weights_for("tax_unit").values == 0).sum()),
            "full_return_source_sha256": _sha(raw_payload),
            "canonical_donor_sha256": _sha(canonical_payload),
            "projection": evidence,
            "model_frame_sha256": frame_sha,
            "source_recid_sha256": _sha(ids.astype("<i8").tobytes()),
            "technical_ids_sha256": _sha(
                np.arange(1, len(ids) + 1, dtype="<i8").tobytes()
            ),
            "crosswalk": "source RECID in tax_unit pandas row order maps to technical IDs 1..ordinary_rows",
            "technical_person_rows_are_source_persons": False,
            "source_admission_issued": False,
            "population_admission_issued": False,
            "current_money_tax_unit_reconstruction": False,
            "release_eligible": False,
        }
        receipt_payload = canonical_json(receipt)
        _require(len(receipt_payload) <= MAX_PROJECTION_BYTES, "PROJECTION_SIZE")
        # Exact immutable expected roster; never compare a result dictionary to
        # another mutable dictionary exposed across final source/file borrows.
        payloads = (raw_payload, canonical_payload, receipt_payload)
        names = ("full_return_source", "canonical_donor", "donor_projection")
        _require(all(type(payload) is bytes for payload in payloads), "OUTPUT_BYTES")
        result = KernelResult(
            frame=frame,
            artifacts=dict(zip(names, payloads, strict=True)),
            receipt=receipt,
        )
        final_buffers = self._read_sources(paths, state)
        self._read_resources(state)
        # No source/store/resource I/O follows. Actual executor source-key and
        # output validation remains mandatory after this kernel returns.
        self._check_state(state)
        self._context(context, state, paths)
        _require(
            buffers == final_buffers and all(type(b) is bytes for b in final_buffers),
            "SOURCE_CHANGED",
        )
        _require(
            type(result) is KernelResult
            and result.frame is frame
            and type(result.frame) is full.Frame
            and not result.columns
            and result.keep is result.expand is result.weights is result.strata is None
            and set(result.artifacts) == set(names)
            and type(result.artifacts) is dict
            and all(type(name) is str for name in result.artifacts)
            and tuple(result.artifacts[name] for name in names) == payloads
            and all(type(result.artifacts[name]) is bytes for name in names)
            and canonical_json(result.receipt) == receipt_payload
            and _frame_seal(result.frame) == frame_sha,
            "FINAL_OUTPUT_CHANGED",
        )
        return result
