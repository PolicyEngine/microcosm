"""Bridge the UK country declarations to the canonical executable full graph.

The F0 compiler remains a static schema/identity projection. This adapter
validates its source projection, then uses the existing UK graph composition;
it never loads a historical candidate H5 or creates a second execution graph.
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Mapping
from dataclasses import replace
from datetime import date
from typing import TYPE_CHECKING

from microcosm.build.country_spec import CountrySpec, load_country_spec
from microcosm.build.uk_runtime.frs_release import load_uk_frs_release

if TYPE_CHECKING:
    from .graph_build import UKFullBuildConfig, UKFullGraph


def validate_uk_country_source_projection(spec: CountrySpec) -> None:
    """Keep raw input declarations and their executable stage pins identical."""
    if spec.country != "uk" or spec.resolved_spec is None:
        raise ValueError(
            "The UK full-build adapter requires a resolved UK country spec."
        )
    resolved = spec.resolved_spec
    sources = resolved.resource("sources").domain.to_wire()
    spine = resolved.resource("spine").domain.to_wire()
    bundle = resolved.resource("bundle").domain.to_wire()
    release = load_uk_frs_release()
    root = next(stage for stage in spec.sources.stages if stage.stage == "frs_spine")
    vintage = f"uk_frs_{release.vintage}"
    expected = [
        {
            "id": f"frs_{artifact['table']}",
            "role": "frs_raw_table",
            "sha256": artifact["sha256"],
            "byte_size": artifact["size_bytes"],
            "loader": "kernel:build_uk_frs_spine",
            "vintages": [f"vintage:{vintage}"],
            **(
                {
                    "vintage_authorities": [
                        {
                            "id": vintage,
                            "kind": "survey_period",
                            "value": release.survey_year,
                        }
                    ]
                }
                if artifact["table"] == "adult"
                else {}
            ),
        }
        for artifact in root.artifacts
        if artifact["role"] == "frs_table"
    ]
    # The atomic-area supports (microcosm#932) and their conditioned forms
    # (microcosm#931) are declared beside the raw FRS tables with their own
    # roles and loader; every other row is an FRS table.
    declared = [row for row in sources["sources"] if row["role"] == "frs_raw_table"]
    if declared != expected:
        raise ValueError(
            "UK country raw-source pins differ from the canonical FRS spine stage."
        )
    from .atomic_area_support import CONDITIONED_SOURCES, IMD_DECILE_COLUMN, SOURCES

    supports = {
        "uk_atomic_area_support": SOURCES,
        "uk_atomic_area_conditioned_support": CONDITIONED_SOURCES,
    }
    others = [row for row in sources["sources"] if row["role"] != "frs_raw_table"]
    if any(
        row["role"] not in supports
        or row["loader"] != "kernel:load_uk_atomic_area_support"
        for row in others
    ):
        raise ValueError(
            "UK country sources beyond the raw FRS tables must be atomic-area supports."
        )
    by_id = {row["id"]: row for row in others}
    if sorted(by_id) != sorted(
        source for ids in supports.values() for source in ids.values()
    ) or any(
        by_id[source]["role"] != role
        for role, ids in supports.items()
        for source in ids.values()
    ):
        raise ValueError(
            "UK atomic-area supports must be each system's support and its "
            "conditioned form."
        )
    # A conditioned support keeps its system's area vintages and adds the
    # vintages of the columns it conditions on.
    for system, source in SOURCES.items():
        if not set(by_id[source]["vintages"]) < set(
            by_id[CONDITIONED_SOURCES[system]]["vintages"]
        ):
            raise ValueError(
                f"The conditioned {system} support must extend its support's vintages."
            )
    # The FRS records each address's region and IMD decile; the conditioned
    # supports carry the decile the geography-first draw matches.
    if spine["channels"] != [
        {
            "id": "frs",
            "source": [row["id"] for row in expected],
            "observed_geography": "region",
            "observed_constraints": [IMD_DECILE_COLUMN],
        }
    ]:
        raise ValueError(
            "UK FRS channel must consume the declared raw FRS tables and observe "
            "the region and the IMD decile."
        )
    if bundle["dataset_run"]["target_period"] != release.calibration_year:
        raise ValueError(
            "UK country target period differs from the FRS release calibration year."
        )


def uk_atomic_support_register(
    spec: CountrySpec | None = None,
    *,
    conditioned: bool = False,
) -> dict[str, dict[str, object]]:
    """The committed identity of each atomic-area support, by system.

    ``sources.yaml`` pins every support's ``sha256`` and ``byte_size`` (the
    provenance register carries the same values). By default it reads the
    base supports, from which ``tools/build_uk_conditioned_area_supports.py``
    builds the conditioned ones; ``conditioned`` reads the conditioned
    supports (microcosm#931), the ones a geography-first spine's areas are
    drawn on. A release builds on exactly those bytes, whatever digests were
    pinned elsewhere: :func:`uk_spine_geography_binding_failures` holds a
    spine's support pins to that register, for the dense build's preparation
    and the dense release pre-flight alike.
    """
    from .atomic_area_support import CONDITIONED_SOURCES, SOURCES

    spec = load_country_spec("uk") if spec is None else spec
    if spec.resolved_spec is None:
        raise ValueError("The atomic-area support register needs a resolved UK spec.")
    role, sources = (
        ("uk_atomic_area_conditioned_support", CONDITIONED_SOURCES)
        if conditioned
        else ("uk_atomic_area_support", SOURCES)
    )
    rows = spec.resolved_spec.resource("sources").domain.to_wire()["sources"]
    by_id = {row["id"]: row for row in rows if row["role"] == role}
    register = {}
    for system, source in sources.items():
        row = by_id.get(source)
        if row is None:
            raise ValueError(
                f"sources.yaml declares no atomic-area support {source!r}."
            )
        register[system] = {
            "sha256": str(row["sha256"]),
            "size_bytes": int(row["byte_size"]),
        }
    return register


_SHA256_HEX = re.compile(r"[0-9a-f]{64}")

#: The register a spine's support pins were held to, by the name its
#: ``geography`` block records under ``register`` (microcosm#931): a release's
#: are held to sources.yaml's conditioned supports
#: (:func:`uk_atomic_support_register` with ``conditioned=True``), which the
#: spine build does before any stage runs. A synthetic fixture's toy supports
#: are held to no register, and its block says so.
UK_SPINE_SUPPORT_REGISTER = "sources.yaml"
UK_SYNTHETIC_SUPPORT_REGISTER = "synthetic"
_SUPPORT_REGISTERS = (UK_SPINE_SUPPORT_REGISTER, UK_SYNTHETIC_SUPPORT_REGISTER)


def uk_spine_geography_binding(
    supports: Mapping[str, bytes], *, register: str
) -> dict[str, object]:
    """A spine's ``geography`` block: how its areas were drawn (microcosm#931).

    ``supports`` holds the three conditioned supports' bytes, by system, as
    the spine's area stages read them, and ``register`` names the register
    their pins were held to (:data:`UK_SPINE_SUPPORT_REGISTER`, or
    :data:`UK_SYNTHETIC_SUPPORT_REGISTER` for a synthetic fixture's toys).
    The block holds, under the roster leaf's keys and in their order
    (``geography_first_roster.SIDECAR_GEOGRAPHY_KEYS``): the atomic law
    (``assignment``), the geography key's column (``identity``), the SHA-256
    of the conditioned definition the area draw builds on those supports at
    its declared seed, as the shared operator digests a definition
    (``definition_sha256``), each system's support SHA-256 and size in bytes
    (``support_pins``, the size an integer), the definition's stream, its seed
    at index 3 (``stream``) and ``register``. The spine build's sidecar writer
    and a spine request's dense build both compose it here, so
    :func:`uk_spine_geography_binding_failures` reads one shape whichever
    route bound the spine.

    Refused: a register other than those two, and every refusal of the
    definition (a support missing or not held to canonical bytes), by the
    shared operator's texts, which name no value.
    """

    from microcosm.graph.canonical import canonical_json

    from .atomic_area_conditioning import (
        IDENTITY_COLUMN,
        UK_SPINE_ATOMIC_AREA_SEED,
        uk_conditioned_assignment_definition,
    )
    from .atomic_area_support import SYSTEMS
    from .geography_first_roster import SIDECAR_GEOGRAPHY_KEYS

    if register not in _SUPPORT_REGISTERS:
        raise ValueError(
            "A spine's geography block names the register its support pins were "
            f"held to: one of {list(_SUPPORT_REGISTERS)}."
        )
    definition = uk_conditioned_assignment_definition(
        supports, seed=UK_SPINE_ATOMIC_AREA_SEED
    )
    values: dict[str, object] = {
        "assignment": "atomic",
        "identity": IDENTITY_COLUMN,
        "definition_sha256": hashlib.sha256(canonical_json(definition)).hexdigest(),
        "support_pins": {
            system: {
                "sha256": hashlib.sha256(supports[system]).hexdigest(),
                "size_bytes": len(supports[system]),
            }
            for system in SYSTEMS
        },
        "stream": [*definition["stream"]],
        "register": register,
    }
    return {key: values[key] for key in SIDECAR_GEOGRAPHY_KEYS}


def uk_spine_geography_binding_failures(
    binding: object,
    *,
    clone_count: int | None = None,
    spec: CountrySpec | None = None,
) -> tuple[str, ...]:
    """What a spine's geography binding fails of a release's (microcosm#931).

    ``binding`` is the spine's ``geography`` block as the dense build binds
    it: the bound spine's build sidecar block, with the spine's
    ``geographic_clone_count`` beside it, which the rowwise manifest records
    as ``geography.assignment``. A release's areas are drawn by the atomic
    law (``assignment`` ``"atomic"``), keyed on each clone's geography key
    (``identity``, the column's name), under a definition its SHA-256 names
    (``definition_sha256``), on exactly the conditioned supports
    ``sources.yaml`` registers (``support_pins``, each system's ``sha256`` and
    ``size_bytes`` as :func:`uk_atomic_support_register` gives them with
    ``conditioned=True``, the size an integer) and no system it does not
    list, on the spine's area stream at its declared seed (``stream``, the
    four entries of ``UK_SPINE_ATOMIC_AREA_DRAW_STREAM``, each of the same
    type as there), with the pins held to sources.yaml's register (``register``,
    :data:`UK_SPINE_SUPPORT_REGISTER`); with ``clone_count``, the block's K
    must be that integer, so a boolean or a float such as ``15.0`` is not it.
    A binding that is not a record fails every field. Its fields are those
    :func:`uk_spine_geography_binding` composes.

    The dense build's preparation holds a release candidate's spine to it,
    and the dense release pre-flight a candidate's manifest. Each failure
    names a field and what it must be, never the value it holds, so a
    refusal carries names and K only.
    """

    from .atomic_area_conditioning import UK_SPINE_ATOMIC_AREA_DRAW_STREAM
    from .atomic_area_support import IDENTITY_COLUMN

    block = binding if isinstance(binding, Mapping) else {}
    failures: list[str] = []
    if block.get("assignment") != "atomic":
        failures.append("assignment is not 'atomic'")
    if block.get("identity") != IDENTITY_COLUMN:
        failures.append(f"identity is not {IDENTITY_COLUMN!r}")
    digest = block.get("definition_sha256")
    if not (isinstance(digest, str) and _SHA256_HEX.fullmatch(digest)):
        failures.append("definition_sha256 is not a SHA-256 hex digest")
    register = uk_atomic_support_register(spec, conditioned=True)
    pins = block.get("support_pins")
    pins = pins if isinstance(pins, Mapping) else {}
    for system, expected in register.items():
        pin = pins.get(system)
        if not (
            isinstance(pin, Mapping)
            and pin.get("sha256") == expected["sha256"]
            and pin.get("size_bytes") == expected["size_bytes"]
            and type(pin.get("size_bytes")) is int
        ):
            failures.append(
                f"support_pins.{system} is not the registered conditioned support"
            )
    if set(pins) - set(register):
        failures.append("support_pins names a system the register does not")
    stream = block.get("stream")
    expected_stream = list(UK_SPINE_ATOMIC_AREA_DRAW_STREAM)
    if not (
        isinstance(stream, list)
        and stream == expected_stream
        and all(
            type(found) is type(wanted)
            for found, wanted in zip(stream, expected_stream, strict=True)
        )
    ):
        failures.append("stream is not the spine's area stream at its declared seed")
    if block.get("register") != UK_SPINE_SUPPORT_REGISTER:
        failures.append(f"register is not {UK_SPINE_SUPPORT_REGISTER!r}")
    if clone_count is not None:
        clones = block.get("geographic_clone_count")
        if type(clones) is not int or clones != clone_count:
            failures.append(f"geographic_clone_count is not K = {clone_count}")
    return tuple(failures)


def build_uk_country_graph(
    config: UKFullBuildConfig | None = None,
    *,
    spec: CountrySpec | None = None,
    engine_identity: str | None = None,
    review_date: date | None = None,
    gate_review_date: date | None = None,
    release_candidate: bool = False,
    skip_holdout: bool = True,
) -> UKFullGraph:
    """Compile the canonical full graph; default target scope is all geographies.

    The dense line runs on the declared spine's version as it stands: the
    spine's clones, areas and sampling are the build's, so the config's clone
    count must be the spine's (one for the declared spine; microcosm#931).
    ``gate_review_date`` is the calibrated gates' clock, by default the solve
    clock ``review_date`` (microcosm#1301).
    """
    from microcosm.graph import compile_graph
    from microcosm.graph.canonical import canonical_json

    from .graph import uk_spine_graph
    from .graph_build import UKFullBuildConfig, uk_full_graph
    from .graph_evidence import add_uk_spine_gate_nodes
    from .graph_terminal import append_uk_full_gate_nodes

    spec = load_country_spec("uk") if spec is None else spec
    validate_uk_country_source_projection(spec)
    release = load_uk_frs_release()
    if config is None:
        config = UKFullBuildConfig(
            calibration_year=release.calibration_year,
            time_period=release.time_period,
            source_year=release.survey_year,
        )
    if engine_identity is None:
        # The same provenance the spine and full builds hash, so a URL
        # install's commit keys this graph too.
        from microcosm.build.uk_runtime.spine_build import _rules_engine_provenance

        engine_identity = hashlib.sha256(
            canonical_json(_rules_engine_provenance())
        ).hexdigest()
    review_date = date.today() if review_date is None else review_date
    spine = add_uk_spine_gate_nodes(
        uk_spine_graph(spec, source_mode="split"),
        spec=spec,
        engine_identity=engine_identity,
        release_candidate=release_candidate,
    )
    full = uk_full_graph(
        config,
        spine=spine,
        review_date=review_date.isoformat(),
    )
    graph = append_uk_full_gate_nodes(
        full.graph,
        calibration=full.calibration,
        # The same roster uk_spine_graph builds: every manifest stage.
        spine_stage_names=tuple(stage.stage for stage in spec.sources.stages),
        engine_identity=engine_identity,
        review_date=review_date,
        gate_review_date=gate_review_date,
        sample_fraction=config.effective_sample_fraction,
        release_candidate=release_candidate,
        skip_holdout=skip_holdout,
    )
    compile_graph(graph)
    return replace(full, graph=graph)
