"""The dense line's spine posture (microcosm#931): the geography-first names
and the roster check that refuses a spine lacking any of them.

Every roster here is invented, read off the packaged spec, or written by the
spine build's own sidecar writer over a synthetic frame; no sidecar of a built
spine is read.
"""

import ast
import inspect
import json
from types import MappingProxyType, SimpleNamespace

import pytest

from microcosm.build.country_spec import load_country_spec
from microcosm.build.uk_runtime import geography_first_roster
from microcosm.build.uk_runtime.geography_first_roster import (
    GEOGRAPHY_FIRST_OPERATIONS,
    GEOGRAPHY_FIRST_STAGES,
    SIDECAR_GEOGRAPHY_KEYS,
)
from microcosm.build.uk_runtime.spine_posture import (
    require_geography_first_spine,
    roster_failures,
    spec_roster,
    stage_roster,
)
from test_support.microcosm_build.uk_full_build_cli import (
    geography_first_sidecar_fields,
    synthetic_spine_geography,
)

#: What a geography-first spine runs that a spine built from the packaged spec
#: does not yet: the area stages and the main-residence localiser. The
#: geography declarations shorten this list until it is empty.
_PACKAGED_ROSTER_LACKS = (
    "stage atomic_area_assign",
    "stage local_authority",
    "stage rfdr",
    "stage area_brma",
    "stage spi_copy_areas",
    "stage private_rent",
    "operation localise_main_residence_value in was_wealth",
)


def _passing():
    roster = geography_first_sidecar_fields("frs_spine")
    return roster["stages"], roster["operations"], roster["geography"]


def test_the_leaf_names_the_stages_the_operation_and_the_sidecar_keys():
    assert GEOGRAPHY_FIRST_STAGES == (
        "geographic_expand",
        "geographic_identity",
        "atomic_area_assign",
        "local_authority",
        "rfdr",
        "area_brma",
        "spi_copy_areas",
        "private_rent",
    )
    assert isinstance(GEOGRAPHY_FIRST_OPERATIONS, MappingProxyType)
    assert dict(GEOGRAPHY_FIRST_OPERATIONS) == {
        "was_wealth": ("localise_main_residence_value",)
    }
    assert SIDECAR_GEOGRAPHY_KEYS == (
        "assignment",
        "identity",
        "definition_sha256",
        "support_pins",
        "stream",
        "register",
    )
    for names in (
        GEOGRAPHY_FIRST_STAGES,
        SIDECAR_GEOGRAPHY_KEYS,
        *GEOGRAPHY_FIRST_OPERATIONS.values(),
    ):
        assert isinstance(names, tuple) and len(set(names)) == len(names)
    # The localiser runs inside a stage that is not one of the roster's own.
    assert not set(GEOGRAPHY_FIRST_OPERATIONS) & set(GEOGRAPHY_FIRST_STAGES)
    with pytest.raises(TypeError):
        GEOGRAPHY_FIRST_OPERATIONS["was_lisa"] = ()  # type: ignore[index]


def test_the_leaf_imports_nothing_first_party():
    """A names-only leaf: a module that reads it reaches no stage code."""
    imported = set()
    for node in ast.walk(ast.parse(inspect.getsource(geography_first_roster))):
        if isinstance(node, ast.ImportFrom):
            assert node.level == 0, node.module
            imported.add(node.module)
        elif isinstance(node, ast.Import):
            imported.update(alias.name for alias in node.names)
    assert imported == {"__future__", "collections.abc", "types"}


def test_each_roster_stage_is_the_stage_its_transform_runs_as():
    """The leaf restates the names its stages' transforms declare, since it
    imports none of them; this holds the two together."""
    from microcosm.build.uk_runtime import (
        geographic_expand,
        geographic_identity,
        spine_geography,
        spine_private_rent,
    )

    assert GEOGRAPHY_FIRST_STAGES == (
        geographic_expand.GEOGRAPHIC_EXPAND_STAGE_NAME,
        geographic_identity.GEOGRAPHIC_IDENTITY_STAGE_NAME,
        spine_geography.ATOMIC_AREA_ASSIGN_STAGE_NAME,
        spine_geography.LOCAL_AUTHORITY_STAGE_NAME,
        spine_geography.RFDR_STAGE_NAME,
        spine_geography.AREA_BRMA_STAGE_NAME,
        spine_geography.SPI_COPY_AREAS_STAGE_NAME,
        spine_private_rent.PRIVATE_RENT_STAGE_NAME,
    )


def test_the_packaged_roster_is_refused_naming_what_it_lacks(monkeypatch):
    """A spine built on this code lacks the area stages and the localiser,
    and the sidecar the spine build's own writer records for it has no
    geography block: so the dense line refuses every spine until the
    geography declarations land (microcosm#931). The stages the spine already
    runs pass."""
    from microcosm.build.uk_runtime import spine_build
    from test_support.microcosm_build.uk_full_population_graph import source_frame

    spec = load_country_spec("uk")
    stages, operations = spec_roster(spec)
    assert {"geographic_expand", "geographic_identity", "was_wealth"} <= set(stages)
    assert roster_failures(stages, operations, from_sidecar=False) == (
        _PACKAGED_ROSTER_LACKS
    )
    # The writer every spine build calls, over a synthetic frame and the
    # stages in the spine graph's order, read back through JSON as the dense
    # preparation reads it. Only the engine's provenance stands in.
    monkeypatch.setattr(
        spine_build, "_rules_engine_provenance", lambda: {"version": "stand-in"}
    )
    by_name = spec.sources.stage_map()
    written = spine_build._build_sidecar(
        frame=source_frame(),
        stages=[by_name[name] for name in spine_build._uk_spine_stage_names(spec)],
        records=[],
        artifact_pins={},
        resource_pins={},
        input_artifact_pins={},
        hmrc_replay={},
        stochastic_contract_sha256="0" * 64,
        frs_vintage="stand-in",
        sampling=None,
    )
    sidecar = json.loads(json.dumps(written))
    recorded = (sidecar["stages"], sidecar["operations"], sidecar.get("geography"))
    assert set(sidecar["stages"]) == set(stages)
    assert roster_failures(*recorded) == (*_PACKAGED_ROSTER_LACKS, "geography block")
    with pytest.raises(ValueError, match="builds only on a geography-first spine"):
        require_geography_first_spine(*recorded)


def test_spec_roster_reads_the_declared_names_and_kinds_in_order():
    def operation(kind):
        return SimpleNamespace(kind=kind)

    spec = SimpleNamespace(
        sources=SimpleNamespace(
            stages=(
                SimpleNamespace(
                    stage="second", operations=(operation("b"), operation("a"))
                ),
                SimpleNamespace(stage="first", operations=()),
            )
        )
    )
    assert spec_roster(spec) == (
        ("second", "first"),
        {"second": ("b", "a"), "first": ()},
    )
    # The same reading of any declared stages, a composed spine's among them,
    # in the order they come.
    assert stage_roster(iter(spec.sources.stages)) == spec_roster(spec)
    assert stage_roster(reversed(spec.sources.stages)) == (
        ("first", "second"),
        {"first": (), "second": ("b", "a")},
    )
    with pytest.raises(ValueError, match="declares no source stages"):
        spec_roster(SimpleNamespace(sources=None))


def test_a_geography_first_roster_passes_and_extra_names_never_fail():
    stages, operations, geography = _passing()
    assert roster_failures(stages, operations, geography) == ()
    assert require_geography_first_spine(stages, operations, geography) is None
    # The stand-in's block holds the leaf's keys, each once.
    assert tuple(synthetic_spine_geography()) == SIDECAR_GEOGRAPHY_KEYS
    wider = (
        ["frs_take_up", *stages, "nts_bus_travel"],
        {
            **operations,
            "was_wealth": ["derive", *operations["was_wealth"], "support_clip"],
            "nts_bus_travel": ["support_clip"],
        },
        {**geography, "seed": 0},
    )
    assert roster_failures(*wider) == ()
    # Tuples and read-only mappings read as lists and dicts do.
    assert (
        roster_failures(
            tuple(stages),
            MappingProxyType({key: tuple(kinds) for key, kinds in operations.items()}),
            MappingProxyType(geography),
        )
        == ()
    )
    # Only the block's keys are read, never their values.
    assert roster_failures(stages, operations, dict.fromkeys(geography)) == ()


def test_each_missing_name_fails_alone_in_the_leafs_order():
    stages, operations, geography = _passing()
    for stage in GEOGRAPHY_FIRST_STAGES:
        kept = [name for name in stages if name != stage]
        assert roster_failures(kept, operations, geography) == (f"stage {stage}",)
    for stage, required in GEOGRAPHY_FIRST_OPERATIONS.items():
        for kind in required:
            fewer = {
                **operations,
                stage: [name for name in operations[stage] if name != kind],
            }
            assert roster_failures(stages, fewer, geography) == (
                f"operation {kind} in {stage}",
            )
            # A stage the roster records no kinds for holds none, and an
            # operation counts only in a stage the roster ran.
            unlisted = {
                name: kinds for name, kinds in operations.items() if name != stage
            }
            assert roster_failures(stages, unlisted, geography) == (
                f"operation {kind} in {stage}",
            )
            unrun = [name for name in stages if name != stage]
            assert roster_failures(unrun, operations, geography) == (
                f"operation {kind} in {stage}",
            )
    for key in SIDECAR_GEOGRAPHY_KEYS:
        fewer = {name: value for name, value in geography.items() if name != key}
        assert roster_failures(stages, operations, fewer) == (f"geography key {key}",)
    for block in (None, [], "atomic", ["assignment"]):
        assert roster_failures(stages, operations, block) == ("geography block",)
    assert roster_failures([], {}, {}) == (
        *(f"stage {stage}" for stage in GEOGRAPHY_FIRST_STAGES),
        *(
            f"operation {kind} in {stage}"
            for stage, required in GEOGRAPHY_FIRST_OPERATIONS.items()
            for kind in required
        ),
        *(f"geography key {key}" for key in SIDECAR_GEOGRAPHY_KEYS),
    )


def test_an_unreadable_roster_fails_by_its_name():
    """A roster that is not names fails as a whole, in place of the names it
    would have held; a stage whose kinds are not names holds none."""
    stages, operations, geography = _passing()
    for bad in (None, "geographic_expand", b"geographic_expand", {"x": 1}, [1]):
        assert roster_failures(bad, operations, geography) == ("stage roster",)
    assert roster_failures([*stages, None], operations, geography) == ("stage roster",)
    assert roster_failures(None, operations, None) == (
        "stage roster",
        "geography block",
    )
    for bad in (None, [], ["was_wealth"], "was_wealth"):
        assert roster_failures(stages, bad, geography) == ("operation roster",)
    for kinds in (
        None,
        "localise_main_residence_value",
        [None],
        {"localise_main_residence_value": True},
    ):
        assert roster_failures(
            stages, {**operations, "was_wealth": kinds}, geography
        ) == ("operation localise_main_residence_value in was_wealth",)


def test_a_roster_read_without_a_sidecar_reads_no_geography_block():
    """A spine request's spine runs inside the dense build's own graph, so
    its roster has no block; one passed with it is a caller's error."""
    stages, operations, geography = _passing()
    assert roster_failures(stages, operations, from_sidecar=False) == ()
    assert roster_failures([], {}, from_sidecar=False) == (
        tuple(
            failure
            for failure in roster_failures([], {}, {})
            if not failure.startswith("geography key ")
        )
    )
    with pytest.raises(ValueError, match="no geography block to check"):
        roster_failures(stages, operations, geography, from_sidecar=False)
    with pytest.raises(ValueError, match="no geography block to check"):
        require_geography_first_spine(stages, operations, geography, from_sidecar=False)


def test_the_refusal_names_what_the_roster_lacks_and_nothing_else():
    stages, operations, geography = _passing()
    kept = [name for name in stages if name not in {"area_brma", "private_rent"}]
    fewer = {key: value for key, value in geography.items() if key != "stream"}
    with pytest.raises(ValueError) as sidecar:
        require_geography_first_spine(kept, operations, fewer)
    assert str(sidecar.value) == (
        "The dense line builds only on a geography-first spine, one that "
        "assigns every geographic clone its area (microcosm#931); the bound "
        "spine's build sidecar lacks: stage area_brma; stage private_rent; "
        "geography key stream."
    )
    with pytest.raises(ValueError) as request:
        require_geography_first_spine(kept, operations, from_sidecar=False)
    assert str(request.value).endswith(
        "the spine request's stage roster lacks: stage area_brma; stage private_rent."
    )
    # Names only: what else the roster holds, figures among it, never enters
    # the refusal, which names the leaf's missing names alone.
    with pytest.raises(ValueError) as figures:
        require_geography_first_spine(
            [*kept, "stage_with_4321_rows"],
            {**operations, "stage_with_4321_rows": ["kind_9876"]},
            {**fewer, "support_pins": {"uk_system": {"size_bytes": 5555}}},
        )
    assert str(figures.value) == str(sidecar.value)
    with pytest.raises(ValueError) as everything:
        require_geography_first_spine([], {}, {})
    assert "geography key definition_sha256" in str(everything.value)
    for refused in (sidecar, request, everything):
        text = str(refused.value).replace("microcosm#931", "").replace("sha256", "")
        assert not any(character.isdigit() for character in text)
