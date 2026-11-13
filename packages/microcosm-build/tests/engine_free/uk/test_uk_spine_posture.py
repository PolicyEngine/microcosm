"""The UK lines' spine posture (microcosm#931): the geography-first names and
the roster check that refuses a spine lacking any of them, and the national
line's own checks: a geography-first spine of one geographic clone, and the
area-basis exclusion signed for the national line and in force on the build's
review date.

Every roster here is invented, read off the packaged spec, or written by the
spine build's own sidecar writer over a synthetic frame; no sidecar of a built
spine is read. Every register is a test register with invented dates, but for
one test of the committed register as it stands, on a fixed date inside its
entry's window; no test reads the wall clock.
"""

import ast
import inspect
import json
from datetime import date, datetime, timedelta
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
    NATIONAL_AREA_EXCLUSION,
    NATIONAL_CLONE_COUNT,
    NATIONAL_LINE,
    require_geography_first_spine,
    require_national_exclusion_signed,
    require_national_spine,
    roster_failures,
    spec_roster,
    stage_roster,
)
from microcosm.build.uk_runtime.weighted_integrity import (
    UK_DEGENERATE_EXCLUSION_REGISTER_RESOURCE,
    UK_RELEASE_LINES,
    load_uk_reviewed_exclusion_register,
)
from test_support.microcosm_build.uk_full_build_cli import (
    NATIONAL_EXCLUSION_WINDOW,
    geography_first_sidecar_fields,
    synthetic_spine_geography,
    write_national_exclusion_register,
)

#: What a geography-first spine runs that a spine built from the packaged spec
#: does not: nothing. The area stages and the main-residence localiser inside
#: the WAS chain are declared (microcosm#931), so the dense line builds on a
#: packaged spine whose sidecar records its geography block.
_PACKAGED_ROSTER_LACKS: tuple[str, ...] = ()


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


def _written_sidecar(monkeypatch, spec, geography):
    """The sidecar the spine build's own writer records over a synthetic frame
    and the packaged stages in the spine graph's order, with ``geography``,
    read back through JSON as the dense preparation reads it. Only the
    engine's provenance stands in."""
    from microcosm.build.uk_runtime import spine_build
    from test_support.microcosm_build.uk_full_population_graph import source_frame

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
        geography=geography,
    )
    return json.loads(json.dumps(written))


def _toy_block(register: str = "synthetic") -> dict:
    """The block the spine build composes for toy conditioned supports."""
    from microcosm.build.uk_runtime.country_adapter import uk_spine_geography_binding
    from test_support.microcosm_build.uk_conditioned_support_fixtures import (
        toy_conditioned_support_payloads,
    )

    return uk_spine_geography_binding(
        toy_conditioned_support_payloads(), register=register
    )


def test_the_packaged_roster_is_geography_first(monkeypatch):
    """A spine built on this code runs every area stage and the
    main-residence localiser inside the WAS chain, and the sidecar the spine
    build's own writer records for it carries the geography block: the dense
    line accepts it (microcosm#931). A spine whose writer records no block
    still fails, on that alone."""
    spec = load_country_spec("uk")
    stages, operations = spec_roster(spec)
    assert set(GEOGRAPHY_FIRST_STAGES) | {"was_wealth"} <= set(stages)
    assert "localise_main_residence_value" in operations["was_wealth"]
    assert roster_failures(stages, operations, from_sidecar=False) == (
        _PACKAGED_ROSTER_LACKS
    )
    require_geography_first_spine(stages, operations, from_sidecar=False)
    sidecar = _written_sidecar(monkeypatch, spec, _toy_block())
    recorded = (sidecar["stages"], sidecar["operations"], sidecar.get("geography"))
    assert set(sidecar["stages"]) == set(stages)
    assert roster_failures(*recorded) == _PACKAGED_ROSTER_LACKS
    require_geography_first_spine(*recorded)
    # A spine whose writer records no block fails on that alone.
    bare = _written_sidecar(monkeypatch, spec, None)
    assert "geography" not in bare
    assert roster_failures(bare["stages"], bare["operations"], None) == (
        *_PACKAGED_ROSTER_LACKS,
        "geography block",
    )
    with pytest.raises(ValueError, match="lacks: geography block[.]$"):
        require_geography_first_spine(bare["stages"], bare["operations"], None)


def test_every_stage_the_leaf_names_is_declared_in_its_order():
    """The packaged spec declares every geography-first stage, in the leaf's
    order, and every stage a geography-first operation runs in (microcosm#931)."""
    spec = load_country_spec("uk")
    stages, _ = spec_roster(spec)
    assert [name for name in stages if name in GEOGRAPHY_FIRST_STAGES] == list(
        GEOGRAPHY_FIRST_STAGES
    )
    assert set(GEOGRAPHY_FIRST_OPERATIONS) <= set(stages)


def test_the_sidecar_writer_writes_the_block_under_the_leafs_keys(monkeypatch):
    """The spine build's sidecar writer records the block it is handed under
    the leaf's keys, in their order, ints as ints, through JSON
    (microcosm#931)."""
    spec = load_country_spec("uk")
    block = _toy_block(register="sources.yaml")
    sidecar = _written_sidecar(monkeypatch, spec, block)
    assert tuple(sidecar["geography"]) == SIDECAR_GEOGRAPHY_KEYS
    assert sidecar["geography"] == block
    assert type(sidecar["geography"]["stream"][3]) is int
    assert all(
        type(pin["size_bytes"]) is int
        for pin in sidecar["geography"]["support_pins"].values()
    )


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


# The national line (microcosm#931): a geography-first spine of one geographic
# clone, and the area-basis exclusion signed for the national line.

_NATIONAL_LEAD = (
    "The national line builds only on a geography-first spine of one "
    "geographic clone, one that assigns every household its area "
    "(microcosm#931); "
)


def _national_sidecar(**fields) -> dict:
    """A stand-in build sidecar the national line builds on: the stand-in
    geography-first roster and block at one clone, with ``fields`` replacing
    any of its fields."""
    return {
        **geography_first_sidecar_fields("frs_spine"),
        "geographic_clone_count": 1,
        **fields,
    }


def test_the_national_names_are_the_lines_the_clone_count_and_the_entry():
    assert (NATIONAL_LINE, NATIONAL_CLONE_COUNT, NATIONAL_AREA_EXCLUSION) == (
        "national",
        1,
        "household.atomic_area_basis",
    )
    assert NATIONAL_LINE in UK_RELEASE_LINES


def test_the_national_line_builds_on_a_geography_first_spine_of_one_clone():
    assert require_national_spine(_national_sidecar()) is None
    # A sidecar that records no clone count describes a spine of one clone,
    # as the checkpoint authentication reads it.
    unrecorded = _national_sidecar()
    del unrecorded["geographic_clone_count"]
    assert require_national_spine(unrecorded) is None
    # Extra names and fields never fail, and read-only forms read alike.
    stages = _national_sidecar()["stages"]
    assert (
        require_national_spine(
            _national_sidecar(stages=[*stages, "nts_bus_travel"], sampling=None)
        )
        is None
    )
    assert require_national_spine(MappingProxyType(_national_sidecar())) is None


def test_the_national_line_refuses_a_spine_of_another_clone_count():
    """K is the int one: any other count, or a value that is not an int (a
    bool among them), is refused in one sentence naming K and nothing else."""
    for clones in (0, 2, 3, 15):
        with pytest.raises(ValueError) as refused:
            require_national_spine(_national_sidecar(geographic_clone_count=clones))
        assert str(refused.value) == (
            _NATIONAL_LEAD
            + f"the bound spine's geographic clone count is K = {clones}."
        )
    for clones in (True, "1", 1.0, None):
        with pytest.raises(ValueError) as refused:
            require_national_spine(_national_sidecar(geographic_clone_count=clones))
        assert str(refused.value) == (
            _NATIONAL_LEAD
            + f"the bound spine's geographic clone count is K = {clones!r}."
        )


def test_the_national_line_refuses_a_roster_that_is_not_geography_first():
    """A one-clone spine built before the area stages (the roster group (b)
    left, with no geography block) is refused, naming what its roster lacks
    in the leaf's order; K, where it is wrong too, comes first."""
    pre_areas = ["frs_spine", "geographic_expand", "geographic_identity", "was_wealth"]
    operations = {name: [] for name in pre_areas}
    lacking = "; ".join(
        (
            *(f"stage {stage}" for stage in GEOGRAPHY_FIRST_STAGES[2:]),
            "operation localise_main_residence_value in was_wealth",
            "geography block",
        )
    )
    with pytest.raises(ValueError) as refused:
        require_national_spine(
            _national_sidecar(stages=pre_areas, operations=operations, geography=None)
        )
    assert str(refused.value) == (
        _NATIONAL_LEAD + f"the bound spine's build sidecar lacks: {lacking}."
    )
    without_block = _national_sidecar(geographic_clone_count=3)
    del without_block["geography"]
    with pytest.raises(ValueError) as both:
        require_national_spine(without_block)
    assert str(both.value) == (
        _NATIONAL_LEAD + "the bound spine's geographic clone count is K = 3; "
        "the bound spine's build sidecar lacks: geography block."
    )
    for sidecar in (None, [], "sidecar", 1):
        with pytest.raises(ValueError, match="build sidecar is a JSON object"):
            require_national_spine(sidecar)


def test_the_national_spine_refusal_names_k_and_names_only():
    """What else the sidecar holds, figures among it, never enters the
    refusal: it names K and the leaf's missing names alone."""
    sidecar = _national_sidecar(
        geographic_clone_count=2,
        stages=[
            name
            for name in _national_sidecar()["stages"]
            if name not in {"rfdr", "private_rent"}
        ],
        entity_row_counts={"household": 4321},
        household_weight_total=98765.0,
    )
    sidecar["geography"]["support_pins"] = {"uk_system": {"size_bytes": 5555}}
    with pytest.raises(ValueError) as refused:
        require_national_spine(sidecar)
    text = str(refused.value)
    assert text == (
        _NATIONAL_LEAD + "the bound spine's geographic clone count is K = 2; "
        "the bound spine's build sidecar lacks: stage rfdr; stage private_rent."
    )
    rest = text.replace("microcosm#931", "").replace("K = 2", "")
    assert not any(character.isdigit() for character in rest)


_WINDOW = tuple(
    date.fromisoformat(NATIONAL_EXCLUSION_WINDOW[name])
    for name in ("approved_on", "expires_on")
)
#: A fixed review date inside the test registers' window.
_REVIEW = date(2027, 1, 15)
_EXCLUSION_LEAD = (
    "The national line builds only once the degenerate-surface exclusion "
    "household.atomic_area_basis is signed for the national line and in force "
    "on the build's review date (microcosm#931); "
)


def test_the_test_registers_review_date_lies_inside_their_window():
    approved, expires = _WINDOW
    assert approved < _REVIEW < expires


def test_an_entry_signed_for_the_national_line_passes_through_its_window(tmp_path):
    """In force from approved_on through expires_on, both days included, on
    the national line; an entry that names no line applies on every one."""
    approved, expires = _WINDOW
    for lines in (("local", "national"), ("national",), None):
        register = write_national_exclusion_register(
            tmp_path / "register.json", release_lines=lines
        )
        for day in (approved, _REVIEW, expires):
            assert require_national_exclusion_signed(day, source=register) is None


def test_an_entry_not_signed_for_the_national_line_is_refused(tmp_path):
    register = write_national_exclusion_register(
        tmp_path / "register.json", release_lines=("local",)
    )
    for day in (*_WINDOW, _REVIEW):
        with pytest.raises(ValueError) as refused:
            require_national_exclusion_signed(day, source=register)
        assert str(refused.value) == (
            _EXCLUSION_LEAD + "its entry, in force from 2026-12-01 through "
            "2027-06-01, does not list the national line."
        )


def test_an_entry_out_of_its_window_is_refused(tmp_path):
    """Expired strictly after expires_on, premature strictly before
    approved_on, as the gates read the register."""
    register = write_national_exclusion_register(tmp_path / "register.json")
    approved, expires = _WINDOW
    for day in (expires + timedelta(days=1), date(2031, 1, 1)):
        with pytest.raises(ValueError) as expired:
            require_national_exclusion_signed(day, source=register)
        assert str(expired.value) == (
            _EXCLUSION_LEAD + "its entry has expired: it was in force from "
            "2026-12-01 through 2027-06-01."
        )
    for day in (approved - timedelta(days=1), date(2026, 1, 1)):
        with pytest.raises(ValueError) as premature:
            require_national_exclusion_signed(day, source=register)
        assert str(premature.value) == (
            _EXCLUSION_LEAD + "its entry is not yet in force: it is in force "
            "from 2026-12-01 through 2027-06-01."
        )
    # A register that does not list the national line refuses on that first,
    # in any window.
    local = write_national_exclusion_register(
        tmp_path / "local.json", release_lines=("local",)
    )
    with pytest.raises(ValueError, match="does not list the national line"):
        require_national_exclusion_signed(expires + timedelta(days=1), source=local)


def test_a_register_without_the_entry_is_refused(tmp_path):
    register = write_national_exclusion_register(tmp_path / "register.json")
    payload = json.loads(register.read_text())
    payload["exclusions"] = {
        "household.another_column": payload["exclusions"][NATIONAL_AREA_EXCLUSION]
    }
    register.write_text(json.dumps(payload))
    with pytest.raises(ValueError) as refused:
        require_national_exclusion_signed(_REVIEW, source=register)
    assert str(refused.value) == _EXCLUSION_LEAD + "the register has no such entry."


def test_the_exclusion_check_takes_the_review_date_and_reads_no_clock(tmp_path):
    """The review date is the build's, a ``date``; a ``datetime``, a string or
    nothing is refused before any register is read."""
    missing = tmp_path / "absent.json"
    for bad in (None, datetime(2027, 1, 15, 12), "2027-01-15"):
        with pytest.raises(TypeError, match="datetime.date"):
            require_national_exclusion_signed(bad, source=missing)
    assert (
        inspect.signature(require_national_exclusion_signed)
        .parameters["review_date"]
        .default
        is inspect.Parameter.empty
    )
    source = inspect.getsource(require_national_exclusion_signed)
    assert "today" not in source and "now(" not in source


def test_the_committed_register_is_not_yet_signed_for_the_national_line():
    """The committed register as it stands, on a fixed date inside its entry's
    window: the entry applies on the local line alone, so the national line
    refuses. Signing the entry for the national line (a register edit and the
    contract re-pin, in their own commit) lifts the refusal with no code
    change, and flips this test."""
    entry = load_uk_reviewed_exclusion_register(
        None, resource=UK_DEGENERATE_EXCLUSION_REGISTER_RESOURCE
    )[NATIONAL_AREA_EXCLUSION]
    day = date(2027, 1, 15)
    assert not entry.premature(day) and not entry.expired(day)
    assert not entry.applies_on(NATIONAL_LINE)
    with pytest.raises(ValueError) as refused:
        require_national_exclusion_signed(day)
    assert str(refused.value) == (
        _EXCLUSION_LEAD + f"its entry, in force from {entry.approved_on} through "
        f"{entry.expires_on}, does not list the national line."
    )


def test_the_exclusion_refusals_name_the_entry_the_line_and_the_window_only(
    tmp_path,
):
    approved, expires = _WINDOW
    refusals = []
    for lines, day in (
        (("local",), _REVIEW),
        (("national",), expires + timedelta(days=1)),
        (("national",), approved - timedelta(days=1)),
    ):
        register = write_national_exclusion_register(
            tmp_path / "register.json", release_lines=lines
        )
        with pytest.raises(ValueError) as refused:
            require_national_exclusion_signed(day, source=register)
        refusals.append(str(refused.value))
    for text in refusals:
        rest = (
            text.replace("microcosm#931", "")
            .replace(NATIONAL_EXCLUSION_WINDOW["approved_on"], "")
            .replace(NATIONAL_EXCLUSION_WINDOW["expires_on"], "")
        )
        assert NATIONAL_AREA_EXCLUSION in rest and "national line" in rest
        assert not any(character.isdigit() for character in rest), text
        # The review date is not the register's, and is not named.
        assert _REVIEW.isoformat() not in text
