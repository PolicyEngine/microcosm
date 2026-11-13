"""The UK country adapter uses raw FRS sources and the canonical full graph."""

import shutil

import pytest

from microcosm.build.country_spec import load_country_spec
from microcosm.build.uk_runtime.country_adapter import (
    UK_SPINE_SUPPORT_REGISTER,
    UK_SYNTHETIC_SUPPORT_REGISTER,
    uk_atomic_support_register,
    uk_spine_geography_binding,
    uk_spine_geography_binding_failures,
    validate_uk_country_source_projection,
)
from test_support.paths import paths_for


def test_country_raw_source_projection_is_current():
    spec = load_country_spec("uk")
    validate_uk_country_source_projection(spec)
    sources = spec.resolved_spec.resource("sources").domain.to_wire()["sources"]
    frs = [row for row in sources if row["role"] == "frs_raw_table"]
    assert all(row["loader"] == "kernel:build_uk_frs_spine" for row in frs)
    assert {"frs_adult", "frs_benefits", "frs_child", "frs_househol"} <= {
        row["id"] for row in frs
    }
    # The three atomic-area supports (microcosm#932) and their conditioned
    # forms (microcosm#931) sit beside the FRS tables.
    supports = {
        row["id"]: row["role"] for row in sources if row["role"] != "frs_raw_table"
    }
    systems = (
        "uk_ew_output_area_2021",
        "uk_scotland_output_area_2022",
        "uk_ni_data_zone_2021",
    )
    assert supports == {
        **{f"{system}_support": "uk_atomic_area_support" for system in systems},
        **{
            f"{system}_conditioned_support": "uk_atomic_area_conditioned_support"
            for system in systems
        },
    }
    assert all(
        row["loader"] == "kernel:load_uk_atomic_area_support"
        for row in sources
        if row["id"] in supports
    )
    assert "uk_national_candidate_2023" not in str(sources)
    spine = spec.resolved_spec.resource("spine").domain.to_wire()
    assert spine["channels"][0]["observed_constraints"] == ["imd_decile"]


def _copied_package(tmp_path):
    source = paths_for("microcosm-build").package / "src/microcosm/build/uk"
    destination = tmp_path / "uk"
    shutil.copytree(source, destination)
    return destination


def test_country_source_projection_refuses_divergent_header_pin(tmp_path):
    destination = _copied_package(tmp_path)
    path = destination / "spec/sources.yaml"
    text = path.read_text()
    first_sha = text.split("  sha256: ", 1)[1].splitlines()[0]
    path.write_text(text.replace(first_sha, "f" * 64, 1))
    spec = load_country_spec(destination)
    with pytest.raises(ValueError, match="raw-source pins differ"):
        validate_uk_country_source_projection(spec)


def test_country_source_projection_refuses_a_spine_without_the_decile(tmp_path):
    destination = _copied_package(tmp_path)
    path = destination / "spec/spine.yaml"
    text = path.read_text()
    assert text.count("  observed_constraints:\n  - imd_decile\n") == 1
    path.write_text(text.replace("  observed_constraints:\n  - imd_decile\n", ""))
    with pytest.raises(ValueError, match="observe the region and the IMD decile"):
        validate_uk_country_source_projection(load_country_spec(destination))


def test_country_source_projection_refuses_a_support_without_its_conditioned_form(
    tmp_path,
):
    destination = _copied_package(tmp_path)
    path = destination / "spec/sources.yaml"
    text = path.read_text()
    renamed = text.replace(
        "id: uk_ni_data_zone_2021_conditioned_support",
        "id: uk_ni_data_zone_2021_other_support",
    )
    assert renamed != text
    path.write_text(renamed)
    vintages = destination / "spec/vintages.yaml"
    vintages.write_text(
        vintages.read_text().replace(
            "source:uk_ni_data_zone_2021_conditioned_support",
            "source:uk_ni_data_zone_2021_other_support",
        )
    )
    with pytest.raises(ValueError, match="each system's support and its conditioned"):
        validate_uk_country_source_projection(load_country_spec(destination))


def test_conditioned_support_register_is_the_committed_provenance():
    import json

    from microcosm.build.uk_runtime.atomic_area_support import SYSTEMS

    register = uk_atomic_support_register(conditioned=True)
    assert tuple(register) == SYSTEMS
    package = paths_for("microcosm-build").package / "src/microcosm/build/uk"
    provenance = json.loads(
        (package / "uk_atomic_area_conditioned_supports.provenance.json").read_text()
    )
    base = uk_atomic_support_register()
    for system, row in register.items():
        entry = provenance["supports"][system]
        assert row == {"sha256": entry["sha256"], "size_bytes": entry["size_bytes"]}
        assert entry["base"]["sha256"] == base[system]["sha256"]
        assert row["sha256"] != base[system]["sha256"]


def test_atomic_support_register_is_the_committed_provenance():
    """The register a release candidate is checked against (microcosm#932
    round 1) is sources.yaml's, and it agrees with the provenance resource."""
    import json

    from microcosm.build.uk_runtime.atomic_area_support import SYSTEMS

    register = uk_atomic_support_register()
    assert tuple(register) == SYSTEMS
    package = paths_for("microcosm-build").package / "src/microcosm/build/uk"
    provenance = json.loads(
        (package / "uk_atomic_area_supports.provenance.json").read_text()
    )
    for system, row in register.items():
        assert row == {
            "sha256": provenance["supports"][system]["sha256"],
            "size_bytes": provenance["supports"][system]["size_bytes"],
        }


def _release_binding(register: dict, clones: int = 15) -> dict:
    """A spine's geography binding as a release's, with invented digests."""
    return {
        "assignment": "atomic",
        "identity": "geography_household_key",
        "definition_sha256": "a" * 64,
        "support_pins": {system: dict(row) for system, row in register.items()},
        "stream": ["sha256-u53-v1", "uk-spine-atomic-area-v1", 0, 0],
        "register": "sources.yaml",
        "geographic_clone_count": clones,
    }


def _pins_failures() -> tuple[str, ...]:
    from microcosm.build.uk_runtime.atomic_area_support import SYSTEMS

    return tuple(
        f"support_pins.{system} is not the registered conditioned support"
        for system in SYSTEMS
    )


def _edited(binding: dict, **changes) -> dict:
    return {**binding, **changes}


@pytest.mark.parametrize(
    ("edit", "failures"),
    [
        (lambda b: _edited(b, assignment="legacy"), ("assignment is not 'atomic'",)),
        (
            lambda b: _edited(b, identity="household_id"),
            ("identity is not 'geography_household_key'",),
        ),
        (
            lambda b: _edited(b, definition_sha256="A" * 64),
            ("definition_sha256 is not a SHA-256 hex digest",),
        ),
        (
            lambda b: _edited(b, definition_sha256="a" * 63),
            ("definition_sha256 is not a SHA-256 hex digest",),
        ),
        (
            lambda b: _edited(b, support_pins=uk_atomic_support_register()),
            _pins_failures(),
        ),
        (lambda b: _edited(b, support_pins=None), _pins_failures()),
        (
            lambda b: _edited(
                b,
                support_pins={
                    **b["support_pins"],
                    "uk_other_area_2021": {"sha256": "9" * 64, "size_bytes": 1},
                },
            ),
            ("support_pins names a system the register does not",),
        ),
        (
            lambda b: _edited(
                b, stream=["sha256-u53-v1", "uk-spine-atomic-area-v1", 0, 3]
            ),
            ("stream is not the spine's area stream at its declared seed",),
        ),
        (
            lambda b: _edited(
                b, stream=["sha256-u53-v1", "uk-post-clone-atomic-area-v1", 0, 0]
            ),
            ("stream is not the spine's area stream at its declared seed",),
        ),
        (
            lambda b: _edited(
                b, stream=["sha256-u53-v1", "uk-spine-atomic-area-v1", 0, 0.0]
            ),
            ("stream is not the spine's area stream at its declared seed",),
        ),
        (
            lambda b: _edited(
                b, stream=["sha256-u53-v1", "uk-spine-atomic-area-v1", False, 0]
            ),
            ("stream is not the spine's area stream at its declared seed",),
        ),
        (
            lambda b: _edited(
                b, stream=("sha256-u53-v1", "uk-spine-atomic-area-v1", 0, 0)
            ),
            ("stream is not the spine's area stream at its declared seed",),
        ),
        (
            lambda b: _edited(b, register="synthetic"),
            ("register is not 'sources.yaml'",),
        ),
        (
            lambda b: _edited(b, geographic_clone_count=7),
            ("geographic_clone_count is not K = 15",),
        ),
        (
            lambda b: _edited(b, geographic_clone_count=15.0),
            ("geographic_clone_count is not K = 15",),
        ),
        (
            lambda b: _edited(
                b,
                support_pins={
                    system: {**pin, "size_bytes": float(pin["size_bytes"])}
                    for system, pin in b["support_pins"].items()
                },
            ),
            _pins_failures(),
        ),
        (
            lambda b: None,
            (
                "assignment is not 'atomic'",
                "identity is not 'geography_household_key'",
                "definition_sha256 is not a SHA-256 hex digest",
                *_pins_failures(),
                "stream is not the spine's area stream at its declared seed",
                "register is not 'sources.yaml'",
                "geographic_clone_count is not K = 15",
            ),
        ),
    ],
    ids=[
        "law",
        "identity",
        "digest case",
        "digest length",
        "base register",
        "no pins",
        "extra system",
        "seed",
        "stream",
        "float seed",
        "boolean offset",
        "tuple stream",
        "synthetic register",
        "K",
        "float K",
        "float sizes",
        "no binding",
    ],
)
def test_a_spine_geography_binding_is_held_to_a_releases(edit, failures):
    """A release's areas are drawn by the atomic law on each clone's
    geography key, under a definition its SHA-256 names, on exactly the
    conditioned supports sources.yaml registers, on the spine's area stream
    at its declared seed, with the pins held to sources.yaml's register, at
    the K asked for (microcosm#931). Each failure names its field and what it
    must be, never the value it holds."""
    register = uk_atomic_support_register(conditioned=True)
    binding = _release_binding(register)
    assert uk_spine_geography_binding_failures(binding, clone_count=15) == ()
    found = uk_spine_geography_binding_failures(edit(binding), clone_count=15)
    assert found == failures
    assert not any(
        value in line
        for line in found
        for value in ("legacy", "household_id'", "A" * 8, "9" * 8, "uk_other")
    )


def test_a_binding_without_a_clone_count_check_reads_no_k():
    register = uk_atomic_support_register(conditioned=True)
    binding = _release_binding(register, clones=7)
    assert uk_spine_geography_binding_failures(binding) == ()
    assert uk_spine_geography_binding_failures(binding, clone_count=7) == ()
    # A boolean is no clone count, however it compares.
    assert uk_spine_geography_binding_failures(
        _edited(binding, geographic_clone_count=True), clone_count=1
    ) == ("geographic_clone_count is not K = 1",)


def test_a_spine_geography_block_is_composed_under_the_leafs_keys():
    """The block the spine build's sidecar writer and a spine request's dense
    build both compose (microcosm#931): the atomic law on the geography key,
    the digest of the conditioned definition the area draw builds on the
    supports at its declared seed, each support's SHA-256 and size, that
    definition's stream with the seed at index 3, and the register, under the
    roster leaf's keys in their order. On toy supports registered in place of
    sources.yaml's it is a release's binding; under the synthetic register it
    fails on that field alone."""
    import hashlib
    import json

    from microcosm.build.uk_runtime.atomic_area_conditioning import (
        UK_SPINE_ATOMIC_AREA_SEED,
        uk_conditioned_assignment_definition,
    )
    from microcosm.build.uk_runtime.atomic_area_support import SYSTEMS
    from microcosm.build.uk_runtime.geography_first_roster import (
        SIDECAR_GEOGRAPHY_KEYS,
    )
    from microcosm.graph.canonical import canonical_json
    from test_support.microcosm_build.uk_conditioned_support_fixtures import (
        toy_conditioned_support_payloads,
    )

    payloads = toy_conditioned_support_payloads()
    block = uk_spine_geography_binding(payloads, register=UK_SPINE_SUPPORT_REGISTER)
    assert tuple(block) == SIDECAR_GEOGRAPHY_KEYS
    definition = uk_conditioned_assignment_definition(
        payloads, seed=UK_SPINE_ATOMIC_AREA_SEED
    )
    pins = {
        system: {
            "sha256": hashlib.sha256(payloads[system]).hexdigest(),
            "size_bytes": len(payloads[system]),
        }
        for system in SYSTEMS
    }
    assert block == {
        "assignment": "atomic",
        "identity": "geography_household_key",
        "definition_sha256": hashlib.sha256(canonical_json(definition)).hexdigest(),
        "support_pins": pins,
        "stream": ["sha256-u53-v1", "uk-spine-atomic-area-v1", 0, 0],
        "register": "sources.yaml",
    }
    assert block["stream"][3] == UK_SPINE_ATOMIC_AREA_SEED
    assert all(type(pin["size_bytes"]) is int for pin in block["support_pins"].values())
    # Through JSON, as a sidecar records it, against the toys registered in
    # place of sources.yaml's supports.
    recorded = json.loads(json.dumps(block))
    assert uk_spine_geography_binding_failures(recorded, spec=_registered(pins)) == ()
    synthetic = uk_spine_geography_binding(
        payloads, register=UK_SYNTHETIC_SUPPORT_REGISTER
    )
    assert {key: value for key, value in synthetic.items() if key != "register"} == {
        key: value for key, value in block.items() if key != "register"
    }
    assert uk_spine_geography_binding_failures(synthetic, spec=_registered(pins)) == (
        "register is not 'sources.yaml'",
    )
    with pytest.raises(ValueError, match="names the register its support pins"):
        uk_spine_geography_binding(payloads, register="sources")


def _registered(pins: dict):
    """A stand-in spec whose conditioned register holds ``pins``."""
    from types import SimpleNamespace

    from microcosm.build.uk_runtime.atomic_area_support import CONDITIONED_SOURCES

    rows = [
        {
            "id": CONDITIONED_SOURCES[system],
            "role": "uk_atomic_area_conditioned_support",
            "sha256": pin["sha256"],
            "byte_size": pin["size_bytes"],
        }
        for system, pin in pins.items()
    ]
    wire = SimpleNamespace(to_wire=lambda: {"sources": rows})
    resource = SimpleNamespace(domain=wire)
    return SimpleNamespace(
        resolved_spec=SimpleNamespace(resource=lambda name: resource)
    )
