"""The UK country adapter uses raw FRS sources and the canonical full graph."""

import shutil

import pytest

from microcosm.build.country_spec import load_country_spec
from microcosm.build.uk_runtime.country_adapter import (
    uk_atomic_support_register,
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
        "K",
        "float K",
        "float sizes",
        "no binding",
    ],
)
def test_a_spine_geography_binding_is_held_to_a_releases(edit, failures):
    """A release's areas are drawn by the atomic law on each clone's
    geography key, under a definition its SHA-256 names, on exactly the
    conditioned supports sources.yaml registers, at the K asked for
    (microcosm#931). Each failure names its field and what it must be, never
    the value it holds."""
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
