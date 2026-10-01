"""Axiom concept mappings against the committed, engine-generated surfaces.

CI has no Axiom engine. The input surfaces under
``tests/fixtures/axiom_input_surfaces/`` were compiled by a real engine build from
pinned rulespec commits, so every Axiom binding is checked against them here:
the module, engine entity and input name must exist, and the binding's
canonical request name must be the engine's.
"""

import json

import pytest

from microcosm.frame.adapters.axiom import AxiomEngine, axiom_concept_mapping
from microcosm.frame.adapters.axiom_input_surface import (
    AXIOM_INPUT_SURFACE_FORMAT,
    AxiomInputSurface,
)
from microcosm.frame.concept_mapping import (
    InputDeclaration,
    coverage_report,
)
from microcosm.frame.concepts import (
    CanonicalConceptKind,
    canonical_concept_kind,
)
from test_support.microcosm_frame.concept_mappings import (
    AXIOM_COUNTRIES,
    axiom_input_surface,
    coverage_golden,
)

COUNTRIES = AXIOM_COUNTRIES
load_axiom_input_surface = axiom_input_surface


@pytest.mark.parametrize("country", COUNTRIES)
def test_surfaces_load_with_provenance(country) -> None:
    surface = load_axiom_input_surface(country)
    assert surface.country == country
    assert len(surface.rulespec_commit) == 40
    assert len(surface.engine_commit) == 40
    assert surface.modules
    refs = surface.refs()
    assert list(refs) == sorted(set(refs))
    for module in surface.modules.values():
        assert module.path.split("/", 1)[0].split("-", 1)[0] == country
        assert len(module.sha256) == 64
        assert module.status in ("compiled", "compile_failed", "dense_unsupported")
        if module.status == "compile_failed":
            assert not module.inputs
        assert all(module.canonical_inputs.get(ref.name) for ref in module.refs())


def test_an_unknown_surface_format_is_refused() -> None:
    with pytest.raises(ValueError, match=AXIOM_INPUT_SURFACE_FORMAT):
        AxiomInputSurface.from_dict({"format": "other"})


@pytest.mark.parametrize("country", COUNTRIES)
def test_mapping_records_the_untyped_input_gap(country) -> None:
    mapping = axiom_concept_mapping(country)
    assert mapping.engine == f"axiom:{country}"
    assert mapping.input_declaration is InputDeclaration.USAGE_INFERRED
    assert dict(mapping.entity_correspondence) == {
        "person": "Person",
        "household": "Household",
    }
    assert mapping.engine_version == load_axiom_input_surface(country).rulespec_commit


@pytest.mark.parametrize("country", COUNTRIES)
def test_every_binding_exists_in_the_engine_surface(country) -> None:
    mapping = axiom_concept_mapping(country)
    surface = load_axiom_input_surface(country)
    refs = set(surface.refs())
    for binding in mapping.bindings:
        assert binding.ref in refs, binding.ref.label()
        assert binding.canonical_input == surface.canonical_input(binding.ref)
        assert canonical_concept_kind(binding.canonical_input) is (
            CanonicalConceptKind.LEGAL
        )


@pytest.mark.parametrize("country", COUNTRIES)
def test_committed_coverage_report_matches_the_surface(country) -> None:
    mapping = axiom_concept_mapping(country)
    report = coverage_report(mapping, load_axiom_input_surface(country).refs())
    assert report.unknown_inputs == ()
    committed = json.loads(
        coverage_golden(f"axiom-{country}").read_text(encoding="utf-8")
    )
    assert committed == json.loads(json.dumps(report.to_dict())), (
        "Regenerate with: uv run --no-sync python tools/refresh_concept_coverage.py "
        f"--engine axiom-{country}"
    )


@pytest.mark.parametrize("country", COUNTRIES)
def test_the_adapter_finds_its_country_mapping(country, tmp_path) -> None:
    root = tmp_path / f"rulespec-{country}"
    module = root / country / "statutes" / "x.yaml"
    adapter = AxiomEngine(module, rulespec_roots=(root,))
    assert adapter.rulespec_country() == country
    assert adapter.concept_mapping() is axiom_concept_mapping(country)


def test_the_country_comes_from_the_module_tree_not_the_root_name(tmp_path) -> None:
    root = tmp_path / "rulespec-nz-worktree"
    adapter = AxiomEngine(root / "nz" / "statutes" / "x.yaml", rulespec_roots=(root,))
    assert adapter.rulespec_country() == "nz"
    brussels = AxiomEngine(root / "be-bru" / "x.yaml", rulespec_roots=(root,))
    assert brussels.rulespec_country() == "be"
    loose = AxiomEngine(root / "x.yaml", rulespec_roots=(root,))
    with pytest.raises(ValueError, match="country tree"):
        loose.rulespec_country()
    with pytest.raises(ValueError, match="No RuleSpec root"):
        AxiomEngine(
            tmp_path / "elsewhere.yaml", rulespec_roots=(root,)
        ).rulespec_country()
