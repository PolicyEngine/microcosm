"""The US plan declaration: complete or nothing, every donor cited."""

from __future__ import annotations

import pytest

from populace.build.us import US_DONORS, US_STAGE_NAMES, BuildConfig, us_plan


def _noop_implementations() -> dict:
    return {name: (lambda frame: frame) for name in US_STAGE_NAMES}


class TestUsPlan:
    def test_assembles_with_all_stages_and_donor_citations(self) -> None:
        plan = us_plan(_noop_implementations())
        assert tuple(stage.name for stage in plan.stages) == US_STAGE_NAMES
        donor_stages = dict(plan.donors())
        # every declared donor is attached to its stage
        assert set(donor_stages) == set(US_DONORS)
        for spec in donor_stages.values():
            assert spec.source.startswith("https://")

    def test_missing_stage_refuses_to_assemble(self) -> None:
        implementations = _noop_implementations()
        del implementations["org_wages"]
        with pytest.raises(ValueError, match="missing \\['org_wages'\\]"):
            us_plan(implementations)

    def test_unknown_stage_is_refused(self) -> None:
        implementations = _noop_implementations()
        implementations["org_wages_fallback"] = lambda frame: frame
        with pytest.raises(ValueError, match="Unknown stage implementation"):
            us_plan(implementations)

    def test_no_ecps_anywhere_in_the_donor_graph(self) -> None:
        """The enhanced CPS is the benchmark, never a build input."""
        for spec in US_DONORS.values():
            text = f"{spec.survey} {spec.source} {spec.notes}".lower()
            assert "enhanced cps" not in text
            assert "ecps" not in text


class TestBuildConfig:
    def test_manifest_round_trip(self) -> None:
        config = BuildConfig(
            year=2024,
            seed=0,
            max_weight_ratio=50.0,
            registry_path="registry/us-2024.json",
            extra={"scf_vintage": 2022},
        )
        manifest = config.to_manifest()
        assert manifest["max_weight_ratio"] == 50.0
        assert manifest["registry_path"] == "registry/us-2024.json"
        assert manifest["extra"] == {"scf_vintage": 2022}

    def test_bad_knobs_refused(self) -> None:
        with pytest.raises(ValueError, match="max_weight_ratio"):
            BuildConfig(year=2024, max_weight_ratio=0.0)
        with pytest.raises(ValueError, match="mass"):
            BuildConfig(year=2024, mass="leaky")
        with pytest.raises(ValueError, match="survey year"):
            BuildConfig(year=1900)


class TestUsSources:
    def test_sources_module_imports_without_donor_deps(self) -> None:
        """Donor loaders import lazily: the module itself must import on a
        base install; each stage fails loudly only when actually called
        without its donor available."""
        from populace.build.us import sources

        for stage in (
            "add_scf_wealth",
            "add_sipp_tips",
            "add_org_wages",
            "add_meps_esi_premiums",
            "add_prior_year_income",
            "add_mortgage_conversion",
            "add_acs_rent",
            "add_vehicle_assets",
        ):
            assert callable(getattr(sources, stage))
