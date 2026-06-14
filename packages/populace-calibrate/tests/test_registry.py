"""The target registry: declared facts with provenance, versioned by content.

Pins the structural promises: every fact carries a citation, negative values
must be declared (the net-STCG sign guard), the version is a content hash
(same facts -> same id, any edit -> a new id), the JSON artifact round-trips
and detects hand-edits, registries compile into solver TargetSets that a real
calibration consumes, and surface ingestion refuses families without sources.
"""

from __future__ import annotations

import json

import pytest

from populace.calibrate import (
    TargetRegistry,
    TargetSpec,
    calibrate,
    specs_from_pe_surface,
)


def _spec(**overrides) -> TargetSpec:
    base = dict(
        name="census/population",
        entity="household",
        aggregation="count",
        value=1000.0,
        source="Census ACS 2024 table B01003",
        family="census",
    )
    base.update(overrides)
    return TargetSpec(**base)


class TestTargetSpec:
    def test_source_is_required(self) -> None:
        with pytest.raises(ValueError, match="source is required"):
            _spec(source="")

    def test_negative_value_requires_signed(self) -> None:
        with pytest.raises(ValueError, match="requires signed=True"):
            _spec(value=-76.8e9)

    def test_signed_negative_value_is_accepted(self) -> None:
        spec = _spec(
            name="puf/net_short_term_capital_gains",
            value=-76.8e9,
            signed=True,
        )
        assert spec.value < 0

    def test_callable_measure_is_refused(self) -> None:
        with pytest.raises(TypeError, match="callables do not serialize"):
            _spec(measure=lambda frame: frame)

    def test_positive_se_only(self) -> None:
        with pytest.raises(ValueError, match="se must be positive"):
            _spec(se=0.0)

    def test_to_target_compiles(self) -> None:
        target = _spec(
            tolerance=5.0,
            metadata={"kind": "neutralize_variable", "output_variable": "income_tax"},
        ).to_target()
        assert target.name == "census/population"
        assert target.value == 1000.0
        assert target.tolerance == 5.0
        assert target.source == "Census ACS 2024 table B01003"
        assert target.metadata == {
            "kind": "neutralize_variable",
            "output_variable": "income_tax",
        }

    def test_metadata_needs_non_empty_keys_and_values(self) -> None:
        with pytest.raises(ValueError, match="metadata"):
            _spec(metadata={"": "neutralize_variable"})
        with pytest.raises(ValueError, match="metadata"):
            _spec(metadata={"kind": ""})


class TestRegistryIdentity:
    def test_duplicate_keys_rejected(self) -> None:
        with pytest.raises(ValueError, match="Duplicate target key"):
            TargetRegistry([_spec(), _spec()], country="us")

    def test_same_facts_same_version(self) -> None:
        a = TargetRegistry([_spec()], country="us")
        b = TargetRegistry([_spec()], country="us")
        assert a.version == b.version

    def test_any_edit_changes_version(self) -> None:
        a = TargetRegistry([_spec()], country="us")
        b = TargetRegistry([_spec(value=1001.0)], country="us")
        c = TargetRegistry([_spec(source="another table")], country="us")
        assert len({a.version, b.version, c.version}) == 3

    def test_families_and_select(self) -> None:
        registry = TargetRegistry(
            [
                _spec(),
                _spec(name="irs_soi/agi", family="irs_soi", value=5e12),
            ],
            country="us",
        )
        assert registry.families() == ("census", "irs_soi")
        soi = registry.select(family="irs_soi")
        assert len(soi) == 1 and next(iter(soi)).name == "irs_soi/agi"


class TestArtifactRoundTrip:
    def test_round_trip_preserves_everything(self, tmp_path) -> None:
        registry = TargetRegistry(
            [
                _spec(
                    se=12.5,
                    notes="ACS 1-year",
                    metadata={"kind": "neutralize_variable"},
                ),
                _spec(
                    name="puf/net_short_term_capital_gains",
                    value=-76.8e9,
                    signed=True,
                    family="irs_soi",
                    source="IRS SOI Table 1.4, uprated",
                ),
            ],
            country="us",
        )
        path = registry.to_json(tmp_path / "registry.json")
        loaded = TargetRegistry.from_json(path)
        assert loaded.version == registry.version
        assert loaded.specs == registry.specs
        assert loaded.country == "us"

    def test_hand_edit_is_detected(self, tmp_path) -> None:
        registry = TargetRegistry([_spec()], country="us")
        path = registry.to_json(tmp_path / "registry.json")
        payload = json.loads(path.read_text())
        payload["specs"][0]["value"] = 999999.0  # silent steering attempt
        path.write_text(json.dumps(payload))
        with pytest.raises(ValueError, match="corrupt"):
            TargetRegistry.from_json(path)

    def test_non_registry_file_is_refused(self, tmp_path) -> None:
        path = tmp_path / "other.json"
        path.write_text(json.dumps({"hello": 1}))
        with pytest.raises(ValueError, match="not a populace target registry"):
            TargetRegistry.from_json(path)


class TestEndToEnd:
    def test_registry_drives_a_real_calibration(self, feasible_frame) -> None:
        frame, truths = feasible_frame(n=80)
        registry = TargetRegistry(
            [
                _spec(value=truths["population"] * 1.2),
                TargetSpec(
                    name="acs/income",
                    entity="household",
                    aggregation="sum",
                    measure="income",
                    value=truths["income"] * 1.2,
                    source="ACS B19025",
                    family="acs",
                ),
            ],
            country="us",
        )
        result = calibrate(frame, registry.to_target_set(), epochs=300, seed=0)
        assert result.final_loss < result.initial_loss * 1e-2
        for diag in result.diagnostics:
            assert abs(diag.relative_error) < 0.02


class TestSurfaceIngest:
    def test_parses_families_and_signs(self) -> None:
        specs = specs_from_pe_surface(
            ["census/population", "puf/net_short_term_capital_gains"],
            [334e6, -76.8e9],
            family_sources={
                "census": "Census ACS",
                "puf": "IRS SOI / PUF uprated",
            },
            signed_names=["puf/net_short_term_capital_gains"],
        )
        assert specs[0].family == "census"
        assert specs[1].signed and specs[1].value < 0

    def test_unknown_family_is_refused(self) -> None:
        with pytest.raises(ValueError, match="no source in"):
            specs_from_pe_surface(
                ["mystery/thing"], [1.0], family_sources={"census": "x"}
            )

    def test_unprefixed_name_is_refused(self) -> None:
        with pytest.raises(ValueError, match="no 'family/...' prefix"):
            specs_from_pe_surface(["thing"], [1.0], family_sources={})

    def test_misaligned_lengths_refused(self) -> None:
        with pytest.raises(ValueError, match="must align"):
            specs_from_pe_surface(["a/b"], [1.0, 2.0], family_sources={"a": "s"})


class TestVersionStability:
    def test_int_and_float_values_hash_identically(self) -> None:
        """Authoring style must not change the fact's identity."""
        a = TargetRegistry([_spec(value=1000)], country="us")
        b = TargetRegistry([_spec(value=1000.0)], country="us")
        assert a.version == b.version
        c = TargetRegistry([_spec(se=12)], country="us")
        d = TargetRegistry([_spec(se=12.0)], country="us")
        assert c.version == d.version
