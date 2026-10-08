"""Tests split from packages/microcosm-build/tests/test_uk_release_input_coverage_manifest.py."""

# ruff: noqa: F403, F405
from test_support.microcosm_build.uk_release_input_coverage_manifest import *


def test_candidate_evidence_is_sha_pinned_and_covers_reference() -> None:
    reference = _resource("efrs_parity_reference.json")
    gaps = _resource("efrs_parity_known_gaps.json")
    evidence = gaps["candidate_evidence"]
    assert evidence["source"]["sha256"] == (
        "f17306ccb2aad7ff0130be3589b560afb2e2a12a943570911cd0c77f07934833"
    )
    assert evidence["source"]["revision"] == (
        "populace-uk-2023-dd68c73-4aa4b14-20260619T023711Z"
    )
    assert evidence["source"]["tier"] == "frs"
    assert set(evidence["nondefault_shares"]) == set(reference["nonzero_shares"])
    assert set(evidence["effective_nondefault_mass_shares"]) == set(
        reference["nonzero_shares"]
    )
    assert evidence["column_entities"] == reference["input_entities"]
    assert all(float(share) > 0 for share in evidence["nondefault_shares"].values())
    # Nonzero is not the gate criterion for default-True flags or zero-weight
    # support rows. These pin the default-aware extraction rather than a naive
    # nonzero comparison.
    assert evidence["nonzero_shares"]["household_owns_tv"] == 0.951073
    assert evidence["nondefault_shares"]["household_owns_tv"] == 0.048927
    assert evidence["nonzero_shares"]["household_weight"] == 0.626224
    assert evidence["nondefault_shares"]["household_weight"] == 1.0
    assert evidence["nondefault_shares"]["employment_income"] == 0.478282
    assert evidence["effective_signal_columns"] == 143
    assert evidence["insufficient_effective_mass_columns"] == [
        "charitable_investment_gifts",
        "gift_aid",
    ]
    assert evidence["effective_nondefault_mass_shares"]["gift_aid"] == 0.0
    assert (
        evidence["effective_nondefault_mass_shares"]["charitable_investment_gifts"]
        == 0.0
    )
    assert (
        evidence["effective_nondefault_mass_shares"]["employment_income"]
        > evidence["effective_mass_coverage"]["minimum_nondefault_mass_share"]
    )


def test_known_gap_register_records_post_candidate_restoration_separately() -> None:
    gaps = _resource("efrs_parity_known_gaps.json")
    assert gaps["known_gaps"] == {}
    # microcosm#1106: property_wealth is formula-owned in policyengine-uk, so
    # the release leaves it for the engine to derive. It is a declared
    # exclusion in its own section, never parity debt in known_gaps.
    assert set(gaps["engine_derived_exclusions"]) == {"property_wealth"}
    assert gaps["engine_derived_exclusions"]["property_wealth"]["reason"] == (
        "formula-owned in policyengine-uk and derived from persisted "
        "components; a persisted copy overrides the formula and is not uprated"
    )
    assert (
        "microcosm#1106"
        in (gaps["engine_derived_exclusions"]["property_wealth"]["tracking_note"])
    )
    assert set(gaps["restored_required_columns"]) == {
        "charitable_investment_gifts",
        "gift_aid",
    }
    assert gaps["candidate_evidence"]["missing_columns"] == []
    assert gaps["candidate_evidence"]["default_only_columns"] == []
    assert gaps["candidate_evidence"]["signal_columns"] == 145
    assert gaps["candidate_evidence"]["effective_signal_columns"] == 143
    assert gaps["exclusion_policy"]["reason"] == (
        "not yet ported from enhanced FRS pipeline — pending review"
    )
    assert gaps["exclusion_policy"]["tracking_note"].strip()
    for name, evidence in gaps["restored_required_columns"].items():
        assert evidence["stage"] == "hmrc_spi_income"
        assert evidence["current_producer_stage"] == "hmrc_spi_income_spine"
        assert evidence["support_channel"] == "spi"
        assert (
            evidence["effective_signal_mass_share"]
            >= (evidence["minimum_nondefault_mass_share"])
        ), name


def test_committed_manifest_matches_regeneration() -> None:
    generator = _load_generator()
    committed = _resource("release_input_coverage_manifest.json")
    assert generator.build_manifest() == committed


def test_frozen_candidate_retains_its_original_engine_provenance() -> None:
    evidence = _resource("efrs_parity_known_gaps.json")["candidate_evidence"]
    assert evidence["engine"]["version"] == "2.89.0"


def test_hmrc_family_period_fields_come_from_the_bytes_their_hash_names() -> None:
    import hashlib

    family = _resource("release_input_coverage_manifest.json")["family_coverage"][
        "hmrc_spi_income"
    ]
    canonical_bytes = files(_UK_PACKAGE).joinpath("source_stages.json").read_bytes()
    assert family["source_manifest"] == "source_stages.json"
    assert (
        family["source_manifest_sha256"] == hashlib.sha256(canonical_bytes).hexdigest()
    )
    stage = next(
        stage
        for stage in json.loads(canonical_bytes)["stages"]
        if stage["stage"] == "hmrc_spi_income_spine"
    )
    surface = next(
        artifact
        for artifact in stage["artifacts"]
        if artifact["role"] == "published_fact_surface"
    )
    assert family["source_vintages"]["mapped_build_period"] == str(
        surface["mapped_build_period"]
    )
    assert family["source_vintages"]["period_mapping"] == surface["period_mapping"]
    assert "canonical_source_manifest" not in family


def test_promoted_manifest_requires_the_full_reference_surface() -> None:
    reference = _resource("efrs_parity_reference.json")
    manifest = _resource("release_input_coverage_manifest.json")
    assert manifest["counts"] == {
        "required": 144,
        "reviewed_exclusion": 1,
        "total": 145,
    }
    assert set(manifest["columns"]) == set(reference["nonzero_shares"])
    # Every reference input is required except the engine-derived total; its
    # three persisted components stay required (microcosm#1106).
    excluded = {
        name: entry
        for name, entry in manifest["columns"].items()
        if entry != {"status": "required"}
    }
    assert set(excluded) == {"property_wealth"}
    assert excluded["property_wealth"]["status"] == "reviewed_exclusion"
    for component in (
        "main_residence_value",
        "other_residential_property_value",
        "non_residential_property_value",
    ):
        assert manifest["columns"][component] == {"status": "required"}
    assert manifest["restoration_evidence"] == {
        "derived_from": "efrs_parity_known_gaps.json",
        "required_columns": ["charitable_investment_gifts", "gift_aid"],
    }
    assert manifest["candidate_evidence"]["tier"] == "frs"
    assert manifest["schema_version"] == 3
    assert manifest["effective_mass_coverage"] == {
        "weight_source": "household_weight",
        "minimum_nondefault_mass_share": 1e-6,
        "reviewed_on": "2026-07-11",
        "rationale": (
            "One part per million rejects zero-weight support and numerical "
            "dust while remaining about 100 times below the rarest populated "
            "record share in the pinned enhanced-FRS reference."
        ),
    }


def test_hmrc_stage_is_required_while_the_208_fact_replay_remains_fenced() -> None:
    manifest = _resource("release_input_coverage_manifest.json")
    family = manifest["family_coverage"]["hmrc_spi_income"]

    assert family["status"] == "required_at_build"
    assert family["restoration_status"] == "adjudicated_partial_replay"
    assert family["source_manifest"] == "source_stages.json"
    assert len(family["source_manifest_sha256"]) == 64
    assert family["base_candidate_tier"] == "frs"
    assert family["source_vintages"] == {
        "spi_donor": "2022-23",
        "hmrc_surface": "2023-24",
        "mapped_build_period": "2024",
        "period_mapping": "latest_published_tax_year",
    }
    assert family["spi_prior_national_household_mass_share"] == 0.5
    assert family["required_predecessor_stages"] == [
        "frs_hmrc_spine_leaves",
        "spi_support_channel",
        "spi_income_band_donors",
    ]
    # The declared allocation's own sentence: a pension-age share is declared
    # since microcosm#1069 c6.
    assert family["required_mass_change_reason"] == (
        "Allocate 0.5 of certified UK national household prior mass in "
        "households without, and 0.2 in households with, a member at or over "
        "State Pension age to the rebuilt 2022-23 SPI support channel; total "
        "national mass is conserved."
    )
    assert family["input_weight_kind"] == "importance"
    assert family["output_weight_kind"] == "importance"
    assert family["calibration_permitted"] is False
    assert family["required_target_count"] == 208
    assert family["band_measure"] == "hmrc_spi_assessable_income"
    assert family["fact_outcome_counts"] == {
        "exact_pass": 0,
        "exact_fail": 0,
        "directional_pass": 0,
        "directional_fail": 0,
        "excluded_with_fence": 208,
    }
    assert family["fact_fence_id"] == "full_frs_tei_band_unavailable"
    assert len(family["reviewed_fence_ids"]) == 8
    assert family["retained_frs_constituents"] == {
        "full": [
            "hmrc_spi_pay",
            "hmrc_spi_unemployment_benefit_income",
            "hmrc_spi_incapacity_benefit_income",
        ],
        "named_subsets": [
            "ossben_identifiable_subset",
            "srp_regular_code5",
        ],
        "source_absent": ["EPB", "EXPS", "TAXTERM", "MOTHINC", "OTHERINC"],
    }
    assert family["effective_mass_requirements"] == {
        "gift_aid": {
            "status": "distributional_required",
            "minimum_nondefault_mass_share": 1e-6,
            "support_channel_column": "person_support_channel",
            "required_support_channel": "spi",
            "mass_share_denominator": "all_person_effective_mass",
        },
        "charitable_investment_gifts": {
            "status": "distributional_required",
            "minimum_nondefault_mass_share": 1e-6,
            "support_channel_column": "person_support_channel",
            "required_support_channel": "spi",
            "mass_share_denominator": "all_person_effective_mass",
        },
    }


def test_generator_assigns_exact_reason_and_tracking_note_to_a_real_gap() -> None:
    generator = _load_generator()
    reference = _resource("efrs_parity_reference.json")
    evidence = _resource("efrs_parity_known_gaps.json")["candidate_evidence"]
    tampered = json.loads(json.dumps(evidence))
    tampered["effective_nondefault_mass_shares"]["dividend_income"] = 0.0
    gaps = generator.build_known_gaps(tampered, reference=reference)
    entry = gaps["known_gaps"]["dividend_income"]
    assert entry["reason"] == (
        "not yet ported from enhanced FRS pipeline — pending review"
    )
    assert entry["tracking_note"].strip()


def test_generator_never_reintroduces_a_pinned_restoration_as_a_gap() -> None:
    generator = _load_generator()
    reference = _resource("efrs_parity_reference.json")
    evidence = _resource("efrs_parity_known_gaps.json")["candidate_evidence"]

    gaps = generator.build_known_gaps(evidence, reference=reference)

    assert gaps["known_gaps"] == {}
    assert set(gaps["restored_required_columns"]) == {
        "charitable_investment_gifts",
        "gift_aid",
    }


def test_generator_keeps_the_engine_derived_section_on_a_candidate_refresh() -> None:
    generator = _load_generator()
    reference = _resource("efrs_parity_reference.json")
    evidence = _resource("efrs_parity_known_gaps.json")["candidate_evidence"]

    gaps = generator.build_known_gaps(evidence, reference=reference)

    assert gaps["engine_derived_exclusions"] == generator.ENGINE_DERIVED_EXCLUSIONS
    assert "property_wealth" not in gaps["known_gaps"]


def test_manifest_generation_rejects_engine_derived_drift_and_overlap() -> None:
    generator = _load_generator()
    reference = _resource("efrs_parity_reference.json")
    drifted = _resource("efrs_parity_known_gaps.json")
    drifted["engine_derived_exclusions"]["property_wealth"]["reason"] = "edited"
    with pytest.raises(ValueError, match="engine_derived_exclusions disagrees"):
        generator.build_manifest(reference=reference, known_gaps_payload=drifted)

    dropped = _resource("efrs_parity_known_gaps.json")
    del dropped["engine_derived_exclusions"]
    with pytest.raises(ValueError, match="engine_derived_exclusions disagrees"):
        generator.build_manifest(reference=reference, known_gaps_payload=dropped)

    # Only a formula-owned input the reference persists can be left to the
    # engine; an ordinary input never becomes an engine-derived exclusion.
    unowned = json.loads(json.dumps(reference))
    unowned["engine"]["formula_owned_persisted_overrides_included"].remove(
        "property_wealth"
    )
    with pytest.raises(ValueError, match="must be formula-owned inputs"):
        generator.build_manifest(
            reference=unowned,
            known_gaps_payload=_resource("efrs_parity_known_gaps.json"),
        )


def test_manifest_generation_rejects_candidate_entity_mismatch() -> None:
    generator = _load_generator()
    reference = _resource("efrs_parity_reference.json")
    gaps = _resource("efrs_parity_known_gaps.json")
    gaps["candidate_evidence"]["column_entities"]["employment_income"] = "household"

    with pytest.raises(ValueError, match="owning-entity evidence disagrees"):
        generator.build_manifest(reference=reference, known_gaps_payload=gaps)


def test_manifest_generation_rejects_candidate_tier_drift() -> None:
    generator = _load_generator()
    reference = _resource("efrs_parity_reference.json")
    gaps = _resource("efrs_parity_known_gaps.json")
    gaps["candidate_evidence"]["source"]["tier"] = "cps-transfer"

    with pytest.raises(ValueError, match="bundled UKDS-licensed FRS lineage"):
        generator.build_manifest(reference=reference, known_gaps_payload=gaps)


def test_hmrc_family_rejects_canonical_source_contract_drift(
    monkeypatch, tmp_path
) -> None:
    generator = _load_generator()
    source_stages = _resource("source_stages.json")
    stage = next(
        stage
        for stage in source_stages["stages"]
        if stage["stage"] == "spi_support_channel"
    )
    operation = next(
        operation
        for operation in stage["operations"]
        if operation["kind"] == "allocate_zero_weight_prior_mass"
    )
    operation["share"] = 0.1
    drifted = tmp_path / "source_stages.json"
    drifted.write_text(json.dumps(source_stages), encoding="utf-8")
    monkeypatch.setattr(generator, "SOURCE_STAGES_PATH", drifted)
    with pytest.raises(ValueError, match="prior.mass_share"):
        generator._hmrc_family_coverage_contract(candidate_source={"tier": "frs"})


def test_candidate_signal_helpers_reject_null_and_blank_only_columns() -> None:
    generator = _load_generator()
    nulls = pd.Series([np.nan, None], dtype=object)
    blanks = pd.Series(["", "  "], dtype=object)
    for column in (nulls, blanks):
        assert generator._nonzero_share(column) == 0.0
        assert generator._nondefault_share(column, 0.0) == 0.0
    assert generator._nondefault_share(pd.Series([0, None], dtype=object), 0.0) == 0.0


def test_effective_signal_helper_ignores_zero_mass_rows() -> None:
    generator = _load_generator()
    column = pd.Series([500.0, 0.0])

    assert generator._nondefault_share(column, 0.0) == 0.5
    assert (
        generator._effective_nondefault_mass_share(
            column,
            0.0,
            np.asarray([0.0, 1_000.0]),
        )
        == 0.0
    )


def test_implicit_candidate_resolution_requires_revision_mapping(monkeypatch) -> None:
    generator = _load_generator()
    import huggingface_hub

    monkeypatch.setattr(generator, "_hf_token", lambda: None)
    monkeypatch.setattr(huggingface_hub, "try_to_load_from_cache", lambda **_: None)
    calls = []

    def download(**kwargs):
        calls.append(kwargs)
        raise RuntimeError("download required")

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", download)
    with pytest.raises(RuntimeError, match="download required"):
        generator.resolve_candidate_h5()

    assert calls == [
        {
            "repo_id": generator.CANDIDATE_REPO_ID,
            "filename": generator.CANDIDATE_FILENAME,
            "revision": generator.CANDIDATE_REVISION,
            "repo_type": generator.CANDIDATE_REPO_TYPE,
            "token": None,
        }
    ]
