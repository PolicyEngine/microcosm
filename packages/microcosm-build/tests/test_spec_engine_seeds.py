from __future__ import annotations

import copy
import hashlib
import random
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

import microcosm.build.spec_engine.seed_callsite_coverage as callsite_coverage
import microcosm.build.spec_engine.seeds as seeds_module
from microcosm.build.spec_engine.canonical import canonical_json_bytes
from microcosm.build.spec_engine.seed_callsite_coverage import (
    HASH_CLASSIFICATION_KINDS,
    LEGACY_V1_HASH_CLASSIFICATIONS,
    LEGACY_V1_PRODUCTION_BINDINGS,
    LEGACY_V1_PRODUCTION_EXEMPTIONS,
    SOURCE_NAMESPACE_EXEMPTIONS,
    UK_ONLY_SOURCE_PREFIXES,
    assert_exact_production_callsite_coverage,
    discover_exempted_source_modules,
    discover_production_callsites,
    discover_production_source_modules,
)
from microcosm.build.spec_engine.seeds import (
    LEGACY_V1_PROTOCOL,
    SeedProtocol,
    source_inventory_sha256,
    validate_seed_protocol_wire,
)
from microcosm.build.us_runtime import sipp_tips, ssi_disability_criteria
from microcosm.build.us_runtime.acs_transfer import (
    _family_seed,
    _pattern_name,
    _pattern_seed,
)
from microcosm.build.us_runtime.multispine_pool import POOL_RANDOM_SEED
from microcosm.build.us_runtime.puf_source_agi import (
    PUF_AGGREGATE_DISAGGREGATION_SEED,
)

EXPECTED_LEGACY_V1_SITES = {
    "acs_qrf_fit_draw",
    "acs_rent_qrf_model",
    "acs_rent_archived_training_cap",
    "acs_transfer_family_seed",
    "acs_transfer_pattern_seed",
    "adult_care_weighted_prefix_assignment",
    "capital_gains_tail_random_rank",
    "child_support_training_cap",
    "child_support_puf_qrf_model",
    "childcare_training_cap",
    "childcare_puf_qrf_model",
    "disability_benefits_training_cap",
    "disability_benefits_puf_qrf_model",
    "eitc_take_up_assignment",
    "energy_subsidy_training_cap",
    "energy_subsidy_puf_qrf_model",
    "exact_k_pcg64_selection",
    "housing_inputs_training_cap",
    "housing_assistance_puf_qrf_model",
    "immigration_ead_students_assignment",
    "immigration_ead_workers_assignment",
    "legacy_congressional_district_assignment",
    "legacy_geography_ladder",
    "legacy_puma_ladder",
    "medicaid_take_up_assignment",
    "other_health_insurance_training_cap",
    "other_health_insurance_puf_qrf_model",
    "org_union_hash_lottery",
    "org_wages_qrf_model",
    "pregnancy_assignment",
    "primary_qrf_fit_draw",
    "primary_puf_monolithic_qrf_model",
    "prior_year_income_training_cap",
    "prior_year_income_puf_qrf_model",
    "puf_archived_aggregate_disaggregation",
    "puf_clone_attachment",
    "puf_live_aggregate_disaggregation",
    "retirement_contributions_training_cap",
    "retirement_contributions_puf_qrf_model",
    "retirement_distributions_training_cap",
    "retirement_distributions_puf_qrf_model",
    "scf_household_source_selector",
    "scf_auto_loan_qrf_model",
    "scf_financial_asset_qrf_model",
    "scf_net_worth_qrf_model",
    "sipp_financial_asset_qrf_models",
    "sipp_financial_asset_training_cap",
    "sipp_tip_training_cap",
    "sipp_tip_qrf_model",
    "sipp_head_start_qrf_model",
    "sipp_vehicle_count_random_forest_model",
    "sipp_vehicle_qrf_model",
    "sipp_vehicle_training_cap",
    "snap_discretionary_exemption_assignment",
    "snap_state_take_up_assignment",
    "snap_take_up_assignment",
    "source_aca_assignment",
    "source_count_calibration",
    "source_joint_count_calibration",
    "ssi_archived_qrf_model",
    "ssi_take_up_assignment",
    "ssi_weighted_replacement_training",
    "survey_sample_acs",
    "survey_sample_asec",
    "tanf_take_up_assignment",
    "torch_calibration_reseed",
    "voluntary_filing_qrf_model",
    "weeks_unemployed_training_cap",
    "weeks_unemployed_puf_qrf_model",
    "wic_claim_assignment",
    "workers_compensation_training_cap",
    "workers_compensation_puf_qrf_model",
}

# Independent total audit oracle: every logical ledger id retains a current
# source home even when several ids intentionally share one physical helper.
AUDITED_SOURCE_BY_SITE = {
    **dict.fromkeys(
        ("survey_sample_asec", "survey_sample_acs"),
        "packages/microcosm-build/src/microcosm/build/frame_sampling.py",
    ),
    **dict.fromkeys(
        ("puf_clone_attachment", "primary_puf_monolithic_qrf_model"),
        "packages/microcosm-build/src/microcosm/build/us_runtime/puf_support.py",
    ),
    "puf_archived_aggregate_disaggregation": "packages/microcosm-build/src/microcosm/build/us_runtime/puf_source_agi.py",
    "puf_live_aggregate_disaggregation": "packages/microcosm-build/src/microcosm/build/us_runtime/puf_aggregate_records.py",
    **dict.fromkeys(
        ("ssi_weighted_replacement_training", "ssi_archived_qrf_model"),
        "packages/microcosm-build/src/microcosm/build/us_runtime/ssi_disability_criteria.py",
    ),
    **dict.fromkeys(
        (
            "sipp_vehicle_training_cap",
            "sipp_vehicle_qrf_model",
            "sipp_vehicle_count_random_forest_model",
        ),
        "packages/microcosm-build/src/microcosm/build/us_runtime/sipp_vehicles.py",
    ),
    **dict.fromkeys(
        ("sipp_financial_asset_training_cap", "sipp_financial_asset_qrf_models"),
        "packages/microcosm-build/src/microcosm/build/us_runtime/sipp_financial_assets.py",
    ),
    **dict.fromkeys(
        (
            "acs_rent_archived_training_cap",
            "housing_inputs_training_cap",
            "acs_rent_qrf_model",
            "housing_assistance_puf_qrf_model",
        ),
        "packages/microcosm-build/src/microcosm/build/us_runtime/housing_inputs.py",
    ),
    **dict.fromkeys(
        ("sipp_tip_training_cap", "sipp_tip_qrf_model"),
        "packages/microcosm-build/src/microcosm/build/us_runtime/sipp_tips.py",
    ),
    **dict.fromkeys(
        (
            "scf_household_source_selector",
            "scf_financial_asset_qrf_model",
            "scf_net_worth_qrf_model",
        ),
        "packages/microcosm-build/src/microcosm/build/us_runtime/scf_wealth.py",
    ),
    "scf_auto_loan_qrf_model": "packages/microcosm-build/src/microcosm/build/us_runtime/scf_auto_loans.py",
    **dict.fromkeys(
        ("acs_transfer_family_seed", "acs_transfer_pattern_seed", "acs_qrf_fit_draw"),
        "packages/microcosm-build/src/microcosm/build/us_runtime/acs_transfer.py",
    ),
    "primary_qrf_fit_draw": "packages/microcosm-fit/src/microcosm/fit/qrf.py",
    **dict.fromkeys(
        (
            "source_aca_assignment",
            "source_count_calibration",
            "source_joint_count_calibration",
        ),
        "packages/microcosm-build/src/microcosm/build/us_runtime/source_runtime.py",
    ),
    "snap_take_up_assignment": "packages/microcosm-build/src/microcosm/build/us_runtime/snap_take_up.py",
    "pregnancy_assignment": "packages/microcosm-build/src/microcosm/build/us_runtime/pregnancy.py",
    "wic_claim_assignment": "packages/microcosm-build/src/microcosm/build/us_runtime/wic_claim.py",
    "snap_discretionary_exemption_assignment": "packages/microcosm-build/src/microcosm/build/us_runtime/snap_discretionary_exemption.py",
    **dict.fromkeys(
        (
            "immigration_ead_workers_assignment",
            "immigration_ead_students_assignment",
        ),
        "packages/microcosm-build/src/microcosm/build/us_runtime/immigration.py",
    ),
    "ssi_take_up_assignment": "packages/microcosm-build/src/microcosm/build/us_runtime/ssi_take_up.py",
    "medicaid_take_up_assignment": "packages/microcosm-build/src/microcosm/build/us_runtime/medicaid_take_up.py",
    "snap_state_take_up_assignment": "packages/microcosm-build/src/microcosm/build/us_runtime/snap_state_take_up.py",
    **dict.fromkeys(
        ("tanf_take_up_assignment", "eitc_take_up_assignment"),
        "packages/microcosm-build/src/microcosm/build/us_runtime/take_up.py",
    ),
    "adult_care_weighted_prefix_assignment": "packages/microcosm-build/src/microcosm/build/us_runtime/adult_care.py",
    "capital_gains_tail_random_rank": "packages/microcosm-build/src/microcosm/build/us_runtime/puf_capital_gains_tail.py",
    "torch_calibration_reseed": "packages/microcosm-calibrate/src/microcosm/calibrate/solve.py",
    "exact_k_pcg64_selection": "packages/microcosm-calibrate/src/microcosm/calibrate/exact_k.py",
    **dict.fromkeys(
        ("prior_year_income_training_cap", "prior_year_income_puf_qrf_model"),
        "packages/microcosm-build/src/microcosm/build/us_runtime/prior_year_income.py",
    ),
    **dict.fromkeys(
        ("childcare_training_cap", "childcare_puf_qrf_model"),
        "packages/microcosm-build/src/microcosm/build/us_runtime/childcare.py",
    ),
    **dict.fromkeys(
        (
            "retirement_contributions_training_cap",
            "retirement_contributions_puf_qrf_model",
        ),
        "packages/microcosm-build/src/microcosm/build/us_runtime/retirement_contributions.py",
    ),
    **dict.fromkeys(
        ("disability_benefits_training_cap", "disability_benefits_puf_qrf_model"),
        "packages/microcosm-build/src/microcosm/build/us_runtime/disability_benefits.py",
    ),
    **dict.fromkeys(
        ("workers_compensation_training_cap", "workers_compensation_puf_qrf_model"),
        "packages/microcosm-build/src/microcosm/build/us_runtime/workers_compensation.py",
    ),
    **dict.fromkeys(
        (
            "retirement_distributions_training_cap",
            "retirement_distributions_puf_qrf_model",
        ),
        "packages/microcosm-build/src/microcosm/build/us_runtime/retirement_distributions.py",
    ),
    **dict.fromkeys(
        ("child_support_training_cap", "child_support_puf_qrf_model"),
        "packages/microcosm-build/src/microcosm/build/us_runtime/child_support.py",
    ),
    **dict.fromkeys(
        ("energy_subsidy_training_cap", "energy_subsidy_puf_qrf_model"),
        "packages/microcosm-build/src/microcosm/build/us_runtime/energy_subsidy.py",
    ),
    **dict.fromkeys(
        (
            "other_health_insurance_training_cap",
            "other_health_insurance_puf_qrf_model",
        ),
        "packages/microcosm-build/src/microcosm/build/us_runtime/other_health_insurance.py",
    ),
    **dict.fromkeys(
        ("weeks_unemployed_training_cap", "weeks_unemployed_puf_qrf_model"),
        "packages/microcosm-build/src/microcosm/build/us_runtime/weeks_unemployed.py",
    ),
    **dict.fromkeys(
        ("org_union_hash_lottery", "org_wages_qrf_model"),
        "packages/microcosm-build/src/microcosm/build/us_runtime/org_wages.py",
    ),
    "sipp_head_start_qrf_model": "packages/microcosm-build/src/microcosm/build/us_runtime/sipp_head_start.py",
    "voluntary_filing_qrf_model": "packages/microcosm-build/src/microcosm/build/us_runtime/voluntary_filing.py",
    "legacy_geography_ladder": "packages/microcosm-build/src/microcosm/build/us_runtime/geography_ladder.py",
    "legacy_puma_ladder": "packages/microcosm-build/src/microcosm/build/us_runtime/puma_ladder.py",
    "legacy_congressional_district_assignment": "packages/microcosm-build/src/microcosm/build/us_runtime/congressional_district_geography.py",
}


def test_legacy_v1_draw_site_inventory_is_exhaustive_and_typed() -> None:
    assert {site.id for site in LEGACY_V1_PROTOCOL.sites} == EXPECTED_LEGACY_V1_SITES
    assert len(LEGACY_V1_PROTOCOL.streams) == 14

    for site in LEGACY_V1_PROTOCOL.sites:
        wire = site.to_wire()
        assert set(wire) == {
            "id",
            "stream",
            "value_source",
            "default",
            "rng_family",
            "rng_version",
            "kernel",
            "seed_material",
            "consumption_order",
            "reset_boundary",
            "draw_condition",
            "derivation",
        }
        assert wire["stream"].startswith("stream:")
        assert wire["seed_material"]
        assert wire["consumption_order"]
        assert wire["reset_boundary"]
        assert wire["derivation"]
        assert wire["rng_version"]
        assert wire["kernel"]


def test_legacy_v1_literals_match_live_constants() -> None:
    assert (
        LEGACY_V1_PROTOCOL.site("survey_sample_asec").default
        == LEGACY_V1_PROTOCOL.site("survey_sample_acs").default
        == 578
    )
    assert LEGACY_V1_PROTOCOL.site("puf_clone_attachment").default == 578
    assert POOL_RANDOM_SEED == 0
    assert (
        LEGACY_V1_PROTOCOL.site("puf_live_aggregate_disaggregation").default
        == POOL_RANDOM_SEED
    )
    assert (
        LEGACY_V1_PROTOCOL.site("puf_archived_aggregate_disaggregation").default
        == PUF_AGGREGATE_DISAGGREGATION_SEED
        == 42
    )
    assert (
        LEGACY_V1_PROTOCOL.site("ssi_weighted_replacement_training").default
        == ssi_disability_criteria._TRAINING_SAMPLE_SEED
        == 8_386_123_572_872_638_692
    )
    assert (
        LEGACY_V1_PROTOCOL.site("ssi_archived_qrf_model").default
        == ssi_disability_criteria._ARCHIVED_MODEL_SEED
        == 42
    )
    assert (
        LEGACY_V1_PROTOCOL.site("sipp_tip_training_cap").default
        == sipp_tips._TIP_TRAINING_SAMPLE_SEED
        == 5_559_651_045_748_063_828
    )


def test_adaptive_training_cap_sites_seal_literal_fill_and_draw_boundaries() -> None:
    vehicle = LEGACY_V1_PROTOCOL.site("sipp_vehicle_training_cap")
    assert vehicle.draw_condition == (
        "union_observed_row_count_gt_20000; fill_only_if_remaining_n_gt_0_"
        "and_remaining_rows_nonempty"
    )

    financial = LEGACY_V1_PROTOCOL.site("sipp_financial_asset_training_cap")
    assert financial.seed_material == (
        "calibration_sipp_asset_training_sample",
        "target_or_fill_salt",
    )
    assert financial.consumption_order == (
        "bank_account_assets",
        "stock_assets",
        "bond_assets",
        "fill",
    )
    assert financial.draw_condition == (
        "max_train_samples_is_not_none_and_union_observed_row_count_gt_"
        "max_train_samples; fill_only_if_remaining_n_gt_0_and_remaining_"
        "rows_nonempty"
    )

    rent = LEGACY_V1_PROTOCOL.site("acs_rent_archived_training_cap")
    assert rent.seed_material == (
        "legacy_acs_rent_training_sample",
        "target_or_fill_salt",
    )
    assert rent.consumption_order == ("rent", "real_estate_taxes", "fill")
    assert rent.draw_condition == (
        "union_observed_row_count_gt_10000; fill_only_if_remaining_n_gt_0_"
        "and_remaining_rows_nonempty"
    )


def test_protocol_digest_covers_every_draw_site_field() -> None:
    first = LEGACY_V1_PROTOCOL.sites[0]
    changed_site = replace(first, reset_boundary="mutated_reset_boundary")
    changed = SeedProtocol(
        id=LEGACY_V1_PROTOCOL.id,
        implementation_id=LEGACY_V1_PROTOCOL.implementation_id,
        kernels=LEGACY_V1_PROTOCOL.kernels,
        sites=(changed_site, *LEGACY_V1_PROTOCOL.sites[1:]),
    )

    assert changed.implementation_sha256 != LEGACY_V1_PROTOCOL.implementation_sha256


def test_f0_legacy_protocol_contains_no_block_first_draw_site() -> None:
    assert "geography_block_draw" not in LEGACY_V1_PROTOCOL.streams
    assert all("block" not in site.id for site in LEGACY_V1_PROTOCOL.sites)


def test_audited_source_manifest_is_independent_and_total() -> None:
    assert set(AUDITED_SOURCE_BY_SITE) == EXPECTED_LEGACY_V1_SITES
    assert set(AUDITED_SOURCE_BY_SITE) == {site.id for site in LEGACY_V1_PROTOCOL.sites}
    root = Path(__file__).resolve().parents[3]
    kernel_by_id = {kernel.id: kernel for kernel in LEGACY_V1_PROTOCOL.kernels}
    for site_id, relative in AUDITED_SOURCE_BY_SITE.items():
        assert (root / relative).is_file(), site_id
        module = (
            relative.split("/src/", maxsplit=1)[1].removesuffix(".py").replace("/", ".")
        )
        site = LEGACY_V1_PROTOCOL.site(site_id)
        assert module in kernel_by_id[site.kernel].source_modules, site_id


def test_production_callsites_are_independent_exact_classified_and_attested() -> None:
    root = Path(__file__).resolve().parents[3]
    kernel_by_id = {kernel.id: kernel for kernel in LEGACY_V1_PROTOCOL.kernels}
    source_modules_by_site = {
        site.id: frozenset(kernel_by_id[site.kernel].source_modules)
        for site in LEGACY_V1_PROTOCOL.sites
    }
    modules = discover_production_source_modules(root)
    callsites = discover_production_callsites(root)
    assert len(modules) == 213
    assert len(callsites) == 285
    assert len(LEGACY_V1_PRODUCTION_BINDINGS) == 119
    assert len(LEGACY_V1_PRODUCTION_EXEMPTIONS) == 166
    assert len(LEGACY_V1_HASH_CLASSIFICATIONS) == 165
    assert {row.kind for row in LEGACY_V1_HASH_CLASSIFICATIONS} == (
        HASH_CLASSIFICATION_KINDS
    )
    assert_exact_production_callsite_coverage(
        root,
        protocol_site_ids=frozenset(EXPECTED_LEGACY_V1_SITES),
        kernel_source_modules_by_site=source_modules_by_site,
    )


def test_production_source_universe_has_only_typed_country_exclusions() -> None:
    root = Path(__file__).resolve().parents[3]
    matches = discover_exempted_source_modules(root)
    assert set(matches) == set(SOURCE_NAMESPACE_EXEMPTIONS)
    assert all(row.reason for row in matches)
    assert all(matches.values())
    assert UK_ONLY_SOURCE_PREFIXES == (
        "microcosm.build.uk_runtime",
        "microcosm.build.stochastic_assignment",
    )
    included = discover_production_source_modules(root)
    assert "microcosm.build.spec_engine.seed_callsite_coverage" in included
    assert "microcosm.data.contract" in included
    assert "microcosm.frame" in included
    assert "microcosm.fit" in included
    assert "tools.build_us_multispine_pool" in included


def test_scanner_finds_new_module_random_seed_sequence_and_hash(tmp_path: Path) -> None:
    tool = tmp_path / "tools/build_us_multispine_pool.py"
    tool.parent.mkdir(parents=True)
    tool.write_text("", encoding="utf-8")
    source = (
        tmp_path
        / "packages/microcosm-build/src/microcosm/build/us_runtime/new_stage.py"
    )
    source.parent.mkdir(parents=True)
    source.write_text(
        "import hashlib\n"
        "import numpy as np\n"
        "def stage():\n"
        "    seed = np.random.SeedSequence(4)\n"
        "    child_seed = seed.spawn(1)[0]\n"
        "    value = child_seed.generate_state(1)\n"
        "    return np.random.default_rng(value).random(), hashlib.sha256(b'x')\n",
        encoding="utf-8",
    )
    discovered = {row.callsite.api for row in discover_production_callsites(tmp_path)}
    assert discovered == {
        "hashlib.sha256",
        "numpy.random.Generator.random",
        "numpy.random.SeedSequence",
        "numpy.random.SeedSequence.generate_state",
        "numpy.random.SeedSequence.spawn",
        "numpy.random.default_rng",
    }


def test_scanner_closes_alias_receiver_hash_uuid_and_escape_paths(
    tmp_path: Path,
) -> None:
    tool = tmp_path / "tools/build_us_multispine_pool.py"
    tool.parent.mkdir(parents=True)
    tool.write_text("", encoding="utf-8")
    source = (
        tmp_path
        / "packages/microcosm-build/src/microcosm/build/us_runtime/new_stage.py"
    )
    source.parent.mkdir(parents=True)
    source.write_text(
        "import hashlib\n"
        "import numpy as np\n"
        "import uuid\n"
        "from uuid import uuid4 as imported_uuid4\n"
        "def parameter(engine: np.random.Generator):\n"
        "    return engine.gamma(2)\n"
        "def locals_():\n"
        "    g = np.random.default_rng(1)\n"
        "    rs = np.random.RandomState(2)\n"
        "    draw = g.gamma\n"
        "    draw_alias = draw\n"
        "    return g.random(), rs.choice(3), draw_alias(2)\n"
        "class Holder:\n"
        "    def __init__(self):\n"
        "        self.g = np.random.default_rng(3)\n"
        "        self.draw = self.g.gamma\n"
        "    def run(self):\n"
        "        return self.draw(2), self.g.permuted([1, 2])\n"
        "def hashes():\n"
        "    digest = hashlib.sha3_256\n"
        "    constructor = hashlib.new\n"
        "    return digest(b'x'), constructor('sha256', b'x')\n"
        "def uuids():\n"
        "    bound = uuid.uuid4\n"
        "    return uuid.uuid4(), imported_uuid4(), bound()\n"
        "def escapes(engine: np.random.Generator, name):\n"
        "    stored = [engine.gamma]\n"
        "    consume(engine.normal)\n"
        "    dynamic = getattr(engine, name)\n"
        "    return engine.choice, stored, dynamic\n",
        encoding="utf-8",
    )
    rows = discover_production_callsites(tmp_path)
    observed = {(row.callsite.qualname, row.callsite.api) for row in rows}
    assert {
        ("parameter", "numpy.random.Generator.gamma"),
        ("locals_", "numpy.random.default_rng"),
        ("locals_", "numpy.random.RandomState"),
        ("locals_", "numpy.random.Generator.random"),
        ("locals_", "numpy.random.RandomState.choice"),
        ("locals_", "numpy.random.Generator.gamma"),
        ("Holder.__init__", "numpy.random.default_rng"),
        ("Holder.run", "numpy.random.Generator.gamma"),
        ("Holder.run", "numpy.random.Generator.permuted"),
        ("hashes", "hashlib.sha3_256"),
        ("hashes", "hashlib.new"),
        ("uuids", "uuid.uuid4"),
        ("escapes", "stochastic.unresolved.bound_method_escape"),
        ("escapes", "stochastic.unresolved.dynamic_getattr"),
    } <= observed
    uuid_rows = [row for row in rows if row.callsite.api == "uuid.uuid4"]
    assert len(uuid_rows) == 3
    escape_rows = [
        row
        for row in rows
        if row.callsite.qualname == "escapes"
        and row.callsite.api == "stochastic.unresolved.bound_method_escape"
    ]
    assert len(escape_rows) == 3


def test_scanner_lexical_facts_kill_stale_precise_labels(tmp_path: Path) -> None:
    tool = tmp_path / "tools/build_us_multispine_pool.py"
    tool.parent.mkdir(parents=True)
    tool.write_text("", encoding="utf-8")
    source = tmp_path / "packages/microcosm-fit/src/microcosm/fit/scope_probe.py"
    source.parent.mkdir(parents=True)
    source.write_text(
        "import numpy as np\n"
        "def precise():\n"
        "    engine = np.random.default_rng(1)\n"
        "    return engine.gamma(2)\n"
        "def killed():\n"
        "    engine = np.random.default_rng(2)\n"
        "    engine = object()\n"
        "    return engine.gamma(2)\n"
        "def sibling():\n"
        "    return engine.gamma(2)\n",
        encoding="utf-8",
    )
    rows = discover_production_callsites(tmp_path)
    by_qualname = {
        row.callsite.qualname: row.callsite.api
        for row in rows
        if row.callsite.api.endswith("gamma")
    }
    assert by_qualname == {
        "precise": "numpy.random.Generator.gamma",
        "killed": "stochastic.unresolved.gamma",
        "sibling": "stochastic.unresolved.gamma",
    }


def test_scanner_namespace_alias_chains_and_branch_joins_are_sound(
    tmp_path: Path,
) -> None:
    tool = tmp_path / "tools/build_us_multispine_pool.py"
    tool.parent.mkdir(parents=True)
    tool.write_text("", encoding="utf-8")
    source = tmp_path / "packages/microcosm-fit/src/microcosm/fit/alias_probe.py"
    source.parent.mkdir(parents=True)
    source.write_text(
        "import hashlib\n"
        "import numpy as np\n"
        "import uuid\n"
        "def factory_chain():\n"
        "    nr = np.random\n"
        "    make = nr.default_rng\n"
        "    engine = make(1)\n"
        "    draw = engine.gamma\n"
        "    again = draw\n"
        "    return again(2)\n"
        "def unresolved_chain(source):\n"
        "    draw = source.gamma\n"
        "    again = draw\n"
        "    return again(2)\n"
        "def false_precision():\n"
        "    rng = object()\n"
        "    return rng.gamma()\n"
        "def hash_namespace():\n"
        "    h = hashlib\n"
        "    make = h.sha3_256\n"
        "    return make(b'x')\n"
        "def uuid_namespace():\n"
        "    u = uuid\n"
        "    make = u.uuid4\n"
        "    return make()\n"
        "def branch_merge(flag):\n"
        "    if flag:\n"
        "        rng = np.random.default_rng(1)\n"
        "    else:\n"
        "        rng = np.random.RandomState(1)\n"
        "    return rng.random()\n"
        "class AnnotatedHolder:\n"
        "    engine: np.random.Generator\n"
        "    def draw(self):\n"
        "        return self.engine.gamma(2)\n"
        "class BranchConfigured:\n"
        "    def configure(self, flag):\n"
        "        if flag:\n"
        "            self.engine = np.random.default_rng(1)\n"
        "        else:\n"
        "            self.engine = np.random.RandomState(1)\n"
        "    def draw(self):\n"
        "        return self.engine.gamma(2)\n"
        "class SetterHolder:\n"
        "    def set_generator(self):\n"
        "        self.engine = np.random.default_rng(1)\n"
        "    def set_random_state(self):\n"
        "        self.engine = np.random.RandomState(1)\n"
        "    def draw(self):\n"
        "        return self.engine.gamma(2)\n"
        "class ReverseSetterHolder:\n"
        "    def set_random_state(self):\n"
        "        self.engine = np.random.RandomState(1)\n"
        "    def set_generator(self):\n"
        "        self.engine = np.random.default_rng(1)\n"
        "    def draw(self):\n"
        "        return self.engine.gamma(2)\n",
        encoding="utf-8",
    )
    rows = discover_production_callsites(tmp_path)
    observed = {(row.callsite.qualname, row.callsite.api) for row in rows}
    assert len(rows) == 19
    assert observed == {
        ("factory_chain", "numpy.random.default_rng"),
        ("factory_chain", "numpy.random.Generator.gamma"),
        ("unresolved_chain", "stochastic.unresolved.gamma"),
        ("false_precision", "stochastic.unresolved.gamma"),
        ("hash_namespace", "hashlib.sha3_256"),
        ("uuid_namespace", "uuid.uuid4"),
        ("branch_merge", "numpy.random.default_rng"),
        ("branch_merge", "numpy.random.RandomState"),
        ("branch_merge", "stochastic.unresolved.random"),
        ("AnnotatedHolder.draw", "numpy.random.Generator.gamma"),
        ("BranchConfigured.configure", "numpy.random.default_rng"),
        ("BranchConfigured.configure", "numpy.random.RandomState"),
        ("BranchConfigured.draw", "stochastic.unresolved.gamma"),
        ("SetterHolder.set_generator", "numpy.random.default_rng"),
        ("SetterHolder.set_random_state", "numpy.random.RandomState"),
        ("SetterHolder.draw", "stochastic.unresolved.gamma"),
        ("ReverseSetterHolder.set_random_state", "numpy.random.RandomState"),
        ("ReverseSetterHolder.set_generator", "numpy.random.default_rng"),
        ("ReverseSetterHolder.draw", "stochastic.unresolved.gamma"),
    }


def test_scanner_api_vocabulary_tracks_runtime_public_callables() -> None:
    def public_callables(value: type[object]) -> frozenset[str]:
        return frozenset(
            name
            for name in dir(value)
            if not name.startswith("_") and callable(getattr(value, name))
        )

    generator_public = public_callables(np.random.Generator)
    assert (
        generator_public - callsite_coverage._NUMPY_GENERATOR_CONTROL_METHODS
        == callsite_coverage._NUMPY_GENERATOR_METHODS
    )
    random_state_public = public_callables(np.random.RandomState)
    assert (
        random_state_public - callsite_coverage._NUMPY_RANDOMSTATE_CONTROL_METHODS
        == callsite_coverage._NUMPY_RANDOMSTATE_DRAW_METHODS
    )
    assert (
        callsite_coverage._NUMPY_RANDOMSTATE_CONTROL_METHODS
        <= callsite_coverage._NUMPY_RANDOMSTATE_METHODS
    )
    python_random_public = public_callables(random.Random)
    assert (
        python_random_public - callsite_coverage._PYTHON_RANDOM_CONTROL_METHODS
        == callsite_coverage._PYTHON_RANDOM_DRAW_METHODS
    )
    assert (
        callsite_coverage._PYTHON_RANDOM_CONTROL_METHODS
        <= callsite_coverage._PYTHON_RANDOM_METHODS
    )
    hashlib_public = frozenset(
        f"hashlib.{name}"
        for name in dir(hashlib)
        if not name.startswith("_") and callable(getattr(hashlib, name))
    )
    assert hashlib_public == callsite_coverage._HASHLIB_APIS


def test_every_uuid4_callsite_is_exclusively_operationally_exempt() -> None:
    root = Path(__file__).resolve().parents[3]
    discovered = {
        row.callsite
        for row in discover_production_callsites(root)
        if row.callsite.api == "uuid.uuid4"
    }
    exemptions = {
        row.callsite: row
        for row in LEGACY_V1_PRODUCTION_EXEMPTIONS
        if row.callsite.api == "uuid.uuid4"
    }
    bindings = {row.callsite for row in LEGACY_V1_PRODUCTION_BINDINGS}
    hash_classifications = {row.callsite for row in LEGACY_V1_HASH_CLASSIFICATIONS}
    assert len(discovered) == 9
    assert discovered == exemptions.keys()
    assert {row.kind for row in exemptions.values()} == {"operational_nonce"}
    assert not discovered & bindings
    assert not discovered & hash_classifications


def test_exact_blake2b_salts_keys_candidates_and_absence_conditions() -> None:
    worker = LEGACY_V1_PROTOCOL.site("immigration_ead_workers_assignment")
    student = LEGACY_V1_PROTOCOL.site("immigration_ead_students_assignment")
    assert "literal_salt=immigration:ead_workers" in worker.seed_material
    assert "literal_salt=immigration:ead_students" in student.seed_material
    assert worker.consumption_order[0] == "all_person_rows_then_worker_candidate_mask"
    assert student.consumption_order[0] == "all_person_rows_then_student_candidate_mask"

    count = LEGACY_V1_PROTOCOL.site("source_count_calibration")
    joint = LEGACY_V1_PROTOCOL.site("source_joint_count_calibration")
    assert "literal_salt=calibrate:{variable}" in count.seed_material
    assert "literal_salt=joint-calibrate:{variable}" in joint.seed_material
    assert count.draw_condition == "only_when_declared_draw_column_is_absent"
    assert joint.draw_condition == "only_when_declared_draw_column_is_absent"

    vectors = (
        (0, "snap_take_up", "2024:10:2", 9_193_979_365_434_741_258),
        (0, "immigration:ead_workers", "2024:2", 3_846_788_339_087_460_008),
        (
            578,
            "aca:takes_up_aca_if_eligible",
            "99",
            6_849_034_126_297_294_090,
        ),
    )
    for seed, salt, key, expected in vectors:
        actual = int.from_bytes(
            hashlib.blake2b(
                f"{seed}:{salt}:{key}".encode(),
                digest_size=8,
            ).digest(),
            "big",
        )
        assert actual == expected


def test_adult_tail_and_live_puf_orders_are_literal() -> None:
    adult = LEGACY_V1_PROTOCOL.site("adult_care_weighted_prefix_assignment")
    assert adult.seed_material == ("build_model_seed",)
    assert adult.consumption_order == (
        "eligible_unit_ids_stable_sorted",
        "one_permutation",
        "permuted_weight_prefix_through_usage_target",
    )
    tail = LEGACY_V1_PROTOCOL.site("capital_gains_tail_random_rank")
    assert tail.seed_material == ("build_model_seed",)
    assert tail.consumption_order[0].startswith("mergesort_recipient_household")
    live_puf = LEGACY_V1_PROTOCOL.site("puf_live_aggregate_disaggregation")
    assert live_puf.consumption_order == (
        "one_shared_generator",
        "aggregate_recids_in_runtime_declared_order",
        "per_bucket_assign_s006_choice_then_donor_choice",
    )


def test_acs_family_pattern_seed_and_label_golden_vectors() -> None:
    assert _family_seed(0, entity="person", family="family_a") == 2_974_888_678
    assert _family_seed(578, entity="tax_unit", family="transfers") == 2_219_304_948
    vectors = (
        ((), 2_696_554_395, "pattern_03_e3b0c442"),
        (("age",), 2_406_131_924, "pattern_03_013f5440"),
        (
            ("age", "employment_income"),
            3_012_383_218,
            "pattern_03_1e060688",
        ),
    )
    for observed, expected_seed, expected_name in vectors:
        assert (
            _pattern_seed(
                0,
                entity="person",
                family="benefits",
                observed_optional=observed,
            )
            == expected_seed
        )
        assert _pattern_name(3, observed) == expected_name
    derivation = LEGACY_V1_PROTOCOL.site("acs_transfer_pattern_seed").derivation
    assert "pattern_" in derivation
    assert "nul_joined_ordered_optional_predictors" in derivation


def test_kernel_attestations_are_recomputed_from_logical_source_inventory() -> None:
    for kernel in LEGACY_V1_PROTOCOL.kernels:
        assert all("/" not in module for module in kernel.source_modules)
        assert source_inventory_sha256(kernel.source_modules) == kernel.source_sha256
    assert all(site.rng_version for site in LEGACY_V1_PROTOCOL.sites)
    qrf_sites = [
        site for site in LEGACY_V1_PROTOCOL.sites if site.kernel == "regime_gated_qrf"
    ]
    assert qrf_sites
    assert all("numpy==" in site.rng_version for site in qrf_sites)


@pytest.mark.parametrize("module_name", ["microcosm.fit", "microcosm.fit.model"])
def test_qrf_kernel_digest_covers_public_dispatch_and_weight_resolution(
    monkeypatch: pytest.MonkeyPatch,
    module_name: str,
) -> None:
    qrf_kernel = next(
        kernel
        for kernel in LEGACY_V1_PROTOCOL.kernels
        if kernel.id == "regime_gated_qrf"
    )
    assert {
        "microcosm.fit",
        "microcosm.fit.model",
        "microcosm.fit.qrf",
    } <= set(qrf_kernel.source_modules)
    baseline = source_inventory_sha256(qrf_kernel.source_modules)
    real_find_spec = seeds_module.find_spec
    real_spec = real_find_spec(module_name)
    assert real_spec is not None and real_spec.origin is not None
    original = Path(real_spec.origin).read_bytes()

    def mutated_find_spec(name: str) -> object:
        if name != module_name:
            return real_find_spec(name)
        loader = SimpleNamespace(
            get_data=lambda _path: original + b"\n# simulated semantic mutation\n"
        )
        return SimpleNamespace(origin=real_spec.origin, loader=loader)

    monkeypatch.setattr(seeds_module, "find_spec", mutated_find_spec)
    assert source_inventory_sha256(qrf_kernel.source_modules) != baseline


def test_shared_qrf_contract_pins_fit_draw_reset_and_runtime_orders() -> None:
    fit = (
        "per_target_fit=detect_regime:no_rng; gated:gate_random_state_integer_then_"
        "HistGradientBoostingClassifier_fit_with_resolved_sample_weight_no_bootstrap; "
        "present_sign_forests_positive_then_negative; each_forest:random_state_"
        "integer_then_external_weighted_bootstrap_choice_if_weighted_then_"
        "RandomForestQuantileRegressor_fit; ungated:single_forest_sequence; "
        "degenerate_zero:no_rng"
    )
    draw = (
        "per_target_draw=degenerate_zero:no_rng; otherwise_one_quantile_uniform_"
        "per_recipient_row; gated_then_one_sign_class_uniform_per_recipient_row; "
        "forest_prediction:no_rng"
    )
    reset = (
        "fresh_SeedSequence(seed).spawn(2)_per_fit_or_start_chain; fit_child_0_"
        "shared_across_declared_targets; draw_child_1_shared_across_declared_"
        "targets_and_successive_predict_calls; checkpoint_resume_restores_then_"
        "advances_both_exact_PCG64_states_without_reseeding_or_entropy"
    )
    qrf_sites = [
        site for site in LEGACY_V1_PROTOCOL.sites if site.kernel == "regime_gated_qrf"
    ]
    assert len(qrf_sites) == 25
    for site in qrf_sites:
        assert fit in site.consumption_order, site.id
        assert draw in site.consumption_order, site.id
        assert (
            "append_each_raw_float64_draw_to_later_recipient_predictors"
            in site.consumption_order
        ), site.id
        assert reset in site.reset_boundary, site.id

    primary = LEGACY_V1_PROTOCOL.site("primary_puf_monolithic_qrf_model")
    assert primary.consumption_order[1] == (
        "declared_target_order=tuple(person_outputs)+tuple(tax_unit_outputs)"
    )
    head_start = LEGACY_V1_PROTOCOL.site("sipp_head_start_qrf_model")
    assert (
        "production_loader_preserves_pinned_source_file_order"
        in (head_start.consumption_order[0])
    )
    assert "eligible_age_3_to_5_subset" in head_start.consumption_order[3]


def test_org_hash_contract_pins_implicit_defaults_full_frame_and_skip_condition() -> (
    None
):
    site = LEGACY_V1_PROTOCOL.site("org_union_hash_lottery")
    assert site.value_source == "literal"
    assert site.seed_material == (
        "implicit_pandas_default_hash_key=0123456789123456",
        "index=False",
        "categorize=True_default",
        "ORG_PREDICTORS_rounded_to_4_decimals",
    )
    assert site.consumption_order[0] == (
        "hash_every_recipient_row_in_current_frame_order_before_eligibility"
    )
    assert site.reset_boundary == (
        "one_stateless_full_recipient_frame_hash_per_assign_union_call"
    )
    assert site.draw_condition.startswith("at_least_one_recipient_row_satisfies_")


def test_protocol_rejects_duplicate_sites_out_of_range_seeds_and_bad_wire_digest() -> (
    None
):
    first = LEGACY_V1_PROTOCOL.sites[0]
    with pytest.raises(ValueError, match="outside uint64"):
        replace(first, default=-1)
    with pytest.raises(ValueError, match="outside uint64"):
        replace(first, default=2**64)
    with pytest.raises(ValueError, match="not unique"):
        SeedProtocol(
            id="legacy-v1",
            implementation_id="duplicate-test",
            kernels=LEGACY_V1_PROTOCOL.kernels,
            sites=(first, first),
        )

    wire = copy.deepcopy(LEGACY_V1_PROTOCOL.to_wire())
    wire["sites"][0]["reset_boundary"] = "mutated"
    with pytest.raises(ValueError, match="implementation_sha256"):
        validate_seed_protocol_wire(wire)
    duplicate = copy.deepcopy(LEGACY_V1_PROTOCOL.to_wire())
    duplicate["sites"].append(copy.deepcopy(duplicate["sites"][0]))
    with pytest.raises(ValueError, match="not unique"):
        validate_seed_protocol_wire(duplicate)


def test_qrf_multi_regime_chain_has_golden_shared_draw_state() -> None:
    from microcosm.fit import RegimeGatedQRF

    row_count = 18
    x = np.linspace(-1.0, 1.0, row_count)
    donor = pd.DataFrame(
        {
            "x": x,
            "zero": np.zeros(row_count),
            "positive": np.arange(1, row_count + 1, dtype=float),
            "gated": np.where(
                np.arange(row_count) % 3 == 0,
                0.0,
                np.arange(1, row_count + 1, dtype=float),
            ),
        }
    )
    fitted = RegimeGatedQRF(n_estimators=3, seed=19, max_samples_leaf=4).fit(
        donor,
        predictors=["x"],
        targets=["zero", "positive", "gated"],
        weights="none",
    )
    assert fitted.regimes() == {
        "zero": "degenerate_zero",
        "positive": "positive_only",
        "gated": "zero_inflated_positive",
    }
    pre_state = hashlib.sha256(
        canonical_json_bytes(fitted._rng.bit_generator.state)
    ).hexdigest()
    assert (
        pre_state == "3d8eec7262f958ee65bb946f583cb6fb41b2ece24a8dbf37b4fe1ee926ef634e"
    )
    drawn = fitted.predict(donor.iloc[:6][["x"]])
    assert drawn["zero"].tolist() == [0.0] * 6
    assert drawn["positive"].to_numpy() == pytest.approx(
        [1.0, 1.4124880203324373, 3.0, 4.0, 4.405932801487469, 6.0]
    )
    assert drawn["gated"].to_numpy() == pytest.approx(
        [2.0, 3.910256579708736, 0.0, 3.235027586673718, 0.0, 0.0]
    )
    post_state = hashlib.sha256(
        canonical_json_bytes(fitted._rng.bit_generator.state)
    ).hexdigest()
    assert (
        post_state == "de1c423f97685d93be539e0d8748b61d13251bcb8b04f63ee01ca37072f7ab43"
    )
