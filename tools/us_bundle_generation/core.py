"""Deterministic extraction of the generation-0 US core bundle domains.

This module is a migration boundary, not a runtime authority.  Every value is
read from a live constant, typed API, packaged compatibility resource, or the
canonical content-verified Logbook receipt.  The returned objects contain only
JSON/YAML scalar, sequence, and mapping types so the one-shot bundle generator
can render them without adding semantic choices.

The compatibility payloads are deliberately complete.  They let the compiler
prove exact generation-0 projection equality while the direct JSON consumers
are migrated to ``ResolvedCountrySpec``; silently dropping fields here would
make that proof circular.
"""

from __future__ import annotations

import copy
import hashlib
import importlib
import inspect
import json
import re
from collections.abc import Mapping
from datetime import UTC, datetime
from importlib import resources as importlib_resources
from pathlib import Path
from typing import Any

from microcosm.build.spec_engine.publication_semantics import (
    compile_publication_regex,
    publication_rung_rows,
)
from microcosm.build.spec_engine.publication_semantics import (
    project_publication_legacy_release as _project_publication_legacy_release,
)
from microcosm.build.spec_engine.publication_semantics import (
    project_spine_legacy_sampling as _project_spine_legacy_sampling,
)

__all__ = [
    "build_bundle",
    "build_catalogs",
    "build_core_specs",
    "build_geography",
    "build_publication",
    "build_sources",
    "build_spine",
    "build_vintages",
    "canonical_input_pins",
    "project_publication_legacy_release",
    "project_spine_legacy_sampling",
]


_REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
_CANONICAL_INPUT_PIN_BUILD_ID = "populace-us-2024-pool-inc2-run7"
_CD_CROSSWALK_PACKAGE = "microcosm.build.us_runtime.data"
_CD_CROSSWALK_RESOURCE = "congressional_district_vintage_crosswalk.csv"
_CD_CROSSWALK_PROVENANCE_RESOURCE = f"{_CD_CROSSWALK_RESOURCE}.provenance.json"
_CD_CROSSWALK_SOURCE_ID = (
    "us_congressional_district_vintage_crosswalk_117_to_119"
)
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_EXPECTED_INPUT_ROLES = frozenset(
    {
        "asec_raw_stage",
        "acs_household",
        "acs_person",
        "acs_rent_donor",
        "processed_puf",
        "puf_source_year",
    }
)
_SOURCE_ROLE_ORDER = (
    "asec_raw_stage",
    "acs_household",
    "acs_person",
    "acs_rent_donor",
    "processed_puf",
    "puf_source_year",
)
_LOADER_BY_ROLE = {
    "asec_raw_stage": "kernel:load_asec_raw_stage_checkpoint",
    "acs_household": "kernel:build_acs_pums_unit_frame",
    "acs_person": "kernel:build_acs_pums_unit_frame",
    "acs_rent_donor": "kernel:load_acs_2022_rent_donor",
    "processed_puf": "kernel:load_puf_tax_unit_donor",
    "puf_source_year": "kernel:load_puf_tax_unit_donor",
}
_VINTAGES_BY_ROLE = {
    "asec_raw_stage": ("vintage:asec_2024", "vintage:asec_2023"),
    "acs_household": ("vintage:acs_2024",),
    "acs_person": ("vintage:acs_2024",),
    "acs_rent_donor": ("vintage:acs_2022",),
    "processed_puf": ("vintage:tax_2015", "vintage:target_2024"),
    "puf_source_year": ("vintage:tax_2015",),
}
_DTYPE = {
    "bool": "bool",
    "float": "float64",
    "int": "int64",
    "str": "string",
}

# RFC v3 decision D6 (docs/spec-engine.md, approved 2026-08-16).  This is
# intentionally embedded in the one-shot migration builder: ``specs/us`` is a
# drafting location that is retired after generation, so it cannot be an input
# to a reproducibility check.  The migration builder reads the live rung map
# once; compiled token lists and regexes are compatibility projections only.
_APPROVED_D6_PUBLICATION: dict[str, Any] = {
    "attempts": {
        "model": "append_only_events_then_terminal_seal",
        "terminal_states": ["landed", "failed", "expired"],
    },
    "promotion": {
        "latest_flip": "human_gate",
        "idempotency": "required_key",
        "recovery": [
            "seal_ok_append_fail",
            "append_ok_alias_fail",
            "orphan_reconciliation",
            "expiry_reconciliation",
        ],
    },
    "release": {
        "line": {
            "value": "microcosm-us-2024",
            "normative": True,
            "legacy_prefixes": ["populace-us-2024"],
            "note": (
                "HF dataset destination is NOT part of D6 — unchanged until a "
                "separate explicit ruling."
            ),
        },
        "pattern": "{line}-stacked-f{rung}-s{seed}",
    },
    "audit_chain": {"kind": "strict_linear", "store": "supabase:logbook"},
    "release_graph": {"relations": ["derived_from", "supersedes", "revokes"]},
}


def _json_package_resource(package: str, filename: str) -> dict[str, Any]:
    resource = importlib_resources.files(package).joinpath(filename)
    payload = json.loads(resource.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{package}/{filename} must contain a JSON object.")
    return payload


def _package_resource_receipt(package: str, filename: str) -> dict[str, str]:
    """Return the compatibility asset identity without making its path semantic."""

    raw = importlib_resources.files(package).joinpath(filename).read_bytes()
    return {
        "id": "source_stages",
        "path": f"{package}/{filename}",
        "sha256": hashlib.sha256(raw).hexdigest(),
    }


def _stacked_tool() -> Any:
    return importlib.import_module("tools.build_us_multispine_pool")


def _stacked_parser_defaults() -> dict[str, int | float]:
    parser = _stacked_tool()._parser()
    return {
        "sample_fraction": float(parser.get_default("sample_fraction")),
        "sample_seed": int(parser.get_default("sample_seed")),
        "clone_attachment_fraction": float(
            parser.get_default("clone_attachment_fraction")
        ),
        "clone_attachment_seed": int(parser.get_default("clone_attachment_seed")),
    }


def _rung_rows() -> list[dict[str, int | float | str]]:
    mapping = _stacked_tool()._STACKED_SAMPLE_RUNG_TOKENS
    return [
        {
            "fraction": float(fraction),
            "token": str(token),
            "percent_basis_points": int(round(float(fraction) * 10_000)),
        }
        for fraction, token in mapping.items()
    ]


def _publication_rung_rows(
    publication: Mapping[str, Any],
) -> list[dict[str, int | float | str]]:
    return publication_rung_rows(publication)

def _compiled_publication_regex(
    *,
    pattern: str,
    line: str,
    rung_tokens: list[str],
) -> str:
    return compile_publication_regex(
        pattern=pattern,
        line=line,
        rung_tokens=rung_tokens,
    )

def project_publication_legacy_release(
    publication: Mapping[str, Any],
) -> dict[str, Any]:
    return _project_publication_legacy_release(publication)

def project_spine_legacy_sampling(
    spine: Mapping[str, Any],
    *,
    publication: Mapping[str, Any],
) -> dict[str, Any]:
    return _project_spine_legacy_sampling(spine, publication=publication)

def _source_stage_compatibility() -> dict[str, Any]:
    from microcosm.build.source_manifest import SourceManifest

    payload = _json_package_resource("microcosm.build.us", "source_stages.json")
    SourceManifest.from_mapping(payload)
    return payload


def _support_spine_compatibility() -> dict[str, Any]:
    from microcosm.build.source_manifest import SupportSpineManifest

    payload = _json_package_resource("microcosm.build.us", "support_spine.json")
    SupportSpineManifest.from_mapping(payload)
    return payload


def _puma_provenance() -> dict[str, Any]:
    payload = _json_package_resource(
        "microcosm.build.us_runtime", "us_puma_ladder.provenance.json"
    )
    digest = payload.get("output_sha256")
    if not isinstance(digest, str) or not _SHA256.fullmatch(digest):
        raise ValueError("US PUMA ladder provenance has no valid output_sha256.")
    return payload


def _congressional_district_vintage_crosswalk_provenance() -> dict[str, Any]:
    """Verify and return the packaged 117th-to-119th CD authority receipt."""

    from microcosm.build.us_runtime.congressional_district_vintage import (
        CURRENT_CONGRESSIONAL_DISTRICT_VINTAGE,
        load_default_congressional_district_vintage_crosswalk,
    )

    payload = _json_package_resource(
        _CD_CROSSWALK_PACKAGE,
        _CD_CROSSWALK_PROVENANCE_RESOURCE,
    )
    expected_digest = payload.get("crosswalk_sha256")
    if not isinstance(expected_digest, str) or not _SHA256.fullmatch(
        expected_digest
    ):
        raise ValueError("US CD-vintage crosswalk has no valid crosswalk_sha256.")
    raw = (
        importlib_resources.files(_CD_CROSSWALK_PACKAGE)
        .joinpath(_CD_CROSSWALK_RESOURCE)
        .read_bytes()
    )
    actual_digest = hashlib.sha256(raw).hexdigest()
    if actual_digest != expected_digest:
        raise ValueError(
            "US CD-vintage crosswalk differs from its packaged provenance: "
            f"expected={expected_digest}, actual={actual_digest}."
        )
    if payload.get("source_geography_vintage") != "117th_congress":
        raise ValueError("US CD-vintage crosswalk source must be 117th_congress.")
    if payload.get("target_geography_vintage") != (
        CURRENT_CONGRESSIONAL_DISTRICT_VINTAGE
    ):
        raise ValueError(
            "US CD-vintage crosswalk target differs from the runtime current "
            "congressional-district vintage."
        )

    # This validates the supported assignment input's columns, district rosters,
    # weights, and population-conservation contract in addition to its byte pin.
    load_default_congressional_district_vintage_crosswalk()
    return {**payload, "byte_size": len(raw)}


def canonical_input_pins() -> dict[str, dict[str, int | str]]:
    """Return the six immutable file identities used by the canonical stack.

    Three roles are not Python constants.  Their current authority is the
    chained run-7 terminal receipt, so this extractor selects that named row,
    verifies the complete Logbook chain and input-pins digest, and then checks
    the independently packaged ACS and rent constants where those exist.
    """

    from microcosm.build.logbook import canonical_json_bytes, load_logbook_file
    from microcosm.build.us_runtime.acs_sources import load_acs_source_manifest
    from microcosm.build.us_runtime.housing_inputs import (
        ACS_2022_RENT_ARTIFACT_SHA256,
    )

    rows = load_logbook_file(_REPOSITORY_ROOT / "logbook" / "us.jsonl")
    matches = [row for row in rows if row.build_id == _CANONICAL_INPUT_PIN_BUILD_ID]
    if len(matches) != 1:
        raise ValueError(
            "Canonical US input-pin Logbook row must resolve exactly once; "
            f"build_id={_CANONICAL_INPUT_PIN_BUILD_ID!r}, matches={len(matches)}."
        )
    row = matches[0]
    terminal = row.gate_verdicts.get("terminal")
    if not isinstance(terminal, Mapping):
        raise ValueError("Canonical US input-pin receipt has no terminal gate object.")
    raw_pins = terminal.get("input_pins")
    if not isinstance(raw_pins, Mapping) or set(raw_pins) != _EXPECTED_INPUT_ROLES:
        raise ValueError(
            "Canonical US input-pin receipt roles differ: "
            f"got={sorted(raw_pins) if isinstance(raw_pins, Mapping) else raw_pins!r}."
        )

    pins: dict[str, dict[str, int | str]] = {}
    for role in sorted(_EXPECTED_INPUT_ROLES):
        raw_pin = raw_pins[role]
        if not isinstance(raw_pin, Mapping) or set(raw_pin) != {
            "sha256",
            "size_bytes",
        }:
            raise ValueError(f"Canonical input pin {role!r} has an invalid shape.")
        digest = raw_pin["sha256"]
        size = raw_pin["size_bytes"]
        if not isinstance(digest, str) or not _SHA256.fullmatch(digest):
            raise ValueError(f"Canonical input pin {role!r} has an invalid SHA-256.")
        if isinstance(size, bool) or not isinstance(size, int) or size < 1:
            raise ValueError(f"Canonical input pin {role!r} has an invalid byte size.")
        pins[role] = {"sha256": digest, "size_bytes": size}

    observed_digest = hashlib.sha256(canonical_json_bytes(pins)).hexdigest()
    if observed_digest != row.input_pins_digest:
        raise ValueError(
            "Canonical US input-pin payload differs from its Logbook digest: "
            f"{observed_digest} != {row.input_pins_digest}."
        )

    acs = load_acs_source_manifest()
    for manifest_role, pin_role in (
        ("household", "acs_household"),
        ("person", "acs_person"),
    ):
        artifact = acs.artifact(manifest_role)
        expected = {"sha256": artifact.sha256, "size_bytes": artifact.size_bytes}
        if pins[pin_role] != expected:
            raise ValueError(
                f"Canonical {pin_role} receipt differs from packaged ACS authority."
            )
    if pins["acs_rent_donor"]["sha256"] != ACS_2022_RENT_ARTIFACT_SHA256:
        raise ValueError(
            "Canonical ACS-rent receipt differs from the runtime artifact pin."
        )
    return pins


def build_bundle() -> dict[str, Any]:
    """Build settings selecting the compiler-owned legacy-v1 protocol."""

    from microcosm.build.spec_engine.seeds import LEGACY_V1_PROTOCOL
    from microcosm.build.us_runtime.multispine_pool import POOL_TIME_PERIOD

    return {
        "country": "us",
        "dataset_run": {"target_period": POOL_TIME_PERIOD},
        "identity_generation": 1,
        "seed_protocol": LEGACY_V1_PROTOCOL.id,
    }


def build_sources() -> dict[str, Any]:
    """Build pinned inputs and first-class typed source-stage declarations."""

    from microcosm.build.us_runtime.acs_pums import ACS_2024_1YR_VINTAGE
    from microcosm.build.us_runtime.multispine_pool import POOL_TIME_PERIOD
    from microcosm.build.us_runtime.puf_source_agi import PUF_SOURCE_YEAR
    from microcosm.build.us_runtime.weeks_unemployed import (
        ASEC_2023_WEEKS_UNEMPLOYED_SOURCE_YEAR,
    )

    pins = canonical_input_pins()
    rows: list[dict[str, Any]] = []
    for role in _SOURCE_ROLE_ORDER:
        pin = pins[role]
        rows.append(
            {
                "id": role,
                "role": role,
                "sha256": pin["sha256"],
                "byte_size": pin["size_bytes"],
                "loader": _LOADER_BY_ROLE[role],
                "vintages": list(_VINTAGES_BY_ROLE[role]),
            }
        )

    authority_rows = {
        "asec_raw_stage": [
            {
                "id": "asec_2022",
                "kind": "survey_period",
                "value": ASEC_2023_WEEKS_UNEMPLOYED_SOURCE_YEAR,
            },
            {
                "id": "asec_2023",
                "kind": "survey_period",
                "value": POOL_TIME_PERIOD - 1,
            },
            {
                "id": "asec_2024",
                "kind": "survey_period",
                "value": POOL_TIME_PERIOD,
            },
        ],
        "acs_household": [
            {
                "id": "acs_2024",
                "kind": "survey_period",
                "value": ACS_2024_1YR_VINTAGE,
            }
        ],
        "acs_rent_donor": [{"id": "acs_2022", "kind": "survey_period", "value": 2022}],
        "puf_source_year": [
            {"id": "tax_2015", "kind": "tax_period", "value": PUF_SOURCE_YEAR}
        ],
    }
    for row in rows:
        authorities = authority_rows.get(str(row["id"]))
        if authorities is not None:
            row["vintage_authorities"] = copy.deepcopy(authorities)

    puma = _puma_provenance()
    rows.append(
        {
            "id": "us_puma_ladder_2020",
            "role": "us_puma_ladder",
            "sha256": puma["output_sha256"],
            "loader": "kernel:load_us_puma_ladder",
            "vintages": [
                "vintage:puma_2020",
                "vintage:census_2020",
                "vintage:cd_119",
            ],
            "vintage_authorities": [
                {
                    "id": "cd_119",
                    "kind": "geography_vintage",
                    "value": puma["layer_vintages"]["congressional_district"],
                },
                {
                    "id": "census_2020",
                    "kind": "geography_vintage",
                    "value": puma["layer_vintages"]["county"],
                },
                {
                    "id": "puma_2020",
                    "kind": "geography_vintage",
                    "value": puma["layer_vintages"]["puma"],
                },
            ],
        }
    )
    crosswalk = _congressional_district_vintage_crosswalk_provenance()
    rows.append(
        {
            "id": _CD_CROSSWALK_SOURCE_ID,
            "role": "congressional_district_vintage_crosswalk",
            "sha256": crosswalk["crosswalk_sha256"],
            "byte_size": crosswalk["byte_size"],
            "loader": "kernel:load_congressional_district_vintage_crosswalk",
            "vintages": ["vintage:cd_117", "vintage:cd_119"],
            "vintage_authorities": [
                {
                    "id": "cd_117",
                    "kind": "geography_vintage",
                    "value": crosswalk["source_geography_vintage"],
                }
            ],
        }
    )
    stage_manifest = _source_stage_compatibility()
    stages = stage_manifest.get("stages")
    if not isinstance(stages, list):
        raise ValueError("US source-stage authority has no ordered stages array.")
    return {
        "sources": rows,
        "stage_asset": _package_resource_receipt(
            "microcosm.build.us", "source_stages.json"
        ),
        "stage_manifest": {
            key: copy.deepcopy(value)
            for key, value in stage_manifest.items()
            if key != "stages"
        },
        "stages": copy.deepcopy(stages),
    }


def build_spine() -> dict[str, Any]:
    """Build stacked-spine, sampling, PUF support, and legacy support data."""

    from microcosm.build.frame_sampling import EXACT_COUNT_RULE
    from microcosm.build.us_runtime.multispine_pool import POOL_HOUSEHOLD_MASS_SHARES
    from microcosm.build.us_runtime.puf_capital_gains_tail import (
        PUF_CAPITAL_GAINS_TAIL_MANIFEST_SCHEMA_VERSION,
        puf_capital_gains_tail_support_contract_identity,
    )
    from microcosm.build.us_runtime.stacked_spine import (
        ACS_STACKED_SUPPORT_CHANNEL,
        DEFAULT_STACKED_HOUSEHOLD_MASS_SHARES,
    )
    from microcosm.build.us_runtime.support_provenance import (
        BASE_ASEC_SUPPORT_CHANNEL,
        PUF_TAX_DETAIL_CLONE_INDEX,
    )
    from tools.us_bundle_generation.identity_contracts import (
        build_pipeline_contract,
        build_seed_site_bindings,
    )

    pool_shares = {
        str(channel): float(share)
        for channel, share in POOL_HOUSEHOLD_MASS_SHARES.items()
    }
    stack_shares = {
        str(channel): float(share)
        for channel, share in DEFAULT_STACKED_HOUSEHOLD_MASS_SHARES.items()
    }
    if pool_shares != stack_shares:
        raise ValueError("Pool and stacked-spine household mass shares differ.")
    expected_channels = [BASE_ASEC_SUPPORT_CHANNEL, ACS_STACKED_SUPPORT_CHANNEL]
    if list(pool_shares) != expected_channels:
        raise ValueError(
            "Stacked household mass-share channel order differs from assembly."
        )

    defaults = _stacked_parser_defaults()
    support_manifest = _support_spine_compatibility()
    support_source_pool = support_manifest.get("support_spine")
    if not isinstance(support_source_pool, Mapping):
        raise ValueError("US support-spine authority has no support_spine object.")
    source_document = _source_stage_compatibility()
    return {
        "pipeline_contract": build_pipeline_contract(),
        "seed_site_bindings": build_seed_site_bindings(source_document),
        "channels": [
            {
                "id": BASE_ASEC_SUPPORT_CHANNEL,
                "source": "asec_raw_stage",
                "observed_geography": "state",
            },
            {
                "id": ACS_STACKED_SUPPORT_CHANNEL,
                "source": ["acs_household", "acs_person"],
                "observed_geography": "puma",
            },
        ],
        "assembly": {
            "mass_anchor_channel": BASE_ASEC_SUPPORT_CHANNEL,
            "shared_dtype_policy": "canonical_string_storage",
            "household_mass_shares": pool_shares,
        },
        "sampling": {
            "channels": expected_channels,
            "grain": "household",
            "fraction": {
                "surface": "run_request",
                "default": defaults["sample_fraction"],
                "rungs_ref": {
                    "domain": "publication",
                    "pointer": "/release/rung_fractions",
                },
            },
            "seed": {
                "surface": "run_request",
                "default": defaults["sample_seed"],
                "streams": {
                    BASE_ASEC_SUPPORT_CHANNEL: "stream:sampling_asec",
                    ACS_STACKED_SUPPORT_CHANNEL: "stream:sampling_acs",
                },
                "same_value_new_generator_per_channel": True,
            },
            "exact_count_rule": EXACT_COUNT_RULE,
            "inventory_order": "sorted_household_id",
            "rng": "numpy.default_rng.choice_without_replacement",
            "full_fraction_is_noop": True,
            "normalize_each_channel_to_full_source_household_mass": True,
        },
        "support_roles": [
            {
                "id": "puf_tax_detail",
                "kind": "puf_attachment",
                "clone_index": PUF_TAX_DETAIL_CLONE_INDEX,
                "attachment": {
                    "fraction": {
                        "surface": "run_request",
                        "default": defaults["clone_attachment_fraction"],
                    },
                    "seed": {
                        "surface": "run_request",
                        "stream": "stream:puf_clone_attachment",
                        "default": defaults["clone_attachment_seed"],
                    },
                },
                "tail_support": {
                    "strata": "filing_status",
                    "policy": "declared_min_support_with_receipt",
                    "manifest_schema_version": (
                        PUF_CAPITAL_GAINS_TAIL_MANIFEST_SCHEMA_VERSION
                    ),
                    "legacy_contract": (
                        puf_capital_gains_tail_support_contract_identity()
                    ),
                },
            }
        ],
        "support_source_pool_metadata": {
            key: copy.deepcopy(value)
            for key, value in support_manifest.items()
            if key != "support_spine"
        },
        "support_source_pool": copy.deepcopy(dict(support_source_pool)),
    }


def build_geography() -> dict[str, Any]:
    """Build the exact generation-0 PUMA-anchored geography declaration."""

    from microcosm.build.us_runtime.congressional_district_vintage import (
        CURRENT_CONGRESSIONAL_DISTRICT_VINTAGE,
    )
    from microcosm.build.us_runtime.puma_ladder import assign_us_puma_ladder

    puma = _puma_provenance()
    crosswalk = _congressional_district_vintage_crosswalk_provenance()
    layer_vintages = puma.get("layer_vintages")
    if not isinstance(layer_vintages, Mapping):
        raise ValueError("US PUMA provenance has no layer_vintages object.")
    if layer_vintages.get("congressional_district") != (
        CURRENT_CONGRESSIONAL_DISTRICT_VINTAGE
    ):
        raise ValueError("US PUMA and runtime congressional-district vintages differ.")
    vintage_ref_by_value = {
        "117th_congress": "vintage:cd_117",
        "119th_congress": "vintage:cd_119",
        "2020_census": "vintage:census_2020",
        "2020_puma": "vintage:puma_2020",
    }
    unknown_vintages = sorted(
        {str(value) for value in layer_vintages.values()} - vintage_ref_by_value.keys()
    )
    if unknown_vintages:
        raise ValueError(
            "US PUMA provenance contains unregistered layer vintages: "
            f"{unknown_vintages!r}."
        )
    default_seed = inspect.signature(assign_us_puma_ladder).parameters["seed"].default
    if isinstance(default_seed, bool) or not isinstance(default_seed, int):
        raise ValueError("US PUMA assignment seed default must be an integer.")

    return {
        "phase": "legacy",
        "assignment": {
            "anchor": "puma",
            "order": "before_gap_fill",
            "kernels": {
                "assign": "kernel:assign_us_puma_ladder",
                "validate": "kernel:us_puma_ladder_gate",
            },
            "draw": {
                "asec": {
                    "universe": "puma_within_state",
                    "weight": "puma_population_2020",
                },
                "congressional_district": {
                    "universe": "congressional_district_within_puma",
                    "weight": "block_population_overlap",
                },
                "county": {
                    "universe": "county_within_puma",
                    "weight": "block_population_overlap",
                },
            },
            "derive": ["puma", "congressional_district_geoid", "county_fips"],
            "assertions": [
                "observed_acs_puma_preserved",
                "geography_state_prefix_consistent",
            ],
            "ladder_source": "source:us_puma_ladder_2020",
            "congressional_district_vintage_crosswalk": {
                "source_ref": f"source:{_CD_CROSSWALK_SOURCE_ID}",
                "source_vintage": vintage_ref_by_value[
                    str(crosswalk["source_geography_vintage"])
                ],
                "target_vintage": vintage_ref_by_value[
                    str(crosswalk["target_geography_vintage"])
                ],
            },
            "seed": "stream:geography_legacy",
            "default_seed": default_seed,
            "assign_tract": False,
            "layer_vintages": {
                str(layer): vintage_ref_by_value[str(vintage)]
                for layer, vintage in sorted(layer_vintages.items())
            },
            "validation": ["puma_ladder_gate", "vintage_refusal"],
        },
    }


def build_publication() -> dict[str, Any]:
    """Build the approved D6 publication contract and exact rung grammar."""

    tool = _stacked_tool()
    payload = copy.deepcopy(_APPROVED_D6_PUBLICATION)
    release = payload.get("release")
    if not isinstance(release, dict):
        raise ValueError("Approved publication draft has no release object.")
    line = release.get("line")
    if not isinstance(line, dict):
        raise ValueError("Approved publication draft has no release line object.")
    line_value = line.get("value")
    if line_value != "microcosm-us-2024":
        raise ValueError(f"Unexpected approved D6 release line {line_value!r}.")
    legacy_prefixes = line.get("legacy_prefixes")
    if legacy_prefixes != ["populace-us-2024"]:
        raise ValueError(
            f"Unexpected approved legacy release prefixes {legacy_prefixes!r}."
        )
    rung_rows = _rung_rows()
    rung_tokens = [str(row["token"]) for row in rung_rows]
    template = (
        "{line}-stacked-{rung}-s{seed}-asec{asec_households}-"
        "acs{acs_households}-{timestamp}-{nonce}"
    )
    release["pattern"] = template
    release["rung_fractions"] = rung_rows

    projected_release = project_publication_legacy_release(payload)
    legacy_regexes = projected_release["legacy_compiled_regexes"]
    if not isinstance(legacy_regexes, list) or len(legacy_regexes) != 1:
        raise ValueError("Approved publication must have one legacy release prefix.")
    legacy_regex = str(legacy_regexes[0])
    if legacy_regex != tool._STACKED_RELEASE_ID_PATTERN.pattern:
        raise ValueError(
            "Derived publication release regex differs from the live writer."
        )
    if projected_release["rungs"] != rung_tokens:
        raise ValueError("Derived publication rung tokens changed ordering.")

    instant = datetime(2024, 1, 2, 3, 4, 5, tzinfo=UTC)
    legacy_vector = tool._new_stacked_release_id(
        sample_fraction=0.25,
        sample_seed=578,
        realized_asec_households=11,
        realized_acs_households=13,
        timestamp=instant,
        nonce="deadbeef",
    )
    rendered_vector = template.format(
        line=legacy_prefixes[0],
        rung="f025",
        seed=578,
        asec_households=11,
        acs_households=13,
        timestamp="20240102T030405Z",
        nonce="deadbeef",
    )
    if rendered_vector != legacy_vector or not re.fullmatch(
        legacy_regex, legacy_vector
    ):
        raise ValueError("Compiled publication grammar differs from the live writer.")
    return payload


def _require_stage_text(payload: Mapping[str, Any], text: str) -> None:
    if text not in json.dumps(payload, sort_keys=True):
        raise ValueError(
            f"Source-stage compatibility data no longer mentions {text!r}."
        )


def build_vintages() -> dict[str, Any]:
    """Build the typed index; values remain on their single authorities."""

    source_stages = _source_stage_compatibility()
    _require_stage_text(source_stages, "SIPP 2023")
    _require_stage_text(source_stages, "SCF 2022")

    records = [
        {
            "id": "acs_2022",
            "kind": "survey_period_ref",
            "authority_ref": {
                "kind": "source_record",
                "source": "source:acs_rent_donor",
                "authority": "acs_2022",
            },
        },
        {
            "id": "acs_2024",
            "kind": "survey_period_ref",
            "authority_ref": {
                "kind": "source_record",
                "source": "source:acs_household",
                "authority": "acs_2024",
            },
        },
        {
            "id": "asec_2022",
            "kind": "survey_period_ref",
            "authority_ref": {
                "kind": "source_record",
                "source": "source:asec_raw_stage",
                "authority": "asec_2022",
            },
        },
        {
            "id": "asec_2023",
            "kind": "survey_period_ref",
            "authority_ref": {
                "kind": "source_record",
                "source": "source:asec_raw_stage",
                "authority": "asec_2023",
            },
        },
        {
            "id": "asec_2024",
            "kind": "survey_period_ref",
            "authority_ref": {
                "kind": "source_record",
                "source": "source:asec_raw_stage",
                "authority": "asec_2024",
            },
        },
        {
            "id": "cd_117",
            "kind": "geography_vintage_ref",
            "authority_ref": {
                "kind": "source_record",
                "source": (
                    "source:us_congressional_district_vintage_crosswalk_117_to_119"
                ),
                "authority": "cd_117",
            },
        },
        {
            "id": "cd_119",
            "kind": "geography_vintage_ref",
            "authority_ref": {
                "kind": "source_record",
                "source": "source:us_puma_ladder_2020",
                "authority": "cd_119",
            },
        },
        {
            "id": "census_2020",
            "kind": "geography_vintage_ref",
            "authority_ref": {
                "kind": "source_record",
                "source": "source:us_puma_ladder_2020",
                "authority": "census_2020",
            },
        },
        {
            "id": "org_2024",
            "kind": "survey_period_ref",
            "authority_ref": {
                "kind": "source_stage_artifact",
                "stage": "org_wages",
                "artifact_index": 0,
                "field": "vintage",
            },
        },
        {
            "id": "policyengine_us_surface",
            "kind": "policy_engine_surface_ref",
            "authority_ref": {
                "kind": "engine_abi_lock",
                "pointer": "/engine/version",
            },
        },
        {
            "id": "puma_2020",
            "kind": "geography_vintage_ref",
            "authority_ref": {
                "kind": "source_record",
                "source": "source:us_puma_ladder_2020",
                "authority": "puma_2020",
            },
        },
        {
            "id": "release_us_2024",
            "kind": "release_series_ref",
            "authority_ref": {
                "kind": "publication_release",
                "pointer": "/release/line/value",
            },
        },
        {
            "id": "scf_2022",
            "kind": "survey_period_ref",
            "authority_ref": {
                "kind": "source_stage_artifact",
                "stage": "scf_wealth",
                "artifact_index": 0,
                "field": "vintage",
            },
        },
        {
            "id": "sipp_2023",
            "kind": "survey_period_ref",
            "authority_ref": {
                "kind": "source_stage_artifact",
                "stage": "scf_wealth",
                "artifact_index": 2,
                "field": "vintage",
            },
        },
        {
            "id": "target_2024",
            "kind": "target_period_ref",
            "authority_ref": {
                "kind": "dataset_run",
                "pointer": "/dataset_run/target_period",
            },
        },
        {
            "id": "tax_2015",
            "kind": "tax_period_ref",
            "authority_ref": {
                "kind": "source_record",
                "source": "source:puf_source_year",
                "authority": "tax_2015",
            },
        },
    ]
    # Compatibility is an explicitly reviewed relation, not an inference from
    # coincident integer values.  The target-period star records which source
    # periods and runtime/release surfaces generation 0 was reviewed against;
    # the two additional cliques are the actual multi-vintage source groups
    # consumed by the pooled ASEC input and the PUMA geography ladder.
    reviewed_compatibility = {
        "acs_2022": {"target_2024"},
        "acs_2024": {"target_2024"},
        "asec_2022": {"target_2024"},
        "asec_2023": {"asec_2024", "target_2024"},
        "asec_2024": {"asec_2023", "target_2024"},
        "cd_117": {"cd_119"},
        "cd_119": {"cd_117", "census_2020", "puma_2020"},
        "census_2020": {"cd_119", "puma_2020"},
        "org_2024": {"target_2024"},
        "policyengine_us_surface": {"target_2024"},
        "puma_2020": {"cd_119", "census_2020"},
        "release_us_2024": {"target_2024"},
        "scf_2022": {"target_2024"},
        "sipp_2023": {"target_2024"},
        "target_2024": {
            "acs_2022",
            "acs_2024",
            "asec_2022",
            "asec_2023",
            "asec_2024",
            "org_2024",
            "policyengine_us_surface",
            "release_us_2024",
            "scf_2022",
            "sipp_2023",
            "tax_2015",
        },
        "tax_2015": {"target_2024"},
    }
    record_ids = {str(row["id"]) for row in records}
    if set(reviewed_compatibility) != record_ids:
        raise ValueError(
            "US reviewed vintage compatibility does not cover every record."
        )
    for row in records:
        row_id = str(row["id"])
        row["compatible_with"] = [
            f"vintage:{target}" for target in sorted(reviewed_compatibility[row_id])
        ]
    return {"records": sorted(records, key=lambda row: str(row["id"]))}


def build_catalogs() -> dict[str, Any]:
    """Build the complete physical column contract catalog.

    PolicyEngine metadata covers the 142 modeled inputs.  The producer graph
    additionally emits generation-0 Frame structure, linkage, support
    provenance, and resolved weights which are deliberately outside the engine
    variable index; those contracts are derived from the canonical producer
    output authority below instead of being hand-maintained alongside it.
    """

    from microcosm.build.us_runtime.multispine_pool import (
        POOL_DEFERRED_TRANSFER_INPUTS,
        pool_input_surface,
    )
    from microcosm.build.us_runtime.us_late_producer_registry import (
        CANONICAL_US_LATE_PRODUCER_SCHEDULE,
    )
    from microcosm.frame.adapters.policyengine_us import (
        PolicyEngineUSVariableMetadataIndex,
    )

    deferred = frozenset(POOL_DEFERRED_TRANSFER_INPUTS)
    index = PolicyEngineUSVariableMetadataIndex()
    columns: list[dict[str, Any]] = []
    seen: set[str] = set()
    for entry in pool_input_surface():
        metadata = index.variable_metadata(entry.variable)
        if metadata.entity != entry.entity:
            raise ValueError(
                f"Pool catalog entity differs for {entry.variable!r}: "
                f"registry={entry.entity!r}, engine={metadata.entity!r}."
            )
        key = f"{entry.entity}.{entry.variable}"
        if key in seen:
            raise ValueError(f"Duplicate pool catalog key {key!r}.")
        seen.add(key)
        try:
            dtype = _DTYPE[metadata.dtype]
        except KeyError as error:
            raise ValueError(
                f"Unsupported engine dtype {metadata.dtype!r} for {entry.variable!r}."
            ) from error
        if dtype == "bool":
            unit = "boolean"
            unit_waiver = None
        elif dtype in {"string", "category"}:
            unit = "categorical"
            unit_waiver = None
        else:
            unit = "unit_not_declared_by_engine_metadata"
            unit_waiver = "policyengine_us_unit_unavailable"
        contract = {
            "entity": entry.entity,
            "dtype": dtype,
            "unit": unit,
            "definition_period": metadata.period,
            "vintage": "vintage:target_2024",
            "nullable": entry.variable in deferred,
            "domain": entry.family,
            "public_stability": "internal",
        }
        if unit_waiver is not None:
            contract["unit_waiver"] = unit_waiver
        columns.append(
            {
                "key": key,
                "contract": contract,
            }
        )

    schedule = json.loads(CANONICAL_US_LATE_PRODUCER_SCHEDULE.canonical_json)
    graph_physical_keys = {
        f"{output['entity']}.{output['column']}"
        for producer in schedule["contracts"]
        for output in producer["outputs"]
        if not str(output["column"]).startswith("@")
        or output["column"] == "@resolved_weight"
    }
    missing_graph_contracts = graph_physical_keys - seen
    if len(graph_physical_keys) != 134 or len(missing_graph_contracts) != 34:
        raise ValueError(
            "Canonical producer/catalog closure changed; "
            f"physical_outputs={len(graph_physical_keys)}, "
            f"missing_contracts={len(missing_graph_contracts)}."
        )
    for key in sorted(missing_graph_contracts):
        entity, column = key.split(".", 1)
        if column == "@resolved_weight":
            dtype = "float64"
            unit = "count"
            definition_period = "year"
            nullable = False
            domain = "frame_weight"
        elif column == "TYPEHUGQ":
            # TYPEHUGQ is source-categorical and only defined on ACS rows.
            dtype = "float64"
            unit = "categorical"
            definition_period = "year"
            nullable = True
            domain = "source_structure"
        elif column.endswith("_support_channel"):
            dtype = "string"
            unit = "categorical"
            definition_period = "eternity"
            nullable = False
            domain = "support_provenance"
        elif column.endswith("_support_clone_index"):
            dtype = "int64"
            unit = "count"
            definition_period = "eternity"
            nullable = False
            domain = "support_provenance"
        elif column.endswith("_source_id"):
            dtype = "int64"
            unit = "count"
            definition_period = "eternity"
            nullable = False
            domain = "source_identity"
        elif column == f"{entity}_id":
            dtype = "int64"
            unit = "count"
            definition_period = "eternity"
            nullable = False
            domain = "frame_identity"
        elif (
            entity == "person"
            and column.startswith("person_")
            and column.endswith("_id")
        ):
            dtype = "int64"
            unit = "count"
            definition_period = "eternity"
            nullable = False
            domain = "frame_membership"
        else:
            raise ValueError(
                f"Producer output {key!r} lacks a closed structural contract rule."
            )
        columns.append(
            {
                "key": key,
                "contract": {
                    "entity": entity,
                    "dtype": dtype,
                    "unit": unit,
                    "definition_period": definition_period,
                    "vintage": "vintage:target_2024",
                    "nullable": nullable,
                    "domain": domain,
                    "public_stability": "internal",
                },
            }
        )
        seen.add(key)
    if not deferred <= {key.split(".", 1)[1] for key in seen}:
        raise ValueError("Deferred pool inputs are absent from the generated catalog.")
    if len(seen) != 176:
        raise ValueError(f"Closed US catalog must contain 176 keys, got {len(seen)}.")
    return {
        "metadata_waivers": [
            {
                "id": "policyengine_us_unit_unavailable",
                "field": "unit",
                "authority": "PolicyEngineUSVariableMetadataIndex",
                "public": False,
                "expires_on": "2026-11-16",
                "reason": (
                    "The import-free installed-engine index exposes entity, dtype, "
                    "and period but no physical unit. Numeric columns remain "
                    "internal until a reviewed unit authority lands."
                ),
            }
        ],
        "columns": sorted(columns, key=lambda row: str(row["key"])),
    }


def build_core_specs() -> dict[str, dict[str, Any]]:
    """Return every D3 core domain under its eventual package-data filename."""

    return {
        "bundle.yaml": build_bundle(),
        "catalogs.yaml": build_catalogs(),
        "geography.yaml": build_geography(),
        "publication.yaml": build_publication(),
        "sources.yaml": build_sources(),
        "spine.yaml": build_spine(),
        "vintages.yaml": build_vintages(),
    }
