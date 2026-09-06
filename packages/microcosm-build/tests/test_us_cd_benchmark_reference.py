"""Bounded, candidate-independent projection of pinned CD published references.

Every bundle here is a tiny invented public-shaped fixture. **A fixture is never
evidence about the real 38.7 MB reference inventory**, which this suite does not
open, parse, hash or characterise. Nothing here scores a candidate, registers a
target, writes a ledger or claims statistical readiness.
"""

import copy
import hashlib
import json
import os
import subprocess
import sys
from decimal import Decimal, localcontext
from pathlib import Path

import pytest

import microcosm.build
from microcosm.build.cd_benchmark.evaluation import evaluate_block
from microcosm.build.cd_benchmark.metrics import ARITHMETIC
from microcosm.build.cd_benchmark.protocol import (
    APPROVED_PROTOCOL_BYTES,
    APPROVED_PROTOCOL_SHA256,
    DistributionBlock,
    PointBlock,
    PrecisionKind,
    canonical_protocol,
    load_canonical_bytes,
)
from microcosm.build.cd_benchmark.reasons import (
    ExposureStatus,
    PrecisionStatus,
    Reason,
)
from microcosm.build.cd_benchmark.reference import ReferenceCell
from microcosm.build.cd_benchmark.universe import PINNED_ADAPTER_MODULES
from microcosm.build.us_runtime import cd_reference, cd_reference_sources
from microcosm.build.us_runtime.cd_benchmark_reference import (
    PROJECTION_CLAIMS_NOT_ESTABLISHED,
    RESERVED_FAMILY_TABLES,
    ReferenceInventory,
    ReferenceProjectionError,
    block_cell_roster,
    cell_units,
    cell_universe,
    count_units,
    district_roster,
    is_reserved_family,
    load_pinned_inventory,
    main,
    pinned_inventory_sha256,
    pinned_inventory_size_bytes,
    project_block,
    project_block_cells,
    project_block_precision,
    projection_completeness,
    require_pinned_adapters,
    roster_scope_reasons,
    table_for_variable,
)

BLOCK_IDS = (
    "population_total",
    "occupied_household_total",
    "age_distribution",
    "income_distribution",
    "tenure_distribution",
)

#: Three invented districts: an ordinary one, DC (published ``98``, canonical
#: ``00``) and one more. These are shapes, not observations.
DISTRICTS = (("01", "01"), ("06", "12"), ("11", "98"))

DC_CANONICAL = "5001900US1100"
DC_PUBLISHED = "5001900US1198"


@pytest.fixture(scope="module")
def protocol():
    return canonical_protocol()


def _stems(protocol, table):
    """Every variable the mandatory blocks read from one table, derived."""
    wanted = []
    for block_id in BLOCK_IDS:
        for request in block_cell_roster(protocol, protocol.block(block_id)):
            if table_for_variable(request.variable) == table:
                wanted.append(request.variable)
    return sorted(dict.fromkeys(wanted))


def _default_value(table, stem):
    """Invented, internally consistent published values. Not real ACS data."""
    if table == "S0101":
        if stem == "S0101_C01_001":
            return ("1000", "50", None, None)
        if stem.startswith("S0101_C01_"):
            return ("50", "10", None, None)
        return ("5.0", "0.5", None, None)
    if table == "B19001":
        if stem == "B19001_001":
            return ("400", "20", None, None)
        return ("25", "8", None, None)
    return {
        "B25003_001": ("400", "20", None, None),
        "B25003_002": ("250", "15", None, None),
        "B25003_003": ("150", "12", None, None),
    }[stem]


def _acs_table(protocol, table, overrides):
    stems = _stems(protocol, table)
    metadata = {"variables": {}}
    for stem in stems:
        for suffix in ("E", "M", "EA", "MA"):
            metadata["variables"][stem + suffix] = {
                "label": f"Estimate!!Invented!!{stem}",
                "concept": f"INVENTED {table}",
                "predicateType": "string" if suffix.endswith("A") else "int",
            }
    header = ["NAME", "GEO_ID", "state", "congressional district"]
    for stem in stems:
        header.extend(stem + suffix for suffix in ("E", "M", "EA", "MA"))
    rows = [header]
    for state, district in DISTRICTS:
        values = [
            f"Invented District {district} (119th Congress), {state}",
            f"5001900US{state}{district}",
            state,
            district,
        ]
        for stem in stems:
            values.extend(
                overrides.get((table, state, district, stem))
                or _default_value(table, stem)
            )
        rows.append(values)
    return cd_reference.acs_table_inventory(table, metadata, rows)


def _bundle(protocol, overrides=None):
    """A public-shaped bundle built through the shipped producer function."""
    overrides = overrides or {}
    captures = []
    for request in cd_reference_sources.census_requests():
        payload = json.dumps(request, sort_keys=True).encode()
        captures.append(
            {
                **request,
                "sha256": hashlib.sha256(payload).hexdigest(),
                "path": f"raw/{hashlib.sha256(payload).hexdigest()}.json",
                "size_bytes": len(payload),
                "retrieved_at": "2026-01-01T00:00:00+00:00",
            }
        )
    acs = {
        table: _acs_table(protocol, table, overrides)
        for table in sorted(RESERVED_FAMILY_TABLES | {"S0101"})
    }
    return {
        "schema_version": 1,
        "kind": "us_cd_reference_inventory",
        "purpose": "reference_inventory_only",
        "candidate_evaluation": "not_performed",
        "target_activation": False,
        "authorities": dict(cd_reference.AUTHORITIES),
        "scope": {
            "country": "us",
            "states": "50_states_and_dc",
            "puerto_rico": "excluded",
            "acs_year": 2024,
            "target_congress": 119,
            "irs_tax_year": 2022,
            "irs_congress": 117,
        },
        "sources": {"census": captures, "local": []},
        "acs": acs,
        "irs": {"rows": [], "row_count": 0, "lineage": "published"},
        "alignment": {"targets": []},
        "reconciliation": {
            "irs_published_district_ids": [],
            "irs_at_large_state_proxy_ids": [],
            "irs_missing_crosswalk_source_ids": [],
            "irs_unmapped_source_ids": [],
            "acs_missing_crosswalk_target_ids": [],
            "acs_unmapped_target_ids": [],
        },
        "lineage_products": {
            "published": "included",
            "crosswalked_values": "not_generated",
            "state_rescaled_values": "not_generated",
        },
    }


@pytest.fixture(scope="module")
def bundle(protocol):
    return _bundle(protocol)


@pytest.fixture
def inventory(protocol, bundle):
    return ReferenceInventory.from_document(copy.deepcopy(bundle), protocol=protocol)


def _write(tmp_path, document):
    payload = (
        json.dumps(document, sort_keys=True, separators=(",", ":"), allow_nan=False)
    ).encode() + b"\n"
    path = tmp_path / "reference-inventory.json"
    path.write_bytes(payload)
    return path, hashlib.sha256(payload).hexdigest()


# --------------------------------------------------------------------------
# The frozen things this lane must leave exactly as it found them.
# --------------------------------------------------------------------------


def test_protocol_and_pure_adapters_are_byte_identical_to_the_contract(protocol):
    """No bound is re-selected and no pinned adapter is touched by this lane."""
    raw = load_canonical_bytes()
    assert hashlib.sha256(raw).hexdigest() == APPROVED_PROTOCOL_SHA256
    assert len(raw) == APPROVED_PROTOCOL_BYTES
    assert protocol.canonical is True
    assert all(require_pinned_adapters(protocol).values())
    assert set(PINNED_ADAPTER_MODULES) == {
        "acs_housing_universe.py",
        "asec_housing_universe.py",
        "cd_reference.py",
    }


def test_pure_benchmark_package_still_loads_no_census_runtime():
    """The projection lives outside ``cd_benchmark`` so that stays pure."""
    src = Path(microcosm.build.__file__).resolve().parents[2]
    code = (
        "import sys\n"
        "import microcosm.build.cd_benchmark.evaluation\n"
        "import microcosm.build.cd_benchmark.ledger\n"
        "import microcosm.build.cd_benchmark.reference\n"
        "import microcosm.build.cd_benchmark.universe\n"
        "assert 'microcosm.build.us_runtime' not in sys.modules, "
        "'the pure package pulled us_runtime'\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        env={**os.environ, "PYTHONPATH": str(src)},
        check=False,
    )
    assert result.returncode == 0, result.stderr


# --------------------------------------------------------------------------
# Loading: the digest is checked before anything is parsed.
# --------------------------------------------------------------------------


def test_a_bad_digest_refuses_before_the_bytes_are_parsed(tmp_path):
    """Unparseable bytes still fail on the digest, proving the order."""
    path = tmp_path / "reference-inventory.json"
    path.write_bytes(b"this is not JSON at all")
    with pytest.raises(ReferenceProjectionError, match="refusing before parsing"):
        load_pinned_inventory(path, expected_sha256="0" * 64)


def test_a_malformed_bundle_with_a_true_digest_is_refused_on_shape(tmp_path):
    path = tmp_path / "reference-inventory.json"
    path.write_bytes(b"{}")
    with pytest.raises(ReferenceProjectionError, match="top-level keys disagree"):
        load_pinned_inventory(path, expected_sha256=hashlib.sha256(b"{}").hexdigest())


def test_duplicate_json_keys_are_refused_by_the_strict_loader(tmp_path):
    path = tmp_path / "reference-inventory.json"
    path.write_bytes(b'{"kind":1,"kind":2}')
    with pytest.raises(ReferenceProjectionError, match="not canonical JSON"):
        load_pinned_inventory(
            path,
            expected_sha256=hashlib.sha256(b'{"kind":1,"kind":2}').hexdigest(),
        )


def test_the_pinned_size_is_enforced_from_the_file_size(tmp_path, protocol):
    """Claiming the pinned digest also claims the pinned byte count."""
    path = tmp_path / "reference-inventory.json"
    path.write_bytes(b"{}")
    assert pinned_inventory_size_bytes() > 0
    with pytest.raises(ReferenceProjectionError, match="refusing before reading"):
        load_pinned_inventory(path, expected_sha256=pinned_inventory_sha256(protocol))


def test_loading_requires_a_stated_digest(tmp_path, bundle):
    """There is no default digest: an unpinned inventory is not an authority."""
    path, _ = _write(tmp_path, bundle)
    with pytest.raises(TypeError):
        load_pinned_inventory(path)
    with pytest.raises(ReferenceProjectionError, match="lowercase 64-character"):
        load_pinned_inventory(path, expected_sha256="not-a-digest")


def test_a_matching_bundle_loads_and_records_its_identity(tmp_path, bundle):
    path, sha256 = _write(tmp_path, bundle)
    loaded = load_pinned_inventory(path, expected_sha256=sha256)
    assert loaded.sha256 == sha256
    assert loaded.size_bytes == path.stat().st_size
    assert loaded["kind"] == "us_cd_reference_inventory"
    assert set(loaded.tables) == RESERVED_FAMILY_TABLES | {"S0101"}


# --------------------------------------------------------------------------
# Shape refusals: duplicates, disagreements and identity defects.
# --------------------------------------------------------------------------


def test_a_duplicate_district_row_is_refused(protocol, bundle):
    document = copy.deepcopy(bundle)
    table = document["acs"]["S0101"]
    table["geographies"].append(dict(table["geographies"][0]))
    with pytest.raises(ReferenceProjectionError, match="repeats district"):
        ReferenceInventory.from_document(document, protocol=protocol)


def test_a_duplicate_cell_is_refused(protocol, bundle):
    document = copy.deepcopy(bundle)
    table = document["acs"]["B25003"]
    table["cells"].append(copy.deepcopy(table["cells"][0]))
    with pytest.raises(ReferenceProjectionError, match="repeats cell"):
        ReferenceInventory.from_document(document, protocol=protocol)


def test_disagreeing_district_sets_across_tables_are_refused(protocol, bundle):
    document = copy.deepcopy(bundle)
    table = document["acs"]["B19001"]
    dropped = table["geographies"].pop()
    table["geography_ids"].remove(dropped["geography_id"])
    with pytest.raises(ReferenceProjectionError, match="district set differs"):
        ReferenceInventory.from_document(document, protocol=protocol)


def test_a_missing_required_table_is_refused(protocol, bundle):
    document = copy.deepcopy(bundle)
    del document["acs"]["B25003"]
    with pytest.raises(ReferenceProjectionError, match="no ACS table"):
        ReferenceInventory.from_document(document, protocol=protocol)


def test_a_scope_disagreement_with_the_protocol_is_refused(protocol, bundle):
    document = copy.deepcopy(bundle)
    document["scope"]["acs_year"] = 2023
    with pytest.raises(ReferenceProjectionError, match="ACS year differs"):
        ReferenceInventory.from_document(document, protocol=protocol)


def test_a_relaxed_reservation_on_a_reserved_table_is_refused(protocol, bundle):
    document = copy.deepcopy(bundle)
    document["acs"]["B19001"]["reservation"]["targets_allowed"] = True
    with pytest.raises(ReferenceProjectionError, match="refuse targets and tuning"):
        ReferenceInventory.from_document(document, protocol=protocol)


def test_an_adapter_disagreement_with_the_inventory_is_refused(protocol, bundle):
    """A drifted classifier is a scope review, never a silent reclassification."""
    document = copy.deepcopy(bundle)
    for cell in document["acs"]["S0101"]["cells"]:
        if cell["variable"] == "S0101_C01_001":
            cell["estimate"]["class"] = "controlled_estimate"
            break
    inventory = ReferenceInventory.from_document(document, protocol=protocol)
    with pytest.raises(ReferenceProjectionError, match="classifies"):
        project_block(
            inventory,
            block_id="population_total",
            geography_id="5001900US0101",
            protocol=protocol,
        )


def test_a_negative_ordinary_numeric_estimate_is_refused(protocol, bundle):
    """``/reference_status/count_domains``: never a negative population count."""
    document = copy.deepcopy(bundle)
    for cell in document["acs"]["B25003"]["cells"]:
        if cell["variable"] == "B25003_002":
            cell["estimate"].update(raw="-4", numeric_value=-4.0, class_="numeric")
            cell["estimate"]["class"] = "numeric"
            break
    inventory = ReferenceInventory.from_document(document, protocol=protocol)
    with pytest.raises(ReferenceProjectionError, match="Negative ordinary numeric"):
        project_block(
            inventory,
            block_id="tenure_distribution",
            geography_id="5001900US0101",
            protocol=protocol,
        )


# --------------------------------------------------------------------------
# Rosters derived from the protocol, never retyped.
# --------------------------------------------------------------------------


def test_block_cell_rosters_are_derived_from_the_compiled_protocol(protocol):
    sizes = {
        block_id: len(block_cell_roster(protocol, protocol.block(block_id)))
        for block_id in BLOCK_IDS
    }
    assert sizes == {
        "population_total": 1,
        "occupied_household_total": 1,
        # denominator + 18 published age bins + 18 published percentage columns
        "age_distribution": 37,
        # denominator + 16 published income bins
        "income_distribution": 17,
        # denominator + owner + renter
        "tenure_distribution": 3,
    }
    age = protocol.block("age_distribution")
    assert age.precision_kind is PrecisionKind.PUBLISHED_PERCENTAGE_MOE
    assert cell_units(age, role="precision") == "percent"
    assert cell_units(age, role="bin") == "persons"
    assert cell_universe(protocol, age) == "population"
    income = protocol.block("income_distribution")
    assert count_units(income) == "households"
    assert cell_universe(protocol, income) == "occupied_housing_units"
    assert count_units(protocol.block("population_total")) == "persons"


def test_reserved_families_are_named_only_where_the_protocol_reserves_them():
    assert RESERVED_FAMILY_TABLES == frozenset({"B19001", "B25003"})
    assert is_reserved_family("B19001")
    assert is_reserved_family("B25003_002")
    assert not is_reserved_family("S0101")
    assert not is_reserved_family("S0101_C02_004")
    with pytest.raises(ReferenceProjectionError):
        is_reserved_family("")


def test_table_is_the_prefix_rule_the_shipped_evaluator_applies():
    assert table_for_variable("S0101_C02_019") == "S0101"
    assert table_for_variable("B19001_017") == "B19001"
    with pytest.raises(ReferenceProjectionError):
        table_for_variable("")


# --------------------------------------------------------------------------
# The district roster and the shipped scope check.
# --------------------------------------------------------------------------


def test_the_roster_retains_both_canonical_and_published_dc_identities(
    protocol, inventory
):
    roster = district_roster(inventory, protocol=protocol)
    assert roster.geography_ids == (
        "5001900US0101",
        "5001900US0612",
        DC_CANONICAL,
    )
    assert roster.published_geography_ids[DC_CANONICAL] == DC_PUBLISHED
    assert roster.published_geography_ids["5001900US0101"] == "5001900US0101"


def _synthetic_roster(count, *, dc_published=DC_PUBLISHED):
    from microcosm.build.us_runtime.cd_benchmark_reference import DistrictRoster

    identifiers = [DC_CANONICAL]
    for state in range(1, 60):
        for district in range(1, 40):
            if len(identifiers) >= count:
                break
            candidate = f"5001900US{state:02d}{district:02d}"
            if candidate != DC_CANONICAL:
                identifiers.append(candidate)
        if len(identifiers) >= count:
            break
    identifiers = tuple(sorted(identifiers))
    assert len(identifiers) == count
    published = {
        identifier: (dc_published if identifier == DC_CANONICAL else identifier)
        for identifier in identifiers
    }
    return DistrictRoster(
        geography_ids=identifiers,
        published_geography_ids=published,
        tables=("B19001", "B25003", "S0101"),
    )


def test_a_complete_roster_passes_and_a_short_one_trips_the_scope_check(protocol):
    complete = _synthetic_roster(protocol.scope.district_count)
    reasons, notes = roster_scope_reasons(complete, protocol=protocol)
    assert reasons == ()
    assert notes["required_district_count"] == protocol.scope.district_count

    short = _synthetic_roster(protocol.scope.district_count - 1)
    reasons, notes = roster_scope_reasons(short, protocol=protocol)
    assert Reason.REQUIRED_CELL_OR_DISTRICT_MISSING in reasons
    assert notes["declared_district_count"] == protocol.scope.district_count
    assert notes["required_district_count"] == protocol.scope.district_count - 1
    assert DC_CANONICAL in short.geography_ids
    assert short.published_geography_ids[DC_CANONICAL] == DC_PUBLISHED


def test_an_incorrect_dc_alias_is_reported_and_never_repaired(protocol):
    wrong = _synthetic_roster(protocol.scope.district_count, dc_published=DC_CANONICAL)
    reasons, notes = roster_scope_reasons(wrong, protocol=protocol)
    assert Reason.IDENTITY_MISMATCH in reasons
    assert notes["dc_alias_mismatch"] is True
    assert wrong.published_geography_ids[DC_CANONICAL] == DC_CANONICAL


def test_a_wrong_published_alias_in_the_bundle_is_reported_not_repaired(
    protocol, bundle
):
    document = copy.deepcopy(bundle)
    for table in document["acs"].values():
        for record in table["geographies"]:
            if record["geography_id"] == DC_CANONICAL:
                record["published_geo_id"] = DC_CANONICAL
    inventory = ReferenceInventory.from_document(document, protocol=protocol)
    projection = project_block(
        inventory,
        block_id="population_total",
        geography_id=DC_CANONICAL,
        protocol=protocol,
    )
    assert Reason.IDENTITY_MISMATCH in projection.reasons
    assert projection.published_geo_id == DC_CANONICAL
    assert projection.evidence["expected_published_geo_id"] == DC_PUBLISHED


# --------------------------------------------------------------------------
# Projection proper.
# --------------------------------------------------------------------------


def test_a_population_block_projects_one_typed_cell_and_one_reading(
    protocol, inventory
):
    projection = project_block(
        inventory,
        block_id="population_total",
        geography_id="5001900US0101",
        protocol=protocol,
    )
    assert projection.complete is True
    assert projection.reserved_family is False
    assert projection.missing_variables == ()
    (cell,) = projection.reference_cells
    assert isinstance(cell, ReferenceCell)
    assert cell.variable == "S0101_C01_001"
    assert cell.table == "S0101"
    assert cell.dataset == "acs/acs1/subject"
    assert cell.units == "persons"
    assert cell.universe == "population"
    assert cell.year == protocol.scope.acs_year
    assert cell.congress == protocol.scope.congress
    assert cell.canonical_geography_id == "5001900US0101"
    assert cell.published_geo_id == "5001900US0101"
    (reading,) = projection.precision_results
    assert reading.kind == "relative_moe"
    assert reading.status is PrecisionStatus.ADEQUATE
    assert reading.measured == Decimal("50") / Decimal("1000")
    assert (
        reading.ceiling
        == protocol.block("population_total").reference_relative_moe_ceiling
    )
    assert projection.evidence["candidate_measurement"] == "not_read"
    assert projection.evidence["exposure_status"] == "not_determined_by_projection"


def test_dc_projects_under_its_canonical_identity_and_published_alias(
    protocol, inventory
):
    projection = project_block(
        inventory,
        block_id="age_distribution",
        geography_id=DC_CANONICAL,
        protocol=protocol,
    )
    assert projection.complete is True
    assert projection.published_geo_id == DC_PUBLISHED
    assert Reason.IDENTITY_MISMATCH not in projection.reasons
    assert {cell.published_geo_id for cell in projection.reference_cells} == {
        DC_PUBLISHED
    }
    assert len(projection.precision_results) == 18
    assert {item.kind for item in projection.precision_results} == {
        "published_percentage_moe_pp"
    }
    assert {item.status for item in projection.precision_results} == {
        PrecisionStatus.ADEQUATE
    }


@pytest.mark.parametrize("block_id", BLOCK_IDS)
def test_projected_values_are_accepted_by_the_shipped_evaluator(
    protocol, inventory, block_id
):
    """The evaluator's own composition check is the pin on our labels.

    ``metric=None`` because projection reads no candidate: the block is
    correctly incomplete, and it is incomplete for the *stated* reasons rather
    than because a label was wrong. Exposure is supplied as unknown because a
    projection determines no freshness.
    """
    projection = project_block(
        inventory,
        block_id=block_id,
        geography_id="5001900US0612",
        protocol=protocol,
    )
    assert projection.complete is True
    evaluation = evaluate_block(
        protocol.block(block_id),
        geography_id="5001900US0612",
        metric=None,
        precision=projection.precision_results,
        support=(),
        exposure=ExposureStatus.EXPOSURE_HISTORY_UNKNOWN,
        scope=protocol.scope,
    )
    assert Reason.IDENTITY_MISMATCH not in evaluation.reasons
    assert Reason.REFERENCE_INVALID not in evaluation.reasons
    assert Reason.REFERENCE_PRECISION_INADEQUATE not in evaluation.reasons
    # Absent candidate measurement and absent origin lineage, stated honestly.
    assert Reason.REQUIRED_CELL_OR_DISTRICT_MISSING in evaluation.reasons
    assert Reason.UNRESOLVED_SOURCE_LINEAGE in evaluation.reasons
    assert Reason.EXPOSURE_HISTORY_UNKNOWN in evaluation.reasons
    assert evaluation.scope_complete is False


def test_reserved_blocks_carry_reserved_provenance_and_others_do_not(
    protocol, inventory
):
    reserved = project_block(
        inventory,
        block_id="income_distribution",
        geography_id="5001900US0101",
        protocol=protocol,
    )
    assert reserved.reserved_family is True
    assert reserved.exposure_families == frozenset({"B19001"})
    assert all(item.reserved_family for item in reserved.cells)
    assert all(item.reserved_family for item in reserved.precision)
    assert reserved.evidence["reserved_exposure_families"] == ("B19001",)

    occupied = project_block(
        inventory,
        block_id="occupied_household_total",
        geography_id="5001900US0101",
        protocol=protocol,
    )
    assert occupied.reserved_family is True
    assert occupied.exposure_families == frozenset({"B19001", "B25003"})
    assert occupied.evidence["duplicate_reference_aliases_not_projected"] == (
        "B19001_001",
    )
    assert occupied.evidence["count_once"] is True
    assert [cell.variable for cell in occupied.reference_cells] == ["B25003_001"]

    plain = project_block(
        inventory,
        block_id="age_distribution",
        geography_id="5001900US0101",
        protocol=protocol,
    )
    assert plain.reserved_family is False
    assert not any(item.reserved_family for item in plain.cells)
    assert plain.evidence["reserved_exposure_families"] == ()


def test_a_controlled_moe_stays_controlled_and_never_becomes_adequate(protocol):
    """``/reference_status/controlled_estimate``: valid point, no adequacy claim."""
    document = _bundle(
        protocol,
        {
            ("S0101", "01", "01", "S0101_C01_001"): (
                "1000",
                "-555555555",
                None,
                "*****",
            )
        },
    )
    inventory = ReferenceInventory.from_document(document, protocol=protocol)
    projection = project_block(
        inventory,
        block_id="population_total",
        geography_id="5001900US0101",
        protocol=protocol,
    )
    (reading,) = projection.precision_results
    assert reading.status is PrecisionStatus.CONTROLLED_UNKNOWN
    assert reading.status is not PrecisionStatus.ADEQUATE
    assert reading.measured is None
    assert reading.reasons == ()
    assert reading.evidence["raw_moe"] == "-555555555"
    assert reading.evidence["numeric_moe"] is None
    assert Reason.REFERENCE_INVALID not in projection.reasons
    (cell,) = projection.reference_cells
    assert cell.estimate.usable is True
    assert cell.moe.adapter_class == "controlled_estimate"


def test_a_sentinel_estimate_is_invalid_and_yields_no_precision_claim(protocol):
    document = _bundle(
        protocol,
        {("S0101", "01", "01", "S0101_C01_001"): ("-666666666", "50", None, None)},
    )
    inventory = ReferenceInventory.from_document(document, protocol=protocol)
    projection = project_block(
        inventory,
        block_id="population_total",
        geography_id="5001900US0101",
        protocol=protocol,
    )
    assert Reason.REFERENCE_INVALID in projection.reasons
    (cell,) = projection.reference_cells
    assert cell.estimate.adapter_class == "insufficient_sample"
    assert cell.estimate.numeric_value is None
    (reading,) = projection.precision_results
    assert reading.status is PrecisionStatus.UNAVAILABLE
    assert reading.measured is None
    assert Reason.REFERENCE_PRECISION_INADEQUATE in reading.reasons


def test_the_subset_share_negative_radicand_uses_the_plus_fallback(protocol):
    """``100*sqrt(Mx^2-(X/Y)^2*My^2)/Y`` with the documented plus fallback."""
    document = _bundle(
        protocol,
        {
            ("B19001", "01", "01", "B19001_001"): ("1000", "200", None, None),
            ("B19001", "01", "01", "B19001_002"): ("100", "5", None, None),
        },
    )
    inventory = ReferenceInventory.from_document(document, protocol=protocol)
    projection = project_block(
        inventory,
        block_id="income_distribution",
        geography_id="5001900US0101",
        protocol=protocol,
    )
    block = protocol.block("income_distribution")
    first = projection.precision_results[0]
    assert first.kind == "subset_share_moe_pp"
    assert [cell.variable for cell in first.cells] == ["B19001_002", "B19001_001"]
    assert first.evidence["negative_radicand_plus_fallback"] is True
    with localcontext(ARITHMETIC):
        share = Decimal(100) / Decimal(1000)
        expected = (
            Decimal(100)
            * (Decimal(5) ** 2 + share**2 * Decimal(200) ** 2).sqrt()
            / Decimal(1000)
        )
    assert first.measured == expected
    assert first.measured > block.reference_moe_ceiling_pp
    assert first.status is PrecisionStatus.INADEQUATE
    assert Reason.REFERENCE_PRECISION_INADEQUATE in projection.reasons
    # Every other bin in this district keeps its own reading.
    assert all(
        item.evidence.get("negative_radicand_plus_fallback") is False
        for item in projection.precision_results[1:]
    )


def test_a_missing_cell_gives_a_complete_missing_result_and_a_strict_refusal(
    protocol, bundle
):
    document = copy.deepcopy(bundle)
    table = document["acs"]["S0101"]
    table["cells"] = [
        cell
        for cell in table["cells"]
        if not (
            cell["geography_id"] == "5001900US0101"
            and cell["variable"] == "S0101_C02_004"
        )
    ]
    inventory = ReferenceInventory.from_document(document, protocol=protocol)
    projection = project_block(
        inventory,
        block_id="age_distribution",
        geography_id="5001900US0101",
        protocol=protocol,
    )
    assert projection.complete is False
    assert projection.missing_variables == ("S0101_C02_004",)
    assert Reason.REQUIRED_CELL_OR_DISTRICT_MISSING in projection.reasons
    assert projection.precision == ()
    assert len(projection.cells) == 36
    assert projection.evidence["missing_variables"] == ("S0101_C02_004",)
    with pytest.raises(ReferenceProjectionError, match="missing published cells"):
        project_block_cells(
            inventory,
            block_id="age_distribution",
            geography_id="5001900US0101",
            protocol=protocol,
        )
    # A different district in the same bundle is unaffected: nothing is borrowed.
    other = project_block(
        inventory,
        block_id="age_distribution",
        geography_id="5001900US0612",
        protocol=protocol,
    )
    assert other.complete is True


def test_a_missing_district_reports_every_required_cell_as_missing(protocol, inventory):
    projection = project_block(
        inventory,
        block_id="tenure_distribution",
        geography_id="5001900US9901",
        protocol=protocol,
    )
    assert projection.complete is False
    assert projection.cells == ()
    assert projection.precision == ()
    assert projection.published_geo_id is None
    assert set(projection.missing_variables) == {
        "B25003_001",
        "B25003_002",
        "B25003_003",
    }
    assert Reason.REQUIRED_CELL_OR_DISTRICT_MISSING in projection.reasons


def test_projection_requires_an_explicit_block_and_district(protocol, inventory):
    with pytest.raises(TypeError):
        project_block(inventory, geography_id="5001900US0101")
    with pytest.raises(TypeError):
        project_block(inventory, block_id="population_total")
    with pytest.raises(ValueError, match="Unknown block id"):
        project_block(
            inventory,
            block_id="everything",
            geography_id="5001900US0101",
            protocol=protocol,
        )


def test_precision_refuses_a_short_or_repeated_cell_set(protocol, inventory):
    cells = project_block_cells(
        inventory,
        block_id="tenure_distribution",
        geography_id="5001900US0101",
        protocol=protocol,
    )
    assert (
        len(project_block_precision(protocol, cells, block_id="tenure_distribution"))
        == 2
    )
    with pytest.raises(ReferenceProjectionError, match="needs cell"):
        project_block_precision(protocol, cells[:1], block_id="tenure_distribution")
    with pytest.raises(ReferenceProjectionError, match="supplied twice"):
        project_block_precision(
            protocol, (*cells, cells[0]), block_id="tenure_distribution"
        )


# --------------------------------------------------------------------------
# Aggregate completeness and the command line.
# --------------------------------------------------------------------------


def test_completeness_reports_counts_and_no_value_sum_or_verdict(protocol, inventory):
    roster = district_roster(inventory, protocol=protocol)
    projections = [
        project_block(
            inventory,
            block_id=block_id,
            geography_id=geography_id,
            protocol=protocol,
        )
        for block_id in BLOCK_IDS
        for geography_id in roster.geography_ids
    ]
    summary = projection_completeness(
        projections, protocol=protocol, roster=roster, inventory=inventory
    )
    payload = summary.payload()
    assert payload["protocol_sha256"] == APPROVED_PROTOCOL_SHA256
    assert payload["candidate_scoring"] is False
    assert payload["target_registration"] is False
    assert payload["exposure_ledger_write"] is False
    assert payload["threshold_selection"] is False
    assert payload["national_reference_cells"].startswith("not_available")
    assert (
        "national published reference conformity" in (payload["claims_not_established"])
    )
    assert "candidate scoring" in payload["claims_not_established"]
    by_block = {item["block_id"]: item for item in payload["blocks"]}
    assert set(by_block) == set(BLOCK_IDS)
    assert by_block["age_distribution"]["districts_projected"] == 3
    assert by_block["age_distribution"]["districts_complete"] == 3
    assert by_block["age_distribution"]["cells_projected"] == 111
    assert by_block["age_distribution"]["precision_readings"] == 54
    assert by_block["income_distribution"]["reserved_family"] is True
    assert by_block["population_total"]["reserved_family"] is False
    assert by_block["population_total"]["precision_status_counts"] == {"adequate": 3}
    # A short roster is reported, never silently accepted.
    assert (
        Reason.REQUIRED_CELL_OR_DISTRICT_MISSING.value
        in (payload["roster_scope_reasons"])
    )
    # Nothing in the payload is a published value.
    text = json.dumps(payload)
    assert "1000" not in text
    assert PROJECTION_CLAIMS_NOT_ESTABLISHED[-1] == "statistical readiness"


def test_the_command_line_is_aggregate_only_and_gated(tmp_path, bundle, capsys):
    path, sha256 = _write(tmp_path, bundle)
    output = tmp_path / "completeness.json"
    base = [
        "--inventory",
        str(path),
        "--expect-sha256",
        sha256,
        "--aggregate-only",
        "--all-roster-districts",
    ]

    assert main([*base, "--block", "income_distribution"]) == 1
    assert "--include-reserved" in capsys.readouterr().err

    assert main([*base]) == 1
    assert "never sweeps blocks" in capsys.readouterr().err

    assert (
        main(
            [
                "--inventory",
                str(path),
                "--expect-sha256",
                sha256,
                "--aggregate-only",
                "--block",
                "population_total",
            ]
        )
        == 1
    )
    assert "there is no default" in capsys.readouterr().err

    assert main([*base, "--block", "population_total", "--output", str(output)]) == 0
    payload = json.loads(output.read_text())
    assert payload["kind"] == "us_cd_reference_projection_completeness"
    assert payload["inventory_sha256"] == sha256
    assert payload["roster_district_count"] == 3
    assert [item["block_id"] for item in payload["blocks"]] == ["population_total"]
    assert payload["blocks"][0]["cells_projected"] == 3
    assert payload["candidate_scoring"] is False

    assert (
        main(
            [
                *base,
                "--block",
                "tenure_distribution",
                "--include-reserved",
                "--output",
                str(output),
            ]
        )
        == 0
    )
    payload = json.loads(output.read_text())
    assert payload["blocks"][0]["reserved_family"] is True


def test_the_command_line_refuses_a_digest_mismatch(tmp_path, bundle, capsys):
    path, _ = _write(tmp_path, bundle)
    code = main(
        [
            "--inventory",
            str(path),
            "--expect-sha256",
            "0" * 64,
            "--aggregate-only",
            "--block",
            "population_total",
            "--all-roster-districts",
        ]
    )
    assert code == 1
    assert "refusing before parsing" in capsys.readouterr().err


def test_the_module_exposes_no_national_or_scoring_entry_point():
    """No district-summing producer and no approval object are offered here."""
    from microcosm.build.us_runtime import cd_benchmark_reference as module

    exported = set(module.__all__)
    assert not any(name for name in exported if "national" in name or "score" in name)
    assert not any(name for name in exported if "approve" in name or "register" in name)
    assert not any(name for name in exported if "ledger" in name)
    assert not hasattr(module, "sum_districts")
    for name in ("PointBlock", "DistributionBlock"):
        assert name not in exported
    assert isinstance(module.RESERVED_FAMILY_TABLES, frozenset)


def test_types_are_the_shipped_ones_not_copies(protocol, inventory):
    projection = project_block(
        inventory,
        block_id="tenure_distribution",
        geography_id="5001900US0101",
        protocol=protocol,
    )
    block = protocol.block("tenure_distribution")
    assert isinstance(block, DistributionBlock)
    assert isinstance(protocol.block("population_total"), PointBlock)
    for cell in projection.reference_cells:
        assert type(cell) is ReferenceCell
    for reading in projection.precision_results:
        assert reading.ceiling == block.reference_moe_ceiling_pp
        assert reading.cells and all(
            type(cell) is ReferenceCell for cell in reading.cells
        )
