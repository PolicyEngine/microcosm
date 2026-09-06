"""Bounded, candidate-independent projection of pinned CD published references.

Every bundle here is a tiny invented public-shaped fixture. **A fixture is never
evidence about the real 38.7 MB reference inventory**, which this suite does not
open, parse, hash or characterise. Nothing here scores a candidate, registers a
target, writes a ledger or claims statistical readiness.
"""

import copy
import dataclasses
import hashlib
import json
import operator
import os
import subprocess
import sys
from decimal import Decimal, localcontext
from pathlib import Path

import pytest

import microcosm.build
import microcosm.build.cd_benchmark
import microcosm.frame
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

#: The invented point cell root's counterexample mutated after a verified load.
POPULATION_CELL = ("S0101", "5001900US0101", "S0101_C01_001")


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


def _acs_table(protocol, table, overrides, *, include_geo_id=True):
    stems = _stems(protocol, table)
    metadata = {"variables": {}}
    for stem in stems:
        for suffix in ("E", "M", "EA", "MA"):
            metadata["variables"][stem + suffix] = {
                "label": f"Estimate!!Invented!!{stem}",
                "concept": f"INVENTED {table}",
                "predicateType": "string" if suffix.endswith("A") else "int",
            }
    header = ["NAME", "state", "congressional district"]
    if include_geo_id:
        header.insert(1, "GEO_ID")
    for stem in stems:
        header.extend(stem + suffix for suffix in ("E", "M", "EA", "MA"))
    rows = [header]
    for state, district in DISTRICTS:
        values = [
            f"Invented District {district} (119th Congress), {state}",
            state,
            district,
        ]
        if include_geo_id:
            values.insert(1, f"5001900US{state}{district}")
        for stem in stems:
            values.extend(
                overrides.get((table, state, district, stem))
                or _default_value(table, stem)
            )
        rows.append(values)
    return cd_reference.acs_table_inventory(table, metadata, rows)


def _bundle(protocol, overrides=None, *, include_geo_id=True):
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
        table: _acs_table(protocol, table, overrides, include_geo_id=include_geo_id)
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
    """The projection lives outside ``cd_benchmark`` so that stays pure.

    The child interpreter drops every ``packages/microcosm-*/src`` entry that is
    not one of *this* checkout's, because an editable install in the ambient
    environment can point at a different worktree — without that, this would
    assert some other tree's purity rather than the tree under test.
    """
    src = Path(microcosm.build.__file__).resolve().parents[2]
    packages = src.parents[1]
    allowed = sorted(
        str(shard / "src")
        for shard in packages.iterdir()
        if shard.name.startswith("microcosm-") and (shard / "src").is_dir()
    )
    assert str(src) in allowed
    code = (
        "import os, sys\n"
        f"allowed = {allowed!r}\n"
        "sys.path[:] = [\n"
        "    entry\n"
        "    for entry in sys.path\n"
        "    if not (\n"
        "        os.path.realpath(entry).endswith(os.sep + 'src')\n"
        "        and os.sep + 'packages' + os.sep + 'microcosm-'\n"
        "        in os.path.realpath(entry)\n"
        "    )\n"
        "    or os.path.realpath(entry) in {os.path.realpath(p) for p in allowed}\n"
        "]\n"
        "import microcosm.build.cd_benchmark.evaluation as evaluation\n"
        "import microcosm.build.cd_benchmark.ledger\n"
        "import microcosm.build.cd_benchmark.reference\n"
        "import microcosm.build.cd_benchmark.universe\n"
        "loaded = os.path.realpath(evaluation.__file__)\n"
        "assert any(\n"
        "    loaded.startswith(os.path.realpath(entry) + os.sep) for entry in allowed\n"
        "), 'the purity check ran against a foreign checkout: ' + loaded\n"
        "assert 'microcosm.build.us_runtime' not in sys.modules, (\n"
        "    'the pure package pulled us_runtime'\n"
        ")\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        env={
            **os.environ,
            "PYTHONPATH": os.pathsep.join(allowed),
        },
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
            cell["estimate"].update(
                raw="-4", numeric_value=-4.0, **{"class": "numeric"}
            )
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


@pytest.mark.parametrize("geography_id", ("5001900US0612", DC_CANONICAL))
@pytest.mark.parametrize("block_id", BLOCK_IDS)
def test_projected_values_are_accepted_by_the_shipped_evaluator(
    protocol, inventory, block_id, geography_id
):
    """The evaluator's own composition check is the pin on our labels.

    ``metric=None`` because projection reads no candidate: the block is
    correctly incomplete, and it is incomplete for the *stated* reasons rather
    than because a label was wrong. Exposure is supplied as unknown because a
    projection determines no freshness.

    DC is included because its canonical identity and its published ``…1198``
    alias differ, and the shipped evaluator checks that pair itself: an
    ordinary district would never exercise that arm.
    """
    projection = project_block(
        inventory,
        block_id=block_id,
        geography_id=geography_id,
        protocol=protocol,
    )
    assert projection.complete is True
    assert projection.published_geo_id == (
        DC_PUBLISHED if geography_id == DC_CANONICAL else geography_id
    )
    evaluation = evaluate_block(
        protocol.block(block_id),
        geography_id=geography_id,
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
    assert payload["inventory_identity_verified"] is True
    assert payload["kind"] == "us_cd_reference_projection_completeness"
    assert payload["inventory_sha256"] == sha256
    assert payload["roster_district_count"] == 3
    assert [item["block_id"] for item in payload["blocks"]] == ["population_total"]
    assert payload["blocks"][0]["cells_projected"] == 3
    assert payload["candidate_scoring"] is False

    # A separate success needs a separate path: this tool replaces no record.
    reserved_output = tmp_path / "completeness-reserved.json"
    assert (
        main(
            [
                *base,
                "--block",
                "tenure_distribution",
                "--include-reserved",
                "--output",
                str(reserved_output),
            ]
        )
        == 0
    )
    payload = json.loads(reserved_output.read_text())
    assert payload["blocks"][0]["reserved_family"] is True
    assert output.exists()


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


# --------------------------------------------------------------------------
# Identity: only bytes that were checked may be labelled.
# --------------------------------------------------------------------------


def test_from_document_refuses_to_state_an_unverified_identity(protocol, bundle):
    """A digest nothing checked is worse in an artifact than no digest at all.

    ``from_document`` never saw a file, so it can verify neither the digest nor
    the byte count. It used to accept both as keyword arguments and store them
    unchecked, and ``projection_completeness`` then reported them as
    ``inventory_sha256`` with nothing in the payload to say they were merely
    claimed.
    """
    document = copy.deepcopy(bundle)
    for forged in ({"sha256": "0" * 64}, {"size_bytes": 38686741}):
        # TypeError, not ReferenceProjectionError: the defect is in the call,
        # not in the bundle, and a caller that reports every
        # ReferenceProjectionError as a rejected bundle must not be handed its
        # own mistake as a statement about the inventory.
        with pytest.raises(TypeError, match="cannot label an inventory's identity"):
            ReferenceInventory.from_document(document, protocol=protocol, **forged)

    # Only those two are named, so a typo keeps Python's own diagnostic rather
    # than being accused of forging an identity it never mentioned.
    with pytest.raises(TypeError, match="unexpected keyword argument 'protocl'"):
        ReferenceInventory.from_document(document, protocl=protocol)

    inventory = ReferenceInventory.from_document(document, protocol=protocol)
    assert inventory.sha256 is None
    assert inventory.size_bytes is None
    assert inventory.identity_verified is False
    payload = projection_completeness(
        [], protocol=protocol, inventory=inventory
    ).payload()
    assert payload["inventory_sha256"] is None
    assert payload["inventory_size_bytes"] is None
    assert payload["inventory_identity_verified"] is False


def test_a_verified_identity_cannot_be_forged_onto_an_inventory(
    tmp_path, protocol, bundle
):
    """The label is not reachable through the constructor or ``replace`` either.

    ``identity_verified`` is only worth reading if nothing but a byte-verifying
    load can set it. Before ``sha256`` and ``size_bytes`` became ``init=False``,
    ``dataclasses.replace(inventory, sha256=...)`` produced an inventory built
    from a three-district invented fixture that reported the real 38.7 MB
    bundle's pinned digest and byte count under a verified label — review
    finding 1 re-entering through a second door.
    """
    inventory = ReferenceInventory.from_document(
        copy.deepcopy(bundle), protocol=protocol
    )
    assert inventory.identity_verified is False

    with pytest.raises(TypeError, match="init=False"):
        dataclasses.replace(inventory, sha256=pinned_inventory_sha256(protocol))
    with pytest.raises(TypeError, match="sha256"):
        ReferenceInventory(
            document=inventory.document,
            tables=inventory.tables,
            datasets=inventory.datasets,
            geographies=inventory.geographies,
            published_geography_ids=inventory.published_geography_ids,
            cells=inventory.cells,
            sha256=pinned_inventory_sha256(protocol),
        )

    # A byte-verifying load is still the one route that sets it.
    path, sha256 = _write(tmp_path, bundle)
    assert load_pinned_inventory(path, expected_sha256=sha256).identity_verified


def test_only_a_byte_verifying_load_reports_a_verified_identity(
    tmp_path, protocol, bundle
):
    path, sha256 = _write(tmp_path, bundle)
    loaded = load_pinned_inventory(path, expected_sha256=sha256)
    assert loaded.identity_verified is True
    assert loaded.sha256 == hashlib.sha256(path.read_bytes()).hexdigest()
    assert loaded.size_bytes == path.stat().st_size
    payload = projection_completeness(
        [
            project_block(
                loaded,
                block_id="population_total",
                geography_id="5001900US0101",
                protocol=protocol,
            )
        ],
        protocol=protocol,
        inventory=loaded,
    ).payload()
    assert payload["inventory_identity_verified"] is True
    assert payload["inventory_sha256"] == sha256
    assert payload["inventory_size_bytes"] == path.stat().st_size


def test_indexing_refuses_a_protocol_that_is_not_the_canonical_document(
    protocol, bundle
):
    """Canonical facts are never attributed to a contract that never stated them.

    The audited key list, the audited ``candidate_evaluation`` scalar and the
    pinned byte count are read from the packaged canonical document whatever
    ``protocol`` says, so mixing them with another compiled protocol is refused
    rather than documented away.
    """
    other = dataclasses.replace(protocol, sha256="0" * 64)
    assert other.canonical is False
    with pytest.raises(ReferenceProjectionError, match="non-canonical protocol"):
        ReferenceInventory.from_document(copy.deepcopy(bundle), protocol=other)
    with pytest.raises(ReferenceProjectionError, match="compiled protocol is required"):
        ReferenceInventory.from_document(copy.deepcopy(bundle), protocol="canonical")


# --------------------------------------------------------------------------
# A validated inventory cannot move under the evidence it produced.
# --------------------------------------------------------------------------


def _relative_moe(inventory, protocol):
    return (
        project_block(
            inventory,
            block_id="population_total",
            geography_id="5001900US0101",
            protocol=protocol,
        )
        .precision_results[0]
        .measured
    )


@pytest.mark.parametrize("route", ("from_document", "load_pinned_inventory"))
def test_a_validated_inventory_cannot_be_changed_after_the_fact(
    tmp_path, protocol, bundle, route
):
    """Root's counterexample: halve a relative MOE while the file stands still.

    ``inv.cells[...]["estimate"]["raw"] = "2000"`` after a byte-verified load
    doubled the denominator and halved the reading, while the digest, the
    recorded identity and the file on disk were all unchanged. Both
    construction routes are pinned, because the fixture route is the one a
    caller holds its own reference through.
    """
    document = copy.deepcopy(bundle)
    if route == "from_document":
        inventory = ReferenceInventory.from_document(document, protocol=protocol)
    else:
        path, sha256 = _write(tmp_path, document)
        inventory = load_pinned_inventory(path, expected_sha256=sha256)
    before = _relative_moe(inventory, protocol)
    assert before == Decimal("50") / Decimal("1000")

    record = inventory.cells[POPULATION_CELL]
    # Real subscript assignment, which is what root's counterexample performed.
    assignments = (
        lambda: operator.setitem(record, "estimate", {"raw": "2000"}),
        lambda: operator.setitem(record["estimate"], "raw", "2000"),
        lambda: operator.setitem(inventory.cells, POPULATION_CELL, {}),
        lambda: operator.setitem(inventory.document, "kind", "something_else"),
        lambda: operator.setitem(
            inventory.geographies["5001900US0101"], "published_geo_id", DC_PUBLISHED
        ),
        lambda: operator.setitem(inventory["acs"], "S0101", {}),
        lambda: operator.setitem(inventory["acs"]["S0101"]["cells"], 0, {}),
        lambda: operator.setitem(
            inventory["acs"]["S0101"]["geographies"][0],
            "published_geo_id",
            DC_PUBLISHED,
        ),
        lambda: operator.setitem(inventory["acs"]["S0101"]["geography_ids"], 0, "x"),
    )
    for attempt in assignments:
        with pytest.raises(TypeError):
            attempt()
    # And the mutating methods are not merely guarded, they are absent.
    for attempt in (
        lambda: record["estimate"].update(raw="2000"),
        lambda: record["estimate"].pop("raw"),
        lambda: inventory["acs"]["S0101"]["cells"].append({}),
    ):
        with pytest.raises(AttributeError):
            attempt()
    assert _relative_moe(inventory, protocol) == before

    # The caller's own object is severed too, not merely wrapped: mutating the
    # document that was handed in changes nothing that was already validated.
    for cell in document["acs"]["S0101"]["cells"]:
        if cell["variable"] == "S0101_C01_001":
            cell["estimate"]["raw"] = "2000"
            cell["estimate"]["numeric_value"] = 2000.0
    assert inventory.cells[POPULATION_CELL]["estimate"]["raw"] == "1000"
    assert _relative_moe(inventory, protocol) == before


def test_the_frozen_document_keeps_the_json_it_was_parsed_from(
    protocol, bundle, inventory
):
    """Freezing removes mutability and nothing else."""
    assert inventory["kind"] == bundle["kind"]
    assert inventory["target_activation"] is False
    assert inventory["scope"]["acs_year"] == bundle["scope"]["acs_year"]
    assert inventory["irs"]["rows"] == ()
    assert inventory["acs"]["S0101"]["geographies"][0]["published_geo_id"] == (
        "5001900US0101"
    )
    round_tripped = json.loads(
        json.dumps(inventory.document, default=dict, sort_keys=True)
    )
    assert round_tripped == json.loads(json.dumps(bundle, sort_keys=True))


def test_a_document_carrying_non_json_data_is_refused(protocol, bundle):
    """``from_document`` stays exactly as strict about shape as the loader."""
    document = copy.deepcopy(bundle)
    document["acs"]["S0101"]["geography_ids"] = tuple(
        document["acs"]["S0101"]["geography_ids"]
    )
    with pytest.raises(ReferenceProjectionError, match="not JSON data: tuple"):
        ReferenceInventory.from_document(document, protocol=protocol)

    document = copy.deepcopy(bundle)
    document["scope"]["acs_year"] = Decimal("2024")
    with pytest.raises(ReferenceProjectionError, match="not JSON data: Decimal"):
        ReferenceInventory.from_document(document, protocol=protocol)

    document = copy.deepcopy(bundle)
    document["acs"]["S0101"][2024] = "not a text key"
    with pytest.raises(ReferenceProjectionError, match="object key that is not text"):
        ReferenceInventory.from_document(document, protocol=protocol)


# --------------------------------------------------------------------------
# Refusals the producer's own fixtures never exercised.
# --------------------------------------------------------------------------


def test_a_capture_without_a_published_geo_id_is_refused(protocol):
    """The producer writes ``row.get("GEO_ID")``; absent, that is ``None``.

    ``published_GEO_ID`` is a required raw field and the contract requires both
    identities retained, so an inventory built from captures without the column
    is refused rather than projected under the canonical identity alone.
    """
    document = _bundle(protocol, include_geo_id=False)
    assert document["acs"]["S0101"]["geographies"][0]["published_geo_id"] is None
    with pytest.raises(ReferenceProjectionError, match="published GEO_ID"):
        ReferenceInventory.from_document(document, protocol=protocol)


def test_a_capture_dataset_that_disagrees_with_the_block_is_refused(protocol, bundle):
    """The block declares a dataset; the bundle records what was captured."""
    document = copy.deepcopy(bundle)
    for capture in document["sources"]["census"]:
        if capture["table"] == "S0101":
            capture["dataset"] = "acs/acs1"
    assert protocol.block("population_total").dataset == "acs/acs1/subject"
    inventory = ReferenceInventory.from_document(document, protocol=protocol)
    with pytest.raises(ReferenceProjectionError, match="was captured from dataset"):
        project_block(
            inventory,
            block_id="population_total",
            geography_id="5001900US0101",
            protocol=protocol,
        )


def test_one_table_captured_under_two_datasets_is_refused(protocol, bundle):
    document = copy.deepcopy(bundle)
    for capture in document["sources"]["census"]:
        if capture["table"] == "S0101" and capture["kind"] == "data":
            capture["dataset"] = "acs/acs1"
    with pytest.raises(ReferenceProjectionError, match="two datasets"):
        ReferenceInventory.from_document(document, protocol=protocol)


# --------------------------------------------------------------------------
# Completeness counts what was actually requested.
# --------------------------------------------------------------------------


def test_the_same_block_and_district_cannot_be_counted_twice(protocol, inventory):
    """Root's counterexample: two districts projected, one district requested."""
    projection = project_block(
        inventory,
        block_id="population_total",
        geography_id="5001900US0101",
        protocol=protocol,
    )
    with pytest.raises(ReferenceProjectionError, match="supplied twice"):
        projection_completeness(
            [projection, projection], protocol=protocol, inventory=inventory
        )
    single = projection_completeness(
        [projection], protocol=protocol, inventory=inventory
    ).payload()
    assert single["blocks"][0]["districts_projected"] == 1
    assert single["districts_requested"] == ["5001900US0101"]

    # The same district under a different block is not a duplicate.
    other = project_block(
        inventory,
        block_id="age_distribution",
        geography_id="5001900US0101",
        protocol=protocol,
    )
    both = projection_completeness(
        [projection, other], protocol=protocol, inventory=inventory
    ).payload()
    assert [item["districts_projected"] for item in both["blocks"]] == [1, 1]
    assert both["districts_requested"] == ["5001900US0101"]

    with pytest.raises(ReferenceProjectionError, match="BlockProjection"):
        projection_completeness(["population_total"], protocol=protocol)


def test_an_empty_projection_stays_empty_and_is_never_roster_complete(
    protocol, inventory
):
    """Zero projections used to default to the whole roster as present."""
    payload = projection_completeness(
        [], protocol=protocol, inventory=inventory
    ).payload()
    assert payload["districts_requested"] == []
    assert payload["blocks"] == []
    assert payload["roster_district_count"] == 3
    assert (
        Reason.REQUIRED_CELL_OR_DISTRICT_MISSING.value
        in payload["roster_scope_reasons"]
    )
    assert payload["roster_scope_notes"]["missing_districts"] == [
        "5001900US0101",
        "5001900US0612",
        DC_CANONICAL,
    ]


def test_dc_cannot_be_counted_under_both_of_its_published_identities(
    protocol, inventory
):
    """One real district, two published ids, and only one of them is a district.

    DC is the one district whose canonical id and published GEO_ID differ, and
    ``…1198`` is exactly what the raw ACS ``GEO_ID`` column carries, so an
    operator naming both is a natural mistake. Keyed on the identity string
    alone, that produced ``districts_projected`` 4 against a
    ``roster_district_count`` of 3 — more districts projected than the roster
    holds, which is root's counterexample in a second form.
    """
    roster = district_roster(inventory, protocol=protocol)
    canonical = project_block(
        inventory,
        block_id="population_total",
        geography_id=DC_CANONICAL,
        protocol=protocol,
    )
    alias = project_block(
        inventory,
        block_id="population_total",
        geography_id=DC_PUBLISHED,
        protocol=protocol,
    )
    with pytest.raises(ReferenceProjectionError, match="the same district"):
        projection_completeness(
            [canonical, alias], protocol=protocol, roster=roster, inventory=inventory
        )

    # Either identity alone still counts once, and the roster is never exceeded.
    for single in (canonical, alias):
        payload = projection_completeness(
            [single], protocol=protocol, roster=roster, inventory=inventory
        ).payload()
        assert payload["blocks"][0]["districts_projected"] == 1
        assert (
            payload["blocks"][0]["districts_projected"]
            <= (payload["roster_district_count"])
        )


def test_the_payload_names_only_reserved_families_it_actually_read(protocol, inventory):
    """Declared exposure families are not the same as cells read into values.

    ``occupied_household_total`` declares both reserved families but projects a
    single ``B25003_001`` cell; the same run records ``B19001_001`` in
    ``duplicate_reference_aliases_not_projected``. Reporting B19001 as projected
    would have the artifact contradict its own evidence.
    """
    projection = project_block(
        inventory,
        block_id="occupied_household_total",
        geography_id="5001900US0101",
        protocol=protocol,
    )
    assert "B19001" in projection.exposure_families
    assert projection.evidence["duplicate_reference_aliases_not_projected"] == (
        "B19001_001",
    )
    payload = projection_completeness(
        [projection], protocol=protocol, inventory=inventory
    ).payload()
    assert payload["reserved_families_projected_into_values"] == ["B25003"]
    assert {item.cell.table for item in projection.cells} == {"B25003"}


def test_a_payload_with_no_inventory_states_nothing_about_reserved_tables(protocol):
    """The reserved-indexing sentence is a claim about bytes that were indexed."""
    payload = projection_completeness([], protocol=protocol).payload()
    assert payload["inventory_sha256"] is None
    assert payload["inventory_identity_verified"] is False
    assert payload["reserved_families_indexed_for_structure"] == []
    assert "no inventory was indexed" in payload["reserved_table_indexing"]

    with pytest.raises(ReferenceProjectionError, match="non-canonical protocol"):
        projection_completeness(
            [], protocol=dataclasses.replace(protocol, sha256="0" * 64)
        )


def test_the_payload_states_that_reserved_tables_were_indexed(protocol, inventory):
    """A non-reserved run still validated the reserved tables' structure."""
    projection = project_block(
        inventory,
        block_id="population_total",
        geography_id="5001900US0101",
        protocol=protocol,
    )
    payload = projection_completeness(
        [projection], protocol=protocol, inventory=inventory
    ).payload()
    assert payload["reserved_families_indexed_for_structure"] == ["B19001", "B25003"]
    assert payload["reserved_families_projected_into_values"] == []
    assert "no reserved" in payload["reserved_table_indexing"]

    reserved = project_block(
        inventory,
        block_id="tenure_distribution",
        geography_id="5001900US0101",
        protocol=protocol,
    )
    named = projection_completeness(
        [reserved], protocol=protocol, inventory=inventory
    ).payload()
    assert named["reserved_families_projected_into_values"] == ["B25003"]


# --------------------------------------------------------------------------
# The command line: bytes it can be held to.
# --------------------------------------------------------------------------


def test_the_command_line_writes_bytes_it_can_be_held_to(tmp_path, bundle, capsys):
    """The printed digest is the digest of the file, and no record is replaced.

    The printed line used to hash the canonical form of the payload while the
    file held an indented form, so hashing the artifact gave a different value
    from the one printed; and the output path was created with ``write_text``,
    which silently erased any prior record at that path.
    """
    path, sha256 = _write(tmp_path, bundle)
    output = tmp_path / "fresh" / "completeness.json"
    argv = [
        "--inventory",
        str(path),
        "--expect-sha256",
        sha256,
        "--aggregate-only",
        "--all-roster-districts",
        "--block",
        "population_total",
        "--output",
        str(output),
    ]

    assert main(argv) == 0
    written = output.read_bytes()
    printed = capsys.readouterr().out
    assert written.endswith(b"\n")
    assert hashlib.sha256(written).hexdigest() in printed
    assert f"{len(written)} bytes" in printed
    assert json.loads(written.decode("utf-8"))["kind"] == (
        "us_cd_reference_projection_completeness"
    )

    # A rerun refuses, and the record it refused to replace is byte-identical.
    assert main(argv) == 1
    assert "already exists" in capsys.readouterr().err
    assert output.read_bytes() == written

    # A second success takes a second path, deliberately.
    second = tmp_path / "fresh" / "completeness-second.json"
    assert main([*argv[:-1], str(second)]) == 0
    assert second.read_bytes() == written
    assert hashlib.sha256(second.read_bytes()).hexdigest() in capsys.readouterr().out


def test_the_command_line_judges_its_arguments_before_it_opens_the_inventory(
    tmp_path, bundle, capsys
):
    """A 38.7 MB read is not where a contradictory argument should surface."""
    _, sha256 = _write(tmp_path, bundle)
    absent = tmp_path / "not-here.json"
    base = [
        "--inventory",
        str(absent),
        "--expect-sha256",
        sha256,
        "--aggregate-only",
        "--block",
        "population_total",
    ]

    assert main([*base, "--all-roster-districts", "--district", "5001900US0101"]) == 1
    assert "not both" in capsys.readouterr().err

    assert main(base) == 1
    assert "there is no default" in capsys.readouterr().err

    assert (
        main([*base, "--district", "5001900US0101", "--district", "5001900US0101"]) == 1
    )
    assert "district was requested twice" in capsys.readouterr().err

    assert not absent.exists()

    # The same arguments over a real path get past the selector checks and are
    # refused by the loader instead, which is what proves the ordering.
    assert main([*base, "--district", "5001900US0101"]) == 1
    assert "No inventory file" in capsys.readouterr().err


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
