"""The reviewed-fill consumer register and its release gate (microcosm#1022).

Engine-free: the gate reads fill manifests and the packaged register only. The
engine-tier twin (tests/engine/us/test_us_acs_local_reviewed_fill_consumers.py)
recomputes each entry's consumers from the installed policyengine-us.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass

import pytest

from microcosm.build.spec_engine.errors import SpecParseError
from microcosm.build.us_runtime.acs_local_reviewed_fill_consumers import (
    ACS_LOCAL_REVIEWED_FILL_CONSUMER_GATE_NAME,
    ACS_LOCAL_REVIEWED_FILL_CONSUMER_REGISTER,
    ACS_LOCAL_REVIEWED_FILL_CONSUMERS_ISSUE,
    acs_local_reviewed_fill_consumer_gate,
    load_reviewed_fill_consumer_register,
    means_tested_consumers,
    reviewed_fill_register_failures,
    reviewed_fill_register_sha256,
    reviewed_fill_register_summary,
)

_ACS = "acs_2024_1yr"
_DONOR = "asec_puf"
_MANIFEST = "consumer_reviewed_null_fills.json"


def _register() -> dict:
    """A small register: one harmless-for-SNAP fill, one TANF known bias, and
    one fill with no means-tested consumer."""

    return {
        "schema_version": 1,
        "issue": ACS_LOCAL_REVIEWED_FILL_CONSUMERS_ISSUE,
        "reviewed_against": {"policyengine_us": "2.2.1"},
        "programs": {
            "snap": {"label": "SNAP", "roots": ["snap"]},
            "tanf": {"label": "TANF", "roots": ["tanf"]},
        },
        "entries": [
            {
                "entity": "person",
                "column": "weeks_unemployed",
                "fill_value": "0",
                "spines": [_ACS],
                "rows": "every ACS person",
                "summary": "Read only by state unemployment calculators.",
                "consumers": [],
                "notes": [],
            },
            {
                "entity": "spm_unit",
                "column": "invented_energy_amount",
                "fill_value": "0",
                "spines": [_ACS],
                "rows": "every ACS SPM unit",
                "summary": "An invented input SNAP reads.",
                "consumers": ["snap"],
                "notes": [
                    {
                        "programs": ["snap"],
                        "verdict": "harmless",
                        "note": "SNAP reads has_heating_cooling_expense, not this.",
                    }
                ],
            },
            {
                "entity": "household",
                "column": "household_vehicles_value",
                "fill_value": "0",
                "spines": [_ACS],
                "rows": "every ACS household",
                "summary": "No ACS value item.",
                "consumers": ["snap", "tanf"],
                "notes": [
                    {
                        "programs": ["tanf"],
                        "verdict": "known_bias",
                        "direction": "overstates",
                        "note": "Zero equity passes six state TANF resource tests.",
                    },
                    {
                        "programs": ["snap"],
                        "verdict": "known_bias",
                        "direction": "mixed",
                        "via": ["tanf"],
                        "note": "Texas BBCE, and TANF income elsewhere.",
                    },
                ],
            },
        ],
    }


def _fill(entity, column, fill_value="0", spines=(_ACS,), rows=10) -> dict:
    return {
        "entity": entity,
        "column": column,
        "filled_rows": rows,
        "rows": 100,
        "fill_value": fill_value,
        "fill_kind": "int",
        "register_missing_rows": rows,
        "missing_rows_by_spine": {spine: rows for spine in spines},
    }


def _manifest(*fills) -> dict:
    return {
        "period": 2024,
        "columns_filled": len(fills),
        "count_mismatch_warnings": [],
        "unregistered_violations": [],
        "fills": list(fills),
    }


def _applied() -> dict:
    return {
        _MANIFEST: _manifest(
            _fill("person", "weeks_unemployed"),
            _fill("spm_unit", "invented_energy_amount"),
            _fill("household", "household_vehicles_value"),
        )
    }


def _gate(manifests=None, document=None, **kwargs):
    return acs_local_reviewed_fill_consumer_gate(
        _applied() if manifests is None else manifests,
        document=_register() if document is None else document,
        **kwargs,
    )


def test_the_passing_case_records_each_fill_and_its_verdicts() -> None:
    gate = _gate(installed_engine_version="2.2.1")

    assert gate.name == ACS_LOCAL_REVIEWED_FILL_CONSUMER_GATE_NAME
    assert gate.passed, gate.failures
    assert gate.details["applied_fills"] == 3
    assert gate.details["register_sha256"] == reviewed_fill_register_sha256(_register())
    assert gate.details["installed_policyengine_us"] == "2.2.1"
    assert gate.details["reviewed_against"] == {"policyengine_us": "2.2.1"}
    vehicles = gate.details["fills"]["household.household_vehicles_value"]
    assert vehicles["known_bias"] == {"snap": "mixed", "tanf": "overstates"}
    assert vehicles["harmless"] == []
    assert vehicles["filled_rows"] == {_MANIFEST: 10}
    energy = gate.details["fills"]["spm_unit.invented_energy_amount"]
    assert energy["harmless"] == ["snap"]
    assert gate.details["known_bias"] == {
        "household.household_vehicles_value": {"snap": "mixed", "tanf": "overstates"}
    }
    assert gate.details["unused_entries"] == []
    assert "warnings" not in gate.details


def test_a_fill_without_an_entry_fails() -> None:
    manifests = _applied()
    manifests[_MANIFEST]["fills"].append(_fill("person", "invented_new_input"))

    gate = _gate(manifests)

    assert not gate.passed
    (failure,) = gate.failures
    assert failure.startswith("person.invented_new_input: a reviewed default fill (0)")
    assert ACS_LOCAL_REVIEWED_FILL_CONSUMER_REGISTER in failure


def test_an_entry_under_another_entity_does_not_cover_the_fill() -> None:
    manifests = {_MANIFEST: _manifest(_fill("household", "weeks_unemployed"))}

    gate = _gate(manifests)

    assert not gate.passed
    assert gate.failures[0].startswith("household.weeks_unemployed: a reviewed")


def test_a_declared_consumer_without_a_note_fails() -> None:
    document = _register()
    vehicles = document["entries"][2]
    vehicles["notes"] = [vehicles["notes"][0]]  # SNAP keeps no note.

    gate = _gate(document=document)

    assert not gate.passed
    assert gate.failures == (
        "register: household.household_vehicles_value: declared consumer 'snap' "
        "has no harmless or known_bias note.",
    )


def test_a_note_for_an_undeclared_consumer_fails() -> None:
    document = _register()
    document["entries"][0]["notes"] = [
        {"programs": ["snap"], "verdict": "harmless", "note": "Not read."}
    ]

    failures = reviewed_fill_register_failures(document)

    assert failures == [
        "person.weeks_unemployed note 0 has a note for 'snap', which is not a "
        "declared consumer of the column."
    ]


@pytest.mark.parametrize(
    ("mutate", "expected"),
    [
        (
            lambda note: note.pop("direction"),
            "is a known_bias note without a direction",
        ),
        (
            lambda note: note.update(verdict="harmless"),
            "is harmless but carries a direction",
        ),
        (lambda note: note.update(verdict="fine"), "verdict 'fine' is not one of"),
        (lambda note: note.update(note="  "), "has no note text"),
        (lambda note: note.update(programs=[]), "programs must be a non-empty list"),
        (lambda note: note.update(extra="x"), "carries unknown keys ['extra']"),
    ],
    ids=["no-direction", "harmless-direction", "verdict", "blank", "empty", "key"],
)
def test_every_note_is_well_formed(mutate, expected) -> None:
    document = _register()
    mutate(document["entries"][2]["notes"][0])

    failures = reviewed_fill_register_failures(document)

    assert any(expected in failure for failure in failures), failures


def test_via_must_name_another_declared_consumer() -> None:
    document = _register()
    document["entries"][2]["notes"][1]["via"] = ["snap"]

    failures = reviewed_fill_register_failures(document)

    assert failures == [
        "household.household_vehicles_value note 1 via ['snap'] must name other "
        "declared consumers."
    ]


def test_a_consumer_covered_twice_fails() -> None:
    document = _register()
    document["entries"][2]["notes"][1]["programs"] = ["snap", "tanf"]
    document["entries"][2]["notes"][1].pop("via")

    failures = reviewed_fill_register_failures(document)

    assert failures == [
        "household.household_vehicles_value: consumer 'tanf' has 2 notes."
    ]


def test_an_entry_the_release_did_not_use_is_reported_not_failed() -> None:
    """Which cells are missing depends on the data: a capped smoke can hold no
    group-quarters household, so a GQ-only fill does not happen. An unused
    entry cannot hide an unreviewed fill, so it warns."""

    manifests = {_MANIFEST: _manifest(_fill("person", "weeks_unemployed"))}

    gate = _gate(manifests)

    assert gate.passed, gate.failures
    assert gate.details["unused_entries"] == [
        "household.household_vehicles_value",
        "spm_unit.invented_energy_amount",
    ]
    assert gate.details["warnings"] == [
        "2 register entries describe a fill this release did not apply "
        "(reported, not failed)."
    ]


def test_an_entry_for_a_never_default_filled_column_fails() -> None:
    """A column the release tool refuses to default-fill cannot be a reviewed
    fill, so its entry is stale by construction (the ACS vehicle count left
    NEVER_DEFAULT_FILLED in the change that added its entry, after the
    microcosm#1064 review)."""

    never = {("household", "household_vehicles_value")}

    gate = _gate(never_default_filled=never)

    assert not gate.passed
    assert gate.failures == (
        "register: household.household_vehicles_value: the release tool never "
        "default-fills this column (NEVER_DEFAULT_FILLED), so the entry is stale; "
        "remove it.",
    )


def test_a_fill_value_other_than_the_reviewed_one_fails() -> None:
    manifests = {_MANIFEST: _manifest(_fill("person", "weeks_unemployed", "1"))}

    gate = _gate(manifests)

    assert gate.failures == (
        "person.weeks_unemployed: the release filled 1 but the entry reviewed 0.",
    )


def test_a_fill_on_a_spine_the_notes_do_not_cover_fails() -> None:
    manifests = {
        _MANIFEST: _manifest(_fill("person", "weeks_unemployed", spines=(_ACS, _DONOR)))
    }

    gate = _gate(manifests)

    assert gate.failures == (
        f"person.weeks_unemployed: filled on spine(s) ['{_DONOR}'] the entry's "
        f"notes do not cover (reviewed for ['{_ACS}']).",
    )


def test_both_engine_passes_are_graded_together() -> None:
    manifests = {
        "reviewed_null_fills.json": _manifest(_fill("person", "weeks_unemployed")),
        _MANIFEST: _manifest(_fill("person", "weeks_unemployed", rows=7)),
    }

    gate = _gate(manifests)

    assert gate.passed, gate.failures
    assert gate.details["fills"]["person.weeks_unemployed"]["filled_rows"] == {
        "reviewed_null_fills.json": 10,
        _MANIFEST: 7,
    }
    manifests[_MANIFEST]["fills"][0]["fill_value"] = "1"
    assert (
        "the manifests disagree on the fill value (0 vs 1)"
        in _gate(manifests).failures[0]
    )


@pytest.mark.parametrize(
    ("manifests", "expected"),
    [
        ({}, "no reviewed-null fill manifest was supplied."),
        (
            {_MANIFEST: {}},
            f"{_MANIFEST} records no fills list, so the fills the engine pass "
            "applied cannot be established; re-run the stage that writes it.",
        ),
        (
            {_MANIFEST: {"columns_filled": []}},
            f"{_MANIFEST} records no fills list, so the fills the engine pass "
            "applied cannot be established; re-run the stage that writes it.",
        ),
        (
            {_MANIFEST: {**_manifest(), "unregistered_violations": [{"x": 1}]}},
            f"{_MANIFEST} records unregistered null violations.",
        ),
        (
            {_MANIFEST: _manifest({"entity": "person", "column": "age"})},
            f"{_MANIFEST} holds a malformed fill record",
        ),
    ],
    ids=["none", "empty", "pre-ledger", "violations", "malformed"],
)
def test_a_manifest_that_cannot_say_what_was_filled_fails(manifests, expected):
    gate = _gate(manifests)

    assert not gate.passed
    assert gate.failures[0].startswith(expected)


@pytest.mark.parametrize(
    ("mutate", "expected"),
    [
        (
            lambda doc: doc["entries"].append(copy.deepcopy(doc["entries"][0])),
            "person.weeks_unemployed: more than one entry for the column.",
        ),
        (
            lambda doc: doc["entries"][1]["consumers"].append("wic"),
            "spm_unit.invented_energy_amount: unknown consumer program(s) ['wic'].",
        ),
        (
            lambda doc: doc["programs"]["tanf"]["roots"].append("snap"),
            "root 'snap' belongs to both 'snap' and 'tanf'.",
        ),
        (
            lambda doc: doc["entries"][0].update(entity="benunit"),
            "benunit.weeks_unemployed: unknown entity 'benunit'.",
        ),
        (
            lambda doc: doc["entries"][0].update(fill_value=0),
            "person.weeks_unemployed: fill_value must be the manifest's repr string.",
        ),
        (
            lambda doc: doc["entries"][0].pop("summary"),
            "entry 0 must carry exactly",
        ),
        (lambda doc: doc.update(schema_version=2), "schema_version must be 1."),
        (
            lambda doc: doc.update(reviewed_against={}),
            "reviewed_against.policyengine_us must name the engine.",
        ),
    ],
    ids=[
        "duplicate",
        "unknown-program",
        "shared-root",
        "entity",
        "fill-value",
        "keys",
        "schema",
        "engine",
    ],
)
def test_the_register_is_validated(mutate, expected) -> None:
    document = _register()
    mutate(document)

    failures = reviewed_fill_register_failures(document)

    assert any(failure.startswith(expected) for failure in failures), failures
    assert not _gate(document=document).passed


def test_the_digest_binds_content_not_layout() -> None:
    document = _register()
    reordered = dict(reversed(list(document.items())))
    changed = _register()
    changed["entries"][1]["notes"][0]["note"] = "SNAP reads something else."

    assert reviewed_fill_register_sha256(reordered) == reviewed_fill_register_sha256(
        document
    )
    assert reviewed_fill_register_sha256(changed) != reviewed_fill_register_sha256(
        document
    )


def test_the_summary_lists_the_known_biases() -> None:
    summary = reviewed_fill_register_summary(_register())

    assert summary == {
        "issue": ACS_LOCAL_REVIEWED_FILL_CONSUMERS_ISSUE,
        "register": ACS_LOCAL_REVIEWED_FILL_CONSUMER_REGISTER,
        "sha256": reviewed_fill_register_sha256(_register()),
        "reviewed_against": {"policyengine_us": "2.2.1"},
        "entries": 3,
        "known_bias": {"household.household_vehicles_value": ["snap", "tanf"]},
    }


# ---------------------------------------------------------------------------
# The committed register
# ---------------------------------------------------------------------------


def test_the_committed_register_is_sound() -> None:
    document = load_reviewed_fill_consumer_register()

    assert reviewed_fill_register_failures(document) == []
    columns = [entry["column"] for entry in document["entries"]]
    # The release tool's test pins the columns against release 767312d60 and
    # NEVER_DEFAULT_FILLED (test_the_register_covers_every_remaining_reviewed_fill).
    assert len(columns) == len(set(columns))
    assert document["reviewed_against"] == {"policyengine_us": "2.2.1"}


def test_the_committed_register_carries_the_triage_verdicts_for_snap() -> None:
    """microcosm#1022's triage: these defaults are harmless for SNAP (SNAP is
    no consumer, or its note says harmless), and the vehicle value is not."""

    document = load_reviewed_fill_consumer_register()
    entries = {entry["column"]: entry for entry in document["entries"]}

    def snap_verdict(column):
        for note in entries[column]["notes"]:
            if "snap" in note["programs"]:
                return note["verdict"]
        assert "snap" not in entries[column]["consumers"]
        return "not a consumer"

    for column in (
        "weeks_unemployed",
        "spm_unit_energy_subsidy",
        "net_worth",
        "health_insurance_premiums",
        "real_estate_taxes",
        "pre_subsidy_rent",
        "tenure_type",
        "spm_unit_tenure_type",
    ):
        assert snap_verdict(column) in {"harmless", "not a consumer"}, column
    assert snap_verdict("household_vehicles_value") == "known_bias"
    energy = entries["spm_unit_energy_subsidy"]
    assert energy["consumers"] == ["liheap"]
    assert energy["notes"][0]["verdict"] == "known_bias"
    tanf = next(
        note
        for note in entries["household_vehicles_value"]["notes"]
        if "tanf" in note["programs"]
    )
    assert (tanf["verdict"], tanf["direction"]) == ("known_bias", "overstates")


def test_the_committed_register_notes_the_acs_owned_vehicle_count() -> None:
    """microcosm#1064 review: ACS VEH counts vehicles available, so the owned
    count is a reviewed fill at 0. Its count-only readers (San Francisco CAAP,
    Los Angeles General Relief) overstate eligibility; it is harmless for SNAP
    while the vehicle value is also 0."""

    document = load_reviewed_fill_consumer_register()
    (entry,) = [
        entry
        for entry in document["entries"]
        if entry["column"] == "household_vehicles_owned"
    ]
    assert (entry["entity"], entry["fill_value"], entry["spines"]) == (
        "household",
        "0",
        ["acs_2024_1yr"],
    )
    verdicts = {
        program: (note["verdict"], note.get("direction"))
        for note in entry["notes"]
        for program in note["programs"]
    }
    assert set(verdicts) == set(entry["consumers"])
    assert verdicts["general_assistance"] == ("known_bias", "overstates")
    assert verdicts["snap"] == ("harmless", None)
    text = " ".join([entry["summary"], *(note["note"] for note in entry["notes"])])
    for fragment in (
        "vehicles available",
        "microcosm#1064 review",
        "ca_sf_caap_vehicle_eligible",
        "la_general_relief_motor_vehicle_value_eligible",
        "lives_in_vehicle",
        "meets_tanf_non_cash_asset_test",
    ):
        assert fragment in text, fragment


def test_the_loader_refuses_duplicate_keys(tmp_path) -> None:
    path = tmp_path / "register.yaml"
    path.write_text("schema_version: 1\nschema_version: 1\n")

    with pytest.raises(SpecParseError):
        load_reviewed_fill_consumer_register(path)


# ---------------------------------------------------------------------------
# The dependency walk
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _Receipt:
    consumer: str


class _Index:
    """column -> [consumers]; an unknown name raises like the engine index."""

    def __init__(self, graph):
        self._graph = graph

    def consumer_receipts(self, name):
        if name not in self._graph:
            raise ValueError(f"Unknown PolicyEngine-US source variable {name!r}.")
        return tuple(_Receipt(consumer) for consumer in self._graph[name])


_PROGRAMS = {"snap": ["snap"], "tanf": ["tanf"], "liheap": ["energy_subsidy"]}


def test_the_walk_reaches_programs_through_other_programs() -> None:
    index = _Index(
        {
            "vehicle_value": ["tx_bbce_asset_test", "tx_tanf_resources"],
            "tx_bbce_asset_test": ["snap"],
            "tx_tanf_resources": ["tanf"],
            "tanf": ["snap_unearned_income"],
            "snap_unearned_income": ["snap"],
            "snap": [],
        }
    )

    reached = means_tested_consumers("vehicle_value", index, _PROGRAMS)

    assert reached == {
        "snap": ("vehicle_value", "tx_bbce_asset_test", "snap"),
        "tanf": ("vehicle_value", "tx_tanf_resources", "tanf"),
    }


def test_a_column_that_is_a_root_reaches_its_own_program() -> None:
    index = _Index({"energy_subsidy": ["spm_unit_benefits"]})

    assert means_tested_consumers("energy_subsidy", index, _PROGRAMS) == {
        "liheap": ("energy_subsidy",)
    }


def test_the_walk_stops_at_variables_the_index_does_not_know() -> None:
    index = _Index({"weeks_unemployed": ["al_ui"]})

    assert means_tested_consumers("weeks_unemployed", index, _PROGRAMS) == {}
    with pytest.raises(ValueError, match="Unknown"):
        means_tested_consumers("not_a_variable", index, _PROGRAMS)
