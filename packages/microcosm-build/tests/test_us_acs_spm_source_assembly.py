"""Invented source rosters only; no population files or country calculation."""

import hashlib
from dataclasses import replace
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import pandas as pd
import pytest

from microcosm.build import acs_spm_partition as partition
from microcosm.build.acs_spm_source_assembly import (
    RECEIPT_KEY,
    AcsSpmSourceAssemblyOptions,
    assemble_acs_spm_source,
)
from microcosm.frame import US_SCHEMA, Frame, WeightKind, Weights

OPTIONS = AcsSpmSourceAssemblyOptions(partition.ACS_SPM_DEVELOPMENT_POLICY, True)
CEILING = 10_000
requires_assembler = pytest.mark.xfail(
    not partition.probe_acs_spm_assembler().supported,
    raises=partition.UnsupportedAssembler,
    strict=True,
    reason="Explicit source construction requires the canonical assembler capability",
)


def source_frame(relations=(20, 34, 36), ages=(45, 30, 16), *, kind=1, offset=0):
    count = len(relations)
    people = pd.DataFrame(
        {
            "person_id": np.arange(count, dtype=np.int64) + 100 + offset,
            "person_household_id": np.full(count, 10 + offset, dtype=np.int64),
            "person_spm_unit_id": np.full(count, 20 + offset, dtype=np.int64),
            "person_tax_unit_id": np.arange(count, dtype=np.int64) + 30 + offset,
            "person_family_id": np.full(count, 40 + offset, dtype=np.int64),
            "person_marital_unit_id": np.arange(count, dtype=np.int64) + 50 + offset,
            "SPORDER": pd.Series([str(i + 1) for i in range(count)], dtype="string"),
            "RELSHIPP": relations,
            "AGEP": ages,
            "MAR": [5] * count,
            "is_household_head": [value == 20 for value in relations],
            "tax_unit_role_input": ["HEAD"] * count,
            "is_related_to_head_or_spouse": [value == 20 for value in relations],
            "source_year": [2024] * count,
            "native_income": pd.Series([1.5] * count, dtype="Float64"),
        }
    )
    tables = {"person": people}
    for entity in US_SCHEMA.group_entities:
        tables[entity] = pd.DataFrame(
            {f"{entity}_id": sorted(set(people[f"person_{entity}_id"]))}, dtype="int64"
        )
    tables["tax_unit"]["filing_status"] = "SINGLE"
    tables["household"]["SERIALNO"] = f"2024TEST{offset}"
    tables["household"]["TYPEHUGQ"] = kind
    tables["household"]["NP"] = count
    tables["household"]["TEN"] = 3 if kind == 1 else pd.NA
    tables["household"]["geography_unmodified"] = "06001"
    return Frame(
        tables,
        US_SCHEMA,
        {"household": Weights([7.25], WeightKind.DESIGN)},
        pd.Series(["acs_2024_1yr"] * count),
        metadata={"original": {"kept": True}},
    )


def rebuild(frame, *, tables=None, weights=None):
    return Frame(
        tables or {entity: frame.table(entity) for entity in frame.entities},
        frame.schema,
        weights or dict(frame._weights),
        frame.strata,
        metadata=frame.metadata,
        mass_log=frame.mass_log,
    )


def assemble(frame, **kwargs):
    return assemble_acs_spm_source(
        frame,
        options=kwargs.pop("options", OPTIONS),
        id_ceiling=kwargs.pop("id_ceiling", CEILING),
        **kwargs,
    )


def assert_preserved(before, after):
    for entity in before.entities:
        if entity == "spm_unit":
            continue
        columns = [
            name for name in before.table(entity) if name != "person_spm_unit_id"
        ]
        pd.testing.assert_frame_equal(
            before.table(entity)[columns],
            after.table(entity)[columns],
            check_exact=True,
        )
    pd.testing.assert_series_equal(before.strata, after.strata, check_exact=True)
    np.testing.assert_array_equal(
        before.weights_for("household").values, after.weights_for("household").values
    )
    assert after.weights_for("household").kind is before.weights_for("household").kind
    assert before.mass_log == after.mass_log
    assert before.metadata["original"] == after.metadata["original"]
    assert "is_spm_independent_minor_role" not in after.person
    assert "spm_unit_spm_universe_status" not in after.table("spm_unit")


@requires_assembler
def test_split_preserves_all_tax_inputs_and_returns_separate_development_evidence():
    frame = source_frame()
    original = rebuild(frame)
    result = assemble(frame)
    assert result.frame.n("spm_unit") == 3
    assert_preserved(frame, result.frame)
    for entity in frame.entities:
        pd.testing.assert_frame_equal(
            frame.table(entity), original.table(entity), check_exact=True
        )
    assert frame.metadata == original.metadata
    assert RECEIPT_KEY not in frame.metadata
    assert result.receipt["source_file_authentication"] is False
    assert result.receipt["engine_role_delivered"] is False
    assert result.receipt["annual_universe_declared"] is False
    assert result.receipt["unit_amounts_copied"] is False
    assert result.receipt["units_with_household_source_uncertainty"] > 0
    assert result.unit_evidence.household_source_uncertain.all()
    assert "modeled_assumption" in set(result.partition.membership.role_source)
    assert result.receipt["assembler"]["supported"] is True
    assert (
        result.receipt["implementation_file_sha256"]["partition_helper"]
        == hashlib.sha256(Path(partition.__file__).read_bytes()).hexdigest()
    )
    assert result.registry.id_ceiling == CEILING
    assert (
        result.frame.metadata[RECEIPT_KEY]["registry_sha256"] == result.registry.sha256
    )


@requires_assembler
@pytest.mark.parametrize(
    "relations,ages,count",
    [
        ((20, 25), (45, 12), 1),
        ((20, 36), (45, 12), 1),
        ((20, 34), (45, 30), 2),
        ((20, 36), (45, 21), 2),
        ((20, 35), (45, 21), 1),
        ((20, 35), (45, 22), 2),
    ],
)
def test_native_rules_and_foster_boundary(relations, ages, count):
    frame = source_frame(relations, ages)
    result = assemble(frame)
    assert result.frame.n("spm_unit") == count
    assert_preserved(frame, result.frame)
    if count == 1:
        assert result.frame.person.person_spm_unit_id.eq(20).all()
    if relations == (20, 36) and ages == (45, 12):
        assert result.partition.membership.partition_assumption.eq(
            "parent_unknown_reference_pooling"
        ).any()
        assert result.partition.links.empty


@requires_assembler
@pytest.mark.parametrize("kind,relation", [(2, 37), (3, 38)])
def test_gq_retains_membership_and_missing_roles(kind, relation):
    frame = source_frame((relation,), (30,), kind=kind)
    result = assemble(frame)
    pd.testing.assert_frame_equal(frame.person, result.frame.person, check_exact=True)
    assert result.partition.membership.independent_minor_role.isna().all()
    assert result.unit_evidence.outside_acs_household_universe.all()
    assert_preserved(frame, result.frame)


@requires_assembler
def test_partner_sensitivity_changes_only_returned_role_evidence():
    frame = source_frame((20, 22), (45, 16))
    yes = assemble(frame)
    no = assemble(frame, options=replace(OPTIONS, minor_partner_role=False))
    for entity in frame.entities:
        pd.testing.assert_frame_equal(
            yes.frame.table(entity), no.frame.table(entity), check_exact=True
        )
    assert yes.partition.membership.independent_minor_role.iloc[1]
    assert not no.partition.membership.independent_minor_role.iloc[1]
    assert yes.registry.sha256 != no.registry.sha256


def paired_frame():
    first, second = source_frame(), source_frame((20, 25), (40, 7), offset=1000)
    tables = {
        entity: pd.concat(
            [first.table(entity), second.table(entity)], ignore_index=True
        )
        for entity in first.entities
    }
    return Frame(
        tables,
        US_SCHEMA,
        {"household": Weights([7.25, 11.5], WeightKind.DESIGN)},
        pd.Series(["acs_2024_1yr"] * 5),
        metadata=first.metadata,
    )


@requires_assembler
def test_one_native_registry_applies_to_shuffle_and_complete_household_chunks():
    frame = paired_frame()
    whole = assemble(frame)
    tables = {entity: frame.table(entity).copy() for entity in frame.entities}
    tables["person"] = tables["person"].sample(frac=1, random_state=3)
    shuffled = Frame(
        tables,
        US_SCHEMA,
        dict(frame._weights),
        frame.strata.reindex(tables["person"].index),
        metadata=frame.metadata,
    )
    replay = assemble(shuffled, registry=whole.registry)
    pd.testing.assert_series_equal(
        whole.frame.person.set_index("person_id").person_spm_unit_id.sort_index(),
        replay.frame.person.set_index("person_id").person_spm_unit_id.sort_index(),
    )
    assert assemble(shuffled).registry == whole.registry
    for household_id in frame.table("household").household_id:
        chunk = frame.select(
            frame.person.person_household_id.eq(household_id).to_numpy()
        )
        applied = assemble(chunk, registry=whole.registry)
        assert applied.receipt["registry_persons"] == 5
        assert applied.registry == whole.registry
        expected = whole.frame.select(
            whole.frame.person.person_household_id.eq(household_id).to_numpy()
        )
        for entity in expected.entities:
            pd.testing.assert_frame_equal(
                applied.frame.table(entity), expected.table(entity), check_exact=True
            )


@requires_assembler
def test_ids_remain_exact_above_float_integer_precision_and_selection_is_bound():
    frame = source_frame(offset=2**53)
    result = assemble(frame, id_ceiling=np.iinfo(np.int64).max)
    assert result.frame.table("spm_unit").spm_unit_id.dtype == np.dtype("int64")
    assert result.crosswalk.person_id.tolist() == frame.person.person_id.tolist()
    other = source_frame()
    with pytest.raises(ValueError, match="REGISTRY_ROSTER_DOMAIN"):
        assemble(other, registry=result.registry, id_ceiling=np.iinfo(np.int64).max)


@pytest.mark.parametrize("value", [True, np.bool_(False), 3.0, -1, 2**63])
def test_ceiling_requires_exact_int64(value):
    with pytest.raises(ValueError, match="EXACT_INTEGER"):
        assemble(source_frame(), id_ceiling=value)


@requires_assembler
def test_registry_overflow_cannot_wrap_or_collide():
    with pytest.raises(ValueError, match="REGISTRY_ID_OVERFLOW"):
        assemble(source_frame(), id_ceiling=21)
    result = assemble(source_frame())
    entries = list(result.registry.entries)
    entries[1] = replace(entries[1], new_spm_unit_id=entries[0].new_spm_unit_id)
    with pytest.raises(ValueError, match="REGISTRY_ASSIGNMENT"):
        assemble(
            source_frame(), registry=replace(result.registry, entries=tuple(entries))
        )


@requires_assembler
@pytest.mark.parametrize("change", ["source", "assembler", "options", "native_key"])
def test_registry_rejects_changed_domain_source_or_contract(change):
    frame = source_frame()
    registry = assemble(frame).registry
    if change == "source":
        frame.person.loc[0, "AGEP"] += 1
    elif change == "assembler":
        registry = replace(registry, assembler_sha256="0" * 64)
    elif change == "options":
        registry = replace(registry, options=replace(OPTIONS, minor_partner_role=False))
    else:
        frame.table("household").loc[0, "SERIALNO"] = "DIFFERENT_NATIVE_DOMAIN"
    with pytest.raises(ValueError, match="REGISTRY_"):
        assemble(frame, registry=registry)


@pytest.mark.parametrize(
    "field", ["childcare", "receives_housing_assistance", "spm_unit_source_id"]
)
def test_existing_spm_values_are_never_regrouped_or_copied(field):
    frame = source_frame()
    frame.table("spm_unit")[field] = 0
    with pytest.raises(ValueError, match="ID_ONLY_SPM_TABLE"):
        assemble(frame)


@pytest.mark.parametrize(
    "defect",
    [
        "partial_np",
        "duplicate_native",
        "bad_code",
        "gq_count",
        "universe_conflict",
        "engine_role",
        "nonacs",
        "wrong_weight",
    ],
)
def test_source_refusals_are_not_hidden_by_unsupported_runtime(defect):
    frame = source_frame()
    if defect == "partial_np":
        frame.table("household").loc[0, "NP"] += 1
    elif defect == "duplicate_native":
        frame.person.loc[1, "SPORDER"] = "1"
    elif defect == "bad_code":
        frame.person.loc[0, "RELSHIPP"] = 99
    elif defect == "gq_count":
        frame.table("household")["TYPEHUGQ"] = 2
    elif defect == "universe_conflict":
        frame.person["TYPEHUGQ"] = 2
    elif defect == "engine_role":
        frame.person["is_spm_independent_minor_role"] = False
    elif defect == "nonacs":
        frame.person["PERIDNUM"] = "asec-person"
    elif defect == "wrong_weight":
        frame = rebuild(
            frame, weights={"household": Weights([7.25], WeightKind.IMPORTANCE)}
        )
    with pytest.raises(ValueError):
        assemble(frame)


@pytest.mark.parametrize(
    "defect",
    ["duplicate_person", "orphan_spm", "duplicate_household", "missing_native_key"],
)
def test_mutated_frame_identity_refuses_before_assembler(defect):
    frame = source_frame()
    if defect == "duplicate_person":
        frame.person.loc[1, "person_id"] = frame.person.loc[0, "person_id"]
    elif defect == "orphan_spm":
        frame.person.loc[1, "person_spm_unit_id"] = 999
    elif defect == "duplicate_household":
        frame._tables["household"] = pd.concat(
            [frame.table("household")] * 2, ignore_index=True
        )
    else:
        frame.table("household").drop(columns="SERIALNO", inplace=True)
    with pytest.raises(ValueError):
        assemble(frame)


def test_options_refuse_implicit_or_unknown_choices():
    with pytest.raises(ValueError, match="PARTNER_ROLE_BOOL"):
        AcsSpmSourceAssemblyOptions(partition.ACS_SPM_DEVELOPMENT_POLICY, 1)
    with pytest.raises(ValueError, match="POLICY"):
        AcsSpmSourceAssemblyOptions("invented-policy", True)


def test_explicit_request_never_falls_back_when_assembler_is_unavailable(monkeypatch):
    probe = partition.AcsSpmAssemblerProbe(False, "assembler_unavailable", None, None)
    monkeypatch.setattr(partition, "probe_acs_spm_assembler", lambda: probe)
    with pytest.raises(partition.UnsupportedAssembler):
        assemble(source_frame())


def invented_archives(tmp_path):
    from microcosm.build.us_runtime.acs_pums import AcsPumsSource

    household = {
        "SERIALNO": "2024TEST001",
        "ST": "06",
        "PUMA": "00100",
        "WGTP": 10,
        "NP": 3,
        "ADJHSG": 1_000_000,
        "TEN": 3,
        "RNTP": 100,
        "GRNTP": 150,
        "TAXAMT": 0,
        "TYPEHUGQ": 1,
    }
    persons = []
    for order, relation, age in ((1, 20, 45), (2, 34, 30), (3, 36, 16)):
        persons.append(
            {
                "SERIALNO": household["SERIALNO"],
                "SPORDER": order,
                "RELSHIPP": relation,
                "AGEP": age,
                "SEX": 1,
                "MAR": 5,
                "ADJINC": 1_000_000,
                "WAGP": 50_000,
                "SEMP": 0,
                "SSP": 0,
                "SSIP": 0,
                "RETP": 0,
                "INTP": 0,
                "PWGTP": 10,
            }
        )
    for name, member, rows in (
        ("csv_hus.zip", "psam_husa.csv", [household]),
        ("csv_pus.zip", "psam_pusa.csv", persons),
    ):
        with ZipFile(tmp_path / name, "w") as archive:
            archive.writestr(member, pd.DataFrame(rows).to_csv(index=False))
    return AcsPumsSource(tmp_path / "csv_hus.zip", tmp_path / "csv_pus.zip")


@requires_assembler
def test_real_constructor_maps_tenure_on_every_split_and_housing_only_to_head(tmp_path):
    from microcosm.build.us_runtime.acs_inputs import map_acs_native_inputs
    from microcosm.build.us_runtime.acs_pums import build_acs_pums_unit_frame
    from microcosm.build.us_runtime.housing_participation import route_participation
    from microcosm.build.us_runtime.operator_column_contracts import (
        PUF_SUPPORT_MAX_CLONE_SAFE_SOURCE_ID,
    )

    source = invented_archives(tmp_path)
    old, old_receipt = build_acs_pums_unit_frame(source, chunksize=1)
    new, receipt = build_acs_pums_unit_frame(
        source, chunksize=1, spm_construction=OPTIONS
    )
    # The constructor does not change unrelated frame metadata or inputs.
    for entity in old.entities:
        if entity == "spm_unit":
            continue
        columns = [name for name in old.table(entity) if name != "person_spm_unit_id"]
        pd.testing.assert_frame_equal(
            old.table(entity)[columns], new.table(entity)[columns], check_exact=True
        )
    assert old.n("spm_unit") == 1 and new.n("spm_unit") == 3
    assert RECEIPT_KEY not in old_receipt
    assert (
        receipt[RECEIPT_KEY]["registry_id_ceiling"]
        == PUF_SUPPORT_MAX_CLONE_SAFE_SOURCE_ID
    )
    mapped = map_acs_native_inputs(new).frame
    assert mapped.table("spm_unit").spm_unit_tenure_type.eq("RENTER").all()
    mapped.person["is_household_head"] = mapped.person.RELSHIPP.eq(20)
    flags = route_participation(
        mapped, pd.Series(True, index=mapped.table("household").household_id)
    )
    head_spm = mapped.person.loc[
        mapped.person.RELSHIPP.eq(20), "person_spm_unit_id"
    ].iloc[0]
    assert flags.receives_housing_assistance.sum() == 1
    assert bool(flags.loc[head_spm, "receives_housing_assistance"])
    assert flags.takes_up_housing_assistance_if_eligible.equals(
        flags.receives_housing_assistance
    )


def test_constructor_refuses_before_source_io_and_default_does_not_probe(
    tmp_path, monkeypatch
):
    from microcosm.build.us_runtime import acs_pums

    source = invented_archives(tmp_path)
    probe = partition.AcsSpmAssemblerProbe(
        False, "incompatible_parent_link_order", "invented", "0" * 64
    )
    calls = []

    def unavailable():
        calls.append(True)
        return probe

    monkeypatch.setattr(partition, "probe_acs_spm_assembler", unavailable)
    old, receipt = acs_pums.build_acs_pums_unit_frame(source)
    assert calls == [] and old.n("spm_unit") == 1 and RECEIPT_KEY not in receipt

    def forbidden(*args, **kwargs):
        raise AssertionError("Source reader reached after unsupported capability")

    monkeypatch.setattr(acs_pums, "load_acs_pums_tables", forbidden)
    with pytest.raises(partition.UnsupportedAssembler):
        acs_pums.build_acs_pums_unit_frame(source, spm_construction=OPTIONS)
    assert calls == [True]


@requires_assembler
def test_registry_component_and_duplicate_native_key_corruption_refuse():
    frame = source_frame()
    registry = assemble(frame).registry
    changed = replace(registry.entries[0], component_sha256="0" * 64)
    with pytest.raises(ValueError, match="REGISTRY_COMPONENT_IDENTITY"):
        assemble(
            frame, registry=replace(registry, entries=(changed, *registry.entries[1:]))
        )
    with pytest.raises(ValueError, match="REGISTRY_NATIVE_KEYS"):
        assemble(
            frame,
            registry=replace(
                registry, entries=(*registry.entries, registry.entries[-1])
            ),
        )
