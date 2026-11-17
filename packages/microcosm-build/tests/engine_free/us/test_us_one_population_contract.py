"""d2283 PR 2: the one-population contract library, phase 0 (the foundation).

These tests pin the module's surface and engine-free import, the injected
engine variable index (the default adapter's rules over a fake inner index,
and the stand-in built from the engine ABI lock), PR 1's public key digest,
and the generated population every later clause test corrupts: uncorrupted,
it passes the SPM composition check, the SPM role signal gate (bands
included) with a record computed by the gate's rule, and the assembly
validator with a clone-0, three-channel record.
"""

from __future__ import annotations

import ast
import hashlib
import inspect
import json

import numpy as np
import pandas as pd
import pytest

from microcosm.build.us_runtime import one_population_contract as contract
from microcosm.build.us_runtime import one_population_selection as selection
from microcosm.build.us_runtime.acs_pums import ACS_2024_1YR_SPINE
from microcosm.build.us_runtime.base_pool import ASEC_PUF_SPINE
from microcosm.build.us_runtime.multispine_pool import (
    POOL_ENGINE_INPUT_PROJECTION_CONTRACT,
)
from microcosm.build.us_runtime.one_population_contract import (
    ACS_SUPPORT_CHANNEL,
    ASSEMBLED_CHANNELS,
    CPS_SPINE_TAG,
    ENGINE_PERIOD,
    STRUCTURAL_COLUMNS,
    EngineVariableIndex,
    OnePopulationContractError,
    PolicyEngineUSVariableIndex,
    _engine_columns,
)
from microcosm.build.us_runtime.puf_capital_gains_tail import (
    PUF_CAPITAL_GAINS_TAIL_APPLIED_COLUMN,
    PUF_CAPITAL_GAINS_TAIL_DONOR_SOURCE_ID_COLUMN,
)
from microcosm.build.us_runtime.spm_composition import check_spm_composition
from microcosm.build.us_runtime.spm_independence_role import (
    us_spm_independence_role_signal_gate,
)
from microcosm.build.us_runtime.support_provenance import (
    BASE_ASEC_SUPPORT_CHANNEL,
    PUF_TAX_DETAIL_SUPPORT_CHANNEL,
    spine_source_id_column,
    support_channel_column,
    support_clone_index_column,
    support_source_id_column,
    validate_assembly_provenance,
)
from microcosm.data.stored_inputs import (
    US_STORED_NON_VARIABLE_COLUMNS,
    undefined_stored_inputs,
)
from microcosm.frame import US_SCHEMA, Frame
from test_support.microcosm_build.us_one_population_contract import (
    RELEASE_WRITTEN_ON_ACS,
    SPM_ROLE,
    STAND_IN_COMPAT_AGGREGATES,
    STAND_IN_COMPUTED_AT_2024,
    STAND_IN_FORMULA_AFTER_2024,
    STAND_IN_STRUCTURAL_FORMULAS,
    TAIL_PROVENANCE_DEFAULTS,
    StandInVariableIndex,
    clone_zero_assembly_record,
    generate_population,
    joined_frame,
    lock_engine_inputs,
    role_shares,
    spm_role_gate_record,
    with_spm_role_record,
    with_staging_union_columns,
)

_SEEDS = (0, 1, 2)


@pytest.fixture(scope="module")
def population():
    return generate_population(0)


@pytest.fixture(scope="module")
def frame(population):
    return joined_frame(population)


@pytest.fixture(scope="module")
def stand_in():
    return StandInVariableIndex.from_lock()


# ----------------------------------------------------------- the surface


class TestModuleSurface:
    def test_public_names_resolve_and_are_sorted(self) -> None:
        assert contract.__all__ == sorted(contract.__all__, key=_all_sort_key)
        for name in contract.__all__:
            assert hasattr(contract, name), name

    def test_signatures_take_the_injected_inputs_by_keyword(self) -> None:
        def keyword_only(function) -> set[str]:
            return {
                name
                for name, parameter in inspect.signature(function).parameters.items()
                if parameter.kind is inspect.Parameter.KEYWORD_ONLY
            }

        assert list(inspect.signature(contract.to_assembled_provenance).parameters) == [
            "cps_tables",
            "acs_tables",
        ]
        assert keyword_only(contract.require_one_population) == {
            "parent_tail_donor_ids",
            "declared_absences",
            "variable_index",
            "register",
        }
        assert list(
            inspect.signature(contract.combined_spm_role_record).parameters
        ) == ["cps_record", "staging_receipt", "frame"]
        assert keyword_only(contract.absences) == {
            "fill_manifest",
            "zero_written",
            "declared",
        }
        assert list(inspect.signature(contract.cps_unit_ids).parameters) == ["person"]

    def test_constants_match_staging_and_provenance(self) -> None:
        # Staging's ACS channel and spine tag, and its donor-row spine tag.
        assert ACS_SUPPORT_CHANNEL == ACS_2024_1YR_SPINE
        assert CPS_SPINE_TAG == ASEC_PUF_SPINE
        assert ASSEMBLED_CHANNELS == (
            BASE_ASEC_SUPPORT_CHANNEL,
            PUF_TAX_DETAIL_SUPPORT_CHANNEL,
            ACS_2024_1YR_SPINE,
        )
        assert ENGINE_PERIOD == 2024
        assert STRUCTURAL_COLUMNS == {
            "person_id",
            "household_id",
            "tax_unit_id",
            "spm_unit_id",
            "family_id",
            "marital_unit_id",
            "person_household_id",
            "person_tax_unit_id",
            "person_spm_unit_id",
            "person_family_id",
            "person_marital_unit_id",
        }

    def test_module_imports_no_engine_at_import_time(self) -> None:
        tree = ast.parse(inspect.getsource(contract))
        top_level = [
            node.module if isinstance(node, ast.ImportFrom) else alias.name
            for node in tree.body
            if isinstance(node, ast.Import | ast.ImportFrom)
            for alias in (node.names if isinstance(node, ast.Import) else [None])
        ]
        assert not [name for name in top_level if "policyengine" in str(name)]

    def test_error_is_a_value_error(self) -> None:
        assert issubclass(OnePopulationContractError, ValueError)


def _all_sort_key(name: str) -> tuple[int, str]:
    # The repo's __all__ order: constants, then classes, then functions.
    if name.isupper():
        return (0, name)
    return (1, name) if name[0].isupper() else (2, name)


# ------------------------------------------- the default adapter (M2)


class _FakeMetadataIndex:
    """The three methods the adapter calls, over fixed name sets."""

    def __init__(self) -> None:
        self.inputs = {"age", "household_id", "person_household_id", "tip_income"}
        self.formula_owned = {"person_id", "adjusted_gross_income", "weeks_worked"}
        self.computed_2024 = {"adjusted_gross_income"}
        self.compat = {"social_security"}
        self.periods: list[object] = []

    def variables(self) -> list[str]:
        return sorted(self.inputs)

    def formula_owned_outputs(self, names) -> set[str]:
        names = set(names)
        return (names & self.compat) | (names & self.formula_owned)

    def _engine_computed_columns(self, tables, *, period) -> set[str]:
        self.periods.append(period)
        present = {column for table in tables.values() for column in table.columns}
        return (present & self.compat) | (
            (present & self.computed_2024) - STRUCTURAL_COLUMNS
        )


class TestDefaultAdapter:
    def test_builds_no_inner_index_until_asked(self, monkeypatch) -> None:
        import microcosm.frame.adapters.policyengine_us as adapter_module

        built: list[str] = []

        class Sentinel:
            def __init__(self) -> None:
                built.append("built")
                raise RuntimeError("inner index built")

        monkeypatch.setattr(
            adapter_module, "PolicyEngineUSVariableMetadataIndex", Sentinel
        )
        index = PolicyEngineUSVariableIndex()
        assert built == []
        assert isinstance(index, EngineVariableIndex)
        with pytest.raises(RuntimeError, match="inner index built"):
            index.known_variables(["age"])
        assert built == ["built"]

    def test_classes_follow_the_release_guard(self) -> None:
        fake = _FakeMetadataIndex()
        index = PolicyEngineUSVariableIndex(fake)
        tables = {
            "person": pd.DataFrame(
                columns=[
                    "person_id",
                    "person_household_id",
                    "age",
                    "tip_income",
                    "weeks_worked",
                    "adjusted_gross_income",
                    "social_security",
                    "not_a_variable",
                ]
            ),
            "household": pd.DataFrame(columns=["household_id"]),
        }
        present = {column for table in tables.values() for column in table}

        assert index.known_variables(present) == present - {"not_a_variable"}
        assert index.computed_at_2024(tables) == {
            "adjusted_gross_income",
            "social_security",
        }
        # weeks_worked: known, no formula at 2024, so an input at 2024 although
        # the year-blind variables() leaves it out. Structural ids never count.
        assert index.inputs_at_2024(tables) == {"age", "tip_income", "weeks_worked"}
        assert set(fake.periods) == {ENGINE_PERIOD}

    def test_coherent_answers_pass_the_contract_check(self) -> None:
        tables = {"person": pd.DataFrame(columns=["person_id", "age", "ssi"])}
        classes = _engine_columns(tables, StandInVariableIndex.from_lock())
        assert classes.inputs == {"age"}
        assert classes.computed == {"ssi"}
        assert classes.known == {"person_id", "age", "ssi"}

    @pytest.mark.parametrize(
        ("broken", "message"),
        [
            ("computed", "inputs also computed"),
            ("structural", "inputs include structural columns"),
            ("unknown", "inputs not known"),
            ("absent", "names columns no table holds"),
        ],
    )
    def test_incoherent_answers_are_refused(self, broken, message) -> None:
        class Broken(StandInVariableIndex):
            def inputs_at_2024(self, tables):
                inputs = set(super().inputs_at_2024(tables))
                extra = {
                    "computed": "ssi",
                    "structural": "person_household_id",
                    "unknown": "not_a_variable",
                    "absent": "tip_income",
                }[broken]
                return frozenset(inputs | {extra})

        tables = {
            "person": pd.DataFrame(
                columns=["person_household_id", "age", "ssi", "not_a_variable"]
            )
        }
        index = Broken(inputs=frozenset(lock_engine_inputs()))
        with pytest.raises(OnePopulationContractError, match=message):
            _engine_columns(tables, index)


# --------------------------------------------------- the stand-in (M2)


class TestStandInIndex:
    def test_lock_projection_is_the_pool_contract(self) -> None:
        # lock_engine_inputs checks the lock's own receipt; this pins the
        # same names to the pool's installed-engine contract.
        inputs = lock_engine_inputs()
        payload = [
            {"entity": entity, "variable": variable}
            for variable, entity in sorted(inputs.items())
        ]
        digest = hashlib.sha256(
            json.dumps(
                payload, allow_nan=False, separators=(",", ":"), sort_keys=True
            ).encode("utf-8")
        ).hexdigest()
        assert len(inputs) == POOL_ENGINE_INPUT_PROJECTION_CONTRACT.input_count == 926
        assert digest == POOL_ENGINE_INPUT_PROJECTION_CONTRACT.sha256

    def test_explicit_sets_are_disjoint_from_the_lock(self, stand_in) -> None:
        for names in (
            STAND_IN_COMPUTED_AT_2024,
            STAND_IN_COMPAT_AGGREGATES,
            STAND_IN_FORMULA_AFTER_2024,
            STAND_IN_STRUCTURAL_FORMULAS,
        ):
            assert not names & stand_in.inputs

    def test_a_post_2024_formula_is_an_input_at_2024(self, stand_in) -> None:
        # weeks_worked has only formula_2025 at policyengine-us 2.2.1: absent
        # from the year-blind 926, not computed at 2024, so an input at 2024.
        tables = {"person": pd.DataFrame(columns=["weeks_worked"])}
        assert "weeks_worked" not in lock_engine_inputs()
        assert stand_in.computed_at_2024(tables) == frozenset()
        assert stand_in.inputs_at_2024(tables) == {"weeks_worked"}

    def test_compat_aggregates_are_computed_and_never_inputs(self, stand_in) -> None:
        tables = {"person": pd.DataFrame(columns=sorted(STAND_IN_COMPAT_AGGREGATES))}
        assert stand_in.computed_at_2024(tables) == STAND_IN_COMPAT_AGGREGATES
        assert stand_in.inputs_at_2024(tables) == frozenset()

    def test_structural_columns_are_known_but_neither_class(self, stand_in) -> None:
        tables = {"person": pd.DataFrame(columns=sorted(STRUCTURAL_COLUMNS))}
        assert stand_in.known_variables(STRUCTURAL_COLUMNS) == STRUCTURAL_COLUMNS
        assert stand_in.computed_at_2024(tables) == frozenset()
        assert stand_in.inputs_at_2024(tables) == frozenset()

    def test_lock_inputs_the_clauses_read_are_inputs(self, stand_in) -> None:
        names = [
            "age",
            "employer_sponsored_insurance_premiums",
            SPM_ROLE,
            "takes_up_ssi_if_eligible",
            "congressional_district_geoid",
            "tip_income",
        ]
        tables = {"person": pd.DataFrame(columns=names)}
        assert stand_in.inputs_at_2024(tables) == set(names)


# ------------------------------------------------ PR 1's key digest (Q7)


class TestKeysDigest:
    def test_keys_sha256_is_public_and_is_the_manifest_rule(self) -> None:
        assert "keys_sha256" in selection.__all__
        assert not hasattr(selection, "_keys_sha256")
        keys = ["2024|cps:2", "2023|cps:1", "acs:9"]
        assert (
            selection.keys_sha256(keys)
            == hashlib.sha256("\n".join(sorted(keys)).encode("utf-8")).hexdigest()
        )


# ---------------------------------------------- the generated population


class TestGeneratedPopulation:
    def test_is_deterministic_in_its_seed(self) -> None:
        first, again, other = (generate_population(seed) for seed in (3, 3, 4))
        for entity in US_SCHEMA.entities:
            pd.testing.assert_frame_equal(
                first.cps_tables[entity], again.cps_tables[entity]
            )
            pd.testing.assert_frame_equal(
                first.acs_tables[entity], again.acs_tables[entity]
            )
        assert first.cps_keys == again.cps_keys
        assert first.cps_keys != other.cps_keys

    def test_cps_rows_carry_route_a_labels(self, population) -> None:
        multiplier = population.clone_multiplier
        for entity in US_SCHEMA.entities:
            table = population.cps_tables[entity]
            ids = table[US_SCHEMA.entity_id_column(entity)].to_numpy()
            channel = table[support_channel_column(entity)].to_numpy()
            clone = table[support_clone_index_column(entity)].to_numpy()
            source = table[support_source_id_column(entity)].to_numpy()
            assert set(clone) == {0, 1, 2}
            assert ((channel == BASE_ASEC_SUPPORT_CHANNEL) == (clone == 0)).all()
            assert set(channel[clone > 0]) == {PUF_TAX_DETAIL_SUPPORT_CHANNEL}
            # puf_support._remap_ids, then the tail's one more multiplier.
            assert (ids == source + clone * multiplier).all()
            assert set(source[clone > 0]) <= set(ids[clone == 0])
            assert spine_source_id_column(entity) not in table
            assert f"{entity}_spine" not in table

    def test_acs_rows_carry_staging_labels(self, population) -> None:
        for entity in US_SCHEMA.entities:
            table = population.acs_tables[entity]
            id_column = US_SCHEMA.entity_id_column(entity)
            cps_ids = population.cps_tables[entity][id_column]
            assert (table[support_channel_column(entity)] == ACS_2024_1YR_SPINE).all()
            assert (table[f"{entity}_spine"] == ACS_2024_1YR_SPINE).all()
            assert (table[support_clone_index_column(entity)] == 0).all()
            shift = table[id_column] - table[support_source_id_column(entity)]
            assert set(shift) == {population.acs_id_offsets[entity]}
            assert table[id_column].min() == cps_ids.max() + 1
            assert spine_source_id_column(entity) not in table
        person = population.acs_tables["person"]
        for group in US_SCHEMA.group_entities:
            membership = person[US_SCHEMA.membership_column(group)]
            assert set(membership) == set(
                population.acs_tables[group][US_SCHEMA.id_column(group)]
            )

    def test_raw_lineage_columns(self, population) -> None:
        cps = population.cps_tables["person"]
        native = cps[cps[support_clone_index_column("person")] == 0]
        peridnum = native["PERIDNUM"]
        assert peridnum.str.fullmatch(r"[1-9][0-9]{21}").all()
        assert peridnum.is_unique
        assert (native["source_person_id"] == peridnum).all()
        assert (
            peridnum.str[:20].groupby(native["person_household_id"]).nunique() == 1
        ).all()
        assert set(native["source_year"]) == {2023, 2024}
        assert (
            len(population.cps_keys)
            == len(set(population.cps_keys))
            == len(
                population.cps_tables["household"].query(
                    "household_support_clone_index == 0"
                )
            )
        )

        household = population.acs_tables["household"]
        person = population.acs_tables["person"]
        assert household["SERIALNO"].is_unique
        assert set(household["TYPEHUGQ"]) == {1, 2, 3}
        serial = person["person_household_id"].map(
            household.set_index("household_id")["SERIALNO"]
        )
        assert (
            not pd.DataFrame({"s": serial, "o": person["SPORDER"]}).duplicated().any()
        )
        assert (person["source_person_id"] == person["SPORDER"].astype(str)).all()
        assert (person["source_year"] == 2024).all()
        native_household = person["person_household_id"].map(
            household.set_index("household_id")["household_source_id"]
        )
        assert (person["source_household_id"] == native_household).all()
        assert population.acs_keys == tuple(
            f"acs:{value}" for value in household["SERIALNO"]
        )

    def test_districts_carry_their_state_prefix(self, frame) -> None:
        household = frame.table("household")
        cd = household["congressional_district_geoid"]
        assert (cd > 0).all()
        assert (cd // 100 == household["state_fips"]).all()

    def test_tail_columns_on_every_tax_unit(self, population, frame) -> None:
        tax_unit = frame.table("tax_unit")
        for column in TAIL_PROVENANCE_DEFAULTS:
            assert tax_unit[column].notna().all(), column
        applied = tax_unit[PUF_CAPITAL_GAINS_TAIL_APPLIED_COLUMN].to_numpy()
        clone = tax_unit[support_clone_index_column("tax_unit")].to_numpy()
        assert applied.any()
        assert (clone[applied] == 2).all()
        donors = tax_unit.loc[applied, PUF_CAPITAL_GAINS_TAIL_DONOR_SOURCE_ID_COLUMN]
        assert set(donors) == population.tail_donor_ids
        assert (
            tax_unit.loc[~applied, PUF_CAPITAL_GAINS_TAIL_DONOR_SOURCE_ID_COLUMN] == -1
        ).all()

    def test_every_person_resolves_to_one_row_per_group(self, frame) -> None:
        person = frame.table("person")
        assert person["person_id"].is_unique
        for group in US_SCHEMA.group_entities:
            table = frame.table(group)
            assert table[US_SCHEMA.id_column(group)].is_unique
            joined = person[US_SCHEMA.membership_column(group)].map(
                table.set_index(US_SCHEMA.id_column(group))[
                    support_channel_column(group)
                ]
            )
            assert (joined == person[support_channel_column("person")]).all()

    def test_shared_columns_keep_one_dtype(self, population, frame) -> None:
        for entity in US_SCHEMA.entities:
            cps = population.cps_tables[entity]
            acs = population.acs_tables[entity]
            for column in set(cps.columns) & set(acs.columns):
                assert cps[column].dtype == acs[column].dtype, (entity, column)
        assert frame.table("person")[SPM_ROLE].dtype == np.dtype(bool)

    @pytest.mark.parametrize("seed", _SEEDS)
    def test_spm_composition_passes(self, seed) -> None:
        result = check_spm_composition(joined_frame(generate_population(seed)))
        assert result.status == "PASS", result.failures
        assert result.details["role_source"] == "source_column"

    def test_a_role_classifies_an_adult_on_its_own(self, frame) -> None:
        # The ACS household headed by a 17-year-old passes only through the role.
        person = frame.table("person")
        minor_heads = person[(person["age"] == 17) & person[SPM_ROLE]]
        assert len(minor_heads) == 1
        unit = minor_heads["person_spm_unit_id"].iloc[0]
        assert (person.loc[person["person_spm_unit_id"] == unit, "age"] < 18).all()

    @pytest.mark.parametrize("seed", _SEEDS)
    def test_role_signal_gate_passes_with_a_record_by_its_rule(self, seed) -> None:
        frame = joined_frame(generate_population(seed))
        gated = with_spm_role_record(frame, spm_role_gate_record(frame))
        result = us_spm_independence_role_signal_gate(gated)
        assert result.passed, result.failures
        share, minor_share = role_shares(frame)
        assert result.details["role_share"] == pytest.approx(share)
        assert result.details["minor_role_share"] == pytest.approx(minor_share)
        lower, upper = result.details["minor_role_share_band"]
        assert lower < minor_share < upper

    def test_role_signal_gate_fails_without_the_record(self, frame) -> None:
        result = us_spm_independence_role_signal_gate(frame)
        assert not result.passed
        assert "no source derivation provenance" in " ".join(result.failures)

    def test_assembly_validator_passes_on_a_clone_zero_record(self, frame) -> None:
        record = clone_zero_assembly_record(frame)
        assembled = Frame(
            {entity: frame.table(entity) for entity in frame.entities},
            frame.schema,
            {"household": frame.weights_for("household")},
            frame.strata,
            mass_log=frame.mass_log,
            metadata={**frame.metadata, **record},
        )
        manifest = validate_assembly_provenance(assembled, boundary="phase 0 test")
        assert tuple(manifest["channels"]) == ASSEMBLED_CHANNELS
        counts = manifest["native_row_counts"]
        assert all(
            counts[entity][PUF_TAX_DETAIL_SUPPORT_CHANNEL] == 0
            for entity in US_SCHEMA.entities
        )

    def test_stored_columns_are_engine_variables_or_registered(
        self, frame, stand_in
    ) -> None:
        columns = {
            column for entity in frame.entities for column in frame.table(entity)
        }
        assert (
            undefined_stored_inputs(
                columns,
                engine_variables=stand_in.known_variables(columns),
                register=US_STORED_NON_VARIABLE_COLUMNS,
            )
            == ()
        )

    def test_no_column_is_computed_at_2024(self, frame, stand_in) -> None:
        tables = {entity: frame.table(entity) for entity in frame.entities}
        assert stand_in.computed_at_2024(tables) == frozenset()

    def test_inputs_are_filled_except_the_release_written(
        self, frame, stand_in
    ) -> None:
        tables = {entity: frame.table(entity) for entity in frame.entities}
        inputs = stand_in.inputs_at_2024(tables)
        assert {"age", SPM_ROLE, "employer_sponsored_insurance_premiums"} <= inputs
        nulls = {
            column
            for table in tables.values()
            for column in inputs & set(table.columns)
            if table[column].isna().any()
        }
        assert nulls == RELEASE_WRITTEN_ON_ACS
        person = frame.table("person")
        acs = person[support_channel_column("person")] == ACS_2024_1YR_SPINE
        for column in RELEASE_WRITTEN_ON_ACS:
            assert person.loc[acs, column].isna().all()
            assert person.loc[~acs, column].notna().all()

    def test_raw_upper_case_columns_sit_on_one_origin(self, frame) -> None:
        person = frame.table("person")
        household = frame.table("household")
        acs = person[support_channel_column("person")] == ACS_2024_1YR_SPINE
        acs_household = (
            household[support_channel_column("household")] == ACS_2024_1YR_SPINE
        )
        for column in ("SSI_VAL", "PERIDNUM"):
            assert person.loc[acs, column].isna().all()
            assert person.loc[~acs, column].notna().all()
        assert person.loc[acs, "SPORDER"].notna().all()
        assert person.loc[~acs, "SPORDER"].isna().all()
        for column in ("SERIALNO", "TYPEHUGQ"):
            assert household.loc[acs_household, column].notna().all()
            assert household.loc[~acs_household, column].isna().all()


class TestStagingUnionForm:
    """The ACS tables in staging's pooled form: every CPS column, all null."""

    def test_acs_rows_gain_every_cps_column_all_null(self, population) -> None:
        union = with_staging_union_columns(population)
        for entity in US_SCHEMA.entities:
            trimmed = population.acs_tables[entity]
            pooled = union.acs_tables[entity]
            cps_only = set(population.cps_tables[entity]) - set(trimmed)
            assert set(pooled) == set(trimmed) | set(population.cps_tables[entity])
            for column in cps_only:
                assert pooled[column].isna().all(), (entity, column)
            for column in trimmed:
                assert (
                    pooled[column].astype(object).tolist()
                    == trimmed[column].astype(object).tolist()
                ), (entity, column)
        person = union.acs_tables["person"]
        assert person["takes_up_ssi_if_eligible"].isna().all()
        # An ACS-only integer column comes back as float64 from staging.
        assert person["SPORDER"].dtype == np.dtype("float64")
        assert union.acs_tables["household"]["TYPEHUGQ"].dtype == np.dtype("float64")

    def test_union_form_passes_the_same_checks(self, population) -> None:
        frame = joined_frame(with_staging_union_columns(population))
        assert check_spm_composition(frame).status == "PASS"
        gated = with_spm_role_record(frame, spm_role_gate_record(frame))
        assert us_spm_independence_role_signal_gate(gated).passed
        assembled = Frame(
            {entity: frame.table(entity) for entity in frame.entities},
            frame.schema,
            {"household": frame.weights_for("household")},
            frame.strata,
            mass_log=frame.mass_log,
            metadata={**frame.metadata, **clone_zero_assembly_record(frame)},
        )
        validate_assembly_provenance(assembled, boundary="phase 0 union test")
