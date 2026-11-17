"""d2283 PR 2: the one-population contract library.

Phase 0 (the foundation) pins the module's surface and engine-free import,
the injected engine variable index (the default adapter's rules over a fake
inner index, and the stand-in built from the engine ABI lock), PR 1's public
key digest, and the generated population every later clause test corrupts:
uncorrupted, it passes the SPM composition check, the SPM role signal gate
(bands included) with a record computed by the gate's rule, and the assembly
validator with a clone-0, three-channel record.

Phase 1 covers the provenance rewrite (``to_assembled_provenance``: X3, X4,
X6 and one refusal per P5, P6, P6b and P7 rule) and ``cps_unit_ids``.
"""

from __future__ import annotations

import ast
import hashlib
import inspect
import json
import re

import numpy as np
import pandas as pd
import pytest

from microcosm.build.gates import GateResult
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
    cps_unit_ids,
    to_assembled_provenance,
)
from microcosm.build.us_runtime.puf_capital_gains_tail import (
    PUF_CAPITAL_GAINS_TAIL_APPLIED_COLUMN,
    PUF_CAPITAL_GAINS_TAIL_DONOR_SOURCE_ID_COLUMN,
)
from microcosm.build.us_runtime.spine_agreement import spine_agreement_gate
from microcosm.build.us_runtime.spm_composition import check_spm_composition
from microcosm.build.us_runtime.spm_independence_role import (
    us_spm_independence_role_signal_gate,
)
from microcosm.build.us_runtime.support_provenance import (
    BASE_ASEC_SUPPORT_CHANNEL,
    PUF_TAX_DETAIL_SUPPORT_CHANNEL,
    SPINE_ASSEMBLY_MANIFEST_KEY,
    require_assembled_support_provenance,
    spine_source_id_column,
    support_channel_column,
    support_clone_index_column,
    support_copy_rank_series,
    support_gate_source_channel_series,
    support_role_series,
    support_source_id_column,
    validate_assembly_provenance,
)
from microcosm.data.stored_inputs import (
    US_STORED_NON_VARIABLE_COLUMNS,
    undefined_stored_inputs,
)
from microcosm.frame import US_SCHEMA, Frame, WeightKind
from test_support.microcosm_build.us_one_population_contract import (
    RELEASE_WRITTEN_ON_ACS,
    SPM_ROLE,
    STAND_IN_COMPAT_AGGREGATES,
    STAND_IN_COMPUTED_AT_2024,
    STAND_IN_FORMULA_AFTER_2024,
    STAND_IN_STRUCTURAL_FORMULAS,
    TAIL_PROVENANCE_DEFAULTS,
    StandInVariableIndex,
    assembled_frame,
    clone_zero_assembly_record,
    generate_population,
    joined_frame,
    lock_engine_inputs,
    role_shares,
    spm_role_gate_record,
    staging_lineage,
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

        parameters = inspect.signature(contract.to_assembled_provenance).parameters
        assert list(parameters)[:2] == ["cps_tables", "acs_tables"]
        assert keyword_only(contract.to_assembled_provenance) == {
            "cps_household_weights",
            "acs_household_weights",
            "staging_lineage",
        }
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


# --------------------------------- phase 1: the provenance rewrite (P1-P8)

_FORMS = ("trimmed", "union")


def _form(population, form: str):
    return with_staging_union_columns(population) if form == "union" else population


@pytest.fixture(scope="module", params=_FORMS)
def rewritten(request, population):
    """(input population, rewritten frame) in the trimmed and union forms."""

    source = _form(population, request.param)
    return source, assembled_frame(source)


def _by_id(table: pd.DataFrame, entity: str) -> pd.DataFrame:
    return table.set_index(US_SCHEMA.entity_id_column(entity), drop=False)


def _acs_rows(table: pd.DataFrame, entity: str) -> pd.DataFrame:
    return table[table[support_channel_column(entity)] == ACS_SUPPORT_CHANNEL]


def _person_lineage(frame: Frame) -> set[tuple[str, int]]:
    household = frame.table("household").set_index("household_id")
    person = _acs_rows(frame.table("person"), "person")
    serial = person["person_household_id"].map(household["SERIALNO"])
    return set(
        zip(
            serial.tolist(),
            (int(value) for value in person["SPORDER"].tolist()),
            strict=True,
        )
    )


class TestProvenanceRewrite:
    def test_returns_a_frame_with_only_the_clone_zero_record(self, rewritten) -> None:
        _, frame = rewritten
        assert isinstance(frame, Frame)
        assert set(frame.metadata) == {SPINE_ASSEMBLY_MANIFEST_KEY}
        assert frame.mass_log == ()
        record = frame.metadata[SPINE_ASSEMBLY_MANIFEST_KEY]
        expected = clone_zero_assembly_record(frame)[SPINE_ASSEMBLY_MANIFEST_KEY]
        assert tuple(record["channels"]) == ASSEMBLED_CHANNELS
        assert {
            entity: dict(counts)
            for entity, counts in record["native_row_counts"].items()
        } == expected["native_row_counts"]

    def test_cps_rows_keep_route_a_labels_and_gain_the_spine_id(
        self, rewritten
    ) -> None:
        population, frame = rewritten
        for entity in US_SCHEMA.entities:
            before = _by_id(population.cps_tables[entity], entity)
            after = _by_id(frame.table(entity), entity).loc[before.index]
            for column in (
                support_source_id_column(entity),
                support_channel_column(entity),
                support_clone_index_column(entity),
            ):
                assert after[column].tolist() == before[column].tolist(), column
            assert (
                after[spine_source_id_column(entity)].tolist()
                == before[support_source_id_column(entity)].tolist()
            )
            assert (after[f"{entity}_spine"] == CPS_SPINE_TAG).all()

    def test_no_row_is_labelled_acs_and_every_row_has_a_spine_tag(
        self, rewritten
    ) -> None:
        # P3: the predictor join's "acs" channel is never written.
        _, frame = rewritten
        for entity in US_SCHEMA.entities:
            table = frame.table(entity)
            assert "acs" not in set(table[support_channel_column(entity)])
            assert table[f"{entity}_spine"].notna().all()
            acs = table[support_channel_column(entity)] == ACS_SUPPORT_CHANNEL
            assert (table.loc[acs, f"{entity}_spine"] == ACS_SUPPORT_CHANNEL).all()
            assert (table.loc[~acs, f"{entity}_spine"] == CPS_SPINE_TAG).all()

    def test_household_weights_follow_their_households(self, rewritten) -> None:
        population, frame = rewritten
        expected = {
            **dict(
                zip(
                    population.cps_tables["household"]["household_id"].tolist(),
                    population.cps_household_weights.tolist(),
                    strict=True,
                )
            ),
            **dict(
                zip(
                    population.acs_tables["household"]["household_id"].tolist(),
                    population.acs_household_weights.tolist(),
                    strict=True,
                )
            ),
        }
        weights = frame.weights_for("household")
        assert weights.kind is WeightKind.IMPORTANCE
        actual = dict(
            zip(
                frame.table("household")["household_id"].tolist(),
                weights.values.tolist(),
                strict=True,
            )
        )
        assert actual == expected
        # The sort moved ACS households among CPS ones only if ids interleave;
        # either way the household table is sorted by id.
        ids = frame.table("household")["household_id"].to_numpy()
        assert (np.diff(ids) > 0).all()

    def test_inputs_are_not_mutated(self, population) -> None:
        cps = {entity: table.copy() for entity, table in population.cps_tables.items()}
        acs = {entity: table.copy() for entity, table in population.acs_tables.items()}
        assembled_frame(population)
        for entity in US_SCHEMA.entities:
            pd.testing.assert_frame_equal(population.cps_tables[entity], cps[entity])
            pd.testing.assert_frame_equal(population.acs_tables[entity], acs[entity])


class TestRewriteInvariants:
    """X3: after the rewrite, across both origins."""

    def test_source_id_and_rank_are_unique_across_origins(self, rewritten) -> None:
        _, frame = rewritten
        for entity in US_SCHEMA.entities:
            table = frame.table(entity)
            pairs = pd.DataFrame(
                {
                    "source": table[support_source_id_column(entity)].to_numpy(),
                    "rank": support_copy_rank_series(table, entity=entity).to_numpy(),
                }
            )
            assert not pairs.duplicated().any(), entity
            # Both origins are present, so uniqueness is across them.
            assert set(table[support_channel_column(entity)]) == set(ASSEMBLED_CHANNELS)

    def test_every_acs_spine_id_is_its_native_id(self, rewritten) -> None:
        population, frame = rewritten
        for entity in US_SCHEMA.entities:
            native = _by_id(population.acs_tables[entity], entity)
            after = _by_id(_acs_rows(frame.table(entity), entity), entity)
            assert sorted(after.index) == sorted(native.index)
            after = after.loc[native.index]
            # Native id: staging's source id before the rewrite.
            assert (
                after[spine_source_id_column(entity)].tolist()
                == native[support_source_id_column(entity)].tolist()
            )
            # The new source id is the pooled id.
            assert (
                after[support_source_id_column(entity)].tolist()
                == after[US_SCHEMA.entity_id_column(entity)].tolist()
            )
            assert (after[support_clone_index_column(entity)] == 0).all()

    def test_every_acs_serialno_and_sporder_survives(self, rewritten) -> None:
        population, frame = rewritten
        assert _person_lineage(frame) == staging_lineage(population)
        native = _by_id(population.acs_tables["household"], "household")
        after = _by_id(frame.table("household"), "household").loc[native.index]
        assert after["SERIALNO"].tolist() == native["SERIALNO"].tolist()
        assert after["TYPEHUGQ"].tolist() == native["TYPEHUGQ"].tolist()


class TestAssembledValidators:
    """X6: the readers that check assembled provenance accept the frame."""

    @pytest.mark.parametrize("seed", _SEEDS)
    @pytest.mark.parametrize("form", _FORMS)
    def test_validator_passes_on_three_channels_with_clones(self, seed, form) -> None:
        population = _form(generate_population(seed), form)
        frame = assembled_frame(population)
        manifest = validate_assembly_provenance(frame, boundary="phase 1 test")
        assert tuple(manifest["channels"]) == ASSEMBLED_CHANNELS
        counts = manifest["native_row_counts"]
        for entity in US_SCHEMA.entities:
            cps = population.cps_tables[entity]
            clone = cps[support_clone_index_column(entity)]
            assert dict(counts[entity]) == {
                BASE_ASEC_SUPPORT_CHANNEL: int((clone == 0).sum()),
                PUF_TAX_DETAIL_SUPPORT_CHANNEL: 0,
                ACS_SUPPORT_CHANNEL: len(population.acs_tables[entity]),
            }
            assert set(clone) == {0, 1, 2}
            require_assembled_support_provenance(frame.table(entity), entity=entity)

    def test_spine_agreement_gate_does_not_raise_with_the_record(
        self, rewritten
    ) -> None:
        _, frame = rewritten
        result = spine_agreement_gate(frame)
        assert isinstance(result, GateResult)
        # It fails, but as a result: puf_tax_detail is a clone role to it.
        assert not result.passed
        assert any("clone role" in failure for failure in result.failures)

    def test_spine_agreement_gate_raises_without_the_record(self, rewritten) -> None:
        _, frame = rewritten
        bare = Frame(
            {entity: frame.table(entity) for entity in frame.entities},
            frame.schema,
            {"household": frame.weights_for("household")},
            frame.strata,
        )
        with pytest.raises(ValueError, match="has no assembly manifest"):
            spine_agreement_gate(bare)


# ------------------------------------------------ X4: a Route A fixture


@pytest.fixture(scope="module")
def route_a():
    """A Route A base made by the production operators, not by hand.

    ``clone_us_frame_for_puf_support`` clones the release-base ASEC fixture
    (``us_release_base_input.asec_frame``) into native rows (clone 0,
    ``asec``) and PUF tax-detail copies (clone 1, ``puf_tax_detail``) on all
    six entities, and ``with_capital_gains_tail_copies`` adds tail copies
    (clone 2) through ``puf_capital_gains_tail._clone_and_transfer``. The
    generator's CPS half restates the same rules by hand; this fixture is
    used so the differential does not rest on that restatement.
    """

    from microcosm.build.us_runtime.puf_support import clone_us_frame_for_puf_support
    from test_support.microcosm_build.us_release_base_input import asec_frame
    from test_support.microcosm_build.us_tail_clone_fixtures import (
        with_capital_gains_tail_copies,
    )

    cloned = clone_us_frame_for_puf_support(asec_frame(40, seed=5))
    household = cloned.table("household")
    sources = household.loc[
        household["household_support_clone_index"] == 1, "household_source_id"
    ].tolist()[::9]
    return with_capital_gains_tail_copies(cloned, sources)


def test_role_readers_match_route_a_with_no_acs_row(route_a, population) -> None:
    """X4: with no ACS row, the role readers return Route A's series."""

    tables = {entity: route_a.table(entity) for entity in route_a.entities}
    empty_acs = {
        entity: population.acs_tables[entity].iloc[:0] for entity in US_SCHEMA.entities
    }
    frame = to_assembled_provenance(
        tables,
        empty_acs,
        cps_household_weights=route_a.weights_for("household").values,
        acs_household_weights=np.array([]),
        staging_lineage=frozenset(),
    )
    for entity in US_SCHEMA.entities:
        before, after = tables[entity], frame.table(entity)
        id_column = US_SCHEMA.entity_id_column(entity)
        assert spine_source_id_column(entity) not in before
        assert spine_source_id_column(entity) in after
        assert set(before[support_clone_index_column(entity)]) == {0, 1, 2}
        for column in before:
            assert after[column].dtype == before[column].dtype, (entity, column)
        for reader in (
            support_role_series,
            support_gate_source_channel_series,
            support_copy_rank_series,
        ):
            expected = pd.Series(
                reader(before, entity=entity).to_numpy(),
                index=before[id_column].to_numpy(),
            ).sort_index()
            actual = pd.Series(
                reader(after, entity=entity).to_numpy(),
                index=after[id_column].to_numpy(),
            ).sort_index()
            pd.testing.assert_series_equal(actual, expected, check_names=False)
    validate_assembly_provenance(frame, boundary="phase 1 X4")
    assert dict(
        zip(
            frame.table("household")["household_id"].tolist(),
            frame.weights_for("household").values.tolist(),
            strict=True,
        )
    ) == dict(
        zip(
            tables["household"]["household_id"].tolist(),
            route_a.weights_for("household").values.tolist(),
            strict=True,
        )
    )


# ------------------------------------ refusals: one per P5, P6, P6b, P7 rule


def _copies(population, origin: str) -> dict[str, pd.DataFrame]:
    tables = population.cps_tables if origin == "cps" else population.acs_tables
    return {entity: table.copy() for entity, table in tables.items()}


def _set(table: pd.DataFrame, row: int, column: str, value: object) -> None:
    """Set one cell, upcasting the column to object when the dtype refuses."""

    try:
        table.loc[table.index[row], column] = value
    except (TypeError, ValueError):
        table[column] = table[column].astype(object)
        table.loc[table.index[row], column] = value


def _first(table: pd.DataFrame, entity: str, clone: int) -> int:
    rows = np.flatnonzero(table[support_clone_index_column(entity)].to_numpy() == clone)
    return int(rows[0])


def _cps_missing_column(p):
    cps = _copies(p, "cps")
    cps["household"] = cps["household"].drop(columns=["household_support_channel"])
    return {"cps": cps}


def _cps_null_source(p):
    cps = _copies(p, "cps")
    _set(cps["person"], 0, "person_source_id", None)
    return {"cps": cps}


def _cps_non_integer_source(p):
    cps = _copies(p, "cps")
    _set(cps["person"], 0, "person_source_id", 1.5)
    return {"cps": cps}


def _cps_unknown_channel(p):
    cps = _copies(p, "cps")
    table = cps["tax_unit"]
    _set(table, _first(table, "tax_unit", 1), "tax_unit_support_channel", "acs")
    return {"cps": cps}


def _cps_asec_on_a_clone(p):
    cps = _copies(p, "cps")
    table = cps["spm_unit"]
    _set(table, _first(table, "spm_unit", 1), "spm_unit_support_channel", "asec")
    return {"cps": cps}


def _cps_clone_index_three(p):
    cps = _copies(p, "cps")
    table = cps["family"]
    _set(table, _first(table, "family", 2), "family_support_clone_index", 3)
    return {"cps": cps}


def _cps_spine_source_present(p):
    cps = _copies(p, "cps")
    table = cps["marital_unit"]
    table["marital_unit_spine_source_id"] = table["marital_unit_source_id"]
    return {"cps": cps}


def _cps_wrong_spine_tag(p):
    cps = _copies(p, "cps")
    cps["household"]["household_spine"] = ACS_SUPPORT_CHANNEL
    return {"cps": cps}


def _acs_missing_column(p):
    acs = _copies(p, "acs")
    acs["tax_unit"] = acs["tax_unit"].drop(columns=["tax_unit_support_clone_index"])
    return {"acs": acs}


def _acs_null_channel(p):
    acs = _copies(p, "acs")
    _set(acs["spm_unit"], 0, "spm_unit_support_channel", None)
    return {"acs": acs}


def _acs_channel_acs(p):
    acs = _copies(p, "acs")
    acs["person"]["person_support_channel"] = "acs"
    return {"acs": acs}


def _acs_clone_index(p):
    acs = _copies(p, "acs")
    _set(acs["household"], 0, "household_support_clone_index", 1)
    return {"acs": acs}


def _acs_offset_varies(p):
    acs = _copies(p, "acs")
    _set(
        acs["family"],
        0,
        "family_source_id",
        int(acs["family"]["family_source_id"].iloc[0]) + 1,
    )
    return {"acs": acs}


def _acs_spine_source_present(p):
    acs = _copies(p, "acs")
    table = acs["household"]
    table["household_spine_source_id"] = table["household_source_id"]
    return {"acs": acs}


def _acs_wrong_spine_tag(p):
    acs = _copies(p, "acs")
    _set(acs["person"], 0, "person_spine", CPS_SPINE_TAG)
    return {"acs": acs}


def _acs_null_spine_tag(p):
    acs = _copies(p, "acs")
    _set(acs["tax_unit"], 0, "tax_unit_spine", None)
    return {"acs": acs}


def _acs_spine_tag_absent(p):
    acs = _copies(p, "acs")
    acs["household"] = acs["household"].drop(columns=["household_spine"])
    return {"acs": acs}


def _weights_misaligned(p):
    return {"acs_weights": p.acs_household_weights[:-1]}


def _ids_collide_across_origins(p):
    # Staged on another donor: staging gave the ACS tax units no offset, so
    # their native ids sit on the export's.
    acs = _copies(p, "acs")
    offset = p.acs_id_offsets["tax_unit"]
    acs["tax_unit"]["tax_unit_id"] -= offset
    acs["person"]["person_tax_unit_id"] -= offset
    return {"acs": acs}


def _membership_crosses_origins(p):
    acs = _copies(p, "acs")
    person = acs["person"]
    sizes = person["person_spm_unit_id"].map(
        person["person_spm_unit_id"].value_counts()
    )
    row = int(np.flatnonzero(sizes.to_numpy() > 1)[0])
    cps_unit = int(p.cps_tables["spm_unit"]["spm_unit_id"].iloc[0])
    _set(person, row, "person_spm_unit_id", cps_unit)
    return {"acs": acs}


def _group_without_a_person(p):
    acs = _copies(p, "acs")
    family = acs["family"]
    extra = family.iloc[[-1]].copy()
    extra["family_id"] += 1
    extra["family_source_id"] += 1
    acs["family"] = pd.concat([family, extra], ignore_index=True)
    return {"acs": acs}


def _rank_repeats_within_cps(p):
    cps = _copies(p, "cps")
    table = cps["person"]
    _set(table, _first(table, "person", 2), "person_support_clone_index", 1)
    return {"cps": cps}


def _rank_repeats_across_origins(p):
    cps = _copies(p, "cps")
    acs_id = int(p.acs_tables["household"]["household_id"].iloc[0])
    table = cps["household"]
    _set(table, _first(table, "household", 0), "household_source_id", acs_id)
    return {"cps": cps}


def _serialno_column_missing(p):
    acs = _copies(p, "acs")
    acs["household"] = acs["household"].drop(columns=["SERIALNO"])
    return {"acs": acs}


def _serialno_null(p):
    acs = _copies(p, "acs")
    _set(acs["household"], 0, "SERIALNO", None)
    return {"acs": acs}


def _serialno_repeats(p):
    acs = _copies(p, "acs")
    _set(acs["household"], 1, "SERIALNO", acs["household"]["SERIALNO"].iloc[0])
    return {"acs": acs}


def _sporder_null(p):
    acs = _copies(p, "acs")
    _set(acs["person"], 0, "SPORDER", None)
    return {"acs": acs}


def _sporder_repeats(p):
    acs = _copies(p, "acs")
    person = acs["person"]
    second = int(np.flatnonzero(person["SPORDER"].to_numpy() == 2)[0])
    _set(person, second, "SPORDER", 1)
    return {"acs": acs}


def _source_household_id_moves(p):
    acs = _copies(p, "acs")
    person = acs["person"]
    _set(
        person,
        0,
        "source_household_id",
        int(person["source_household_id"].iloc[0]) + 1000,
    )
    return {"acs": acs}


def _source_person_id_moves(p):
    acs = _copies(p, "acs")
    _set(acs["person"], 0, "source_person_id", "99")
    return {"acs": acs}


def _source_year_moves(p):
    acs = _copies(p, "acs")
    _set(acs["person"], 0, "source_year", 2023)
    return {"acs": acs}


def _staging_has_a_person_the_selection_lost(p):
    return {"lineage": staging_lineage(p) | {("2024HU9999999", 1)}}


def _staging_lacks_a_selected_person(p):
    lineage = sorted(staging_lineage(p))
    return {"lineage": frozenset(lineage[1:])}


def _staging_pair_malformed(p):
    return {"lineage": [*staging_lineage(p), ("2024HU0000001",)]}


_REFUSALS = [
    # P5 on CPS rows (Route A's labels, complete).
    (
        _cps_missing_column,
        "CPS rows lack complete support metadata: 'household' is missing",
    ),
    (
        _cps_null_source,
        "CPS rows lack complete support metadata: 'person_source_id' has 1 null",
    ),
    (
        _cps_non_integer_source,
        "CPS 'person_source_id' must hold source ids: 1 value(s) are not integers",
    ),
    (_cps_unknown_channel, "CPS rows must carry channel 'asec' or 'puf_tax_detail'"),
    (
        _cps_asec_on_a_clone,
        "CPS rows must carry channel 'asec' exactly on clone index 0",
    ),
    (_cps_clone_index_three, "CPS clone indices must be 0, 1 or 2"),
    (
        _cps_spine_source_present,
        "CPS rows already carry 'marital_unit_spine_source_id'",
    ),
    (_cps_wrong_spine_tag, "CPS rows carry a spine tag other than 'asec_puf'"),
    # P5 on ACS rows (staging's labels, complete, one offset).
    (
        _acs_missing_column,
        "ACS rows lack complete support metadata: 'tax_unit' is missing",
    ),
    (
        _acs_null_channel,
        "ACS rows lack complete support metadata: 'spm_unit_support_channel' has 1 null",
    ),
    (_acs_channel_acs, "ACS rows must carry channel 'acs_2024_1yr'"),
    (_acs_clone_index, "ACS rows must carry clone index 0"),
    (_acs_offset_varies, "ACS 'family' ids must sit one constant offset"),
    (_acs_spine_source_present, "ACS rows already carry 'household_spine_source_id'"),
    (_acs_wrong_spine_tag, "ACS rows carry a spine tag other than 'acs_2024_1yr'"),
    # Q3: a null spine tag after the rewrite.
    (_acs_null_spine_tag, "A null spine tag: 1 ACS 'tax_unit' row(s)"),
    (_acs_spine_tag_absent, "A null spine tag: 24 ACS 'household' row(s)"),
    (
        _weights_misaligned,
        "ACS household weights must align with the ACS household table",
    ),
    # P6b: frame ids and memberships across origins.
    (
        _ids_collide_across_origins,
        "Frame ids collide: 'tax_unit_id' repeats across origins",
    ),
    (
        _membership_crosses_origins,
        "Memberships do not resolve: 1 ACS person(s)' 'person_spm_unit_id' match no ACS spm_unit row",
    ),
    (
        _group_without_a_person,
        "Memberships do not resolve: 1 ACS family row(s) hold no ACS person",
    ),
    # P6: a repeated (source id, rank) pair.
    (_rank_repeats_within_cps, "A (source id, rank) pair repeats in 'person'"),
    (_rank_repeats_across_origins, "A (source id, rank) pair repeats in 'household'"),
    # P7: SERIALNO/SPORDER lineage.
    (
        _serialno_column_missing,
        "ACS lineage does not survive: the ACS household table lacks ['SERIALNO']",
    ),
    (
        _serialno_null,
        "ACS lineage does not survive: every ACS household needs a non-empty string SERIALNO",
    ),
    (
        _serialno_repeats,
        "ACS lineage does not survive: SERIALNO repeats across ACS households",
    ),
    (
        _sporder_null,
        "ACS lineage does not survive: every ACS person needs an integral SPORDER: 1 value(s) are null",
    ),
    (
        _sporder_repeats,
        "ACS lineage does not survive: (SERIALNO, SPORDER) repeats on 2 ACS person(s)",
    ),
    (
        _source_household_id_moves,
        "ACS lineage does not survive: source_household_id differs from the linked household's native id",
    ),
    (
        _source_person_id_moves,
        "ACS lineage does not survive: source_person_id differs from str(SPORDER) on 1",
    ),
    (
        _source_year_moves,
        "ACS lineage does not survive: source_year must be 2024 on every ACS person; 1 differ",
    ),
    (
        _staging_has_a_person_the_selection_lost,
        "differs from staging's for the selected households; 1 missing",
    ),
    (
        _staging_lacks_a_selected_person,
        "differs from staging's for the selected households; 0 missing (such as []), 1 extra",
    ),
    (
        _staging_pair_malformed,
        "ACS lineage does not survive: staging's lineage holds ('2024HU0000001',), not a (SERIALNO, SPORDER) pair",
    ),
]


def _rewrite(population, corruption: dict) -> Frame:
    return to_assembled_provenance(
        corruption.get("cps", population.cps_tables),
        corruption.get("acs", population.acs_tables),
        cps_household_weights=corruption.get(
            "cps_weights", population.cps_household_weights
        ),
        acs_household_weights=corruption.get(
            "acs_weights", population.acs_household_weights
        ),
        staging_lineage=corruption.get("lineage", staging_lineage(population)),
    )


@pytest.mark.parametrize(
    ("corrupt", "message"),
    _REFUSALS,
    ids=[corrupt.__name__.lstrip("_") for corrupt, _ in _REFUSALS],
)
def test_each_rule_refuses_with_its_own_message(population, corrupt, message) -> None:
    with pytest.raises(OnePopulationContractError, match=re.escape(message)):
        _rewrite(population, corrupt(population))


def test_refusal_messages_are_distinct() -> None:
    messages = [message for _, message in _REFUSALS]
    assert len(set(messages)) == len(messages)


def test_union_form_refuses_a_fractional_sporder(population) -> None:
    union = with_staging_union_columns(population)
    acs = {entity: table.copy() for entity, table in union.acs_tables.items()}
    assert acs["person"]["SPORDER"].dtype == np.dtype("float64")
    acs["person"].loc[acs["person"].index[0], "SPORDER"] = 1.5
    with pytest.raises(
        OnePopulationContractError,
        match=re.escape(
            "every ACS person needs an integral SPORDER: 1 value(s) are not integers"
        ),
    ):
        _rewrite(union, {"acs": acs, "lineage": staging_lineage(population)})


def test_a_corrupted_input_raises_before_the_frame_does(population) -> None:
    # The contract's own refusal, not the Frame constructor's ValueError.
    corruption = _ids_collide_across_origins(population)
    with pytest.raises(OnePopulationContractError):
        _rewrite(population, corruption)


# ------------------------------------------------------------ cps_unit_ids


class TestCpsUnitIds:
    def test_keys_are_the_selection_keys(self, population) -> None:
        person = population.cps_tables["person"]
        household = population.cps_tables["household"].set_index("household_id")
        ids = cps_unit_ids(person)
        assert ids.index.name == "household_id"
        assert ids.index.tolist() == sorted(household.index)
        assert ids["income_year"].dtype == np.dtype("int64")
        native = household["household_support_clone_index"] == 0
        native_keys = ids.loc[native[native].index, "key"].tolist()
        assert sorted(native_keys) == sorted(population.cps_keys)
        assert selection.keys_sha256(native_keys) == selection.keys_sha256(
            population.cps_keys
        )
        # A clone household shares its source household's key.
        source = household["household_source_id"]
        assert (ids["key"] == ids["key"].loc[source.loc[ids.index]].to_numpy()).all()
        assert (ids["unit_id"].str.len() == 20).all()
        assert (
            ids["key"]
            == [
                selection.cps_unit_key(year, unit)
                for year, unit in zip(ids["income_year"], ids["unit_id"], strict=True)
            ]
        ).all()

    def test_reads_source_person_id_when_peridnum_is_absent(self, population) -> None:
        person = population.cps_tables["person"]
        pd.testing.assert_frame_equal(
            cps_unit_ids(person.drop(columns=["PERIDNUM"])), cps_unit_ids(person)
        )

    @pytest.mark.parametrize(
        ("corrupt", "message"),
        [
            (
                lambda person: person.drop(columns=["PERIDNUM"]).assign(
                    source_person_id=np.arange(len(person))
                ),
                "need source_person_id as a string of at least 20 digits",
            ),
            (
                lambda person: person.assign(PERIDNUM=person["PERIDNUM"].astype(float)),
                "need PERIDNUM as a string of at least 20 digits",
            ),
            (
                lambda person: person.assign(PERIDNUM=person["PERIDNUM"].str[:19]),
                "need PERIDNUM as a string of at least 20 digits",
            ),
            (
                lambda person: person.assign(
                    PERIDNUM=person["PERIDNUM"].where(person.index != 0, None)
                ),
                "need PERIDNUM as a string of at least 20 digits",
            ),
            (
                lambda person: person.drop(columns=["PERIDNUM", "source_person_id"]),
                "need PERIDNUM or source_person_id",
            ),
            (
                lambda person: person.assign(
                    source_year=person["source_year"]
                    .astype(float)
                    .where(person.index != 0, np.nan)
                ),
                "need an integral source_year",
            ),
        ],
        ids=["integer_fallback", "float", "short", "null", "absent", "null_year"],
    )
    def test_refuses_a_malformed_id(self, population, corrupt, message) -> None:
        person = population.cps_tables["person"].copy()
        with pytest.raises(OnePopulationContractError, match=re.escape(message)):
            cps_unit_ids(corrupt(person))

    def test_refuses_acs_rows(self, frame) -> None:
        with pytest.raises(
            OnePopulationContractError, match="as a string of at least 20 digits"
        ):
            cps_unit_ids(frame.table("person"))

    @pytest.mark.parametrize(
        ("column", "message"),
        [
            ("PERIDNUM", "disagree on PERIDNUM[:20] (H_IDNUM)"),
            ("source_year", "disagree on source_year"),
        ],
    )
    def test_refuses_a_household_whose_persons_disagree(
        self, population, column, message
    ) -> None:
        person = population.cps_tables["person"].copy()
        sizes = person["person_household_id"].map(
            person["person_household_id"].value_counts()
        )
        row = person.index[int(np.flatnonzero(sizes.to_numpy() > 1)[0])]
        if column == "PERIDNUM":
            value = person.loc[row, "PERIDNUM"]
            person.loc[row, "PERIDNUM"] = ("1" if value[0] != "1" else "2") + value[1:]
        else:
            person.loc[row, "source_year"] = person.loc[row, "source_year"] - 1
        with pytest.raises(OnePopulationContractError, match=re.escape(message)):
            cps_unit_ids(person)
