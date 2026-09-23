"""Tests split from packages/microcosm-build/tests/test_us_spm_independence_role.py."""

# ruff: noqa: F403, F405
from test_support.microcosm_build.us_spm_independence_role import *


class TestManifestAndPlan:
    def test_stage_spec_pins_the_source_rule(self) -> None:
        spec = us_spm_independence_role_stage_spec()

        assert spec.stage == US_SPM_INDEPENDENCE_ROLE_STAGE_NAME
        assert (
            tuple(spec.outputs) == US_SPM_INDEPENDENCE_ROLE_OUTPUT_COLUMNS == (_ROLE,)
        )
        assert US_SPM_INDEPENDENCE_ROLE_NONCONSTANT_PERSON_COLUMNS == (_ROLE,)
        assert [operation.kind for operation in spec.operations] == [
            "read_table",
            US_SPM_INDEPENDENCE_ROLE_OPERATION_KIND,
        ]
        assert (
            "SPM_HEAD == 1 OR (A_FAMTYP in {1,4} AND A_FAMREL in {1,2})" in spec.notes
        )
        assert "derive_spm_role_source" in spec.notes
        assert "exactly one SPM_HEAD" in spec.notes
        assert "no SPM unit is left without a classified adult" in spec.notes

    def test_handler_and_plan_are_wired_between_relationships_and_puf_support(
        self,
    ) -> None:
        handlers = us_source_operation_handlers()
        assert (
            handlers[US_SPM_INDEPENDENCE_ROLE_OPERATION_KIND]
            is derive_us_spm_independence_role_from_manifest
        )
        assert US_SPM_INDEPENDENCE_ROLE_STAGE_NAME in US_DONORS
        assert (
            US_DONORS[US_SPM_INDEPENDENCE_ROLE_STAGE_NAME].survey == "Census CPS ASEC"
        )
        order = US_STAGE_NAMES.index
        assert (
            order(US_RELATIONSHIP_INPUTS_STAGE_NAME)
            < order(US_SPM_INDEPENDENCE_ROLE_STAGE_NAME)
            < order(US_PUF_SUPPORT_STAGE_NAME)
        )

    def test_required_columns_are_the_derivations_own(self) -> None:
        """The stage restates only the frame identity; the raw tail is imported."""

        assert US_SPM_INDEPENDENCE_ROLE_REQUIRED_SOURCE_COLUMNS == (
            "person_id",
            "person_spm_unit_id",
            "source_year",
            "PERIDNUM",
            "age",
            "source_household_id",
            "source_person_id",
            "source_row_id",
            *_REQUIRED_RAW_CHECKS,
        )
        assert US_SPM_INDEPENDENCE_ROLE_OPTIONAL_SOURCE_COLUMNS == _OPTIONAL_RAW_CHECKS

    @pytest.mark.parametrize("column", US_SPM_INDEPENDENCE_ROLE_REQUIRED_SOURCE_COLUMNS)
    def test_the_derivation_itself_requires_each_column(
        self, tmp_path: Path, column: str
    ) -> None:
        """Consistency: every column the stage names, derive_spm_role_source refuses."""

        source = _hand_built_source()
        path, pin = _write_source(tmp_path, source)
        person = _person_table(source).drop(columns=[column])
        parent = tmp_path / "parent.h5"
        person.to_hdf(parent, key="person")
        pd.DataFrame(
            {
                "spm_unit_id": np.sort(
                    _person_table(source)["person_spm_unit_id"].unique()
                )
            }
        ).to_hdf(parent, key="spm_unit")
        with pytest.raises(ValueError, match=f"missing required columns.*{column}"):
            derive_spm_role_source(
                parent,
                {_INCOME_YEAR: path},
                expected_parent_sha256=_digest(parent),
                source_pins={_INCOME_YEAR: pin},
            )

    def test_resolver_refuses_unpinned_years_and_keeps_explicit_paths(
        self, tmp_path: Path
    ) -> None:
        explicit = tmp_path / "pppub25.csv"
        resolved = resolve_asec_spm_role_source_paths(
            {_INCOME_YEAR: explicit}, income_years=(_INCOME_YEAR,)
        )
        assert resolved == {_INCOME_YEAR: explicit}
        with pytest.raises(ValueError, match="pinned income years"):
            resolve_asec_spm_role_source_paths({1999: explicit}, income_years=(1999,))
        with pytest.raises(ValueError, match="have no pinned ASEC SPM role source"):
            resolve_asec_spm_role_source_paths(None, income_years=(1999,))
        assert set(ASEC_SPM_ROLE_SOURCES) == {2022, 2023, 2024}


class TestDerivation:
    def test_roles_equal_the_certified_derivation_and_the_raw_rule(
        self, population
    ) -> None:
        source, path, pin, frame = population

        result = _run(frame, path, pin)

        person = result.table("person")
        role = person[_ROLE].to_numpy()
        assert person[_ROLE].dtype == bool
        # The raw rule on the source rows, joined by PERIDNUM (clones repeat it).
        expected = (
            person["PERIDNUM"]
            .map(
                dict(zip(source.PERIDNUM, independent_minor_role(source), strict=True))
            )
            .to_numpy(dtype=bool)
        )
        assert np.array_equal(role, expected)
        assert role.tolist() == [
            True,
            True,
            False,
            True,
            False,
            True,
            True,
            True,
            False,
        ]
        # Unit 100's clone (native unit 40) carries the same roles as the native.
        assert person.loc[person.person_spm_unit_id.eq(13), _ROLE].tolist() == [
            True,
            True,
            False,
        ]
        # The stage is the certified derivation, run on a projection of the frame.
        parent = path.parent / "direct.h5"
        frame.table("person").to_hdf(parent, key="person")
        frame.table("spm_unit").to_hdf(parent, key="spm_unit")
        direct = derive_spm_role_source(
            parent,
            {_INCOME_YEAR: path},
            expected_parent_sha256=_digest(parent),
            source_pins={_INCOME_YEAR: pin},
        )
        assert np.array_equal(direct.role, role)

    def test_provenance_rides_on_the_frame(self, population) -> None:
        _source, path, pin, frame = population

        result = _run(frame, path, pin)

        provenance = result.metadata[US_SPM_INDEPENDENCE_ROLE_PROVENANCE_KEY]
        assert provenance["unmatched_persons"] == 0
        assert provenance["adult_child_person_count_mismatch_units"] == 0
        assert provenance["native_spm_units"] == 4
        assert provenance["total_source_units"] == 3
        assert provenance["minor_only_units_resolved"] == 3
        assert provenance["independent_minor_persons"] == 5
        assert provenance["weights_used"] is False
        assert provenance["ages_changed"] is False
        assert len(provenance["frame_projection_sha256"]) == 64
        assert provenance["dataset_sha256"] == provenance["frame_projection_sha256"]
        assert set(provenance["frame_projection_columns"]) == set(
            US_SPM_INDEPENDENCE_ROLE_REQUIRED_SOURCE_COLUMNS
        ) | {"A_FAMTYP", "A_FAMREL", "A_SPOUSE", "PECOHAB"}

    def test_refuses_a_unit_with_two_heads(self, tmp_path: Path) -> None:
        source = _hand_built_source()
        source.loc[source.index[1], "SPM_HEAD"] = 1
        path, pin = _write_source(tmp_path, source)
        frame = _frame(_person_table(source))
        with pytest.raises(SourceRuntimeError, match="exactly one SPM head per unit"):
            _run(frame, path, pin)

    def test_refuses_a_unit_with_no_head(self, tmp_path: Path) -> None:
        source = _hand_built_source()
        source.loc[source.index[3], "SPM_HEAD"] = 0
        path, pin = _write_source(tmp_path, source)
        frame = _frame(_person_table(source))
        with pytest.raises(SourceRuntimeError, match="exactly one SPM head per unit"):
            _run(frame, path, pin)

    def test_refuses_a_unit_left_without_a_classified_adult(
        self, tmp_path: Path
    ) -> None:
        """A 14-year-old head: the role is True but the age gate leaves no adult."""

        source = _hand_built_source()
        source.loc[source.index[5], "A_AGE"] = 14
        source.loc[source.index[5], "SPM_HAGE"] = 14
        source.loc[source.index[5], ["SPM_NUMADULTS", "SPM_NUMKIDS"]] = [0, 1]
        path, pin = _write_source(tmp_path, source)
        frame = _frame(_person_table(source))
        with pytest.raises(SourceRuntimeError, match="unresolved zero-adult SPM unit"):
            _run(frame, path, pin)

    def test_refuses_a_person_with_no_asec_origin(self, population) -> None:
        _source, path, pin, frame = population
        person = frame.table("person").copy()
        person.loc[person.index[0], ["PERIDNUM", "source_person_id"]] = "9" * 22
        with pytest.raises(SourceRuntimeError, match="unmatched parent persons"):
            _run(_frame(person), path, pin)

    def test_refuses_a_census_count_that_does_not_reconcile(
        self, tmp_path: Path
    ) -> None:
        source = _hand_built_source()
        source.loc[source.SPM_ID.eq(200), "SPM_NUMADULTS"] = 2
        path, pin = _write_source(tmp_path, source)
        frame = _frame(_person_table(source))
        with pytest.raises(
            SourceRuntimeError, match="adult/child/person count reconciliation"
        ):
            _run(frame, path, pin)

    def test_refuses_a_native_unit_mixing_two_source_units(self, population) -> None:
        _source, path, pin, frame = population
        person = frame.table("person").copy()
        # Move unit 200's child into native unit 10 (source unit 100).
        person.loc[person.index[4], "person_spm_unit_id"] = 10
        with pytest.raises(SourceRuntimeError, match="combines distinct source units"):
            _run(_frame(person), path, pin)

    @pytest.mark.parametrize("column", US_SPM_INDEPENDENCE_ROLE_REQUIRED_SOURCE_COLUMNS)
    def test_missing_source_column_is_named(self, population, column: str) -> None:
        """The handler names the column; the frame wrapper refuses it first when
        the frame cannot even be built or its income years cannot be read."""

        _source, path, pin, frame = population
        person = frame.table("person").drop(columns=[column])
        with pytest.raises(SourceRuntimeError, match=column):
            derive_us_spm_independence_role_from_manifest(
                person, _operation(), _context(frame, path, pin)
            )
        if column in ("person_id", "person_spm_unit_id"):
            return  # the Frame schema refuses a person table without these
        if column == "source_year":
            with pytest.raises(ValueError, match=column):
                _run(_frame(person), path, pin)
            return
        with pytest.raises(SourceRuntimeError, match=column):
            _run(_frame(person), path, pin)

    def test_handler_refuses_wrong_operation_missing_table_context_and_paths(
        self, population
    ) -> None:
        _source, path, pin, frame = population
        person = frame.table("person")
        wrong = SourceStageSpec.from_mapping(
            {
                "stage": "test",
                "survey": "test",
                "source": "https://example.com",
                "grain": "person",
                "operations": [{"kind": "derive"}],
                "outputs": [_ROLE],
            }
        ).operations[0]
        with pytest.raises(SourceRuntimeError, match="unexpected operation"):
            derive_us_spm_independence_role_from_manifest(
                person, wrong, _context(frame, path, pin)
            )
        with pytest.raises(SourceRuntimeError, match="person table"):
            derive_us_spm_independence_role_from_manifest(
                None, _operation(), _context(frame, path, pin)
            )
        with pytest.raises(SourceRuntimeError, match="runtime context"):
            derive_us_spm_independence_role_from_manifest(person, _operation(), None)
        no_spm = SourceRuntimeContext(
            config=_context(frame, path, pin).config, tables={}
        )
        with pytest.raises(SourceRuntimeError, match="spm_unit"):
            derive_us_spm_independence_role_from_manifest(person, _operation(), no_spm)
        no_paths = SourceRuntimeContext(
            config=SourceRuntimeConfig(seed=0, target_year=_INCOME_YEAR),
            tables={"spm_unit": frame.table("spm_unit")},
        )
        with pytest.raises(SourceRuntimeError, match="CSV paths"):
            derive_us_spm_independence_role_from_manifest(
                person, _operation(), no_paths
            )

    def test_refuses_an_unpinned_csv(self, population) -> None:
        _source, path, pin, frame = population
        with pytest.raises(SourceRuntimeError, match="CSV SHA-256"):
            _run(frame, path, replace(pin, csv_sha256="0" * 64))

    def test_missing_csv_refusal_names_the_stage(self, population) -> None:
        _source, path, pin, frame = population
        with pytest.raises(SourceRuntimeError, match="US SPM independence role"):
            _run(frame, path.with_name("missing.csv"), pin)


class TestFrameAndGate:
    def test_gate_details_are_json_serializable(self, population) -> None:
        _source, path, pin, frame = population
        gate = us_spm_independence_role_signal_gate(_run(frame, path, pin))
        serialized = json.loads(json.dumps(gate.details, allow_nan=False))
        assert serialized["derivation"]["persons_joined"] == frame.n("person")
        assert serialized["derivation"]["source_checks"]

    @pytest.mark.parametrize("change", ["swap_roles", "change_age", "stale_count"])
    def test_gate_refuses_stale_derivation(self, population, change) -> None:
        _source, path, pin, frame = population
        result = _run(frame, path, pin)
        tables = {entity: result.table(entity).copy() for entity in result.entities}
        metadata = json.loads(
            json.dumps(us_spm_independence_role_summary(result)["derivation"])
        )
        person = tables["person"]
        if change == "swap_roles":
            indices = [
                person.index[person[_ROLE]].tolist()[0],
                person.index[~person[_ROLE]].tolist()[0],
            ]
            person.loc[indices, _ROLE] = ~person.loc[indices, _ROLE]
        elif change == "change_age":
            person.loc[person.index[0], "age"] += 1
        else:
            metadata["persons_joined"] += 1
        altered = Frame(
            tables,
            result.schema,
            {entity: result.weights_for(entity) for entity in result.weighted_entities},
            result.strata,
            metadata={US_SPM_INDEPENDENCE_ROLE_PROVENANCE_KEY: metadata},
        )
        gate = us_spm_independence_role_signal_gate(altered)
        assert not gate.passed
        assert any("provenance" in failure for failure in gate.failures)

    def test_gate_refuses_integer_role_and_missing_age(self, population) -> None:
        _source, path, pin, frame = population
        person = _run(frame, path, pin).table("person").copy()
        person[_ROLE] = person[_ROLE].astype(int)
        gate = us_spm_independence_role_signal_gate(_frame(person))
        assert not gate.passed
        assert any("non-Boolean" in failure for failure in gate.failures)
        gate = us_spm_independence_role_signal_gate(_frame(person.drop(columns="age")))
        assert not gate.passed
        assert gate.details["missing"] == ["age"]

    def test_idempotent_and_leaves_everything_else_untouched(self, population) -> None:
        _source, path, pin, frame = population

        result = _run(frame, path, pin)

        assert _ROLE not in frame.table("person")
        assert set(result.table("person").columns) == set(
            frame.table("person").columns
        ) | {_ROLE}
        for entity in frame.entities:
            if entity == "person":
                continue
            assert result.table(entity).equals(frame.table(entity))
        assert np.array_equal(
            result.table("person")["person_spm_unit_id"].to_numpy(),
            frame.table("person")["person_spm_unit_id"].to_numpy(),
        )
        repeated = _run(result, path, pin)
        assert repeated.table("person").equals(result.table("person"))
        assert (
            repeated.metadata[US_SPM_INDEPENDENCE_ROLE_PROVENANCE_KEY][
                "unmatched_persons"
            ]
            == 0
        )

    def test_existing_role_cannot_bypass_the_source_checks(self, population) -> None:
        _source, path, pin, frame = population
        result = _run(frame, path, pin)
        with pytest.raises(SourceRuntimeError, match="CSV SHA-256"):
            _run(result, path, replace(pin, csv_sha256="0" * 64))

        person = result.table("person").copy()
        person.loc[person.index[0], _ROLE] = not person.loc[person.index[0], _ROLE]
        with pytest.raises(ValueError, match="disagrees with the pinned Census"):
            _run(_frame(person), path, pin)

    @pytest.mark.parametrize("invalid", [None, "False", 0])
    def test_existing_role_refuses_non_boolean_or_missing_values(
        self, population, invalid
    ) -> None:
        _source, path, pin, frame = population
        person = _run(frame, path, pin).table("person").copy()
        person[_ROLE] = person[_ROLE].astype(object)
        person.loc[person.index[0], _ROLE] = invalid
        with pytest.raises(ValueError, match="non-null Boolean observations"):
            _run(_frame(person), path, pin)

    def test_role_without_source_provenance_fails_gate(self, population) -> None:
        _source, path, pin, frame = population
        person = _run(frame, path, pin).table("person")
        gate = us_spm_independence_role_signal_gate(_frame(person))
        assert not gate.passed
        assert (
            "SPM independence role has no source derivation provenance."
            in gate.failures
        )

    def test_requires_the_us_schema_and_source_year(self, population) -> None:
        _source, path, pin, frame = population
        person = frame.table("person").drop(columns=["source_year"])
        with pytest.raises(ValueError, match="source_year"):
            with_us_spm_independence_role(
                _frame(person),
                seed=0,
                time_period=2024,
                asec_spm_role_source_paths={_INCOME_YEAR: path},
                source_pins={_INCOME_YEAR: pin},
            )
        with pytest.raises(ValueError, match="explicit CSV paths"):
            with_us_spm_independence_role(
                frame, seed=0, time_period=2024, source_pins={_INCOME_YEAR: pin}
            )

    def test_gate_passes_on_a_realistic_population_and_reports_the_composition(
        self, tmp_path: Path
    ) -> None:
        source = _random_source(seed=7, n_units=300)
        path, pin = _write_source(tmp_path, source)
        frame = _frame(_person_table(source))

        result = _run(frame, path, pin)
        gate = us_spm_independence_role_signal_gate(result)

        assert gate.passed, gate.failures
        summary = us_spm_independence_role_summary(result)
        assert summary["spm_composition"]["status"] == "PASS"
        assert summary["spm_composition"]["role_source"] == "source_column"
        assert summary["spm_composition"]["n_units_without_classified_adult"] == 0
        assert summary["independent_minor_persons"] == int(
            (source.A_AGE.between(15, 17) & source.SPM_HEAD.eq(1)).sum()
        )
        assert 0.40 <= summary["role_share"] <= 0.75
        assert 0.003 <= summary["minor_role_share"] <= 0.06
        assert summary["derivation"]["unmatched_persons"] == 0
        assert check_spm_composition(result).status == "PASS"

    def test_gate_names_a_missing_column(self, population) -> None:
        _source, _path, _pin, frame = population
        gate = us_spm_independence_role_signal_gate(frame)
        assert not gate.passed
        assert gate.details["missing"] == [_ROLE]

    def test_gate_fails_a_frame_whose_role_leaves_a_unit_unclassified(
        self, population
    ) -> None:
        """A stored False on the only teen of a minor-only unit: the engine refuses."""

        _source, path, pin, frame = population
        result = _run(frame, path, pin)
        person = result.table("person").copy()
        person.loc[person.person_spm_unit_id.eq(12), _ROLE] = False
        gate = us_spm_independence_role_signal_gate(_frame(person))

        assert not gate.passed
        assert any("no classified adult" in failure for failure in gate.failures)
        assert any("SPM_COMPOSITION_REQUIRED" in failure for failure in gate.failures)

    def test_gate_fails_a_degenerate_or_missing_valued_role(self, population) -> None:
        _source, path, pin, frame = population
        result = _run(frame, path, pin)
        person = result.table("person").copy()
        person[_ROLE] = True
        gate = us_spm_independence_role_signal_gate(_frame(person))
        assert not gate.passed
        assert any("degenerate" in failure for failure in gate.failures)

        person = result.table("person").copy()
        person[_ROLE] = person[_ROLE].astype(object)
        person.loc[person.index[0], _ROLE] = None
        gate = us_spm_independence_role_signal_gate(_frame(person))
        assert not gate.passed
        assert any("missing for 1 person" in failure for failure in gate.failures)


@pytest.mark.parametrize("seed", range(12))
def test_seeded_agreement_with_the_raw_rule_and_the_derivation(
    tmp_path: Path, seed: int
) -> None:
    """On random populations the stage, the raw rule and the derivation agree.

    The stage adds nothing to the rule: for every person its role equals
    ``independent_minor_role`` evaluated on that person's Census row, and
    equals ``derive_spm_role_source`` run directly on an H5 of the same frame;
    afterwards no unit is left without a classified adult, and the counts the
    derivation reconciled are the frame's own.
    """

    rng = np.random.default_rng(seed)
    source = _random_source(seed=seed, n_units=int(rng.integers(40, 160)))
    clone_units = tuple(
        int(unit)
        for unit in rng.choice(
            source.SPM_ID.unique(), size=int(rng.integers(0, 6)), replace=False
        )
    )
    path, pin = _write_source(tmp_path, source)
    person = _person_table(source, clone_units=clone_units)
    frame = _frame(
        person,
        weights=rng.uniform(
            10.0, 500.0, size=int(person["person_household_id"].nunique())
        ),
    )

    result = _run(frame, path, pin)

    person = result.table("person")
    role = person[_ROLE].to_numpy(dtype=bool)
    by_source = dict(zip(source.PERIDNUM, independent_minor_role(source), strict=True))
    assert np.array_equal(role, person["PERIDNUM"].map(by_source).to_numpy(dtype=bool))

    parent = tmp_path / f"direct-{seed}.h5"
    frame.table("person").to_hdf(parent, key="person")
    frame.table("spm_unit").to_hdf(parent, key="spm_unit")
    direct = derive_spm_role_source(
        parent,
        {_INCOME_YEAR: path},
        expected_parent_sha256=_digest(parent),
        source_pins={_INCOME_YEAR: pin},
    )
    assert np.array_equal(direct.role, role)

    composition = check_spm_composition(result)
    assert composition.status == "PASS", composition.failures
    provenance = result.metadata[US_SPM_INDEPENDENCE_ROLE_PROVENANCE_KEY]
    assert provenance["persons_joined"] == len(person)
    assert provenance["native_spm_units"] == result.n("spm_unit")
    assert provenance["unmatched_persons"] == 0
    assert provenance["adult_child_person_count_mismatch_units"] == 0
    teens = person["age"].between(15, 17).to_numpy()
    assert provenance["independent_minor_persons"] == int((role & teens).sum())
