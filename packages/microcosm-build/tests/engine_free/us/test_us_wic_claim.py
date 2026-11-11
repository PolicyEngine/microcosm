"""Tests split from packages/microcosm-build/tests/test_us_wic_claim.py."""

# ruff: noqa: F403, F405
from test_support.microcosm_build.us_wic_claim import *


class TestManifestAndProvenance:
    def test_stage_locks_the_exact_official_category_rate_contract(self) -> None:
        spec = us_wic_claim_stage_spec()

        assert spec.stage == US_WIC_CLAIM_STAGE_NAME
        assert tuple(spec.outputs) == US_WIC_CLAIM_OUTPUT_COLUMNS == (_OUTPUT,)
        assert US_WIC_CLAIM_NONCONSTANT_PERSON_COLUMNS == (_OUTPUT,)
        assert US_WIC_CLAIM_REQUIRED_SOURCE_COLUMNS == (
            "age",
            "is_female",
            "is_pregnant",
            "own_children_in_household",
            "person_family_id",
        )
        assert [operation.kind for operation in spec.operations] == [
            "read_table",
            "derive_wic_claim",
        ]
        assert _operation().parameters == {
            "seed_from_build_config": True,
            "category_rates": {
                "source": WIC_CLAIM_FNS_SOURCE_URL,
                "vintage": "CY2022",
                "values": _RATES,
            },
        }
        assert "after pregnancy" in spec.notes
        assert "no breastfeeding assessment" in spec.notes

    def test_source_urls_lock_archived_files_lines_and_fns_pdf(self) -> None:
        commit = "42ed5d45c56df80d754fbe24cce21cfeb8d05cbe"
        assert commit in WIC_CLAIM_ARCHIVED_DERIVATION_URL
        assert WIC_CLAIM_ARCHIVED_DERIVATION_URL.endswith(
            "/datasets/cps/cps.py#L684-L691"
        )
        assert WIC_CLAIM_ARCHIVED_PARAMETERS_URL.endswith(
            "/parameters/take_up/wic_takeup.yaml#L1-L33"
        )
        assert WIC_CLAIM_ARCHIVED_RANDOMNESS_URL.endswith("/utils/randomness.py#L5-L28")
        assert WIC_CLAIM_FNS_SOURCE_URL == (
            "https://fns-prod.azureedge.us/sites/default/files/resource-files/"
            "wic-eer-2022-summary.pdf"
        )

    def test_handler_and_plan_run_wic_after_pregnancy_before_support(self) -> None:
        assert (
            us_source_operation_handlers()["derive_wic_claim"]
            is derive_us_wic_claim_from_manifest
        )
        assert US_WIC_CLAIM_STAGE_NAME in US_DONORS
        assert (
            US_STAGE_NAMES.index(US_PREGNANCY_STAGE_NAME)
            < US_STAGE_NAMES.index(US_WIC_CLAIM_STAGE_NAME)
            < US_STAGE_NAMES.index(US_PUF_SUPPORT_STAGE_NAME)
        )


class TestDerivation:
    def test_categories_follow_pe_order_with_collapsed_postpartum(self) -> None:
        rows = [
            {
                "person_family_id": 1,
                "age": 28,
                "is_female": True,
                "is_pregnant": True,
                "own_children_in_household": 1,
            },
            {"person_family_id": 1, "age": 0},
            {
                "person_family_id": 2,
                "age": 30,
                "is_female": True,
                "own_children_in_household": 1,
            },
            {"person_family_id": 2, "age": 0},
            {
                "person_family_id": 3,
                "age": 30,
                "is_female": False,
                "own_children_in_household": 1,
            },
            {"person_family_id": 3, "age": 0},
            {"age": 4},
            {"age": 5},
        ]
        derived = with_us_wic_claim_input(_frame(rows), seed=11, time_period=2024)
        summary = us_wic_claim_summary(derived)

        assert summary["category_assignment_order"] == [
            "pregnant",
            "postpartum",
            "infant",
            "child",
            "none",
        ]
        assert summary["category_counts"] == {
            "pregnant": 1,
            "postpartum": 1,
            "infant": 3,
            "child": 1,
            "none": 2,
        }
        assert summary["breastfeeding_source_available"] is False
        assert summary["breastfeeding_rate_validated_but_unassigned"] == 0.663

    def test_draws_are_reproducible_and_keyed_by_source_identity(self) -> None:
        rows = [
            {
                "person_id": 1,
                "age": 3,
                "source_year": 2024,
                "source_household_id": 50,
                "source_person_id": 7,
            },
            {
                "person_id": 2,
                "age": 3,
                "source_year": 2024,
                "source_household_id": 50,
                "source_person_id": 7,
            },
        ]
        person = _frame(rows).table("person")
        first = _derive(person, seed=9)
        second = _derive(person, seed=9)

        assert first[_OUTPUT].tolist() == second[_OUTPUT].tolist()
        assert first[_OUTPUT].iloc[0] == first[_OUTPUT].iloc[1]

        many = _frame([{"age": 3} for _ in range(500)]).table("person")
        assert not np.array_equal(
            _derive(many, seed=1)[_OUTPUT].to_numpy(),
            _derive(many, seed=2)[_OUTPUT].to_numpy(),
        )

    def test_multispine_identity_precedes_source_local_identity(self) -> None:
        person = _frame(
            [
                {
                    "person_source_id": 10,
                    "person_spine_source_id": 1,
                    "source_year": 2024,
                    "source_household_id": 50,
                    "source_person_id": 7,
                },
                {
                    "person_source_id": 20,
                    "person_spine_source_id": 2,
                    "source_year": 2024,
                    "source_household_id": 50,
                    "source_person_id": 7,
                },
                {
                    "person_source_id": 10,
                    "person_spine_source_id": 1,
                    "source_year": 2024,
                    "source_household_id": 50,
                    "source_person_id": 7,
                },
            ]
        ).table("person")

        assert _stable_person_keys(person).tolist() == [
            "source:10",
            "source:20",
            "source:10",
        ]

        person.loc[person.index[0], "person_source_id"] = np.nan
        with pytest.raises(SourceRuntimeError, match="assembly source identity"):
            _stable_person_keys(person)

    @pytest.mark.parametrize("column", US_WIC_CLAIM_REQUIRED_SOURCE_COLUMNS)
    def test_missing_source_columns_fail_closed(self, column: str) -> None:
        person = _frame([{}]).table("person").drop(columns=[column])
        with pytest.raises(SourceRuntimeError, match=column):
            _derive(person)

    @pytest.mark.parametrize(
        ("updates", "message"),
        [
            ({"age": np.nan}, "age"),
            ({"age": -1}, "age"),
            ({"is_female": 2}, "is_female"),
            ({"is_pregnant": None}, "is_pregnant"),
            ({"own_children_in_household": 1.5}, "own_children"),
            ({"is_female": False, "is_pregnant": True}, "nonfemale"),
        ],
    )
    def test_invalid_category_sources_fail_closed(
        self,
        updates: dict[str, object],
        message: str,
    ) -> None:
        with pytest.raises(SourceRuntimeError, match=message):
            _derive(_frame([updates]).table("person"))

    def test_missing_family_membership_fails_closed_at_raw_stage_boundary(self) -> None:
        person = _frame([{}]).table("person").copy()
        person.loc[0, "person_family_id"] = np.nan
        with pytest.raises(SourceRuntimeError, match="person_family_id"):
            _derive(person)

    def test_partial_stable_identity_fails_instead_of_changing_clone_key(self) -> None:
        person = _frame([{"age": 3}]).table("person").drop(columns=["source_person_id"])
        with pytest.raises(SourceRuntimeError, match="partial"):
            _derive(person)

    def test_wrong_operation_missing_frame_and_parameter_drift_are_rejected(
        self,
    ) -> None:
        person = _frame([{}]).table("person")
        with pytest.raises(SourceRuntimeError, match="unexpected operation"):
            derive_us_wic_claim_from_manifest(
                person,
                SourceOperationSpec(kind="wrong", parameters={}),
                _context(),
            )
        with pytest.raises(SourceRuntimeError, match="person table"):
            derive_us_wic_claim_from_manifest(None, _operation(), _context())

        mutations = []
        for mutate in ("source", "vintage", "child", "extra"):
            parameters = deepcopy(dict(_operation().parameters))
            if mutate == "source":
                parameters["category_rates"]["source"] = "https://example.com"
            elif mutate == "vintage":
                parameters["category_rates"]["vintage"] = "CY2021"
            elif mutate == "child":
                parameters["category_rates"]["values"]["child"] = 0.99
            else:
                parameters["category_rates"]["values"]["extra"] = 0.1
            mutations.append(parameters)
        for parameters in mutations:
            with pytest.raises(SourceRuntimeError):
                derive_us_wic_claim_from_manifest(
                    person,
                    SourceOperationSpec(
                        kind="derive_wic_claim",
                        parameters=parameters,
                    ),
                    _context(),
                )


class TestFrameAndGate:
    def test_wrapper_recomputes_stale_surface_and_is_idempotent_by_equality(
        self,
    ) -> None:
        frame = _frame(_plausible_rows())
        derived = with_us_wic_claim_input(frame, seed=0, time_period=2024)
        assert us_wic_claim_signal_gate(derived).passed
        assert with_us_wic_claim_input(derived, seed=0, time_period=2024) is derived

        stale_person = derived.table("person").copy()
        stale_person[_OUTPUT] = ~stale_person[_OUTPUT]
        stale = _replace_person(derived, stale_person)
        healed = with_us_wic_claim_input(stale, seed=0, time_period=2024)
        np.testing.assert_array_equal(
            healed.table("person")[_OUTPUT],
            derived.table("person")[_OUTPUT],
        )

    def test_constant_default_true_is_recomputed(self) -> None:
        frame = _frame([{**row, _OUTPUT: True} for row in _plausible_rows()])
        derived = with_us_wic_claim_input(frame, seed=0, time_period=2024)
        assert derived.table("person")[_OUTPUT].nunique() == 2
        assert us_wic_claim_signal_gate(derived).passed

    def test_support_clones_keep_identical_claims_and_channel_signal(self) -> None:
        derived = with_us_wic_claim_input(
            _frame(_plausible_rows()), seed=0, time_period=2024
        )
        cloned = clone_us_frame_for_puf_support(derived)
        refreshed = with_us_wic_claim_input(cloned, seed=0, time_period=2024)
        summary = us_wic_claim_summary(refreshed)

        assert refreshed is cloned
        assert summary["clone_group_count"] == len(derived.table("person"))
        assert summary["clone_claim_mismatch_count"] == 0
        assert summary["clone_category_mismatch_count"] == 0
        assert set(summary["channel_weighted_claim_shares"]) == {
            "asec",
            PUF_TAX_DETAIL_SUPPORT_CHANNEL,
        }
        assert us_wic_claim_signal_gate(refreshed).passed

    def test_gate_allows_unpaired_selected_channels_but_rejects_clone_mismatch(
        self,
    ) -> None:
        derived = with_us_wic_claim_input(
            _frame(_plausible_rows()), seed=0, time_period=2024
        )
        selected_person = derived.table("person").copy()
        selected_person["person_support_channel"] = np.where(
            np.arange(len(selected_person)) % 2,
            "asec",
            PUF_TAX_DETAIL_SUPPORT_CHANNEL,
        )
        selected = _replace_person(derived, selected_person)
        assert us_wic_claim_signal_gate(selected).passed

        cloned = clone_us_frame_for_puf_support(derived)
        broken_person = cloned.table("person").copy()
        duplicate = broken_person["source_person_id"].duplicated(keep="first")
        row = int(np.flatnonzero(duplicate.to_numpy())[0])
        broken_person.loc[row, _OUTPUT] = not bool(broken_person.loc[row, _OUTPUT])
        broken = _replace_person(cloned, broken_person)
        gate = us_wic_claim_signal_gate(broken)
        assert not gate.passed
        assert any("clone claim mismatch" in failure for failure in gate.failures)

    def test_gate_rejects_missing_constant_and_none_category_claim(self) -> None:
        assert not us_wic_claim_signal_gate(_frame([{}])).passed

        constant = _frame([{_OUTPUT: True} for _ in range(10)])
        assert not us_wic_claim_signal_gate(constant).passed

        derived = with_us_wic_claim_input(
            _frame(_plausible_rows()), seed=0, time_period=2024
        )
        person = derived.table("person").copy()
        none_row = person.index[person["age"] == 35][0]
        person.loc[none_row, _OUTPUT] = True
        gate = us_wic_claim_signal_gate(_replace_person(derived, person))
        assert not gate.passed
        assert any("none weighted claim share" in failure for failure in gate.failures)

        invalid = derived.table("person").copy()
        invalid[_OUTPUT] = invalid[_OUTPUT].astype(object)
        invalid.loc[invalid.index[0], _OUTPUT] = "not-a-boolean"
        gate = us_wic_claim_signal_gate(_replace_person(derived, invalid))
        assert not gate.passed
        assert any("nonnumeric" in failure for failure in gate.failures)

    def test_non_us_schema_is_rejected(self) -> None:
        frame = Frame(
            {
                "person": pd.DataFrame({"person_id": [1], "person_household_id": [1]}),
                "household": pd.DataFrame({"household_id": [1]}),
            },
            EntitySchema(group_entities=("household",)),
            {"household": Weights(np.ones(1), WeightKind.DESIGN)},
        )
        with pytest.raises(ValueError, match="US WIC"):
            with_us_wic_claim_input(frame, seed=0, time_period=2024)


def test_release_promotion_probe_and_retired_gap_removal() -> None:
    manifest = load_release_input_coverage_manifest()
    assert _OUTPUT in RESTORED_REFERENCE_ECPS_REQUIRED_INPUTS
    assert _OUTPUT in manifest.required_columns
    assert _OUTPUT not in manifest.reviewed_exclusions
    assert _OUTPUT in US_RELEASE_REQUIRED_PERSON_SOURCE_COLUMNS

    probe = next(
        probe
        for probe in us_release_reform_coverage_probes()
        if probe.id == "wic_claim_neutralization"
    )
    assert probe.neutralized_variable == _OUTPUT
    assert probe.binding_inputs == (_OUTPUT,)
    assert probe.budget_measure == "wic"
    assert probe.effect_direction == "baseline_minus_reform"
    assert probe.expected_sign == "positive"
    assert probe.min_abs_effect == 25_000_000.0

    known_gaps = __import__("json").loads(
        files("microcosm.build.us").joinpath("ecps_parity_known_gaps.json").read_text()
    )["known_gaps"]
    assert _OUTPUT not in known_gaps
