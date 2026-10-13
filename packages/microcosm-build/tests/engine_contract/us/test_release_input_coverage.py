"""Tests split from packages/microcosm-build/tests/test_release_input_coverage.py."""

# ruff: noqa: F403, F405
from test_support.microcosm_build.release_input_coverage import *


class TestReleaseInputCoverageGate:
    def test_full_required_set_with_signal_passes(self) -> None:
        # Case 1: both required columns present and carrying signal; the excluded
        # column is absent (dormant), which is reported, not failed.
        frame = _person_frame(
            {
                "employment_income": np.asarray([0.0, 52_000.0, 12_000.0]),
                "stock_assets": np.asarray([0.0, 1_500.0, 0.0]),
            }
        )
        result = us_release_input_coverage_gate(
            frame, _StubEngine(_DEFAULTS), manifest=_CONTRACT
        )
        assert result.passed
        assert result.failures == ()
        assert result.details["dormant_exclusions"] == ["alimony_income"]

    def test_missing_required_column_fails(self) -> None:
        # Case 2: stock_assets is absent from the export entirely — the silent
        # zero the #368 launch failure rode in on.
        frame = _person_frame({"employment_income": np.asarray([0.0, 52_000.0])})
        result = us_release_input_coverage_gate(
            frame, _StubEngine(_DEFAULTS), manifest=_CONTRACT
        )
        assert not result.passed
        assert "stock_assets" in result.details["missing"]
        assert any(
            "stock_assets" in failure and "absent" in failure
            for failure in result.failures
        )

    def test_degenerate_required_column_without_exclusion_fails(self) -> None:
        # Case 3: stock_assets is present but every value is the engine default,
        # so the export writer's default-broadcast makes it indistinguishable
        # from absence — and there is no reviewed exclusion to accept it.
        frame = _person_frame(
            {
                "employment_income": np.asarray([0.0, 52_000.0]),
                "stock_assets": np.asarray([0.0, 0.0]),
            }
        )
        result = us_release_input_coverage_gate(
            frame, _StubEngine(_DEFAULTS), manifest=_CONTRACT
        )
        assert not result.passed
        assert "stock_assets" in result.details["degenerate_required"]
        assert any(
            "stock_assets" in failure and "default" in failure
            for failure in result.failures
        )

    def test_all_nonfinite_required_asset_columns_fail(self) -> None:
        assets = tuple(sorted(SSI_COUNTABLE_RESOURCE_ASSETS))
        assert assets == ("bank_account_assets", "bond_assets", "stock_assets")
        manifest = _manifest(
            tuple(ReleaseInputColumn(name, "required") for name in assets)
        )
        frame = _person_frame(
            {name: np.asarray([np.nan, np.nan], dtype=np.float64) for name in assets}
        )

        result = us_release_input_coverage_gate(
            frame,
            _StubEngine({name: 0.0 for name in assets}),
            manifest=manifest,
        )

        assert not result.passed
        assert result.details["missing"] == []
        assert result.details["degenerate_required"] == list(assets)
        assert result.details["no_observed_required"] == list(assets)
        assert len(result.failures) == 3
        for name, failure in zip(assets, result.failures, strict=True):
            assert name in failure
            assert "no finite/non-null observed values" in failure

    @pytest.mark.parametrize(
        ("name", "values", "default"),
        [
            (
                "business_is_sstb",
                pd.Series([pd.NA, pd.NA], dtype="boolean"),
                False,
            ),
            (
                "ssn_card_type",
                pd.Series([pd.NA, pd.NA], dtype="string"),
                "CITIZEN",
            ),
            (
                "stock_assets",
                np.asarray([pd.NA, pd.NA], dtype=object),
                0.0,
            ),
            (
                "event_date",
                pd.Series([pd.NaT, pd.NaT], dtype="datetime64[ns]"),
                pd.Timestamp("2000-01-01"),
            ),
        ],
        ids=["nullable_boolean", "nullable_string", "object_pd_na", "datetime_nat"],
    )
    def test_required_all_pandas_missing_column_has_named_no_observed_finding(
        self,
        name: str,
        values: np.ndarray | pd.Series,
        default: object,
    ) -> None:
        frame = _person_frame({name: values})
        manifest = _manifest((ReleaseInputColumn(name, "required"),))

        result = us_release_input_coverage_gate(
            frame,
            _StubEngine({name: default}),
            manifest=manifest,
        )

        assert not result.passed
        assert result.details["missing"] == []
        assert result.details["degenerate_required"] == [name]
        assert result.details["no_observed_required"] == [name]
        assert len(result.failures) == 1
        assert result.failures[0].startswith(f"{name}: required eCPS input column")
        assert "no finite/non-null observed values" in result.failures[0]

    @pytest.mark.parametrize(
        ("name", "values", "default"),
        [
            (
                "business_is_sstb",
                pd.Series([pd.NA, True], dtype="boolean"),
                False,
            ),
            (
                "string_input",
                pd.Series([pd.NA, "x"], dtype="string"),
                "",
            ),
            (
                "stock_assets",
                np.asarray([np.nan, 125.0], dtype=np.float64),
                0.0,
            ),
        ],
        ids=["nullable_boolean", "nullable_string", "float_nan"],
    )
    def test_one_observed_nondefault_value_is_release_signal(
        self,
        name: str,
        values: np.ndarray | pd.Series,
        default: object,
    ) -> None:
        frame = _person_frame({name: values})
        manifest = _manifest((ReleaseInputColumn(name, "required"),))

        result = us_release_input_coverage_gate(
            frame,
            _StubEngine({name: default}),
            manifest=manifest,
        )

        assert result.passed
        assert result.failures == ()
        assert result.details["degenerate_required"] == []
        assert result.details["no_observed_required"] == []

    def test_real_us_nullable_boolean_is_no_observed_required(self) -> None:
        __import__("policyengine_us")
        from microcosm.frame.adapters.policyengine_us import PolicyEngineUSEngine

        name = "business_is_sstb"
        frame = _us_person_frame({name: pd.Series([pd.NA, pd.NA], dtype="boolean")})
        manifest = load_release_input_coverage_manifest()
        engine = PolicyEngineUSEngine()

        assert frame.table("person")[name].dtype == pd.BooleanDtype()
        assert name in manifest.required_columns
        assert engine.default_values([name]) == {name: False}

        result = us_release_input_coverage_gate(frame, engine, manifest=manifest)

        assert not result.passed
        assert name not in result.details["missing"]
        assert name in result.details["degenerate_required"]
        assert result.details["no_observed_required"] == [name]
        assert any(
            failure.startswith(f"{name}: required eCPS input column")
            and "no finite/non-null observed values" in failure
            for failure in result.failures
        )

    def test_stale_reviewed_exclusion_fails(self) -> None:
        # Case 4: alimony_income is a reviewed exclusion, but the data caught up
        # — it is now present with signal, so the exclusion is stale and must be
        # promoted to a hard requirement (#286 cannot-rot).
        frame = _person_frame(
            {
                "employment_income": np.asarray([0.0, 52_000.0]),
                "stock_assets": np.asarray([0.0, 1_500.0]),
                "alimony_income": np.asarray([0.0, 800.0]),
            }
        )
        result = us_release_input_coverage_gate(
            frame, _StubEngine(_DEFAULTS), manifest=_CONTRACT
        )
        assert not result.passed
        assert result.details["stale_exclusions"] == ["alimony_income"]
        assert any(
            "Stale reviewed exclusions" in failure for failure in result.failures
        )

    def test_absent_and_degenerate_excluded_column_passes(self) -> None:
        # The exclusion accepts both an absent column and a degenerate one: with
        # alimony_income present-but-all-default, the reviewed exclusion holds.
        frame = _person_frame(
            {
                "employment_income": np.asarray([0.0, 52_000.0]),
                "stock_assets": np.asarray([0.0, 1_500.0]),
                "alimony_income": np.asarray([0.0, 0.0]),
            }
        )
        result = us_release_input_coverage_gate(
            frame, _StubEngine(_DEFAULTS), manifest=_CONTRACT
        )
        assert result.passed
        assert result.details["reviewed_exclusions"] == {
            "alimony_income": "Residual income-source layer not yet sourced; tracked."
        }

    def test_typed_household_weights_count_as_persisted_input_signal(self) -> None:
        manifest = _manifest((ReleaseInputColumn("household_weight", "required"),))
        frame = _household_weight_frame(np.asarray([125.0, 275.0]))
        assert "household_weight" not in frame.table("household")

        result = us_release_input_coverage_gate(
            frame,
            _StubEngine({"household_weight": 1.0}),
            manifest=manifest,
        )

        assert result.passed
        assert result.failures == ()

    def test_typed_household_weights_override_stale_table_column(self) -> None:
        manifest = _manifest((ReleaseInputColumn("household_weight", "required"),))
        frame = _household_weight_frame(
            np.asarray([1.0, 1.0]),
            stored_values=np.asarray([125.0, 275.0]),
        )

        result = us_release_input_coverage_gate(
            frame,
            _StubEngine({"household_weight": 1.0}),
            manifest=manifest,
        )

        assert not result.passed
        assert result.details["degenerate_required"] == ["household_weight"]


class TestAsecReportedReceiptInputGuarantee:
    """PolicyEngine/microcosm#978 option 1: the three ASEC reported-receipt
    inputs are hard requirements of the national release coverage gate. The
    base build derives them and the ACS local-area transfer requires them in
    its donor, but nothing national required them, so the published default
    shipped without them and the local chain could only stage from an
    unpublished receipt-qualified child the publish contract refuses."""

    def test_constant_names_the_three_receipt_inputs(self) -> None:
        assert US_ASEC_REPORTED_RECEIPT_REQUIRED_INPUTS == (
            "receives_wic",
            "receives_snap",
            "receives_tanf",
        )

    def test_receipt_inputs_are_the_asec_carry_stage_outputs(self) -> None:
        # The requirement names exactly what derive_us_cps_carried_inputs
        # produces, on the entity it produces it: WIC on person, SNAP/TANF on
        # spm_unit. A rename on either side breaks this before it breaks a build.
        assert "receives_wic" in CPS_CARRIED_PERSON_INPUTS
        assert {"receives_snap", "receives_tanf"} <= CPS_CARRIED_SPM_UNIT_INPUTS
        assert (
            set(US_ASEC_REPORTED_RECEIPT_REQUIRED_INPUTS)
            <= CPS_CARRIED_PERSON_INPUTS | CPS_CARRIED_SPM_UNIT_INPUTS
        )

    def test_receipt_inputs_are_post_reference_hard_requirements(self) -> None:
        # Absent from the frozen reference eCPS surface, so they enter the
        # contract through the post-reference set; the shipped manifest must
        # carry them required, unexcused, and annotated with the owning issue.
        assert (
            set(US_ASEC_REPORTED_RECEIPT_REQUIRED_INPUTS)
            <= POST_REFERENCE_ECPS_REQUIRED_INPUTS
        )
        manifest = load_release_input_coverage_manifest()
        for column in US_ASEC_REPORTED_RECEIPT_REQUIRED_INPUTS:
            assert column in manifest.required_columns
            assert column not in manifest.reviewed_exclusions
            entry = next(entry for entry in manifest.columns if entry.name == column)
            assert "PolicyEngine/microcosm#978" in entry.note

    @pytest.mark.parametrize("column", US_ASEC_REPORTED_RECEIPT_REQUIRED_INPUTS)
    def test_receipt_input_cannot_regress_to_reviewed_exclusion(
        self, column: str
    ) -> None:
        manifest = load_release_input_coverage_manifest()
        demoted = ReleaseInputCoverageManifest(
            reference=manifest.reference,
            columns=tuple(
                ReleaseInputColumn(
                    name=entry.name,
                    status="reviewed_exclusion",
                    reason="regression attempt",
                    issue="PolicyEngine/microcosm#978",
                )
                if entry.name == column
                else entry
                for entry in manifest.columns
            ),
            probes=manifest.probes,
            schema_version=manifest.schema_version,
        )

        with pytest.raises(ValueError, match=column):
            assert_release_input_coverage_manifest_current(manifest=demoted)

    @pytest.mark.parametrize("column", US_ASEC_REPORTED_RECEIPT_REQUIRED_INPUTS)
    def test_receipt_input_cannot_be_dropped_from_the_manifest(
        self, column: str
    ) -> None:
        manifest = load_release_input_coverage_manifest()
        dropped = ReleaseInputCoverageManifest(
            reference=manifest.reference,
            columns=tuple(entry for entry in manifest.columns if entry.name != column),
            probes=manifest.probes,
            schema_version=manifest.schema_version,
        )

        with pytest.raises(ValueError, match=rf"{column}: ASEC reported-receipt"):
            assert_release_input_coverage_manifest_current(manifest=dropped)

    @pytest.mark.parametrize("column", US_ASEC_REPORTED_RECEIPT_REQUIRED_INPUTS)
    def test_candidate_missing_a_receipt_input_fails_the_shipped_gate(
        self, column: str
    ) -> None:
        # Against the SHIPPED manifest: the candidate that drops one receipt
        # column is failed with that column named as absent, while the other
        # two (present with signal) are not flagged.
        frame = _receipt_candidate(omit=column)
        result = us_release_input_coverage_gate(
            frame,
            _StubEngine(_RECEIPT_DEFAULTS),
            manifest=load_release_input_coverage_manifest(),
        )

        assert not result.passed
        assert column in result.details["missing"]
        others = set(US_ASEC_REPORTED_RECEIPT_REQUIRED_INPUTS) - {column}
        assert not others & set(result.details["missing"])
        assert not others & set(result.details["degenerate_required"])
        assert any(
            column in failure and "absent" in failure for failure in result.failures
        )

    @pytest.mark.parametrize("column", US_ASEC_REPORTED_RECEIPT_REQUIRED_INPUTS)
    def test_candidate_missing_a_receipt_input_fails_the_receipt_contract(
        self, column: str
    ) -> None:
        # Isolated to the three-column contract so the missing column is the
        # ONLY failure.
        frame = _receipt_candidate(omit=column)
        result = us_release_input_coverage_gate(
            frame, _StubEngine(_RECEIPT_DEFAULTS), manifest=_RECEIPT_CONTRACT
        )

        assert not result.passed
        assert result.details["missing"] == [column]
        assert result.details["degenerate_required"] == []
        assert len(result.failures) == 1

    def test_candidate_carrying_all_receipt_inputs_with_signal_passes(self) -> None:
        frame = _receipt_candidate()
        result = us_release_input_coverage_gate(
            frame, _StubEngine(_RECEIPT_DEFAULTS), manifest=_RECEIPT_CONTRACT
        )
        assert result.passed
        assert result.failures == ()

        # And under the shipped manifest none of the three is flagged (the
        # other required columns this bare candidate lacks are, by design).
        shipped = us_release_input_coverage_gate(
            frame,
            _StubEngine(_RECEIPT_DEFAULTS),
            manifest=load_release_input_coverage_manifest(),
        )
        flagged = set(shipped.details["missing"]) | set(
            shipped.details["degenerate_required"]
        )
        assert not flagged & set(US_ASEC_REPORTED_RECEIPT_REQUIRED_INPUTS)

    def test_candidate_carrying_default_only_receipt_inputs_fails(self) -> None:
        # Present but every value the engine default (False) is the export
        # writer's default-broadcast: indistinguishable from absent, and there
        # is no reviewed exclusion to accept it.
        frame = _receipt_candidate(all_default=True)
        result = us_release_input_coverage_gate(
            frame, _StubEngine(_RECEIPT_DEFAULTS), manifest=_RECEIPT_CONTRACT
        )

        assert not result.passed
        assert result.details["missing"] == []
        assert result.details["degenerate_required"] == sorted(
            US_ASEC_REPORTED_RECEIPT_REQUIRED_INPUTS
        )

    def test_real_engine_treats_receipt_inputs_as_false_default_input_leaves(
        self,
    ) -> None:
        __import__("policyengine_us")
        from microcosm.frame.adapters.policyengine_us import PolicyEngineUSEngine

        engine = PolicyEngineUSEngine()
        names = list(US_ASEC_REPORTED_RECEIPT_REQUIRED_INPUTS)
        assert engine.default_values(names) == {name: False for name in names}
        assert set(names) <= set(engine.variables())

        passing = us_release_input_coverage_gate(
            _receipt_candidate(), engine, manifest=_RECEIPT_CONTRACT
        )
        assert passing.passed

        failing = us_release_input_coverage_gate(
            _receipt_candidate(all_default=True), engine, manifest=_RECEIPT_CONTRACT
        )
        assert not failing.passed
        assert failing.details["degenerate_required"] == sorted(names)
