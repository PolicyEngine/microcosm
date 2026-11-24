"""Tests split from packages/microcosm-build/tests/test_us_weeks_unemployed.py."""

# ruff: noqa: F403, F405
from functools import cache

from hypothesis import given, settings
from hypothesis import strategies as st

from microcosm.build.gates import GateResult
from microcosm.build.us_runtime.one_population_contract import ACS_SUPPORT_CHANNEL
from microcosm.build.us_runtime.support_provenance import (
    support_channel_column,
    support_clone_index_column,
)
from test_support.microcosm_build.us_one_population_contract import (
    MixedOriginReaderPopulation,
    mixed_origin_reader_population,
)
from test_support.microcosm_build.us_weeks_unemployed import *

#: The weeks gate's one line for d2283's ACS channel until a reviewed rule
#: lands (lead call P3-L2).
_ACS_PLACEHOLDER = (
    "acs_2024_1yr weeks_unemployed has no reviewed rule yet (d2283 spike)."
)
_UC_FAILURE = (
    "weeks_unemployed has 1 row(s) positive outside the "
    "unemployment-compensation constraint."
)
#: The stored channels of CPS rows: physical ASEC rows and their clones.
_CPS_CHANNELS = ("asec", "puf_tax_detail")


@cache
def _mixed_origin() -> MixedOriginReaderPopulation:
    return mixed_origin_reader_population()


@cache
def _mixed_origin_gate() -> GateResult:
    return us_weeks_unemployed_signal_gate(_mixed_origin().mixed_origin)


def _acs_person_rows(frame: Frame) -> np.ndarray:
    channel = frame.table("person")[support_channel_column("person")]
    return channel.eq(ACS_SUPPORT_CHANNEL).to_numpy()


def _with_person_column(frame: Frame, column: str, values: np.ndarray) -> Frame:
    person = frame.table("person").copy()
    person[column] = values
    return module._replace_person_table(frame, person)


def test_public_stage_contract_is_exactly_manifest_pinned() -> None:
    spec = us_weeks_unemployed_stage_spec()

    assert spec.stage == "weeks_unemployed_input"
    assert [operation.kind for operation in spec.operations] == [
        "read_table",
        "derive_weeks_unemployed",
        "impute_weeks_unemployed_to_puf_support",
    ]
    assert dict(spec.operations[0].parameters) == WEEKS_UNEMPLOYED_READ_PARAMETERS
    assert dict(spec.operations[1].parameters) == WEEKS_UNEMPLOYED_DERIVE_PARAMETERS
    assert (
        dict(spec.operations[2].parameters)
        == WEEKS_UNEMPLOYED_PUF_IMPUTATION_PARAMETERS
    )
    artifact = next(
        item
        for item in spec.artifacts
        if item.get("member") == ASEC_2023_WEEKS_UNEMPLOYED_MEMBER
    )
    assert artifact["size_bytes"] == ASEC_2023_WEEKS_UNEMPLOYED_ZIP_SIZE_BYTES
    assert artifact["sha256"] == ASEC_2023_WEEKS_UNEMPLOYED_ZIP_SHA256
    assert artifact["member_size_bytes"] == ASEC_2023_WEEKS_UNEMPLOYED_MEMBER_SIZE_BYTES
    assert artifact["member_crc32"] == ASEC_2023_WEEKS_UNEMPLOYED_MEMBER_CRC32
    assert artifact["member_sha256"] == ASEC_2023_WEEKS_UNEMPLOYED_MEMBER_SHA256


def test_loader_verifies_zip_member_identity_and_attaches_audit(tmp_path: Path) -> None:
    path, pins = _mini_source(tmp_path)

    source = load_asec_2023_weeks_unemployed_source(
        path,
        **pins,
        expected_rows=3,
        expected_unique_keys=3,
    )

    assert tuple(source.columns) == ASEC_2023_WEEKS_UNEMPLOYED_SOURCE_COLUMNS
    assert source["PERIDNUM"].str.len().eq(22).all()
    assert source["LKWEEKS"].tolist() == [0, 4, -1]
    assert source.attrs["source_audit"] == {
        "raw_rows": 3,
        "unique_keys": 3,
        "positive_rows": 1,
        "niu_rows": 1,
        "minimum": -1,
        "maximum": 4,
        "weighted_source_share": pytest.approx(1 / 3),
        "weighted_weeks": pytest.approx(8.0),
        "pinned_transform": 0,
    }


def test_fetch_verifies_both_parent_and_member_and_reuses_cache(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    member = _person_csv()
    payload = _zip_bytes(member)
    pins = _pins(payload, member)
    calls = 0

    def _urlopen(_request: object, *, timeout: int) -> io.BytesIO:
        nonlocal calls
        assert timeout == 180
        calls += 1
        return io.BytesIO(payload)

    monkeypatch.setattr(module.urllib.request, "urlopen", _urlopen)
    result = fetch_asec_2023_weeks_unemployed_source(tmp_path, **pins, chunk_size=17)
    assert result.read_bytes() == member
    assert calls == 1

    result_again = fetch_asec_2023_weeks_unemployed_source(
        tmp_path, **pins, chunk_size=17
    )
    assert result_again == result
    assert calls == 1


@pytest.mark.parametrize("tamper", ["parent", "member"])
def test_loader_rejects_tampered_source(
    tmp_path: Path,
    tamper: str,
) -> None:
    path, pins = _mini_source(tmp_path)
    if tamper == "parent":
        path.write_bytes(path.read_bytes() + b"tamper")
    else:
        pins["expected_member_sha256"] = "0" * 64

    with pytest.raises(ValueError, match="SHA-256|byte length"):
        load_asec_2023_weeks_unemployed_source(
            path,
            **pins,
            expected_rows=3,
            expected_unique_keys=3,
        )


def test_loader_rejects_duplicate_or_non_fixed_width_person_keys(
    tmp_path: Path,
) -> None:
    raw = pd.read_csv(io.BytesIO(_person_csv()), dtype={"PERIDNUM": "string"})
    raw.loc[1, "PERIDNUM"] = raw.loc[0, "PERIDNUM"]
    member = raw.to_csv(index=False).encode()
    payload = _zip_bytes(member)
    path = tmp_path / "duplicate.zip"
    path.write_bytes(payload)

    with pytest.raises(ValueError, match="unique|duplicate"):
        load_asec_2023_weeks_unemployed_source(
            path,
            **_pins(payload, member),
            expected_rows=3,
            expected_unique_keys=3,
        )

    raw.loc[1, "PERIDNUM"] = "123"
    member = raw.to_csv(index=False).encode()
    payload = _zip_bytes(member)
    path.write_bytes(payload)
    with pytest.raises(ValueError, match="22-digit"):
        load_asec_2023_weeks_unemployed_source(
            path,
            **_pins(payload, member),
            expected_rows=3,
            expected_unique_keys=3,
        )


def test_fill_repairs_only_2022_and_accepts_cloned_duplicate_peridnum() -> None:
    source = pd.DataFrame(
        {
            "PH_SEQ": [101, 102],
            "P_SEQ": [1, 1],
            "A_LINENO": [1, 1],
            "PERIDNUM": [f"{value:022d}" for value in (1, 2)],
            "LKWEEKS": [7, 9],
        }
    )
    person = pd.DataFrame(
        {
            "source_year": [2022, 2023, 2022],
            "source_household_id": [101, 999, 101],
            "P_SEQ": [1, 1, 1],
            "A_LINENO": [1, 1, 1],
            "PERIDNUM": [f"{value:022d}" for value in (1, 3, 1)],
            "LKWEEKS": [np.nan, 11.0, np.nan],
            "person_support_channel": ["asec", "asec", "puf_tax_detail"],
        }
    )

    result = fill_asec_2022_weeks_unemployed_source(person, source)

    assert result["LKWEEKS"].tolist() == [7.0, 11.0, 7.0]
    assert person["LKWEEKS"].isna().sum() == 2


def test_fill_prefers_raw_source_household_id_over_transformed_ph_seq() -> None:
    source = pd.DataFrame(
        {
            "PH_SEQ": [32],
            "P_SEQ": [1],
            "A_LINENO": [1],
            "PERIDNUM": ["0000000000000000000001"],
            "LKWEEKS": [7],
        }
    )
    person = pd.DataFrame(
        {
            "source_year": [2022],
            # Pooled PH_SEQ is globally transformed; source_household_id is raw.
            "PH_SEQ": [112_033],
            "source_household_id": [32],
            "P_SEQ": [1],
            "A_LINENO": [1],
            "PERIDNUM": ["0000000000000000000001"],
            "LKWEEKS": [np.nan],
        }
    )

    result = fill_asec_2022_weeks_unemployed_source(person, source)

    assert result["LKWEEKS"].tolist() == [7.0]


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ({"PERIDNUM": "0000000000000000000099"}, "does not cover"),
        ({"source_household_id": 999}, "identity mismatch"),
        ({"LKWEEKS": 8.0}, "disagrees"),
    ],
)
def test_fill_fails_closed_on_key_identity_or_existing_value_mismatch(
    mutation: dict[str, object],
    message: str,
) -> None:
    source = pd.DataFrame(
        {
            "PH_SEQ": [101],
            "P_SEQ": [1],
            "A_LINENO": [1],
            "PERIDNUM": ["0000000000000000000001"],
            "LKWEEKS": [7],
        }
    )
    row: dict[str, object] = {
        "source_year": 2022,
        "source_household_id": 101,
        "P_SEQ": 1,
        "A_LINENO": 1,
        "PERIDNUM": "0000000000000000000001",
        "LKWEEKS": np.nan,
    }
    row.update(mutation)

    with pytest.raises(ValueError, match=message):
        fill_asec_2022_weeks_unemployed_source(pd.DataFrame([row]), source)


def test_direct_carry_maps_only_minus_one_and_rejects_invalid_values() -> None:
    operation = _operation("derive_weeks_unemployed")
    source = pd.DataFrame({"LKWEEKS": [-1, 0, 1, 51, 52]})

    result = derive_us_weeks_unemployed_from_manifest(source, operation, None)

    assert result[_OUTPUT].tolist() == [0, 0, 1, 51, 52]
    for bad in (np.nan, np.inf, -2, 53, 1.5):
        with pytest.raises(SourceRuntimeError, match="integer -1 or 0--52"):
            derive_us_weeks_unemployed_from_manifest(
                pd.DataFrame({"LKWEEKS": [bad]}), operation, None
            )


def test_puf_qrf_uses_seed_weights_round_clip_and_uc_zero_rule(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _CapturingQRF.instances.clear()
    _CapturingQRF.prediction = pd.DataFrame({_OUTPUT: [1.6, 80.4]})
    monkeypatch.setattr(module, "QRF", _CapturingQRF)
    context = SourceRuntimeContext(
        SourceRuntimeConfig(seed=918, target_year=2024), tables={}
    )

    result = impute_us_weeks_unemployed_to_puf_support_from_manifest(
        _imputation_table(),
        _operation("impute_weeks_unemployed_to_puf_support"),
        context,
    )

    assert result[_OUTPUT].tolist() == [0.0, 6.0, 2.0, 0.0]
    fitted = _CapturingQRF.instances[-1]
    assert (fitted.n_estimators, fitted.seed) == (100, 918)
    assert fitted.predictors == [*_REQUIRED_PREDICTORS, "unemployment_compensation"]
    assert fitted.weights.tolist() == [2.0, 3.0]


def test_puf_qrf_omits_uc_predictor_and_rule_when_source_is_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _CapturingQRF.instances.clear()
    _CapturingQRF.prediction = pd.DataFrame({_OUTPUT: [-2.4, 99.0]})
    monkeypatch.setattr(module, "QRF", _CapturingQRF)

    result = impute_us_weeks_unemployed_to_puf_support_from_manifest(
        _imputation_table(include_uc=False),
        _operation("impute_weeks_unemployed_to_puf_support"),
        SourceRuntimeContext(SourceRuntimeConfig(seed=1), tables={}),
    )

    assert result[_OUTPUT].tolist() == [0.0, 6.0, 0.0, 52.0]
    assert _CapturingQRF.instances[-1].predictors == list(_REQUIRED_PREDICTORS)


@pytest.mark.parametrize(
    ("prediction", "message"),
    [
        (pd.DataFrame({_OUTPUT: [np.nan, 2.0]}), "nonfinite"),
        (pd.DataFrame({"wrong": [1.0, 2.0]}), "missing"),
        (pd.DataFrame({_OUTPUT: [1.0]}), "expected 2"),
    ],
)
def test_puf_qrf_rejects_adversarial_predictions(
    monkeypatch: pytest.MonkeyPatch,
    prediction: pd.DataFrame,
    message: str,
) -> None:
    _CapturingQRF.prediction = prediction
    monkeypatch.setattr(module, "QRF", _CapturingQRF)

    with pytest.raises(SourceRuntimeError, match=message):
        impute_us_weeks_unemployed_to_puf_support_from_manifest(
            _imputation_table(),
            _operation("impute_weeks_unemployed_to_puf_support"),
            SourceRuntimeContext(SourceRuntimeConfig(seed=1), tables={}),
        )


def test_puf_qrf_caps_training_at_5000_with_build_seed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    row_count = 5_003
    frame = pd.DataFrame(
        {
            "person_support_channel": ["asec"] * row_count + ["puf_tax_detail"] * 2,
            "person_weight": np.arange(1, row_count + 3, dtype=np.float64),
            _OUTPUT: np.resize([0.0, 4.0], row_count + 2),
        }
    )
    for index, predictor in enumerate(_REQUIRED_PREDICTORS):
        frame[_PREFIX + predictor] = np.arange(row_count + 2) + index
    _CapturingQRF.instances.clear()
    _CapturingQRF.prediction = pd.DataFrame({_OUTPUT: [1.0, 2.0]})
    monkeypatch.setattr(module, "QRF", _CapturingQRF)

    impute_us_weeks_unemployed_to_puf_support_from_manifest(
        frame,
        _operation("impute_weeks_unemployed_to_puf_support"),
        SourceRuntimeContext(SourceRuntimeConfig(seed=42), tables={}),
    )

    fitted = _CapturingQRF.instances[-1]
    assert fitted.training is not None
    assert len(fitted.training) == 5_000
    assert fitted.weights is not None
    assert len(fitted.weights) == 5_000


def test_wrapper_runs_before_and_after_support_cloning(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(module, "us_weeks_unemployed_stage_spec", _stage_spec)
    direct = with_us_weeks_unemployed(_frame(channels=False), seed=7, time_period=2024)
    assert direct.table("person")[_OUTPUT].tolist() == [2, 4]

    _CapturingQRF.prediction = pd.DataFrame({_OUTPUT: [3.2, 9.8]})
    monkeypatch.setattr(module, "QRF", _CapturingQRF)
    supported = with_us_weeks_unemployed(_frame(), seed=7, time_period=2024)
    assert supported.table("person")[_OUTPUT].tolist() == [2.0, 4.0, 3.0, 0.0]


def test_signal_gate_requires_exact_asec_and_nondefault_integer_both_channels() -> None:
    frame = _gate_frame()
    assert us_weeks_unemployed_signal_gate(frame).passed

    person = frame.table("person").copy()
    person.loc[0, _OUTPUT] = 3.0
    bad = module._replace_person_table(frame, person)
    gate = us_weeks_unemployed_signal_gate(bad)
    assert not gate.passed
    assert any("reconciliation" in failure for failure in gate.failures)

    person = frame.table("person").copy()
    person.loc[1_000, _OUTPUT] = 1.5
    bad = module._replace_person_table(frame, person)
    gate = us_weeks_unemployed_signal_gate(bad)
    assert not gate.passed
    assert any("noninteger" in failure for failure in gate.failures)

    person = frame.table("person").copy()
    person.loc[person["person_support_channel"].eq("puf_tax_detail"), _OUTPUT] = 0.0
    bad = module._replace_person_table(frame, person)
    gate = us_weeks_unemployed_signal_gate(bad)
    assert not gate.passed
    assert any("puf_tax_detail" in failure for failure in gate.failures)


def test_signal_gate_derives_legacy_asec_puf_roster_and_constraint_scope() -> None:
    frame = _gate_frame()
    summary = module.us_weeks_unemployed_summary(frame)

    assert summary["channel_roster"] == ["asec", "puf_tax_detail"]
    assert summary["source_rows"] == 1_000
    assert summary["source_reconciliation_rows"] == 1_000
    assert summary["uc_constraint_rows"] == 2_000
    assert summary["uc_constraint_mismatch_count"] == 0

    person = frame.table("person").copy()
    first_puf_carrier = person.index[
        person["person_support_channel"].eq("puf_tax_detail") & person[_OUTPUT].gt(0.0)
    ][0]
    person.loc[first_puf_carrier, "unemployment_compensation"] = 0.0
    gate = us_weeks_unemployed_signal_gate(module._replace_person_table(frame, person))

    assert not gate.passed
    assert any("unemployment-compensation constraint" in item for item in gate.failures)


def test_signal_gate_derives_stacked_asec_acs_roster_and_reviewed_scopes() -> None:
    frame = _stacked_gate_frame()
    gate = us_weeks_unemployed_signal_gate(frame)
    summary = gate.details

    assert gate.passed
    assert summary["channel_roster"] == ["asec", "acs"]
    assert set(summary["channels"]) == {"asec", "acs"}
    assert summary["source_rows"] == 1_000
    assert summary["source_reconciliation_rows"] == 500
    assert summary["source_invalid"] == 0
    assert summary["source_mismatch_count"] == 0
    assert summary["uc_constraint_rows"] == 1_500
    assert summary["uc_constraint_mismatch_count"] == 0

    # The UC rule takes the 500 ASEC clones and the 1,000 ACS natives; the
    # 500 ASEC natives and the 1,000 ACS natives would give the same count.
    person = frame.table("person")
    asec = person["person_support_channel"].eq("asec").to_numpy()
    native = person["person_support_clone_index"].eq(0).to_numpy()
    uc_constraint_rows = module._weeks_unemployed_gate_scopes(person)[4]
    np.testing.assert_array_equal(
        uc_constraint_rows, (asec & ~native) | (~asec & native)
    )
    # The stacked pool's acs keeps the PUF bands; only d2283's acs_2024_1yr
    # loses them.
    assert summary["channels"]["acs"]["positive_share_band"] == list(
        module._CHANNEL_POSITIVE_SHARE_BANDS["puf_tax_detail"]
    )
    assert summary["channels"]["acs"]["weighted_mean_weeks_band"] == list(
        module._CHANNEL_WEIGHTED_MEAN_WEEKS_BANDS["puf_tax_detail"]
    )


def test_signal_gate_stacked_source_and_uc_checks_ignore_unowned_rows() -> None:
    frame = _stacked_gate_frame()
    person = frame.table("person").copy()
    acs_clone = person["person_support_channel"].eq("acs")
    person.loc[acs_clone, "LKWEEKS"] = 99.0
    ignored = module._replace_person_table(frame, person)
    assert us_weeks_unemployed_signal_gate(ignored).passed

    asec_clone_one = person["person_support_channel"].eq("asec") & person[
        "person_support_clone_index"
    ].eq(1)
    person.loc[person.index[asec_clone_one][0], "LKWEEKS"] = 99.0
    invalid_source = us_weeks_unemployed_signal_gate(
        module._replace_person_table(frame, person)
    )
    assert not invalid_source.passed
    assert invalid_source.details["source_invalid"] == 1

    person = frame.table("person").copy()
    acs_native = person["person_support_channel"].eq("acs") & person[
        "person_support_clone_index"
    ].eq(0)
    person.loc[person.index[acs_native][0], _OUTPUT] = 1.0
    uc_mismatch = us_weeks_unemployed_signal_gate(
        module._replace_person_table(frame, person)
    )
    assert not uc_mismatch.passed
    assert uc_mismatch.details["uc_constraint_mismatch_count"] == 1


@st.composite
def _assembled_people(
    draw: st.DrawFn, *, channels: tuple[str, ...], with_asec: bool
) -> pd.DataFrame:
    """An assembled person table: spine and source ids, a stored channel and
    a clone index on every row, and nothing else."""

    rows = draw(st.integers(min_value=1, max_value=60))
    channel = draw(st.lists(st.sampled_from(channels), min_size=rows, max_size=rows))
    if with_asec:
        channel[draw(st.integers(min_value=0, max_value=rows - 1))] = "asec"
    ids = st.lists(
        st.integers(min_value=0, max_value=2**40), min_size=rows, max_size=rows
    )
    clone = st.lists(st.sampled_from((0, 1, 2)), min_size=rows, max_size=rows)
    return pd.DataFrame(
        {
            "person_spine_source_id": np.asarray(draw(ids), dtype=np.int64),
            "person_source_id": np.asarray(draw(ids), dtype=np.int64),
            "person_support_channel": pd.Series(channel, dtype=object),
            "person_support_clone_index": np.asarray(draw(clone), dtype=np.int64),
        }
    )


@settings(max_examples=200, deadline=None)
@given(
    person=_assembled_people(
        channels=("asec", "puf_tax_detail", ACS_SUPPORT_CHANNEL, "acs"),
        with_asec=True,
    )
)
def test_no_acs_row_enters_lkweeks_reconciliation(person: pd.DataFrame) -> None:
    """On any assembled person table (P1, d2283 DESIGN :324), the scope
    takes only stored ``asec`` rows as source rows and reconciles only the
    native ones to ``LKWEEKS``, so no ACS row (``acs_2024_1yr`` or the
    stacked pool's ``acs``) enters either check. The UC rule takes every CPS
    clone, with a stored channel of ``asec`` or ``puf_tax_detail``, and
    every native of another channel.
    """

    channel = person["person_support_channel"].to_numpy(dtype=object)
    native = person["person_support_clone_index"].to_numpy() == 0
    asec = channel == "asec"
    cps = np.isin(channel, _CPS_CHANNELS)

    (
        source_channel,
        roster,
        source_rows,
        source_reconciliation_rows,
        uc_constraint_rows,
    ) = module._weeks_unemployed_gate_scopes(person)

    np.testing.assert_array_equal(source_channel, channel)
    assert roster == ("asec", *sorted(set(channel) - {"asec"}))
    np.testing.assert_array_equal(source_rows, asec)
    np.testing.assert_array_equal(source_reconciliation_rows, asec & native)
    assert not ((source_rows | source_reconciliation_rows) & ~asec).any()
    np.testing.assert_array_equal(uc_constraint_rows, (cps & ~native) | (~cps & native))


@settings(max_examples=100, deadline=None)
@given(
    person=_assembled_people(
        channels=("puf_tax_detail", ACS_SUPPORT_CHANNEL, "acs"), with_asec=False
    )
)
def test_an_assembled_table_without_asec_rows_refuses(person: pd.DataFrame) -> None:
    """An assembled table with no stored ``asec`` row has no source rows,
    and the scope refuses it (P1's second property)."""

    with pytest.raises(ValueError, match="has no ASEC source rows"):
        module._weeks_unemployed_gate_scopes(person)


#: Values an ACS row's ``LKWEEKS`` cell might hold: null, the
#: not-in-universe code, a valid week count, out of range, fractional, text.
_ACS_LKWEEKS_VALUES = st.one_of(
    st.sampled_from((np.nan, -1.0, 99.0, 3.5, "not a week count")),
    st.integers(min_value=0, max_value=52).map(float),
)


@settings(max_examples=50, deadline=None)
@given(data=st.data())
def test_acs_lkweeks_cells_never_move_the_source_checks(data: st.DataObject) -> None:
    """On the mixed-origin frame (P2), whatever the ACS rows' ``LKWEEKS``
    cells hold, the gate's source rows, reconciliation rows, invalid count,
    mismatch count, every other statistic and its failures keep their
    values. The source checks read only the rows the scope returns."""

    mixed = _mixed_origin().mixed_origin
    baseline = _mixed_origin_gate()
    acs = np.flatnonzero(_acs_person_rows(mixed))
    assert len(acs) > 0
    assert mixed.table("person")["LKWEEKS"].iloc[acs].isna().all()
    assert baseline.details["source_rows"] > 0
    assert baseline.details["source_reconciliation_rows"] > 0

    rows = data.draw(st.lists(st.sampled_from(acs.tolist()), min_size=1, unique=True))
    lkweeks = mixed.table("person")["LKWEEKS"].to_numpy(dtype=object, copy=True)
    for row in rows:
        lkweeks[row] = data.draw(_ACS_LKWEEKS_VALUES)
    gate = us_weeks_unemployed_signal_gate(
        _with_person_column(mixed, "LKWEEKS", lkweeks)
    )

    for key in (
        "source_rows",
        "source_reconciliation_rows",
        "source_invalid",
        "source_mismatch_count",
    ):
        assert gate.details[key] == baseline.details[key], key
    assert gate.details == baseline.details
    assert (gate.passed, gate.failures) == (baseline.passed, baseline.failures)


def test_acs_native_weeks_channel_fails_closed_until_reviewed() -> None:
    """The gate fails d2283's ACS channel with one line until a reviewed
    rule lands (lead call P3-L2), and keeps the channel's statistics in its
    details.

    Its two bands read None; the ``asec`` and ``puf_tax_detail`` entries keep
    their keys, in order, and their bands. The line stands in for the
    default-only and band checks: ACS weeks far outside the PUF bands still
    give that one line.
    """

    mixed = _mixed_origin().mixed_origin
    acs = _acs_person_rows(mixed)
    gate = _mixed_origin_gate()

    assert not gate.passed
    assert gate.failures == (_ACS_PLACEHOLDER,)
    channels = gate.details["channels"]
    assert gate.details["channel_roster"] == [
        "asec",
        ACS_SUPPORT_CHANNEL,
        "puf_tax_detail",
    ]
    acs_channel = channels[ACS_SUPPORT_CHANNEL]
    assert acs_channel["rows"] == int(acs.sum())
    assert acs_channel["positive_rows"] == 0
    assert acs_channel["positive_share_band"] is None
    assert acs_channel["weighted_mean_weeks_band"] is None
    for name in _CPS_CHANNELS:
        assert list(channels[name]) == list(acs_channel), name
        assert channels[name]["positive_share_band"] == list(
            module._CHANNEL_POSITIVE_SHARE_BANDS[name]
        ), name
        assert channels[name]["weighted_mean_weeks_band"] == list(
            module._CHANNEL_WEIGHTED_MEAN_WEEKS_BANDS[name]
        ), name

    person = mixed.table("person")
    compensated = acs & (person["unemployment_compensation"].to_numpy() > 0.0)
    weeks = person["weeks_unemployed"].to_numpy(dtype=np.float64, copy=True)
    weeks[compensated] = 52.0
    busy = us_weeks_unemployed_signal_gate(
        _with_person_column(mixed, "weeks_unemployed", weeks)
    )

    assert busy.failures == (_ACS_PLACEHOLDER,)
    busy_acs = busy.details["channels"][ACS_SUPPORT_CHANNEL]
    assert busy_acs["positive_rows"] == int(compensated.sum()) > 0
    share_high = module._CHANNEL_POSITIVE_SHARE_BANDS["puf_tax_detail"][1]
    mean_high = module._CHANNEL_WEIGHTED_MEAN_WEEKS_BANDS["puf_tax_detail"][1]
    assert busy_acs["positive_share"] > share_high
    assert busy_acs["weighted_mean_weeks"] > mean_high


def test_acs_native_rows_keep_the_uc_rule() -> None:
    """An ACS native with weeks unemployed and no unemployment compensation
    breaks the UC rule, as a native of any non-CPS channel does: the gate
    counts one mismatch and reports it beside the ACS channel's line."""

    mixed = _mixed_origin().mixed_origin
    baseline = _mixed_origin_gate()
    person = mixed.table("person")
    clone_index = person[support_clone_index_column("person")].to_numpy()
    uncompensated = _acs_person_rows(mixed) & (
        person["unemployment_compensation"].to_numpy() == 0.0
    )
    row = int(np.flatnonzero(uncompensated)[0])
    assert clone_index[row] == 0
    assert baseline.details["uc_constraint_mismatch_count"] == 0

    weeks = person["weeks_unemployed"].to_numpy(dtype=np.float64, copy=True)
    weeks[row] = 3.0
    gate = us_weeks_unemployed_signal_gate(
        _with_person_column(mixed, "weeks_unemployed", weeks)
    )

    assert gate.details["uc_constraint_mismatch_count"] == 1
    assert gate.details["uc_constraint_rows"] == baseline.details["uc_constraint_rows"]
    assert gate.failures == (_UC_FAILURE, _ACS_PLACEHOLDER)


def test_cps_clones_are_in_the_uc_rule_on_every_form() -> None:
    """The UC rule takes every CPS clone on Route A's frame, on its ASEC-only
    assembled form (where clones keep the ``puf_tax_detail`` label) and on the
    mixed-origin frame, and on the last also every ACS row, all native."""

    population = _mixed_origin()
    for form in ("route_a", "assembled", "mixed_origin"):
        frame = getattr(population, form)
        person = frame.table("person")
        clone = person[support_clone_index_column("person")].to_numpy() > 0
        acs = _acs_person_rows(frame)
        assert clone.any(), form
        assert acs.any() == (form == "mixed_origin"), form

        uc_constraint_rows = module._weeks_unemployed_gate_scopes(person)[4]
        summary = module.us_weeks_unemployed_summary(frame)

        np.testing.assert_array_equal(uc_constraint_rows, clone | acs)
        assert not (clone & acs).any(), form
        assert summary["uc_constraint_rows"] == int(clone.sum() + acs.sum()), form


def test_weeks_acs_channel_label_matches_the_contract() -> None:
    """The weeks gate names d2283's ACS channel without importing the
    contract; the two labels must agree. The contract's tests pin
    ``acs_pums``'s spine tag to the same label."""

    assert module._ACS_NATIVE_CHANNEL == ACS_SUPPORT_CHANNEL


def test_signal_gate_rejects_collapsed_puf_share_and_weighted_weeks() -> None:
    frame = _gate_frame()
    person = frame.table("person").copy()
    puf = person["person_support_channel"].eq("puf_tax_detail")
    person.loc[puf, _OUTPUT] = 0.0
    first_puf = person.index[puf][0]
    person.loc[first_puf, _OUTPUT] = 16.0
    collapsed_share = module._replace_person_table(frame, person)

    gate = us_weeks_unemployed_signal_gate(collapsed_share)

    assert not gate.passed
    assert any("positive share" in failure for failure in gate.failures)

    person.loc[puf, _OUTPUT] = 0.0
    first_four = person.index[puf][:4]
    person.loc[first_four, _OUTPUT] = 1.0
    person.loc[first_four, "unemployment_compensation"] = 100.0
    collapsed_weeks = module._replace_person_table(frame, person)

    gate = us_weeks_unemployed_signal_gate(collapsed_weeks)

    assert not gate.passed
    assert not any("positive share" in failure for failure in gate.failures)
    assert any("weighted mean weeks" in failure for failure in gate.failures)


def test_pinned_weighted_weeks_tolerates_only_binary_float_residue() -> None:
    audit = {
        "raw_rows": ASEC_2023_WEEKS_UNEMPLOYED_RAW_ROWS,
        "unique_keys": ASEC_2023_WEEKS_UNEMPLOYED_UNIQUE_KEYS,
        "positive_rows": ASEC_2023_WEEKS_UNEMPLOYED_POSITIVE_ROWS,
        "weighted_source_share": ASEC_2023_WEEKS_UNEMPLOYED_WEIGHTED_SOURCE_SHARE,
        "weighted_weeks": ASEC_2023_WEEKS_UNEMPLOYED_WEIGHTED_WEEKS + 2e-8,
    }
    module._assert_pinned_source_audit(audit)
    audit["weighted_weeks"] = ASEC_2023_WEEKS_UNEMPLOYED_WEIGHTED_WEEKS + 2e-6
    with pytest.raises(ValueError, match="weighted_weeks"):
        module._assert_pinned_source_audit(audit)


def test_optional_full_pinned_source_audit() -> None:
    candidates = [
        Path(value).expanduser()
        for value in (
            os.environ.get("POPULACE_ASEC_2023_WEEKS_SOURCE", ""),
            str(
                Path.home()
                / ".cache"
                / "microcosm"
                / "cps"
                / "asec_2023"
                / "asecpub23csv.zip"
            ),
            "/private/tmp/asecpub23csv.zip",
        )
        if value
    ]
    path = next((candidate for candidate in candidates if candidate.is_file()), None)
    if path is None:
        pytest.skip("pinned ASEC 2023 archive is not mounted")

    source = load_asec_2023_weeks_unemployed_source(path)
    audit = source.attrs["source_audit"]
    assert audit["pinned_transform"]
    assert audit["raw_rows"] == ASEC_2023_WEEKS_UNEMPLOYED_RAW_ROWS
    assert audit["unique_keys"] == ASEC_2023_WEEKS_UNEMPLOYED_UNIQUE_KEYS
    assert audit["positive_rows"] == ASEC_2023_WEEKS_UNEMPLOYED_POSITIVE_ROWS
    assert audit["weighted_source_share"] == pytest.approx(
        ASEC_2023_WEEKS_UNEMPLOYED_WEIGHTED_SOURCE_SHARE, abs=1e-12
    )
    assert audit["weighted_weeks"] == pytest.approx(
        ASEC_2023_WEEKS_UNEMPLOYED_WEIGHTED_WEEKS, abs=1e-6
    )
