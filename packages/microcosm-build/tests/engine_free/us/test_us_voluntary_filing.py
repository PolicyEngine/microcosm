"""Tests split from packages/microcosm-build/tests/test_us_voluntary_filing.py."""

# ruff: noqa: F403, F405
from test_support.microcosm_build.us_voluntary_filing import *


def test_archived_and_pinned_source_coordinates_are_exact() -> None:
    assert US_VOLUNTARY_FILING_STAGE_NAME == "voluntary_filing_input"
    assert SIPP_2023_VOLUNTARY_FILING_DONOR_REVISION == (
        "21280dca5995e978d706740a8a4b9b7860cfd7b6"
    )
    assert SIPP_2023_VOLUNTARY_FILING_DONOR_SHA256 == (
        "5c30439e365fc26483318ef61d1d8f4bb2f0e9d6bb47c22c06756a7698733ee2"
    )
    assert SIPP_2023_VOLUNTARY_FILING_DONOR_SIZE_BYTES == 3_726_010_471
    assert SIPP_2023_VOLUNTARY_FILING_DONOR_REVISION in (
        SIPP_2023_VOLUNTARY_FILING_DONOR_URL
    )
    assert VOLUNTARY_FILING_ARCHIVED_DERIVATION_URL.endswith(
        "datasets/cps/cps.py#L726-L747"
    )
    assert VOLUNTARY_FILING_ARCHIVED_PARAMETERS_URL.endswith(
        "parameters/take_up/voluntary_filing.yaml#L1-L43"
    )
    assert VOLUNTARY_FILING_SIPP_DICTIONARY_URL.endswith(
        "2023_SIPP_Data_Dictionary.pdf"
    )

def test_exact_source_columns_predictors_outputs_and_manifest_stage() -> None:
    assert SIPP_VOLUNTARY_FILING_SOURCE_COLUMNS == (
        "SSUID",
        "PNUM",
        "MONTHCODE",
        "WPFINWGT",
        "TAGE",
        "ESEX",
        "EPNSPOUSE",
        "AFILING",
        "EFILING",
        "AWILLFILE",
        "EWILLFILE",
        "EDEPCLM",
        "TJB1_MSUM",
        "TJB2_MSUM",
        "TJB3_MSUM",
        "TJB4_MSUM",
        "TJB5_MSUM",
        "TJB6_MSUM",
        "TJB7_MSUM",
    )
    assert SIPP_VOLUNTARY_FILING_MODEL_PREDICTORS == (
        "employment_income",
        "reference_age",
        "reference_is_female",
        "reference_is_married",
        "count_under_18",
    )
    assert US_VOLUNTARY_FILING_OUTPUT_COLUMNS == (_OUTPUT,)
    assert US_VOLUNTARY_FILING_NONCONSTANT_TAX_UNIT_COLUMNS == (_OUTPUT,)
    spec = us_voluntary_filing_stage_spec()
    assert spec.stage == "voluntary_filing_input"
    assert spec.grain == "tax_unit"
    assert spec.outputs == (_OUTPUT,)
    assert [operation.kind for operation in spec.operations] == [
        "read_table",
        "fit_weighted_qrf",
    ]

def test_loader_uses_reported_answers_drops_dependents_and_pairs_spouses(
    tmp_path: Path,
) -> None:
    path = _write_source(tmp_path, _synthetic_source_rows())

    donor = load_sipp_2023_voluntary_filing_donor(
        path, expected_size_bytes=None
    ).set_index("source_tax_unit_key")

    assert len(donor) == 2
    married_key = next(key for key in donor.index if str(key).startswith("1:"))
    singleton_key = next(key for key in donor.index if str(key).startswith("2:"))
    assert bool(donor.loc[married_key, _OUTPUT])
    assert not bool(donor.loc[singleton_key, _OUTPUT])
    assert donor.loc[married_key, "employment_income"] == pytest.approx(18_000.0)
    assert donor.loc[married_key, "reference_age"] == pytest.approx(40.0)
    assert donor.loc[married_key, "reference_is_female"] == pytest.approx(0.0)
    assert donor.loc[married_key, "reference_is_married"] == pytest.approx(1.0)
    assert donor.loc[married_key, "count_under_18"] == pytest.approx(1.0)
    assert donor.loc[married_key, "tax_unit_weight"] == pytest.approx(10.0)
    # Total income is deliberately absent: only TJB*_MSUM feeds wages.
    assert "TPTOTINC" not in SIPP_VOLUNTARY_FILING_SOURCE_COLUMNS
    # The November extreme and the dependent/imputed/zero-weight rows vanished.
    assert donor["employment_income"].max() < 100_000.0

def test_loader_rejects_reciprocal_spouse_target_disagreement(tmp_path: Path) -> None:
    rows = [
        _source_row(1, 101, spouse=102, filing=1),
        _source_row(
            1,
            102,
            spouse=101,
            filing=2,
            will_file=2,
            will_file_status=1,
        ),
    ]
    path = _write_source(tmp_path, rows)
    with pytest.raises(ValueError, match="spouses disagree"):
        load_sipp_2023_voluntary_filing_donor(path, expected_size_bytes=None)

def test_loader_reference_is_minimum_pnum_before_response_filter(
    tmp_path: Path,
) -> None:
    rows = [
        _source_row(
            1,
            101,
            spouse=102,
            age=63,
            sex=2,
            weight=31,
            filing=np.nan,
            filing_status=0,
            monthly_wages=100,
        ),
        _source_row(
            1,
            102,
            spouse=101,
            age=41,
            sex=1,
            weight=97,
            filing=1,
            monthly_wages=200,
        ),
        _source_row(
            2,
            101,
            filing=2,
            will_file=2,
            will_file_status=1,
        ),
    ]
    donor = load_sipp_2023_voluntary_filing_donor(
        _write_source(tmp_path, rows), expected_size_bytes=None
    ).set_index("source_tax_unit_key")
    married = donor.loc[next(key for key in donor.index if str(key).startswith("1:"))]

    assert married["reference_age"] == pytest.approx(63)
    assert married["reference_is_female"] == pytest.approx(1)
    assert married["tax_unit_weight"] == pytest.approx(31)
    assert married["employment_income"] == pytest.approx((100 + 200) * 12)

def test_loader_rejects_missing_columns_bad_hash_and_constant_target(
    tmp_path: Path,
) -> None:
    path = _write_source(tmp_path, _synthetic_source_rows())
    with pytest.raises(ValueError, match="sha-256 verification"):
        load_sipp_2023_voluntary_filing_donor(
            path,
            expected_sha256="0" * 64,
            expected_size_bytes=None,
        )

    missing = tmp_path / "missing.csv"
    pd.DataFrame({"SSUID": [1], "MONTHCODE": [12]}).to_csv(
        missing, sep="|", index=False
    )
    with pytest.raises(ValueError, match="missing column"):
        load_sipp_2023_voluntary_filing_donor(missing, expected_size_bytes=None)

    constant = tmp_path / "constant.csv"
    _write_source(
        tmp_path,
        [_source_row(10, 101), _source_row(20, 101)],
    ).replace(constant)
    with pytest.raises(ValueError, match="target is constant"):
        load_sipp_2023_voluntary_filing_donor(constant, expected_size_bytes=None)

def test_cached_full_donor_matches_locked_response_and_weight_facts() -> None:
    snapshot = (
        Path.home()
        / ".cache"
        / "huggingface"
        / "hub"
        / ("models--policyengine--policyengine-" + "us-data")
        / "snapshots"
        / SIPP_2023_VOLUNTARY_FILING_DONOR_REVISION
        / "pu2023.csv"
    )
    if not snapshot.is_file():
        pytest.skip("the 3.73 GB pinned SIPP donor is not mounted")

    donor = load_sipp_2023_voluntary_filing_donor(
        snapshot,
        expected_sha256=SIPP_2023_VOLUNTARY_FILING_DONOR_SHA256,
        expected_size_bytes=SIPP_2023_VOLUNTARY_FILING_DONOR_SIZE_BYTES,
    )
    weights = donor["tax_unit_weight"].to_numpy(dtype=np.float64)
    target = donor[_OUTPUT].to_numpy(dtype=bool)
    audit = donor.attrs["source_audit"]
    assert audit["december_rows"] == 39_513
    assert audit["observed_response_rows"] == 30_510
    assert audit["observed_response_true_rows"] == 24_473
    assert audit["claimed_dependent_observed_rows"] == 526
    assert audit["claimed_dependent_observed_true_rows"] == 526
    assert audit["spouse_target_disagreement_units"] == 0
    assert audit["canonical_preweight_units"] == 22_313
    assert audit["canonical_preweight_true_units"] == 16_820
    assert audit["positive_finite_weight_units"] == 22_296
    assert audit["positive_finite_weight_true_units"] == 16_817
    assert len(donor) == 22_296
    assert int(target.sum()) == 16_817
    assert float(weights.sum()) == pytest.approx(178_696_583.4878655, abs=1e-4)
    assert float(weights[target].sum() / weights.sum()) == pytest.approx(
        0.760308456312741, abs=1e-12
    )
    assert audit["positive_finite_weight_sum"] == pytest.approx(
        178_696_583.4878655, abs=1e-4
    )
    assert audit["weighted_true_share"] == pytest.approx(0.760308456312741, abs=1e-12)

def test_fetch_streams_verifies_atomically_and_reuses_cache(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    payload = b"small synthetic pinned filing donor"
    digest = hashlib.sha256(payload).hexdigest()
    response = _ChunkedResponse(payload)
    monkeypatch.setattr(urllib.request, "urlopen", lambda *_args: response)

    path = fetch_sipp_2023_voluntary_filing_donor(
        tmp_path,
        expected_sha256=digest,
        expected_size_bytes=len(payload),
        chunk_size=4,
    )
    assert path.read_bytes() == payload
    assert len(response.read_sizes) > 2
    assert all(size == 4 for size in response.read_sizes)
    assert not (tmp_path / "pu2023.csv.part").exists()

    monkeypatch.setattr(
        urllib.request,
        "urlopen",
        lambda *_args: (_ for _ in ()).throw(
            AssertionError("valid cache must be reused")
        ),
    )
    assert (
        fetch_sipp_2023_voluntary_filing_donor(
            tmp_path,
            expected_sha256=digest,
            expected_size_bytes=len(payload),
            chunk_size=4,
        )
        == path
    )

def test_receiver_uses_unit_wages_head_spouse_and_full_household_children() -> None:
    frame = _frame(6)
    receiver = module._recipient_tax_unit_predictor_table(frame)

    # Household/tax unit 6 has a head, spouse, and child.
    row = receiver.loc[106]
    assert row["employment_income"] == pytest.approx(20_000.0)
    assert row["reference_age"] == pytest.approx(31.0)
    assert row["reference_is_female"] == pytest.approx(0.0)
    assert row["reference_is_married"] == pytest.approx(1.0)
    assert row["count_under_18"] == pytest.approx(1.0)
    # Household 5 has neither spouse nor child.
    assert receiver.loc[105, "reference_is_married"] == pytest.approx(0.0)
    assert receiver.loc[105, "count_under_18"] == pytest.approx(0.0)

def test_qrf_predicts_once_per_source_unit_and_fans_out_identical_clones(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    expanded = clone_us_frame_for_puf_support(_frame(10))
    calls: dict[str, object] = {}

    class FakeFitted:
        def predict(self, receiver: pd.DataFrame) -> pd.DataFrame:
            calls["receiver"] = receiver.copy()
            return pd.DataFrame(
                {_OUTPUT: (np.arange(len(receiver)) % 4) != 0},
                index=receiver.index,
            )

    class FakeQRF:
        def __init__(self, **kwargs: object) -> None:
            calls["init"] = kwargs

        def fit(
            self,
            training: pd.DataFrame,
            *,
            predictors: list[str],
            targets: list[str],
            weights: np.ndarray,
        ) -> FakeFitted:
            calls["training"] = training.copy()
            calls["predictors"] = predictors
            calls["targets"] = targets
            calls["weights"] = weights.copy()
            return FakeFitted()

    monkeypatch.setattr(module, "QRF", FakeQRF)
    predicted = impute_us_voluntary_filing(expanded, _donor(), seed=17, n_estimators=9)

    assert calls["init"] == {"n_estimators": 9, "seed": 17}
    assert calls["predictors"] == list(SIPP_VOLUNTARY_FILING_MODEL_PREDICTORS)
    assert calls["targets"] == [_OUTPUT]
    assert len(calls["receiver"]) == 10
    assert len(predicted) == 20
    tax_unit = expanded.table("tax_unit")
    by_source = pd.DataFrame(
        {
            "source": tax_unit["tax_unit_source_id"],
            "predicted": predicted.to_numpy(),
        }
    ).groupby("source")["predicted"]
    assert (by_source.nunique() == 1).all()

def test_puf_only_survivor_units_predict_from_the_surviving_clone(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A unit whose ASEC row was dropped by selection still predicts once.

    Build M's sparse run died here: the certified frozen-support selection
    keeps only the PUF clone for some source units (the L0-survivor case the
    SSI reporter lineage already handles), and the receiver demanded exactly
    one ASEC row per unit. The surviving clone carries the unit's source
    predictors, so it serves as the prediction row; duplicated ASEC rows
    remain a hard error.
    """

    expanded = clone_us_frame_for_puf_support(_frame(10))
    person = expanded.table("person")
    tax_unit = expanded.table("tax_unit")
    dropped_source = tax_unit["tax_unit_source_id"].iloc[0]
    dropped_units = tax_unit.loc[
        tax_unit["tax_unit_source_id"].eq(dropped_source)
        & tax_unit["tax_unit_support_channel"].eq("asec"),
        "tax_unit_id",
    ]
    dropped = person["person_tax_unit_id"].isin(dropped_units)
    sparse = expanded.select(~dropped.to_numpy())

    class FakeFitted:
        def predict(self, receiver: pd.DataFrame) -> pd.DataFrame:
            return pd.DataFrame(
                {_OUTPUT: np.ones(len(receiver), dtype=bool)},
                index=receiver.index,
            )

    class FakeQRF:
        def __init__(self, **_kwargs: object) -> None:
            pass

        def fit(self, *_args: object, **_kwargs: object) -> FakeFitted:
            return FakeFitted()

    monkeypatch.setattr(module, "QRF", FakeQRF)
    predicted = impute_us_voluntary_filing(sparse, _donor(), seed=17)
    survivors = sparse.table("tax_unit")["tax_unit_source_id"].eq(dropped_source)
    assert survivors.any()
    assert len(predicted) == len(sparse.table("tax_unit"))
    assert predicted[survivors.to_numpy()].all()

def test_duplicate_same_role_source_rows_fail_closed() -> None:
    expanded = clone_us_frame_for_puf_support(_frame(10))
    tax_unit = expanded.table("tax_unit")
    asec_rows = tax_unit.index[tax_unit["tax_unit_support_channel"].eq("asec")].tolist()
    tax_unit.loc[asec_rows[1], "tax_unit_source_id"] = tax_unit.loc[
        asec_rows[0], "tax_unit_source_id"
    ]

    with pytest.raises(ValueError, match="duplicated support copies"):
        impute_us_voluntary_filing(expanded, _donor(), seed=17)

def test_historical_tail_copy_fans_the_source_decision_to_every_copy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    frame = _historical_tail_frame()
    tax_unit = frame.table("tax_unit")
    assert "tax_unit_spine_source_id" not in tax_unit
    assert tax_unit["tax_unit_support_clone_index"].value_counts().to_dict() == {
        0: 6,
        1: 6,
        2: 2,
    }
    assert set(
        tax_unit.loc[
            tax_unit["tax_unit_support_clone_index"].eq(2),
            "tax_unit_support_channel",
        ]
    ) == {"puf_tax_detail"}

    _IncomeThresholdQRF.receivers.clear()
    monkeypatch.setattr(module, "QRF", _IncomeThresholdQRF)
    predicted = impute_us_voluntary_filing(
        _divergent_tail_wages(frame), _donor(), seed=17
    )
    (receiver,) = _IncomeThresholdQRF.receivers
    assert len(receiver) == 6  # one canonical row per source tax unit
    by_source = pd.DataFrame(
        {"source": tax_unit["tax_unit_source_id"], "value": predicted}
    ).groupby("source")["value"]
    assert (by_source.nunique() == 1).all()
    # Unit wages are 3,000 x household id (+2,000 per spouse): units 3 to 6
    # file. The zero-wage tail copies inherit the native copy's decision.
    assert by_source.first().to_dict() == {
        101: False,
        102: False,
        103: True,
        104: True,
        105: True,
        106: True,
    }
    tail = tax_unit["tax_unit_support_clone_index"].eq(2).to_numpy()
    assert predicted[tail].all()

    materialized = _replace_tax_unit(frame, **{_OUTPUT: predicted.to_numpy()})
    summary = us_voluntary_filing_summary(materialized)
    assert summary["clone_source_units"] == 6
    assert summary["clone_mismatch_source_units"] == 0

    flipped = predicted.to_numpy().copy()
    flipped[np.flatnonzero(tail)[0]] = False
    mismatch = _replace_tax_unit(frame, **{_OUTPUT: flipped})
    mismatch_summary = us_voluntary_filing_summary(mismatch)
    assert mismatch_summary["clone_mismatch_source_units"] == 1
    gate = us_voluntary_filing_signal_gate(mismatch)
    assert any("disagree for 1 source unit" in failure for failure in gate.failures)

def test_historical_puf_only_survivor_predicts_from_the_primary_detail_copy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    frame = _divergent_tail_wages(_historical_tail_frame())
    tax_unit = frame.table("tax_unit")
    person = frame.table("person")
    native_unit_six = tax_unit.loc[
        tax_unit["tax_unit_source_id"].eq(106)
        & tax_unit["tax_unit_support_clone_index"].eq(0),
        "tax_unit_id",
    ]
    survivors = frame.select(
        ~person["person_tax_unit_id"].isin(native_unit_six).to_numpy()
    )

    _IncomeThresholdQRF.receivers.clear()
    monkeypatch.setattr(module, "QRF", _IncomeThresholdQRF)
    predicted = impute_us_voluntary_filing(survivors, _donor(), seed=17)
    surviving_units = survivors.table("tax_unit")
    source_six = surviving_units["tax_unit_source_id"].eq(106).to_numpy()
    surviving_clones = surviving_units.loc[source_six, "tax_unit_support_clone_index"]
    assert surviving_clones.tolist() == [1, 2]
    # Copy rank picks the primary PUF-detail copy (wages 18,000), never the
    # zero-wage tail copy.
    assert predicted[source_six].tolist() == [True, True]

def test_historical_duplicate_clone_index_still_fails_closed() -> None:
    frame = _historical_tail_frame()
    tax_unit = frame.table("tax_unit")
    tail = tax_unit["tax_unit_support_clone_index"].eq(2).to_numpy()
    clone_index = tax_unit["tax_unit_support_clone_index"].to_numpy().copy()
    clone_index[tail] = 1
    duplicated = _replace_tax_unit(
        frame, **{"tax_unit_support_clone_index": clone_index}
    )

    with pytest.raises(ValueError, match=r"duplicated support copies.*\('103', 1\)"):
        impute_us_voluntary_filing(duplicated, _donor(), seed=17)

@pytest.mark.parametrize("clone_index", [3, 7, 2**62])
@pytest.mark.parametrize("dtype", [np.int64, np.float64], ids=["int", "float"])
def test_historical_out_of_domain_clone_refused_before_prediction(
    clone_index: int,
    dtype,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    frame = _historical_tail_frame()
    tax_unit = frame.table("tax_unit")
    clone_indices = tax_unit["tax_unit_support_clone_index"].to_numpy(
        dtype=dtype, copy=True
    )
    clone_indices[clone_indices == 2] = clone_index
    frame = _replace_tax_unit(frame, tax_unit_support_clone_index=clone_indices)
    receiver = pd.DataFrame(
        {
            predictor: np.zeros(len(tax_unit))
            for predictor in SIPP_VOLUNTARY_FILING_MODEL_PREDICTORS
        },
        index=tax_unit["tax_unit_id"],
    )

    def unexpected_qrf(**kwargs):
        pytest.fail("Invalid historical clone indices must fail before QRF")

    monkeypatch.setattr(module, "QRF", unexpected_qrf)
    with pytest.raises(ValueError, match="historical"):
        module._source_receiver_rows(frame, receiver)
    with pytest.raises(ValueError, match="historical"):
        impute_us_voluntary_filing(frame, _donor(), seed=17)

def test_assembled_frame_without_clone_indices_fails_closed() -> None:
    # Microcosm #992 gate finding: with the clone-index column dropped, an
    # assembled ASEC-channel frame once fell back to role ranks. Assembled
    # channels name physical sources, so the copy rank needs clone indices.
    tax_unit = pd.DataFrame(
        {
            "tax_unit_id": [100, 200],
            "tax_unit_source_id": [10, 20],
            "tax_unit_spine_source_id": [1, 2],
            "tax_unit_support_channel": ["asec", "asec"],
        }
    )
    receiver = pd.DataFrame(
        {
            predictor: np.arange(2, dtype=np.float64)
            for predictor in SIPP_VOLUNTARY_FILING_MODEL_PREDICTORS
        },
        index=tax_unit["tax_unit_id"],
    )

    class TaxUnitFrame:
        def table(self, entity: str) -> pd.DataFrame:
            assert entity == "tax_unit"
            return tax_unit

    with pytest.raises(
        ValueError,
        match=r"assembled support metadata requires 'tax_unit_support_clone_index'",
    ):
        module._source_receiver_rows(TaxUnitFrame(), receiver)

@pytest.mark.parametrize(
    "channels",
    [["asec"] * 4, ["asec", "puf_tax_detail"] * 2],
    ids=["all_asec", "role_labels"],
)
@pytest.mark.parametrize(
    "output",
    [[False, True, True, True], [False, False, True, True]],
    ids=["copies_disagree", "copies_agree"],
)
def test_gate_refuses_assembled_frame_without_clone_indices(
    channels: list[str],
    output: list[bool],
) -> None:
    # Microcosm #992 gate finding (c), exactly as reported: source IDs
    # [10, 10, 20, 20] with matching spine IDs, all 'asec' channels and
    # outputs [F, T, T, T]. With the clone-index column gone the summary fell
    # back to (source, role) occurrence pairing, reported 0 mismatched source
    # units and the gate passed; the base grouped by source ID and failed.
    # When the copies agree the fallback still under-reported
    # clone_source_units (0 instead of 2). The gate now refuses the table.
    frame = _assembled_gate_frame(output, channels=channels)
    pattern = (
        r"assembled support metadata requires 'tax_unit_support_clone_index' "
        r"alongside 'tax_unit_spine_source_id'"
    )
    with pytest.raises(ValueError, match=pattern):
        us_voluntary_filing_summary(frame)
    with pytest.raises(ValueError, match=pattern):
        us_voluntary_filing_signal_gate(frame)

@pytest.mark.parametrize(
    ("channels", "clone_indices", "missing"),
    [
        pytest.param(
            None,
            None,
            r"'tax_unit_support_channel' and 'tax_unit_support_clone_index'",
            id="no_channel_no_clone",
        ),
        pytest.param(
            None,
            [0, 1, 0, 1],
            r"'tax_unit_support_channel'",
            id="clone_index_without_channel",
        ),
    ],
)
def test_gate_refuses_assembled_frame_without_support_channel(
    channels: list[str] | None,
    clone_indices: list[int] | None,
    missing: str,
) -> None:
    # Before #992 the gate skipped the clone comparison entirely for a table
    # with neither provenance column, even with raw spine IDs and copies that
    # disagree. An assembled table must carry complete provenance.
    frame = _assembled_gate_frame(
        [False, True, True, True],
        channels=channels,
        clone_indices=clone_indices,
    )
    with pytest.raises(
        ValueError,
        match=r"assembled support metadata requires " + missing,
    ):
        us_voluntary_filing_signal_gate(frame)

@pytest.mark.parametrize(
    "consumer", [us_voluntary_filing_summary, us_voluntary_filing_signal_gate]
)
@pytest.mark.parametrize(
    "missing_columns",
    [
        ("tax_unit_support_channel",),
        ("tax_unit_support_clone_index",),
        ("tax_unit_support_channel", "tax_unit_support_clone_index"),
    ],
    ids=["no_channel", "no_clone_index", "no_support_metadata"],
)
@pytest.mark.parametrize("empty", [False, True], ids=["missing_output", "empty"])
def test_gate_validates_assembled_provenance_before_output_or_weights(
    consumer,
    missing_columns: tuple[str, ...],
    empty: bool,
) -> None:
    # The earlier fix checked provenance only after reading the output and
    # weights. The gate's missing-output return bypassed that check entirely.
    # Empty tables cannot form a valid Frame, but direct callers still need
    # the same provenance refusal before any output/weight processing.
    tax_unit = pd.DataFrame(
        {
            "tax_unit_id": [100],
            "tax_unit_source_id": [10],
            "tax_unit_spine_source_id": [1],
            "tax_unit_support_channel": ["asec"],
            "tax_unit_support_clone_index": [0],
        }
    ).drop(columns=list(missing_columns))
    if empty:
        tax_unit = tax_unit.iloc[:0].assign(**{_OUTPUT: pd.Series(dtype=bool)})

    class TaxUnitFrame:
        def table(self, entity: str) -> pd.DataFrame:
            assert entity == "tax_unit"
            return tax_unit

        def resolve_weights(self, entity: str):
            pytest.fail("Incomplete assembled provenance must precede weights")

    with pytest.raises(ValueError, match="assembled support metadata requires"):
        consumer(TaxUnitFrame())

def test_receiver_refuses_assembled_frame_without_support_metadata() -> None:
    # Unique source IDs give the one-row-per-source path nothing to reject,
    # so the stripped assembled table must be refused by provenance alone.
    tax_unit = pd.DataFrame(
        {
            "tax_unit_id": [100, 200, 300, 400],
            "tax_unit_source_id": [10, 20, 30, 40],
            "tax_unit_spine_source_id": [1, 2, 3, 4],
        }
    )
    receiver = pd.DataFrame(
        {
            predictor: np.arange(4, dtype=np.float64)
            for predictor in SIPP_VOLUNTARY_FILING_MODEL_PREDICTORS
        },
        index=tax_unit["tax_unit_id"],
    )

    class TaxUnitFrame:
        def table(self, entity: str) -> pd.DataFrame:
            assert entity == "tax_unit"
            return tax_unit

    with pytest.raises(
        ValueError,
        match=(
            r"assembled support metadata requires 'tax_unit_support_channel' "
            r"and 'tax_unit_support_clone_index'"
        ),
    ):
        module._source_receiver_rows(TaxUnitFrame(), receiver)

def test_gate_compares_every_historical_copy_with_repeated_clone_indices() -> None:
    # A historical table whose tail copy repeats clone index 1 is malformed
    # (both imputers refuse it). The gate still compares every copy of the
    # source unit, so the duplicated copy's disagreement is caught; the old
    # (source, role) occurrence pairing left it unpaired and unchecked.
    frame = _historical_tail_frame()
    tax_unit = frame.table("tax_unit")
    clone_index = tax_unit["tax_unit_support_clone_index"].to_numpy().copy()
    tail = clone_index == 2
    clone_index[tail] = 1
    output = tax_unit["tax_unit_source_id"].isin([103, 104, 105, 106]).to_numpy()
    agreeing = _replace_tax_unit(
        frame,
        **{"tax_unit_support_clone_index": clone_index, _OUTPUT: output},
    )
    summary = us_voluntary_filing_summary(agreeing)
    assert summary["clone_source_units"] == 6
    assert summary["clone_mismatch_source_units"] == 0

    disagreeing_output = output.copy()
    disagreeing_output[np.flatnonzero(tail)[0]] = False
    disagreeing = _replace_tax_unit(agreeing, **{_OUTPUT: disagreeing_output})
    assert us_voluntary_filing_summary(disagreeing)["clone_mismatch_source_units"] == 1
    gate = us_voluntary_filing_signal_gate(disagreeing)
    assert any("disagree for 1 source unit" in failure for failure in gate.failures)

@pytest.mark.parametrize(
    "output",
    [[False, False, True, True], [False, True, True, True]],
    ids=["copies_agree", "copies_disagree"],
)
def test_gate_flags_duplicate_sources_without_support_metadata(
    output: list[bool],
) -> None:
    frame = _replace_tax_unit(
        _frame(4),
        tax_unit_source_id=np.asarray([10, 10, 20, 20]),
        **{_OUTPUT: np.asarray(output)},
    )
    summary = us_voluntary_filing_summary(frame)
    assert summary["clone_metadata_missing"] is True
    gate = us_voluntary_filing_signal_gate(frame)
    assert not gate.passed
    assert any("provenance" in failure for failure in gate.failures)

@pytest.mark.parametrize(
    "consumer", [us_voluntary_filing_summary, us_voluntary_filing_signal_gate]
)
@pytest.mark.parametrize(
    "output",
    [
        [False, False, False, True, True],
        [False, False, True, True, True],
    ],
    ids=["copies_agree", "unpaired_copy_disagrees"],
)
def test_channel_only_gate_refuses_duplicated_source_role(
    consumer,
    output: list[bool],
) -> None:
    frame = _replace_tax_unit(
        _frame(5),
        tax_unit_source_id=np.asarray([10, 10, 10, 20, 20]),
        tax_unit_support_channel=np.asarray(
            ["asec", "puf_tax_detail", "puf_tax_detail", "asec", "puf_tax_detail"]
        ),
        **{_OUTPUT: np.asarray(output)},
    )
    with pytest.raises(ValueError, match="duplicated same-role rows"):
        consumer(frame)

def test_unique_sources_without_metadata_keep_the_unexpanded_summary() -> None:
    frame = _replace_tax_unit(
        _frame(4), **{_OUTPUT: np.asarray([False, True, True, True])}
    )
    source_frame = _replace_tax_unit(
        frame, tax_unit_source_id=np.asarray([10, 20, 30, 40])
    )
    assert us_voluntary_filing_summary(source_frame) == us_voluntary_filing_summary(
        frame
    )
    assert us_voluntary_filing_signal_gate(source_frame).passed

def test_channel_only_unique_source_role_pairs_keep_valid_clone_diagnostics() -> None:
    frame = _replace_tax_unit(
        _frame(4),
        tax_unit_source_id=np.asarray([10, 10, 20, 20]),
        tax_unit_support_channel=np.asarray(["asec", "puf_tax_detail"] * 2),
        **{_OUTPUT: np.asarray([False, False, True, True])},
    )
    summary = us_voluntary_filing_summary(frame)
    assert summary["clone_source_units"] == 2
    assert summary["clone_mismatch_source_units"] == 0
    assert summary["clone_metadata_missing"] is False
    assert us_voluntary_filing_signal_gate(frame).passed

@pytest.mark.parametrize("assembled", [True, False], ids=["assembled", "historical"])
def test_clone_index_past_int64_fails_closed(assembled: bool) -> None:
    # float(2**63) is the first float past int64; the base let it wrap to
    # INT64_MAX and both the receiver and the gate ran normally.
    malformed = (
        r"PUF support metadata column 'tax_unit_support_clone_index' must "
        r"contain nonnegative integers \(finite and representable as int64\)"
    )
    if assembled:
        frame = _assembled_gate_frame(
            [False, False, True, True],
            channels=["acs"] * 4,
            clone_indices=[0, 1, 0, 1],
        )
        frame = _replace_tax_unit(
            frame,
            tax_unit_support_clone_index=np.asarray([0.0, 1.0, 0.0, float(2**63)]),
        )
    else:
        frame = _historical_tail_frame()
        clone_index = frame.table("tax_unit")["tax_unit_support_clone_index"]
        values = clone_index.to_numpy(dtype=np.float64).copy()
        values[clone_index.eq(2).to_numpy()] = float(2**63)
        frame = _replace_tax_unit(
            frame,
            **{
                "tax_unit_support_clone_index": values,
                _OUTPUT: np.ones(len(values), dtype=bool),
            },
        )
    with pytest.raises(ValueError, match=malformed):
        us_voluntary_filing_signal_gate(frame)
    receiver = pd.DataFrame(
        {
            predictor: np.zeros(len(frame.table("tax_unit")))
            for predictor in SIPP_VOLUNTARY_FILING_MODEL_PREDICTORS
        },
        index=frame.table("tax_unit")["tax_unit_id"],
    )
    with pytest.raises(ValueError, match=malformed):
        module._source_receiver_rows(frame, receiver)

def test_assembled_clone_two_uses_explicit_index_and_checks_every_clone() -> None:
    tax_unit = pd.DataFrame(
        {
            "tax_unit_id": [100, 101, 102, 200, 201],
            "tax_unit_source_id": [10, 10, 10, 20, 20],
            "tax_unit_spine_source_id": [1, 1, 1, 2, 2],
            "tax_unit_support_channel": ["acs"] * 5,
            "tax_unit_support_clone_index": [0, 1, 2, 1, 2],
        }
    )
    receiver = pd.DataFrame(
        {
            predictor: np.arange(5, dtype=np.float64) + offset
            for offset, predictor in enumerate(SIPP_VOLUNTARY_FILING_MODEL_PREDICTORS)
        },
        index=tax_unit["tax_unit_id"],
    )

    class TaxUnitFrame:
        def table(self, entity: str) -> pd.DataFrame:
            assert entity == "tax_unit"
            return tax_unit

    prediction_rows, fan_keys = module._source_receiver_rows(TaxUnitFrame(), receiver)
    assert prediction_rows.index.tolist() == ["10", "20"]
    assert fan_keys.tolist() == ["10", "10", "10", "20", "20"]
    # Source 10 prefers clone 0; source 20 has no native survivor and picks
    # the lowest surviving clone index, clone 1.
    assert prediction_rows.iloc[:, 0].tolist() == [0.0, 3.0]

    summary_frame = _replace_tax_unit(
        _frame(3),
        **{
            _OUTPUT: np.asarray([False, False, True]),
            "tax_unit_source_id": np.asarray([7, 7, 7]),
            "tax_unit_spine_source_id": np.asarray([70, 70, 70]),
            "tax_unit_support_channel": np.asarray(["acs", "acs", "acs"]),
            "tax_unit_support_clone_index": np.asarray([0, 1, 2]),
        },
    )
    summary = us_voluntary_filing_summary(summary_frame)
    assert summary["clone_source_units"] == 1
    assert summary["clone_mismatch_source_units"] == 1

def test_real_qrf_recomputation_is_deterministic() -> None:
    frame = _frame(14)
    donor = _donor(120)
    first = impute_us_voluntary_filing(frame, donor, seed=31, n_estimators=8)
    second = impute_us_voluntary_filing(
        frame,
        donor.sample(frac=1.0, random_state=99),
        seed=31,
        n_estimators=8,
    )
    pd.testing.assert_series_equal(first, second)

def test_wrapper_recomputes_stale_signal_and_is_idempotent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    frame = _replace_tax_unit(
        _frame(10),
        **{_OUTPUT: np.asarray([True, False] * 5)},
    )
    expected = pd.Series(
        np.asarray([False, True, True, True, True] * 2),
        index=frame.table("tax_unit").index,
        name=_OUTPUT,
    )
    monkeypatch.setattr(module, "us_voluntary_filing_stage_spec", lambda: object())
    monkeypatch.setattr(
        module,
        "impute_us_voluntary_filing",
        lambda *_args, **_kwargs: expected,
    )

    restored = with_us_voluntary_filing_input(
        frame,
        seed=1,
        time_period=2024,
        sipp_donor=_donor(),
    )
    assert np.array_equal(restored.table("tax_unit")[_OUTPUT], expected)
    assert restored is not frame
    repeated = with_us_voluntary_filing_input(
        restored,
        seed=1,
        time_period=2024,
        sipp_donor=_donor(),
    )
    assert repeated is restored

def test_signal_gate_requires_boolean_plausible_and_clone_consistent() -> None:
    expanded = clone_us_frame_for_puf_support(_frame(10))
    values = np.tile(
        np.asarray([False, True, True, True, True, False, True, True, False, True]),
        2,
    )
    valid = _replace_tax_unit(expanded, **{_OUTPUT: values})
    summary = us_voluntary_filing_summary(valid)
    assert summary["clone_source_units"] == 10
    assert summary["clone_mismatch_source_units"] == 0
    gate = us_voluntary_filing_signal_gate(valid)
    assert gate.passed, gate.failures

    mismatched_values = values.copy()
    mismatched_values[10] = ~mismatched_values[0]
    mismatch = _replace_tax_unit(expanded, **{_OUTPUT: mismatched_values})
    mismatch_gate = us_voluntary_filing_signal_gate(mismatch)
    assert not mismatch_gate.passed
    assert any("clones disagree" in failure for failure in mismatch_gate.failures)

    constant = _replace_tax_unit(_frame(10), **{_OUTPUT: np.ones(10, dtype=bool)})
    constant_gate = us_voluntary_filing_signal_gate(constant)
    assert not constant_gate.passed
    assert any("constant" in failure for failure in constant_gate.failures)

@pytest.mark.parametrize(
    ("column", "value", "message"),
    [
        ("employment_income", np.nan, "finite and nonnegative"),
        ("tax_unit_weight", 0.0, "finite and positive"),
        (_OUTPUT, 2, "target must be boolean"),
    ],
)
def test_imputer_fails_closed_on_invalid_donor(
    column: str, value: float, message: str
) -> None:
    donor = _donor()
    if column == _OUTPUT:
        donor[column] = donor[column].astype(np.int8)
    donor.loc[0, column] = value
    with pytest.raises(ValueError, match=message):
        impute_us_voluntary_filing(_frame(), donor, seed=0, n_estimators=4)
