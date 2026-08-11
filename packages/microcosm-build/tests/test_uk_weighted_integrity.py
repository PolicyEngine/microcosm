"""UK weighted-integrity gates: input-mass parity + QRF tail concentration.

Increment 4 of the UK parity plan (#609): both gates reuse the shared
implementations in ``microcosm.build.gates`` and add the national-table
evidence plumbing plus the universal reviewed-exclusion discipline
(mandatory reason, dormant reported, stale fails).
"""

from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from microcosm.build.uk_runtime import weighted_integrity
from microcosm.build.uk_runtime.spi_support import (
    FRS_ONLY_SPI_FILL_PERSON_COLUMNS,
    SPI_INCOME_QRF_OUTPUT_COLUMNS,
)
from microcosm.build.uk_runtime.weighted_integrity import (
    UK_INPUT_MASS_EXCLUSION_REGISTER_RESOURCE,
    UK_QRF_TAIL_EXCLUSION_REGISTER_RESOURCE,
    UKInputMassParityPolicy,
    UKInputMassReference,
    UKQRFTailConcentrationPolicy,
    load_uk_input_mass_reference,
    load_uk_reviewed_exclusion_register,
    uk_dataset_input_mass_totals,
    uk_input_mass_parity_gate,
    uk_qrf_tail_concentration_columns,
    uk_qrf_tail_concentration_gate,
)


def _entry(reason: str, *, expires_on: str = "2027-02-10") -> dict[str, str]:
    """A valid schema-2 approval receipt around the fixture's reason."""

    return {
        "reason": reason,
        "approved_by": "test-reviewer",
        "adjudication": "microcosm#610",
        "approved_on": "2026-08-10",
        "expires_on": expires_on,
    }


def _dataset(
    *,
    n: int = 4,
    weights: list[float] | np.ndarray | None = None,
    person_columns: dict[str, object] | None = None,
):
    if weights is None:
        weights = np.ones(n, dtype=float)
    household_ids = np.arange(1, n + 1, dtype=np.int64)
    person = {
        "person_id": np.arange(101, 101 + n, dtype=np.int64),
        "person_household_id": household_ids,
        "person_benunit_id": np.arange(201, 201 + n, dtype=np.int64),
        "employment_income": np.arange(1, n + 1, dtype=float),
    }
    person.update(person_columns or {})
    return SimpleNamespace(
        person=pd.DataFrame(person),
        benunit=pd.DataFrame({"benunit_id": np.arange(201, 201 + n, dtype=np.int64)}),
        household=pd.DataFrame(
            {
                "household_id": household_ids,
                "household_weight": np.asarray(weights, dtype=float),
                "council_tax": np.full(n, 10.0),
            }
        ),
    )


def _reference(totals, **overrides) -> UKInputMassReference:
    fields = {
        "filename": "enhanced_frs_2023_24.h5",
        "revision": "655dd07e4bb9c777b00dac044949611f1feb824f",
        "sha256": ("584ae33d80ca0431254610a3f8254d132da73477d31966d6446282861ecae50d"),
        "vintage": "2023_24",
    }
    fields.update(overrides)
    return UKInputMassReference(totals=totals, **fields)


def _policy(**overrides) -> UKInputMassParityPolicy:
    fields = {"relative_tolerance": 0.5, "minimum_reference_total": 0.0}
    fields.update(overrides)
    return UKInputMassParityPolicy(**fields)


def _synthetic_input_mass_gate(*args, **kwargs):
    """Exercise gate semantics with small totals, outside the licensed pin."""

    with patch.object(
        weighted_integrity,
        "_validate_input_mass_reference",
        return_value=None,
    ):
        return uk_input_mass_parity_gate(*args, **kwargs)


def test_dataset_totals_broadcast_household_weights_by_membership() -> None:
    dataset = _dataset(weights=[2.0, 1.0, 1.0, 1.0])

    totals = uk_dataset_input_mass_totals(dataset)

    # person values 1..4 under weights [2,1,1,1] -> 2 + 2 + 3 + 4.
    assert totals["person.employment_income"] == 11.0
    assert totals["household.council_tax"] == 50.0
    assert "household.household_weight" not in totals
    assert "person.person_household_id" not in totals


def test_dataset_totals_fail_closed_on_memberless_benunits() -> None:
    dataset = _dataset()
    dataset.benunit = pd.DataFrame({"benunit_id": [201, 202, 203, 204, 999]})

    with pytest.raises(ValueError, match="no member persons"):
        uk_dataset_input_mass_totals(dataset)


def test_dataset_totals_fail_closed_on_benunit_split_across_households() -> None:
    """A split benunit has no single household weight to inherit."""

    dataset = _dataset()
    # Person 4 keeps household 4 but joins benunit 201, which sits in
    # household 1: benunit 201 would silently take whichever came first.
    dataset.person.loc[3, "person_benunit_id"] = 201

    with pytest.raises(ValueError, match="exactly one household"):
        uk_dataset_input_mass_totals(dataset)


def test_zeroed_input_column_fails_by_name_at_any_tolerance() -> None:
    dataset = _dataset(person_columns={"employment_income": [0.0, 0.0, 0.0, 0.0]})
    reference = _reference({"person.employment_income": 10.0})

    gate = _synthetic_input_mass_gate(
        uk_dataset_input_mass_totals(dataset),
        reference,
        policy=_policy(relative_tolerance=1e9),
    )

    assert gate.name == "input_mass_parity"
    assert not gate.passed
    assert "person.employment_income" in gate.failures[0]
    assert "mass is zero" in gate.failures[0]


def test_material_mass_loss_fails_and_within_tolerance_passes() -> None:
    reference = _reference({"person.employment_income": 10.0})
    lost = _synthetic_input_mass_gate(
        {"person.employment_income": 0.01},
        reference,
        policy=_policy(),
    )
    kept = _synthetic_input_mass_gate(
        {"person.employment_income": 9.0, "person.pension_income": 5.0},
        reference,
        policy=_policy(),
    )

    assert not lost.passed
    assert "-99.9%" in lost.failures[0]
    assert kept.passed
    # Candidate-only columns are reported, never failed.
    assert kept.details["candidate_only_columns"] == ["person.pension_income"]


def test_input_mass_reference_identity_is_recorded() -> None:
    reference = _reference({"person.employment_income": 10.0})

    gate = _synthetic_input_mass_gate(
        {"person.employment_income": 10.0},
        reference,
        policy=_policy(),
    )

    assert gate.details["reference_identity"] == {
        "filename": "enhanced_frs_2023_24.h5",
        "revision": "655dd07e4bb9c777b00dac044949611f1feb824f",
        "sha256": ("584ae33d80ca0431254610a3f8254d132da73477d31966d6446282861ecae50d"),
        "vintage": "2023_24",
    }


def test_input_mass_reference_rejects_substituted_totals_at_approved_identity(
    tmp_path,
) -> None:
    caller_self_reference = _reference(
        {"person.employment_income": 1.0},
    )

    with pytest.raises(ValueError, match="reference totals must match the reviewed"):
        uk_input_mass_parity_gate(
            {"person.employment_income": 1.0},
            caller_self_reference,
            policy=_policy(),
        )
    path = tmp_path / "self-reference.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "identity": caller_self_reference.identity,
                "totals": dict(caller_self_reference.totals),
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="reference totals must match the reviewed"):
        load_uk_input_mass_reference(path)


def test_input_mass_reference_identity_pin_cannot_be_shadowed_from_cwd(
    tmp_path,
    monkeypatch,
) -> None:
    shadow = {
        "schema_version": 3,
        "source": {
            "repo_id": "caller/repo",
            "repo_type": "model",
            "filename": "caller.h5",
            "revision": "caller",
            "sha256": "b" * 64,
            "url": "https://example.invalid/caller.h5",
            "vintage": "caller",
            "period": "2023",
            "size_bytes": 1,
        },
        "nonzero_shares": {"employment_income": 1.0},
        "input_entities": {"employment_income": "person"},
    }
    (tmp_path / "efrs_parity_reference.json").write_text(
        json.dumps(shadow), encoding="utf-8"
    )
    monkeypatch.chdir(tmp_path)
    caller_reference = _reference(
        {"person.employment_income": 1.0},
        filename="caller.h5",
        revision="caller",
        sha256="b" * 64,
        vintage="caller",
    )

    with pytest.raises(ValueError, match="identity must match the reviewed"):
        uk_input_mass_parity_gate(
            {"person.employment_income": 1.0},
            caller_reference,
            policy=_policy(),
        )


def test_input_mass_exclusion_discipline_live_stale_dormant() -> None:
    reason = "Seeded reviewed loss for the fixture."
    reference = _reference({"person.employment_income": 10.0, "person.tiny_layer": 0.5})
    policy = _policy(
        minimum_reference_total=1.0,
        reviewed_exclusions={
            "person.employment_income": _entry(reason),
            "person.tiny_layer": _entry(reason),
            "person.never_shipped": _entry(reason),
        },
    )

    live = _synthetic_input_mass_gate(
        {"person.employment_income": 0.0},
        reference,
        policy=policy,
    )
    stale = _synthetic_input_mass_gate(
        {"person.employment_income": 10.0},
        reference,
        policy=policy,
    )

    # A live exclusion suppresses the zeroed-column failure and is recorded.
    assert live.passed
    assert live.details["reviewed_exclusions"]["person.employment_income"] == reason
    # Below-floor and absent-from-reference entries are dormant, not failing.
    assert live.details["dormant_exclusions"] == [
        "person.never_shipped",
        "person.tiny_layer",
    ]
    # A column now within tolerance is a rotted register entry and fails.
    assert not stale.passed
    assert stale.details["stale_exclusions"] == ["person.employment_income"]
    assert "Stale reviewed input-mass exclusions" in stale.failures[0]


def test_input_mass_policy_and_reference_validation() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        _policy(relative_tolerance=-0.1)
    with pytest.raises(ValueError, match="reason must be a non-empty string"):
        _policy(reviewed_exclusions={"person.employment_income": _entry("  ")})
    with pytest.raises(ValueError, match="lowercase sha256"):
        _reference({"person.employment_income": 1.0}, sha256="nope")
    with pytest.raises(ValueError, match="non-empty mapping"):
        _reference({})
    with pytest.raises(ValueError, match="finite"):
        _reference({"person.employment_income": float("nan")})
    with pytest.raises(ValueError, match="non-empty, trimmed column names"):
        _policy(reviewed_exclusions={1: _entry("Seeded invalid name.")})
    with pytest.raises(TypeError, match="must be an object with fields"):
        _policy(reviewed_exclusions={"person.employment_income": None})


def test_qrf_surface_is_derived_from_the_source_manifest() -> None:
    from microcosm.build.uk_runtime.hmrc_source_contract import (
        uk_hmrc_weighted_qrf_output_columns,
    )

    declared = uk_hmrc_weighted_qrf_output_columns()

    # Stage-1 outputs come first and match the executable runtime surface;
    # stage-2 declares the FRS-only fill columns minus reviewed absences.
    assert declared[: len(SPI_INCOME_QRF_OUTPUT_COLUMNS)] == (
        SPI_INCOME_QRF_OUTPUT_COLUMNS
    )
    assert set(declared[len(SPI_INCOME_QRF_OUTPUT_COLUMNS) :]) <= set(
        FRS_ONLY_SPI_FILL_PERSON_COLUMNS
    )
    assert len(declared) == len(set(declared))


def test_qrf_columns_check_every_declared_output_regardless_of_density() -> None:
    n = 6
    dense = np.full(n, 5.0)  # 100% nonzero: the US 5% cutoff would skip it.
    dataset = _dataset(n=n, person_columns={"self_employment_income": dense})

    values, weights, surface = uk_qrf_tail_concentration_columns(dataset)

    assert "self_employment_income" in values
    assert values["self_employment_income"].shape == (n,)
    assert weights["self_employment_income"].shape == (n,)
    assert surface["checked_columns"] == ["self_employment_income"]
    assert "density_filter" in surface
    # Declared outputs the fixture does not carry are reported, not invented.
    assert "dividend_income" in surface["absent_columns"]
    assert surface["declared_qrf_outputs"] >= 47


def test_declared_absent_qrf_output_is_a_named_gate_failure() -> None:
    values, weights, surface = uk_qrf_tail_concentration_columns(
        _dataset(),
        output_columns=("declared_but_absent",),
    )

    assert set(values) == {"declared_but_absent"}
    assert set(weights) == {"declared_but_absent"}
    gate = uk_qrf_tail_concentration_gate(
        values,
        weights,
        policy=UKQRFTailConcentrationPolicy(
            top_k=1,
            max_top_share=0.5,
            min_nonzero_records=2,
        ),
        surface=surface,
    )

    assert not gate.passed
    assert gate.details["columns_checked"] == 0
    assert gate.details["surface"]["absent_columns"] == ["declared_but_absent"]
    assert "declared_but_absent" in gate.failures[0]
    assert "absent" in gate.failures[0]


def test_declared_nonnumeric_qrf_output_is_a_named_gate_failure() -> None:
    values, weights, surface = uk_qrf_tail_concentration_columns(
        _dataset(person_columns={"declared_nonnumeric": ["x"] * 4}),
        output_columns=("declared_nonnumeric",),
    )

    assert set(values) == {"declared_nonnumeric"}
    assert set(weights) == {"declared_nonnumeric"}
    gate = uk_qrf_tail_concentration_gate(
        values,
        weights,
        policy=UKQRFTailConcentrationPolicy(
            top_k=1,
            max_top_share=0.5,
            min_nonzero_records=2,
        ),
        surface=surface,
    )

    assert not gate.passed
    assert gate.details["columns_checked"] == 0
    assert gate.details["surface"]["non_numeric_columns"] == ["declared_nonnumeric"]
    assert "declared_nonnumeric" in gate.failures[0]
    assert "not numeric" in gate.failures[0]


def test_qrf_surface_rejects_partially_omitted_declared_outputs() -> None:
    gate = uk_qrf_tail_concentration_gate(
        {"self_employment_income": np.ones(4)},
        {"self_employment_income": np.ones(4)},
        policy=UKQRFTailConcentrationPolicy(
            top_k=1,
            max_top_share=0.75,
            min_nonzero_records=2,
        ),
        surface={
            "declared_qrf_outputs": 2,
            "checked_columns": ["self_employment_income"],
            "absent_columns": [],
            "non_numeric_columns": [],
            "density_filter": "none: every declared output is checked (#609)",
        },
    )

    assert not gate.passed
    assert "QRF surface declarations must reconcile exactly" in gate.failures[0]


def test_concentrated_qrf_column_fails_by_name() -> None:
    n = 10
    concentrated = np.ones(n)
    concentrated[0] = 1_000.0
    dataset = _dataset(n=n, person_columns={"self_employment_income": concentrated})
    values, weights, surface = uk_qrf_tail_concentration_columns(dataset)

    gate = uk_qrf_tail_concentration_gate(
        values,
        weights,
        policy=UKQRFTailConcentrationPolicy(
            top_k=1,
            max_top_share=0.5,
            min_nonzero_records=2,
        ),
        surface=surface,
    )

    assert gate.name == "qrf_tail_concentration"
    assert not gate.passed
    assert "self_employment_income" in gate.failures[0]
    assert gate.details["surface"]["declared_qrf_outputs"] >= 47


def test_thin_qrf_column_is_reported_not_checked() -> None:
    n = 4
    dataset = _dataset(n=n, person_columns={"self_employment_income": np.ones(n)})
    values, weights, _surface = uk_qrf_tail_concentration_columns(dataset)

    gate = uk_qrf_tail_concentration_gate(
        values,
        weights,
        policy=UKQRFTailConcentrationPolicy(
            top_k=10,
            max_top_share=0.5,
            min_nonzero_records=100,
        ),
    )

    assert not gate.passed
    assert gate.details["thin_columns"]["self_employment_income"] == 4
    assert "No declared QRF output" in gate.failures[0]


def test_thin_qrf_exclusion_is_classified_as_dormant() -> None:
    values = {"person.x": np.ones(4)}
    weights = {"person.x": np.ones(4)}

    gate = uk_qrf_tail_concentration_gate(
        values,
        weights,
        policy=UKQRFTailConcentrationPolicy(
            top_k=10,
            max_top_share=0.5,
            min_nonzero_records=100,
            reviewed_exclusions={"person.x": _entry("Seeded thin entry.")},
        ),
    )

    assert not gate.passed
    assert gate.details["thin_columns"] == {"person.x": 4}
    assert gate.details["reviewed_exclusions"] == {}
    assert gate.details["stale_exclusions"] == []
    assert gate.details["dormant_exclusions"] == ["person.x"]


def test_qrf_stale_exclusion_fails_and_dormant_is_reported() -> None:
    n = 10
    dataset = _dataset(n=n, person_columns={"self_employment_income": np.ones(n)})
    values, weights, _surface = uk_qrf_tail_concentration_columns(dataset)

    gate = uk_qrf_tail_concentration_gate(
        values,
        weights,
        policy=UKQRFTailConcentrationPolicy(
            top_k=1,
            max_top_share=0.5,
            min_nonzero_records=2,
            reviewed_exclusions={
                "self_employment_income": _entry("Seeded stale entry."),
                "dividend_income": _entry("Seeded dormant entry."),
            },
        ),
    )

    assert not gate.passed
    assert "Stale reviewed exclusions" in gate.failures[0]
    assert gate.details["stale_exclusions"] == ["self_employment_income"]
    assert gate.details["dormant_exclusions"] == ["dividend_income"]


def test_qrf_policy_validation() -> None:
    with pytest.raises(ValueError, match="strict subset"):
        UKQRFTailConcentrationPolicy(
            top_k=10, max_top_share=0.5, min_nonzero_records=10
        )
    with pytest.raises(ValueError, match=r"in \(0, 1\)"):
        UKQRFTailConcentrationPolicy(top_k=1, max_top_share=1.0, min_nonzero_records=2)
    with pytest.raises(ValueError, match="reason must be a non-empty string"):
        UKQRFTailConcentrationPolicy(
            top_k=1,
            max_top_share=0.5,
            min_nonzero_records=2,
            reviewed_exclusions={"self_employment_income": _entry("")},
        )


def test_committed_exclusion_registers_load_and_are_empty() -> None:
    input_mass = load_uk_reviewed_exclusion_register(
        None,
        resource=UK_INPUT_MASS_EXCLUSION_REGISTER_RESOURCE,
    )
    qrf_tail = load_uk_reviewed_exclusion_register(
        None,
        resource=UK_QRF_TAIL_EXCLUSION_REGISTER_RESOURCE,
    )

    assert input_mass == {}
    assert qrf_tail == {}


def test_register_loader_rejects_missing_reasons_and_bad_schema(tmp_path) -> None:
    bad_reason = tmp_path / "register.json"
    bad_reason.write_text(
        json.dumps(
            {
                "schema_version": 2,
                "description": "Seeded register.",
                "exclusions": {"person.x": _entry("")},
            }
        )
    )
    legacy_schema = tmp_path / "schema.json"
    legacy_schema.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "description": "Seeded register.",
                "exclusions": {},
            }
        )
    )
    bad_expiry = tmp_path / "expiry.json"
    bad_expiry.write_text(
        json.dumps(
            {
                "schema_version": 2,
                "description": "Seeded register.",
                "exclusions": {
                    "person.x": _entry(
                        "Seeded reversed dates.", expires_on="2026-08-09"
                    )
                },
            }
        )
    )
    bad_date = tmp_path / "date.json"
    bad_date.write_text(
        json.dumps(
            {
                "schema_version": 2,
                "description": "Seeded register.",
                "exclusions": {
                    "person.x": _entry("Seeded bad date.", expires_on="soon")
                },
            }
        )
    )

    with pytest.raises(ValueError, match="reason must be a non-empty string"):
        load_uk_reviewed_exclusion_register(
            bad_reason,
            resource=UK_INPUT_MASS_EXCLUSION_REGISTER_RESOURCE,
        )
    with pytest.raises(ValueError, match="schema_version must be 2"):
        load_uk_reviewed_exclusion_register(
            legacy_schema,
            resource=UK_INPUT_MASS_EXCLUSION_REGISTER_RESOURCE,
        )
    with pytest.raises(ValueError, match="expires_on must be after approved_on"):
        load_uk_reviewed_exclusion_register(
            bad_expiry,
            resource=UK_INPUT_MASS_EXCLUSION_REGISTER_RESOURCE,
        )
    with pytest.raises(ValueError, match="must be an ISO date"):
        load_uk_reviewed_exclusion_register(
            bad_date,
            resource=UK_INPUT_MASS_EXCLUSION_REGISTER_RESOURCE,
        )


@pytest.mark.parametrize(
    ("raw", "match"),
    [
        (
            '{"schema_version":2,"description":"x","exclusions":'
            '{"person.x":"first","person.x":null}}',
            "duplicate JSON key",
        ),
        (
            '{"schema_version":2,"description":"x","exclusions":{"person.x":null}}',
            "must be an object with fields",
        ),
        (
            '{"schema_version":2,"description":"x",'
            '"exclusions":{"person.x":{"ticket":"610"}}}',
            "fields must be exactly",
        ),
        (
            '{"schema_version":2,"description":"x","exclusions":'
            '{"person.x":"a plain schema-1 reason"}}',
            "must be an object with fields",
        ),
        (
            '{"schema_version":2,"description":"x","exclusions":{"person.x":7}}',
            "must be an object with fields",
        ),
        (
            '{"schema_version":2,"description":"x","exclusions":[[1,"reason"]]}',
            "'exclusions' object",
        ),
        ("null", "JSON object"),
        ('{"schema_version":2', "malformed JSON"),
    ],
)
def test_register_loader_rejects_malformed_or_coerced_entries(
    tmp_path,
    raw: str,
    match: str,
) -> None:
    path = tmp_path / "register.json"
    path.write_text(raw, encoding="utf-8")

    with pytest.raises((TypeError, ValueError), match=match):
        load_uk_reviewed_exclusion_register(
            path,
            resource=UK_INPUT_MASS_EXCLUSION_REGISTER_RESOURCE,
        )


def test_uk_totals_handle_shuffled_ids_benunits_and_entity_name_collisions() -> None:
    dataset = SimpleNamespace(
        person=pd.DataFrame(
            {
                "person_id": [3, 1, 2],
                "person_household_id": [20, 10, 20],
                "person_benunit_id": [200, 100, 200],
                "shared": [11.0, 5.0, 7.0],
            }
        ),
        benunit=pd.DataFrame(
            {
                "benunit_id": [200, 100],
                "shared": [13.0, 17.0],
            }
        ),
        household=pd.DataFrame(
            {
                "household_id": [20, 10],
                "household_weight": [3.0, 2.0],
                "shared": [19.0, 23.0],
            }
        ),
    )

    totals = uk_dataset_input_mass_totals(dataset)

    assert totals["person.shared"] == 64.0
    assert totals["benunit.shared"] == 73.0
    assert totals["household.shared"] == 103.0


def test_uk_totals_match_the_shared_frame_helper_on_equivalent_data() -> None:
    """The UK table-layout helper must not reinvent the US numeric semantics.

    Same records, same weights, one expressed as a microcosm Frame (the US
    path) and one as UK national tables: per-column weighted totals must be
    identical, differing only in the ``entity.`` namespace the UK layout
    needs because its tables do not enforce globally unique column names.
    """

    from microcosm.build.input_mass import input_mass_totals
    from microcosm.frame import EntitySchema, Frame, WeightKind, Weights

    person = pd.DataFrame(
        {
            "person_id": [101, 102, 103],
            "person_household_id": [1, 1, 2],
            "employment_income": [30_000.0, np.nan, 12_000.0],
            "is_disabled": pd.array([True, False, pd.NA], dtype="boolean"),
            "occupation": ["a", "b", "c"],  # strings are skipped on both paths
        }
    )
    household = pd.DataFrame({"household_id": [1, 2], "council_tax": [900.0, 1_500.0]})
    weights = np.array([2.0, 5.0])

    frame_totals = input_mass_totals(
        Frame(
            {"person": person, "household": household},
            EntitySchema(group_entities=("household",)),
            {"household": Weights(weights, WeightKind.DESIGN)},
        )
    )
    uk_totals = uk_dataset_input_mass_totals(
        SimpleNamespace(
            person=person.assign(person_benunit_id=[201, 201, 202]),
            benunit=pd.DataFrame({"benunit_id": [201, 202]}),
            household=household.assign(household_weight=weights),
        )
    )

    assert frame_totals == {
        "employment_income": uk_totals["person.employment_income"],
        "is_disabled": uk_totals["person.is_disabled"],
        "council_tax": uk_totals["household.council_tax"],
    }
    # NaN fills to 0, booleans total weighted True mass, weights broadcast
    # through membership — asserted against hand computation once, so both
    # paths are pinned to the same semantics rather than merely to each other.
    assert frame_totals["employment_income"] == 30_000.0 * 2.0 + 12_000.0 * 5.0
    assert frame_totals["is_disabled"] == 2.0
    assert "occupation" not in frame_totals
    assert "person.occupation" not in uk_totals


def test_uk_input_mass_gate_is_the_shared_gate_plus_recorded_identity() -> None:
    """Without exclusions the UK wrapper must reproduce the US gate verbatim.

    Same failure lines, same verdict, same details — the UK result may only
    add the reference identity and the (empty) stale/dormant register fields
    the #609 discipline requires on top of the shared gate.
    """

    from microcosm.build.gates import input_mass_parity_gate

    candidate = {
        "person.employment_income": 0.0,  # the #278 signature
        "person.pension_income": 4.0,  # -60% drift
        "person.new_layer": 7.0,  # candidate-only
    }
    reference = _reference(
        {
            "person.employment_income": 10.0,
            "person.pension_income": 10.0,
            "person.tiny": 0.5,  # below the floor
        }
    )
    policy = _policy(relative_tolerance=0.5, minimum_reference_total=1.0)

    shared = input_mass_parity_gate(
        candidate,
        reference.totals,
        candidate_name="uk_release_candidate",
        reference_name=reference.filename,
        relative_tolerance=policy.relative_tolerance,
        minimum_reference_total=policy.minimum_reference_total,
    )
    ported = _synthetic_input_mass_gate(candidate, reference, policy=policy)

    assert ported.name == shared.name == "input_mass_parity"
    assert ported.passed == shared.passed
    assert ported.failures == shared.failures
    shared_details = dict(shared.details)
    assert {
        key: value for key, value in ported.details.items() if key in shared_details
    } == shared_details
    assert set(ported.details) - set(shared_details) == {
        "stale_exclusions",
        "dormant_exclusions",
        "expired_exclusions",
        "exclusions_evaluated_on",
        "reference_identity",
    }


def test_uk_tail_gate_is_the_shared_gate_under_the_uk_name() -> None:
    """Exclusion discipline included: the shared US gate already carries it."""

    from microcosm.build.gates import tail_concentration_gate

    n = 12
    concentrated = np.ones(n)
    concentrated[0] = 500.0
    values = {"self_employment_income": concentrated, "dividend_income": np.ones(n)}
    weights = {name: np.ones(n) for name in values}
    reason = "Seeded stale entry."

    shared = tail_concentration_gate(
        values,
        weights,
        top_k=1,
        max_top_share=0.5,
        min_nonzero_records=2,
        reviewed_exclusions={"dividend_income": reason},
    )
    ported = uk_qrf_tail_concentration_gate(
        values,
        weights,
        policy=UKQRFTailConcentrationPolicy(
            top_k=1,
            max_top_share=0.5,
            min_nonzero_records=2,
            reviewed_exclusions={"dividend_income": _entry(reason)},
        ),
    )

    assert shared.name == "tail_concentration"
    assert ported.name == "qrf_tail_concentration"
    assert ported.passed == shared.passed
    assert ported.failures == shared.failures
    shared_details = dict(shared.details)
    assert {
        key: value for key, value in ported.details.items() if key in shared_details
    } == shared_details
    assert set(ported.details) - set(shared_details) == {
        "expired_exclusions",
        "exclusions_evaluated_on",
    }


def test_input_mass_reference_round_trips_the_measurement_schema(tmp_path) -> None:
    path = tmp_path / "reference.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "identity": {
                    "filename": "enhanced_frs_2023_24.h5",
                    "revision": "655dd07e4bb9c777b00dac044949611f1feb824f",
                    "sha256": (
                        "584ae33d80ca0431254610a3f8254d132da73477d31966d6446282861ecae50d"
                    ),
                    "vintage": "2023_24",
                },
                "totals": {"person.employment_income": 10.5},
            }
        )
    )

    # The licensed 131-column totals are intentionally unavailable to CI;
    # bypass only the reviewed digest while checking the measurement schema.
    with patch.object(
        weighted_integrity,
        "_validate_input_mass_reference",
        return_value=None,
    ):
        reference = load_uk_input_mass_reference(path)

    assert reference.filename == "enhanced_frs_2023_24.h5"
    assert dict(reference.totals) == {"person.employment_income": 10.5}
    with pytest.raises(ValueError, match="schema_version"):
        path.write_text(json.dumps({"schema_version": 9}))
        load_uk_input_mass_reference(path)


def test_expired_exclusion_stops_suppressing_and_names_its_receipt() -> None:
    """Honored through expires_on; strictly after it, renew-or-remove fails."""

    from datetime import date

    reason = "Seeded reviewed loss for the fixture."
    reference = _reference({"person.employment_income": 10.0})
    policy = _policy(
        minimum_reference_total=1.0,
        reviewed_exclusions={"person.employment_income": _entry(reason)},
    )
    candidate = {"person.employment_income": 0.0}

    honored = _synthetic_input_mass_gate(
        candidate, reference, policy=policy, now=date(2027, 2, 10)
    )
    expired = _synthetic_input_mass_gate(
        candidate, reference, policy=policy, now=date(2027, 2, 11)
    )

    assert honored.passed
    assert honored.details["expired_exclusions"] == []
    assert honored.details["exclusions_evaluated_on"] == "2027-02-10"
    assert not expired.passed
    assert expired.details["expired_exclusions"] == ["person.employment_income"]
    assert any(
        "renew the adjudication or remove the entries" in failure
        and "test-reviewer" in failure
        and "microcosm#610" in failure
        for failure in expired.failures
    )
