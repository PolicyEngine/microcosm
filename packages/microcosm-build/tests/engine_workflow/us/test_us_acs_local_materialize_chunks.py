"""The ACS local release's materialize step gives one target matrix at any chunk size.

``materialize_chunked`` (``tools/build_us_acs_local_release.py``) runs
policyengine-us over contiguous household chunks (``--hh-chunk``), each in
engine batches (``--batch``), and assembles the targets-by-households CSR
float32 matrix of the targets it compiles. This test runs it with the real
engine on six households, unpatched: the input projection, the reviewed-null
fill, the population-aggregate refusal that several chunks turn on, and chunk
0's direct check of one district row per carrier. Every layout must give the
one-chunk, one-batch run's compiled target names and matrix bit for bit. So,
on these six households and layouts, no formula these targets reach changed a
stored value by reading another household; the scan of the installed engine's
sources in ``test_us_batched_target_materialization.py`` is the check over the
whole engine.

The targets are the local build's default families: SNAP amounts and
household counts, Medicaid enrollment, and IRS SOI state rows with district
rows split from geography-free carriers under state parents.
``engine_free/us/test_us_acs_local_sparse_cd.py`` checks the same comparison
over more layouts with a fake engine.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from microcosm.frame import US_SCHEMA, Frame, WeightKind, Weights
from test_support.microcosm_build.us_acs_local_reviewed_nulls import load_tool_module
from test_support.microcosm_build.us_batched_target_materialization import (
    _soi,
    _spec,
)
from test_support.microcosm_build.us_local_materialize_chunks import (
    assert_same_surface,
)

#: (household_id, state_fips, congressional_district_geoid). District 602
#: has two households, every other district one.
_HOUSEHOLDS = (
    (1, 6, 601),
    (2, 6, 602),
    (3, 36, 3601),
    (4, 6, 602),
    (5, 24, 2401),
    (6, 36, 3602),
)
#: (household_id, tax_unit_id, marital_unit_id, age,
#: employment_income_before_lsr, taxable_interest_income). Married couples
#: share a marital unit; NaN is a registered null the fill sets to 0.
_PERSONS = (
    (1, 10, 101, 34.0, 18_000.0, 0.0),
    (1, 10, 102, 6.0, 0.0, 0.0),
    (1, 10, 103, 3.0, 0.0, 0.0),
    (2, 20, 201, 45.0, 210_000.0, 4_000.0),
    (2, 20, 201, 43.0, 95_000.0, np.nan),
    (3, 30, 301, 71.0, 0.0, 12_000.0),
    (3, 31, 302, 24.0, 9_000.0, 0.0),
    (4, 40, 401, 29.0, 0.0, 0.0),
    (4, 40, 402, 1.0, 0.0, 0.0),
    (5, 50, 501, 52.0, 64_000.0, 300.0),
    (5, 50, 501, 50.0, 31_000.0, 0.0),
    (5, 50, 502, 15.0, 2_000.0, 0.0),
    (6, 60, 601, 27.0, 26_000.0, 0.0),
    (6, 60, 601, 26.0, 12_000.0, 0.0),
    (6, 60, 602, 2.0, 0.0, 0.0),
)
_DISTRICTS = (
    ("06", "601"),
    ("06", "602"),
    ("36", "3601"),
    ("36", "3602"),
    ("24", "2401"),
)

#: (hh_chunk, batch, chunk sizes, households of each engine batch, the
#: district of the rows chunk 0 materializes directly). Chunk 0 checks the
#: district with the most households in it, the first declared on a tie.
_REFERENCE_LAYOUT = (6, 6, (6,), (6,), "602")
_LAYOUTS = (
    # Two equal chunks, each in batches of 2 and 1.
    (3, 2, (3, 3), (2, 1, 2, 1), "601"),
    # An uneven last chunk.
    (4, 4, (4, 2), (4, 2), "602"),
    # One household per chunk.
    (1, 1, (1,) * 6, (1,) * 6, "601"),
)


def _frame() -> Frame:
    """Six households in three states and five districts; every group nests
    in its household, and one household holds two tax units."""

    persons = pd.DataFrame(
        {
            "person_id": np.arange(1, len(_PERSONS) + 1, dtype="int64"),
            "person_household_id": np.asarray([row[0] for row in _PERSONS], "int64"),
            "person_tax_unit_id": np.asarray([row[1] for row in _PERSONS], "int64"),
            "person_spm_unit_id": np.asarray(
                [row[0] * 100 for row in _PERSONS], "int64"
            ),
            "person_family_id": np.asarray(
                [row[0] * 1000 for row in _PERSONS], "int64"
            ),
            "person_marital_unit_id": np.asarray([row[2] for row in _PERSONS], "int64"),
            "age": np.asarray([row[3] for row in _PERSONS]),
            "employment_income_before_lsr": np.asarray([row[4] for row in _PERSONS]),
            "taxable_interest_income": np.asarray([row[5] for row in _PERSONS]),
        }
    )
    tables = {"person": persons}
    for entity in ("tax_unit", "spm_unit", "family", "marital_unit"):
        ids = np.unique(persons[f"person_{entity}_id"].to_numpy())
        tables[entity] = pd.DataFrame({f"{entity}_id": ids})
    tables["household"] = pd.DataFrame(
        {
            "household_id": np.asarray([row[0] for row in _HOUSEHOLDS], "int64"),
            "state_fips": np.asarray([row[1] for row in _HOUSEHOLDS], "int64"),
            "congressional_district_geoid": np.asarray(
                [row[2] for row in _HOUSEHOLDS], "int64"
            ),
        }
    )
    return Frame(
        tables,
        US_SCHEMA,
        {"household": Weights(np.arange(1.0, len(_HOUSEHOLDS) + 1), WeightKind.DESIGN)},
    )


def _admin_variable(name, family, base_variable, **metadata):
    return _spec(
        name,
        family,
        {
            "materializer": "policyengine_variable",
            "base_variable": base_variable,
            **metadata,
        },
    )


def _specs() -> tuple:
    specs = [
        _admin_variable(
            "snap_ca", "usda_snap", "snap", measure_mode="sum", state_fips="06"
        ),
        _admin_variable(
            "snap_households_ny",
            "usda_snap",
            "snap",
            measure_mode="indicator_sum",
            indicator_map_to="household",
            state_fips="36",
        ),
        _admin_variable(
            "medicaid_enrollment_ca",
            "cms_medicaid",
            "medicaid_enrolled",
            measure_mode="indicator_sum",
            state_fips="06",
            target_role="medicaid_enrollment",
        ),
    ]
    for state in ("06", "36", "24"):
        specs += [
            _soi(f"state_{state}_agi", "adjusted_gross_income", state_fips=state),
            _soi(f"state_{state}_returns", "count", state_fips=state),
        ]
    for state, district in _DISTRICTS:
        located = {"state_fips": state, "congressional_district_geoid": district}
        specs += [
            _soi(
                f"cd_{district}_agi",
                "adjusted_gross_income",
                state_cd_parent_target_name=f"state_{state}_agi",
                state_cd_vintage_rule="fixture",
                **located,
            ),
            _soi(
                f"cd_{district}_returns",
                "count",
                state_cd_parent_target_name=f"state_{state}_returns",
                state_cd_vintage_rule="fixture",
                **located,
            ),
            _soi(
                f"cd_{district}_agi_under_50k_returns",
                "count",
                agi_lower_bound="1",
                agi_upper_bound="50000",
                **located,
            ),
            _soi(f"cd_{district}_eitc", "eitc", **located),
            _soi(
                f"cd_{district}_joint_wages",
                "employment_income",
                filing_status="Married Filing Jointly/Surviving Spouse",
                **located,
            ),
        ]
    return tuple(specs)


def _summary(tmp_path):
    path = tmp_path / "staging.summary.json"
    path.write_text(
        json.dumps(
            {
                "reviewed_engine_input_nulls": [
                    {
                        "entity": "person",
                        "column": "taxable_interest_income",
                        "missing_rows": 1,
                        "rows": len(_PERSONS),
                    }
                ]
            }
        )
    )
    return path


def _directly_checked_districts(materialized) -> set[str]:
    """Districts of the rows chunk 0 materialized directly, one per carrier."""

    return {
        entry["target"].split("_")[1] for entry in materialized.carrier_check["checked"]
    }


def test_every_chunk_and_batch_size_gives_the_one_chunk_target_matrix(
    monkeypatch, tmp_path
) -> None:
    module = load_tool_module()
    import build_us_fiscal_refresh_release as release

    summary = _summary(tmp_path)
    # Each engine batch builds one dataset; record its households, nothing else.
    batch_households: list[int] = []
    own_dataset_from_frame = release._dataset_from_frame

    def recording_dataset_from_frame(frame, **options):
        batch_households.append(frame.n("household"))
        return own_dataset_from_frame(frame, **options)

    monkeypatch.setattr(release, "_dataset_from_frame", recording_dataset_from_frame)

    def materialize(hh_chunk, batch, chunk_sizes, batches, checked_district, label):
        batch_households.clear()
        materialized = module.materialize_chunked(
            _frame(), _specs(), hh_chunk=hh_chunk, batch=batch, summary_path=summary
        )
        assert [stat["households"] for stat in materialized.chunk_stats] == list(
            chunk_sizes
        ), label
        assert batch_households == list(batches), label
        assert _directly_checked_districts(materialized) == {checked_district}, label
        assert materialized.carrier_check["all_equal"] is True, label
        # Three state AGI and three state return-count parents, every chunk.
        assert materialized.carrier_check["parent_blocks_checked"] == 6 * len(
            chunk_sizes
        ), label
        return materialized

    reference = materialize(*_REFERENCE_LAYOUT, label="reference")
    # Every declared target compiled, in declared order.
    assert [spec.name for spec in reference.compiled_specs] == [
        spec.name for spec in _specs()
    ]
    # The fixture reaches real engine output in each family.
    dense = dict(zip(reference.names, reference.matrix.toarray(), strict=True))
    for name in (
        "snap_ca",
        "snap_households_ny",
        "medicaid_enrollment_ca",
        "state_06_agi",
        "state_36_returns",
        "cd_601_eitc",
        "cd_602_joint_wages",
        "cd_3601_agi_under_50k_returns",
        "cd_3602_eitc",
    ):
        assert dense[name].any(), name
    for _state, district in _DISTRICTS:
        assert dense[f"cd_{district}_agi"].any(), district
    # Some stored values are not whole numbers (the EITC of household 6, in
    # the credit's phase-out range), so float32 rounding is compared too.
    assert np.any(reference.matrix.data % 1 != 0)
    assert reference.carrier_check["rows_without_parent_check"] == 3 * len(_DISTRICTS)

    save = module.cd_surface.save_target_matrix
    reference_sha256 = save(tmp_path / "reference.npz", reference.matrix)
    for layout in _LAYOUTS:
        hh_chunk, batch = layout[:2]
        label = f"hh_chunk={hh_chunk}, batch={batch}"
        materialized = materialize(*layout, label=label)
        assert_same_surface(reference, materialized, label=label)
        assert save(tmp_path / f"{hh_chunk}-{batch}.npz", materialized.matrix) == (
            reference_sha256
        ), label
