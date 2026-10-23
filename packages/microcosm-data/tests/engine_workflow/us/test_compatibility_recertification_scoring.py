"""The recertification scorer under the installed policyengine-us.

On a three-household H5 in two states, scored with real engines:

- every household-batched total is the same at batch sizes 1 and 1,000, so
  the scored metric set is additive over households for this fixture (a
  check of these paths, not a proof for every household);
- the stated-by-derived filing-status table and the role comparison count
  every tax unit and person exactly once;
- the counterfactual withholds exactly the newly read columns.

The wrapper package is not in this environment and no wheels are at hand,
so the scorer's runtime proof is stubbed; the real loader replay and wheel
proof run in the qualification environments.
"""

from __future__ import annotations

import json

import pandas as pd
import pytest

from microcosm.data import compatibility_recertification as recert
from test_support.microcosm_data.compatibility_recertification import (
    CRITICAL_METADATA,
    FIXTURE_PRODUCER,
    load_tool,
)


def _write_h5(path):
    person = pd.DataFrame(
        {
            "person_id": [1, 2, 3, 4, 5, 6],
            "person_household_id": [1, 1, 1, 2, 3, 3],
            "person_tax_unit_id": [1, 1, 1, 2, 3, 4],
            "person_spm_unit_id": [1, 1, 1, 2, 3, 3],
            "person_family_id": [1, 1, 1, 2, 3, 3],
            "person_marital_unit_id": [1, 1, 2, 3, 4, 5],
            "age": [40.0, 38.0, 6.0, 70.0, 30.0, 25.0],
            "employment_income": [30_000.0, 12_000.0, 0.0, 0.0, 60_000.0, 8_000.0],
            "social_security": [0.0, 0.0, 0.0, 22_000.0, 0.0, 0.0],
            "tax_unit_role_input": [
                "HEAD",
                "SPOUSE",
                "DEPENDENT",
                "HEAD",
                "HEAD",
                "HEAD",
            ],
        }
    )
    tables = {
        "person": person,
        "household": pd.DataFrame(
            {
                "household_id": [1, 2, 3],
                "household_weight": [100.0, 50.0, 80.0],
                "state_fips": [6, 36, 6],
                "county_fips": [6037, 36061, 6037],
            }
        ),
        "tax_unit": pd.DataFrame(
            {
                "tax_unit_id": [1, 2, 3, 4],
                "filing_status_input": ["JOINT", "SINGLE", "SINGLE", "SINGLE"],
            }
        ),
        "spm_unit": pd.DataFrame({"spm_unit_id": [1, 2, 3]}),
        "family": pd.DataFrame({"family_id": [1, 2, 3]}),
        "marital_unit": pd.DataFrame({"marital_unit_id": [1, 2, 3, 4, 5]}),
    }
    with pd.HDFStore(path, "w") as store:
        store.put("_time_period", pd.Series([2024]), format="table")
        for key, frame in tables.items():
            store.put(key, frame, format="table", data_columns=True)


@pytest.fixture
def scorable(tmp_path, monkeypatch):
    # The scorer compiles the engine into an isolated cache; this module runs
    # in its own process, so isolate before anything imports the engine.
    recert.isolate_bytecode()
    artifact_root = tmp_path / "artifacts"
    artifact_root.mkdir()
    h5 = artifact_root / "populace_us_2024.h5"
    _write_h5(h5)
    release_dir = tmp_path / "releases" / "populace-us-2024-scoring-fixture"
    release_dir.mkdir(parents=True)
    dataset = {
        "artifact": "populace_us_2024",
        "filename": h5.name,
        "sha256": recert.sha256_file(h5),
    }
    (release_dir / recert.REPORT_FILE).write_text(json.dumps({"dataset": dataset}))
    targets = [
        {
            "name": name,
            "metadata": metadata,
            "registry": {"family": name.split(".", 1)[0]},
            "target": 1.0,
            "final_estimate": 1.0,
        }
        for name, metadata in CRITICAL_METADATA.items()
    ]
    for target in targets:
        if target["name"].startswith(
            "irs_soi.ty2022.historic_table_2.us.all.income_tax"
        ):
            target["metadata"]["target_role"] = "federal_income_tax_total"
    (release_dir / "calibration_diagnostics.json").write_text(
        json.dumps(
            {
                "targets": targets,
                "build": {"post_export_scoring": {"spm": {"geography_kind": "county"}}},
            }
        )
    )
    tool = load_tool()
    monkeypatch.setattr(tool, "_producer_identity", lambda: dict(FIXTURE_PRODUCER))
    monkeypatch.setattr(
        tool,
        "scoring_runtime",
        lambda wheels: {
            name: {
                "version": "0",
                "files": 0,
                "content_sha256": "0" * 64,
                "wheel_sha256": "0" * 64,
                "pypi_sha256_verified": True,
            }
            for name in recert.COMPATIBILITY_PACKAGES
        },
    )
    return tool, release_dir, artifact_root


def test_household_batched_totals_do_not_depend_on_the_batch_size(scorable, tmp_path):
    tool, release_dir, artifact_root = scorable
    documents = [
        tool.score_candidate(
            release_dir=release_dir,
            artifact_root=artifact_root,
            output=tmp_path / f"scores-{size}.json",
            variant=recert.VARIANT_AS_STORED,
            compatibility_wheels=(),
            batch_households=size,
        )
        for size in (1, 1_000)
    ]
    one, whole = (document["metrics"] for document in documents)
    assert set(one) == set(whole)
    for metric_id in one:
        assert one[metric_id]["value"] == pytest.approx(
            whole[metric_id]["value"], rel=1e-12, abs=1e-6
        ), metric_id
    # The fixture reaches real engine output.
    assert whole["income_tax_net"]["value"] != 0
    assert whole["social_security_total"]["value"] == pytest.approx(50 * 22_000.0)
    statuses = documents[1]["filing_status"]
    assert (
        sum(cell["tax_units"] for row in statuses.values() for cell in row.values())
        == 4
    )
    assert documents[1]["tax_unit_roles"]["persons"] == 6
    assert documents[0]["scoring"]["passes"][0]["batches"] == 3
    assert [item["households"] for item in documents[0]["scoring"]["passes"]] == [3, 3]


def test_the_counterfactual_withholds_exactly_the_newly_read_columns(
    scorable, tmp_path
):
    from microcosm.data import stored_inputs

    tool, release_dir, artifact_root = scorable
    tables = stored_inputs.h5_stored_tables(artifact_root / "populace_us_2024.h5")
    changes = recert.stored_column_changes(
        tables,
        engine_variables=stored_inputs.installed_us_engine().variables,
        stored_input_register=stored_inputs.US_STORED_NON_VARIABLE_COLUMNS,
    )
    read = sorted(item["column"] for item in changes[recert.CHANGE_READ])
    if not read:
        with pytest.raises(ValueError, match="newly reads no stored column"):
            tool.score_candidate(
                release_dir=release_dir,
                artifact_root=artifact_root,
                output=tmp_path / "counterfactual.json",
                variant=recert.VARIANT_WITHOUT_NEWLY_READ,
                compatibility_wheels=(),
            )
        return
    document = tool.score_candidate(
        release_dir=release_dir,
        artifact_root=artifact_root,
        output=tmp_path / "counterfactual.json",
        variant=recert.VARIANT_WITHOUT_NEWLY_READ,
        compatibility_wheels=(),
    )
    assert document["variant"] == {
        "id": recert.VARIANT_WITHOUT_NEWLY_READ,
        "withheld_columns": read,
    }
