"""Add the "rebase onto a one-sided Table 1.4 control" arm (#1228 option 2).

The capital-gains fix (#1036) keeps the model's one-sided concept and binds
each HT2 state row as its share of a national Table 1.4 control. This builds
the same for business and rental income, from ``prepare.py``'s outputs, so
the option can be measured before Chronicle carries the legs
(PolicyEngine/chronicle#346):

* **Business.** The model concept is the positive part of
  ``total_self_employment_income``, counted when positive. The control is
  Table 1.4 TY2022 "Business or profession: Net income": 21,969,832 returns
  and $543,564,980 thousand.
* **Rental.** The model concept is the positive part of ``rental_income +
  farm_rent_income``, counted when positive. The control is Table 1.4 TY2022
  "Total rental and royalty: Net income", which includes farm rental:
  6,380,131 returns and $148,879,455 thousand.
* **Partnership and S corporation.** Table 1.4 has had no combined net-income
  column since TY2020, so there is no control. These rows keep the
  concept-exact binding.

A state row's value is its HT2 value / the HT2 US value × the control; the US
row's value is the control. Amounts are aged by the rows' own factor,
1.12034597479376, and counts are not aged, as in the release.

Writes ``rebase_rows.npz`` (the 312 rows on this arm's concepts, CSR, with
their values) and ``rebase_meta.parquet`` (each target's value and loss weight
on this arm, with the weights recomputed by ``target_loss_weights``).

Run: ``uv run --no-sync python experiments/1228-net-income-less-loss/prepare_rebase.py --inputs DIR``
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
from prepare import DIAGNOSTICS, concept_stem, measure_id
from scipy import sparse

AGING = 1.12034597479376
CONTROLS = {
    "schedule_c_income_returns": ("schedule_c_total", "pos_count", 21_969_832.0),
    "schedule_c_income_amount": (
        "schedule_c_total",
        "pos_sum",
        543_564_980e3 * AGING,
    ),
    "rental_royalty_income_returns": (
        "rental_plus_farm_rent",
        "pos_count",
        6_380_131.0,
    ),
    "rental_royalty_income_amount": (
        "rental_plus_farm_rent",
        "pos_sum",
        148_879_455e3 * AGING,
    ),
}
#: Model concept ids for the loss-weight stand-in specs.
ONE_SIDED = {
    "schedule_c_income": "business_net_profits",
    "rental_royalty_income": "rent_and_royalty_net_income",
    "partnership_scorp_income": "partnership_and_s_corp_net_income_less_loss",
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", type=Path, required=True)
    args = parser.parse_args()
    inputs: Path = args.inputs

    entries = json.loads(DIAGNOSTICS.read_text())["targets"]
    meta = pd.read_parquet(inputs / "targets_meta.parquet")
    totals = dict(np.load(inputs / "concept_totals.npz"))
    state = totals["state_fips"]
    with np.load(inputs / "concept_rows.npz") as payload:
        concept_rows = payload["rows"]
        exact = sparse.csr_array(
            (payload["data"], payload["indices"], payload["indptr"]),
            shape=tuple(payload["shape"]),
        )
    values = meta["value"].to_numpy().copy()
    us_value = {
        measure: float(
            meta.loc[
                (meta["measure_id"] == measure)
                & (meta["geography_level"] == "country"),
                "value",
            ].iloc[0]
        )
        for measure in CONTROLS
    }
    rows = []
    for position, row in enumerate(concept_rows):
        entry = entries[int(row)]
        measure = measure_id(entry)
        if measure not in CONTROLS:
            rows.append(exact[[position]].toarray()[0])
            continue
        concept, part, control = CONTROLS[measure]
        column = totals[f"{concept}__{part}"].astype(np.float64)
        state_fips = entry["metadata"].get("state_fips")
        if state_fips:
            column = np.where(state == int(state_fips), column, 0.0)
            values[int(row)] = (
                float(entry["compiled_target"]) / us_value[measure] * control
            )
        else:
            values[int(row)] = control
        rows.append(column)
    rebase = sparse.csr_array(np.vstack(rows))

    from microcosm.build.us_runtime.target_loss_weights import target_loss_weights

    specs = []
    for row, entry in enumerate(entries):
        metadata = dict(entry["metadata"])
        stem = concept_stem(entry)
        if stem is not None:
            metadata["source_variable"] = ONE_SIDED[stem]
            metadata["variable"] = ONE_SIDED[stem]
        specs.append(
            SimpleNamespace(
                name=entry["target_name"],
                value=float(values[row]),
                family=entry["registry"]["family"],
                entity=entry["entity"],
                period=entry["period"],
                filter=entry["filter"],
                metadata=metadata,
            )
        )
    weights = target_loss_weights(specs)
    pd.DataFrame(
        {
            "row": np.arange(len(entries)),
            "value_rebase_positive_part": values,
            "loss_weight_rebase_positive_part": weights,
        }
    ).to_parquet(inputs / "rebase_meta.parquet", index=False)
    np.savez_compressed(
        inputs / "rebase_rows.npz",
        rows=np.asarray(concept_rows, dtype=np.int64),
        data=rebase.data,
        indices=rebase.indices,
        indptr=rebase.indptr,
        shape=np.asarray(rebase.shape),
    )
    manifest_path = inputs / "MANIFEST.json"
    manifest = json.loads(manifest_path.read_text())
    for name in ("rebase_rows.npz", "rebase_meta.parquet"):
        manifest["outputs"][name] = hashlib.sha256(
            (inputs / name).read_bytes()
        ).hexdigest()
    manifest["rebase_positive_part"] = {
        "controls": {measure: control for measure, (_, _, control) in CONTROLS.items()},
        "us_values": us_value,
        "changed_values": int(np.sum(values != meta["value"].to_numpy())),
    }
    manifest_path.write_text(json.dumps(manifest, indent=1, sort_keys=True))
    print(json.dumps(manifest["rebase_positive_part"], indent=1))


if __name__ == "__main__":
    main()
