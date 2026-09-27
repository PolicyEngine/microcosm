"""Measure every byte transport this lane censuses, through each module's own encoder.

The row-count lane established the full-source counts from the recovered
1/1000 artifact's catalogues (1,531,614 selectable ACS households, 3,422,888
ACS persons, 55,762 ASEC households, 142,125 ASEC persons, 1,587,376 stacked
households, 3,565,013 stacked persons, 3,174,752 clone households). This
measures the *bytes* each transport meets at those counts:

* the ACS archive is streamed in full -- every person record and every
  household record of the pilot's captured public ACS PUMS archive -- so the
  coverage body, the pre-allocation charge, the selected-key list and the
  serialno list are sums over the real rows, not samples scaled up;
* each JSON roster document is encoded through its own module's encoder on
  rows shaped at full-source id widths, and the marginal cost per row is a
  difference between two row counts so fixed preambles cancel;
* each binary artifact is computed from the module's own width constants.

Where a count is a roster ratio scaled from 1/1000 rather than a catalogue
count (tax units per household, eligible children), the receipt says so.

Nothing outside the output path is written. Not a build, not a certification,
not release eligible.

    python byte_census_measure.py <acs-snapshot-dir> <preparation.json> <out.json>
"""

from __future__ import annotations

import csv
import io
import json
import pathlib
import sys
import time
import zipfile
from types import SimpleNamespace

import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path[:0] = [str(path) for path in sorted((ROOT / "packages").glob("*/src"))]

from microcosm.build.us_runtime import acs_person_coverage_authentication as auth
from microcosm.build.us_runtime import acs_person_coverage_columns as literal
from microcosm.build.us_runtime import current_survey_household_roles as roles
from microcosm.build.us_runtime import current_survey_predictors as predictors
from microcosm.build.us_runtime import graph_child_property_income as child_graph
from microcosm.build.us_runtime import graph_combined_clone as clone
from microcosm.build.us_runtime import (
    graph_current_survey_household_roles as roles_graph,
)
from microcosm.build.us_runtime import graph_survey_age_artifact as ages
from microcosm.build.us_runtime import graph_survey_calibration as numeric
from microcosm.build.us_runtime import graph_survey_population as graph
from microcosm.build.us_runtime import puf_detail_transfer as detail
from microcosm.build.us_runtime import survey_origin_budget as budget_owner
from microcosm.fit import graph_joint_empirical as adapter

for _module in (auth, literal, graph, budget_owner, numeric, ages, roles, adapter):
    if not pathlib.Path(_module.__file__).resolve().is_relative_to(ROOT):
        raise SystemExit(f"{_module.__name__} resolved outside {ROOT}")

# Measured full-source counts; see experiments/native-row-ceilings/roster-census.json.
ACS_HOUSEHOLDS = 1_531_614
ACS_PERSONS = 3_422_888
ASEC_HOUSEHOLDS = 55_762
ASEC_PERSONS = 142_125
STACKED_HOUSEHOLDS = 1_587_376
STACKED_PERSONS = 3_565_013
CLONE_HOUSEHOLDS = 3_174_752
CLONE_PERSONS = 7_130_026
MIB = 1024**2


def _members(archive, prefix):
    with zipfile.ZipFile(archive) as zf:
        for info in sorted(zf.infolist(), key=lambda m: m.filename):
            name = info.filename.casefold()
            if name.startswith(prefix) and name.endswith(".csv"):
                yield zf, info


def _rows(stream, first):
    """Raw record bytes and decoded cells, the way the owner decodes them."""
    header = None
    for raw in stream:
        text = raw.decode("utf-8-sig" if header is None and first else "utf-8")
        values = next(csv.reader(io.StringIO(text, newline="")))
        if header is None:
            header = values
            yield raw, header, None
            continue
        yield raw, header, values


def acs_persons(archive, contract):
    fields = contract["fields"]
    out = {
        "records": 0,
        "raw_bytes": 0,
        "raw_max": 0,
        "old_charge_bytes": 0,  # 6 * len(raw) + 1024, the shipped pre-charge
        "body_bytes": 0,  # exact NDJSON body: _json(row) + newline
        "body_row_max": 0,
        "cells_json_bytes": 0,  # _json([*READ_COLUMNS cells, member, ordinal])
        "cells_json_max": 0,
        "key_pair_bytes": 0,  # _json([SERIALNO, SPORDER]) per person, plus commas
        "sporder_two_digit_rows": 0,
        "states": {},
        "members": {},
    }
    started = time.monotonic()
    for zf, info in _members(archive, "psam_pus"):
        positions = None
        ordinal = 0
        with zf.open(info) as stream:
            for raw, header, values in _rows(stream, True):
                if values is None:
                    positions = [header.index(c) for c in literal.READ_COLUMNS]
                    continue
                ordinal += 1
                cells = [values[p] for p in positions]
                serial, sporder, agep, mil, esr = cells
                out["records"] += 1
                out["raw_bytes"] += len(raw)
                out["raw_max"] = max(out["raw_max"], len(raw))
                out["old_charge_bytes"] += 6 * len(raw) + 1024
                states = [
                    literal._field_state(
                        agep,
                        value,
                        minimum_age=fields[name]["minimum_age"],
                        codes=fields[name]["codes"],
                    )
                    for name, value in (("MIL", mil), ("ESR", esr))
                ]
                for state in states:
                    out["states"][state] = out["states"].get(state, 0) + 1
                row = auth._json(
                    [
                        serial,
                        int(sporder),
                        agep,
                        mil,
                        esr,
                        *states,
                        info.filename,
                        ordinal,
                    ],
                    auth.MAX_RECORD_BYTES - 1,
                )
                out["body_bytes"] += len(row) + 1
                out["body_row_max"] = max(out["body_row_max"], len(row) + 1)
                cells_json = len(
                    auth._json(
                        [*cells, info.filename, ordinal], auth.MAX_RECORD_BYTES - 1
                    )
                )
                out["cells_json_bytes"] += cells_json
                out["cells_json_max"] = max(out["cells_json_max"], cells_json)
                out["key_pair_bytes"] += len(auth._json([serial, int(sporder)])) + 1
                if int(sporder) >= 10:
                    out["sporder_two_digit_rows"] += 1
        out["members"][info.filename] = ordinal
    out["key_list_bytes"] = out["key_pair_bytes"] - 1 + 2  # no trailing comma; brackets
    out["seconds"] = round(time.monotonic() - started, 1)
    return out


def acs_households(archive):
    out = {
        "records": 0,
        "occupied_records": 0,
        "raw_bytes": 0,
        "old_charge_bytes_all": 0,
        "old_charge_bytes_occupied": 0,
        "np_total": 0,
        "serialno_widths": {},
        "members": {},
    }
    started = time.monotonic()
    for zf, info in _members(archive, "psam_hus"):
        positions = None
        ordinal = 0
        with zf.open(info) as stream:
            for raw, header, values in _rows(stream, True):
                if values is None:
                    positions = [header.index(c) for c in ("SERIALNO", "NP")]
                    continue
                ordinal += 1
                serial, np_ = (values[p] for p in positions)
                out["records"] += 1
                out["raw_bytes"] += len(raw)
                charge = 6 * len(raw) + 1024
                out["old_charge_bytes_all"] += charge
                width = str(len(serial))
                out["serialno_widths"][width] = out["serialno_widths"].get(width, 0) + 1
                n = int(np_)
                if n > 0:
                    out["occupied_records"] += 1
                    out["old_charge_bytes_occupied"] += charge
                    out["np_total"] += n
        out["members"][info.filename] = ordinal
    # 16 canonical bytes per 13-character key: quotes, 13 chars, separator.
    out["serialno_list_bytes_full_source"] = 16 * out["occupied_records"] + 1
    out["seconds"] = round(time.monotonic() - started, 1)
    return out


def origin_budget():
    from fractions import Fraction

    design, probability, share = Fraction(137, 1), Fraction(1, 1), Fraction(1, 2)
    actual = float(design * share / probability)
    reference = budget_owner._reference(design, probability, share, actual)
    rows = {}
    for magnitude in (STACKED_HOUSEHOLDS, CLONE_HOUSEHOLDS):
        record = {
            "source": "acs",
            "source_year": 2024,
            "survey_year": 2024,
            "raw_native_id": f"2024HU{magnitude % 10_000_000:07d}",
            "selected_receiving_household_id": magnitude,
            "combined_household_id": magnitude,
            "statistical_unit": "occupied_housing_unit",
            "original_design_float64_bytes": "0" * 16,
            **reference,
            "members": [[magnitude, 0], [magnitude * 2, 1]],
            "incoming_clone_float64_bytes": ["0" * 16, "0" * 16],
        }
        encoded = len(budget_owner._json(record))
        ids = len(str(magnitude * 2)) * 2 + 2
        groups = len(str(magnitude)) * 2 + 2
        rows[str(magnitude)] = {
            "record_bytes": encoded,
            "per_group_bytes": encoded + 1 + ids + groups,
        }
    per_group = rows[str(STACKED_HOUSEHOLDS)]["per_group_bytes"]
    return {
        "encoder": "survey_origin_budget._json (graph._bounded_json) per record",
        "rows": rows,
        "per_group_bytes": per_group,
        "full_source_bytes": per_group * STACKED_HOUSEHOLDS,
        "tenth_bytes": per_group * (STACKED_HOUSEHOLDS // 10),
        "households_the_64MiB_cap_admits": budget_owner.MAX_PAYLOAD_BYTES // per_group,
        # The two header lists alone, each one entry per clone household row.
        "household_ids_list_bytes": (len(str(CLONE_HOUSEHOLDS)) + 1) * CLONE_HOUSEHOLDS
        + 1,
        "group_indices_list_bytes": (len(str(STACKED_HOUSEHOLDS)) + 1)
        * CLONE_HOUSEHOLDS
        + 1,
    }


def numeric_bounds():
    def document(groups):
        ids = list(range(CLONE_HOUSEHOLDS - 2 * groups, CLONE_HOUSEHOLDS))
        indices = [STACKED_HOUSEHOLDS - groups + i // 2 for i in range(2 * groups)]
        upper = [float(137 * 2 * (1 + i / 7)).hex() for i in range(groups)]
        row_upper = [float(137 * 8 * (1 + i / 7)).hex() for i in range(2 * groups)]
        incoming = [float(137 * (1 + i / 7)).hex() for i in range(2 * groups)]
        return {
            "protocol": numeric.BOUNDS_PROTOCOL,
            "budget_sha256": "a" * 64,
            "population": clone.COMBINED_CLONE_NODE,
            "household_ids": ids,
            "group_indices": indices,
            "group_upper_hex": upper,
            "row_upper_hex": row_upper,
            "incoming_hex": incoming,
        }

    small, large = 2, 2_002
    bytes_small = len(graph._bounded_json(document(small), numeric.MAX_BYTES))
    bytes_large = len(graph._bounded_json(document(large), numeric.MAX_BYTES))
    per_group = (bytes_large - bytes_small) / (large - small)
    return {
        "encoder": "graph._bounded_json(document, graph_survey_calibration.MAX_BYTES)",
        "bytes_at_groups": {str(small): bytes_small, str(large): bytes_large},
        "per_group_bytes": round(per_group, 2),
        "per_clone_row_bytes": round(per_group / 2, 2),
        "full_source_bytes": int(per_group * STACKED_HOUSEHOLDS),
        "tenth_bytes": int(per_group * (STACKED_HOUSEHOLDS // 10)),
        "clone_rows_the_64MiB_cap_admits": int(numeric.MAX_BYTES // (per_group / 2)),
        "max_rows": numeric.MAX_ROWS,
        "clone_rows_full_source": CLONE_HOUSEHOLDS,
    }


def age_artifact():
    header = ages._json(
        {
            "protocol": "microcosm.us.survey-household-age-counts.v1",
            "population": clone.COMBINED_CLONE_NODE,
            "columns": list(ages._COLUMNS),
            "rows": CLONE_HOUSEHOLDS,
            "people": CLONE_PERSONS,
            "age_convention": "integer_years_completed",
        }
    )
    row = ages._WIDTH * 8
    fixed = len(ages.MAGIC) + 4 + len(header)
    reserved = len(ages.MAGIC) + 4 + ages.MAX_HEADER_BYTES
    return {
        "encoder": "MAGIC + header + values.tobytes(order='C'), values <i8 (rows, 1+bands)",
        "row_bytes": row,
        "header_bytes": len(header),
        "full_source_bytes": fixed + row * CLONE_HOUSEHOLDS,
        "tenth_bytes": fixed + row * (CLONE_HOUSEHOLDS // 10),
        "rows_the_64MiB_cap_admits": (ages.MAX_BYTES - reserved) // row,
        "clone_rows_full_source": CLONE_HOUSEHOLDS,
        "max_people": ages.MAX_PEOPLE,
        "clone_persons_full_source": CLONE_PERSONS,
    }


def roles_projection():
    def table(rows):
        pattern = {
            roles.CANONICAL_COLUMN: [True, False, True],
            roles.SURVEY_COLUMN: ["acs", "asec", "acs"],
            roles.NATIVE_ID_COLUMN: [1, 2, 3],
            roles.CODE_COLUMN: [20, 25, 37],
            roles.CODE_KNOWN_COLUMN: [True, True, False],
            roles.ROLE_STATE_COLUMN: ["observed_reference_person"] * 3,
            roles.UNIVERSE_COLUMN: ["housing_unit"] * 3,
        }
        frame = pd.concat([pd.DataFrame(pattern)] * ((rows + 2) // 3)).head(rows)
        frame.index = pd.Index(
            range(STACKED_PERSONS - rows, STACKED_PERSONS), name="person_id"
        )
        return frame

    small, large = 3, 1_203
    a, b = (
        len(roles._projection_bytes(table(small))),
        len(roles._projection_bytes(table(large))),
    )
    per_row = (b - a) / (large - small)
    return {
        "encoder": 'table.reset_index().to_json(orient="table", index=False).encode()',
        "per_person_bytes": round(per_row, 2),
        "full_source_bytes": int(per_row * STACKED_PERSONS),
        "tenth_bytes": int(per_row * (STACKED_PERSONS // 10)),
        "persons_the_64MiB_cap_admits": int(roles_graph.MAX_ARTIFACT_BYTES // per_row),
        "max_persons_row_bound": roles.MAX_PERSONS,
        "reachability": "49/51-node variants only; the 45-node pilot disabled roles",
    }


def predictor_projection():
    def document(rows):
        origins = [
            [STACKED_PERSONS - i, STACKED_PERSONS - i, "acs" if i % 25 else "asec"]
            for i in range(rows)
        ]
        return {"origins": origins, "protocol": predictors.PROTOCOL}

    small, large = 1, 1_001
    a = len(graph._bounded_json(document(small), 64 * MIB))
    b = len(graph._bounded_json(document(large), 64 * MIB))
    per_row = (b - a) / (large - small)
    columns = predictors.feature_columns(True)
    matrix_row = 8 * (1 + len(columns))
    return {
        "encoder": "graph._bounded_json({**evidence, 'origins': [[person_id, native_person_id, source], ...]}, survey_population_preparation.MAX_PAYLOAD_BYTES)",
        "per_person_origin_row_bytes": round(per_row, 2),
        "full_source_origins_bytes": int(per_row * STACKED_PERSONS),
        "tenth_origins_bytes": int(per_row * (STACKED_PERSONS // 10)),
        "persons_the_64MiB_cap_admits": int(64 * MIB // per_row),
        "matrix": {
            "encoder": "model_input.encode_recipient_matrix over ACS persons",
            "feature_columns_with_demographic_conditioning": list(columns),
            "bytes_per_acs_person": matrix_row,
            "full_source_bytes": matrix_row * ACS_PERSONS,
            "cap": "none in encode_recipient_matrix beyond the header; stored whole by put_bytes",
        },
    }


def diagnostic_matrix(preparation):
    entities = preparation["origins"]["entities"]
    households = len(preparation["origins"]["households"])
    units = entities["tax_unit"]
    ratio = len(units["rows"] if isinstance(units, dict) else units) / households
    row = 8 * (1 + len(detail.FEATURES))
    clone_one = int(round(ratio * STACKED_HOUSEHOLDS))
    return {
        "encoder": f"puf_detail_transfer.recipient_matrix -> encode_recipient_matrix, FEATURES={len(detail.FEATURES)}",
        "bytes_per_tax_unit": row,
        "tax_units_per_household_at_1_1000": round(ratio, 6),
        "clone_one_tax_units_full_source_SCALED_FROM_1_1000": clone_one,
        "full_source_bytes_SCALED": row * clone_one,
        "tenth_bytes_SCALED": row * (clone_one // 10),
        "tax_units_the_64MiB_cap_admits": (64 * MIB) // row,
        "note": "tax units per household is a roster ratio from the 1/1000 artifact, not a catalogue count",
    }


def child_draw():
    key = ("acs", 2024, 2024, "2024HU0001234567", "12", "12")
    n = 1_001
    draw = SimpleNamespace(
        values=np.tile(np.array([[1234.5, 0.0]]), (n, 1)),
        patterns=np.ones(n, dtype=np.int64),
        donor_keys=[("asec", 2024, 2025, "1234567", "12", "1")] * n,
        model_sha256="b" * 64,
    )
    params = {
        "support_sha256": "c" * 64,
        "donor_source_sha256": "d" * 64,
        "source_sha256": "e" * 64,
        "scenario_sha256": "f" * 64,
        "transport_json": '{"kind":"nearest"}',
        "transport_sha256": "0" * 64,
        "stream": ["child-property", 1],
        "coordinate_suffix": ("child-property-v1",),
    }

    def encoded(rows):
        ids = np.arange(STACKED_PERSONS - rows, STACKED_PERSONS)
        keys = [key] * rows
        partial = SimpleNamespace(
            values=draw.values[:rows],
            patterns=draw.patterns[:rows],
            donor_keys=draw.donor_keys[:rows],
            model_sha256=draw.model_sha256,
        )
        return len(adapter._json(adapter._draw_document(partial, ids, keys, params)))

    a, b = encoded(1), encoded(n)
    per_row = (b - a) / (n - 1)
    return {
        "encoder": "graph_child_property_income._json(adapter._draw_document(...)) [producer]; adapter._document(payload, MAX_DRAW_BYTES) [consumer]",
        "per_eligible_child_bytes": round(per_row, 2),
        "eligible_children_full_source_ESTIMATE": None,
        "children_the_64MiB_cap_admits": int(64 * MIB // per_row),
        "max_recipients_fit": adapter.MAX_RECIPIENTS,
        "max_draw_bytes_fit": adapter.MAX_DRAW_BYTES,
        "max_artifact_bytes_graph": child_graph.MAX_ARTIFACT_BYTES,
        "note": "eligible-child count is not in the 1/1000 artifact; a census agent estimates it from ACS/ASEC age shares",
    }


def selection_block(preparation):
    selected = preparation["selection"]["selected"]

    def enc(v):
        return len(
            json.dumps(
                v, sort_keys=True, separators=(",", ":"), ensure_ascii=False
            ).encode()
        )

    per_row = enc(selected) / len(selected)
    return {
        "encoder": "survey_origin_budget._json(view.receipt['selection']) for selection_sha256 (graph._bounded_json, 64 MiB)",
        "per_selected_row_bytes_at_1_1000": round(per_row, 2),
        "full_source_bytes_SCALED": int(per_row * STACKED_HOUSEHOLDS),
        "tenth_bytes_SCALED": int(per_row * (STACKED_HOUSEHOLDS // 10)),
        "rows_the_64MiB_cap_admits": int(64 * MIB // per_row),
        "note": "row widths at 1/1000 (inclusion_probability pairs, ids) differ slightly from full-source widths; scaled, not encoded at full-source widths",
    }


def main() -> int:
    snapshot = pathlib.Path(sys.argv[1])
    preparation = json.loads(pathlib.Path(sys.argv[2]).read_bytes())
    out = pathlib.Path(sys.argv[3])
    contract = literal.coverage_field_contract()
    record = {
        "scope": (
            "Byte measurements for the byte-transport lane, through each module's own "
            "encoder, over the recovered 1/1000 artifact's catalogue counts and the full "
            "captured public ACS PUMS archive. Not a build, not a certification, not "
            "release eligible."
        ),
        "release_eligible": False,
        "counts": {
            "acs_households": ACS_HOUSEHOLDS,
            "acs_persons": ACS_PERSONS,
            "asec_households": ASEC_HOUSEHOLDS,
            "asec_persons": ASEC_PERSONS,
            "stacked_households": STACKED_HOUSEHOLDS,
            "stacked_persons": STACKED_PERSONS,
            "clone_households": CLONE_HOUSEHOLDS,
            "clone_persons": CLONE_PERSONS,
        },
        "origin_budget": origin_budget(),
        "numeric_bounds": numeric_bounds(),
        "age_artifact": age_artifact(),
        "roles_projection": roles_projection(),
        "predictor_projection": predictor_projection(),
        "diagnostic_matrix": diagnostic_matrix(preparation),
        "child_draw": child_draw(),
        "selection_block": selection_block(preparation),
    }
    out.write_text(json.dumps(record, indent=1) + "\n")
    print("encoder measurements written; streaming the ACS archive ...", flush=True)
    record["acs_household_archive"] = acs_households(snapshot / "csv_hus.zip")
    out.write_text(json.dumps(record, indent=1) + "\n")
    print(
        "household archive done",
        record["acs_household_archive"]["seconds"],
        "s",
        flush=True,
    )
    record["acs_person_archive"] = acs_persons(snapshot / "csv_pus.zip", contract)
    people = record["acs_person_archive"]
    people["derived"] = {
        "old_charge_admits_rows_at_64MiB": int(
            auth.MAX_BODY_BYTES / (people["old_charge_bytes"] / people["records"])
        ),
        "old_charge_full_source_over_64MiB": people["old_charge_bytes"]
        / auth.MAX_BODY_BYTES,
        "body_full_source_over_64MiB": people["body_bytes"] / auth.MAX_BODY_BYTES,
        "body_admits_rows_at_64MiB": int(
            auth.MAX_BODY_BYTES / (people["body_bytes"] / people["records"])
        ),
        "key_list_full_source_over_64MiB": people["key_list_bytes"]
        / auth.MAX_BODY_BYTES,
        "body_bytes_per_row_mean": people["body_bytes"] / people["records"],
        "old_charge_over_body": people["old_charge_bytes"] / people["body_bytes"],
    }
    hh = record["acs_household_archive"]
    hh["derived"] = {
        "old_charge_occupied_over_64MiB": hh["old_charge_bytes_occupied"]
        / auth.MAX_BODY_BYTES,
        "np_total_equals_person_records": hh["np_total"] == people["records"],
        "serialno_list_over_1MiB": hh["serialno_list_bytes_full_source"] / MIB,
        "serialno_list_over_2MiB": hh["serialno_list_bytes_full_source"] / (2 * MIB),
        "serialnos_the_1MiB_cap_admits": (MIB - 1) // 16,
    }
    out.write_text(json.dumps(record, indent=1) + "\n")
    print(
        json.dumps({"people": people["derived"], "households": hh["derived"]}, indent=1)
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
