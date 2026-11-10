"""Cross-tabulate two stage runs' immigration labels (millions of persons).

Usage: python -I xtab_labels.py MAIN_LABELS.pkl HEAD_LABELS.pkl [OUT.json]

Rows align on (source_year, source_person_id); the script refuses unequal
row sets or person weights. Prints the README's main-by-head table.
"""

import json
import sys

import numpy as np
import pandas as pd

ORDER = [
    "CITIZEN",
    "LEGAL_PERMANENT_RESIDENT",
    "UNDOCUMENTED",
    "DACA",
    "CUBAN_HAITIAN_ENTRANT",
    "PAROLED_ONE_YEAR",
    "REFUGEE",
    "ASYLEE",
    "TPS",
]


def load(path):
    frame = pd.read_pickle(path)
    key = frame["source_year"].astype(str) + ":" + frame["source_person_id"].astype(str)
    if key.duplicated().any():
        raise SystemExit(f"{path}: duplicate person keys")
    return frame.set_index(key)


main_path, head_path = sys.argv[1:3]
main, head = load(main_path), load(head_path)
if set(main.index) != set(head.index):
    raise SystemExit("the two runs hold different persons")
head = head.loc[main.index]
if not np.allclose(main["person_weight"], head["person_weight"], rtol=0, atol=1e-9):
    raise SystemExit("the two runs weight persons differently")
weight = main["person_weight"].to_numpy()
table = (
    pd.DataFrame(
        {
            "main": main["immigration_status_str"].to_numpy(),
            "head": head["immigration_status_str"].to_numpy(),
            "w": weight,
        }
    )
    .groupby(["main", "head"])["w"]
    .sum()
    .div(1e6)
    .unstack(fill_value=0.0)
)
statuses = [s for s in ORDER if s in set(table.index) | set(table.columns)]
statuses += sorted((set(table.index) | set(table.columns)) - set(statuses))
table = table.reindex(index=statuses, columns=statuses, fill_value=0.0)
pd.set_option("display.width", 200)
print(table.round(3).to_string())
totals = {
    "main": {s: round(float(table.loc[s].sum()), 3) for s in statuses},
    "head": {s: round(float(table[s].sum()), 3) for s in statuses},
}
print(json.dumps(totals, indent=1))
none_main = main["ssn_card_type"].eq("NONE").to_numpy()
none_head = head["ssn_card_type"].eq("NONE").to_numpy()
und_main = main["immigration_status_str"].eq("UNDOCUMENTED").to_numpy()
und_head = head["immigration_status_str"].eq("UNDOCUMENTED").to_numpy()
print(
    "NONE iff UNDOCUMENTED:",
    bool((none_main == und_main).all()),
    bool((none_head == und_head).all()),
)
if len(sys.argv) > 3:
    out = {
        "main_labels": main_path,
        "head_labels": head_path,
        "millions": {
            r: {c: round(float(table.loc[r, c]), 6) for c in statuses} for r in statuses
        },
        "totals": totals,
    }
    with open(sys.argv[3], "w") as stream:
        json.dump(out, stream, indent=1, sort_keys=True)
