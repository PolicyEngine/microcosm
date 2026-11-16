"""S3: which Route A QRF columns above 5% top-1 share a largest record or a household?

Reads route_a_top1.json (the PR's evidence) for each column's largest record at
final weights, and the release H5's person table (pandas HDFStore, read with
h5py, engine-free) to map tax units and persons to households.

Usage: shared_largest_records.py <route_a_top1.json> <populace_us_2024.h5>
"""

import json
import sys
from collections import defaultdict

import h5py

evidence = json.load(open(sys.argv[1]))
with h5py.File(sys.argv[2], "r") as f:
    person = f["person/table"].fields(
        ["person_id", "person_tax_unit_id", "person_household_id"]
    )[:]
tax_unit_households = defaultdict(set)
for tax_unit, household in zip(  # noqa: B905
    person["person_tax_unit_id"].tolist(), person["person_household_id"].tolist()
):
    tax_unit_households[tax_unit].add(household)
person_household = dict(
    zip(person["person_id"].tolist(), person["person_household_id"].tolist())  # noqa: B905
)
assert all(len(h) == 1 for h in tax_unit_households.values()), (
    "a tax unit spans households"
)

over = evidence["over"]["final_weights_over_5pct"]
rows = {row["column"]: row for row in evidence["rows"]}
by_record, by_household = defaultdict(list), defaultdict(list)
print(f"{'column':45s} {'share':>6s} entity     record    household  weight")
for column in over:
    row = rows[column]
    largest = row["final_largest_record"]
    entity, record = largest["entity"], largest["record_id"]
    if entity == "person":
        household = person_household[record]
        assert largest["household_id"] in (None, household), column
    elif entity == "tax_unit":
        (household,) = tax_unit_households[record]
    else:
        household = record if entity == "household" else None
    by_record[(entity, record)].append(column)
    by_household[household].append(column)
    print(
        f"{column:45s} {row['final']['top_1_share']:6.3f} {entity:9s} {record:8d} {household!s:>10s}  {largest['weight']!r}"
    )
print()
print(
    "largest records shared by columns:",
    {f"{e} {r}": c for (e, r), c in by_record.items() if len(c) > 1},
)
print(
    "households holding more than one column's largest record:",
    {h: c for h, c in by_household.items() if len(c) > 1},
)
