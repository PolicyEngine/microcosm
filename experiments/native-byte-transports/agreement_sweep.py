"""Seeded agreement sweep over every byte transport this lane changed the shape of.

Max's standard from the retention lane (its report §11, question 3): wherever a
predicate becomes a fold, a seeded sweep runs beside the hand-built battery.
Five predicates became folds here, one per encoder, and each is swept against
the whole-bytes oracle it replaces:

* ``graph_survey_population._segmented_json`` against ``_bounded_json`` at
  random segment sizes down to the longest token; ``_json_sha256`` against
  ``_sha(_bounded_json)``; ``_json_matches`` against ``==`` on the whole
  bytes, on every mutation (a flipped byte, a truncation, an extension, an
  empty payload) and on the unmutated bytes.
* ``acs_person_coverage_authentication._json_roster`` against ``_json`` at
  random segment sizes; ``_json_sha256`` against ``_sha(_json)``.
* ``graph_child_property_income._json`` against ``adapter._json``
  (``json.dumps`` with the adapter's settings) at random accumulation sizes.
* ``survey_origin_budget``'s streamed ``household_ids``/``group_indices``
  lists: ``append_list`` is exercised through the module by encoding a list
  element by element and comparing with ``_json(list)``; the sweep reproduces
  the loop's bytes here because the loop is a closure inside ``_document``.

Every document is drawn from a seeded generator over nested dicts and lists of
ints, floats (finite, including -0.0 and subnormals), ASCII and non-ASCII
strings, None and booleans, at random depth and width. Refusals are also
swept: below the longest token every segmented encoder must refuse with its
per-accumulation code, and one byte over the total every one must refuse with
its total code.

Not a build, not a certification, not release eligible.

    python agreement_sweep.py <out.json> [pairs]
"""

from __future__ import annotations

import hashlib
import json
import pathlib
import random
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path[:0] = [str(path) for path in sorted((ROOT / "packages").glob("*/src"))]

from microcosm.build.us_runtime import acs_person_coverage_authentication as auth
from microcosm.build.us_runtime import graph_child_property_income as child
from microcosm.build.us_runtime import graph_survey_population as graph
from microcosm.build.us_runtime import survey_origin_budget as budget
from microcosm.fit import graph_joint_empirical as adapter

for _module in (auth, child, graph, budget, adapter):
    if not pathlib.Path(_module.__file__).resolve().is_relative_to(ROOT):
        raise SystemExit(f"{_module.__name__} resolved outside {ROOT}")

SEED = 20260918
ALPHABET = "abcXYZ019 _-./\\\"'\n\té中\U0001f600"


def _scalar(rng, floats):
    # The coverage encoder admits str, int, bool, None, list and dict only
    # (acs_person_coverage_authentication._json_size refuses CANONICAL_TYPE on
    # anything else), so floats are drawn only for the encoders that take them.
    kind = rng.random()
    if kind < 0.25:
        return rng.randint(-(2**62), 2**62)
    if kind < 0.45 and floats:
        return rng.choice(
            [0.0, -0.0, 1.5, 137.0 / 7, 5e-324, 1e300, rng.random() * 1e6, -rng.random()]
        )
    if kind < 0.75:
        return "".join(rng.choice(ALPHABET) for _ in range(rng.randint(0, 24)))
    if kind < 0.85:
        return None
    return rng.random() < 0.5


def _document(rng, depth=0, floats=True):
    if depth >= 3 or rng.random() < 0.3:
        return _scalar(rng, floats)
    if rng.random() < 0.5:
        return [_document(rng, depth + 1, floats) for _ in range(rng.randint(0, 12))]
    keys = ["".join(rng.choice(ALPHABET[:12]) for _ in range(rng.randint(1, 8))) for _ in range(rng.randint(0, 8))]
    return {k: _document(rng, depth + 1, floats) for k in keys}


def _roster(rng, floats=True):
    # A document with a dict at the top and a long list somewhere, like every
    # whole-roster document on the path.
    return {
        "protocol": "invented/sweep",
        "rows": [_document(rng, 1, floats) for _ in range(rng.randint(1, 60))],
        "header": _document(rng, 1, floats),
    }


def _mutations(rng, whole):
    yield "truncated", whole[:-1]
    yield "extended", whole + b"x"
    yield "empty", b""
    if whole:
        position = rng.randrange(len(whole))
        flipped = bytearray(whole)
        flipped[position] ^= 1 << rng.randrange(8)
        yield "flipped", bytes(flipped)


def sweep_graph(rng, receipt):
    document = _roster(rng)
    whole = graph._bounded_json(document, 64 * 1024**2)
    pieces = list(graph._canonical_pieces(document, 64 * 1024**2))
    longest = max(map(len, pieces))
    for _ in range(3):
        segment = rng.randint(longest, max(longest, len(whole)) + 8)
        receipt["comparisons"] += 1
        if graph._segmented_json(document, segment=segment, maximum=len(whole)) != whole:
            receipt["disagreements"].append(("graph._segmented_json", segment, whole[:80]))
    receipt["comparisons"] += 1
    if graph._json_sha256(document, maximum=len(whole)) != hashlib.sha256(whole).hexdigest():
        receipt["disagreements"].append(("graph._json_sha256", whole[:80]))
    receipt["comparisons"] += 1
    if graph._json_matches(document, whole, maximum=len(whole)) is not True:
        receipt["disagreements"].append(("graph._json_matches whole", whole[:80]))
    for name, mutated in _mutations(rng, whole):
        receipt["comparisons"] += 1
        if graph._json_matches(document, mutated, maximum=2**40) is not False:
            receipt["disagreements"].append(("graph._json_matches " + name, whole[:80]))
    if longest > 1 and longest < len(whole):
        receipt["refusals"] += 1
        try:
            graph._segmented_json(document, segment=longest - 1, maximum=len(whole))
            receipt["disagreements"].append(("graph accumulation refusal missing", whole[:80]))
        except graph.SurveyPopulationGraphError as error:
            if str(error) != "TRANSPORT_LIMIT":
                receipt["disagreements"].append(("graph accumulation code", str(error)))
    receipt["refusals"] += 1
    try:
        graph._segmented_json(document, segment=longest, maximum=len(whole) - 1)
        receipt["disagreements"].append(("graph total refusal missing", whole[:80]))
    except graph.SurveyPopulationGraphError as error:
        if str(error) != "TRANSPORT_ROSTER_LIMIT":
            receipt["disagreements"].append(("graph total code", str(error)))


def sweep_coverage(rng, receipt):
    document = _roster(rng, floats=False)
    whole = auth._json(document, auth.MAX_ROSTER_BYTES)
    pieces = list(auth._json_chunks(document, auth.MAX_ROSTER_BYTES))
    longest = max(map(len, pieces))
    for _ in range(3):
        segment = rng.randint(longest, max(longest, len(whole)) + 8)
        receipt["comparisons"] += 1
        if auth._json_roster(document, segment=segment, maximum=len(whole)) != whole:
            receipt["disagreements"].append(("auth._json_roster", segment, whole[:80]))
    receipt["comparisons"] += 1
    if auth._json_sha256(document, len(whole)) != hashlib.sha256(whole).hexdigest():
        receipt["disagreements"].append(("auth._json_sha256", whole[:80]))
    if longest > 1 and longest < len(whole):
        receipt["refusals"] += 1
        try:
            auth._json_roster(document, segment=longest - 1, maximum=len(whole))
            receipt["disagreements"].append(("auth accumulation refusal missing", whole[:80]))
        except auth.ACSCoverageAuthenticationError as error:
            if str(error) != "CANONICAL_SIZE":
                receipt["disagreements"].append(("auth accumulation code", str(error)))
    receipt["refusals"] += 1
    try:
        auth._json_roster(document, segment=longest, maximum=len(whole) - 1)
        receipt["disagreements"].append(("auth total refusal missing", whole[:80]))
    except auth.ACSCoverageAuthenticationError as error:
        if str(error) != "CANONICAL_SIZE":
            receipt["disagreements"].append(("auth total code", str(error)))


def sweep_child(rng, receipt, monkey):
    document = _roster(rng)
    whole = adapter._json(document)
    encoder = json.JSONEncoder(sort_keys=True, separators=(",", ":"), allow_nan=False)
    longest = max(len(piece.encode()) for piece in encoder.iterencode(document))
    for _ in range(3):
        segment = rng.randint(longest, max(longest, len(whole)) + 8)
        monkey(child, "MAX_ARTIFACT_BYTES", segment)
        monkey(child, "MAX_ROSTER_BYTES", len(whole))
        receipt["comparisons"] += 1
        if child._json(document) != whole:
            receipt["disagreements"].append(("child._json", segment, whole[:80]))
    if longest > 1 and longest < len(whole):
        monkey(child, "MAX_ARTIFACT_BYTES", longest - 1)
        monkey(child, "MAX_ROSTER_BYTES", 2**40)
        receipt["refusals"] += 1
        try:
            child._json(document)
            receipt["disagreements"].append(("child accumulation refusal missing", whole[:80]))
        except ValueError as error:
            if "GRAPH_ARTIFACT_SIZE" not in str(error):
                receipt["disagreements"].append(("child accumulation code", str(error)))
    monkey(child, "MAX_ARTIFACT_BYTES", 64 * 1024**2)
    monkey(child, "MAX_ROSTER_BYTES", len(whole) - 1)
    receipt["refusals"] += 1
    try:
        child._json(document)
        receipt["disagreements"].append(("child total refusal missing", whole[:80]))
    except ValueError as error:
        if "GRAPH_ARTIFACT_LIMIT" not in str(error):
            receipt["disagreements"].append(("child total code", str(error)))


def sweep_budget_lists(rng, receipt):
    # The two header lists are streamed one element at a time inside
    # _document's append_list; the bytes those appends produce are
    # b"[" + b",".join(_json(v)) + b"]", which must equal _json(list).
    values = [rng.randint(-(2**62), 2**62) for _ in range(rng.randint(0, 200))]
    streamed = b"[" + b",".join(budget._json(v) for v in values) + b"]"
    receipt["comparisons"] += 1
    if streamed != budget._json(values):
        receipt["disagreements"].append(("budget append_list", values[:5]))


def main() -> int:
    out = pathlib.Path(sys.argv[1])
    pairs = int(sys.argv[2]) if len(sys.argv) > 2 else 2000
    rng = random.Random(SEED)
    receipt = {
        "scope": "Seeded agreement sweep over the segmented, digested and matched byte transports this lane introduced, against their whole-bytes oracles. Not a build, not a certification, not release eligible.",
        "release_eligible": False,
        "seed": SEED,
        "documents": pairs,
        "comparisons": 0,
        "refusals": 0,
        "disagreements": [],
    }
    saved = {}

    def monkey(module, name, value):
        saved.setdefault((module, name), getattr(module, name))
        setattr(module, name, value)

    try:
        for _ in range(pairs):
            sweep_graph(rng, receipt)
            sweep_coverage(rng, receipt)
            sweep_child(rng, receipt, monkey)
            sweep_budget_lists(rng, receipt)
    finally:
        for (module, name), value in saved.items():
            setattr(module, name, value)
    receipt["disagreements"] = [
        [str(part)[:200] for part in item] for item in receipt["disagreements"]
    ]
    out.write_text(json.dumps(receipt, indent=1) + "\n")
    print(
        f"documents={pairs} comparisons={receipt['comparisons']} "
        f"refusals={receipt['refusals']} disagreements={len(receipt['disagreements'])}"
    )
    return 1 if receipt["disagreements"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
