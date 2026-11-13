#!/usr/bin/env python3
"""Extract NZ golden-08 from two pinned axiom-oracles files into a small JSON.

The sources are ``requests.json`` and ``full-responses.json`` from
TheAxiomFoundation/axiom-oracles at 6275afcdbe7590d0a56298c4916970a46ed936d0.
Pass local copies; each must match its pinned SHA-256 and schema id.

The extract keeps golden-08's request (``requests[8]``: 28 Family inputs,
including the reg 17 base rate and the reg 18 cutout), the three
Accommodation Supplement outputs it queries (the full responses'
``requests[8].response.results[0]``), and the inputs of golden-00 and
golden-07, whose entity ids share golden-08's scenario stem and wage point and
whose period is golden-08's. The golden-08 differential case reads person facts
from those two. Every value records its source and RFC 6901 pointer. The script
resolves each pointer in its source, compares it with a direct walk of the same
path, and refuses any mismatch, so the extract holds only values read there.
For all three requests it records the full responses' ``request_sha256`` and
recomputes it from the requests file, requires every input interval to equal
the query period, and refuses a response whose metadata shows a fallback
(a ``fallback_reason`` or an ``actual_mode`` other than its
``requested_mode``). It also recomputes golden-08's ``response_sha256``.

``--check`` rebuilds the extract from the given files and fails if the
committed extract differs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "packages/microcosm-build/tests/golden/nz_golden08_oracle_extract.json"
FORMAT = "microcosm.nz.golden08_oracle_extract@1"
REPOSITORY = "TheAxiomFoundation/axiom-oracles"
COMMIT = "6275afcdbe7590d0a56298c4916970a46ed936d0"
SOURCES = {
    "requests": {
        "file": "requests.json",
        "schema": "axiom_oracles.nz_executable_requests.v1",
        "sha256": "ed83b26c3221c253b13a9df307567e09948b2e9337ff5289729276b6b6ed6272",
    },
    "full_responses": {
        "file": "full-responses.json",
        "schema": "axiom_oracles.nz_executable_full_responses.v1",
        "sha256": "085daf12c203e0d1431234deaf733cf02caed725e3fd609e77f9f9ca803960eb",
    },
}
GOLDEN = ("golden-08", 8, "Family")
HOUSEHOLD_REQUESTS = (("golden-00", 0, "Person"), ("golden-07", 7, "Person"))
STEM = "single_parent_three_children_area1_rent"
#: golden-08 input position -> name: the two amounts the bridge calls produce.
BRIDGE_INPUTS = {
    2: "accommodation_supplement_base_rate_weekly_amount",
    11: "accommodation_supplement_non_beneficiary_income_cutout_weekly_amount",
}
#: golden-08's queried outputs; the extract refuses a request that queries others.
OUTPUTS = (
    "accommodation_supplement_rounded_weekly_payment",
    "accommodation_supplement_weekly_amount_before_rounding",
    "accommodation_supplement_weekly_qualifying_accommodation_costs",
)


def _token(value):
    return str(value).replace("~", "~0").replace("/", "~1")


def pointer(*tokens):
    """The RFC 6901 pointer for a path of keys and indices."""
    return "".join("/" + _token(token) for token in tokens)


def resolve(document, text):
    """Resolve an RFC 6901 pointer, refusing noncanonical indices and escapes."""
    if text == "":
        return document
    if not text.startswith("/"):
        raise ValueError(f"{text!r} is not an RFC 6901 pointer.")
    value = document
    for raw in text[1:].split("/"):
        if re.search(r"~(?:[^01]|$)", raw):
            raise ValueError(f"{text!r} contains an invalid escape.")
        token = raw.replace("~1", "/").replace("~0", "~")
        if isinstance(value, list):
            if re.fullmatch(r"0|[1-9][0-9]*", token) is None or int(token) >= len(
                value
            ):
                raise ValueError(f"{text!r} has a bad array index {token!r}.")
            value = value[int(token)]
        elif isinstance(value, dict) and token in value:
            value = value[token]
        else:
            raise ValueError(f"{text!r} does not resolve.")
    return value


def _canonical_sha256(value):
    text = json.dumps(value, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _read(documents, source, *tokens):
    """Read one value by pointer and check it against a direct walk."""
    text = pointer(*tokens)
    value = resolve(documents[source], text)
    direct = documents[source]
    for token in tokens:
        direct = direct[token]
    if type(direct) is not type(value) or direct != value:
        raise ValueError(f"{source} {text} disagrees with its direct path.")
    return {"source": source, "pointer": text, "value": value}


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _load(path, source):
    data = Path(path).read_bytes()
    digest = hashlib.sha256(data).hexdigest()
    pinned = SOURCES[source]
    if digest != pinned["sha256"]:
        raise SystemExit(
            f"{path} has SHA-256 {digest}; the pinned {pinned['file']} has "
            f"{pinned['sha256']}."
        )
    document = json.loads(data)
    _require(
        document.get("schema") == pinned["schema"],
        f"{path} is not a {pinned['schema']} document.",
    )
    return document


def _request(documents, request_id, index, entity):
    """One request's metadata and inputs, each with its source pointer."""
    base = ("requests", index)
    identity = _read(documents, "requests", *base, "id")
    _require(identity["value"] == request_id, f"requests[{index}] is not {request_id}.")
    request = documents["requests"]["requests"][index]["request"]
    _require(request["dataset"]["relations"] == [], f"{request_id} declares relations.")
    queries = request["queries"]
    _require(len(queries) == 1, f"{request_id} must carry one query.")
    rows = request["dataset"]["inputs"]
    entity_ids = {row["entity_id"] for row in rows} | {queries[0]["entity_id"]}
    _require(len(entity_ids) == 1, f"{request_id} names more than one entity id.")
    entity_id = entity_ids.pop()
    _require(entity_id.split(":")[0] == STEM, f"{request_id} is not on {STEM!r}.")
    _require(
        {row["entity"] for row in rows} == {entity},
        f"{request_id} inputs must all be {entity}.",
    )
    period = queries[0]["period"]
    interval = {"start": period["start"], "end": period["end"]}
    _require(
        all(row["interval"] == interval for row in rows),
        f"{request_id} has an input interval other than its query period.",
    )
    inputs = {}
    for position, row in enumerate(rows):
        module, separator, name = row["name"].partition("#input.")
        _require(separator and name not in inputs, f"Bad input name {row['name']!r}.")
        path = (*base, "request", "dataset", "inputs", position, "value")
        read = _read(documents, "requests", *path, "value")
        inputs[name] = {
            "name": row["name"],
            "kind": _read(documents, "requests", *path, "kind")["value"],
            "value": read["value"],
            "source": "requests",
            "pointer": read["pointer"],
        }
    return {
        "id": identity,
        "program": _read(documents, "requests", *base, "program"),
        "mode": _read(documents, "requests", *base, "request", "mode"),
        "entity": entity,
        "entity_id": _read(
            documents, "requests", *base, "request", "queries", 0, "entity_id"
        ),
        "period": _read(
            documents, "requests", *base, "request", "queries", 0, "period"
        ),
        "queried_outputs": _read(
            documents, "requests", *base, "request", "queries", 0, "outputs"
        ),
        "inputs": inputs,
    }


def _full_entry(documents, request_id, index, request):
    """A request's full-responses entry, its id and its ``request_sha256``.

    Refuses an entry for another request or program, a ``request_sha256`` that
    does not hash the requests file's request, and a response whose metadata
    shows a fallback.
    """
    base = ("requests", index)
    identity = _read(documents, "full_responses", *base, "id")
    _require(identity["value"] == request_id, f"full responses [{index}] mismatch.")
    entry = documents["full_responses"]["requests"][index]
    _require(entry["program"] == request["program"]["value"], "Programs differ.")
    source_request = documents["requests"]["requests"][index]["request"]
    request_sha256 = _read(documents, "full_responses", *base, "request_sha256")
    _require(
        request_sha256["value"] == _canonical_sha256(source_request),
        f"{request_id} request_sha256 does not hash the requests file's request.",
    )
    metadata = entry["response"]["metadata"]
    _require(
        metadata["fallback_reason"] is None
        and metadata["actual_mode"] == metadata["requested_mode"],
        f"{request_id}'s response shows a fallback: {metadata}.",
    )
    return entry, identity, request_sha256


def _response(documents, request_id, index, request):
    """golden-08's compared outputs, with the request/response digest checks."""
    base = ("requests", index)
    entry, identity, request_sha256 = _full_entry(documents, request_id, index, request)
    response_sha256 = _read(documents, "full_responses", *base, "response_sha256")
    _require(
        response_sha256["value"] == _canonical_sha256(entry["response"]),
        f"{request_id} response_sha256 does not hash its response.",
    )
    results = entry["response"]["results"]
    _require(len(results) == 1, f"{request_id} must carry one result.")
    _require(
        results[0]["entity_id"] == request["entity_id"]["value"]
        and results[0]["period"] == request["period"]["value"],
        f"{request_id} result names another entity or period.",
    )
    outputs = {}
    for name in OUTPUTS:
        keys = [
            key for key, row in results[0]["outputs"].items() if row["name"] == name
        ]
        _require(len(keys) == 1, f"{request_id} must report {name} once.")
        key = keys[0]
        _require(key in request["queried_outputs"]["value"], f"{key} was not queried.")
        path = (*base, "response", "results", 0, "outputs", key)
        read = _read(documents, "full_responses", *path, "value", "value")
        outputs[name] = {
            "id": key,
            "kind": _read(documents, "full_responses", *path, "value", "kind")["value"],
            "unit": _read(documents, "full_responses", *path, "unit")["value"],
            "value": read["value"],
            "source": "full_responses",
            "pointer": read["pointer"],
        }
    _require(
        len(outputs) == len(request["queried_outputs"]["value"]),
        f"{request_id} queries outputs the extract does not keep.",
    )
    return {
        "id": identity,
        "request_sha256": request_sha256,
        "response_sha256": response_sha256,
        "outputs": outputs,
    }


def extract(requests_path, full_responses_path):
    """Build the extract document from local copies of the two pinned files."""
    documents = {
        "requests": _load(requests_path, "requests"),
        "full_responses": _load(full_responses_path, "full_responses"),
    }
    request_id, index, entity = GOLDEN
    request = _request(documents, request_id, index, entity)
    names = list(request["inputs"])
    for position, name in BRIDGE_INPUTS.items():
        _require(names[position] == name, f"golden-08 input {position} is not {name}.")
    household_requests = {}
    for name, position, kind in HOUSEHOLD_REQUESTS:
        household = _request(documents, name, position, kind)
        _require(
            household["entity_id"]["value"].split(":")[-1]
            == request["entity_id"]["value"].split(":")[-1]
            and household["period"]["value"] == request["period"]["value"],
            f"{name} is not at golden-08's wage point and period.",
        )
        _, _, request_sha256 = _full_entry(documents, name, position, household)
        household_requests[name] = {**household, "request_sha256": request_sha256}
    return {
        "format": FORMAT,
        "description": (
            "Golden-08's Accommodation Supplement request and outputs, and the "
            "inputs of two main-benefit requests on the same scenario household, "
            "read by pointer from two pinned axiom-oracles files. Built by "
            "tools/extract_nz_golden08_oracle.py; the golden-08 differential case "
            "compares against it."
        ),
        "oracle": {"repository": REPOSITORY, "commit": COMMIT},
        "sources": SOURCES,
        "golden_08": {
            "request": request,
            "bridge_inputs": [
                BRIDGE_INPUTS[position] for position in sorted(BRIDGE_INPUTS)
            ],
            "response": _response(documents, request_id, index, request),
        },
        "household_requests": household_requests,
    }


def render(document):
    return json.dumps(document, indent=2, ensure_ascii=False) + "\n"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--requests", type=Path, required=True)
    parser.add_argument("--full-responses", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args(argv)
    text = render(extract(args.requests, args.full_responses))
    if args.check:
        if not args.out.exists() or args.out.read_text(encoding="utf-8") != text:
            raise SystemExit(f"Golden-08 oracle extract is stale: {args.out}")
        print("Golden-08 oracle extract matches its pinned sources.")
        return
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(text, encoding="utf-8")
    print(f"Wrote {args.out}.")


if __name__ == "__main__":
    main()
