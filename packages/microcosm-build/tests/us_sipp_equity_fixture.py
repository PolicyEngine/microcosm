"""INVENTED SIPP-shaped December-2022 fixtures. NOT genuine SIPP data.

Every SSUID, PNUM, weight, equity value and flag here was authored by hand.
No genuine SIPP file, projection, receipt or content store was opened to build
it. The lexical shapes are transcribed from the field contracts in
``microcosm.build.sipp_measurements`` so the real measurement API accepts them;
the VALUES are fiction and carry no source authority.

The pins in the invented contract below are computed from these invented bytes
through the producer's own key formulas. That is the explicit test-only
construction route: a caller-supplied pin never confers production authority.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path

import microcosm.build.sipp_december_source as s
import microcosm.build.sipp_equity_contract as contract_module
import microcosm.build.sipp_measurements as m
from microcosm.graph.canonical import canonical_json
from microcosm.graph.store import ContentStore

INVENTED_PARENT_SHA256 = (
    "f" * 8 + "1" * 8 + "0" * 8 + "a" * 8 + "b" * 8 + "c" * 8 + "d" * 16
)
INVENTED_PARENT_KEY = "e" * 64
INVENTED_REGISTRY_SHA256 = "9" * 64
INVENTED_DICTIONARY_SHA256 = "8" * 64
INVENTED_BINDING = "7" * 64
_MISSING_EQUITY = "-10000998"


def _row(ssuid, pnum, residence, weight, equity, flag, **over):
    values = {
        "SSUID": ssuid,
        "PNUM": str(pnum),
        "MONTHCODE": "12",
        "ERESIDENCEID": str(residence),
        "WPFINWGT": weight,
        "THEQ_VEH": equity,
        "AHEQ_VEH": flag,
        "THVAL_VEH": "0",
        "AHVAL_VEH": "9",
        "THDEBT_VEH": "0",
        "AHDEBT_VEH": "9",
        "TVEH_NUM": "1",
        "AVEH_NUM": "1",
        "TVAL_BANK": "0",
        "AVAL_BANK": "1",
        "TAGE_EHC": "40",
    }
    values.update(over)
    return values


#: (ssuid, residence, per-member weights, equity, flag) -> one invented residence.
#: Flag 9 is the genuine source's only observed AHEQ_VEH flag, so most donors
#: carry it: admission keeps them and never relabels the quality.
_ADMITTED = (
    ("00000001", 100001, ("1140.5",), "-4200", "9"),
    ("00000002", 100002, ("980.25", "1010.75"), "0", "9"),
    ("00000003", 100003, ("1500.0", "1450.5", "1490.5"), "15800", "9"),
    ("00000004", 100004, ("760.0",), "-950", "9"),
    ("00000005", 100005, ("2100.0", "2050.0"), "27400", "9"),
    ("00000006", 100006, ("890.0", "905.0", "870.0", "915.0"), "0", "9"),
    ("00000007", 100007, ("1320.75", "1280.25"), "8600", "1"),
    ("00000008", 100008, ("1710.0", "1690.0", "1730.0"), "-12500", "9"),
    # A member with an exact zero weight; the mean stays strictly positive.
    ("00000009", 100009, ("640.5", "0"), "3100", "9"),
    # One SSUID, two December residences.
    ("00000010", 100010, ("1180.0", "1220.0"), "5100", "9"),
    ("00000010", 100011, ("1990.25",), "41200", "9"),
)
#: Excluded before donor CREATE, retained in full coverage.
_ALL_ZERO_WEIGHT = ("00000020", 200001, ("0", "0"), "7700", "9")
_UNKNOWN_WEIGHT = ("00000021", 200002, ("1050.0", None), "6700", "9")
_CONFLICTING_EQUITY = ("00000022", 200003, ("1260.0", "1300.0"), ("5400", "9100"), "9")
_UNKNOWN_EQUITY = ("00000023", 200004, ("1440.5",), _MISSING_EQUITY, "9")


def invented_rows():
    """Every invented December row, admitted and excluded alike."""
    rows = []
    used: dict[str, int] = {}
    for ssuid, residence, weights, equity, flag in _ADMITTED:
        # PNUM is unique per SSUID, so a second residence continues the count.
        base = used.get(ssuid, 101)
        for offset, weight in enumerate(weights):
            rows.append(_row(ssuid, base + offset, residence, weight, equity, flag))
        used[ssuid] = base + len(weights)
    ssuid, residence, weights, equity, flag = _ALL_ZERO_WEIGHT
    for offset, weight in enumerate(weights):
        rows.append(_row(ssuid, 101 + offset, residence, weight, equity, flag))
    ssuid, residence, weights, equity, flag = _UNKNOWN_WEIGHT
    for offset, weight in enumerate(weights):
        row = _row(ssuid, 101 + offset, residence, weight or "", equity, flag)
        rows.append(row)
    ssuid, residence, weights, equities, flag = _CONFLICTING_EQUITY
    for offset, (weight, equity) in enumerate(zip(weights, equities, strict=True)):
        rows.append(_row(ssuid, 101 + offset, residence, weight, equity, flag))
    ssuid, residence, weights, equity, flag = _UNKNOWN_EQUITY
    for offset, weight in enumerate(weights):
        rows.append(_row(ssuid, 101 + offset, residence, weight, equity, flag))
    return rows


#: A second, equally valid invented residence set: the same eleven admitted
#: donors plus one more two-person residence. It keeps every admitted household
#: size, so the same recipient population stays inside the support, and it
#: changes the projection bytes, so it is a *different valid source*, not a
#: corrupted one.
_VARIANT_EXTRA = ("00000012", 100013, ("1425.25", "1385.75"), "13300", "9")

ADMITTED_RESIDENCES = len(_ADMITTED)
EXCLUDED_ZERO_WEIGHT = 1
EXCLUDED_UNKNOWN_TARGET = 2  # conflicting household copies, missing sentinel
EXCLUDED_UNKNOWN_WEIGHT = 1
ADMITTED_SIZES = sorted({len(spec[2]) for spec in _ADMITTED})
DENOMINATOR_RESIDENCES = (
    ADMITTED_RESIDENCES
    + EXCLUDED_ZERO_WEIGHT
    + EXCLUDED_UNKNOWN_TARGET
    + EXCLUDED_UNKNOWN_WEIGHT
)
VARIANT_ADMITTED_RESIDENCES = ADMITTED_RESIDENCES + 1

#: Hand-authored expectations, read off the invented specs above rather than
#: computed by the code under test. The genuine contract's own 17,588 / 20 / 38
#: are untouched; these are the fixture's own counts for the same three pins.
EXPECTED = {
    "expected_admitted_residences": ADMITTED_RESIDENCES,
    "expected_excluded_zero_weight": EXCLUDED_ZERO_WEIGHT,
    "expected_excluded_unknown_target": EXCLUDED_UNKNOWN_TARGET,
}
VARIANT_EXPECTED = dict(
    EXPECTED, expected_admitted_residences=VARIANT_ADMITTED_RESIDENCES
)


def variant_rows():
    """A second valid invented source: one more admitted two-person residence."""
    ssuid, residence, weights, equity, flag = _VARIANT_EXTRA
    extra = [
        _row(ssuid, 101 + offset, residence, weight, equity, flag)
        for offset, weight in enumerate(weights)
    ]
    return invented_rows() + extra


def build_projection(rows=None):
    """Serialize invented rows exactly as the accepted producer would."""
    rows = invented_rows() if rows is None else rows
    return canonical_json(
        {
            "format": "sipp-december-lexical/1",
            "fields": s.FIELDS,
            "rows": [tuple(row.get(f) for f in s.FIELDS) for row in rows],
        }
    )


def build_receipt(
    projection,
    rows=None,
    *,
    binding=INVENTED_BINDING,
    counts=None,
    ssuid_digest=None,
    genuine_shaped=False,
):
    """Build an invented receipt with the producer's own digests and formulas.

    ``genuine_shaped`` adds the fields the accepted producer also writes —
    ``parser`` and ``aggregates`` — so a receipt shaped like the genuine one,
    with its extra fields, is exercised rather than only the minimal shape.
    """
    rows = invented_rows() if rows is None else rows
    source = m.SippSource(INVENTED_PARENT_SHA256, m.SIPP_2023_MASTER_SHA256, 2023, 2022)
    supplied = tuple(
        m.SippRow(dict(zip(s.FIELDS, [r.get(f) for f in s.FIELDS], strict=True)))
        for r in rows
    )
    households = m.december_households(
        supplied, source=source, complete_membership=True, copied_fields=s._COPIED
    )
    bank = m.december_bank_assets(supplied, source=source)
    observed = s._counts(households) if counts is None else counts
    # The producer's own selector and key formula over the same roster.
    selected = s._selected_ssuids(tuple(h.key for h in households), "all_ssuids")
    digest = (
        s._key("sipp-december-ssuid-selection/1", selected)
        if ssuid_digest is None
        else ssuid_digest
    )
    receipt = {
        "format": "sipp_december_source_v1",
        "producer_binding": binding,
        "contract": {
            "parent": {
                "name": "sipp_2023_pu2023",
                "sha256": INVENTED_PARENT_SHA256,
                "bytes": 1234,
                "content_key": INVENTED_PARENT_KEY,
                "revision": "0" * 40,
            },
            "master_sha256": m.SIPP_2023_MASTER_SHA256,
            "registry_sha256": INVENTED_REGISTRY_SHA256,
            "dictionary_sha256": INVENTED_DICTIONARY_SHA256,
            "source_years": {"survey": 2023, "reference": 2022},
        },
        "selection": "all_ssuids",
        "selected_ssuid_digest": digest,
        "full_counts": observed,
        "selected_counts": observed,
        "full_membership_digest": s._membership(households),
        "selected_membership_digest": s._membership(households),
        "projection": {
            "key": s._key(
                "sipp-december-projection/1",
                (binding, "all_ssuids", s._sha(projection), len(projection)),
            ),
            "sha256": s._sha(projection),
            "bytes": len(projection),
        },
        "measurement_digest": s._measurement_digest(households, bank),
    }
    if genuine_shaped:
        receipt["parser"] = {
            "header_columns": len(s.FIELDS),
            "physical_records": len(rows),
            "month_records": {"12": len(rows)},
        }
        receipt["aggregates"] = {
            "bank": s._profile(bank),
            "vehicles": {
                f: s._profile(h.measurements[i] for h in households)
                for i, f in enumerate(s._COPIED)
            },
            "person_weights": s._weights(
                w for h in households for w in h.person_weights
            ),
            "household_weights": s._weights(h.weight for h in households),
        }
    return canonical_json(receipt)


def publish(root: Path, projection: bytes, receipt: bytes, *, binding=INVENTED_BINDING):
    """Publish the two invented objects into a real ContentStore."""
    store = ContentStore(Path(root))
    receipt_key = s._key(
        "sipp-december-receipt/1", (binding, s._sha(receipt), len(receipt))
    )
    projection_key = json.loads(receipt)["projection"]["key"]
    store.put_bytes(projection_key, projection, node_key=binding)
    store.put_json(
        receipt_key,
        json.loads(receipt),
        kind="sipp_december_source_v1",
        node_key=binding,
    )
    return store, receipt_key, projection_key


def invented_contract(
    receipt,
    projection,
    receipt_key,
    projection_key,
    *,
    binding=INVENTED_BINDING,
    expected=None,
):
    """The explicit test-only contract construction route."""
    document = json.loads(
        (
            Path(contract_module.__file__).parent / contract_module.CONTRACT_RESOURCE
        ).read_bytes()
    )
    parsed = json.loads(receipt)
    document["admission"].update(EXPECTED if expected is None else expected)
    document["source"].update(
        {
            "producer_binding": binding,
            "projection_key": projection_key,
            "projection_sha256": s._sha(projection),
            "projection_bytes": len(projection),
            "receipt_key": receipt_key,
            "receipt_sha256": s._sha(receipt),
            "receipt_bytes": len(receipt),
            "full_membership_digest": parsed["full_membership_digest"],
            "selected_membership_digest": parsed["selected_membership_digest"],
            "measurement_digest": parsed["measurement_digest"],
            "selected_ssuid_digest": parsed["selected_ssuid_digest"],
            "parent_sha256": INVENTED_PARENT_SHA256,
            "parent_content_key": INVENTED_PARENT_KEY,
            "parent_bytes": 1234,
            "parent_revision": "0" * 40,
            "registry_sha256": INVENTED_REGISTRY_SHA256,
            "dictionary_sha256": INVENTED_DICTIONARY_SHA256,
            "counts": parsed["full_counts"],
        }
    )
    return contract_module._contract_from_document(document)


def accepted_store(
    tmp_path: Path,
    *,
    rows=None,
    counts=None,
    binding=INVENTED_BINDING,
    expected=None,
    ssuid_digest=None,
    genuine_shaped=False,
):
    """One invented accepted store plus its matching test-only contract."""
    projection = build_projection(rows)
    receipt = build_receipt(
        projection,
        rows,
        binding=binding,
        counts=counts,
        ssuid_digest=ssuid_digest,
        genuine_shaped=genuine_shaped,
    )
    root = tmp_path / "accepted-store"
    _, receipt_key, projection_key = publish(root, projection, receipt, binding=binding)
    contract = invented_contract(
        receipt,
        projection,
        receipt_key,
        projection_key,
        binding=binding,
        expected=expected,
    )
    return root, contract, projection, receipt


def recipient_source(directory: Path, sizes, *, household_ids=None, extra_column=None):
    """An invented recipient population as csv-tables: complete rosters only."""
    directory.mkdir(parents=True, exist_ok=True)
    ids = list(household_ids or range(900001, 900001 + len(sizes)))
    person_lines = ["person_id,person_household_id"]
    person_id = 0
    for household_id, size in zip(ids, sizes, strict=True):
        for _ in range(size):
            person_id += 1
            person_lines.append(f"{person_id},{household_id}")
    header = "household_id" + (f",{extra_column[0]}" if extra_column else "")
    household_lines = [header]
    for offset, household_id in enumerate(ids):
        row = str(household_id)
        if extra_column:
            row += f",{extra_column[1][offset]}"
        household_lines.append(row)
    (directory / "person.csv").write_text("\n".join(person_lines) + "\n")
    (directory / "household.csv").write_text("\n".join(household_lines) + "\n")
    (directory / "schema.json").write_text(
        json.dumps(
            {
                "schema": {"person_entity": "person", "group_entities": ["household"]},
                "weights": {"household": {"kind": "design"}},
            }
        )
    )
    (directory / "weights.json").write_text(
        json.dumps({"household": {"values": [1.0] * len(ids)}})
    )
    return directory


def mutate_same_size(payload: bytes) -> bytes:
    """Change one byte of a payload without changing its length."""
    body = bytearray(payload)
    for index, value in enumerate(body):
        if chr(value).isdigit() and chr(value) != "9":
            body[index] = value + 1
            return bytes(body)
    raise AssertionError("no digit to mutate")


def forge_metadata(root: Path, key: str, *, node_key: str) -> None:
    """Rewrite a stored object's metadata node_key in place."""
    obj = ContentStore(root).object_path(key)
    meta = json.loads((obj / "meta.json").read_bytes())
    meta["node_key"] = node_key
    (obj / "meta.json").write_bytes(canonical_json(meta))


def copied(document):
    return copy.deepcopy(document)
