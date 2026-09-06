"""Bounded readback of an already accepted December projection and receipt.

This is a public sibling of :mod:`microcosm.build.sipp_december_source`, added
so a consumer can reuse a content-pinned source artifact without rereading the
3.7 GB parent. It reuses that producer's private bounded-object, count,
membership and measurement-digest helpers behind this reviewed wrapper; that
narrow internal dependency is deliberate, tested, and part of consumer identity.
The producer module itself is unchanged, so its accepted historical output
identity still holds.

What this does NOT claim: it does not reauthenticate the original CSV, does not
recompute the historical producer binding from this interpreter, and confers no
authority on caller-supplied pins. It reuses a previously accepted, closed,
content-addressed artifact and checks it against packaged pins by comparison.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from microcosm.graph.store import ContentStore

from . import sipp_december_source as s
from . import sipp_measurements as m

__all__ = [
    "PROJECTION_FORMAT",
    "ReadbackError",
    "SippDecemberReadback",
    "read_sipp_december_projection",
]

PROJECTION_FORMAT = "sipp-december-lexical/1"
_RECEIPT_KIND = "sipp_december_source_v1"
_ALL_SSUIDS = "all_ssuids"
_SSUID_SELECTION_DOMAIN = "sipp-december-ssuid-selection/1"


class ReadbackError(ValueError):
    """This wrapper's own refusal at the accepted-artifact readback boundary.

    Only refusals raised *here* carry this type, and their messages are closed
    codes rather than paths or payload bytes. Errors raised beneath this
    boundary are **not** wrapped: a ``ContentStore`` integrity failure or a
    producer-helper refusal propagates as its own ``StoreCorrupt``,
    ``StoreMiss`` or ``SourceError``, and its message is whatever that layer
    wrote. Callers that need one exception type must catch those too.
    """


@dataclass(frozen=True)
class SippDecemberReadback:
    """Reconstructed December evidence from an accepted, closed projection.

    Nothing here is target admission: every residence in the immutable source
    roster is present, including unknown and zero-weight ones.
    """

    households: tuple[m.DecemberHousehold, ...]
    bank: tuple[m.Measurement, ...]
    rows: tuple[m.SippRow, ...]
    receipt: dict
    receipt_bytes: int
    source: m.SippSource
    producer_binding: str
    projection_key: str
    receipt_key: str
    selected_ssuid_digest: str


def _require(condition: object, code: str) -> None:
    if not condition:
        raise ReadbackError(code)


def _pinned_bytes(
    store: ContentStore,
    key: str,
    scratch: Path,
    *,
    kind: str,
    binding: str,
    sha256: str,
    size: int,
    code: str,
) -> bytes:
    # A pinned object the store does not hold is a sanitized refusal here, not
    # an operating-system error escaping the authority boundary.
    obj = Path(store.object_path(key))
    if obj.is_symlink() or not obj.is_dir():
        raise ReadbackError(code + ":OBJECT_ABSENT")
    payload = s._bounded_object(store, key, scratch, kind=kind, binding=binding)
    _require(len(payload) == size and s._sha(payload) == sha256, code)
    return payload


def read_sipp_december_projection(
    store: ContentStore, scratch: Path, *, contract
) -> SippDecemberReadback:
    """Reconstruct the accepted all-SSUID December evidence from two objects.

    Args:
        store: A ContentStore over a prepared, read-only source store holding
            exactly the accepted receipt and projection objects.
        scratch: A private bounded-copy namespace OUTSIDE ``store.root``.
        contract: The packaged consumer contract carrying the closed pins.

    Raises:
        ReadbackError: On any mismatch against the packaged pins, the receipt's
            own contract, or the reconstruction of the roster and measurements.
    """
    source_pins = contract.source
    # The SSUID-selection digest below is recomputed from the reconstructed
    # roster, which equals the producer's full roster only under the all-SSUID
    # selection. Under any narrower selection the producer's digest covers a
    # different set, so refuse rather than compare an incomparable value. The
    # accepted artifact this consumer exists for is the all-SSUID one.
    _require(source_pins["selection"] == _ALL_SSUIDS, "SELECTION_NOT_ALL_SSUIDS")
    scratch = Path(scratch)
    _require(scratch.is_dir(), "SCRATCH_NOT_A_DIRECTORY")
    root = Path(store.root).resolve()
    resolved_scratch = scratch.resolve()
    # The store root is itself "inside" the store: checking only the parents
    # would have let ``scratch == store.root`` write bounded copies into the
    # closed roster the codec just validated.
    _require(
        resolved_scratch != root and root not in resolved_scratch.parents,
        "SCRATCH_INSIDE_SOURCE_STORE",
    )
    binding = source_pins["producer_binding"]

    receipt_bytes = _pinned_bytes(
        store,
        source_pins["receipt_key"],
        scratch,
        kind=_RECEIPT_KIND,
        binding=binding,
        sha256=source_pins["receipt_sha256"],
        size=source_pins["receipt_bytes"],
        code="RECEIPT_PIN",
    )
    receipt = json.loads(receipt_bytes)
    _require(receipt.get("format") == _RECEIPT_KIND, "RECEIPT_FORMAT")
    # Comparison only. The historical binding embeds the producer's interpreter
    # and dependency versions and is never recomputed from this process.
    _require(receipt.get("producer_binding") == binding, "RECEIPT_BINDING")
    _require(receipt.get("selection") == source_pins["selection"], "RECEIPT_SELECTION")
    contract_fields = receipt.get("contract")
    _require(isinstance(contract_fields, dict), "RECEIPT_CONTRACT_SHAPE")
    parent = contract_fields.get("parent")
    _require(isinstance(parent, dict), "RECEIPT_PARENT_SHAPE")
    for pinned, observed in (
        (source_pins["parent_sha256"], parent.get("sha256")),
        (source_pins["parent_content_key"], parent.get("content_key")),
        (source_pins["parent_bytes"], parent.get("bytes")),
        (source_pins["parent_name"], parent.get("name")),
        (source_pins["parent_revision"], parent.get("revision")),
        (source_pins["master_sha256"], contract_fields.get("master_sha256")),
        (source_pins["registry_sha256"], contract_fields.get("registry_sha256")),
        (source_pins["dictionary_sha256"], contract_fields.get("dictionary_sha256")),
    ):
        _require(pinned == observed, "RECEIPT_CONTRACT_PIN")
    years = contract_fields.get("source_years")
    _require(
        isinstance(years, dict)
        and years.get("survey") == source_pins["survey_year"]
        and years.get("reference") == source_pins["reference_year"],
        "RECEIPT_SOURCE_YEARS",
    )
    # Consistency check of the closed key against the producer's key formula.
    _require(
        s._key(
            "sipp-december-receipt/1",
            (binding, s._sha(receipt_bytes), len(receipt_bytes)),
        )
        == source_pins["receipt_key"],
        "RECEIPT_KEY_FORMULA",
    )

    declared = receipt.get("projection")
    _require(isinstance(declared, dict), "RECEIPT_PROJECTION_SHAPE")
    for pinned, observed in (
        (source_pins["projection_key"], declared.get("key")),
        (source_pins["projection_sha256"], declared.get("sha256")),
        (source_pins["projection_bytes"], declared.get("bytes")),
    ):
        _require(pinned == observed, "RECEIPT_PROJECTION_PIN")

    projection_bytes = _pinned_bytes(
        store,
        source_pins["projection_key"],
        scratch,
        kind="bytes",
        binding=binding,
        sha256=source_pins["projection_sha256"],
        size=source_pins["projection_bytes"],
        code="PROJECTION_PIN",
    )
    _require(
        s._key(
            "sipp-december-projection/1",
            (
                binding,
                source_pins["selection"],
                s._sha(projection_bytes),
                len(projection_bytes),
            ),
        )
        == source_pins["projection_key"],
        "PROJECTION_KEY_FORMULA",
    )

    projection = json.loads(projection_bytes)
    _require(projection.get("format") == PROJECTION_FORMAT, "PROJECTION_FORMAT")
    _require(tuple(projection.get("fields", ())) == s.FIELDS, "PROJECTION_FIELDS")
    raw_rows = projection.get("rows")
    _require(isinstance(raw_rows, list), "PROJECTION_ROWS_SHAPE")
    rows = []
    for raw in raw_rows:
        _require(isinstance(raw, list) and len(raw) == len(s.FIELDS), "PROJECTION_ROW")
        rows.append(m.SippRow(dict(zip(s.FIELDS, raw, strict=True))))
    rows = tuple(rows)

    # The measurement source identity carries the parent and master hashes, so
    # the reconstructed digest is only comparable when both are the pinned ones.
    source = m.SippSource(
        source_pins["parent_sha256"],
        source_pins["master_sha256"],
        source_pins["survey_year"],
        source_pins["reference_year"],
    )
    households = m.december_households(
        rows,
        source=source,
        complete_membership=True,
        copied_fields=s._COPIED,
        target_year=source_pins["reference_year"],
    )
    bank = m.december_bank_assets(
        rows, source=source, target_year=source_pins["reference_year"]
    )

    counts = s._counts(households)
    _require(counts == dict(source_pins["counts"]), "RECONSTRUCTED_COUNTS_PIN")
    _require(counts == receipt.get("selected_counts"), "RECEIPT_SELECTED_COUNTS")
    _require(counts == receipt.get("full_counts"), "RECEIPT_FULL_COUNTS")
    membership = s._membership(households)
    _require(membership == source_pins["selected_membership_digest"], "MEMBERSHIP_PIN")
    _require(
        membership == receipt.get("selected_membership_digest")
        and membership == receipt.get("full_membership_digest")
        and source_pins["full_membership_digest"] == membership,
        "RECEIPT_MEMBERSHIP",
    )
    digest = s._measurement_digest(households, bank)
    _require(digest == source_pins["measurement_digest"], "MEASUREMENT_DIGEST_PIN")
    _require(digest == receipt.get("measurement_digest"), "RECEIPT_MEASUREMENT_DIGEST")

    # The producer's own selector and key formula, not a restatement of them:
    # ``_selected_ssuids`` is the function that produced the pinned digest, and
    # it validates the identity tuples it is given. Under ``all_ssuids`` the
    # reconstructed roster is the producer's full roster, which the counts and
    # membership assertions above have already established.
    ssuid_digest = s._key(
        _SSUID_SELECTION_DOMAIN,
        s._selected_ssuids(tuple(h.key for h in households), source_pins["selection"]),
    )
    _require(
        ssuid_digest == source_pins["selected_ssuid_digest"],
        "SELECTED_SSUID_DIGEST_PIN",
    )
    _require(
        ssuid_digest == receipt.get("selected_ssuid_digest"),
        "RECEIPT_SELECTED_SSUID_DIGEST",
    )

    return SippDecemberReadback(
        households=households,
        bank=bank,
        rows=rows,
        receipt=receipt,
        receipt_bytes=len(receipt_bytes),
        source=source,
        producer_binding=binding,
        projection_key=source_pins["projection_key"],
        receipt_key=source_pins["receipt_key"],
        selected_ssuid_digest=ssuid_digest,
    )
