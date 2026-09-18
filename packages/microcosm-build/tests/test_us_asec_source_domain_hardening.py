"""Invented mutation/domain controls for source-only amount qualifiers."""

import copy
import hashlib
import json
from dataclasses import replace
from types import SimpleNamespace

import pytest

from microcosm.build.us_runtime import current_asec_child_support_source as child
from microcosm.build.us_runtime import current_asec_income_routing_source as routing
from microcosm.build.us_runtime import current_asec_interest_source as interest

OWNERS = ((routing, "PNSN_VAL"), (interest, "INT_VAL"), (child, "CSP_VAL"))


def getter(owner):
    return owner.printed_amount_entries if owner is routing else owner.amount_entries


def document():
    return json.loads(
        routing.resources.files(routing.__package__)
        .joinpath(routing.DOMAINS_RESOURCE)
        .read_bytes()
    )


def patch_document(monkeypatch, owner, raw):
    payload = json.dumps(raw).encode()
    resource = SimpleNamespace(read_bytes=lambda: payload)
    monkeypatch.setattr(
        owner.resources,
        "files",
        lambda *_: SimpleNamespace(joinpath=lambda *_: resource),
    )
    monkeypatch.setattr(routing, "DOMAINS_SHA256", hashlib.sha256(payload).hexdigest())
    getter(owner).cache_clear()


@pytest.mark.parametrize("owner,field", OWNERS)
def test_public_cache_mapping_cannot_change_later_calls(owner, field):
    get = getter(owner)
    get.cache_clear()
    expected = copy.deepcopy(get())
    try:
        get()[field] = "untrusted replacement"
        assert get() == expected
        get().clear()
        assert get() == expected
    finally:
        get.cache_clear()


def test_child_nested_domain_and_temporal_metadata_are_detached():
    child.amount_entries.cache_clear()
    expected = copy.deepcopy(child.amount_entries())
    try:
        changed = child.amount_entries()
        changed["CSP_VAL"]["width"] = 6
        changed["CSP_VAL"]["domain"]["zero_semantics"] = "valid_zero_dollars"
        changed["CHSP_VAL"]["domain"]["declared_negative_nonmoney_codes"].append(-1)
        changed["CHSP_VAL"]["temporal_authority"]["pdf_pages_1based"].append(999)
        assert child.amount_entries() == expected
    finally:
        child.amount_entries.cache_clear()


@pytest.mark.parametrize("owner,field", OWNERS)
@pytest.mark.parametrize("defect", ["missing", "duplicate"])
def test_current_vintage_must_exist_exactly_once(monkeypatch, owner, field, defect):
    raw = document()
    entry = next(f for f in raw["fields"] if f["name"] == field)
    current = next(v for v in entry["vintages"] if v["income_year"] == 2024)
    if defect == "missing":
        entry["vintages"].remove(current)
    else:
        entry["vintages"].append(copy.deepcopy(current))
    patch_document(monkeypatch, owner, raw)
    try:
        with pytest.raises(ValueError, match="VINTAGE"):
            getter(owner)()
    finally:
        getter(owner).cache_clear()


@pytest.mark.parametrize("owner,field", OWNERS)
@pytest.mark.parametrize("defect", ["missing", "duplicate"])
def test_amount_field_must_exist_exactly_once(monkeypatch, owner, field, defect):
    raw = document()
    entry = next(f for f in raw["fields"] if f["name"] == field)
    if defect == "missing":
        raw["fields"].remove(entry)
    else:
        raw["fields"].append(copy.deepcopy(entry))
    patch_document(monkeypatch, owner, raw)
    try:
        with pytest.raises(ValueError, match="(ROSTER|FIELD)"):
            getter(owner)()
    finally:
        getter(owner).cache_clear()


@pytest.mark.parametrize(
    "owner,field", [(interest, "INT_VAL"), (child, "CSP_VAL"), (child, "CHSP_VAL")]
)
@pytest.mark.parametrize(
    "defect",
    [
        "negative_bound",
        "negative_dollars",
        "negative_niu",
        "other_missing",
        "excluded_dollars",
        "extra_niu",
        "zero_semantics",
        "range_header",
        "numeric_type",
        "entity",
        "grain",
        "column",
    ],
)
def test_repinned_unsupported_amount_domains_refuse(monkeypatch, owner, field, defect):
    raw = document()
    entry = next(f for f in raw["fields"] if f["name"] == field)
    domain = entry["domain"]
    if defect == "negative_bound":
        domain["encoded_range_inclusive"]["minimum"] = -1
    elif defect == "negative_dollars":
        domain["negative_dollars_permitted"] = True
    elif defect == "negative_niu":
        domain["declared_negative_nonmoney_codes"] = [-1]
    elif defect == "other_missing":
        domain["declared_other_missing_codes"] = [-2]
    elif defect == "excluded_dollars":
        domain["valid_dollar_range_excludes"] = [2]
    elif defect == "extra_niu":
        domain["declared_niu_codes"] = [0, 17]
    elif defect == "zero_semantics":
        domain["zero_semantics"] = "valid_zero_dollars"
    elif defect == "range_header":
        next(v for v in entry["vintages"] if v["income_year"] == 2024)["range_header"][
            "maximum"
        ] += 1
    elif defect == "numeric_type":
        domain["numeric_type"] = "decimal_US_dollars"
    elif defect in ("entity", "grain"):
        entry[defect] = "household"
    else:
        entry["column"] = "different_native_amount"
    patch_document(monkeypatch, owner, raw)
    try:
        with pytest.raises(ValueError, match="DOMAIN"):
            getter(owner)()
    finally:
        getter(owner).cache_clear()


def ready_domain():
    return SimpleNamespace(
        bindings=SimpleNamespace(
            spec=SimpleNamespace(
                fields=tuple(
                    routing.money.MoneyDomain(
                        **{
                            k: f[k]
                            for k in (
                                "name",
                                "entity",
                                "column",
                                "grain",
                                "minimum",
                                "maximum",
                                "zero_semantics",
                            )
                        }
                    )
                    for f in document()["fields"]
                )
            )
        )
    )


@pytest.mark.parametrize("owner,field", OWNERS)
@pytest.mark.parametrize(
    "attribute", ["entity", "column", "grain", "minimum", "maximum", "zero_semantics"]
)
def test_live_ready_domain_must_agree(owner, field, attribute):
    ready = ready_domain()
    owner._domain_agreement(ready)
    value = {
        "entity": "household",
        "column": "different_native_amount",
        "grain": "household",
        "minimum": -5,
        "maximum": 999999999,
        "zero_semantics": "changed_zero_meaning",
    }[attribute]
    ready.bindings.spec.fields = tuple(
        replace(d, **{attribute: value}) if d.name == field else d
        for d in ready.bindings.spec.fields
    )
    with pytest.raises(ValueError, match="DOMAIN_DISAGREEMENT"):
        owner._domain_agreement(ready)


@pytest.mark.parametrize("owner,field", OWNERS)
def test_live_ready_domain_duplicates_are_not_overwritten(owner, field):
    ready = ready_domain()
    ready.bindings.spec.fields += tuple(
        d for d in ready.bindings.spec.fields if d.name == field
    )
    with pytest.raises(ValueError, match="DOMAIN"):
        owner._domain_agreement(ready)
