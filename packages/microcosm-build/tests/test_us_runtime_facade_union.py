"""Frozen B/F/J facade contract, without donor or country-engine loading."""

import hashlib
from importlib import import_module
from types import SimpleNamespace

import pytest

from microcosm.build import us_runtime as facade

# Derived independently from __all__ in pinned B 2ebca68e, F eba103bb and
# J 6f6087cb. Hash preimage is UTF-8 newline-joined sorted unique export names.
_UNION_SHA256 = "213ae67eab1640925fade5075a10eae62d7f219b15c1c04090a9feac8c692aa3"
_CONSTANTS = (
    "PUF_CAPITAL_GAINS_TAIL_APPLIED_COLUMN",
    "PUF_CAPITAL_GAINS_TAIL_DONOR_AGI_BAND_COLUMN",
    "PUF_CAPITAL_GAINS_TAIL_DONOR_FILING_STATUS_COLUMN",
    "PUF_CAPITAL_GAINS_TAIL_DONOR_SOURCE_ID_COLUMN",
    "PUF_CAPITAL_GAINS_TAIL_DONOR_SYNTHETIC_COLUMN",
    "PUF_CAPITAL_GAINS_TAIL_PERSON_COLUMNS",
    "PUF_CAPITAL_GAINS_TAIL_TAX_UNIT_COLUMNS",
    "PUF_CAPITAL_GAINS_TAIL_TRANSFER_WEIGHT_COLUMN",
    "PUF_TAX_DETAIL_DEFAULT_PERSON_OUTPUTS",
    "PUF_TAX_DETAIL_DEFAULT_TAX_UNIT_OUTPUTS",
    "US_PUF_SUPPORT_STAGE_NAME",
    "US_QBI_BOOLEAN_OUTPUT_COLUMNS",
    "US_QBI_NONNEGATIVE_OUTPUT_COLUMNS",
    "US_QBI_OUTPUT_COLUMNS",
)


def test_facade_preserves_exact_906_name_union_without_duplicates():
    assert len(facade.__all__) == len(set(facade.__all__)) == 906
    preimage = "\n".join(sorted(facade.__all__)).encode()
    assert hashlib.sha256(preimage).hexdigest() == _UNION_SHA256
    assert set(facade.__all__) <= set(dir(facade))
    assert all(
        name in facade.__dict__
        or name in facade._LAZY_EXPORTS
        or name in facade._SOURCE_EXPORTS
        or name in facade._SUPPORT_EXPORTS
        or name in {"US_DONORS", "US_STAGE_NAMES"}
        for name in facade.__all__
    )


def test_graph_constants_keep_canonical_owner_and_object_identity():
    module_name = "microcosm.build.us_runtime.operator_column_contracts"
    owner = import_module(module_name)
    for name in _CONSTANTS:
        assert facade._LAZY_EXPORTS[name] == (module_name, name)
        assert getattr(facade, name) is getattr(owner, name)


@pytest.mark.parametrize(
    ("name", "module"),
    [
        ("ASEC_RAW_STAGE_COVERAGE_SCHEMA_VERSION", "asec_checkpoint"),
        ("load_asec_raw_stage_checkpoint_v4", "asec_checkpoint"),
        ("us_puma_ladder_joint_support_gate", "puma_ladder"),
    ],
)
def test_new_union_members_resolve_to_real_owner(name, module):
    owner = import_module(f"microcosm.build.us_runtime.{module}")
    assert getattr(facade, name) is getattr(owner, name)


def test_lazy_resolution_preserves_override_installed_during_import(monkeypatch):
    name = "us_puma_ladder_joint_support_gate"
    imported, override = object(), object()
    monkeypatch.delitem(facade.__dict__, name, raising=False)

    def loader(module):
        assert module == "microcosm.build.us_runtime.puma_ladder"
        monkeypatch.setitem(facade.__dict__, name, override)
        return SimpleNamespace(**{name: imported})

    monkeypatch.setattr(facade, "import_module", loader)
    assert facade._resolve_export(name) is override
    assert getattr(facade, name) is override


def test_existing_override_avoids_lazy_import(monkeypatch):
    name = "us_puma_ladder_joint_support_gate"
    override = object()
    monkeypatch.setitem(facade.__dict__, name, override)

    def refuse_import(module):
        raise AssertionError(f"unexpected import of {module}")

    monkeypatch.setattr(facade, "import_module", refuse_import)
    assert facade._resolve_export(name) is override


def test_plan_keeps_missing_and_unknown_stage_refusal(monkeypatch):
    # Ordinary injected plan avoids reading source/donor manifest resources.
    monkeypatch.setitem(facade.__dict__, "US_STAGE_NAMES", ("ordinary_stage",))
    monkeypatch.setitem(facade.__dict__, "US_DONORS", {})
    with pytest.raises(ValueError, match="missing"):
        facade.us_plan({})
    with pytest.raises(ValueError, match="Unknown stage"):
        facade.us_plan(
            {"ordinary_stage": lambda frame: frame, "typo": lambda frame: frame}
        )

    def transform(frame):
        return frame

    plan = facade.us_plan({"ordinary_stage": transform})
    assert len(plan.stages) == 1
    assert plan.stages[0].name == "ordinary_stage"
    assert plan.stages[0].transform is transform
