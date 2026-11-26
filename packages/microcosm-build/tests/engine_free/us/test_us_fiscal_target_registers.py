"""The US compile registers read from declared data as the old literals did.

microcosm#1728 moved the reviewed registers of the US fiscal target compile
from Python literals in ``us_runtime/fiscal_targets.py`` and
``us_runtime/source_vintage.py`` to ``us/fiscal_target_registers.json``, read
by the shared ``microcosm.build.declared_registers``. The digests below were
measured on the literals at ``904a46d86``, before the move. A digest covers
the keys, the values, their Python types (a tuple is not a list) and the entry
order, so an equal digest is the same register.

The pinned-feed tests compile the pinned ``41b2fb2`` feed under four flag sets
and pin the full registry JSON, and pin the exclusion receipt. They run only
when ``MICROCOSM_US_CHRONICLE_FACTS`` names the pinned feed.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import os
from pathlib import Path

import pytest

from microcosm.build.declared_registers import package_registers
from microcosm.build.us_runtime import fiscal_targets, source_vintage
from microcosm.build.us_runtime.fiscal_target_registers import (
    US_FISCAL_TARGET_REGISTERS_RESOURCE,
    us_fiscal_target_register,
)

#: (module, name) -> digest of the literal it replaced.
PINNED_REGISTERS = {
    ("fiscal_targets", "SOI_AMOUNT_MEASURE_VARIABLES"): (
        "e9de59d33a9f522ae88ebf34e76365cc010dfc7ee04d12a42a632460dffb7330"
    ),
    ("fiscal_targets", "SOI_RETURN_MEASURE_VARIABLES"): (
        "62b1cd21abd3dbbc469e68722e8fd6e60e8d0775f12cf4cf1008aac30f9d248d"
    ),
    ("fiscal_targets", "SOI_VARIABLE_MAP"): (
        "fdadeab36c1c42177bc57d8987395e6628944c182c8cb30a63cf92e564ac121e"
    ),
    ("fiscal_targets", "_SOI_BASE_VARIABLE_OVERRIDES"): (
        "3104dff5c88f8de11c3f80c9e362da7e7c6d2a4dfe5102b7a063a22142b93a17"
    ),
    ("fiscal_targets", "DIRECT_LEDGER_TARGETS"): (
        "c991f34caf1e9bde876777a2785f42f736b0331013ce0a72c4e25c627863e9e5"
    ),
    ("fiscal_targets", "INDICATOR_LEDGER_TARGETS"): (
        "dbe04076e64d03cd561a0ce2087db3d36e87e93d7f7b1405633d23267e098d49"
    ),
    ("fiscal_targets", "US_FISCAL_TARGET_SUPPORT_EXCLUSIONS"): (
        "0557e4028686d83baffb643546ce607f3ccdad925fdd3069d20461fac4ef45ac"
    ),
    ("fiscal_targets", "US_FISCAL_TARGET_ALL_VINTAGE_SUPPORT_EXCLUSIONS"): (
        "8bf5ae5682a618737d7d2868e72a2f0433b3febc6d63741cbaa4a59074019444"
    ),
    ("fiscal_targets", "US_FISCAL_TARGET_EXCLUSION_VINTAGE_BYPASSES"): (
        "42adf0d3280d340b19fe97e4e275dfe458029cfa2c8f287e44f0099b444fed47"
    ),
    ("fiscal_targets", "US_FISCAL_TARGET_SOURCE_COLUMN_EXCLUSIONS"): (
        "0209724942fcbafcd460630749e066fe46d0efdf2d44f61b18a8ce6279e63ed4"
    ),
    ("fiscal_targets", "_M_CHIP_STATE_FIPS"): (
        "c952168ebc0e0c1d346306e248ff51a570e163a8bc5b26cdbecdaf8ec99346e8"
    ),
    ("source_vintage", "US_RESTAMPED_SOURCE_PACKAGES"): (
        "2c259b40a92efbe2a582a5f7f2c336f3a30f2ffcf2873bc306c5b8a3ddf26ab1"
    ),
    ("source_vintage", "US_LATER_PERIOD_OBSERVATION_EXEMPTIONS"): (
        "42adf0d3280d340b19fe97e4e275dfe458029cfa2c8f287e44f0099b444fed47"
    ),
}

#: The compile of the pinned feed under four flag sets, before the move:
#: (targets, TargetRegistry.version, sha256 of the registry JSON).
PINNED_COMPILES = {
    "unaged": (
        31_747,
        "8cb6469905f8",
        "efded864baf7c0b18d4326c7dd9515556f03c495df6c92963972b9fbab5a61fc",
    ),
    "aged": (
        31_747,
        "6fe934db05ea",
        "c756b266e2e7f074c906817248f9cb99c0c5c930be80bde3639abe195a0b8fa9",
    ),
    "unaged_student_loans": (
        31_877,
        "d8c4e2f13bee",
        "0e733bfd69d4e6c67b2e5a8b0372a2d2aa9f7f7fd4efff31b261914bf8c310ea",
    ),
    "aged_student_loans": (
        31_877,
        "fe558dd37fb4",
        "6767f33ac5fad73ffc56333ca0634b77b041e21a3106d424a3751ae56b98b756",
    ),
}
COMPILE_FLAGS = {
    "unaged": {"allow_unaged_dollar_targets": True},
    "aged": {"age_targets": True},
    "unaged_student_loans": {
        "allow_unaged_dollar_targets": True,
        "student_loan_targets": True,
    },
    "aged_student_loans": {"age_targets": True, "student_loan_targets": True},
}
PINNED_EXCLUSION_RECEIPT_SHA256 = (
    "04a74b9cdcc674465497c1bff28ed953173690e36d2bda5c1480a92c1bffe22f"
)

#: The register sections of the file, and the names the code gives them.
SECTIONS = {
    "support_exclusions": "US_FISCAL_TARGET_SUPPORT_EXCLUSIONS",
    "all_vintage_support_exclusions": "US_FISCAL_TARGET_ALL_VINTAGE_SUPPORT_EXCLUSIONS",
    "exclusion_vintage_bypasses": "US_FISCAL_TARGET_EXCLUSION_VINTAGE_BYPASSES",
    "source_column_exclusions": "US_FISCAL_TARGET_SOURCE_COLUMN_EXCLUSIONS",
    "soi_amount_measure_variables": "SOI_AMOUNT_MEASURE_VARIABLES",
    "soi_return_measure_variables": "SOI_RETURN_MEASURE_VARIABLES",
    "soi_variables": "SOI_VARIABLE_MAP",
    "soi_base_variable_overrides": "_SOI_BASE_VARIABLE_OVERRIDES",
    "direct_targets": "DIRECT_LEDGER_TARGETS",
    "indicator_targets": "INDICATOR_LEDGER_TARGETS",
    "m_chip_state_fips": "_M_CHIP_STATE_FIPS",
    "restamped_source_packages": "US_RESTAMPED_SOURCE_PACKAGES",
    "later_period_observation_exemptions": "US_LATER_PERIOD_OBSERVATION_EXEMPTIONS",
}


def _canon(value):
    """Keys, values, Python types and order, as JSON (the baseline's encoding)."""

    if isinstance(value, dict) or (
        hasattr(value, "items") and not isinstance(value, (str, bytes))
    ):
        return ["<map>", [[_canon(k), _canon(v)] for k, v in value.items()]]
    if isinstance(value, (frozenset, set)):
        return ["<set>", sorted(_canon(v) for v in value)]
    if isinstance(value, (tuple, list)):
        kind = "<tuple>" if isinstance(value, tuple) else "<list>"
        return [kind, [_canon(v) for v in value]]
    if dataclasses.is_dataclass(value):
        return [
            "<dc:" + type(value).__name__ + ">",
            [
                [field.name, _canon(getattr(value, field.name))]
                for field in dataclasses.fields(value)
            ],
        ]
    return [type(value).__name__, value]


def _digest(value) -> str:
    return hashlib.sha256(json.dumps(_canon(value), default=str).encode()).hexdigest()


@pytest.mark.parametrize(
    ("module", "name"), sorted(PINNED_REGISTERS), ids=lambda part: str(part)
)
def test_each_register_reads_back_as_the_literal_it_replaced(
    module: str, name: str
) -> None:
    """The declared register equals the literal: content, types and order.

    A ``fiscal_targets`` register is read fresh from the file, because other
    tests patch the module's dict in place, and ``monkeypatch.delitem``
    restores a key at the end, which changes the order but not the content.
    The module's object must hold the same content. The ``source_vintage``
    registers are read-only mapping proxies, so the module's objects are read.
    """

    if module == "fiscal_targets":
        section = {value: key for key, value in SECTIONS.items()}[name]
        declared = us_fiscal_target_register(section)
        assert getattr(fiscal_targets, name) == declared
    else:
        declared = getattr(source_vintage, name)
    assert _digest(declared) == PINNED_REGISTERS[(module, name)]


def test_the_file_declares_exactly_the_registers_the_code_reads() -> None:
    registers = package_registers(
        "microcosm.build.us", US_FISCAL_TARGET_REGISTERS_RESOURCE
    )
    assert set(registers) == set(SECTIONS)
    assert sorted(SECTIONS.values()) == sorted(
        name for _module, name in PINNED_REGISTERS
    )


def _pinned_feed() -> str:
    feed = os.environ.get("MICROCOSM_US_CHRONICLE_FACTS", "")
    if not feed or not Path(feed).exists():
        pytest.skip("set MICROCOSM_US_CHRONICLE_FACTS to the pinned 41b2fb2 feed")
    return feed


@pytest.fixture(scope="module")
def pinned_facts():
    from microcosm.build.ledger_artifact import load_ledger_consumer_artifact
    from microcosm.build.us_runtime import (
        default_congressional_district_vintage_crosswalk_path,
        load_congressional_district_vintage_crosswalk,
    )

    facts = load_ledger_consumer_artifact(_pinned_feed()).facts
    crosswalk = load_congressional_district_vintage_crosswalk(
        default_congressional_district_vintage_crosswalk_path()
    )
    return facts, crosswalk


@pytest.mark.parametrize("label", sorted(PINNED_COMPILES))
def test_pinned_feed_compiles_are_unchanged(pinned_facts, label: str, tmp_path) -> None:
    facts, crosswalk = pinned_facts
    registry = fiscal_targets.compile_us_fiscal_target_registry(
        facts,
        target_period=2024,
        congressional_district_vintage_crosswalk=crosswalk,
        **COMPILE_FLAGS[label],
    )
    path = registry.to_json(tmp_path / f"{label}.json")
    assert (
        len(registry.specs),
        registry.version,
        hashlib.sha256(path.read_bytes()).hexdigest(),
    ) == PINNED_COMPILES[label]


def test_pinned_feed_exclusion_receipt_is_unchanged(pinned_facts) -> None:
    facts, crosswalk = pinned_facts
    receipt = fiscal_targets.us_fiscal_target_exclusion_receipt(
        facts, target_period=2024, congressional_district_vintage_crosswalk=crosswalk
    )
    digest = hashlib.sha256(
        json.dumps(receipt, indent=1, default=str).encode()
    ).hexdigest()
    assert digest == PINNED_EXCLUSION_RECEIPT_SHA256
