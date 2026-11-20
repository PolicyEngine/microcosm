"""The four US gate scopes: one battery per build tool (microcosm#1585 item 11).

A :class:`~microcosm.build.gate_battery.GateBatteryRun` runs one manifest,
each phase once and in order, so the US gives each build tool its own scope
of ``us/gates.json``, as the UK does for its spine build, calibration seam,
release cut and local candidate: the entries the tool evaluates, the phases
it evaluates them in, and a policy suffix, so each scope has its own
``gates_manifest_sha256``, ``policy_sha256`` and ``spec_fingerprint`` to pin.

- ``us_base``: ``tools/build_us_puf_support_base.py``, Route A's base.
- ``us_pool``: ``tools/build_us_multispine_pool.py``.
- ``us_national``: ``tools/build_us_fiscal_refresh_release.py``.
- ``us_local``: ``tools/build_us_acs_local_release.py``.

A scope owns the entries whose id starts with its name and an underscore
(``us_base_weights_audit`` belongs to ``us_base``). The partition is checked
closed-world in both directions, as the UK's is: an entry no scope owns, or
one bound to a phase its scope does not evaluate, is refused when the
manifest is scoped.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass
from functools import cache
from pathlib import Path
from types import MappingProxyType

from microcosm.build.country_spec import GatesManifest, load_gates_manifest
from microcosm.build.gate_battery import (
    GateBatteryRun,
    GateBinding,
    scoped_gate_manifest,
)

__all__ = [
    "US_GATE_SCOPES",
    "USGateScope",
    "check_us_gate_scopes",
    "us_gate_scope_digests",
    "us_gates_manifest",
    "us_scope_entry_ids",
    "us_scoped_gate_manifest",
]


@dataclass(frozen=True)
class USGateScope:
    """One US build tool's share of ``us/gates.json``.

    Attributes:
        name: The scope's name; its entries' id prefix and policy suffix stem.
        tool: The repository path of the tool that evaluates it.
        phases: The phases it evaluates, in order.
    """

    name: str
    tool: str
    phases: tuple[str, ...]

    @property
    def prefix(self) -> str:
        return f"{self.name}_"

    @property
    def policy_suffix(self) -> str:
        return f"{self.name}_scope"


US_GATE_SCOPES: Mapping[str, USGateScope] = MappingProxyType(
    {
        scope.name: scope
        for scope in (
            USGateScope(
                name="us_base",
                tool="tools/build_us_puf_support_base.py",
                phases=("assembled", "transferred", "terminal"),
            ),
            USGateScope(
                name="us_pool",
                tool="tools/build_us_multispine_pool.py",
                phases=("assembled", "transferred", "terminal"),
            ),
            USGateScope(
                name="us_national",
                tool="tools/build_us_fiscal_refresh_release.py",
                phases=("preflight", "assembled", "terminal"),
            ),
            USGateScope(
                name="us_local",
                tool="tools/build_us_acs_local_release.py",
                phases=("assembled", "terminal"),
            ),
        )
    }
)


@cache
def us_gates_manifest() -> GatesManifest:
    """The committed ``us/gates.json``, read once per process."""

    return load_gates_manifest("us")


def _owner(entry_id: str) -> list[str]:
    return [
        scope.name
        for scope in US_GATE_SCOPES.values()
        if entry_id.startswith(scope.prefix)
    ]


def check_us_gate_scopes(gates: GatesManifest) -> None:
    """Refuse a manifest the four scopes do not partition exactly.

    Raises:
        RuntimeError: An entry no scope owns, one two scopes' prefixes both
            match, or an entry bound to a phase its scope does not evaluate.
    """

    unowned: list[str] = []
    ambiguous: list[str] = []
    stray: list[str] = []
    for entry in gates.gates:
        owners = _owner(entry.id)
        if not owners:
            unowned.append(entry.id)
        elif len(owners) > 1:
            ambiguous.append(entry.id)
        elif entry.phase not in US_GATE_SCOPES[owners[0]].phases:
            stray.append(f"{entry.id} ({entry.phase})")
    if unowned or ambiguous or stray:
        raise RuntimeError(
            "US gate scopes must partition us/gates.json: owned by no scope "
            f"{sorted(unowned)}, by two {sorted(ambiguous)}, bound to a phase "
            f"their scope does not evaluate {sorted(stray)}."
        )


def us_scope_entry_ids(
    scope: str, gates: GatesManifest | None = None
) -> tuple[str, ...]:
    """The ids ``scope`` owns, in manifest order."""

    declared = _scope(scope)
    source = us_gates_manifest() if gates is None else gates
    return tuple(
        entry.id for entry in source.gates if entry.id.startswith(declared.prefix)
    )


def _scope(scope: str) -> USGateScope:
    try:
        return US_GATE_SCOPES[scope]
    except KeyError:
        raise ValueError(
            f"unknown US gate scope {scope!r}; declared: {sorted(US_GATE_SCOPES)}."
        ) from None


def us_scoped_gate_manifest(
    scope: str, *, source: GatesManifest | None = None
) -> GatesManifest:
    """One tool's scope of ``us/gates.json``.

    ``source`` lets a caller that already holds a manifest, or a hermetic
    test, supply it; the default is the committed package spec.
    """

    declared = _scope(scope)
    gates = us_gates_manifest() if source is None else source
    check_us_gate_scopes(gates)
    return scoped_gate_manifest(
        gates,
        us_scope_entry_ids(scope, gates),
        phases=declared.phases,
        policy_suffix=declared.policy_suffix,
    )


def us_gate_scope_digests(
    scope: str, registry: Mapping[str, GateBinding]
) -> dict[str, str]:
    """The scope's three pinned digests, as its run would attest them."""

    run = GateBatteryRun(
        us_scoped_gate_manifest(scope),
        release_id="us-gate-scope-digest-derivation",
        report_path=Path(os.devnull),
        release_candidate=False,
        registry=registry,
    )
    return {
        "gates_manifest_sha256": run.gates_manifest_sha256,
        "policy_sha256": run.policy_sha256,
        "spec_fingerprint": run.spec_fingerprint,
    }
