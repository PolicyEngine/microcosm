"""Committed registry of the keys allowed to sign UK gate-battery reports.

Gate reports, release-cut certifications and dense gate reports are signed
with HMAC-SHA256 under the key in ``MICROCOSM_UK_TERMINAL_GATE_SIGNING_KEY``.
Before this registry, every verifier trusted whichever key was in the
environment: a report signed by any 32-byte key verified, provided the
verifier held the same key. Verification now also requires the key's SHA-256
fingerprint to be registered here, so a cut can only be certified,
assembled or published under a key the repository names.

HMAC is symmetric: verifying a report needs the secret that signed it. So a
cut is signed, certified and published end to end by one key holder, and
the registry names one key per builder rather than one shared secret. A new
builder gets their own key and a reviewed entry; nobody has to share a key.

Statuses:

* ``active``: may sign new reports and verify.
* ``retired``: verifies reports already signed, but signs nothing new.
  Use it for a routine rotation.
* ``revoked``: neither signs nor verifies. Use it if a key leaks; every cut
  it signed must then be re-signed or withdrawn.

It lives in ``microcosm-data`` because the publisher verifies, and the
build shard imports it for its signing-side checks. Entries are
source-coded; add, retire or revoke one only in a reviewed pull request
(the procedure is in ``docs/uk-cut-anyone.md``).

Run ``python -m microcosm.data.gate_signing_keys check`` to see whether the
key in your environment may sign.
"""

from __future__ import annotations

import argparse
import base64
import binascii
import getpass
import hashlib
import json
import os
import re
import secrets
import subprocess
import sys
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from types import MappingProxyType
from typing import Literal

__all__ = [
    "GATE_SIGNING_KEY_STATUSES",
    "UK_GATE_SIGNING_KEYS",
    "UK_GATE_SIGNING_KEY_ENV",
    "GateSigningKey",
    "GateSigningKeyError",
    "decode_gate_signing_key",
    "gate_signing_key_fingerprint",
    "registered_gate_signing_key",
    "require_registered_gate_signing_key",
    "store_in_keychain",
    "unregistered_gate_signing_key_problem",
]

#: The environment variable every UK signer and verifier reads. Lockstep with
#: ``microcosm.build.gate_battery.gate_signing_key_env("uk")``.
UK_GATE_SIGNING_KEY_ENV = "MICROCOSM_UK_TERMINAL_GATE_SIGNING_KEY"

GateSigningKeyStatus = Literal["active", "retired", "revoked"]
GateSigningKeyPurpose = Literal["sign", "verify"]
GATE_SIGNING_KEY_STATUSES: frozenset[str] = frozenset({"active", "retired", "revoked"})
_PURPOSE_STATUSES: Mapping[str, frozenset[str]] = MappingProxyType(
    {"sign": frozenset({"active"}), "verify": frozenset({"active", "retired"})}
)


class GateSigningKeyError(RuntimeError):
    """The key in the environment is absent, malformed or not registered."""


@dataclass(frozen=True)
class GateSigningKey:
    """One registered signing key, identified by its fingerprint only.

    The fingerprint is the SHA-256 of the 32 decoded key bytes: the value
    every report records as ``attestation.signing_key_sha256``. The secret
    itself is never committed.
    """

    key_sha256: str
    label: str
    status: GateSigningKeyStatus
    custodian: str
    held_in: str
    registered: str
    evidence: str
    retired_on: str | None = None


UK_GATE_SIGNING_KEYS: Mapping[str, GateSigningKey] = MappingProxyType(
    {
        "66bf0e29dbdabc2e3d7a79f21f43ac28d17b592faaf91e67c9bcf9b30c09a6ff": (
            GateSigningKey(
                key_sha256=(
                    "66bf0e29dbdabc2e3d7a79f21f43ac28d17b592faaf91e67c9bcf9b30c09a6ff"
                ),
                label="uk-gate-juaristi22-2026",
                status="active",
                custodian="María Juaristi (GitHub juaristi22)",
                held_in="her build machine; not in a shared store",
                registered="2026-10-22",
                evidence=(
                    "Signed every UK gate report on record from 2026-08-18 "
                    "(uk/national_staging_build_record.json, #635) through the "
                    "2026-10-19 and 2026-10-21 national cuts and the "
                    "2026-10-19 dense cut."
                ),
            )
        ),
        "dd31e32563854596d983a71a6519c6b7860a7c1f63ffb80dbe679d5d34c4de8c": (
            GateSigningKey(
                key_sha256=(
                    "dd31e32563854596d983a71a6519c6b7860a7c1f63ffb80dbe679d5d34c4de8c"
                ),
                label="uk-gate-team-2026",
                status="active",
                custodian="Max Ghenis (GitHub MaxGhenis)",
                held_in=(
                    "agent-secret service microcosm/uk-gate-signing-key-team, "
                    "account maxghenis, on Max's build Mac; agent sessions on "
                    "that Mac may read it"
                ),
                registered="2026-10-22",
                evidence=(
                    "Generated 2026-10-22 for team-built cuts (microcosm#1348); "
                    "never printed or committed."
                ),
            )
        ),
    }
)


def _registry(country: str) -> Mapping[str, GateSigningKey] | None:
    """The registry for a country, or ``None`` when the country has none.

    Countries without a registry keep the old behaviour (any well-formed key
    in the environment). Tests replace this function to admit their keys.
    """

    return UK_GATE_SIGNING_KEYS if country == "uk" else None


def gate_signing_key_fingerprint(key: bytes) -> str:
    """The SHA-256 a report records for the key that signed it."""

    return hashlib.sha256(key).hexdigest()


def decode_gate_signing_key(encoded: str | None, *, env: str) -> bytes:
    """Decode a base64 signing key, refusing anything but 32 bytes."""

    if not encoded:
        raise GateSigningKeyError(f"{env} must contain a base64-encoded 32-byte key.")
    try:
        key = base64.b64decode(encoded.strip(), validate=True)
    except (binascii.Error, ValueError) as exc:
        raise GateSigningKeyError(f"{env} must be valid base64.") from exc
    if len(key) != 32:
        raise GateSigningKeyError(f"{env} must decode to exactly 32 bytes.")
    return key


def registered_gate_signing_key(country: str, key_sha256: str) -> GateSigningKey | None:
    """The registry entry for a fingerprint, or ``None`` if unregistered."""

    registry = _registry(country)
    if registry is None:
        return None
    return registry.get(key_sha256)


def unregistered_gate_signing_key_problem(
    country: str, key: bytes, *, purpose: GateSigningKeyPurpose
) -> str | None:
    """Why ``key`` may not sign or verify for ``country``, or ``None`` if it may.

    Returns ``None`` for a country without a registry.
    """

    if purpose not in _PURPOSE_STATUSES:
        raise ValueError(f"purpose must be 'sign' or 'verify', got {purpose!r}.")
    registry = _registry(country)
    if registry is None:
        return None
    fingerprint = gate_signing_key_fingerprint(key)
    entry = registry.get(fingerprint)
    if entry is None:
        return (
            f"the {country.upper()} gate signing key with fingerprint "
            f"{fingerprint} is not registered in "
            "microcosm.data.gate_signing_keys; a key signs or verifies "
            "release evidence only after a reviewed pull request registers it."
        )
    if entry.status not in _PURPOSE_STATUSES[purpose]:
        return (
            f"the {country.upper()} gate signing key {entry.label} "
            f"({fingerprint}) is {entry.status}; a {entry.status} key may not "
            f"{purpose}."
        )
    return None


def require_registered_gate_signing_key(
    country: str, key: bytes, *, purpose: GateSigningKeyPurpose
) -> bytes:
    """Return ``key`` if the registry lets it ``purpose``; raise otherwise."""

    problem = unregistered_gate_signing_key_problem(country, key, purpose=purpose)
    if problem is not None:
        raise GateSigningKeyError(problem)
    return key


# ---------------------------------------------------------------------------
# Operator CLI
# ---------------------------------------------------------------------------


def _entry_payload(entry: GateSigningKey) -> dict[str, object]:
    return asdict(entry)


def _check(args: argparse.Namespace) -> int:
    env = args.env or UK_GATE_SIGNING_KEY_ENV
    try:
        key = decode_gate_signing_key(os.environ.get(env), env=env)
    except GateSigningKeyError as exc:
        print(json.dumps({"ok": False, "env": env, "problem": str(exc)}, indent=2))
        return 1
    fingerprint = gate_signing_key_fingerprint(key)
    problem = unregistered_gate_signing_key_problem(
        args.country, key, purpose=args.purpose
    )
    entry = registered_gate_signing_key(args.country, fingerprint)
    print(
        json.dumps(
            {
                "ok": problem is None,
                "env": env,
                "country": args.country,
                "purpose": args.purpose,
                "key_sha256": fingerprint,
                "registered": _entry_payload(entry) if entry else None,
                "problem": problem,
            },
            indent=2,
        )
    )
    return 0 if problem is None else 1


def _list(args: argparse.Namespace) -> int:
    registry = _registry(args.country) or {}
    print(json.dumps([_entry_payload(entry) for entry in registry.values()], indent=2))
    return 0


#: Account and service names ``security -i`` can take unquoted.
_KEYCHAIN_NAME = re.compile(r"^[A-Za-z0-9._@-]+$")


def store_in_keychain(key: bytes, *, service: str, account: str) -> None:
    """Store a key in the macOS login keychain without putting it in any argv.

    ``security`` reads the command from stdin (``security -i``), so the key
    never appears in a process list. The stored value is then read back and
    compared, and is never printed. An existing item is never replaced: a
    retired key's cuts verify only for whoever still holds it, so rotating
    stores the new key under a new service name.
    """

    for name in (service, account):
        if not _KEYCHAIN_NAME.match(name):
            raise GateSigningKeyError(f"keychain name {name!r} must be plain.")
    existing = subprocess.run(
        ["security", "find-generic-password", "-a", account, "-s", service],
        capture_output=True,
        text=True,
        check=False,
    )
    if existing.returncode == 0:
        raise GateSigningKeyError(
            f"keychain service {service} already holds a key for {account}; it "
            "is left as it is. Store a new key under a new --keychain-service "
            "name, so the old key, and the cuts it signed, stay verifiable."
        )
    encoded = base64.b64encode(key).decode()
    subprocess.run(
        ["security", "-i"],
        input=f"add-generic-password -a {account} -s {service} -w {encoded}\n",
        capture_output=True,
        text=True,
        check=True,
    )
    stored = subprocess.run(
        ["security", "find-generic-password", "-a", account, "-s", service, "-w"],
        capture_output=True,
        text=True,
        check=False,
    )
    if stored.returncode != 0 or stored.stdout.strip() != encoded:
        raise GateSigningKeyError(
            f"the key did not read back from keychain service {service}."
        )


def _generate(args: argparse.Namespace) -> int:
    key = secrets.token_bytes(32)
    fingerprint = f"key_sha256 {gate_signing_key_fingerprint(key)}"
    if args.keychain_service:
        try:
            store_in_keychain(
                key, service=args.keychain_service, account=getpass.getuser()
            )
        except (GateSigningKeyError, OSError, subprocess.CalledProcessError) as exc:
            print(f"key not stored: {exc}", file=sys.stderr)
            return 1
        # Only the fingerprint is printed: it is what a pull request registers.
        print(fingerprint)
        return 0
    if sys.stdout.isatty():
        # A key printed to a terminal stays in its scrollback and in any
        # agent transcript.
        print(
            "refusing to print a signing key to a terminal: pass "
            "--keychain-service NAME to store it in the login keychain, or pipe "
            "stdout into a secret store.",
            file=sys.stderr,
        )
        return 1
    # The key goes to stdout for the operator to pipe into a secret store;
    # the fingerprint goes to stderr so it can be pasted into a pull request.
    print(base64.b64encode(key).decode())
    print(fingerprint, file=sys.stderr)
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m microcosm.data.gate_signing_keys",
        description=(
            "Inspect the committed UK gate signing key registry and the key "
            "in the environment."
        ),
    )
    sub = parser.add_subparsers(dest="command", required=True)
    check = sub.add_parser(
        "check",
        help="exit 0 when the key in the environment may sign (or verify)",
    )
    check.add_argument("--country", default="uk")
    check.add_argument("--purpose", choices=("sign", "verify"), default="sign")
    check.add_argument(
        "--env",
        default=None,
        help=f"variable to read (default {UK_GATE_SIGNING_KEY_ENV})",
    )
    check.set_defaults(func=_check)
    listing = sub.add_parser("list", help="print the registered keys")
    listing.add_argument("--country", default="uk")
    listing.set_defaults(func=_list)
    generate = sub.add_parser(
        "generate",
        help=(
            "make a fresh key and print its fingerprint: stored in the login "
            "keychain with --keychain-service, otherwise written to stdout for "
            "a pipe into a secret store (never to a terminal)"
        ),
    )
    generate.add_argument(
        "--keychain-service",
        metavar="NAME",
        help="store the key in the macOS login keychain under this service",
    )
    generate.set_defaults(func=_generate)
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
