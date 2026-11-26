"""A child process reuses its parent's lock-file check through a record.

The full check of ``engine_abi.lock.json`` derives the lock afresh from the
installed policyengine-us and compares bytes. A process that passed it can hand
its children a record of what it compared: the lock file's SHA-256, the
installed policyengine-us version and the SHA-256 of the take-up and sources
domains. A child that receives a record from its parent compares those three
values instead of deriving the lock again, and refuses to run when any of them
differs. A process without a record from its parent still runs the full check.
"""

# ruff: noqa: F403, F405
import hashlib
import importlib.metadata
import os

from microcosm.build.spec_engine.canonical import canonical_json_bytes
from test_support.microcosm_build.spec_engine_engine_abi import *

RECORD_VARIABLE = "MICROCOSM_ENGINE_ABI_LOCK_CHECK_RECORD"


def _write_fresh_lock(root: Path, domains: dict[str, object]) -> bytes:
    lock_bytes = engine_abi_lock_bytes_from_domains(domains)
    (root / "engine_abi.lock.json").write_bytes(lock_bytes)
    return lock_bytes


def _domains_sha256(domains: dict[str, object]) -> str:
    return hashlib.sha256(
        canonical_json_bytes(
            {"take_up": domains["take_up"], "sources": domains["sources"]}
        )
    ).hexdigest()


def _check(root: Path, domains: dict[str, object]) -> object:
    return assert_engine_abi_lock_current(
        root,
        domains,
        schema_registry=load_schema_registry(),
    )


def _issued_record(root: Path) -> dict[str, object]:
    environment = engine_abi_module.engine_abi_lock_check_environment(root)
    assert set(environment) == {RECORD_VARIABLE}
    return json.loads(environment[RECORD_VARIABLE])


def _receive_record_from_parent(
    monkeypatch: pytest.MonkeyPatch,
    record: dict[str, object],
) -> None:
    """Make this process look like the child of the record's issuer."""

    received = {**record, "issuer_pid": os.getppid()}
    monkeypatch.setenv(RECORD_VARIABLE, json.dumps(received))


def _refuse_fresh_derivation(monkeypatch: pytest.MonkeyPatch) -> None:
    def derivation_must_not_run(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("the fresh lock derivation ran despite a record")

    monkeypatch.setattr(
        engine_abi_module,
        "_fresh_policyengine_us_contract",
        derivation_must_not_run,
    )
    monkeypatch.setattr(
        engine_abi_module,
        "_fresh_remaining_stage_input_manifest",
        derivation_must_not_run,
    )


def test_a_passed_check_issues_a_record_of_what_it_compared(
    tmp_path: Path,
    fake_fresh_engine: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv(RECORD_VARIABLE, raising=False)
    domains = _fake_domains()
    lock_bytes = _write_fresh_lock(tmp_path, domains)

    _check(tmp_path, domains)
    record = _issued_record(tmp_path)

    assert record == {
        "schema_version": 1,
        "issuer_pid": os.getpid(),
        "lock_sha256": hashlib.sha256(lock_bytes).hexdigest(),
        "engine": {"package": "policyengine-us", "version": "1.2.3"},
        "domains_sha256": _domains_sha256(domains),
    }


def test_no_record_is_issued_without_a_passed_check(
    tmp_path: Path,
    fake_fresh_engine: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv(RECORD_VARIABLE, raising=False)
    assert engine_abi_module.engine_abi_lock_check_environment(tmp_path) == {}

    # Without policyengine-us the lock is only checked structurally, so there is
    # no engine version to record and no record to hand on.
    lock_root = tmp_path / "unattested"
    lock_root.mkdir()
    domains = _fake_domains()
    _write_fresh_lock(lock_root, domains)

    def absent(package: str) -> str:
        raise importlib.metadata.PackageNotFoundError(package)

    monkeypatch.setattr(engine_abi_module, "_installed_engine_version", absent)
    _check(lock_root, domains)
    assert engine_abi_module.engine_abi_lock_check_environment(lock_root) == {}


def test_a_record_from_the_parent_process_replaces_the_fresh_derivation(
    tmp_path: Path,
    fake_fresh_engine: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv(RECORD_VARIABLE, raising=False)
    domains = _fake_domains()
    lock_bytes = _write_fresh_lock(tmp_path, domains)
    _check(tmp_path, domains)
    record = _issued_record(tmp_path)

    _receive_record_from_parent(monkeypatch, record)
    _refuse_fresh_derivation(monkeypatch)

    assert _check(tmp_path, domains) == json.loads(lock_bytes)
    # The child hands the same comparison on to its own children.
    assert _issued_record(tmp_path) == record


def test_a_child_refuses_a_lock_file_that_differs_from_the_parents_record(
    tmp_path: Path,
    fake_fresh_engine: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv(RECORD_VARIABLE, raising=False)
    domains = _fake_domains()
    _write_fresh_lock(tmp_path, domains)
    _check(tmp_path, domains)
    record = _issued_record(tmp_path)

    # A schema-valid, canonical lock that is not the one the parent checked.
    replacement = engine_abi_lock_payload_from_domains(domains)
    replacement["programs"]["program_00"]["default"] = False
    (tmp_path / "engine_abi.lock.json").write_bytes(
        canonical_json_bytes(replacement) + b"\n"
    )
    _receive_record_from_parent(monkeypatch, record)
    _refuse_fresh_derivation(monkeypatch)

    with pytest.raises(SpecValidationError, match="lock file differs"):
        _check(tmp_path, domains)


def test_a_child_refuses_an_installed_engine_that_differs_from_the_parents_record(
    tmp_path: Path,
    fake_fresh_engine: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv(RECORD_VARIABLE, raising=False)
    domains = _fake_domains()
    _write_fresh_lock(tmp_path, domains)
    _check(tmp_path, domains)
    record = _issued_record(tmp_path)

    _receive_record_from_parent(monkeypatch, record)
    _refuse_fresh_derivation(monkeypatch)
    monkeypatch.setattr(
        engine_abi_module,
        "_installed_engine_version",
        lambda package: "9.9.9",
    )

    with pytest.raises(SpecValidationError, match="installed policyengine-us"):
        _check(tmp_path, domains)


def test_a_child_refuses_when_policyengine_us_is_missing_but_the_parent_had_it(
    tmp_path: Path,
    fake_fresh_engine: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv(RECORD_VARIABLE, raising=False)
    domains = _fake_domains()
    _write_fresh_lock(tmp_path, domains)
    _check(tmp_path, domains)
    record = _issued_record(tmp_path)

    _receive_record_from_parent(monkeypatch, record)
    _refuse_fresh_derivation(monkeypatch)

    def absent(package: str) -> str:
        raise importlib.metadata.PackageNotFoundError(package)

    monkeypatch.setattr(engine_abi_module, "_installed_engine_version", absent)

    with pytest.raises(SpecValidationError, match="installed policyengine-us"):
        _check(tmp_path, domains)


@pytest.mark.parametrize("changed_domain", ["take_up", "sources"])
def test_a_child_refuses_domains_that_differ_from_the_parents_record(
    tmp_path: Path,
    fake_fresh_engine: None,
    monkeypatch: pytest.MonkeyPatch,
    changed_domain: str,
) -> None:
    monkeypatch.delenv(RECORD_VARIABLE, raising=False)
    domains = _fake_domains()
    _write_fresh_lock(tmp_path, domains)
    _check(tmp_path, domains)
    record = _issued_record(tmp_path)

    changed = copy.deepcopy(domains)
    changed[changed_domain]["edited_after_the_parent_checked"] = True
    _receive_record_from_parent(monkeypatch, record)
    _refuse_fresh_derivation(monkeypatch)

    with pytest.raises(SpecValidationError, match="take-up and sources domains"):
        _check(tmp_path, changed)


@pytest.mark.parametrize(
    "raw_record",
    [
        "not json",
        json.dumps([]),
        json.dumps({"schema_version": 2}),
        "with an unexpected field",
        "with a malformed digest",
    ],
)
def test_a_child_refuses_a_malformed_record(
    tmp_path: Path,
    fake_fresh_engine: None,
    monkeypatch: pytest.MonkeyPatch,
    raw_record: str,
) -> None:
    monkeypatch.delenv(RECORD_VARIABLE, raising=False)
    domains = _fake_domains()
    _write_fresh_lock(tmp_path, domains)
    _check(tmp_path, domains)
    record = {**_issued_record(tmp_path), "issuer_pid": os.getppid()}
    if raw_record == "with an unexpected field":
        raw_record = json.dumps({**record, "skip": True})
    elif raw_record == "with a malformed digest":
        raw_record = json.dumps({**record, "lock_sha256": "ABC"})
    monkeypatch.setenv(RECORD_VARIABLE, raw_record)
    _refuse_fresh_derivation(monkeypatch)

    with pytest.raises(SpecValidationError, match="malformed"):
        _check(tmp_path, domains)


def test_a_record_issued_by_another_process_is_ignored(
    tmp_path: Path,
    fake_fresh_engine: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A record inherited from further up the process tree is not a parent's.

    Such a process runs the full check, so a stale lock is still refused.
    """

    monkeypatch.delenv(RECORD_VARIABLE, raising=False)
    domains = _fake_domains()
    _write_fresh_lock(tmp_path, domains)
    _check(tmp_path, domains)
    record = _issued_record(tmp_path)
    # This process issued the record itself, so it is not its own parent.
    assert record["issuer_pid"] != os.getppid()
    monkeypatch.setenv(RECORD_VARIABLE, json.dumps(record))

    derivations: list[object] = []

    def counted_contract() -> dict[str, dict[str, object]]:
        derivations.append(None)
        return _fake_engine_contract()

    monkeypatch.setattr(
        engine_abi_module,
        "_fresh_policyengine_us_contract",
        counted_contract,
    )
    _check(tmp_path, domains)
    assert len(derivations) == 1

    stale = engine_abi_lock_payload_from_domains(domains)
    stale["programs"]["program_00"]["default"] = False
    (tmp_path / "engine_abi.lock.json").write_bytes(canonical_json_bytes(stale) + b"\n")
    with pytest.raises(SpecValidationError, match="stale or non-canonical"):
        _check(tmp_path, domains)


def test_without_a_record_the_full_check_runs(
    tmp_path: Path,
    fake_fresh_engine: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv(RECORD_VARIABLE, raising=False)
    domains = _fake_domains()
    _write_fresh_lock(tmp_path, domains)
    derivations: list[object] = []

    def counted_contract() -> dict[str, dict[str, object]]:
        derivations.append(None)
        return _fake_engine_contract()

    monkeypatch.setattr(
        engine_abi_module,
        "_fresh_policyengine_us_contract",
        counted_contract,
    )

    _check(tmp_path, domains)
    _check(tmp_path, domains)

    assert len(derivations) == 2
