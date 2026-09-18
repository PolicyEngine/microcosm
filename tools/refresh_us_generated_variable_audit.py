#!/usr/bin/env python3
"""Re-derive the PolicyEngine-US generated-variable audit from installed wheels.

The frame adapter reads PolicyEngine-US variable metadata by parsing ordinary
``class ...(Variable)`` declarations out of the installed source tree, without
importing the engine.  Some of the default system's variables are not declared
that way: they are built at construction time by generator functions
(``create_50_state_variables``, the PUF leaf factory, the state MFS factory,
the Michigan surtax reform, and — since policyengine-us 2.x — spm-calculator's
``build_policyengine_variables``).  Those variables carry a compact audited
snapshot in ``microcosm.frame.adapters.policyengine_us``, fingerprinted against
every source and activation file that produced it, so an unreviewed wheel fails
closed instead of silently omitting a formula-owned output.

This tool is that snapshot's generator.  It:

1. imports the installed engine once, and diffs the constructed default
   system's variables against the statically parsed class declarations — the
   same subtraction the adapter's contract describes;
2. reads each remaining variable's metadata (entity, dtype, period, whether the
   engine owns a formula for it) off the engine object rather than off prose;
3. resolves each generated variable's *defining* module to a real file through
   :func:`inspect.getfile`, and refuses to write a snapshot whose defining files
   are not all in the reviewed pin set — a new generator surface is a review
   event, not a silent digest move;
4. fingerprints the pin set (defining files plus the reviewed *activation*
   files that invoke the generators) in both distributions; and
5. rewrites the sentinel-delimited block in the adapter, or with ``--check``
   refuses when the checked-in block differs.

Never hand-edit the block this tool owns: run

    uv run python tools/refresh_us_generated_variable_audit.py

with the engine extra installed, and review the diff.
"""

from __future__ import annotations

import argparse
import collections
import inspect
import sys
from collections.abc import Iterable, Mapping
from hashlib import sha256
from importlib.metadata import distribution
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
ADAPTER_PATH = (
    REPOSITORY_ROOT
    / "packages"
    / "microcosm-frame"
    / "src"
    / "microcosm"
    / "frame"
    / "adapters"
    / "policyengine_us.py"
)
BEGIN_SENTINEL = "# --- BEGIN GENERATED VARIABLE AUDIT ---"
END_SENTINEL = "# --- END GENERATED VARIABLE AUDIT ---"

#: Files that *activate* generation without declaring the variables themselves:
#: the system constructor that calls every generator, the reform registry that
#: routes the Michigan surtax, the star-export surface the generators are
#: written against, and ``spm.py``, whose ``DATASET_SOURCE_INPUTS`` declares
#: which generated variables a producer delivers from the source despite a
#: fallback formula (the snapshot records those as input leaves).  Defining
#: files are derived, never listed; these are the reviewed activation surface,
#: so adding one is a deliberate edit here.
ACTIVATION_FILES: Mapping[str, tuple[str, ...]] = {
    "policyengine-us": ("model_api.py", "reforms/reforms.py", "spm.py", "system.py"),
    "spm-calculator": (),
}

#: Distribution name -> importable package directory name.
PACKAGE_DIRECTORIES: Mapping[str, str] = {
    "policyengine-us": "policyengine_us",
    "spm-calculator": "spm_calculator",
}

#: The snapshot literal each distribution's pins are emitted into.
PIN_SYMBOLS: Mapping[str, tuple[str, str]] = {
    "policyengine-us": ("_GENERATED_SOURCE_VERSION", "_GENERATED_SOURCE_SHA256"),
    "spm-calculator": (
        "_GENERATED_SPM_SOURCE_VERSION",
        "_GENERATED_SPM_SOURCE_SHA256",
    ),
}


class AuditError(RuntimeError):
    """The audit cannot be derived, or the checked-in snapshot is stale."""


def _package_root(name: str) -> Path:
    return Path(distribution(name).locate_file(PACKAGE_DIRECTORIES[name]))


def _version(name: str) -> str:
    return distribution(name).version


def _declared_variable_names(package_root: Path) -> set[str]:
    """Names the adapter's static parser already sees as ordinary classes."""

    sys.path.insert(0, str(REPOSITORY_ROOT / "packages" / "microcosm-frame" / "src"))
    from microcosm.frame.adapters._policyengine_us_source_index import (
        _index_policyengine_us_sources,
    )

    return set(_index_policyengine_us_sources(package_root / "variables").definitions)


def _generated_variables() -> dict[str, object]:
    """The constructed default system minus the statically declared classes."""

    package_root = _package_root("policyengine-us")
    declared = _declared_variable_names(package_root)
    from policyengine_us.system import system

    missing = sorted(declared - set(system.variables))
    if missing:
        raise AuditError(
            "Parsed PolicyEngine-US variable classes that the constructed "
            f"default system does not carry: {missing}."
        )
    return {
        name: variable
        for name, variable in system.variables.items()
        if name not in declared
    }


def _formula_owned(variable: object) -> bool:
    """Whether the engine computes this variable at every period.

    Mirrors the static parser's ``always_computed``: an undated ``formula``,
    or a non-empty ``adds`` / ``subtracts``.  A *dated* formula would need the
    snapshot to carry formula starts, which it deliberately does not model, so
    :func:`_derive_groups` refuses one rather than flattening it.
    """

    formulas = getattr(variable, "formulas", None) or {}
    return (
        bool(formulas)
        or bool(getattr(variable, "adds", None))
        or bool(getattr(variable, "subtracts", None))
    )


def _dataset_source_inputs(generated: Mapping[str, object]) -> frozenset[str]:
    """The engine's declared source-deliverable inputs, validated like the adapter.

    ``policyengine_us.spm.DATASET_SOURCE_INPUTS`` overrides formula presence:
    a producer must retain its observed value for these names, so the
    snapshot records them as input leaves.  The declaration is read off the
    installed engine through the same validation the adapter's live path
    applies, and every declared name must be one of the generated variables
    this snapshot covers — a declared ordinary class would need the static
    parser to learn the rule, which is a review event, not a silent regenerate.
    """

    sys.path.insert(0, str(REPOSITORY_ROOT / "packages" / "microcosm-frame" / "src"))
    import policyengine_us

    from microcosm.frame.adapters.policyengine_us import (
        _engine_dataset_source_inputs,
    )

    declared = _engine_dataset_source_inputs(policyengine_us)
    unknown = sorted(declared - set(generated))
    if unknown:
        raise AuditError(
            "policyengine_us.spm.DATASET_SOURCE_INPUTS names variable(s) the "
            f"generated-variable snapshot does not cover: {unknown}. Extend the "
            "static parser before regenerating."
        )
    return declared


_EPOCH_START = "0001-01-01"
_DTYPE_BY_VALUE_TYPE = {float: "float", int: "int", bool: "bool", str: "str"}
_PERIOD_BY_DEFINITION = {"year": "year", "month": "month"}


def _defining_file(variable: object) -> Path:
    return Path(inspect.getfile(type(variable))).resolve()


def _relative_to_distribution(path: Path) -> tuple[str, str]:
    """Return ``(distribution name, package-relative posix path)``."""

    for name in PACKAGE_DIRECTORIES:
        root = _package_root(name).resolve()
        if path.is_relative_to(root):
            return name, path.relative_to(root).as_posix()
    raise AuditError(
        f"A generated PolicyEngine-US variable is defined outside the audited "
        f"distributions: {path}. Pin its distribution before regenerating."
    )


def _derive_groups(
    variables: Mapping[str, object],
) -> tuple[tuple[tuple[str, ...], str, str, str, bool], ...]:
    """Bucket the generated variables by defining file and metadata."""

    buckets: dict[tuple[str, str, str, str, str, bool], list[str]] = (
        collections.defaultdict(list)
    )
    source_inputs = _dataset_source_inputs(variables)
    for name, variable in sorted(variables.items()):
        formulas = getattr(variable, "formulas", None) or {}
        dated = [start for start in formulas if start != _EPOCH_START]
        if dated:
            raise AuditError(
                f"Generated variable {name!r} has dated formulas {dated}; the "
                "audited snapshot models only undated formula ownership. "
                "Extend the snapshot before regenerating."
            )
        dtype = _DTYPE_BY_VALUE_TYPE.get(variable.value_type, "str")
        period = _PERIOD_BY_DEFINITION.get(variable.definition_period, "point")
        source, relative = _relative_to_distribution(_defining_file(variable))
        buckets[
            (
                source,
                relative,
                variable.entity.key,
                dtype,
                period,
                name not in source_inputs and _formula_owned(variable),
            )
        ].append(name)
    return tuple(
        (tuple(sorted(names)), entity, dtype, period, owned)
        for (
            _source,
            _relative,
            entity,
            dtype,
            period,
            owned,
        ), names in sorted(buckets.items())
    )


def _pin_set(variables: Mapping[str, object]) -> dict[str, dict[str, str]]:
    """Digest every defining file plus the reviewed activation files."""

    relative_paths: dict[str, set[str]] = {
        name: set(files) for name, files in ACTIVATION_FILES.items()
    }
    for variable in variables.values():
        name, relative = _relative_to_distribution(_defining_file(variable))
        relative_paths[name].add(relative)
    pins: dict[str, dict[str, str]] = {}
    for name, paths in relative_paths.items():
        root = _package_root(name)
        pins[name] = {}
        for relative in sorted(paths):
            source = root / relative
            if not source.is_file():
                raise AuditError(f"Audited {name} source is unavailable: {source}.")
            pins[name][relative] = sha256(source.read_bytes()).hexdigest()
    return pins


def _render_mapping(symbol: str, pins: Mapping[str, str]) -> str:
    lines = [f"{symbol}: dict[str, str] = {{"]
    for relative, digest in pins.items():
        entry = f'    "{relative}": "{digest}",'
        if len(entry) <= 88:
            lines.append(entry)
        else:
            lines.append(f'    "{relative}": (')
            lines.append(f'        "{digest}"')
            lines.append("    ),")
    lines.append("}")
    return "\n".join(lines)


def _render_names(names: Iterable[str]) -> list[str]:
    """Emit a name tuple as a wrapped ``"...".split()`` literal."""

    names = list(names)
    if len(names) == 1:
        return [f'        ("{names[0]}",),']
    chunks: list[str] = []
    current = ""
    for name in names:
        candidate = f"{current}{name} "
        if len(candidate) > 62 and current:
            chunks.append(current)
            current = f"{name} "
        else:
            current = candidate
    chunks.append(current.rstrip())
    if len(chunks) == 1:
        return [f'        tuple("{chunks[0]}".split()),']
    lines = ["        tuple("]
    lines.extend(f'            "{chunk}"' for chunk in chunks[:-1])
    lines.append(f'            "{chunks[-1]}".split()')
    lines.append("        ),")
    return lines


def _render_groups(
    groups: tuple[tuple[tuple[str, ...], str, str, str, bool], ...],
) -> str:
    lines = [
        "_GENERATED_VARIABLE_GROUPS: tuple["
        "tuple[tuple[str, ...], str, str, str, bool], ..."
        "] = (",
    ]
    for names, entity, dtype, period, owned in groups:
        lines.append("    (")
        lines.extend(_render_names(names))
        lines.append(f'        "{entity}",')
        lines.append(f'        "{dtype}",')
        lines.append(f'        "{period}",')
        lines.append(f"        {owned},")
        lines.append("    ),")
    lines.append(")")
    return "\n".join(lines)


def render_block() -> str:
    variables = _generated_variables()
    pins = _pin_set(variables)
    groups = _derive_groups(variables)
    counted = sum(len(names) for names, *_rest in groups)
    if counted != len(variables):
        raise AuditError(
            f"Rendered {counted} audited names for {len(variables)} generated "
            "variables."
        )
    sections = [
        BEGIN_SENTINEL,
        "# Regenerate with:",
        "#   uv run python tools/refresh_us_generated_variable_audit.py",
        "# The block below is generated from the installed wheels; never edit a",
        f"# digest or a name by hand. {counted} default-system variables are",
        "# created outside ordinary top-level ``class ...(Variable)``",
        "# declarations, so the snapshot is tied to every source and activation",
        "# surface that produced it: a changed wheel fails closed until this",
        "# audit is refreshed, and never silently omits a newly generated",
        "# formula-owned output. A name policyengine_us.spm.DATASET_SOURCE_INPUTS",
        "# declares source-deliverable is recorded as an input leaf despite its",
        "# fallback formula; spm.py is pinned so that declaration cannot drift.",
    ]
    for name, (version_symbol, digest_symbol) in PIN_SYMBOLS.items():
        sections.append(f'{version_symbol} = "{_version(name)}"')
        sections.append(_render_mapping(digest_symbol, pins[name]))
    sections.append(_render_groups(groups))
    sections.append(END_SENTINEL)
    return "\n".join(sections) + "\n"


def _replace_block(text: str, block: str) -> str:
    start = text.index(BEGIN_SENTINEL)
    end = text.index(END_SENTINEL) + len(END_SENTINEL) + 1
    return text[:start] + block + text[end:]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check",
        action="store_true",
        help="refuse if the checked-in audit differs from the installed wheels",
    )
    args = parser.parse_args(argv)

    block = render_block()
    text = ADAPTER_PATH.read_text()
    if BEGIN_SENTINEL not in text or END_SENTINEL not in text:
        raise AuditError(f"{ADAPTER_PATH} has no generated-variable audit sentinels.")
    updated = _replace_block(text, block)
    if updated == text:
        print("US generated-variable audit is current")
        return 0
    if args.check:
        raise SystemExit(
            "US generated-variable audit is stale for policyengine-us "
            f"{_version('policyengine-us')} / spm-calculator "
            f"{_version('spm-calculator')}: regenerate with "
            "`uv run python tools/refresh_us_generated_variable_audit.py`."
        )
    ADAPTER_PATH.write_text(updated)
    print(f"updated {ADAPTER_PATH.relative_to(REPOSITORY_ROOT).as_posix()}")
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI
    raise SystemExit(main())
