"""Check CLI flags against add_argument declarations without importing an engine."""

from __future__ import annotations

import ast
import sys
from collections.abc import Iterable


def declared_flags(source: str) -> set[str]:
    """Read literal option names from the tool's argument parser."""
    return {
        argument.value
        for node in ast.walk(ast.parse(source))
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "add_argument"
        for argument in node.args
        if isinstance(argument, ast.Constant)
        and isinstance(argument.value, str)
        and argument.value.startswith("-")
    }


def missing_flags(source: str, flags: Iterable[str]) -> list[str]:
    """Return undeclared options, including options supplied as --flag=value."""
    return sorted({flag.split("=", 1)[0] for flag in flags} - declared_flags(source))


def main() -> int:
    missing = missing_flags(sys.stdin.read(), sys.argv[1:])
    if missing:
        print("release parser does not declare: " + " ".join(missing), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
