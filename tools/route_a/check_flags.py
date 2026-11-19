"""Check CLI flags against add_argument declarations without importing an engine.

``check_flags.py FLAG...`` reads a tool's source on stdin and exits 1 naming
every flag none of its ``add_argument`` calls declares. ``--in-function NAME``
first narrows the source to the function ``NAME``, for a tool with two parsers:
the release tool's certification parser is ``_certify_parser``, and its build
parser declares flags of the same names (microcosm#1585 item 11).
"""

from __future__ import annotations

import ast
import sys
from collections.abc import Iterable


def declared_flags(source: str, *, function: str | None = None) -> set[str]:
    """Read literal option names from the tool's argument parser.

    With ``function``, only the ``add_argument`` calls inside that function
    count; a source without it declares nothing.
    """
    tree: ast.AST = ast.parse(source)
    if function is not None:
        scopes = [
            node
            for node in ast.walk(tree)
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            and node.name == function
        ]
    else:
        scopes = [tree]
    return {
        argument.value
        for scope in scopes
        for node in ast.walk(scope)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "add_argument"
        for argument in node.args
        if isinstance(argument, ast.Constant)
        and isinstance(argument.value, str)
        and argument.value.startswith("-")
    }


def missing_flags(
    source: str, flags: Iterable[str], *, function: str | None = None
) -> list[str]:
    """Return undeclared options, including options supplied as --flag=value."""
    return sorted(
        {flag.split("=", 1)[0] for flag in flags}
        - declared_flags(source, function=function)
    )


def main() -> int:
    flags = sys.argv[1:]
    function = None
    if flags[:1] == ["--in-function"]:
        if len(flags) < 2:
            print("usage: check_flags.py [--in-function NAME] FLAG...", file=sys.stderr)
            return 2
        function, flags = flags[1], flags[2:]
    missing = missing_flags(sys.stdin.read(), flags, function=function)
    if missing:
        parser = function if function else "release parser"
        print(f"{parser} does not declare: " + " ".join(missing), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
