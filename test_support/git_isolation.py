"""Keep the suite's git commands in the repositories its tests name.

Git exports an absolute ``GIT_DIR`` to the hooks, ``rebase --exec`` commands
and shell aliases it runs in a linked worktree. ``GIT_DIR`` outranks ``-C``
and the working directory, so under one a fixture's ``git init``, ``commit``
or ``update-ref`` in a temporary directory runs against the developer's
checked-out branch instead. The root ``conftest.py`` calls
:func:`drop_inherited_git_repository` before any test runs.
"""

from __future__ import annotations

import subprocess
from collections.abc import MutableMapping

#: These carry ``git -c`` options. Git keeps them when it runs a command in
#: another repository, and so does the suite.
GIT_CONFIG_CARRIERS = frozenset({"GIT_CONFIG_PARAMETERS", "GIT_CONFIG_COUNT"})


def drop_inherited_git_repository(environ: MutableMapping[str, str]) -> list[str]:
    """Remove the variables that pick git's repository; return their names.

    The variables are the ones the installed git calls repository-local
    (``git rev-parse --local-env-vars``), less GIT_CONFIG_CARRIERS. Without
    git there is nothing for a fixture to redirect, and nothing is removed.
    """

    try:
        listed = subprocess.run(
            ["git", "rev-parse", "--local-env-vars"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.split()
    except (OSError, subprocess.CalledProcessError):
        return []
    dropped = sorted(
        name for name in listed if name not in GIT_CONFIG_CARRIERS and name in environ
    )
    for name in dropped:
        del environ[name]
    return dropped
