"""Git options for the throwaway repositories tests create."""

from __future__ import annotations

from collections.abc import Mapping
from types import MappingProxyType

#: ``git -c`` options that keep background maintenance out of a test repository.
#:
#: After its own work, ``git commit`` starts ``git maintenance run --auto``.
#: That process takes ``.git/objects/maintenance.lock``, detaches, and removes
#: the lock only when it finishes, which can be after the commit has returned.
#: A test that lists or hashes ``.git`` straight after a commit can therefore
#: list a lock file that is gone by the time it is read.
#:
#: ``maintenance.auto=false`` stops git starting that process, and so does
#: ``gc.auto=0`` when ``maintenance.auto`` is unset. If a maintenance run
#: starts all the same, ``maintenance.autoDetach=false`` keeps it in the
#: foreground, so it releases its lock before the command returns. Read from
#: git 2.55.0: ``prepare_auto_maintenance`` in ``run-command.c`` and
#: ``maintenance_run_tasks`` in ``builtin/gc.c``.
#:
#: Pass them before the subcommand: ``["git", *NO_BACKGROUND_MAINTENANCE, ...]``.
NO_BACKGROUND_MAINTENANCE: tuple[str, ...] = (
    "-c",
    "maintenance.auto=false",
    "-c",
    "gc.auto=0",
    "-c",
    "maintenance.autoDetach=false",
)

#: The same settings as ``GIT_CONFIG_COUNT`` environment variables, for git
#: that the code under test runs from ``os.environ``, where no ``-c`` option
#: reaches it. They override any ``GIT_CONFIG_COUNT`` entries already in the
#: environment; a ``GIT_CONFIG_PARAMETERS`` there (an outer ``git -c``) still
#: applies, after them. Code that strips git's repository-local variables
#: (``git rev-parse --local-env-vars``) drops them.
NO_BACKGROUND_MAINTENANCE_ENV: Mapping[str, str] = MappingProxyType(
    {
        "GIT_CONFIG_COUNT": str(len(NO_BACKGROUND_MAINTENANCE) // 2),
        **{
            f"GIT_CONFIG_{part}_{index}": value
            for index, setting in enumerate(NO_BACKGROUND_MAINTENANCE[1::2])
            for part, value in zip(("KEY", "VALUE"), setting.split("=", 1), strict=True)
        },
    }
)
