"""Git options for the throwaway repositories tests create."""

from __future__ import annotations

#: ``git -c`` options that keep background maintenance out of a test repository.
#:
#: After its own work, ``git commit`` starts ``git maintenance run --auto``.
#: That process takes ``.git/objects/maintenance.lock``, detaches, and removes
#: the lock only when it finishes, which can be after the commit has returned.
#: A test that lists or hashes ``.git`` straight after a commit can therefore
#: list a lock file that is gone by the time it is read.
#:
#: ``maintenance.auto=false`` stops git starting that process, and so does
#: ``gc.auto=0`` when ``maintenance.auto`` is unset. ``gc.autoDetach=false``
#: (git's fallback for ``maintenance.autoDetach``) keeps a maintenance run that
#: does start in the foreground, so it releases its lock before the command
#: returns. Read from git 2.55.0: ``prepare_auto_maintenance`` in
#: ``run-command.c`` and ``maintenance_run_tasks`` in ``builtin/gc.c``.
#:
#: Pass them before the subcommand: ``["git", *NO_BACKGROUND_MAINTENANCE, ...]``.
NO_BACKGROUND_MAINTENANCE: tuple[str, ...] = (
    "-c",
    "maintenance.auto=false",
    "-c",
    "gc.auto=0",
    "-c",
    "gc.autoDetach=false",
)
