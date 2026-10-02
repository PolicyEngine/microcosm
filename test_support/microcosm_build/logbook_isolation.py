"""Discover literal Logbook aliases without triggering foreign lazy imports."""

import sys
from types import ModuleType


def logbook_urlopen_holders(real_urlopen: object) -> list[ModuleType]:
    """Return imported modules binding the Logbook HTTP function directly.

    The CLI imports ``urlopen`` by value, so the isolation fixture must patch
    aliases as well as the defining module. Inspect module globals rather than
    looking up attributes: unrelated modules may resolve missing attributes
    through ``__getattr__`` hooks that import or initialize native libraries.
    """
    return [
        module
        for module in list(sys.modules.values())
        if isinstance(module, ModuleType)
        and vars(module).get("urlopen") is real_urlopen
    ]
