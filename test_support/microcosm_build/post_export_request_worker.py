"""What a spawned pool worker receives of the post-export requests sent to it.

This module imports nothing beyond the standard library, so a spawned process
that runs it loads only what unpickling the requests needs. The release tool
(``tools/build_us_fiscal_refresh_release.py``) is loaded by file path, outside
``sys.modules``; a type defined there cannot be unpickled in such a process.
"""

from __future__ import annotations

import multiprocessing
import os


def requests_as_a_spawned_worker_receives_them(requests):
    """Return the unpickled requests with where this process got their types."""
    types = {type(request) for request in requests}
    types |= {
        type(request.reform) for request in requests if request.reform is not None
    }
    return {
        "requests": requests,
        "process_id": os.getpid(),
        "start_method": multiprocessing.get_start_method(),
        "type_modules": sorted({kind.__module__ for kind in types}),
    }
