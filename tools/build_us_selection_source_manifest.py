"""CLI wrapper: distill a source H5 into a committed selection-source manifest.

See microcosm#328. The manifest names a frozen support's stable record
identities so the sparse US default is reproducible without re-downloading the
source H5.
"""

from microcosm.build.us_runtime.warm_start_selection import main

if __name__ == "__main__":
    main()
