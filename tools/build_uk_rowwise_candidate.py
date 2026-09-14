"""Compatibility entry point for the canonical UK full build.

The independent rowwise candidate driver is retired. Both command names now
use the same graph and default to all applicable target geographies. Request
--target-geographies country explicitly for a country-only target filter.
Legacy census-only, independent solver, and logbook options are no longer
accepted; use --help for the maintained full-build request and output options.
"""

from microcosm.build.uk_runtime.full_build_cli import main

if __name__ == "__main__":
    raise SystemExit(main())
