"""Build a UK rowwise candidate in either release role through the graph driver.

Both roles are served by ``microcosm-build-uk``
(:mod:`microcosm.build.uk_runtime.full_build_cli`): ``--release-role dense``
builds the K-clone joint national + local surface through the graph, and
``--release-role national`` dispatches to the retained calibration seam
(:mod:`microcosm.build.uk_runtime.national_role`). This historical entry point
re-exports the driver's ``main`` (microcosm#901 phase 4).
"""

from microcosm.build.uk_runtime.full_build_cli import main

if __name__ == "__main__":
    raise SystemExit(main())
