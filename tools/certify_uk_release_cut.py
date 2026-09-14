"""Materialize unsigned certification readiness from the canonical UK full graph.

Use --graph-manifest, --graph-store, --candidate-h5 and --certification-json.
The historical independent national/seam battery CLI is retired. Native and
matched-size scorecards enter as declared sources of the full graph build.
"""

from microcosm.build.uk_runtime.full_certification import main

if __name__ == "__main__":
    raise SystemExit(main())
