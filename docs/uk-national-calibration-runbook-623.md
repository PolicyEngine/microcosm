# UK calibration runbook

The separate national calibration driver is retired. Use the [UK full-build graph runbook](uk-full-build-graph.md) for current commands and evidence requirements.

The standard build calibrates all applicable geographies together. A country-only request uses `tools/build_uk_full.py --target-geographies country` with the same source, pool, solver, sizing and packaging machinery. This filter excludes regional and local target rows; it does not bypass source validation or geography integrity. K geographic copies and k output households remain separate explicit settings.
