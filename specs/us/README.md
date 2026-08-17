# US specification drafting location retired

The authoritative US bundle is packaged at
`packages/microcosm-build/src/microcosm/build/us/spec/` and is generated from
the constants-era authority surfaces by
`tools/generate_us_bundle_from_constants.py` during the F0 migration.

Do not author a second copy here. Run the generator with `--check` to verify
that the packaged bundle and its generated legacy compatibility projections
match the current migration source.
