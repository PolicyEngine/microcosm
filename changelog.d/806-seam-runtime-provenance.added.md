The UK calibration seam records the calibrating environment's package
versions (`python`, `policyengine-core`, `policyengine-uk`, and the microcosm
shards) into the signed diagnostics build block. The release assembler pins
manifests only from that authenticated provenance — a `--runtime-version`
override may re-assert a signed value but never replace it — so a release
can no longer describe an invented assembly environment.
