# Frame checkpoint formats

`microcosm.build.frame_checkpoint.write_frame_checkpoint` writes deterministic,
local HDF5 checkpoints one column at a time. `load_frame_checkpoint` restores
tables, indexes and axis names, entity schema, typed weights, strata, mass history
and caller-owned external JSON metadata. These are build artifacts, not release
certification or an untrusted interchange format.

The writer selects the earliest required schema version:

| Version | Required feature |
| --- | --- |
| 2 | Existing NumPy, scalar object and pandas string storage |
| 3 | At least one nullable pandas `boolean`, without nullable integer storage |
| 4 | At least one nullable pandas integer column, index or strata series |

Version 4 accepts `Int8`, `Int16`, `Int32`, `Int64`, `UInt8`, `UInt16`, `UInt32`
and `UInt64`, matching the graph store's pandas integer widths. Its
`nullable_integer_v1` encoding records the exact pandas dtype and a boolean
`has_null_mask`. HDF5 `values` use the corresponding signed or unsigned NumPy
integer dtype. If any values are null, an aligned one-dimensional uint8
`null_mask` contains 0/1 indicators. Masked storage values are canonical zero;
an observed zero has mask 0. Integers never pass through floating-point storage,
including IDs above 2**53. Existing nullable booleans retain their version 3
encoding unchanged, including when carried in a version 4 checkpoint.

The reader rejects mismatched encodings, integer widths, malformed or unexpected
masks and nonzero masked storage. Versions 2 and 3 cannot declare nullable
integer storage; version 4 requires an integer spec. Readers predating version 4
refuse it explicitly instead of silently converting values. Frames that do not
need version 4 retain the previous version 2 or 3 bytes in the same runtime.
HDF5 whole-file bytes are runtime/platform dependent; no cross-platform byte
identity is claimed.

The metadata API is unchanged: `metadata=` stores external JSON. Intrinsic
`Frame.metadata` must be bound separately by the caller and passed through
`load_frame_checkpoint(..., frame_metadata=...)`, for example after validating
the graph's typed US context artifact. Merely loading the HDF5 file does not
authenticate that separate context.

Other extension types, including nullable floats and Arrow-backed integer
arrays, remain unsupported. This change does not expand graph column declaration
types, modify weights or authorize a release.
