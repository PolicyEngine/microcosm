Add an explicit raw-byte mode to the graph source-codec registry:
`SourceCodecRegistry.register_bytes`/`load_bytes` over a `SourceBytesCodec`
callable type, a built-in file-only `raw-bytes-v1` loader bounded at 64 MiB,
and `load_source_bytes`. A typed-artifact import kernel can now decode a real
lookup source (an NPZ of ratios, a crosswalk CSV) without a Frame codec
registered as a pretence that a lookup table is a population. Frame-mode
`register`/`load` are unchanged and still require a Frame; codec names stay
unique across both modes, and reading a codec in the other mode raises.
Each mode is enumerated by its own pair (`names`/`as_mapping`,
`bytes_names`/`as_bytes_mapping`).

Operators: `source_hash` attests a kernel's whole defining module, so any
kernel whose `implementation_hash` binds `load_source` — `example.transfer.source@1`
and `example.transfer.evaluate@1`, the only two in this repo — re-keys on this
change and recomputes from a warm store (or refuses under `resume="require"`).
The US and UK production kernels do not bind it and are unaffected.
