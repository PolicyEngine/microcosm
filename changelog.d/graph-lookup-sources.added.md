Add an explicit raw-byte mode to the graph source-codec registry:
`SourceCodecRegistry.register_bytes`/`load_bytes` over a `SourceBytesCodec`
callable type, a built-in file-only `raw-bytes-v1` loader bounded at 64 MiB,
and `load_source_bytes`. A typed-artifact import kernel can now decode a real
lookup source (an NPZ of ratios, a crosswalk CSV) without a Frame codec
registered as a pretence that a lookup table is a population. Frame-mode
`register`/`load` are unchanged and still require a Frame; codec names stay
unique across both modes, and reading a codec in the other mode raises.
