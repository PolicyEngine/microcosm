Add the transport skeleton composer, its local-only build driver and
`tools/build_transport.py`. A country's `transport_graph.json` spec resource
declares the skeleton graph; each node binds only the spec resources it
selects, every rules node runs through the one `simulate.rules_by_ref@1`
router, and later packages append branches through a typed extension point
that cannot re-key the skeleton. The driver runs ancestor-closed checkpoints
on `<out>/.graph-store` and writes every output under `--out`. Activation checks every selected resource and reference set and every null
scenario knob before donor preparation. Calibration ancestry refuses hold-out
selections and sources, benefit-unit engine references, and graph digests.
Selected resource lists remain literal data. The driver writes deterministic
HDF5 container bytes while readback continues to hash the complete file.
CREATE inventories are cached by implementation, parameters and source bytes;
a cold build runs CREATE twice, and a warm inventory does not run the probe.
Cold `--resume require` builds refuse before preparation. Local spec directories
are protected from output overlap, node ids must be safe filenames, and
extensions receive independent spec copies.
Extensions can declare new sources and pre-export checkpoints, and use
`transport_rules_node` to obtain a binding's engine reference and period.
The locality property is parametrized over every selected resource.
