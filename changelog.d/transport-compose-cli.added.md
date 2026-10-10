Add the transport skeleton composer, its local-only build driver and
`tools/build_transport.py`. A country's `transport_graph.json` spec resource
declares the skeleton graph; each node binds only the spec resources it
selects, every rules node runs through the one `simulate.rules_by_ref@1`
router, and later packages append branches through a typed extension point.
Explicit artifact edges into the terminal package receipt may re-key that
receipt; every other skeleton predecessor set stays fixed. The driver runs ancestor-closed checkpoints
on `<out>/.graph-store` and writes every output under `--out`; it refuses an
existing store tree that contains a link, which could redirect store writes.
Activation checks run before registry preparation, CREATE or any graph source
read. Resource selections must name present resources; value and JSON
selections must resolve existing paths to non-null selected values. Parameter
resolution shares these checks: skeleton JSON selections require objects;
entitlement JSON selections also accept lists, and value selections reject
objects. Optional null fields within selected JSON remain allowed.
Preflight also checks the dependent-child age limit, mandatory and
selected reference activation rows, required engine commit and wheel pin
presence, and every null scenario knob. The receipt kernel's shared validator
checks contract fields, required text, program and exclusion structure, and
declared rates; checks needing person data, weights or target surfaces run during
receipt assignment. Calibration ancestry refuses hold-out selections and sources,
benefit-unit engine references, and graph digests.
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
