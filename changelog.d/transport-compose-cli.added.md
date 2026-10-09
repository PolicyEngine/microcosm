Add the transport skeleton composer, its local-only build driver and
`tools/build_transport.py`. A country's `transport_graph.json` spec resource
declares the skeleton graph; each node binds only the spec resources it
selects, every rules node runs through the one `simulate.rules_by_ref@1`
router, and later packages append branches through a typed extension point
that cannot re-key the skeleton. The driver runs ancestor-closed checkpoints
on `<out>/.graph-store` and writes every output under `--out`. Unresolved
country evidence is refused with each missing item named.
