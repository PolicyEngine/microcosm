Resolve each declared money node's own declaration, profile selection and factor
context before a new transport run starts CREATE. Check contexts separately so
nodes that share a factor id cannot hide expiry. Preserve historical replay with
`--resume require` and skip factor requirements for identity families.
