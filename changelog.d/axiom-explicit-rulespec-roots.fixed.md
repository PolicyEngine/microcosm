Require every `AxiomEngine` caller to provide a non-empty explicit sequence of
canonical RuleSpec roots and forward that exact authority boundary to the
current Axiom dense loader. The adapter no longer relies on the retired
implicit-root interface, and its canonical-layout fixture plus engine-free
regressions pin missing, empty, scalar, and forwarded-root behavior. Root and
module validation errors also propagate instead of being mistaken for an
optional entity with no derived rules.
