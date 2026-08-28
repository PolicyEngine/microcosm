Give the synthetic import-entry CLI tests a subprocess budget that covers the
rules-engine import their CLI pays before doing any work, so they stop timing
out on contended runners.
