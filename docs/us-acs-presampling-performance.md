# ACS preparation before sampling

The composed survey CREATE prepares full ASEC and ACS source populations before
its existing household sampling step. This change removes two avoidable costs
from ACS preparation without changing that order or the sampling policy.

`_read_archive` constructs a lexical pandas index for each household inventory
once per archive. Its indexer is reused across person chunks. Every person is
still checked against the complete household inventory before selection, so an
orphan cannot disappear because its household was not retained. Leading zeros,
member sorting, source order, blank handling and output dtypes remain part of
the loader contract.

`_with_structural_columns` keeps the existing household validation loop and
marital/relationship error order. It writes its five integer outputs into
preallocated arrays and installs each column once. Label-based row lookup keeps
the previous supported integer-index behavior. This does not change or replace
microunit's tax-unit rules.

The bounded developer runner executes the exact seven loader/relationship
function bodies, constants and source class from this checkout, and compares
them with frozen source `669edf780d4b45e09eb4e38a9e458798b04352d2`.
It uses real pandas/NumPy and small invented ZIP members. It calls named test
bodies directly to avoid pytest discovering unrelated ancestor directories.
It does not execute the package initializer, construct a Frame, run microunit,
compile the graph or read genuine survey data. A later small actual graph
lifecycle must qualify this source for integration.

The timing witness is limited to 10,000 invented households and 30,000 people,
with 120 CPU seconds, 180 wall seconds and a 2 GiB resident-memory watchdog.
These measurements describe the tested functions, not a forecast of real build
time. Source preparation still includes full-table materialization and actual
microunit construction.

The existing implementation inventory binds all bytes of `acs_pums.py` in
`acs_codec_2024`, `acs_housing_universe_2024`, `assembly_prepare`,
`composed_population_v1` and `composed_asec_binding_v1`. Their source identities
and dependent graph keys must change. This change neither relabels old cache
entries nor claims their reuse; the scoped checks validate the unchanged
dependency contract but do not measure new graph keys.
