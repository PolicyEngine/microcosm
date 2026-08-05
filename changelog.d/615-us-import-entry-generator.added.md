Add the US synthetic import-entry generator P2 (#615): a deterministic,
wall-clock-free build over the P1-authenticated margins publication that
reproduces every (HTS-10 × country × month) customs-value margin exactly
via integer largest-remainder stratification of a within-cell lognormal
size assumption; both free parameters are anchored to CBP's published
fiscal-year-to-date totals (the informal-share sigma calibration is a
disclosed proxy moment, not a statutory identification), every row is
labeled synthetic (id prefix, row-level `is_synthetic` column, schema
metadata, assumptions register), and the emitted schema satisfies a
sha-pinned engine input-surface contract.
