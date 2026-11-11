"""Country-neutral graph kernels for donor-based (transport) countries.

A transport country builds its population from another country's donor
records and calibrates it to its own published facts. Everything specific to
one destination country (node ids, target references, gate thresholds,
scenario lists) is spec data under ``microcosm/build/<cc>/``; this package
holds only the shared, country-parameterized kernels those specs compose.
"""
