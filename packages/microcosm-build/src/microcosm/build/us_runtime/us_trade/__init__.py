"""US import-entry unit family: ledger-margin ingest and entry synthesis.

The synthetic import-entry family (PolicyEngine/microcosm#615) is built from
the ledger and engine legs only: official Census/CBP series are admitted as
ledger-contract facts with full source provenance, and entries are generated
from — and calibrated exactly back to — those margins. No entry-level
microdata exists for imports; every distributional assumption beyond the
margins is explicit, documented, and labeled synthetic.

Modules:

- :mod:`microcosm.build.us_runtime.us_trade.imdb_bulk` — the primary ingest: Census
  monthly bulk IMDB archives (full HTS-10 × country × district × rate-
  provision detail with transport splits), parsed per the archives' own
  record layouts and reconciled exactly against their control totals.
- :mod:`microcosm.build.us_runtime.us_trade.census_imports` — Census International Trade
  API ingest, retained as the independent cross-check leg: the same margin
  series fetched over a second official channel.
- :mod:`microcosm.build.us_runtime.us_trade.census_country_bridge` — the vendored Census
  Schedule C → ISO-2 bridge (fail-closed).
- :mod:`microcosm.build.us_runtime.us_trade.cbp_entry_stats` — CBP fiscal-year entry
  summary counts from the archived public statistics page.
- :mod:`microcosm.build.us_runtime.us_trade.import_entry_facts` — ledger
  consumer-artifact emission for the margin series.
"""

from microcosm.build.us_runtime.us_trade.cbp_entry_stats import (
    CBP_TRADE_STATS_URL,
    CbpEntryStats,
    parse_cbp_trade_stats,
)
from microcosm.build.us_runtime.us_trade.census_country_bridge import (
    CensusCountryBridge,
    load_census_country_bridge,
)
from microcosm.build.us_runtime.us_trade.census_imports import (
    CENSUS_IMPORTS_HS_ENDPOINT,
    CensusImportsMonth,
    CensusImportsPull,
    assemble_margins_table,
    fetch_imports_month,
    latest_published_month,
    month_range,
    parse_imports_response,
)
from microcosm.build.us_runtime.us_trade.imdb_bulk import (
    IMDB_URL_TEMPLATE,
    ImdbBulkAssembly,
    ImdbMonth,
    ImdbMonthSummary,
    assemble_bulk_margins,
    ensure_imdb_archive,
    imdb_archive_name,
    imdb_archive_url,
    latest_available_imdb_month,
    load_imdb_month,
    summarize_imdb_month,
)
from microcosm.build.us_runtime.us_trade.import_entry_facts import (
    IMDB_BULK_SOURCE_LEG,
    IMDB_DISTRICT_SOURCE_LEG,
    IMPORT_ENTRY_FACT_GRAINS,
    FactSourceLeg,
    build_cbp_entry_fact_rows,
    build_district_entry_fact_rows,
    build_import_entry_fact_rows,
    default_generator_block,
    write_consumer_artifact,
)

__all__ = [
    "CBP_TRADE_STATS_URL",
    "CENSUS_IMPORTS_HS_ENDPOINT",
    "CbpEntryStats",
    "CensusCountryBridge",
    "CensusImportsMonth",
    "CensusImportsPull",
    "FactSourceLeg",
    "IMDB_BULK_SOURCE_LEG",
    "IMDB_DISTRICT_SOURCE_LEG",
    "IMDB_URL_TEMPLATE",
    "IMPORT_ENTRY_FACT_GRAINS",
    "ImdbBulkAssembly",
    "ImdbMonth",
    "ImdbMonthSummary",
    "assemble_bulk_margins",
    "assemble_margins_table",
    "build_cbp_entry_fact_rows",
    "build_district_entry_fact_rows",
    "build_import_entry_fact_rows",
    "default_generator_block",
    "ensure_imdb_archive",
    "fetch_imports_month",
    "imdb_archive_name",
    "imdb_archive_url",
    "latest_available_imdb_month",
    "latest_published_month",
    "load_census_country_bridge",
    "load_imdb_month",
    "month_range",
    "parse_cbp_trade_stats",
    "parse_imports_response",
    "summarize_imdb_month",
    "write_consumer_artifact",
]
