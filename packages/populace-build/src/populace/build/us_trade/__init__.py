"""US import-entry unit family: ledger-margin ingest and entry synthesis.

The synthetic import-entry family (PolicyEngine/populace#615) is built from
the ledger and engine legs only: official Census/CBP series are admitted as
ledger-contract facts with full source provenance, and entries are generated
from — and calibrated exactly back to — those margins. No entry-level
microdata exists for imports; every distributional assumption beyond the
margins is explicit, documented, and labeled synthetic.

Modules:

- :mod:`populace.build.us_trade.census_imports` — Census International Trade
  API ingest: HTS-10 × country × month customs value, calculated duty,
  dutiable value, and quantities, with retrieval manifests.
- :mod:`populace.build.us_trade.census_country_bridge` — the vendored Census
  Schedule C → ISO-2 bridge (fail-closed).
- :mod:`populace.build.us_trade.cbp_entry_stats` — CBP fiscal-year entry
  summary counts from the archived public statistics page.
- :mod:`populace.build.us_trade.import_entry_facts` — ledger
  consumer-artifact emission for the margin series.
"""

from populace.build.us_trade.cbp_entry_stats import (
    CBP_TRADE_STATS_URL,
    CbpEntryStats,
    parse_cbp_trade_stats,
)
from populace.build.us_trade.census_country_bridge import (
    CensusCountryBridge,
    load_census_country_bridge,
)
from populace.build.us_trade.census_imports import (
    CENSUS_IMPORTS_HS_ENDPOINT,
    CensusImportsMonth,
    CensusImportsPull,
    assemble_margins_table,
    fetch_imports_month,
    latest_published_month,
    month_range,
    parse_imports_response,
)
from populace.build.us_trade.import_entry_facts import (
    IMPORT_ENTRY_FACT_GRAINS,
    build_cbp_entry_fact_rows,
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
    "IMPORT_ENTRY_FACT_GRAINS",
    "assemble_margins_table",
    "build_cbp_entry_fact_rows",
    "build_import_entry_fact_rows",
    "default_generator_block",
    "fetch_imports_month",
    "latest_published_month",
    "load_census_country_bridge",
    "month_range",
    "parse_cbp_trade_stats",
    "parse_imports_response",
    "write_consumer_artifact",
]
