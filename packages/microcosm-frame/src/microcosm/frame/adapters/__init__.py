"""Rules-engine adapters.

Each adapter implements the :class:`microcosm.frame.rules.RulesEngine` protocol
for one concrete engine, importing that engine lazily so the adapter module
(and microcosm-frame itself) imports without it.
"""
