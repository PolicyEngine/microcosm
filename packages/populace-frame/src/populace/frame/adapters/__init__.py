"""Rules-engine adapters.

Each adapter implements the :class:`populace.frame.rules.RulesEngine` protocol
for one concrete engine, importing that engine lazily so the adapter module
(and populace-frame itself) imports without it.
"""
