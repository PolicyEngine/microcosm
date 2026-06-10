"""Rules-engine adapters.

Each adapter implements the :class:`microframe.rules.RulesEngine` protocol
for one concrete engine, importing that engine lazily so the adapter module
(and microframe itself) imports without it.
"""
