"""Compatibility exports for #1099 telemetry clients.

Lazy loading keeps the general emitter's authentication independent of the
telemetry package's import order.
"""

from importlib import import_module

_EXPORTS = {
    "CollectorDelivery": "collector",
    "EmitterService": "runtime",
    "EventSpool": "spool",
    "PRODUCTION_COLLECTOR_URL": "constants",
    "ProcessTreeSampler": "resources",
}
__all__ = list(_EXPORTS)


def __getattr__(name):
    if name not in _EXPORTS:
        raise AttributeError(name)
    return getattr(import_module(f"{__name__}.{_EXPORTS[name]}"), name)
