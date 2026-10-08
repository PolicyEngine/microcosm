"""Public build-emitter handle, with telemetry and completed-graph operations.

The original telemetry import remains supported for #1099 callers. Service
composition and authentication are independent of that compatibility API.
"""

from .telemetry_emitter import (
    LocalTelemetryEmitter as LocalBuildEmitter,
)
from .telemetry_emitter import (
    start_local_telemetry_emitter_service as start_local_build_emitter_service,
)

__all__ = ["LocalBuildEmitter", "start_local_build_emitter_service"]
