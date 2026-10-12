"""Module execution entry point for the telemetry emitter service."""

import signal

from microcosm.build.telemetry_emitter_service.main import main

# The service runs in its own session, so an interrupt reaches it only when
# sent on purpose. End at once, as SIGTERM does, rather than print a
# KeyboardInterrupt traceback on the build's stderr.
signal.signal(signal.SIGINT, signal.SIG_DFL)
raise SystemExit(main())
