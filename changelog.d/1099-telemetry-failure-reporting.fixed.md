Report failed UK builds and active stages as failed, isolate telemetry in automated tests, and keep local telemetry setup errors from aborting builds. Rename the initial telemetry migration module so wheel validation accepts it without changing its revision identifier.

Report nonzero US dry-run results as failures, delay telemetry delivery retries during shutdown, and retain reaped workers' CPU usage in the Linux fallback sampler.
