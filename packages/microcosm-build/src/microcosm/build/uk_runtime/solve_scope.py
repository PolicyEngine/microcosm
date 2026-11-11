"""The UK country resources the dense solve never reads (microcosm#1301).

A local release is built twice at most. The first build's failing cells and
areas are signed in gate-only registers, and the second build, at the
sign-off's head, must reuse the first build's solve and re-run only the
calibrated gates, the export, the package and certification. Every node keys
on the country spec, so the nodes the solve depends on key on the *solve
fingerprint*: every packaged UK resource but :data:`UK_GATE_ONLY_RESOURCES`.

The calibrated gate node keeps the full fingerprint, so a sign-off re-keys
it. Membership is a claim that no target compilation, solve, spine or
generator module reads the file: ``test_uk_solve_scope.py`` scans the source
for readers and perturbs each file against every solve kernel's key. It
does not say whether a sign-off may edit the file; that is the sign-off
rule's question. The degenerate register stays out, since the release
contract pins its digest.

The names are literals rather than imports of the register constants
(``weighted_integrity``), which import the gate machinery; a test holds the
two in lockstep with the ``gates.json`` resource parameters.
"""

from __future__ import annotations

from microcosm.build import country_spec

__all__ = ["UK_GATE_ONLY_RESOURCES", "uk_solve_fingerprint"]

UK_GATE_ONLY_RESOURCES: frozenset[str] = frozenset(
    {
        # uk_target_fit: national rows signed for a release line.
        "target_fit_reviewed_exclusions.json",
        # uk_local_target_fit: local cells, family/area_code/metric.
        "local_target_fit_reviewed_exclusions.json",
        # uk_input_mass_parity: per-reference input-mass columns.
        "input_mass_reviewed_exclusions.json",
        # uk_local_area_support: both blocks; the reference generator keeps
        # its own deferred areas and bound families since the gate-only
        # area-support change (microcosm#1301).
        "local_area_support_exclusions.json",
    }
)


def uk_solve_fingerprint(spec: country_spec.ResolvedCountrySpec | None = None) -> str:
    """The UK country fingerprint without the gate-only registers.

    ``spec`` defaults to the packaged UK spec, looked up when called so a
    test can substitute one.
    """

    resolved = country_spec.load_country_spec("uk") if spec is None else spec
    return resolved.fingerprint_excluding(UK_GATE_ONLY_RESOURCES)
