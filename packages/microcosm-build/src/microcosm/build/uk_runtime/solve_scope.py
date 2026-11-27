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

The land side of the stock cell table (microcosm#1524) leaves the
fingerprint too, as :data:`UK_POST_SOLVE_RESOURCES`: its building costs, its
rollups by authority and constituency, and its provenance. No solve module
reads them, and a rebuild of the land, which the level's inputs from ONS
will prompt, must not re-key the solve. The stock cells' value block stays
in, since the frame's home values will be read from it. Membership is the
same claim as for the gate-only registers, and the same suite tests it.

The declared target sets (``spec/target_sets.yaml``, microcosm#1759) stay in
the fingerprint, but by their content, as :data:`UK_CONTENT_KEYED_RESOURCES`.
They are a typed resource of the spec bundle, and the spec engine separates a
typed resource's normative content from its documentation. The selections
read the clauses, levels, exclusions and rules; no node reads a description
or a rationale, and most of the file is that prose. So the fingerprint takes
the hash of the file's normative projection
(``ResolvedCountrySpec.content_hashes``): a reworded rationale merged between
the two builds of a release does not re-key the solve, and a changed clause
re-keys every node that keys on the fingerprint. The calibrated gate node and
the release manifests keep the byte hash. The other typed resources still
enter by their bytes; each can join this set after a review of what its
surfaces hold.
"""

from __future__ import annotations

from microcosm.build import country_spec

__all__ = [
    "UK_CONTENT_KEYED_RESOURCES",
    "UK_GATE_ONLY_RESOURCES",
    "UK_POST_SOLVE_RESOURCES",
    "uk_solve_fingerprint",
]

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

UK_POST_SOLVE_RESOURCES: frozenset[str] = frozenset(
    {
        # The stock cell table's building costs and central land per cell.
        "stock_cells_structure.json",
        # Land and stock value by authority and constituency, every variant.
        "la_land_shares.json",
        "constituency_land_totals.json",
        # The inputs, level, checks and measurements of that build.
        "stock_cells.provenance.json",
    }
)


#: The solve-scoped typed resources that enter the fingerprint by the hash of
#: their normative projection, which leaves out their documentation (see the
#: module docstring).
UK_CONTENT_KEYED_RESOURCES: frozenset[str] = frozenset({"spec/target_sets.yaml"})


def uk_solve_fingerprint(spec: country_spec.ResolvedCountrySpec | None = None) -> str:
    """The UK country fingerprint without the gate-only and post-solve resources.

    The declared target sets enter by their content
    (:data:`UK_CONTENT_KEYED_RESOURCES`). ``spec`` defaults to the packaged
    UK spec, looked up when called so a test can substitute one.
    """

    resolved = country_spec.load_country_spec("uk") if spec is None else spec
    return resolved.fingerprint_excluding(
        UK_GATE_ONLY_RESOURCES | UK_POST_SOLVE_RESOURCES,
        by_content=UK_CONTENT_KEYED_RESOURCES,
    )
