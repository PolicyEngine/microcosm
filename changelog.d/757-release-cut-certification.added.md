The release-cut certification producer (#757 B5): the 16 declared national
preflight/terminal gates get their executable home back
(`tools/certify_uk_release_cut.py` over
`uk_runtime/release_certification.py`), running as a scoped release-candidate
battery against the calibrated candidate with evidence reconstructed from
the persisted artifacts - the spine sidecar (which now carries each fitting
stage's `FitWeightRecord`s across the run boundary), the seam's diagnostics
and build record, and the per-run licensed input-mass reference. The
multi-part certification the 5413502559 audit specified composes over the
spine, seam, and release-cut reports: union to the full declared entry set,
no gap, no overlap beyond `uk_aggregate_admin`, per-part signatures and
committed-spec scoped digests, full phase coverage, a closed identity join
(spine report -> sidecar -> build record -> diagnostics -> candidate
bytes), the doctrine and its receipted overrides recorded verbatim, and the
rule-1 score receipt cross-pinned. A candidate's shippability verdict comes
only from the certification.
