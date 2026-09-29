"""UK-owned configuration for the country-neutral staging implementation.

Two destinations, one run id. Telemetry (reviewed aggregate JSON) goes to the
staging repository under ``runs/<run_id>/``; the finished dataset bundle a
build's manifest vouches for goes to the private artifact repository under
``staged/<run_id>/`` (:mod:`microcosm.build.staging_dataset`), where it is
inspectable without being published: ``releases/`` and ``latest.json`` are
never touched by staging.
"""

from microcosm.build.staging_storage import StagingRepositoryConfig

__all__ = [
    "UK_STAGED_DATASET_PREFIX",
    "UK_STAGED_DATASET_REPOSITORY",
    "UK_STAGING_REPOSITORY",
]

UK_STAGING_REPOSITORY = StagingRepositoryConfig(
    default_repo_id="policyengine/populace-uk-staging",
    repo_id_environment_variable="POPULACE_UK_STAGING_REPO_ID",
)

UK_STAGED_DATASET_REPOSITORY = StagingRepositoryConfig(
    default_repo_id="policyengine/populace-uk-private",
    repo_id_environment_variable="POPULACE_UK_STAGED_DATASET_REPO_ID",
)

UK_STAGED_DATASET_PREFIX = "staged"
