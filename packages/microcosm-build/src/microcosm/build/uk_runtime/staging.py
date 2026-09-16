"""UK-owned configuration for the country-neutral staging implementation."""

from microcosm.build.staging_storage import StagingRepositoryConfig

__all__ = ["UK_STAGING_REPOSITORY"]

UK_STAGING_REPOSITORY = StagingRepositoryConfig(
    default_repo_id="policyengine/populace-uk-staging",
    repo_id_environment_variable="POPULACE_UK_STAGING_REPO_ID",
)
