-- Scope the Logbook chain per dataset line (microcosm#665, PR #702).
--
-- The chain's job is tamper-evident lineage per dataset line. A single global
-- total order across scopes buys nothing and forces cross-operator
-- coordination: the caller supplies prev_row_digest at build time, so while a
-- US ladder appends concurrently, any UK row recorded against a stale global
-- tail is permanently orphaned -- its row_digest bakes in that predecessor, so
-- no reconciliation can repair it. Per-scope chains remove the coupling
-- without weakening anything: each scope keeps one genesis, one tail, and no
-- forks.
--
-- The chain key is derived from the pipeline rather than stored in a new
-- column. Two reasons: every existing row_digest stays valid (the hashed
-- payload is untouched), and the key stays *inside* that hashed payload, so a
-- row cannot be moved between chains without breaking its own digest. The live
-- data requires this choice -- the 28 archived rows span three pipeline values
-- (us-2024-release, us-pool-inc2, us-stacked-pool) with chain links crossing
-- them, so pipeline itself cannot be the key. Those three legacy names are
-- hardcoded to the grandfathered `us` scope and keep extending that mixed
-- chain forever; nothing is re-scoped retroactively.

-- Scope of a build, derived from the ratified `<country>-<dataset-line>-...`
-- pipeline convention. NULL for a pipeline that does not declare one, which
-- the CHECK below refuses: an unscoped pipeline has no chain to belong to.
CREATE OR REPLACE FUNCTION logbook.chain_scope(p_pipeline text)
RETURNS text
LANGUAGE sql
IMMUTABLE
PARALLEL SAFE
SET search_path = pg_catalog
AS $function$
    SELECT CASE
        WHEN p_pipeline IN (
            'us-2024-release',
            'us-pool-inc2',
            'us-stacked-pool'
        ) THEN 'us'
        WHEN p_pipeline ~ '^[a-z]{2}-[a-z0-9_]+(-[a-z0-9_-]+)?$'
            THEN left(p_pipeline, 2) || '/' || split_part(p_pipeline, '-', 2)
    END;
$function$;

GRANT EXECUTE ON FUNCTION logbook.chain_scope(text)
    TO logbook_writer, logbook_exporter, logbook_break_glass_admin;

ALTER TABLE logbook.builds
    ADD CONSTRAINT builds_pipeline_declares_scope
    CHECK (logbook.chain_scope(pipeline) IS NOT NULL);

-- One genesis per scope, not one table-wide.
DROP INDEX IF EXISTS logbook.builds_single_genesis;
CREATE UNIQUE INDEX builds_single_genesis_per_scope
    ON logbook.builds (logbook.chain_scope(pipeline))
    WHERE prev_row_digest IS NULL;

-- builds_unique_predecessor stays global and unchanged: row digests are
-- unique across the table, so "two rows claiming one predecessor" is a fork
-- wherever it happens. Scoping it per dataset line would weaken it.

CREATE OR REPLACE FUNCTION logbook.enforce_build_chain()
RETURNS trigger
LANGUAGE plpgsql
SECURITY DEFINER
SET search_path = pg_catalog, logbook, extensions
AS $function$
DECLARE
    computed_digest logbook.sha256_hex;
    existing_build logbook.builds%ROWTYPE;
    expected_predecessor logbook.sha256_hex;
    predecessor_scope text;
    new_scope text;
    build_count bigint;
    tail_count bigint;
BEGIN
    new_scope := logbook.chain_scope(NEW.pipeline);
    IF new_scope IS NULL THEN
        RAISE EXCEPTION
            'Logbook build % pipeline % does not declare a chain scope',
            NEW.build_id,
            NEW.pipeline
            USING ERRCODE = '23514';
    END IF;

    -- Transaction-scoped and per scope, so two scopes can append
    -- concurrently while appends within one scope still serialize.
    PERFORM pg_advisory_xact_lock(628, hashtext(new_scope));

    computed_digest := logbook.expected_build_row_digest(NEW);
    IF NEW.row_digest IS NOT NULL AND NEW.row_digest <> computed_digest THEN
        RAISE EXCEPTION
            'Logbook build % supplied row_digest %, expected %',
            NEW.build_id,
            NEW.row_digest,
            computed_digest
            USING ERRCODE = '23514';
    END IF;
    NEW.row_digest := computed_digest;

    -- BEFORE INSERT triggers run before ON CONFLICT is resolved.  Recognize an
    -- exact replay here so `resolution=ignore-duplicates` remains idempotent,
    -- while refusing a reused build_id whose immutable content diverges.
    SELECT *
    INTO existing_build
    FROM logbook.builds
    WHERE build_id = NEW.build_id;

    IF FOUND THEN
        IF logbook.build_hash_payload(existing_build)
                IS DISTINCT FROM logbook.build_hash_payload(NEW)
            OR existing_build.prev_row_digest
                IS DISTINCT FROM NEW.prev_row_digest
            OR existing_build.row_digest IS DISTINCT FROM NEW.row_digest
        THEN
            RAISE EXCEPTION
                'Logbook build_id % already exists with divergent content',
                NEW.build_id
                USING ERRCODE = '23505';
        END IF;
        RETURN NEW;
    END IF;

    SELECT count(*)
    INTO build_count
    FROM logbook.builds AS existing
    WHERE logbook.chain_scope(existing.pipeline) = new_scope;

    IF build_count = 0 THEN
        IF NEW.prev_row_digest IS NOT NULL THEN
            RAISE EXCEPTION
                'Logbook genesis build % for scope % must have null '
                'prev_row_digest, got %',
                NEW.build_id,
                new_scope,
                NEW.prev_row_digest
                USING ERRCODE = '23514';
        END IF;
    ELSE
        -- The scope's tail: its row with no successor in the same scope.
        SELECT count(*), min(candidate.row_digest::text)::logbook.sha256_hex
        INTO tail_count, expected_predecessor
        FROM logbook.builds AS candidate
        WHERE logbook.chain_scope(candidate.pipeline) = new_scope
        AND NOT EXISTS (
            SELECT 1
            FROM logbook.builds AS successor
            WHERE successor.prev_row_digest = candidate.row_digest
            AND logbook.chain_scope(successor.pipeline) = new_scope
        );

        IF tail_count <> 1 THEN
            RAISE EXCEPTION
                'Logbook chain for scope % is corrupt: expected one tail, '
                'found %',
                new_scope,
                tail_count
                USING ERRCODE = '23514';
        END IF;

        IF NEW.prev_row_digest IS DISTINCT FROM expected_predecessor THEN
            RAISE EXCEPTION
                'Logbook build % has prev_row_digest %, current % tail is %',
                NEW.build_id,
                NEW.prev_row_digest,
                new_scope,
                expected_predecessor
                USING ERRCODE = '23514';
        END IF;
    END IF;

    -- A chain never crosses scopes. The tail check above already implies
    -- this for a well-formed table; stating it explicitly means a corrupt or
    -- hand-edited predecessor cannot smuggle one chain into another.
    IF NEW.prev_row_digest IS NOT NULL THEN
        SELECT logbook.chain_scope(predecessor.pipeline)
        INTO predecessor_scope
        FROM logbook.builds AS predecessor
        WHERE predecessor.row_digest = NEW.prev_row_digest;

        IF predecessor_scope IS DISTINCT FROM new_scope THEN
            RAISE EXCEPTION
                'Logbook build % (scope %) extends a % row; chains never '
                'cross scopes',
                NEW.build_id,
                new_scope,
                coalesce(predecessor_scope, 'missing')
                USING ERRCODE = '23514';
        END IF;
    END IF;

    RETURN NEW;
END;
$function$;
