-- Give exact-record-count US releases a distinct pipeline identity while
-- preserving the existing mixed US Logbook sequence.

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
            'us-exact-k-release',
            'us-pool-inc2',
            'us-stacked-pool'
        ) THEN 'us'
        WHEN p_pipeline ~ '^[a-z]{2}-[a-z0-9_]+(-[a-z0-9_-]+)?$'
            THEN left(p_pipeline, 2) || '/' || split_part(p_pipeline, '-', 2)
    END;
$function$;

GRANT EXECUTE ON FUNCTION logbook.chain_scope(text)
    TO logbook_writer, logbook_exporter, logbook_break_glass_admin;

COMMENT ON FUNCTION logbook.chain_scope(text) IS
    'Derives the immutable Logbook sequence scope from a build pipeline name.';
