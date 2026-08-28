-- Ratify uk/local for the local-areas line's first archived run (#761).
CREATE OR REPLACE FUNCTION logbook.scope_declared(p_scope text)
RETURNS boolean
LANGUAGE sql
IMMUTABLE
PARALLEL SAFE
SET search_path = pg_catalog
AS $function$
    SELECT p_scope IN ('us', 'uk/frs', 'uk/local');
$function$;

GRANT EXECUTE ON FUNCTION logbook.scope_declared(text)
    TO logbook_writer, logbook_exporter, logbook_break_glass_admin;
