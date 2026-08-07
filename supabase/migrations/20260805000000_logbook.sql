-- Logbook: the append-only build and prediction ledger.
--
-- The build hash contract is deliberately implemented in SQL as well as in
-- the Python client.  Canonical JSON is UTF-8, compact, object-key sorted in
-- C/Unicode code-point order, and renders JSON numbers as plain decimals.
-- A build digest is:
--
--   sha256(canonical_json(build_without_chain_fields) || raw_previous_digest)
--
-- Both prev_row_digest and row_digest are excluded from the canonical object;
-- the predecessor is appended once, without a delimiter (it is either empty
-- for the genesis row or exactly 64 lowercase hexadecimal characters).

CREATE SCHEMA IF NOT EXISTS extensions;
CREATE EXTENSION IF NOT EXISTS pgcrypto WITH SCHEMA extensions;

-- IF NOT EXISTS leaves an already-installed extension in its original schema.
-- Logbook hard-qualifies digest() below, so normalize a relocatable pgcrypto
-- installation instead of depending on the database's prior search path.
DO $pgcrypto_schema$
DECLARE
    installed_schema text;
BEGIN
    SELECT namespace.nspname
    INTO installed_schema
    FROM pg_catalog.pg_extension AS extension
    JOIN pg_catalog.pg_namespace AS namespace
        ON namespace.oid = extension.extnamespace
    WHERE extension.extname = 'pgcrypto';

    IF installed_schema IS DISTINCT FROM 'extensions' THEN
        ALTER EXTENSION pgcrypto SET SCHEMA extensions;
    END IF;
END;
$pgcrypto_schema$;

CREATE SCHEMA IF NOT EXISTS logbook;

CREATE DOMAIN logbook.sha256_hex AS text
    CHECK (VALUE ~ '^[0-9a-f]{64}$');

CREATE TYPE logbook.build_disposition AS ENUM (
    'iterating',
    'billed',
    'published',
    'certified',
    'failed',
    'superseded',
    'discarded'
);

-- Python's str.strip() recognizes these 29 Unicode whitespace code points.
-- Keep direct database inserts on the same nonempty/trimmed text contract as
-- the fail-closed Logbook client.
CREATE OR REPLACE FUNCTION logbook.nonempty_trimmed_text(p_value text)
RETURNS boolean
LANGUAGE sql
IMMUTABLE
STRICT
PARALLEL SAFE
SET search_path = pg_catalog, logbook
AS $function$
    SELECT p_value <> '' AND p_value = btrim(
        p_value,
        U&'\0009\000A\000B\000C\000D\001C\001D\001E\001F\0020' ||
        U&'\0085\00A0\1680\2000\2001\2002\2003\2004\2005\2006' ||
        U&'\2007\2008\2009\200A\2028\2029\202F\205F\3000'
    )
$function$;

CREATE TABLE logbook.predictions (
    id text PRIMARY KEY CHECK (logbook.nonempty_trimmed_text(id)),
    ts timestamptz NOT NULL CHECK (isfinite(ts)),
    claim text CHECK (
        claim IS NULL OR logbook.nonempty_trimmed_text(claim)
    ),
    type text CHECK (
        type IS NULL OR logbook.nonempty_trimmed_text(type)
    ),
    predicted jsonb,
    p numeric CHECK (p IS NULL OR (p >= 0 AND p <= 1)),
    resolved timestamptz CHECK (resolved IS NULL OR isfinite(resolved)),
    resolves text CHECK (
        resolves IS NULL
        OR logbook.nonempty_trimmed_text(resolves)
    ),
    outcome text NOT NULL CHECK (logbook.nonempty_trimmed_text(outcome)),
    actual jsonb,
    note text,
    CONSTRAINT predictions_event_shape CHECK (
        (
            resolves IS NULL
            AND claim IS NOT NULL
            AND type IS NOT NULL
        )
        OR (
            resolves IS NOT NULL
            AND claim IS NULL
            AND type IS NULL
        )
    ),
    CONSTRAINT predictions_resolves_other CHECK (
        resolves IS NULL OR resolves <> id
    ),
    CONSTRAINT predictions_resolves_fk FOREIGN KEY (resolves)
        REFERENCES logbook.predictions (id)
        ON UPDATE RESTRICT
        ON DELETE RESTRICT
        DEFERRABLE INITIALLY DEFERRED
);

CREATE INDEX predictions_resolves_idx
    ON logbook.predictions (resolves)
    WHERE resolves IS NOT NULL;

CREATE OR REPLACE FUNCTION logbook.valid_build_phases(p_value jsonb)
RETURNS boolean
LANGUAGE plpgsql
IMMUTABLE
STRICT
PARALLEL SAFE
SET search_path = pg_catalog, logbook
AS $function$
DECLARE
    item jsonb;
    phase text;
    phases_seen text[] := ARRAY[]::text[];
BEGIN
    IF jsonb_typeof(p_value) <> 'array' THEN
        RETURN false;
    END IF;
    IF jsonb_array_length(p_value) = 0 THEN
        RETURN false;
    END IF;
    FOR item IN SELECT value FROM jsonb_array_elements(p_value)
    LOOP
        IF jsonb_typeof(item) <> 'string' THEN
            RETURN false;
        END IF;
        phase := item #>> '{}';
        IF NOT logbook.nonempty_trimmed_text(phase)
            OR phase = ANY(phases_seen)
        THEN
            RETURN false;
        END IF;
        phases_seen := array_append(phases_seen, phase);
    END LOOP;
    RETURN true;
END;
$function$;

CREATE OR REPLACE FUNCTION logbook.valid_gate_verdicts(p_value jsonb)
RETURNS boolean
LANGUAGE plpgsql
IMMUTABLE
STRICT
PARALLEL SAFE
SET search_path = pg_catalog, logbook
AS $function$
DECLARE
    member record;
    verdict text;
    receipt_pointer text;
BEGIN
    IF jsonb_typeof(p_value) <> 'object' OR p_value = '{}'::jsonb THEN
        RETURN false;
    END IF;
    FOR member IN SELECT key, value FROM jsonb_each(p_value)
    LOOP
        IF NOT logbook.nonempty_trimmed_text(member.key)
            OR jsonb_typeof(member.value) <> 'object'
            OR jsonb_typeof(member.value -> 'verdict') IS DISTINCT FROM 'string'
            OR jsonb_typeof(member.value -> 'receipt') IS DISTINCT FROM 'string'
        THEN
            RETURN false;
        END IF;
        verdict := member.value ->> 'verdict';
        receipt_pointer := member.value ->> 'receipt';
        IF NOT logbook.nonempty_trimmed_text(verdict)
            OR NOT logbook.nonempty_trimmed_text(receipt_pointer)
        THEN
            RETURN false;
        END IF;
    END LOOP;
    RETURN true;
END;
$function$;

CREATE TABLE logbook.builds (
    build_id text PRIMARY KEY CHECK (
        logbook.nonempty_trimmed_text(build_id)
        AND build_id ~ '^[A-Za-z0-9][A-Za-z0-9._:-]{0,254}$'
    ),
    ts timestamptz NOT NULL CHECK (isfinite(ts)),
    pipeline text NOT NULL CHECK (logbook.nonempty_trimmed_text(pipeline)),
    rung text NOT NULL,
    seed bigint CHECK (seed IS NULL OR seed >= 0),
    code_pin text NOT NULL CHECK (logbook.nonempty_trimmed_text(code_pin)),
    input_pins_digest logbook.sha256_hex NOT NULL,
    identity_digest logbook.sha256_hex NOT NULL,
    phases_reached jsonb NOT NULL
        CHECK (logbook.valid_build_phases(phases_reached)),
    gate_verdicts jsonb NOT NULL
        CHECK (logbook.valid_gate_verdicts(gate_verdicts)),
    wall_seconds numeric,
    cost_usd numeric,
    artifact_location text CHECK (
        artifact_location IS NULL
        OR logbook.nonempty_trimmed_text(artifact_location)
    ),
    disposition logbook.build_disposition NOT NULL,
    prediction_id text CHECK (
        prediction_id IS NULL
        OR logbook.nonempty_trimmed_text(prediction_id)
    ),
    prev_row_digest logbook.sha256_hex,
    row_digest logbook.sha256_hex NOT NULL,
    CONSTRAINT builds_rung_fraction_token CHECK (
        rung IN ('f001', 'f010', 'f100')
    ),
    CONSTRAINT builds_wall_seconds_nonnegative_finite CHECK (
        wall_seconds IS NULL
        OR (wall_seconds >= 0 AND wall_seconds < 'Infinity'::numeric)
    ),
    CONSTRAINT builds_cost_usd_nonnegative_finite CHECK (
        cost_usd IS NULL
        OR (cost_usd >= 0 AND cost_usd < 'Infinity'::numeric)
    ),
    CONSTRAINT builds_published_artifact_required CHECK (
        disposition NOT IN ('published', 'certified')
        OR (
            artifact_location IS NOT NULL
            AND btrim(artifact_location) <> ''
        )
    ),
    CONSTRAINT builds_prediction_fk FOREIGN KEY (prediction_id)
        REFERENCES logbook.predictions (id)
        ON UPDATE RESTRICT
        ON DELETE RESTRICT
        DEFERRABLE INITIALLY DEFERRED,
    CONSTRAINT builds_row_digest_unique UNIQUE (row_digest),
    CONSTRAINT builds_predecessor_fk FOREIGN KEY (prev_row_digest)
        REFERENCES logbook.builds (row_digest)
        ON UPDATE RESTRICT
        ON DELETE RESTRICT
        DEFERRABLE INITIALLY IMMEDIATE
);

-- A digest may be the predecessor of at most one row.  Together with the
-- trigger's unique-tail check, this prevents a fork even if concurrent writers
-- race to append after the same build.
CREATE UNIQUE INDEX builds_unique_predecessor
    ON logbook.builds (prev_row_digest)
    WHERE prev_row_digest IS NOT NULL;

-- The trigger enforces this too; the index keeps the single-genesis invariant
-- declarative if a privileged operator ever disables triggers temporarily.
CREATE UNIQUE INDEX builds_single_genesis
    ON logbook.builds ((1))
    WHERE prev_row_digest IS NULL;

CREATE INDEX builds_ts_idx ON logbook.builds (ts, build_id);
CREATE INDEX builds_disposition_ts_idx
    ON logbook.builds (disposition, ts, build_id);
CREATE INDEX builds_prediction_id_idx
    ON logbook.builds (prediction_id)
    WHERE prediction_id IS NOT NULL;

CREATE OR REPLACE FUNCTION logbook.canonical_json_text(p_value jsonb)
RETURNS text
LANGUAGE plpgsql
IMMUTABLE
STRICT
PARALLEL SAFE
SET search_path = pg_catalog, logbook
AS $function$
DECLARE
    value_kind text;
    rendered text;
    member record;
    first_member boolean;
BEGIN
    value_kind := jsonb_typeof(p_value);

    IF value_kind = 'object' THEN
        rendered := '{';
        first_member := true;
        FOR member IN
            SELECT entry.key, entry.value
            FROM jsonb_each(p_value) AS entry(key, value)
            ORDER BY entry.key COLLATE "C"
        LOOP
            IF NOT first_member THEN
                rendered := rendered || ',';
            END IF;
            rendered := rendered
                || to_jsonb(member.key)::text
                || ':'
                || logbook.canonical_json_text(member.value);
            first_member := false;
        END LOOP;
        RETURN rendered || '}';
    END IF;

    IF value_kind = 'array' THEN
        rendered := '[';
        first_member := true;
        FOR member IN
            SELECT entry.value
            FROM jsonb_array_elements(p_value) WITH ORDINALITY
                AS entry(value, ordinal)
            ORDER BY entry.ordinal
        LOOP
            IF NOT first_member THEN
                rendered := rendered || ',';
            END IF;
            rendered := rendered
                || logbook.canonical_json_text(member.value);
            first_member := false;
        END LOOP;
        RETURN rendered || ']';
    END IF;

    IF value_kind = 'number' THEN
        -- jsonb stores JSON numbers as PostgreSQL numeric.  numeric's text
        -- output is a plain decimal.  trim_scale removes insignificant
        -- fractional zeroes and normalizes signed zero exactly like the
        -- Python Decimal encoder.
        RETURN trim_scale((p_value #>> '{}')::numeric)::text;
    END IF;

    IF value_kind IN ('string', 'boolean', 'null') THEN
        -- jsonb scalar text supplies the required JSON quoting and escaping.
        RETURN p_value::text;
    END IF;

    RAISE EXCEPTION 'unsupported JSON value kind: %', value_kind
        USING ERRCODE = '22023';
END;
$function$;

CREATE OR REPLACE FUNCTION logbook.build_hash_payload(
    p_build logbook.builds
)
RETURNS jsonb
LANGUAGE sql
STABLE
STRICT
SET search_path = pg_catalog, logbook
AS $function$
    SELECT jsonb_build_object(
        'artifact_location', p_build.artifact_location,
        'build_id', p_build.build_id,
        'code_pin', p_build.code_pin,
        'cost_usd', p_build.cost_usd,
        'disposition', p_build.disposition::text,
        'gate_verdicts', p_build.gate_verdicts,
        'identity_digest', p_build.identity_digest,
        'input_pins_digest', p_build.input_pins_digest,
        'phases_reached', p_build.phases_reached,
        'pipeline', p_build.pipeline,
        'prediction_id', p_build.prediction_id,
        'rung', p_build.rung,
        'seed', p_build.seed,
        'ts', to_char(
            p_build.ts AT TIME ZONE 'UTC',
            'YYYY-MM-DD"T"HH24:MI:SS.US"Z"'
        ),
        'wall_seconds', p_build.wall_seconds
    )
$function$;

CREATE OR REPLACE FUNCTION logbook.expected_build_row_digest(
    p_build logbook.builds
)
RETURNS logbook.sha256_hex
LANGUAGE sql
STABLE
STRICT
SET search_path = pg_catalog, logbook, extensions
AS $function$
    SELECT encode(
        extensions.digest(
            convert_to(
                logbook.canonical_json_text(
                    logbook.build_hash_payload(p_build)
                ) || coalesce(p_build.prev_row_digest::text, ''),
                'UTF8'
            ),
            'sha256'
        ),
        'hex'
    )::logbook.sha256_hex
$function$;

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
    build_count bigint;
    tail_count bigint;
BEGIN
    -- Use a fixed, transaction-scoped lock so concurrent transactions cannot
    -- both observe and append after the same tail.
    PERFORM pg_advisory_xact_lock(628, 20260805);

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

    SELECT count(*) INTO build_count FROM logbook.builds;

    IF build_count = 0 THEN
        IF NEW.prev_row_digest IS NOT NULL THEN
            RAISE EXCEPTION
                'Logbook genesis build % must have null prev_row_digest, got %',
                NEW.build_id,
                NEW.prev_row_digest
                USING ERRCODE = '23514';
        END IF;
    ELSE
        SELECT count(*), min(candidate.row_digest::text)::logbook.sha256_hex
        INTO tail_count, expected_predecessor
        FROM logbook.builds AS candidate
        WHERE NOT EXISTS (
            SELECT 1
            FROM logbook.builds AS successor
            WHERE successor.prev_row_digest = candidate.row_digest
        );

        IF tail_count <> 1 THEN
            RAISE EXCEPTION
                'Logbook chain is corrupt: expected one tail, found %',
                tail_count
                USING ERRCODE = '23514';
        END IF;

        IF NEW.prev_row_digest IS DISTINCT FROM expected_predecessor THEN
            RAISE EXCEPTION
                'Logbook build % has prev_row_digest %, current tail is %',
                NEW.build_id,
                NEW.prev_row_digest,
                expected_predecessor
                USING ERRCODE = '23514';
        END IF;
    END IF;

    RETURN NEW;
END;
$function$;

CREATE TRIGGER builds_enforce_chain_before_insert
BEFORE INSERT ON logbook.builds
FOR EACH ROW
EXECUTE FUNCTION logbook.enforce_build_chain();

-- Only the explicitly safe build projection is exposed.  In particular, the
-- view carries neither cost_usd nor gate_verdicts (where private failure
-- diagnostics and receipt detail can live).  Artifact locations are public
-- only after publication or certification; failed-build locations may expose
-- private runtime paths.
CREATE VIEW logbook.builds_public
WITH (security_barrier = true)
AS
SELECT
    build_id,
    ts,
    pipeline,
    rung,
    seed,
    code_pin,
    input_pins_digest,
    identity_digest,
    phases_reached,
    wall_seconds,
    CASE
        WHEN disposition IN ('published', 'certified')
            THEN artifact_location
        ELSE NULL
    END AS artifact_location,
    disposition,
    prediction_id,
    prev_row_digest,
    row_digest
FROM logbook.builds;

-- Custom PostgREST roles.  They are NOLOGIN group roles: Supabase JWTs may
-- select the insert-only writer or private read-only exporter through
-- authenticator, while break-glass membership is granted out of band to a
-- human-controlled administrative identity.
DO $roles$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM pg_roles WHERE rolname = 'logbook_writer'
    ) THEN
        EXECUTE
            'CREATE ROLE logbook_writer NOLOGIN NOINHERIT NOBYPASSRLS';
    END IF;
    IF NOT EXISTS (
        SELECT 1 FROM pg_roles WHERE rolname = 'logbook_exporter'
    ) THEN
        EXECUTE
            'CREATE ROLE logbook_exporter NOLOGIN NOINHERIT NOBYPASSRLS';
    END IF;
    IF NOT EXISTS (
        SELECT 1 FROM pg_roles
        WHERE rolname = 'logbook_break_glass_admin'
    ) THEN
        EXECUTE
            'CREATE ROLE logbook_break_glass_admin '
            'NOLOGIN NOINHERIT NOBYPASSRLS';
    END IF;
END;
$roles$;

REVOKE ALL ON SCHEMA logbook FROM PUBLIC;
REVOKE ALL ON ALL TABLES IN SCHEMA logbook FROM PUBLIC;
REVOKE ALL ON ALL FUNCTIONS IN SCHEMA logbook FROM PUBLIC;

GRANT USAGE ON SCHEMA logbook
    TO logbook_writer, logbook_exporter, logbook_break_glass_admin;
GRANT USAGE ON TYPE logbook.sha256_hex, logbook.build_disposition
    TO logbook_writer, logbook_exporter, logbook_break_glass_admin;
GRANT EXECUTE ON FUNCTION
    logbook.nonempty_trimmed_text(text),
    logbook.valid_build_phases(jsonb),
    logbook.valid_gate_verdicts(jsonb)
    TO logbook_writer, logbook_break_glass_admin;
GRANT INSERT ON logbook.builds, logbook.predictions
    TO logbook_writer;
-- The client's only INSERT shape is PostgREST's idempotent replay
-- (?on_conflict=build_id + resolution=ignore-duplicates), which plans as
-- INSERT ... ON CONFLICT (build_id) DO NOTHING. PostgreSQL requires
-- plan-time SELECT privilege on the conflict-target column, and FORCE RLS
-- additionally requires a SELECT policy for the conflict check. The
-- column ACL keeps every other column (cost_usd included) unreadable.
GRANT SELECT (build_id) ON logbook.builds
    TO logbook_writer;
GRANT SELECT ON logbook.builds
    TO logbook_exporter;
GRANT SELECT, INSERT, UPDATE, DELETE
    ON logbook.builds, logbook.predictions
    TO logbook_break_glass_admin;
GRANT SELECT ON logbook.builds_public
    TO logbook_break_glass_admin;

ALTER TABLE logbook.builds ENABLE ROW LEVEL SECURITY;
ALTER TABLE logbook.builds FORCE ROW LEVEL SECURITY;
ALTER TABLE logbook.predictions ENABLE ROW LEVEL SECURITY;
ALTER TABLE logbook.predictions FORCE ROW LEVEL SECURITY;

CREATE POLICY builds_writer_insert
    ON logbook.builds
    FOR INSERT
    TO logbook_writer
    WITH CHECK (true);

-- Row visibility for the writer is still bounded by the column ACL above:
-- this policy only lets the ON CONFLICT (build_id) check run under FORCE
-- ROW LEVEL SECURITY.
CREATE POLICY builds_writer_conflict_select
    ON logbook.builds
    FOR SELECT
    TO logbook_writer
    USING (true);

CREATE POLICY predictions_writer_insert
    ON logbook.predictions
    FOR INSERT
    TO logbook_writer
    WITH CHECK (true);

CREATE POLICY builds_exporter_select
    ON logbook.builds
    FOR SELECT
    TO logbook_exporter
    USING (true);

CREATE POLICY builds_break_glass_all
    ON logbook.builds
    FOR ALL
    TO logbook_break_glass_admin
    USING (true)
    WITH CHECK (true);

CREATE POLICY predictions_break_glass_all
    ON logbook.predictions
    FOR ALL
    TO logbook_break_glass_admin
    USING (true)
    WITH CHECK (true);

-- Keep this migration usable in plain PostgreSQL test databases, where the
-- Supabase API roles do not exist.  On Supabase, grant only the safe view to
-- anonymous/authenticated readers, explicitly strip service_role privileges,
-- and let PostgREST assume the insert-only writer role.
DO $supabase_roles$
DECLARE
    api_role text;
BEGIN
    FOREACH api_role IN ARRAY ARRAY['anon', 'authenticated']
    LOOP
        IF EXISTS (SELECT 1 FROM pg_roles WHERE rolname = api_role) THEN
            EXECUTE format(
                'GRANT USAGE ON SCHEMA logbook TO %I', api_role
            );
            EXECUTE format(
                'GRANT SELECT ON logbook.builds_public TO %I', api_role
            );
        END IF;
    END LOOP;

    IF EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'service_role') THEN
        EXECUTE
            'REVOKE ALL ON SCHEMA logbook FROM service_role';
        EXECUTE
            'REVOKE ALL ON logbook.builds, logbook.predictions, '
            'logbook.builds_public FROM service_role';
    END IF;

    IF EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'authenticator') THEN
        EXECUTE
            'GRANT logbook_writer, logbook_exporter TO authenticator';
    END IF;
END;
$supabase_roles$;

ALTER DEFAULT PRIVILEGES IN SCHEMA logbook
    REVOKE ALL ON TABLES FROM PUBLIC;
ALTER DEFAULT PRIVILEGES IN SCHEMA logbook
    REVOKE ALL ON FUNCTIONS FROM PUBLIC;

COMMENT ON SCHEMA logbook IS
    'Logbook append-only build and prediction ledger.';
COMMENT ON TABLE logbook.builds IS
    'Immutable build attempts linked in one advisory-locked SHA-256 chain.';
COMMENT ON TABLE logbook.predictions IS
    'Prediction ledger rows matching the populace prediction JSONL schema.';
COMMENT ON VIEW logbook.builds_public IS
    'Public-safe build projection without cost or gate/failure diagnostics.';
COMMENT ON FUNCTION logbook.canonical_json_text(jsonb) IS
    'Compact sorted-key JSON with PostgreSQL numeric values in plain-decimal form.';
COMMENT ON FUNCTION logbook.expected_build_row_digest(logbook.builds) IS
    'SHA-256 of canonical build content followed by its raw predecessor digest.';
