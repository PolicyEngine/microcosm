-- Chronicle: the append-only build and prediction ledger.
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
-- Chronicle hard-qualifies digest() below, so normalize a relocatable pgcrypto
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

CREATE SCHEMA IF NOT EXISTS chronicle;

CREATE DOMAIN chronicle.sha256_hex AS text
    CHECK (VALUE ~ '^[0-9a-f]{64}$');

CREATE TYPE chronicle.build_disposition AS ENUM (
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
-- the fail-closed Chronicle client.
CREATE OR REPLACE FUNCTION chronicle.nonempty_trimmed_text(p_value text)
RETURNS boolean
LANGUAGE sql
IMMUTABLE
STRICT
PARALLEL SAFE
SET search_path = pg_catalog, chronicle
AS $function$
    SELECT p_value <> '' AND p_value = btrim(
        p_value,
        U&'\0009\000A\000B\000C\000D\001C\001D\001E\001F\0020' ||
        U&'\0085\00A0\1680\2000\2001\2002\2003\2004\2005\2006' ||
        U&'\2007\2008\2009\200A\2028\2029\202F\205F\3000'
    )
$function$;

CREATE TABLE chronicle.predictions (
    id text PRIMARY KEY CHECK (chronicle.nonempty_trimmed_text(id)),
    ts timestamptz NOT NULL CHECK (isfinite(ts)),
    claim text CHECK (
        claim IS NULL OR chronicle.nonempty_trimmed_text(claim)
    ),
    type text CHECK (
        type IS NULL OR chronicle.nonempty_trimmed_text(type)
    ),
    predicted jsonb,
    p numeric CHECK (p IS NULL OR (p >= 0 AND p <= 1)),
    resolved timestamptz CHECK (resolved IS NULL OR isfinite(resolved)),
    resolves text CHECK (
        resolves IS NULL
        OR chronicle.nonempty_trimmed_text(resolves)
    ),
    outcome text NOT NULL CHECK (chronicle.nonempty_trimmed_text(outcome)),
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
        REFERENCES chronicle.predictions (id)
        ON UPDATE RESTRICT
        ON DELETE RESTRICT
        DEFERRABLE INITIALLY DEFERRED
);

CREATE INDEX predictions_resolves_idx
    ON chronicle.predictions (resolves)
    WHERE resolves IS NOT NULL;

CREATE OR REPLACE FUNCTION chronicle.valid_build_phases(p_value jsonb)
RETURNS boolean
LANGUAGE plpgsql
IMMUTABLE
STRICT
PARALLEL SAFE
SET search_path = pg_catalog, chronicle
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
        IF NOT chronicle.nonempty_trimmed_text(phase)
            OR phase = ANY(phases_seen)
        THEN
            RETURN false;
        END IF;
        phases_seen := array_append(phases_seen, phase);
    END LOOP;
    RETURN true;
END;
$function$;

CREATE OR REPLACE FUNCTION chronicle.valid_gate_verdicts(p_value jsonb)
RETURNS boolean
LANGUAGE plpgsql
IMMUTABLE
STRICT
PARALLEL SAFE
SET search_path = pg_catalog, chronicle
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
        IF NOT chronicle.nonempty_trimmed_text(member.key)
            OR jsonb_typeof(member.value) <> 'object'
            OR jsonb_typeof(member.value -> 'verdict') IS DISTINCT FROM 'string'
            OR jsonb_typeof(member.value -> 'receipt') IS DISTINCT FROM 'string'
        THEN
            RETURN false;
        END IF;
        verdict := member.value ->> 'verdict';
        receipt_pointer := member.value ->> 'receipt';
        IF NOT chronicle.nonempty_trimmed_text(verdict)
            OR NOT chronicle.nonempty_trimmed_text(receipt_pointer)
        THEN
            RETURN false;
        END IF;
    END LOOP;
    RETURN true;
END;
$function$;

CREATE TABLE chronicle.builds (
    build_id text PRIMARY KEY CHECK (
        chronicle.nonempty_trimmed_text(build_id)
        AND build_id ~ '^[A-Za-z0-9][A-Za-z0-9._:-]{0,254}$'
    ),
    ts timestamptz NOT NULL CHECK (isfinite(ts)),
    pipeline text NOT NULL CHECK (chronicle.nonempty_trimmed_text(pipeline)),
    rung text NOT NULL,
    seed bigint CHECK (seed IS NULL OR seed >= 0),
    code_pin text NOT NULL CHECK (chronicle.nonempty_trimmed_text(code_pin)),
    input_pins_digest chronicle.sha256_hex NOT NULL,
    identity_digest chronicle.sha256_hex NOT NULL,
    phases_reached jsonb NOT NULL
        CHECK (chronicle.valid_build_phases(phases_reached)),
    gate_verdicts jsonb NOT NULL
        CHECK (chronicle.valid_gate_verdicts(gate_verdicts)),
    wall_seconds numeric,
    cost_usd numeric,
    artifact_location text CHECK (
        artifact_location IS NULL
        OR chronicle.nonempty_trimmed_text(artifact_location)
    ),
    disposition chronicle.build_disposition NOT NULL,
    prediction_id text CHECK (
        prediction_id IS NULL
        OR chronicle.nonempty_trimmed_text(prediction_id)
    ),
    prev_row_digest chronicle.sha256_hex,
    row_digest chronicle.sha256_hex NOT NULL,
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
        REFERENCES chronicle.predictions (id)
        ON UPDATE RESTRICT
        ON DELETE RESTRICT
        DEFERRABLE INITIALLY DEFERRED,
    CONSTRAINT builds_row_digest_unique UNIQUE (row_digest),
    CONSTRAINT builds_predecessor_fk FOREIGN KEY (prev_row_digest)
        REFERENCES chronicle.builds (row_digest)
        ON UPDATE RESTRICT
        ON DELETE RESTRICT
        DEFERRABLE INITIALLY IMMEDIATE
);

-- A digest may be the predecessor of at most one row.  Together with the
-- trigger's unique-tail check, this prevents a fork even if concurrent writers
-- race to append after the same build.
CREATE UNIQUE INDEX builds_unique_predecessor
    ON chronicle.builds (prev_row_digest)
    WHERE prev_row_digest IS NOT NULL;

-- The trigger enforces this too; the index keeps the single-genesis invariant
-- declarative if a privileged operator ever disables triggers temporarily.
CREATE UNIQUE INDEX builds_single_genesis
    ON chronicle.builds ((1))
    WHERE prev_row_digest IS NULL;

CREATE INDEX builds_ts_idx ON chronicle.builds (ts, build_id);
CREATE INDEX builds_disposition_ts_idx
    ON chronicle.builds (disposition, ts, build_id);
CREATE INDEX builds_prediction_id_idx
    ON chronicle.builds (prediction_id)
    WHERE prediction_id IS NOT NULL;

CREATE OR REPLACE FUNCTION chronicle.canonical_json_text(p_value jsonb)
RETURNS text
LANGUAGE plpgsql
IMMUTABLE
STRICT
PARALLEL SAFE
SET search_path = pg_catalog, chronicle
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
                || chronicle.canonical_json_text(member.value);
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
                || chronicle.canonical_json_text(member.value);
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

CREATE OR REPLACE FUNCTION chronicle.build_hash_payload(
    p_build chronicle.builds
)
RETURNS jsonb
LANGUAGE sql
STABLE
STRICT
SET search_path = pg_catalog, chronicle
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

CREATE OR REPLACE FUNCTION chronicle.expected_build_row_digest(
    p_build chronicle.builds
)
RETURNS chronicle.sha256_hex
LANGUAGE sql
STABLE
STRICT
SET search_path = pg_catalog, chronicle, extensions
AS $function$
    SELECT encode(
        extensions.digest(
            convert_to(
                chronicle.canonical_json_text(
                    chronicle.build_hash_payload(p_build)
                ) || coalesce(p_build.prev_row_digest::text, ''),
                'UTF8'
            ),
            'sha256'
        ),
        'hex'
    )::chronicle.sha256_hex
$function$;

CREATE OR REPLACE FUNCTION chronicle.enforce_build_chain()
RETURNS trigger
LANGUAGE plpgsql
SECURITY DEFINER
SET search_path = pg_catalog, chronicle, extensions
AS $function$
DECLARE
    computed_digest chronicle.sha256_hex;
    existing_build chronicle.builds%ROWTYPE;
    expected_predecessor chronicle.sha256_hex;
    build_count bigint;
    tail_count bigint;
BEGIN
    -- Use a fixed, transaction-scoped lock so concurrent transactions cannot
    -- both observe and append after the same tail.
    PERFORM pg_advisory_xact_lock(628, 20260805);

    computed_digest := chronicle.expected_build_row_digest(NEW);
    IF NEW.row_digest IS NOT NULL AND NEW.row_digest <> computed_digest THEN
        RAISE EXCEPTION
            'Chronicle build % supplied row_digest %, expected %',
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
    FROM chronicle.builds
    WHERE build_id = NEW.build_id;

    IF FOUND THEN
        IF chronicle.build_hash_payload(existing_build)
                IS DISTINCT FROM chronicle.build_hash_payload(NEW)
            OR existing_build.prev_row_digest
                IS DISTINCT FROM NEW.prev_row_digest
            OR existing_build.row_digest IS DISTINCT FROM NEW.row_digest
        THEN
            RAISE EXCEPTION
                'Chronicle build_id % already exists with divergent content',
                NEW.build_id
                USING ERRCODE = '23505';
        END IF;
        RETURN NEW;
    END IF;

    SELECT count(*) INTO build_count FROM chronicle.builds;

    IF build_count = 0 THEN
        IF NEW.prev_row_digest IS NOT NULL THEN
            RAISE EXCEPTION
                'Chronicle genesis build % must have null prev_row_digest, got %',
                NEW.build_id,
                NEW.prev_row_digest
                USING ERRCODE = '23514';
        END IF;
    ELSE
        SELECT count(*), min(candidate.row_digest::text)::chronicle.sha256_hex
        INTO tail_count, expected_predecessor
        FROM chronicle.builds AS candidate
        WHERE NOT EXISTS (
            SELECT 1
            FROM chronicle.builds AS successor
            WHERE successor.prev_row_digest = candidate.row_digest
        );

        IF tail_count <> 1 THEN
            RAISE EXCEPTION
                'Chronicle chain is corrupt: expected one tail, found %',
                tail_count
                USING ERRCODE = '23514';
        END IF;

        IF NEW.prev_row_digest IS DISTINCT FROM expected_predecessor THEN
            RAISE EXCEPTION
                'Chronicle build % has prev_row_digest %, current tail is %',
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
BEFORE INSERT ON chronicle.builds
FOR EACH ROW
EXECUTE FUNCTION chronicle.enforce_build_chain();

-- Only the explicitly safe build projection is exposed.  In particular, the
-- view carries neither cost_usd nor gate_verdicts (where private failure
-- diagnostics and receipt detail can live).  Artifact locations are public
-- only after publication or certification; failed-build locations may expose
-- private runtime paths.
CREATE VIEW chronicle.builds_public
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
FROM chronicle.builds;

-- Custom PostgREST roles.  They are NOLOGIN group roles: Supabase JWTs may
-- select the insert-only writer or private read-only exporter through
-- authenticator, while break-glass membership is granted out of band to a
-- human-controlled administrative identity.
DO $roles$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM pg_roles WHERE rolname = 'chronicle_writer'
    ) THEN
        EXECUTE
            'CREATE ROLE chronicle_writer NOLOGIN NOINHERIT NOBYPASSRLS';
    END IF;
    IF NOT EXISTS (
        SELECT 1 FROM pg_roles WHERE rolname = 'chronicle_exporter'
    ) THEN
        EXECUTE
            'CREATE ROLE chronicle_exporter NOLOGIN NOINHERIT NOBYPASSRLS';
    END IF;
    IF NOT EXISTS (
        SELECT 1 FROM pg_roles
        WHERE rolname = 'chronicle_break_glass_admin'
    ) THEN
        EXECUTE
            'CREATE ROLE chronicle_break_glass_admin '
            'NOLOGIN NOINHERIT NOBYPASSRLS';
    END IF;
END;
$roles$;

REVOKE ALL ON SCHEMA chronicle FROM PUBLIC;
REVOKE ALL ON ALL TABLES IN SCHEMA chronicle FROM PUBLIC;
REVOKE ALL ON ALL FUNCTIONS IN SCHEMA chronicle FROM PUBLIC;

GRANT USAGE ON SCHEMA chronicle
    TO chronicle_writer, chronicle_exporter, chronicle_break_glass_admin;
GRANT USAGE ON TYPE chronicle.sha256_hex, chronicle.build_disposition
    TO chronicle_writer, chronicle_exporter, chronicle_break_glass_admin;
GRANT EXECUTE ON FUNCTION
    chronicle.nonempty_trimmed_text(text),
    chronicle.valid_build_phases(jsonb),
    chronicle.valid_gate_verdicts(jsonb)
    TO chronicle_writer, chronicle_break_glass_admin;
GRANT INSERT ON chronicle.builds, chronicle.predictions
    TO chronicle_writer;
GRANT SELECT ON chronicle.builds
    TO chronicle_exporter;
GRANT SELECT, INSERT, UPDATE, DELETE
    ON chronicle.builds, chronicle.predictions
    TO chronicle_break_glass_admin;
GRANT SELECT ON chronicle.builds_public
    TO chronicle_break_glass_admin;

ALTER TABLE chronicle.builds ENABLE ROW LEVEL SECURITY;
ALTER TABLE chronicle.builds FORCE ROW LEVEL SECURITY;
ALTER TABLE chronicle.predictions ENABLE ROW LEVEL SECURITY;
ALTER TABLE chronicle.predictions FORCE ROW LEVEL SECURITY;

CREATE POLICY builds_writer_insert
    ON chronicle.builds
    FOR INSERT
    TO chronicle_writer
    WITH CHECK (true);

CREATE POLICY predictions_writer_insert
    ON chronicle.predictions
    FOR INSERT
    TO chronicle_writer
    WITH CHECK (true);

CREATE POLICY builds_exporter_select
    ON chronicle.builds
    FOR SELECT
    TO chronicle_exporter
    USING (true);

CREATE POLICY builds_break_glass_all
    ON chronicle.builds
    FOR ALL
    TO chronicle_break_glass_admin
    USING (true)
    WITH CHECK (true);

CREATE POLICY predictions_break_glass_all
    ON chronicle.predictions
    FOR ALL
    TO chronicle_break_glass_admin
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
                'GRANT USAGE ON SCHEMA chronicle TO %I', api_role
            );
            EXECUTE format(
                'GRANT SELECT ON chronicle.builds_public TO %I', api_role
            );
        END IF;
    END LOOP;

    IF EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'service_role') THEN
        EXECUTE
            'REVOKE ALL ON SCHEMA chronicle FROM service_role';
        EXECUTE
            'REVOKE ALL ON chronicle.builds, chronicle.predictions, '
            'chronicle.builds_public FROM service_role';
    END IF;

    IF EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'authenticator') THEN
        EXECUTE
            'GRANT chronicle_writer, chronicle_exporter TO authenticator';
    END IF;
END;
$supabase_roles$;

ALTER DEFAULT PRIVILEGES IN SCHEMA chronicle
    REVOKE ALL ON TABLES FROM PUBLIC;
ALTER DEFAULT PRIVILEGES IN SCHEMA chronicle
    REVOKE ALL ON FUNCTIONS FROM PUBLIC;

COMMENT ON SCHEMA chronicle IS
    'Chronicle append-only build and prediction ledger.';
COMMENT ON TABLE chronicle.builds IS
    'Immutable build attempts linked in one advisory-locked SHA-256 chain.';
COMMENT ON TABLE chronicle.predictions IS
    'Prediction ledger rows matching the populace prediction JSONL schema.';
COMMENT ON VIEW chronicle.builds_public IS
    'Public-safe build projection without cost or gate/failure diagnostics.';
COMMENT ON FUNCTION chronicle.canonical_json_text(jsonb) IS
    'Compact sorted-key JSON with PostgreSQL numeric values in plain-decimal form.';
COMMENT ON FUNCTION chronicle.expected_build_row_digest(chronicle.builds) IS
    'SHA-256 of canonical build content followed by its raw predecessor digest.';
