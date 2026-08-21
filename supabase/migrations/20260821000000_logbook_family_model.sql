-- Add authenticated build cardinality and relational dataset-family records.
--
-- This migration depends on 20260818000000_logbook_chain_scopes.sql. It does
-- not replace chain_scope(), scope_declared(), or enforce_build_chain(): the
-- current one-predecessor-per-scope behavior remains the build insertion rule.

ALTER TABLE logbook.builds
    ADD COLUMN row_format_version smallint,
    ADD COLUMN requested_k bigint,
    ADD COLUMN realized_k bigint,
    ADD COLUMN record_unit text;

ALTER TABLE logbook.builds
    ALTER COLUMN rung DROP NOT NULL,
    DROP CONSTRAINT builds_rung_fraction_token;

ALTER TABLE logbook.builds
    ADD CONSTRAINT builds_rung_by_row_format CHECK (
        (
            row_format_version IS NULL
            AND rung IS NOT NULL
            AND rung IN ('f001', 'f004', 'f010', 'f025', 'f100')
        )
        OR (
            row_format_version IS NOT DISTINCT FROM 2
            AND (
                rung IS NULL
                OR rung IN ('f001', 'f004', 'f010', 'f025', 'f100')
            )
        )
    ),
    ADD CONSTRAINT builds_row_format_and_cardinality CHECK (
        (
            row_format_version IS NULL
            AND requested_k IS NULL
            AND realized_k IS NULL
            AND record_unit IS NULL
        )
        OR (
            row_format_version IS NOT DISTINCT FROM 2
            AND (requested_k IS NULL OR requested_k > 0)
            AND (realized_k IS NULL OR realized_k > 0)
            AND (
                (
                    requested_k IS NULL
                    AND realized_k IS NULL
                    AND record_unit IS NULL
                )
                OR (
                    (requested_k IS NOT NULL OR realized_k IS NOT NULL)
                    AND record_unit IS NOT NULL
                    AND logbook.nonempty_trimmed_text(record_unit)
                    AND record_unit = lower(record_unit)
                    AND record_unit ~ '^[a-z][a-z0-9_]*$'
                )
            )
        )
    ),
    ADD CONSTRAINT builds_published_cardinality_matches CHECK (
        disposition NOT IN ('published', 'certified')
        OR requested_k IS NULL
        OR (
            realized_k IS NOT NULL
            AND realized_k = requested_k
        )
    );

CREATE OR REPLACE FUNCTION logbook.build_hash_payload(
    p_build logbook.builds
)
RETURNS jsonb
LANGUAGE plpgsql
STABLE
STRICT
SET search_path = pg_catalog, logbook
AS $function$
DECLARE
    payload jsonb;
BEGIN
    payload := jsonb_build_object(
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
    );

    IF p_build.row_format_version IS NULL THEN
        IF p_build.requested_k IS NOT NULL
            OR p_build.realized_k IS NOT NULL
            OR p_build.record_unit IS NOT NULL
        THEN
            RAISE EXCEPTION
                'Legacy Logbook build % has version-2 cardinality values',
                p_build.build_id
                USING ERRCODE = '23514';
        END IF;
        RETURN payload;
    END IF;

    IF p_build.row_format_version = 2 THEN
        RETURN payload || jsonb_build_object(
            'realized_k', p_build.realized_k,
            'record_unit', p_build.record_unit,
            'requested_k', p_build.requested_k,
            'row_format_version', p_build.row_format_version
        );
    END IF;

    RAISE EXCEPTION
        'Unsupported Logbook row_format_version % for build %',
        p_build.row_format_version,
        p_build.build_id
        USING ERRCODE = '23514';
END;
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

CREATE OR REPLACE VIEW logbook.builds_public
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
    row_digest,
    requested_k,
    realized_k,
    record_unit
FROM logbook.builds;

CREATE TYPE logbook.family_action_type AS ENUM (
    'revokes',
    'supersedes'
);

CREATE TABLE logbook.families (
    family_id uuid PRIMARY KEY,
    chain_scope text NOT NULL CHECK (
        logbook.scope_declared(chain_scope)
    ),
    source_pool_sha256 logbook.sha256_hex NOT NULL,
    CONSTRAINT families_scope_source_unique
        UNIQUE (chain_scope, source_pool_sha256)
);

CREATE TABLE logbook.family_members (
    family_id uuid NOT NULL,
    build_id text NOT NULL,
    CONSTRAINT family_members_pk PRIMARY KEY (family_id, build_id),
    CONSTRAINT family_members_build_unique UNIQUE (build_id),
    CONSTRAINT family_members_family_fk FOREIGN KEY (family_id)
        REFERENCES logbook.families (family_id)
        ON UPDATE RESTRICT
        ON DELETE RESTRICT,
    CONSTRAINT family_members_build_fk FOREIGN KEY (build_id)
        REFERENCES logbook.builds (build_id)
        ON UPDATE RESTRICT
        ON DELETE RESTRICT
);

CREATE TABLE logbook.family_actions (
    action_id uuid PRIMARY KEY,
    family_id uuid NOT NULL,
    build_id text NOT NULL,
    action_type logbook.family_action_type NOT NULL,
    related_build_id text,
    recorded_at timestamptz NOT NULL CHECK (isfinite(recorded_at)),
    actor text NOT NULL CHECK (logbook.nonempty_trimmed_text(actor)),
    reason text NOT NULL CHECK (logbook.nonempty_trimmed_text(reason)),
    evidence_location text CHECK (
        evidence_location IS NULL
        OR logbook.nonempty_trimmed_text(evidence_location)
    ),
    CONSTRAINT family_actions_shape CHECK (
        (
            action_type = 'revokes'
            AND related_build_id IS NULL
        )
        OR (
            action_type = 'supersedes'
            AND related_build_id IS NOT NULL
            AND related_build_id <> build_id
        )
    ),
    CONSTRAINT family_actions_member_fk
        FOREIGN KEY (family_id, build_id)
        REFERENCES logbook.family_members (family_id, build_id)
        ON UPDATE RESTRICT
        ON DELETE RESTRICT,
    CONSTRAINT family_actions_related_member_fk
        FOREIGN KEY (family_id, related_build_id)
        REFERENCES logbook.family_members (family_id, build_id)
        ON UPDATE RESTRICT
        ON DELETE RESTRICT
);

CREATE UNIQUE INDEX family_actions_one_direct_replacement
    ON logbook.family_actions (family_id, related_build_id)
    WHERE action_type = 'supersedes';

CREATE INDEX families_scope_idx
    ON logbook.families (chain_scope, family_id);
CREATE INDEX families_source_idx
    ON logbook.families (source_pool_sha256, family_id);
CREATE INDEX family_members_build_idx
    ON logbook.family_members (build_id, family_id);
CREATE INDEX family_actions_revocation_idx
    ON logbook.family_actions (family_id, build_id, recorded_at)
    WHERE action_type = 'revokes';
CREATE INDEX family_actions_replacement_idx
    ON logbook.family_actions (family_id, related_build_id, recorded_at)
    WHERE action_type = 'supersedes';

CREATE OR REPLACE FUNCTION logbook.enforce_family_insert()
RETURNS trigger
LANGUAGE plpgsql
SECURITY DEFINER
SET search_path = pg_catalog, logbook
AS $function$
DECLARE
    by_id logbook.families%ROWTYPE;
    by_source logbook.families%ROWTYPE;
    found_id boolean := false;
    found_source boolean := false;
BEGIN
    PERFORM pg_advisory_xact_lock(
        628,
        hashtext('family:' || NEW.family_id::text)
    );

    SELECT *
    INTO by_id
    FROM logbook.families
    WHERE family_id = NEW.family_id;
    found_id := FOUND;

    SELECT *
    INTO by_source
    FROM logbook.families
    WHERE chain_scope = NEW.chain_scope
      AND source_pool_sha256 = NEW.source_pool_sha256;
    found_source := FOUND;

    IF found_id AND (
        by_id.chain_scope IS DISTINCT FROM NEW.chain_scope
        OR by_id.source_pool_sha256
            IS DISTINCT FROM NEW.source_pool_sha256
    ) THEN
        RAISE EXCEPTION
            'Logbook family_id % already exists with divergent content',
            NEW.family_id
            USING ERRCODE = '23505';
    END IF;

    IF found_source
        AND by_source.family_id IS DISTINCT FROM NEW.family_id
    THEN
        RAISE EXCEPTION
            'Logbook source % in scope % already belongs to family %',
            NEW.source_pool_sha256,
            NEW.chain_scope,
            by_source.family_id
            USING ERRCODE = '23505';
    END IF;

    RETURN NEW;
END;
$function$;

CREATE TRIGGER families_enforce_insert_before_insert
BEFORE INSERT ON logbook.families
FOR EACH ROW
EXECUTE FUNCTION logbook.enforce_family_insert();

CREATE OR REPLACE FUNCTION logbook.enforce_family_membership()
RETURNS trigger
LANGUAGE plpgsql
SECURITY DEFINER
SET search_path = pg_catalog, logbook
AS $function$
DECLARE
    family_scope text;
    build_scope text;
BEGIN
    SELECT family.chain_scope
    INTO family_scope
    FROM logbook.families AS family
    WHERE family.family_id = NEW.family_id;

    IF NOT FOUND THEN
        RAISE EXCEPTION
            'Logbook family % does not exist',
            NEW.family_id
            USING ERRCODE = '23503';
    END IF;

    SELECT logbook.chain_scope(build.pipeline)
    INTO build_scope
    FROM logbook.builds AS build
    WHERE build.build_id = NEW.build_id;

    IF NOT FOUND THEN
        RAISE EXCEPTION
            'Logbook build % does not exist',
            NEW.build_id
            USING ERRCODE = '23503';
    END IF;

    IF build_scope IS DISTINCT FROM family_scope THEN
        RAISE EXCEPTION
            'Logbook build % scope % does not match family % scope %',
            NEW.build_id,
            coalesce(build_scope, 'missing'),
            NEW.family_id,
            family_scope
            USING ERRCODE = '23514';
    END IF;

    RETURN NEW;
END;
$function$;

CREATE TRIGGER family_members_enforce_before_insert
BEFORE INSERT ON logbook.family_members
FOR EACH ROW
EXECUTE FUNCTION logbook.enforce_family_membership();

CREATE OR REPLACE FUNCTION logbook.enforce_family_action()
RETURNS trigger
LANGUAGE plpgsql
SECURITY DEFINER
SET search_path = pg_catalog, logbook
AS $function$
DECLARE
    existing_action logbook.family_actions%ROWTYPE;
    replacement_requested bigint;
    replacement_unit text;
    replaced_requested bigint;
    replaced_unit text;
BEGIN
    PERFORM pg_advisory_xact_lock(
        628,
        hashtext('family-action:' || NEW.action_id::text)
    );

    SELECT *
    INTO existing_action
    FROM logbook.family_actions
    WHERE action_id = NEW.action_id;

    IF FOUND THEN
        IF existing_action.family_id IS DISTINCT FROM NEW.family_id
            OR existing_action.build_id IS DISTINCT FROM NEW.build_id
            OR existing_action.action_type IS DISTINCT FROM NEW.action_type
            OR existing_action.related_build_id
                IS DISTINCT FROM NEW.related_build_id
            OR existing_action.recorded_at IS DISTINCT FROM NEW.recorded_at
            OR existing_action.actor IS DISTINCT FROM NEW.actor
            OR existing_action.reason IS DISTINCT FROM NEW.reason
            OR existing_action.evidence_location
                IS DISTINCT FROM NEW.evidence_location
        THEN
            RAISE EXCEPTION
                'Logbook action_id % already exists with divergent content',
                NEW.action_id
                USING ERRCODE = '23505';
        END IF;
        RETURN NEW;
    END IF;

    IF NEW.action_type = 'supersedes' THEN
        SELECT build.requested_k, build.record_unit
        INTO replacement_requested, replacement_unit
        FROM logbook.family_members AS member
        JOIN logbook.builds AS build
          ON build.build_id = member.build_id
        WHERE member.family_id = NEW.family_id
          AND member.build_id = NEW.build_id;

        IF NOT FOUND THEN
            RAISE EXCEPTION
                'Replacement build % is not a member of family %',
                NEW.build_id,
                NEW.family_id
                USING ERRCODE = '23503';
        END IF;

        SELECT build.requested_k, build.record_unit
        INTO replaced_requested, replaced_unit
        FROM logbook.family_members AS member
        JOIN logbook.builds AS build
          ON build.build_id = member.build_id
        WHERE member.family_id = NEW.family_id
          AND member.build_id = NEW.related_build_id;

        IF NOT FOUND THEN
            RAISE EXCEPTION
                'Replaced build % is not a member of family %',
                NEW.related_build_id,
                NEW.family_id
                USING ERRCODE = '23503';
        END IF;

        IF replacement_requested IS DISTINCT FROM replaced_requested
            OR replacement_unit IS DISTINCT FROM replaced_unit
        THEN
            RAISE EXCEPTION
                'Superseding builds must have matching requested_k and record_unit'
                USING ERRCODE = '23514';
        END IF;
    END IF;

    RETURN NEW;
END;
$function$;

CREATE TRIGGER family_actions_enforce_before_insert
BEFORE INSERT ON logbook.family_actions
FOR EACH ROW
EXECUTE FUNCTION logbook.enforce_family_action();

CREATE VIEW logbook.families_public
WITH (security_barrier = true)
AS
SELECT
    family_id,
    chain_scope,
    source_pool_sha256
FROM logbook.families;

CREATE VIEW logbook.family_members_public
WITH (security_barrier = true)
AS
SELECT
    family.family_id,
    family.chain_scope,
    family.source_pool_sha256,
    build.build_id,
    build.ts,
    build.pipeline,
    build.rung,
    build.seed,
    build.code_pin,
    build.input_pins_digest,
    build.identity_digest,
    build.phases_reached,
    build.wall_seconds,
    build.artifact_location,
    build.disposition,
    build.prediction_id,
    build.prev_row_digest,
    build.row_digest,
    build.requested_k,
    build.realized_k,
    build.record_unit
FROM logbook.family_members AS member
JOIN logbook.families_public AS family
  ON family.family_id = member.family_id
JOIN logbook.builds_public AS build
  ON build.build_id = member.build_id;

CREATE VIEW logbook.family_actions_public
WITH (security_barrier = true)
AS
SELECT
    action.action_id,
    action.family_id,
    family.chain_scope,
    action.build_id,
    action.action_type,
    action.related_build_id,
    action.recorded_at,
    action.actor,
    action.reason,
    action.evidence_location
FROM logbook.family_actions AS action
JOIN logbook.families AS family
  ON family.family_id = action.family_id;

CREATE VIEW logbook.family_member_status_public
WITH (security_barrier = true)
AS
SELECT
    member.family_id,
    member.build_id,
    EXISTS (
        SELECT 1
        FROM logbook.family_actions AS action
        WHERE action.family_id = member.family_id
          AND action.build_id = member.build_id
          AND action.action_type = 'revokes'
    ) AS revoked,
    (
        SELECT action.build_id
        FROM logbook.family_actions AS action
        WHERE action.family_id = member.family_id
          AND action.related_build_id = member.build_id
          AND action.action_type = 'supersedes'
    ) AS superseded_by_build_id
FROM logbook.family_members AS member;

REVOKE ALL ON
    logbook.families,
    logbook.family_members,
    logbook.family_actions,
    logbook.families_public,
    logbook.family_members_public,
    logbook.family_actions_public,
    logbook.family_member_status_public
FROM PUBLIC;
REVOKE ALL ON FUNCTION
    logbook.enforce_family_insert(),
    logbook.enforce_family_membership(),
    logbook.enforce_family_action()
FROM PUBLIC;

GRANT USAGE ON TYPE logbook.family_action_type
    TO logbook_writer, logbook_exporter, logbook_break_glass_admin;
GRANT INSERT ON
    logbook.families,
    logbook.family_members,
    logbook.family_actions
TO logbook_writer;
GRANT SELECT (family_id) ON logbook.families
    TO logbook_writer;
GRANT SELECT (family_id, build_id) ON logbook.family_members
    TO logbook_writer;
GRANT SELECT (action_id) ON logbook.family_actions
    TO logbook_writer;
GRANT SELECT ON
    logbook.families,
    logbook.family_members,
    logbook.family_actions,
    logbook.families_public,
    logbook.family_members_public,
    logbook.family_actions_public,
    logbook.family_member_status_public
TO logbook_exporter;
GRANT SELECT, INSERT, UPDATE, DELETE ON
    logbook.families,
    logbook.family_members,
    logbook.family_actions
TO logbook_break_glass_admin;
GRANT SELECT ON
    logbook.families_public,
    logbook.family_members_public,
    logbook.family_actions_public,
    logbook.family_member_status_public
TO logbook_break_glass_admin;

ALTER TABLE logbook.families ENABLE ROW LEVEL SECURITY;
ALTER TABLE logbook.families FORCE ROW LEVEL SECURITY;
ALTER TABLE logbook.family_members ENABLE ROW LEVEL SECURITY;
ALTER TABLE logbook.family_members FORCE ROW LEVEL SECURITY;
ALTER TABLE logbook.family_actions ENABLE ROW LEVEL SECURITY;
ALTER TABLE logbook.family_actions FORCE ROW LEVEL SECURITY;

CREATE POLICY families_writer_insert
    ON logbook.families
    FOR INSERT
    TO logbook_writer
    WITH CHECK (true);
CREATE POLICY families_writer_conflict_select
    ON logbook.families
    FOR SELECT
    TO logbook_writer
    USING (true);
CREATE POLICY family_members_writer_insert
    ON logbook.family_members
    FOR INSERT
    TO logbook_writer
    WITH CHECK (true);
CREATE POLICY family_members_writer_conflict_select
    ON logbook.family_members
    FOR SELECT
    TO logbook_writer
    USING (true);
CREATE POLICY family_actions_writer_insert
    ON logbook.family_actions
    FOR INSERT
    TO logbook_writer
    WITH CHECK (true);
CREATE POLICY family_actions_writer_conflict_select
    ON logbook.family_actions
    FOR SELECT
    TO logbook_writer
    USING (true);

CREATE POLICY families_exporter_select
    ON logbook.families
    FOR SELECT
    TO logbook_exporter
    USING (true);
CREATE POLICY family_members_exporter_select
    ON logbook.family_members
    FOR SELECT
    TO logbook_exporter
    USING (true);
CREATE POLICY family_actions_exporter_select
    ON logbook.family_actions
    FOR SELECT
    TO logbook_exporter
    USING (true);

CREATE POLICY families_break_glass_all
    ON logbook.families
    FOR ALL
    TO logbook_break_glass_admin
    USING (true)
    WITH CHECK (true);
CREATE POLICY family_members_break_glass_all
    ON logbook.family_members
    FOR ALL
    TO logbook_break_glass_admin
    USING (true)
    WITH CHECK (true);
CREATE POLICY family_actions_break_glass_all
    ON logbook.family_actions
    FOR ALL
    TO logbook_break_glass_admin
    USING (true)
    WITH CHECK (true);

DO $supabase_family_roles$
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
                'GRANT SELECT ON logbook.families_public, '
                'logbook.family_members_public, '
                'logbook.family_actions_public, '
                'logbook.family_member_status_public TO %I',
                api_role
            );
        END IF;
    END LOOP;

    IF EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'service_role') THEN
        EXECUTE
            'REVOKE ALL ON logbook.families, logbook.family_members, '
            'logbook.family_actions, logbook.families_public, '
            'logbook.family_members_public, '
            'logbook.family_actions_public, '
            'logbook.family_member_status_public FROM service_role';
    END IF;
END;
$supabase_family_roles$;

COMMENT ON COLUMN logbook.builds.row_format_version IS
    'Logbook row representation version; NULL is the unchanged legacy shape.';
COMMENT ON COLUMN logbook.builds.rung IS
    'Sampling-fraction category; NULL in version 2 for absolute-count builds.';
COMMENT ON COLUMN logbook.builds.requested_k IS
    'Requested record cardinality after resolving any symbolic request.';
COMMENT ON COLUMN logbook.builds.realized_k IS
    'Validated record cardinality in the completed dataset.';
COMMENT ON COLUMN logbook.builds.record_unit IS
    'Normalized entity counted by requested_k and realized_k.';
COMMENT ON TABLE logbook.families IS
    'Dataset families identified separately from their prepared-input checksum.';
COMMENT ON TABLE logbook.family_members IS
    'Two-column association from a dataset family to an immutable build row.';
COMMENT ON TABLE logbook.family_actions IS
    'Append-only revocation and direct-replacement decisions for family members.';

-- Reserved for the spec-engine run identity and deliberately absent here:
-- identity_generation, source_grammar_receipt, spec_binding,
-- authority_versions, code_inventory_digest, artifact_protocol_inventory,
-- run_request, execution_receipt, and schema_version.
