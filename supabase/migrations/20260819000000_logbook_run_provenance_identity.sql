-- Add the optional D3 run identity without rewriting historic chain digests.
-- Existing rows remain NULL in PostgreSQL and absent on their canonical wire.

CREATE OR REPLACE FUNCTION logbook.valid_run_provenance_identity(p_value jsonb)
RETURNS boolean
LANGUAGE plpgsql
IMMUTABLE
STRICT
PARALLEL SAFE
SET search_path = pg_catalog, logbook
AS $function$
DECLARE
    generation text;
    grammar jsonb;
    binding jsonb;
    migration jsonb;
BEGIN
    IF jsonb_typeof(p_value) <> 'object'
        OR (SELECT count(*) FROM jsonb_object_keys(p_value)) <> 8
        OR NOT (p_value ?& ARRAY[
            'identity_generation',
            'source_grammar_receipt',
            'spec_binding',
            'authority_versions',
            'code_inventory_digest',
            'artifact_protocol_inventory',
            'run_request',
            'execution_receipt'
        ])
        OR jsonb_typeof(p_value -> 'identity_generation') <> 'number'
        OR (p_value -> 'identity_generation')::text NOT IN ('0', '1')
        OR jsonb_typeof(p_value -> 'authority_versions') <> 'object'
        OR jsonb_typeof(p_value -> 'code_inventory_digest') <> 'string'
        OR (p_value ->> 'code_inventory_digest') !~ '^[0-9a-f]{64}$'
        OR jsonb_typeof(p_value -> 'artifact_protocol_inventory') <> 'object'
        OR jsonb_typeof(p_value -> 'run_request') <> 'object'
        OR jsonb_typeof(p_value -> 'execution_receipt') <> 'object'
    THEN
        RETURN false;
    END IF;

    generation := p_value ->> 'identity_generation';
    grammar := p_value -> 'source_grammar_receipt';
    binding := p_value -> 'spec_binding';
    IF generation = '0' THEN
        RETURN jsonb_typeof(grammar) = 'null'
            AND jsonb_typeof(binding) = 'null';
    END IF;

    IF jsonb_typeof(grammar) <> 'object'
        OR (SELECT count(*) FROM jsonb_object_keys(grammar)) <> 3
        OR NOT (grammar ?& ARRAY[
            'schema_version',
            'canonicalizer_version',
            'migration_chain'
        ])
        OR jsonb_typeof(grammar -> 'schema_version') <> 'number'
        OR (grammar -> 'schema_version')::text !~ '^[1-9][0-9]*$'
        OR jsonb_typeof(grammar -> 'canonicalizer_version') <> 'number'
        OR (grammar -> 'canonicalizer_version')::text !~ '^[1-9][0-9]*$'
        OR jsonb_typeof(grammar -> 'migration_chain') <> 'array'
    THEN
        RETURN false;
    END IF;
    FOR migration IN SELECT value FROM jsonb_array_elements(
        grammar -> 'migration_chain'
    )
    LOOP
        IF jsonb_typeof(migration) <> 'object'
            OR (SELECT count(*) FROM jsonb_object_keys(migration)) <> 2
            OR NOT (migration ?& ARRAY['id', 'sha256'])
            OR jsonb_typeof(migration -> 'id') <> 'string'
            OR NOT logbook.nonempty_trimmed_text(migration ->> 'id')
            OR jsonb_typeof(migration -> 'sha256') <> 'string'
            OR (migration ->> 'sha256') !~ '^[0-9a-f]{64}$'
        THEN
            RETURN false;
        END IF;
    END LOOP;

    IF jsonb_typeof(binding) <> 'object'
        OR (SELECT count(*) FROM jsonb_object_keys(binding)) <> 6
        OR NOT (binding ?& ARRAY[
            'country',
            'schema_id',
            'schema_version',
            'canonicalizer_version',
            'spec_sha256',
            'attestation'
        ])
        OR jsonb_typeof(binding -> 'country') <> 'string'
        OR NOT logbook.nonempty_trimmed_text(binding ->> 'country')
        OR jsonb_typeof(binding -> 'schema_id') <> 'string'
        OR NOT logbook.nonempty_trimmed_text(binding ->> 'schema_id')
        OR jsonb_typeof(binding -> 'schema_version') <> 'number'
        OR (binding -> 'schema_version')::text !~ '^[1-9][0-9]*$'
        OR jsonb_typeof(binding -> 'canonicalizer_version') <> 'number'
        OR (binding -> 'canonicalizer_version')::text !~ '^[1-9][0-9]*$'
        OR jsonb_typeof(binding -> 'spec_sha256') <> 'string'
        OR (binding ->> 'spec_sha256') !~ '^[0-9a-f]{64}$'
        OR jsonb_typeof(binding -> 'attestation') <> 'string'
        OR (binding ->> 'attestation') NOT IN (
            'mirror-attested',
            'bundle-authoritative'
        )
        OR (grammar -> 'schema_version')
            <> (binding -> 'schema_version')
        OR (grammar -> 'canonicalizer_version')
            <> (binding -> 'canonicalizer_version')
    THEN
        RETURN false;
    END IF;
    RETURN true;
END;
$function$;

ALTER TABLE logbook.builds
    ADD COLUMN run_provenance_identity jsonb
    CHECK (
        run_provenance_identity IS NULL
        OR logbook.valid_run_provenance_identity(run_provenance_identity)
    );

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
    ) || CASE
        WHEN p_build.run_provenance_identity IS NULL THEN '{}'::jsonb
        ELSE jsonb_build_object(
            'run_provenance_identity',
            p_build.run_provenance_identity
        )
    END
$function$;

GRANT EXECUTE ON FUNCTION logbook.valid_run_provenance_identity(jsonb)
    TO logbook_writer, logbook_break_glass_admin;

COMMENT ON COLUMN logbook.builds.run_provenance_identity IS
    'Closed D3 run identity; NULL only for historical pre-F1 rows.';
