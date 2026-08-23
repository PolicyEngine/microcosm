-- #624 ladder revision: add the 25% probe rung token to the builds rung
-- constraint. Rung grammar history: f001/f010/f100 at genesis; f004 added
-- when 1% sat below the capital-gains tail's support floor; f025 added as
-- the largest rung that fits local hardware (128 GB), for memory-curve
-- probes and pre-full battery reads during the calibration era.
ALTER TABLE logbook.builds
    DROP CONSTRAINT builds_rung_fraction_token;
ALTER TABLE logbook.builds
    ADD CONSTRAINT builds_rung_fraction_token
    CHECK (rung IN ('f001', 'f004', 'f010', 'f025', 'f100'));
