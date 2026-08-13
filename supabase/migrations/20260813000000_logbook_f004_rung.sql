-- #624 ladder revision: add the 4% smoke rung token to the builds rung
-- constraint. Rung grammar history: f001/f010/f100 at genesis; f004 added
-- when measurement showed 1% sits below the tail's own support floor while
-- 4% clears every filing-status stratum naturally.
ALTER TABLE logbook.builds
    DROP CONSTRAINT builds_rung_fraction_token;
ALTER TABLE logbook.builds
    ADD CONSTRAINT builds_rung_fraction_token
    CHECK (rung IN ('f001', 'f004', 'f010', 'f100'));
