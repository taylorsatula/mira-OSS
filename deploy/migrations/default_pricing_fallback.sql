-- Migration: add the reserved default pricing fallback
--
-- Any endpoint/model pair without an explicit pricing key should bill at
-- $5/$25 through the __default__ key.

BEGIN;

INSERT INTO usage_pricing (name) VALUES ('__default__')
ON CONFLICT (name) DO NOTHING;

UPDATE usage_pricing
SET input_price_per_mtok = 5.000000,
    output_price_per_mtok = 25.000000
WHERE name = '__default__';

COMMIT;
