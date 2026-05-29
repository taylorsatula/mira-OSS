-- Switch relationship classification from Sonnet to Haiku
-- The task is well-scoped classification: two short memory texts, 7-type taxonomy
-- with explicit decision tree, 3-field JSON output. Haiku handles this reliably
-- at lower cost per classification.

UPDATE internal_llm
SET model = 'claude-haiku-4-5-20251001'
WHERE name = 'relationship'
  AND tier = 'cof';
