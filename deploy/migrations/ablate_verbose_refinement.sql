-- Ablate verbose refinement pipeline
-- The verbose refinement pipeline (trim/split/do_nothing for memories >70 chars) contradicts
-- the extraction prompt which deliberately produces 130-250 char pre-stitched memories.
-- The sync path had a KeyError bug (wrong template variable) confirming it never ran successfully.
-- No link transfer on trim/split means archived memories leave dangling references.

-- Drop verbose-refinement-only columns from memories
ALTER TABLE memories DROP COLUMN IF EXISTS is_refined;
ALTER TABLE memories DROP COLUMN IF EXISTS last_refined_at;
ALTER TABLE memories DROP COLUMN IF EXISTS refinement_rejection_count;

-- Update batch type constraint to remove verbose_refinement
ALTER TABLE post_processing_batches
    DROP CONSTRAINT post_processing_batches_batch_type_check;
ALTER TABLE post_processing_batches
    ADD CONSTRAINT post_processing_batches_batch_type_check
    CHECK (batch_type IN ('relationship_classification', 'consolidation'));

-- Remove orphaned internal_llm entries (no code calls get_internal_llm('refinement'))
DELETE FROM internal_llm WHERE name = 'refinement';

-- Remove orphaned usage_pricing entry (table may not exist in all environments)
DO $$ BEGIN
    DELETE FROM usage_pricing WHERE name = 'refinement';
EXCEPTION WHEN undefined_table THEN NULL;
END $$;
