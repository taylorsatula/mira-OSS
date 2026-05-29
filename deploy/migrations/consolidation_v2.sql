-- Consolidation v2: connected-components clustering, single-pass LLM review, rejection tracking
-- Replaces hub-based candidate selection + two-pass consolidation pipeline

-- Add consolidation rejection tracking (mirrors refinement_rejection_count)
ALTER TABLE memories ADD COLUMN consolidation_rejection_count INTEGER DEFAULT 0;

-- Remove 'consolidation_review' from batch type constraint, add 'verbose_refinement'
ALTER TABLE post_processing_batches
    DROP CONSTRAINT post_processing_batches_batch_type_check;
ALTER TABLE post_processing_batches
    ADD CONSTRAINT post_processing_batches_batch_type_check
    CHECK (batch_type IN ('relationship_classification', 'consolidation', 'verbose_refinement'));

COMMENT ON COLUMN memories.consolidation_rejection_count IS 'Number of times LLM declined to merge this memory. Excluded from consolidation candidates at 3.';
