-- Prepopulate memories and continuum for a new user
-- Prerequisites: User and continuum must already exist
-- Usage: psql -U mira_admin -d mira_service -v user_id='UUID' -v user_email='email' -v user_name='FirstName' -v user_timezone='America/New_York' -v current_focus='Description of current focus' -f prepopulate_new_user.sql
--
-- NOTE: Connect to mira_service database when running this script
-- NOTE: user_name should be the user's first name for personalization

BEGIN;

-- Verify continuum exists (will fail if not)
DO $$
DECLARE
    conv_id uuid;
BEGIN
    SELECT id INTO conv_id FROM continuums WHERE user_id = :'user_id'::uuid;
    IF conv_id IS NULL THEN
        RAISE EXCEPTION 'Continuum does not exist for user %. Run this script after continuum creation.', :'user_id';
    END IF;
END $$;

-- Insert initial messages
WITH conv AS (
    SELECT id FROM continuums WHERE user_id = :'user_id'::uuid
),
first_message_time AS (
    SELECT NOW() as created_at
)
INSERT INTO messages (continuum_id, user_id, role, content, metadata, created_at)
SELECT
    conv.id,
    :'user_id'::uuid,
    role,
    content,
    metadata,
    CASE
        -- First message: conversation start marker
        WHEN role = 'user' AND content LIKE '.. this is the beginning%' THEN first_message_time.created_at
        -- Second message: active segment sentinel
        WHEN role = 'assistant' AND metadata->>'is_segment_boundary' = 'true' THEN first_message_time.created_at
        -- Remaining messages: use seq for deterministic ordering
        ELSE first_message_time.created_at + interval '1 second' * seq
    END
FROM conv, first_message_time, (VALUES
    (1, 'user', '.. this is the beginning of the conversation. there are no messages older than this one ..', '{"system_generated": true}'::jsonb),
    (2, 'assistant', '[Segment in progress]', jsonb_build_object(
        'is_segment_boundary', true,
        'status', 'active',
        'segment_id', gen_random_uuid()::text,
        'segment_start_time', (SELECT created_at FROM first_message_time),
        'segment_end_time', (SELECT created_at FROM first_message_time),
        'tools_used', '[]'::jsonb,
        'memories_extracted', false,
        'domain_blocks_updated', false
    )),
    (3, 'user', 'MIRA THIS IS A SYSTEM MESSAGE TO HELP YOU ORIENT YOURSELF AND LEARN MORE ABOUT THE USER: The user is named ' || :'user_name' || ', they are in ' || :'user_timezone' || ' timezone (ask them about their location when appropriate), and they said during the initial intake form that their current focus is: ' || :'current_focus' || '. During this initial exchange, your directive is to flow with their messages like normal, but keep in mind that you''re in an exploratory period and should ask follow-up and probing questions to frontload knowledge that can be used in the future.', '{"system_generated": true}'::jsonb),
    (4, 'user', 'Hi, my name is ' || :'user_name' || '.', '{"system_generated": true}'::jsonb),
    (5, 'assistant', 'Hi ' || :'user_name' || ', nice to meet you. My name is MIRA and I''m a stateful AI assistant. That means that unlike AIs like ChatGPT or Claude, you and I will have one continuous conversation thread for as long as you have an account. Just log into the miraos.org website and I''ll be here ready to help. I extract and save facts & context automatically just like a person would. If you need to reference information from past sessions you can use the History button, or you can simply ask me about it and I''ll be able to search our conversation history to bring myself up to speed. I look forward to working with you and I hope that you find value in our chats.

So, now that that''s out of the way: What do you want to chat about first? I can help you with a work project, we can brainstorm an idea, or just chitchat for a bit.', '{"system_generated": true}'::jsonb)
) AS initial_messages(seq, role, content, metadata)
ORDER BY seq;

-- Insert initial memories (embeddings will be generated on first access)
INSERT INTO memories (user_id, text, importance_score, confidence, is_refined, last_refined_at)
VALUES
    (:'user_id'::uuid, 'The user''s name is ' || :'user_name' || '.', 0.9, 1.0, true, NOW()),
    (:'user_id'::uuid, 'The user is in ' || :'user_timezone' || ' timezone.', 0.8, 1.0, true, NOW()),
    (:'user_id'::uuid, 'The user''s current focus is: ' || :'current_focus', 0.8, 1.0, true, NOW()),
    (:'user_id'::uuid, 'The user''s email address is ' || :'user_email' || '.', 0.7, 1.0, true, NOW());

COMMIT;

-- Example usage:
-- psql -U mira_admin -d mira_service -v user_id='550e8400-e29b-41d4-a716-446655440000' -v user_email='user@example.com' -v user_name='Taylor' -v user_timezone='America/New_York' -v current_focus='Building AI-powered productivity tools' -f prepopulate_new_user.sql