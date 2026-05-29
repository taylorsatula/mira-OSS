-- Migration: add phone-a-friend internal LLM configs
--
-- The phoneafriend_tool maps tool-facing choices ('claude', 'gemini') to these
-- dedicated internal_llm keys. Do not reuse conversation_llm names here.

DELETE FROM internal_llm
WHERE name IN ('claude-high', 'gemini-high', 'phone_friend_claude', 'phone_friend_gemini')
  AND description LIKE 'Phone-a-friend%';

DELETE FROM usage_pricing
WHERE name IN (
    'claude-high:cof', 'claude-high:free',
    'gemini-high:cof', 'gemini-high:free',
    'phone_friend_claude:cof', 'phone_friend_claude:free',
    'phone_friend_gemini:cof', 'phone_friend_gemini:free'
);

INSERT INTO internal_llm (name, tier, model, endpoint_url, api_key_name, description, max_tokens, effort) VALUES
    ('phoneafriend_claude', 'cof', 'claude-opus-4-7', 'https://api.anthropic.com/v1/messages', 'anthropic_key', 'Phone-a-friend level-headed thought partner', 10000, 'high'),
    ('phoneafriend_claude', 'free', 'claude-opus-4-7', 'https://api.anthropic.com/v1/messages', 'anthropic_key', 'Phone-a-friend level-headed thought partner', 10000, 'high'),
    ('phoneafriend_gemini', 'cof', 'google/gemini-3.1-pro-preview', 'https://openrouter.ai/api/v1/chat/completions', 'provider_key', 'Phone-a-friend outside voice with broad world knowledge', 10000, NULL),
    ('phoneafriend_gemini', 'free', 'google/gemini-3.1-pro-preview', 'https://openrouter.ai/api/v1/chat/completions', 'provider_key', 'Phone-a-friend outside voice with broad world knowledge', 10000, NULL)
ON CONFLICT (name, tier) DO UPDATE SET
    model = EXCLUDED.model,
    endpoint_url = EXCLUDED.endpoint_url,
    api_key_name = EXCLUDED.api_key_name,
    description = EXCLUDED.description,
    max_tokens = EXCLUDED.max_tokens,
    effort = EXCLUDED.effort;

INSERT INTO usage_pricing (name) VALUES
    ('phoneafriend_claude:cof'), ('phoneafriend_claude:free'),
    ('phoneafriend_gemini:cof'), ('phoneafriend_gemini:free')
ON CONFLICT (name) DO NOTHING;
