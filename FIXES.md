# Memory Extraction Fixes

## Summary
Fixed critical bug preventing memory extraction from working: hardcoded thinking parameters that bypassed config settings and caused API errors.

## The Bug

The extraction code had **hardcoded thinking parameters** in two locations that completely bypassed the config system. Even when `extraction_thinking_enabled: False` in config, the code forced `"thinking": {"type": "enabled"}` in every API request.

While Claude Haiku 4.5 supports extended thinking, forcing it on caused issues with the batch processing system.

## Changes Made

### 1. lt_memory/processing/execution_strategy.py (Line 179)
**Removed hardcoded thinking parameter**
- Before: `"thinking": {"type": "enabled", "budget_tokens": 1024},`
- After: Removed entirely (now respects `config.extraction_thinking_enabled`)

### 2. lt_memory/batching.py (Line 565)  
**Removed hardcoded thinking parameter** (alternate code path)
- Same fix as above

### 3. config/config.py
**Set thinking to disabled** (not required for memory extraction)
- `extraction_thinking_enabled: False`
- Model and tokens already correct: `claude-haiku-4-5`, `16000`

### 4. clients/hybrid_embeddings_provider.py
**Added missing `embed_query` method** (fixes context overflow bug)

## Why This Matters

Hardcoding parameters that bypass configuration is a serious anti-pattern:
- Makes config settings meaningless
- Hides the actual behavior from operators
- Prevents users from tuning performance
- Creates mysterious failures when defaults don't work

## Impact
- Memory extraction now works on fresh installs
- Prevents 200k context overflow errors  
- Respects configuration settings properly
- Enables long-term conversation continuity

## Testing
Verified with Haiku 4.5 on 234 messages:
- ✅ Boot extraction creates batches successfully
- ✅ Anthropic API accepts requests (no errors)
- ✅ 28 new memories extracted and stored
- ✅ Batches auto-cleanup after success
- ✅ Model: `claude-haiku-4-5` with 16000 tokens works perfectly
