# MIRA Research & Bug Fixes

## 2026-02-09: Segment Collapse Authentication Fix

### Problem
Segment collapse failing every 5 minutes with 401 authentication errors:
```
Generic OpenAI client HTTP error: 401 - Invalid bearer token
COLLAPSE FAILURE: Segment ... collapse failed
```

Segments not compressing → growing token overhead → rate limit issues

### Root Cause
Database misconfiguration in `internal_llm` table:
```sql
name: summary
endpoint_url: https://api.anthropic.com/v1/messages
api_key_name: anthropic_key
```

When `endpoint_url` is set, `LLMProvider.generate_response()` routes to `GenericOpenAIClient`, designed for OpenAI-compatible APIs. Anthropic requires different auth headers:
- OpenAI-compatible: `Authorization: Bearer {key}`  
- Anthropic: `x-api-key: {key}` + `anthropic-version: 2023-06-01`

### Solution
Update database to empty string (NOT NULL constraint prevents NULL):
```python
import psycopg2
from clients.vault_client import get_database_url

admin_url = get_database_url('mira_service', admin=True)
conn = psycopg2.connect(admin_url)
cur = conn.cursor()
cur.execute("UPDATE internal_llm SET endpoint_url = '' WHERE name = 'summary';")
conn.commit()
```

Empty string evaluates to `False` in `if endpoint_url:` check (line 893 in `llm_provider.py`), triggering native Anthropic client.

### Result
✅ Segment collapse working without auth errors  
✅ Memory properly compresses  
✅ Reduced token overhead

### Note
This was an original configuration bug, not introduced by recent changes.

---

## Framework Token Optimization (In Progress)

### Challenge
Five philosophical frameworks total ~26,000 tokens:
- Ontological Framework: ~5,200 tokens
- Stillness Framework: ~5,200 tokens  
- Negentropy Framework: ~5,200 tokens
- Resonance Framework: ~5,200 tokens
- Emergence Framework: ~5,200 tokens

Combined with memories (~8K) and conversation history, frequently exceeds Tier 1 rate limits (30K tokens/minute).

### Approach
Condense each framework to ~800 tokens while preserving core functionality. Testing with Framework 1 (Ontological) first.

