# clients/ — External infrastructure clients and service adapters

## Rules

All secrets come from Vault via `get_database_url()`, `get_api_key()`, or `get_service_config(field)`. `VaultClient` is the only place env vars (`VAULT_ADDR`, `VAULT_ROLE_ID`, `VAULT_SECRET_ID`) are the primary config source — all other clients delegate to it.

Most clients are module-level singletons accessed through factory functions (`get_valkey()`, `get_hybrid_embeddings_provider()`, `get_lattice_client()`). `LLMProvider` is **not** a singleton — instantiate per use. `FilesManager` is **not** a singleton — requires a live `anthropic.Anthropic` instance.

All LLM calls in application code go through `LLMProvider.generate_response()` or `LLMProvider.stream_events()`. Provider-specific code belongs in `clients/llm/dialects/`.

`ModelResolver` in `clients/llm/resolver.py` is the single source of truth for resolving `internal_llm`, conversation LLM names, dialect names, model IDs, endpoint URLs, Vault key names, max tokens, and effort. Callers pass intent (`internal_llm=...`, `conversation_llm=...`, or explicit overrides); they do not fetch provider API keys or route by provider enum.

Adding a new provider is a drop-a-file operation: write a new `clients/llm/dialects/<name>.py`, declare a `Dialect` subclass with `dialect_name` in `DialectName`, implement `from_selection`, and `DialectRegistry` picks it up at startup. The registry validates each candidate (fail-loud) — malformed dialects raise at boot, not at first request. Adding a new dialect_name value also requires extending the `DialectName` literal in `clients/llm/types.py` and the CHECK constraint in the schema.

`SQLiteClient` has no RLS. Manual `user_id` filtering in queries is the only isolation mechanism — it is not redundant.

`PostgresClient` pools are class-level, keyed by database name. `admin=True` uses a separate `{name}_admin` pool with BYPASSRLS role.

## Files

- `vault_client.py` — Vault AppRole auth and secret retrieval; the only permitted source of credentials for all other clients. `preload_secrets()` bulk-loads at startup and raises `RuntimeError` on any failure.
- `postgres_client.py` — Pooled raw-SQL client with automatic RLS context (`SET app.current_user_id`) on connection checkout. All `execute_*` methods are monkey-patched by `utils/perf.py` when `mira.perf` logger is at INFO or DEBUG.
- `valkey_client.py` — Caching, sessions, and rate limiting. Exposes sync, async, and binary clients from a single module-level pool. Pings Valkey on construction — fails immediately if unreachable.
- `sqlite_client.py` — Per-user tool data storage. No pooling; fresh connection per request. Factory `get_sqlite_client(db_path, user_id)` is cached per `user_id:path`.
- `llm_provider.py` — Universal LLM entry point. Builds neutral `Request` objects, resolves model selection through `ModelResolver`, instantiates dialects via `DialectRegistry`, and delegates one provider request's completion/fallback/billing policy to `LLMLifecycle`. Returns `clients.llm.types.Result`. Public methods accept `effort=`/`thinking_tokens=` convenience kwargs OR `thinking=ThinkingConfig(...)` directly (mixing the two raises `ValueError`).
- `llm/` — Provider-neutral LLM layer. `types.py` owns strict `Request`, `Result`, `ThinkingConfig`, `Usage`, `ToolDefinition`, `ToolCall`, `ToolResult`, and `ReasoningArtifact` contracts plus the `DialectName` literal; `thinking.py` owns the `TranslationNote` frozen dataclass and the Anthropic-specific `uses_adaptive_thinking()` model classifier; `events.py` owns provider-neutral streaming event dataclasses; `lifecycle.py` owns one live provider request policy; `accounting.py` owns explicit billing/usage lifecycle policy; `resolver.py` owns model selection (`ModelSelection` carries `dialect_name`); `dialect_registry.py` discovers concrete `Dialect` subclasses by walking the `dialects/` package at first access; `dialects/` owns provider serialization/parsing.
- `llm/dialects/` — Four concrete dialect modules plus shared infrastructure. `base.py` declares the `Dialect` ABC (`native_thinking_fields`, `from_selection`, `_log_translation`). `openai_chat_base.py` extracts the OpenAI Chat Completions transport (message conversion, streaming SSE, tool parsing, usage parsing) with hook methods for dialect-specific thinking serialization, reasoning extraction, cache fields, and foreign-field stripping. `anthropic.py` (Anthropic native, `("effort", "budget")`), `openai.py` (top-level `reasoning_effort`, `("effort",)`), `openrouter.py` (nested `reasoning` block + `reasoning_details`, `("effort", "budget")`), and `groq.py` (`reasoning_effort` + `reasoning_format=parsed`, `("effort",)`). Each dialect translates non-native thinking knobs and emits a `TranslationNote` at WARNING when information is lost.
- `hybrid_embeddings_provider.py` — Local asymmetric embeddings (`mdbr-leaf-ir-asym`, 768-dim) with Valkey-backed cache. `encode_realtime()` for queries; `encode_deep()` for documents.
- `lattice_client.py` — Thin HTTP client for Lattice federation. Only consumed by `pager_tool`. Not exported from `__init__.py`.
- `files_manager.py` — Anthropic Files API upload/delete with persistent per-user segment tracking in `UserDataManager`. Cleanup deletes tracked file IDs across manager instances; transient delete failures keep tracking rows for retry. Not exported from `__init__.py`.
- `__init__.py` — Re-exports `HybridEmbeddingsProvider`, `get_hybrid_embeddings_provider`, `LLMProvider`, `PostgresClient`, `SQLiteClient`, `ValkeyClient`, `get_valkey`, `get_valkey_client`, and selected vault functions. `LatticeClient` and `FilesManager` are excluded — import directly.

## Wiring

`build_batch_params(purpose, system_prompt, messages)` is the standard construction path for Anthropic Batch API param dicts. Batch result extraction normalizes Anthropic SDK messages to `Result` before returning to callers.

Local tool execution is orchestrator-owned. `cns/services/tool_loop.py` propagates contextvars to `ThreadPoolExecutor` workers via `contextvars.copy_context()`. Any new threaded tool execution path must do the same for RLS enforcement to hold.

Usage accounting is an explicit `LLMLifecycle` policy. Hosted builds with the `billing` package present require provider usage and user context before a completion is emitted; missing pricing raises `BillingConfigurationError`. OSS-style builds where the `billing` package is absent disable billing policy explicitly instead of treating hosted billing failures as optional.
