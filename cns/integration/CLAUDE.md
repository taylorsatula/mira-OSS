# cns/integration/ — Event bus and CNS dependency wiring

## Rules

`CNSIntegrationFactory` is the sole entry point for constructing the CNS graph. Never instantiate `ContinuumOrchestrator`, `WorkingMemory`, `ToolRepository`, live context compaction services, or peripheral services directly outside this factory — the initialization order encodes hard dependency constraints.

The factory initializes `AsyncWorkBarrier` before the retrieval-backed tool-result summarizer. The summarizer receives the pool's existing `ValkeyMessageCache`; do not resolve the pool again from its worker thread.

Event handlers registered via `event_bus.subscribe()` must be synchronous. Exceptions in handlers are caught and logged by the bus; non-critical handlers must not let errors propagate. For async work inside a handler, spawn a thread with `contextvars.copy_context()`.

`event_type` strings passed to `subscribe()` must match the class `__name__` exactly (`'TurnCompletedEvent'`, not `'turn_completed'`).

Trinkets self-register: constructing `TimeManager(event_bus, working_memory)` is the registration. Add new trinkets in `_get_working_memory()`, after `tool_repo` if they depend on tools. `LiveContextCompactionTrinket` and `LiveContextCompactionService` each instantiate their own `LiveContextCompactionStore` locally; there is no shared store instance.

Peripheral services (`SegmentCollapseHandler`, `ManifestQueryService`, `PeanutGalleryService`, `InboxPollerService`) are initialized via `_initialize_*()` methods that subscribe to events internally — do not subscribe to their event types from outside the factory.

## Files

- `event_bus.py` — Owns pub/sub routing. Keyed by event class name string; callbacks execute synchronously in `publish()` order; exceptions are caught per-callback.
- `factory.py` — Owns the full CNS dependency graph. `create_orchestrator()` is the authoritative construction path; `create_cns_orchestrator()` is the public convenience wrapper. Wires the live-context compaction service and trinket; each creates its own `LiveContextCompactionStore` internally.
