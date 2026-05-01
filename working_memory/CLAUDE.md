# working_memory/ — Event-driven system prompt composition via trinkets

## Rules

All trinkets extend `EventAwareTrinket` and must define `variable_name: str` as a class attribute — omitting it raises `TypeError` at init. Trinkets with turn-scoped state extend `StatefulTrinket` instead and implement `_expire_items() -> bool` and `_clear_all_state()`. Never subscribe a trinket to `SegmentCollapsedEvent` directly — `WorkingMemory._flush_stateful_trinkets()` handles all stateful trinket cleanup centrally on collapse.

Trinket placement in the composed prompt is controlled exclusively by `SECTION_LAYOUT` in `composer.py`. Sections not listed there default to `system` placement with a warning. To place a new trinket, add its `variable_name` to the appropriate placement list in `SECTION_LAYOUT` — do not route placement inside `generate_content()`.

Infrastructure failures (DB, Valkey) must propagate out of `generate_content()` — the isolation boundary is `WorkingMemory._handle_update_trinket()`, not inside the trinket. No `try/except` around infrastructure calls in trinket implementations.

Portrait injection (`{user_context}`) and `{first_name}` / `{relative time since account creation}` substitutions happen in `core.py:_handle_compose_prompt()`, not in any trinket.

## Files

- `types.py` — `ComposedPrompt`, `TrinketState`, `TrinketStatesMeta`, `AllTrinketStates` TypedDicts shared across composer and core
- `composer.py` — `SystemPromptComposer`: owns section routing via `SECTION_LAYOUT` and prompt assembly into `ComposedPrompt`; sections not in `SECTION_LAYOUT` land in `system` with a warning
- `core.py` — `WorkingMemory`: owns trinket registration, event subscriptions, portrait cache (`_portrait_cache`), and `TrinketState` retrieval from Valkey
- `trinkets/base.py` — `EventAwareTrinket` (ABC) and `StatefulTrinket`; owns Valkey persistence (`TRINKET_KEY_PREFIX`) and `_clear_from_valkey()`
- `trinkets/domaindoc_trinket.py` — domain knowledge document injection with per-document collapse/expand state. Supports shared domaindocs via `utils.domaindoc_shares.get_accepted_shares()`, reads shared docs from owner's `UserDataManager`, renders with `shared_by` attribute
- `trinkets/asyncactivity_trinket.py` — sidebar agent activity feed (`EventAwareTrinket`); SQLite-backed, reads from `sidebar_activity` on each render; items persist until dismissed via `sidebaragents_tool`
- `trinkets/forage_trinket.py` — background forage agent results (`StatefulTrinket`, TTL-scoped errors, dismiss support). Lifecycle: `pending → in_progress (stacked) → success|timeout|failed`. `in_progress` summaries from overwatch accumulate per-iteration so the primary LLM sees the full research arc; cleared when terminal state arrives.
- `trinkets/whilethecatsaway_trinket.py` — curiosity research results (`StatefulTrinket`, all results auto-expire after `RESULT_TTL_TURNS=8`); surfaces what the agent learned and which memories were stored
- `trinkets/location_trinket.py` — user location + weather from Valkey cache (`cache_policy=True`)
- `trinkets/lora_trinket.py` — user model observations from the feedback synthesis pipeline
- `trinkets/manifest_trinket.py` — conversation segment manifest (stable, cached content)
- `trinkets/email_trinket.py` — unread email headers in HUD (`StatefulTrinket`). Thin renderer — receives inbox data from `InboxPollerService` via `UpdateTrinketEvent`, renders `<inbox_status>` XML with nudge instruction. Zero I/O.
- `trinkets/peanutgallery_trinket.py` — high-salience metacognitive directives with TTL expiry (`StatefulTrinket`). Preserves per-guidance `critical` severity and renders standard vs. critical guidance in separate HUD sections so fourth-wall repair instructions apply only to critical items.
- `trinkets/proactive_memory_trinket.py` — surfaced long-term memories; exposes `get_cached_memories()` for orchestrator retention evaluation
- `trinkets/reminder_manager.py` — active reminders fetched and formatted for notification center
- `trinkets/time_manager.py` — current datetime injection into notification center
- `trinkets/__init__.py` — package init, no logic

## Wiring

Composition flow (all synchronous, single request):

`ComposeSystemPromptEvent` → `WorkingMemory._handle_compose_prompt()` broadcasts `UpdateTrinketEvent` per trinket → each trinket's `handle_update_request()` calls `generate_content()`, persists to Valkey, publishes `TrinketContentEvent` → `WorkingMemory._handle_trinket_content()` calls `composer.add_section()` → `composer.compose()` routes by `SECTION_LAYOUT` → `SystemPromptComposedEvent`

`SegmentCollapsedEvent` → `WorkingMemory._flush_stateful_trinkets()` calls `_clear_all_state()` + `_clear_from_valkey()` on every registered `StatefulTrinket`.

`TurnCompletedEvent` → each `StatefulTrinket._on_turn_completed()` increments turn counter, calls `_expire_items()`, and if items were removed triggers `publish_trinket_update()` for a mid-turn refresh.
