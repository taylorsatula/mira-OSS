# config/prompts/ — LLM Prompt Templates

## Rules

- Naming: `{feature}_system.txt` = system message (instructions, output format); `{feature}_user.txt` = user message with `{variable}` placeholders. Some prompts are a single combined file with no separate user template.
- Template variables use Python `.format()` syntax: `{variable_name}`. Literal braces in JSON examples require doubling: `{{` / `}}`.
- Wrap runtime data in descriptive XML tags within user templates: `<conversation>`, `<entity_groups>`, `<candidate_memories>`, etc. — not bare text.
- All prompt loading must go through `config.prompts.loader.load_prompt(filename)` (importable as `from config.prompts import load_prompt`). Never roll your own `open()`/`read_text()` for prompt files. The loader guarantees UTF-8 encoding, existence checks with descriptive errors, and consistent `.strip()`. Use `load_prompt("file.txt", required=False)` for optional addendum prompts that may not exist.
- `agents/` holds prompts for autonomous sidebar agents. `base_system.txt` is the shared loop-mechanics preamble; agent-specific rubrics are `{agent_id}_system.txt`. Loaded via `load_prompt("agents/{agent_id}_system.txt")`.
- `variants/` holds experimental subcortical prompt variants for tuning. Nothing in `variants/` is loaded in production.

## Files

- `memory_extraction_system.txt` / `memory_extraction_user.txt` — LT_Memory extraction: pulls durable memories from conversation segments. Variable: `{formatted_messages}`. Consumer: `lt_memory/processing/extraction_engine.py`.
- `segment_summary_system.txt` / `segment_summary_user.txt` — Segment collapse diarist: first-person memory traces with 2-sentence precis. Output tags: `<mira:precis>`, `<mira:display_title>`, `<mira:complexity>`. Variables: `{previous_summaries}`, `{conversation_text}`, `{tools_used}`. Consumer: `cns/services/summary_generator.py`.
- `synthesis_summary_system.txt` / `synthesis_summary_user.txt` — Merges multi-chunk partial summaries into a single unified memory trace with precis. Output tags: `<mira:precis>`, `<mira:display_title>`, `<mira:complexity>`. Consumer: `cns/services/summary_generator.py`.
- `live_context_compaction_system.txt` / `live_context_compaction_user.txt` — Active-chat continuation brief for live request shaping. Variables are replaced by `LiveContextCompactionService` with simple string replacement (`{covered_end}`, `{new_history}`), not Python `.format()`, because the output contract includes literal JSON braces. The recovery instruction (search-based `continuum_tool` path) and the `<mira:compacted_active_context>` wrapper are appended programmatically by `_wrap_brief()` in the service, not by the LLM. Consumer: `cns/services/live_context_compaction_service.py`.
- `assessment_extraction_system.txt` / `assessment_extraction_user.txt` — Evaluates conversation against system prompt sections; produces alignment/misalignment/contextual_pass signals with evidence. XML output. Consumer: `cns/services/assessment_extractor.py`.
- `thinking_block_instructions.txt` — Addendum injected into assessment extraction when thinking content exceeds 70%. Not a standalone prompt. Consumer: `cns/services/assessment_extractor.py`.
- `user_model_synthesis_system.txt` / `user_model_synthesis_user.txt` — Evolves user model observations from assessment signals. Section-anchored XML output. Consumer: `cns/services/user_model_synthesizer.py`.
- `user_model_critic_system.txt` / `user_model_critic_user.txt` — Quality critic for user model drafts: catches observation laundering, personality labels, contradictions. Pass/fail XML output. Consumer: `cns/services/user_model_synthesizer.py`.
- `portrait_synthesis_system.txt` / `portrait_synthesis_user.txt` — Produces concise factual user portrait injected via `{user_context}` into `config/system_prompt.txt`. Variable: `{segment_summaries}`. Consumer: `cns/services/portrait_service.py`.
- `portrait_refinement_system.txt` — Refines existing portrait from user instructions. Minimal-change editing: preserve style, length (4-5 sentences ~100-150 words), and factual tone. No user template — user message built inline from existing portrait + instructions. Consumer: `cns/services/portrait_service.py:refine_portrait()`.
- `subcortical_system.txt` / `subcortical_user.txt` — Pre-LLM IR stage: entity extraction, passage filtering, query expansion, complexity assessment. XML output. Consumer: `cns/services/subcortical.py`.
- `domaindoc_summary_system.txt` / `domaindoc_summary_user.txt` — One-sentence section summaries (max 100 chars). Plain text output. Consumer: `cns/services/domaindoc_summary_service.py`.
- `peanutgallery_system.txt` / `peanutgallery_user.txt` — Metacognitive observer: receives conversation + execution trace, emits noop/concern/coaching signal as out-of-band corrective guidance. Consumer: `cns/services/peanutgallery_model.py`.
- `repulsion_rewriter_system.txt` / `repulsion_rewriter_user.txt` — Register-aware rewrite prompt for Repulsed feedback captures. Variables: `{user_message}`, `{ai_response}`, `{matched_tells}`. Consumer: `cns/api/actions.py:FeedbackDomainHandler`.
- `behavioral_primer.txt` — Static synthetic dialogue (4 turns, user/assistant/user/assistant) injected between collapsed segment summaries and continuity messages as ambient behavioral priming for authenticity directives. Role-delimited format: `[role]` header + content, `---` separator. No template variables. Consumer: `cns/core/segment_cache_loader.py`.
- `agents/base_system.txt` — Shared agent loop preamble: identity, loop mechanics, complete_task requirement. Prepended to agent-specific prompts when `inherit_base_prompt=True`. Consumer: `agents/base.py`.
- `agents/forage_system.txt` — Background research agent rubric: quality rubric, output format. Consumer: `agents/implementations/forage_agent.py`.
- `agents/whilethecatsaway_system.txt` — Curiosity-driven background research rubric: open-ended exploration, memory storage with pending_id tracking, structured completion summary. Consumer: `agents/implementations/whilethecatsaway_agent.py`.
- `agents/memory_curator_integration.txt` — Memory curator integration-mode rubric: link / merge / stand-alone decisions for newly extracted memories. Self-contained (no `base_system.txt` preamble). Consumer: `agents/implementations/memory_curator_agent.py`.
- `agents/memory_curator_floor.txt` — Memory curator floor-mode rubric: archive / salvage triage for sampled low-value memories. Self-contained. Consumer: `agents/implementations/memory_curator_agent.py`.
- `entity_merge_system.txt` / `entity_merge_user.txt` — Entity dedup judge: given groups of similar-named entities, decide which are the same and pick a canonical to merge into. JSON output. Consumer: `lt_memory/entity_merge.py`.
