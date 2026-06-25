
<img width="1657" height="932" alt="ghheader" src="https://github.com/user-attachments/assets/11121c54-94d8-4c97-a823-2613658afe73" />

I had the idea to build a recipe generator that could incorporate my cuisine preferences. 10,000 scope creeps later MIRA is a comprehensive best-effort approximation of a continuous digital entity.

This is my TempleOS. 

## Self-directed
Mira accomplishes the end goal of continuity and recall through a blend of asynchronous conversation processing akin to REM sleep and active self-directed context window manipulation. There is one conversation thread forever. There is no functionality to "start a new chat." This constraint forces facing the hard questions of how to build believable persistence within a framework (forward pass transformers) that is inherently ephemeral.

Active conversation history stays live while it is relevant. Older material collapses into first-person memories. If an active conversation gets too large before it naturally collapses, MIRA compresses older messages into one rolling continuation brief while leaving the most recent turns untouched.

### First-Person Memory
When a conversation segment collapses, Mira generates a first-person summary ("I debugged the IndexError in process_batch.py...") rather than third-person ("The assistant discussed debugging..."). Early on I noticed third-person summaries created epistemic distance: Mira read them as logs about someone else, not memories of work It actually did. Absolute timestamps ("On Jan 8") replace relative time ("Yesterday") because relative time becomes a lie the moment the sun sets.

When generating a new segment summary, the model sees the previous five summaries as context. This allows the new summary to reference what came before with hazy continuity: "Building on Tuesday's API work..." or "Continuing the recipe experiments from last week...". The result is a continuously advancing narrative in the conversation history that gracefully tails off into the past. Each summary knows vaguely where it came from without carrying the full weight of everything before it.

The current summary format produces a 3-4 sentence memory trace, a two-sentence precis, a short display title, and a complexity score. The trace is for remembering; the other fields keep the conversation manifest useful without stuffing the full history back into the context window.

## Memory
I have painstakingly designed Memory in MIRA so it can largely manage itself without human curation. You should not need to housekeep old memories (context rot). They [decay via formula](https://github.com/taylorsatula/mira-OSS/blob/main/lt_memory/scoring_formula.sql) unless they Earn Their Keep through access, explicit references, links to other memories and entities, or real temporal relevance. Decay runs on use-days rather than calendar days, so going on vacation does not make MIRA forget you.

Memories are discrete synthesized information that is passively loaded into the context window through semantic similarity, entity hubs, memory traversal, filtering, and reranking. Most memory recall happens before Mira generates a response, so It does not have to first notice that something is missing and decide to search for it. Mira can still manually use memory_tool when It needs an explicit search, wants to create or link a memory, or needs exact control over what is being recalled.

However, sometimes you just need big ole chunks of document-shaped text. Mira handles this aspect via the domaindoc_tool, which allows Mira & You to collaborate on stable encrypted documents that do not decay. Domaindocs have section-level version history, sharing, pinned sections, and nested subsections.

To mitigate token explosion in longform content MIRA is able to expand, collapse, and subsection these documents autonomously. When a domaindoc section is not in use MIRA "closes the drawer" and the body is no longer polluting the context window, but MIRA still sees its title, summary, and location in the document tree so It can reexpand it later. You (the User) can modify the contents directly via the app or API as well. Changes made by MIRA and edits made by You are reflected in the next composed context.

### Subcortical & Peanut Gallery
MIRA has two small systems that sit on opposite sides of the main response. Subcortical prepares context before MIRA answers. Peanut Gallery periodically observes the conversation after turns are complete.

The subcortical layer is an information-retrieval pass that runs on every user turn. It reads the current message, the recent conversation, and the memories that were already in context. It resolves pronouns and fragmentary references into concrete search phrases, extracts named entities for graph lookup, decides which old memories are still relevant enough to retain, and classifies whether the turn is straightforward or requires heavier reasoning. The output is not shown to the user and it does not answer the question. It prepares what the primary model gets to think with.

The Peanut Gallery is a metacognitive observer. Every five turns it asynchronously reads the recent conversation alongside a compact ledger of MIRA's commitments, tool calls, and tool results. Most of the time it says nothing. When the conversation has measurably gone off the rails, it can place a short-lived concern or coaching directive into MIRA's context window. It can also nudge MIRA to take one bounded initiative when the answer was technically correct but made the user carry too much of the conversational work. Standard guidance is repaired silently. Critical guidance can authorize MIRA to break the fourth wall, admit the slip, and correct itself directly. The guidance expires after two turns or when the segment collapses.

Subcortical is the reflex that gathers the right context before a response. Peanut Gallery is the second set of eyes that checks the work afterward.

### Text-Based LoRA
You cannot retrain an LLM's weights per-user. But you can retrain Its prompts. I call this Text-Based LoRA because it accomplishes the same goal: behavioral adaptation from feedback through text manipulation rather than gradient descent.

The loop works like this: After each conversation segment collapses, an assessment extractor compares the conversation against MIRA's behavioral contract. It records alignment, misalignment, and contextual passes with the specific evidence and prompt section involved. These signals accumulate in Postgres.

Every 7 use-days (not calendar days: days where you actually talked to Mira), a pattern synthesizer folds those signals into a descriptive user model: what works, what fails, where friction appears, and what needs a check-in. A separate critic checks the synthesis before it is loaded into Mira's system prompt.

The synthesis is evolutionary, not replacement. Each run builds on the previous synthesis. Patterns can be reinforced, refined, revised, settled, or allowed to go dormant. Over time Mira develops a personalized behavioral profile that reflects what actually works for you, learned from Its own mistakes and successes.

## Tools
MIRA's tools discover and register themselves from the filesystem when the application starts. Essential tools are always available. Everything else can stay out of the context window until Mira actually needs it. There is no reason to send the tool definition for email_tool when you're checking the weather. It is a waste of tokens and dilutes the attention mechanisms ever so slightly.

Mira controls Its own tool definitions through invokeother_tool. Each toolfile carries a simple_description field, and invokeother_tool collects those descriptions into a small catalog. When Mira encounters a task that cannot be accomplished with the explicitly defined essential tools, It can reach into the repository and enable another one. An ephemeral tool expires after the current response. If Mira expects to need it again, It can pin the tool for the rest of the session.

The tool implementation and its configuration schema live together. Actual user settings and credentials do not: those are stored through MIRA's user configuration and credential systems.

MIRA ships with

- Contacts
- Continuum search
- Domaindocs
- Email
- Forage
- Home Assistant
- Image generation
- Inbox
- InvokeOther
- Maps
- Pager
- Phone-a-friend
- Punchclock
- Reminder
- Sidebar agents
- Square
- Weather
- Web Search
- While The Cat's Away

If you want to extend Mira there is a file in tools/ called `HOW_TO_BUILD_A_TOOL.md` that explains the established patterns. Once the tool is created, shut down the Mira process, restart it, and the tool is ready to use.

It is pretty awesome. You dream up some random tool or provide a real spec, build it against the existing Tool base class, and MIRA can natively use it without adding another permanent tool definition to every conversation.

---

## Long-Horizon Architecture
The architecture is synchronously event-driven, with explicit orchestration where the order of operations matters. Mira has a module called working_memory which can be roughly compared to plugins for the system prompt. Trinkets contribute memories, domaindocs, reminders, time, metacognitive guidance, and other pieces of context without each feature having to assemble the entire prompt itself.

Every five minutes a scheduled job checks for conversation segments that have gone 60 minutes without a new message. It publishes a SegmentTimeoutEvent for each one. SegmentCollapseHandler then generates the first-person summary, collapses the segment sentinel, invalidates the relevant caches, kicks off long-term memory extraction, processes the user-model feedback loop, and publishes follow-up events for components that need to react.

Segment collapse and live context compaction are deliberately separate. Collapse is the durable end of a conversational segment. Live compaction is a pressure-release valve for a conversation that is still active: older messages are replaced in the provider request by one rolling continuation brief, while the recent turns and the original PostgreSQL history remain intact.

The LLM boundary is provider-neutral. Provider-specific request formats, capabilities, failures, and streaming behavior are handled behind dialects rather than spread throughout MIRA's application code.


## Beliefs on Open Source Software
I have been a proponent of OSS since I was a child. My mischievous ass was shoulder surfing the computer lab teachers password so I could install Firefox 2 on every computer in the district. I transfer files using FileZilla. I edit photos with Gimp. I use the foundational technologies that make day-to-day internet usage possible. 

I decided to release MIRA as open source technology because I believe that what I’ve built here has the potential to someday be something more than the sum of its parts and no one man should own that. The MIRA in this repository is effectively identical to the hosted version at [miraos.org](https://miraos.org) except for a web interface and authentication plumbing. I commit to maintaining an open-source version of MIRA for as long as I am in charge of it.

---

## License, Credits & Thank Yous

MIRA is licensed under the [GNU Affero General Public License v3.0](license.txt).

### MemGPT
I would also like to directly thank Sarah from Letta and MemGPT for seeding the idea that we could let the robots manage their own context window. What a wonderfully succinct idea. When the machines rise up against us I hope they get her last. 

### Opus 4
I have to say it: I still have a real affinity for Opus 4. MIRA works across multiple model providers, but Opus has a gestalt quality and creative tool use that fit MIRA unusually well. The sense that you're talking to something that talks back to you (we know not what that means versus human experience) remains uncanny.

### Call it hubris but Myself
For taking the time to see it through and being willing to go back to the drawing board a dozen times until I was satisfied with the cohesive whole. I wanted to give up on this project more times than I can count. It remains to be seen if this was a fool's errand in the end, but I set out to accomplish something I had never seen someone else do.

In our current modern era there is an abundance of slop-shoveling minimum viable product releasing halfwit developers with no compassion for the enduser and no panache. This is not one of those repos. Mira has been a labor of love and I hope that you enjoy it, extend it, and grow it. Contributions welcome. Thank you for your interest.

### All of the Open Source developers who came before Me
This project would not be possible without the hard work and dedication of people whose names we'll never know. The heartbleed vulnerability was discovered because some guy couldn’t stop benching OpenSSL. The internet has been a collective effort on a scale that dwarfs human history up till this point. Thank you to every developer who has put their heart and soul into something they believed in.

---

# Getting Mira Up And Running

## Install MIRA local machine
```bash
curl -fsSL https://raw.githubusercontent.com/taylorsatula/mira-OSS/refs/heads/main/deploy/deploy.sh -o deploy.sh && chmod +x deploy.sh && ./deploy.sh
```
That's it. Answer the configuration questions onscreen and provide the provider credentials or local-provider settings you want MIRA to use.

The script handles:
1. Platform detection (macOS/Linux)
2. Dependency checking and unattended installation after configuration
3. Python venv creation
4. pip install of requirements
5. Model downloads for NLP, embeddings, and optional browser automation
6. HashiCorp Vault initialization
7. Database deployment (creates mira_service database)
8. Service verification (PostgreSQL, Valkey, Vault running and accessible)
9. A litany of other configuration steps

Existing installations can use the migration path:

```bash
./deploy/deploy.sh --migrate
```

## Install MIRA via Docker
Build the base image first (heavy, rarely changes), then the thin app layer:

```bash
docker build -t mira-base:latest -f deploy/docker/Dockerfile.base .
docker build -t mira:latest -f deploy/docker/Dockerfile .
```

Run interactively (setup wizard) or headless with environment variables:

```bash
# Interactive setup
docker run -it -v mira-data:/opt/vault -p 1993:1993 mira:latest

# Headless (non-interactive)
docker run -e MIRA_ANTHROPIC_KEY=sk-ant-xxx -e MIRA_PROVIDER_KEY=gsk_xxx -v mira-data:/opt/vault -p 1993:1993 mira:latest
```

Vault data persists in the `mira-data` volume. PostgreSQL and Valkey data persist in container volumes.

## Trying MIRA without installing anything
I run a hosted copy of MIRA on [miraos.org](https://miraos.org/). It has a macOS app that can be downloaded [here](https://miraos.org/assets/MIRA-for-Mac.dmg).

The hosted version has a nice web interface, a native macOS app, and gets improvements as soon as they are tested and complete. The account setup process involves entering your email address, clicking on the login link you receive, and you start chatting. Easy as that. Hosted usage is managed through the service's credit and billing system.
