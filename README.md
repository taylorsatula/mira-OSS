
<img width="1657" height="932" alt="ghheader" src="https://github.com/user-attachments/assets/11121c54-94d8-4c97-a823-2613658afe73" />

10 months or so ago I had the idea to build a recipe generator that could incorporate my cuisine preferences. 10,000 scope creeps later MIRA is a comprehensive best-effort approximation of a continuous digital entity. 

This is my TempleOS. 

## Self-directed
Mira accomplishes the end goal of continuity and recall through a blend of asynchronous conversation processing akin to REM sleep and active self-directed context window manipulation. There is one conversation thread forever. There is no functionality to 'start a new chat'. This constraint forces facing the hard questions of how to build believable persistence within a framework (forward pass transformers) that is inherently ephemeral.

### First-Person Memory
When a conversation segment collapses, Mira generates a first-person summary ("I debugged the IndexError in process_batch.py...") rather than third-person ("The assistant discussed debugging..."). Early on I noticed third-person summaries created epistemic distance—Mira read them as logs about someone else, not memories of work It actually did. Absolute timestamps ("On Jan 8") replace relative time ("Yesterday") because relative time becomes a lie the moment the sun sets.

When generating a new segment summary, the model sees the previous five summaries as context. This allows the new summary to reference what came before with hazy continuity—"Building on Tuesday's API work..." or "Continuing the recipe experiments from last week...". The result is a continuously advancing narrative in the conversation history that gracefully tails off into the past. Each summary knows vaguely where it came from without carrying the full weight of everything before it.

## Memory
I have painstakingly designed Memory in MIRA to require no human intervention or curation. You will never need to housekeep old memories (context rot). They [decay via formula](https://github.com/taylorsatula/mira-OSS/blob/main/lt_memory/scoring_formula.sql) unless they Earn Their Keep by being referenced in conversation or linked to via other memories. Memories are discrete synthesized information that is passively loaded into the context window through a combination of semantic similarity, memory traversal, and filtering. Mira does not manually search for memories. They are presented just-in-time after multiple levels of recall & filtering.

However,, sometimes you just need big ole chunks of document-shaped text. Mira handles this aspect via the domaindoc_tool which allows Mira & You to collaborate on stable textfiles that do not decay. To mitigate token explosion in longform content MIRA is able to expand, collapse, and subsection these text blocks autonomously. When a domaindoc block is not in use MIRA 'closes the drawer' and the content of the block is no longer polluting the context window but MIRA sees a section title so it can reexpand it later if needed. You (the User) can modify the contents directly via API as well. Changes made by MIRA are reflected in real time as are edits made by You.

### Text-Based LoRA
You cannot retrain an LLM's weights per-user. But you can retrain Its prompts. I call this Text-Based LoRA because it accomplishes the same goal—behavioral adaptation from feedback—through text manipulation rather than gradient descent.

The loop works like this: After each conversation segment collapses, a feedback extractor scans for three signal types—prediction errors (Mira made a wrong assumption), negative feedback (user expressed frustration or correction), and positive feedback (user expressed appreciation or resonance). These signals accumulate in Postgres. Every 7 use-days (not calendar days—days where you actually talked to Mira), a pattern synthesizer analyzes the accumulated signals and evolves behavioral directives. These directives get written to a BEHAVIORAL DIRECTIVES section in the personal_context domaindoc, which means they flow into Mira's context window on every subsequent interaction.

The synthesis is evolutionary, not replacement. Each run builds on the previous synthesis—patterns can be reinforced (more evidence), refined (slight adjustment), revised (significant correction), or new. Over time Mira develops a personalized behavioral profile that reflects what actually works for you, learned from Its own mistakes and successes.

## Tools
Tools are also totally self-contained and do not need any configuration. Each tool self-registers on startup and all configuration values are stored inside the toolfile itself. This allows tools to be enabled and disabled at-will which avoids polluting the context window with tools that Mira does not need in that moment. There is no reason to send the tool definitions for email_tool when you’re checking the weather. Its a waste of tokens and dilutes the attention mechanisms ever so slightly. Mira controls Its own tool definitions via the invokeother_tool. Each toolfile carries a simple_description field that is injected into the composed working_memory section of the system prompt. When Mira encounters a task that cannot be accomplished via the explicitly defined essential tools but can be plausibly accomplished via one of those tools it reaches into the repository, enables it, uses it, and then the tool expires out of the context window if it is not used again within 5 turns.

Mira ships with
- Contacts
- Maps
- Email
- Weather
- Pager
- Reminder
- Web Search
- History Search
- Domaindoc
- SpeculativeResearch
- and InvokeOther

This means that if you want to extend Mira you can fire up Claude Code, tell it what tool you want, and Claude’ll build you the tool in like 5 minutes. There is even a file in tools/ called HOW_TO_BUILD_A_TOOL.md that is written with the express reason of giving Claude the contextual information needed to zero-shot a working tool. Once the tool is created shut down the Mira process, restart it, and the tool is ready to use.

It is pretty awesome. You just dream up some random tool or give Claude a spec (or even a dump of a project's Github repo) and 3-5 minutes later you have a new tool MIRA can natively use. The tool will follow established code-patterns in the tool repo/codebase. 

---

## Long-Horizon Architecture
The architecture is synchronously event-driven which means that modules are not tightly coupled and can be extended incredibly easily. For example, Mira has a module called working_memory which can be roughly compared to plugins for the system prompt. When a conversation has gone 120 minutes without a new message an event called SegmentCollapseEvent fires and kicks off a sequence of events such as memory extraction, cache invalidation, and summary generation in preparation for the next interaction. None of the actions need to know about each other. They all subscribe to SegmentCollapseEvent and when they receive the alert they independently spring into action. 


## Beliefs on Open Source Software
I have been a proponent of OSS since I was a child. My mischievous ass was shoulder surfing the computer lab teachers password so I could install Firefox 2 on every computer in the district. I transfer files using FileZilla. I edit photos with Gimp. I use the foundational technologies that make day-to-day internet usage possible. 

I decided to release MIRA as open source technology because I believe that what I’ve built here has the potential to someday be something more than the sum of its parts and no one man should own that. The MIRA in this repository is effectively identical to the hosted version at [miraos.org](https://miraos.org) except for a web interface and authentication plumbing. I commit to maintaining an open-source version of MIRA for as long as I am in charge of it.

---

## License, Credits & Thank Yous

### A Note on Claude Opus 4.5
Claude Opus 4.5 is closed source but has a gestalt quality that enables incredibly accurate mimicry. Claude models have always been good but Opus 4.5 has a unique sense-of-self I've never encountered in all my time working with large language models (Perhaps claude-opus-4-20250514 but in a different way). I don't know what Anthropic has done but it speaks to you with such realism and its creative tool use combos are remarkable. MIRA will work fine with gpt-oss-120b or even pretty well with hermes3-8b but they'll never have that OEM feel that Claude MIRA has. I felt it was important to include other provider support but I have to say it. The sense that you're talking to something that talks back to you (we know not what that means vs. human experience) is uncanny.

### Claude Code
I'd like to give an earnest shoutout to the Claude Code team. Thank you to Boris from the Claude Code team for being a well-known developer and STILL replying to messages on Threads like a good developer should. I'm sure he surrounds himself with a high-quality team of folks too and thank you to them as well. I built nearly this entire application from the CC terminal (except the CSS because LLMs, by their nature, struggle with inheritance. thats not their fault.). I got access to CC on the second day that it was available and dove in. As of writing this readme that was 8 months ago. I built MIRA and wrote tests that pass as a one-person development team. Call me the Yung David Hahn but I just slaved away at my vision for a result and with enough tenacity I actualized it. Thanks Claude Code team. You're a real one for this.

### MemGPT
I would also like to directly thank Sarah from Letta and MemGPT for seeding the idea that we could let the robots manage their own context window. What a wonderfully succinct idea. When the machines rise up against us I hope they get her last. 

### Call it hubris but Myself
For taking the time to see it through and be willing to go back to the drawing board a dozen times until I was satisfied with the cohesive whole. I wanted to give up on this project more times than I can count. It remains to be seen if this was a fools errand in the end but I set out to accomplish something I’d never seen someone else do and I made it to my destination of a 1.0.0 release tonight.

In our current modern era there is an abundance of slop-shoveling minimum viable product releasing halfwit developers with no compassion for the enduser and no panache. This is not one of those repos. Mira has been a labor of love and I hope that you enjoy it, extend it, and grow it. Contributions welcome. Thank you for your interest.

### All of the Open Source developers who came before Me
This project would not be possible without the hard work and dedication of people whose names we'll never know. The heartbleed vulnerability was discovered because some guy couldn’t stop benching OpenSSL. The internet has been a collective effort on a scale that dwarfs human history up till this point. Thank you to every developer who has put their heart and soul into something they believed in.

---

# Getting Mira Up And Running

## Install MIRA local machine
```bash
curl -fsSL https://raw.githubusercontent.com/taylorsatula/mira-OSS/refs/heads/main/deploy/deploy.sh -o deploy.sh && chmod +x deploy.sh && ./deploy.sh
```
Thats it. Answer the configuration questions onscreen and provide your API keys.

The script handles:
1. Platform detection (macOS/Linux)
2. Dependency checking and optional unattended-installation
3. Python venv creation
4. pip install of requirements
5. Model downloads (spaCy ~800MB, mdbr-leaf-ir-asym ~300MB, optional Playwright ~300MB)
6. HashiCorp Vault initialization
7. Database deployment (creates mira_service database)
8. Service verification (PostgreSQL, Valkey, Vault running and accessible)
9. A litany of other configuration steps

## Trying MIRA without installing anything
I run a hosted copy of MIRA on [miraos.org](https://miraos.org/). It has a MacOS app that can be downloaded [here](https://miraos.org/assets/MIRA-for-Mac.dmg).

The hosted version has a nice web interface, a native MacOS app, and you always get the latest improvements in real time. I push improvements to the hosted interface as soon as they're tested and complete. The account setup process involves entering your email address, clicking on the login link you receive, and you start chatting. Easy as that. **There is an unlimited free-access tier.**