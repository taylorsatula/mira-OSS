# docs/ — Operator-facing documentation

## Rules

- Keep deployment docs aligned with `deploy/` scripts. If a script prints a path in `docs/`, that file must exist and describe the current workflow.
- Prefer operational checklists over design essays. These files are read when installation automation cannot continue or when the operator needs to prepare external resources.
- Do not document removed helper scripts as if they still exist. Point to the active script, command, or manual procedure instead.

## Files

- `MANUAL_INSTALL.md` — Manual setup notes for platforms outside the automated installer path.
- `OFFLINE_MODELS.md` — Local-model preparation notes for offline/local LLM installs.
- `SEGMENT_SYSTEM.md` — Current segment collapse, sentinel, and live-compaction overview.
