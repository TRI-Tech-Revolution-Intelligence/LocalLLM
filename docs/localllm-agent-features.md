# LocalLLM Agent Features

This documents the LocalLLM Agent tab: its built-in agents, skills, and tool set.

## Built-in agents

- **Architect** — plans changes, maps files, and avoids edits until the approach is clear.
- **Ask** — answers questions and explains code without changing the workspace (read-only).
- **Coder** — implements scoped code changes and verifies them.
- **Debugger** — reproduces failures, inspects logs, patches narrowly, and retests.
- **Reviewer** — reviews for bugs, regressions, and missing tests without editing by default.
- **Tester** — finds and runs the right tests, then reports failures clearly.
- **Orchestrator** — breaks work into tasks and delegates to specialist agents.

A **Plan/Edit** toggle gates workspace changes: Plan mode blocks edits, moves, copies, pastes, and shell commands so the agent investigates and proposes a plan; Edit mode allows changes.

## Tool set

Read: `read_file`, `list_files`, `search_files`, `find_files`, `go_to_line`, `search_lines`, `list_code_definition_names`, `semantic_search`.

Edit: `edit_file`, `write_to_file`, `append_to_file`, `insert_content`, `edit_lines`, `replace_in_file`, `apply_diff`, `delete_file`, `copy_file`, `move_file`, `paste_file`.

Web/browser: `web_search` (DuckDuckGo), `web_fetch` (readable Markdown extraction), `download_file`, `browse_url`, `browser_action`.

Command/MCP: `shell`/`execute_command`, `mcp_call`, `use_mcp_tool`, `access_mcp_resource`.

Workflow: `switch_mode`, `new_task`, `delegate_agent`, `ask_followup_question`, `attempt_completion`, `todo_write`, `update_todo_list`, `todoread`, `compact_context`, `skill_load`, `subagent_start`, `local_chat`.

Tool calls may also use the short alias names `read`, `glob`, `grep`, `edit`, `write`, `bash`, `apply_patch`, `webfetch`, `websearch`, `question`, `task`, `todowrite`, `skill`, which map onto the tools above.

## Skills (SKILL.md)

Folder-based Agent Skills (the open `SKILL.md` format) are discovered from:

- the local workspace `.localllm/skills/`,
- the LocalLLM default folder (the app config directory `skills/`),
- the default home `~/.localllm/skills/`.

The workspace takes precedence on name conflicts. `SKILL.md` frontmatter provides `name`, `description`, and optional `modes:` to limit a skill to specific agents. Only metadata is read at discovery and included in the agent prompt; the full instructions are loaded on demand when the agent invokes the skill.

## Chat and task process

The agent keeps the full conversation for a task and feeds it back each turn, so follow-up messages remember earlier tool calls, results, and answers. A new task starts from a clean slate.

## Web content

`web_fetch` returns readable Markdown (headings, links, lists, code blocks) extracted from the main content of a page. HTTP requests use a native Rust client (ureq/rustls) with a curl fallback.

## Local fallbacks and not-yet-complete areas

- `semantic_search` uses a keyword fallback; full semantic search needs an embedding provider and vector index.
- `browser_action` supports open, navigate, search, and close; full automation (click, type, scroll, screenshot) is not implemented.
- MCP is local HTTP JSON-RPC only.
- `new_task` delegates to local specialist prompts; persistent parent/child task trees are not implemented.
- Token-by-token streaming, checkpoints/revert, inline autocomplete, and remote skill sources are not implemented yet.
