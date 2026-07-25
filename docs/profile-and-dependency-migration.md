# Profile and dependency migration

## Agent profile schema 2

LocalLLM now stores Agent profiles in the `localllm-agent-profiles-v2`
application-storage entry. A store contains a schema version, active/default
profile identifiers, and validated profiles.

On first launch after upgrading, the previous Agent role, goal, thinking level,
execution mode, approval settings, and tool permissions are migrated into a
profile named **Default**. Existing legacy keys remain populated for backward
compatibility with older LocalLLM builds.

Each profile owns:

- name and built-in Agent role;
- system, reusable task, and goal instructions;
- normalized workspace folder;
- thinking level and Plan/Edit mode;
- tool permissions and approval behavior;
- temperature, timeout, retries, and maximum local-agent steps;
- creation and update timestamps.

Older imports that use `mode`, `instructions`, or `workingDirectory` are mapped
to `role`, `systemInstructions`, and `workspaceRoot`. Unknown or invalid enum
values fall back to safe defaults. Numeric limits are clamped to the supported
ranges. Invalid imports are rejected without replacing existing profiles.

Profile-store saves copy the last valid JSON to a backup entry before replacing
the current value. If the current entry is malformed, LocalLLM loads that backup
and shows a recovery warning. A malformed current and backup store produces a
new safe Default profile rather than crashing.

Unsaved profile edits are marked in the interface. Switching profiles or
resetting/deleting data requires confirmation where data could be lost, and the
window close path warns while a profile is dirty.

## Application config persistence

`config.json` and the model cache now use staged writes:

1. Serialize to a same-directory temporary file.
2. Flush and sync the staged file.
3. Move the current valid file to `.bak`.
4. Atomically rename the staged file into place.
5. Restore the backup if the final rename fails.

Load operations fall back to `.bak` when the primary file is missing or invalid.
Reset removes both primary and backup files.

## Dependency compatibility

The July 2026 update uses:

- `@earendil-works/pi-coding-agent` 0.82.0;
- `@tauri-apps/api` 2.11.1;
- `@tauri-apps/cli` 2.11.4;
- `@tauri-apps/plugin-dialog` 2.7.2;
- `playwright-core` 1.62.0;
- TypeScript 5.6.3 and Vite 6.4.3 (kept on their existing major lines).
- Rust lockfile updates including Tauri 2.11.5, tauri-build 2.6.3, and
  tauri-plugin-dialog 2.7.2.

Pi 0.82 requires Node.js 22.19 or newer for development. The packaged desktop
app continues to use the compiled Pi sidecar and does not require Node.js at
runtime.

The Pi 0.80 SDK changes do not affect LocalLLM because it invokes Pi through the
documented CLI boundary rather than importing the removed SDK entry points.
LocalLLM deliberately does not pass reasoning flags to Pi; the active Pi model
selects its own supported thinking levels. The local llama.cpp fallback receives
only the reasoning fields that it already supports.

Pi 0.82 adds capability metadata for constrained tools and provider-verified
reasoning levels. LocalLLM keeps unsupported settings out of Pi CLI arguments
and explains this distinction next to the thinking control.

## Known dependency advisory

Pi 0.82.0's published npm shrinkwrap pins `brace-expansion` 5.0.7 through
`minimatch`. npm currently reports GHSA-mh99-v99m-4gvg against that nested
version even though 0.82.0 is the newest published Pi release. Root-level npm
overrides do not replace dependencies pinned by Pi's shrinkwrap, so LocalLLM
does not claim the advisory is resolved. The compiled sidecar is not given
untrusted glob patterns by LocalLLM's profile UI. Update Pi again when an
upstream release includes `brace-expansion` 5.0.8 or newer.
