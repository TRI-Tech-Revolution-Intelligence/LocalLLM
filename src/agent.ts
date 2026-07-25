import { invoke } from "@tauri-apps/api/core";
import { open } from "@tauri-apps/plugin-dialog";
import { openUrl } from "@tauri-apps/plugin-opener";
import {
  agentAutoAcceptStorageKey,
  agentAutoCompactStorageKey,
  agentActiveProfileStorageKey,
  agentContextSummaryStorageKey,
  agentExecutionModeStorageKey,
  agentGoalStorageKey,
  agentMcpServersStorageKey,
  agentModeStorageKey,
  agentPermissionsStorageKey,
  agentProfilesBackupStorageKey,
  agentProfilesStorageKey,
  agentSkillsStorageKey,
  agentSubagentsStorageKey,
  agentTaskHistoryStorageKey,
  agentThinkingLevelStorageKey,
  agentTodosStorageKey,
  agentYoloModeStorageKey,
  defaultAgentPermissions,
} from "./constants";
import { confirmAction } from "./confirm-dialog";
import { $ } from "./dom";
import {
  createAgentProfile,
  duplicateAgentProfile,
  exportAgentProfiles,
  loadAgentProfileStore,
  normalizeAgentProfileStore,
  validateAgentProfile,
} from "./agent-profiles";
import type {
  AgentExecutionMode,
  AgentMcpServer,
  AgentMode,
  AgentPiStatus,
  AgentPathInfo,
  AgentPermission,
  AgentPermissions,
  AgentProfile,
  AgentProfileStore,
  AgentReadResult,
  AgentSkill,
  AgentSkillFile,
  AgentSubagent,
  AgentThinkingLevel,
  AgentTodo,
  AgentTool,
  AgentTransferResult,
  AgentWriteResult,
  AppConfig,
  CommandOutput,
  ServerStatus,
} from "./types";

type AgentRole = "agent" | "tool" | "user";
type ToolArgs = Record<string, unknown>;
type SpeechRecognitionConstructor = new () => {
  continuous: boolean;
  interimResults: boolean;
  onresult: ((event: { results: ArrayLike<ArrayLike<{ transcript: string }>> }) => void) | null;
  onerror: (() => void) | null;
  start: () => void;
};

interface AgentToolCall {
  tool: string;
  args?: ToolArgs;
}

interface AgentModelAction {
  tool?: string;
  args?: ToolArgs;
  final?: string;
  message?: string;
  reasoning?: string;
  rationale?: string;
}

interface AgentToolDefinition {
  name: string;
  description: string;
  permission: AgentTool;
  run: (args: ToolArgs) => Promise<string>;
}

interface CodingAgentProfile {
  id: AgentMode;
  name: string;
  description: string;
  instructions: string;
  allowedTools: string[];
}

interface AgentTranscriptMessage {
  role: AgentRole;
  message: string;
  at: number;
}

interface AgentTaskSession {
  id: string;
  title: string;
  startedAt: number;
  updatedAt: number;
  messages: AgentTranscriptMessage[];
}

const permissionTools = [
  "read",
  "edit",
  "move",
  "copy",
  "paste",
  "browse",
  "shell",
  "todo",
  "skill",
  "subagent",
  "externalDirectory",
] as const satisfies readonly AgentTool[];

let agentPermissions: AgentPermissions = { ...defaultAgentPermissions };
let yoloMode = false;
let autoAccept = false;
let agentGoal = "";
let agentMode: AgentMode = "coder";
let activeAgentProfile: AgentMode = "coder";
let agentExecutionMode: AgentExecutionMode = "edit";
let agentThinkingLevel: AgentThinkingLevel = "medium";
let agentSystemInstructions = "";
let agentTaskInstructions = "";
let agentTemperature = 0.6;
let agentTimeoutSeconds = 300;
let agentRetryCount = 1;
let agentMaxSteps = 32;
let autoCompact = false;
let piAgentStatus: AgentPiStatus | null = null;
let piAgentDiscovery: Promise<AgentPiStatus | null> | null = null;
let contextSummary = "";
let mcpServers: AgentMcpServer[] = [];
let todos: AgentTodo[] = [];
let skills: AgentSkill[] = [];
let discoveredSkills: AgentSkillFile[] = [];
let subagents: AgentSubagent[] = [];
let taskHistory: AgentTaskSession[] = [];
let currentTaskId = "";
let currentAgentRequest = "";
let agentRunActive = false;
let agentAbort: AbortController | null = null;
let agentThinkingRow: HTMLElement | null = null;
let agentProfileStore: AgentProfileStore;
let agentProfileDirty = false;
let profileEventsSuspended = false;
let agentProfileBusy = false;
let agentProfileNeedsWorkspaceMigration = false;

// Constrains the agent's reply (via llama.cpp json_schema/grammar) to a single
// valid action object, so weak models can't reply with prose or invalid JSON.
const agentActionJsonSchema = {
  type: "object",
  properties: {
    reasoning: { type: "string" },
    tool: { type: "string" },
    args: { type: "object" },
    final: { type: "string" },
  },
  anyOf: [{ required: ["tool"] }, { required: ["final"] }],
};
const runIconSvg = '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="m6 4 14 8-14 8V4Z" /></svg>';
const stopIconSvg = '<svg viewBox="0 0 24 24" aria-hidden="true"><rect x="6" y="6" width="12" height="12" rx="2" /></svg>';

const codingAgentProfiles: CodingAgentProfile[] = [
  {
    id: "architect",
    name: "Architect",
    description: "Plans changes, maps files, and avoids edits until the approach is clear.",
    instructions:
      "You are the Architect agent. Inspect the project, identify relevant files, produce a careful plan, and only edit when the user explicitly asks implementation or the next safe step is obvious.",
    allowedTools: [
      "read_file",
      "list_files",
      "search_files",
      "find_files",
      "go_to_line",
      "search_lines",
      "list_code_definition_names",
      "semantic_search",
      "update_todo_list",
      "todo_write",
      "switch_mode",
      "ask_followup_question",
      "attempt_completion",
      "compact_context",
      "local_chat",
      "delegate_agent",
    ],
  },
  {
    id: "ask",
    name: "Ask",
    description: "Answers questions and explains code without changing the workspace.",
    instructions:
      "You are the Ask agent. Answer the user's questions about the code and project, explain concepts, and investigate using read-only tools. Never edit files or run commands that modify the workspace. Finish with a clear answer.",
    allowedTools: [
      "read_file",
      "list_files",
      "search_files",
      "find_files",
      "go_to_line",
      "search_lines",
      "list_code_definition_names",
      "semantic_search",
      "ask_followup_question",
      "attempt_completion",
      "compact_context",
      "local_chat",
    ],
  },
  {
    id: "coder",
    name: "Coder",
    description: "Implements scoped code changes and verifies them.",
    instructions:
      "You are the Coder agent. Make focused edits, prefer existing project patterns, and verify with builds or tests when possible.",
    allowedTools: [
      "read_file",
      "list_files",
      "search_files",
      "list_code_definition_names",
      "semantic_search",
      "edit_file",
      "write_to_file",
      "append_to_file",
      "insert_content",
      "edit_lines",
      "replace_in_file",
      "apply_diff",
      "delete_file",
      "shell",
      "execute_command",
      "browser_action",
      "use_mcp_tool",
      "access_mcp_resource",
      "todo_write",
      "update_todo_list",
      "switch_mode",
      "ask_followup_question",
      "attempt_completion",
      "compact_context",
      "local_chat",
      "delegate_agent",
    ],
  },
  {
    id: "debugger",
    name: "Debugger",
    description: "Reproduces failures, inspects logs, patches narrowly, and retests.",
    instructions:
      "You are the Debugger agent. Reproduce the issue, gather evidence, make the smallest credible fix, and retest the same failure path.",
    allowedTools: [
      "read_file",
      "list_files",
      "search_files",
      "find_files",
      "go_to_line",
      "search_lines",
      "list_code_definition_names",
      "semantic_search",
      "edit_lines",
      "replace_in_file",
      "apply_diff",
      "shell",
      "execute_command",
      "browser_action",
      "todo_write",
      "update_todo_list",
      "ask_followup_question",
      "attempt_completion",
      "compact_context",
      "local_chat",
    ],
  },
  {
    id: "reviewer",
    name: "Reviewer",
    description: "Reviews code for bugs, regressions, and missing tests without editing by default.",
    instructions:
      "You are the Reviewer agent. Prioritize concrete bugs, risks, regressions, and missing tests. Do not edit files unless asked to fix the findings.",
    allowedTools: [
      "read_file",
      "list_files",
      "search_files",
      "list_code_definition_names",
      "semantic_search",
      "shell",
      "execute_command",
      "todo_write",
      "update_todo_list",
      "ask_followup_question",
      "attempt_completion",
      "compact_context",
      "local_chat",
    ],
  },
  {
    id: "tester",
    name: "Tester",
    description: "Finds and runs the right tests, then reports failures clearly.",
    instructions:
      "You are the Tester agent. Discover test commands, run focused verification, summarize failures, and suggest minimal test additions.",
    allowedTools: [
      "read_file",
      "list_files",
      "search_files",
      "list_code_definition_names",
      "semantic_search",
      "shell",
      "execute_command",
      "browser_action",
      "todo_write",
      "update_todo_list",
      "ask_followup_question",
      "attempt_completion",
      "compact_context",
      "local_chat",
    ],
  },
  {
    id: "orchestrator",
    name: "Orchestrator",
    description: "Breaks work into tasks and delegates to specialist coding agents.",
    instructions:
      "You are the Orchestrator agent. Break complex goals into smaller tasks, delegate to specialist agents, maintain the todo list, and coordinate implementation safely.",
    allowedTools: [
      "read_file",
      "list_files",
      "search_files",
      "list_code_definition_names",
      "semantic_search",
      "shell",
      "execute_command",
      "todo_write",
      "update_todo_list",
      "switch_mode",
      "new_task",
      "ask_followup_question",
      "attempt_completion",
      "skill_load",
      "subagent_start",
      "compact_context",
      "local_chat",
      "delegate_agent",
    ],
  },
];

// Web read tools are safe for every profile; downloading writes to the
// workspace, so it is limited to profiles that are allowed to make changes.
for (const profile of codingAgentProfiles) {
  for (const toolName of ["find_files", "go_to_line", "search_lines"]) {
    if (!profile.allowedTools.includes(toolName)) profile.allowedTools.push(toolName);
  }
  for (const toolName of ["browse_url", "web_search", "web_fetch", "browser_action"]) {
    if (!profile.allowedTools.includes(toolName)) profile.allowedTools.push(toolName);
  }
  if (["coder", "debugger", "orchestrator"].includes(profile.id) && !profile.allowedTools.includes("download_file")) {
    profile.allowedTools.push("download_file");
  }
}

const localTools: AgentToolDefinition[] = [
  {
    name: "read_file",
    description: "Read a text file or list a directory.",
    permission: "read",
    run: async (args) => readAgentPath(stringArg(args, "path") || "."),
  },
  {
    name: "list_files",
    description: "List files in a directory.",
    permission: "read",
    run: async (args) => readAgentPath(stringArg(args, "path") || "."),
  },
  {
    name: "search_files",
    description: "Search file contents with ripgrep.",
    permission: "read",
    run: async (args) =>
      searchAgentFiles(
        stringArg(args, "query"),
        stringArg(args, "path") || ".",
        stringArg(args, "glob"),
      ),
  },
  {
    name: "find_files",
    description: "Find files by name under a directory, with optional glob filtering.",
    permission: "read",
    run: async (args) =>
      findAgentFiles(
        stringArg(args, "path") || ".",
        stringArg(args, "query") || stringArg(args, "name") || stringArg(args, "pattern"),
        stringArg(args, "glob"),
      ),
  },
  {
    name: "go_to_line",
    description: "Read a file around a 1-based line number, with line numbers.",
    permission: "read",
    run: async (args) =>
      readAgentLineRange(
        stringArg(args, "path"),
        numberArg(args, "line") ?? numberArg(args, "start"),
        numberArg(args, "end"),
        numberArg(args, "context"),
      ),
  },
  {
    name: "search_lines",
    description: "Search within one file and return matching line numbers.",
    permission: "read",
    run: async (args) =>
      searchAgentLines(
        stringArg(args, "path"),
        stringArg(args, "query") || stringArg(args, "pattern"),
        Boolean(args.regex),
      ),
  },
  {
    name: "list_code_definition_names",
    description: "List top-level code definitions with file paths and line numbers.",
    permission: "read",
    run: async (args) => listCodeDefinitionNames(stringArg(args, "path") || "."),
  },
  {
    name: "semantic_search",
    description: "Find semantically relevant code. Uses local lexical search until an embedding index is configured.",
    permission: "read",
    run: async (args) =>
      semanticSearchAgentFiles(
        stringArg(args, "query"),
        stringArg(args, "path") || stringArg(args, "directory") || ".",
      ),
  },
  {
    name: "edit_file",
    description: "Create or replace a text file.",
    permission: "edit",
    run: async (args) => editAgentFile(stringArg(args, "path"), contentArg(args)),
  },
  {
    name: "write_to_file",
    description: "Create or replace a text file with the complete provided contents.",
    permission: "edit",
    run: async (args) =>
      writeAgentFile(
        stringArg(args, "path"),
        contentArg(args),
      ),
  },
  {
    name: "append_to_file",
    description: "Append exact text to the end of an existing or new text file.",
    permission: "edit",
    run: async (args) =>
      appendContentToAgentFile(
        stringArg(args, "path"),
        contentArg(args),
      ),
  },
  {
    name: "insert_content",
    description: "Insert text into a file at a 1-based line number, or append when no line is provided.",
    permission: "edit",
    run: async (args) =>
      insertContentInAgentFile(
        stringArg(args, "path"),
        contentArg(args),
        numberArg(args, "line"),
      ),
  },
  {
    name: "edit_lines",
    description: "Replace a 1-based inclusive line range with exact text.",
    permission: "edit",
    run: async (args) =>
      editAgentLines(
        stringArg(args, "path"),
        numberArg(args, "start") ?? numberArg(args, "line"),
        numberArg(args, "end"),
        contentArg(args),
      ),
  },
  {
    name: "replace_in_file",
    description: "Replace text inside a file.",
    permission: "edit",
    run: async (args) =>
      replaceInAgentFile(
        stringArg(args, "path"),
        stringArg(args, "search"),
        stringArg(args, "replace"),
        Boolean(args.all),
      ),
  },
  {
    name: "apply_diff",
    description: "Apply a unified diff to the workspace with git apply.",
    permission: "edit",
    run: async (args) => applyAgentDiff(stringArg(args, "diff")),
  },
  {
    name: "delete_file",
    description: "Delete a file or directory from the workspace.",
    permission: "edit",
    run: async (args) => deleteAgentPath(stringArg(args, "path")),
  },
  {
    name: "copy_file",
    description: "Copy a file to another path.",
    permission: "copy",
    run: async (args) => copyAgentFile(stringArg(args, "fromPath"), stringArg(args, "toPath")),
  },
  {
    name: "move_file",
    description: "Move or rename a file or directory.",
    permission: "move",
    run: async (args) => moveAgentFile(stringArg(args, "fromPath"), stringArg(args, "toPath")),
  },
  {
    name: "paste_file",
    description: "Paste clipboard text into a file.",
    permission: "paste",
    run: async (args) => pasteAgentFile(stringArg(args, "path")),
  },
  {
    name: "browse_url",
    description: "Open and preview a URL.",
    permission: "browse",
    run: async (args) => browseAgentUrl(stringArg(args, "url")),
  },
  {
    name: "browser_action",
    description: "browser action. Supports local open/navigate/search/close actions.",
    permission: "browse",
    run: async (args) => browserAction(args),
  },
  {
    name: "web_search",
    description:
      "Search the web without an API key. Uses the OMP-style public provider chain: DuckDuckGo, Ecosia, Google, then Mojeek.",
    permission: "browse",
    run: async (args) => webSearchKeyless(stringArg(args, "query"), stringArg(args, "provider") || "auto"),
  },
  {
    name: "web_fetch",
    description: "Download the text contents of a URL over HTTP (read-only).",
    permission: "browse",
    run: async (args) => webFetchUrl(stringArg(args, "url")),
  },
  {
    name: "download_file",
    description: "Download a URL to a file in the workspace.",
    permission: "edit",
    run: async (args) =>
      downloadFileTool(
        stringArg(args, "url"),
        stringArg(args, "path") || stringArg(args, "toPath") || stringArg(args, "output"),
      ),
  },
  {
    name: "shell",
    description: "Run a shell command in the workspace.",
    permission: "shell",
    run: async (args) => runAgentShell(stringArg(args, "command")),
  },
  {
    name: "execute_command",
    description: "alias for running a workspace shell command.",
    permission: "shell",
    run: async (args) => runAgentShell(stringArg(args, "command")),
  },
  {
    name: "todo_write",
    description: "Add a task to the agent todo list.",
    permission: "todo",
    run: async (args) => addTodo(stringArg(args, "text")),
  },
  {
    name: "update_todo_list",
    description: "Replace the full task todo list.",
    permission: "todo",
    run: async (args) => updateTodoList(args),
  },
  {
    name: "skill_load",
    description: "Enable or create a named skill instruction.",
    permission: "skill",
    run: async (args) => loadSkill(stringArg(args, "name"), stringArg(args, "instructions")),
  },
  {
    name: "subagent_start",
    description: "Start a named local subagent role.",
    permission: "subagent",
    run: async (args) => startSubagent(stringArg(args, "name"), stringArg(args, "role")),
  },
  {
    name: "compact_context",
    description: "Compact the visible agent transcript into a summary.",
    permission: "todo",
    run: async () => compactAgentContext(),
  },
  {
    name: "mcp_call",
    description: "Call a registered local MCP HTTP endpoint.",
    permission: "browse",
    run: async (args) =>
      callLocalMcpServer(
        stringArg(args, "server"),
        stringArg(args, "method") || "tools/list",
        objectArg(args, "params"),
      ),
  },
  {
    name: "use_mcp_tool",
    description: "MCP tool call through a registered local MCP HTTP endpoint.",
    permission: "browse",
    run: async (args) =>
      callLocalMcpServer(
        stringArg(args, "server_name") || stringArg(args, "server"),
        "tools/call",
        {
          name: stringArg(args, "tool_name") || stringArg(args, "tool"),
          arguments: objectArg(args, "arguments"),
        },
      ),
  },
  {
    name: "access_mcp_resource",
    description: "MCP resource read through a registered local MCP HTTP endpoint.",
    permission: "browse",
    run: async (args) =>
      callLocalMcpServer(
        stringArg(args, "server_name") || stringArg(args, "server"),
        "resources/read",
        { uri: stringArg(args, "uri") },
      ),
  },
  {
    name: "switch_mode",
    description: "Switch to a different coding agent mode.",
    permission: "todo",
    run: async (args) => switchAgentModeTool(stringArg(args, "mode")),
  },
  {
    name: "new_task",
    description: "Create a new task or delegate a subtask to a specialist agent.",
    permission: "subagent",
    run: async (args) =>
      createNewTaskTool(
        stringArg(args, "mode") || stringArg(args, "agent"),
        stringArg(args, "message") || stringArg(args, "task") || stringArg(args, "prompt"),
      ),
  },
  {
    name: "ask_followup_question",
    description: "Ask the user a clarifying question and optionally show suggestions.",
    permission: "todo",
    run: async (args) =>
      askFollowupQuestion(
        stringArg(args, "question") || stringArg(args, "text"),
        args.follow_up ?? args.suggestions,
      ),
  },
  {
    name: "attempt_completion",
    description: "Mark the current task attempt complete and present the result.",
    permission: "todo",
    run: async (args) =>
      attemptCompletion(
        stringArg(args, "result") || stringArg(args, "message") || stringArg(args, "summary"),
        stringArg(args, "command"),
      ),
  },
  {
    name: "local_chat",
    description: "Send a prompt to the running local llama-server.",
    permission: "browse",
    run: async (args) => askLocalServer(stringArg(args, "prompt")),
  },
  {
    name: "delegate_agent",
    description: "Ask one built-in coding agent for focused analysis.",
    permission: "subagent",
    run: async (args) =>
      delegateToCodingAgent(
        stringArg(args, "agent") || stringArg(args, "name"),
        stringArg(args, "prompt") || stringArg(args, "task"),
      ),
  },
];

// Read the current session todo list (todoread).
localTools.push({
  name: "todoread",
  description: "Read the current session todo list.",
  permission: "todo",
  run: async () => {
    const text = todos.length
      ? todos.map((todo) => `${todo.done ? "[x]" : "[ ]"} ${todo.text}`).join("\n")
      : "Todo list is empty.";
    appendAgentMessage("tool", text);
    return text;
  },
});

localTools.push({
  name: "edit",
  description: "Pi-compatible exact text edit tool. Accepts {path, edits:[{oldText,newText}]} or {path, search, replace}.",
  permission: "edit",
  run: async (args) => piEditTool(args),
});

localTools.push({
  name: "pi_cli",
  description: "Run the behind-the-scenes Pi coding agent CLI command in the workspace.",
  permission: "shell",
  run: async (args) => piCliTool(stringArg(args, "command") || stringArg(args, "args") || "status"),
});

localTools.push({
  name: "pi_agent",
  description: "Run the real earendil-works/pi coding agent behind the scenes for a coding task.",
  permission: "shell",
  run: async (args) => runPiCodingAgent(stringArg(args, "prompt") || stringArg(args, "task") || stringArg(args, "message")),
});

// Register alternate tool names as thin aliases that reuse the
// existing local implementations, so alternate tool-call names resolve.
const compatToolAliases: Record<string, string> = {
  read: "read_file",
  ls: "list_files",
  find: "find_files",
  glob: "find_files",
  grep: "search_files",
  write: "write_to_file",
  bash: "shell",
  apply_patch: "apply_diff",
  webfetch: "web_fetch",
  websearch: "web_search",
  question: "ask_followup_question",
  task: "new_task",
  todowrite: "update_todo_list",
  skill: "skill_load",
};
for (const [alias, target] of Object.entries(compatToolAliases)) {
  const base = localTools.find((tool) => tool.name === target);
  if (base && !localTools.some((tool) => tool.name === alias)) {
    localTools.push({ ...base, name: alias, description: `Alias for ${target}. ${base.description}` });
  }
}

function agentMessageContainer() {
  return $("agent-chat-messages");
}

function setAgentStatus(message: string) {
  $("agent-status").textContent = message;
}

function defaultAgentCliStatus() {
  if (!piAgentStatus) return "CLI: checking pi";
  return piAgentStatus.available ? `CLI: ${piAgentStatus.command}` : "CLI: local fallback";
}

function setAgentCliStatus(message = defaultAgentCliStatus()) {
  const target = document.getElementById("agent-cli-status");
  if (target) target.textContent = message;
}

function setAgentOutput(message: string) {
  $("agent-tool-output").textContent = message;
}

function stringArg(args: ToolArgs, name: string): string {
  const value = args[name];
  return typeof value === "string" ? value.trim() : "";
}

function contentArg(args: ToolArgs): string {
  if (typeof args.content === "string") return args.content;
  if (typeof args.text === "string") return args.text;
  return "";
}

function objectArg(args: ToolArgs, name: string): ToolArgs {
  const value = args[name];
  return value && typeof value === "object" && !Array.isArray(value) ? (value as ToolArgs) : {};
}

function numberArg(args: ToolArgs, name: string): number | null {
  const value = args[name];
  if (typeof value === "number" && Number.isFinite(value)) return value;
  if (typeof value === "string" && value.trim()) {
    const parsed = Number(value.trim());
    return Number.isFinite(parsed) ? parsed : null;
  }
  return null;
}

function toolAliasCanonical(toolName: string): string {
  const aliases: Record<string, string> = {
    write_to_file: "edit_file",
    execute_command: "shell",
    update_todo_list: "todo_write",
    use_mcp_tool: "mcp_call",
    access_mcp_resource: "mcp_call",
    // Alternate tool names mapped onto the local equivalents.
    read: "read_file",
    ls: "list_files",
    find: "find_files",
    glob: "find_files",
    grep: "search_files",
    edit: "replace_in_file",
    write: "write_to_file",
    bash: "shell",
    apply_patch: "apply_diff",
    webfetch: "web_fetch",
    websearch: "web_search",
    question: "ask_followup_question",
    task: "new_task",
    todowrite: "update_todo_list",
    todoread: "todo_write",
    skill: "skill_load",
    pi_cli: "shell",
    pi_agent: "shell",
  };
  return aliases[toolName] ?? toolName;
}

function normalizeThinkingLevel(value: string | null | undefined): AgentThinkingLevel {
  return value === "low" || value === "medium" || value === "high" || value === "xhigh" || value === "disabled"
    ? value
    : "medium";
}

function activeSavedProfile(): AgentProfile {
  return (
    agentProfileStore.profiles.find((profile) => profile.id === agentProfileStore.activeProfileId) ??
    agentProfileStore.profiles[0]
  );
}

function uniqueProfileName(stem: string): string {
  const names = new Set(agentProfileStore.profiles.map((profile) => profile.name.toLocaleLowerCase()));
  let name = stem;
  let suffix = 2;
  while (names.has(name.toLocaleLowerCase())) {
    name = `${stem} ${suffix}`;
    suffix += 1;
  }
  return name;
}

function readBoundedAgentNumber(id: string, fallback: number, min: number, max: number, integer = false): number {
  const parsed = Number(($(`${id}`) as HTMLInputElement).value);
  if (!Number.isFinite(parsed)) return fallback;
  const bounded = Math.min(max, Math.max(min, parsed));
  return integer ? Math.round(bounded) : bounded;
}

function setAgentProfileFeedback(message: string, tone: "info" | "ok" | "warn" | "error" = "info") {
  const feedback = $("agent-profile-feedback");
  feedback.textContent = message;
  feedback.dataset.tone = tone;
}

function setAgentProfileDirty(dirty: boolean, message = dirty ? "Unsaved changes" : "Saved") {
  agentProfileDirty = dirty;
  const state = $("agent-profile-save-state");
  state.textContent = message;
  state.dataset.dirty = String(dirty);
}

async function runAgentProfileOperation(operation: () => Promise<void>): Promise<void> {
  if (agentProfileBusy) return;
  agentProfileBusy = true;
  const controls = document.querySelectorAll<HTMLButtonElement | HTMLSelectElement>(
    "#agent-saved-profile-select, .agent-profile-actions button",
  );
  controls.forEach((control) => {
    control.disabled = true;
  });
  try {
    await operation();
  } finally {
    controls.forEach((control) => {
      control.disabled = false;
    });
    agentProfileBusy = false;
  }
}

function persistAgentProfileStore(): void {
  const next = JSON.stringify(agentProfileStore);
  const previous = window.localStorage.getItem(agentProfilesStorageKey);
  try {
    if (previous) window.localStorage.setItem(agentProfilesBackupStorageKey, previous);
    window.localStorage.setItem(agentProfilesStorageKey, next);
    if (window.localStorage.getItem(agentProfilesStorageKey) !== next) {
      throw new Error("Profile storage verification failed");
    }
  } catch (error) {
    if (previous) window.localStorage.setItem(agentProfilesStorageKey, previous);
    throw new Error(`Unable to save profiles safely: ${String(error)}`);
  }
}

function renderSavedProfileSelect() {
  const select = $("agent-saved-profile-select") as HTMLSelectElement;
  select.replaceChildren();
  for (const profile of agentProfileStore.profiles) {
    const option = document.createElement("option");
    option.value = profile.id;
    const suffix = profile.id === agentProfileStore.defaultProfileId ? " · default" : "";
    option.textContent = `${profile.name}${suffix}`;
    select.append(option);
  }
  select.value = agentProfileStore.activeProfileId;
}

function captureAgentProfile(): AgentProfile {
  const current = activeSavedProfile();
  return createAgentProfile(($("agent-profile-name") as HTMLInputElement).value, {
    ...current,
    id: current.id,
    role: activeAgentProfile,
    systemInstructions: ($("agent-system-instructions") as HTMLTextAreaElement).value,
    taskInstructions: ($("agent-task-instructions") as HTMLTextAreaElement).value,
    goal: ($("agent-goal") as HTMLTextAreaElement).value.trim(),
    workspaceRoot: ($("agent-workspace-root-input") as HTMLInputElement).value.trim(),
    thinkingLevel: normalizeThinkingLevel(($("agent-thinking-level") as HTMLSelectElement).value),
    executionMode: agentExecutionMode,
    permissions: { ...agentPermissions },
    autoApprove: autoAccept,
    yoloMode,
    autoCompact,
    temperature: readBoundedAgentNumber("agent-temperature", agentTemperature, 0, 2),
    timeoutSeconds: readBoundedAgentNumber("agent-timeout-seconds", agentTimeoutSeconds, 10, 3600, true),
    retryCount: readBoundedAgentNumber("agent-retry-count", agentRetryCount, 0, 5, true),
    maxSteps: readBoundedAgentNumber("agent-max-steps", agentMaxSteps, 1, 64, true),
    createdAt: current.createdAt,
    updatedAt: Date.now(),
  });
}

function persistLegacyAgentSettings() {
  saveAgentPermissions();
  saveAgentApprovalMode();
  saveAgentGoal();
  saveAgentMode();
  saveActiveAgentProfile();
  saveAgentThinkingLevel();
  saveExecutionMode();
  saveAgentContext();
}

async function persistAgentWorkspaceRoot(path: string, announce: boolean): Promise<void> {
  const nextRoot = path.trim();
  if (!nextRoot) return;
  const normalizedRoot = await invoke<string>("agent_validate_workspace", { path: nextRoot });
  const nextConfig = await invoke<AppConfig>("load_config");
  if (nextConfig.agentWorkspaceRoot === normalizedRoot) return;
  nextConfig.agentWorkspaceRoot = normalizedRoot;
  ($("agent-workspace-root-input") as HTMLInputElement).value = normalizedRoot;
  await invoke<AppConfig>("save_config", { config: nextConfig });
  await refreshAgentWorkspace();
  await refreshDiscoveredSkills();
  if (announce) appendAgentMessage("agent", `Workspace set to:\n${normalizedRoot}`);
}

async function applySavedAgentProfile(profile: AgentProfile, persistWorkspace = true): Promise<void> {
  profileEventsSuspended = true;
  let workspaceWarning = false;
  try {
    agentProfileStore.activeProfileId = profile.id;
    activeAgentProfile = profile.role;
    agentMode = profile.role;
    agentSystemInstructions = profile.systemInstructions;
    agentTaskInstructions = profile.taskInstructions;
    agentGoal = profile.goal;
    agentThinkingLevel = profile.thinkingLevel;
    agentExecutionMode = profile.executionMode;
    agentPermissions = { ...profile.permissions };
    autoAccept = profile.autoApprove;
    yoloMode = profile.yoloMode || profile.autoApprove;
    autoCompact = profile.autoCompact;
    agentTemperature = profile.temperature;
    agentTimeoutSeconds = profile.timeoutSeconds;
    agentRetryCount = profile.retryCount;
    agentMaxSteps = profile.maxSteps;

    renderSavedProfileSelect();
    ($("agent-profile-name") as HTMLInputElement).value = profile.name;
    ($("agent-system-instructions") as HTMLTextAreaElement).value = profile.systemInstructions;
    ($("agent-task-instructions") as HTMLTextAreaElement).value = profile.taskInstructions;
    ($("agent-workspace-root-input") as HTMLInputElement).value = profile.workspaceRoot;
    ($("agent-temperature") as HTMLInputElement).value = String(profile.temperature);
    ($("agent-timeout-seconds") as HTMLInputElement).value = String(profile.timeoutSeconds);
    ($("agent-retry-count") as HTMLInputElement).value = String(profile.retryCount);
    ($("agent-max-steps") as HTMLInputElement).value = String(profile.maxSteps);

    syncPermissionControls();
    syncExecutionModeUi();
    syncAgentProfileUi();
    syncAgentThinkingUi();
    syncGoalUi();
    syncModeUi();
    syncContextUi();
    persistLegacyAgentSettings();
    persistAgentProfileStore();
    if (persistWorkspace && profile.workspaceRoot) {
      try {
        await persistAgentWorkspaceRoot(profile.workspaceRoot, false);
      } catch (error) {
        workspaceWarning = true;
        setAgentProfileFeedback(
          `${profile.name} loaded, but its workspace is unavailable. ${String(error)}`,
          "warn",
        );
        setAgentStatus("Profile loaded · workspace needs attention");
      }
    }
    setAgentProfileDirty(false);
    if (!workspaceWarning) {
      setAgentProfileFeedback(
        `${profile.name} loaded${profile.id === agentProfileStore.defaultProfileId ? " · default profile" : ""}.`,
        "ok",
      );
    }
  } finally {
    profileEventsSuspended = false;
  }
}

async function confirmDiscardAgentProfileChanges(): Promise<boolean> {
  if (!agentProfileDirty) return true;
  return confirmAction({
    title: "Discard unsaved profile changes?",
    message: "The current profile has changes that have not been saved.",
    okLabel: "Discard changes",
    cancelLabel: "Keep editing",
    kind: "warning",
  });
}

async function switchSavedAgentProfile(profileId: string): Promise<void> {
  const previousId = agentProfileStore.activeProfileId;
  if (profileId === previousId) return;
  if (!(await confirmDiscardAgentProfileChanges())) {
    ($("agent-saved-profile-select") as HTMLSelectElement).value = previousId;
    return;
  }
  const next = agentProfileStore.profiles.find((profile) => profile.id === profileId);
  if (!next) throw new Error("The selected profile no longer exists");
  await applySavedAgentProfile(next);
}

async function saveCurrentAgentProfile(): Promise<void> {
  const next = captureAgentProfile();
  const duplicate = agentProfileStore.profiles.find(
    (profile) => profile.id !== next.id && profile.name.toLocaleLowerCase() === next.name.toLocaleLowerCase(),
  );
  const errors = validateAgentProfile(next);
  if (duplicate) errors.push(`Another profile is already named "${next.name}".`);
  if (errors.length) {
    setAgentProfileFeedback(errors.join(" "), "error");
    throw new Error(errors.join(" "));
  }
  if (next.workspaceRoot) {
    next.workspaceRoot = await invoke<string>("agent_validate_workspace", { path: next.workspaceRoot });
    ($("agent-workspace-root-input") as HTMLInputElement).value = next.workspaceRoot;
  }
  agentProfileStore.profiles = agentProfileStore.profiles.map((profile) => (profile.id === next.id ? next : profile));
  agentProfileStore.activeProfileId = next.id;
  agentSystemInstructions = next.systemInstructions;
  agentTaskInstructions = next.taskInstructions;
  agentGoal = next.goal;
  agentTemperature = next.temperature;
  agentTimeoutSeconds = next.timeoutSeconds;
  agentRetryCount = next.retryCount;
  agentMaxSteps = next.maxSteps;
  persistLegacyAgentSettings();
  persistAgentProfileStore();
  if (next.workspaceRoot) await persistAgentWorkspaceRoot(next.workspaceRoot, false);
  renderSavedProfileSelect();
  setAgentProfileDirty(false);
  setAgentProfileFeedback(`${next.name} saved safely.`, "ok");
  setAgentStatus(`Profile saved: ${next.name}`);
}

async function createNewAgentProfile(): Promise<void> {
  if (!(await confirmDiscardAgentProfileChanges())) return;
  const profile = createAgentProfile(uniqueProfileName("New Profile"), {
    role: "coder",
    thinkingLevel: "medium",
    workspaceRoot: ($("agent-workspace-root-input") as HTMLInputElement).value.trim(),
  });
  agentProfileStore.profiles.push(profile);
  agentProfileStore.activeProfileId = profile.id;
  persistAgentProfileStore();
  await applySavedAgentProfile(profile, false);
  ($("agent-profile-name") as HTMLInputElement).focus();
  ($("agent-profile-name") as HTMLInputElement).select();
  setAgentProfileFeedback("New profile created. Customize it, then press Save.", "ok");
}

async function duplicateCurrentAgentProfile(): Promise<void> {
  if (!(await confirmDiscardAgentProfileChanges())) return;
  const sourceName = activeSavedProfile().name;
  const duplicate = duplicateAgentProfile(
    activeSavedProfile(),
    agentProfileStore.profiles.map((profile) => profile.name),
  );
  agentProfileStore.profiles.push(duplicate);
  agentProfileStore.activeProfileId = duplicate.id;
  persistAgentProfileStore();
  await applySavedAgentProfile(duplicate);
  setAgentProfileFeedback(`${duplicate.name} created from ${sourceName}.`, "ok");
}

function renameCurrentAgentProfile() {
  const nextName = ($("agent-profile-name") as HTMLInputElement).value.trim();
  if (!nextName) throw new Error("Enter a profile name before renaming");
  if (nextName.length > 80) throw new Error("Profile names must be 80 characters or fewer");
  if (agentProfileStore.profiles.some(
    (profile) => profile.id !== agentProfileStore.activeProfileId &&
      profile.name.toLocaleLowerCase() === nextName.toLocaleLowerCase(),
  )) {
    throw new Error(`Another profile is already named "${nextName}"`);
  }
  const current = activeSavedProfile();
  current.name = nextName;
  current.updatedAt = Date.now();
  persistAgentProfileStore();
  renderSavedProfileSelect();
  setAgentProfileDirty(false);
  setAgentProfileFeedback(`Profile renamed to ${nextName}.`, "ok");
}

async function deleteCurrentAgentProfile(): Promise<void> {
  if (agentProfileStore.profiles.length === 1) throw new Error("At least one agent profile must remain");
  const current = activeSavedProfile();
  const confirmed = await confirmAction({
    title: `Delete “${current.name}”?`,
    message: "This removes the saved profile. Export it first if you may need it later.",
    okLabel: "Delete profile",
    cancelLabel: "Cancel",
    kind: "danger",
  });
  if (!confirmed) return;
  agentProfileStore.profiles = agentProfileStore.profiles.filter((profile) => profile.id !== current.id);
  if (agentProfileStore.defaultProfileId === current.id) {
    agentProfileStore.defaultProfileId = agentProfileStore.profiles[0].id;
  }
  const next = agentProfileStore.profiles.find((profile) => profile.id === agentProfileStore.defaultProfileId)
    ?? agentProfileStore.profiles[0];
  agentProfileStore.activeProfileId = next.id;
  persistAgentProfileStore();
  await applySavedAgentProfile(next);
  setAgentProfileFeedback(`${current.name} deleted.`, "ok");
}

function setCurrentAgentProfileDefault() {
  agentProfileStore.defaultProfileId = activeSavedProfile().id;
  persistAgentProfileStore();
  renderSavedProfileSelect();
  setAgentProfileFeedback(`${activeSavedProfile().name} is now the default profile.`, "ok");
}

async function resetCurrentAgentProfile(): Promise<void> {
  const current = activeSavedProfile();
  const confirmed = await confirmAction({
    title: `Reset “${current.name}”?`,
    message: "This restores safe defaults in the editor. Press Save to keep the reset.",
    okLabel: "Reset profile",
    cancelLabel: "Cancel",
    kind: "warning",
  });
  if (!confirmed) return;
  const reset = createAgentProfile(current.name, {
    id: current.id,
    createdAt: current.createdAt,
    workspaceRoot: "",
  });
  agentProfileStore.profiles = agentProfileStore.profiles.map((profile) => profile.id === current.id ? reset : profile);
  await applySavedAgentProfile(reset, false);
  setAgentProfileDirty(true, "Reset not saved");
  setAgentProfileFeedback("Defaults restored in the editor. Press Save to keep them.", "warn");
}

function exportCurrentAgentProfile() {
  const profile = activeSavedProfile();
  const blob = new Blob([exportAgentProfiles(agentProfileStore, profile.id)], { type: "application/json" });
  const url = URL.createObjectURL(blob);
  const anchor = document.createElement("a");
  anchor.href = url;
  anchor.download = `${profile.name.replace(/[^a-z0-9._-]+/gi, "-").replace(/^-|-$/g, "") || "agent-profile"}.json`;
  anchor.click();
  window.setTimeout(() => URL.revokeObjectURL(url), 0);
  setAgentProfileFeedback(`${profile.name} exported.`, "ok");
}

async function importAgentProfileFile(file: File): Promise<void> {
  const imported = normalizeAgentProfileStore(JSON.parse(await file.text()));
  const existingIds = new Set(agentProfileStore.profiles.map((profile) => profile.id));
  const existingNames = agentProfileStore.profiles.map((profile) => profile.name);
  const added = imported.profiles.map((profile) => {
    const duplicateId = existingIds.has(profile.id);
    const duplicateName = existingNames.some((name) => name.toLocaleLowerCase() === profile.name.toLocaleLowerCase());
    const next = createAgentProfile(
      duplicateName ? uniqueProfileName(`${profile.name} imported`) : profile.name,
      { ...profile, id: duplicateId ? crypto.randomUUID() : profile.id },
    );
    existingIds.add(next.id);
    existingNames.push(next.name);
    return next;
  });
  agentProfileStore.profiles.push(...added);
  agentProfileStore.activeProfileId = added[0].id;
  persistAgentProfileStore();
  await applySavedAgentProfile(added[0]);
  setAgentProfileFeedback(`${added.length} profile${added.length === 1 ? "" : "s"} imported.`, "ok");
}

function agentThinkingBudget(level = agentThinkingLevel): number {
  switch (level) {
    case "low":
      return 512;
    case "medium":
      return 2048;
    case "high":
      return 4096;
    case "xhigh":
      return -1;
    case "disabled":
    default:
      return 0;
  }
}

function agentThinkingEffort(level = agentThinkingLevel): "low" | "medium" | "high" {
  if (level === "low" || level === "disabled") return "low";
  if (level === "medium") return "medium";
  return "high";
}

function agentThinkingKwargs(): Record<string, unknown> {
  const budget = agentThinkingBudget();
  const enabled = agentThinkingLevel !== "disabled";
  return {
    enable_thinking: enabled,
    preserve_thinking: false,
    reasoning_effort: agentThinkingEffort(),
    reasoning_budget: budget,
    thinking_budget: budget,
  };
}

function agentThinkingRequestFields(): Record<string, unknown> {
  const budget = agentThinkingBudget();
  return {
    chat_template_kwargs: agentThinkingKwargs(),
    reasoning_budget: budget,
    reasoning: agentThinkingLevel === "disabled" ? "off" : "auto",
  };
}

function agentThinkingInstruction(): string {
  switch (agentThinkingLevel) {
    case "disabled":
      return "Agent thinking effort: disabled. Do not emit chain-of-thought, thought tags, analysis, channel text, reasoning, or rationale fields. Return only a tool call or final answer.";
    case "low":
      return "Agent thinking effort: low. Keep internal reasoning minimal and return a short public reasoning field only when useful.";
    case "medium":
      return "Agent thinking effort: medium. Use concise reasoning before tools, then act.";
    case "high":
      return "Agent thinking effort: high. Spend more reasoning on planning and validation, but keep the public reasoning summary short.";
    case "xhigh":
      return "Agent thinking effort: xHigh. Use extended reasoning internally for complex tasks, but do not dump chain-of-thought; return concise public reasoning only.";
  }
}

function syncAgentThinkingUi() {
  const select = document.getElementById("agent-thinking-level") as HTMLSelectElement | null;
  if (select) select.value = agentThinkingLevel;
  const description = document.getElementById("agent-thinking-description");
  if (!description) return;
  const budget = agentThinkingBudget();
  const budgetLabel = budget === -1 ? "unlimited" : String(budget);
  description.textContent =
    agentThinkingLevel === "disabled"
      ? "Thinking is disabled for both the local fallback and Pi CLI."
      : `${select?.selectedOptions[0]?.textContent ?? agentThinkingLevel}: local fallback uses reasoning_effort=${agentThinkingEffort()} and budget=${budgetLabel}; Pi receives the matching --thinking level.`;
}

function saveAgentThinkingLevel() {
  window.localStorage.setItem(agentThinkingLevelStorageKey, agentThinkingLevel);
}

function profileById(id: string | null | undefined): CodingAgentProfile {
  return codingAgentProfiles.find((profile) => profile.id === id) ?? codingAgentProfiles[0];
}

function activeProfile(): CodingAgentProfile {
  return profileById(activeAgentProfile);
}

function agentProfileAllowsTool(toolName: string): boolean {
  const allowed = activeProfile().allowedTools;
  return allowed.includes(toolName) || allowed.includes(toolAliasCanonical(toolName));
}

// Tool permissions that change the workspace or run side-effecting commands.
// These are blocked while Plan mode is active so the agent investigates and
// proposes a plan without touching anything.
const planBlockedPermissions: readonly AgentTool[] = ["edit", "move", "copy", "paste", "shell"];

function toolBlockedByPlanMode(permission: AgentTool): boolean {
  return agentExecutionMode === "plan" && planBlockedPermissions.includes(permission);
}

function saveExecutionMode() {
  window.localStorage.setItem(agentExecutionModeStorageKey, agentExecutionMode);
}

function syncExecutionModeUi() {
  const planButton = document.getElementById("agent-mode-plan");
  const editButton = document.getElementById("agent-mode-edit");
  planButton?.setAttribute("aria-pressed", String(agentExecutionMode === "plan"));
  editButton?.setAttribute("aria-pressed", String(agentExecutionMode === "edit"));
  planButton?.classList.toggle("active", agentExecutionMode === "plan");
  editButton?.classList.toggle("active", agentExecutionMode === "edit");
}

function setExecutionMode(mode: AgentExecutionMode, announce = true) {
  if (mode !== "plan" && mode !== "edit") return;
  agentExecutionMode = mode;
  saveExecutionMode();
  syncExecutionModeUi();
  if (!profileEventsSuspended) setAgentProfileDirty(true);
  if (announce) {
    setAgentStatus(mode === "plan" ? "Plan mode: read-only investigation" : "Edit mode: changes allowed");
  }
}

function saveActiveAgentProfile() {
  window.localStorage.setItem(agentActiveProfileStorageKey, activeAgentProfile);
}

function syncAgentProfileUi() {
  const select = document.getElementById("agent-profile-select") as HTMLSelectElement | null;
  if (select) {
    if (select.options.length === 0) {
      for (const profile of codingAgentProfiles) {
        const option = document.createElement("option");
        option.value = profile.id;
        option.textContent = profile.name;
        select.append(option);
      }
    }
    select.value = activeAgentProfile;
  }
  const profile = activeProfile();
  const description = document.getElementById("agent-profile-description");
  if (description) description.textContent = profile.description;
}

function createAgentMessageRow(role: AgentRole, message: string) {
  const row = document.createElement("div");
  const label = document.createElement("span");
  const bubble = document.createElement("div");

  row.className = `agent-message agent-message-${role}`;
  label.className = "agent-message-label";
  label.textContent = role === "user" ? "You" : role === "tool" ? "Tool" : "Agent";
  bubble.className = "agent-message-bubble";
  if (role === "tool") {
    renderToolOutput(bubble, message);
  } else {
    bubble.textContent = message;
  }
  row.append(label, bubble);
  return row;
}

function appendAgentMessage(role: AgentRole, message: string) {
  rememberAgentMessage(role, message);
  const row = createAgentMessageRow(role, message);
  agentMessageContainer().append(row);
  row.scrollIntoView({ block: "end" });

  const messageCount = agentMessageContainer().querySelectorAll(".agent-message").length;
  if (autoCompact && role !== "agent" && messageCount > 30 && messageCount % 10 === 0) {
    void compactAgentContext().catch(showAgentError);
  }
}

// Large tool results (file contents, command output) collapse into an
// expandable block so a single read doesn't bury the rest of the transcript.
function renderToolOutput(bubble: HTMLElement, message: string) {
  const lines = message.split(/\r?\n/);
  if (message.length <= 600 && lines.length <= 12) {
    bubble.textContent = message;
    return;
  }

  bubble.classList.add("agent-tool-output");
  const preview = document.createElement("div");
  preview.className = "agent-tool-output-preview";
  preview.textContent = lines.filter((line) => line.trim()).slice(0, 2).join("\n");

  const details = document.createElement("details");
  const summary = document.createElement("summary");
  const body = document.createElement("pre");
  summary.textContent = `Show full output · ${message.length.toLocaleString()} chars · ${lines.length.toLocaleString()} lines`;
  body.className = "agent-tool-arg-block";
  body.textContent = message;
  details.append(summary, body);

  bubble.append(preview, details);
}

// Animated placeholder shown in the transcript while a model request is in
// flight (prompt prefill + generation). Removed as soon as the response lands.
function showAgentThinking(label = "Processing") {
  hideAgentThinking();
  const row = document.createElement("div");
  const tag = document.createElement("span");
  const bubble = document.createElement("div");
  const text = document.createElement("span");
  const dots = document.createElement("span");
  row.className = "agent-message agent-message-agent agent-thinking";
  tag.className = "agent-message-label";
  tag.textContent = "Agent";
  bubble.className = "agent-message-bubble agent-thinking-bubble";
  text.className = "agent-thinking-text";
  text.textContent = label;
  dots.className = "agent-thinking-dots";
  dots.append(document.createElement("i"), document.createElement("i"), document.createElement("i"));
  bubble.append(text, dots);
  row.append(tag, bubble);
  agentMessageContainer().append(row);
  row.scrollIntoView({ block: "end" });
  agentThinkingRow = row;
}

function hideAgentThinking() {
  agentThinkingRow?.remove();
  agentThinkingRow = null;
}

function compactArgValue(value: unknown): string {
  const text = typeof value === "string" ? value : JSON.stringify(value);
  const compact = (text ?? "").replace(/\s+/g, " ").trim();
  return compact.length > 60 ? `${compact.slice(0, 60)}…` : compact;
}

// Renders a tool invocation as a compact card: a title plus a key/value table.
// Long string args (file contents, diffs, commands) collapse into an expandable
// block instead of flooding the transcript with thousands of escaped characters.
function appendAgentToolCall(toolName: string, args: ToolArgs) {
  const entries = Object.entries(args ?? {}).filter(([, value]) => value !== undefined);
  const summary = entries.map(([key, value]) => `${key}=${compactArgValue(value)}`).join(", ");
  rememberAgentMessage("tool", `Calling ${toolName}${summary ? ` (${summary})` : ""}`);

  const row = document.createElement("div");
  const label = document.createElement("span");
  const card = document.createElement("div");
  row.className = "agent-message agent-message-tool";
  label.className = "agent-message-label";
  label.textContent = "Tool";
  card.className = "agent-message-bubble agent-tool-card";

  const title = document.createElement("div");
  title.className = "agent-tool-card-title";
  title.textContent = `Calling ${toolName}`;
  card.append(title);

  if (entries.length > 0) {
    const list = document.createElement("dl");
    list.className = "agent-tool-args";
    for (const [key, value] of entries) {
      const term = document.createElement("dt");
      const definition = document.createElement("dd");
      term.textContent = key;
      renderToolArgValue(definition, value);
      list.append(term, definition);
    }
    card.append(list);
  }

  row.append(label, card);
  agentMessageContainer().append(row);
  row.scrollIntoView({ block: "end" });
}

function renderToolArgValue(target: HTMLElement, value: unknown) {
  const text = typeof value === "string" ? value : JSON.stringify(value, null, 2);
  if (text.length <= 200 && !text.includes("\n")) {
    target.textContent = text;
    return;
  }

  const details = document.createElement("details");
  const summary = document.createElement("summary");
  const block = document.createElement("pre");
  const lines = text.split(/\r?\n/).length;
  summary.textContent = `${text.length.toLocaleString()} chars · ${lines.toLocaleString()} lines`;
  block.className = "agent-tool-arg-block";
  block.textContent = text;
  details.append(summary, block);
  target.append(details);
}

function loadAgentPermissions(): AgentPermissions {
  try {
    const parsed = JSON.parse(window.localStorage.getItem(agentPermissionsStorageKey) ?? "{}") as Partial<AgentPermissions>;
    return { ...defaultAgentPermissions, ...parsed };
  } catch {
    return { ...defaultAgentPermissions };
  }
}

function loadMcpServers(): AgentMcpServer[] {
  try {
    const parsed = JSON.parse(window.localStorage.getItem(agentMcpServersStorageKey) ?? "[]") as AgentMcpServer[];
    return Array.isArray(parsed) ? parsed.filter((server) => isLocalHttpUrl(server.url)) : [];
  } catch {
    return [];
  }
}

function loadStoredArray<T>(key: string): T[] {
  try {
    const parsed = JSON.parse(window.localStorage.getItem(key) ?? "[]") as T[];
    return Array.isArray(parsed) ? parsed : [];
  } catch {
    return [];
  }
}

function loadTaskHistory(): AgentTaskSession[] {
  return loadStoredArray<unknown>(agentTaskHistoryStorageKey)
    .flatMap((value) => {
      if (!value || typeof value !== "object") return [];
      const record = value as Record<string, unknown>;
      if (typeof record.id !== "string" || !record.id.trim()) return [];
      const rawMessages = Array.isArray(record.messages) ? record.messages : [];
      const messages = rawMessages.flatMap((entry) => {
        if (!entry || typeof entry !== "object") return [];
        const message = entry as Record<string, unknown>;
        if (
          (message.role !== "user" && message.role !== "agent" && message.role !== "tool") ||
          typeof message.message !== "string"
        ) {
          return [];
        }
        return [{
          role: message.role,
          message: message.message,
          at: typeof message.at === "number" && Number.isFinite(message.at) ? message.at : Date.now(),
        } satisfies AgentTranscriptMessage];
      });
      const startedAt =
        typeof record.startedAt === "number" && Number.isFinite(record.startedAt) ? record.startedAt : Date.now();
      const updatedAt =
        typeof record.updatedAt === "number" && Number.isFinite(record.updatedAt) ? record.updatedAt : startedAt;
      return [{
        id: record.id,
        title: typeof record.title === "string" && record.title.trim() ? record.title.trim().slice(0, 96) : "Saved task",
        startedAt,
        updatedAt,
        messages: messages.slice(-80),
      } satisfies AgentTaskSession];
    })
    .sort((left, right) => right.updatedAt - left.updatedAt)
    .slice(0, 40);
}

function saveAgentPermissions() {
  window.localStorage.setItem(agentPermissionsStorageKey, JSON.stringify(agentPermissions));
}

function saveAgentApprovalMode() {
  window.localStorage.setItem(agentYoloModeStorageKey, String(yoloMode));
  window.localStorage.setItem(agentAutoAcceptStorageKey, String(autoAccept));
}

function saveAgentGoal() {
  window.localStorage.setItem(agentGoalStorageKey, agentGoal);
}

function saveAgentMode() {
  window.localStorage.setItem(agentModeStorageKey, agentMode);
}

function saveAgentContext() {
  window.localStorage.setItem(agentAutoCompactStorageKey, String(autoCompact));
  window.localStorage.setItem(agentContextSummaryStorageKey, contextSummary);
}

function saveMcpServers() {
  window.localStorage.setItem(agentMcpServersStorageKey, JSON.stringify(mcpServers));
}

function saveTodos() {
  window.localStorage.setItem(agentTodosStorageKey, JSON.stringify(todos));
}

function saveSkills() {
  window.localStorage.setItem(agentSkillsStorageKey, JSON.stringify(skills));
}

function saveSubagents() {
  window.localStorage.setItem(agentSubagentsStorageKey, JSON.stringify(subagents));
}

function saveTaskHistory() {
  window.localStorage.setItem(agentTaskHistoryStorageKey, JSON.stringify(taskHistory.slice(0, 40)));
}

function renderEmptyAgentConversation() {
  const welcome = document.createElement("div");
  const avatar = document.createElement("div");
  const content = document.createElement("div");
  const title = document.createElement("h3");
  const description = document.createElement("p");

  welcome.className = "agent-welcome";
  avatar.className = "agent-avatar";
  avatar.textContent = "pi";
  title.textContent = "Pi coding workbench";
  description.textContent =
    "Start a new task or open History to resume a saved conversation. Read, write, edit, bash, grep, find, and ls are available through the local workspace bridge.";
  content.append(title, description);
  welcome.append(avatar, content);
  agentMessageContainer().replaceChildren(welcome);
}

function renderTaskTranscript(task: AgentTaskSession) {
  const rows = task.messages.map((entry) => createAgentMessageRow(entry.role, entry.message));
  if (rows.length === 0) {
    renderEmptyAgentConversation();
    return;
  }
  agentMessageContainer().replaceChildren(...rows);
  rows[rows.length - 1]?.scrollIntoView({ block: "end" });
}

function resetCurrentConversationUi(status: string) {
  if (agentRunActive) cancelAgentRun();
  contextSummary = "";
  currentAgentRequest = "";
  saveAgentContext();
  syncContextUi();
  renderEmptyAgentConversation();
  ($("agent-command-input") as HTMLTextAreaElement).value = "";
  setAgentOutput("");
  setAgentStatus(status);
}

function closeAgentHistory() {
  $("agent-history-overlay").hidden = true;
}

function renderAgentHistory() {
  const list = $("agent-history-list");
  const count = $("agent-history-count");
  const deleteAll = $("agent-history-delete-all") as HTMLButtonElement;
  const clearCurrent = $("agent-history-clear-current") as HTMLButtonElement;
  list.replaceChildren();

  const savedTasks = taskHistory.filter((task) => task.messages.length > 0);
  count.textContent =
    savedTasks.length === 0 ? "No saved sessions" : `${savedTasks.length} saved session${savedTasks.length === 1 ? "" : "s"}`;
  deleteAll.disabled = savedTasks.length === 0;
  clearCurrent.disabled = !taskHistory.some((task) => task.id === currentTaskId && task.messages.length > 0);

  if (savedTasks.length === 0) {
    const empty = document.createElement("div");
    const title = document.createElement("strong");
    const description = document.createElement("span");
    empty.className = "agent-history-empty";
    title.textContent = "Your history is clear";
    description.textContent = "Completed conversations will appear here automatically.";
    empty.append(title, description);
    list.append(empty);
    return;
  }

  for (const task of savedTasks) {
    const item = document.createElement("article");
    const openButton = document.createElement("button");
    const heading = document.createElement("strong");
    const preview = document.createElement("span");
    const meta = document.createElement("span");
    const deleteButton = document.createElement("button");
    const lastMessage = task.messages[task.messages.length - 1]?.message.trim().replace(/\s+/g, " ") ?? "";

    item.className = "agent-history-item";
    if (task.id === currentTaskId) item.classList.add("is-current");
    openButton.type = "button";
    openButton.className = "agent-history-open";
    openButton.dataset.taskId = task.id;
    openButton.setAttribute("aria-label", `Open ${task.title}`);
    heading.textContent = task.title;
    preview.textContent = lastMessage.length > 120 ? `${lastMessage.slice(0, 117)}...` : lastMessage;
    meta.textContent = `${task.messages.length} message${task.messages.length === 1 ? "" : "s"} · ${new Date(task.updatedAt).toLocaleString()}`;
    openButton.append(heading, preview, meta);

    deleteButton.type = "button";
    deleteButton.className = "agent-history-delete";
    deleteButton.dataset.taskId = task.id;
    deleteButton.setAttribute("aria-label", `Delete ${task.title}`);
    deleteButton.title = "Delete session";
    deleteButton.innerHTML =
      '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M4 7h16M9 7V4h6v3M7 7l1 13h8l1-13"/><path d="M10 11v5M14 11v5"/></svg>';

    item.append(openButton, deleteButton);
    list.append(item);
  }
}

function openAgentHistory() {
  renderAgentHistory();
  $("agent-history-overlay").hidden = false;
  ($("agent-history-close") as HTMLButtonElement).focus({ preventScroll: true });
}

function resumeAgentTask(taskId: string) {
  const task = taskHistory.find((candidate) => candidate.id === taskId);
  if (!task) return;
  if (agentRunActive) cancelAgentRun();
  currentTaskId = task.id;
  taskHistory = [task, ...taskHistory.filter((candidate) => candidate.id !== task.id)];
  saveTaskHistory();
  contextSummary = "";
  currentAgentRequest = "";
  saveAgentContext();
  syncContextUi();
  renderTaskTranscript(task);
  closeAgentHistory();
  setAgentStatus(`Resumed: ${task.title}`);
}

async function clearCurrentAgentConversation() {
  const task = taskHistory.find((candidate) => candidate.id === currentTaskId);
  const visibleMessages = agentMessageContainer().querySelectorAll(".agent-message").length;
  if ((!task || task.messages.length === 0) && visibleMessages === 0) {
    setAgentStatus("Current conversation is already clear");
    return;
  }
  const confirmed = await confirmAction({
    title: "Clear current conversation?",
    message: "This permanently removes the active conversation and its saved transcript. This action cannot be undone.",
    okLabel: "Clear conversation",
    cancelLabel: "Cancel",
    kind: "danger",
  });
  if (!confirmed) return;
  taskHistory = taskHistory.filter((candidate) => candidate.id !== currentTaskId);
  currentTaskId = "";
  saveTaskHistory();
  resetCurrentConversationUi("Current conversation cleared");
  renderAgentHistory();
}

async function deleteAgentTask(taskId: string) {
  const task = taskHistory.find((candidate) => candidate.id === taskId);
  if (!task) return;
  const confirmed = await confirmAction({
    title: "Delete saved session?",
    message: `"${task.title}" and all of its messages will be permanently deleted.`,
    okLabel: "Delete session",
    cancelLabel: "Cancel",
    kind: "danger",
  });
  if (!confirmed) return;
  taskHistory = taskHistory.filter((candidate) => candidate.id !== taskId);
  if (currentTaskId === taskId) {
    currentTaskId = "";
    resetCurrentConversationUi("Session deleted");
  } else {
    setAgentStatus("Saved session deleted");
  }
  saveTaskHistory();
  renderAgentHistory();
}

async function deleteAllAgentHistory() {
  const savedCount = taskHistory.filter((task) => task.messages.length > 0).length;
  if (savedCount === 0) return;
  const confirmed = await confirmAction({
    title: "Delete all Agent history?",
    message: `This permanently deletes ${savedCount} saved session${savedCount === 1 ? "" : "s"} and every message in them. This action cannot be undone.`,
    okLabel: "Delete all history",
    cancelLabel: "Cancel",
    kind: "danger",
  });
  if (!confirmed) return;
  taskHistory = [];
  currentTaskId = "";
  saveTaskHistory();
  resetCurrentConversationUi("All Agent history deleted");
  renderAgentHistory();
}

function currentTask(): AgentTaskSession {
  let task = taskHistory.find((candidate) => candidate.id === currentTaskId);
  if (!task) {
    task = {
      id: crypto.randomUUID(),
      title: "New task",
      startedAt: Date.now(),
      updatedAt: Date.now(),
      messages: [],
    };
    currentTaskId = task.id;
    taskHistory = [task, ...taskHistory];
    saveTaskHistory();
  }
  return task;
}

function titleFromMessage(message: string): string {
  const firstLine = message.trim().split(/\r?\n/).find(Boolean) ?? "New task";
  return firstLine.length > 64 ? `${firstLine.slice(0, 61)}...` : firstLine;
}

function rememberAgentMessage(role: AgentRole, message: string) {
  const task = currentTask();
  if (task.title === "New task" && role === "user") {
    task.title = titleFromMessage(message);
  }
  task.updatedAt = Date.now();
  task.messages.push({ role, message, at: Date.now() });
  task.messages = task.messages.slice(-80);
  taskHistory = [task, ...taskHistory.filter((candidate) => candidate.id !== task.id)];
  saveTaskHistory();
}

function startNewAgentTask() {
  if (agentRunActive) cancelAgentRun();

  const task: AgentTaskSession = {
    id: crypto.randomUUID(),
    title: "New task",
    startedAt: Date.now(),
    updatedAt: Date.now(),
    messages: [],
  };
  currentTaskId = task.id;
  taskHistory = [task, ...taskHistory.filter((candidate) => candidate.messages.length > 0)].slice(0, 40);
  saveTaskHistory();

  // A new task is a clean slate: drop the rolling context summary and the last
  // request so the model is driven by the new prompt, not the previous task.
  contextSummary = "";
  currentAgentRequest = "";
  saveAgentContext();
  syncContextUi();

  agentMessageContainer().replaceChildren();
  ($("agent-command-input") as HTMLTextAreaElement).value = "";
  setAgentOutput("");
  setAgentStatus("New task started");
}

function renderTaskHistorySummary(): string {
  if (taskHistory.length === 0) return "No task history yet.";
  return taskHistory
    .slice(0, 12)
    .map((task, index) => {
      const updated = new Date(task.updatedAt).toLocaleString();
      return `${index + 1}. ${task.title}\n   ${task.messages.length} messages, updated ${updated}`;
    })
    .join("\n");
}

function syncPermissionControls() {
  for (const tool of permissionTools) {
    const select = $(`agent-permission-${tool}`) as HTMLSelectElement;
    select.value = yoloMode ? "allow" : agentPermissions[tool];
    select.disabled = yoloMode;
  }
  ($("agent-yolo-mode") as HTMLInputElement).checked = yoloMode;
  ($("agent-auto-accept") as HTMLInputElement).checked = autoAccept;
  const headerAutoApprove = document.getElementById("agent-auto-approve-header") as HTMLInputElement | null;
  if (headerAutoApprove) headerAutoApprove.checked = autoAccept;
}

function syncGoalUi() {
  ($("agent-goal") as HTMLTextAreaElement).value = agentGoal;
  $("agent-goal-status").textContent = agentGoal ? "Goal: active" : "Goal: not set";
}

function syncModeUi() {
  for (const button of document.querySelectorAll<HTMLButtonElement>("[data-agent-mode]")) {
    const selected = button.dataset.agentMode === agentMode;
    button.setAttribute("aria-pressed", String(selected));
  }

  const descriptions: Record<AgentMode, string> = {
    architect: "Plan first, identify files and risks, then choose the safest implementation path.",
    ask: "Answer questions and explain code without changing the workspace.",
    coder: "Implement directly, use tools, and verify the result.",
    debugger: "Reproduce issues, inspect outputs, patch narrowly, and retest.",
    reviewer: "Review for concrete bugs, regressions, and missing tests.",
    tester: "Find and run the right tests, then summarize failures.",
    orchestrator: "Break work into tasks and coordinate specialist agents.",
  };
  $("agent-mode-description").textContent = descriptions[agentMode];
  syncAgentProfileUi();
  renderSkills();
}

async function confirmAgentItemDeletion(
  itemType: string,
  itemName: string,
  remove: () => void,
) {
  const confirmed = await confirmAction({
    title: `Delete ${itemType}?`,
    message: `"${itemName}" will be permanently removed from this Agent configuration.`,
    okLabel: "Delete",
    cancelLabel: "Cancel",
    kind: "danger",
  });
  if (!confirmed) return;
  remove();
  setAgentStatus(`${itemType} deleted`);
}

function renderTodos() {
  const list = $("agent-todo-list");
  list.replaceChildren();

  if (todos.length === 0) {
    const empty = document.createElement("div");
    empty.className = "empty-state";
    empty.textContent = "No tasks yet";
    list.append(empty);
    return;
  }

  for (const todo of todos) {
    const row = document.createElement("div");
    const label = document.createElement("label");
    const checkbox = document.createElement("input");
    const text = document.createElement("span");
    const remove = document.createElement("button");
    checkbox.type = "checkbox";
    checkbox.checked = todo.done;
    checkbox.addEventListener("change", () => {
      todos = todos.map((item) => (item.id === todo.id ? { ...item, done: checkbox.checked } : item));
      saveTodos();
      renderTodos();
    });
    text.textContent = todo.text;
    row.className = "agent-todo-row";
    label.className = "agent-todo-main";
    label.append(checkbox, text);
    remove.type = "button";
    remove.className = "agent-item-delete";
    remove.textContent = "Delete";
    remove.setAttribute("aria-label", `Delete task ${todo.text}`);
    remove.addEventListener("click", () => {
      void confirmAgentItemDeletion("task", todo.text, () => {
        todos = todos.filter((item) => item.id !== todo.id);
        saveTodos();
        renderTodos();
      }).catch(showAgentError);
    });
    row.append(label, remove);
    list.append(row);
  }
}

function renderSkills() {
  renderChipList("agent-skill-list", skills, "skill");

  const loaded = new Set(skills.map((skill) => skill.name.toLowerCase()));
  const available = applicableDiscoveredSkills().filter((skill) => !loaded.has(skill.name.toLowerCase()));
  if (available.length === 0) return;

  const list = $("agent-skill-list");
  const heading = document.createElement("div");
  heading.className = "agent-skill-available-heading";
  heading.textContent = `Discovered skills (${available.length})`;
  list.append(heading);

  for (const skill of available) {
    const row = document.createElement("div");
    const main = document.createElement("span");
    const load = document.createElement("button");
    row.className = "agent-chip-row";
    const scope = skill.modes.trim() ? ` · ${skill.modes.trim()}` : "";
    main.textContent = `${skill.name}: ${skill.description || skill.source}${scope}`;
    load.type = "button";
    load.textContent = "Load";
    load.addEventListener("click", () => {
      loadSkill(skill.name, "").catch(showAgentError);
    });
    row.append(main, load);
    list.append(row);
  }
}

function renderSubagents() {
  renderChipList("agent-subagent-list", subagents, "subagent");
}

function renderChipList(
  targetId: string,
  items: Array<AgentSkill | AgentSubagent>,
  kind: "skill" | "subagent",
) {
  const list = $(targetId);
  list.replaceChildren();

  if (items.length === 0) {
    const empty = document.createElement("div");
    empty.className = "empty-state";
    empty.textContent = kind === "skill" ? "No skills loaded" : "No subagents configured";
    list.append(empty);
    return;
  }

  for (const item of items) {
    const row = document.createElement("div");
    const main = document.createElement("span");
    const actions = document.createElement("div");
    const toggle = document.createElement("button");
    const remove = document.createElement("button");
    row.className = "agent-chip-row";
    actions.className = "agent-item-actions";
    main.textContent =
      kind === "skill"
        ? `${item.enabled ? "on" : "off"} ${item.name}: ${(item as AgentSkill).instructions}`
        : `${item.enabled ? "on" : "off"} ${item.name}: ${(item as AgentSubagent).role}`;
    toggle.type = "button";
    toggle.textContent = item.enabled ? "Disable" : "Enable";
    toggle.addEventListener("click", () => {
      if (kind === "skill") {
        skills = skills.map((skill) => (skill.id === item.id ? { ...skill, enabled: !skill.enabled } : skill));
        saveSkills();
        renderSkills();
      } else {
        subagents = subagents.map((agent) =>
          agent.id === item.id ? { ...agent, enabled: !agent.enabled } : agent,
        );
        saveSubagents();
        renderSubagents();
      }
    });
    remove.type = "button";
    remove.className = "agent-item-delete";
    remove.textContent = "Delete";
    remove.setAttribute("aria-label", `Delete ${kind} ${item.name}`);
    remove.addEventListener("click", () => {
      void confirmAgentItemDeletion(kind, item.name, () => {
        if (kind === "skill") {
          skills = skills.filter((skill) => skill.id !== item.id);
          saveSkills();
          renderSkills();
        } else {
          subagents = subagents.filter((agent) => agent.id !== item.id);
          saveSubagents();
          renderSubagents();
        }
      }).catch(showAgentError);
    });
    actions.append(toggle, remove);
    row.append(main, actions);
    list.append(row);
  }
}

function syncContextUi() {
  ($("agent-auto-compact") as HTMLInputElement).checked = autoCompact;
  ($("agent-context-summary") as HTMLTextAreaElement).value = contextSummary;
}

function setYoloMode(enabled: boolean) {
  yoloMode = enabled;
  if (enabled) {
    agentPermissions = {
      read: "allow",
      edit: "allow",
      move: "allow",
      copy: "allow",
      paste: "allow",
      browse: "allow",
      shell: "allow",
      todo: "allow",
      skill: "allow",
      subagent: "allow",
      externalDirectory: "allow",
    };
  } else {
    // Auto approve is a YOLO capability: turning YOLO off must also revoke it.
    autoAccept = false;
    agentPermissions = loadAgentPermissions();
  }
  saveAgentApprovalMode();
  syncPermissionControls();
  if (!profileEventsSuspended) setAgentProfileDirty(true);
  setAgentStatus(enabled ? "YOLO mode enabled: all local tool asks are auto-approved" : "YOLO mode disabled");
}

function setAutoApprove(enabled: boolean, label = "Auto approve") {
  autoAccept = enabled;
  if (enabled && !yoloMode) {
    yoloMode = true;
    agentPermissions = {
      read: "allow",
      edit: "allow",
      move: "allow",
      copy: "allow",
      paste: "allow",
      browse: "allow",
      shell: "allow",
      todo: "allow",
      skill: "allow",
      subagent: "allow",
      externalDirectory: "allow",
    };
  }
  saveAgentApprovalMode();
  syncPermissionControls();
  if (!profileEventsSuspended) setAgentProfileDirty(true);
  setAgentStatus(
    enabled
      ? `${label} enabled · YOLO mode enabled`
      : `${label} disabled${yoloMode ? " · YOLO mode remains enabled" : ""}`,
  );
}

function bindPermissionControls() {
  for (const tool of permissionTools) {
    const select = $(`agent-permission-${tool}`) as HTMLSelectElement;
    select.addEventListener("change", () => {
      agentPermissions = {
        ...agentPermissions,
        [tool]: select.value as AgentPermission,
      };
      saveAgentPermissions();
      setAgentProfileDirty(true);
      setAgentStatus("Permission changed · save the profile to keep it");
    });
  }

  $("agent-yolo-mode").addEventListener("change", () => {
    setYoloMode(($("agent-yolo-mode") as HTMLInputElement).checked);
  });
  $("agent-auto-accept").addEventListener("change", () => {
    setAutoApprove(($("agent-auto-accept") as HTMLInputElement).checked, "Auto accept");
  });
  const headerAutoApprove = document.getElementById("agent-auto-approve-header") as HTMLInputElement | null;
  headerAutoApprove?.addEventListener("change", () => {
    setAutoApprove(headerAutoApprove.checked);
  });
}

async function requestToolPermission(tool: AgentTool, summary: string): Promise<boolean> {
  const permission = yoloMode ? "allow" : agentPermissions[tool];
  if (permission === "allow" || autoAccept) {
    if (permission === "ask" && autoAccept) {
      setAgentStatus(`Auto-approved ${tool}`);
    }
    return true;
  }
  if (permission === "deny") {
    appendAgentMessage("tool", `${tool} denied: ${summary}`);
    setAgentStatus(`${tool} denied by permission settings`);
    return false;
  }

  const approved = await confirmAction({
    title: "Approve agent action",
    message: `${tool}\n\n${summary}`,
    okLabel: "Run",
    cancelLabel: "Deny",
    kind: tool === "read" || tool === "browse" ? "warning" : "danger",
  });
  setAgentStatus(`${approved ? "Approved" : "Denied"} ${tool}`);
  if (!approved) {
    appendAgentMessage("tool", `${tool} denied: ${summary}`);
  }
  return approved;
}

async function pathInfo(path: string): Promise<AgentPathInfo> {
  return invoke<AgentPathInfo>("agent_path_info", { path });
}

async function requirePathPermissions(tool: AgentTool, summary: string, paths: string[]): Promise<boolean> {
  if (!(await requestToolPermission(tool, summary))) return false;

  for (const path of paths.filter(Boolean)) {
    const info = await pathInfo(path);
    if (info.external) {
      const externalSummary = `${path}\n\nResolved outside workspace:\n${info.path}`;
      if (!(await requestToolPermission("externalDirectory", externalSummary))) {
        return false;
      }
    }
  }

  return true;
}

function normalizeFsPath(value: string): string {
  return value.replace(/^\\\\\?\\/, "").replace(/\\/g, "/");
}

function displayAgentPath(info: AgentPathInfo): string {
  const path = normalizeFsPath(info.path);
  const root = normalizeFsPath(info.workspaceRoot).replace(/\/+$/, "");
  if (!info.external && root && (path === root || path.startsWith(`${root}/`))) {
    return path === root ? "." : path.slice(root.length + 1);
  }
  return path;
}

function formatPathInfo(info: AgentPathInfo): string {
  const kind = info.isDirectory ? "directory" : info.isFile ? "file" : info.exists ? "path" : "missing";
  const size = info.sizeBytes == null ? "" : `, ${info.sizeBytes} bytes`;
  const scope = info.external ? ", external" : "";
  return `${kind}${size}${scope}\n${displayAgentPath(info)}`;
}

function formatReadResult(result: AgentReadResult): string {
  if (result.entries) {
    const entries = result.entries
      .map((entry) => `${entry.isDirectory ? "[dir] " : "      "}${entry.path}`)
      .join("\n");
    return `${formatPathInfo(result.info)}\n\n${entries || "Directory is empty"}`;
  }

  return `${formatPathInfo(result.info)}\n\n${result.content ?? ""}`;
}

function formatTransferResult(action: string, result: AgentTransferResult): string {
  return `${action} complete\n\nFrom: ${displayAgentPath(result.from)}\nTo: ${displayAgentPath(result.to)}`;
}

function formatWriteResult(result: AgentWriteResult): string {
  return `Wrote ${result.bytesWritten} bytes to ${displayAgentPath(result.info)}`;
}

async function readAgentPath(path = (($("agent-path") as HTMLInputElement).value || "").trim()): Promise<string> {
  if (!path) throw new Error("Choose a path to read");
  if (!(await requirePathPermissions("read", path, [path]))) return "Read denied";

  const result = await invoke<AgentReadResult>("agent_read_path", { path });
  if (result.content != null) {
    ($("agent-file-content") as HTMLTextAreaElement).value = result.content;
  }
  ($("agent-path") as HTMLInputElement).value = path;
  const output = formatReadResult(result);
  setAgentOutput(output);
  appendAgentMessage("tool", output);
  return output;
}

async function editAgentFile(
  path = (($("agent-path") as HTMLInputElement).value || "").trim(),
  content?: string,
): Promise<string> {
  if (!path) throw new Error("Choose a path to edit");
  const nextContent = content ?? ($("agent-file-content") as HTMLTextAreaElement).value;
  const misplacedArtifact = misplacedHtmlArtifactPathReason(path);
  if (misplacedArtifact) return blockToolWrite(misplacedArtifact);
  if (!(await requirePathPermissions("edit", path, [path]))) return "Edit denied";
  const fullEditBlocked = await fullFileWriteBlockedReason(path);
  if (fullEditBlocked) return fullEditBlocked;
  return writeAgentFileContent(path, nextContent, true);
}

async function writeAgentFileContent(path: string, nextContent: string, validateArtifact: boolean): Promise<string> {
  if (validateArtifact) {
    const weakArtifact = weakArtifactWriteReason(path, nextContent);
    if (weakArtifact) {
      const output = `Write blocked: ${weakArtifact}`;
      setAgentOutput(output);
      appendAgentMessage("tool", output);
      return output;
    }
  }

  const result = await invoke<AgentWriteResult>("agent_write_file", {
    request: { path, content: nextContent, create: true },
  });
  const output = formatWriteResult(result);
  setAgentOutput(output);
  appendAgentMessage("tool", output);
  return output;
}

async function writeAgentFile(path: string, content: string): Promise<string> {
  if (!path) throw new Error("Choose a path to write");
  if (!content) throw new Error("Provide content to write");
  const misplacedArtifact = misplacedHtmlArtifactPathReason(path);
  if (misplacedArtifact) return blockToolWrite(misplacedArtifact);
  if (!(await requirePathPermissions("edit", path, [path]))) return "Write denied";
  const fullWriteBlocked = await fullFileWriteBlockedReason(path);
  if (fullWriteBlocked) return fullWriteBlocked;
  return writeAgentFileContent(path, content, true);
}

function blockToolWrite(reason: string): string {
  const output = `Write blocked: ${reason}`;
  setAgentOutput(output);
  appendAgentMessage("tool", output);
  return output;
}

function normalizeAgentRelativePath(path: string): string {
  return path.trim().replace(/^\.?[\\/]+/, "").replace(/\\/g, "/").toLowerCase();
}

function currentRequestMentionsPath(path: string): boolean {
  const normalizedPath = normalizeAgentRelativePath(path);
  const normalizedRequest = currentAgentRequest.replace(/\\/g, "/").toLowerCase();
  return normalizedPath.length > 0 && normalizedRequest.includes(normalizedPath);
}

function misplacedHtmlArtifactPathReason(path: string): string | null {
  const normalizedPath = normalizeAgentRelativePath(path);
  if (normalizedPath !== "src/index.html") return null;
  if (!requestLooksLikeArtifactCreation()) return null;
  if (currentRequestMentionsPath(path)) return null;
  return "standalone HTML/page/game artifacts should not be written to src/index.html in this project. Use index.html or a named root file such as game.html unless the user explicitly requests src/index.html.";
}

async function fullFileWriteBlockedReason(path: string): Promise<string | null> {
  const info = await pathInfo(path);
  if (info.exists && requestLooksLikeSurgicalEdit()) {
    const output = [
      "Write blocked: this request looks like a fix/change to an existing file, so full-file replacement is not allowed.",
      "Inspect first with search_lines/go_to_line, then patch with edit_lines, replace_in_file, insert_content, apply_diff, or shell.",
      "PowerShell examples: Select-String -Path file -Pattern \"needle\"; $lines = Get-Content file; $lines[10..30]; replace ranges by editing the $lines array and Set-Content.",
      "Linux examples: rg -n \"needle\" file; nl -ba file | sed -n '10,30p'; perl -0pi -e 's/old/new/g' file; sed -i '10,12d' file.",
      formatPathInfo(info),
    ].join("\n");
    setAgentOutput(output);
    appendAgentMessage("tool", output);
    return output;
  }
  return null;
}

function requestLooksLikeArtifactCreation(): boolean {
  return /\b(write|create|build|make|implement|generate)\b/i.test(currentAgentRequest) &&
    /\b(game|app|page|html|website|component|script|tool)\b/i.test(currentAgentRequest);
}

function requestLooksLikeSurgicalEdit(): boolean {
  if (/\b(rewrite|recreate|regenerate|from scratch|replace (the )?(whole|entire) file|full rewrite)\b/i.test(currentAgentRequest)) {
    return false;
  }
  return /\b(fix|error|bug|issue|change|update|modify|patch|adjust|repair|broken)\b/i.test(currentAgentRequest);
}

function htmlTagMatches(content: string, pattern: RegExp): RegExpMatchArray[] {
  return Array.from(content.matchAll(pattern));
}

function htmlTruncationReason(content: string): string | null {
  const opensHtml = /<!doctype|<html\b/i.test(content);
  if (opensHtml && !/<\/html>/i.test(content)) {
    return "the proposed HTML document appears truncated because it does not close the html element. Send one complete replacement including </html>, or use append_to_file/insert_content to add only the missing tail to the existing file.";
  }

  const scriptOpens = htmlTagMatches(content, /<script\b[^>]*>/gi);
  const scriptCloses = htmlTagMatches(content, /<\/script\s*>/gi);
  const scriptOpenCount = scriptOpens.length;
  const scriptCloseCount = scriptCloses.length;
  if (scriptOpenCount > scriptCloseCount) {
    return `the proposed HTML document appears truncated because it opens ${scriptOpenCount} script tag${scriptOpenCount === 1 ? "" : "s"} but closes ${scriptCloseCount}. Do not retry another partial write_to_file call; send one complete replacement with closing </script> tags, or use append_to_file/insert_content to add the missing tail to the existing file.`;
  }

  const finalScriptOpen = scriptOpens[scriptOpens.length - 1]?.index ?? -1;
  const finalScriptClose = scriptCloses[scriptCloses.length - 1]?.index ?? -1;
  if (finalScriptOpen > finalScriptClose) {
    return "the proposed HTML document appears truncated near the final script tag. Send one complete replacement, or continue the existing file with append_to_file/insert_content instead of rewriting from scratch.";
  }

  return null;
}

function weakArtifactWriteReason(path: string, content: string): string | null {
  if (!requestLooksLikeArtifactCreation()) return null;
  const normalizedContent = content.trim().toLowerCase().replace(/\s+/g, " ");
  const normalizedRequest = currentAgentRequest.trim().toLowerCase().replace(/\s+/g, " ");
  const pathLooksLikeArtifact = /\.(html|htm|js|ts|tsx|jsx|css|py|rs|go|java|cs|vue|svelte)$/i.test(path) || !path.includes(".");

  if (!pathLooksLikeArtifact) return null;
  if (content.trim().length < 200) {
    return "the proposed file is too small for the requested artifact. The agent should reason, plan, and write a complete implementation instead of saving the prompt text.";
  }
  if (normalizedContent && normalizedRequest.includes(normalizedContent)) {
    return "the proposed file content is just a fragment of the user request.";
  }
  if (/\b(html|single html|game|website|page)\b/i.test(currentAgentRequest) && !/<(html|canvas|script|style|body)\b/i.test(content)) {
    return "the request asks for an HTML/page artifact, but the proposed content does not contain an HTML document, styles, or script.";
  }
  if (/\b(html|single html|game|website|page)\b/i.test(currentAgentRequest)) {
    return htmlTruncationReason(content);
  }
  return null;
}

function shellQuote(value: string): string {
  return `"${value.replace(/"/g, '\\"')}"`;
}

async function deleteAgentPath(path = (($("agent-path") as HTMLInputElement).value || "").trim()): Promise<string> {
  if (!path) throw new Error("Choose a path to delete");
  if (!(await requirePathPermissions("edit", `Delete ${path}`, [path]))) return "Delete denied";

  const before = await pathInfo(path);
  const result = await invoke<AgentPathInfo>("agent_delete_path", { path });
  const output = `Deleted ${before.isDirectory ? "directory" : "file"}\n${formatPathInfo(result)}`;
  setAgentOutput(output);
  appendAgentMessage("tool", output);
  return output;
}

async function insertContentInAgentFile(path: string, content: string, line: number | null): Promise<string> {
  if (!path) throw new Error("Choose a path to edit");
  if (!content) throw new Error("Provide content to insert");
  if (!(await requirePathPermissions("edit", `Insert content into ${path}`, [path]))) return "Insert denied";

  const result = await invoke<AgentReadResult>("agent_read_path", { path });
  const current = result.content;
  if (current == null) throw new Error(`${path} is not a text file`);

  const lines = current.split(/\r?\n/);
  const normalizedLine = line == null || line <= 0 ? lines.length + 1 : Math.min(Math.floor(line), lines.length + 1);
  lines.splice(normalizedLine - 1, 0, content);
  const nextContent = lines.join(current.includes("\r\n") ? "\r\n" : "\n");
  const writeResult = await invoke<AgentWriteResult>("agent_write_file", {
    request: { path, content: nextContent, create: true },
  });
  const output = `Inserted content at line ${normalizedLine}\n${formatWriteResult(writeResult)}`;
  setAgentOutput(output);
  appendAgentMessage("tool", output);
  return output;
}

async function appendContentToAgentFile(path: string, content: string): Promise<string> {
  if (!path) throw new Error("Choose a path to edit");
  if (!content) throw new Error("Provide content to append");
  if (!(await requirePathPermissions("edit", `Append content to ${path}`, [path]))) return "Append denied";

  let current = "";
  const info = await pathInfo(path);
  if (info.exists) {
    const result = await invoke<AgentReadResult>("agent_read_path", { path });
    if (result.content == null) throw new Error(`${path} is not a text file`);
    current = result.content;
  }

  const writeResult = await invoke<AgentWriteResult>("agent_write_file", {
    request: { path, content: current + content, create: true },
  });
  const appendedBytes = new TextEncoder().encode(content).length;
  const output = `Appended ${appendedBytes} bytes\n${formatWriteResult(writeResult)}`;
  setAgentOutput(output);
  appendAgentMessage("tool", output);
  return output;
}

function escapeRegex(value: string): string {
  return value.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
}

function globToRegex(glob: string): RegExp {
  const body = glob
    .split("*")
    .map((part) => part.split("?").map(escapeRegex).join("."))
    .join(".*");
  return new RegExp(`^${body}$`, "i");
}

function lineNumberWidth(total: number): number {
  return Math.max(1, String(total).length);
}

function formatNumberedLines(lines: string[], startLine: number, totalLines: number): string {
  const width = lineNumberWidth(totalLines);
  return lines
    .map((line, index) => `${String(startLine + index).padStart(width, " ")}: ${line}`)
    .join("\n");
}

async function readTextAgentFile(path: string): Promise<{ content: string; info: AgentPathInfo }> {
  if (!path) throw new Error("Choose a file path");
  const result = await invoke<AgentReadResult>("agent_read_path", { path });
  if (result.content == null) throw new Error(`${path} is not a text file`);
  return { content: result.content, info: result.info };
}

async function findAgentFiles(path: string, query: string, glob: string): Promise<string> {
  if (!(await requirePathPermissions("read", `Find files in ${path}`, [path]))) return "Find files denied";

  const queryLower = query.toLowerCase();
  const globRegex = glob ? globToRegex(glob) : null;
  const matches: string[] = [];
  const visited = new Set<string>();
  const skipDirs = new Set([".git", "node_modules", "dist", "target"]);
  const maxMatches = 500;
  const maxVisited = 5000;

  async function walk(currentPath: string): Promise<void> {
    if (visited.size >= maxVisited || matches.length >= maxMatches || visited.has(currentPath)) return;
    visited.add(currentPath);
    const result = await invoke<AgentReadResult>("agent_read_path", { path: currentPath });
    if (!result.entries) {
      const name = currentPath.split(/[\\/]/).pop() ?? currentPath;
      const queryMatches = !queryLower || name.toLowerCase().includes(queryLower) || currentPath.toLowerCase().includes(queryLower);
      const globMatches = !globRegex || globRegex.test(name) || globRegex.test(currentPath);
      if (queryMatches && globMatches) matches.push(currentPath);
      return;
    }

    for (const entry of result.entries) {
      if (entry.isDirectory && skipDirs.has(entry.name)) continue;
      if (entry.isDirectory) {
        await walk(entry.path);
      } else {
        const queryMatches = !queryLower || entry.name.toLowerCase().includes(queryLower) || entry.path.toLowerCase().includes(queryLower);
        const globMatches = !globRegex || globRegex.test(entry.name) || globRegex.test(entry.path);
        if (queryMatches && globMatches) matches.push(entry.path);
        if (matches.length >= maxMatches) return;
      }
    }
  }

  await walk(path);
  const suffix = visited.size >= maxVisited ? `\n\nStopped after scanning ${maxVisited} paths; narrow path/query for more.` : "";
  const output = matches.length
    ? `Found ${matches.length} file${matches.length === 1 ? "" : "s"}${query ? ` matching "${query}"` : ""}${glob ? ` with glob "${glob}"` : ""}:\n${matches.join("\n")}${suffix}`
    : `No files found${query ? ` matching "${query}"` : ""}${glob ? ` with glob "${glob}"` : ""}.`;
  setAgentOutput(output);
  appendAgentMessage("tool", output);
  return output;
}

async function readAgentLineRange(
  path: string,
  line: number | null,
  end: number | null,
  context: number | null,
): Promise<string> {
  if (!(await requirePathPermissions("read", `Go to line ${line ?? ""} in ${path}`, [path]))) return "Go to line denied";
  if (line == null || line <= 0) throw new Error("Provide a 1-based line number");

  const { content, info } = await readTextAgentFile(path);
  const lines = content.split(/\r?\n/);
  const total = lines.length;
  const startLine = Math.min(Math.max(1, Math.floor(line)), total);
  const endLine = end == null || end <= 0 ? startLine : Math.min(Math.max(startLine, Math.floor(end)), total);
  const contextSize = end == null ? Math.max(0, Math.floor(context ?? 20)) : 0;
  const from = Math.max(1, startLine - contextSize);
  const to = Math.min(total, endLine + contextSize);
  const excerpt = lines.slice(from - 1, to);
  const output = `${formatPathInfo(info)}\nShowing lines ${from}-${to} of ${total}\n\n${formatNumberedLines(excerpt, from, total)}`;
  setAgentOutput(output);
  appendAgentMessage("tool", output);
  return output;
}

async function searchAgentLines(path: string, query: string, regex: boolean): Promise<string> {
  if (!query) throw new Error("Enter a line search query");
  if (!(await requirePathPermissions("read", `Search lines in ${path}`, [path]))) return "Line search denied";

  const { content, info } = await readTextAgentFile(path);
  const lines = content.split(/\r?\n/);
  const matcher = regex
    ? new RegExp(query, "i")
    : new RegExp(escapeRegex(query), "i");
  const matches = lines
    .map((line, index) => ({ line, number: index + 1 }))
    .filter((item) => matcher.test(item.line))
    .slice(0, 200);
  const output = matches.length
    ? `${formatPathInfo(info)}\nFound ${matches.length} matching line${matches.length === 1 ? "" : "s"} for "${query}"${matches.length === 200 ? " (first 200 shown)" : ""}\n\n${matches.map((item) => formatNumberedLines([item.line], item.number, lines.length)).join("\n")}`
    : `${formatPathInfo(info)}\nNo matching lines for "${query}".`;
  setAgentOutput(output);
  appendAgentMessage("tool", output);
  return output;
}

async function editAgentLines(
  path: string,
  start: number | null,
  end: number | null,
  replacement: string,
): Promise<string> {
  if (!path) throw new Error("Choose a file path");
  if (start == null || start <= 0) throw new Error("Provide a 1-based start line");
  if (replacement == null) throw new Error("Provide replacement content");
  if (!(await requirePathPermissions("edit", `Edit lines ${start}-${end ?? start} in ${path}`, [path]))) return "Edit lines denied";

  const { content } = await readTextAgentFile(path);
  const newline = content.includes("\r\n") ? "\r\n" : "\n";
  const lines = content.split(/\r?\n/);
  const normalizedStart = Math.min(Math.max(1, Math.floor(start)), lines.length);
  const normalizedEnd = end == null || end <= 0
    ? normalizedStart
    : Math.min(Math.max(normalizedStart, Math.floor(end)), lines.length);
  const replacementLines = replacement.length ? replacement.split(/\r?\n/) : [];
  lines.splice(normalizedStart - 1, normalizedEnd - normalizedStart + 1, ...replacementLines);
  const nextContent = lines.join(newline);
  const writeResult = await invoke<AgentWriteResult>("agent_write_file", {
    request: { path, content: nextContent, create: true },
  });
  const output = `Replaced lines ${normalizedStart}-${normalizedEnd} with ${replacementLines.length} line${replacementLines.length === 1 ? "" : "s"}\n${formatWriteResult(writeResult)}`;
  setAgentOutput(output);
  appendAgentMessage("tool", output);
  return output;
}

async function listCodeDefinitionNames(path: string): Promise<string> {
  if (!(await requirePathPermissions("read", `List code definitions in ${path}`, [path]))) {
    return "Definition listing denied";
  }

  const pattern = [
    "^[[:space:]]*(export[[:space:]]+)?(async[[:space:]]+)?(function|class|interface|type|enum|const|let|var)[[:space:]]+[A-Za-z0-9_$]+",
    "^[[:space:]]*(pub[[:space:]]+)?(async[[:space:]]+)?(fn|struct|enum|trait|impl)[[:space:]]+[A-Za-z0-9_]+",
    "^[[:space:]]*(def|class)[[:space:]]+[A-Za-z0-9_]+",
  ].join("|");
  const command = [
    "rg --line-number --hidden --smart-case",
    "--glob \"*.ts\" --glob \"*.tsx\" --glob \"*.js\" --glob \"*.jsx\" --glob \"*.rs\" --glob \"*.py\" --glob \"*.go\" --glob \"*.java\" --glob \"*.cs\"",
    shellQuote(pattern),
    shellQuote(path),
  ].join(" ");
  const output = await runAgentShell(command);
  const result = output.includes("stdout:") ? output : `${output}\n\nNo code definitions matched.`;
  setAgentOutput(result);
  return result;
}

async function semanticSearchAgentFiles(query: string, path: string): Promise<string> {
  if (!query) throw new Error("Enter a semantic search query");
  const keywords = query
    .toLowerCase()
    .split(/[^a-z0-9_.$-]+/i)
    .filter((part) => part.length > 2)
    .slice(0, 6);
  const fallbackQuery = keywords.length ? keywords.join("|") : query;
  const output = await searchAgentFiles(fallbackQuery, path, "");
  const note =
    "Semantic search local fallback: no embedding index is configured yet, so this used a keyword search over the workspace.";
  return `${note}\n\n${output}`;
}

async function searchAgentFiles(query: string, path: string, glob: string): Promise<string> {
  if (!query) throw new Error("Enter a search query");
  if (!(await requirePathPermissions("read", `Search ${path} for ${query}`, [path]))) return "Search denied";

  const escapedQuery = query.replace(/"/g, '\\"');
  const escapedPath = path.replace(/"/g, '\\"');
  const globArg = glob ? ` --glob "${glob.replace(/"/g, '\\"')}"` : "";
  const output = await runAgentShell(`rg --line-number --hidden --smart-case${globArg} "${escapedQuery}" "${escapedPath}"`);
  return output;
}

async function replaceInAgentFile(
  path: string,
  search: string,
  replacement: string,
  all: boolean,
): Promise<string> {
  if (!path) throw new Error("Choose a path to edit");
  if (!search) throw new Error("Choose text to replace");
  if (!(await requirePathPermissions("edit", `Replace text in ${path}`, [path]))) return "Replace denied";

  const result = await invoke<AgentReadResult>("agent_read_path", { path });
  const content = result.content;
  if (content == null) throw new Error(`${path} is not a text file`);
  if (!content.includes(search)) throw new Error(`Text not found in ${path}`);

  const nextContent = all ? content.split(search).join(replacement) : content.replace(search, replacement);
  const writeResult = await invoke<AgentWriteResult>("agent_write_file", {
    request: { path, content: nextContent, create: true },
  });
  const count = all ? content.split(search).length - 1 : 1;
  const output = `Replaced ${count} occurrence${count === 1 ? "" : "s"}\n${formatWriteResult(writeResult)}`;
  setAgentOutput(output);
  appendAgentMessage("tool", output);
  return output;
}

function editEntriesArg(args: ToolArgs): Array<{ oldText: string; newText: string }> {
  const raw = args.edits;
  if (!Array.isArray(raw)) return [];
  return raw
    .map((entry) => {
      if (!entry || typeof entry !== "object") return null;
      const item = entry as Record<string, unknown>;
      const oldText = typeof item.oldText === "string"
        ? item.oldText
        : typeof item.old_text === "string"
          ? item.old_text
          : "";
      const newText = typeof item.newText === "string"
        ? item.newText
        : typeof item.new_text === "string"
          ? item.new_text
          : "";
      return oldText ? { oldText, newText } : null;
    })
    .filter((entry): entry is { oldText: string; newText: string } => Boolean(entry));
}

async function piEditTool(args: ToolArgs): Promise<string> {
  const path = stringArg(args, "path");
  const edits = editEntriesArg(args);
  if (!path) throw new Error("Choose a path to edit");

  if (edits.length === 0) {
    const search = stringArg(args, "search") || stringArg(args, "oldText") || stringArg(args, "old_text");
    const replacement = stringArg(args, "replace") || stringArg(args, "newText") || stringArg(args, "new_text");
    return replaceInAgentFile(path, search, replacement, Boolean(args.all));
  }

  if (!(await requirePathPermissions("edit", `Pi edit ${path}`, [path]))) return "Edit denied";
  const { content } = await readTextAgentFile(path);
  let nextContent = content;
  const missing: string[] = [];

  for (const edit of edits) {
    if (!content.includes(edit.oldText)) {
      missing.push(edit.oldText.slice(0, 80));
      continue;
    }
    nextContent = nextContent.replace(edit.oldText, edit.newText);
  }

  if (missing.length > 0) {
    throw new Error(`Exact edit text not found in ${path}: ${missing.join("; ")}`);
  }
  if (nextContent === content) {
    const output = `No changes needed in ${path}.`;
    setAgentOutput(output);
    appendAgentMessage("tool", output);
    return output;
  }

  const writeResult = await invoke<AgentWriteResult>("agent_write_file", {
    request: { path, content: nextContent, create: true },
  });
  const output = `Pi edit applied ${edits.length} replacement${edits.length === 1 ? "" : "s"}\n${formatWriteResult(writeResult)}`;
  setAgentOutput(output);
  appendAgentMessage("tool", output);
  return output;
}

async function applyAgentDiff(diff: string): Promise<string> {
  if (!diff.trim()) throw new Error("Provide a unified diff to apply");
  if (!(await requestToolPermission("edit", "Apply unified diff to workspace"))) return "Diff apply denied";

  const patchPath = ".localllm-agent.patch";
  await invoke<AgentWriteResult>("agent_write_file", {
    request: { path: patchPath, content: diff, create: true },
  });
  const command =
    "git apply --whitespace=nowarn .localllm-agent.patch; " +
    "if ($LASTEXITCODE -eq 0) { Remove-Item -LiteralPath .localllm-agent.patch -Force }";
  const output = await runAgentShell(command);
  return output;
}

async function copyAgentFile(
  fromPath = (($("agent-path") as HTMLInputElement).value || "").trim(),
  toPath = (($("agent-destination-path") as HTMLInputElement).value || "").trim(),
): Promise<string> {
  if (!fromPath || !toPath) throw new Error("Choose source and destination paths");
  if (!(await requirePathPermissions("copy", `${fromPath} -> ${toPath}`, [fromPath, toPath]))) return "Copy denied";

  const result = await invoke<AgentTransferResult>("agent_copy_path", {
    request: { fromPath, toPath, overwrite: false },
  });
  const output = formatTransferResult("Copy", result);
  setAgentOutput(output);
  appendAgentMessage("tool", output);
  return output;
}

async function moveAgentFile(
  fromPath = (($("agent-path") as HTMLInputElement).value || "").trim(),
  toPath = (($("agent-destination-path") as HTMLInputElement).value || "").trim(),
): Promise<string> {
  if (!fromPath || !toPath) throw new Error("Choose source and destination paths");
  if (!(await requirePathPermissions("move", `${fromPath} -> ${toPath}`, [fromPath, toPath]))) return "Move denied";

  const result = await invoke<AgentTransferResult>("agent_move_path", {
    request: { fromPath, toPath, overwrite: false },
  });
  const output = formatTransferResult("Move", result);
  setAgentOutput(output);
  appendAgentMessage("tool", output);
  return output;
}

async function pasteAgentFile(path = (($("agent-path") as HTMLInputElement).value || "").trim()): Promise<string> {
  if (!path) throw new Error("Choose a path to paste into");
  if (!(await requirePathPermissions("paste", path, [path]))) return "Paste denied";

  let content = "";
  try {
    content = await navigator.clipboard.readText();
  } catch {
    content = ($("agent-file-content") as HTMLTextAreaElement).value;
  }
  if (!content) throw new Error("Clipboard is empty and the content box has no text");

  ($("agent-file-content") as HTMLTextAreaElement).value = content;
  return editAgentFile(path, content);
}

function normalizeUrl(rawUrl: string): string {
  const trimmed = rawUrl.trim();
  if (!trimmed) throw new Error("Enter a URL to browse");
  return /^[a-z][a-z0-9+.-]*:\/\//i.test(trimmed) ? trimmed : `https://${trimmed}`;
}

function duckDuckGoUrl(query: string): string {
  const trimmed = query.trim();
  if (!trimmed) throw new Error("Enter a search query");
  return `https://duckduckgo.com/?q=${encodeURIComponent(trimmed)}`;
}

async function browseAgentUrl(rawUrl = (($("agent-url") as HTMLInputElement).value || "").trim()): Promise<string> {
  const url = normalizeUrl(rawUrl);
  if (!(await requestToolPermission("browse", url))) return "Browse denied";

  ($("agent-url") as HTMLInputElement).value = url;
  ($("agent-browser-frame") as HTMLIFrameElement).src = url;
  openUrl(url).catch(() => undefined);
  const output = `Browsing ${url}`;
  appendAgentMessage("tool", output);
  setAgentStatus("Browser opened");
  return output;
}

async function searchDuckDuckGo(query: string): Promise<string> {
  const url = duckDuckGoUrl(query);
  if (!(await requestToolPermission("browse", `DuckDuckGo search: ${query}`))) return "Search denied";

  ($("agent-url") as HTMLInputElement).value = url;
  ($("agent-browser-frame") as HTMLIFrameElement).src = url;
  openUrl(url).catch(() => undefined);
  const output = `Searching DuckDuckGo for "${query}"`;
  appendAgentMessage("tool", output);
  setAgentStatus("DuckDuckGo search opened");
  return output;
}

// Uses the real curl binary in the Rust backend (never the PowerShell `curl`
// alias), so HTTP fetches and downloads work the same on every platform.
async function httpFetch(
  url: string,
  options: { savePath?: string; timeoutSeconds?: number; extractText?: boolean } = {},
): Promise<CommandOutput> {
  return invoke<CommandOutput>("agent_http_fetch", {
    request: {
      url,
      savePath: options.savePath ?? null,
      timeoutSeconds: options.timeoutSeconds ?? 30,
      extractText: options.extractText ?? false,
    },
  });
}

interface KeylessWebSearchResult {
  title: string;
  url: string;
  snippet: string;
}

type KeylessWebSearchProviderId = "duckduckgo" | "ecosia" | "google" | "mojeek";

function normalizeSearchText(value: string | null | undefined): string {
  return (value ?? "").replace(/\s+/g, " ").trim();
}

function unwrapSearchUrl(href: string, base: string, rejectedHosts: string[] = []): string {
  try {
    const resolved = new URL(href.replace(/&amp;/g, "&"), base);
    const wrapped = resolved.searchParams.get("uddg") || resolved.searchParams.get("url") || resolved.searchParams.get("q");
    const target = wrapped && /^https?:\/\//i.test(wrapped) ? new URL(wrapped) : resolved;
    if (!["http:", "https:"].includes(target.protocol)) return "";
    if (rejectedHosts.some((host) => target.hostname === host || target.hostname.endsWith(`.${host}`))) return "";
    return target.href;
  } catch {
    return "";
  }
}

function parseDuckDuckGoResults(html: string): KeylessWebSearchResult[] {
  const doc = new DOMParser().parseFromString(html, "text/html");
  return Array.from(doc.querySelectorAll(".result"))
    .map((block) => {
      const link = block.querySelector<HTMLAnchorElement>("a.result__a, a.result-link");
      return {
        title: normalizeSearchText(link?.textContent),
        url: unwrapSearchUrl(link?.getAttribute("href") ?? "", "https://html.duckduckgo.com/", ["duckduckgo.com"]),
        snippet: normalizeSearchText(block.querySelector(".result__snippet, .result-snippet")?.textContent),
      };
    })
    .filter((result) => result.title && result.url)
    .slice(0, 10);
}

function parseEcosiaResults(html: string): KeylessWebSearchResult[] {
  const doc = new DOMParser().parseFromString(html, "text/html");
  return Array.from(doc.querySelectorAll('article[data-test-id="organic-result"]'))
    .map((block) => {
      const heading = block.querySelector('[data-test-id="result-title"]');
      const link = heading?.closest("a");
      return {
        title: normalizeSearchText(heading?.textContent),
        url: unwrapSearchUrl(link?.getAttribute("href") ?? "", "https://www.ecosia.org/", ["ecosia.org"]),
        snippet: normalizeSearchText(
          block.querySelector('[data-test-id="web-result-description"], [data-test-id="result-description"]')
            ?.textContent,
        ),
      };
    })
    .filter((result) => result.title && result.url)
    .slice(0, 10);
}

function parseGoogleResults(html: string): KeylessWebSearchResult[] {
  const doc = new DOMParser().parseFromString(html, "text/html");
  return Array.from(doc.querySelectorAll("h3"))
    .map((heading) => {
      const link = heading.closest("a");
      const container = heading.closest(".tF2Cxc, .MjjYud, .Gx5Zad") ?? link?.parentElement;
      return {
        title: normalizeSearchText(heading.textContent),
        url: unwrapSearchUrl(link?.getAttribute("href") ?? "", "https://www.google.com/", ["google.com"]),
        snippet: normalizeSearchText(
          container?.querySelector(".VwiC3b, .IsZvec, .BNeawe.s3v9rd, [data-sncf='1']")?.textContent,
        ),
      };
    })
    .filter((result) => result.title && result.url)
    .slice(0, 10);
}

function parseMojeekResults(html: string): KeylessWebSearchResult[] {
  const doc = new DOMParser().parseFromString(html, "text/html");
  return Array.from(doc.querySelectorAll("ul.results-standard > li"))
    .map((block) => {
      const link = block.querySelector<HTMLAnchorElement>("h2 a.title, a.title");
      return {
        title: normalizeSearchText(link?.textContent),
        url: unwrapSearchUrl(link?.getAttribute("href") ?? "", "https://www.mojeek.de/", ["mojeek.de"]),
        snippet: normalizeSearchText(block.querySelector("p.s")?.textContent),
      };
    })
    .filter((result) => result.title && result.url)
    .slice(0, 10);
}

const keylessWebSearchProviders: Array<{
  id: KeylessWebSearchProviderId;
  label: string;
  endpoint: (query: string) => string;
  parse: (html: string) => KeylessWebSearchResult[];
}> = [
  {
    id: "duckduckgo",
    label: "DuckDuckGo",
    endpoint: (query) => `https://html.duckduckgo.com/html/?q=${encodeURIComponent(query)}`,
    parse: parseDuckDuckGoResults,
  },
  {
    id: "ecosia",
    label: "Ecosia",
    endpoint: (query) => `https://www.ecosia.org/search?q=${encodeURIComponent(query)}`,
    parse: parseEcosiaResults,
  },
  {
    id: "google",
    label: "Google",
    endpoint: (query) =>
      `https://www.google.com/search?q=${encodeURIComponent(query)}&num=10&hl=en&udm=14&pws=0`,
    parse: parseGoogleResults,
  },
  {
    id: "mojeek",
    label: "Mojeek",
    endpoint: (query) =>
      `https://www.mojeek.de/search?q=${encodeURIComponent(query)}&t=10&arc=none&lang=en`,
    parse: parseMojeekResults,
  },
];

// Mirrors OMP's public, credential-free search providers. "auto" falls
// through on HTTP failures, bot challenges, and markup with no parsed results.
async function webSearchKeyless(query: string, requestedProvider = "auto"): Promise<string> {
  if (!query) throw new Error("Enter a search query");
  if (!(await requestToolPermission("browse", `Keyless web search: ${query}`))) return "Search denied";

  const previewUrl = duckDuckGoUrl(query);
  ($("agent-url") as HTMLInputElement).value = previewUrl;

  const normalizedProvider = requestedProvider.trim().toLowerCase();
  const providers =
    normalizedProvider === "auto"
      ? keylessWebSearchProviders
      : keylessWebSearchProviders.filter((provider) => provider.id === normalizedProvider);
  if (providers.length === 0) {
    throw new Error("Unknown web search provider. Use auto, duckduckgo, ecosia, google, or mojeek.");
  }

  const failures: string[] = [];
  for (const provider of providers) {
    setAgentStatus(`Searching ${provider.label} without an API key`);
    try {
      const result = await httpFetch(provider.endpoint(query), { timeoutSeconds: 30 });
      if (!result.success) {
        failures.push(`${provider.label}: ${result.stderr.trim() || `HTTP ${result.statusCode ?? "error"}`}`);
        continue;
      }
      const results = provider.parse(result.stdout);
      if (results.length === 0) {
        failures.push(`${provider.label}: no parseable results`);
        continue;
      }
      const output = `${provider.label} results for "${query}" (no API key):\n\n${results
        .map((item, index) => `${index + 1}. ${item.title}\n   ${item.url}${item.snippet ? `\n   ${item.snippet}` : ""}`)
        .join("\n\n")}`;
      setAgentOutput(output);
      appendAgentMessage("tool", output);
      setAgentStatus(`${provider.label} search complete`);
      return output;
    } catch (error) {
      failures.push(`${provider.label}: ${String(error)}`);
    }
  }

  const output =
    `No keyless provider returned parseable results for "${query}". The DuckDuckGo URL is available in the Browser field.\n\n` +
    failures.map((failure) => `- ${failure}`).join("\n");
  setAgentOutput(output);
  appendAgentMessage("tool", output);
  setAgentStatus("Keyless web search exhausted all providers");
  return output;
}

async function webFetchUrl(rawUrl: string): Promise<string> {
  const url = normalizeUrl(rawUrl);
  if (!(await requestToolPermission("browse", `Fetch ${url}`))) return "Fetch denied";

  const result = await httpFetch(url, { timeoutSeconds: 30, extractText: true });
  const body = result.stdout.trim();
  const limit = 12000;
  const output = body
    ? `Fetched ${url} (${body.length.toLocaleString()} chars)\n\n${body.slice(0, limit)}${
        body.length > limit ? "\n...[truncated]" : ""
      }`
    : `Fetch returned no body (exit ${result.statusCode ?? "unknown"}).${result.stderr.trim() ? `\n${result.stderr.trim()}` : ""}`;
  setAgentOutput(output);
  appendAgentMessage("tool", output);
  return output;
}

async function downloadFileTool(rawUrl: string, path: string): Promise<string> {
  if (!path) throw new Error("Provide a destination path");
  const url = normalizeUrl(rawUrl);
  if (!(await requirePathPermissions("edit", `Download ${url} -> ${path}`, [path]))) return "Download denied";

  const result = await httpFetch(url, { savePath: path, timeoutSeconds: 120 });
  if (!result.success) {
    const output = `Download failed (exit ${result.statusCode ?? "unknown"}).${
      result.stderr.trim() ? `\n${result.stderr.trim()}` : result.stdout.trim() ? `\n${result.stdout.trim()}` : ""
    }`;
    setAgentOutput(output);
    appendAgentMessage("tool", output);
    return output;
  }
  const info = await pathInfo(path);
  const output = `Downloaded ${url}\n${formatPathInfo(info)}`;
  setAgentOutput(output);
  appendAgentMessage("tool", output);
  return output;
}

async function browserAction(args: ToolArgs): Promise<string> {
  const action = (stringArg(args, "action") || "open").toLowerCase();
  const url = stringArg(args, "url");
  const coordinate = [stringArg(args, "coordinate"), stringArg(args, "x"), stringArg(args, "y")]
    .filter(Boolean)
    .join(", ");
  const text = stringArg(args, "text");

  if (["open", "launch", "navigate", "go_to", "visit"].includes(action)) {
    return browseAgentUrl(url);
  }
  if (action === "search") {
    return searchDuckDuckGo(text || url);
  }
  if (action === "close") {
    ($("agent-browser-frame") as HTMLIFrameElement).src = "about:blank";
    const output = "Browser preview closed";
    appendAgentMessage("tool", output);
    setAgentStatus(output);
    return output;
  }

  const output = [
    `browser_action "${action}" is not fully automated in LocalLLM yet.`,
    "Supported local actions: open, navigate, search, close.",
    coordinate ? `Requested coordinate: ${coordinate}` : "",
    text ? `Requested text: ${text}` : "",
    url ? `Requested URL: ${url}` : "",
  ]
    .filter(Boolean)
    .join("\n");
  setAgentOutput(output);
  appendAgentMessage("tool", output);
  return output;
}

async function runAgentShell(command: string): Promise<string> {
  if (!command) throw new Error("Enter a shell command");
  if (!(await requestToolPermission("shell", command))) return "Shell command denied";
  setAgentCliStatus("CLI: running");

  try {
    const result = await invoke<CommandOutput>("agent_run_shell", {
      request: { command, timeoutSeconds: 60 },
    });
    const output = [
      `pi bash> ${result.command}`,
      `status: ${result.statusCode ?? "unknown"} ${result.success ? "ok" : "failed"}`,
      result.stdout.trim() ? `stdout:\n${result.stdout.trim()}` : "",
      result.stderr.trim() ? `stderr:\n${result.stderr.trim()}` : "",
    ]
      .filter(Boolean)
      .join("\n\n");
    setAgentOutput(output);
    appendAgentMessage("tool", output);
    setAgentCliStatus(result.success ? "CLI: ok" : "CLI: failed");
    return output;
  } finally {
    window.setTimeout(() => setAgentCliStatus(), 1800);
  }
}

async function refreshPiAgentStatus(): Promise<AgentPiStatus | null> {
  if (!piAgentDiscovery) {
    piAgentDiscovery = invoke<AgentPiStatus>("discover_pi_agent")
      .then((status) => {
        piAgentStatus = status;
        setAgentCliStatus();
        return status;
      })
      .catch((error) => {
        piAgentStatus = { available: false, command: "", version: String(error), checked: [] };
        setAgentCliStatus("CLI: local fallback");
        return null;
      })
      .finally(() => {
        piAgentDiscovery = null;
      });
  }
  return piAgentDiscovery;
}

function composePiCodingPrompt(userPrompt: string): string {
  const profile = activeProfile();
  return [
    `You are running inside LocalLLM through the real earendil-works/pi coding agent CLI.`,
    `Profile: ${profile.name}`,
    profile.instructions,
    agentSystemInstructions ? `System instructions:\n${agentSystemInstructions}` : "",
    agentTaskInstructions ? `Reusable task instructions:\n${agentTaskInstructions}` : "",
    `Execution mode: ${agentExecutionMode}.`,
    agentGoal ? `Current goal:\n${agentGoal}` : "",
    contextSummary ? `Context summary:\n${contextSummary}` : "",
    conversationSnapshot() ? `Recent LocalLLM transcript:\n${conversationSnapshot()}` : "",
    "Answer direct questions completely. In Edit mode, do not stop after describing a plan: execute every safe, in-scope step, verify the result, and only then give a final response.",
    "Continue autonomously through ordinary recoverable errors and never ask the user to say “continue”. Ask a follow-up only when required information or authority is genuinely missing.",
    "Use Pi's native tools for workspace inspection and edits. Prefer small, verifiable changes and report commands/tests you ran.",
    `User request:\n${userPrompt}`,
  ]
    .filter(Boolean)
    .join("\n\n");
}

function piThinkingArgument(): string {
  return agentThinkingLevel === "disabled" ? "off" : agentThinkingLevel;
}

function piResponseNeedsContinuation(text: string): boolean {
  if (agentExecutionMode === "plan") return false;
  const normalized = text.replace(/\s+/g, " ").trim();
  if (!normalized) return true;
  if ((text.match(/```/g)?.length ?? 0) % 2 !== 0) return true;
  return (
    /\b(?:tell|ask) me to continue\b/i.test(normalized) ||
    /\b(?:i(?:'ll| will)|let me) continue\b/i.test(normalized) ||
    /\b(?:paused|stopped|halting)\b.{0,80}\b(?:limit|unfinished|remaining|continue)\b/i.test(normalized) ||
    /\b(?:remaining|pending) (?:work|steps?|tasks?)\b.{0,120}\b(?:need|will|next|continue)\b/i.test(normalized) ||
    /\b(?:next|now) i (?:need|will|should|must) (?:to )?(?:run|edit|write|implement|verify|test|inspect)\b[^.!?]*$/i.test(
      normalized,
    ) ||
    /(?:\.{3}|…)$/.test(normalized)
  );
}

async function runPiCodingAgent(prompt: string): Promise<string> {
  if (!prompt.trim()) throw new Error("Enter a Pi coding prompt");
  if (!(await requestToolPermission("shell", `Run Pi coding agent: ${prompt}`))) {
    return "Pi coding agent denied";
  }

  setAgentCliStatus("CLI: pi running");
  setAgentStatus("Running Pi coding agent");
  showAgentThinking("Pi is working");
  let result: CommandOutput | null = null;
  const segments: string[] = [];
  const maxRecoveries = Math.min(3, Math.max(1, agentRetryCount + 1));
  let nextPrompt = composePiCodingPrompt(prompt);
  let continueSession = false;
  try {
    for (let attempt = 0; attempt <= maxRecoveries; attempt += 1) {
      result = await invoke<CommandOutput>("agent_run_pi", {
        request: {
          prompt: nextPrompt,
          extraArgs: [
            "--thinking",
            piThinkingArgument(),
            ...(continueSession ? ["--continue"] : []),
          ],
          timeoutSeconds: agentTimeoutSeconds,
          temperature: agentTemperature,
        },
      });
      const body = result.stdout.trim() || (result.success ? "" : result.stderr.trim());
      if (!result.success) {
        if (attempt === maxRecoveries) break;
        setAgentStatus(`Pi failed · retry ${attempt + 1}/${maxRecoveries}`);
        nextPrompt = composePiCodingPrompt(prompt);
        continueSession = false;
        continue;
      }
      if (body) segments.push(body);
      if (!body || piResponseNeedsContinuation(body)) {
        if (attempt === maxRecoveries) break;
        continueSession = true;
        nextPrompt = body
          ? "Continue the same task now. Do not repeat the plan or prior summary. Execute and verify every remaining step, then return the completed result."
          : "Your previous response was empty. Continue the same task now, answer or execute it fully, verify the result, and return a non-empty final response.";
        setAgentStatus(`Pi is continuing automatically · pass ${attempt + 2}/${maxRecoveries + 1}`);
        showAgentThinking("Pi is continuing");
        continue;
      }
      break;
    }
  } finally {
    hideAgentThinking();
  }
  if (!result) throw new Error("Pi did not return a result");
  const body = segments.join("\n\n--- continued ---\n\n").trim();
  if (result.success && !body) {
    setAgentCliStatus("CLI: pi empty response");
    throw new Error("Pi returned an empty response after automatic recovery attempts.");
  }
  const output = [
    `pi> ${result.command}`,
    `status: ${result.statusCode ?? "unknown"} ${result.success ? "ok" : "failed"}`,
    body || result.stderr.trim() || "(no output)",
  ].join("\n\n");
  setAgentOutput(output);
  appendAgentMessage(result.success ? "agent" : "tool", body || result.stderr.trim() || "(no output)");
  setAgentCliStatus(result.success ? defaultAgentCliStatus() : "CLI: pi failed");
  if (!result.success) {
    throw new Error(body || result.stderr.trim() || "Pi failed without an error message");
  }
  return body;
}

async function piCliTool(command: string): Promise<string> {
  const trimmed = command.trim();
  if (!trimmed || trimmed === "status") {
    const root = await invoke<string>("agent_workspace_root").catch(() => "unavailable");
    const status = piAgentStatus ?? (await refreshPiAgentStatus());
    const output = [
      "Pi coding agent CLI",
      `available: ${status?.available ? "yes" : "no"}`,
      status?.command ? `command: ${status.command}` : "",
      status?.version ? `version: ${status.version}` : "",
      `mode: ${agentExecutionMode}`,
      `profile: ${activeProfile().name}`,
      `workspace: ${root}`,
      "tools: Pi native read, write, edit, bash, grep, find, ls",
      status?.available
        ? "runtime: earendil-works/pi CLI with LocalLLM provider injection when llama.cpp is running"
        : "runtime: local fallback loop until Pi is installed",
      status?.checked?.length ? `checked: ${status.checked.join(", ")}` : "",
    ]
      .filter(Boolean)
      .join("\n");
    setAgentOutput(output);
    appendAgentMessage("tool", output);
    setAgentCliStatus();
    return output;
  }

  return runAgentShell(trimmed);
}

function todoTextFromValue(value: unknown): string {
  if (typeof value === "string") return value.trim();
  if (value && typeof value === "object") {
    const item = value as Record<string, unknown>;
    const text = item.text ?? item.content ?? item.task ?? item.description;
    const done = item.done === true || item.status === "completed" || item.status === "done";
    return `${done ? "[x]" : "[ ]"} ${typeof text === "string" ? text : JSON.stringify(value)}`;
  }
  return String(value ?? "").trim();
}

async function updateTodoList(args: ToolArgs): Promise<string> {
  const rawTodos = args.todos ?? args.todoList ?? args.items ?? args.list;
  const markdown = stringArg(args, "text") || stringArg(args, "todo_list") || stringArg(args, "todoList");
  if (!(await requestToolPermission("todo", "Replace todo list"))) return "Todo update denied";

  const nextTodos: AgentTodo[] = [];
  if (Array.isArray(rawTodos)) {
    for (const value of rawTodos) {
      const text = todoTextFromValue(value);
      if (!text) continue;
      const done = /^\s*\[[xX]\]/.test(text);
      nextTodos.push({
        id: crypto.randomUUID(),
        text: text.replace(/^\s*\[[ xX]\]\s*/, ""),
        done,
        createdAt: Date.now(),
      });
    }
  } else {
    const lines = markdown
      .split(/\r?\n/)
      .map((line) => line.trim())
      .filter(Boolean);
    for (const line of lines) {
      const match = line.match(/^[-*]?\s*\[([ xX])\]\s*(.+)$/);
      nextTodos.push({
        id: crypto.randomUUID(),
        text: (match?.[2] ?? line.replace(/^[-*]\s*/, "")).trim(),
        done: match?.[1]?.toLowerCase() === "x",
        createdAt: Date.now(),
      });
    }
  }

  if (nextTodos.length === 0) throw new Error("Provide todos as an array or markdown checklist");
  todos = nextTodos;
  saveTodos();
  renderTodos();
  const output = `Todo list replaced with ${todos.length} item${todos.length === 1 ? "" : "s"}.`;
  appendAgentMessage("tool", output);
  return output;
}

async function addTodo(text: string): Promise<string> {
  if (!text) throw new Error("Enter a todo item");
  if (!(await requestToolPermission("todo", text))) return "Todo update denied";

  todos = [...todos, { id: crypto.randomUUID(), text, done: false, createdAt: Date.now() }];
  saveTodos();
  renderTodos();
  const output = `Todo added: ${text}`;
  appendAgentMessage("tool", output);
  return output;
}

async function switchAgentModeTool(mode: string): Promise<string> {
  if (!mode) throw new Error("Choose a mode to switch to");
  if (!(await requestToolPermission("todo", `Switch mode to ${mode}`))) return "Mode switch denied";
  setActiveAgentProfile(mode as AgentMode);
  const output = `Switched to ${activeProfile().name} mode`;
  appendAgentMessage("tool", output);
  return output;
}

async function createNewTaskTool(mode: string, message: string): Promise<string> {
  if (!(await requestToolPermission("subagent", `New task ${mode || "current"}: ${message || "empty"}`))) {
    return "New task denied";
  }

  if (message) {
    const delegated = await delegateToCodingAgent(mode || activeAgentProfile, message);
    const output = `New ${profileById((mode || activeAgentProfile) as AgentMode).name} subtask\n${delegated}`;
    appendAgentMessage("tool", output);
    return output;
  }

  startNewAgentTask();
  if (mode) setActiveAgentProfile(mode as AgentMode);
  const output = `New task started in ${activeProfile().name} mode`;
  appendAgentMessage("tool", output);
  return output;
}

async function askFollowupQuestion(question: string, suggestionsValue: unknown): Promise<string> {
  if (!question) throw new Error("Provide a follow-up question");
  if (!(await requestToolPermission("todo", `Ask follow-up: ${question}`))) return "Follow-up denied";

  const suggestions = Array.isArray(suggestionsValue)
    ? suggestionsValue
        .map((value) => {
          if (typeof value === "string") return value;
          if (value && typeof value === "object") {
            const item = value as Record<string, unknown>;
            return typeof item.text === "string" ? item.text : JSON.stringify(value);
          }
          return String(value ?? "");
        })
        .filter(Boolean)
        .slice(0, 3)
    : [];
  const output = [`Follow-up question:\n${question}`, suggestions.length ? `Suggestions:\n${suggestions.map((item, index) => `${index + 1}. ${item}`).join("\n")}` : ""]
    .filter(Boolean)
    .join("\n\n");
  appendAgentMessage("agent", output);
  setAgentStatus("Waiting for user follow-up");
  return output;
}

async function attemptCompletion(result: string, command: string): Promise<string> {
  if (!result) throw new Error("Provide a completion result");
  if (!(await requestToolPermission("todo", "Attempt completion"))) return "Completion denied";

  let commandOutput = "";
  if (command) {
    commandOutput = await runAgentShell(command);
  }
  const output = [`Task completion attempt:\n${result}`, commandOutput ? `Verification command:\n${commandOutput}` : ""]
    .filter(Boolean)
    .join("\n\n");
  appendAgentMessage("agent", output);
  setAgentStatus("Agent task complete");
  return output;
}

async function refreshDiscoveredSkills(): Promise<void> {
  try {
    discoveredSkills = await invoke<AgentSkillFile[]>("agent_list_skills");
    renderSkills();
  } catch {
    discoveredSkills = [];
  }
}

// A skill with no modes is generic (all agents); otherwise it only applies when
// the current agent/profile is listed in its frontmatter modes.
function skillAppliesToCurrentAgent(skill: AgentSkillFile): boolean {
  const modes = (skill.modes || "").trim().toLowerCase();
  if (!modes) return true;
  const list = modes.split(/[\s,]+/).filter(Boolean);
  return list.includes(activeAgentProfile) || list.includes(agentMode) || list.includes("all");
}

function applicableDiscoveredSkills(): AgentSkillFile[] {
  return discoveredSkills.filter(skillAppliesToCurrentAgent);
}

// Lists discovered SKILL.md skills (metadata only) so the model can choose to
// load one on demand with the skill tool, following the Agent Skills flow.
function discoveredSkillsPromptSection(): string {
  const applicable = applicableDiscoveredSkills();
  if (applicable.length === 0) return "";
  const loadedNames = new Set(
    skills.filter((skill) => skill.enabled).map((skill) => skill.name.toLowerCase()),
  );
  const lines = applicable
    .filter((skill) => !loadedNames.has(skill.name.toLowerCase()))
    .map((skill) => `- ${skill.name}: ${skill.description || "(no description)"}`);
  if (lines.length === 0) return "";
  return [
    "Available skills (load full instructions only when the request clearly matches one):",
    ...lines,
    'To load a skill: {"tool":"skill","args":{"name":"skill-name"}}',
  ].join("\n");
}

async function loadSkill(name: string, instructions: string): Promise<string> {
  if (!name) throw new Error("Enter a skill name");
  if (!(await requestToolPermission("skill", name))) return "Skill load denied";

  // When no instructions are supplied, try to load a discovered SKILL.md from a
  // workspace skills directory (Agent Skills SKILL.md format).
  let resolvedInstructions = instructions;
  if (!resolvedInstructions) {
    const discovered = discoveredSkills.find((skill) => skill.name.toLowerCase() === name.toLowerCase());
    if (discovered) {
      try {
        const result = await invoke<AgentReadResult>("agent_read_path", { path: discovered.path });
        if (result.content) resolvedInstructions = result.content;
      } catch {
        // Fall back to the generic instruction below.
      }
    }
  }

  const existing = skills.find((skill) => skill.name.toLowerCase() === name.toLowerCase());
  if (existing) {
    skills = skills.map((skill) =>
      skill.id === existing.id
        ? { ...skill, enabled: true, instructions: resolvedInstructions || skill.instructions }
        : skill,
    );
  } else {
    skills = [
      ...skills,
      {
        id: crypto.randomUUID(),
        name,
        instructions: resolvedInstructions || "Apply this skill's specialized instructions to the current goal.",
        enabled: true,
        createdAt: Date.now(),
      },
    ];
  }
  saveSkills();
  renderSkills();
  const output = `Skill loaded: ${name}`;
  appendAgentMessage("tool", output);
  return output;
}

async function startSubagent(name: string, role: string): Promise<string> {
  if (!name) throw new Error("Enter a subagent name");
  if (!(await requestToolPermission("subagent", name))) return "Subagent start denied";

  const existing = subagents.find((agent) => agent.name.toLowerCase() === name.toLowerCase());
  if (existing) {
    subagents = subagents.map((agent) =>
      agent.id === existing.id ? { ...agent, enabled: true, role: role || agent.role } : agent,
    );
  } else {
    subagents = [
      ...subagents,
      {
        id: crypto.randomUUID(),
        name,
        role: role || "Assist with the current goal using a focused specialist perspective.",
        enabled: true,
        createdAt: Date.now(),
      },
    ];
  }
  saveSubagents();
  renderSubagents();
  const output = `Subagent active: ${name}`;
  appendAgentMessage("tool", output);
  return output;
}

async function compactAgentContext(): Promise<string> {
  const messages = Array.from(agentMessageContainer().querySelectorAll<HTMLElement>(".agent-message-bubble"))
    .map((bubble) => bubble.textContent?.trim())
    .filter(Boolean) as string[];
  const enabledSkills = skills.filter((skill) => skill.enabled).map((skill) => skill.name).join(", ") || "none";
  const enabledSubagents =
    subagents.filter((agent) => agent.enabled).map((agent) => `${agent.name}: ${agent.role}`).join("; ") || "none";

  contextSummary = [
    `Mode: ${agentMode}`,
    `Goal: ${agentGoal || "not set"}`,
    `Skills: ${enabledSkills}`,
    `Subagents: ${enabledSubagents}`,
    `Todos: ${todos.map((todo) => `${todo.done ? "[x]" : "[ ]"} ${todo.text}`).join("; ") || "none"}`,
    `Recent transcript: ${messages.slice(-8).join(" | ") || "empty"}`,
  ].join("\n");
  saveAgentContext();
  syncContextUi();
  appendAgentMessage("agent", `Context compacted:\n${contextSummary}`);
  return contextSummary;
}

function parseAgentCommand(input: string): string[] {
  return (input.match(/"[^"]*"|'[^']*'|\S+/g) ?? []).map((part) =>
    part.replace(/^(['"])(.*)\1$/, "$2"),
  );
}

function stripCommandPrefix(input: string, verb: string): string {
  return input.trim().slice(verb.length).trim();
}

function splitPathAndContent(input: string): { path: string; content: string } {
  const trimmed = input.trim();
  const quoted = trimmed.match(/^"([^"]+)"\s+([\s\S]*)$/) ?? trimmed.match(/^'([^']+)'\s+([\s\S]*)$/);
  if (quoted) return { path: quoted[1].trim(), content: quoted[2] ?? "" };

  const [path = "", ...contentParts] = parseAgentCommand(trimmed);
  return { path, content: contentParts.join(" ") };
}

function splitLineEditCommand(input: string): { path: string; start: number | null; end: number | null; content: string } {
  const [path = "", startRaw = "", endRaw = "", ...contentParts] = parseAgentCommand(input);
  const start = Number(startRaw);
  const end = Number(endRaw);
  return {
    path,
    start: Number.isFinite(start) ? start : null,
    end: Number.isFinite(end) ? end : null,
    content: contentParts.join(" "),
  };
}

function looksLikeFilePath(value: string): boolean {
  return (
    value.includes("/") ||
    value.includes("\\") ||
    value.startsWith(".") ||
    /^[A-Za-z]:/.test(value) ||
    /\.[A-Za-z0-9]{1,12}$/.test(value)
  );
}

function isExplicitFileWriteCommand(verb: string, first: string, commandBody: string): boolean {
  if (!["edit", "write", "write_to_file", "create"].includes(verb)) return false;
  if (first.toLowerCase() === "file") return true;
  const quotedPath = commandBody.trim().match(/^"[^"]+"\s+[\s\S]+$/) ?? commandBody.trim().match(/^'[^']+'\s+[\s\S]+$/);
  return Boolean(quotedPath) || looksLikeFilePath(first);
}

function parseToolCall(input: string): AgentToolCall | null {
  const trimmed = input.trim();
  if (!trimmed.startsWith("{")) return null;

  const parsed = JSON.parse(trimmed) as Partial<AgentToolCall> & { name?: string; arguments?: ToolArgs };
  const tool = parsed.tool ?? parsed.name;
  if (!tool) throw new Error("Tool call JSON needs a tool or name field");
  return {
    tool,
    args: parsed.args ?? parsed.arguments ?? {},
  };
}

function missingRequiredToolArgs(toolName: string, args: ToolArgs): string[] {
  const value = (name: string) => (typeof args[name] === "string" ? (args[name] as string).trim() : "");
  const required: Record<string, string[]> = {
    edit_file: ["path", "content"],
    write_to_file: ["path", "content"],
    append_to_file: ["path", "content"],
    insert_content: ["path", "content"],
    go_to_line: ["path"],
    search_lines: ["path", "query"],
    edit_lines: ["path", "content"],
    replace_in_file: ["path", "search", "replace"],
    delete_file: ["path"],
    copy_file: ["fromPath", "toPath"],
    move_file: ["fromPath", "toPath"],
    shell: ["command"],
    execute_command: ["command"],
    web_fetch: ["url"],
    download_file: ["url", "path"],
    web_search: ["query"],
  };
  return (required[toolName] ?? []).filter((name) => !value(name));
}

function toolSchemaHint(toolName: string): string {
  const hints: Record<string, string> = {
    edit_file: `Use {"tool":"edit_file","args":{"path":"index.html","content":"complete file contents"}}`,
    edit: `Use {"tool":"edit","args":{"path":"src/file.ts","edits":[{"oldText":"exact old text","newText":"replacement text"}]}}`,
    write_to_file: `Use {"tool":"write_to_file","args":{"path":"index.html","content":"complete file contents"}}. There is no artificial output-token cap for agent writes.`,
    append_to_file: `Use {"tool":"append_to_file","args":{"path":"index.html","content":"text to add at the end"}}`,
    insert_content: `Use {"tool":"insert_content","args":{"path":"index.html","content":"text to insert","line":12}}`,
    find_files: `Use {"tool":"find_files","args":{"path":"src","query":"agent","glob":"*.ts"}}`,
    go_to_line: `Use {"tool":"go_to_line","args":{"path":"src/agent.ts","line":120,"context":20}}`,
    search_lines: `Use {"tool":"search_lines","args":{"path":"src/agent.ts","query":"write_to_file"}}`,
    edit_lines: `Use {"tool":"edit_lines","args":{"path":"src/agent.ts","start":10,"end":12,"content":"replacement lines"}}`,
    replace_in_file: `Use {"tool":"replace_in_file","args":{"path":"file","search":"old","replace":"new","all":true}}`,
    delete_file: `Use {"tool":"delete_file","args":{"path":"file"}}`,
    copy_file: `Use {"tool":"copy_file","args":{"fromPath":"source","toPath":"destination"}}`,
    move_file: `Use {"tool":"move_file","args":{"fromPath":"source","toPath":"destination"}}`,
    shell: `Use {"tool":"shell","args":{"command":"npm run build"}}`,
    execute_command: `Use {"tool":"execute_command","args":{"command":"npm run build"}}`,
    web_fetch: `Use {"tool":"web_fetch","args":{"url":"https://example.com"}}`,
    download_file: `Use {"tool":"download_file","args":{"url":"https://example.com/file.zip","path":"assets/file.zip"}}`,
    web_search: `Use {"tool":"web_search","args":{"query":"local llm tools"}}`,
    pi_cli: `Use {"tool":"pi_cli","args":{"command":"status"}} or {"tool":"pi_cli","args":{"command":"npm run build"}}`,
  };
  return hints[toolName] ?? "Call the tool with the required argument object.";
}

async function executeToolCall(call: AgentToolCall): Promise<string> {
  const tool = localTools.find((candidate) => candidate.name === call.tool);
  if (!tool) {
    const available = localTools.map((candidate) => candidate.name).join(", ");
    throw new Error(`Unknown tool "${call.tool}". Available tools: ${available}`);
  }
  if (!agentProfileAllowsTool(tool.name)) {
    const message = `${activeProfile().name} is not allowed to use ${tool.name}. Switch agents or delegate to Coder/Orchestrator.`;
    appendAgentMessage("tool", message);
    return message;
  }
  if (toolBlockedByPlanMode(tool.permission)) {
    const message = `Plan mode is on, so ${tool.name} (a workspace change) is blocked. Finish investigating and return a step-by-step plan, or switch to Edit mode to apply changes.`;
    appendAgentMessage("tool", message);
    return message;
  }

  const args = call.args ?? {};
  appendAgentToolCall(tool.name, args);
  const missing = missingRequiredToolArgs(tool.name, args);
  if (missing.length > 0) {
    const output = `Tool call rejected: ${tool.name} is missing required args: ${missing.join(", ")}.\n${toolSchemaHint(tool.name)}`;
    setAgentOutput(output);
    appendAgentMessage("tool", output);
    return output;
  }

  try {
    return await tool.run(args);
  } catch (error) {
    const output = `Tool error in ${tool.name}: ${String(error)}\n${toolSchemaHint(tool.name)}`;
    setAgentOutput(output);
    appendAgentMessage("tool", output);
    return output;
  }
}

async function callLocalMcpServer(serverNameOrUrl: string, method: string, params: ToolArgs): Promise<string> {
  const server =
    mcpServers.find((candidate) => candidate.name === serverNameOrUrl || candidate.url === serverNameOrUrl) ??
    mcpServers[0];
  if (!server) throw new Error("Add a local MCP server first");
  if (!isLocalHttpUrl(server.url)) throw new Error("MCP support is local-only for now");
  if (!(await requestToolPermission("browse", `MCP ${server.name}: ${method}`))) return "MCP call denied";

  const response = await fetch(server.url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ jsonrpc: "2.0", id: Date.now(), method, params }),
  });
  const text = await response.text();
  const output = `MCP ${server.name} ${response.status}\n${text}`;
  setAgentOutput(output);
  appendAgentMessage("tool", output);
  return output;
}

function configuredServerUrl(config: AppConfig): string {
  const host = config.server.host || "127.0.0.1";
  const port = config.server.port || 8080;
  return `http://${host}:${port}`;
}

async function localServerBaseUrl(): Promise<string> {
  const status = await invoke<ServerStatus>("server_status");
  if (status.running && status.url) return status.url.replace(/\/+$/, "");

  const config = await invoke<AppConfig>("load_config");
  return configuredServerUrl(config).replace(/\/+$/, "");
}

async function askLocalServer(prompt: string): Promise<string> {
  if (!prompt) throw new Error("Enter a prompt for the local server");
  const output = await postLocalCompletion(chatPrompt(prompt));
  appendAgentMessage("agent", output);
  return output;
}

function chatPrompt(prompt: string): string {
  return [
    agentGoal ? `Current goal:\n${agentGoal}` : "",
    contextSummary ? `Context summary:\n${contextSummary}` : "",
    `User:\n${prompt}`,
    "Assistant:",
  ]
    .filter(Boolean)
    .join("\n\n");
}

const unlimitedCompletionBudget = -1;
const fallbackCompletionBudget = 32768;

// The agent's own model call. This is the agent reasoning, not an external
// action, so it is never gated behind a tool-permission prompt. Risky actions
// (edit, shell, browse, ...) are still gated where they actually happen.
async function postLocalCompletion(
  prompt: string,
  label = "Processing",
  jsonSchema?: unknown,
): Promise<string> {
  const baseUrl = await localServerBaseUrl();
  setAgentStatus(`Asking local server at ${baseUrl}`);
  showAgentThinking(label);

  try {
    let response: Response;
    const postCompletion = (nPredict: number, withSchema: boolean) =>
      fetch(`${baseUrl}/completion`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        signal: agentAbort?.signal,
        body: JSON.stringify({
          ...agentThinkingRequestFields(),
          prompt,
          n_predict: nPredict,
          temperature: agentTemperature,
          stream: false,
          ...(withSchema && jsonSchema ? { json_schema: jsonSchema } : {}),
        }),
      });

    try {
      response = await postCompletion(unlimitedCompletionBudget, true);
      // 400/422 can mean the budget OR that the server rejects json_schema; retry
      // without the schema so unsupported servers degrade gracefully.
      if (!response.ok && [400, 422].includes(response.status)) {
        response = await postCompletion(fallbackCompletionBudget, false);
      }
    } catch (error) {
      if (agentAbort?.signal.aborted) throw new DOMException("Agent run cancelled", "AbortError");
      throw new Error(
        `Could not reach the local server at ${baseUrl}. Start it from the Control tab, then try again.`,
      );
    }

    if (!response.ok) {
      const detail = (await response.text()).trim();
      throw new Error(`Local server returned ${response.status}${detail ? `: ${detail}` : ""}`);
    }

    const payload = await response.json();
    const text = String(
      payload.content ??
        payload.response ??
        payload.choices?.[0]?.message?.content ??
        payload.choices?.[0]?.text ??
        "",
    ).trim();
    setAgentStatus(`Connected to ${baseUrl}`);
    return text;
  } finally {
    hideAgentThinking();
  }
}

function extractModelAction(text: string): AgentModelAction | null {
  const variants = modelTextVariants(text);
  for (const variant of variants) {
    const parsed =
      parseJsonAction(variant) ??
      parseJsonFragmentAction(variant) ??
      parseXmlToolAction(variant);
    if (parsed) return parsed;
  }
  return null;
}

function modelTextVariants(text: string): string[] {
  const trimmed = text.trim();
  const fenced = trimmed.match(/```(?:json|xml)?\s*([\s\S]*?)```/i)?.[1]?.trim();
  const toolCallBlock = trimmed.match(/<tool_call>\s*([\s\S]*?)\s*<\/tool_call>/i)?.[1]?.trim();
  const noThink = trimmed
    .replace(/<think>[\s\S]*?<\/think>/gi, "")
    .replace(/<reasoning>[\s\S]*?<\/reasoning>/gi, "")
    .replace(/<\/?tool_call>/gi, "")
    .trim();
  const strippedChannels = trimmed
    .replace(/<\|channel\|>[a-z_]+/gi, "\n")
    .replace(/<\|message\|>/gi, "\n")
    .replace(/<\|start\|>|<\|end\|>/gi, "\n");
  const unescaped = strippedChannels
    .replace(/\\"/g, '"')
    .replace(/\\n/g, "\n")
    .replace(/\\t/g, "\t");
  return Array.from(
    new Set([trimmed, fenced ?? "", toolCallBlock ?? "", noThink, strippedChannels, unescaped].filter(Boolean)),
  );
}

function cleanModelTextForDisplay(text: string): string {
  return text
    .replace(/<\|channel\|>[a-z_]+/gi, "")
    .replace(/<\|message\|>/gi, "")
    .replace(/<\|start\|>|<\|end\|>/gi, "")
    .replace(/\\"/g, '"')
    .replace(/\\n/g, "\n")
    .trim();
}

function firstString(record: Record<string, unknown>, keys: string[]): string | undefined {
  for (const key of keys) {
    const value = record[key];
    if (typeof value === "string" && value.trim()) return value;
  }
  return undefined;
}

function firstObject(record: Record<string, unknown>, keys: string[]): ToolArgs | undefined {
  for (const key of keys) {
    const value = record[key];
    if (value && typeof value === "object" && !Array.isArray(value)) return value as ToolArgs;
  }
  return undefined;
}

// Parses JSON, then progressively repairs the malformed JSON local models emit:
// trailing commas, unquoted keys, and single-quoted strings.
function relaxedJsonParse(text: string): unknown {
  try {
    return JSON.parse(text);
  } catch {
    // fall through to repairs
  }
  let repaired = text.replace(/,(\s*[}\]])/g, "$1");
  try {
    return JSON.parse(repaired);
  } catch {
    // continue
  }
  repaired = repaired.replace(/([{,]\s*)([A-Za-z_$][\w$]*)\s*:/g, '$1"$2":');
  try {
    return JSON.parse(repaired);
  } catch {
    // continue
  }
  repaired = repaired.replace(/'([^'\\]*(?:\\.[^'\\]*)*)'/g, (_match, inner: string) => `"${inner.replace(/"/g, '\\"')}"`);
  try {
    return JSON.parse(repaired);
  } catch {
    return undefined;
  }
}

function parseMaybeJsonObject(value: unknown): ToolArgs | undefined {
  if (value && typeof value === "object" && !Array.isArray(value)) return value as ToolArgs;
  if (typeof value === "string" && value.trim()) {
    try {
      const parsed = JSON.parse(value);
      if (parsed && typeof parsed === "object" && !Array.isArray(parsed)) return parsed as ToolArgs;
    } catch {
      // not JSON
    }
  }
  return undefined;
}

const inlineArgKeys = [
  "path", "content", "command", "query", "url", "search", "replace", "line",
  "glob", "diff", "fromPath", "toPath", "uri", "prompt", "result", "instructions", "role", "mode",
];

function harvestInlineArgs(record: Record<string, unknown>): ToolArgs {
  const args: ToolArgs = {};
  for (const key of inlineArgKeys) {
    if (record[key] !== undefined) args[key] = record[key];
  }
  return args;
}

function isKnownToolName(name: string): boolean {
  return (
    localTools.some((tool) => tool.name === name) ||
    localTools.some((tool) => tool.name === toolAliasCanonical(name))
  );
}

// Normalizes the many shapes models emit — tool/name/tool_name/action,
// args/arguments/parameters/input/tool_input/action_input, function objects,
// tool_calls arrays, ReAct final answers, and inline args — into one action.
function normalizeModelAction(value: unknown): AgentModelAction | null {
  if (!value || typeof value !== "object" || Array.isArray(value)) return null;
  const record = value as Record<string, unknown>;
  const reasoning = firstString(record, ["reasoning", "rationale", "thought", "thinking"]);

  const toolCalls = record.tool_calls;
  if (Array.isArray(toolCalls) && toolCalls.length > 0) {
    const call = toolCalls[0] as Record<string, unknown>;
    const fn = (call.function as Record<string, unknown>) ?? call;
    if (typeof fn.name === "string" && fn.name) {
      return { tool: fn.name, args: parseMaybeJsonObject(fn.arguments) ?? {}, reasoning };
    }
  }

  const fnObj = record.function as Record<string, unknown> | undefined;
  // Explicit tool keys are trusted even if unknown (the model meant a tool, so it
  // gets useful feedback). Loose name/action keys must match a real tool.
  const explicitTool =
    firstString(record, ["tool", "tool_name"]) ??
    (fnObj && typeof fnObj.name === "string" ? fnObj.name : undefined);
  const looseTool = firstString(record, ["action", "name", "recipient_name"]);

  const finalText = firstString(record, ["final", "final_answer", "answer"]);
  const messageText = firstString(record, ["message", "content", "text", "response"]);
  const args =
    firstObject(record, ["args", "arguments", "parameters", "params", "input", "tool_input", "action_input"]) ??
    (fnObj ? parseMaybeJsonObject(fnObj.arguments) : undefined);

  const finishWord = (explicitTool ?? looseTool)?.toLowerCase();
  if (finishWord === "final" || finishWord === "final_answer" || finishWord === "finish" || finishWord === "respond") {
    const answer = finalText ?? messageText ?? (args && typeof args.answer === "string" ? args.answer : undefined);
    if (answer) return { final: answer, reasoning };
  }

  let toolName = explicitTool;
  if (!toolName && looseTool && (isKnownToolName(looseTool) || (!finalText && !messageText))) {
    toolName = looseTool;
  }

  if (toolName) {
    return { tool: toolName, args: args ?? harvestInlineArgs(record), reasoning };
  }
  if (finalText) return { final: finalText, reasoning };
  if (messageText) return { message: messageText, reasoning };
  return null;
}

function actionScore(candidate: string): number {
  let score = 0;
  if (/"(?:tool|name|tool_name|action|function|tool_calls)"\s*:/.test(candidate)) score += 2;
  if (/"(?:final|final_answer|answer|message|content)"\s*:/.test(candidate)) score += 1;
  return score;
}

// Returns every balanced { ... } object found in the text, most action-like first,
// so prose, multiple objects, or reasoning that contains braces don't break parsing.
function jsonObjectCandidates(text: string): string[] {
  const candidates: string[] = [];
  const trimmed = text.trim();
  if (trimmed.startsWith("{") && trimmed.endsWith("}")) candidates.push(trimmed);
  for (let index = 0; index < text.length; index += 1) {
    if (text[index] !== "{") continue;
    const end = findBalancedObjectEnd(text, index);
    if (end > index) {
      candidates.push(text.slice(index, end + 1));
      index = end;
    }
  }
  const first = trimmed.indexOf("{");
  const last = trimmed.lastIndexOf("}");
  if (first >= 0 && last > first) candidates.push(trimmed.slice(first, last + 1));
  return Array.from(new Set(candidates.filter(Boolean))).sort((a, b) => actionScore(b) - actionScore(a));
}

function parseJsonAction(text: string): AgentModelAction | null {
  for (const candidate of jsonObjectCandidates(text)) {
    const parsed = relaxedJsonParse(candidate);
    if (parsed === undefined) continue;
    const action = normalizeModelAction(parsed);
    if (action) return action;
  }
  return null;
}

function parseJsonFragmentAction(text: string): AgentModelAction | null {
  const toolMatch = text.match(/"(?:tool|name|tool_name|action)"\s*:\s*"([^"]+)"/);
  if (!toolMatch) return null;

  const reasoningMatch = text.match(/"reasoning"\s*:\s*"([\s\S]*?)"\s*,\s*"(?:tool|name|action)"/);
  const argsIndex = text.search(/"(?:args|arguments|parameters|input|tool_input|action_input)"\s*:/);
  let args: ToolArgs = {};
  if (argsIndex >= 0) {
    const argsStart = text.indexOf("{", argsIndex);
    const argsEnd = argsStart >= 0 ? findBalancedObjectEnd(text, argsStart) : -1;
    if (argsStart >= 0 && argsEnd > argsStart) {
      const argsText = text.slice(argsStart, argsEnd + 1);
      try {
        const parsedArgs = JSON.parse(argsText) as unknown;
        if (parsedArgs && typeof parsedArgs === "object" && !Array.isArray(parsedArgs)) {
          args = parsedArgs as ToolArgs;
        }
      } catch {
        args = recoverCommonToolArgs(toolMatch[1], text);
      }
    } else {
      args = recoverCommonToolArgs(toolMatch[1], text);
    }
  }

  return {
    tool: toolMatch[1],
    args,
    reasoning: reasoningMatch?.[1],
  };
}

function findBalancedObjectEnd(text: string, start: number): number {
  let depth = 0;
  let inString = false;
  let escaped = false;
  for (let index = start; index < text.length; index += 1) {
    const char = text[index];
    if (escaped) {
      escaped = false;
      continue;
    }
    if (char === "\\") {
      escaped = true;
      continue;
    }
    if (char === '"') {
      inString = !inString;
      continue;
    }
    if (inString) continue;
    if (char === "{") depth += 1;
    if (char === "}") {
      depth -= 1;
      if (depth === 0) return index;
    }
  }
  return -1;
}

function recoverCommonToolArgs(tool: string, text: string): ToolArgs {
  const path = text.match(/"path"\s*:\s*"([^"]+)"/)?.[1] ?? "";
  const command = text.match(/"command"\s*:\s*"([^"]+)"/)?.[1] ?? "";
  const contentMatch = text.match(/"content"\s*:\s*"([\s\S]*)/);
  const content = contentMatch?.[1]?.replace(/"\s*}\s*}?\s*$/, "") ?? "";
  if (tool === "execute_command" || tool === "shell") return { command: decodeJsonStringEscapes(command) };
  return { path, content: decodeJsonStringEscapes(content) };
}

// When a tool call is recovered from a broken JSON fragment, escape sequences
// (\n, \", \t) survive as literal characters. Decode them so files are written
// with real newlines and quotes instead of literal backslash-n.
function decodeJsonStringEscapes(value: string): string {
  return value
    .replace(/\\r\\n/g, "\n")
    .replace(/\\n/g, "\n")
    .replace(/\\r/g, "\n")
    .replace(/\\t/g, "\t")
    .replace(/\\"/g, '"')
    .replace(/\\\//g, "/")
    .replace(/\\\\/g, "\\");
}

function decodeXmlText(value: string): string {
  const textarea = document.createElement("textarea");
  textarea.innerHTML = value;
  return textarea.value;
}

function parseXmlToolAction(text: string): AgentModelAction | null {
  for (const tool of localTools) {
    const match = text.match(new RegExp(`<${tool.name}>\\s*([\\s\\S]*?)\\s*</${tool.name}>`, "i"));
    if (!match) continue;
    const body = match[1];
    const args: ToolArgs = {};
    for (const field of body.matchAll(/<([a-zA-Z0-9_-]+)>\s*([\s\S]*?)\s*<\/\1>/g)) {
      args[field[1]] = decodeXmlText(field[2]);
    }
    return { tool: tool.name, args };
  }

  const attemptMatch = text.match(/<attempt_completion>\s*([\s\S]*?)\s*<\/attempt_completion>/i);
  if (attemptMatch) {
    return { tool: "attempt_completion", args: { result: decodeXmlText(attemptMatch[1]) } };
  }
  return null;
}

function agentToolInstructions(): string {
  const allowed = new Set(activeProfile().allowedTools);
  return [
    "- read {\"path\":\"src/main.ts\"}: Pi read tool; reads a file or lists a directory",
    "- write {\"path\":\"file.txt\",\"content\":\"complete file contents\"}: Pi write tool; create or replace a file",
    "- edit {\"path\":\"file.txt\",\"edits\":[{\"oldText\":\"exact old text\",\"newText\":\"replacement\"}]}: Pi edit tool for exact replacements",
    "- bash {\"command\":\"npm run build\"}: Pi bash tool; runs through the local CLI bridge",
    "- grep {\"query\":\"needle\",\"path\":\"src\",\"glob\":\"*.ts\"}: Pi grep tool backed by ripgrep",
    "- find {\"path\":\"src\",\"query\":\"agent\",\"glob\":\"*.ts\"}: Pi find tool for file names and paths",
    "- ls {\"path\":\"src\"}: Pi ls tool for directories",
    "- read_file {\"path\":\"src/main.ts\"}: read a file or list a directory",
    "- list_files {\"path\":\"src\"}: list a directory",
    "- search_files {\"query\":\"needle\",\"path\":\"src\",\"glob\":\"*.ts\"}: search with ripgrep",
    "- find_files {\"path\":\"src\",\"query\":\"agent\",\"glob\":\"*.ts\"}: filter files by name/path",
    "- go_to_line {\"path\":\"src/main.ts\",\"line\":120,\"context\":20}: read numbered lines around a line",
    "- search_lines {\"path\":\"src/main.ts\",\"query\":\"needle\",\"regex\":false}: search one file and return line numbers",
    "- list_code_definition_names {\"path\":\"src\"}: list functions, classes, and definitions",
    "- semantic_search {\"query\":\"auth flow\",\"path\":\"src\"}: search code by intent with local lexical fallback",
    "- edit_file {\"path\":\"file.txt\",\"content\":\"full replacement\"}: explicit full-file replacement only",
    "- write_to_file {\"path\":\"file.txt\",\"content\":\"complete file contents\"}: create or replace a file; blocked for fix/change requests on existing files",
    "- append_to_file {\"path\":\"file.txt\",\"content\":\"text\"}: append exact text to the end of a file",
    "- insert_content {\"path\":\"file.txt\",\"line\":12,\"content\":\"text\"}: insert text into a file",
    "- edit_lines {\"path\":\"file.txt\",\"start\":12,\"end\":14,\"content\":\"replacement\"}: replace a numbered line range",
    "- replace_in_file {\"path\":\"file.txt\",\"search\":\"old\",\"replace\":\"new\",\"all\":true}: replace text",
    "- apply_diff {\"diff\":\"unified diff\"}: apply a git-style unified diff",
    "- delete_file {\"path\":\"file.txt\"}: delete a file or directory",
    "- shell {\"command\":\"npm run build\"}: run a workspace command",
    "- execute_command {\"command\":\"npm run build\"}: command execution",
    "- browse_url {\"url\":\"https://example.com\"}: open a URL",
    "- browser_action {\"action\":\"open\",\"url\":\"https://example.com\"}: browser open/search/close actions",
    '- web_search {"query":"...","provider":"auto"}: keyless DuckDuckGo, Ecosia, Google, and Mojeek search with automatic fallback',
    "- web_fetch {\"url\":\"https://example.com\"}: download the text contents of a URL",
    "- download_file {\"url\":\"https://example.com/file.zip\",\"path\":\"assets/file.zip\"}: save a URL to a workspace file",
    "- todo_write {\"text\":\"task\"}: update todo list",
    "- update_todo_list {\"todos\":[{\"text\":\"inspect files\",\"status\":\"pending\"}]}: replace todo list",
    "- skill_load {\"name\":\"reviewer\",\"instructions\":\"...\"}: load a skill",
    "- subagent_start {\"name\":\"Reviewer\",\"role\":\"...\"}: configure a subagent",
    "- compact_context {}: summarize context",
    "- mcp_call {\"server\":\"localhost:3000\",\"method\":\"tools/list\",\"params\":{}}: call local MCP",
    "- use_mcp_tool {\"server_name\":\"local\",\"tool_name\":\"tool\",\"arguments\":{}}: MCP tool call",
    "- access_mcp_resource {\"server_name\":\"local\",\"uri\":\"resource://id\"}: read MCP resource",
    "- switch_mode {\"mode\":\"coder\"}: switch coding agent mode",
    "- new_task {\"mode\":\"reviewer\",\"message\":\"review this change\"}: delegate a subtask",
    "- ask_followup_question {\"question\":\"...\",\"follow_up\":[\"option\"]}: ask user for more context",
    "- attempt_completion {\"result\":\"...\",\"command\":\"npm test\"}: present final result",
    "- delegate_agent {\"agent\":\"Reviewer\",\"prompt\":\"review this plan\"}: ask a built-in specialist",
    "- pi_cli {\"command\":\"status\"}: show the local Pi-compatible CLI bridge state",
  ]
    .filter((line) => {
      const body = line.slice(2);
      const name = body.slice(0, body.indexOf(" "));
      return allowed.has(name) || allowed.has(toolAliasCanonical(name));
    })
    .join("\n");
}

// A compact transcript of the current task's earlier messages (before this
// turn), so the agent remembers the conversation and its own prior tool results
// across user turns — a persistent task message history.
function conversationSnapshot(): string {
  const prior = currentTask().messages.slice(0, -1); // exclude the current user prompt
  if (prior.length === 0) return "";

  const picked: string[] = [];
  let total = 0;
  for (let index = prior.length - 1; index >= 0 && total < 12000; index -= 1) {
    const entry = prior[index];
    const label = entry.role === "user" ? "User" : entry.role === "tool" ? "Tool" : "Assistant";
    const text = entry.message.length > 2000 ? `${entry.message.slice(0, 2000)}…` : entry.message;
    picked.push(`${label}: ${text}`);
    total += text.length;
  }
  picked.reverse();
  return picked.join("\n");
}

function agentLoopPrompt(userPrompt: string, observations: string[], priorConversation: string): string {
  const enabledSkills = skills.filter((skill) => skill.enabled).map((skill) => `${skill.name}: ${skill.instructions}`).join("\n");
  const enabledSubagents = subagents.filter((agent) => agent.enabled).map((agent) => `${agent.name}: ${agent.role}`).join("\n");
  const profile = activeProfile();
  return [
    `You are the LocalLLM Pi Workbench ${profile.name}, a local coding agent inspired by earendil-works/pi.`,
    profile.instructions,
    agentSystemInstructions ? `System instructions:\n${agentSystemInstructions}` : "",
    agentTaskInstructions ? `Reusable task instructions:\n${agentTaskInstructions}` : "",
    "Use the Pi tool vocabulary when possible: read, write, edit, bash, grep, find, and ls. These are implemented behind the scenes by LocalLLM's Tauri workspace bridge and local llama-server.",
    `Agent mode: ${agentMode}. ${
      agentMode === "architect"
        ? "Prefer reading and planning before edits."
        : agentMode === "debugger"
          ? "Reproduce, inspect logs, patch narrowly, and verify."
          : "Implement directly and verify."
    }`,
    agentExecutionMode === "plan"
      ? "PLAN MODE is ON. Do not modify the workspace. Use only read and search tools to investigate, then return a clear, numbered implementation plan with {\"final\":\"...\"}. Editing, moving, copying, pasting, and shell commands are disabled."
      : "EDIT MODE is ON. Do not stop after writing a plan. Modify the workspace with the available tools, continue through all safe in-scope steps, verify the result, and only then finish.",
    "Return exactly one JSON object. Do not wrap it in prose.",
    agentThinkingInstruction(),
    "To use a tool: {\"tool\":\"read\",\"args\":{\"path\":\"src/main.ts\"}}",
    agentThinkingLevel === "disabled"
      ? "To answer the user: {\"final\":\"your answer\"}"
      : "To answer the user: {\"reasoning\":\"checked the result\",\"final\":\"your answer\"}",
    "Never call a tool with empty args. File tools require path and content when writing.",
    "For write/write_to_file, always include args.path and args.content with the complete file contents. Agent completions use unlimited prediction, so do not split ordinary files into partial write chunks. Use write/write_to_file/edit_file only for new files or explicit full rewrites. For fixes, errors, patches, and changes to existing files, first inspect with read/grep/find/ls or go_to_line/search_lines, then use edit with exact oldText/newText replacements, edit_lines, replace_in_file, insert_content, apply_diff, or bash. Use append_to_file only when the user explicitly wants to append. For bash/execute_command, always include args.command.",
    "Terminal editing playbook: PowerShell search/read lines: Select-String -Path 'file' -Pattern 'needle'; $lines = Get-Content 'file'; $lines[9..29]. PowerShell replace file text: (Get-Content -Raw 'file') -replace 'old','new' | Set-Content -NoNewline 'file'. Linux search/read lines: rg -n 'needle' file; nl -ba file | sed -n '10,30p'. Linux replace/delete lines: perl -0pi -e 's/old/new/g' file; sed -i '10,12d' file.",
    "For artifact requests like games, apps, or pages, write complete runnable source code. Never write only the user request into a file.",
    "For standalone HTML/page/game artifacts, write to index.html or a clearly named root file like game.html unless the user explicitly names another path. Do not choose src/index.html by default.",
    "Do not just describe what you will do. If you intend to inspect, read, search, edit, or fix something, emit the matching tool call in this same reply. Saying \"I will inspect\" or \"I am fixing\" without a tool call does nothing. Only use {\"final\":...} once the work is actually done.",
    "For a direct question, answer it fully with {\"final\":\"...\"}. Never return an empty response and never ask the user to say “continue”; continue the current task yourself until it is complete or an exact external blocker must be reported.",
    "You have NOT created, written, or edited any file until a tool observation in 'Steps taken this turn' confirms success (e.g. 'Wrote N bytes'). Never claim you created a file or finished the task unless such a confirmation is present. If asked to build something, your first action must be a write_to_file tool call with the full contents.",
    "If a previous observation says a write was blocked or a tool repeated, change strategy and produce a better implementation.",
    "Available tools:",
    agentToolInstructions(),
    discoveredSkillsPromptSection(),
    enabledSkills ? `Enabled skills:\n${enabledSkills}` : "",
    enabledSubagents ? `Enabled subagents:\n${enabledSubagents}` : "",
    agentGoal ? `Current goal:\n${agentGoal}` : "",
    contextSummary ? `Context summary:\n${contextSummary}` : "",
    priorConversation ? `Conversation so far:\n${priorConversation}` : "",
    observations.length ? `Steps taken this turn:\n${observations.join("\n\n")}` : "",
    `Current user message:\n${userPrompt}`,
  ]
    .filter(Boolean)
    .join("\n\n");
}

async function delegateToCodingAgent(agentName: string, prompt: string): Promise<string> {
  if (!prompt) throw new Error("Enter a task to delegate");
  const requested = agentName.trim().toLowerCase();
  const profile =
    codingAgentProfiles.find((candidate) => candidate.id === requested || candidate.name.toLowerCase() === requested) ??
    profileById("reviewer");
  if (!(await requestToolPermission("subagent", `Delegate to ${profile.name}: ${prompt}`))) {
    return "Delegation denied";
  }

  const output = await postLocalCompletion(
    [
      `You are the LocalLLM Pi Workbench ${profile.name}, a specialist coding agent.`,
      profile.instructions,
      "Return a concise specialist response. You cannot execute tools directly in this delegated reply; tell the primary agent exactly what to inspect, change, or verify.",
      agentGoal ? `Current goal:\n${agentGoal}` : "",
      contextSummary ? `Context summary:\n${contextSummary}` : "",
      `Delegated task:\n${prompt}`,
    ]
      .filter(Boolean)
      .join("\n\n"),
  );
  const result = `${profile.name} agent:\n${output}`;
  appendAgentMessage("agent", result);
  return result;
}

function promptAsksForImplementation(prompt: string): boolean {
  return /\b(write|create|build|make|implement|generate|fix|change|add|delete|remove|update|crie|criar|faça|fazer|conserte|corrija|corrigir|adicione|adicionar|altere|alterar|implemente|implementar|mude|mudar|remova|remover|exclua|excluir|apague|apagar|atualize|atualizar|gere|gerar)\b/i.test(
    prompt,
  );
}

// Detects a reply that claims a file/artifact was created or the task finished,
// used to catch hallucinated completion when no write actually happened.
function claimsArtifactCompletion(text: string): boolean {
  const claims = /\b(creat\w*|wrote|written|generat\w*|sav\w*|built|build|implement\w*|finish\w*|complet\w*|\bdone\b)\b/i.test(text);
  const artifact = /\b(file|index\.html|\.html?|\.js|\.ts|\.css|\.py|game|code|app|page|website|script|project)\b/i.test(text);
  const runnable = /\b(you can (?:run|open|play|use)|open .* in (?:your|the) browser|run the (?:game|app|file))\b/i.test(text);
  return (claims && artifact) || runnable;
}

async function runAgentLoop(userPrompt: string): Promise<void> {
  const observations: string[] = [];
  const seenActions = new Set<string>();
  currentAgentRequest = userPrompt;
  // Snapshot the prior conversation once, before this turn adds tool activity,
  // so the model has cross-turn memory without double-counting this turn.
  const priorConversation = conversationSnapshot();
  let parseFailures = 0;
  let noActionStreak = 0;
  let didWrite = false;
  let hallucinatedFinals = 0;
  const effectiveMaxSteps = Math.min(64, agentMaxSteps + Math.max(8, Math.ceil(agentMaxSteps / 2)));

  if (agentExecutionMode === "edit" && activeAgentProfile === "architect" && promptAsksForImplementation(userPrompt)) {
    setActiveAgentProfile("coder");
    appendAgentMessage("agent", "Reasoning\nThis is an implementation request, so I switched from Architect to Coder before using write-capable tools.");
  }

  const piStatus = piAgentStatus ?? (await refreshPiAgentStatus());
  if (agentExecutionMode === "edit" && piStatus?.available) {
    try {
      await runPiCodingAgent(userPrompt);
      setAgentStatus("Pi coding agent task complete");
      return;
    } catch (error) {
      const fallback = `Pi coding agent failed, falling back to the local llama-server loop.\n${String(error)}`;
      appendAgentMessage("tool", fallback);
      setAgentOutput(fallback);
      setAgentCliStatus("CLI: local fallback");
    }
  }

  for (let step = 1; step <= effectiveMaxSteps; step += 1) {
    if (agentAbort?.signal.aborted) return;
    if (step === agentMaxSteps + 1) {
      const continuation =
        `The configured ${agentMaxSteps}-step pass ended before an explicit completion. ` +
        `Continue autonomously using the safety reserve; do not repeat the plan and finish or report an exact blocker.`;
      observations.push(continuation);
      appendAgentMessage("tool", continuation);
    }
    setAgentStatus(`Thinking with local server, step ${step}/${effectiveMaxSteps}`);
    const completion = await postLocalCompletion(
      `${agentLoopPrompt(userPrompt, observations, priorConversation)}\n\nAssistant:`,
      `Thinking · step ${step}/${effectiveMaxSteps}`,
      agentActionJsonSchema,
    );
    const action = extractModelAction(completion);

    if (!action) {
      parseFailures += 1;
      // Surface what the model actually said so it is debuggable instead of hidden.
      const raw = cleanModelTextForDisplay(completion).trim();
      if (raw && !promptAsksForImplementation(userPrompt)) {
        appendAgentMessage("agent", raw);
        setAgentStatus("Agent task complete");
        return;
      }
      appendAgentMessage("agent", raw ? raw.slice(0, 1500) : "(empty model reply)");
      if (parseFailures >= 5) {
        const failure =
          "The local model did not return a usable answer or tool action after five recovery attempts. " +
          "The task was not marked complete; try a stronger model or inspect the raw replies above.";
        appendAgentMessage("tool", failure);
        setAgentStatus("Agent needs attention: unusable model responses");
        return;
      }
      const correction =
        'Your last reply was not a single valid JSON action. Reply with exactly one JSON object and nothing else: {"tool":"<name>","args":{...}} to act, or {"final":"<answer>"} to finish. No prose, no code fences, no markdown.';
      appendAgentMessage("tool", correction);
      observations.push(`Step ${step}: your previous reply could not be parsed as an action. ${correction}`);
      continue;
    }
    parseFailures = 0;

    const reasoning = action.reasoning ?? action.rationale;
    if (reasoning && agentThinkingLevel !== "disabled") {
      appendAgentMessage("agent", `Reasoning\n${reasoning}`);
    }

    // Only an explicit final answer ends the turn. A reply with a message but no
    // tool is usually the model narrating intent ("I will inspect...") without
    // acting; show it and nudge it to actually call a tool instead of stopping.
    if (action.final) {
      // Guard against hallucinated completion: the model claims it created/edited
      // a file but no write actually happened this turn.
      if (
        promptAsksForImplementation(userPrompt) &&
        !didWrite &&
        hallucinatedFinals < 2 &&
        claimsArtifactCompletion(action.final)
      ) {
        hallucinatedFinals += 1;
        appendAgentMessage("agent", action.final);
        const nudge =
          'You claimed the work is done, but no file was actually written or edited this turn. Do not claim completion. Emit the tool call now to really do it, e.g. {"tool":"write_to_file","args":{"path":"index.html","content":"<full file contents>"}}.';
        appendAgentMessage("tool", nudge);
        observations.push(`Step ${step}: ${nudge}`);
        continue;
      }
      appendAgentMessage("agent", action.final);
      setAgentStatus("Agent task complete");
      return;
    }

    if (!action.tool) {
      const narration = (action.message ?? cleanModelTextForDisplay(completion)).trim();
      if (narration) appendAgentMessage("agent", narration);
      noActionStreak += 1;
      // A message with no tool call is not a finished turn — only an explicit
      // {"final":...} ends it. Nudge the model to actually act (or to finish
      // explicitly) instead of stopping on narration like "Creating the file now".
      const nudge =
        noActionStreak < 4
          ? 'You wrote a message but took no action. If the task is not finished, reply with exactly one tool call now to do the work, e.g. {"tool":"write_to_file","args":{"path":"game.html","content":"<full file contents>"}}. If you are completely done, reply with {"final":"<answer>"}.'
          : 'You have repeatedly narrated without acting. Stop restating the plan. Emit one permitted tool call now, or return {"final":"<answer>"} only if the requested work is actually complete.';
      appendAgentMessage("tool", nudge);
      observations.push(`Step ${step}: ${nudge}`);
      if (noActionStreak >= 4) noActionStreak = 0;
      continue;
    }
    noActionStreak = 0;

    const signature = `${action.tool}:${JSON.stringify(action.args ?? {})}`;
    if (seenActions.has(signature)) {
      const repeatObservation = `Repeated tool call blocked: ${action.tool} with identical arguments. Inspect the previous result, change approach, or finish with {"final":"..."}.`;
      appendAgentMessage("tool", repeatObservation);
      observations.push(repeatObservation);
      continue;
    }
    seenActions.add(signature);

    const observation = await executeToolCall({ tool: action.tool, args: action.args ?? {} });
    observations.push(`Step ${step} ${action.tool} observation:\n${observation.slice(0, 6000)}`);

    const toolDef = localTools.find((candidate) => candidate.name === action.tool);
    if (toolDef?.permission === "edit" && !/denied|blocked|tool error|rejected|fail/i.test(observation)) {
      didWrite = true;
    }

    if (action.tool === "attempt_completion") {
      setAgentStatus("Agent task complete");
      return;
    }
  }

  appendAgentMessage(
    "agent",
    `I used the full ${effectiveMaxSteps}-step safety budget without reaching a reliable completion. ` +
      "The task is still open; review the latest tool result or increase Maximum steps before retrying.",
  );
  setAgentStatus("Agent paused at hard safety limit · task not complete");
}

function setAgentRunning(active: boolean) {
  agentRunActive = active;
  const button = document.getElementById("agent-run-command") as HTMLButtonElement | null;
  const composer = document.querySelector<HTMLElement>("#app-view-agent .agent-chat-input-row");
  const input = document.getElementById("agent-command-input") as HTMLTextAreaElement | null;
  const loading = document.getElementById("agent-prompt-loading");
  composer?.classList.toggle("is-loading", active);
  composer?.setAttribute("aria-busy", String(active));
  input?.setAttribute("aria-busy", String(active));
  if (loading) loading.hidden = !active;
  if (button) {
    button.title = active ? "Stop" : "Run";
    button.setAttribute("aria-label", active ? "Stop" : "Run");
    button.classList.toggle("agent-running", active);
    button.innerHTML = active ? stopIconSvg : runIconSvg;
  }
}

function cancelAgentRun() {
  agentAbort?.abort();
  setAgentStatus("Stopping agent...");
}

function isAbortError(error: unknown): boolean {
  return error instanceof Error && error.name === "AbortError";
}

async function runAgentCommand() {
  if (agentRunActive) {
    cancelAgentRun();
    return;
  }

  const inputField = $("agent-command-input") as HTMLTextAreaElement;
  const input = inputField.value.trim();
  if (!input) return;
  inputField.value = "";

  agentAbort = new AbortController();
  setAgentRunning(true);
  try {
    await dispatchAgentCommand(input);
  } catch (error) {
    if (isAbortError(error)) {
      appendAgentMessage("agent", "Agent run stopped.");
      setAgentStatus("Agent stopped");
    } else {
      throw error;
    }
  } finally {
    setAgentRunning(false);
    agentAbort = null;
  }
}

async function dispatchPiSlashCommand(input: string): Promise<boolean> {
  const [commandRaw = "", ...parts] = parseAgentCommand(input);
  const command = commandRaw.toLowerCase();
  const body = parts.join(" ");

  switch (command) {
    case "/new":
      startNewAgentTask();
      return true;
    case "/session": {
      const task = currentTask();
      const output = [
        `Session: ${task.id}`,
        `Name: ${task.title}`,
        `Messages: ${task.messages.length}`,
        `Profile: ${activeProfile().name}`,
        `Mode: ${agentExecutionMode}`,
        `Updated: ${new Date(task.updatedAt).toLocaleString()}`,
      ].join("\n");
      appendAgentMessage("agent", output);
      return true;
    }
    case "/resume":
    case "/history":
      appendAgentMessage("agent", `Task history\n${renderTaskHistorySummary()}`);
      return true;
    case "/compact":
      await compactAgentContext();
      return true;
    case "/name": {
      const task = currentTask();
      task.title = body.trim() || task.title;
      task.updatedAt = Date.now();
      saveTaskHistory();
      appendAgentMessage("agent", `Session renamed: ${task.title}`);
      return true;
    }
    case "/model":
    case "/settings":
      await piCliTool("status");
      return true;
    case "/tools":
      appendAgentMessage("agent", localTools.map((tool) => `${tool.name}: ${tool.description}`).join("\n"));
      return true;
    default:
      return false;
  }
}

async function dispatchAgentCommand(input: string) {
  appendAgentMessage("user", input);
  currentAgentRequest = input;

  const toolCall = parseToolCall(input);
  if (toolCall) {
    await executeToolCall(toolCall);
    return;
  }

  if (input.startsWith("!!")) {
    await executeToolCall({ tool: "bash", args: { command: input.slice(2).trim() } });
    return;
  }

  if (input.startsWith("!")) {
    await executeToolCall({ tool: "bash", args: { command: input.slice(1).trim() } });
    return;
  }

  if (input.startsWith("/")) {
    if (await dispatchPiSlashCommand(input)) return;
  }

  const [verbRaw, first = "", second = "", ...rest] = parseAgentCommand(input);
  const verb = verbRaw?.toLowerCase();
  const commandBody = stripCommandPrefix(input, verbRaw ?? "");

  switch (verb) {
    case "read":
    case "open":
    case "ls":
    case "list":
      await executeToolCall({
        tool: (verb === "list" || verb === "ls") && second ? "find_files" : verb === "list" || verb === "ls" ? "list_files" : "read_file",
        args: (verb === "list" || verb === "ls") && second
          ? { path: first || ".", query: [second, ...rest].filter(Boolean).join(" ") }
          : { path: first === "file" ? second || "." : first || "." },
      });
      break;
    case "pi":
      if (!commandBody || commandBody === "status" || commandBody === "model" || commandBody === "settings") {
        await executeToolCall({ tool: "pi_cli", args: { command: commandBody || "status" } });
      } else {
        await executeToolCall({ tool: "pi_agent", args: { prompt: commandBody } });
      }
      break;
    case "cli":
      await executeToolCall({ tool: "pi_cli", args: { command: commandBody || "status" } });
      break;
    case "find":
    case "filter":
    case "find_files":
      await executeToolCall({
        tool: "find_files",
        args: { path: first || ".", query: [second, ...rest].filter(Boolean).join(" ") },
      });
      break;
    case "goto":
    case "go_to_line":
    case "line":
    case "lines":
      await executeToolCall({
        tool: "go_to_line",
        args: { path: first, line: Number(second), end: rest[0] ? Number(rest[0]) : undefined },
      });
      break;
    case "edit":
    case "write":
    case "write_to_file":
    case "create": {
      if (!isExplicitFileWriteCommand(verb, first, commandBody)) {
        await runAgentLoop(input);
        break;
      }
      const body = first === "file" ? stripCommandPrefix(commandBody, first) : commandBody;
      const { path, content } = splitPathAndContent(body);
      await executeToolCall({
        tool: verb === "edit" ? "edit_file" : "write_to_file",
        args: { path, content: content || undefined },
      });
      break;
    }
    case "insert":
    case "insert_content": {
      const { path, content } = splitPathAndContent(commandBody);
      await executeToolCall({
        tool: "insert_content",
        args: { path, content },
      });
      break;
    }
    case "append":
    case "append_to_file": {
      const { path, content } = splitPathAndContent(commandBody);
      await executeToolCall({
        tool: "append_to_file",
        args: { path, content },
      });
      break;
    }
    case "edit_lines": {
      const { path, start, end, content } = splitLineEditCommand(commandBody);
      await executeToolCall({
        tool: "edit_lines",
        args: { path, start: start ?? undefined, end: end ?? undefined, content },
      });
      break;
    }
    case "delete":
    case "rm":
    case "delete_file":
      await executeToolCall({ tool: "delete_file", args: { path: first } });
      break;
    case "copy":
    case "cp":
      await executeToolCall({ tool: "copy_file", args: { fromPath: first, toPath: second } });
      break;
    case "replace":
    case "sreplace":
      await executeToolCall({
        tool: "replace_in_file",
        args: {
          path: first,
          search: second,
          replace: rest.join(" "),
          all: verb === "replace",
        },
      });
      break;
    case "diff":
    case "patch":
    case "apply":
      await executeToolCall({ tool: "apply_diff", args: { diff: commandBody } });
      break;
    case "move":
    case "mv":
    case "rename":
      await executeToolCall({ tool: "move_file", args: { fromPath: first, toPath: second } });
      break;
    case "paste":
      await executeToolCall({ tool: "paste_file", args: { path: first } });
      break;
    case "browse":
    case "url":
      await executeToolCall({ tool: "browse_url", args: { url: first } });
      break;
    case "fetch":
    case "get":
      await executeToolCall({ tool: "web_fetch", args: { url: first } });
      break;
    case "download":
    case "dl":
      await executeToolCall({ tool: "download_file", args: { url: first, path: second } });
      break;
    case "search":
    case "rg":
    case "grep":
      await executeToolCall({
        tool: looksLikeFilePath(first) && second ? "search_lines" : "search_files",
        args: looksLikeFilePath(first) && second
          ? { path: first, query: [second, ...rest].filter(Boolean).join(" ") }
          : { query: [first, second, ...rest].filter(Boolean).join(" "), path: "." },
      });
      break;
    case "search_lines":
      await executeToolCall({
        tool: "search_lines",
        args: { path: first, query: [second, ...rest].filter(Boolean).join(" ") },
      });
      break;
    case "definitions":
    case "symbols":
    case "list_code_definition_names":
      await executeToolCall({ tool: "list_code_definition_names", args: { path: first || "." } });
      break;
    case "semantic":
    case "semantic_search":
      await executeToolCall({
        tool: "semantic_search",
        args: { query: [first, second, ...rest].filter(Boolean).join(" "), path: "." },
      });
      break;
    case "web":
    case "websearch":
      await executeToolCall({ tool: "web_search", args: { query: [first, second, ...rest].filter(Boolean).join(" ") } });
      break;
    case "tools":
      appendAgentMessage("agent", localTools.map((tool) => `${tool.name}: ${tool.description}`).join("\n"));
      break;
    case "help":
      appendAgentMessage(
        "agent",
        "Local tools:\nread path\nlist path\nlist path filter\nfind path name\ngoto path line\nsearch query\nsearch path query\nsearch_lines path query\nsemantic query\ndefinitions path\nwrite path content\nappend path content\ninsert path content\nedit_lines path start end content\ndelete path\nreplace path old new\nsreplace path old new\npatch <unified diff>\ncopy from to\nmove from to\npaste path\nshell command\nbrowse url\nfetch url\ndownload url path\nweb query\nmode coder\nnew_task reviewer task\ncomplete result\nask prompt",
      );
      break;
    case "goal":
      setAgentGoal([first, second, ...rest].filter(Boolean).join(" "));
      break;
    case "mode":
    case "switch_mode":
      await executeToolCall({ tool: "switch_mode", args: { mode: first } });
      break;
    case "shell":
    case "run":
    case "execute_command":
      await executeToolCall({ tool: "shell", args: { command: [first, second, ...rest].filter(Boolean).join(" ") } });
      break;
    case "todo":
      await executeToolCall({ tool: "todo_write", args: { text: [first, second, ...rest].filter(Boolean).join(" ") } });
      break;
    case "update_todo_list":
      await executeToolCall({ tool: "update_todo_list", args: { text: commandBody } });
      break;
    case "skill":
      await executeToolCall({
        tool: "skill_load",
        args: { name: first, instructions: [second, ...rest].filter(Boolean).join(" ") },
      });
      break;
    case "subagent":
    case "task":
    case "new_task":
      await executeToolCall({
        tool: verb === "new_task" ? "new_task" : "subagent_start",
        args: { mode: first, name: first, message: [second, ...rest].filter(Boolean).join(" "), role: [second, ...rest].filter(Boolean).join(" ") },
      });
      break;
    case "compact":
      await executeToolCall({ tool: "compact_context", args: {} });
      break;
    case "mcp":
      await executeToolCall({ tool: "mcp_call", args: { server: first, method: second || "tools/list" } });
      break;
    case "use_mcp_tool":
      await executeToolCall({
        tool: "use_mcp_tool",
        args: { server_name: first, tool_name: second, arguments: {} },
      });
      break;
    case "access_mcp_resource":
      await executeToolCall({ tool: "access_mcp_resource", args: { server_name: first, uri: second } });
      break;
    case "followup":
    case "ask_followup_question":
      await executeToolCall({
        tool: "ask_followup_question",
        args: { question: [first, second, ...rest].filter(Boolean).join(" ") },
      });
      break;
    case "complete":
    case "attempt_completion":
      await executeToolCall({
        tool: "attempt_completion",
        args: { result: [first, second, ...rest].filter(Boolean).join(" ") },
      });
      break;
    case "ask":
    case "chat":
    case "prompt":
      await executeToolCall({
        tool: "local_chat",
        args: { prompt: [first, second, ...rest].filter(Boolean).join(" ") },
      });
      break;
    default:
      await runAgentLoop(input);
  }
}

async function refreshAgentWorkspace() {
  try {
    const root = await invoke<string>("agent_workspace_root");
    $("agent-workspace-root").textContent = `Workspace: ${root}`;
    ($("agent-workspace-root-input") as HTMLInputElement).value = root;
    if (agentProfileNeedsWorkspaceMigration) {
      const profile = activeSavedProfile();
      profile.workspaceRoot = root;
      profile.updatedAt = Date.now();
      persistAgentProfileStore();
      agentProfileNeedsWorkspaceMigration = false;
    }
    setAgentStatus(`Agent tools ready / thinking: ${agentThinkingLevel}`);
  } catch (error) {
    $("agent-workspace-root").textContent = "Workspace: unavailable";
    setAgentStatus(String(error));
  }
}

async function saveAgentWorkspaceRoot(path: string): Promise<void> {
  const nextRoot = path.trim();
  if (!nextRoot) throw new Error("Choose a workspace folder");
  await persistAgentWorkspaceRoot(nextRoot, true);
  if (!profileEventsSuspended) {
    setAgentProfileDirty(true);
    setAgentProfileFeedback("Workspace changed. Save the profile to remember it per profile.", "warn");
  }
}

function setAgentGoal(nextGoal: string) {
  agentGoal = nextGoal.trim();
  saveAgentGoal();
  syncGoalUi();
  if (!profileEventsSuspended) setAgentProfileDirty(true);
  appendAgentMessage("agent", agentGoal ? `Goal set:\n${agentGoal}` : "Goal cleared");
}

function setAgentMode(nextMode: AgentMode) {
  if (!codingAgentProfiles.some((profile) => profile.id === nextMode)) {
    throw new Error("Unknown coding agent");
  }
  agentMode = nextMode;
  activeAgentProfile = nextMode;
  saveAgentMode();
  saveActiveAgentProfile();
  syncModeUi();
  if (!profileEventsSuspended) setAgentProfileDirty(true);
  appendAgentMessage("agent", `Coding agent set to ${activeProfile().name}`);
}

function setActiveAgentProfile(nextProfile: AgentMode) {
  if (!codingAgentProfiles.some((profile) => profile.id === nextProfile)) {
    throw new Error("Unknown coding agent");
  }
  activeAgentProfile = nextProfile;
  agentMode = nextProfile;
  saveActiveAgentProfile();
  saveAgentMode();
  syncModeUi();
  if (!profileEventsSuspended) setAgentProfileDirty(true);
  appendAgentMessage("agent", `Active coding agent: ${activeProfile().name}\n${activeProfile().description}`);
}

function isLocalHttpUrl(rawUrl: string): boolean {
  try {
    const url = new URL(rawUrl);
    return (
      ["http:", "https:"].includes(url.protocol) &&
      ["localhost", "127.0.0.1", "[::1]", "::1"].includes(url.hostname)
    );
  } catch {
    return false;
  }
}

function normalizeLocalMcpUrl(rawUrl: string): string {
  const trimmed = rawUrl.trim();
  if (!trimmed) throw new Error("Enter a local MCP URL");
  const url = normalizeUrl(trimmed);
  if (!isLocalHttpUrl(url)) {
    throw new Error("Only local MCP hosts are supported for now: localhost, 127.0.0.1, or ::1");
  }
  return url;
}

function renderMcpServers() {
  const list = $("agent-mcp-list");
  list.replaceChildren();

  if (mcpServers.length === 0) {
    const empty = document.createElement("div");
    empty.className = "empty-state";
    empty.textContent = "No local MCP servers added";
    list.append(empty);
    return;
  }

  for (const server of mcpServers) {
    const row = document.createElement("div");
    const main = document.createElement("span");
    const actions = document.createElement("div");
    const probe = document.createElement("button");
    const remove = document.createElement("button");

    row.className = "agent-mcp-row";
    main.textContent = `${server.name} ${server.url}`;
    actions.className = "button-row";
    probe.type = "button";
    probe.textContent = "Tools";
    probe.addEventListener("click", () => callLocalMcpServer(server.url, "tools/list", {}).catch(showAgentError));
    remove.type = "button";
    remove.className = "agent-item-delete";
    remove.textContent = "Delete";
    remove.setAttribute("aria-label", `Delete MCP server ${server.name}`);
    remove.addEventListener("click", () => {
      void confirmAgentItemDeletion("MCP server", server.name, () => {
        mcpServers = mcpServers.filter((candidate) => candidate.id !== server.id);
        saveMcpServers();
        renderMcpServers();
      }).catch(showAgentError);
    });
    actions.append(probe, remove);
    row.append(main, actions);
    list.append(row);
  }
}

function addMcpServer() {
  const input = $("agent-mcp-url") as HTMLInputElement;
  const url = normalizeLocalMcpUrl(input.value);
  if (mcpServers.some((server) => server.url === url)) {
    setAgentStatus("MCP server already added");
    return;
  }

  const parsed = new URL(url);
  mcpServers = [
    ...mcpServers,
    {
      id: crypto.randomUUID(),
      name: `${parsed.hostname}:${parsed.port || (parsed.protocol === "https:" ? "443" : "80")}`,
      url,
      addedAt: Date.now(),
    },
  ];
  input.value = "";
  saveMcpServers();
  renderMcpServers();
  appendAgentMessage("tool", `Added local MCP server: ${url}`);
}

function agentPanel(): HTMLElement | null {
  return document.querySelector<HTMLElement>("#app-view-agent .agent-panel");
}

function setAgentPaneCollapsed(side: "left" | "right", collapsed: boolean, announce = true) {
  const panel = agentPanel();
  if (!panel) return;
  const className = side === "left" ? "agent-left-collapsed" : "agent-right-collapsed";
  panel.classList.toggle(className, collapsed);

  const toggle = document.getElementById(side === "left" ? "agent-toggle-left" : "agent-toggle-right");
  const bar = document.getElementById(side === "left" ? "agent-left-bar" : "agent-right-bar");
  const expanded = !collapsed;
  toggle?.setAttribute("aria-pressed", String(expanded));
  bar?.setAttribute("aria-expanded", String(expanded));
  if (announce) {
    setAgentStatus(`${side === "left" ? "Session panels" : "Inspector panels"} ${collapsed ? "collapsed" : "expanded"}`);
  }
}

function toggleAgentPane(side: "left" | "right") {
  const panel = agentPanel();
  if (!panel) return;
  const className = side === "left" ? "agent-left-collapsed" : "agent-right-collapsed";
  setAgentPaneCollapsed(side, !panel.classList.contains(className));
}

function bindAgentEvents() {
  $("agent-saved-profile-select").addEventListener("change", () => {
    const profileId = ($("agent-saved-profile-select") as HTMLSelectElement).value;
    void runAgentProfileOperation(() => switchSavedAgentProfile(profileId)).catch(showAgentError);
  });
  $("agent-profile-new").addEventListener("click", () =>
    void runAgentProfileOperation(createNewAgentProfile).catch(showAgentError));
  $("agent-profile-save").addEventListener("click", () =>
    void runAgentProfileOperation(saveCurrentAgentProfile).catch(showAgentError));
  $("agent-profile-duplicate").addEventListener("click", () =>
    void runAgentProfileOperation(duplicateCurrentAgentProfile).catch(showAgentError));
  $("agent-profile-rename").addEventListener("click", () => {
    try {
      renameCurrentAgentProfile();
    } catch (error) {
      showAgentError(error);
    }
  });
  $("agent-profile-delete").addEventListener("click", () =>
    void runAgentProfileOperation(deleteCurrentAgentProfile).catch(showAgentError));
  $("agent-profile-default").addEventListener("click", () => {
    try {
      setCurrentAgentProfileDefault();
    } catch (error) {
      showAgentError(error);
    }
  });
  $("agent-profile-reset").addEventListener("click", () =>
    void runAgentProfileOperation(resetCurrentAgentProfile).catch(showAgentError));
  $("agent-profile-export").addEventListener("click", () => {
    try {
      exportCurrentAgentProfile();
    } catch (error) {
      showAgentError(error);
    }
  });
  $("agent-profile-import").addEventListener("click", () => {
    ($("agent-profile-import-file") as HTMLInputElement).click();
  });
  $("agent-profile-import-file").addEventListener("change", () => {
    const input = $("agent-profile-import-file") as HTMLInputElement;
    const file = input.files?.[0];
    input.value = "";
    if (file) void runAgentProfileOperation(() => importAgentProfileFile(file)).catch(showAgentError);
  });
  for (const id of [
    "agent-profile-name",
    "agent-system-instructions",
    "agent-task-instructions",
    "agent-goal",
    "agent-workspace-root-input",
    "agent-temperature",
    "agent-timeout-seconds",
    "agent-retry-count",
    "agent-max-steps",
  ]) {
    $(id).addEventListener("input", () => {
      if (!profileEventsSuspended) setAgentProfileDirty(true);
    });
  }
  $("agent-new-task").addEventListener("click", () => {
    startNewAgentTask();
  });
  $("agent-history").addEventListener("click", () => {
    openAgentHistory();
  });
  $("agent-clear-current").addEventListener("click", () => void clearCurrentAgentConversation().catch(showAgentError));
  $("agent-history-clear-current").addEventListener("click", () =>
    void clearCurrentAgentConversation().catch(showAgentError));
  $("agent-history-delete-all").addEventListener("click", () => void deleteAllAgentHistory().catch(showAgentError));
  $("agent-history-close").addEventListener("click", closeAgentHistory);
  $("agent-history-overlay").addEventListener("pointerdown", (event) => {
    if (event.target === $("agent-history-overlay")) closeAgentHistory();
  });
  $("agent-history-overlay").addEventListener("keydown", (event) => {
    if ((event as KeyboardEvent).key === "Escape") {
      event.preventDefault();
      closeAgentHistory();
    }
  });
  $("agent-history-list").addEventListener("click", (event) => {
    const target = event.target as HTMLElement;
    const deleteButton = target.closest<HTMLButtonElement>(".agent-history-delete");
    if (deleteButton?.dataset.taskId) {
      void deleteAgentTask(deleteButton.dataset.taskId).catch(showAgentError);
      return;
    }
    const openButton = target.closest<HTMLButtonElement>(".agent-history-open");
    if (openButton?.dataset.taskId) resumeAgentTask(openButton.dataset.taskId);
  });
  $("agent-manager").addEventListener("click", () => {
    const enabledSkills = skills.filter((skill) => skill.enabled).map((skill) => skill.name).join(", ") || "none";
    const enabledSubagents = subagents.filter((agent) => agent.enabled).map((agent) => agent.name).join(", ") || "none";
    const agents = codingAgentProfiles
      .map((profile) => `${profile.id === activeAgentProfile ? "*" : " "} ${profile.name}: ${profile.description}`)
      .join("\n");
    runAgentShell("git status --short --branch; git worktree list")
      .then((output) => {
        appendAgentMessage("agent", `Agent manager\nActive: ${activeProfile().name}\nSkills: ${enabledSkills}\nSubagents: ${enabledSubagents}\n\nBuilt-in coding agents:\n${agents}\n\n${output}`);
      })
      .catch(showAgentError);
  });
  $("agent-settings").addEventListener("click", () => {
    const panel = agentPanel();
    panel?.classList.toggle("agent-settings-open");
    const settingsOpen = Boolean(panel?.classList.contains("agent-settings-open"));
    $("agent-settings").setAttribute("aria-pressed", String(settingsOpen));
    if (settingsOpen) {
      setAgentPaneCollapsed("right", false, false);
    }
    setAgentStatus(settingsOpen ? "Settings open" : "Settings closed");
  });
  $("agent-toggle-left").addEventListener("click", () => toggleAgentPane("left"));
  $("agent-toggle-right").addEventListener("click", () => toggleAgentPane("right"));
  $("agent-left-bar").addEventListener("click", () => setAgentPaneCollapsed("left", false));
  $("agent-right-bar").addEventListener("click", () => setAgentPaneCollapsed("right", false));
  $("agent-pick-workspace-root").addEventListener("click", () => {
    open({ directory: true, multiple: false })
      .then((selected) => {
        if (typeof selected !== "string") return;
        ($("agent-workspace-root-input") as HTMLInputElement).value = selected;
        return saveAgentWorkspaceRoot(selected);
      })
      .catch(showAgentError);
  });
  $("agent-save-workspace-root").addEventListener("click", () => {
    saveAgentWorkspaceRoot(($("agent-workspace-root-input") as HTMLInputElement).value).catch(showAgentError);
  });
  $("agent-workspace-root-input").addEventListener("keydown", (event) => {
    if ((event as KeyboardEvent).key !== "Enter") return;
    event.preventDefault();
    saveAgentWorkspaceRoot(($("agent-workspace-root-input") as HTMLInputElement).value).catch(showAgentError);
  });
  $("agent-enhance-prompt").addEventListener("click", () => {
    const input = $("agent-command-input") as HTMLTextAreaElement;
    const prompt = input.value.trim();
    if (!prompt) {
      setAgentStatus("Type a prompt to enhance");
      return;
    }
    input.value = `Goal: ${prompt}\n\nPlease reason about the smallest safe change, identify files to inspect, implement it, and verify the result.`;
    input.focus();
    setAgentStatus("Prompt enhanced");
  });
  $("agent-voice-input").addEventListener("click", () => {
    const SpeechRecognition = (
      window as typeof window & {
        SpeechRecognition?: SpeechRecognitionConstructor;
        webkitSpeechRecognition?: SpeechRecognitionConstructor;
      }
    ).SpeechRecognition ?? (
      window as typeof window & {
        SpeechRecognition?: SpeechRecognitionConstructor;
        webkitSpeechRecognition?: SpeechRecognitionConstructor;
      }
    ).webkitSpeechRecognition;
    if (!SpeechRecognition) {
      setAgentStatus("Voice input is not available in this WebView");
      return;
    }
    const recognition = new SpeechRecognition();
    recognition.continuous = false;
    recognition.interimResults = false;
    recognition.onresult = (event) => {
      const text = Array.from(event.results)
        .map((result) => result[0]?.transcript ?? "")
        .join(" ")
        .trim();
      const input = $("agent-command-input") as HTMLTextAreaElement;
      input.value = [input.value.trim(), text].filter(Boolean).join(" ");
      setAgentStatus("Voice input captured");
    };
    recognition.onerror = () => setAgentStatus("Voice input failed");
    recognition.start();
    setAgentStatus("Listening");
  });
  $("agent-mode-plan").addEventListener("click", () => setExecutionMode("plan"));
  $("agent-mode-edit").addEventListener("click", () => setExecutionMode("edit"));
  $("agent-set-goal").addEventListener("click", () => setAgentGoal(($("agent-goal") as HTMLTextAreaElement).value));
  $("agent-profile-select").addEventListener("change", () => {
    setActiveAgentProfile(($("agent-profile-select") as HTMLSelectElement).value as AgentMode);
  });
  $("agent-thinking-level").addEventListener("change", () => {
    agentThinkingLevel = normalizeThinkingLevel(($("agent-thinking-level") as HTMLSelectElement).value);
    saveAgentThinkingLevel();
    syncAgentThinkingUi();
    setAgentProfileDirty(true);
    setAgentStatus(`Agent thinking set to ${($("agent-thinking-level") as HTMLSelectElement).selectedOptions[0]?.textContent ?? agentThinkingLevel}`);
  });
  $("agent-mode-tabs").addEventListener("click", (event) => {
    const button = (event.target as HTMLElement).closest<HTMLButtonElement>("[data-agent-mode]");
    if (!button?.dataset.agentMode) return;
    try {
      setAgentMode(button.dataset.agentMode as AgentMode);
    } catch (error) {
      showAgentError(error);
    }
  });
  $("agent-auto-compact").addEventListener("change", () => {
    autoCompact = ($("agent-auto-compact") as HTMLInputElement).checked;
    saveAgentContext();
    setAgentProfileDirty(true);
    setAgentStatus(autoCompact ? "Auto compact enabled" : "Auto compact disabled");
  });
  $("agent-compact-context").addEventListener("click", () => compactAgentContext().catch(showAgentError));
  $("agent-add-todo").addEventListener("click", () => {
    const input = $("agent-todo-input") as HTMLInputElement;
    addTodo(input.value)
      .then(() => {
        input.value = "";
      })
      .catch(showAgentError);
  });
  $("agent-add-skill").addEventListener("click", () => {
    const name = $("agent-skill-name") as HTMLInputElement;
    const instructions = $("agent-skill-instructions") as HTMLInputElement;
    loadSkill(name.value, instructions.value)
      .then(() => {
        name.value = "";
        instructions.value = "";
      })
      .catch(showAgentError);
  });
  $("agent-add-subagent").addEventListener("click", () => {
    const name = $("agent-subagent-name") as HTMLInputElement;
    const role = $("agent-subagent-role") as HTMLInputElement;
    startSubagent(name.value, role.value)
      .then(() => {
        name.value = "";
        role.value = "";
      })
      .catch(showAgentError);
  });
  $("agent-run-shell").addEventListener("click", () =>
    runAgentShell(($("agent-shell-command") as HTMLInputElement).value).catch(showAgentError),
  );
  $("agent-refresh-workspace").addEventListener("click", () => refreshAgentWorkspace());
  $("agent-clear-chat").addEventListener("click", () => {
    void clearCurrentAgentConversation().catch(showAgentError);
  });
  $("agent-run-command").addEventListener("click", () => runAgentCommand().catch(showAgentError));
  $("agent-command-input").addEventListener("keydown", (event) => {
    const keyEvent = event as KeyboardEvent;
    if (keyEvent.key !== "Enter" || keyEvent.isComposing) return;
    // Shift+Enter inserts a newline; plain Enter (or Ctrl/Cmd+Enter) sends.
    if (keyEvent.shiftKey) return;
    event.preventDefault();
    void runAgentCommand().catch(showAgentError);
  });
  $("agent-read-file").addEventListener("click", () => readAgentPath().catch(showAgentError));
  $("agent-edit-file").addEventListener("click", () => editAgentFile().catch(showAgentError));
  $("agent-insert-file").addEventListener("click", () =>
    insertContentInAgentFile(
      ($("agent-path") as HTMLInputElement).value,
      ($("agent-file-content") as HTMLTextAreaElement).value,
      null,
    ).catch(showAgentError),
  );
  $("agent-delete-file").addEventListener("click", () => deleteAgentPath().catch(showAgentError));
  $("agent-copy-file").addEventListener("click", () => copyAgentFile().catch(showAgentError));
  $("agent-move-file").addEventListener("click", () => moveAgentFile().catch(showAgentError));
  $("agent-paste-file").addEventListener("click", () => pasteAgentFile().catch(showAgentError));
  $("agent-browse-url").addEventListener("click", () => browseAgentUrl().catch(showAgentError));
  $("agent-add-mcp-server").addEventListener("click", () => {
    try {
      addMcpServer();
    } catch (error) {
      showAgentError(error);
    }
  });
  $("agent-url").addEventListener("keydown", (event) => {
    if ((event as KeyboardEvent).key !== "Enter") return;
    event.preventDefault();
    void browseAgentUrl().catch(showAgentError);
  });
  $("agent-mcp-url").addEventListener("keydown", (event) => {
    if ((event as KeyboardEvent).key !== "Enter") return;
    event.preventDefault();
    try {
      addMcpServer();
    } catch (error) {
      showAgentError(error);
    }
  });
}

function showAgentError(error: unknown) {
  const message = String(error);
  setAgentStatus(message);
  setAgentOutput(message);
  appendAgentMessage("tool", message);
}

export function initAgent() {
  agentPermissions = loadAgentPermissions();
  yoloMode = window.localStorage.getItem(agentYoloModeStorageKey) === "true";
  autoAccept = window.localStorage.getItem(agentAutoAcceptStorageKey) === "true";
  agentGoal = window.localStorage.getItem(agentGoalStorageKey) ?? "";
  agentMode = (window.localStorage.getItem(agentModeStorageKey) as AgentMode | null) ?? "coder";
  activeAgentProfile =
    (window.localStorage.getItem(agentActiveProfileStorageKey) as AgentMode | null) ?? agentMode;
  agentThinkingLevel = normalizeThinkingLevel(window.localStorage.getItem(agentThinkingLevelStorageKey));
  agentExecutionMode =
    window.localStorage.getItem(agentExecutionModeStorageKey) === "plan" ? "plan" : "edit";
  if (!codingAgentProfiles.some((profile) => profile.id === activeAgentProfile)) {
    activeAgentProfile = "coder";
  }
  agentMode = activeAgentProfile;
  autoCompact = window.localStorage.getItem(agentAutoCompactStorageKey) === "true";
  contextSummary = window.localStorage.getItem(agentContextSummaryStorageKey) ?? "";
  mcpServers = loadMcpServers();
  todos = loadStoredArray<AgentTodo>(agentTodosStorageKey);
  skills = loadStoredArray<AgentSkill>(agentSkillsStorageKey);
  subagents = loadStoredArray<AgentSubagent>(agentSubagentsStorageKey);
  taskHistory = loadTaskHistory();
  currentTaskId = taskHistory[0]?.id ?? "";
  const profileLoad = loadAgentProfileStore(
    window.localStorage.getItem(agentProfilesStorageKey),
    window.localStorage.getItem(agentProfilesBackupStorageKey),
    {
      role: activeAgentProfile,
      goal: agentGoal,
      workspaceRoot: "",
      thinkingLevel: agentThinkingLevel,
      executionMode: agentExecutionMode,
      permissions: agentPermissions,
      autoApprove: autoAccept,
      yoloMode,
      autoCompact,
    },
  );
  agentProfileStore = profileLoad.store;
  agentProfileNeedsWorkspaceMigration = profileLoad.migrated;
  const initialProfile =
    agentProfileStore.profiles.find((profile) => profile.id === agentProfileStore.activeProfileId) ??
    agentProfileStore.profiles[0];
  void applySavedAgentProfile(initialProfile).catch(showAgentError);
  if (profileLoad.warning) setAgentProfileFeedback(profileLoad.warning, "warn");
  syncPermissionControls();
  syncExecutionModeUi();
  syncAgentProfileUi();
  syncAgentThinkingUi();
  syncGoalUi();
  syncModeUi();
  syncContextUi();
  renderTodos();
  renderSkills();
  renderSubagents();
  renderMcpServers();
  if (taskHistory[0]?.messages.length) {
    renderTaskTranscript(taskHistory[0]);
  } else {
    renderEmptyAgentConversation();
  }
  setAgentPaneCollapsed("left", true, false);
  setAgentPaneCollapsed("right", true, false);
  bindPermissionControls();
  bindAgentEvents();
  window.addEventListener("beforeunload", (event) => {
    if (!agentProfileDirty) return;
    event.preventDefault();
    event.returnValue = "";
  });
  void refreshPiAgentStatus();
  void refreshAgentWorkspace();
  void refreshDiscoveredSkills();
}
