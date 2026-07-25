import type { AgentPermissions, ServerConfig } from "./types";

export const preserveThinkingDefault = '{"preserve_thinking": true}';
export const themeStorageKey = "localllm-theme";
export const paneWidthStorageKey = "localllm-left-pane-width";
export const benchmarkSettingsStorageKey = "localllm-benchmark-settings";
export const agentPermissionsStorageKey = "localllm-agent-permissions";
export const agentYoloModeStorageKey = "localllm-agent-yolo-mode";
export const agentAutoAcceptStorageKey = "localllm-agent-auto-accept";
export const agentGoalStorageKey = "localllm-agent-goal";
export const agentModeStorageKey = "localllm-agent-mode";
export const agentExecutionModeStorageKey = "localllm-agent-execution-mode";
export const agentActiveProfileStorageKey = "localllm-agent-active-profile";
export const agentThinkingLevelStorageKey = "localllm-agent-thinking-level";
export const agentTodosStorageKey = "localllm-agent-todos";
export const agentSkillsStorageKey = "localllm-agent-skills";
export const agentSubagentsStorageKey = "localllm-agent-subagents";
export const agentAutoCompactStorageKey = "localllm-agent-auto-compact";
export const agentContextSummaryStorageKey = "localllm-agent-context-summary";
export const agentMcpServersStorageKey = "localllm-agent-mcp-servers";
export const agentTaskHistoryStorageKey = "localllm-agent-task-history";
export const agentProfilesStorageKey = "localllm-agent-profiles-v2";
export const agentProfilesBackupStorageKey = "localllm-agent-profiles-v2-backup";
export const benchmarkDefaultPrompt =
  "Write a concise field report about a local AI server benchmark. Include one bottleneck, one strength, and one practical tuning idea.";
export const benchmarkPrefillTargets = [2048, 4098, 8192];
export const themes = new Set(["spotify", "sage", "graphite", "paper", "webmcp", "liquidglass"]);
export const minLeftPaneWidth = 300;
export const minRightPaneWidth = 320;

export const defaultAgentPermissions: AgentPermissions = {
  read: "allow",
  edit: "ask",
  move: "ask",
  copy: "ask",
  paste: "ask",
  browse: "ask",
  shell: "ask",
  todo: "allow",
  skill: "ask",
  subagent: "ask",
  externalDirectory: "ask",
};

export const defaultServerConfig: ServerConfig = {
  modelPath: "",
  host: "127.0.0.1",
  port: 8080,
  ctxSize: 4096,
  gpuLayers: "",
  threads: 0,
  batchSize: 2048,
  ubatchSize: 512,
  parallel: -1,
  enableKvCacheOptions: true,
  cacheTypeK: "q8_0",
  cacheTypeV: "q8_0",
  flashAttention: "",
  enableGpuMemoryOptions: false,
  kvOffload: "",
  noHost: false,
  opOffload: "",
  fit: "",
  fitTarget: "",
  fitCtx: 0,
  device: "",
  tensorSplit: "",
  splitMode: "",
  mainGpu: "",
  cpuMoe: false,
  enableSamplingOptions: false,
  temperature: "",
  topK: "",
  topP: "",
  minP: "",
  typicalP: "",
  repeatPenalty: "",
  presencePenalty: "",
  frequencyPenalty: "",
  enableSpeculativeOptions: true,
  specType: "",
  specDraftNMax: 0,
  specDraftNMin: 0,
  specDraftPMin: "",
  specDraftPSplit: "",
  noMmap: false,
  mlock: false,
  specNgramModNMatch: 0,
  specNgramModNMin: 0,
  specNgramModNMax: 0,
  noCpuMoe: 0,
  enableReasoningOptions: true,
  preserveThinking: true,
  reasoningFormat: "",
  reasoningBudget: "",
  chatTemplateKwargs: preserveThinkingDefault,
  reasoning: "",
  enableMultimodalOptions: true,
  mmproj: "",
  embeddings: false,
  toolsAll: false,
  jinja: false,
  verbose: false,
  terminalMode: "visible",
  extraArgs: "",
};
