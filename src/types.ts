export type Tone = "info" | "ok" | "warn" | "error";

export interface ModelEntry {
  id: string;
  name: string;
  path: string;
  source: string;
  sizeBytes: number;
  metadata?: GgufModelMetadata | null;
}

export interface GgufModelMetadata {
  architecture?: string | null;
  contextLength?: number | null;
  blockCount?: number | null;
  embeddingLength?: number | null;
  attentionHeadCount?: number | null;
  attentionHeadCountKv?: number | null;
  attentionKeyLength?: number | null;
  attentionValueLength?: number | null;
}

export interface ModelProfile {
  modelPath: string;
  name: string;
  server: ServerConfig;
}

export interface ServerConfig {
  modelPath: string;
  host: string;
  port: number;
  ctxSize: number;
  gpuLayers: string;
  threads: number;
  batchSize: number;
  ubatchSize: number;
  parallel: number;
  enableKvCacheOptions: boolean;
  cacheTypeK: string;
  cacheTypeV: string;
  flashAttention: string;
  kvu: boolean;
  enableGpuMemoryOptions: boolean;
  kvOffload: string;
  noHost: boolean;
  opOffload: string;
  fit: string;
  fitTarget: string;
  fitCtx: number;
  device: string;
  tensorSplit: string;
  splitMode: string;
  mainGpu: string;
  cpuMoe: boolean;
  enableSamplingOptions: boolean;
  temperature: string;
  topK: string;
  topP: string;
  minP: string;
  typicalP: string;
  repeatPenalty: string;
  presencePenalty: string;
  frequencyPenalty: string;
  enableSpeculativeOptions: boolean;
  specType: string;
  specDraftNMax: number;
  specDraftNMin: number;
  specDraftPMin: string;
  specDraftPSplit: string;
  noMmap: boolean;
  mlock: boolean;
  specNgramModNMatch: number;
  specNgramModNMin: number;
  specNgramModNMax: number;
  specDraftModelPath: string;
  noCpuMoe: number;
  enableReasoningOptions: boolean;
  preserveThinking?: boolean;
  reasoningPreserve: string;
  reasoningFormat: string;
  reasoningBudget: string;
  chatTemplateKwargs: string;
  reasoning: string;
  enableMultimodalOptions: boolean;
  mmproj: string;
  embeddings: boolean;
  toolsAll: boolean;
  jinja: boolean;
  verbose: boolean;
  terminalMode: string;
  extraArgs: string;
}

export interface AppConfig {
  llamaServerPath: string;
  llamaCliPath: string;
  llamaBenchPath: string;
  agentWorkspaceRoot: string;
  modelDir: string;
  hfToken: string;
  manualModels: ModelEntry[];
  modelProfiles: ModelProfile[];
  server: ServerConfig;
}

export interface ToolDiscovery {
  llamaServer: string | null;
  llamaCli: string | null;
  llamaBench: string | null;
  hfCli: string | null;
  huggingfaceCli: string | null;
}

export interface ServerStatus {
  running: boolean;
  pid: number | null;
  command: string;
  url: string;
  modelPath: string;
  logPath: string;
  startedAt: number | null;
}

export interface CommandOutput {
  success: boolean;
  statusCode: number | null;
  command: string;
  stdout: string;
  stderr: string;
}

export interface LlamaServerProcess {
  pid: number;
  commandLine: string;
}

export interface DownloadRequest {
  repoId: string;
  pattern: string;
  revision: string;
  targetDir: string;
  token: string;
  force: boolean;
  maxWorkers: number;
}

export interface HfRepoFile {
  path: string;
  sizeBytes: number | null;
}

export type HfModelSort = "trending" | "updated" | "downloads";
export type AppTab = "control" | "benchmark" | "webui" | "agent";
export type ConfirmKind = "warning" | "danger";
export type BenchmarkMode = "performance" | "evaluation";
export type EvalBenchmarkType =
  | "humaneval"
  | "mbpp"
  | "gsm8k"
  | "mmlu"
  | "arc"
  | "hellaswag"
  | "truthfulqa"
  | "winogrande";

export interface EvalBenchmarkSettings {
  benchmarkType: EvalBenchmarkType;
  sampleCount: string;
  temperature: string;
}

export interface EvalSampleResult {
  index: number;
  prompt: string;
  expected: string;
  actual: string;
  passed: boolean;
}

export interface EvalBenchmarkResult {
  benchmarkType: EvalBenchmarkType;
  totalSamples: number;
  passedSamples: number;
  score: number;
  scorePercent: number;
  samples: EvalSampleResult[];
  elapsedMs: number;
  error?: string;
}
export type BenchmarkPreset = "quick" | "standard" | "long";
export type AgentPermission = "allow" | "ask" | "deny";
export type AgentTool =
  | "read"
  | "edit"
  | "move"
  | "copy"
  | "paste"
  | "browse"
  | "shell"
  | "todo"
  | "skill"
  | "subagent"
  | "externalDirectory";

export interface HfModelSummary {
  id: string;
  downloads: number | null;
  likes: number | null;
  lastModified: string | null;
  createdAt: string | null;
  pipelineTag: string | null;
  libraryName: string | null;
  trendingScore: number | null;
}

export interface LlamaCppInstallRequest {
  package: string;
  targetDir: string;
}

export interface LlamaCppInstallResult {
  releaseTag: string;
  assetName: string;
  installDir: string;
  llamaServerPath: string;
  llamaCliPath: string;
  llamaBenchPath: string;
  command: string;
  stdout: string;
  stderr: string;
}

export interface ConfirmActionOptions {
  title: string;
  message: string;
  okLabel: string;
  cancelLabel: string;
  kind?: ConfirmKind;
}

export interface BenchmarkSettings {
  preset: BenchmarkPreset;
  runs: string;
  generateTokens: string;
  prompt: string;
}

export interface BenchmarkPrefillResult {
  targetTokens: number;
  promptTokens: number;
  promptMs: number;
  tokensPerSecond: number;
}

export interface BenchmarkRunResult {
  index: number;
  prefill: BenchmarkPrefillResult[];
  generationTokens: number;
  generationMs: number;
  generationTokensPerSecond: number;
  totalMs: number;
  text: string;
  error?: string;
}

export interface AgentPermissions {
  read: AgentPermission;
  edit: AgentPermission;
  move: AgentPermission;
  copy: AgentPermission;
  paste: AgentPermission;
  browse: AgentPermission;
  shell: AgentPermission;
  todo: AgentPermission;
  skill: AgentPermission;
  subagent: AgentPermission;
  externalDirectory: AgentPermission;
}

export interface AgentPathInfo {
  workspaceRoot: string;
  path: string;
  exists: boolean;
  isFile: boolean;
  isDirectory: boolean;
  sizeBytes: number | null;
  modifiedAt: number | null;
  external: boolean;
}

export interface AgentFileEntry {
  name: string;
  path: string;
  isDirectory: boolean;
  sizeBytes: number | null;
}

export interface AgentReadResult {
  info: AgentPathInfo;
  content: string | null;
  entries: AgentFileEntry[] | null;
}

export interface AgentWriteRequest {
  path: string;
  content: string;
  create: boolean;
}

export interface AgentWriteResult {
  info: AgentPathInfo;
  bytesWritten: number;
}

export interface AgentTransferRequest {
  fromPath: string;
  toPath: string;
  overwrite: boolean;
}

export interface AgentTransferResult {
  from: AgentPathInfo;
  to: AgentPathInfo;
}

export interface AgentMcpServer {
  id: string;
  url: string;
  name: string;
  addedAt: number;
}

export type AgentMode = "architect" | "ask" | "coder" | "debugger" | "reviewer" | "tester" | "orchestrator";

export type AgentExecutionMode = "plan" | "edit";
export type AgentThinkingLevel = "disabled" | "low" | "medium" | "high" | "xhigh";

export interface AgentProfile {
  id: string;
  schemaVersion: 3;
  name: string;
  role: AgentMode;
  systemInstructions: string;
  taskInstructions: string;
  goal: string;
  workspaceRoot: string;
  thinkingLevel: AgentThinkingLevel;
  executionMode: AgentExecutionMode;
  permissions: AgentPermissions;
  autoApprove: boolean;
  yoloMode: boolean;
  autoCompact: boolean;
  temperature: number;
  timeoutSeconds: number;
  retryCount: number;
  maxSteps: number;
  createdAt: number;
  updatedAt: number;
}

export interface AgentProfileStore {
  schemaVersion: 3;
  activeProfileId: string;
  defaultProfileId: string;
  profiles: AgentProfile[];
}

export interface AgentTodo {
  id: string;
  text: string;
  done: boolean;
  createdAt: number;
}

export interface AgentShellRequest {
  command: string;
  timeoutSeconds: number;
}

export interface AgentPiRequest {
  prompt: string;
  extraArgs: string[];
  timeoutSeconds: number;
  temperature: number;
}

export interface AgentPiStatus {
  available: boolean;
  command: string;
  version: string;
  checked: string[];
}

export interface AgentSkill {
  id: string;
  name: string;
  instructions: string;
  enabled: boolean;
  createdAt: number;
}

export interface AgentSkillFile {
  name: string;
  description: string;
  path: string;
  source: string;
  modes: string;
}

export interface AgentSubagent {
  id: string;
  name: string;
  role: string;
  enabled: boolean;
  createdAt: number;
}
