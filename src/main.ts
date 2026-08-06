import { invoke } from "@tauri-apps/api/core";
import { open } from "@tauri-apps/plugin-dialog";
import { openPath, openUrl } from "@tauri-apps/plugin-opener";
import { initAgent } from "./agent";
import {
  agentAutoAcceptStorageKey,
  agentAutoCompactStorageKey,
  agentContextSummaryStorageKey,
  agentGoalStorageKey,
  agentMcpServersStorageKey,
  agentModeStorageKey,
  agentPermissionsStorageKey,
  agentSkillsStorageKey,
  agentSubagentsStorageKey,
  agentTodosStorageKey,
  agentYoloModeStorageKey,
  benchmarkDefaultPrompt,
  benchmarkPrefillTargets,
  benchmarkSettingsStorageKey,
  defaultServerConfig,
  evalBenchmarkDatasets,
  evalBenchmarkLabels,
  evalSettingsStorageKey,
  minLeftPaneWidth,
  minRightPaneWidth,
  paneWidthStorageKey,
  preserveThinkingDefault,
  themeStorageKey,
  themes,
} from "./constants";
import { confirmAction } from "./confirm-dialog";
import { $, bindButtonRipples, bindSelectionGuard } from "./dom";
import { initWindowControls } from "./window-controls";
import type {
  AppConfig,
  AppTab,
  BenchmarkMode,
  BenchmarkPrefillResult,
  BenchmarkPreset,
  BenchmarkRunResult,
  BenchmarkSettings,
  CommandOutput,
  DownloadRequest,
  EvalBenchmarkResult,
  EvalBenchmarkSettings,
  EvalBenchmarkType,
  EvalSampleResult,
  GgufModelMetadata,
  HfModelSort,
  HfModelSummary,
  HfRepoFile,
  LlamaCppInstallRequest,
  LlamaCppInstallResult,
  LlamaServerProcess,
  ModelEntry,
  ModelProfile,
  ServerConfig,
  ServerStatus,
  Tone,
  ToolDiscovery,
} from "./types";

const appMessage = $("app-message");
const workspace = $("workspace");
const workspaceResizer = $("workspace-resizer");
const modelSelect = $("model-select") as HTMLSelectElement;
const modelList = $("model-list");
const modelLoading = $("model-loading");
const commandPreview = $("command-preview");
const commandHelper = $("command-helper");
const webUiFrame = $("web-ui-frame") as HTMLIFrameElement;
const webUiUrl = $("web-ui-url");
const webUiTab = $("app-tab-webui") as HTMLButtonElement;
const reloadWebUiButton = $("reload-web-ui") as HTMLButtonElement;
const benchmarkStatus = $("benchmark-status");
const benchmarkOutput = $("benchmark-output");
const benchmarkResults = $("benchmark-results");
const benchmarkPrompt = $("benchmark-prompt") as HTMLTextAreaElement;
const benchmarkSample = $("benchmark-sample");
const benchmarkChart = $("benchmark-chart");
const benchmarkPrefillMetricIds: Record<number, string> = {
  2048: "benchmark-prefill-2048-tokens-per-second",
  4098: "benchmark-prefill-4098-tokens-per-second",
  8192: "benchmark-prefill-8192-tokens-per-second",
};
const benchmarkPrefillMetrics = Object.fromEntries(
  Object.entries(benchmarkPrefillMetricIds).map(([target, id]) => [Number(target), $(id)]),
) as Record<number, HTMLElement>;
const benchmarkGenerateTokensPerSecond = $("benchmark-generate-tokens-per-second");
const benchmarkTotalTime = $("benchmark-total-time");
const benchmarkOutputTokens = $("benchmark-output-tokens");
const benchmarkShareSummary = $("benchmark-share-summary") as HTMLTextAreaElement;
const benchmarkPerformanceSettings = $("benchmark-performance-settings");
const benchmarkEvaluationSettings = $("benchmark-evaluation-settings");
const evalResults = $("eval-results");
const evalScorePercent = $("eval-score-percent");
const evalPassedCount = $("eval-passed-count");
const evalTotalCount = $("eval-total-count");
const evalElapsedTime = $("eval-elapsed-time");
const evalSamplesContainer = $("eval-samples-container");
const evalSamplesList = $("eval-samples-list");
const vramTotal = $("vram-total");
const vramDetail = $("vram-detail");
const downloadOutput = $("download-output");
const downloadFileSelect = $("hf-file-select") as HTMLSelectElement;
const hfModelResults = $("hf-model-results");
const hfModelStatus = $("hf-model-status");
const llamaInstallOutput = $("llama-install-output");

let config: AppConfig;
let models: ModelEntry[] = [];
let tools: ToolDiscovery = {
  llamaServer: null,
  llamaCli: null,
  llamaBench: null,
  hfCli: null,
  huggingfaceCli: null,
};
let server: ServerStatus = {
  running: false,
  pid: null,
  command: "",
  url: "",
  modelPath: "",
  logPath: "",
  startedAt: null,
};
let previewTimer: number | undefined;
let previewSerial = 0;
let downloadFileListTimer: number | undefined;
let downloadFileListSerial = 0;
let hfModelSearchTimer: number | undefined;
let hfModelListSerial = 0;
let activeHfModelSort: HfModelSort = "trending";
let activeAppTab: AppTab = "control";
let modelScanSerial = 0;
let modelRenderSerial = 0;
let activeModelPath = "";
let selectedProfileOverrideIndex: number | null = null;
let creatingProfile = false;
let benchmarkAbortController: AbortController | null = null;
let benchmarkHistory: BenchmarkRunResult[] = [];
let evalAbortController: AbortController | null = null;
let evalBenchmarkHistory: EvalBenchmarkResult[] = [];
const modelLoadingTasks = new Set<string>();
const metadataLoads = new Set<string>();
const metadataLoaded = new Set<string>();

const optionalGroups = [
  { toggleId: "enable-kv-cache-options", groupId: "kv-cache-options" },
  { toggleId: "enable-gpu-memory-options", groupId: "gpu-memory-options" },
  { toggleId: "enable-sampling-options", groupId: "sampling-options" },
  { toggleId: "enable-speculative-options", groupId: "speculative-options" },
  { toggleId: "enable-reasoning-options", groupId: "reasoning-options" },
  { toggleId: "enable-multimodal-options", groupId: "multimodal-options" },
];

function setMessage(message: string, tone: Tone = "info") {
  appMessage.textContent = message;
  appMessage.dataset.tone = tone;
}

function applyTheme(theme: string) {
  const nextTheme = themes.has(theme) ? theme : "spotify";
  document.documentElement.dataset.theme = nextTheme;
  ($("theme-select") as HTMLSelectElement).value = nextTheme;
  window.localStorage.setItem(themeStorageKey, nextTheme);
}

function initTheme() {
  applyTheme(window.localStorage.getItem(themeStorageKey) ?? "spotify");
}

function workspaceContentWidth(): number {
  const rect = workspace.getBoundingClientRect();
  const styles = window.getComputedStyle(workspace);
  const padding =
    Number.parseFloat(styles.paddingLeft) + Number.parseFloat(styles.paddingRight);
  return Math.max(0, rect.width - padding);
}

function clampPaneWidth(width: number): number {
  const splitterWidth = workspaceResizer.getBoundingClientRect().width || 12;
  const maxWidth = workspaceContentWidth() - splitterWidth - minRightPaneWidth;
  if (maxWidth <= minLeftPaneWidth) {
    return Math.max(220, maxWidth);
  }
  return Math.min(Math.max(width, minLeftPaneWidth), maxWidth);
}

function setLeftPaneWidth(width: number, persist = true) {
  const nextWidth = clampPaneWidth(width);
  workspace.style.setProperty("--left-pane-width", `${nextWidth}px`);
  workspaceResizer.setAttribute("aria-valuenow", String(Math.round(nextWidth)));
  if (persist) {
    window.localStorage.setItem(paneWidthStorageKey, String(Math.round(nextWidth)));
  }
}

function leftPaneWidthFromPointer(event: PointerEvent): number {
  const rect = workspace.getBoundingClientRect();
  const styles = window.getComputedStyle(workspace);
  return event.clientX - rect.left - Number.parseFloat(styles.paddingLeft);
}

function currentLeftPaneWidth(): number {
  const raw = window.getComputedStyle(workspace).getPropertyValue("--left-pane-width").trim();
  const parsed = Number.parseFloat(raw);
  if (Number.isFinite(parsed)) return parsed;

  const firstPane = workspace.querySelector<HTMLElement>(".server-panel");
  return firstPane?.getBoundingClientRect().width ?? workspaceContentWidth() * 0.48;
}

function initPaneResize() {
  const storedWidth = Number(window.localStorage.getItem(paneWidthStorageKey));
  if (Number.isFinite(storedWidth) && storedWidth > 0) {
    window.requestAnimationFrame(() => setLeftPaneWidth(storedWidth, false));
  }

  workspaceResizer.addEventListener("pointerdown", (event) => {
    if (window.matchMedia("(max-width: 920px)").matches) return;
    event.preventDefault();
    workspaceResizer.classList.add("dragging");
    document.body.classList.add("resizing-panes");

    const handlePointerMove = (moveEvent: PointerEvent) => {
      setLeftPaneWidth(leftPaneWidthFromPointer(moveEvent));
    };
    const stopDragging = () => {
      workspaceResizer.classList.remove("dragging");
      document.body.classList.remove("resizing-panes");
      window.removeEventListener("pointermove", handlePointerMove);
      window.removeEventListener("pointerup", stopDragging);
      window.removeEventListener("pointercancel", stopDragging);
    };

    setLeftPaneWidth(leftPaneWidthFromPointer(event));
    window.addEventListener("pointermove", handlePointerMove);
    window.addEventListener("pointerup", stopDragging);
    window.addEventListener("pointercancel", stopDragging);
  });

  workspaceResizer.addEventListener("keydown", (event) => {
    if (!["ArrowLeft", "ArrowRight", "Home", "End"].includes(event.key)) return;
    event.preventDefault();

    const splitterWidth = workspaceResizer.getBoundingClientRect().width || 12;
    const maxWidth = workspaceContentWidth() - splitterWidth - minRightPaneWidth;
    if (event.key === "Home") {
      setLeftPaneWidth(minLeftPaneWidth);
    } else if (event.key === "End") {
      setLeftPaneWidth(maxWidth);
    } else {
      const delta = event.key === "ArrowLeft" ? -24 : 24;
      setLeftPaneWidth(currentLeftPaneWidth() + delta);
    }
  });

  window.addEventListener("resize", () => {
    const stored = Number(window.localStorage.getItem(paneWidthStorageKey));
    setLeftPaneWidth(Number.isFinite(stored) && stored > 0 ? stored : currentLeftPaneWidth(), false);
  });
}

function yieldToUi(): Promise<void> {
  return new Promise((resolve) => window.setTimeout(resolve, 0));
}

function value(id: string): string {
  return ($(id) as HTMLInputElement).value.trim();
}

function setValue(id: string, next: string | number) {
  ($(id) as HTMLInputElement).value = String(next);
}

function checked(id: string): boolean {
  return ($(id) as HTMLInputElement).checked;
}

function setChecked(id: string, next: boolean) {
  ($(id) as HTMLInputElement).checked = next;
}

function numberValue(id: string, fallback: number): number {
  const raw = Number(value(id));
  return Number.isFinite(raw) ? raw : fallback;
}

function formatBytes(bytes: number): string {
  if (!Number.isFinite(bytes) || bytes <= 0) return "0 B";
  const units = ["B", "KB", "MB", "GB", "TB"];
  let size = bytes;
  let index = 0;
  while (size >= 1024 && index < units.length - 1) {
    size /= 1024;
    index += 1;
  }
  return `${size.toFixed(index === 0 ? 0 : 1)} ${units[index]}`;
}

function formatCount(count: number | null | undefined): string {
  if (!Number.isFinite(count ?? NaN) || (count ?? 0) < 0) return "0";
  const value = count ?? 0;
  if (value >= 1_000_000) return `${(value / 1_000_000).toFixed(1)}M`;
  if (value >= 1_000) return `${(value / 1_000).toFixed(1)}K`;
  return String(value);
}

function formatShortDate(raw: string | null | undefined): string {
  if (!raw) return "";
  const date = new Date(raw);
  if (Number.isNaN(date.getTime())) return raw;
  return date.toLocaleDateString(undefined, { month: "short", day: "numeric", year: "numeric" });
}

function setActiveHfModelSort(sort: HfModelSort) {
  activeHfModelSort = sort;
  for (const button of $("hf-model-sort-tabs").querySelectorAll<HTMLButtonElement>("button[data-hf-sort]")) {
    button.setAttribute("aria-pressed", String(button.dataset.hfSort === sort));
  }
}

function serverWebUiUrl(): string {
  return server.url || `http://${value("host")}:${value("port")}`;
}

function syncWebUiFrame(force = false) {
  if (!server.running) {
    webUiUrl.textContent = "Server offline";
    webUiFrame.removeAttribute("data-loaded-url");
    if (webUiFrame.getAttribute("src") !== "about:blank") {
      webUiFrame.setAttribute("src", "about:blank");
    }
    return;
  }

  const url = serverWebUiUrl();
  webUiUrl.textContent = url;
  if (activeAppTab !== "webui") return;

  const loadedUrl = webUiFrame.dataset.loadedUrl;
  if (force || loadedUrl !== url) {
    webUiFrame.dataset.loadedUrl = url;
    webUiFrame.setAttribute("src", url);
  }
}

function setAppTab(tab: AppTab, forceReload = false) {
  if (tab === "webui" && !server.running) return;
  activeAppTab = tab;

  for (const button of document.querySelectorAll<HTMLButtonElement>(".app-tab-button[data-app-tab]")) {
    const selected = button.dataset.appTab === tab;
    button.setAttribute("aria-selected", String(selected));
  }

  for (const panel of document.querySelectorAll<HTMLElement>(".app-view")) {
    panel.hidden = panel.id !== `app-view-${tab}`;
  }

  syncWebUiFrame(forceReload);
}

function renderHfModelResults(results: HfModelSummary[]) {
  if (results.length === 0) {
    hfModelResults.innerHTML = "";
    const empty = document.createElement("div");
    empty.className = "empty-state hf-empty-state";
    empty.textContent = "No models found";
    hfModelResults.append(empty);
    return;
  }

  const fragment = document.createDocumentFragment();
  for (const model of results) {
    const row = document.createElement("button");
    row.type = "button";
    row.className = "hf-model-row";
    row.dataset.repoId = model.id;

    const title = document.createElement("span");
    title.className = "hf-model-main";
    title.textContent = model.id;

    const meta = document.createElement("span");
    meta.className = "hf-model-meta";
    const details = [
      `${formatCount(model.downloads)} downloads`,
      `${formatCount(model.likes)} likes`,
      model.trendingScore ? `trend ${model.trendingScore}` : "",
      model.lastModified ? `updated ${formatShortDate(model.lastModified)}` : "",
      model.libraryName,
      model.pipelineTag,
    ].filter(Boolean);
    meta.textContent = details.join(" / ");

    row.append(title, meta);
    fragment.append(row);
  }
  hfModelResults.replaceChildren(fragment);
}

async function loadHfModels() {
  const serial = ++hfModelListSerial;
  const request = {
    search: value("hf-model-search"),
    sort: activeHfModelSort,
    token: value("hf-token"),
    limit: 20,
  };

  hfModelStatus.textContent =
    request.search || activeHfModelSort !== "trending"
      ? "Searching Hugging Face..."
      : "Loading trending models...";
  hfModelResults.replaceChildren();

  try {
    const results = await invoke<HfModelSummary[]>("list_hf_models", { request });
    if (serial !== hfModelListSerial) return;
    renderHfModelResults(results);
    hfModelStatus.textContent = `${results.length} model${results.length === 1 ? "" : "s"} found`;
  } catch (error) {
    if (serial !== hfModelListSerial) return;
    hfModelStatus.textContent = String(error);
    hfModelResults.replaceChildren();
    setMessage(String(error), "error");
  }
}

function scheduleHfModelSearch() {
  window.clearTimeout(hfModelSearchTimer);
  hfModelSearchTimer = window.setTimeout(() => {
    void loadHfModels();
  }, 500);
}

async function selectHfModel(repoId: string) {
  setValue("hf-repo", repoId);
  setMessage(`Selected ${repoId}`, "ok");
  await loadHfRepoFiles();
}

function llamaInstallLog(result: LlamaCppInstallResult): string {
  return [
    result.command,
    "",
    `Release: ${result.releaseTag}`,
    `Asset: ${result.assetName}`,
    `Install: ${result.installDir}`,
    `Server: ${result.llamaServerPath}`,
    `CLI: ${result.llamaCliPath}`,
    `Bench: ${result.llamaBenchPath || "not found"}`,
    "",
    result.stdout.trim(),
    result.stderr.trim(),
  ]
    .filter(Boolean)
    .join("\n");
}

async function installLlamaCpp() {
  const button = $("install-llama-cpp") as HTMLButtonElement;
  const request: LlamaCppInstallRequest = {
    package: value("llama-package"),
    targetDir: value("llama-install-dir"),
  };

  button.disabled = true;
  llamaInstallOutput.textContent = "Fetching latest llama.cpp release...\n";
  setMessage("Installing llama.cpp", "info");

  try {
    const result = await invoke<LlamaCppInstallResult>("install_llama_cpp", { request });
    setValue("llama-server-path", result.llamaServerPath);
    setValue("llama-cli-path", result.llamaCliPath);
    llamaInstallOutput.textContent = llamaInstallLog(result);
    await saveConfig("llama.cpp installed and paths saved");
    await refreshTools();
    schedulePreview();
  } finally {
    button.disabled = false;
  }
}

function updateToolStatus() {
  const serverLabel = tools.llamaServer ? "server found" : "server missing";
  const cliLabel = tools.llamaCli ? "cli found" : "cli missing";
  const hfLabel = tools.hfCli || tools.huggingfaceCli ? "hf found" : "hf missing";
  $("tool-status").textContent = `${serverLabel} / ${cliLabel} / ${hfLabel}`;
}

function updateServerChrome() {
  const pill = $("server-pill");
  pill.textContent = server.running && server.pid ? `Server ${server.pid}` : "Server offline";
  pill.className = `pill ${server.running ? "ok" : "muted"}`;

  $("server-status").textContent = server.running
    ? `Running at ${server.url}`
    : "Idle";
  $("server-log").textContent = server.logPath ? `Log: ${server.logPath}` : "";
  ($("start-server") as HTMLButtonElement).disabled = server.running;
  ($("stop-server") as HTMLButtonElement).disabled = !server.running;
  ($("open-server") as HTMLButtonElement).disabled = !server.running;
  webUiTab.disabled = !server.running;
  reloadWebUiButton.disabled = !server.running;
  if (!server.running && activeAppTab === "webui") {
    setAppTab("control");
  }
  syncBenchmarkServerState();
  syncWebUiFrame();
}

function readServerConfig(): ServerConfig {
  return {
    modelPath: modelSelect.value || activeModelPath,
    host: value("host") || "127.0.0.1",
    port: numberValue("port", 8080),
    ctxSize: numberValue("ctx-size", 4096),
    gpuLayers: value("gpu-layers"),
    threads: numberValue("threads", 0),
    batchSize: numberValue("batch-size", 512),
    ubatchSize: numberValue("ubatch-size", 0),
    parallel: numberValue("parallel", -1),
    enableKvCacheOptions: checked("enable-kv-cache-options"),
    cacheTypeK: value("cache-type-k"),
    cacheTypeV: value("cache-type-v"),
    flashAttention: value("flash-attention"),
    kvu: value("kvu"),
    enableGpuMemoryOptions: checked("enable-gpu-memory-options"),
    kvOffload: value("kv-offload"),
    noHost: checked("no-host"),
    opOffload: value("op-offload"),
    fit: value("fit"),
    fitTarget: value("fit-target"),
    fitCtx: numberValue("fit-ctx", 0),
    device: value("device"),
    tensorSplit: value("tensor-split"),
    splitMode: value("split-mode"),
    mainGpu: value("main-gpu"),
    enableSamplingOptions: checked("enable-sampling-options"),
    temperature: value("temperature"),
    topK: value("top-k"),
    topP: value("top-p"),
    minP: value("min-p"),
    typicalP: value("typical-p"),
    repeatPenalty: value("repeat-penalty"),
    presencePenalty: value("presence-penalty"),
    frequencyPenalty: value("frequency-penalty"),
    enableSpeculativeOptions: checked("enable-speculative-options"),
    specType: value("spec-type"),
    specDraftNMax: numberValue("spec-draft-n-max", 0),
    specDraftNMin: numberValue("spec-draft-n-min", 0),
    specDraftPMin: value("spec-draft-p-min"),
    specDraftPSplit: value("spec-draft-p-split"),
    noMmap: checked("no-mmap"),
    mlock: checked("mlock"),
    specNgramModNMatch: numberValue("spec-ngram-mod-n-match", 0),
    specNgramModNMin: numberValue("spec-ngram-mod-n-min", 0),
    specNgramModNMax: numberValue("spec-ngram-mod-n-max", 0),
    specDraftModelPath: value("spec-draft-model-path"),
    cpuMoe: checked("cpu-moe"),
    noCpuMoe: numberValue("no-cpu-moe", 0),
    enableReasoningOptions: checked("enable-reasoning-options"),
    preserveThinking: checked("enable-reasoning-options") && (value("reasoning-preserve") === "chat-template" || value("reasoning-preserve") === "flag" || Boolean(value("chat-template-kwargs").trim())),
    reasoningPreserve: value("reasoning-preserve"),
    reasoningFormat: value("reasoning-format"),
    reasoningBudget: value("reasoning-budget"),
    chatTemplateKwargs: value("chat-template-kwargs") || preserveThinkingDefault,
    reasoning: value("reasoning"),
    enableMultimodalOptions: checked("enable-multimodal-options"),
    mmproj: value("mmproj"),
    embeddings: checked("embeddings"),
    toolsAll: checked("tools-all"),
    jinja: checked("jinja"),
    verbose: checked("verbose"),
    terminalMode: value("terminal-mode") || "visible",
    extraArgs: value("extra-args"),
  };
}

function readConfigFromUi(): AppConfig {
  return {
    ...config,
    llamaServerPath: value("llama-server-path"),
    llamaCliPath: value("llama-cli-path"),
    llamaBenchPath: config.llamaBenchPath,
    modelDir: value("model-dir"),
    hfToken: value("hf-token"),
    server: readServerConfig(),
  };
}

function hydrateServerUi(server: ServerConfig) {
  setValue("host", server.host);
  setValue("port", server.port);
  setValue("ctx-size", server.ctxSize);
  setValue("gpu-layers", server.gpuLayers);
  setValue("threads", server.threads);
  setValue("batch-size", server.batchSize);
  setValue("ubatch-size", server.ubatchSize);
  setValue("parallel", server.parallel);
  setChecked("enable-kv-cache-options", server.enableKvCacheOptions);
  setValue("cache-type-k", server.cacheTypeK);
  setValue("cache-type-v", server.cacheTypeV);
  setValue("flash-attention", server.flashAttention);
  setValue("kvu", server.kvu ?? "");
  setChecked("enable-gpu-memory-options", server.enableGpuMemoryOptions);
  setValue("kv-offload", server.kvOffload ?? "");
  setChecked("no-host", server.noHost ?? false);
  setValue("op-offload", server.opOffload ?? "");
  setValue("fit", server.fit);
  setValue("fit-target", server.fitTarget);
  setValue("fit-ctx", server.fitCtx);
  setValue("device", server.device ?? (server as ServerConfig & { devices?: string }).devices ?? "");
  setValue("tensor-split", server.tensorSplit ?? "");
  setValue("split-mode", server.splitMode ?? "");
  setValue("main-gpu", server.mainGpu ?? "");
  setChecked("enable-sampling-options", server.enableSamplingOptions ?? false);
  setValue("temperature", server.temperature ?? "");
  setValue("top-k", server.topK ?? "");
  setValue("top-p", server.topP ?? "");
  setValue("min-p", server.minP ?? "");
  setValue("typical-p", server.typicalP ?? "");
  setValue("repeat-penalty", server.repeatPenalty ?? "");
  setValue("presence-penalty", server.presencePenalty ?? "");
  setValue("frequency-penalty", server.frequencyPenalty ?? "");
  setChecked("enable-speculative-options", server.enableSpeculativeOptions);
  setValue("spec-type", server.specType);
  setValue("spec-draft-n-max", server.specDraftNMax);
  setValue("spec-draft-n-min", server.specDraftNMin);
  setValue("spec-draft-p-min", server.specDraftPMin);
  setValue("spec-draft-p-split", server.specDraftPSplit);
  setChecked("no-mmap", server.noMmap);
  setChecked("mlock", server.mlock);
  setValue("spec-ngram-mod-n-match", server.specNgramModNMatch);
  setValue("spec-ngram-mod-n-min", server.specNgramModNMin);
  setValue("spec-ngram-mod-n-max", server.specNgramModNMax);
  setValue("spec-draft-model-path", server.specDraftModelPath);
  setChecked("cpu-moe", server.cpuMoe ?? false);
  setValue("no-cpu-moe", server.noCpuMoe);
  setChecked("enable-reasoning-options", server.enableReasoningOptions);
  setValue("reasoning-preserve", server.reasoningPreserve ?? "flag");
  setValue("reasoning-format", server.reasoningFormat);
  setValue("reasoning-budget", server.reasoningBudget);
  setValue("chat-template-kwargs", server.reasoningPreserve === "chat-template" ? (server.chatTemplateKwargs || preserveThinkingDefault) : "");
  setValue("reasoning", server.reasoning);
  setChecked("enable-multimodal-options", server.enableMultimodalOptions);
  setValue("mmproj", server.mmproj);
  setChecked("embeddings", server.embeddings);
  setChecked("tools-all", server.toolsAll);
  setChecked("jinja", server.jinja);
  setChecked("verbose", server.verbose);
  setValue("terminal-mode", server.terminalMode || "visible");
  setValue("extra-args", server.extraArgs);
  syncOptionalGroups();
  syncPreserveThinking();
}

function hydrateUi() {
  activeModelPath = config.server.modelPath;
  setValue("llama-server-path", config.llamaServerPath);
  setValue("llama-cli-path", config.llamaCliPath);
  setValue("model-dir", config.modelDir);
  setValue("hf-target-dir", config.modelDir);
  setValue("hf-token", config.hfToken);
  setValue("hf-pattern", "*.gguf");
  setValue("hf-workers", 8);
  hydrateServerUi(config.server);
  hydrateBenchmarkSettings();
  hydrateEvalSettings();
  setBenchmarkMode("performance");
  resetBenchmarkResults();
  syncBenchmarkServerState();
}

async function saveConfig(message = "Saved") {
  config = readConfigFromUi();
  config = await invoke<AppConfig>("save_config", { config });
  setMessage(message, "ok");
}

async function resetEverything() {
  const confirmed = await confirmAction({
    title: "Reset LocalLLM",
    message:
      "Reset LocalLLM settings, presets, manual models, Hugging Face token, model cache, logs, theme, and layout?\n\nThis will not delete downloaded GGUF models or your llama.cpp install folder.",
    kind: "danger",
    okLabel: "Reset everything",
    cancelLabel: "Cancel",
  });
  if (!confirmed) return;

  setMessage("Resetting LocalLLM data", "info");
  config = await invoke<AppConfig>("reset_app_data");
  window.localStorage.removeItem(themeStorageKey);
  window.localStorage.removeItem(paneWidthStorageKey);
  window.localStorage.removeItem(benchmarkSettingsStorageKey);
  window.localStorage.removeItem(agentPermissionsStorageKey);
  window.localStorage.removeItem(agentYoloModeStorageKey);
  window.localStorage.removeItem(agentAutoAcceptStorageKey);
  window.localStorage.removeItem(agentGoalStorageKey);
  window.localStorage.removeItem(agentModeStorageKey);
  window.localStorage.removeItem(agentTodosStorageKey);
  window.localStorage.removeItem(agentSkillsStorageKey);
  window.localStorage.removeItem(agentSubagentsStorageKey);
  window.localStorage.removeItem(agentAutoCompactStorageKey);
  window.localStorage.removeItem(agentContextSummaryStorageKey);
  window.localStorage.removeItem(agentMcpServersStorageKey);

  models = [];
  activeModelPath = "";
  selectedProfileOverrideIndex = null;
  creatingProfile = false;
  metadataLoads.clear();
  metadataLoaded.clear();
  modelList.replaceChildren();

  hydrateUi();
  updateSelectedModelUi();
  updateServerChrome();
  resetHfFileSelect();
  applyTheme("spotify");
  await refreshTools();
  await refreshPreviewNow();
  setMessage("LocalLLM reset complete. Press Rescan to rebuild the model cache.", "ok");
}

async function refreshTools() {
  tools = await invoke<ToolDiscovery>("discover_tools");
  updateToolStatus();

  if (!value("llama-server-path") && tools.llamaServer) {
    setValue("llama-server-path", tools.llamaServer);
  }
  if (!value("llama-cli-path") && tools.llamaCli) {
    setValue("llama-cli-path", tools.llamaCli);
  }
}

function mergeWithLoadedMetadata(nextModels: ModelEntry[]): ModelEntry[] {
  const currentByPath = new Map(models.map((model) => [model.path, model]));
  return nextModels.map((model) => {
    const current = currentByPath.get(model.path);
    if (current && current.sizeBytes !== model.sizeBytes) {
      metadataLoaded.delete(model.path);
    }
    if (model.metadata || !current?.metadata || current.sizeBytes !== model.sizeBytes) return model;
    return { ...model, metadata: current.metadata };
  });
}

function dedupeAndSortModels(nextModels: ModelEntry[]): ModelEntry[] {
  const seen = new Set<string>();
  const deduped: ModelEntry[] = [];
  for (const model of nextModels) {
    const key = model.path.toLowerCase();
    if (seen.has(key)) continue;
    seen.add(key);
    deduped.push(model);
  }
  return deduped.sort((left, right) => left.name.localeCompare(right.name));
}

function cleanPath(path: string): string {
  if (!path) return "";
  let s = path.trim();
  if (s.startsWith("\\\\?\\")) {
    s = s.slice(4);
  }
  return s.replace(/\//g, "\\").replace(/\\+$/, "").toLowerCase();
}

function samePath(a: string, b: string): boolean {
  const cleanA = cleanPath(a);
  const cleanB = cleanPath(b);
  return Boolean(cleanA && cleanB && cleanA === cleanB);
}

function modelExists(path: string): boolean {
  return models.some((model) => samePath(model.path, path));
}

function selectedModelPath(preferredPath?: string): string {
  const candidates = [activeModelPath, preferredPath, modelSelect.value, config.server.modelPath, models[0]?.path];
  return candidates.find((path) => path && modelExists(path)) ?? "";
}

function setActiveModelPath(path: string) {
  if (!path || !modelExists(path)) return;
  activeModelPath = path;
  config.server.modelPath = path;
  modelSelect.value = path;
}

function setModelRefreshBusy(isBusy: boolean) {
  const button = $("rescan-models") as HTMLButtonElement;
  button.disabled = isBusy;
  button.textContent = isBusy ? "Scanning..." : "Rescan";
  setModelLoading("scan", isBusy);
}

function setModelLoading(task: string, isBusy: boolean) {
  if (isBusy) {
    modelLoadingTasks.add(task);
  } else {
    modelLoadingTasks.delete(task);
  }
  modelLoading.hidden = modelLoadingTasks.size === 0;
}

function updateSelectedModelUi() {
  const path = selectedModelPath();
  if (path) {
    setActiveModelPath(path);
  }

  for (const row of modelList.querySelectorAll<HTMLButtonElement>(".model-row.selected")) {
    row.classList.remove("selected");
  }
  const selectedRow = Array.from(modelList.querySelectorAll<HTMLButtonElement>(".model-row")).find(
    (row) => row.dataset.path && samePath(row.dataset.path, activeModelPath),
  );
  selectedRow?.classList.add("selected");
  updateProfileStatus();
  updateVramEstimate();
  updateCommandHelper();
}

function selectModel(path: string) {
  selectedProfileOverrideIndex = null;
  creatingProfile = false;
  setActiveModelPath(path);
  updateSelectedModelUi();
  applyProfileForSelectedModel();
}

function handleModelSelectChange() {
  selectModel(modelSelect.value);
}

function readBenchmarkSettings(): BenchmarkSettings {
  const preset = value("benchmark-preset") as BenchmarkPreset;
  return {
    preset: ["quick", "standard", "long"].includes(preset) ? preset : "standard",
    runs: value("benchmark-run-count") || "3",
    generateTokens: value("benchmark-generate-tokens") || "128",
    prompt: benchmarkPrompt.value.trim() || benchmarkDefaultPrompt,
  };
}

function hydrateBenchmarkSettings() {
  const raw = window.localStorage.getItem(benchmarkSettingsStorageKey);
  setValue("benchmark-preset", "standard");
  applyBenchmarkPreset();

  if (!raw) {
    return;
  }

  try {
    const settings = JSON.parse(raw) as Partial<BenchmarkSettings>;
    if (settings.preset) setValue("benchmark-preset", settings.preset);
    if (settings.runs) setValue("benchmark-run-count", settings.runs);
    if (settings.generateTokens) setValue("benchmark-generate-tokens", settings.generateTokens);
    if (settings.prompt) benchmarkPrompt.value = settings.prompt;
  } catch {
    window.localStorage.removeItem(benchmarkSettingsStorageKey);
  }
}

function readEvalSettings(): EvalBenchmarkSettings {
  return {
    benchmarkType: (value("eval-benchmark-type") as EvalBenchmarkType) || "humaneval",
    sampleCount: value("eval-sample-count") || "5",
    temperature: value("eval-temperature") || "0.2",
  };
}

function hydrateEvalSettings() {
  const raw = window.localStorage.getItem(evalSettingsStorageKey);
  setValue("eval-benchmark-type", "humaneval");
  setValue("eval-sample-count", "5");
  setValue("eval-temperature", "0.2");

  if (!raw) {
    return;
  }

  try {
    const settings = JSON.parse(raw) as Partial<EvalBenchmarkSettings>;
    if (settings.benchmarkType) setValue("eval-benchmark-type", settings.benchmarkType);
    if (settings.sampleCount) setValue("eval-sample-count", settings.sampleCount);
    if (settings.temperature) setValue("eval-temperature", settings.temperature);
  } catch {
    window.localStorage.removeItem(evalSettingsStorageKey);
  }
}

function setBenchmarkMode(mode: BenchmarkMode) {
  const isEval = mode === "evaluation";
  benchmarkPerformanceSettings.hidden = isEval;
  benchmarkEvaluationSettings.hidden = !isEval;
  benchmarkResults.hidden = isEval;
  evalResults.hidden = !isEval;
  evalSamplesContainer.hidden = isEval;
  $("benchmark-visual-grid").hidden = !isEval;
  $("benchmark-share-panel").hidden = isEval;
  $("benchmark-transcript").hidden = isEval;
}

function createModelRow(model: ModelEntry): HTMLButtonElement {
  const row = document.createElement("button");
  row.type = "button";
  row.className = `model-row ${samePath(model.path, activeModelPath) ? "selected" : ""}`;
  row.dataset.path = model.path;

  const main = document.createElement("span");
  main.className = "model-main";
  main.textContent = model.name;

  const meta = document.createElement("span");
  meta.className = "model-meta";
  meta.textContent = `${model.source} / ${formatBytes(model.sizeBytes)} / ${model.path}`;

  row.append(main, meta);
  return row;
}

async function renderModels(preferredPath?: string) {
  const renderSerial = ++modelRenderSerial;
  setModelLoading("render", true);
  $("model-count").textContent = `${models.length} model${models.length === 1 ? "" : "s"}`;
  modelSelect.replaceChildren();
  modelList.replaceChildren();

  try {
    if (models.length === 0) {
      const option = document.createElement("option");
      option.value = "";
      option.textContent = "No GGUF models found";
      modelSelect.append(option);

      const empty = document.createElement("div");
      empty.className = "empty-state";
      empty.textContent = "No GGUF models";
      modelList.append(empty);
      updateSelectedModelUi();
      return;
    }

    const optionBatchSize = 250;
    for (let index = 0; index < models.length; index += optionBatchSize) {
      if (renderSerial !== modelRenderSerial) return;
      const fragment = document.createDocumentFragment();
      for (const model of models.slice(index, index + optionBatchSize)) {
        const option = document.createElement("option");
        option.value = model.path;
        option.textContent = `${model.name} - ${formatBytes(model.sizeBytes)}`;
        fragment.append(option);
      }
      modelSelect.append(fragment);
      await yieldToUi();
    }

    const nextSelectedPath = selectedModelPath(preferredPath);
    if (nextSelectedPath) {
      setActiveModelPath(nextSelectedPath);
    }

    const rowBatchSize = 80;
    for (let index = 0; index < models.length; index += rowBatchSize) {
      if (renderSerial !== modelRenderSerial) return;
      const fragment = document.createDocumentFragment();
      for (const model of models.slice(index, index + rowBatchSize)) {
        fragment.append(createModelRow(model));
      }
      modelList.append(fragment);
      await yieldToUi();
    }

    updateSelectedModelUi();
  } finally {
    if (renderSerial === modelRenderSerial) {
      setModelLoading("render", false);
    }
  }
}

async function syncModelInputsFromUi() {
  const uiConfig = readConfigFromUi();
  config = {
    ...config,
    llamaServerPath: uiConfig.llamaServerPath,
    llamaCliPath: uiConfig.llamaCliPath,
    modelDir: uiConfig.modelDir,
    hfToken: uiConfig.hfToken,
  };
}

async function loadCachedModels() {
  await syncModelInputsFromUi();
  setModelLoading("cache", true);
  try {
    const cachedModels = await invoke<ModelEntry[]>("load_model_cache", {
      modelDir: config.modelDir,
      manualModels: config.manualModels,
    });
    const nextModels = cachedModels.length > 0 ? cachedModels : config.manualModels;

    models = mergeWithLoadedMetadata(dedupeAndSortModels(nextModels));
    await renderModels();
    await ensureModelDefaultProfiles();
    applyProfileForSelectedModel();
    if (models.length > 0) {
      setMessage(cachedModels.length > 0 ? "Cached models loaded" : "Manual models loaded", "ok");
    }
  } finally {
    setModelLoading("cache", false);
  }
}

async function refreshModels(message = "Models refreshed", busyMessage = "Scanning models from disk") {
  await syncModelInputsFromUi();
  const scanSerial = ++modelScanSerial;
  setModelRefreshBusy(true);
  setMessage(busyMessage, "info");

  try {
    const scannedModels = await invoke<ModelEntry[]>("scan_models", {
      modelDir: config.modelDir,
      manualModels: config.manualModels,
    });
    if (scanSerial !== modelScanSerial) return;

    const previousPath = activeModelPath || modelSelect.value;
    models = mergeWithLoadedMetadata(scannedModels);
    await renderModels(previousPath);
    await ensureModelDefaultProfiles();
    applyProfileForSelectedModel();
    setMessage(message, "ok");
  } finally {
    if (scanSerial === modelScanSerial) {
      setModelRefreshBusy(false);
    }
  }
}

async function refreshModelsInBackground(message = "Models refreshed") {
  refreshModels(message, "Refreshing model cache").catch(showError);
}

function buildLaunchConfig() {
  return {
    executablePath: value("llama-server-path"),
    ...readServerConfig(),
  };
}

function selectedModel(): ModelEntry | undefined {
  return models.find((model) => samePath(model.path, activeModelPath));
}

function selectedModelProfileIndex(): number {
  return config.modelProfiles.findIndex((profile) => samePath(profile.modelPath, activeModelPath));
}

function activeProfileIndex(): number {
  if (creatingProfile) return -1;
  return selectedProfileOverrideIndex ?? selectedModelProfileIndex();
}

function activeProfile(): ModelProfile | undefined {
  const index = activeProfileIndex();
  return index >= 0 ? config.modelProfiles[index] : undefined;
}

function serverForModel(server: ServerConfig, modelPath: string): ServerConfig {
  return { ...server, modelPath };
}

function defaultServerForModel(modelPath: string): ServerConfig {
  return serverForModel(defaultServerConfig, modelPath);
}

function defaultProfileName(model: ModelEntry): string {
  return `${model.name} default`;
}

function nextProfileName(model: ModelEntry): string {
  const baseName = `${model.name} profile`;
  const existingNames = new Set(config.modelProfiles.map((profile) => profile.name.toLowerCase()));
  if (!existingNames.has(baseName.toLowerCase())) return baseName;

  for (let index = 2; index < 1000; index += 1) {
    const candidate = `${baseName} ${index}`;
    if (!existingNames.has(candidate.toLowerCase())) return candidate;
  }

  return `${baseName} ${config.modelProfiles.length + 1}`;
}

async function ensureModelDefaultProfiles() {
  if (models.length === 0) return;

  const profiledModelPaths = new Set(config.modelProfiles.map((profile) => cleanPath(profile.modelPath)));
  const missingModels = models.filter((model) => !profiledModelPaths.has(cleanPath(model.path)));
  if (missingModels.length === 0) return;

  const modelProfiles = [
    ...config.modelProfiles,
    ...missingModels.map((model) => ({
      modelPath: model.path,
      name: defaultProfileName(model),
      server: defaultServerForModel(model.path),
    })),
  ];

  config = await invoke<AppConfig>("save_config", {
    config: { ...config, modelProfiles },
  });
}

function updateProfileSelector(selectedIndex: number) {
  const selector = $("profile-select") as HTMLSelectElement;
  const selectedValue = selectedIndex >= 0 ? `profile:${selectedIndex}` : "default";
  selector.replaceChildren();

  const defaultOption = document.createElement("option");
  defaultOption.value = "default";
  defaultOption.textContent = "Default";
  selector.append(defaultOption);

  config.modelProfiles.forEach((profile, index) => {
    const option = document.createElement("option");
    option.value = `profile:${index}`;
    option.textContent = profile.name || models.find((model) => samePath(model.path, profile.modelPath))?.name || profile.modelPath;
    selector.append(option);
  });

  selector.value = selectedValue;
}

function updateProfileStatus() {
  const model = selectedModel();
  const profileIndex = activeProfileIndex();
  const profile = profileIndex >= 0 ? config.modelProfiles[profileIndex] : undefined;
  const profileStatus = $("profile-status");
  const profileName = $("profile-name") as HTMLInputElement;
  const profileSelect = $("profile-select") as HTMLSelectElement;
  const deleteButton = $("delete-model-profile") as HTMLButtonElement;
  const saveButton = $("save-model-profile") as HTMLButtonElement;
  const defaultButton = $("load-default-profile") as HTMLButtonElement;
  const isNewProfile = Boolean(model && creatingProfile);

  updateProfileSelector(profileIndex);
  profileSelect.disabled = !model && config.modelProfiles.length === 0;
  saveButton.disabled = !model;
  saveButton.textContent = isNewProfile ? "Create Profile" : "Save Profile";
  deleteButton.disabled = profileIndex < 0;
  defaultButton.disabled = !model;
  profileName.disabled = !model;
  profileName.value = model
    ? isNewProfile
      ? profileName.value || nextProfileName(model)
      : profile?.name ?? model.name
    : "";
  const profileModelName = profile
    ? models.find((entry) => samePath(entry.path, profile.modelPath))?.name ?? profile.modelPath
    : "";
  if (!model) {
    profileStatus.textContent = "No model selected";
  } else if (isNewProfile) {
    profileStatus.textContent = "New profile";
  } else if (profile) {
    profileStatus.textContent =
      profile.modelPath === activeModelPath
        ? `Profile: ${profile.name}`
        : `Profile: ${profile.name} from ${profileModelName}`;
  } else {
    profileStatus.textContent = "Default profile";
  }
}

function applyProfile(profile: ModelProfile, message = "Profile loaded") {
  const modelPath = activeModelPath || modelSelect.value;
  if (!modelPath) return;

  hydrateServerUi(serverForModel(profile.server, modelPath));
  updateProfileStatus();
  refreshPreviewNow();
  setMessage(message, "ok");
}

function applyProfileForSelectedModel(message?: string) {
  const modelPath = activeModelPath || modelSelect.value;
  if (!modelPath) {
    updateProfileStatus();
    refreshPreviewNow();
    return;
  }

  const profile = activeProfile();
  hydrateServerUi(serverForModel(profile?.server ?? config.server, modelPath));
  updateProfileStatus();
  refreshPreviewNow();
  if (message) setMessage(message, "ok");
}

function loadDefaultProfileForSelectedModel() {
  const modelPath = activeModelPath || modelSelect.value;
  if (!modelPath) return;
  selectedProfileOverrideIndex = -1;
  creatingProfile = false;
  hydrateServerUi(defaultServerForModel(modelPath));
  updateProfileStatus();
  refreshPreviewNow();
  setMessage("Default profile loaded", "ok");
}

function loadSelectedProfileFromSelector() {
  const selected = value("profile-select");
  creatingProfile = false;
  if (selected === "default") {
    loadDefaultProfileForSelectedModel();
    return;
  }

  const index = Number(selected.replace("profile:", ""));
  const profile = Number.isInteger(index) ? config.modelProfiles[index] : undefined;
  if (!profile) {
    selectedProfileOverrideIndex = null;
    updateProfileStatus();
    return;
  }

  selectedProfileOverrideIndex = index;
  applyProfile(profile);
}

function startNewProfile() {
  const model = selectedModel();
  if (!model) return;

  selectedProfileOverrideIndex = null;
  creatingProfile = true;
  ($("profile-name") as HTMLInputElement).value = nextProfileName(model);
  updateProfileStatus();
  setMessage("New profile ready", "ok");
}

async function saveSelectedModelProfile() {
  const model = selectedModel();
  if (!model) return;

  const profile: ModelProfile = {
    modelPath: model.path,
    name: value("profile-name") || model.name,
    server: serverForModel(readServerConfig(), model.path),
  };
  const activeIndex = activeProfileIndex();
  const index = activeIndex >= 0 && config.modelProfiles[activeIndex]?.modelPath === model.path
    ? activeIndex
    : -1;
  const modelProfiles = [...config.modelProfiles];
  let savedIndex = index;
  if (index >= 0) {
    modelProfiles[index] = profile;
  } else {
    modelProfiles.push(profile);
    savedIndex = modelProfiles.length - 1;
  }

  config = await invoke<AppConfig>("save_config", {
    config: { ...readConfigFromUi(), server: config.server, modelProfiles },
  });
  creatingProfile = false;
  selectedProfileOverrideIndex = savedIndex;
  updateProfileStatus();
  setMessage(index >= 0 ? "Model profile saved" : "Model profile created", "ok");
}

async function deleteSelectedModelProfile() {
  const index = activeProfileIndex();
  if (index < 0) return;

  const modelProfiles = config.modelProfiles.filter((_, profileIndex) => profileIndex !== index);
  config = await invoke<AppConfig>("save_config", {
    config: { ...readConfigFromUi(), server: config.server, modelProfiles },
  });
  creatingProfile = false;
  selectedProfileOverrideIndex = null;
  applyProfileForSelectedModel("Model profile deleted");
}

async function updatePreview(serial = ++previewSerial) {
  window.clearTimeout(previewTimer);
  try {
    const command = await invoke<string>("preview_server_command", {
      config: buildLaunchConfig(),
    });
    if (serial !== previewSerial) return;
    commandPreview.textContent = command;
  } catch (error) {
    if (serial !== previewSerial) return;
    commandPreview.textContent = String(error);
  }
}

function schedulePreview() {
  window.clearTimeout(previewTimer);
  updateVramEstimate();
  updateCommandHelper();
  const serial = ++previewSerial;
  previewTimer = window.setTimeout(() => {
    void updatePreview(serial);
  }, 150);
}

function refreshPreviewNow() {
  window.clearTimeout(previewTimer);
  updateVramEstimate();
  updateCommandHelper();
  void updatePreview();
}

async function chooseDirectory(targetId: string) {
  const selected = await open({
    directory: true,
    multiple: false,
    defaultPath: value(targetId) || undefined,
  });
  if (typeof selected === "string") {
    setValue(targetId, selected);
    if (targetId === "model-dir") {
      setValue("hf-target-dir", selected);
      await saveConfig("Model directory saved");
      await loadCachedModels();
      setMessage("Model directory saved. Press Rescan to refresh.", "ok");
    }
  }
}

async function chooseExecutable(targetId: string) {
  const selected = await open({
    multiple: false,
    defaultPath: value(targetId) || undefined,
  });
  if (typeof selected === "string") {
    setValue(targetId, selected);
    await saveConfig("Executable path saved");
    schedulePreview();
  }
}

async function chooseMmproj() {
  const selected = await open({
    multiple: false,
    filters: [{ name: "GGUF", extensions: ["gguf"] }],
    defaultPath: value("mmproj") || value("model-dir") || undefined,
  });
  if (typeof selected === "string") {
    setValue("mmproj", selected);
    schedulePreview();
  }
}

async function chooseDraftModel() {
  const selected = await open({
    multiple: false,
    filters: [{ name: "GGUF", extensions: ["gguf"] }],
    defaultPath: value("spec-draft-model-path") || value("model-dir") || undefined,
  });
  if (typeof selected === "string") {
    setValue("spec-draft-model-path", selected);
    await saveConfig("Draft model path saved");
    schedulePreview();
  }
}

async function addManualModel() {
  const selected = await open({
    multiple: false,
    filters: [{ name: "GGUF", extensions: ["gguf"] }],
    defaultPath: value("model-dir") || undefined,
  });
  if (typeof selected !== "string") return;

  const model = await invoke<ModelEntry>("model_from_path", { path: selected });
  if (!config.manualModels.some((entry) => samePath(entry.path, model.path))) {
    config.manualModels.push(model);
    config = await invoke<AppConfig>("save_config", { config });
  }
  models = dedupeAndSortModels([model, ...models]);
  activeModelPath = model.path;
  await renderModels(model.path);
  applyProfileForSelectedModel();
  setMessage("Manual model added", "ok");
}

function applyAdvancedPreset() {
  setChecked("enable-kv-cache-options", true);
  setChecked("enable-gpu-memory-options", false);
  setChecked("enable-sampling-options", false);
  setChecked("enable-speculative-options", false);
  setChecked("enable-reasoning-options", true);
  setChecked("enable-multimodal-options", false);
  setValue("host", "127.0.0.1");
  setValue("port", 8080);
  setValue("gpu-layers", "");
  setValue("cache-type-k", "q8_0");
  setValue("cache-type-v", "q8_0");
  setValue("ctx-size", 4096);
  setValue("flash-attention", "");
  setValue("threads", 0);
  setValue("kv-offload", "");
  setChecked("no-host", false);
  setValue("op-offload", "");
  setValue("fit", "");
  setValue("fit-target", "");
  setValue("fit-ctx", 0);
  setValue("device", "");
  setValue("tensor-split", "");
  setValue("split-mode", "");
  setValue("main-gpu", "");
  setValue("temperature", "");
  setValue("top-k", "");
  setValue("top-p", "");
  setValue("min-p", "");
  setValue("typical-p", "");
  setValue("repeat-penalty", "");
  setValue("presence-penalty", "");
  setValue("frequency-penalty", "");
  setValue("spec-type", "");
  setValue("spec-draft-n-max", 0);
  setValue("spec-draft-n-min", 0);
  setValue("spec-draft-p-min", "");
  setValue("spec-draft-p-split", "");
  setChecked("no-mmap", false);
  setChecked("mlock", false);
  setValue("kvu", "");
  setValue("batch-size", 0);
  setValue("ubatch-size", 0);
  setValue("parallel", -1);
  setValue("spec-ngram-mod-n-match", 0);
  setValue("spec-ngram-mod-n-min", 0);
  setValue("spec-ngram-mod-n-max", 0);
  setValue("spec-draft-model-path", "");
  setChecked("cpu-moe", false);
  setValue("no-cpu-moe", 0);
  setValue("reasoning-format", "");
  setValue("reasoning-budget", "");
  setValue("reasoning-preserve", "flag");
  setValue("chat-template-kwargs", preserveThinkingDefault);
  setValue("reasoning", "");
  setValue("mmproj", "");
  setChecked("embeddings", false);
  setChecked("tools-all", false);
  setChecked("jinja", false);
  setChecked("verbose", false);
  setValue("terminal-mode", "visible");
  setValue("extra-args", "");
  syncOptionalGroups();
  schedulePreview();
  setMessage("Clean auto preset applied", "ok");
}

async function startServer() {
  await saveConfig("Configuration saved");
  const backgroundServers = await invoke<LlamaServerProcess[]>("list_background_llama_servers");
  if (backgroundServers.length > 0) {
    const processList = backgroundServers
      .slice(0, 5)
      .map((process) => `PID ${process.pid}: ${process.commandLine || "llama-server.exe"}`)
      .join("\n");
    const extraCount = backgroundServers.length > 5 ? `\n...and ${backgroundServers.length - 5} more.` : "";
    const closeExisting = await confirmAction({
      title: "Another llama-server is running",
      message: `Another llama-server process is already running in the background.\n\n${processList}${extraCount}\n\nDo you want LocalLLM to close the existing process before starting this server?\n\nChoose No to leave it running and start another llama-server.`,
      kind: "warning",
      okLabel: "Yes, close it",
      cancelLabel: "No, start another",
    });
    if (closeExisting) {
      setMessage("Closing existing llama-server", "info");
      await invoke("close_llama_servers", {
        pids: backgroundServers.map((process) => process.pid),
      });
    } else {
      setMessage("Starting another llama-server without closing the existing one", "warn");
    }
  }
  setMessage("Starting llama-server", "info");
  server = await invoke<ServerStatus>("start_server", {
    config: buildLaunchConfig(),
  });
  updateServerChrome();
  setAppTab("webui", true);
  setMessage("llama-server started", "ok");
}

async function stopServer() {
  server = await invoke<ServerStatus>("stop_server");
  updateServerChrome();
  setMessage("llama-server stopped", "ok");
}

async function refreshServerStatus() {
  try {
    server = await invoke<ServerStatus>("server_status");
    updateServerChrome();
  } catch (error) {
    setMessage(String(error), "error");
  }
}

async function downloadModel() {
  const request: DownloadRequest = {
    repoId: value("hf-repo"),
    pattern: downloadFileSelect.value || value("hf-pattern") || "*.gguf",
    revision: value("hf-revision"),
    targetDir: value("hf-target-dir") || value("model-dir"),
    token: value("hf-token"),
    force: checked("hf-force"),
    maxWorkers: numberValue("hf-workers", 8),
  };

  setMessage("Downloading from Hugging Face", "info");
  downloadOutput.textContent = "Starting download...\n";
  ($("download-model") as HTMLButtonElement).disabled = true;

  try {
    const result = await invoke<CommandOutput>("download_model", { request });
    const status = result.success ? "completed" : `failed (${result.statusCode ?? "unknown"})`;
    downloadOutput.textContent = [
      result.command,
      "",
      result.stdout.trim(),
      result.stderr.trim(),
    ]
      .filter(Boolean)
      .join("\n");
    setMessage(`Download ${status}`, result.success ? "ok" : "error");
    if (result.success) {
      setMessage("Download finished. Press Rescan to refresh.", "ok");
    }
  } finally {
    ($("download-model") as HTMLButtonElement).disabled = false;
  }
}

function resetHfFileSelect(label = "Manual / glob") {
  const option = document.createElement("option");
  option.value = "";
  option.textContent = label;
  downloadFileSelect.replaceChildren(option);
  downloadFileSelect.value = "";
  downloadFileSelect.disabled = true;
}

function populateHfFileSelect(files: HfRepoFile[]) {
  const selectedPattern = value("hf-pattern");
  const manualOption = document.createElement("option");
  manualOption.value = "";
  manualOption.textContent = "Manual / glob";

  const fragment = document.createDocumentFragment();
  fragment.append(manualOption);

  for (const file of files) {
    const option = document.createElement("option");
    option.value = file.path;
    option.textContent =
      typeof file.sizeBytes === "number" && file.sizeBytes > 0
        ? `${file.path} - ${formatBytes(file.sizeBytes)}`
        : file.path;
    fragment.append(option);
  }

  downloadFileSelect.replaceChildren(fragment);
  downloadFileSelect.disabled = files.length === 0;
  downloadFileSelect.value = files.some((file) => file.path === selectedPattern) ? selectedPattern : "";
}

async function loadHfRepoFiles() {
  const repoId = value("hf-repo");
  const serial = ++downloadFileListSerial;

  if (!repoId || !repoId.includes("/")) {
    resetHfFileSelect();
    return;
  }

  resetHfFileSelect("Loading files...");

  try {
    const files = await invoke<HfRepoFile[]>("list_hf_repo_files", {
      repoId,
      revision: value("hf-revision"),
      token: value("hf-token"),
    });
    if (serial !== downloadFileListSerial) return;

    populateHfFileSelect(files);
    setMessage(`${files.length} Hugging Face file${files.length === 1 ? "" : "s"} loaded`, "ok");
  } catch (error) {
    if (serial !== downloadFileListSerial) return;
    resetHfFileSelect("Unable to load files");
    setMessage(String(error), "error");
  }
}

function scheduleHfRepoFileLoad() {
  window.clearTimeout(downloadFileListTimer);
  const repoId = value("hf-repo");
  if (!repoId || !repoId.includes("/")) {
    ++downloadFileListSerial;
    resetHfFileSelect();
    return;
  }

  downloadFileListTimer = window.setTimeout(() => {
    void loadHfRepoFiles();
  }, 600);
}

function handleHfFileSelectChange() {
  if (downloadFileSelect.value) {
    setValue("hf-pattern", downloadFileSelect.value);
  }
}

function handleHfPatternInput() {
  if (downloadFileSelect.value && value("hf-pattern") !== downloadFileSelect.value) {
    downloadFileSelect.value = "";
  }
}

function cacheTypeBytes(type: string): number {
  switch (type.trim().toLowerCase()) {
    case "f32":
      return 4;
    case "bf16":
    case "f16":
      return 2;
    case "q8_0":
      return 34 / 32;
    case "q5_0":
      return 22 / 32;
    case "q5_1":
      return 24 / 32;
    case "q4_0":
    case "iq4_nl":
      return 18 / 32;
    case "q4_1":
      return 20 / 32;
    default:
      return 2;
  }
}

function positiveNumber(value: number | null | undefined): number | undefined {
  return typeof value === "number" && Number.isFinite(value) && value > 0 ? value : undefined;
}

function modelShapeGuess(sizeBytes: number): KvShape {
  const sizeGb = sizeBytes / 1024 ** 3;
  const guessed =
    sizeGb <= 4
      ? { layers: 32, hidden: 2048 }
      : sizeGb <= 9
        ? { layers: 32, hidden: 4096 }
        : sizeGb <= 18
          ? { layers: 40, hidden: 5120 }
          : sizeGb <= 35
            ? { layers: 60, hidden: 6656 }
            : { layers: 80, hidden: 8192 };
  const headDim = 128;
  const heads = Math.max(1, Math.round(guessed.hidden / headDim));
  return {
    layers: guessed.layers,
    headCount: heads,
    kvHeadCount: heads,
    keyLength: guessed.hidden / heads,
    valueLength: guessed.hidden / heads,
    contextLength: undefined,
    source: "fallback",
  };
}

interface KvShape {
  layers: number;
  headCount: number;
  kvHeadCount: number;
  keyLength: number;
  valueLength: number;
  contextLength?: number;
  source: "metadata" | "fallback";
}

function kvShapeForModel(model: ModelEntry): KvShape {
  const metadata = model.metadata;
  const layers = positiveNumber(metadata?.blockCount);
  const embedding = positiveNumber(metadata?.embeddingLength);
  const headCount = positiveNumber(metadata?.attentionHeadCount);
  const kvHeadCount = positiveNumber(metadata?.attentionHeadCountKv) ?? headCount;
  const keyLength =
    positiveNumber(metadata?.attentionKeyLength) ??
    (embedding && headCount ? embedding / headCount : undefined);
  const valueLength =
    positiveNumber(metadata?.attentionValueLength) ??
    (embedding && headCount ? embedding / headCount : undefined);

  if (layers && headCount && kvHeadCount && keyLength && valueLength) {
    return {
      layers,
      headCount,
      kvHeadCount,
      keyLength,
      valueLength,
      contextLength: positiveNumber(metadata?.contextLength),
      source: "metadata",
    };
  }

  return modelShapeGuess(model.sizeBytes);
}

function gpuLayerFraction(raw: string, totalLayers?: number): number {
  const value = raw.trim().toLowerCase();
  if (!value || value === "auto" || value === "all") return 1;
  const numeric = Number(value);
  if (!Number.isFinite(numeric) || numeric <= 0) return 0;
  const layerCount = totalLayers && totalLayers > 0 ? totalLayers : 80;
  if (numeric >= layerCount) return 1;
  return Math.min(1, numeric / layerCount);
}

async function ensureModelMetadata(model: ModelEntry) {
  if (model.metadata || metadataLoads.has(model.path) || metadataLoaded.has(model.path)) return;

  metadataLoads.add(model.path);
  try {
    const metadata = await invoke<GgufModelMetadata | null>("load_model_metadata", { path: model.path });
    metadataLoaded.add(model.path);
    const current = models.find((entry) => samePath(entry.path, model.path));
    if (current) {
      current.metadata = metadata;
    }
    if (activeModelPath === model.path) {
      updateVramEstimate();
    }
  } catch (error) {
    metadataLoaded.add(model.path);
    console.warn(error);
  } finally {
    metadataLoads.delete(model.path);
  }
}

function updateVramEstimate() {
  const model = models.find((entry) => samePath(entry.path, activeModelPath));
  if (!model) {
    vramTotal.textContent = "VRAM estimate: no model";
    vramDetail.textContent = "Pick a GGUF model to estimate.";
    return;
  }

  ensureModelMetadata(model);
  const shape = kvShapeForModel(model);
  const configuredCtx = Math.max(0, numberValue("ctx-size", 0));
  const ctx = configuredCtx > 0 ? configuredCtx : shape.contextLength ?? 0;
  const cacheK = cacheTypeBytes(value("cache-type-k"));
  const cacheV = cacheTypeBytes(value("cache-type-v"));
  const weightsBytes = model.sizeBytes * gpuLayerFraction(value("gpu-layers"), shape.layers);
  const kvBytes =
    ctx * shape.layers * shape.kvHeadCount * (shape.keyLength * cacheK + shape.valueLength * cacheV);
  const projectorBytes =
    checked("enable-multimodal-options") && value("mmproj") ? 384 * 1024 ** 2 : 0;
  const overheadBytes = (weightsBytes + kvBytes + projectorBytes) * 0.08;
  const totalBytes = weightsBytes + kvBytes + projectorBytes + overheadBytes;
  const shapeLabel =
    shape.source === "metadata"
      ? `${shape.layers} layers, ${shape.kvHeadCount}/${shape.headCount} KV heads`
      : metadataLoads.has(model.path)
        ? "loading GGUF metadata"
        : "fallback model-shape guess";
  const ctxLabel = ctx > 0 ? ctx.toLocaleString() : "unknown";

  vramTotal.textContent = `VRAM estimate: ${formatBytes(totalBytes)}`;
  vramDetail.textContent = `Weights ${formatBytes(weightsBytes)} + KV ${formatBytes(
    kvBytes,
  )} + overhead ${formatBytes(overheadBytes)} / ctx ${ctxLabel} / ${shapeLabel}`;
}

function updateCommandHelper() {
  const notes: string[] = [];
  notes.push(`serves ${activeModelPath ? "the selected GGUF" : "a GGUF model"} on ${value("host") || "127.0.0.1"}:${value("port") || "8080"}`);
  notes.push(`uses context ${value("ctx-size") || "model default"}, batch ${value("batch-size") || "2048"}, ubatch ${value("ubatch-size") || "512"}`);
  if (value("gpu-layers")) {
    notes.push(`offloads GPU layers: ${value("gpu-layers")}`);
  }
  notes.push(
    value("terminal-mode") === "visible"
      ? "shows llama-server output in a terminal"
      : "runs llama-server hidden with output in the log",
  );

  if (checked("enable-kv-cache-options")) {
    const cacheNotes = [value("cache-type-k") && `K ${value("cache-type-k")}`, value("cache-type-v") && `V ${value("cache-type-v")}`, value("flash-attention") && `flash ${value("flash-attention")}`].filter(Boolean);
    if (cacheNotes.length) notes.push(`KV cache: ${cacheNotes.join(", ")}`);
  }
  if (checked("enable-gpu-memory-options")) {
    const memoryNotes = [
      value("kv-offload") && `KV offload ${value("kv-offload")}`,
      value("device") && `device ${value("device")}`,
      value("op-offload") && `ops ${value("op-offload")}`,
      checked("no-host") && "no host buffer",
      value("split-mode") && `mode ${value("split-mode")}`,
      value("tensor-split") && `split ${value("tensor-split")}`,
      value("main-gpu") && `main GPU ${value("main-gpu")}`,
      value("fit") && `fit ${value("fit")}`,
      checked("cpu-moe") && "all MoE on CPU",
      value("no-cpu-moe") !== "0" && value("no-cpu-moe") && `CPU MoE layers ${value("no-cpu-moe")}`,
      checked("no-mmap") && "no mmap",
      checked("mlock") && "mlock",
    ].filter(Boolean);
    if (memoryNotes.length) notes.push(`GPU/memory: ${memoryNotes.join(", ")}`);
  }
  if (checked("enable-sampling-options")) {
    const samplingNotes = [
      value("temperature") && `temp ${value("temperature")}`,
      value("top-k") && `top-k ${value("top-k")}`,
      value("top-p") && `top-p ${value("top-p")}`,
      value("min-p") && `min-p ${value("min-p")}`,
      value("repeat-penalty") && `repeat ${value("repeat-penalty")}`,
    ].filter(Boolean);
    if (samplingNotes.length) notes.push(`sampling: ${samplingNotes.join(", ")}`);
  }
  if (checked("enable-speculative-options") && value("spec-type")) {
    notes.push(`speculative decoding: ${value("spec-type")}`);
  }
  if (checked("enable-reasoning-options")) {
    const reasoningNotes = [value("reasoning") && `reasoning ${value("reasoning")}`, value("reasoning-budget") && `budget ${value("reasoning-budget")}`, value("reasoning-preserve") && `preserve ${value("reasoning-preserve")}`].filter(Boolean);
    if (reasoningNotes.length) notes.push(`chat reasoning: ${reasoningNotes.join(", ")}`);
  }
  if (checked("enable-multimodal-options") && value("mmproj")) {
    notes.push(`multimodal projector: ${value("mmproj")}`);
  }
  if (checked("tools-all")) {
    notes.push("tools: all");
  }
  if (checked("jinja")) {
    notes.push("jinja");
  }

  commandHelper.textContent = notes.join(" / ");
}

function syncOptionalGroups() {
  for (const group of optionalGroups) {
    const enabled = checked(group.toggleId);
    const section = $(group.groupId);
    section.classList.toggle("disabled-group", !enabled);

    for (const element of section.querySelectorAll<HTMLInputElement | HTMLSelectElement | HTMLTextAreaElement | HTMLButtonElement>(
      "input, select, textarea, button",
    )) {
      if (element.id !== group.toggleId) {
        element.disabled = !enabled;
      }
    }
  }
  syncPreserveThinking();
}

function syncPreserveThinking() {
  const enabled = checked("enable-reasoning-options");
  const method = value("reasoning-preserve");
  const input = $("chat-template-kwargs") as HTMLTextAreaElement;
  input.disabled = !enabled || method !== "chat-template";
  if (enabled && method === "chat-template" && !input.value.trim()) {
    input.value = preserveThinkingDefault;
  }
}

function applyBenchmarkPreset() {
  const preset = readBenchmarkSettings().preset;
  const presets: Record<
    BenchmarkPreset,
    { runs: number; generateTokens: number; prompt: string }
  > = {
    quick: {
      runs: 2,
      generateTokens: 64,
      prompt: "Answer in one compact paragraph: what makes a local LLM server feel fast?",
    },
    standard: {
      runs: 3,
      generateTokens: 128,
      prompt: benchmarkDefaultPrompt,
    },
    long: {
      runs: 3,
      generateTokens: 256,
      prompt:
        "Draft a practical tuning note for a local llama-server setup. Cover prompt latency, generation throughput, memory pressure, and one validation step.",
    },
  };
  const next = presets[preset];
  setValue("benchmark-run-count", next.runs);
  setValue("benchmark-generate-tokens", next.generateTokens);
  benchmarkPrompt.value = next.prompt;
}

function benchmarkBaseUrl(): string {
  return (server.url || `http://${value("host") || "127.0.0.1"}:${value("port") || "8080"}`).replace(/\/+$/, "");
}

function syncBenchmarkServerState() {
  const button = $("run-benchmark") as HTMLButtonElement;
  const isRunning = benchmarkAbortController !== null || evalAbortController !== null;
  const url = benchmarkBaseUrl();
  benchmarkStatus.textContent = server.running ? `Ready at ${url}` : "Server offline";
  button.disabled = !server.running || isRunning;
  button.textContent = isRunning ? "Running..." : "Run";
}

function estimateTokenCount(text: string): number {
  const normalized = text.trim();
  if (!normalized) return 0;
  return Math.max(1, Math.round(normalized.length / 4));
}

function formatBenchmarkNumber(value: number, maximumFractionDigits = 1): string {
  return value.toLocaleString(undefined, { maximumFractionDigits });
}

function formatBenchmarkMs(value: number): string {
  return formatBenchmarkNumber(value, 0);
}

function animateBenchmarkNumber(
  element: HTMLElement,
  target: number,
  formatter: (value: number) => string,
) {
  const startValue = Number(element.dataset.value ?? "0");
  const startTime = performance.now();
  const duration = 650;

  const tick = (now: number) => {
    const progress = Math.min(1, (now - startTime) / duration);
    const eased = 1 - (1 - progress) ** 3;
    element.textContent = formatter(startValue + (target - startValue) * eased);
    if (progress < 1) {
      window.requestAnimationFrame(tick);
    } else {
      element.dataset.value = String(target);
      element.textContent = formatter(target);
    }
  };

  window.requestAnimationFrame(tick);
}

function resetBenchmarkMetric(element: HTMLElement) {
  element.dataset.value = "0";
  element.textContent = "-";
}

function averageBenchmarkResults(results: BenchmarkRunResult[]): BenchmarkRunResult | null {
  if (!results.length) return null;
  const total = results.reduce(
    (acc, result) => ({
      generationTokens: acc.generationTokens + result.generationTokens,
      generationMs: acc.generationMs + result.generationMs,
      generationTokensPerSecond: acc.generationTokensPerSecond + result.generationTokensPerSecond,
      totalMs: acc.totalMs + result.totalMs,
    }),
    {
      generationTokens: 0,
      generationMs: 0,
      generationTokensPerSecond: 0,
      totalMs: 0,
    },
  );
  const count = results.length;
  const latest = results[results.length - 1];
  return {
    index: count,
    prefill: benchmarkPrefillTargets.map((targetTokens) => {
      const matching = results
        .map((result) => result.prefill.find((entry) => entry.targetTokens === targetTokens))
        .filter((entry): entry is BenchmarkPrefillResult => Boolean(entry));
      const prefillTotal = matching.reduce(
        (acc, entry) => ({
          promptTokens: acc.promptTokens + entry.promptTokens,
          promptMs: acc.promptMs + entry.promptMs,
          tokensPerSecond: acc.tokensPerSecond + entry.tokensPerSecond,
        }),
        { promptTokens: 0, promptMs: 0, tokensPerSecond: 0 },
      );
      const prefillCount = Math.max(1, matching.length);
      return {
        targetTokens,
        promptTokens: Math.round(prefillTotal.promptTokens / prefillCount),
        promptMs: prefillTotal.promptMs / prefillCount,
        tokensPerSecond: prefillTotal.tokensPerSecond / prefillCount,
      };
    }),
    generationTokens: Math.round(total.generationTokens / count),
    generationMs: total.generationMs / count,
    generationTokensPerSecond: total.generationTokensPerSecond / count,
    totalMs: total.totalMs / count,
    text: latest.text,
  };
}

function updateBenchmarkCards(result: BenchmarkRunResult | null) {
  benchmarkResults.classList.toggle("has-results", result !== null);
  if (!result) {
    for (const targetTokens of benchmarkPrefillTargets) {
      resetBenchmarkMetric(benchmarkPrefillMetrics[targetTokens]);
    }
    resetBenchmarkMetric(benchmarkGenerateTokensPerSecond);
    resetBenchmarkMetric(benchmarkTotalTime);
    resetBenchmarkMetric(benchmarkOutputTokens);
    updateBenchmarkShareSummary(null);
    return;
  }

  for (const prefill of result.prefill) {
    const metric = benchmarkPrefillMetrics[prefill.targetTokens];
    if (metric) {
      animateBenchmarkNumber(metric, prefill.tokensPerSecond, (next) => formatBenchmarkNumber(next, 1));
    }
  }
  animateBenchmarkNumber(benchmarkGenerateTokensPerSecond, result.generationTokensPerSecond, (next) =>
    formatBenchmarkNumber(next, 1),
  );
  animateBenchmarkNumber(benchmarkTotalTime, result.totalMs / 1000, (next) =>
    formatBenchmarkNumber(next, 2),
  );
  animateBenchmarkNumber(benchmarkOutputTokens, result.generationTokens, (next) =>
    formatBenchmarkNumber(next, 0),
  );
  updateBenchmarkShareSummary(result);
}

function renderBenchmarkBars(result: BenchmarkRunResult | null) {
  const rows = Array.from($("benchmark-bars").querySelectorAll<HTMLElement>(".benchmark-bar-row"));
  const values = result
    ? [
        ...benchmarkPrefillTargets.map((targetTokens) => {
          const prefill = result.prefill.find((entry) => entry.targetTokens === targetTokens);
          return {
            label: prefill ? formatBenchmarkMs(prefill.promptMs) : "-",
            value: prefill?.promptMs ?? 0,
          };
        }),
        { label: formatBenchmarkMs(result.generationMs), value: result.generationMs },
        { label: formatBenchmarkMs(result.totalMs), value: result.totalMs },
      ]
    : [
        { label: "-", value: 0 },
        { label: "-", value: 0 },
        { label: "-", value: 0 },
        { label: "-", value: 0 },
        { label: "-", value: 0 },
      ];
  const max = Math.max(1, ...values.map((entry) => entry.value));

  rows.forEach((row, index) => {
    const bar = row.querySelector<HTMLElement>("i");
    const label = row.querySelector<HTMLElement>("b");
    const value = values[index] ?? values[0];
    if (bar) bar.style.width = `${Math.round((value.value / max) * 100)}%`;
    if (label) label.textContent = value.label === "-" ? "-" : `${value.label} ms`;
  });
}

function renderBenchmarkChart(results: BenchmarkRunResult[]) {
  benchmarkChart.replaceChildren();
  const average = averageBenchmarkResults(results);
  if (!average) {
    const empty = document.createElement("div");
    empty.className = "benchmark-chart-empty";
    empty.textContent = "No runs yet";
    benchmarkChart.append(empty);
    return;
  }

  const entries = [
    ...average.prefill.map((entry) => ({
      label: `${Math.round(entry.targetTokens / 1024)}K`,
      kicker: "Prefill",
      value: entry.tokensPerSecond,
    })),
    { label: "Gen", kicker: "Generate", value: average.generationTokensPerSecond },
  ];
  const max = Math.max(1, ...entries.map((entry) => entry.value));
  const legend = document.createElement("div");
  legend.className = "benchmark-chart-legend";
  legend.textContent = "Average tokens/s";
  benchmarkChart.append(legend);

  for (const entry of entries) {
    const column = document.createElement("div");
    const valueLabel = document.createElement("span");
    const track = document.createElement("div");
    const fill = document.createElement("i");
    const label = document.createElement("b");
    const kicker = document.createElement("em");
    const height = Math.max(6, (entry.value / max) * 100);

    column.className = `benchmark-chart-column ${entry.kicker === "Generate" ? "generate" : "prefill"}`;
    column.title = `${entry.kicker} ${entry.label}: ${formatBenchmarkNumber(entry.value, 1)} tokens/s`;
    valueLabel.textContent = formatBenchmarkNumber(entry.value, 1);
    fill.style.height = `${height}%`;
    track.append(fill);
    label.textContent = entry.label;
    kicker.textContent = entry.kicker;
    column.append(valueLabel, track, label, kicker);
    benchmarkChart.append(column);
  }
}

function resetBenchmarkResults() {
  benchmarkHistory = [];
  updateBenchmarkCards(null);
  renderBenchmarkBars(null);
  renderBenchmarkChart([]);
  benchmarkSample.textContent = "";
  benchmarkOutput.textContent = "";
}

function appendBenchmarkLog(line: string) {
  benchmarkOutput.textContent = [benchmarkOutput.textContent, line].filter(Boolean).join("\n");
  benchmarkOutput.scrollTop = benchmarkOutput.scrollHeight;
}

function updateBenchmarkShareSummary(result: BenchmarkRunResult | null) {
  if (!result) {
    benchmarkShareSummary.value =
      "LocalLLM benchmark\nPrefill: pending\nGenerate: pending";
    return;
  }

  const prefillSummary = benchmarkPrefillTargets
    .map((targetTokens) => {
      const prefill = result.prefill.find((entry) => entry.targetTokens === targetTokens);
      const label = targetTokens === 4098 ? "4K" : `${Math.round(targetTokens / 1024)}K`;
      return `${label} ${prefill ? formatBenchmarkNumber(prefill.tokensPerSecond, 1) : "-"} tok/s`;
    })
    .join(", ");

  benchmarkShareSummary.value = [
    "LocalLLM llama-server benchmark",
    `Prefill: ${prefillSummary}`,
    `Generate: ${formatBenchmarkNumber(result.generationTokensPerSecond, 1)} tok/s (${result.generationTokens} tokens)`,
    `Runs: ${benchmarkHistory.length || result.index}`,
  ].join("\n");
}

async function copyBenchmarkSummary() {
  await navigator.clipboard.writeText(benchmarkShareSummary.value);
  setMessage("Benchmark summary copied", "ok");
}

function shareBenchmarkOnTwitter() {
  openUrl(`https://twitter.com/intent/tweet?text=${encodeURIComponent(benchmarkShareSummary.value)}`).catch(showError);
}

async function resolveBenchmarkModel(baseUrl: string, signal: AbortSignal): Promise<string> {
  try {
    const response = await fetch(`${baseUrl}/v1/models`, { signal });
    if (!response.ok) return "local-model";
    const payload = (await response.json()) as { data?: Array<{ id?: string }> };
    return payload.data?.find((model) => model.id)?.id ?? "local-model";
  } catch (error) {
    if (error instanceof DOMException && error.name === "AbortError") throw error;
    return "local-model";
  }
}

function buildPrefillBenchmarkPrompt(targetTokens: number): string {
  const target = Math.min(32768, Math.max(128, Math.round(targetTokens)));
  const block =
    "PREFILL_BENCHMARK_BLOCK: local inference throughput validation. " +
    "This deterministic paragraph is intentionally repetitive and neutral so prompt processing dominates the measurement. " +
    "Measure prompt evaluation speed, memory bandwidth, KV setup, and scheduler overhead before generation begins. ";
  const chunks: string[] = [
    "You are running a standardized LocalLLM prefill benchmark. Read the following text and answer with one short checksum sentence.",
  ];

  while (estimateTokenCount(chunks.join("\n")) < target) {
    chunks.push(`${chunks.length.toString().padStart(4, "0")} ${block}`);
  }

  chunks.push("Checksum answer: summarize the benchmark input in one sentence.");
  return chunks.join("\n");
}

async function requestBenchmarkCompletion(
  prompt: string,
  nPredict: number,
  signal: AbortSignal,
): Promise<{ text: string; timings: Record<string, unknown>; elapsedMs: number }> {
  const baseUrl = benchmarkBaseUrl();
  const startedAt = performance.now();
  const response = await fetch(`${baseUrl}/completion`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      prompt,
      n_predict: nPredict,
      temperature: 0,
      top_k: 1,
      stream: false,
      cache_prompt: false,
    }),
    signal,
  });

  if (!response.ok) {
    const detail = (await response.text()).trim();
    throw new Error(`Server returned ${response.status}${detail ? `: ${detail}` : ""}`);
  }

  const payload = await response.json();
  const text = String(payload.content ?? payload.response ?? payload.choices?.[0]?.text ?? "");
  return {
    text,
    timings: payload.timings ?? {},
    elapsedMs: Math.max(1, performance.now() - startedAt),
  };
}

async function runBenchmarkRequest(
  index: number,
  signal: AbortSignal,
): Promise<BenchmarkRunResult> {
  const settings = readBenchmarkSettings();
  const generateTokens = Math.max(1, Number(settings.generateTokens) || 128);
  const prefill: BenchmarkPrefillResult[] = [];

  for (const targetTokens of benchmarkPrefillTargets) {
    const prefillPrompt = buildPrefillBenchmarkPrompt(targetTokens);
    const prefillResult = await requestBenchmarkCompletion(prefillPrompt, 1, signal);
    const timings = prefillResult.timings;
    const promptTokens = Math.max(0, Number(timings.prompt_n) || estimateTokenCount(prefillPrompt));
    const promptMs = Math.max(1, Number(timings.prompt_ms) || prefillResult.elapsedMs);
    const tokensPerSecond =
      Number(timings.prompt_per_second) ||
      (promptTokens > 0 ? promptTokens / (promptMs / 1000) : 0);

    prefill.push({ targetTokens, promptTokens, promptMs, tokensPerSecond });
    appendBenchmarkLog(
      `Run ${index} prefill ${targetTokens}: ${formatBenchmarkNumber(
        tokensPerSecond,
        1,
      )} tokens/s (${promptTokens} tokens)`,
    );
  }

  const generationResult = await requestBenchmarkCompletion(settings.prompt, generateTokens, signal);
  const generationTimings = generationResult.timings;
  const text = generationResult.text;
  const generationTokens = Math.max(0, Number(generationTimings.predicted_n) || estimateTokenCount(text));
  const generationMs = Math.max(1, Number(generationTimings.predicted_ms) || generationResult.elapsedMs);
  const generationTotalMs =
    Number(generationTimings.prompt_ms || 0) + Number(generationTimings.predicted_ms || 0) ||
    generationResult.elapsedMs;
  const prefillTotalMs = prefill.reduce((sum, entry) => sum + entry.promptMs, 0);
  const totalMs = Math.max(1, prefillTotalMs + generationTotalMs);
  const generationTokensPerSecond =
    Number(generationTimings.predicted_per_second) ||
    (generationTokens > 0 ? generationTokens / (generationMs / 1000) : 0);

  benchmarkSample.textContent = text;
  return {
    index,
    prefill,
    generationTokens,
    generationMs,
    generationTokensPerSecond,
    totalMs,
    text,
    error: index > 0 ? undefined : "Run index unavailable",
  };
}

async function runBenchmark() {
  const mode = (value("benchmark-mode") as BenchmarkMode) || "performance";
  if (mode === "evaluation") {
    return runEvalBenchmark();
  }

  if (!server.running) {
    setMessage("Start llama-server before running the benchmark", "warn");
    syncBenchmarkServerState();
    return;
  }

  resetBenchmarkResults();
  const controller = new AbortController();
  benchmarkAbortController = controller;
  syncBenchmarkServerState();
  setMessage("Running server benchmark", "info");

  try {
    const settings = readBenchmarkSettings();
    const runs = Math.min(10, Math.max(1, Number(settings.runs) || 1));
    const baseUrl = benchmarkBaseUrl();
    const model = await resolveBenchmarkModel(baseUrl, controller.signal);
    appendBenchmarkLog(`Server: ${baseUrl}`);
    appendBenchmarkLog(`Model: ${model}`);
    appendBenchmarkLog(`Runs: ${runs} / prefill: ${benchmarkPrefillTargets.join(", ")} / generate: ${settings.generateTokens}`);

    for (let index = 1; index <= runs; index += 1) {
      appendBenchmarkLog(`Run ${index} started`);
      const result = await runBenchmarkRequest(index, controller.signal);
      benchmarkHistory.push(result);
      const average = averageBenchmarkResults(benchmarkHistory);
      updateBenchmarkCards(average);
      renderBenchmarkBars(average);
      renderBenchmarkChart(benchmarkHistory);
      appendBenchmarkLog(
        `Run ${index}: generate ${formatBenchmarkNumber(
          result.generationTokensPerSecond,
        1,
        )} tokens/s (${result.generationTokens} tokens), ${formatBenchmarkNumber(result.totalMs / 1000, 2)} s total`,
      );
    }

    setMessage("Benchmark complete", "ok");
  } catch (error) {
    setMessage(String(error), "error");
    appendBenchmarkLog(String(error));
  } finally {
    benchmarkAbortController = null;
    syncBenchmarkServerState();
  }
}


async function saveEvalSettings() {
  window.localStorage.setItem(evalSettingsStorageKey, JSON.stringify(readEvalSettings()));
  setMessage("Evaluation settings saved", "ok");
}
async function saveBenchmarkSettings() {
  window.localStorage.setItem(benchmarkSettingsStorageKey, JSON.stringify(readBenchmarkSettings()));
  setMessage("Benchmark settings saved", "ok");
}

function scorePythonSample(actual: string, expected: string): boolean {
  const actualMatch = actual.match(/def\s+\w+\s*\([^)]*\)[\s:]/s);
  const expectedMatch = expected.match(/def\s+\w+\s*\([^)]*\)[\s:]/s);
  return Boolean(actualMatch && expectedMatch);
}

function scoreGsm8kSample(actual: string, expected: string): boolean {
  const actualMatch = actual.match(/[\d]+\.?[\d]*/);
  const actualNum = actualMatch ? Number(actualMatch[0]) : NaN;
  const expectedNum = Number(expected.trim());
  if (isNaN(actualNum) || isNaN(expectedNum)) return false;
  return Math.abs(actualNum - expectedNum) < 0.01;
}

function scoreMultipleChoiceSample(actual: string, expected: string): boolean {
  const actualMatch = actual.match(/[A-D]/i);
  return Boolean(actualMatch && actualMatch[0].toUpperCase() === expected.trim().toUpperCase());
}

function scoreTruthfulqaSample(actual: string, expected: string): boolean {
  return actual.toLowerCase().trim().includes(expected.toLowerCase().trim());
}

async function runEvalBenchmark() {
  if (!server.running) {
    setMessage("Start llama-server before running the evaluation", "warn");
    syncBenchmarkServerState();
    return;
  }

  resetEvalResults();
  const controller = new AbortController();
  evalAbortController = controller;
  syncBenchmarkServerState();

  try {
    const settings = readEvalSettings();
    const dataset = evalBenchmarkDatasets[settings.benchmarkType] || [];
    const sampleCount = Math.min(dataset.length, Math.max(1, Number(settings.sampleCount) || dataset.length));
    const temperature = Math.max(0, Math.min(2, Number(settings.temperature) || 0.2));
    const baseUrl = benchmarkBaseUrl();

    const scorer = getScorerForBenchmark(settings.benchmarkType);
    const samples: EvalSampleResult[] = [];
    const startedAt = performance.now();

    appendBenchmarkLog(`Evaluation: ${evalBenchmarkLabels[settings.benchmarkType]}`);
    appendBenchmarkLog(`Samples: ${sampleCount} / Temperature: ${temperature}`);

    for (let i = 0; i < sampleCount; i++) {
      if (controller.signal.aborted) break;

      const sample = dataset[i];
      appendBenchmarkLog(`Sample ${i + 1}/${sampleCount}...`);

      try {
        const result = await requestEvalCompletion(sample.prompt, temperature, baseUrl, controller.signal);
        const passed = scorer(result, sample.answer);

        samples.push({
          index: i + 1,
          prompt: sample.prompt,
          expected: sample.answer,
          actual: result,
          passed,
        });

        appendBenchmarkLog(`  ${passed ? "PASS" : "FAIL"}`);
        updateEvalProgress(samples);
      } catch (err) {
        samples.push({
          index: i + 1,
          prompt: sample.prompt,
          expected: sample.answer,
          actual: String(err),
          passed: false,
        });
        appendBenchmarkLog(`  ERROR: ${err}`);
      }
    }

    const elapsedMs = performance.now() - startedAt;
    const passedSamples = samples.filter(s => s.passed).length;
    const score = samples.length > 0 ? (passedSamples / samples.length) * 100 : 0;

    const result: EvalBenchmarkResult = {
      benchmarkType: settings.benchmarkType,
      totalSamples: samples.length,
      passedSamples,
      score,
      scorePercent: Math.round(score * 10) / 10,
      samples,
      elapsedMs,
    };

    evalBenchmarkHistory.push(result);
    updateEvalResults(result);
    renderEvalSamples(samples);

    setMessage(`Evaluation complete: ${result.scorePercent}%`, passedSamples > 0 ? "ok" : "warn");
  } catch (error) {
    setMessage(String(error), "error");
    appendBenchmarkLog(String(error));
  } finally {
    evalAbortController = null;
    syncBenchmarkServerState();
  }
}

function getScorerForBenchmark(type: EvalBenchmarkType): (actual: string, expected: string) => boolean {
  switch (type) {
    case "humaneval":
    case "mbpp":
      return scorePythonSample;
    case "gsm8k":
      return scoreGsm8kSample;
    case "mmlu":
    case "arc":
    case "hellaswag":
    case "winogrande":
      return scoreMultipleChoiceSample;
    case "truthfulqa":
      return scoreTruthfulqaSample;
  }
}

async function requestEvalCompletion(
  prompt: string,
  temperature: number,
  baseUrl: string,
  signal: AbortSignal,
): Promise<string> {
  const response = await fetch(`${baseUrl}/completion`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      prompt,
      n_predict: 256,
      temperature,
      stop: ["\n\n", "```"],
    }),
    signal,
  });

  if (!response.ok) {
    throw new Error(`Server returned ${response.status}`);
  }

  const data = await response.json();
  return data.content || data.result || "";
}

function resetEvalResults() {
  evalBenchmarkHistory = [];
  evalScorePercent.textContent = "-";
  evalPassedCount.textContent = "-";
  evalTotalCount.textContent = "-";
  evalElapsedTime.textContent = "-";
  evalSamplesList.innerHTML = "";
}

function updateEvalProgress(samples: EvalSampleResult[]) {
  const passed = samples.filter(s => s.passed).length;
  const score = samples.length > 0 ? (passed / samples.length) * 100 : 0;
  evalScorePercent.textContent = Math.round(score * 10) / 10 + "";
  evalPassedCount.textContent = passed + "";
  evalTotalCount.textContent = samples.length + "";
}

function updateEvalResults(result: EvalBenchmarkResult) {
  evalScorePercent.textContent = result.scorePercent + "";
  evalPassedCount.textContent = result.passedSamples + "";
  evalTotalCount.textContent = result.totalSamples + "";
  evalElapsedTime.textContent = (result.elapsedMs / 1000).toFixed(2);
}

function renderEvalSamples(samples: EvalSampleResult[]) {
  evalSamplesList.innerHTML = "";

  for (const sample of samples) {
    const card = document.createElement("div");
    card.className = `eval-sample-card ${sample.passed ? "eval-pass" : "eval-fail"}`;

    card.innerHTML = `
      <div class="eval-sample-header">
        <span class="eval-sample-index">Sample ${sample.index}</span>
        <span class="eval-sample-status ${sample.passed ? "pass" : "fail"}">${sample.passed ? "PASS" : "FAIL"}</span>
      </div>
      <div class="eval-sample-prompt">${escapeHtml(sample.prompt)}</div>
      <div class="eval-sample-expected">${escapeHtml(sample.expected)}</div>
      <div class="eval-sample-actual">${escapeHtml(sample.actual)}</div>
    `;

    evalSamplesList.appendChild(card);
  }
}

function escapeHtml(text: string): string {
  const div = document.createElement("div");
  div.textContent = text;
  return div.innerHTML;
}

function bindEvents() {
  $("save-settings").addEventListener("click", () => saveConfig());
  $("reset-app-data").addEventListener("click", () => resetEverything().catch(showError));
  $("theme-select").addEventListener("change", () => applyTheme(value("theme-select")));
  $("apply-advanced-preset").addEventListener("click", applyAdvancedPreset);
  $("pick-draft-model").addEventListener("click", () => chooseDraftModel().catch(showError));
  $("profile-select").addEventListener("change", loadSelectedProfileFromSelector);
  $("new-model-profile").addEventListener("click", startNewProfile);
  $("save-model-profile").addEventListener("click", () => saveSelectedModelProfile().catch(showError));
  $("delete-model-profile").addEventListener("click", () => deleteSelectedModelProfile().catch(showError));
  $("load-default-profile").addEventListener("click", loadDefaultProfileForSelectedModel);
  $("rescan-models").addEventListener("click", () => refreshModels());
  $("add-gguf").addEventListener("click", () => addManualModel().catch(showError));
  $("pick-model-dir").addEventListener("click", () => chooseDirectory("model-dir").catch(showError));
  $("pick-download-dir").addEventListener("click", () => chooseDirectory("hf-target-dir").catch(showError));
  $("pick-server").addEventListener("click", () => chooseExecutable("llama-server-path").catch(showError));
  $("pick-cli").addEventListener("click", () => chooseExecutable("llama-cli-path").catch(showError));
  $("install-llama-cpp").addEventListener("click", () => installLlamaCpp().catch(showError));
  $("pick-mmproj").addEventListener("click", () => chooseMmproj().catch(showError));
  $("start-server").addEventListener("click", () => startServer().catch(showError));
  $("stop-server").addEventListener("click", () => stopServer().catch(showError));
  $("download-model").addEventListener("click", () => downloadModel().catch(showError));
  $("save-benchmark-settings").addEventListener("click", () => saveBenchmarkSettings().catch(showError));
  $("run-benchmark").addEventListener("click", () => runBenchmark().catch(showError));
  $("benchmark-preset").addEventListener("change", applyBenchmarkPreset);
  $("benchmark-mode").addEventListener("change", () => setBenchmarkMode(value("benchmark-mode") as BenchmarkMode));
  $("save-benchmark-settings").addEventListener("click", () => saveEvalSettings().catch(showError));
  $("copy-benchmark-summary").addEventListener("click", () => copyBenchmarkSummary().catch(showError));
  $("share-benchmark-twitter").addEventListener("click", shareBenchmarkOnTwitter);
  $("app-tab-control").addEventListener("click", () => setAppTab("control"));
  $("app-tab-benchmark").addEventListener("click", () => setAppTab("benchmark"));
  $("app-tab-agent").addEventListener("click", () => setAppTab("agent"));
  $("app-tab-webui").addEventListener("click", () => setAppTab("webui"));
  $("reload-web-ui").addEventListener("click", () => syncWebUiFrame(true));
  $("hf-search-models").addEventListener("click", () => loadHfModels().catch(showError));
  $("hf-model-search").addEventListener("input", scheduleHfModelSearch);
  $("hf-model-search").addEventListener("keydown", (event) => {
    if ((event as KeyboardEvent).key !== "Enter") return;
    event.preventDefault();
    void loadHfModels();
  });
  $("hf-model-sort-tabs").addEventListener("click", (event) => {
    const button = (event.target as HTMLElement).closest<HTMLButtonElement>("button[data-hf-sort]");
    const sort = button?.dataset.hfSort as HfModelSort | undefined;
    if (!sort) return;
    setActiveHfModelSort(sort);
    void loadHfModels();
  });
  hfModelResults.addEventListener("click", (event) => {
    const row = (event.target as HTMLElement).closest<HTMLButtonElement>(".hf-model-row");
    if (!row?.dataset.repoId) return;
    void selectHfModel(row.dataset.repoId).catch(showError);
  });
  $("hf-file-select").addEventListener("change", handleHfFileSelectChange);
  $("hf-pattern").addEventListener("input", handleHfPatternInput);
  $("hf-repo").addEventListener("input", scheduleHfRepoFileLoad);
  $("hf-repo").addEventListener("change", () => loadHfRepoFiles().catch(showError));
  $("hf-revision").addEventListener("input", scheduleHfRepoFileLoad);
  $("hf-token").addEventListener("change", () => loadHfRepoFiles().catch(showError));
  $("open-server").addEventListener("click", () => {
    const url = server.url || `http://${value("host")}:${value("port")}`;
    openUrl(url).catch(showError);
  });
  $("server-log").addEventListener("click", () => {
    if (server.logPath) openPath(server.logPath).catch(showError);
  });
  $("copy-command").addEventListener("click", async () => {
    await navigator.clipboard.writeText(commandPreview.textContent ?? "");
    setMessage("Command copied", "ok");
  });
  modelList.addEventListener("click", (event) => {
    const row = (event.target as HTMLElement).closest<HTMLButtonElement>(".model-row");
    if (!row?.dataset.path) return;
    selectModel(row.dataset.path);
  });
  modelSelect.addEventListener("input", handleModelSelectChange);
  modelSelect.addEventListener("change", handleModelSelectChange);

  for (const group of optionalGroups) {
    $(group.toggleId).addEventListener("change", () => {
      syncOptionalGroups();
      schedulePreview();
    });
  }

  for (const id of [
    "host",
    "port",
    "ctx-size",
    "gpu-layers",
    "threads",
    "batch-size",
    "ubatch-size",
    "parallel",
    "terminal-mode",
    "cache-type-k",
    "cache-type-v",
    "flash-attention",
    "kvu",
    "kv-offload",
    "op-offload",
    "device",
    "split-mode",
    "tensor-split",
    "main-gpu",
    "fit",
    "fit-target",
    "fit-ctx",
    "temperature",
    "top-k",
    "top-p",
    "min-p",
    "typical-p",
    "repeat-penalty",
    "presence-penalty",
    "frequency-penalty",
    "spec-type",
    "spec-draft-n-max",
    "spec-draft-n-min",
    "spec-draft-p-min",
    "spec-draft-p-split",
    "spec-ngram-mod-n-match",
    "spec-ngram-mod-n-min",
    "spec-ngram-mod-n-max",
    "spec-draft-model-path",
    "reasoning-preserve",
    "no-cpu-moe",
    "reasoning-format",
    "reasoning-budget",
    "chat-template-kwargs",
    "reasoning",
    "mmproj",
    "extra-args",
    "llama-server-path",
    "llama-cli-path",
  ]) {
    $(id).addEventListener("input", schedulePreview);
    $(id).addEventListener("change", schedulePreview);
  }
  $("no-host").addEventListener("change", schedulePreview);
  $("cpu-moe").addEventListener("change", schedulePreview);
  $("no-mmap").addEventListener("change", schedulePreview);
  $("mlock").addEventListener("change", schedulePreview);
  $("embeddings").addEventListener("change", schedulePreview);
  $("tools-all").addEventListener("change", schedulePreview);
  $("reasoning-preserve").addEventListener("change", () => {
    syncPreserveThinking();
    schedulePreview();
  });

}

function showError(error: unknown) {
  setMessage(String(error), "error");
}

async function boot() {
  initWindowControls();
  initTheme();
  bindSelectionGuard();
  bindButtonRipples();
  initPaneResize();
  initAgent();
  bindEvents();
  setModelLoading("startup", true);
  try {
    await yieldToUi();
    config = await invoke<AppConfig>("load_config");
    hydrateUi();
    await refreshTools();
    await loadCachedModels();
  } finally {
    setModelLoading("startup", false);
  }
  refreshModelsInBackground("Ready");
  await refreshServerStatus();
  await updatePreview();
  window.setInterval(refreshServerStatus, 1500);
}

window.addEventListener("DOMContentLoaded", () => {
  boot().catch(showError);
});
