import { invoke } from "@tauri-apps/api/core";
import { ask, open } from "@tauri-apps/plugin-dialog";
import { openPath, openUrl } from "@tauri-apps/plugin-opener";

type Tone = "info" | "ok" | "warn" | "error";

interface ModelEntry {
  id: string;
  name: string;
  path: string;
  source: string;
  sizeBytes: number;
  metadata?: GgufModelMetadata | null;
}

interface GgufModelMetadata {
  architecture?: string | null;
  contextLength?: number | null;
  blockCount?: number | null;
  embeddingLength?: number | null;
  attentionHeadCount?: number | null;
  attentionHeadCountKv?: number | null;
  attentionKeyLength?: number | null;
  attentionValueLength?: number | null;
}

interface ModelProfile {
  modelPath: string;
  name: string;
  server: ServerConfig;
}

interface ServerConfig {
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
  enableGpuMemoryOptions: boolean;
  fit: string;
  fitTarget: string;
  fitCtx: number;
  devices: string;
  tensorSplit: string;
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
  noCpuMoe: number;
  enableReasoningOptions: boolean;
  preserveThinking: boolean;
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

interface AppConfig {
  llamaServerPath: string;
  llamaCliPath: string;
  modelDir: string;
  hfToken: string;
  manualModels: ModelEntry[];
  modelProfiles: ModelProfile[];
  server: ServerConfig;
}

interface ToolDiscovery {
  llamaServer: string | null;
  llamaCli: string | null;
  hfCli: string | null;
  huggingfaceCli: string | null;
}

interface ServerStatus {
  running: boolean;
  pid: number | null;
  command: string;
  url: string;
  modelPath: string;
  logPath: string;
  startedAt: number | null;
}

interface CommandOutput {
  success: boolean;
  statusCode: number | null;
  command: string;
  stdout: string;
  stderr: string;
}

interface LlamaServerProcess {
  pid: number;
  commandLine: string;
}

interface DownloadRequest {
  repoId: string;
  pattern: string;
  revision: string;
  targetDir: string;
  token: string;
  force: boolean;
  maxWorkers: number;
}

interface HfRepoFile {
  path: string;
  sizeBytes: number | null;
}

type HfModelSort = "trending" | "updated" | "downloads";
type AppTab = "control" | "webui";

interface HfModelSummary {
  id: string;
  downloads: number | null;
  likes: number | null;
  lastModified: string | null;
  createdAt: string | null;
  pipelineTag: string | null;
  libraryName: string | null;
  trendingScore: number | null;
}

interface LlamaCppInstallRequest {
  package: string;
  targetDir: string;
}

interface LlamaCppInstallResult {
  releaseTag: string;
  assetName: string;
  installDir: string;
  llamaServerPath: string;
  llamaCliPath: string;
  command: string;
  stdout: string;
  stderr: string;
}

const $ = <T extends HTMLElement>(id: string): T => {
  const element = document.getElementById(id);
  if (!element) {
    throw new Error(`Missing element #${id}`);
  }
  return element as T;
};

const editableInputTypes = new Set(["email", "number", "password", "search", "tel", "text", "url"]);

function isEditableTextbox(element: Element | null): element is HTMLInputElement | HTMLTextAreaElement {
  if (element instanceof HTMLTextAreaElement) {
    return !element.disabled && !element.readOnly;
  }
  if (element instanceof HTMLInputElement) {
    return editableInputTypes.has(element.type) && !element.disabled && !element.readOnly;
  }
  return false;
}

function editableTextboxFromTarget(target: EventTarget | null): HTMLInputElement | HTMLTextAreaElement | null {
  if (!(target instanceof Element)) return null;
  const textbox = target.closest("input, textarea");
  return isEditableTextbox(textbox) ? textbox : null;
}

function clearDocumentSelection() {
  const selection = window.getSelection();
  if (selection && !selection.isCollapsed) {
    selection.removeAllRanges();
  }
}

function clearActiveTextboxSelection() {
  const activeElement = document.activeElement;
  if (!isEditableTextbox(activeElement)) return;

  const caretPosition = activeElement.value.length;
  try {
    activeElement.setSelectionRange(caretPosition, caretPosition);
  } catch {
    // Some editable input types, such as number, do not expose setSelectionRange.
  }
}

function bindSelectionGuard() {
  document.addEventListener("contextmenu", (event) => {
    event.preventDefault();
  });

  document.addEventListener(
    "pointerdown",
    (event) => {
      if (!editableTextboxFromTarget(event.target)) {
        clearActiveTextboxSelection();
        clearDocumentSelection();
      }
    },
    true,
  );

  document.addEventListener("selectstart", (event) => {
    if (editableTextboxFromTarget(event.target)) return;
    event.preventDefault();
    clearActiveTextboxSelection();
    clearDocumentSelection();
  });

  document.addEventListener("selectionchange", () => {
    if (isEditableTextbox(document.activeElement)) return;
    clearDocumentSelection();
  });

  document.addEventListener("keydown", (event) => {
    if (!(event.ctrlKey || event.metaKey) || event.key.toLowerCase() !== "a") return;
    if (editableTextboxFromTarget(event.target)) return;
    event.preventDefault();
    clearActiveTextboxSelection();
    clearDocumentSelection();
  });
}

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
const vramTotal = $("vram-total");
const vramDetail = $("vram-detail");
const downloadOutput = $("download-output");
const downloadFileSelect = $("hf-file-select") as HTMLSelectElement;
const hfModelResults = $("hf-model-results");
const hfModelStatus = $("hf-model-status");
const llamaInstallOutput = $("llama-install-output");
const preserveThinkingDefault = '{"preserve_thinking": true}';
const themeStorageKey = "localllm-theme";
const paneWidthStorageKey = "localllm-left-pane-width";
const themes = new Set(["spotify", "sage", "graphite", "paper", "webmcp"]);
const minLeftPaneWidth = 300;
const minRightPaneWidth = 320;

let config: AppConfig;
let models: ModelEntry[] = [];
let tools: ToolDiscovery = {
  llamaServer: null,
  llamaCli: null,
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
    enableGpuMemoryOptions: checked("enable-gpu-memory-options"),
    fit: value("fit"),
    fitTarget: value("fit-target"),
    fitCtx: numberValue("fit-ctx", 0),
    devices: value("devices"),
    tensorSplit: value("tensor-split"),
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
    noCpuMoe: numberValue("no-cpu-moe", 0),
    enableReasoningOptions: checked("enable-reasoning-options"),
    preserveThinking: checked("preserve-thinking"),
    reasoningFormat: value("reasoning-format"),
    reasoningBudget: value("reasoning-budget"),
    chatTemplateKwargs: checked("preserve-thinking")
      ? value("chat-template-kwargs") || preserveThinkingDefault
      : "",
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
  setChecked("enable-gpu-memory-options", server.enableGpuMemoryOptions);
  setValue("fit", server.fit);
  setValue("fit-target", server.fitTarget);
  setValue("fit-ctx", server.fitCtx);
  setValue("devices", server.devices ?? "");
  setValue("tensor-split", server.tensorSplit ?? "");
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
  setValue("no-cpu-moe", server.noCpuMoe);
  setChecked("enable-reasoning-options", server.enableReasoningOptions);
  setChecked("preserve-thinking", server.preserveThinking ?? true);
  setValue("reasoning-format", server.reasoningFormat);
  setValue("reasoning-budget", server.reasoningBudget);
  setValue("chat-template-kwargs", server.chatTemplateKwargs || (server.preserveThinking === false ? "" : preserveThinkingDefault));
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
}

async function saveConfig(message = "Saved") {
  config = readConfigFromUi();
  config = await invoke<AppConfig>("save_config", { config });
  setMessage(message, "ok");
}

async function resetEverything() {
  const confirmed = await ask(
    "Reset LocalLLM settings, presets, manual models, Hugging Face token, model cache, logs, theme, and layout?\n\nThis will not delete downloaded GGUF models or your llama.cpp install folder.",
    {
      title: "Reset LocalLLM",
      kind: "warning",
      okLabel: "Reset everything",
      cancelLabel: "Cancel",
    },
  );
  if (!confirmed) return;

  setMessage("Resetting LocalLLM data", "info");
  config = await invoke<AppConfig>("reset_app_data");
  window.localStorage.removeItem(themeStorageKey);
  window.localStorage.removeItem(paneWidthStorageKey);

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

function modelExists(path: string): boolean {
  return models.some((model) => model.path === path);
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
    (row) => row.dataset.path === activeModelPath,
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

function createModelRow(model: ModelEntry): HTMLButtonElement {
  const row = document.createElement("button");
  row.type = "button";
  row.className = `model-row ${model.path === activeModelPath ? "selected" : ""}`;
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
  return models.find((model) => model.path === activeModelPath);
}

function selectedModelProfileIndex(): number {
  return config.modelProfiles.findIndex((profile) => profile.modelPath === activeModelPath);
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

  const profiledModelPaths = new Set(config.modelProfiles.map((profile) => profile.modelPath));
  const missingModels = models.filter((model) => !profiledModelPaths.has(model.path));
  if (missingModels.length === 0) return;

  const modelProfiles = [
    ...config.modelProfiles,
    ...missingModels.map((model) => ({
      modelPath: model.path,
      name: defaultProfileName(model),
      server: serverForModel(config.server, model.path),
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
    const modelName = models.find((model) => model.path === profile.modelPath)?.name ?? profile.modelPath;
    const option = document.createElement("option");
    option.value = `profile:${index}`;
    option.textContent = `${profile.name} - ${modelName}`;
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
    ? models.find((entry) => entry.path === profile.modelPath)?.name ?? profile.modelPath
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
  selectedProfileOverrideIndex = null;
  creatingProfile = false;
  hydrateServerUi(serverForModel(config.server, modelPath));
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

async function addManualModel() {
  const selected = await open({
    multiple: false,
    filters: [{ name: "GGUF", extensions: ["gguf"] }],
    defaultPath: value("model-dir") || undefined,
  });
  if (typeof selected !== "string") return;

  const model = await invoke<ModelEntry>("model_from_path", { path: selected });
  if (!config.manualModels.some((entry) => entry.path === model.path)) {
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
  setValue("fit", "");
  setValue("fit-target", "");
  setValue("fit-ctx", 0);
  setValue("devices", "");
  setValue("tensor-split", "");
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
  setValue("batch-size", 0);
  setValue("ubatch-size", 0);
  setValue("parallel", -1);
  setValue("spec-ngram-mod-n-match", 0);
  setValue("spec-ngram-mod-n-min", 0);
  setValue("spec-ngram-mod-n-max", 0);
  setValue("no-cpu-moe", 0);
  setValue("reasoning-format", "");
  setValue("reasoning-budget", "");
  setChecked("preserve-thinking", true);
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
    const closeExisting = await ask(
      `Another llama-server process is already running in the background.\n\n${processList}${extraCount}\n\nDo you want LocalLLM to close the existing process before starting this server?\n\nChoose No to leave it running and start another llama-server.`,
      {
        title: "Another llama-server is running",
        kind: "warning",
        okLabel: "Yes, close it",
        cancelLabel: "No, start another",
      },
    );
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
    const current = models.find((entry) => entry.path === model.path);
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
  const model = models.find((entry) => entry.path === activeModelPath);
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
      value("devices") && `devices ${value("devices")}`,
      value("tensor-split") && `split ${value("tensor-split")}`,
      value("fit") && `fit ${value("fit")}`,
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
    const reasoningNotes = [value("reasoning") && `reasoning ${value("reasoning")}`, value("reasoning-budget") && `budget ${value("reasoning-budget")}`, checked("preserve-thinking") && "preserve thinking"].filter(Boolean);
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
  const enabled = checked("enable-reasoning-options") && checked("preserve-thinking");
  const input = $("chat-template-kwargs") as HTMLTextAreaElement;
  input.disabled = !enabled;
  if (enabled && !input.value.trim()) {
    input.value = preserveThinkingDefault;
  }
}

function bindEvents() {
  $("save-settings").addEventListener("click", () => saveConfig());
  $("reset-app-data").addEventListener("click", () => resetEverything().catch(showError));
  $("theme-select").addEventListener("change", () => applyTheme(value("theme-select")));
  $("apply-advanced-preset").addEventListener("click", applyAdvancedPreset);
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
  $("app-tab-control").addEventListener("click", () => setAppTab("control"));
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
    "devices",
    "tensor-split",
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
  $("no-mmap").addEventListener("change", schedulePreview);
  $("mlock").addEventListener("change", schedulePreview);
  $("embeddings").addEventListener("change", schedulePreview);
  $("tools-all").addEventListener("change", schedulePreview);
  $("jinja").addEventListener("change", schedulePreview);
  $("verbose").addEventListener("change", schedulePreview);
  $("preserve-thinking").addEventListener("change", () => {
    syncPreserveThinking();
    schedulePreview();
  });
}

function showError(error: unknown) {
  setMessage(String(error), "error");
}

async function boot() {
  initTheme();
  bindSelectionGuard();
  initPaneResize();
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
