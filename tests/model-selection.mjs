import { spawn, spawnSync } from "node:child_process";
import { existsSync } from "node:fs";
import { setTimeout as delay } from "node:timers/promises";
import { fileURLToPath } from "node:url";
import { chromium } from "playwright-core";

const appPort = 14200 + Math.floor(Math.random() * 1000);
const appUrl = `http://127.0.0.1:${appPort}/`;
const chromePath = findChromePath();
const repoRoot = fileURLToPath(new URL("..", import.meta.url));

const models = [
  model("a", "alpha.gguf", "\\\\?\\D:\\LMStudio\\models\\alpha.gguf", 1.7),
  model("b", "bravo.gguf", "\\\\?\\D:\\LMStudio\\models\\bravo.gguf", 15.6),
  model("c", "charlie.gguf", "\\\\?\\D:\\LMStudio\\models\\charlie.gguf", 17.4),
];

function model(id, name, path, gb) {
  return {
    id,
    name,
    path,
    source: "directory",
    sizeBytes: Math.round(gb * 1024 ** 3),
    metadata: null,
  };
}

async function isAppUp() {
  try {
    const response = await fetch(appUrl);
    return response.ok;
  } catch {
    return false;
  }
}

async function ensureDevServer() {
  if (await isAppUp()) return undefined;

  const devServer = devServerCommand();
  const child = spawn(devServer.command, devServer.args, {
    cwd: repoRoot,
    shell: false,
    stdio: "pipe",
  });

  for (let attempt = 0; attempt < 50; attempt += 1) {
    if (await isAppUp()) return child;
    await delay(100);
  }

  child.kill();
  throw new Error("Vite dev server did not start");
}

async function expectValue(page, selector, expected, label) {
  const actual = await page.locator(selector).inputValue();
  if (actual !== expected) {
    throw new Error(`${label}: expected ${expected}, got ${actual}`);
  }
}

async function expectTextIncludes(page, selector, expected, label) {
  const actual = await page.locator(selector).textContent();
  if (!actual?.includes(expected)) {
    throw new Error(`${label}: expected text to include ${expected}, got ${actual}`);
  }
}

async function waitTextIncludes(page, selector, expected, label) {
  await page.waitForFunction(
    ({ selector: targetSelector, expectedText }) =>
      document.querySelector(targetSelector)?.textContent?.includes(expectedText),
    { selector, expectedText: expected },
    { timeout: 5000 },
  );
  await expectTextIncludes(page, selector, expected, label);
}

async function waitValueIncludes(page, selector, expected, label) {
  await page.waitForFunction(
    ({ selector: targetSelector, expectedText }) => {
      const element = document.querySelector(targetSelector);
      return element && "value" in element && String(element.value).includes(expectedText);
    },
    { selector, expectedText: expected },
    { timeout: 5000 },
  );
  const actual = await page.locator(selector).inputValue();
  if (!actual.includes(expected)) {
    throw new Error(`${label}: expected value to include ${expected}, got ${actual}`);
  }
}

function devServerCommand() {
  const npmArgs = ["run", "dev", "--", "--host", "127.0.0.1", "--port", String(appPort), "--strictPort"];
  if (process.platform === "win32") {
    return { command: "cmd.exe", args: ["/c", "npm", ...npmArgs] };
  }
  return { command: "npm", args: npmArgs };
}

function commandWorks(command) {
  const result = spawnSync(command, ["--version"], {
    shell: false,
    stdio: "ignore",
  });
  return result.status === 0;
}

function resolveCommandPath(command) {
  const lookup = process.platform === "win32" ? "where" : "which";
  const result = spawnSync(lookup, [command], {
    encoding: "utf8",
    shell: false,
    stdio: ["ignore", "pipe", "ignore"],
  });
  const path = result.stdout?.split(/\r?\n/).find(Boolean)?.trim();
  return path || (commandWorks(command) ? command : "");
}

function findChromePath() {
  if (process.env.CHROME_PATH) return process.env.CHROME_PATH;
  if (process.env.PLAYWRIGHT_CHROMIUM_EXECUTABLE_PATH) {
    return process.env.PLAYWRIGHT_CHROMIUM_EXECUTABLE_PATH;
  }

  const candidates =
    process.platform === "win32"
      ? [
          "C:/Program Files/Google/Chrome/Application/chrome.exe",
          "C:/Program Files (x86)/Google/Chrome/Application/chrome.exe",
          "C:/Program Files/Microsoft/Edge/Application/msedge.exe",
        ]
      : process.platform === "darwin"
        ? [
            "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
            "/Applications/Chromium.app/Contents/MacOS/Chromium",
            "/Applications/Microsoft Edge.app/Contents/MacOS/Microsoft Edge",
          ]
        : [
            "google-chrome",
            "google-chrome-stable",
            "chromium",
            "chromium-browser",
            "microsoft-edge",
            "/usr/bin/google-chrome",
            "/usr/bin/chromium",
            "/usr/bin/chromium-browser",
          ];

  for (const candidate of candidates) {
    if (candidate.includes("/") || candidate.includes("\\")) {
      if (existsSync(candidate)) return candidate;
    } else {
      const resolved = resolveCommandPath(candidate);
      if (resolved) return resolved;
    }
  }

  throw new Error("Unable to find Chrome/Chromium. Set CHROME_PATH to run this test.");
}

async function expectDarkControl(page, selector, label) {
  const colors = await page.locator(selector).evaluate((element) => {
    const styles = window.getComputedStyle(element);
    const rootStyles = window.getComputedStyle(document.documentElement);
    return {
      backgroundColor: styles.backgroundColor,
      colorScheme: rootStyles.colorScheme,
    };
  });
  if (colors.colorScheme !== "dark") {
    throw new Error(`${label}: expected dark color-scheme, got ${colors.colorScheme}`);
  }
  if (colors.backgroundColor === "rgb(255, 255, 255)") {
    throw new Error(`${label}: expected non-white background in dark theme`);
  }
}

async function run() {
  const devServer = await ensureDevServer();
  const browser = await chromium.launch({
    executablePath: chromePath,
    headless: true,
  });

  try {
    const page = await browser.newPage();
    await page.addInitScript((modelFixtures) => {
      const initialServer = {
        modelPath: modelFixtures[0].path,
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
        kvu: false,
        enableGpuMemoryOptions: true,
        fit: "",
        fitTarget: "",
        fitCtx: 0,
        device: "",
        tensorSplit: "",
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
        specDraftTypeK: "",
        specDraftTypeV: "",
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
        reasoningPreserve: "flag",
        reasoningFormat: "",
        reasoningBudget: "",
        chatTemplateKwargs: "",
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
      window.isTauri = true;
      const originalFetch = window.fetch.bind(window);
      let prefillCallIndex = 0;
      window.__benchmarkRequests = [];
      window.fetch = async (input, init) => {
        const url = input instanceof Request ? input.url : String(input);
        if (url.endsWith("/v1/models")) {
          return new Response(JSON.stringify({ data: [{ id: "bench-model" }] }), {
            headers: { "content-type": "application/json" },
          });
        }
        if (url.endsWith("/completion")) {
          const body = JSON.parse(init?.body || "{}");
          window.__benchmarkRequests.push(body);
          const isPrefill = String(body.prompt || "").includes("PREFILL_BENCHMARK_BLOCK");
          const prefillTargets = [2048, 4098, 8192];
          const prefillScores = [300, 240, 180];
          if (isPrefill) {
            const index = Math.min(prefillCallIndex, prefillTargets.length - 1);
            prefillCallIndex += 1;
            const promptTokens = prefillTargets[index];
            const promptScore = prefillScores[index];
            return new Response(
              JSON.stringify({
                content: "prefill checksum",
                timings: {
                  prompt_n: promptTokens,
                  prompt_ms: (promptTokens / promptScore) * 1000,
                  prompt_per_second: promptScore,
                  predicted_n: 1,
                  predicted_ms: 25,
                  predicted_per_second: 40,
                },
              }),
              { headers: { "content-type": "application/json" } },
            );
          }
          return new Response(
            JSON.stringify({
              content: "Benchmark response from the loaded llama-server model.",
              timings: {
                prompt_n: 96,
                prompt_ms: 300,
                prompt_per_second: 320,
                predicted_n: 64,
                predicted_ms: 1600,
                predicted_per_second: 40,
              },
            }),
            { headers: { "content-type": "application/json" } },
          );
        }
        return originalFetch(input, init);
      };
      window.__TAURI_INTERNALS__ = {
        transformCallback: () => Math.floor(Math.random() * 1_000_000),
        unregisterCallback: () => {},
        convertFileSrc: (path) => path,
        invoke: async (cmd, args) => {
          switch (cmd) {
            case "load_config":
              return {
                llamaServerPath: "llama-server",
                llamaCliPath: "llama-cli",
                llamaBenchPath: "llama-bench",
                modelDir: "C:/models",
                hfToken: "",
                manualModels: [],
                modelProfiles: [
                  {
                    modelPath: modelFixtures[2].path,
                    name: "Charlie profile",
                    server: { ...initialServer, modelPath: modelFixtures[2].path, threads: 6 },
                  },
                ],
                server: initialServer,
              };
            case "discover_tools":
              return {
                llamaServer: "llama-server",
                llamaCli: "llama-cli",
                llamaBench: "llama-bench",
                hfCli: "hf",
                huggingfaceCli: null,
              };
            case "save_config":
              return args.config;
            case "load_model_cache":
              return modelFixtures;
            case "scan_models":
              await new Promise((resolve) => setTimeout(resolve, 450));
              return modelFixtures;
            case "server_status":
              return {
                running: true,
                pid: 4242,
                command: "",
                url: "http://127.0.0.1:8080",
                modelPath: modelFixtures[0].path,
                logPath: "",
                startedAt: 1,
              };
            case "preview_server_command":
              return [
                "llama-server",
                "--model",
                args.config.modelPath,
                args.config.threads > 0 ? `--threads ${args.config.threads}` : "",
                args.config.ctxSize > 0 ? `--ctx-size ${args.config.ctxSize}` : "",
                args.config.batchSize > 0 ? `-b ${args.config.batchSize}` : "",
                args.config.enableKvCacheOptions && args.config.cacheTypeK
                  ? `-ctk ${args.config.cacheTypeK}`
                  : "",
                args.config.enableKvCacheOptions && args.config.cacheTypeV
                  ? `-ctv ${args.config.cacheTypeV}`
                  : "",
                args.config.enableKvCacheOptions && args.config.kvu ? "-kvu" : "",
                args.config.enableGpuMemoryOptions && args.config.device
                  ? `--device ${args.config.device}`
                  : "",
                args.config.enableGpuMemoryOptions && args.config.tensorSplit
                  ? `--tensor-split ${args.config.tensorSplit}`
                  : "",
                args.config.enableSamplingOptions && args.config.temperature
                  ? `--temp ${args.config.temperature}`
                  : "",
                args.config.enableSamplingOptions && args.config.topK
                  ? `--top-k ${args.config.topK}`
                  : "",
                args.config.enableSamplingOptions && args.config.topP
                  ? `--top-p ${args.config.topP}`
                  : "",
                args.config.enableSamplingOptions && args.config.minP
                  ? `--min-p ${args.config.minP}`
                  : "",
                args.config.enableSamplingOptions && args.config.typicalP
                  ? `--typical-p ${args.config.typicalP}`
                  : "",
                args.config.enableSamplingOptions && args.config.repeatPenalty
                  ? `--repeat-penalty ${args.config.repeatPenalty}`
                  : "",
                args.config.enableSamplingOptions && args.config.presencePenalty
                  ? `--presence-penalty ${args.config.presencePenalty}`
                  : "",
                args.config.enableSamplingOptions && args.config.frequencyPenalty
                  ? `--frequency-penalty ${args.config.frequencyPenalty}`
                  : "",
                args.config.enableSpeculativeOptions && args.config.specDraftTypeK
                  ? `--spec-draft-type-k ${args.config.specDraftTypeK}`
                  : "",
                args.config.enableSpeculativeOptions && args.config.specDraftTypeV
                  ? `--spec-draft-type-v ${args.config.specDraftTypeV}`
                  : "",
                args.config.toolsAll ? "--tools all" : "",
                args.config.jinja ? "--jinja" : "",
                args.config.embeddings ? "--embedding" : "",
                args.config.enableReasoningOptions && args.config.reasoningFormat
                  ? `--reasoning-format ${args.config.reasoningFormat}`
                  : "",
                args.config.enableReasoningOptions && args.config.reasoningBudget
                  ? `--reasoning-budget ${args.config.reasoningBudget}`
                  : "",
                args.config.enableReasoningOptions && args.config.reasoningPreserve === "flag"
                  ? "--reasoning-preserve"
                  : (args.config.enableReasoningOptions && args.config.reasoningPreserve === "chat-template"
                      ? `--chat-template-kwargs ${args.config.chatTemplateKwargs || '{"preserve_thinking": true}'}`
                      : (args.config.enableReasoningOptions && args.config.reasoningPreserve !== "none" && args.config.chatTemplateKwargs
                          ? `--chat-template-kwargs ${args.config.chatTemplateKwargs}`
                          : (args.config.enableReasoningOptions && args.config.reasoningPreserve !== "none" && args.config.preserveThinking
                              ? "--reasoning-preserve"
                              : ""))),
                args.config.enableReasoningOptions && args.config.reasoning
                  ? `-rea ${args.config.reasoning}`
                  : "",
              ]
                .filter(Boolean)
                .join(" ");
            case "download_model":
              return {
                success: true,
                statusCode: 0,
                command: [
                  "hf",
                  "download",
                  args.request.repoId,
                  "--include",
                  args.request.pattern,
                  "--local-dir",
                  args.request.targetDir,
                  args.request.token ? "--token <token>" : "",
                ]
                  .filter(Boolean)
                  .join(" "),
                stdout: "download complete",
                stderr: "",
              };
            case "list_hf_models":
              return [
                {
                  id: "public/repo",
                  downloads: 1250000,
                  likes: 321,
                  lastModified: "2026-06-05T12:00:00Z",
                  createdAt: "2026-01-01T12:00:00Z",
                  pipelineTag: "text-generation",
                  libraryName: "gguf",
                  trendingScore: 99,
                },
                {
                  id: "other/repo",
                  downloads: 4000,
                  likes: 12,
                  lastModified: "2026-06-04T12:00:00Z",
                  createdAt: "2026-01-01T12:00:00Z",
                  pipelineTag: "text-generation",
                  libraryName: "transformers",
                  trendingScore: 4,
                },
              ];
            case "list_hf_repo_files":
              return [
                { path: "README.md", sizeBytes: 512 },
                { path: "model.Q4_K_M.gguf", sizeBytes: 4_294_967_296 },
                { path: "model.Q5_K_M.gguf", sizeBytes: 5_368_709_120 },
              ];
            case "load_model_metadata":
              return null;
            default:
              throw new Error(`Unexpected invoke: ${cmd}`);
          }
        },
      };
    }, models);

    await page.goto(appUrl);
    await page.locator("#model-select option").nth(2).waitFor({ state: "attached" });

    await page.locator("#theme-select").selectOption("graphite");
    await expectDarkControl(page, "#host", "dark theme text input");
    await expectDarkControl(page, "#model-select", "dark theme select");
    await page.locator("#reset-app-data").click();
    await page.locator(".app-confirm").waitFor({ state: "visible" });
    await expectTextIncludes(page, ".app-confirm", "Reset LocalLLM", "reset confirmation modal");
    await expectDarkControl(page, ".app-confirm", "dark theme confirmation modal");
    await page.locator(".app-confirm button", { hasText: "Cancel" }).click();
    await page.locator(".app-confirm").waitFor({ state: "detached" });

    await expectValue(page, "#model-select", models[0].path, "initial selected model");
    await expectValue(page, "#profile-select", "profile:1", "initial model uses auto-created exclusive profile");
    await page.locator("#profile-select").selectOption("profile:0");
    await delay(250);
    await expectValue(page, "#model-select", models[0].path, "profile selector keeps current model");
    await expectValue(page, "#threads", "6", "profile selector hydrates chosen profile fields");
    await expectTextIncludes(page, "#command-preview", models[0].path, "profile command keeps current model");
    await expectTextIncludes(page, "#command-preview", "--threads 6", "command after profile selector");
    const namedProfileLabel = await page.locator('#profile-select option[value="profile:0"]').textContent();
    if (namedProfileLabel !== "Charlie profile") {
      throw new Error(`named profile label should not include model name: ${namedProfileLabel}`);
    }

    await page.locator("#model-select").selectOption(models[1].path);
    await delay(250);
    await expectValue(page, "#model-select", models[1].path, "dropdown selection before scan settles");
    await expectValue(page, "#profile-select", "profile:2", "model change resets to exclusive model profile");
    await expectValue(page, "#threads", "0", "exclusive profile uses default threads");
    await expectValue(page, "#batch-size", "2048", "exclusive profile uses default batch");
    await expectValue(page, "#ubatch-size", "512", "exclusive profile uses default ubatch");
    await expectTextIncludes(page, "#command-preview", models[1].path, "command after dropdown selection");
    const bravoCommand = await page.locator("#command-preview").textContent();
    if (bravoCommand?.includes("--threads 6")) {
      throw new Error(`model-exclusive profile should replace the manually selected profile: ${bravoCommand}`);
    }

    await delay(700);
    await expectValue(page, "#model-select", models[1].path, "dropdown selection after background scan");
    await expectValue(page, "#profile-select", "profile:2", "background scan keeps exclusive model profile");
    await expectTextIncludes(page, "#command-preview", models[1].path, "command after background scan");

    await page.locator(".model-row", { hasText: models[2].name }).click();
    await delay(250);
    await expectValue(page, "#model-select", models[2].path, "row click selection");
    await expectValue(page, "#profile-select", "profile:0", "model defaults to matching saved profile");
    await expectValue(page, "#threads", "6", "matching model profile auto-loads");
    await expectTextIncludes(page, "#command-preview", models[2].path, "command after row click");
    await expectTextIncludes(page, "#command-preview", "--threads 6", "command uses matching model profile");

    await page.locator("#load-default-profile").click();
    await delay(250);
    await expectValue(page, "#profile-select", "default", "defaults button keeps default selected");
    await expectValue(page, "#threads", "0", "defaults button resets threads");
    await expectValue(page, "#batch-size", "2048", "defaults button resets batch");
    await expectValue(page, "#ubatch-size", "512", "defaults button resets ubatch");
    const defaultCommand = await page.locator("#command-preview").textContent();
    if (defaultCommand?.includes("--threads 6")) {
      throw new Error(`default profile should not keep saved profile threads: ${defaultCommand}`);
    }

    await page.locator("#new-model-profile").click();
    await expectTextIncludes(page, "#profile-status", "New profile", "new profile mode");
    await page.locator("#profile-name").fill("Charlie second profile");
    await page.locator("#save-model-profile").click();
    await page.locator('#profile-select option[value="profile:3"]').waitFor({ state: "attached" });
    await expectValue(page, "#profile-select", "profile:3", "new profile remains selected after save");

    await page.locator("#apply-advanced-preset").click();
    await delay(250);
    await expectValue(page, "#threads", "0", "clean preset thread auto");
    await expectValue(page, "#ctx-size", "4096", "clean preset default context");
    await expectValue(page, "#cache-type-k", "q8_0", "clean preset cache K");
    await expectValue(page, "#cache-type-v", "q8_0", "clean preset cache V");
    await expectValue(page, "#terminal-mode", "visible", "clean preset terminal default");
    await expectValue(page, "#batch-size", "0", "clean preset batch auto");
    await expectValue(page, "#no-cpu-moe", "0", "preset ncmoe default");
    await expectValue(page, "#spec-draft-type-k", "", "clean preset spec draft type K");
    await expectValue(page, "#spec-draft-type-v", "", "clean preset spec draft type V");
    if (!(await page.locator("#no-cpu-moe").isDisabled())) {
      throw new Error("ncmoe should be disabled by default in clean preset");
    }
    const cleanCommand = await page.locator("#command-preview").textContent();
    for (const flag of ["--threads", "-b ", "--tools all"]) {
      if (cleanCommand?.includes(flag)) {
        throw new Error(`clean preset should not include ${flag}: ${cleanCommand}`);
      }
    }
    await expectTextIncludes(page, "#command-preview", "--ctx-size 4096", "default context flag");
    await expectTextIncludes(page, "#command-preview", "-ctk q8_0", "default cache K flag");
    await expectTextIncludes(page, "#command-preview", "-ctv q8_0", "default cache V flag");
    await expectTextIncludes(
      page,
      "#command-preview",
      "--reasoning-preserve",
      "preserve-thinking default flag",
    );

    await page.locator("#enable-gpu-memory-options").check();
    await page.locator("#device").fill("CUDA0,Vulkan0");
    await page.locator("#tensor-split").fill("3,1");
    await delay(250);
    await expectTextIncludes(page, "#command-preview", "--device CUDA0,Vulkan0", "device flag");
    await expectTextIncludes(page, "#command-preview", "--tensor-split 3,1", "tensor-split flag");

    await page.locator("#enable-sampling-options").check();
    await page.locator("#temperature").fill("0.7");
    await page.locator("#top-k").fill("40");
    await page.locator("#top-p").fill("0.95");
    await page.locator("#min-p").fill("0.05");
    await page.locator("#typical-p").fill("1");
    await page.locator("#repeat-penalty").fill("1.1");
    await page.locator("#presence-penalty").fill("0.2");
    await page.locator("#frequency-penalty").fill("0.1");
    await delay(250);
    await expectTextIncludes(page, "#command-preview", "--temp 0.7", "temperature flag");
    await expectTextIncludes(page, "#command-preview", "--top-k 40", "top-k flag");
    await expectTextIncludes(page, "#command-preview", "--top-p 0.95", "top-p flag");
    await expectTextIncludes(page, "#command-preview", "--min-p 0.05", "min-p flag");
    await expectTextIncludes(page, "#command-preview", "--typical-p 1", "typical-p flag");
    await expectTextIncludes(page, "#command-preview", "--repeat-penalty 1.1", "repeat-penalty flag");
    await expectTextIncludes(page, "#command-preview", "--presence-penalty 0.2", "presence-penalty flag");
    await expectTextIncludes(page, "#command-preview", "--frequency-penalty 0.1", "frequency-penalty flag");

    await page.locator("#tools-all").check();
    await delay(250);
    await expectTextIncludes(page, "#command-preview", "--tools all", "tools-all flag");

    await page.locator("#enable-speculative-options").check();
    await page.locator("#spec-draft-type-k").selectOption("q8_0");
    await page.locator("#spec-draft-type-v").selectOption("bf16");
    await delay(250);
    await expectTextIncludes(page, "#command-preview", "--spec-draft-type-k q8_0", "spec draft type k flag");
    await expectTextIncludes(page, "#command-preview", "--spec-draft-type-v bf16", "spec draft type v flag");

    await page.locator("#reasoning").selectOption("on");
    await page.locator("#reasoning-format").selectOption("deepseek");
    await page.locator("#reasoning-budget").fill("2048");
    await page.locator("#reasoning-preserve").selectOption("chat-template");
    await page.locator("#chat-template-kwargs").fill('{"preserve_thinking": true, "custom_key": 42}');
    await delay(250);
    await expectTextIncludes(page, "#command-preview", "-rea on", "reasoning flag");
    await expectTextIncludes(page, "#command-preview", "--reasoning-format deepseek", "reasoning format flag");
    await expectTextIncludes(page, "#command-preview", "--reasoning-budget 2048", "reasoning budget flag");
    await expectTextIncludes(page, "#command-preview", 'custom_key', "custom chat template kwargs in command");

    await page.locator("#reasoning-preserve").selectOption("flag");
    await delay(250);
    await expectTextIncludes(page, "#command-preview", "--reasoning-preserve", "reasoning preserve flag");
    const flagCommand = await page.locator("#command-preview").textContent();
    if (flagCommand?.includes("--chat-template-kwargs")) {
      throw new Error(`flag selection should replace chat template kwargs: ${flagCommand}`);
    }

    await page.locator("#enable-reasoning-options").uncheck();
    await delay(250);
    const commandWithoutPreserve = await page.locator("#command-preview").textContent();
    if (commandWithoutPreserve?.includes("preserve_thinking") || commandWithoutPreserve?.includes("--reasoning-preserve")) {
      throw new Error(`preserve-thinking checkbox did not remove preserve flag: ${commandWithoutPreserve}`);
    }

    await page.locator("#app-tab-benchmark").click();
    await expectTextIncludes(page, "#app-view-benchmark", "Prefill", "benchmark prefill metric");
    await expectTextIncludes(page, "#app-view-benchmark", "Generate", "benchmark generation metric");
    if (await page.locator("#benchmark-visual-grid").getAttribute("hidden") !== null) {
      throw new Error("benchmark-visual-grid should be visible in performance mode");
    }
    if (await page.locator("#benchmark-share-panel").getAttribute("hidden") !== null) {
      throw new Error("benchmark-share-panel should be visible in performance mode");
    }
    if (await page.locator("#benchmark-transcript").getAttribute("hidden") !== null) {
      throw new Error("benchmark-transcript should be visible in performance mode");
    }
    if (await page.locator("#eval-samples-container").getAttribute("hidden") === null) {
      throw new Error("eval-samples-container should be hidden in performance mode");
    }
    const sampleBelowLatency = await page.evaluate(() => {
      const visualGrid = document.querySelector("#benchmark-visual-grid");
      const evalSamples = document.querySelector("#eval-samples-container");
      return Boolean(visualGrid && evalSamples && (visualGrid.compareDocumentPosition(evalSamples) & Node.DOCUMENT_POSITION_FOLLOWING));
    });
    if (!sampleBelowLatency) {
      throw new Error("eval-samples-container should be positioned below benchmark-visual-grid in DOM order");
    }
    await page.locator("#benchmark-run-count").fill("1");
    await page.locator("#benchmark-generate-tokens").fill("64");
    await page.locator("#run-benchmark").click();
    await waitTextIncludes(page, "#benchmark-prefill-2048-tokens-per-second", "300", "2k prefill tokens per second");
    await waitTextIncludes(page, "#benchmark-prefill-4098-tokens-per-second", "240", "4k prefill tokens per second");
    await waitTextIncludes(page, "#benchmark-prefill-8192-tokens-per-second", "180", "8k prefill tokens per second");
    await waitTextIncludes(page, "#benchmark-generate-tokens-per-second", "40", "generation tokens per second");
    await waitTextIncludes(page, "#benchmark-output", "prefill 2048: 300", "benchmark log 2k prefill score");
    await waitTextIncludes(page, "#benchmark-output", "prefill 4098: 240", "benchmark log 4k prefill score");
    await waitTextIncludes(page, "#benchmark-output", "prefill 8192: 180", "benchmark log 8k prefill score");
    await waitTextIncludes(page, "#benchmark-output", "generate 40", "benchmark log generation score");
    await waitTextIncludes(page, "#benchmark-chart", "300", "benchmark chart 2k score");
    await waitValueIncludes(page, "#benchmark-share-summary", "2K 300", "share summary 2k score");
    await waitValueIncludes(page, "#benchmark-share-summary", "4K 240", "share summary 4k score");
    await waitValueIncludes(page, "#benchmark-share-summary", "8K 180", "share summary 8k score");
    const benchmarkRequests = await page.evaluate(() => window.__benchmarkRequests);
    if (benchmarkRequests.length !== 4) {
      throw new Error(`benchmark should perform 3 prefill calls + 1 generation call, got ${benchmarkRequests.length}`);
    }
    if (!benchmarkRequests.slice(0, 3).every((request) => request.n_predict === 1 && request.cache_prompt === false)) {
      throw new Error(`prefill benchmark requests should use n_predict=1 and disable prompt cache: ${JSON.stringify(benchmarkRequests)}`);
    }
    if (benchmarkRequests[3].n_predict !== 64) {
      throw new Error(`generation benchmark should use requested n_predict=64: ${JSON.stringify(benchmarkRequests[3])}`);
    }
    await page.locator("#benchmark-mode").selectOption("evaluation");
    if (await page.locator("#eval-samples-container").getAttribute("hidden") !== null) {
      throw new Error("eval-samples-container should be visible in evaluation mode");
    }
    if (await page.locator("#benchmark-visual-grid").getAttribute("hidden") === null) {
      throw new Error("benchmark-visual-grid should be hidden in evaluation mode");
    }
    await page.locator("#benchmark-mode").selectOption("performance");
    await page.locator("#app-tab-control").click();

    await page.locator("#hf-model-search").fill("qwen gguf");
    await page.locator("#hf-search-models").click();
    await page.locator(".hf-model-row", { hasText: "public/repo" }).click();
    await expectValue(page, "#hf-repo", "public/repo", "selected Hugging Face repo from search");
    await page.locator('#hf-file-select option[value="model.Q4_K_M.gguf"]').waitFor({ state: "attached" });
    await page.locator("#hf-file-select").selectOption("model.Q4_K_M.gguf");
    await expectValue(page, "#hf-pattern", "model.Q4_K_M.gguf", "selected Hugging Face file");
    await page.locator("#hf-token").fill("");
    await page.locator("#download-model").click();
    await expectTextIncludes(page, "#download-output", "download complete", "download log output");
    const downloadLog = await page.locator("#download-output").textContent();
    if (!downloadLog?.includes("model.Q4_K_M.gguf")) {
      throw new Error(`download should include selected repo file: ${downloadLog}`);
    }
    if (downloadLog?.includes("--token")) {
      throw new Error(`blank token download should not include --token: ${downloadLog}`);
    }
  } finally {
    await browser.close();
    if (devServer?.pid) {
      spawnSync("taskkill", ["/PID", String(devServer.pid), "/T", "/F"], { stdio: "ignore" });
    }
  }
}

run()
  .then(() => {
    process.exit(0);
  })
  .catch((error) => {
    console.error(error);
    process.exit(1);
  });
