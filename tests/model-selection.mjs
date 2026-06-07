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
        enableGpuMemoryOptions: true,
        fit: "",
        fitTarget: "",
        fitCtx: 0,
        devices: "",
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
        chatTemplateKwargs: "{\"preserve_thinking\": true}",
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
                running: false,
                pid: null,
                command: "",
                url: "",
                modelPath: "",
                logPath: "",
                startedAt: null,
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
                args.config.enableGpuMemoryOptions && args.config.devices
                  ? `--devices ${args.config.devices}`
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
                args.config.toolsAll ? "--tools all" : "",
                args.config.jinja ? "--jinja" : "",
                args.config.embeddings ? "--embedding" : "",
                args.config.preserveThinking
                  ? `--chat-template-kwargs ${args.config.chatTemplateKwargs}`
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

    await page.locator("#model-select").selectOption(models[1].path);
    await delay(250);
    await expectValue(page, "#model-select", models[1].path, "dropdown selection before scan settles");
    await expectValue(page, "#profile-select", "profile:2", "model change resets to exclusive model profile");
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
      '{"preserve_thinking": true}',
      "preserve-thinking default",
    );

    await page.locator("#enable-gpu-memory-options").check();
    await page.locator("#devices").fill("CUDA0,Vulkan0");
    await page.locator("#tensor-split").fill("3,1");
    await delay(250);
    await expectTextIncludes(page, "#command-preview", "--devices CUDA0,Vulkan0", "devices flag");
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

    await page.locator("#preserve-thinking").uncheck();
    await delay(250);
    const commandWithoutPreserve = await page.locator("#command-preview").textContent();
    if (commandWithoutPreserve?.includes("preserve_thinking")) {
      throw new Error(`preserve-thinking checkbox did not remove default kwargs: ${commandWithoutPreserve}`);
    }

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

run().catch((error) => {
  console.error(error);
  process.exit(1);
});
