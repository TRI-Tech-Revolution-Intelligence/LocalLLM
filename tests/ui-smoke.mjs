import assert from "node:assert/strict";
import { existsSync } from "node:fs";
import { chromium } from "playwright-core";
import { createServer } from "vite";

const browserExecutable = [
  process.env.PLAYWRIGHT_CHROMIUM_EXECUTABLE_PATH,
  "C:\\Program Files (x86)\\Microsoft\\Edge\\Application\\msedge.exe",
  "C:\\Program Files\\Microsoft\\Edge\\Application\\msedge.exe",
  "C:\\Program Files\\Google\\Chrome\\Application\\chrome.exe",
  "/usr/bin/google-chrome",
  "/usr/bin/chromium",
  "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
].find((candidate) => candidate && existsSync(candidate));
if (!browserExecutable) {
  throw new Error("Set PLAYWRIGHT_CHROMIUM_EXECUTABLE_PATH to run the UI smoke test.");
}

const appServer = await createServer({
  server: { host: "127.0.0.1", port: 0 },
});
await appServer.listen();
const appUrl = appServer.resolvedUrls?.local[0];
if (!appUrl) throw new Error("The Vite test server did not provide a local URL.");
const browser = await chromium.launch({
  executablePath: browserExecutable,
  headless: true,
});
const page = await browser.newPage({ viewport: { width: 1440, height: 960 }, deviceScaleFactor: 1 });
const consoleErrors = [];
page.on("console", (message) => {
  if (message.type() === "error") consoleErrors.push(message.text());
});
page.on("pageerror", (error) => consoleErrors.push(error.message));

await page.addInitScript(() => {
  const server = {
    modelPath: "", host: "127.0.0.1", port: 8080, ctxSize: 4096, gpuLayers: "", threads: 0,
    batchSize: 2048, ubatchSize: 512, parallel: -1, enableKvCacheOptions: true, cacheTypeK: "q8_0",
    cacheTypeV: "q8_0", flashAttention: "", enableGpuMemoryOptions: false, kvOffload: "", noHost: false,
    opOffload: "", fit: "", fitTarget: "", fitCtx: 0, device: "", tensorSplit: "", splitMode: "",
    mainGpu: "", cpuMoe: false, enableSamplingOptions: false, temperature: "", topK: "", topP: "",
    minP: "", typicalP: "", repeatPenalty: "", presencePenalty: "", frequencyPenalty: "",
    enableSpeculativeOptions: true, specType: "", specDraftNMax: 0, specDraftNMin: 0,
    specDraftPMin: "", specDraftPSplit: "", noMmap: false, mlock: false, specNgramModNMatch: 0,
    specNgramModNMin: 0, specNgramModNMax: 0, noCpuMoe: 0, enableReasoningOptions: true,
    preserveThinking: true, reasoningFormat: "", reasoningBudget: "",
    chatTemplateKwargs: '{"preserve_thinking": true}', reasoning: "", enableMultimodalOptions: true,
    mmproj: "", embeddings: false, toolsAll: false, jinja: false, verbose: false,
    terminalMode: "visible", extraArgs: "",
  };
  let config = {
    llamaServerPath: "", llamaCliPath: "", llamaBenchPath: "", agentWorkspaceRoot: "G:\\LocalLLM",
    modelDir: "G:\\Models", hfToken: "", manualModels: [], modelProfiles: [], server,
  };
  window.__TAURI_INTERNALS__ = {
    invoke: async (command, args = {}) => {
      if (command === "load_config") return structuredClone(config);
      if (command === "save_config") {
        config = structuredClone(args.config);
        return structuredClone(config);
      }
      if (command === "discover_tools") {
        return { llamaServer: null, llamaCli: null, llamaBench: null, hfCli: null, huggingfaceCli: null };
      }
      if (command === "server_status") {
        return {
          running: false, pid: null, command: "", url: "", modelPath: "", logPath: "", startedAt: null,
        };
      }
      if (command === "discover_pi_agent") {
        return { available: true, command: "pi", version: "0.82.0", checked: ["bundled"] };
      }
      if (command === "agent_workspace_root") return config.agentWorkspaceRoot;
      if (command === "agent_validate_workspace") return args.path;
      if (command === "preview_server_command") return "llama-server --host 127.0.0.1 --port 8080";
      if (command === "load_model_cache" || command === "scan_models" || command === "agent_list_skills") return [];
      return null;
    },
    transformCallback: () => 1,
    unregisterCallback: () => {},
  };
  const now = Date.now();
  window.localStorage.setItem("localllm-agent-task-history", JSON.stringify([
    {
      id: "history-refactor",
      title: "Refactor parser",
      startedAt: now - 120000,
      updatedAt: now - 60000,
      messages: [
        { role: "user", message: "Refactor the parser safely.", at: now - 120000 },
        { role: "agent", message: "The parser refactor is complete and verified.", at: now - 60000 },
      ],
    },
    {
      id: "history-tests",
      title: "Write tests",
      startedAt: now - 240000,
      updatedAt: now - 180000,
      messages: [{ role: "user", message: "Add regression coverage.", at: now - 180000 }],
    },
    {
      id: "history-docs",
      title: "Update documentation",
      startedAt: now - 360000,
      updatedAt: now - 300000,
      messages: [{ role: "agent", message: "Documentation updated.", at: now - 300000 }],
    },
  ]));
  window.localStorage.setItem("localllm-agent-todos", JSON.stringify([
    { id: "todo-release", text: "Ship release", done: false, createdAt: now },
  ]));
  window.localStorage.setItem("localllm-agent-skills", JSON.stringify([
    { id: "skill-polish", name: "UI polish", instructions: "Review visual consistency.", enabled: true, createdAt: now },
  ]));
  window.localStorage.setItem("localllm-agent-subagents", JSON.stringify([
    { id: "subagent-reviewer", name: "Reviewer", role: "Review the final changes.", enabled: true, createdAt: now },
  ]));
  window.localStorage.setItem("localllm-agent-mcp-servers", JSON.stringify([
    { id: "mcp-local", name: "127.0.0.1:3000", url: "http://127.0.0.1:3000", addedAt: now },
  ]));
});

try {
  await page.goto(appUrl, { waitUntil: "networkidle" });
  await page.getByRole("tab", { name: "Agent" }).click();

  await page.locator("#agent-history").click();
  assert.equal(await page.locator(".agent-history-item").count(), 3);
  await page.locator(".agent-history-dialog").waitFor({ state: "visible" });
  await page.waitForTimeout(250);
  const historyDialogRect = await page.locator(".agent-history-dialog").boundingBox();
  assert.ok(historyDialogRect && historyDialogRect.width >= 420, "Agent history drawer is not visibly rendered");
  await page.screenshot({ path: "docs/assets/agent-history-management.png", fullPage: false });
  await page.getByRole("button", { name: "Open Refactor parser" }).click();
  assert.equal(await page.locator(".agent-message").count(), 2);
  assert.equal(await page.locator("#agent-history-overlay").isHidden(), true);

  await page.locator("#agent-clear-current").click();
  await page.getByRole("button", { name: "Clear conversation" }).click();
  assert.equal(await page.locator(".agent-message").count(), 0);

  await page.locator("#agent-history").click();
  assert.equal(await page.locator(".agent-history-item").count(), 2);
  await page.getByRole("button", { name: "Delete Write tests" }).click();
  await page.getByRole("button", { name: "Delete session" }).click();
  assert.equal(await page.locator(".agent-history-item").count(), 1);
  await page.locator("#agent-history-delete-all").click();
  await page.locator(".app-confirm").getByRole("button", { name: "Delete all history" }).click();
  assert.equal(await page.locator(".agent-history-item").count(), 0);
  await page.getByText("Your history is clear", { exact: true }).waitFor();
  assert.deepEqual(
    await page.evaluate(() => JSON.parse(window.localStorage.getItem("localllm-agent-task-history") ?? "null")),
    [],
  );
  await page.locator("#agent-history-close").click();

  await page.locator("#agent-toggle-left").click();
  await page.locator("#agent-profile-name").waitFor({ state: "visible" });

  await page.getByRole("button", { name: "Delete task Ship release" }).click();
  await page.locator(".app-confirm").getByRole("button", { name: "Delete" }).click();
  await page.getByText("No tasks yet", { exact: true }).waitFor();
  await page.getByRole("button", { name: "Delete skill UI polish" }).click();
  await page.locator(".app-confirm").getByRole("button", { name: "Delete" }).click();
  await page.getByText("No skills loaded", { exact: true }).waitFor();
  await page.getByRole("button", { name: "Delete subagent Reviewer" }).click();
  await page.locator(".app-confirm").getByRole("button", { name: "Delete" }).click();
  await page.getByText("No subagents configured", { exact: true }).waitFor();

  assert.equal(await page.locator("#agent-saved-profile-select option").count(), 1);
  await page.locator("#agent-system-instructions").fill("Be concise and verify every change.");
  assert.equal(await page.locator("#agent-profile-save-state").textContent(), "Unsaved changes");
  await page.locator("#agent-profile-save").click();
  await page.getByText("Default saved safely.", { exact: true }).waitFor();

  await page.locator("#agent-profile-duplicate").click();
  assert.equal(await page.locator("#agent-saved-profile-select option").count(), 2);
  await page.locator("#agent-profile-default").click();
  assert.match(await page.locator("#agent-saved-profile-select").textContent(), /default/);

  await page.locator("#agent-settings").click();
  assert.equal(await page.locator("#agent-settings").getAttribute("aria-pressed"), "true");
  assert.equal(await page.locator(".agent-workbench > .agent-file-panel").isVisible(), false);
  assert.equal(await page.locator(".agent-shell-panel").isVisible(), false);
  assert.equal(await page.locator(".agent-mcp-panel").isVisible(), true);
  assert.equal(await page.locator(".agent-browser-panel").isVisible(), true);
  const settingsLayout = await page.evaluate(() => {
    const panel = document.querySelector("#app-view-agent .agent-panel");
    const browserPanel = document.querySelector("#app-view-agent .agent-browser-panel");
    const status = document.querySelector("#app-view-agent .agent-status-line");
    if (!panel || !browserPanel || !status) throw new Error("Settings layout elements are missing");
    const panelRect = panel.getBoundingClientRect();
    const browserRect = browserPanel.getBoundingClientRect();
    const statusRect = status.getBoundingClientRect();
    return {
      scrollHeight: panel.scrollHeight,
      clientHeight: panel.clientHeight,
      browserBottom: browserRect.bottom,
      statusTop: statusRect.top,
      panelBottom: panelRect.bottom,
    };
  });
  assert.ok(settingsLayout.scrollHeight <= settingsLayout.clientHeight + 1, "Settings created implicit grid overflow");
  assert.ok(settingsLayout.browserBottom <= settingsLayout.statusTop + 1, "Settings escaped above the status bar");
  assert.ok(settingsLayout.statusTop <= settingsLayout.panelBottom, "Status bar escaped the Agent panel");
  await page.screenshot({ path: "docs/assets/agent-settings-drawer.png", fullPage: false });
  await page.getByRole("button", { name: "Delete MCP server 127.0.0.1:3000" }).click();
  await page.locator(".app-confirm").getByRole("button", { name: "Delete" }).click();
  await page.getByText("No local MCP servers added", { exact: true }).waitFor();
  await page.locator("#agent-settings").click();
  assert.equal(await page.locator("#agent-settings").getAttribute("aria-pressed"), "false");

  await page.setViewportSize({ width: 1024, height: 800 });
  await page.locator("#agent-settings").click();
  const compactSettingsLayout = await page.evaluate(() => {
    const panel = document.querySelector("#app-view-agent .agent-panel");
    const browserPanel = document.querySelector("#app-view-agent .agent-browser-panel");
    const status = document.querySelector("#app-view-agent .agent-status-line");
    if (!panel || !browserPanel || !status) throw new Error("Compact settings layout elements are missing");
    return {
      scrollHeight: panel.scrollHeight,
      clientHeight: panel.clientHeight,
      browserBottom: browserPanel.getBoundingClientRect().bottom,
      statusTop: status.getBoundingClientRect().top,
    };
  });
  assert.ok(
    compactSettingsLayout.scrollHeight <= compactSettingsLayout.clientHeight + 1,
    "Compact Settings created implicit grid overflow",
  );
  assert.ok(
    compactSettingsLayout.browserBottom <= compactSettingsLayout.statusTop + 1,
    "Compact Settings escaped above the status bar",
  );
  assert.equal(await page.locator(".agent-mcp-panel").isVisible(), true);
  assert.equal(await page.locator(".agent-browser-panel").isVisible(), true);
  await page.locator("#agent-settings").click();
  await page.setViewportSize({ width: 1440, height: 960 });

  const liquidGlass = await page.evaluate(() => {
    const topbar = document.querySelector(".topbar");
    const tabs = document.querySelector(".app-tabs");
    const titlebar = document.querySelector(".window-titlebar");
    const panel = document.querySelector("#app-view-agent .agent-panel");
    const chat = document.querySelector("#app-view-agent .agent-chat-container");
    if (!topbar || !tabs || !titlebar || !panel || !chat) throw new Error("Liquid Glass surfaces are missing");
    const styles = [topbar, tabs, titlebar, panel, chat].map((element) => getComputedStyle(element));
    return {
      topbarRadius: Number.parseFloat(styles[0].borderTopLeftRadius),
      tabRadius: Number.parseFloat(styles[1].borderTopLeftRadius),
      titlebarRadius: Number.parseFloat(styles[2].borderTopLeftRadius),
      panelRadius: Number.parseFloat(styles[3].borderTopLeftRadius),
      panelGap: Number.parseFloat(styles[3].gap),
      chatRadius: Number.parseFloat(styles[4].borderTopLeftRadius),
      blurSurfaces: styles.filter((style) => style.backdropFilter !== "none").length,
      windowControls: titlebar.querySelectorAll(".window-control").length,
    };
  });
  assert.ok(liquidGlass.topbarRadius >= 20, "Top bar lost its floating glass shape");
  assert.ok(liquidGlass.tabRadius >= 20, "Navigation lost its capsule shape");
  assert.ok(liquidGlass.titlebarRadius >= 16, "Window titlebar lost its glass shape");
  assert.equal(liquidGlass.windowControls, 3, "Window titlebar controls are incomplete");
  assert.ok(liquidGlass.panelRadius >= 20, "Workbench lost its liquid glass shape");
  assert.ok(liquidGlass.panelGap >= 6, "Workbench panes collapsed into hard dividers");
  assert.ok(liquidGlass.chatRadius >= 16, "Chat surface lost its glass shape");
  assert.ok(liquidGlass.blurSurfaces >= 4, "Liquid Glass blur was not applied to the key surfaces");

  await page.screenshot({ path: "docs/assets/agent-crystal-dark.png", fullPage: false });
  await page.locator("#theme-select").selectOption("paper");
  await page.setViewportSize({ width: 760, height: 900 });
  await page.locator("#agent-profile-name").scrollIntoViewIfNeeded();
  await page.screenshot({ path: "docs/assets/agent-crystal-light-mobile.png", fullPage: false });

  const viewport = await page.evaluate(() => ({
    width: document.documentElement.scrollWidth,
    innerWidth: window.innerWidth,
    profileVisible: Boolean(document.querySelector("#agent-profile-name")?.getBoundingClientRect().height),
    agentColumns: getComputedStyle(document.querySelector("#app-view-agent .agent-panel")).gridTemplateColumns,
  }));
  assert.equal(viewport.profileVisible, true);
  assert.ok(viewport.width <= viewport.innerWidth + 1, `Unexpected page overflow: ${viewport.width}px`);
  assert.equal(viewport.agentColumns.trim().split(/\s+/).length, 1, "Compact Agent layout is not a single column");
  assert.deepEqual(consoleErrors, []);
  console.log("Agent profile UI, Liquid Glass surfaces, responsive layout, and browser console checks passed.");
} finally {
  await browser.close();
  await appServer.close();
}
