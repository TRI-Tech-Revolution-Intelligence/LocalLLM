import assert from "node:assert/strict";
import { mkdtemp, readFile, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { pathToFileURL } from "node:url";
import ts from "typescript";

const root = new URL("../", import.meta.url);
const temporary = await mkdtemp(join(tmpdir(), "localllm-profile-tests-"));

try {
  const transpile = async (sourceName, targetName, transform = (value) => value) => {
    const source = await readFile(new URL(sourceName, root), "utf8");
    const output = ts.transpileModule(source, {
      compilerOptions: {
        target: ts.ScriptTarget.ES2022,
        module: ts.ModuleKind.ES2022,
      },
    }).outputText;
    await writeFile(join(temporary, targetName), transform(output), "utf8");
  };

  await transpile("src/constants.ts", "constants.mjs");
  await transpile(
    "src/agent-profiles.ts",
    "agent-profiles.mjs",
    (output) => output.replaceAll('"./constants"', '"./constants.mjs"'),
  );

  const {
    createAgentProfile,
    duplicateAgentProfile,
    exportAgentProfiles,
    loadAgentProfileStore,
    normalizeAgentProfileStore,
    validateAgentProfile,
  } = await import(`${pathToFileURL(join(temporary, "agent-profiles.mjs")).href}?test=${Date.now()}`);

  const migrated = loadAgentProfileStore(null, null, {
    role: "coder",
    goal: "Ship safely",
    thinkingLevel: "high",
    workspaceRoot: "G:\\work",
  });
  assert.equal(migrated.migrated, true);
  assert.equal(migrated.store.profiles[0].role, "coder");
  assert.equal(migrated.store.profiles[0].goal, "Ship safely");
  assert.equal(migrated.store.profiles[0].schemaVersion, 3);

  const defaults = createAgentProfile("Fresh defaults");
  assert.equal(defaults.role, "coder");
  assert.equal(defaults.thinkingLevel, "medium");
  assert.equal(defaults.temperature, 0.6);
  assert.equal(defaults.maxSteps, 32);
  assert.equal(createAgentProfile("Auto approve", { autoApprove: true }).yoloMode, true);

  const backupProfile = createAgentProfile("Recovered", { role: "reviewer" }, 100);
  const backupStore = {
    schemaVersion: 3,
    activeProfileId: backupProfile.id,
    defaultProfileId: backupProfile.id,
    profiles: [backupProfile],
  };
  const recovered = loadAgentProfileStore("{broken", JSON.stringify(backupStore));
  assert.equal(recovered.recovered, true);
  assert.equal(recovered.store.profiles[0].name, "Recovered");

  const oldFormat = normalizeAgentProfileStore({
    schemaVersion: 1,
    profile: {
      name: "Legacy import",
      mode: "debugger",
      instructions: "Investigate first",
      workingDirectory: "C:\\project",
      timeoutSeconds: 99999,
    },
  });
  assert.equal(oldFormat.profiles[0].role, "debugger");
  assert.equal(oldFormat.profiles[0].systemInstructions, "Investigate first");
  assert.equal(oldFormat.profiles[0].workspaceRoot, "C:\\project");
  assert.equal(oldFormat.profiles[0].timeoutSeconds, 3600);

  const upgradedDefaults = normalizeAgentProfileStore({
    schemaVersion: 2,
    activeProfileId: "legacy-default",
    defaultProfileId: "legacy-default",
    profiles: [{
      id: "legacy-default",
      schemaVersion: 2,
      name: "Default",
      role: "architect",
      thinkingLevel: "disabled",
      temperature: 0.7,
      maxSteps: 16,
    }],
  });
  assert.equal(upgradedDefaults.profiles[0].role, "coder");
  assert.equal(upgradedDefaults.profiles[0].thinkingLevel, "medium");
  assert.equal(upgradedDefaults.profiles[0].temperature, 0.6);
  assert.equal(upgradedDefaults.profiles[0].maxSteps, 32);

  const duplicate = duplicateAgentProfile(
    oldFormat.profiles[0],
    ["Legacy import", "Legacy import copy"],
    200,
  );
  assert.equal(duplicate.name, "Legacy import copy 2");
  assert.notEqual(duplicate.id, oldFormat.profiles[0].id);
  assert.deepEqual(validateAgentProfile(duplicate), []);

  const exported = JSON.parse(exportAgentProfiles(oldFormat, oldFormat.profiles[0].id));
  assert.equal(exported.schemaVersion, 3);
  assert.equal(exported.profiles.length, 1);

  console.log("Agent profile migration, validation, backup recovery, duplication, and export tests passed.");
} finally {
  await rm(temporary, { recursive: true, force: true });
}
