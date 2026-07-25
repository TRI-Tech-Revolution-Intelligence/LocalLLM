import { defaultAgentPermissions } from "./constants";
import type {
  AgentExecutionMode,
  AgentMode,
  AgentPermissions,
  AgentProfile,
  AgentProfileStore,
  AgentThinkingLevel,
} from "./types";

export const AGENT_PROFILE_SCHEMA_VERSION = 3 as const;

const agentModes = new Set<AgentMode>([
  "architect",
  "ask",
  "coder",
  "debugger",
  "reviewer",
  "tester",
  "orchestrator",
]);
const thinkingLevels = new Set<AgentThinkingLevel>(["disabled", "low", "medium", "high", "xhigh"]);
const executionModes = new Set<AgentExecutionMode>(["plan", "edit"]);
const permissionValues = new Set(["allow", "ask", "deny"]);

export interface LegacyAgentProfileSeed {
  role?: string | null;
  goal?: string | null;
  workspaceRoot?: string | null;
  thinkingLevel?: string | null;
  executionMode?: string | null;
  permissions?: Partial<AgentPermissions> | null;
  autoApprove?: boolean;
  yoloMode?: boolean;
  autoCompact?: boolean;
}

export interface AgentProfileLoadResult {
  store: AgentProfileStore;
  recovered: boolean;
  migrated: boolean;
  warning: string;
}

function stringValue(value: unknown, fallback = ""): string {
  return typeof value === "string" ? value : fallback;
}

function boundedInteger(value: unknown, fallback: number, min: number, max: number): number {
  const parsed = typeof value === "number" ? value : Number(value);
  return Number.isFinite(parsed) ? Math.min(max, Math.max(min, Math.round(parsed))) : fallback;
}

function boundedNumber(value: unknown, fallback: number, min: number, max: number): number {
  const parsed = typeof value === "number" ? value : Number(value);
  return Number.isFinite(parsed) ? Math.min(max, Math.max(min, parsed)) : fallback;
}

function normalizeRole(value: unknown): AgentMode {
  return typeof value === "string" && agentModes.has(value as AgentMode) ? (value as AgentMode) : "coder";
}

function normalizeThinking(value: unknown): AgentThinkingLevel {
  return typeof value === "string" && thinkingLevels.has(value as AgentThinkingLevel)
    ? (value as AgentThinkingLevel)
    : "medium";
}

function normalizeExecutionMode(value: unknown): AgentExecutionMode {
  return typeof value === "string" && executionModes.has(value as AgentExecutionMode)
    ? (value as AgentExecutionMode)
    : "edit";
}

function normalizePermissions(value: unknown): AgentPermissions {
  const candidate = value && typeof value === "object" ? (value as Partial<AgentPermissions>) : {};
  const normalized = { ...defaultAgentPermissions };
  for (const key of Object.keys(normalized) as Array<keyof AgentPermissions>) {
    if (permissionValues.has(String(candidate[key]))) normalized[key] = candidate[key]!;
  }
  return normalized;
}

export function createAgentProfile(
  name = "Default",
  overrides: Partial<AgentProfile> = {},
  now = Date.now(),
): AgentProfile {
  return {
    id: overrides.id?.trim() || crypto.randomUUID(),
    schemaVersion: AGENT_PROFILE_SCHEMA_VERSION,
    name: name.trim() || "Untitled Profile",
    role: normalizeRole(overrides.role),
    systemInstructions: stringValue(overrides.systemInstructions),
    taskInstructions: stringValue(overrides.taskInstructions),
    goal: stringValue(overrides.goal),
    workspaceRoot: stringValue(overrides.workspaceRoot),
    thinkingLevel: normalizeThinking(overrides.thinkingLevel),
    executionMode: normalizeExecutionMode(overrides.executionMode),
    permissions: normalizePermissions(overrides.permissions),
    autoApprove: Boolean(overrides.autoApprove),
    yoloMode: Boolean(overrides.yoloMode || overrides.autoApprove),
    autoCompact: Boolean(overrides.autoCompact),
    temperature: boundedNumber(overrides.temperature, 0.6, 0, 2),
    timeoutSeconds: boundedInteger(overrides.timeoutSeconds, 300, 10, 3600),
    retryCount: boundedInteger(overrides.retryCount, 1, 0, 5),
    maxSteps: boundedInteger(overrides.maxSteps, 32, 1, 64),
    createdAt: boundedInteger(overrides.createdAt, now, 0, Number.MAX_SAFE_INTEGER),
    updatedAt: boundedInteger(overrides.updatedAt, now, 0, Number.MAX_SAFE_INTEGER),
  };
}

function normalizeProfile(value: unknown, index: number, upgradeLegacyDefaults = false, now = Date.now()): AgentProfile {
  if (!value || typeof value !== "object") {
    throw new Error(`Profile ${index + 1} is not an object`);
  }
  const candidate = value as Partial<AgentProfile> & { mode?: string; instructions?: string; workingDirectory?: string };
  const isLegacyDefault =
    upgradeLegacyDefaults &&
    stringValue(candidate.name, "Default").trim().toLocaleLowerCase() === "default" &&
    normalizeRole(candidate.role ?? candidate.mode) === "architect" &&
    normalizeThinking(candidate.thinkingLevel) === "disabled" &&
    boundedNumber(candidate.temperature, 0.7, 0, 2) === 0.7 &&
    boundedInteger(candidate.maxSteps, 16, 1, 64) === 16;
  return createAgentProfile(stringValue(candidate.name, `Imported Profile ${index + 1}`), {
    ...candidate,
    id: stringValue(candidate.id) || crypto.randomUUID(),
    role: isLegacyDefault ? "coder" : normalizeRole(candidate.role ?? candidate.mode),
    systemInstructions: stringValue(candidate.systemInstructions, stringValue(candidate.instructions)),
    taskInstructions: stringValue(candidate.taskInstructions),
    workspaceRoot: stringValue(candidate.workspaceRoot, stringValue(candidate.workingDirectory)),
    thinkingLevel: isLegacyDefault ? "medium" : normalizeThinking(candidate.thinkingLevel),
    temperature: isLegacyDefault ? 0.6 : candidate.temperature,
    maxSteps: isLegacyDefault ? 32 : candidate.maxSteps,
  }, now);
}

export function validateAgentProfile(profile: AgentProfile): string[] {
  const errors: string[] = [];
  if (!profile.name.trim()) errors.push("Profile name is required.");
  if (profile.name.trim().length > 80) errors.push("Profile name must be 80 characters or fewer.");
  if (!agentModes.has(profile.role)) errors.push("Choose a supported agent role.");
  if (!thinkingLevels.has(profile.thinkingLevel)) errors.push("Choose a supported thinking level.");
  if (!executionModes.has(profile.executionMode)) errors.push("Choose Plan or Edit execution mode.");
  if (!Number.isFinite(profile.temperature) || profile.temperature < 0 || profile.temperature > 2) {
    errors.push("Temperature must be between 0 and 2.");
  }
  if (!Number.isInteger(profile.timeoutSeconds) || profile.timeoutSeconds < 10 || profile.timeoutSeconds > 3600) {
    errors.push("Timeout must be from 10 to 3600 seconds.");
  }
  if (!Number.isInteger(profile.retryCount) || profile.retryCount < 0 || profile.retryCount > 5) {
    errors.push("Retries must be from 0 to 5.");
  }
  if (!Number.isInteger(profile.maxSteps) || profile.maxSteps < 1 || profile.maxSteps > 64) {
    errors.push("Maximum steps must be from 1 to 64.");
  }
  return errors;
}

export function normalizeAgentProfileStore(value: unknown): AgentProfileStore {
  if (!value || typeof value !== "object") throw new Error("Profile file is not an object");
  const candidate = value as Partial<AgentProfileStore> & { profile?: unknown };
  const rawProfiles = Array.isArray(candidate.profiles)
    ? candidate.profiles
    : candidate.profile
      ? [candidate.profile]
      : [];
  if (rawProfiles.length === 0) throw new Error("Profile file contains no profiles");
  const sourceSchemaVersion = boundedInteger(candidate.schemaVersion, 1, 1, AGENT_PROFILE_SCHEMA_VERSION);
  const profiles = rawProfiles.map((profile, index) =>
    normalizeProfile(profile, index, sourceSchemaVersion < AGENT_PROFILE_SCHEMA_VERSION),
  );
  const ids = new Set<string>();
  for (const profile of profiles) {
    if (ids.has(profile.id)) profile.id = crypto.randomUUID();
    ids.add(profile.id);
    const errors = validateAgentProfile(profile);
    if (errors.length) throw new Error(`${profile.name}: ${errors.join(" ")}`);
  }
  const activeProfileId = profiles.some((profile) => profile.id === candidate.activeProfileId)
    ? String(candidate.activeProfileId)
    : profiles[0].id;
  const defaultProfileId = profiles.some((profile) => profile.id === candidate.defaultProfileId)
    ? String(candidate.defaultProfileId)
    : profiles[0].id;
  return {
    schemaVersion: AGENT_PROFILE_SCHEMA_VERSION,
    activeProfileId,
    defaultProfileId,
    profiles,
  };
}

export function createMigratedProfileStore(seed: LegacyAgentProfileSeed = {}): AgentProfileStore {
  const profile = createAgentProfile("Default", {
    role: seed.role == null ? "coder" : normalizeRole(seed.role),
    goal: stringValue(seed.goal),
    workspaceRoot: stringValue(seed.workspaceRoot),
    thinkingLevel: seed.thinkingLevel == null ? "medium" : normalizeThinking(seed.thinkingLevel),
    executionMode: normalizeExecutionMode(seed.executionMode),
    permissions: normalizePermissions(seed.permissions),
    autoApprove: Boolean(seed.autoApprove),
    yoloMode: Boolean(seed.yoloMode || seed.autoApprove),
    autoCompact: Boolean(seed.autoCompact),
  });
  return {
    schemaVersion: AGENT_PROFILE_SCHEMA_VERSION,
    activeProfileId: profile.id,
    defaultProfileId: profile.id,
    profiles: [profile],
  };
}

export function loadAgentProfileStore(
  primary: string | null,
  backup: string | null,
  seed: LegacyAgentProfileSeed = {},
): AgentProfileLoadResult {
  if (primary) {
    try {
      return {
        store: normalizeAgentProfileStore(JSON.parse(primary)),
        recovered: false,
        migrated: false,
        warning: "",
      };
    } catch (primaryError) {
      if (backup) {
        try {
          return {
            store: normalizeAgentProfileStore(JSON.parse(backup)),
            recovered: true,
            migrated: false,
            warning: `The profile store was invalid and the last backup was restored. ${String(primaryError)}`,
          };
        } catch {
          // Fall through to a safe migration seed.
        }
      }
      return {
        store: createMigratedProfileStore(seed),
        recovered: true,
        migrated: true,
        warning: `The profile store could not be read. A safe Default profile was created. ${String(primaryError)}`,
      };
    }
  }
  return {
    store: createMigratedProfileStore(seed),
    recovered: false,
    migrated: true,
    warning: "",
  };
}

export function duplicateAgentProfile(profile: AgentProfile, existingNames: string[], now = Date.now()): AgentProfile {
  const used = new Set(existingNames.map((name) => name.trim().toLocaleLowerCase()));
  const stem = `${profile.name} copy`;
  let name = stem;
  let suffix = 2;
  while (used.has(name.toLocaleLowerCase())) {
    name = `${stem} ${suffix}`;
    suffix += 1;
  }
  return createAgentProfile(name, {
    ...profile,
    id: crypto.randomUUID(),
    createdAt: now,
    updatedAt: now,
  }, now);
}

export function exportAgentProfiles(store: AgentProfileStore, profileId?: string): string {
  const profiles = profileId ? store.profiles.filter((profile) => profile.id === profileId) : store.profiles;
  if (profiles.length === 0) throw new Error("The selected profile no longer exists");
  return JSON.stringify({
    schemaVersion: AGENT_PROFILE_SCHEMA_VERSION,
    exportedAt: new Date().toISOString(),
    activeProfileId: profiles[0].id,
    defaultProfileId: profiles.some((profile) => profile.id === store.defaultProfileId)
      ? store.defaultProfileId
      : profiles[0].id,
    profiles,
  }, null, 2);
}
