import { cp, mkdir, rm } from "node:fs/promises";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import { spawn } from "node:child_process";

const root = dirname(dirname(fileURLToPath(import.meta.url)));
const piPackage = join(root, "node_modules", "@earendil-works", "pi-coding-agent");
const piDist = join(piPackage, "dist");
const outDir = join(root, "src-tauri", "binaries");

const targetMatrix = {
  "x86_64-pc-windows-msvc": { bunTarget: "bun-windows-x64-baseline", extension: ".exe" },
  "aarch64-pc-windows-msvc": { bunTarget: "bun-windows-arm64", extension: ".exe" },
  "x86_64-unknown-linux-gnu": { bunTarget: "bun-linux-x64-baseline", extension: "" },
  "aarch64-unknown-linux-gnu": { bunTarget: "bun-linux-arm64", extension: "" },
  "x86_64-apple-darwin": { bunTarget: "bun-darwin-x64", extension: "" },
  "aarch64-apple-darwin": { bunTarget: "bun-darwin-arm64", extension: "" },
};

function hostTargetTriple() {
  const key = `${process.platform}-${process.arch}`;
  const targets = {
    "win32-x64": "x86_64-pc-windows-msvc",
    "win32-arm64": "aarch64-pc-windows-msvc",
    "linux-x64": "x86_64-unknown-linux-gnu",
    "linux-arm64": "aarch64-unknown-linux-gnu",
    "darwin-x64": "x86_64-apple-darwin",
    "darwin-arm64": "aarch64-apple-darwin",
  };
  const target = targets[key];
  if (!target) throw new Error(`Unsupported Pi sidecar build host: ${key}`);
  return target;
}

const hostTriple = hostTargetTriple();
const targetTriple =
  process.env.PI_TARGET_TRIPLE ||
  process.env.TAURI_ENV_TARGET_TRIPLE ||
  hostTriple;
const target = targetMatrix[targetTriple];
if (!target) {
  throw new Error(
    `Unsupported Pi sidecar target "${targetTriple}". Supported targets: ${Object.keys(targetMatrix).join(", ")}`,
  );
}
const bunTarget =
  process.env.PI_BUN_TARGET ||
  (targetTriple === hostTriple ? null : target.bunTarget);
const outExe = join(outDir, `pi-${targetTriple}${target.extension}`);

if (process.argv.includes("--describe")) {
  console.log(
    JSON.stringify(
      { targetTriple, bunTarget: bunTarget || "native", output: outExe },
      null,
      2,
    ),
  );
  process.exit(0);
}

function run(command, args) {
  return new Promise((resolve, reject) => {
    const child = spawn(command, args, { cwd: root, stdio: "inherit", shell: false });
    child.on("error", reject);
    child.on("exit", (code) => {
      if (code === 0) resolve();
      else reject(new Error(`${command} ${args.join(" ")} exited with ${code}`));
    });
  });
}

async function copyDirectory(from, to) {
  await rm(to, { recursive: true, force: true });
  await cp(from, to, { recursive: true });
}

async function copyFile(from, to) {
  await cp(from, to);
}

await mkdir(outDir, { recursive: true });
const bunArgs = [
  "build",
  "--compile",
  join(piDist, "bun", "cli.js"),
  "--outfile",
  outExe,
];
if (bunTarget) bunArgs.splice(2, 0, `--target=${bunTarget}`);
await run("bun", bunArgs);

await copyDirectory(join(piDist, "modes", "interactive", "theme"), join(outDir, "theme"));
await copyDirectory(join(piDist, "modes", "interactive", "assets"), join(outDir, "assets"));
await copyDirectory(join(piDist, "core", "export-html"), join(outDir, "export-html"));
await copyDirectory(join(piPackage, "docs"), join(outDir, "docs"));
await copyDirectory(join(piPackage, "examples"), join(outDir, "examples"));
await copyFile(join(piPackage, "package.json"), join(outDir, "package.json"));
await copyFile(join(piPackage, "README.md"), join(outDir, "README.md"));
await copyFile(join(piPackage, "CHANGELOG.md"), join(outDir, "CHANGELOG.md"));

try {
  await copyFile(
    join(root, "node_modules", "@silvia-odwyer", "photon-node", "photon_rs_bg.wasm"),
    join(outDir, "photon_rs_bg.wasm"),
  );
} catch {
  // Optional image support is copied when the package is present.
}
