import { cp, mkdir, rm } from "node:fs/promises";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import { spawn } from "node:child_process";

const root = dirname(dirname(fileURLToPath(import.meta.url)));
const piPackage = join(root, "node_modules", "@earendil-works", "pi-coding-agent");
const piDist = join(piPackage, "dist");
const outDir = join(root, "src-tauri", "binaries");
const outExe = join(outDir, "pi-x86_64-pc-windows-msvc.exe");

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
await run("bun", [
  "build",
  "--compile",
  join(piDist, "bun", "cli.js"),
  "--outfile",
  outExe,
]);

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
