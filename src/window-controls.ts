import { getCurrentWindow } from "@tauri-apps/api/window";
import { $ } from "./dom";

function reportWindowError(action: string, error: unknown) {
  console.error(`Unable to ${action} the application window`, error);
}

export function initWindowControls() {
  const titlebar = $("window-titlebar");
  const minimizeButton = $("window-minimize") as HTMLButtonElement;
  const maximizeButton = $("window-maximize") as HTMLButtonElement;
  const closeButton = $("window-close") as HTMLButtonElement;
  let appWindow;
  try {
    appWindow = getCurrentWindow();
  } catch {
    // The browser preview renders the titlebar without native window actions.
    return;
  }

  const syncMaximizedState = async () => {
    try {
      const isMaximized = await appWindow.isMaximized();
      titlebar.classList.toggle("is-maximized", isMaximized);
      maximizeButton.ariaLabel = isMaximized ? "Restore window" : "Maximize window";
      maximizeButton.title = isMaximized ? "Restore" : "Maximize";
    } catch {
      titlebar.classList.remove("is-maximized");
    }
  };

  minimizeButton.addEventListener("click", () => {
    appWindow.minimize().catch((error) => reportWindowError("minimize", error));
  });

  maximizeButton.addEventListener("click", () => {
    appWindow
      .toggleMaximize()
      .then(syncMaximizedState)
      .catch((error) => reportWindowError("maximize", error));
  });

  closeButton.addEventListener("click", () => {
    appWindow.close().catch((error) => reportWindowError("close", error));
  });

  titlebar.addEventListener("dblclick", (event) => {
    if ((event.target as HTMLElement).closest(".window-controls")) return;
    appWindow
      .toggleMaximize()
      .then(syncMaximizedState)
      .catch((error) => reportWindowError("maximize", error));
  });

  void syncMaximizedState();
  appWindow.onResized(syncMaximizedState).catch(() => {
    // A plain-browser preview has no native window events.
  });
}
