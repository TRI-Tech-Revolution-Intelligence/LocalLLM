import type { ConfirmActionOptions } from "./types";

export function confirmAction(options: ConfirmActionOptions): Promise<boolean> {
  return new Promise((resolve) => {
    const previousFocus = document.activeElement instanceof HTMLElement ? document.activeElement : null;
    const overlay = document.createElement("div");
    const modal = document.createElement("section");
    const titleId = `confirm-title-${Date.now()}`;
    const messageId = `confirm-message-${Date.now()}`;
    const title = document.createElement("h2");
    const message = document.createElement("p");
    const actions = document.createElement("div");
    const cancelButton = document.createElement("button");
    const okButton = document.createElement("button");

    overlay.className = "app-confirm-overlay";
    modal.className = "app-confirm";
    modal.dataset.kind = options.kind ?? "warning";
    modal.setAttribute("role", "dialog");
    modal.setAttribute("aria-modal", "true");
    modal.setAttribute("aria-labelledby", titleId);
    modal.setAttribute("aria-describedby", messageId);
    modal.tabIndex = -1;

    title.id = titleId;
    title.textContent = options.title;
    message.id = messageId;
    message.textContent = options.message;

    actions.className = "button-row app-confirm-actions";
    cancelButton.type = "button";
    cancelButton.textContent = options.cancelLabel;
    okButton.type = "button";
    okButton.className = options.kind === "danger" ? "danger" : "primary";
    okButton.textContent = options.okLabel;

    actions.append(cancelButton, okButton);
    modal.append(title, message, actions);
    overlay.append(modal);

    let settled = false;
    const close = (result: boolean) => {
      if (settled) return;
      settled = true;
      document.removeEventListener("keydown", handleKeydown);
      overlay.remove();
      previousFocus?.focus({ preventScroll: true });
      resolve(result);
    };
    const handleKeydown = (event: KeyboardEvent) => {
      if (event.key === "Escape") {
        event.preventDefault();
        close(false);
      }
    };

    cancelButton.addEventListener("click", () => close(false));
    okButton.addEventListener("click", () => close(true));
    overlay.addEventListener("pointerdown", (event) => {
      if (event.target === overlay) close(false);
    });
    document.addEventListener("keydown", handleKeydown);
    document.body.append(overlay);
    okButton.focus({ preventScroll: true });
  });
}
