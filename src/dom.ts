export const $ = <T extends HTMLElement>(id: string): T => {
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

export function bindSelectionGuard() {
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

export function bindButtonRipples() {
  document.addEventListener(
    "pointerdown",
    (event) => {
      if (!(event.target instanceof Element)) return;
      const button = event.target.closest<HTMLButtonElement>("button");
      if (!button || button.disabled) return;

      const rect = button.getBoundingClientRect();
      button.style.setProperty("--ripple-x", `${event.clientX - rect.left}px`);
      button.style.setProperty("--ripple-y", `${event.clientY - rect.top}px`);
      button.classList.remove("button-ripple");
      void button.offsetWidth;
      button.classList.add("button-ripple");
    },
    true,
  );

  document.addEventListener("animationend", (event) => {
    if (event.animationName !== "button-ripple") return;
    if (event.target instanceof HTMLButtonElement) {
      event.target.classList.remove("button-ripple");
    }
  });
}
