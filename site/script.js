const viewButtons = document.querySelectorAll("[data-view]");
const diagramViews = document.querySelectorAll("[data-diagram]");
const explanation = document.getElementById("diagram-explanation");
const description = document.getElementById("diagram-description");
const playbackNote = document.getElementById("diagram-playback");
const switchIntervalMs = 5000;
let currentView = "shared";
let switchTimer;

const viewContent = {
  separate: {
    explanation: "SGD processes the same ‘cas’ prefix in each separate example before predicting the next token: t, e, or h.",
    description: "The SGD view shows three isolated token sequences. Each repeats c, a, and s before ending in t, e, or h.",
  },
  shared: {
    explanation: "AGPT represents the shared ‘cas’ path once. Each branch contributes evidence back through that shared computation.",
    description: "The AGPT view stores the cas prefix once in a trie before branching to t, e, and h.",
  },
};

function setView(selected) {
  if (!viewContent[selected]) return;
  currentView = selected;
  for (const button of viewButtons) {
    const active = button.dataset.view === selected;
    button.setAttribute("aria-pressed", String(active));
    button.classList.toggle("is-active", active);
  }
  for (const view of diagramViews) {
    view.toggleAttribute("hidden", view.dataset.diagram !== selected);
  }
  explanation.textContent = viewContent[selected].explanation;
  description.textContent = viewContent[selected].description;
}

if (!window.matchMedia("(prefers-reduced-motion: reduce)").matches) {
  playbackNote.hidden = false;
  playbackNote.textContent = "Auto-switching · choose a view to pause";
  switchTimer = window.setInterval(() => {
    if (document.visibilityState === "visible") {
      setView(currentView === "shared" ? "separate" : "shared");
    }
  }, switchIntervalMs);
}

for (const button of viewButtons) {
  button.addEventListener("click", () => {
    if (switchTimer !== undefined) {
      window.clearInterval(switchTimer);
      switchTimer = undefined;
      playbackNote.textContent = "Paused on your selection";
    }
    setView(button.dataset.view);
  });
}
