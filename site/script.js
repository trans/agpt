const viewButtons = document.querySelectorAll("[data-view]");
const diagramViews = document.querySelectorAll("[data-diagram]");
const explanation = document.getElementById("diagram-explanation");
const description = document.getElementById("diagram-description");

const viewContent = {
  separate: {
    explanation: "In separate paths, the common prefix ‘cas’ is processed three times: once for each word.",
    description: "Three separate paths for cast, case, and cash each repeat the cas prefix.",
  },
  shared: {
    explanation: "The shared ‘cas’ path is represented once. Each branch contributes evidence back through that shared computation.",
    description: "A trie stores the cas prefix once before branching to t, e, and h.",
  },
};

for (const button of viewButtons) {
  button.addEventListener("click", () => {
    const selected = button.dataset.view;
    if (!viewContent[selected]) return;
    for (const candidate of viewButtons) {
      const active = candidate.dataset.view === selected;
      candidate.setAttribute("aria-pressed", String(active));
      candidate.classList.toggle("is-active", active);
    }
    for (const view of diagramViews) {
      view.hidden = view.dataset.diagram !== selected;
    }
    explanation.textContent = viewContent[selected].explanation;
    description.textContent = viewContent[selected].description;
  });
}
