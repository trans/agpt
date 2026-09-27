const viewButtons = document.querySelectorAll("[data-view]");
const diagramViews = document.querySelectorAll("[data-diagram]");
const explanation = document.getElementById("diagram-explanation");
const description = document.getElementById("diagram-description");

const viewContent = {
  separate: {
    explanation: "Each isolated row processes the same ‘cas’ prefix again before predicting a different next token: t, e, or h.",
    description: "Three isolated token sequences repeat the c, a, and s prefix, then end with t, e, and h respectively.",
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
      view.toggleAttribute("hidden", view.dataset.diagram !== selected);
    }
    explanation.textContent = viewContent[selected].explanation;
    description.textContent = viewContent[selected].description;
  });
}
