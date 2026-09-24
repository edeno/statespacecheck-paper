// Page entry point: fill reported numbers, then start the interactive pieces.

import { fillMacros, loadJSON } from "./data.js";
import { initExplainer } from "./explainer.js";
import { initPlayground } from "./playground.js";
import { initScenarios, renderReplay } from "./players.js";

function showError(container, what, error) {
  container.innerHTML = "";
  const message = document.createElement("p");
  message.className = "error";
  message.textContent = `Could not load the ${what} (${error.message}).`;
  container.appendChild(message);
  console.error(error);
}

/** Run `start` once `element` is within a screen of the viewport. */
function whenNear(element, start) {
  if (!("IntersectionObserver" in window)) {
    start();
    return;
  }
  const observer = new IntersectionObserver(
    (entries) => {
      if (entries.some((entry) => entry.isIntersecting)) {
        observer.disconnect();
        start();
      }
    },
    { rootMargin: "100% 0px" },
  );
  observer.observe(element);
}

function wireCopyButtons() {
  for (const button of document.querySelectorAll("[data-copy]")) {
    button.addEventListener("click", async () => {
      const text = document.querySelector(button.dataset.copy).textContent;
      try {
        await navigator.clipboard.writeText(text);
        button.textContent = "Copied";
      } catch {
        button.textContent = "Select and copy";
      }
      setTimeout(() => {
        button.textContent = "Copy";
      }, 2000);
    });
  }
}

async function main() {
  wireCopyButtons();
  const explainer = document.querySelector("#filter");
  const playground = document.querySelector("#playground");
  const simulation = document.querySelector("#simulation");
  const recording = document.querySelector("#real-data");

  // The three above-the-fold files are independent: request them together.
  const manifestLoad = loadJSON("data/manifest.json");
  Promise.all([loadJSON("data/filter.json"), manifestLoad])
    .then(([data, manifest]) => initExplainer(explainer, data, manifest))
    .catch((error) => showError(explainer.querySelector("#ft-tracks"), "filter explainer", error));
  Promise.all([loadJSON("data/playground.json"), manifestLoad])
    .then(([data]) => initPlayground(playground, data))
    .catch((error) => showError(playground.querySelector("#pg-chart"), "playground", error));

  let manifest;
  try {
    manifest = await manifestLoad;
  } catch (error) {
    showError(simulation.querySelector("#sc-view"), "simulation", error);
    showError(recording.querySelector("#rp-view"), "recording", error);
    return;
  }
  fillMacros(document, manifest.macros);

  whenNear(simulation, () => initScenarios(simulation, manifest));

  whenNear(recording, () =>
    loadJSON("data/replay.json")
      .then((data) => renderReplay(recording, data, manifest))
      .catch((error) => showError(recording.querySelector("#rp-view"), "recording", error)),
  );
}

main();
