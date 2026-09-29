// Page entry point: fill reported numbers, then start the interactive pieces.

import { fillMacros, loadJSON } from "./data.js";
import { initExplainer } from "./explainer.js";
import { initPlayground } from "./playground.js";
import { initConditions, renderRecording } from "./players.js";

function showError(container, what, error) {
  container.innerHTML = "";
  const message = document.createElement("p");
  message.className = "error";
  message.setAttribute("role", "alert");
  message.textContent = `Could not load the ${what} (${error.message}).`;
  container.appendChild(message);
  console.error(error);
}

/**
 * Run `start` once, when `element` comes within a screen of the viewport or
 * the page has finished loading and the browser is idle, whichever is first.
 * A screen reader's reading cursor need not scroll the viewport, so a section
 * must not wait for scrolling alone.
 */
function whenNearOrIdle(element, start) {
  let started = false;
  let observer = null;
  const run = () => {
    if (started) return;
    started = true;
    observer?.disconnect();
    start();
  };
  if ("IntersectionObserver" in window) {
    observer = new IntersectionObserver(
      (entries) => {
        if (entries.some((entry) => entry.isIntersecting)) run();
      },
      { rootMargin: "100% 0px" },
    );
    observer.observe(element);
  }
  const idle = () =>
    "requestIdleCallback" in window ? requestIdleCallback(run, { timeout: 2000 }) : setTimeout(run, 500);
  if (document.readyState === "complete") idle();
  else window.addEventListener("load", idle, { once: true });
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
    showError(simulation.querySelector("#cond-view"), "simulation", error);
    showError(recording.querySelector("#rec-view"), "recording", error);
    return;
  }
  fillMacros(document, { ...manifest.macros, ...manifest.page_values });

  whenNearOrIdle(simulation, () => initConditions(simulation, manifest));

  whenNearOrIdle(recording, () =>
    loadJSON("data/recording.json")
      .then((data) => renderRecording(recording, data, manifest))
      .catch((error) => showError(recording.querySelector("#rec-view"), "recording", error)),
  );
}

main();
