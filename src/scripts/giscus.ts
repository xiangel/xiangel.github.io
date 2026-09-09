const GISCUS_ORIGIN = "https://giscus.app";

type GiscusTheme = "light" | "dark";

function getCurrentTheme(): GiscusTheme {
  const root = document.documentElement;
  return root.getAttribute("data-theme") === "dark" ? "dark" : "light";
}

function sendGiscusMessage(message: Record<string, unknown>) {
  const iframe = document.querySelector<HTMLIFrameElement>("iframe.giscus-frame");
  if (!iframe?.contentWindow) return;
  iframe.contentWindow.postMessage({ giscus: message }, GISCUS_ORIGIN);
}

function updateGiscusTheme(theme: GiscusTheme) {
  sendGiscusMessage({ setConfig: { theme } });
}

function loadGiscus() {
  const container = document.getElementById("giscus-container");
  if (!container || container.dataset.loaded === "true") return;

  const {
    repo,
    repoId,
    category,
    categoryId,
    mapping = "pathname",
    lang = "zh-CN",
    reactionsEnabled = "1",
  } = container.dataset;

  if (!repo || !repoId || !category || !categoryId) return;

  container.innerHTML = "";
  container.dataset.loaded = "true";

  const script = document.createElement("script");
  script.src = `${GISCUS_ORIGIN}/client.js`;
  script.async = true;
  script.crossOrigin = "anonymous";
  script.setAttribute("data-repo", repo);
  script.setAttribute("data-repo-id", repoId);
  script.setAttribute("data-category", category);
  script.setAttribute("data-category-id", categoryId);
  script.setAttribute("data-mapping", mapping);
  script.setAttribute("data-strict", "0");
  script.setAttribute("data-reactions-enabled", reactionsEnabled);
  script.setAttribute("data-emit-metadata", "0");
  script.setAttribute("data-input-position", "bottom");
  script.setAttribute("data-theme", getCurrentTheme());
  script.setAttribute("data-lang", lang);

  container.appendChild(script);
}

function setupGiscus() {
  loadGiscus();

  window.addEventListener("theme-change", event => {
    const theme = (event as CustomEvent<{ theme: string }>).detail.theme;
    updateGiscusTheme(theme === "dark" ? "dark" : "light");
  });
}

setupGiscus();
document.addEventListener("astro:page-load", setupGiscus);
