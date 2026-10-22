/* Keep Material's search available both locally and on Read the Docs. */
(() => {
  let versions;

  function renderVersions() {
    const topic = document.querySelector(".md-header__topic");
    if (!topic || !versions?.current || !Array.isArray(versions.active)) return;

    const selector = document.createElement("div");
    selector.className = "md-version";
    const button = document.createElement("button");
    button.type = "button";
    button.className = "md-version__current";
    button.setAttribute("aria-label", "Select version");
    button.setAttribute("aria-expanded", "false");
    button.setAttribute("aria-controls", "city2graph-versions");
    button.textContent = versions.current.slug;
    selector.append(button);

    const list = document.createElement("ul");
    list.className = "md-version__list";
    list.id = "city2graph-versions";
    for (const version of versions.active) {
      const item = document.createElement("li");
      item.className = "md-version__item";
      const link = document.createElement("a");
      link.className = "md-version__link";
      link.href = version.urls.documentation;
      link.textContent = version.slug;
      item.append(link);
      list.append(item);
    }
    selector.append(list);
    button.addEventListener("click", () => {
      const open = selector.toggleAttribute("data-open");
      button.setAttribute("aria-expanded", String(open));
      if (open) list.querySelector("a")?.focus();
    });
    selector.addEventListener("keydown", event => {
      if (event.key === "Escape") {
        selector.removeAttribute("data-open");
        button.setAttribute("aria-expanded", "false");
        button.focus();
      }
    });
    selector.addEventListener("focusout", event => {
      if (!selector.contains(event.relatedTarget)) {
        selector.removeAttribute("data-open");
        button.setAttribute("aria-expanded", "false");
      }
    });
    document.querySelector(".md-version")?.remove();
    topic.append(selector);
  }

  document.addEventListener("readthedocs-addons-data-ready", event => {
    versions = event.detail.data().versions;
    renderVersions();
  });
  document$.subscribe(renderVersions);
})();
