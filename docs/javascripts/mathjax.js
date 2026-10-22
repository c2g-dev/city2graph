/* Load MathJax only for pages that contain maths or notebook markdown. */
(() => {
  let loading;
  let typesetting = Promise.resolve();

  window.MathJax = {
    tex: {
      inlineMath: [["\\(", "\\)"], ["$", "$"]],
      displayMath: [["\\[", "\\]"], ["$$", "$$"]],
      processEscapes: true,
      processEnvironments: true
    },
    options: {
      ignoreHtmlClass: ".*",
      processHtmlClass: "arithmatex|jp-RenderedMarkdown"
    },
    startup: { typeset: false }
  };

  function loadMathJax() {
    if (!loading) {
      loading = new Promise((resolve, reject) => {
        const script = document.createElement("script");
        script.src = "https://cdn.jsdelivr.net/npm/mathjax@4.1.3/tex-mml-chtml.js";
        script.async = true;
        script.onload = () => window.MathJax.startup.promise.then(resolve, reject);
        script.onerror = () => {
          script.remove();
          loading = undefined;
          reject(new Error("MathJax could not be loaded"));
        };
        document.head.append(script);
      });
    }
    return loading;
  }

  document$.subscribe(() => {
    const content = document.querySelector(".md-content");
    if (!content) return;
    const notebookMaths = [...content.querySelectorAll(".jp-RenderedMarkdown")].some(node =>
      /\$[^$]+\$|\\\(|\\\[|\\begin\{/.test(node.textContent)
    );
    if (!content.querySelector(".arithmatex") && !notebookMaths) return;

    // Serialise rendering and skip pages replaced while the library was loading.
    typesetting = typesetting.then(async () => {
      await loadMathJax();
      if (!content.isConnected) return;
      window.MathJax.typesetClear();
      window.MathJax.texReset();
      await window.MathJax.typesetPromise([content]);
    }).catch(error => console.warn("Unable to render maths:", error));
  });
})();
