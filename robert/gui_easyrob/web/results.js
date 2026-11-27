"use strict";

(() => {
  let bridge = null;
  let activeView = null;
  let sourceSignature = "";
  let reportPage = 0;
  let predictionPage = 0;
  let sortColumn = "";
  let descending = false;
  let plotType = "observed";
  let plotData = null;
  let requestId = 0;
  let inspectedPoint = "";
  let diagnosticKey = "";
  let predictionNumeric = null;
  const hiddenGroups = new Set();
  const svgNS = "http://www.w3.org/2000/svg";
  // Match the blue training and red test markers in the generated report figures.
  const colors = { CV: "#0000ff", Test: "#ff0000", External: "#ff0000", Data: "#0000ff",
    "Typical 80%": "#0000ff", "High leverage 20%": "#ff0000" };

  function setOptions(element, options) {
    const oldValue = element.value;
    element.replaceChildren();
    for (const option of options) {
      const item = document.createElement("option");
      item.value = option.value;
      item.textContent = option.label;
      element.appendChild(item);
    }
    if (options.some(option => option.value === oldValue)) element.value = oldValue;
  }

  function errorFor(name, message) {
    const element = document.getElementById(`${name}-error`);
    element.hidden = !message;
    element.textContent = message || "";
  }

  function open(name) {
    const ids = { Report: "report-view", Predictions: "prediction-view", "Interactive plots": "plot-view" };
    activeView = ids[name] || null;
    for (const id of Object.values(ids)) document.getElementById(id).hidden = id !== activeView;
    if (!activeView) return;
    document.getElementById(activeView).scrollIntoView({ behavior: "smooth", block: "start" });
    if (name === "Report") loadReport();
    if (name === "Predictions") loadPredictions();
    if (name === "Interactive plots") loadPlot();
  }

  function resetPredictionSelection() {
    predictionPage = 0;
    sortColumn = "";
    descending = false;
    diagnosticKey = "";
    document.getElementById("prediction-search").value = "";
    document.getElementById("prediction-histogram").hidden = true;
    predictionNumeric = null;
    const statistics = document.getElementById("prediction-numeric");
    statistics.dataset.snapshot = "";
    statistics.replaceChildren();
  }

  function render(results) {
    const viewNames = { "report-view": "Report", "prediction-view": "Predictions",
      "plot-view": "Interactive plots" };
    if (activeView && !results.available[viewNames[activeView]]) {
      open(null);
      if (bridge) bridge.performAction("activeResultView", "");
    }
    const signature = JSON.stringify([results.revision, results.selectedModel,
      results.files, results.predictionFiles]);
    if (sourceSignature === signature) return;
    sourceSignature = signature;
    reportPage = 0;
    resetPredictionSelection();
    setOptions(document.getElementById("report-file"), (results.files.Report || []).map((label, index) => ({ value: String(index), label })));
    setOptions(document.getElementById("prediction-file"), (results.predictionFiles || []).map(item => ({ value: item.key, label: item.label })));
    setOptions(document.getElementById("plot-file"), (results.files["Interactive plots"] || []).map((label, index) => ({ value: String(index), label })));
    if (activeView === "report-view") loadReport();
    if (activeView === "prediction-view") loadPredictions();
    if (activeView === "plot-view") loadPlot();
  }

  function loadReport() {
    if (!bridge || !document.getElementById("report-file").options.length) return;
    const token = ++requestId;
    errorFor("report", "");
    bridge.getReportPage(Number(document.getElementById("report-file").value), reportPage,
      Number(document.getElementById("report-zoom").value), payload => {
        if (token !== requestId) return;
        const result = JSON.parse(payload);
        if (result.error) return errorFor("report", result.error);
        reportPage = result.page;
        document.getElementById("report-image").src = result.image;
        document.getElementById("report-position").textContent = `Page ${result.page + 1} / ${result.pages}`;
        document.getElementById("report-previous").disabled = result.page === 0;
        document.getElementById("report-next").disabled = result.page >= result.pages - 1;
      });
  }

  function loadPredictions() {
    const key = document.getElementById("prediction-file").value;
    if (!bridge || !key) return;
    const token = ++requestId;
    errorFor("prediction", "");
    bridge.getPredictionTable(key, document.getElementById("prediction-search").value,
      sortColumn, descending, predictionPage, payload => {
        if (token !== requestId) return;
        const result = JSON.parse(payload);
        if (result.error) return errorFor("prediction", result.error);
        predictionPage = result.page;
        const head = document.getElementById("prediction-head");
        const body = document.getElementById("prediction-body");
        head.replaceChildren();
        body.replaceChildren();
        const header = document.createElement("tr");
        for (const column of result.columns) {
          const cell = document.createElement("th");
          if (column === result.nameColumn) cell.classList.add("identifier-column");
          if (column !== "Image") {
            const button = document.createElement("button");
            button.type = "button";
            button.dataset.sortColumn = column;
            button.textContent = `${column}${sortColumn === column ? (descending ? " ↓" : " ↑") : " ↕"}`;
            button.setAttribute("aria-label", `Sort by ${column}`);
            cell.appendChild(button);
          } else cell.textContent = column;
          header.appendChild(cell);
        }
        head.appendChild(header);
        for (const row of result.rows) {
          const line = document.createElement("tr");
          row.forEach((value, index) => {
            const cell = document.createElement("td");
            if (result.columns[index] === result.nameColumn) cell.classList.add("identifier-column");
            if (result.columns[index] === "Image" && value) {
              const image = document.createElement("img");
              image.src = value;
              image.alt = "Molecule";
              image.loading = "lazy";
              cell.appendChild(image);
            } else cell.textContent = value;
            line.appendChild(cell);
          });
          body.appendChild(line);
        }
        if (!result.rows.length) {
          const line = document.createElement("tr");
          const cell = document.createElement("td");
          cell.colSpan = Math.max(1, result.columns.length);
          cell.className = "prediction-empty";
          cell.textContent = "No rows match this search. Try another identifier.";
          line.appendChild(cell);
          body.appendChild(line);
        }
        document.getElementById("prediction-count").textContent = `${result.total} of ${result.allRows} rows`;
        document.getElementById("prediction-columns").textContent = `${result.columns.length} columns`;
        document.getElementById("prediction-identifier").textContent = result.nameColumn
          ? `Identifier: ${result.nameColumn}` : "No identifier column detected";
        document.getElementById("prediction-position").textContent = `Page ${result.page + 1} / ${result.pages}`;
        document.getElementById("prediction-previous").disabled = result.page === 0;
        document.getElementById("prediction-next").disabled = result.page >= result.pages - 1;
        document.getElementById("prediction-search").placeholder = result.nameColumn ? `Search ${result.nameColumn}` : "Identifier unavailable";
        document.getElementById("prediction-search").disabled = !result.nameColumn;
        renderPredictionScenario(result.scenario);
        predictionNumeric = result.numeric;
        if (document.querySelector(".prediction-statistics").open) renderPredictionStatistics(result.numeric);
        if (document.querySelector(".prediction-insights").open) loadDiagnostics();
      });
  }

  function appendRecommendation(target, value) {
    const almosLink = '<a href="https://github.com/MiguelMartzFdez/almos">ALMOS</a>';
    value.split(almosLink).forEach((part, index) => {
      if (index) {
        const link = document.createElement("button");
        link.type = "button";
        link.className = "inline-link";
        link.dataset.action = "openAlmos";
        link.textContent = "ALMOS ↗";
        target.appendChild(link);
      }
      target.appendChild(document.createTextNode(part));
    });
  }

  function renderPredictionScenario(result) {
    const scenario = document.getElementById("prediction-scenario");
    scenario.replaceChildren();
    scenario.dataset.state = result.state;
    const marker = document.createElement("div");
    marker.className = "scenario-marker";
    marker.textContent = result.state === "FAILED" ? "!" : result.state === "STRONG" ? "✓" : "i";
    const content = document.createElement("div");
    content.className = "scenario-content";
    const heading = document.createElement("div");
    heading.className = "scenario-heading";
    const title = document.createElement("strong");
    title.textContent = "Model assessment";
    const badge = document.createElement("span");
    badge.className = "scenario-state";
    badge.textContent = result.state;
    heading.append(title, badge);
    const message = document.createElement("p");
    message.textContent = (result.messages || []).map(value => value
      .replace(/^CheckML score is /, "Interpolation score (CheckML) is ")
      .replace(/^No valid CheckML score/, "No valid Interpolation score")).join(" ");
    content.append(heading, message);
    if (result.recommendations?.length) {
      const advice = document.createElement("div");
      advice.className = "scenario-advice";
      const label = document.createElement("small");
      label.textContent = "SUGGESTED NEXT STEP";
      const copy = document.createElement("p");
      appendRecommendation(copy, result.recommendations.join(" "));
      advice.append(label, copy);
      content.appendChild(advice);
    }
    scenario.append(marker, content);
  }

  function renderPredictionStatistics(numeric) {
    const grid = document.getElementById("prediction-numeric");
    const signature = JSON.stringify(numeric);
    if (grid.dataset.snapshot === signature) return;
    grid.dataset.snapshot = signature;
    grid.replaceChildren();
    for (const [column, stats] of Object.entries(numeric)) {
      if (stats.mean === null || stats.mean === undefined) continue;
      const card = document.createElement("div");
      card.className = "prediction-stat-card";
      const heading = document.createElement("strong");
      heading.textContent = column;
      const range = document.createElement("p");
      range.textContent = `${Number(stats.min).toPrecision(3)} – ${Number(stats.max).toPrecision(3)} range`;
      const detail = document.createElement("small");
      detail.textContent = `Median ${Number(stats.median).toPrecision(3)} · Mean ${Number(stats.mean).toPrecision(3)}`;
      const histogram = document.createElement("button");
      histogram.type = "button";
      histogram.className = "histogram-link";
      histogram.dataset.histogram = column;
      histogram.textContent = "View distribution →";
      card.append(heading, range, detail, histogram);
      grid.appendChild(card);
    }
    if (!grid.children.length) grid.textContent = "No numeric columns are available for this dataset.";
  }

  function loadDiagnostics() {
    const key = document.getElementById("prediction-file").value;
    if (!bridge || !key || key === diagnosticKey) return;
    diagnosticKey = key;
    bridge.getPredictionDiagnostics(key, payload => {
      if (key !== document.getElementById("prediction-file").value) return;
      const result = JSON.parse(payload);
      const grid = document.getElementById("prediction-diagnostics");
      grid.replaceChildren();
      if (result.error) { grid.textContent = result.error; return; }
      const cards = [
        ["Model performance", result.scoreImage],
        [result.boundaryScore === null ? "Boundary robustness" : `Boundary robustness · ${result.boundaryScore}/10`, result.boundaryImage],
        ["External validation", result.externalImage],
      ];
      for (const [title, source] of cards) {
        if (!source) continue;
        const card = document.createElement("div");
        card.className = "diagnostic-card";
        const heading = document.createElement("strong");
        heading.textContent = title;
        const image = document.createElement("img");
        image.src = source;
        image.alt = title;
        card.append(heading, image);
        grid.appendChild(card);
      }
      if (!grid.children.length) grid.textContent = "No report diagnostics are available for this model.";
    });
  }

  function loadHistogram(column) {
    const key = document.getElementById("prediction-file").value;
    bridge.getPredictionHistogram(key, column, payload => {
      if (key !== document.getElementById("prediction-file").value) return;
      const result = JSON.parse(payload);
      if (result.error) return errorFor("prediction", result.error);
      const panel = document.getElementById("prediction-histogram");
      const svg = document.getElementById("histogram-svg");
      panel.hidden = false;
      document.getElementById("histogram-title").textContent = `Distribution of ${column}`;
      svg.replaceChildren();
      const top = 25, bottom = 205, left = 43, right = 595;
      const maximum = Math.max(1, ...result.counts);
      const width = (right - left) / result.counts.length;
      result.counts.forEach((count, index) => {
        const height = count / maximum * (bottom - top);
        svg.appendChild(svgElement("rect", { x: left + index * width + 2, y: bottom - height,
          width: Math.max(1, width - 4), height, rx: 3, fill: "#0000ff" }));
      });
      svg.appendChild(svgElement("text", { x: left, y: 230, class: "plot-tick" }, Number(result.edges[0]).toPrecision(3)));
      svg.appendChild(svgElement("text", { x: right, y: 230, "text-anchor": "end", class: "plot-tick" }, Number(result.edges.at(-1)).toPrecision(3)));
      svg.appendChild(svgElement("text", { x: left, y: 18, class: "plot-tick" }, `${maximum} rows`));
      panel.scrollIntoView({ behavior: "smooth", block: "nearest" });
    });
  }

  function svgElement(name, attributes = {}, label = "") {
    const element = document.createElementNS(svgNS, name);
    for (const [key, value] of Object.entries(attributes)) element.setAttribute(key, String(value));
    if (label) element.textContent = label;
    return element;
  }

  function drawLine(svg, x1, y1, x2, y2, dashed = false) {
    svg.appendChild(svgElement("line", { x1, y1, x2, y2, stroke: "#808080",
      "stroke-width": 1, "stroke-dasharray": dashed ? "5 5" : "" }));
  }

  function inspectPoint(point, williams) {
    const signature = `${williams}:${point.index}`;
    if (inspectedPoint === signature) return;
    inspectedPoint = signature;
    const format = value => Number(value).toPrecision(4);
    const lines = [point.label, `Group: ${point.group}`];
    if (williams) lines.push(`Leverage: ${format(point.x)}`, `Standardized residual: ${format(point.y)}`);
    if (point.observed !== null) lines.push(`Observed: ${format(point.observed)}`);
    if (point.predicted !== null) lines.push(`Predicted: ${format(point.predicted)}`);
    if (!williams) {
      if (point.residual !== null) lines.push(`Residual: ${format(point.residual)}`);
      if (point.outlier !== null) lines.push(`Outlier score: ${format(point.outlier)}`);
    }
    document.getElementById("plot-point-details").textContent = lines.join("\n");
    const image = document.getElementById("plot-molecule");
    image.hidden = true;
    bridge.getPlotMolecule(Number(document.getElementById("plot-file").value), point.index, williams, source => {
      if (inspectedPoint !== signature) return;
      image.hidden = !source;
      if (source) image.src = source;
    });
  }

  function drawPlot() {
    if (!plotData) return;
    const svg = document.getElementById("plot-svg");
    const fallback = document.getElementById("williams-fallback");
    svg.replaceChildren();
    fallback.hidden = true;
    const williams = plotType === "williams";
    const points = williams ? plotData.williams : plotData.points;
    if (williams && !points.length && plotData.williamsImage) {
      svg.hidden = true;
      fallback.src = plotData.williamsImage;
      fallback.hidden = false;
      document.getElementById("plot-legend").replaceChildren();
      return;
    }
    svg.hidden = false;
    const coordinates = points.map(point => ({ ...point,
      x: williams ? point.x : plotType === "outlier" ? point.outlier : point.observed,
      y: williams ? point.y : plotType === "outlier" ? point.outlier : point.predicted,
    })).filter(point => Number.isFinite(point.x) && Number.isFinite(point.y));
    const valuesX = coordinates.map(point => point.x);
    const valuesY = coordinates.map(point => point.y);
    const threshold = plotData.threshold || 2;
    let minX = valuesX.length ? Math.min(...valuesX) : 0;
    let maxX = valuesX.length ? Math.max(...valuesX) : 1;
    let minY = valuesY.length ? Math.min(...valuesY) : 0;
    let maxY = valuesY.length ? Math.max(...valuesY) : 1;
    if (williams) { minX = 0; maxX = Math.max(maxX, (points[0]?.hStar || 0) * 1.2); minY = Math.min(minY, -3.5); maxY = Math.max(maxY, 3.5); }
    else if (plotType === "outlier") { const limit = Math.max(2.5, Math.abs(minX) + .5, Math.abs(maxX) + .5); minX = minY = -limit; maxX = maxY = limit; }
    else { const low = Math.min(minX, minY); const high = Math.max(maxX, maxY); minX = minY = low; maxX = maxY = high; }
    const paddingX = (maxX - minX || 1) * .07;
    const paddingY = (maxY - minY || 1) * .07;
    minX -= paddingX; maxX += paddingX; minY -= paddingY; maxY += paddingY;
    const left = 62, top = 28, width = 576, height = 340;
    const px = value => left + (value - minX) / (maxX - minX) * width;
    const py = value => top + height - (value - minY) / (maxY - minY) * height;
    for (let tick = 0; tick <= 4; tick++) {
      const x = left + width * tick / 4;
      const y = top + height * tick / 4;
      drawLine(svg, x, top, x, top + height, true);
      drawLine(svg, left, y, left + width, y, true);
      svg.appendChild(svgElement("text", { x, y: top + height + 19, "text-anchor": "middle", class: "plot-tick" }, (minX + (maxX - minX) * tick / 4).toPrecision(3)));
      svg.appendChild(svgElement("text", { x: left - 9, y: y + 4, "text-anchor": "end", class: "plot-tick" }, (maxY - (maxY - minY) * tick / 4).toPrecision(3)));
    }
    if (williams) {
      for (const bound of [-3, 3]) drawLine(svg, left, py(bound), left + width, py(bound), true);
      if (points[0]?.hStar !== null && Number.isFinite(points[0]?.hStar)) drawLine(svg, px(points[0].hStar), top, px(points[0].hStar), top + height, true);
    } else if (plotType === "observed") drawLine(svg, px(Math.max(minX, minY)), py(Math.max(minX, minY)), px(Math.min(maxX, maxY)), py(Math.min(maxX, maxY)), true);
    else {
      for (const bound of [-threshold, threshold]) {
        drawLine(svg, px(bound), top, px(bound), top + height, true);
        drawLine(svg, left, py(bound), left + width, py(bound), true);
      }
    }
    drawLine(svg, left, top + height, left + width, top + height);
    drawLine(svg, left, top, left, top + height);
    const xLabel = williams ? "Leverage" : plotType === "outlier" ? "SD of the errors" : plotData.target;
    const yLabel = williams ? "Standardized residual" : plotType === "outlier" ? "SD of the errors" : `Predicted ${plotData.target}`;
    svg.appendChild(svgElement("text", { x: left + width / 2, y: 423, "text-anchor": "middle", class: "plot-axis" }, xLabel));
    svg.appendChild(svgElement("text", { transform: "translate(17 198) rotate(-90)", "text-anchor": "middle", class: "plot-axis" }, yLabel));
    const legend = document.getElementById("plot-legend");
    legend.replaceChildren();
    for (const group of [...new Set(coordinates.map(point => point.group))]) {
      const button = document.createElement("button");
      button.type = "button";
      button.className = `legend-button ${hiddenGroups.has(group) ? "muted" : ""}`;
      button.dataset.legendGroup = group;
      button.textContent = `${group === "CV" ? plotData.cvLabel : group} (${coordinates.filter(point => point.group === group).length})`;
      button.style.setProperty("--series-color", colors[group] || colors.Data);
      legend.appendChild(button);
    }
    for (const point of coordinates) {
      if (hiddenGroups.has(point.group)) continue;
      if (!williams && plotType === "observed" && point.sd !== null) {
        drawLine(svg, px(point.x), py(point.y - point.sd), px(point.x), py(point.y + point.sd));
      }
      const circle = svgElement("circle", { cx: px(point.x), cy: py(point.y), r: 5.5,
        fill: colors[point.group] || colors.Data, stroke: "#000", "stroke-width": 0.8,
        class: "plot-point", tabindex: 0, "aria-label": `${point.label}: ${point.x}, ${point.y}` });
      circle.addEventListener("mouseenter", () => inspectPoint(point, williams));
      circle.addEventListener("focus", () => inspectPoint(point, williams));
      circle.addEventListener("click", () => inspectPoint(point, williams));
      svg.appendChild(circle);
    }
    if (!coordinates.length) svg.appendChild(svgElement("text", { x: 340, y: 200, "text-anchor": "middle", class: "plot-axis" }, "No measured points are available for this plot"));
  }

  function loadPlot() {
    if (!bridge || !document.getElementById("plot-file").options.length) return;
    const token = ++requestId;
    errorFor("plot", "");
    bridge.getPlotData(Number(document.getElementById("plot-file").value), payload => {
      if (token !== requestId) return;
      const result = JSON.parse(payload);
      if (result.error) return errorFor("plot", result.error);
      plotData = result;
      hiddenGroups.clear();
      inspectedPoint = "";
      document.getElementById("plot-point-details").textContent = "Hover over or select a point.";
      document.getElementById("plot-molecule").hidden = true;
      drawPlot();
    });
  }

  document.addEventListener("click", event => {
    if (event.target.closest("[data-close-result]")) return open(null);
    if (event.target.id === "histogram-close") {
      document.getElementById("prediction-histogram").hidden = true;
      return;
    }
    const histogram = event.target.closest("[data-histogram]");
    if (histogram) return loadHistogram(histogram.dataset.histogram);
    const sort = event.target.closest("[data-sort-column]");
    if (sort) {
      descending = sortColumn === sort.dataset.sortColumn && !descending;
      sortColumn = sort.dataset.sortColumn;
      predictionPage = 0;
      return loadPredictions();
    }
    const mode = event.target.closest("[data-plot-type]");
    if (mode) {
      plotType = mode.dataset.plotType;
      document.querySelectorAll("[data-plot-type]").forEach(button => button.classList.toggle("active", button === mode));
      return drawPlot();
    }
    const legend = event.target.closest("[data-legend-group]");
    if (legend) {
      if (hiddenGroups.has(legend.dataset.legendGroup)) hiddenGroups.delete(legend.dataset.legendGroup);
      else hiddenGroups.add(legend.dataset.legendGroup);
      return drawPlot();
    }
    if (event.target.id === "report-previous") { reportPage -= 1; return loadReport(); }
    if (event.target.id === "report-next") { reportPage += 1; return loadReport(); }
    if (event.target.id === "prediction-previous") { predictionPage -= 1; return loadPredictions(); }
    if (event.target.id === "prediction-next") { predictionPage += 1; return loadPredictions(); }
  });
  document.addEventListener("change", event => {
    if (event.target.id === "report-file") { reportPage = 0; return loadReport(); }
    if (event.target.id === "report-zoom") return loadReport();
    if (event.target.id === "prediction-file") { resetPredictionSelection(); return loadPredictions(); }
    if (event.target.id === "plot-file") return loadPlot();
  });
  let searchTimer;
  document.getElementById("prediction-search").addEventListener("input", () => {
    clearTimeout(searchTimer);
    predictionPage = 0;
    searchTimer = setTimeout(loadPredictions, 200);
  });
  document.querySelector(".prediction-insights").addEventListener("toggle", event => {
    if (event.target.open) loadDiagnostics();
  });
  document.querySelector(".prediction-statistics").addEventListener("toggle", event => {
    if (event.target.open && predictionNumeric) renderPredictionStatistics(predictionNumeric);
  });

  window.easyrobResults = { connect: value => { bridge = value; }, render, open };
})();
