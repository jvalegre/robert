"use strict";

(() => {
  const checkmlMode = new URLSearchParams(window.location.search).get("mode") === "checkml";
  let bridge = null;
  let focusBotDraftAfterSend = false;
  let lastBotProvider = null;
  let lastBotApiSaved = null;
  const fieldSnapshots = new Map();
  const stages = ["CURATE", "GENERATE", "VERIFY", "PREDICT", "REPORT"];
  let galleryStage = "all";
  let galleryState = { stages: [], total: 0 };
  let tutorials = [];
  let tutorialTopic = 0;
  let tutorialStep = 0;
  let chemdrawRows = [];
  const logBuffers = { workflow: "", checkml: "" };
  const chemdrawOptionalColumns = ["charge", "mult", "complex_type", "sample", "geom"];
  const chemdrawColumns = document.getElementById("chemdraw-columns");
  for (const name of chemdrawOptionalColumns) {
    const label = document.createElement("label");
    const input = document.createElement("input");
    input.type = "checkbox";
    input.dataset.chemdrawColumn = name;
    label.append(input, document.createTextNode(name));
    chemdrawColumns.appendChild(label);
  }

  function renderChemdrawTable() {
    const optional = chemdrawOptionalColumns.filter(name => document.querySelector(`[data-chemdraw-column="${name}"]`).checked);
    const columns = ["Image", "SMILES", "code_name", "target", ...optional];
    const head = document.getElementById("chemdraw-head");
    const body = document.getElementById("chemdraw-body");
    head.replaceChildren();
    body.replaceChildren();
    const header = document.createElement("tr");
    for (const column of columns) {
      const cell = document.createElement("th");
      cell.textContent = column === "target" ? document.getElementById("chemdraw-target-name").value || "target" : column;
      header.appendChild(cell);
    }
    head.appendChild(header);
    chemdrawRows.forEach((row, rowIndex) => {
      const line = document.createElement("tr");
      for (const column of columns) {
        const cell = document.createElement("td");
        if (column === "Image") {
          if (row.image) {
            const image = document.createElement("img");
            image.src = row.image;
            image.alt = `Molecule ${rowIndex + 1}`;
            cell.appendChild(image);
          }
        } else if (column === "complex_type") {
          const select = document.createElement("select");
          for (const value of ["", "squareplanar", "squarepyramidal", "linear", "trigonalplanar"]) {
            const option = document.createElement("option");
            option.value = value;
            option.textContent = value || "Select geometry";
            select.appendChild(option);
          }
          select.value = row[column] || "";
          select.dataset.chemdrawRow = String(rowIndex);
          select.dataset.chemdrawField = column;
          cell.appendChild(select);
        } else {
          const input = document.createElement("input");
          input.value = row[column] || "";
          input.dataset.chemdrawRow = String(rowIndex);
          input.dataset.chemdrawField = column;
          input.setAttribute("aria-label", `${column} for molecule ${rowIndex + 1}`);
          cell.appendChild(input);
        }
        line.appendChild(cell);
      }
      body.appendChild(line);
    });
  }

  function setLog(kind, value) {
    logBuffers[kind] = value;
    const log = document.getElementById(kind === "workflow" ? "workflow-log" : "checkml-log");
    const atBottom = log.scrollHeight - log.scrollTop - log.clientHeight < 30;
    log.textContent = value || "No output yet.";
    if (atBottom) log.scrollTop = log.scrollHeight;
  }

  function applyLogChange(payload) {
    const update = JSON.parse(payload);
    if (!(update.kind in logBuffers)) return;
    if (update.replace) {
      setLog(update.kind, update.text);
    } else if (logBuffers[update.kind].length === update.offset) {
      const log = document.getElementById(update.kind === "workflow" ? "workflow-log" : "checkml-log");
      const atBottom = log.scrollHeight - log.scrollTop - log.clientHeight < 30;
      if (!logBuffers[update.kind]) log.textContent = "";
      logBuffers[update.kind] += update.text;
      log.appendChild(document.createTextNode(update.text));
      if (atBottom) log.scrollTop = log.scrollHeight;
    } else if (bridge) {
      bridge.getLogs(snapshot => {
        const logs = JSON.parse(snapshot);
        setLog("workflow", logs.workflow);
        setLog("checkml", logs.checkml);
      });
    }
  }

  function showChemdrawEditor(payload) {
    chemdrawRows = JSON.parse(payload).rows;
    document.getElementById("chemdraw-target-name").value = "target";
    document.querySelectorAll("[data-chemdraw-column]").forEach(input => { input.checked = false; });
    document.getElementById("chemdraw-message").textContent = `${chemdrawRows.length} molecules imported. Complete the name and target columns before saving.`;
    document.getElementById("chemdraw-editor").hidden = false;
    renderChemdrawTable();
    showPage("aqme");
    document.getElementById("chemdraw-editor").scrollIntoView({ behavior: "smooth", block: "start" });
  }

  function saveChemdrawEditor() {
    if (!bridge || !chemdrawRows.length) return;
    const columns = chemdrawOptionalColumns.filter(name => document.querySelector(`[data-chemdraw-column="${name}"]`).checked);
    const payload = JSON.stringify({ rows: chemdrawRows, columns,
      targetColumn: document.getElementById("chemdraw-target-name").value });
    bridge.saveChemdrawCsv(payload, response => {
      const result = JSON.parse(response);
      const message = document.getElementById("chemdraw-message");
      if (result.error) message.textContent = result.error;
      else if (result.saved) message.textContent = `Saved ${result.saved}`;
      else message.textContent = "Save cancelled.";
    });
  }

  function showPage(name) {
    if (checkmlMode && !["checkml", "results", "bot", "tutorial"].includes(name)) return;
    if (name === "bot" || name === "tutorial") {
      openDrawer(name);
      return;
    }
    if (name === "aqme" && document.querySelector('.nav-item[data-page="aqme"]').disabled) return;
    document.querySelectorAll(".page").forEach(page => page.classList.toggle("active", page.id === `page-${name}`));
    document.querySelectorAll(".nav-item[data-page]").forEach(button => button.classList.toggle("active", button.dataset.page === name));
    document.querySelector(".main").scrollTop = 0;
    if (bridge) bridge.performAction("activePage", name);
  }

  function showResult(name) {
    showPage("results");
    window.easyrobResults.open(name);
    if (bridge) bridge.performAction("activeResultView", name);
    const gallery = document.getElementById("embedded-gallery");
    gallery.hidden = name !== "Images";
    if (name === "Images") gallery.scrollIntoView({ behavior: "smooth", block: "start" });
  }

  function openDrawer(name) {
    if (checkmlMode && name === "bot") return;
    const previousTutorialOpen = !document.getElementById("tutorial-drawer").hidden;
    const previousBotOpen = !document.getElementById("bot-drawer").hidden;
    document.body.classList.toggle("drawer-open", name === "bot");
    document.getElementById("tutorial-backdrop").hidden = name !== "tutorial";
    for (const element of document.querySelectorAll(".sidebar, .main, #bot-drawer")) {
      element.inert = name === "tutorial";
    }
    for (const drawerName of ["bot", "tutorial"]) {
      const open = drawerName === name;
      document.getElementById(`${drawerName}-drawer`).hidden = !open;
      const toggle = document.querySelector(`.sidebar [data-${drawerName}-toggle]`);
      toggle.classList.toggle("active", open);
      toggle.setAttribute("aria-expanded", String(open));
    }
    if (name === "tutorial" && !previousTutorialOpen) {
      document.querySelector("#tutorial-drawer .drawer-close").focus({ preventScroll: true });
    } else if (name === "bot" && !previousBotOpen) {
      document.getElementById("bot-draft").focus({ preventScroll: true });
    } else if (!name && previousTutorialOpen) {
      document.querySelector(".sidebar [data-tutorial-toggle]").focus({ preventScroll: true });
    } else if (!name && previousBotOpen) {
      document.querySelector(".sidebar [data-bot-toggle]").focus({ preventScroll: true });
    }
    if (name === "bot" && bridge) bridge.performAction("botOpen", "");
    if (name === "tutorial" && bridge && !tutorials.length) {
      bridge.getTutorials(payload => {
        tutorials = JSON.parse(payload);
        renderTutorial();
      });
    }
  }

  function toggleDrawer(name) {
    openDrawer(document.getElementById(`${name}-drawer`).hidden ? name : null);
  }

  function renderTutorial() {
    if (!tutorials.length) return;
    const topics = document.getElementById("tutorial-topics");
    if (topics.children.length !== tutorials.length) {
      topics.replaceChildren();
      tutorials.forEach((topic, index) => {
        const button = document.createElement("button");
        button.type = "button";
        button.className = "tutorial-topic";
        button.dataset.tutorial = String(index);
        button.textContent = topic.title;
        topics.appendChild(button);
      });
    }
    [...topics.children].forEach((button, index) => {
      button.classList.toggle("active", index === tutorialTopic);
      if (index === tutorialTopic) button.setAttribute("aria-current", "true");
      else button.removeAttribute("aria-current");
    });
    const topic = tutorials[tutorialTopic];
    const step = topic.steps[tutorialStep];
    document.getElementById("tutorial-title").textContent = topic.title;
    document.getElementById("tutorial-count").textContent = `${topic.steps.length} STEPS`;
    document.getElementById("tutorial-position").textContent = `${tutorialStep + 1} / ${topic.steps.length}`;
    document.getElementById("tutorial-image").src = step.image;
    document.getElementById("tutorial-image").alt = `${topic.title}, step ${tutorialStep + 1}`;
    document.getElementById("tutorial-copy").innerHTML = step.html;
    document.querySelector(".tutorial-step").scrollTop = 0;
    document.getElementById("tutorial-previous").disabled = tutorialStep === 0;
    document.getElementById("tutorial-next").disabled = tutorialStep === topic.steps.length - 1;
  }

  function updateSelect(select, options, current) {
    const signature = JSON.stringify(options);
    if (select.dataset.snapshot !== signature) {
      select.dataset.snapshot = signature;
      select.replaceChildren();
      for (const option of options) {
        const item = document.createElement("option");
        item.value = option.value;
        item.textContent = option.label;
        select.appendChild(item);
      }
    }
    select.value = current;
  }

  function renderBot(bot) {
    if ((lastBotProvider !== null && lastBotProvider !== bot.provider)
        || (lastBotApiSaved !== null && lastBotApiSaved !== bot.apiSaved)) {
      document.getElementById("bot-api-key").value = "";
    }
    lastBotProvider = bot.provider;
    lastBotApiSaved = bot.apiSaved;
    updateSelect(document.getElementById("bot-mode"), bot.modes.map(value => ({ value, label: value })), bot.mode);
    updateSelect(document.getElementById("bot-provider"), bot.providers.map(value => ({ value, label: value })), bot.provider);
    updateSelect(document.getElementById("bot-suggestions"), [{ value: "", label: "Choose a suggested question..." },
      ...bot.suggestions.map(item => ({ value: String(item.index), label: item.label }))], "");
    document.getElementById("bot-cloud-settings").hidden = bot.mode !== "Cloud AI";
    document.getElementById("bot-web-search").checked = bot.webSearch;
    document.getElementById("bot-api-status").textContent = bot.apiSaved ? "API key saved locally" : "No saved API key";
    document.getElementById("bot-delete-api").disabled = !bot.apiSaved || bot.busy;
    document.getElementById("bot-send").disabled = !bot.canSend;
    document.getElementById("bot-draft").disabled = bot.busy;
    if (document.activeElement !== document.getElementById("bot-draft")) document.getElementById("bot-draft").value = bot.draft;
    if (focusBotDraftAfterSend && !bot.busy && !document.getElementById("bot-drawer").hidden) {
      document.getElementById("bot-draft").focus({ preventScroll: true });
      focusBotDraftAfterSend = false;
    }
    document.getElementById("bot-last-usage").textContent = bot.lastUsage;
    document.getElementById("bot-session-usage").textContent = bot.sessionUsage;
    document.getElementById("bot-request-stage").textContent = bot.requestStage || "Ask about your workflow, results, or AQME and ROBERT.";
    document.getElementById("bot-summarize").disabled = !bot.canSummarize || bot.busy;
    const history = document.getElementById("bot-messages");
    const signature = JSON.stringify(bot.messages);
    if (history.dataset.snapshot !== signature) {
      history.dataset.snapshot = signature;
      history.replaceChildren();
      for (const message of bot.messages) {
        const bubble = document.createElement("article");
        bubble.className = `bot-message ${message.role.toLowerCase() === "user" ? "user" : "assistant"}`;
        const label = document.createElement("strong");
        label.textContent = message.role;
        const content = document.createElement("div");
        content.className = "bot-message-content";
        if (message.html) content.innerHTML = message.html;
        else content.textContent = message.content;
        bubble.append(label, content);
        history.appendChild(bubble);
      }
      history.scrollTop = history.scrollHeight;
    }
  }

  function displayPath(path) {
    if (!path) return "No file selected";
    return path.replaceAll("\\", "/").split("/").at(-1);
  }

  function renderFields(group, fields) {
    const container = document.getElementById(`${group}-fields`);
    if (!container) return;
    const signature = JSON.stringify(fields);
    if (fieldSnapshots.get(group) === signature) return;
    fieldSnapshots.set(group, signature);
    const active = document.activeElement;
    const draft = container.contains(active) && active.matches('input[type="text"][data-field]')
      ? { key: active.dataset.field, value: active.value,
        start: active.selectionStart, end: active.selectionEnd } : null;
    container.replaceChildren();
    for (const field of fields) {
      const wrapper = document.createElement("div");
      wrapper.className = `field ${field.type === "toggle" ? "toggle-field" : ""}`;
      const label = document.createElement("label");
      label.textContent = field.label;
      const input = document.createElement(field.type === "select" ? "select" : "input");
      input.dataset.field = field.key;
      input.disabled = !field.enabled;
      if (field.type === "select") {
        for (const option of field.options) {
          const item = document.createElement("option");
          item.value = option;
          item.textContent = option;
          input.appendChild(item);
        }
        input.value = field.value;
      } else if (field.type === "toggle") {
        input.type = "checkbox";
        input.checked = field.value;
      } else {
        input.type = "text";
        input.value = field.value;
        input.placeholder = field.placeholder || "Default";
      }
      const inputId = `field-${field.key.replaceAll(".", "-")}`;
      input.id = inputId;
      label.htmlFor = inputId;
      wrapper.appendChild(label);
      wrapper.appendChild(input);
      container.appendChild(wrapper);
    }
    if (draft) {
      const restored = [...container.querySelectorAll('input[type="text"][data-field]')]
        .find(input => input.dataset.field === draft.key && !input.disabled);
      if (restored) {
        restored.value = draft.value;
        restored.focus({ preventScroll: true });
        restored.setSelectionRange(draft.start, draft.end);
        if (bridge && fields.find(field => field.key === draft.key)?.value !== draft.value) {
          bridge.setField(draft.key, draft.value);
        }
      }
    }
  }

  function renderColumns(containerId, values, action) {
    const container = document.getElementById(containerId);
    const signature = JSON.stringify(values);
    if (container.dataset.snapshot === signature) return;
    container.dataset.snapshot = signature;
    container.replaceChildren();
    if (!values.length) {
      const empty = document.createElement("span");
      empty.className = "column-placeholder";
      empty.textContent = "No columns";
      container.appendChild(empty);
      return;
    }
    for (const name of values) {
      const button = document.createElement("button");
      button.className = `column-chip ${action === "restore" ? "restore" : ""}`;
      button.type = "button";
      button.dataset.action = action;
      button.dataset.value = name;
      button.title = action === "ignore" ? "Exclude this column" : "Use this column";
      button.textContent = action === "ignore" ? `${name}  →` : `←  ${name}`;
      container.appendChild(button);
    }
  }

  function renderProgress(states, aqmeState) {
    const list = document.getElementById("workflow-progress");
    const signature = JSON.stringify([states, aqmeState]);
    if (list.dataset.snapshot === signature) return;
    list.dataset.snapshot = signature;
    list.replaceChildren();
    const items = stages.map((name, index) => ({ name, state: states[index] || "idle" }));
    if (aqmeState !== "inactive") items.unshift({ name: "AQME", state: aqmeState });
    const captions = { done: "Complete", active: "Running", pending: "Up next", inactive: "Not selected",
      failed: "Failed", stopped: "Stopped", idle: "Not started" };
    const descriptions = { AQME: "Calculate molecular descriptors", CURATE: "Prepare the dataset",
      GENERATE: "Train candidate models", VERIFY: "Validate model quality",
      PREDICT: "Generate predictions", REPORT: "Assemble the report" };
    const selected = items.filter(item => item.state !== "inactive");
    const completed = selected.filter(item => item.state === "done").length;
    const total = selected.length;
    const percentage = total ? Math.round(completed * 100 / total) : 0;
    document.getElementById("workflow-completion").textContent = `${completed} of ${total} complete`;
    document.getElementById("workflow-percent").textContent = `${percentage}%`;
    document.getElementById("workflow-meter-fill").style.width = `${percentage}%`;
    document.querySelector(".progress-meter").setAttribute("aria-valuenow", String(percentage));
    const overall = selected.some(item => item.state === "failed") ? "Failed"
      : selected.some(item => item.state === "stopped") ? "Stopped"
      : selected.some(item => item.state === "active") ? "Running"
      : total && completed === total ? "Complete" : "Ready";
    const badge = document.getElementById("workflow-state-badge");
    badge.textContent = overall;
    badge.dataset.state = overall.toLowerCase();
    items.forEach((item, index) => {
      const row = document.createElement("div");
      row.className = `progress-row status-${item.state}`;
      row.setAttribute("role", "listitem");
      row.setAttribute("aria-label", `${item.name}: ${captions[item.state] || "Pending"}`);
      const marker = document.createElement("span");
      marker.className = "progress-marker";
      marker.textContent = item.state === "done" ? "✓" : item.state === "failed" ? "!"
        : item.state === "stopped" ? "■" : String(index + 1).padStart(2, "0");
      const copy = document.createElement("span");
      copy.className = "progress-copy";
      const name = document.createElement("strong");
      name.textContent = item.name;
      const description = document.createElement("small");
      description.textContent = descriptions[item.name] || "Processing stage";
      copy.append(name, description);
      const status = document.createElement("span");
      status.className = "progress-status";
      status.textContent = captions[item.state] || "Pending";
      row.append(marker, copy, status);
      list.appendChild(row);
    });
  }

  function renderCheckmlColumns(columns) {
    const container = document.getElementById("checkml-columns");
    const signature = JSON.stringify(columns);
    if (container.dataset.snapshot === signature) return;
    container.dataset.snapshot = signature;
    container.replaceChildren();
    if (!columns.length) {
      const empty = document.createElement("span");
      empty.className = "column-placeholder";
      empty.textContent = "Choose an input CSV first";
      container.appendChild(empty);
    }
    for (const column of columns) {
      const button = document.createElement("button");
      button.type = "button";
      button.className = `column-chip ${column.ignored ? "restore" : ""}`;
      button.dataset.action = "toggleCheckmlIgnore";
      button.dataset.value = column.name;
      button.textContent = `${column.ignored ? "✓ " : "○ "}${column.name}`;
      container.appendChild(button);
    }
  }

  function renderResults(results) {
    window.easyrobResults.render(results);
    const gallery = document.getElementById("embedded-gallery");
    if (!results.available.Images && !gallery.hidden) {
      gallery.hidden = true;
      if (bridge) bridge.performAction("activeResultView", "");
    }
    const toolbar = document.getElementById("result-model-toolbar");
    const selector = document.getElementById("result-model");
    toolbar.hidden = results.models.length < 2;
    const modelSignature = JSON.stringify(results.models);
    if (selector.dataset.snapshot !== modelSignature) {
      selector.dataset.snapshot = modelSignature;
      selector.replaceChildren();
      for (const model of results.models) {
        const option = document.createElement("option");
        option.value = model.value;
        option.textContent = model.label;
        selector.appendChild(option);
      }
    }
    selector.value = results.selectedModel;
    for (const card of document.querySelectorAll(".result-card")) {
      const name = card.dataset.value;
      const available = Boolean(results.available[name]);
      card.disabled = !available;
      const list = card.querySelector(".result-files");
      const files = (results.files[name] || []).slice(0, 3);
      const signature = JSON.stringify(files);
      if (list.dataset.snapshot !== signature) {
        list.dataset.snapshot = signature;
        list.replaceChildren();
        for (const filename of files) {
          const item = document.createElement("li");
          item.textContent = filename;
          list.appendChild(item);
        }
      }
      card.querySelector("em").textContent = available ? (name === "Images" ? "Open gallery ↓" : "Explore here ↓") : "No output yet";
    }
    renderGallery(results.gallery);
  }

  function renderGallery(nextState) {
    const signature = JSON.stringify(nextState);
    if (document.getElementById("gallery-grid").dataset.snapshot === `${galleryStage}:${signature}`) return;
    galleryState = nextState;
    if (galleryStage !== "all" && !galleryState.stages.some(stage => stage.id === galleryStage)) galleryStage = "all";
    const filters = document.getElementById("gallery-filters");
    const grid = document.getElementById("gallery-grid");
    filters.replaceChildren();
    grid.replaceChildren();
    const choices = [{ id: "all", label: "All images", count: galleryState.total },
      ...galleryState.stages.map(stage => ({ id: stage.id, label: stage.label, count: stage.images.length }))];
    for (const choice of choices) {
      const button = document.createElement("button");
      button.type = "button";
      button.className = "gallery-filter";
      button.dataset.stage = choice.id;
      button.setAttribute("aria-pressed", String(choice.id === galleryStage));
      button.textContent = `${choice.label}  ${choice.count}`;
      filters.appendChild(button);
    }
    for (const stage of galleryState.stages) {
      if (galleryStage !== "all" && galleryStage !== stage.id) continue;
      for (const image of stage.images) {
        const card = document.createElement("article");
        card.className = "gallery-card";
        const preview = document.createElement("img");
        preview.src = image.url;
        preview.alt = image.name;
        preview.loading = "lazy";
        const body = document.createElement("div");
        body.className = "gallery-card-body";
        const heading = document.createElement("strong");
        heading.textContent = image.name;
        const stageLabel = document.createElement("small");
        stageLabel.textContent = stage.label;
        const actions = document.createElement("div");
        actions.className = "gallery-actions";
        for (const [action, label] of [["open", "Open"], ["copy", "Copy"], ["save", "Save as"], ["folder", "Folder"]]) {
          const button = document.createElement("button");
          button.type = "button";
          button.className = "text-button";
          button.dataset.action = "imageAction";
          button.dataset.value = `${image.id}|${action}`;
          button.textContent = label;
          actions.appendChild(button);
        }
        body.append(stageLabel, heading, actions);
        card.append(preview, body);
        grid.appendChild(card);
      }
    }
    grid.dataset.snapshot = `${galleryStage}:${signature}`;
  }

  function renderMolecule(molecule) {
    const image = document.getElementById("molecule-image");
    const message = document.getElementById("molecule-message");
    if (image.dataset.snapshot !== molecule.image) {
      image.dataset.snapshot = molecule.image;
      image.src = molecule.image || "";
    }
    image.hidden = !molecule.image;
    image.classList.toggle("selectable", molecule.canSelect);
    message.hidden = Boolean(molecule.image);
    message.textContent = molecule.message || "Choose a CSV with a SMILES column to inspect its common pattern.";
    document.getElementById("molecule-info").textContent = molecule.info;
  }

  function render(state) {
    for (const [group, fields] of Object.entries(state.groups)) renderFields(group, fields);
    for (const [id, path] of Object.entries({
      "train-file": state.files.train,
      "test-file": state.files.test,
      "checkml-train-file": state.files.checkmlTrain,
      "checkml-test-file": state.files.checkmlTest,
      "checkml-params-file": state.files.checkmlParams,
      "checkml-model-file": state.files.checkmlModel,
    })) {
      const label = document.getElementById(id);
      label.textContent = path ? displayPath(path) : id.includes("params") || id.includes("model-file") ? "" : "No file selected";
      label.title = path || "";
    }
    document.querySelector('[data-action="clearTest"]').hidden = !state.files.test;
    document.getElementById("clear-checkml-test").hidden = !state.files.checkmlTest;
    document.getElementById("checkml-params-picker").hidden = state.checkmlSource !== 2;
    document.getElementById("checkml-model-picker").hidden = state.checkmlSource !== 3;
    document.getElementById("available-count").textContent = state.columns.available.length;
    document.getElementById("ignored-count").textContent = state.columns.ignored.length;
    renderColumns("available-columns", state.columns.available, "ignore");
    renderColumns("ignored-columns", state.columns.ignored, "restore");
    renderCheckmlColumns(state.checkmlColumns);
    renderMolecule(state.molecule);
    renderBot(state.bot);
    renderProgress(state.progressStates || [], state.aqmeProgress || "inactive");
    const progressBadge = document.getElementById("workflow-state-badge").textContent;
    const activeStage = document.querySelector(".progress-row.status-active .progress-copy strong")?.textContent;
    document.getElementById("workflow-activity").textContent = state.activity || (activeStage
      ? `${activeStage} is running` : progressBadge === "Complete" ? "Workflow complete"
        : progressBadge === "Failed" ? "Workflow stopped with an error"
          : progressBadge === "Stopped" ? "Workflow stopped" : "Waiting for a run");
    if ("log" in state) setLog("workflow", state.log);
    if ("checkmlLog" in state) setLog("checkml", state.checkmlLog);
    document.getElementById("run-button").disabled = !state.actions.run;
    document.getElementById("aqme-run-button").disabled = !state.actions.runAqme;
    document.getElementById("stop-button").disabled = !state.actions.stop;
    document.getElementById("checkml-run-button").disabled = !state.actions.checkmlRun;
    document.getElementById("checkml-stop-button").disabled = !state.actions.checkmlStop;
    const aqmeNavigation = document.querySelector('.nav-item[data-page="aqme"]');
    aqmeNavigation.disabled = !state.actions.aqme;
    aqmeNavigation.title = state.actions.aqme ? "Configure AQME descriptors" : "Enable Calculate descriptors with AQME in Workflow to configure descriptors";
    if (!state.actions.aqme && document.getElementById("page-aqme").classList.contains("active")) showPage("workflow");
    document.getElementById("results-empty").hidden = state.actions.results;
    renderResults(state.results);
  }

  document.addEventListener("click", event => {
    const botLink = event.target.closest("#bot-messages a[href]");
    if (botLink && bridge) {
      event.preventDefault();
      bridge.performAction("openBotLink", botLink.href);
      return;
    }
    if (bridge && event.target.closest("[data-close-result]")) {
      bridge.performAction("activeResultView", "");
    }
    const consoleToggle = event.target.closest("[data-console-toggle]");
    if (consoleToggle) {
      const log = document.getElementById(consoleToggle.getAttribute("aria-controls"));
      log.hidden = !log.hidden;
      consoleToggle.setAttribute("aria-expanded", String(!log.hidden));
      if (!log.hidden) log.scrollTop = log.scrollHeight;
      return;
    }
    if (event.target.id === "chemdraw-close") {
      document.getElementById("chemdraw-editor").hidden = true;
      return;
    }
    if (event.target.id === "chemdraw-save") {
      saveChemdrawEditor();
      return;
    }
    if (event.target.closest("[data-bot-toggle]")) {
      toggleDrawer("bot");
      return;
    }
    if (event.target.closest("[data-tutorial-toggle]")) {
      toggleDrawer("tutorial");
      return;
    }
    if (event.target.id === "tutorial-backdrop") {
      openDrawer(null);
      return;
    }
    const topic = event.target.closest("[data-tutorial]");
    if (topic) {
      tutorialTopic = Number(topic.dataset.tutorial);
      tutorialStep = 0;
      renderTutorial();
      return;
    }
    if (event.target.id === "tutorial-previous" && tutorialStep > 0) {
      tutorialStep -= 1;
      renderTutorial();
      return;
    }
    if (event.target.id === "tutorial-next" && tutorialStep < tutorials[tutorialTopic].steps.length - 1) {
      tutorialStep += 1;
      renderTutorial();
      return;
    }
    const filter = event.target.closest("[data-stage]");
    if (filter) {
      galleryStage = filter.dataset.stage;
      document.getElementById("gallery-grid").dataset.snapshot = "";
      renderGallery(galleryState);
      return;
    }
    if (event.target.id === "close-gallery") {
      document.getElementById("embedded-gallery").hidden = true;
      if (bridge) bridge.performAction("activeResultView", "");
      return;
    }
    if (event.target.id === "molecule-image" && event.target.classList.contains("selectable") && bridge) {
      const bounds = event.target.getBoundingClientRect();
      const x = Math.floor((event.clientX - bounds.left) * 400 / bounds.width);
      const y = Math.floor((event.clientY - bounds.top) * 400 / bounds.height);
      bridge.performAction("selectAtom", `${x},${y}`);
      return;
    }
    const navigation = event.target.closest("[data-page]");
    if (navigation) {
      showPage(navigation.dataset.page);
      return;
    }
    const action = event.target.closest("[data-action]");
    if (action && bridge && !action.disabled) {
      if (action.dataset.action === "openResult") {
        showResult(action.dataset.value);
        return;
      }
      if (action.dataset.action === "botSend") focusBotDraftAfterSend = true;
      bridge.performAction(action.dataset.action, action.dataset.value || "");
    }
  });

  document.addEventListener("change", event => {
    if (event.target.matches("[data-chemdraw-column]")) {
      renderChemdrawTable();
      return;
    }
    if (event.target.matches("[data-chemdraw-row]")) {
      chemdrawRows[Number(event.target.dataset.chemdrawRow)][event.target.dataset.chemdrawField] = event.target.value;
      return;
    }
    if (bridge && event.target.id === "bot-mode") return bridge.performAction("botMode", event.target.value);
    if (bridge && event.target.id === "bot-provider") {
      document.getElementById("bot-api-key").value = "";
      return bridge.performAction("botProvider", event.target.value);
    }
    if (bridge && event.target.id === "bot-web-search") return bridge.performAction("botWebSearch", String(event.target.checked));
    if (bridge && event.target.id === "bot-suggestions" && event.target.value) return bridge.performAction("botSuggestion", event.target.value);
    if (event.target.id === "result-model" && bridge) {
      bridge.performAction("setResultModel", event.target.value);
      return;
    }
    const input = event.target.closest("[data-field]");
    if (input && bridge) bridge.setField(input.dataset.field, input.type === "checkbox" ? String(input.checked) : input.value);
  });

  document.addEventListener("input", event => {
    if (event.target.id === "chemdraw-target-name") {
      const targetHeader = document.querySelector("#chemdraw-head th:nth-child(4)");
      if (targetHeader) targetHeader.textContent = event.target.value || "target";
    }
    if (event.target.matches("[data-chemdraw-row]")) {
      chemdrawRows[Number(event.target.dataset.chemdrawRow)][event.target.dataset.chemdrawField] = event.target.value;
    }
    if (!bridge) return;
    if (event.target.id === "bot-draft") bridge.setBotDraft(event.target.value);
    if (event.target.id === "bot-api-key") bridge.setBotApiKey(event.target.value);
  });

  document.getElementById("bot-draft").addEventListener("keydown", event => {
    if (event.key === "Enter" && bridge) {
      event.preventDefault();
      focusBotDraftAfterSend = true;
      bridge.performAction("botSend", "");
    }
  });

  document.addEventListener("keydown", event => {
    const tutorial = document.getElementById("tutorial-drawer");
    if (event.key === "Escape" && (!tutorial.hidden || !document.getElementById("bot-drawer").hidden)) {
      openDrawer(null);
      return;
    }
    if (event.key === "Tab" && !tutorial.hidden) {
      const focusable = [...tutorial.querySelectorAll("button:not(:disabled), a[href]")];
      if (!focusable.length) return;
      const first = focusable[0];
      const last = focusable[focusable.length - 1];
      if (event.shiftKey && document.activeElement === first) {
        event.preventDefault();
        last.focus();
      } else if (!event.shiftKey && document.activeElement === last) {
        event.preventDefault();
        first.focus();
      }
    }
  });

  window.easyrobApp = { bridgeReady: false, showPage, showResult, openDrawer };

  if (checkmlMode) {
    document.body.classList.add("checkml-mode");
    document.title = "CheckML";
    document.querySelector(".brand img").src = "checkml_brand.png";
    document.querySelector(".brand img").alt = "CheckML";
    document.querySelector(".topbar-version").textContent = "CheckML · Desktop";
    document.querySelector(".tutorial-drawer .drawer-heading h2").innerHTML = "CheckML tutorial<span class=\"dot\">.</span>";
    document.getElementById("bot-request-stage").textContent = "Ask about CheckML and its results.";
    document.getElementById("bot-draft").placeholder = "Ask about CheckML and its results";
    showPage("checkml");
  }

  if (window.qt && window.qt.webChannelTransport && window.QWebChannel) {
    new QWebChannel(window.qt.webChannelTransport, channel => {
      bridge = channel.objects.workflow;
      window.easyrobResults.connect(bridge);
      window.easyrobApp.bridgeReady = true;
      bridge.stateChanged.connect(payload => render(JSON.parse(payload)));
      bridge.logChanged.connect(applyLogChange);
      bridge.chemdrawReady.connect(showChemdrawEditor);
      bridge.getState(payload => render(JSON.parse(payload)));
    });
  }
})();
