"use strict";

(() => {
  const filters = document.getElementById("stage-filters");
  const gallery = document.getElementById("gallery");
  const imageCount = document.getElementById("image-count");
  const stageCount = document.getElementById("stage-count");
  let state = { stages: [], total: 0 };
  let selectedStage = "all";
  let bridge = null;

  const make = (tag, className, text) => {
    const element = document.createElement(tag);
    element.className = className;
    if (text !== undefined) element.textContent = text;
    return element;
  };

  function renderFilters() {
    filters.replaceChildren();
    const options = [{ id: "all", label: "All images", count: state.total },
      ...state.stages.map(stage => ({ id: stage.id, label: stage.label, count: stage.images.length }))];
    for (const option of options) {
      const button = make("button", "filter", option.label);
      button.type = "button";
      button.setAttribute("aria-pressed", String(selectedStage === option.id));
      button.appendChild(make("span", "filter-count", String(option.count)));
      button.addEventListener("click", () => {
        selectedStage = option.id;
        renderFilters();
        renderGallery();
      });
      filters.appendChild(button);
    }
  }

  function cardFor(image, stage) {
    const card = make("article", "image-card");
    const preview = make("div", "preview");
    const picture = make("img", "");
    picture.src = image.url;
    picture.alt = image.name;
    picture.loading = "lazy";
    preview.appendChild(picture);
    card.appendChild(preview);

    const body = make("div", "card-body");
    body.appendChild(make("p", "card-stage", stage.label));
    body.appendChild(make("h2", "", image.name));
    const actions = make("div", "actions");
    for (const [name, label] of [["open", "Open"], ["copy", "Copy"], ["save", "Save as"], ["folder", "Folder"]]) {
      const button = make("button", `action ${name === "open" ? "primary" : ""}`, label);
      button.type = "button";
      button.dataset.action = name;
      button.addEventListener("click", () => {
        if (bridge) bridge.performAction(image.id, name);
      });
      actions.appendChild(button);
    }
    body.appendChild(actions);
    card.appendChild(body);
    return card;
  }

  function renderGallery() {
    gallery.replaceChildren();
    const stages = selectedStage === "all" ? state.stages : state.stages.filter(stage => stage.id === selectedStage);
    for (const stage of stages) {
      for (const image of stage.images) gallery.appendChild(cardFor(image, stage));
    }
    if (!gallery.children.length) {
      const empty = make("div", "empty");
      empty.appendChild(make("span", "empty-icon", "◌"));
      empty.appendChild(make("strong", "", "No images in this view"));
      empty.appendChild(make("p", "", "Run a workflow to generate visual results."));
      gallery.appendChild(empty);
    }
  }

  window.easyrobGallery = {
    bridgeReady: false,
    setData(nextState) {
      state = nextState;
      if (selectedStage !== "all" && !state.stages.some(stage => stage.id === selectedStage)) {
        selectedStage = "all";
      }
      imageCount.textContent = String(state.total);
      stageCount.textContent = `${state.stages.length} ${state.stages.length === 1 ? "stage" : "stages"}`;
      renderFilters();
      renderGallery();
    }
  };

  window.easyrobGallery.setData(state);
  if (window.qt && window.qt.webChannelTransport && window.QWebChannel) {
    new QWebChannel(window.qt.webChannelTransport, channel => {
      bridge = channel.objects.imageActions;
      window.easyrobGallery.bridgeReady = true;
    });
  }
})();
