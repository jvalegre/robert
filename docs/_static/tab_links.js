// Direct links to sphinx-design tabs.
// Every tab gets a slug from its label text (e.g. "6. Prediction stability" ->
// "#6-prediction-stability"). Opening a page with that hash selects the tab (and every
// parent tab it is nested in) and scrolls to it. Clicking a tab writes its slug to the URL,
// so the address bar always holds a link that can be copied and shared.
(function () {
  'use strict';

  function slugify(text) {
    return text
      .normalize('NFKD').replace(/[̀-ͯ]/g, '')
      .toLowerCase()
      .replace(/[^a-z0-9]+/g, '-')
      .replace(/^-+|-+$/g, '');
  }

  var tabsBySlug = {};

  function registerTabs() {
    var used = {};
    document.querySelectorAll('.sd-tab-set > label.sd-tab-label').forEach(function (label) {
      var input = document.getElementById(label.getAttribute('for'));
      if (!input) { return; }
      var base = slugify(label.textContent) || 'tab';
      var slug = base;
      var n = 2;
      while (used[slug] || document.getElementById(slug)) { slug = base + '-' + n++; }
      used[slug] = true;
      label.dataset.tabSlug = slug;
      tabsBySlug[slug] = { input: input, label: label };
      input.addEventListener('change', function () {
        if (input.checked && window.history && history.replaceState) {
          history.replaceState(null, '', '#' + slug);
        }
      });
    });
  }

  // select the tab and all the tabs that contain it, outermost first
  function openTab(slug) {
    var tab = tabsBySlug[slug];
    if (!tab) { return false; }
    var chain = [];
    var node = tab.label;
    while (node) {
      var set = node.closest ? node.closest('.sd-tab-set') : null;
      if (!set) { break; }
      // the tab content that holds this tab-set belongs to the label right before it
      var content = set.parentElement ? set.parentElement.closest('.sd-tab-content') : null;
      if (node === tab.label) { chain.push(tab.input); }
      if (!content) { break; }
      var parentLabel = content.previousElementSibling;
      if (parentLabel && parentLabel.classList.contains('sd-tab-label')) {
        var parentInput = document.getElementById(parentLabel.getAttribute('for'));
        if (parentInput) { chain.push(parentInput); }
        node = parentLabel;
      } else {
        break;
      }
    }
    chain.reverse().forEach(function (input) { input.checked = true; });
    // scroll to the start of the tab row, so the sibling tabs stay visible too
    tab.label.closest('.sd-tab-set').scrollIntoView({ block: 'start' });
    return true;
  }

  function openFromHash() {
    var hash = decodeURIComponent(window.location.hash.slice(1));
    if (hash) { openTab(hash); }
  }

  document.addEventListener('DOMContentLoaded', function () {
    registerTabs();
    openFromHash();
  });
  window.addEventListener('hashchange', openFromHash);
})();
