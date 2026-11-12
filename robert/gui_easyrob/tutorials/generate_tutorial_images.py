"""Capture the current web workspace and external workflow illustrations.

The example values are illustrative. This script never runs a workflow or
modifies scientific results. Run it from the repository's robert directory.
"""

import json
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("QTWEBENGINE_CHROMIUM_FLAGS", "--disable-gpu --no-sandbox")

from PySide6.QtCore import QTimer, QUrl
from PySide6.QtWidgets import QApplication
from PySide6.QtWebEngineWidgets import QWebEngineView


ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "tutorial_images"


@dataclass(frozen=True)
class Capture:
    page: str
    selector: str
    scene: str = ""


CAPTURES = {
    "overview": [
        Capture("workflow", ".page-heading"),
        Capture("workflow", '.nav-item[data-page="workflow"]'),
        Capture("workflow", "#page-workflow .file-grid"),
        Capture("workflow", "#workflow-fields"),
        Capture("workflow", "#page-workflow .column-grid"),
        Capture("workflow", '[data-field="workflow.aqme_workflow"]'),
        Capture("workflow", '[data-field="workflow.workflow_selector"]'),
        Capture("workflow", ".action-panel", "running"),
        Capture("workflow", ".progress-panel", "running"),
        Capture("aqme", ".molecule-panel"),
        Capture("advanced", ".option-grid"),
        Capture("workflow", '.nav-item[data-action="openMolssi"]'),
        Capture("checkml", "#page-checkml .page-heading"),
        Capture("results", "#page-results .result-grid", "results"),
        Capture("results", "#embedded-gallery", "gallery"),
        Capture("results", "#plot-view", "plot"),
        Capture("workflow", "#bot-drawer", "bot"),
    ],
    "csv": [
        Capture("workflow", "#page-workflow .file-grid"),
        Capture("workflow", "#tutorial-scene-focus", "spreadsheet-blank"),
        Capture("workflow", "#tutorial-scene-focus", "spreadsheet-columns"),
        Capture("workflow", "#tutorial-scene-focus", "smiles-copy"),
        Capture("workflow", "#tutorial-scene-focus", "spreadsheet-complete"),
        Capture("workflow", "#tutorial-scene-focus", "csv-save"),
        Capture("workflow", "#page-workflow .file-grid"),
        Capture("workflow", "#workflow-fields"),
        Capture("workflow", '[data-field="workflow.aqme_workflow"]'),
        Capture("aqme", "#molecule-canvas"),
        Capture("aqme", "#molecule-canvas"),
        Capture("workflow", "#run-button", "running"),
        Capture("workflow", ".progress-panel", "complete"),
        Capture("results", "#page-results .result-grid", "results"),
    ],
    "chemdraw": [
        Capture("aqme", '[data-action="openChemdraw"]'),
        Capture("workflow", '[data-field="workflow.aqme_workflow"]'),
        Capture("aqme", '[data-action="openChemdraw"]'),
        Capture("aqme", "#tutorial-scene-focus", "chemdraw-requirements"),
        Capture("aqme", "#tutorial-scene-focus", "chemdraw-file"),
        Capture("aqme", "#chemdraw-editor", "chemdraw"),
        Capture("aqme", "#chemdraw-save", "chemdraw"),
        Capture("aqme", "#chemdraw-message", "chemdraw"),
        Capture("workflow", "#train-file"),
        Capture("workflow", "#workflow-fields"),
        Capture("workflow", "#run-button", "running"),
        Capture("workflow", ".progress-panel", "running"),
        Capture("results", "#page-results .result-grid", "results"),
    ],
    "predictions": [
        Capture("workflow", "#page-workflow .file-grid"),
        Capture("workflow", "#tutorial-demo-data", "data"),
        Capture("workflow", "#page-workflow .file-grid"),
        Capture("workflow", '[data-field="workflow.workflow_selector"]'),
        Capture("workflow", ".progress-panel", "running"),
        Capture("workflow", ".action-panel", "running"),
        Capture("results", "#prediction-view", "prediction"),
    ],
    "descriptors": [
        Capture("aqme", "#aqme-fields"),
        Capture("workflow", "#page-workflow .file-grid"),
        Capture("workflow", "#aqme-run-button", "running"),
        Capture("workflow", ".progress-panel", "running"),
        Capture("workflow", ".progress-panel", "complete"),
        Capture("workflow", "#page-workflow .file-grid"),
    ],
    "check_model": [
        Capture("checkml", "#page-checkml .page-heading"),
        Capture("checkml", '.nav-item[data-page="checkml"]'),
        Capture("checkml", "#page-checkml .file-grid"),
        Capture("checkml", "#checkml-columns"),
        Capture("checkml", '#checkml-fields .field:last-child select'),
        Capture("checkml", "#checkml-params-picker", "checkml-model"),
        Capture("checkml", "#checkml-run-button"),
    ],
    "robbot": [
        Capture("workflow", '#bot-drawer', "bot"),
        Capture("workflow", '#bot-drawer', "bot"),
        Capture("workflow", '#bot-draft', "bot"),
        Capture("workflow", '#bot-suggestions', "bot"),
        Capture("workflow", '#bot-summarize', "bot"),
        Capture("workflow", '#bot-mode', "bot-settings"),
        Capture("workflow", '#bot-cloud-settings', "bot-settings"),
        Capture("workflow", '#bot-mode', "bot-heuristic"),
    ],
}


DEMO_SCRIPT = r"""
(() => {
  document.querySelector('.nav-item[data-page="aqme"]').disabled = false;
  const makeField = (container, key, label, options, selected) => {
    const wrapper = document.createElement('div');
    wrapper.className = 'field';
    const caption = document.createElement('label');
    caption.textContent = label;
    const select = document.createElement('select');
    select.dataset.field = key;
    for (const value of options) {
      const option = document.createElement('option');
      option.textContent = value;
      option.value = value;
      select.appendChild(option);
    }
    select.value = selected || options[0];
    wrapper.append(caption, select);
    container.appendChild(wrapper);
  };
  const workflow = document.getElementById('workflow-fields');
  makeField(workflow, 'workflow.y_dropdown', 'Target column', ['target', 'yield'], 'target');
  makeField(workflow, 'workflow.type_dropdown', 'Prediction type', ['Regression', 'Classification'], 'Regression');
  makeField(workflow, 'workflow.names_dropdown', 'Name column', ['code_name', 'SMILES'], 'code_name');
  makeField(workflow, 'workflow.workflow_selector', 'Workflow', ['Full Workflow', 'PREDICT', 'REPORT'], 'Full Workflow');
  makeField(workflow, 'workflow.aqme_workflow', 'Calculate descriptors with AQME', ['No', 'Yes'], 'Yes');
  makeField(workflow, 'workflow.all_models_toggle', 'Report every selected model', ['No', 'Yes'], 'No');
  document.getElementById('train-file').textContent = 'example_molecules.csv';
  document.getElementById('test-file').textContent = 'external_molecules.csv';
  document.getElementById('available-count').textContent = '3';
  document.getElementById('ignored-count').textContent = '1';
  for (const [id, labels] of [['available-columns', ['MolWt', 'LogP', 'TPSA']], ['ignored-columns', ['SMILES']]]) {
    const node = document.getElementById(id);
    for (const label of labels) {
      const item = document.createElement('button');
      item.type = 'button';
      item.className = 'column-chip';
      item.textContent = label;
      node.appendChild(item);
    }
  }
  for (const [id, label] of [['general-fields', 'Random seed'], ['curate-fields', 'Descriptor correlation filter'],
    ['generate-fields', 'Optimization metric'], ['predict-fields', 'SHAP plots'],
    ['aqme-fields', 'Descriptor level']]) {
    makeField(document.getElementById(id), id + '.example', label,
      id === 'aqme-fields' ? ['interpret', 'denovo', 'full'] : ['Default', 'Custom'], 'Default');
  }
  for (const [label, options] of [['Target column', ['target', 'yield']],
    ['Name column', ['code_name', 'SMILES']], ['Prediction type', ['Regression', 'Classification']],
    ['Model source', ['Default (MVL)', 'Scikit-learn model', 'Hyperparameters CSV', 'Saved model']]]) {
    makeField(document.getElementById('checkml-fields'), 'checkml.' + label, label, options);
  }
  document.getElementById('checkml-train-file').textContent = 'example_molecules.csv';
  document.getElementById('checkml-test-file').textContent = 'external_molecules.csv';
  const ignored = document.getElementById('checkml-columns');
  for (const [label, checked] of [['SMILES', true], ['notes', true], ['MolWt', false]]) {
    const chip = document.createElement('button');
    chip.className = 'column-chip' + (checked ? ' restore' : '');
    chip.textContent = (checked ? '✓ ' : '○ ') + label;
    ignored.appendChild(chip);
  }
  document.getElementById('molecule-message').textContent = 'A common molecular pattern appears here when the loaded structures share one.';
  document.getElementById('molecule-info').textContent = 'Choose atoms only when atom-based descriptors are needed.';
  const progress = document.getElementById('workflow-progress');
  for (const [index, stage] of ['CURATE', 'GENERATE', 'VERIFY', 'PREDICT', 'REPORT'].entries()) {
    const row = document.createElement('div');
    row.className = 'progress-row status-idle';
    row.innerHTML = '<span class="progress-marker">' + String(index + 1).padStart(2, '0') + '</span>' +
      '<span class="progress-copy"><strong>' + stage + '</strong><small>Workflow stage</small></span>' +
      '<span class="progress-status">Not started</span>';
    progress.appendChild(row);
  }
  const results = document.querySelectorAll('.result-card');
  for (const [index, card] of results.entries()) {
    card.querySelector('.result-files').innerHTML = '<li>' +
      ['ROBERT_report.pdf', 'external_predictions.csv', 'prediction_plot.png', 'model_predictions.csv'][index] + '</li>';
  }
  const gallery = document.getElementById('gallery-grid');
  const plotThumbnail = (title, offset) => {
    const points = [[52, 143], [75, 132], [103, 116], [132, 101], [155, 84], [183, 73], [210, 51]]
      .map(([x, y], index) => '<circle cx="' + x + '" cy="' + (y + (index % 2 ? offset : -offset)) +
        '" r="4" fill="' + (index < 5 ? '#0000ff' : '#ff0000') + '" stroke="#171717" stroke-width="1"/>').join('');
    const svg = '<svg xmlns="http://www.w3.org/2000/svg" width="260" height="165" viewBox="0 0 260 165">' +
      '<rect width="260" height="165" fill="white"/><text x="18" y="21" font-size="11" fill="#333">' + title + '</text>' +
      '<path d="M36 30V145H240" fill="none" stroke="#333"/><path d="M41 140L220 34" stroke="#777" stroke-dasharray="4 3"/>' +
      points + '</svg>';
    return 'data:image/svg+xml;charset=utf-8,' + encodeURIComponent(svg);
  };
  for (const [label, src] of [['Model plot', plotThumbnail('Predicted vs observed', 5)],
    ['CheckML plot', plotThumbnail('CheckML validation', 10)]]) {
    const card = document.createElement('article');
    card.className = 'gallery-card';
    card.innerHTML = '<img src="' + src + '" alt="Illustrative gallery item"><div class="gallery-card-body"><small>PREDICT</small><strong>' + label + '</strong></div>';
    gallery.appendChild(card);
  }
  document.getElementById('prediction-head').innerHTML = '<tr><th>code_name</th><th>target_pred</th><th>Set</th></tr>';
  document.getElementById('prediction-body').innerHTML = '<tr><td>molecule_01</td><td>3.42</td><td>External</td></tr><tr><td>molecule_02</td><td>4.15</td><td>External</td></tr>';
  document.getElementById('prediction-count').textContent = '2 of 2 rows';
  document.getElementById('prediction-columns').textContent = '3 columns';
  document.getElementById('prediction-identifier').textContent = 'Identifier: code_name';
  document.getElementById('prediction-scenario').innerHTML = '<div class="scenario-marker">i</div><div class="scenario-content"><div class="scenario-heading"><strong>Model assessment</strong></div><p>Illustrative prediction results.</p></div>';
  const svg = document.getElementById('plot-svg');
  const ns = 'http://www.w3.org/2000/svg';
  for (const [x, y, color] of [[125, 310, '#0000ff'], [205, 265, '#0000ff'], [305, 213, '#0000ff'],
    [410, 170, '#ff0000'], [520, 115, '#ff0000']]) {
    const point = document.createElementNS(ns, 'circle');
    for (const [name, value] of Object.entries({cx:x, cy:y, r:7, fill:color, stroke:'#000', 'stroke-width':1})) point.setAttribute(name, value);
    svg.appendChild(point);
  }
  const line = document.createElementNS(ns, 'line');
  for (const [name, value] of Object.entries({x1:70, y1:350, x2:605, y2:60, stroke:'#777', 'stroke-width':2})) line.setAttribute(name, value);
  svg.prepend(line);
  document.getElementById('plot-legend').innerHTML = '<button class="legend-button" style="--series-color:#0000ff">CV</button><button class="legend-button" style="--series-color:#ff0000">Test</button>';
  document.getElementById('plot-point-details').textContent = 'molecule_01\nGroup: CV\nObserved: 3.42\nPredicted: 3.38';
  document.getElementById('bot-messages').innerHTML = '<article class="bot-message assistant"><strong>robBOT</strong><p>Ask about your workflow, results, or AQME and ROBERT.</p></article>';
  document.getElementById('bot-draft').value = 'What does VERIFY check in my workflow?';
  document.getElementById('bot-suggestions').innerHTML = '<option>Choose a suggested question...</option><option>Explain my latest result</option>';
  document.getElementById('bot-mode').innerHTML = '<option>Cloud AI</option><option>Heuristic</option>';
  document.getElementById('bot-provider').innerHTML = '<option>Groq</option>';
  const data = document.createElement('section');
  data.id = 'tutorial-demo-data';
  data.className = 'panel';
  data.innerHTML = '<p class="eyebrow">ILLUSTRATIVE CSV EXAMPLE</p><h2>Prepare molecular data</h2><p class="muted">One molecule per row. Save the table as a CSV before loading it.</p>' +
    '<div class="prediction-scroll"><table class="prediction-table"><thead><tr><th>code_name</th><th>target</th><th>SMILES</th></tr></thead>' +
    '<tbody><tr><td>molecule_01</td><td>3.42</td><td>CCO</td></tr><tr><td>molecule_02</td><td>4.15</td><td>c1ccccc1</td></tr></tbody></table></div>';
  document.querySelector('#page-workflow .primary-column').prepend(data);
  data.hidden = true;
  const editor = document.getElementById('chemdraw-editor');
  document.getElementById('chemdraw-head').innerHTML = '<tr><th>Image</th><th>SMILES</th><th>code_name</th><th>target</th></tr>';
  document.getElementById('chemdraw-body').innerHTML = '<tr><td>⌬</td><td>CCO</td><td>molecule_01</td><td>3.42</td></tr><tr><td>⌬</td><td>c1ccccc1</td><td>molecule_02</td><td>4.15</td></tr>';
  document.getElementById('chemdraw-message').textContent = '2 molecules imported. Complete the name and target columns before saving.';
  document.getElementById('checkml-params-picker').hidden = false;
  document.getElementById('checkml-params-file').textContent = 'model_parameters.csv';
})();
"""


EXTERNAL_SCENE_SCRIPT = r"""
(() => {
  const style = document.createElement('style');
  style.textContent = `
    .tutorial-external-scene { position: fixed; z-index: 9000; top: 91px; left: 258px;
      width: 860px; height: 554px; box-sizing: border-box; border: 1px solid #d9d2e1;
      border-radius: 18px; overflow: hidden; background: #fff; box-shadow: 0 22px 60px rgba(23, 13, 37, .27);
      color: #2d213a; font: 15px 'Segoe UI', Arial, sans-serif; }
    .scene-titlebar { display: flex; align-items: center; justify-content: space-between; height: 58px;
      box-sizing: border-box; padding: 0 24px; background: #f6f1fa; border-bottom: 1px solid #e6dced; }
    .scene-titlebar strong { font-size: 18px; }
    .scene-titlebar span { color: #745e86; font-size: 12px; font-weight: 700; letter-spacing: .09em; }
    .scene-content { padding: 26px 30px; }
    .scene-content h2 { margin: 0 0 8px; font-size: 25px; }
    .scene-content p { margin: 0 0 20px; color: #6a5c75; line-height: 1.5; }
    .scene-sheet { border: 1px solid #d7d9df; border-radius: 9px; overflow: hidden; }
    .scene-sheet table { width: 100%; border-collapse: collapse; table-layout: fixed; font-size: 16px; }
    .scene-sheet th, .scene-sheet td { height: 49px; padding: 0 14px; border-right: 1px solid #dfe2e5;
      border-bottom: 1px solid #dfe2e5; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; text-align: left; }
    .scene-sheet th { width: 42px; padding: 0; background: #eef0f2; color: #53606a; text-align: center; }
    .scene-sheet thead td { height: 31px; background: #eef0f2; text-align: center; color: #53606a; font-size: 13px; }
    .scene-sheet .scene-head { background: #e8f3ec; color: #145b35; font-weight: 750; }
    .scene-sheet .scene-entry { background: #f8fbf9; }
    .scene-footer { display: flex; align-items: center; justify-content: space-between; margin-top: 17px;
      color: #655370; font-size: 14px; }
    .scene-pill { display: inline-block; padding: 8px 13px; border-radius: 8px; background: #e9f3ed;
      color: #176c3f; font-size: 13px; font-weight: 750; }
    .scene-chem { display: grid; grid-template-columns: 1fr 1fr; gap: 30px; align-items: stretch; }
    .scene-molecule { display: grid; place-items: center; min-height: 290px; border: 1px solid #e5dceb;
      border-radius: 12px; background: #fcf9fe; }
    .scene-molecule svg { width: 220px; height: 220px; }
    .scene-menu { border: 1px solid #dfd7e6; border-radius: 12px; overflow: hidden; }
    .scene-menu h3 { margin: 0; padding: 17px 21px; background: #f4eff8; font-size: 17px; }
    .scene-menu div { padding: 17px 21px; border-top: 1px solid #eee7f3; }
    .scene-menu .scene-selected { background: #eee3f8; color: #53218b; font-weight: 750; }
    .scene-modal { width: 670px; margin: 10px auto; padding: 28px 34px; box-sizing: border-box;
      border: 1px solid #d9cbe5; border-radius: 15px; background: #fff;
      box-shadow: 0 15px 38px rgba(31, 18, 48, .13); }
    .scene-modal h2 { margin-bottom: 13px; }
    .scene-modal li { margin: 13px 0; line-height: 1.4; }
    .scene-modal .scene-actions { display: flex; justify-content: flex-end; gap: 10px; margin-top: 22px; }
    .scene-modal .scene-actions span { border-radius: 8px; padding: 10px 18px; background: #f2ecf8; }
    .scene-modal .scene-actions .scene-primary { background: #6d35a7; color: white; font-weight: 700; }
    .scene-file { display: flex; justify-content: space-between; align-items: center; padding: 18px;
      border: 1px solid #dfd4e9; border-radius: 9px; background: #f8f4fc; font-weight: 700; }
    .scene-field { display: flex; align-items: center; gap: 14px; margin: 12px 0; }
    .scene-field strong { width: 130px; font-size: 14px; }
    .scene-field span { flex: 1; padding: 11px 14px; border: 1px solid #d9d3df;
      border-radius: 7px; background: #fff; }
  `;
  document.head.appendChild(style);
  const sheet = (complete) => {
    const values = complete ? [
      ['molecule_01', '3.42', 'CCO'], ['molecule_02', '4.15', 'c1ccccc1'],
      ['molecule_03', '2.87', 'CC(=O)O']
    ] : [['molecule_01', '3.42', ''], ['molecule_02', '4.15', ''], ['molecule_03', '2.87', '']];
    return '<div class="scene-sheet" id="tutorial-scene-focus"><table><thead><tr><th></th><td>A</td><td>B</td><td>C</td></tr></thead>' +
      '<tbody><tr><th>1</th><td class="scene-head">code_name</td><td class="scene-head">target</td><td class="scene-head">SMILES</td></tr>' +
      values.map((row, index) => '<tr><th>' + (index + 2) + '</th>' + row.map(value => '<td class="scene-entry">' + value + '</td>').join('') + '</tr>').join('') +
      '</tbody></table></div>';
  };
  const examples = {
    'spreadsheet-blank': '<div class="scene-content"><h2>Start a molecule table</h2><p>Open a blank spreadsheet and use one row per molecule.</p>' +
      '<div class="scene-sheet" id="tutorial-scene-focus"><table><thead><tr><th></th><td>A</td><td>B</td><td>C</td></tr></thead><tbody>' +
      [1,2,3,4,5].map(n => '<tr><th>' + n + '</th><td></td><td></td><td></td></tr>').join('') +
      '</tbody></table></div><div class="scene-footer"><span>Blank workbook</span><span class="scene-pill">One molecule per row</span></div></div>',
    'spreadsheet-columns': '<div class="scene-content"><h2>Define the essential columns</h2><p>Unique name, measured target, and molecular structure.</p>' +
      sheet(false) + '<div class="scene-footer"><span>Fill SMILES after copying each structure.</span><span class="scene-pill">CSV layout</span></div></div>',
    'smiles-copy': '<div class="scene-content"><h2>Copy a structure as SMILES</h2><p>In ChemDraw, select the molecule and choose Edit → Copy As → SMILES.</p>' +
      '<div class="scene-chem"><div class="scene-molecule"><svg viewBox="0 0 220 220" aria-hidden="true">' +
      '<g fill="none" stroke="#372449" stroke-width="4" stroke-linecap="round" stroke-linejoin="round">' +
      '<path d="M110 25 183 67 183 151 110 193 37 151 37 67Z"/>' +
      '<path d="M110 39 169 74M169 144 110 179M51 144V75"/></g></svg></div>' +
      '<div class="scene-menu"><h3>Edit menu</h3><div>Copy</div><div>Copy As</div>' +
      '<div class="scene-selected" id="tutorial-scene-focus">SMILES&nbsp;&nbsp; → &nbsp;&nbsp;c1ccccc1</div>' +
      '<div>Paste into the matching spreadsheet row</div></div></div></div>',
    'spreadsheet-complete': '<div class="scene-content"><h2>Review the finished table</h2><p>Check every molecule before saving the CSV.</p>' +
      sheet(true) + '<div class="scene-footer"><span>3 molecules · no missing values</span><span class="scene-pill">Ready to export</span></div></div>',
    'csv-save': '<div class="scene-content"><h2>Save as a CSV file</h2><p>A spreadsheet workbook is not the training input. Choose comma-separated values.</p>' +
      '<div class="scene-modal"><div class="scene-field"><strong>File name</strong><span>example_molecules.csv</span></div>' +
      '<div class="scene-field"><strong>Save as type</strong><span id="tutorial-scene-focus">CSV (Comma delimited) (*.csv)</span></div>' +
      '<div class="scene-actions"><span>Cancel</span><span class="scene-primary">Save</span></div></div></div>',
    'chemdraw-requirements': '<div class="scene-content"><div class="scene-modal" id="tutorial-scene-focus">' +
      '<h2>Before importing a drawing</h2><p>Check the structure file before easyROB reads the molecules.</p>' +
      '<ul><li>Save ChemDraw structures as <b>CDXML</b>.</li><li>Correct unusual bonds and valences.</li>' +
      '<li>Check charges and disconnected fragments.</li></ul>' +
      '<div class="scene-actions"><span class="scene-primary">Choose structure file</span></div></div></div>',
    'chemdraw-file': '<div class="scene-content"><div class="scene-modal"><h2>Select the structure file</h2>' +
      '<p>Choose the CDXML, SDF, or MOL file containing the molecules to import.</p>' +
      '<div class="scene-file" id="tutorial-scene-focus"><span>◈ &nbsp; sample_structures.cdxml</span><span>CDXML file</span></div>' +
      '<div class="scene-actions"><span>Cancel</span><span class="scene-primary">Open</span></div></div></div>'
  };
  window.easyrobTutorialScene = scene => {
    document.getElementById('tutorial-external-scene')?.remove();
    if (!examples[scene]) return;
    const wrapper = document.createElement('div');
    wrapper.id = 'tutorial-external-scene';
    wrapper.className = 'tutorial-external-scene';
    const application = scene.startsWith('spreadsheet') || scene === 'csv-save' ? 'Spreadsheet' : 'ChemDraw';
    wrapper.innerHTML = '<div class="scene-titlebar"><strong>' + application + ' example</strong>' +
      '<span>ILLUSTRATIVE WORKFLOW</span></div>' + examples[scene];
    document.body.appendChild(wrapper);
  };
})();
"""


def _ordered_images(topic):
    return sorted(
        (OUTPUT / topic).glob(f"{topic}_*.png"),
        key=lambda path: tuple(int(part) if part.isdigit() else part for part in path.stem.split("_")),
    )


def _step_title(block):
    match = re.search(r"<b>(.*?)</b>", block, re.IGNORECASE | re.DOTALL)
    return re.sub(r"<[^>]*>", "", match.group(1)).strip().rstrip(".") if match else "Tutorial step"


def main():
    captures = []
    selected_topic = os.environ.get("TUTORIAL_CAPTURE_TOPIC", "").strip()
    if selected_topic and selected_topic not in CAPTURES:
        raise ValueError(f"Unknown tutorial topic: {selected_topic}")
    for topic, specifications in CAPTURES.items():
        if selected_topic and topic != selected_topic:
            continue
        paths = _ordered_images(topic)
        blocks = (ROOT / f"{topic}.md").read_text(encoding="utf-8").split("---")
        if len(paths) != len(specifications) or len(blocks) != len(specifications):
            raise ValueError(f"Tutorial step and image counts differ for {topic}")
        captures.extend((path, specification, _step_title(block), index)
                        for index, (path, specification, block)
                        in enumerate(zip(paths, specifications, blocks), 1))
    selected_steps = os.environ.get("TUTORIAL_CAPTURE_STEPS", "").strip()
    if selected_steps:
        steps = {int(step.strip()) for step in selected_steps.split(",")}
        captures = [capture for capture in captures if capture[3] in steps]
    limit = int(os.environ.get("TUTORIAL_CAPTURE_LIMIT", "0"))
    if limit:
        captures = captures[:limit]

    app = QApplication(sys.argv)
    view = QWebEngineView()
    view.resize(1160, 720)
    view.show()
    page = view.page()

    def capture(index):
        if index >= len(captures):
            print(f"Captured {len(captures)} tutorial images", flush=True)
            app.quit()
            return
        path, spec, title, step = captures[index]
        settings = json.dumps({"page": spec.page, "selector": spec.selector,
                               "scene": spec.scene, "title": title, "step": step})
        script = f"""(() => {{
          const config = {settings};
          document.querySelectorAll('.tutorial-highlight').forEach(node => node.classList.remove('tutorial-highlight'));
          document.getElementById('tutorial-capture-label')?.remove();
          window.easyrobTutorialScene(config.scene);
          document.getElementById('tutorial-demo-data').hidden = config.scene !== 'data';
          document.getElementById('chemdraw-editor').hidden = config.scene !== 'chemdraw';
          document.getElementById('embedded-gallery').hidden = config.scene !== 'gallery';
          document.getElementById('plot-view').hidden = config.scene !== 'plot';
          document.getElementById('prediction-view').hidden = config.scene !== 'prediction';
          document.getElementById('checkml-params-picker').hidden = config.scene !== 'checkml-model';
          document.getElementById('bot-cloud-settings').hidden = config.scene === 'bot-heuristic';
          document.getElementById('bot-mode').value = config.scene === 'bot-heuristic' ? 'Heuristic' : 'Cloud AI';
          if (config.scene === 'bot' || config.scene === 'bot-settings' || config.scene === 'bot-heuristic') {{
            window.easyrobApp.openDrawer('bot');
            document.querySelector('.bot-settings').open = config.scene !== 'bot';
          }} else window.easyrobApp.openDrawer(null);
          window.easyrobApp.showPage(config.page);
          document.getElementById('workflow-state-badge').textContent = config.scene === 'complete' ? 'Complete' : config.scene === 'running' ? 'Running' : 'Ready';
          document.getElementById('workflow-activity').textContent = config.scene === 'complete' ? 'Workflow complete' : config.scene === 'running' ? 'PREDICT is running' : 'Waiting for a run';
          const target = document.querySelector(config.selector);
          if (!target) return 'Missing selector: ' + config.selector;
          target.classList.add('tutorial-highlight');
          target.scrollIntoView({{block:'center', inline:'nearest'}});
          const label = document.createElement('div');
          label.id = 'tutorial-capture-label';
          label.textContent = String(config.step).padStart(2, '0') + '  ' + config.title;
          document.body.appendChild(label);
          return 'ok';
        }})()"""

        def after_script(result):
            if result != "ok":
                print(f"{path.name}: {result}", flush=True)
                app.exit(1)
                return

            def save():
                if not view.grab().save(str(path)):
                    print(f"Unable to save {path}", flush=True)
                    app.exit(1)
                    return
                capture(index + 1)

            QTimer.singleShot(700, save)

        page.runJavaScript(script, after_script)

    def loaded(success):
        if not success:
            print("Unable to load web workspace", flush=True)
            app.exit(1)
            return
        page.runJavaScript(DEMO_SCRIPT)
        page.runJavaScript(EXTERNAL_SCENE_SCRIPT)
        page.runJavaScript("""(() => {
          const style = document.createElement('style');
          style.textContent = `.tutorial-highlight { outline: 4px solid #f2a93b !important; outline-offset: 5px !important;
            border-radius: 10px; box-shadow: 0 0 0 9px rgba(242,169,59,.20) !important; }
            #tutorial-capture-label { position: fixed; z-index: 9999; right: 22px; bottom: 18px;
            max-width: 360px; padding: 10px 16px; border-radius: 12px; background: #251535;
            color: white; font: 700 13px 'Segoe UI', Arial, sans-serif;
            box-shadow: 0 10px 30px rgba(25,10,45,.25); }`;
          document.head.appendChild(style);
        })()""")
        QTimer.singleShot(700, lambda: capture(0))

    view.loadFinished.connect(loaded)
    view.setUrl(QUrl.fromLocalFile(str(ROOT.parent / "web" / "app.html")))
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
