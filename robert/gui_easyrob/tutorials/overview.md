<b>What you can do here.</b><br><br>easyROB connects two tools. <b>AQME</b> turns molecular structures into numerical descriptors. <b>ROBERT</b> uses those descriptors to prepare data, build models, check their reliability, make predictions, and create reports. Follow this overview once, then use the practical tutorial that matches your starting data.

---

<b>Find your starting point.</b><br><br>The <b>Workflow</b> page opens first. It shows CSV inputs, analysis settings, descriptor columns, run controls, workflow progress, and the process log. Scroll within the page to reach the controls below the fold.

---

<b>Choose the data.</b><br><br>Load the training <b>Input CSV</b>. The <b>External Test CSV</b> is optional; use it for molecules you want to predict after the model is built. If you reopen a CSV in a folder that already contains ROBERT outputs, easyROB detects those files and makes their available Results views accessible again.

---

<b>Tell ROBERT what to learn.</b><br><br>Set <b>Target Column (y)</b> to the property to predict, <b>Prediction Type</b> to Regression for a numerical value or Classification for a category, and <b>name column</b> to each row's identifier. For example, a table with <b>code_name</b> and <b>target</b> should use those columns for name and y respectively.

---

<b>Keep only useful descriptors.</b><br><br>Move metadata that must not influence the model from <b>Available Columns</b> to <b>Ignored Columns</b>. An identifier or free-text note is not a molecular descriptor. Check this list before running: an unintended column can change what the model learns. The chosen target and name columns are handled separately.

---

<b>Decide whether AQME is needed.</b><br><br>Choose <b>Yes</b> for <b>Calculate descriptors with AQME</b> when your CSV contains structures that still need descriptors. The AQME descriptors page becomes available and AQME appears in Workflow progress. Choose <b>No</b> when the CSV already contains the descriptors you want ROBERT to use.

---

<b>Choose the scope of the run.</b><br><br>Use the <b>Workflow</b> selector. <b>Full Workflow</b> is the usual choice for a new model. CURATE, GENERATE, VERIFY, PREDICT, and REPORT run a particular stage when its required inputs or earlier outputs are ready.

---

<b>Start the right process.</b><br><br>Use <b>Run ROBERT</b> to launch the selected workflow. <b>Run AQME only</b> creates descriptors without starting ROBERT. <b>Stop</b> interrupts a running process. Before clicking, review the loaded CSV, target and name columns, AQME choice, and selected workflow.

---

<b>Read progress at a glance.</b><br><br>Workflow progress shows the optional AQME preparation and the ROBERT stages. A green check means a stage completed; an error marks a failed stage. Open <b>Process log</b> beside the progress card for detailed messages.

---

<b>Explore AQME when you need descriptors.</b><br><br>With AQME enabled, open <b>AQME descriptors</b> to adjust settings. The page includes the ChemDraw/SDF import and a molecular viewer. Select atoms when atom-based descriptors are useful to your question.

---

<b>Change advanced settings deliberately.</b><br><br>The <b>Advanced options</b> page groups controls by ROBERT stage. Start with defaults unless you have a specific modelling reason to change one. Use a section's Help control when a setting is unfamiliar.

---

<b>Browse prepared molecular data.</b><br><br><b>MolSSI databases</b> opens the descriptor library browser. Use it to inspect existing datasets. The Workflow input still determines what runs in your project.

---

<b>Evaluate a model you chose.</b><br><br>The <b>CheckML</b> page is separate from ROBERT's automatic model search. It runs CHECKML on a default model, a scikit-learn estimator, parameters from CSV, or a saved model. Open the <b>Check ML</b> tutorial for input requirements.

---

<b>Find outputs in one place.</b><br><br>After a run, or after reopening a CSV beside existing outputs, open <b>Results</b>. Its buttons lead to Report, Predictions, Images, and Interactive plots when those files exist. If several models were generated, use the <b>Model</b> selector first so you inspect the intended model.

---

<b>Inspect workflow figures.</b><br><br>Choose <b>Images</b> inside Results, then use the stage filters to narrow the gallery. Figures can reveal patterns, outliers, or model behaviour that a single score cannot show.

---

<b>Explore predictions visually.</b><br><br><b>Predictions</b> displays external test results when such a CSV was generated. <b>Interactive plots</b> shows model plots even when no external prediction table exists. Hover over a point to inspect it, and click a legend entry to show or hide that set of points.

---

<b>Ask robBOT for help.</b><br><br>Select <b>robBOT</b> in the sidebar. Its drawer stays beside the workspace while you ask about a workflow, result, popup, or setting. The robBOT tutorial explains the available answer modes.
