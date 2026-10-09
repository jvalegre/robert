<b>What you can do here.</b><br><br>easyROB connects two tools. <b>AQME</b> turns molecular structures into numerical descriptors. <b>ROBERT</b> uses those descriptors to prepare data, build models, check their reliability, make predictions, and create reports. Follow this overview once, then use the practical tutorial that matches your starting data.

---

<b>Find your starting point.</b><br><br>The <b>ROBERT</b> tab opens first. Everything needed for a standard run is on this scrollable page: CSV inputs at the top, modelling choices in the middle, and workflow progress, Live log, and action buttons near the bottom. Scroll inside the tab to reach the controls below the fold.

---

<b>Choose the data.</b><br><br>Load the training <b>Input CSV</b>. The <b>External Test CSV</b> is optional; use it for molecules you want to predict after the model is built. If you reopen a CSV in a folder that already contains ROBERT outputs, easyROB detects those files and makes their available Results views accessible again.

---

<b>Tell ROBERT what to learn.</b><br><br>Set <b>Target Column (y)</b> to the property to predict, <b>Prediction Type</b> to Regression for a numerical value or Classification for a category, and <b>name column</b> to each row's identifier. For example, a table with <b>code_name</b> and <b>target</b> should use those columns for name and y respectively.

---

<b>Keep only useful descriptors.</b><br><br>Move metadata that must not influence the model from <b>Available Columns</b> to <b>Ignored Columns</b>. An identifier or free-text note is not a molecular descriptor. Check this list before running: an unintended column can change what the model learns. The chosen target and name columns are handled separately.

---

<b>Decide whether AQME is needed.</b><br><br>Choose <b>Yes</b> for <b>Start by calculating descriptors from SMILES? (AQME)</b> when your CSV contains structures that still need descriptors. The AQME tab becomes available and AQME appears as preparation in Workflow progress. Choose <b>No</b> when the CSV already contains the descriptors you want ROBERT to use.

---

<b>Choose the scope of the run.</b><br><br>Click <b>What do you want to run?</b>. <b>Full Workflow</b> is the usual choice for a new model. CURATE, GENERATE, VERIFY, PREDICT, and REPORT run a particular stage when its required inputs or earlier outputs are ready. Hovering or scrolling over the closed dropdown does not change your selection.

---

<b>Start the right process.</b><br><br>At the bottom, <b>Run ROBERT</b> launches the workflow chosen above. <b>Run AQME (descriptor generation)</b> creates descriptors without starting ROBERT. <b>Stop</b> interrupts a running process. Before clicking, review the loaded CSV, the target and name columns, the AQME choice, and the selected workflow.

---

<b>Read progress at a glance.</b><br><br>Workflow progress shows the optional AQME preparation and the ROBERT stages. The current stage animates; a green check means it completed; an error marks the stage that failed. A stage not selected for an individual run stays inactive. Open <b>Live log</b> beside the cards to see detailed messages and diagnose a warning or failure.

---

<b>Explore AQME when you need descriptors.</b><br><br>With AQME enabled, open its tab to adjust descriptor settings. It also contains the ChemDraw/SDF to CSV tool. When a compatible SMILES dataset is available, the molecular viewer may show a common scaffold; select atoms there only if atom-based descriptors are useful to your question.

---

<b>Change advanced settings deliberately.</b><br><br>The <b>Advanced Options</b> tab groups controls by ROBERT stage. Start with defaults unless you have a specific modelling reason to change one. If a setting is unfamiliar, use its help control before editing it, then note the change so you can interpret the resulting report.

---

<b>Browse prepared molecular data.</b><br><br><b>MolSSI Databases</b> lets you explore descriptor libraries and available molecules. Use it when you want an existing dataset to inspect or analyse. The main ROBERT input still determines what runs in your own workflow.

---

<b>Evaluate a model you chose.</b><br><br>The <b>Check model</b> tab is separate from ROBERT's automatic model search. It runs EVALUATE on a default model, a scikit-learn estimator, parameters from CSV, or a saved model. Open the <b>Check ML</b> tutorial for the input requirements and model choices.

---

<b>Find outputs in one place.</b><br><br>After a run, or after reopening a CSV beside existing outputs, open <b>Results</b>. Its buttons lead to Report, Predictions, Images, and Interactive plots when those files exist. If several models were generated, use the <b>Model</b> selector first so you inspect the intended model.

---

<b>Inspect workflow figures.</b><br><br>Choose <b>Images</b> inside Results, then select a stage such as CURATE or PREDICT from the View menu. Figures can reveal patterns, outliers, or model behaviour that a single score cannot show. A missing stage or empty view means no matching image was found in this project.

---

<b>Explore predictions visually.</b><br><br><b>Predictions</b> displays external test results when such a CSV was generated. <b>Interactive plots</b> shows model plots even when no external prediction table exists. Hover over a point to inspect it, and click a legend entry to show or hide that set of points.

---

<b>Ask robBOT for help.</b><br><br>The <b>robBOT</b> button remains at the top right. It opens a separate window, so you can keep the GUI visible while asking about a workflow, result, popup, or setting. The robBOT tutorial explains how to ask questions, summarize available results, and choose an answer mode.
