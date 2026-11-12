<b>Goal: assess a model you chose.</b><br><br>The GUI calls this tab <b>CheckML</b>; this tutorial calls the route Check ML. It runs ROBERT CHECKML using a model or model family you specify. Choose it when you want ROBERT's evaluation and report for that choice rather than its automatic model search.

---

<b>Open the CheckML page.</b><br><br>Select <b>CheckML</b> in the sidebar. Its CSV inputs, model source, Run CheckML button, and process log belong to this page. Workflow settings do not replace these inputs.

---

<b>Load the data to evaluate.</b><br><br>Choose an <b>Input CSV</b> containing the target and numerical descriptor columns for the model. You may also add an external test CSV, and clear that optional selection without replacing the input. Check that both files correspond to the same problem; CHECKML makes its own internal split using ROBERT's defaults.

---

<b>Map and filter columns.</b><br><br>Select the target column, name column, and prediction type. In <b>Columns to ignore</b>, <b>SMILES is checked by default</b> when the CSV is loaded, as are other text columns. Review the checks so only intended numerical descriptors reach the model.

---

<b>Choose where the model comes from.</b><br><br>The <b>MODEL TO CHECKML</b> menu offers default MVL, a scikit-learn estimator, a hyperparameters CSV, or a saved joblib/pickle model. Pick the option that matches what you actually have. The controls shown below the menu change according to that choice.

---

<b>Supply only the required model details.</b><br><br>For a scikit-learn estimator, choose its name and fill only hyperparameters you want to override; blank fields use its defaults. A hyperparameters CSV needs <b>param,value</b> rows, including a <b>model</b> row. For a saved model, select its joblib/pickle file instead.

---

<b>Run and interpret the result.</b><br><br>Press <b>Run CheckML</b> and follow the process log. Use Stop if needed. Selecting a CSV keeps you on CheckML even when its folder has earlier results. After successful completion, open <b>Results</b> for the PDF and available outputs; if the run stops early, read the log before changing inputs.
