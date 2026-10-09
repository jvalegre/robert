<b>Goal: assess a model you chose.</b><br><br>The GUI calls this tab <b>Check model</b>; this tutorial calls the route Check ML. It runs ROBERT EVALUATE using a model or model family you specify. Choose it when you want ROBERT's evaluation and report for that choice rather than its automatic model search.

---

<b>Open the separate EVALUATE page.</b><br><br>Click <b>Check model</b> in the main tab bar. Its CSV inputs, model source, Run EVALUATE button, progress indicator, and console belong to this page. Settings from the main ROBERT workflow do not replace these inputs.

---

<b>Load the data to evaluate.</b><br><br>Choose an <b>Input CSV</b> containing the target and numerical descriptor columns for the model. You may also add an external test CSV. Check that both files correspond to the same problem; EVALUATE makes its own internal split using ROBERT's defaults.

---

<b>Map and filter columns.</b><br><br>Select <b>y (target column)</b>, <b>names (ID column)</b>, and Regression or Classification. The ignored-column list excludes fields that are not descriptors. Text fields such as SMILES start checked when a CSV is loaded; review the checks so the model receives only the intended numerical features.

---

<b>Choose where the model comes from.</b><br><br>The <b>MODEL TO EVALUATE</b> menu offers default MVL, a scikit-learn estimator, a hyperparameters CSV, or a saved joblib/pickle model. Pick the option that matches what you actually have. The controls shown below the menu change according to that choice.

---

<b>Supply only the required model details.</b><br><br>For a scikit-learn estimator, choose its name and fill only hyperparameters you want to override; blank fields use its defaults. A hyperparameters CSV needs <b>param,value</b> rows, including a <b>model</b> row. For a saved model, select its joblib/pickle file instead.

---

<b>Run and interpret the result.</b><br><br>Press <b>Run EVALUATE</b> and follow the progress indicator and console on this tab. Use Stop if you need to interrupt it. After successful completion, open <b>Results</b> for the PDF and any other available outputs; if the run stops early, read the console before changing the inputs.
