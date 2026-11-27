<b>Goal: predict new molecules with an existing model.</b><br><br>Use this route only after a ROBERT project has produced model outputs. You are reusing that model rather than training from scratch. The new molecules must provide the same kind of input information used for the original model.

---

<b>Prepare the new data.</b><br><br>Save the external test CSV in the existing ROBERT project folder so the saved model can be found. Use a name column for each new molecule. If the trained model used AQME descriptors, include SMILES; if it used your own descriptors, provide the matching descriptor columns.

---

<b>Select both CSVs.</b><br><br>On <b>Workflow</b>, load the original project input CSV, then select the new file under <b>External test CSV</b>. Check both filenames in the Dataset card.

---

<b>Run only the needed stage.</b><br><br>Set the <b>Workflow</b> selector to <b>PREDICT</b>. The other ROBERT stages stay inactive for this run.

---

<b>Start and watch PREDICT.</b><br><br>Click <b>Run ROBERT</b>. The PREDICT progress row becomes active; open <b>Process log</b> to follow file loading and descriptor handling.

---

<b>Match the descriptor method.</b><br><br>If the dialog asks how to handle descriptors, choose <b>Generate descriptors with AQME</b> when the original model was trained with AQME-derived descriptors and your test CSV has SMILES. Choose <b>Descriptors already present</b> only when the test CSV already contains the descriptors expected by that model.

---

<b>Review the prediction output.</b><br><br>Open <b>Results → Predictions</b> after external output is created. Use the Model selector when several models exist. Interactive plots offers visual comparisons, and legend clicks show or hide point groups.
