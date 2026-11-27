<b>Goal: turn drawings into a modelling dataset.</b><br><br>This example starts with structures saved from ChemDraw as CDXML. easyROB extracts the molecules, lets you add identifiers and target values, saves a CSV, and can then calculate descriptors with AQME before ROBERT builds a model.

---

<b>Enable AQME first.</b><br><br>On <b>Workflow</b>, choose <b>Yes</b> for <b>Calculate descriptors with AQME</b>. This unlocks <b>AQME descriptors</b>. Keep it enabled for the later full workflow.

---

<b>Open the conversion tool.</b><br><br>In <b>AQME descriptors</b>, click <b>Import ChemDraw or SDF</b>. This prepares an editable molecule table; it does not start model training.

---

<b>Check the drawing requirements.</b><br><br>Save the ChemDraw drawing as <b>CDXML</b> and inspect unusual bonds, charges, and disconnected fragments before importing. Incorrect drawings can lead to incorrect molecules or descriptors.

---

<b>Select the structure file.</b><br><br>Use the file picker opened by <b>Import ChemDraw or SDF</b> to choose the correct CDXML, SDF, or MOL file. Check the filename before editing the imported table.

---

<b>Complete the extracted table.</b><br><br>The molecule table appears on the AQME descriptors page. Give each molecule a unique <b>code_name</b> and enter its measured property in the target column. Check units and missing values.

---

<b>Save the prepared CSV.</b><br><br>After reviewing the rows, click <b>Save as CSV</b> and choose a clear filename and destination. Remember that location: the workflow outputs will be associated with this dataset's project folder.

---

<b>Confirm the handoff.</b><br><br>After saving, the message below the table shows the selected CSV path. Check that the file was written and loaded before starting ROBERT.

---

<b>Return to Workflow.</b><br><br>On <b>Workflow</b>, check that the new CSV appears as the training input. AQME should still be set to Yes for descriptor generation from the converted structures.

---

<b>Tell ROBERT how to read it.</b><br><br>Choose the measured property as <b>Target Column (y)</b>, select Regression or Classification, and set <b>code_name</b> as the name column. Move any extra metadata that should not become a descriptor to Ignored Columns.

---

<b>Launch the combined workflow.</b><br><br>Choose <b>Full Workflow</b> and click <b>Run ROBERT</b>. AQME preparation is followed by the ROBERT stages. Use <b>Process log</b> if a step reports an error.

---

<b>Read any descriptor notice.</b><br><br>If easyROB announces that it will use structural descriptors, continue when that matches your goal. For atom-based descriptors, return to <b>AQME descriptors</b> and select relevant atoms when a common scaffold is available.

---

<b>Inspect the output.</b><br><br>Open <b>Results → Report</b> after the run to understand model performance. <b>Images</b> shows generated figures. Predictions appears when external prediction files exist, and Interactive plots appears when model plot data is available.
