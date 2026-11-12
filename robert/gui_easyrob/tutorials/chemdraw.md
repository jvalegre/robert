<b>Goal: turn drawings into a modelling dataset.</b><br><br>This example starts with structures saved from ChemDraw as CDXML. easyROB extracts the molecules, lets you add identifiers and target values, saves a CSV, and can then calculate descriptors with AQME before ROBERT builds a model.

---

<b>Enable AQME first.</b><br><br>On <b>ROBERT</b>, choose <b>Yes</b> for <b>Start by calculating descriptors from SMILES? (AQME)</b>. This unlocks the AQME tab. Keep it enabled for the later full workflow because the converted structures still need descriptors.

---

<b>Open the conversion tool.</b><br><br>In <b>AQME</b>, click <b>Generate CSV from ChemDraw Files or SDF file</b>. This prepares a table from the structure file; it does not start model training yet.

---

<b>Check the drawing requirements.</b><br><br>Read the dialog before selecting a file. Save a ChemDraw drawing as <b>CDXML</b> and inspect unusual bonds, charges, and disconnected fragments. If the structure is drawn incorrectly, its exported molecule or descriptors may not represent what you intended.

---

<b>Select the structure file.</b><br><br>Choose the CDXML containing the molecules for this dataset. Confirm the filename before continuing; if you selected the wrong drawing, go back now rather than editing an unrelated table.

---

<b>Complete the extracted table.</b><br><br>Review each molecule. Give it a unique <b>code_name</b> and enter the measured property in the target column. The target is what ROBERT will learn, so check units and missing values. In this example, the property is Yield.

---

<b>Save the prepared CSV.</b><br><br>After reviewing the rows, click <b>Save as CSV</b> and choose a clear filename and destination. Remember that location: the workflow outputs will be associated with this dataset's project folder.

---

<b>Confirm the handoff.</b><br><br>The confirmation dialog indicates that the CSV was written and loaded into easyROB. If the save fails or the wrong location was chosen, resolve that before starting ROBERT.

---

<b>Return to the main tab.</b><br><br>On <b>ROBERT</b>, check that the new CSV appears as the selected training input. This is the moment to catch an accidental older file. AQME should still be set to Yes for descriptor generation from the converted structures.

---

<b>Tell ROBERT how to read it.</b><br><br>Choose the measured property as <b>Target Column (y)</b>, select Regression or Classification, and set <b>code_name</b> as the name column. Move any extra metadata that should not become a descriptor to Ignored Columns.

---

<b>Launch the combined workflow.</b><br><br>Choose <b>Full Workflow</b> and click <b>Run ROBERT</b>. The AQME preparation card is followed by the ROBERT stage cards. Use Live log if the conversion or descriptor step reports an error.

---

<b>Read any descriptor notice.</b><br><br>If easyROB announces that it will use structural descriptors, continue when that matches your goal. For atom-based descriptors, return to the AQME tab and select relevant atoms when a common scaffold is available.

---

<b>Inspect the output.</b><br><br>Open <b>Results → Report</b> after the run to understand model performance. <b>Images</b> shows generated figures. Predictions appears when external prediction files exist, and Interactive plots appears when model plot data is available.
