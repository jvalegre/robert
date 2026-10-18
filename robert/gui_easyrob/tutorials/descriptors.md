<b>Goal: create descriptors without training.</b><br><br>This route runs AQME on molecular structures and saves descriptor CSVs. It is useful for inspecting features or preparing data before deciding whether to build a ROBERT model.

---

<b>Load structures.</b><br><br>On <b>ROBERT</b>, choose a CSV with a <b>SMILES</b> column and identifiers. Check that the selected filename is correct. Set the AQME option to Yes if you want to inspect descriptor settings or atom selections in the AQME tab before running.

---

<b>Start AQME on its own.</b><br><br>Scroll to the action buttons and click <b>Run AQME (descriptor generation)</b>. The AQME card in Workflow progress shows the activity; CURATE through REPORT are not part of this run. You do not need to select Full Workflow or press Run ROBERT.

---

<b>Understand the notice.</b><br><br>If a dialog says structural descriptors will be generated, continue when those features meet your goal. Atom-based descriptors need suitable structures and, when available, atom choices in the AQME tab.

---

<b>Find the files.</b><br><br>When AQME finishes, look for its green check and inspect <b>Live log</b> for output paths or warnings. The generated CSV variants can differ in descriptor coverage; choose the set whose detail matches your analysis needs.

---

<b>Reuse a descriptor dataset.</b><br><br>To model later, load one generated CSV as the ROBERT input. Set the target and name columns, choose <b>No</b> for AQME because descriptors are already present, and then select the desired ROBERT workflow. Check that the target property was carried into the descriptor file before training.
