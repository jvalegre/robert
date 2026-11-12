<b>Goal: create descriptors without training.</b><br><br>This route runs AQME on molecular structures and saves descriptor CSVs. It is useful for inspecting features or preparing data before deciding whether to build a ROBERT model.

---

<b>Load structures.</b><br><br>On <b>Workflow</b>, choose a CSV with a <b>SMILES</b> column and identifiers. Check the displayed filename. Set <b>Calculate descriptors with AQME</b> to Yes if you want to inspect AQME settings first.

---

<b>Start AQME on its own.</b><br><br>Click <b>Run AQME only</b>. Workflow progress shows AQME activity; CURATE through REPORT are not part of this run. You do not need to press Run ROBERT.

---

<b>Understand the notice.</b><br><br>If a dialog says structural descriptors will be generated, continue when those features meet your goal. Atom-based descriptors need suitable structures and, when available, atom choices in the AQME tab.

---

<b>Find the files.</b><br><br>When AQME finishes, look for its completed status and inspect <b>Process log</b> for output paths or warnings. The generated CSV variants differ in descriptor coverage.

---

<b>Reuse a descriptor dataset.</b><br><br>To model later, load one generated CSV on <b>Workflow</b>. Set the target and name columns, choose <b>No</b> for AQME because descriptors are already present, then select the desired workflow. Check that the target property is present before training.
