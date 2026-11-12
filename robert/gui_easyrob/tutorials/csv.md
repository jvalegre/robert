<b>Goal: start from structures in a CSV.</b><br><br>We will prepare identifiers, a target property, and SMILES. AQME will calculate molecular descriptors from the structures; ROBERT will then use those descriptors to build and assess a model. If your CSV already has numerical descriptors, skip the AQME choice later.

---

<b>Open a spreadsheet.</b><br><br>Excel is shown in the image, but any editor that can save a comma-separated CSV works. Use one row per molecule and one header per column. Before importing the file, make sure the first row contains column names rather than data.

---

<b>Set up three essential columns.</b><br><br><b>code_name</b> identifies each molecule, <b>target</b> holds the value or class to learn, and <b>SMILES</b> describes the structure. For regression, target values should be numerical. Give every molecule a distinct identifier and check for missing targets before modelling.

---

<b>Add molecular structures.</b><br><br>In ChemDraw, select a structure and use <b>Edit → Copy As → SMILES</b>, then paste the string into its row. You can also obtain SMILES from a trusted database. Verify that each string belongs to the correct molecule; a shifted spreadsheet row silently pairs the wrong structure with the target.

---

<b>Review the completed table.</b><br><br>Check that every row has an identifier, target, and SMILES, with no accidental blank lines or extra header rows. A quick spot-check of a few molecules now is easier than diagnosing incorrect descriptor generation later.

---

<b>Save a CSV, not a workbook.</b><br><br>Use <b>File → Save As</b> and choose a CSV format. Give the file a simple name without spaces, because ROBERT checks CSV file names. Reopen it once to confirm the three headers and values survived the export.

---

<b>Load the input in easyROB.</b><br><br>On the <b>ROBERT</b> tab, select your training CSV. Add an <b>External Test CSV</b> only when you have a separate set of molecules to predict. Seeing the selected filename under the input control confirms that easyROB is working with the intended file.

---

<b>Map the columns.</b><br><br>Choose <b>target</b> as y, Regression or Classification according to the property, and <b>code_name</b> as the name column. Do not assume the first dropdown item is correct: inspect each selection. Move other non-descriptor metadata to Ignored Columns.

---

<b>Enable descriptor calculation.</b><br><br>Because this example starts from SMILES, choose <b>Yes</b> for <b>Start by calculating descriptors from SMILES? (AQME)</b>. The AQME tab becomes available, and the AQME stage appears before ROBERT in Workflow progress. Choose No only if you intend to use descriptors already present in the CSV.

---

<b>Inspect the AQME viewer.</b><br><br>Open the <b>AQME</b> tab. It may show a common molecular pattern after processing the SMILES. The placeholder in the image indicates where that pattern appears. You can use structural descriptors with the defaults even when there is no suitable shared scaffold.

---

<b>Optional: choose atoms.</b><br><br>If a common scaffold appears and specific atoms matter for your study, select them in the viewer for atom-based descriptors. Check that the highlighted atoms correspond to the chemistry you intend to compare. If no pattern appears, leave atom selections empty and continue with structural descriptors.

---

<b>Run AQME and ROBERT together.</b><br><br>Return to <b>ROBERT</b>, leave <b>Full Workflow</b> selected, and click <b>Run ROBERT</b>. AQME prepares descriptors first; ROBERT then advances through its stages. Watch the cards for the active step, and expand Live log when you need detailed progress.

---

<b>Interpret the finish state.</b><br><br>Green checks show stages that completed. If a card shows an error, open Live log and read the messages around that stage before changing inputs or trying again. A completed run should provide output files in the project folder.

---

<b>Open the results.</b><br><br>In <b>Results</b>, start with <b>Report</b> for the model summary. Use Images and Interactive plots to inspect behaviour; Predictions is available when external test output exists. If multiple models were produced, choose one with the Model selector before comparing its views.
