In this tutorial we start from molecular structures created
in <b>ChemDraw</b>.<br><br>

ChemDraw files contain only chemical structures, without
descriptors or datasets.<br><br>

easyROB can read these structures, convert them into a dataset,
and then use them to train a machine learning model.

---

First, enable the <b>AQME Workflow</b> to unlock the
<b>AQME</b> tab.<br><br>

This tab contains the tools required to process ChemDraw
files and convert them into a dataset that can be used
for modelling.

---

In the <b>AQME</b> tab, click
<b>Generate CSV from ChemDraw files</b>.<br><br>

This option converts ChemDraw structures into a dataset
that can later be used to train the model.

---

A pop-up window will appear explaining the requirements
for ChemDraw files.<br><br>

The structures must be saved in <b>CDXML format</b>, and
the chemical drawings should be correct, since incorrect
structures may lead to errors when the GUI reads them.

---

Next, select the ChemDraw file containing the molecular
structures.<br><br>

Browse to the file location and load it into the program.

---

Once the file is loaded, a table will appear containing
the molecules extracted from the ChemDraw file.<br><br>

Two columns are displayed: <b>code_name</b> and
<b>Target</b>.<br><br>

Assign a name to each molecule in the <b>code_name</b>
column.<br><br>

The <b>Target</b> column should contain the property that
will be predicted by the machine learning model.<br><br>

In this example, the target column is renamed to
<b>Yield</b> and filled with the corresponding values.

---

After completing the information, click <b>Save as CSV</b>.<br><br>

A dialog window will appear asking you to choose the location
and the name of the CSV file to be saved.

---

A confirmation pop-up will then appear indicating that the
CSV file has been successfully saved and automatically
loaded into the GUI.

---

Return to the <b>ROBERT</b> tab.<br><br>

The newly generated CSV dataset will appear automatically
as the training input.

---

Next, configure the model parameters.<br><br>

Select the <b>target column</b> (Yield), choose the
prediction type (<b>regression</b>), and specify the
column containing the molecule identifiers.<br><br>

No columns need to be ignored in this example.

---

Press <b>Run ROBERT</b> to launch the workflow.<br><br>

easyROB will automatically generate molecular descriptors
from the structures and train the machine learning model.

---

A message will appear indicating that only structural
descriptors will be generated in this workflow.<br><br>

Atom-based descriptors can also be generated using the
<b>AQME</b> tab, as shown in the <b>From CSV</b> tutorial.

---

Once the workflow finishes, a pop-up message confirms
that the process has completed successfully.<br><br>

The <b>Reports</b> and <b>Images</b> tabs become available,
providing access to the results generated during the
workflow.<br><br>

These result sections are explained in more detail in the
<b>Overview</b> tutorial.