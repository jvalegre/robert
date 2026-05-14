In this tutorial we will build a machine learning model starting
from a simple CSV dataset.<br><br>

We will first show how to create the input CSV file containing
the molecular structures and the target property to be predicted.<br><br>

In this example we will use <b>SMILES</b> representations so that
molecular descriptors can be generated automatically with AQME,
but if descriptors are already available they can be included
directly in the CSV instead of SMILES.<br><br>

This dataset will then be used to train the machine learning
model within the ROBERT workflow.

---

Open a new spreadsheet in Microsoft Excel. 
Any spreadsheet software can be used to create or edit CSV files.<br><br>

---

In this example we will create a dataset containing three columns:
<b>code_name</b> (unique identifiers for each molecule), 
<b>target</b> (the property that the machine learning model will 
learn to predict), and <b>SMILES</b> (the molecular structures).

---

The <b>SMILES strings</b> can be obtained in several ways.<br><br>

For example, they can be copied from molecular drawing software
such as <b>ChemDraw</b>, or retrieved from online chemical
databases that provide both the structure and the corresponding
SMILES string.<br><br>

In this tutorial we will copy the SMILES directly from ChemDraw,
as shown in the image.<br><br>

Select the structure and go to <b>Edit → Copy As → SMILES</b>.
You can also use the shortcut <b>Ctrl + Alt + C</b>.

---

Once the dataset is complete and all molecular structures have
been added as SMILES, the file must be saved as a CSV file.

---

Go to the <b>File</b> tab and select <b>Save As</b>. 
Choose <b>CSV</b> under the <b>Save as type</b> dropdown menu,
provide a name for the dataset, and click <b>Save</b>.

---

The first step in the workflow is loading the input dataset.<br><br>

Select the <b>training CSV file</b> that will be used to build
the machine learning model.<br><br>

Optionally, an <b>external test dataset</b> can also be provided.
If included, ROBERT will generate predictions for these
molecules after the model has been trained.

---


Next, configure the basic model parameters.<br><br>



Select the <b>target column</b>, choose the prediction type

(in this example <b>regression</b>), and specify the column

containing the molecule identifiers.<br><br>



No columns need to be excluded in this case, since the

SMILES column is handled automatically.



---



Enable the <b>AQME Workflow</b> to generate molecular

descriptors from the SMILES structures.<br><br>



Once this option is activated, the <b>AQME</b> tab becomes

available and can be used to adjust descriptor generation

settings.<br><br>



<b>Note:</b> If the input CSV already contains molecular

descriptors, this step can be skipped and the workflow can

be run directly.



---



In the <b>AQME</b> tab, the interface may detect a common

molecular scaffold across the dataset.<br><br>



In this example, the molecules share a carboxylic acid

functional group, which is displayed in the structure viewer.<br><br>



Atoms from this structure can be selected to generate

additional atom-based descriptors.



---



Here, two atoms from the carboxylic acid group have been

selected: the carbonyl oxygen and the hydroxyl oxygen.<br><br>



These selections will be used during descriptor generation.



---



Once the configuration is complete, return to the

<b>ROBERT</b> tab and press <b>Run ROBERT</b> to launch

the workflow.<br><br>



The program will automatically generate descriptors,

train the machine learning models, and produce the results.



---



After the workflow finishes, a message will indicate that

the process has completed successfully.<br><br>



You can then choose to open the generated report to

inspect the results.



---



The report summarizes the machine learning results, including

model performance metrics and validation information.<br><br>



Once the workflow has generated results, the <b>Reports</b>,

<b>Images</b>, and <b>Predictions</b> tabs become available.<br><br>



The <b>Reports</b> tab displays the PDF summary of the modelling

results, the <b>Images</b> tab provides access to the figures

generated during the workflow, and the <b>Predictions</b> tab

shows the predicted values for the external test dataset used

in this example.<br><br>



These result sections are described in more detail in the

<b>Overview</b> tutorial.

