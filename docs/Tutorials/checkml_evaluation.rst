Perform a CheckML evaluation
============================

Step 1. Dataset
++++++++++++++++

Upload a tabular dataset containing the required columns. An error will appear 
if the dataset does not meet all criteria. In such cases, the program will provide 
a message describing the issue; alternatively, users can 
visit :doc:`checkml_database` for more details.

Step 2. Evaluation setup
++++++++++++++++++++++++

Adjust the following three options:

- **Target column:** select the column with the target values (y values)
- **Name / ID column:** select the column with the names or identifiers
- **Problem type:** select the type of problem (regression or classification)

Optionally, users can also select:

- **SMILES column:** to display molecules in interactive graphs

Step 3. Model configuration
+++++++++++++++++++++++++++

Select your model through any of the following options:

- **Multivariate linear model (MVL)**
- **Other scikit-learn models:** in the interactive table that appears, select a model from the scikit-learn library and provide its hyperparameters
- **From a hyperparameters CSV:** For a hyperparameters CSV, use columns "param,value" and include a row such as "model,RandomForestRegressor"

.. warning::
  Saved pickle/joblib uploads are disabled by default on public Spaces because 
  loading them can execute code.

Step 4. Execute CheckML
+++++++++++++++++++++++

Press the **Evaluate model** button to run the evaluation. The evaluation may take 
a few minutes depending on the dataset size and model complexity.

Step 5. Analyze and report CheckML results
++++++++++++++++++++++++++++++++++++++++++

All scores, warnings, interactive plots, and other results will be available once 
the evaluation is complete. These results can be viewed and downloaded 
from the sections at the bottom of the page (**Report & Downloads**, 
**PDF Preview Figures**, **Interactive Predictions**, etc.).

PDF reports can be downloaded from the **Report & Downloads** section. 
These reports contain all the information needed to document the evaluation 
and can be directly included in the Supporting Information of a publication, 
together with the input CSV used in CheckML.