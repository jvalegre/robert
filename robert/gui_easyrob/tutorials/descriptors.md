This tutorial shows how to generate molecular descriptors
using the <b>AQME</b> workflow.<br><br>

Descriptor generation can be useful when preparing datasets
for machine learning models or when molecular descriptors
are required for other types of analysis.

---

Load a <b>CSV file</b> containing the molecules for which
descriptors should be generated.<br><br>

The dataset must contain a <b>SMILES column</b>, which
AQME will use to compute the molecular descriptors.

---

Press <b>Run AQME</b> to start the descriptor generation
workflow.<br><br>

AQME will automatically generate three descriptor datasets:
<b>denovo</b>, <b>interpret</b>, and <b>full</b>.<br><br>

These files contain increasing numbers of descriptors,
ranging from smaller and more interpretable sets to the
complete descriptor collection.

---

A pop-up window will indicate that only <b>molecular
descriptors</b> will be generated in this workflow.<br><br>

Atomic descriptors can also be generated using the
<b>AQME</b> tab. An example of this workflow is shown
in the <b>From CSV</b> tutorial.

---

Once the process finishes, a message will indicate that
the descriptor datasets have been generated successfully.<br><br>

The three CSV files (<b>denovo</b>, <b>interpret</b>, and
<b>full</b>) will be saved in the working directory.<br><br>

These descriptor datasets can be used directly for further
analysis or loaded into the <b>ROBERT</b> workflow to
build machine learning models.

---

In this example, the <b>interpret</b> dataset is loaded,
but any of the generated descriptor sets can be used
depending on your needs.

At this stage, the descriptors can be inspected, filtered,
or modified before building a machine learning model.<br><br>

If you want to train a predictive model using these descriptors,
the dataset must contain a <b>target column</b> with the property
you want to predict.

If the original CSV did not include a target value, it can be
added later before running the <b>ROBERT</b> workflow.

Alternatively, the generated descriptor datasets can also be
used independently of ROBERT for other analyses or workflows.