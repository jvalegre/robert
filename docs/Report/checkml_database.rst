Prepare a database for CheckML
==============================

Columns included
++++++++++++++++

Before using CheckML to evaluate a model, prepare a tabular database with one
row per compound or data point. Include **ONLY** the following columns:

* **Names or IDs**: one column with a unique name or identifier for each row.
* **Target (y values)**: one column with the target values.
* **Descriptors (X values)**: one column for each descriptor used by the model.

An optional **SMILES** column can be included to display molecules in
interactive graphs. It is additional to the name, target, and descriptor
columns (SMILES strings are not included as descriptors).

.. |br| raw:: html

   <br />

.. warning::
  During testing, we found that some users wanted to include additional columns 
  that were not used by the model, such as unused descriptors or predefined data splits. 
  However, these columns were sometimes inadvertently included in the evaluation 
  rather than excluded using the initially available "Ignore" option, leading to 
  misleading CheckML scores. Similarly, requiring other researchers to manually 
  exclude unused descriptors also increases the potential for reproducibility errors.
  |br| 
  For these reasons, we decided to **accept only the minimum data required to evaluate 
  a model.** This standardized format allows users to include the input CSV directly 
  in the Supporting Information alongside the CheckML PDF report.

For example:

.. csv-table::
   :header-rows: 1

   "Molecule_name", "Target_property", "descriptor_1", "descriptor_2", "...", "descriptor_n"
   "molecule_1", "1.25", "0.41", "2.7", "...", "3.2"
   "molecule_2", "0.82", "0.36", "1.9", "...", "2.5"
   "...", "...", "...", "...", "...", "..."
   "molecule_n", "0.98", "0.46", "1.7", "...", "4.1"

The column names above are illustrative and their order does not affect the results. Use
the column names expected by the model you are evaluating.

And with SMILES:

.. csv-table::
   :header-rows: 1

   "Molecule_name", "Target_property", "descriptor_1", "descriptor_2", "SMILES"
   "molecule_1", "1.25", "0.41", "2.7", "CCO"
   "molecule_2", "0.82", "0.36", "1.9", "CCN"

|br|

Data points included and user-defined data splits
+++++++++++++++++++++++++++++++++++++++++++++++++

As stated in the previous section, **do not specify data partitions in the input CSV.**

CheckML uses a standardized and systematic data-splitting strategy to evaluate models, 
rather than relying on user-defined training, validation, and test splits. This avoids 
potential inconsistencies caused by small differences in data partitioning, which 
can have a substantial impact on model evaluation, particularly for small datasets.

The input database provided to CheckML should therefore contain all data points 
used to train, validate, and test the original model. For example:

.. csv-table::
   :header-rows: 1

   "Original model split","Data points to include in CheckML"
   "80 training + 20 test","All 100 data points"
   "80 training + 20 validation + 20 test","All 120 data points"

For more information, see the :ref:`CheckML score section <robert-score-start>`.
