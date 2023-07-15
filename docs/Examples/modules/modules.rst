Use of individual modules
=========================

Data curation
-------------

Required inputs
+++++++++++++++

* **Robert_example.csv:** CSV file with data to curate. The full CSV file can be found in the `Examples folder of the ROBERT repository <https://github.com/jvalegre/robert/tree/master/Examples/CSV_workflow>`__.

.. csv-table:: 
   :file: ../full_workflow/CSV/Robert_example.csv
   :header-rows: 1

Executing the job
+++++++++++++++++

**Instructions:**

1. First, go to the folder containing the CSV files in your terminal.
2. Run the following command line:

.. code:: shell

    python -m robert --ignore "[Name]" --y Target_values --csv_name Robert_example.csv --curate

**Options used:**

* :code:`--ignore "[Name]"`: Variables ignored in the model. In this case, the column 'Name' that contains the names of the datapoints, which is not included in the model. Quotation marks are included in "[Name]" to avoid problems when using lists in the command line. More variables can be incuded as "[VAR1,VAR2,VAR3...]". 

* :code:`--y Target_values`: Name of the column containing the response y values.  

* :code:`--csv_name Robert_example.csv`: CSV with the data to curate.  

* :code:`--curate`: Use only the CURATE module.  

Execution time
++++++++++++++

Time: ~1 second

System: 4 processors (Intel Xeon Ice Lake 8352Y) using 8.0 GB RAM memory

Results
+++++++

* A CSV file containing the curated database (Robert_example_CURATE.csv) should be created inside the CURATE folder. 


Screening of models
-------------------

Required inputs
+++++++++++++++

* **Robert_example_CURATE.csv:** Curated CSV file with data to use as the training and validation sets. It is created in a previous CURATE job.

.. csv-table:: 
   :file: Robert_example_CURATE.csv
   :header-rows: 1

Executing the job
+++++++++++++++++

**Instructions:**

1. First, go to the folder where CURATE was previously run in your terminal. You should see a folder called CURATE on it.
2. Run the following command line:

.. code:: shell

    python -m robert --csv_name CURATE/Robert_example_CURATE.csv --generate

**Options used:**

* :code:`--csv_name CURATE/Robert_example_CURATE.csv`: CSV with the curated database.  

* :code:`--generate`: Use only the GENERATE module.  

Execution time
++++++++++++++

Time: ~2 min

System: 4 processors (Intel Xeon Ice Lake 8352Y) using 8.0 GB RAM memory

Results
+++++++

* Four CSV files for each combination of ML model/training size (in /GENERATE/Raw_data). Half of the CSVs relate to models with all the variables (No_PFI folder) and the other half for models that use only the msot important features based on PFI (PFI folder). For each pair of CSVs, one contains the parameters of the model and the other contains the database already split into training/validation sets (_db suffix).
* The two best models, for No_PFI (all descriptors) and for PFI (only important descriptors), are stored in /GENERATE/Best_model.
* Two heatmaps with a summary of the results for all the models, created in /GENERATE/Raw_data. 


Verification of predictive ability
----------------------------------

Required inputs
+++++++++++++++

* Previous folder from a GENERATE job.

Executing the job
+++++++++++++++++

**Instructions:**

1. First, go to the main folder where GENERATE was run in your terminal. You should see a folder called GENERATE on it.
2. Run the following command line:

.. code:: shell

    python -m robert --verify

**Options used:**

* :code:`--verify`: Use only the VERIFY module.  

Execution time
++++++++++++++

Time: ~2 seconds

System: 4 processors (Intel Xeon Ice Lake 8352Y) using 8.0 GB RAM memory

Results
+++++++

* Two donut plots (for No_PFI and PFI) with a summary of the results of the four verification tests, created in /VERIFY. 


Predict external test set
-------------------------

Required inputs
+++++++++++++++

* Previous folder from a GENERATE job.
* **Robert_example_test.csv:** CSV file with data to use as the external test set. The full CSV file can be found in the `Examples folder of the ROBERT repository <https://github.com/jvalegre/robert/tree/master/Examples/CSV_workflow>`__.

.. csv-table:: 
   :file: ../full_workflow/CSV/Robert_example_test.csv
   :header-rows: 1

Executing the job
+++++++++++++++++

**Instructions:**

1. First, go to the folder where GENERATE was previously run in your terminal. You should see a folder called GENERATE on it.
2. Run the following command line:

.. code:: shell

    python -m robert --csv_test Robert_example_test.csv --predict

**Options used:**

* :code:`--csv_test Robert_example_test.csv`: CSV with the external test set.  

* :code:`--predict`: Use only the PREDICT module.  

Execution time
++++++++++++++

Time: ~10 seconds

System: 4 processors (Intel Xeon Ice Lake 8352Y) using 8.0 GB RAM memory

Results
+++++++

* Two graphs, for No_PFI and for PFI (in /PREDICT), with: representation of predictions, SHAP feature analysis, PFI feature analysis and outlier analysis .
* Six CSV files with the predictions of each set, for No_PFI and for PFI (in /PREDICT).
