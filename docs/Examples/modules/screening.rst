GENERATE (screening of ML models)
---------------------------------

Overview
++++++++

.. |generate| image:: ../../Modules/images/GENERATE.jpg
   :width: 600

.. centered:: |generate|

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
