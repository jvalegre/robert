PREDICT (predict external test set and feature importance analysis)
-------------------------------------------------------------------

Overview
++++++++

.. |predict| image:: ../../Modules/images/PREDICT.jpg
   :width: 600

.. centered:: |predict|

Required inputs
+++++++++++++++

.. |csv_FW_test| image:: ../images/csv_icon.jpg
   :target: ../../_static/Robert_example_test.csv
   :width: 30

* Previous folder from a GENERATE job.
* **Robert_example_test.csv:** CSV file with data to use as the external test set. The full CSV file can be found in the `Examples folder of the ROBERT repository <https://github.com/jvalegre/robert/tree/master/Examples/CSV_workflow>`__ or downloaded here: |csv_FW_test|

.. csv-table:: 
   :file: ../full_workflow/CSV/Robert_example_test.csv
   :header-rows: 1

Executing the job
+++++++++++++++++

**Instructions:**

1. First, go to the folder where GENERATE was previously run in your terminal. You should see a folder called GENERATE on it.
2. Run the following command line:

.. code:: shell

    python -m robert --csv_test Robert_example_test.csv --names Name --predict

**Options used:**

* :code:`--csv_test Robert_example_test.csv`: CSV with the external test set.  

* :code:`--names Name`: Name of the column containing the names of the datapoints. This feature allows to print the names of the outlier points (if any).  

* :code:`--predict`: Use only the PREDICT module.  

Execution time
++++++++++++++

Time: ~10 seconds

System: 4 processors (Intel Xeon Ice Lake 8352Y) using 8.0 GB RAM memory

Results
+++++++

* Two graphs, for No_PFI and for PFI (in /PREDICT), with: representation of predictions, SHAP feature analysis, PFI feature analysis and outlier analysis .
* Six CSV files with the predictions of each set, for No_PFI and for PFI (in /PREDICT).
