CURATE (data curation)
----------------------

Overview
++++++++

.. |curate| image:: ../../Modules/images/CURATE.jpg
   :width: 600

.. centered:: |curate|

Required inputs
+++++++++++++++

.. |csv_FW| image:: ../images/csv_icon.jpg
   :target: ../../_static/Robert_example.csv
   :width: 30

* **Robert_example.csv:** CSV file with data to curate. The full CSV file can be found in the `Examples folder of the ROBERT repository <https://github.com/jvalegre/robert/tree/master/Examples/CSV_workflow>`__ or downloaded here: |csv_FW|

.. csv-table:: 
   :file: ../full_workflow/CSV/Robert_example.csv
   :header-rows: 1

Executing the job
+++++++++++++++++

**Instructions:**

1. First, go to the folder containing the CSV files in your terminal.
2. Run the following command line:

.. code:: shell

    python -m robert --names Name --y Target_values --csv_name Robert_example.csv --curate

**Options used:** 

* :code:`--names Name`: Name of the column containing the names of the datapoints.  

* :code:`--y Target_values`: Name of the column containing the response y values.  

* :code:`--csv_name Robert_example.csv`: CSV with the data to curate.  

* :code:`--curate`: Use only the CURATE module.  

Execution time
++++++++++++++

Time: ~5 seconds

System: 4 processors (Intel Xeon Ice Lake 8352Y) using 8.0 GB RAM memory

Results
+++++++

* A CSV file containing the curated database (Robert_example_CURATE.csv) should be created inside the CURATE folder. 
