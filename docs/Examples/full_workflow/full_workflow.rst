Full workflow from CSV
======================

Overview
++++++++

.. |fullworkflow_fig| image:: ../../Modules/images/FullWorkflow.jpg
   :width: 600

.. centered:: |fullworkflow_fig|

Required inputs
+++++++++++++++

.. |csv_FW| image:: ../images/csv_icon.jpg
   :target: ../../_static/Robert_example.csv
   :width: 30

* **Robert_example.csv:** CSV file with data to use as the training and validation sets. The full CSV file can be 
  found in the `"Examples" folder of the ROBERT repository <https://github.com/jvalegre/robert/tree/master/Examples/CSV_workflow>`__ 
  or downloaded here: |csv_FW|

.. csv-table:: 
   :file: CSV/Robert_example.csv
   :header-rows: 1

Executing the job
+++++++++++++++++

**Instructions:**

1. Download the **Robert_example.csv** file specified in Required inputs.
2. Go to the folder containing the CSV file in your terminal (using the "cd" command, i.e. :code:`cd C:/Users/test_robert`).
3. Activate the conda environment where ROBERT was installed (:code:`conda activate robert`).
4. Run the following command line:

.. code:: shell

   python -m robert --ignore "[Name]" --names Name --y Target_values --csv_name Robert_example.csv

**Options used:**

* :code:`--ignore "[Name]"`: Variables ignored in the model. In this case, the column 'Name' that contains the names of the datapoints, which is not included in the model. Quotation marks are included in "[Name]" to avoid problems when using lists in the command line. More variables can be incuded as "[VAR1,VAR2,VAR3...]". 

* :code:`--names Name`: Name of the column containing the names of the datapoints. This feature allows to print the names of the outlier points (if any).  

* :code:`--y Target_values`: Name of the column containing the response y values.  

* :code:`--csv_name Robert_example.csv`: CSV with the database.   

Execution time and versions
+++++++++++++++++++++++++++

Time: ~1.5 min

System: 4 processors (Intel Xeon Ice Lake 8352Y) using 8.0 GB RAM memory

ROBERT version: 1.2.0

scikit-learn-intelex version: 2024.5.0

Results
+++++++

.. |pdf_report_test| image:: ../images/pdf_icon.jpg
   :target: ../../_static/ROBERT_report.pdf
   :width: 30

A PDF file called **ROBERT_report.pdf** should be created in the folder where ROBERT was executed. The PDF file can be visualized here: |pdf_report_test|

The PDF report contains all the results of the workflow. In this case, two Neural Network (NN) models with 80% training sizes were the optimal model found from: 

   * Four different models (Gradient Boosting GB, MultiVariate Linear MVL, Neural Network NN, Random Forest RF) 
   * Three different partition sizes (60%, 70%, 80%) 

The first part of the PDF file is shown below as a preview:

.. |pdf_preview| image:: ../images/FW/preview.png
   :width: 400

|pdf_preview|