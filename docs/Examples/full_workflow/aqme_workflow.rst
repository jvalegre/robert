Full workflow from SMILES
=========================

Required inputs
+++++++++++++++

* **solubility_short.csv:** CSV file with SMILES to generate descriptors that will be used as the training and validation sets. The full CSV file can be found in the `Examples folder of the ROBERT repository <https://github.com/jvalegre/robert/tree/master/Examples/AQME_workflow>`__.

.. csv-table:: 
   :file: CSV/solubility_short.csv
   :header-rows: 1

Executing the job
+++++++++++++++++

**Instructions:**

1. First, go to the folder containing the CSV files in your terminal.
2. Run the following command line:

.. code:: shell

    python -m robert --aqme --ignore "[code_name,smiles]" --names code_name --y solubility --csv_name solubility_short.csv

**Options used:**

* :code:`--aqme"`: Calls the AQME module to convert SMILES into RDKit and xTB descriptors, retrieving a new CSV database. 

* :code:`--ignore "[code_name,smiles]"`: Variables ignored in the model. In this case, the columns 'code_name' and 'smiles' contain the names of the datapoints and their SMILES strings, which are not included in the model. Quotation marks are included in "[code_name,smiles]" to avoid problems when using lists in the command line. More variables can be incuded as "[VAR1,VAR2,VAR3...]". 

* :code:`--names code_name`: Name of the column containing the names of the datapoints. This is an optional feature that allows to print the names of the outlier points.  

* :code:`--y solubility`: Name of the column containing the response y values.  

* :code:`--csv_name solubility_short.csv`: CSV with the SMILES strings.  

Execution time
++++++++++++++

XXXXXXX Time: XX min
System: 8 processors (2x Intel Xeon Ice Lake 8352Y) with 16.0 GB RAM memory

Results
+++++++

**Initial AQME workflow**

XXXXXXX CSEARCH CONFORMER SAMPLING (say that conformational sampling with RDKit generates XX SDF files created in /AQME/CSEARCH)
XXXXXXX QDESCP DESCRIPTOR GENERATOR (say that the XX.csv file with XX+ RDKit and XX+ xTB Boltzmann-averaged descriptors is created in /AQME/QDESCP)
XXXXXXX END UP WITH CREATION OF AQME-ROBERT_XXXX.CSV file in the main folder where you call ORBERT

**Following ROBERT workflow**

.. |pdf_report_test| image:: ../images/pdf_icon.jpg
   :target: ../../_static/ROBERT_report_aqme.pdf
   :width: 30

A PDF file called ROBERT_report.pdf should be created in the folder where ROBERT was executed. The PDF 
file can be visualized here: |pdf_report_test|

XXXXXXX The PDF report contains all the results of the workflow. In this case, a XXX (NN) model with XXX% of the data used as the training set was the optimal model found from the combinations of four different models (Gradient Boosting GB, MultiVariate Linear MVL, Neural Network NN, Random Forest RF) with four different partition sizes (60%, 70%, 80%, 90%). This information is summarized below:

.. |CURATE_data| image:: ../images/AQME/CURATE_data.jpg
   :width: 600

.. |Person_heatmap| image:: ../images/AQME/Pearson_heatmap.png
   :width: 400

.. |GENERATE_data| image:: ../images/AQME/GENERATE_data.jpg
   :width: 600

.. |heatmap_no_pfi| image:: ../images/AQME/heatmap_no_pfi.png
   :width: 400

.. |heatmap_pfi| image:: ../images/AQME/heatmap_pfi.png
   :width: 400

.. |VERIFY_dat_no_pfi| image:: ../images/AQME/VERIFY_dat_no_pfi.jpg
   :width: 600

.. |VERIFY_no_pfi| image:: ../images/AQME/VERIFY_no_pfi.png
   :width: 600

.. |VERIFY_dat_pfi| image:: ../images/AQME/VERIFY_dat_pfi.jpg
   :width: 600

.. |VERIFY_pfi| image:: ../images/AQME/VERIFY_pfi.png
   :width: 600

.. |PREDICT_res_no_pfi| image:: ../images/AQME/PREDICT_res_no_pfi.jpg
   :width: 600

.. |PREDICT_graph_no_pfi| image:: ../images/AQME/PREDICT_graph_no_pfi.png
   :width: 600

.. |PREDICT_res_pfi| image:: ../images/AQME/PREDICT_res_pfi.jpg
   :width: 600

.. |PREDICT_graph_pfi| image:: ../images/AQME/PREDICT_graph_pfi.png
   :width: 600

.. |PREDICT_shap_dat_no_pfi| image:: ../images/AQME/PREDICT_shap_dat_no_pfi.jpg
   :width: 600

.. |PREDICT_shap_no_pfi| image:: ../images/AQME/PREDICT_shap_no_pfi.png
   :width: 600

.. |PREDICT_shap_dat_pfi| image:: ../images/AQME/PREDICT_shap_dat_pfi.jpg
   :width: 600

.. |PREDICT_shap_pfi| image:: ../images/AQME/PREDICT_shap_pfi.png
   :width: 600

.. |PREDICT_out_dat_no_pfi| image:: ../images/AQME/PREDICT_out_dat_no_pfi.jpg
   :width: 600

.. |PREDICT_out_no_pfi| image:: ../images/AQME/PREDICT_out_no_pfi.png
   :width: 600

.. |PREDICT_out_dat_pfi| image:: ../images/AQME/PREDICT_out_dat_pfi.jpg
   :width: 600

.. |PREDICT_out_pfi| image:: ../images/AQME/PREDICT_out_pfi.png
   :width: 600

+---------------------------------------------------------------------------------------------------+
|                                                                                                   |
|                         .. centered:: **RESULTS**                                                 |
|                                                                                                   |
+---------------------------------------------------------------------------------------------------+
|            |                                                                                      |
|  .. centered:: /CURATE folder                                                                     |
|                                                                                                   |
+-------------------------------------------------------------+-------------------------------------+
|  .. centered:: CURATE_data.dat                              |    |CURATE_data|                    |
+-------------------------------------------------------------+-------------------------------------+
|  .. centered:: Person_heatmap.png                           |    |Person_heatmap|                 |
+-------------------------------------------------------------+-------------------------------------+
|            |                                                                                      |
|  .. centered:: /GENERATE folder                                                                   |
|                                                                                                   |
+-------------------------------------------------------------+-------------------------------------+
|  .. centered:: GENERATE_data.dat                            |    |GENERATE_data|                  |
+-------------------------------------------------------------+-------------------------------------+
|  .. centered:: Heatmap ML models no                         |    |heatmap_no_pfi|                 |
|  .. centered:: PFI filter.png                               |                                     |
+-------------------------------------------------------------+-------------------------------------+
|  .. centered:: Heatmap ML models with                       |    |heatmap_pfi|                    |
|  .. centered:: PFI filter.png                               |                                     |
+-------------------------------------------------------------+-------------------------------------+
|            |                                                                                      |
|  .. centered:: /VERIFY folder                                                                     |
|                                                                                                   |
+-------------------------------------------------------------+-------------------------------------+
|  .. centered:: VERIFY_tests_NN_80_No_PFI.dat                |    |VERIFY_dat_no_pfi|              |
|  .. centered:: *(using 12 descriptors)*                     |                                     |
+-------------------------------------------------------------+-------------------------------------+
|  .. centered:: VERIFY_tests_NN_80_No_PFI.png                |    |VERIFY_no_pfi|                  |
|  .. centered:: *(using 12 descriptors)*                     |                                     |
+-------------------------------------------------------------+-------------------------------------+
|  .. centered:: VERIFY_tests_NN_80_PFI.dat                   |    |VERIFY_dat_pfi|                 |
|  .. centered:: *(PFI filter applied, using 7 descriptors)*  |                                     |
+-------------------------------------------------------------+-------------------------------------+
|  .. centered:: VERIFY_tests_NN_80_PFI.png                   |    |VERIFY_pfi|                     |
|  .. centered:: *(PFI filter applied, using 7 descriptors)*  |                                     |
+-------------------------------------------------------------+-------------------------------------+
|            |                                                                                      |
|  .. centered:: /PREDICT folder                                                                    |
|                                                                                                   |
+-------------------------------------------------------------+-------------------------------------+
|  .. centered:: Results_NN_80_No_PFI.dat                     |    |PREDICT_res_no_pfi|             |
|  .. centered:: *(using 12 descriptors)*                     |                                     |
+-------------------------------------------------------------+-------------------------------------+
|  .. centered:: Results_NN_80_No_PFI.png                     |    |PREDICT_graph_no_pfi|           |
|  .. centered:: *(using 12 descriptors)*                     |                                     |
+-------------------------------------------------------------+-------------------------------------+
|  .. centered:: SHAP_NN_80_No_PFI.dat                        |    |PREDICT_shap_dat_no_pfi|        |
|  .. centered:: *(using 12 descriptors)*                     |                                     |
+-------------------------------------------------------------+-------------------------------------+
|  .. centered:: SHAP_NN_80_No_PFI.png                        |    |PREDICT_shap_no_pfi|            |
|  .. centered:: *(using 12 descriptors)*                     |                                     |
+-------------------------------------------------------------+-------------------------------------+
|  .. centered:: Outliers_NN_80_No_PFI.dat                    |    |PREDICT_out_dat_no_pfi|         |
|  .. centered:: *(using 12 descriptors)*                     |                                     |
+-------------------------------------------------------------+-------------------------------------+
|  .. centered:: Outliers_NN_80_No_PFI.png                    |    |PREDICT_out_no_pfi|             |
|  .. centered:: *(using 12 descriptors)*                     |                                     |
+-------------------------------------------------------------+-------------------------------------+
|  .. centered:: Results_NN_80_PFI.dat                        |    |PREDICT_res_pfi|                |
|  .. centered:: *(PFI filter applied, using 7 descriptors)*  |                                     |
+-------------------------------------------------------------+-------------------------------------+
|  .. centered:: Results_NN_80_PFI.png                        |    |PREDICT_graph_pfi|              |
|  .. centered:: *(PFI filter applied, using 7 descriptors)*  |                                     |
+-------------------------------------------------------------+-------------------------------------+
|  .. centered:: SHAP_NN_80_PFI.dat                           |    |PREDICT_shap_dat_pfi|           |
|  .. centered:: *(PFI filter applied, using 7 descriptors)*  |                                     |
+-------------------------------------------------------------+-------------------------------------+
|  .. centered:: SHAP_NN_80_PFI.png                           |    |PREDICT_shap_pfi|               |
|  .. centered:: *(PFI filter applied, using 7 descriptors)*  |                                     |
+-------------------------------------------------------------+-------------------------------------+
|  .. centered:: Outliers_NN_80_PFI.dat                       |    |PREDICT_out_dat_pfi|            |
|  .. centered:: *(PFI filter applied, using 7 descriptors)*  |                                     |
+-------------------------------------------------------------+-------------------------------------+
|  .. centered:: Outliers_NN_80_PFI.png                       |    |PREDICT_out_pfi|                |
|  .. centered:: *(PFI filter applied, using 7 descriptors)*  |                                     |
+-------------------------------------------------------------+-------------------------------------+



