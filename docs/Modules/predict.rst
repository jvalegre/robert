.. predict-modules-start

New predictions and feature analysis
------------------------------------

Overview of the PREDICT module
++++++++++++++++++++++++++++++

.. |predict_fig| image:: images/PREDICT.jpg
   :width: 600

.. centered:: |predict_fig|

Input required
++++++++++++++

This module uses a GENERATE folder created in a GENERATE job.

Automated protocols
+++++++++++++++++++

   *  Calculates R2, MAE and RMSE (for regression) or accuracy, F1 score and MCC (for classification) for the training, validation and test sets of the best No PFI and PFI models found in a GENERATE job.
   *  Predicts values for an external test set (if any). If the measured y values are included, the model metrics from the previous point will be included, otherwise only the predicted y values will be retrieved. 
   *  Performs an outlier analysis.
   *  Performs a SHAP feature analysis.
   *  Performs a PFI feature analysis.

Technical information
+++++++

The PREDICT module uses models obtained in the GENERATE module to compute various metrics, including R2, MAE, and RMSE (regression), and accuracy, F1 score, and MCC (classification). This module also enables predictions for an external test dataset, incorporating predictor metrics when measured y-values are available. In cases where measured y-values are absent, the module shows predicted y-values in the resulting PDF report and within the csv_test folder created inside the PREDICT main folder.
Furthermore, this module conducts feature importance analysis through PFI and SHAP methods, which analyze how descriptors impact model performance. The PREDICT module also identifies outliers by measuring the absolute errors between predicted and measured y values. The detection of outliers is based on the “t_value” option, defaulted to two and measured in SD units. This default t-value identifies outliers in predictions exhibiting errors surpassing two SDs (approx. 5% of a normal population).

Example
+++++++

An example is available in **Examples/Use of individual modules**.

.. predict-modules-end
