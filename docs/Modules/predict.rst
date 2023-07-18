.. predict-modules-start

PREDICT
-------

Overview
++++++++

.. |predict_fig| image:: images/PREDICT.jpg
   :width: 600

.. centered:: |predict_fig|

Input required
++++++++++++++

This module uses a GENERATE folder created in a GENERATE job.

Automated protocols
+++++++++++++++++++

   *  Calculates R2, MAE and RMSE (for regression) or accuracy, F1 score and MCC (for classification) for the training and validation sets of the best No PFI and PFI models found in a GENERATE job.
   *  Predicts values for an external test set (if any). If the measured y values are included, the model metrics from the previous point will be included, otherwise only the predicted y values will be retrieved. 
   *  Performs an outlier analysis.
   *  Performs a SHAP feature analysis.
   *  Performs a PFI feature analysis.

Example
+++++++

An example is available in **Examples/Use of individual modules**.

.. predict-modules-end
