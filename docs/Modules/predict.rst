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

   *  Calculates R2, MAE and RMSE (for regression) or accuracy, F1 score and MCC (for classification) for the 10x 5-fold CV (training+validation) and test sets of the best No PFI and PFI models found in a GENERATE job.
   *  Predicts values for the test set or an external test set (if any). If the measured y values are included in the external test set, the model metrics from the previous point will be included, otherwise only the predicted y values will be retrieved. 
   *  Performs an outlier analysis.
   *  Performs a SHAP feature analysis.
   *  Performs a PFI feature analysis.

Technical information
+++++++++++++++++++++

The PREDICT module uses models obtained in the GENERATE module to compute various metrics, including R2, MAE, and RMSE (regression), and accuracy, F1 score, and MCC (classification). This module also enables predictions for an external test dataset, incorporating predictor metrics when measured y-values are available. In cases where measured y-values are absent, the module shows predicted y-values in the resulting PDF report and within the csv_test folder created inside the PREDICT main folder.
Furthermore, it conducts feature importance analysis through PFI and SHAP methods when ``predict_diagnostics`` is True and ``plot_verbosity`` is high enough to emit diagnostic figures (see CLI help / ``var_dict``). It also identifies outliers using the ``t_value`` option (default 2, in approximate SD units).

Prediction CSVs and the :doc:`Python API <../API/robert.api>` can include uncertainty columns alongside ``{y}_pred``:

* ``{y}_pred_sd`` — spread across repeated CV refits (overwritten by ``{y}_pred_uq_total`` when meta UQ is enabled).
* ``{y}_pred_conformal_hw`` — split-conformal half-width (regression; NaN for classification).
* ``{y}_pred_uq_model``, ``{y}_pred_uq_meta``, ``{y}_pred_uq_total`` — meta-model decomposition (opt-in).
* ``{y}_pred_uq_auto``, ``{y}_pred_uq_auto_source`` — auto-selected uncertainty (regression, opt-in).

Full semantics, configuration kwargs, and ``predict(..., return_uncertainty=...)`` modes are documented in :doc:`../API/robert.api`. This includes conformal knobs (``conformal_*``), meta-UQ knobs (``uq_enable_meta``, ``uq_top_k_models``, ``uq_model_weighting``), and auto-UQ knobs (``uq_auto_*``).

Example
+++++++

An example is available in **Examples/Use of individual modules**.

.. predict-modules-end
