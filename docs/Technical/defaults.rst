==================
Default parameters
==================

This documents details the default parameters used in the ROBERT program.

.. contents::
   :local:

AQME
----

.. automodule:: robert.aqme
   :noindex:

CURATE 
------

.. automodule:: robert.curate
   :noindex:

GENERATE
--------

.. automodule:: robert.generate
   :noindex:

PREDICT
-------

.. automodule:: robert.predict
   :noindex:

VERIFY
------

.. automodule:: robert.verify
   :noindex:

REPORT
------

.. automodule:: robert.report
   :noindex:

Uncertainty and model selection
-------------------------------

Defaults below are defined in ``robert.argument_parser.var_dict``. Full semantics,
output columns, and ``predict(..., return_uncertainty=...)`` modes are in
:doc:`../API/robert.api`.

.. list-table::
   :header-rows: 1
   :widths: 28 72

   * - Parameter
     - Default / notes
   * - ``model``
     - ``["RF", "GB", "NN", "MVL"]`` (reg) or ``["RF", "GB", "NN", "AdaB"]`` (clas); add ``"XGB"`` explicitly
   * - ``conformal_enable``
     - ``True`` (regression split-conformal half-width column)
   * - ``conformal_calib_frac``
     - ``0.15``
   * - ``conformal_coverage``
     - ``0.9``
   * - ``uq_enable_meta``
     - ``False``
   * - ``uq_top_k_models``
     - ``3``
   * - ``uq_model_weighting``
     - ``"score_weighted"`` (or ``"uniform"``)
   * - ``uq_auto_enable``
     - ``False`` (regression auto selection)
   * - ``uq_auto_candidates``
     - ``["cv_sd", "conformal", "meta_total"]``
   * - ``uq_auto_scaler``
     - ``"global_multiplicative"`` (also ``"none"``, ``"isotonic"``)
   * - ``uq_auto_metric_weights``
     - ``coverage: 1.0``, ``sharpness: 0.25``, ``nll: 0.5``
   * - ``uq_auto_min_samples``
     - ``12``
   * - ``uq_auto_random_state``
     - ``0``
   * - ``uq_auto_clas_mode``
     - ``"error"`` (raises if auto UQ requested for classification)
   * - ``predict_diagnostics``
     - ``True`` (SHAP, PFI, plots; ``RobertModel.predict`` forces ``False``)
   * - ``plot_verbosity``
     - ``2`` (higher → more diagnostic figures when diagnostics enabled)
