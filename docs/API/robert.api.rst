Python API (sklearn-style)
==========================

:class:`~robert.api.RobertModel` runs the full ROBERT workflow (CURATE, GENERATE,
VERIFY, PREDICT) and exposes ``fit`` / ``predict`` / ``score`` on
:class:`pandas.DataFrame` or :class:`numpy.ndarray` inputs. PREDICT writes CSV
columns aligned with the pipeline:

- ``{y}_pred``: point prediction from the **selected estimator refit on all training
  data** (deployment-style mean), or a **weighted average** / **weighted vote** when
  meta-model uncertainty is enabled (see below).
- ``{y}_pred_sd``: per-row **standard deviation across repeated cross-validation
  predictions** (disagreement between refits on overlapping training folds; related
  to epistemic instability, not a calibrated predictive distribution). With meta UQ
  enabled, this column is set to ``{y}_pred_uq_total``.
- ``{y}_pred_conformal_hw`` (**regression only**): a single **symmetric interval
  half-width** from split-style conformal calibration (absolute residuals on a
  held-out calibration slice of the training set when large enough, otherwise
  residuals vs CV out-of-fold means). Because the reported point predictor is
  refit on **all** training data, finite-sample **coverage is approximate** at the
  nominal ``conformal_coverage`` (default 0.9); tune with ``conformal_enable``,
  ``conformal_calib_frac``, and ``conformal_coverage`` in :class:`~robert.api.RobertModel`
  kwargs. For **classification**, this column is present but filled with NaN;
  ``{y}_pred_sd`` reflects **vote spread** across CV refits, not class probabilities.
- ``{y}_pred_uq_model`` (**meta UQ, opt-in**): within-model component (mean CV spread
  across the top-k candidates; regression uses variance decomposition).
- ``{y}_pred_uq_meta`` (**meta UQ, opt-in**): between-model component (spread of
  top-k point predictions).
- ``{y}_pred_uq_total`` (**meta UQ, opt-in**): combined uncertainty (regression:
  :math:`\sqrt{\mathrm{E}[\sigma^2] + \mathrm{Var}(\hat y)}`; classification uses a
  heuristic combining vote spread and between-model disagreement).
- ``{y}_pred_uq_auto`` (**auto UQ, regression, opt-in**): calibrated sigma-like scale
  from automatic candidate selection (see **Auto uncertainty** below).
- ``{y}_pred_uq_auto_source`` (**auto UQ, opt-in**): name of the selected candidate
  (``cv_sd``, ``conformal``, or ``meta_total``).

``predict`` returns ``{y}_pred`` values aligned to input rows. Uncertainty:

- ``return_std=True`` is equivalent to ``return_uncertainty="cv_sd"`` and returns
  ``(y, sd_cv)``.
- ``return_uncertainty="conformal"`` (**regression only**) returns ``(y, half_width)``.
- ``return_uncertainty="both"`` (**regression only**) returns ``(y, sd_cv, half_width)``.
- ``return_uncertainty="meta"`` returns ``(y, uq_meta)`` (requires ``uq_enable_meta=True``).
- ``return_uncertainty="total"`` returns ``(y, uq_total)`` (requires ``uq_enable_meta=True``).
- ``return_uncertainty="decomposed"`` returns ``(y, uq_model, uq_meta, uq_total)``
  (requires ``uq_enable_meta=True``).
- ``return_uncertainty="auto"`` (**regression only**) enables auto uncertainty for
  that predict call and returns ``(y, uq_auto)`` from ``{y}_pred_uq_auto``.
- ``return_uncertainty="auto_decomposed"`` (**regression only**) returns
  ``(y, uq_auto, metadata)`` where ``metadata`` is loaded from
  ``PREDICT/uq_auto_metadata.json`` when present.
- If both ``return_std`` and ``return_uncertainty`` are set, ``return_uncertainty``
  wins and a warning is issued.

Supported models
----------------

Pass ``model`` as a list of algorithm codes (same as the CLI ``--model`` option).
Defaults are ``["RF", "GB", "NN", "MVL"]`` for regression and ``["RF", "GB", "NN", "AdaB"]``
for classification (when ``auto_type`` switches the problem type).

.. list-table::
   :header-rows: 1
   :widths: 20 80

   * - Code
     - Backend
   * - RF, GB, NN, MVL
     - scikit-learn (default screening set for regression includes MVL instead of AdaB)
   * - GP, AdaB, VR
     - scikit-learn (opt-in; AdaB replaces MVL in the default classification set)
   * - **XGB**
     - XGBoost (:class:`~xgboost.XGBRegressor` / :class:`~xgboost.XGBClassifier`), opt-in;
       hyperoptimized with in-code Bayesian bounds (no packaged ``model_params/XGB_params.yaml``)

Example with XGB:

.. code-block:: python

   model_xgb = RobertModel(
       problem_type="reg",
       workdir="./robert_run_xgb",
       model=["RF", "XGB"],
       n_iter=2,
       init_points=2,
   )
   model_xgb.fit(X_train, y_train)

Configuration (uncertainty kwargs)
------------------------------------

Defaults are defined in ``robert.argument_parser.var_dict``. Every key in that
dictionary can be passed as a :class:`~robert.api.RobertModel` keyword argument or
set in a YAML varfile (``varfile=FILE.yaml``). For XGBoost, install-time dependency
availability (``xgboost``) is distinct from runtime model selection: include
``"XGB"`` in ``model`` to screen XGBoost.

**CLI vs API / YAML.** ``python -m robert --help`` documents ``conformal_enable``,
``conformal_calib_frac``, ``conformal_coverage``, ``uq_enable_meta``,
``uq_top_k_models``, and ``uq_auto_enable``. Set ``uq_model_weighting`` and the
remaining ``uq_auto_*`` keys (candidates, scaler, metric weights, min samples,
random state, clas mode) via a YAML varfile or :class:`~robert.api.RobertModel`
keyword arguments (see list below).

- **Conformal:** ``conformal_enable`` (``True``), ``conformal_calib_frac`` (``0.15``),
  ``conformal_coverage`` (``0.9``).
- **Meta-model:** ``uq_enable_meta`` (``False``), ``uq_top_k_models`` (``3``),
  ``uq_model_weighting`` (``"score_weighted"`` or ``"uniform"``).
- **PREDICT diagnostics:** ``predict_diagnostics`` (``True``). When ``False``, PREDICT
  skips SHAP, PFI, Pearson heatmap, outlier, and distribution plots (and y-vs-pred
  graphs). :class:`~robert.api.RobertModel.predict` sets this to ``False`` automatically.
- **Plot verbosity:** ``plot_verbosity`` (``2``). Higher values emit more diagnostic
  figures during PREDICT when ``predict_diagnostics`` is ``True`` (see CLI help / ``var_dict``).
- **Auto (regression):** ``uq_auto_enable`` (``False``),
  ``uq_auto_candidates`` (``["cv_sd", "conformal", "meta_total"]``),
  ``uq_auto_scaler`` (``"global_multiplicative"``; also ``"none"`` or ``"isotonic"``),
  ``uq_auto_metric_weights`` (default ``{"coverage": 1.0, "sharpness": 0.25, "nll": 0.5}``),
  ``uq_auto_min_samples`` (``12``), ``uq_auto_random_state`` (``0``),
  ``uq_auto_clas_mode`` (``"error"`` — raises if auto is requested for classification).

Meta-model uncertainty
----------------------

Enable with ``uq_enable_meta=True`` on :class:`~robert.api.RobertModel`. PREDICT
re-runs up to ``uq_top_k_models`` estimators ranked by GENERATE
``combined_{error_type}`` scores in ``GENERATE/Raw_data``, then combines predictions
with ``uq_model_weighting``. **Regression** uses a weighted mean and law-of-total-variance
decomposition. **Classification** uses a weighted vote and heuristic uncertainty
components (not class probabilities). If Raw_data has no candidates, CV spread is
used with zero meta component and a warning is issued.

Example (meta UQ):

.. code-block:: python

   model_meta = RobertModel(
       problem_type="reg",
       workdir="./robert_run_meta",
       model=["RF", "GB"],
       uq_enable_meta=True,
       uq_top_k_models=2,
   )
   model_meta.fit(X.iloc[:25], y.iloc[:25])
   y_meta, uq_between = model_meta.predict(X.iloc[25:], return_uncertainty="meta")
   y_dec, uq_m, uq_b, uq_t = model_meta.predict(
       X.iloc[25:], return_uncertainty="decomposed"
   )

Auto uncertainty (Bayesian optimization)
----------------------------------------

For uncertainty-aware acquisition (e.g. expected improvement with a surrogate
variance), use ``return_uncertainty="auto"`` on **regression** tasks. Auto mode
scores candidates ``cv_sd``, ``conformal``, and ``meta_total`` (when available) on
training out-of-fold absolute residuals, fits an optional scaler (``uq_auto_scaler``),
and writes ``{y}_pred_uq_auto``. Enable with ``uq_auto_enable=True``, or rely on
``return_uncertainty="auto"`` / ``"auto_decomposed"`` to enable it per predict call.
Legacy columns ``{y}_pred_sd`` and conformal half-width are unchanged when auto mode
runs. Lower-level helpers live in :mod:`robert.uq_auto`.

Example (auto UQ):

.. code-block:: python

   model_auto = RobertModel(
       problem_type="reg",
       workdir="./robert_run_auto",
       model=["RF"],
       uq_auto_enable=True,
   )
   model_auto.fit(X.iloc[:25], y.iloc[:25])
   y_bo, sigma = model_auto.predict(X.iloc[25:], return_uncertainty="auto")
   y_bo2, sigma2, meta = model_auto.predict(
       X.iloc[25:], return_uncertainty="auto_decomposed"
   )

Pipeline semantics
------------------

- **Single high-level estimator.** Encoding and curation happen in CURATE; training
  matrices are scaled inside ROBERT (``StandardScaler`` on the design matrix in
  ``prepare_sets``). This is not a composable sklearn ``Pipeline`` of separate
  ``TransformerMixin`` steps on :class:`~robert.api.RobertModel` itself.
- **Do not** stack another ``StandardScaler`` (or similar) in front of the same raw
  descriptor table unless you know exactly how it interacts with CURATE outputs;
  you would usually double-scale or break column semantics.
- **Row order.** ``predict`` returns one value per input row, aligned to ``X`` even if
  ROBERT writes prediction CSVs in a different row order (alignment uses the names
  column from CURATE).

Matplotlib
----------

During ``fit`` and ``predict``, Matplotlib is switched to the non-interactive ``Agg``
backend so plotting does not require a GUI; the prior backend is restored afterward.
Figures are still written under ``workdir`` like the CLI workflow.

Example
-------

.. code-block:: python

   from robert import RobertModel
   import pandas as pd

   df = pd.read_csv("Robert_example.csv")
   X = df.drop(columns=["Target_values"])
   y = df["Target_values"]

   model = RobertModel(
       problem_type="reg",
       workdir="./robert_run",
       model=["RF", "XGB"],
       n_iter=2,
       init_points=2,
       conformal_enable=True,
   )
   model.fit(X.iloc[:25], y.iloc[:25])
   preds = model.predict(X.iloc[25:])
   preds, sd_cv = model.predict(X.iloc[25:], return_std=True)
   preds2, hw = model.predict(X.iloc[25:], return_uncertainty="conformal")
   preds3, sd_cv2, hw2 = model.predict(X.iloc[25:], return_uncertainty="both")
   r2 = model.score(X.iloc[25:], y.iloc[25:])

.. autoclass:: robert.api.RobertModel
   :members: fit, predict, score, get_params, set_params
   :no-inherited-members:
