Python API (sklearn-style)
==========================

:class:`~robert.api.RobertModel` runs the full ROBERT workflow (CURATE, GENERATE,
VERIFY, PREDICT) and exposes ``fit`` / ``predict`` / ``score`` on
:class:`pandas.DataFrame` or :class:`numpy.ndarray` inputs. PREDICT writes CSV
columns aligned with the pipeline:

- ``{y}_pred``: point prediction from the **selected estimator refit on all training
  data** (deployment-style mean).
- ``{y}_pred_sd``: per-row **standard deviation across repeated cross-validation
  predictions** (disagreement between refits on overlapping training folds; related
  to epistemic instability, not a calibrated predictive distribution).
- ``{y}_pred_conformal_hw`` (**regression only**): a single **symmetric interval
  half-width** from split-style conformal calibration (absolute residuals on a
  held-out calibration slice of the training set when large enough, otherwise
  residuals vs CV out-of-fold means). Because the reported point predictor is
  refit on **all** training data, finite-sample **coverage is approximate** at the
  nominal ``conformal_coverage`` (default 0.9); tune with ``conformal_enable``,
  ``conformal_calib_frac``, and ``conformal_coverage`` in :class:`~robert.api.RobertModel`
  kwargs. For **classification**, this column is present but filled with NaN;
  ``{y}_pred_sd`` reflects **vote spread** across CV refits, not class probabilities.

``predict`` returns ``{y}_pred`` values aligned to input rows. Uncertainty:

- ``return_std=True`` is equivalent to ``return_uncertainty="cv_sd"`` and returns
  ``(y, sd_cv)``.
- ``return_uncertainty="conformal"`` (regression only) returns ``(y, half_width)``.
- ``return_uncertainty="both"`` returns ``(y, sd_cv, half_width)``.
- If both ``return_std`` and ``return_uncertainty`` are set, ``return_uncertainty``
  wins and a warning is issued.

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
       model=["RF"],
       n_iter=2,
       init_points=2,
   )
   model.fit(X.iloc[:25], y.iloc[:25])
   preds = model.predict(X.iloc[25:])
   preds, sd_cv = model.predict(X.iloc[25:], return_std=True)
   preds2, hw = model.predict(X.iloc[25:], return_uncertainty="conformal")
   preds3, sd_cv2, hw2 = model.predict(X.iloc[25:], return_uncertainty="both")
   r2 = model.score(X.iloc[25:], y.iloc[25:])

.. autoclass:: robert.api.RobertModel
   :members:
   :no-inherited-members:
