.. evaluate-modules-start

Evaluate a pre-specified model
------------------------------

Overview of the EVALUATE module
+++++++++++++++++++++++++++++++

The EVALUATE module skips GENERATE model screening. It prepares the
``GENERATE/Best_model/No_PFI`` folder with a user-chosen sklearn model so later
modules (VERIFY, PREDICT, REPORT) can run as in a standard workflow.

Input required
++++++++++++++

A curated CSV (typically from CURATE) with descriptors and a target column ``y``.

Automated protocols
+++++++++++++++++++

*  Loads and standardizes the database (same path as GENERATE).
*  Writes ``GENERATE/Best_model/No_PFI/{eval_model}.csv`` (model metadata) and
   ``{eval_model}_db.csv`` (database with train/test ``Set`` column).
*  Creates an ``EVALUATE/`` log folder; does **not** produce GENERATE heatmaps or
   Raw_data screening outputs.

Technical information
+++++++++++++++++++++

*  **Supported models:** ``eval_model='MVL'`` (multivariate linear regression via
   sklearn ``LinearRegression``) is the only option today.
*  **Problem type:** regression (``type='reg'``). Classification support is planned.
*  **CLI:** ``python -m robert --csv_name FILE.csv --evaluate`` plus ``--y``,
   ``--names``, and optional ``--eval_model``, ``--kfold``, ``--repeat_kfolds``.
*  **Typical workflow:** CURATE → EVALUATE → VERIFY → PREDICT (or full workflow
   with ``--evaluate`` instead of GENERATE).
*  **UQ / XGB:** EVALUATE does not screen XGB or other GENERATE models. Uncertainty
   columns in PREDICT still follow :doc:`../API/robert.api` when enabled downstream.

Example
+++++++

A minimal regression CSV is in the repository at ``tests/Evaluate_test.csv``:

.. code:: shell

   python -m robert --csv_name tests/Evaluate_test.csv --evaluate --y Target_values --names Name

.. evaluate-modules-end
