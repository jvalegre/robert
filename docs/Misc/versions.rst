.. _versions:

========
Versions
========

Version 2.2.0 [`url <https://github.com/jvalegre/robert/releases/tag/2.2.0>`__]
   *The ROBERT score is now split into two fully independent scores, Interpolation and
   Boundary robustness (each 0-10), instead of one shared score with duplicated sub-metrics on
   both sides. Boundary robustness's predictions now use the same repeated-CV methodology as
   Interpolation, several of its sub-metrics were redesigned for statistical robustness,
   Interpolation gained a new overfitting-focused item, a long-standing bias in VERIFY's
   one-hot test was fixed, VERIFY gained a new cluster-based flawed-model test, a new
   --all_models option runs the full VERIFY/PREDICT/REPORT pipeline for every screened model
   instead of just the best one, and the Bayesian Optimization hyperparameter search space was
   revised for ROBERT's typical (small) dataset sizes.*

   **Interpolation and Boundary robustness are now independent scores**
   -  Section A/B used to show the same combined score on both the Interpolation and
      Boundary robustness (formerly "Extrapolation" - renamed since the metric evaluates
      robustness at the edges of the observed y-range via repeated CV, not genuine
      out-of-domain extrapolation) columns, with the right-hand column mostly duplicating or
      leaving blank its sub-metrics. Both columns now have their own independent 0-10 score,
      their own score icon, and five dedicated sub-metrics each (0-2 points each), scored and
      explained separately

   **Boundary robustness now uses the same repeated-CV methodology as Interpolation**
   -  The Low/High (bottom/top 20% of y, formerly labeled "Q1"/"Q5" - renamed since they are
      sorted-CV folds, not statistical quartiles) predictions and the Applicability Domain
      analysis used to come from a single train-once/test-once fit. They are now computed with
      the same 10x repeated 5-fold CV that Interpolation's "10x 5-fold CV"/"test" plot already
      used, scoped to each extreme's own 80/20 partition of the sorted dataset. This makes
      Boundary robustness's RMSE directly comparable to Interpolation's RMSE (both are averages
      over the same kind of repeated procedure, instead of one being averaged and the other a
      single noisy draw), and it fixed a real bug where the "Scaled RMSE (Low/High, sorted CV)"
      text shown in the report was still silently reading VERIFY's old single-pass values even
      after the plots themselves had already switched to the new repeated-CV predictions

   **Boundary robustness sub-metrics redesigned**
   -  "Sorted CV, top 20% (High)" / "bottom 20% (Low)": scored on scaled RMSE (unchanged
      thresholds), now penalized by how many of the extreme predictions actually cross beyond
      the training range ("real range extension" / crossing rate) instead of by Spearman rank
   -  New "Spearman rank" item: Spearman rank correlation within the Low/High folds is now its
      own dedicated, independently-scored item instead of only acting as an internal penalty
      inside the RMSE items above
   -  "Degradation ratio" item: compares each extreme fold's (Low/High) scaled RMSE against the
      RMSE of its own "remaining 80%" baseline from the same sorted-CV run. Low and High are
      now scored independently (+1 each, <=1.5x tiering) instead of taking the worse (max) of
      the two sides, so a badly-degraded side is no longer invisible just because the other
      side happens to be fine
   -  Applicability domain (leverage): scoring and Williams-plot methodology unchanged, now
      also fed by the repeated-CV predictions instead of a single fit
   -  New unscored "Additional diagnostics" item: shows the global Spearman rank over the
      whole sorted dataset as context (the local, per-extreme Spearman is already scored in
      item 3), without double-counting rank that other items already score
   -  Every Boundary robustness sub-score is now a plain integer (no more .5 values feeding
      into the column total)

   **New Interpolation item: train vs. validation gap**
   -  New item 4, "Train vs validation gap": compares each fold's out-of-fold validation RMSE
      against that same fold's own in-fold training RMSE (scaled RMSE (test) <= 1.25x train:
      +2, <= 1.5x: +1), to catch a model that fits its own training folds much better than it
      generalizes to their held-out validation fold - a form of overfitting that a validation-only
      metric can miss. The existing "CV vs test" comparison (now item 5) was retitled "CV vs
      test consistency" to disambiguate it from this new item, since both compare two error
      values but on different axes (fold-internal train/validation vs. overall CV/test)

   **Interpolation item 6 redesigned into three stability facets**
   -  "Avg. standard deviation (SD)" is renamed "Prediction stability" and now averages three
      facets (each scored 0-2, final score is their average, rounded): (a) SD of the
      repeated-CV test-set predictions (unchanged), (b) SD of the out-of-fold
      train+validation predictions (new), (c) coefficient of variation of the aggregate RMSE
      across the 10 CV repeats (new) - a dataset-wide view of the same "how much does this
      depend on the random split" question that (a)/(b) ask per-point
   -  New plot: out-of-fold CV predictions +- SD, mirroring the existing test-set +- SD plot

   **Report PDF rendering fixes**
   -  Fixed inconsistently cropped titles on the small report thumbnails (VERIFY tests, Low/High,
      applicability domain, Williams plot, SD plots): some showed no title, others a
      half-cut one, depending on each image's exact aspect ratio. The saved PNG files in
      PREDICT/VERIFY keep their full title as before; the report's image containers now use a
      height measured precisely from each image type so the title is cropped out cleanly and
      consistently instead of only in some plots
   -  Fixed a large blank gap in the Pearson correlation heatmap (worst on databases with very
      few descriptors, e.g. only 1 of 4 cells visible with 2 descriptors): masking the upper
      triangle left the always-fully-masked first row and last column as still-valid (if
      invisible) axes content, which ``bbox_inches='tight'`` does not crop away. The heatmap
      now trims that empty row/column from the plot only; the function's returned correlation
      matrix is unchanged (kept full-size, since PREDICT's correlation-pair detection indexes
      into it positionally and would silently miscount pairs if it were trimmed too)

   **VERIFY: new flawed-model test - cluster mean baseline**
   -  Added a fourth flawed-model comparison, alongside ``y_mean``/``y_shuffle``/``onehot``: a
      baseline that clusters the descriptors into two groups with KMeans and predicts each
      point using its own cluster's training-y mean, evaluated out-of-fold with the same
      repeated 10x 5-fold CV used for the real model. This specifically targets bimodal/
      structured datasets, where a trivial "which cluster am I in" rule can otherwise reach
      deceptively good RMSE and go undetected by the existing three tests

   **Bayesian Optimization: integer hyperparameters no longer reported with decimals**
   -  Initial BO points are generated with Latin Hypercube Sampling and probed directly (instead
      of queued lazily), and every suggestion from the optimizer now has its integer-type
      hyperparameters (e.g. ``n_estimators``, ``max_depth``) rounded before being registered.
      The model itself was always fit with the correctly rounded value, but the reported/saved
      "best params" could previously show misleading decimals for these hyperparameters

   **Removed obsolete LOOCV documentation**
   -  The ``--kfold`` help text described a "for databases with less than 50 points, do LOOCV"
      auto-behavior that was never implemented (``kfold`` has always simply defaulted to 5,
      regardless of dataset size). The help text now matches the actual behavior

   **VERIFY module: fixed a systematic bias in the one-hot test**
   -  The one-hot flawed-model test binarized every descriptor with "value == 0 -> 0, else ->
      1". For any strictly-positive continuous descriptor with no literal zeros (e.g. a
      molecular weight), this silently collapsed it into a useless constant column instead of
      a genuine two-way split, which could make the test fail regardless of whether the model
      was actually flawed. Continuous descriptors are now binarized with an unsupervised
      per-descriptor median split (value >= median -> 1, else -> 0), which does not require
      literal zeros and does not look at the target value (keeping the test an unbiased
      sanity check)
   -  Descriptors that already have 2 or fewer unique values are left untouched instead of
      being median-split: applying the median split to an imbalanced already-binary
      descriptor (its minority class under 50%) reproduces the same collapse-to-constant
      failure, just triggered by class imbalance instead of by missing literal zeros. This is
      handled per descriptor, so datasets mixing binary and continuous descriptors are scored
      correctly instead of only checking the dataset as a whole
   -  A discrete descriptor with many ties at its median can still collapse to a constant
      after the split; that single descriptor is now dropped from the one-hot matrix instead
      of letting it invalidate the whole test
   -  The one-hot test is skipped entirely (reported as N/A, not scored) when every descriptor
      already has 2 or fewer unique values, or when every descriptor collapses to a constant
      after binarization, since there is no continuous variation left to destroy and the test
      would otherwise always report a spurious failure

   **Bayesian Optimization: hyperparameter search space revised**
   -  Bounds are now tuned for ROBERT's typical dataset sizes (mostly 20-100 datapoints, up to
      ~10,000 at most) rather than scikit-learn's generic defaults, which assume much larger
      samples. RF/GB's ``max_depth`` are no longer shared: GB (boosting) now searches a
      shallower 2-10 range instead of RF's 5-20, matching scikit-learn's own default of 3 for
      gradient boosting. NN's ``alpha`` lower bound widened down to scikit-learn's own default
      (0.0001), which the previous 0.01 floor excluded from the search entirely
   -  Removed three hyperparameters that were being searched without any real effect on the
      fitted model: GB's ``validation_fraction`` (only used if ``n_iter_no_change`` is set,
      which ROBERT never does, so scikit-learn silently ignores it), RF/GB's
      ``min_weight_fraction_leaf`` (constrains leaf size by weighted sample fraction, but
      ROBERT never passes ``sample_weight`` to ``fit()``, so with implicit uniform weights this
      is a redundant, weaker echo of ``min_samples_leaf``/``min_samples_split``, already
      searched), and NN's ``tol`` (affects when the solver stops, not the quality of the fit)
   -  Integer-type hyperparameters can now define a step size (e.g. ``n_estimators`` in steps
      of 10, NN's ``max_iter`` in steps of 25) instead of every single integer, so each BO
      evaluation explores meaningfully different configurations instead of two iterations
      landing on near-identical values (e.g. 71 vs. 72 trees)

   **New --all_models option: full VERIFY/PREDICT/REPORT for every screened model**
   -  GENERATE already fits all 4 models (RF/GB/NN/MVL) during hyperparameter screening, but
      VERIFY/PREDICT/REPORT only ever ran on the single best one. With ``--all_models True``,
      every model GENERATE screened gets its own full VERIFY -> PREDICT -> REPORT pass and its
      own PDF (e.g. ``ROBERT_report_RF_No_PFI.pdf``, ``ROBERT_report_NN_PFI.pdf``, ...), so the
      user can compare and pick a model manually instead of only ever seeing the
      auto-selected best one. Only this VERIFY/PREDICT/REPORT tail runs once per model - the
      expensive BO search in GENERATE is unaffected, since it already computes all 4 models
      regardless of this option
   -  Section F (Model Screening) shows a different heatmap when this option is active: instead
      of GENERATE's raw combined-RMSE-per-model heatmap (only reflects the BO search
      criterion), it shows each model's final Interpolation/Boundary robustness score (0-10), which
      is a more informative basis for comparing models once VERIFY/PREDICT have finished

   **PREDICT: faster SHAP analysis for RF and GB**
   -  SHAP was always run as ``shap.Explainer(model.predict, X)``, i.e. through the generic
      predict-function wrapper. This hides the tree structure from SHAP, so it fell back to the
      Exact/Permutation explainer, which scales poorly with dataset size and descriptor count.
      RF and GB now use ``shap.TreeExplainer(model)`` directly, which computes the same exact
      Shapley values through the polynomial-time TreeSHAP algorithm instead of the slow
      generic fallback - same results, much faster on large datasets. NN and MVL are unaffected
      (TreeSHAP doesn't apply to them)
   -  Fixed a crash introduced by the change above for RF/GB classification models:
      ``TreeExplainer`` returns SHAP values per class (an extra 3rd array dimension) instead of
      the plain per-descriptor 2D array the generic explainer returned, which broke both the
      summary plot and the printed min/max SHAP values per descriptor. The last class (the
      positive class in binary problems) is now selected right after computing SHAP values, so
      plotting and the printed summary both consistently work on a 2D array again

   **REPORT: classification Interpolation score now reaches 10, same as regression**
   -  Classification's Interpolation score used to max out at 8 (CV predictions, max 3 + test
      predictions, max 3 + the flawed-models penalty, always ≤ 0 + the MCC-difference item, max 2),
      missing the two points regression gets from item 6 (prediction stability) because that item's
      SD-based, %-of-y-range definition doesn't apply to a discrete label. Item 6 is now defined for
      classification too, adapting the same three facets: (a)/(b) the disagreement rate between
      individual CV repeats and the already majority-voted class (test set and out-of-fold
      train+validation, respectively), and (c) the coefficient of variation of the per-repeat MCC
      instead of the per-repeat RMSE - each scored 0-2 with the same ≤15%/≤25% thresholds used by
      regression's facet (c), then averaged and rounded, same as regression. This was a stale
      denominator bug in passing (the percentage-of-max used to look up the score tier was dividing
      by 9 - a leftover from an assumption that the flawed-models item could contribute +1, which it
      never does - before this fix corrected it to 8, now superseded by the real max of 10)

   **GENERATE/REPORT: --test_set 0 now works on its own, and the score is hidden when the
   test set isn't the standard split**
   -  ``--test_set 0`` used to still get silently raised to 0.2 by the ``--auto_test`` safety
      net (needing ``--auto_test False`` on top to actually take effect) even though an
      explicit 0 is an unambiguous, deliberate "train on 100% of the data" choice, not a value
      that needs the safety net - only small-but-nonzero values (a likely accident) still get
      raised
   -  The ROBERT score's thresholds were calibrated assuming the standard ~20% test_set split
      (see the note in score.rst). With no internal test set at all (``--test_set 0``, or any
      other non-standard ``--test_set``), the "test set predictions" score component silently
      computed as 0 (``NaN <= threshold`` comparisons are always ``False`` in Python, so this
      failed silently, no crash and no warning), which could tank an otherwise good model's
      score for a reason invisible in the PDF. Section A now shows a short explanatory notice
      instead of a score in this case, the "Overall assessment" line shows "Not available"
      instead of a verdict based on that score, and Section B (entirely a breakdown of the
      score) is skipped - the rest of the report (SHAP, PFI, outliers, reproducibility, etc.)
      is unaffected

   **GENERATE: fixed the STRATIFIED split for classification, and warn on split methods
   that don't apply to classification**
   -  ``--split stratified`` used to cap the min/max target value and bin the remaining values
      into quantiles via ``pd.qcut()`` before stratifying - a regression-only approach (binning
      a *continuous* target). For classification this ignored the actual class labels entirely,
      relying on ``qcut`` accidentally degenerating to ~2 bins for a binary target to produce a
      passably-balanced split; for 3+ classes it was blocked outright (fell back to RND). Now
      classification stratifies directly on the class labels via ``StratifiedShuffleSplit``,
      which works for any number of classes
   -  While fixing this, found and fixed a real pre-existing bug affecting the *regression*
      STRATIFIED split too: ``StratifiedShuffleSplit.split()`` yields ``(train_idx, test_idx)``
      in that order, but the code unpacked it as ``for test_idx, _ in ...``, taking the train
      indices as if they were the test indices. This had been silently compensated by a second,
      opposite bug (``test_size=(100 - size) / 100`` instead of ``size / 100``, i.e. requesting
      the complement), so the two bugs cancelled out and regression's STRATIFIED split ended up
      correct anyway - fixed to compute both correctly on their own, without relying on that
      accidental cancellation
   -  EVEN/EXTRA_Q1/EXTRA_Q5 rely on a continuous, ordered target value (e.g. "the lowest 20% of
      y") that has no defined meaning for a discrete class label, regardless of class count -
      choosing one of these for a classification run now logs a warning and falls back to
      STRATIFIED (previously this fallback only existed for KN/STRATIFIED with 3+ classes, and
      fell back to RND)
   -  Classification's default split (``--split auto``, or no ``--split`` at all) changed from
      RND to STRATIFIED - preserving class proportions in both train and test is standard
      practice for classification, and a plain random split can, by chance, leave a class
      under/over-represented, especially on small datasets

Version 2.1.1 [`url <https://github.com/jvalegre/robert/releases/tag/2.1.1>`__]
   - Adding RMSE values for each fold to calculate t- and Wilconxon tests
   - Add code for BO function

Version 2.1.0 [`url <https://github.com/jvalegre/robert/releases/tag/2.1.0>`__]
   -  In classification problems now we can use 2 different categorial class labels (e.g., "active"/"inactive").
   -  Changing the way of selecting the 10 first initial points in Bayesian Optimization (now using Latin Hypercube Sampling)
   -  Deleting first the most correlated features in CURATE module
   -  Sorting the columns and rows in the csv files to ensure reproducibility
   -  Using only KNN imputer if you have more than 100 datapoints
   -  Fixing RFECV (each model now has its own set of descriptors after feature selection)
   -  Fixing bug in the AQME module when using --csv_test
   -  Changing pkg_resources to importlib.resources to avoid deprecation warnings
   -  Fixed bug when selecting test set datapoints with the EVEN option
   -  Fixed bug in the name of extra_q1 and extra_q5 splitting methods
   -  Updating packages versions in setup.py
   -  Molssi databases link in easyROB GUI
   -  Default split is 'RND' for classification problems
   -  The sklearn-intelex accelerator was removed

Version 2.0.2 [`url <https://github.com/jvalegre/robert/releases/tag/2.0.2>`__]
   -  Fixed bug in MAC and Linux OS from the GENERATE
   -  Updating AQME version to 1.7.3
   -  Fixing libgfortran library to version 14.2.0

Version 2.0.1 [`url <https://github.com/jvalegre/robert/releases/tag/2.0.1>`__]
   -  The AQME and EVALUATE modules are now fully functional and have been reactivated in this version.
   -  Fixed classification with external predictions
   -  Fixed scores from VERIFY tests in clas
   -  Fixed bug in the detection of automatic classification problems
   -  Fixed bug in 'load_variables' where the model type and target value were not being saved
   -  Fixed bug in 'sort_n_load' to ensure reproducibility of sorted CV across different operating systems

Version 2.0.0 [`url <https://github.com/jvalegre/robert/releases/tag/2.0.0>`__]
   *Adaptation of the code to avoid overfitting and to use with low-data problems*
   -  Fixed a bug in one-hot encoding in the one-hot test
   -  Adding the possibility to disable the automatic standarization of descriptors (--std False)
   -  Changing CV_test (now it standardizes the full database with sklearn functions)
   -  Fixing a bug with the sklearn-intelex accelerator
   -  Fixing a threading bug with matplotlib in SHAP
   -  train:validation split was replaced by a repeated k-fold CV
   -  The program always holds out a test set
   -  The average results of the repeated k-fold CV are used to measure predictive ability and to predict new results
   -  The BayesianOptimization() is used to find the bets model, using a combined metric that depends on interpolation and extrapolation of diferent types of CVs
   -  This version does not work with classification problems and the AQME and EVALUATE modules were disabled until v2.0.1.
   -  Updated ROBERT score, which is more robust towards small data problems

Version 1.3.0 [`url <https://github.com/jvalegre/robert/releases/tag/1.3.0>`__]
   -  Fixing a bug in the KNN imputer (it was incorrectly placing values in the target variable)
   -  Adding a new way of splitting data (stratified) to ensure that the validation points are taken throughout the range of the target values
   -  Fixing bug to work with spaces in descriptor names
   -  Changing the way of selecting the best model (now using a combined error metric, not only the validation error)
   -  Fixing bug in GENERATE when plotting the models' heatmap in case the model had infinite values
   -  Auto_test is now done by default if the database has more than 100 datapoints
   -  90% training size disables for datasets with less than 100 datapoints and 80% for less than 50 datapoints
   -  Changing models paramaters to avoid overgitting in small datasets
   -  Fixing bug (ROBERT was not reading some CSV files correctly when saved as UTF-8)
   -  Fixed bug in the report module when the Target_values had spaces
   -  MVL is replaced with AdaB when ROBERT assigns automated classification problems
   -  Adding automatic checks to ensure compatible classification problems
   -  ROBERT score is printed in the section title in the report to save space
   -  Kmeans clustering is applied individually to the different target values in classification problems to allow for a more compensated training selection

Version 1.2.1 [`url <https://github.com/jvalegre/robert/releases/tag/1.2.1>`__]
   -  NN solver are now set to 'lbfgs' by default in the MLPRegressor to work with small datasets
   -  Thres_x is now set to 0.7 by default in the CURATE module
   -  Fixing bug in the PREDICT module when using EVALUATE module (it was not showing the linear model equation)
   -  Adding linear model equation in the REPORT module
   -  Changing the threshold for correlated features in predict_utils to adjust to the new thres_x
   -  Changing the way missing values are treated (previously filled with 0s, now using KNN imputer)
   -  Adding .csv in --csv_test in case the user forgets to add it
   -  Adding ROBERT score number in the REPORT module
   -  Creating --descp_lvl to select which descriptors to use in the AQME-ROBERT workflow (interpret/denovo/full)
   -  The AQME-ROBERT workflow now uses interpretable descriptors by default (--descp_lvl interpret)

Version 1.2.0 [`url <https://github.com/jvalegre/robert/releases/tag/1.2.0>`__]
   -  Changing cross-validation (CV) in VERIFY to LOOCV for datasets with less than 50 points
   -  Changing MAPIE in PREDICT to LOOCV for datasets with less than 50 points
   -  By default, RFECV uses LOOCV for small datasets and 5-fold CV for larger datasets
   -  The external test set is chosen more evenly along the range of y values (not fully random)
   -  Changing the format of the VERIFY plot, from donut to bar plots
   -  Automatic KN data splitting for databases with less than 250 datapoints
   -  Change CV_test from ShuffleSplit to Kfold
   -  Predictions from CV are now represented in a graph and stored in a CSV
   -  Changing the ROBERT score to depend more heavily on results from CV
   -  Fixing auto_test (now it works as specified in the documentation)
   -  Adding clas predictions to report PDF
   -  Adding new pytests that cover the ROBERT score section from the report PDF
   -  Adding the EVALUATE module to evaluate linear models with user-defined descriptors and partitions
   -  Adding Pearson heatmap in PREDICT for the two models, with individual variable correlation analysis
   -  Adding y-distribution graphs and analysis of uniformity
   -  Major changes to the report PDF file to include sections rather than modules
   -  Improving explanation of the ROBERT score on Read The Docs
   -  Printing coefficients in MVL models inside PREDICT.dat
   -  Fixing bug in RFECV for classification problems, now it uses RandomForestClassifier()
   -  Automatic recognition of classification problems

Version 1.1.2 [`url <https://github.com/jvalegre/robert/releases/tag/1.1.2>`__]
   -  Fixing conda-forge install and making pip install the preferred installation method in ReadtheDocs

Version 1.1.1 [`url <https://github.com/jvalegre/robert/releases/tag/1.1.1>`__]
   -  Hotfix of 1.1.0 in the installation
   -  Add documentation of AQME with versions >=1.6.0, in which SMILES workflows are fully reproducible

Version 1.1.0 [`url <https://github.com/jvalegre/robert/releases/tag/1.1.0>`__]
   -  Adding RFECV in CURATE to fix the maximum number of descriptors to 1/3 of datapoints
   -  Added the possibility to use more than 1 SMILES column in the AQME module
   -  Change the scoring criteria in the PFI workflow (from R2 to RMSE)
   -  Fixing models where R2 in validation is much better than in training (if the validation set is very small or unrepresentative, the model may appear to perform excellently simply by chance)
   -  Fixing PFI_plot bug (now takes all the features into account)
   -  Fixing a bad allocation memory issue in GENERATE
   -  Fixing bug in classification models when more than 2 classes of the target variable are present
   -  Fixing reproducibility when using a specific seed in GENERATE module
   -  Change CV_test from Kfold to ShuffleSplit and adding a random_state to ensure reproducibility
   -  Allows CSV inputs that use ; as separator
   -  Fixing CV_test bug in VERIFY (now it uses equal test size to the model tested)
   -  Adding variability in the prediction with MAPIE python library
   -  Adding sd in the predictions table when using external test set
   -  Fixing error_type bug for classification models
   -  MCC as default metric for classification models (better to check performance in unbalanced datasets)
   -  PFI workflow now uses the same metric as error_type

Version 1.0.5 [`url <https://github.com/jvalegre/robert/releases/tag/1.0.5>`__]
   -  Fixing some overfitted models with train and validation R2 0.99-1
   -  Including the easyROB graphical user interface (GUI)

Version 1.0.4 [`url <https://github.com/jvalegre/robert/releases/tag/1.0.4>`__]
   -  Fixing outlier bug for negative t-values
   -  csv_test is treated separately from the test set from GENERATE
   -  Table of score thresholds in ROBERT_report.pdf
   -  Showing predictions at the end of the PREDICT section of ROBERT_report.pdf
   -  Adding --csv_test to AQME workflows
   -  Adding the --crest option to AQME workflows
   -  Auto adjusting the convergence criteria and xTB accuracy of QDESCP based on number 
      of datapoints

Version 1.0.3 [`url <https://github.com/jvalegre/robert/releases/tag/1.0.3>`__]
   -  Changing default split to RND
   -  Adding the scikit-learn-intelex accelerator (now it's compatible for scikit-learn 1.3)
   -  Changing the thres_test default value to 0.25 (before: 0.20)
   -  Automatic KN data splitting for databases with less than 100 datapoints
   -  Droping 90% and 80% training sizes for small databases (less than 50 and 30 datapoints)
   -  Better print for command lines (more reproducible commands)
   -  Adding more information in the --help option
   -  Introducing SCORE and REPRODUBILITY to ROBERT_report.pdf
   -  Added the auto_test option
   -  Fixed empty spaces in heatmaps from GENERATE
   -  Mantain the ordering of GENERATE heatmaps across No_PFI and PFI 
   -  Added pytest to full workflows with classification and tests
   -  Fixed " separators in command lines with options that had more than one word (i.e. 
      --qdescp_keywords)
   -  Fixed length of outlier names for long words

Version 1.0.2 [`url <https://github.com/jvalegre/robert/releases/tag/1.0.2>`__]
   -  Adding the REPORT module
   -  Adding the ReadTheDocs documentation

Version 1.0.0 [`url <https://github.com/jvalegre/robert/releases/tag/1.0.0>`__]
   -  First estable version of the program
