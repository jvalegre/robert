.. _robert-score-start:

CheckML score
-------------

.. |br| raw:: html

   <br />

.. |u| raw:: html

   <u>

.. |/u| raw:: html

   </u>

.. |space| raw:: html

   &nbsp;

.. _score-ranges:

.. tab-set::
   :class: checkml-main-tabs

   .. tab-item:: Overview and considerations

      **How should we assess whether a published ML model is sufficiently robust, reliable, and applicable
      to the chemical problem it is intended to address?**

      In practice, model quality is often communicated through a small number of statistical metrics
      (e.g., R2, RMSE, MCC). No single train/test split, performance metric, or validation strategy can fully
      establish a model's reliability, making it difficult
      for readers and reviewers to assess the robustness and domain of applicability of the reported models.

      CheckML addresses this limitation through a standardized, multidimensional assessment of descriptor-based
      tabular ML models. The framework examines predictive performance,
      overfitting, generalization, prediction stability, and behavior at the boundaries of the modeled chemical space
      (related to extrapolation), providing quality scores and diagnostic alerts that highlight potential weaknesses.
      These analyses are compiled into standardized PDF reports to make evaluations transparent and easier to review.

      .. figure:: images/checkml_summary.jpg
         :width: 400
         :align: center

      |br|

      **Considerations when using CheckML**

      * **User-defined train/test splits do not receive a score.** The CheckML score requires the standardized
        splitting and CV procedure described in the **Protocols followed by CheckML** tab to keep evaluations
        comparable. A user-defined test set can still be analyzed, but the report will not include a score
        (see the **Model evaluation with user-defined data partitions** tab).
      * **CheckML scores are indicative.** Scores are guides rather than rigid thresholds of model quality
        or usefulness; their interpretation depends on the intended application. Review the underlying metrics
        and diagnostic alerts alongside the scores. Before interpreting a report, read the protocol and scoring
        details in this documentation carefully.

      .. note::

         Please note that the CheckML score was developed based on the following:

         * Established recommendations for robust ML
         * Experience with chemical ML models across a broad community of expert chemists
         * A comprehensive benchmarking process

         **We are completely open to discuss any advice on how to improve the thresholds used in the score or make the score more robust!**

      Please, see the CheckML publication for more details.

      |br|

   .. tab-item:: Protocols followed by CheckML

      Before evaluation, CheckML applies a standardized splitting procedure. The random seed is fixed, and rows
      and columns are sorted so that the results are independent of the order in which data are provided. 80% of
      the data are used for cross-validation (CV), while the remaining 20% (or a minimum of four data points)
      are reserved as an independent test set.

      CheckML uses 10-times repeated 5-fold CV (10x 5-fold CV) as part of its evaluation protocol.
      Importantly, alternative CV strategies have been explored in the literature, including 
      5x 2-fold, 2x 5-fold, and 1x 10-fold CV, among many others. Although alternative CV strategies 
      may also be valid, we selected 10x 5-fold CV because it is a well-established approach widely 
      used in the ML community and is the default repeated CV strategy recommended
      by scikit-learn. Benchmarking across low-data regimes also showed that 10x 5-fold CV
      remains informative across a broad range of dataset sizes (16–4200 data points,
      see the CheckML publication).

      .. note::
        Although leave-one-out CV (LOOCV) is sometimes used for low-data chemical problems,
        each validation estimate is based on a single observation and the resulting
        estimates can be highly correlated. Thus, 10x 5-fold CV provides a more informative
        validation framework for the intended application of CheckML.

      After this initial splitting, different ROBERT routines are applied to evaluate the
      model as explained in the following sections.

      |br|

   .. tab-item:: Model evaluation with user-defined data partitions

      A central motivation for using repeated CV is that performance derived from a single user-defined
      train/test split can be highly sensitive to the selected partition, especially in low-data scenarios.
      To avoid this dependence, CheckML evaluates models using its own standardized data partitions,
      independently of any train/test splits originally defined by the user. The figure below illustrates
      this issue using the same overall dataset and the same linear regression algorithm.

      .. figure:: images/user_partitions.png
         :width: 100%
         :align: center

      *Example illustrating how different train/test partitions of the same dataset can lead to misleading*
      *conclusions about model performance. Each partition contains 22 training points and 6 test points,*
      *with separate linear regression models fitted to each training set. The linear equations show scaled*
      *coefficients to facilitate comparison of feature importance. The two descriptors exhibit limited*
      *correlation (R*\ :sup:`2` *= 0.32), minimizing the risk of misleading feature selection due to collinearity.*

      Models A and B have similar train and test R\ :sup:`2` and RMSE values, and would therefore appear
      comparably successful if judged using conventional approaches. However, the models suggest markedly
      different interpretations of the same problem. In Model A, the fitted relationship is dominated by
      descriptor x\ :sub:`1`, whereas in Model B it is dominated by x\ :sub:`2`. In a chemistry context, such
      differences could translate into substantially different conclusions regarding which molecular property
      is most relevant for the target problem. At the same time, Model C shows that a different partition of
      the same dataset can yield clearly worse performance in both training and test sets.

      Therefore, even when a model appears reliable, its reported performance may not reflect its behavior
      across alternative data partitions. Relying on a single user-defined split can thus be misleading,
      whereas repeated CV provides a more robust and representative assessment. Nevertheless, authors may
      still report results from user-defined splits, using the CheckML scores and warnings to assess potential
      limitations in model reliability and interpretation. When such limitations are identified, authors
      should provide additional evidence or justification to support the model's reliability beyond the
      selected partition and, when relevant, the consistency of conclusions drawn from feature analysis.

      **Analyzing your own test set (no score)**

      To obtain the CheckML score, upload the CSV **without** any partition information (see
      :doc:`checkml_database`). If you also want to see how your own train/test split behaves, run CheckML
      a second time with an extra column named ``Set``:

      * Rows with the value ``Test`` (case-insensitive) are held out as the test set.
      * All the other rows (e.g. ``Training``) are used for the 10x 5-fold CV.

      .. csv-table::
         :header-rows: 1

         "Molecule_name", "Target_property", "descriptor_1", "descriptor_2", "Set"
         "molecule_1", "1.25", "0.41", "2.7", "Training"
         "molecule_2", "0.82", "0.36", "1.9", "Test"
         "molecule_3", "0.98", "0.46", "1.7", "Training"

      The PDF report of that run includes all the analyses (metrics, plots, warnings), but **no CheckML
      score**: the score bars are grayed out and marked as not available, because the score is calibrated
      for the standardized split. The same applies to the command-line ``--split`` option of ROBERT's
      CHECKML module: any value other than the default (``auto``, i.e. EVEN for regression and STRATIFIED
      for classification) gives a report without a score.

      .. note::
         CSV files produced by ROBERT (e.g. ``*_db.csv``) already contain a ``Set`` column. Remove it before
         uploading the file to CheckML if you want to obtain a score.

      |br|

   .. tab-item:: List of warnings

      The tables below reproduce the warning messages shown in the PDF summary, together with their
      problem type, severity, trigger, and suggested actions. Warnings can occur together; resolving one
      does not necessarily resolve the others.

      Within each table, severe warnings appear before moderate warnings. Within each severity group,
      warnings without a section reference come first, followed by B.1, B.2, and subsequent sections in order.

      In the hyperparameter warning, ``{parameter}`` is replaced by the actual parameter name.
      The **Pass**, **Unclear**, and **Warning** indicators in the **CV-Test Split Diagnostic** box are
      explained in their own tab.

      **Interpolation**

      .. list-table::
         :header-rows: 1
         :widths: 28 16 33 23
         :class: checkml-warning-table

         * - Warning as printed in the PDF
           - Problem type
           - Explanation, trigger, and implications
           - Suggested actions
         * - Failing required tests (Section B.1)
           - Regression and classification
           - **Severe.** The flawed-model score is negative and at least
             one applicable VERIFY test is marked FAILED. A trivial or deliberately degraded model
             competes too closely with the original, weakening evidence that meaningful relationships were learned.
           - Inspect the failed y-mean, y-shuffle, onehot, or cluster test. Check leakage, descriptor
             informativeness, and response quality; address the underlying issue and repeat VERIFY.
         * - Overfitting observed between train and validation (Section B.4)
           - Regression and classification
           - **Severe.** The train/validation comparison scores 0/2.
             Out-of-fold RMSE is much worse than the training fit, or training MCC substantially exceeds
             validation MCC. Training performance therefore overstates predictive ability.
           - Reduce model complexity, increase appropriate regularization, and check for leakage or
             duplicates. Select parameters within CV and repeat the standardized evaluation.
         * - Inconsistent results between CV and test (Section B.5)
           - Regression and classification
           - **Severe.** The CV/test consistency score is 0/2. Regression
             test error is much larger than CV error, or classification CV and test MCC differ substantially.
             The CV result may not represent performance on held-out data.
           - Compare coverage and errors across the CV and test pools. Check data consistency and leakage,
             improve representative coverage, and rerun the standard protocol.
         * - Very uneven class distribution (Section C)
           - Classification
           - **Severe.** The largest class contains more than three times as many
             observations as the smallest. A classifier can favor the majority class while performing
             poorly on underrepresented classes.
           - Collect representative minority-class observations and inspect per-class errors and the
             confusion matrix. If using class weights or resampling, apply them within training folds only.
         * - Some tests are unclear or close to the thresholds (Section B.1)
           - Regression and classification
           - **Moderate.** The flawed-model score is negative without
             any test marked FAILED. At least one baseline remains close enough to incur a penalty,
             so the model's advantage over simple or degraded predictions is uncertain.
           - Review each applicable VERIFY result and its margin from the thresholds. Improve informative
             descriptors or data coverage and check whether the advantage persists after reevaluation.
         * - A simpler MVL model performs similarly (Section B.1), consider trying MVL with the same descriptors
           - Regression
           - **Moderate.** VERIFY places the multivariate linear (MVL) baseline within
             its 15% similarity band relative to the selected model. The additional complexity has no
             clear predictive advantage in this comparison.
           - Fit MVL with the same descriptors and compare validation results, interpretability,
             and intended use before choosing the more complex model.
         * - A simpler MVL model performs better (Section B.1), consider trying MVL with the same descriptors
           - Regression
           - **Moderate.** VERIFY finds that the MVL baseline outperforms the selected
             model by its comparison threshold (15%). The more complex model may be unnecessary or poorly tuned.
           - Evaluate MVL as a candidate and review the selected model's parameters and descriptors.
             Confirm the preferred model with the same standardized validation protocol.
         * - A simpler logistic regression model performs similarly (Section B.1), consider trying LOGISTIC with the same descriptors
           - Classification
           - **Moderate.** VERIFY places logistic regression within its 15%
             similarity band relative to the selected classifier. A more complex decision rule has
             not shown a clear predictive advantage.
           - Compare LOGISTIC using the same descriptors and validation settings, including performance
             across classes, before deciding whether the additional complexity is useful.
         * - A simpler logistic regression model performs better (Section B.1), consider trying LOGISTIC with the same descriptors
           - Classification
           - **Moderate.** VERIFY finds that logistic regression outperforms the
             selected classifier by its comparison threshold (15%). The chosen classifier may add
             complexity while reducing predictive performance.
           - Evaluate LOGISTIC as a candidate. Review the chosen classifier's parameters and descriptors,
             and confirm performance with the standard validation protocol.
         * - Weak CV predictions (Section B.2)
           - Regression and classification
           - **Moderate.** The CV prediction component scores 0/2:
             the combined scaled-RMSE score and R2 penalty reach zero in regression, or the MCC-based
             score reaches zero in classification. Predictive performance on unseen CV observations is weak.
           - Check response quality, descriptor relevance, and preprocessing. Compare suitable models and
             parameters within CV, and collect more informative observations where possible.
         * - Weak test-set predictions (Section B.3)
           - Regression and classification
           - **Moderate.** The independent test prediction component
             scores 0/2 using the regression scaled-RMSE/R2 combination or classification MCC.
             Predictions on held-out observations have limited accuracy.
           - Examine individual test errors and domain coverage. Improve the model using training/CV data
             and evaluate subsequent development on fresh independent data.
         * - Imprecise predictions (Section B.6)
           - Regression and classification
           - **Moderate.** In regression, the prediction-stability score
             is 0/2 because of variation across repeated CV predictions and/or unequal performance between
             low- and high-response halves. For classification, the current warning trigger is a zero
             CV/test MCC-consistency score, so it can accompany the Section B.5 warning. These outcomes
             indicate sensitivity to which data are used for evaluation.
           - Inspect repeated-CV variability and errors across response regions or classes. Check noisy
             observations, reduce unstable model complexity, and improve coverage of poorly represented cases.
         * - Predictions do not rank the points in the lower/upper half of the y range (Section B.6)
           - Regression
           - **Moderate.** Inside one half of the response range, the out-of-fold CV predictions are flat or
             in the wrong order (Spearman rank ≤ 0.05 between observed and predicted values), even if the
             overall error looks acceptable. The model cannot tell compounds apart in that zone, so it should
             not be used to select or rank them there.
           - Check whether the descriptors carry information in that region (e.g. plot the response against
             the most important descriptors) and add descriptors or data that cover it; otherwise, restrict
             the use of the model to the zone where it ranks the points correctly.
         * - Response may have two separate groups, classification might fit better (Section C)
           - Regression
           - **Moderate.** A large response gap separates two sizeable groups, or
             VERIFY's cluster test fails. Good apparent regression performance may mainly reflect group
             membership instead of a relationship describing variation within groups.
           - Inspect the response distribution and within-group predictions. Consider classification
             when scientifically meaningful discrete groups match the intended application; otherwise
             improve within-group coverage and investigate the source of the gap.
         * - Uneven y distribution (Section C)
           - Regression
           - **Moderate.** The most populated of four equal-width response intervals
             contains more than four times as many points as the least populated interval. An empty
             interval also triggers this warning. Aggregate metrics can hide weak coverage in sparse regions.
           - Inspect performance across the response range and collect observations in sparse regions.
             Explain coverage limitations and assess their relevance to intended predictions.
         * - Slightly uneven y distribution (Section C)
           - Regression
           - **Moderate.** The largest response-interval population is more than twice,
             but no more than four times, the smallest. Less populated regions can have less reliable
             predictions even when overall metrics look satisfactory.
           - Check errors separately across the four response intervals. Improve coverage where needed
             and report differences in reliability across the response range.
         * - Uneven class distribution (Section C)
           - Classification
           - **Moderate.** The largest class is more than 1.5 times, but no more
             than three times, the smallest. Overall performance can obscure weaker minority-class predictions.
           - Inspect class-specific performance and improve minority-class coverage. Evaluate appropriate
             class weighting within CV and document the balance expected in the intended application.
         * - Low-relevance descriptor(s) detected, try removing them (Section D)
           - Regression and classification
           - **Moderate.** A descriptor other than the leading one has
             a reported PFI or SHAP importance magnitude below 4% of the leading magnitude or below 0.05
             in absolute terms. Such descriptors may add noise or complexity with little predictive benefit.
           - Review PFI, SHAP, and descriptor correlations together. Try removing low-impact descriptors
             and compare CV performance before adopting the reduced model.
         * - Highly correlated features (Section D)
           - Regression and classification
           - **Moderate.** At least one descriptor pair has an absolute
             Pearson correlation above 0.84. Redundant inputs can destabilize fitted coefficients and
             complicate the interpretation of individual descriptor contributions.
           - Review correlated pairs and their scientific meaning. Consider retaining a representative
             descriptor or using appropriate regularization, and compare validation performance.
         * - Moderately correlated features (Section D)
           - Regression and classification
           - **Moderate.** The strongest flagged descriptor correlation
             is above 0.80 and at most 0.84 in absolute value. Overlapping information can add redundancy
             and make individual descriptor importance harder to interpret.
           - Inspect the correlation map alongside PFI and SHAP. Test whether removing redundant
             descriptors simplifies the model without harming validated performance.
         * - Potential "faulty" outliers (Section E)
           - Regression
           - **Moderate.** At least one reported prediction error exceeds 6.5 SDs
             in the outlier analysis. Such an extreme residual may reflect an input or measurement
             problem, or a valid observation outside the model's reliable domain.
           - Check the original measurements, units, descriptors, and observation identity. Correct
             verified errors; retain and explain valid unusual observations or improve domain coverage.
         * - More than 5% of points are outliers, check and explain them (Section E)
           - Regression
           - **Moderate.** More than 5% of observations across the analyzed CV and
             test sets are marked as outliers using the configured ``t_value`` cutoff (default: 2 SDs).
             Many large residuals can indicate noise, missing relationships, or heterogeneous data.
           - Review flagged observations and shared error patterns. Check data consistency and missing
             descriptors, investigate distinct populations, and explain valid outliers without removing
             them solely to improve metrics.
         * - A simpler MVL model scores as high or higher in Interpolation (Section F)
           - Regression
           - **Moderate.** With ``--all_models``, MVL's Interpolation score is at least
             as high as the selected model's. Extra complexity has not improved the assessed
             interpolation reliability.
           - Compare the Section F scores and underlying diagnostics. Consider MVL if interpolation
             is the intended use, while accounting for the boundary-performance tradeoff.
         * - A simpler MVL model scores as high or higher in Interpolation and Boundary robustness (Section F)
           - Regression
           - **Moderate.** With ``--all_models``, MVL matches or exceeds both CheckML
             scores of the selected model. The score comparison provides no reliability advantage
             for the more complex model.
           - Review MVL's metrics and diagnostics as a simpler candidate, then validate the final
             choice for the intended application.
         * - A simpler logistic regression model scores as high or higher in Interpolation (Section F)
           - Classification
           - **Moderate.** With ``--all_models``, LOGISTIC's Interpolation score
             matches or exceeds the selected classifier's. Additional complexity has not improved
             the reliability summarized by that score.
           - Compare LOGISTIC's metrics, class-specific behavior, and diagnostics with the selected
             classifier, and consider using the simpler model.
         * - A simpler MVL model has a similar or lower error in the model screening (Section F)
           - Regression
           - **Moderate.** GENERATE's combined screening metric finds MVL similar
             to or better than the selected model under the 15% tolerance. Screening has not established
             a clear accuracy benefit from the more complex model.
           - Compare the screened MVL model using the same descriptors and full CheckML evaluation.
             Choose the model that best meets the application's accuracy and robustness needs.
         * - A simpler logistic regression model has a similar or lower error in the model screening (Section F)
           - Classification
           - **Moderate.** GENERATE's combined screening metric finds LOGISTIC
             similar to or better than the selected classifier under the 15% tolerance. The selected
             classifier has no clear screening advantage over the linear baseline.
           - Compare LOGISTIC with the same descriptors using the full evaluation, including MCC
             and class-specific errors, before retaining the more complex classifier.
         * - {parameter} = 1 (integer) means ONE sample/feature in scikit-learn, use 1.0 if all of them were intended (Section H)
           - Regression and classification
           - **Moderate.** A parameter accepting both counts and fractions
             was supplied as the integer ``1``. This selects one sample or feature, whereas ``1.0``
             represents all of them, and can unintentionally produce nearly constant predictions.
           - Check the intended parameter meaning and the saved input file. Use ``1.0`` if the full
             fraction was intended, preserve its type when exporting parameters, and rerun the evaluation.

      **Boundary robustness**

      .. list-table::
         :header-rows: 1
         :widths: 28 16 33 23
         :class: checkml-warning-table

         * - Warning as printed in the PDF
           - Problem type
           - Explanation, trigger, and implications
           - Suggested actions
         * - Tree-based algorithm, not designed to extrapolate outside the training range
           - Regression
           - **Severe.** The algorithm is recognized as tree based, so this warning
             appears regardless of its boundary scores. Tree predictions can flatten beyond the training
             domain, making them unsuitable for assuming a continuing response trend outside that domain.
           - Limit use to supported regions, add observations covering the intended prediction range,
             or validate a model with an appropriate extrapolating functional form for the application.
         * - Nearest-neighbour algorithm, not designed to extrapolate outside the training range
           - Regression
           - **Severe.** The algorithm is recognized as a nearest-neighbour model,
             regardless of boundary scores. Predictions average nearby training responses and cannot
             extend beyond the response range represented by those neighbours.
           - Improve training coverage around intended predictions or restrict use to supported regions.
             If extrapolation is required, compare and independently validate an appropriate alternative model.
         * - Difficult to extrapolate new predictions in the upper zone (Section B.1)
           - Regression
           - **Moderate.** The upper-zone total is below 2.0/4.0, summing prediction,
             Spearman-ranking, and degradation components (1a-1c). Predictions of the highest-response
             held-out points have weak accuracy, ordering, or robustness relative to the remaining data.
           - Inspect components 1a-1c to identify the cause. Improve high-response coverage and descriptors,
             and verify that the selected algorithm can represent the expected upper-range trend.
         * - Difficult to extrapolate new predictions in the lower zone (Section B.2)
           - Regression
           - **Moderate.** The lower-zone total is below 2.0/4.0, summing prediction,
             Spearman-ranking, and degradation components (2a-2c). Predictions of the lowest-response
             held-out points have weak accuracy, ordering, or robustness relative to the remaining data.
           - Inspect components 2a-2c to identify the cause. Improve low-response coverage and descriptors,
             and verify that the selected algorithm can represent the expected lower-range trend.
         * - Poor extrapolation to descriptors very different from training (Section B.3)
           - Regression
           - **Moderate.** The applicability-domain score is below 1.0/2.0.
             Prediction error is high on the held-out, high-leverage observations, indicating limited
             reliability for unusual combinations of descriptors.
           - Inspect the leverage and applicability-domain plots. Check unusual descriptors for errors,
             add representative observations from the intended domain, and define supported prediction limits.
         * - A simpler MVL model scores as high or higher in Boundary robustness (Section F)
           - Regression
           - **Moderate.** With ``--all_models``, MVL's Boundary robustness score is at
             least as high as the selected model's. The selected model has no assessed
             boundary-robustness advantage over the linear baseline.
           - Compare boundary components and interpolation performance. Consider MVL if its behavior
             better matches the prediction domain required by the application.
         * - A simpler MVL model scores as high or higher in Interpolation and Boundary robustness (Section F)
           - Regression
           - **Moderate.** With ``--all_models``, MVL matches or exceeds both CheckML
             scores of the selected model. The score comparison provides no reliability advantage
             for the more complex model.
           - Review MVL's metrics and diagnostics as a simpler candidate, then validate the final
             choice for the intended application.

      |br|

   .. tab-item:: CV-Test Split Diagnostic

      .. note::
        A potential concern during development of the scoring system was whether reserving
        20% of the data as an independent test set, in addition to the CV set,
        could adversely affect model performance. See below.

      For each regression or classification model, PREDICT also uses 10x 5-fold
      CV on the full internal dataset (the 80% CV pool plus the 20% test pool). For regression, the
      report's **CV-Test Split Diagnostic** box compares this full-data baseline with
      CV RMSEs from the four CV pools used in CheckML: the interpolation CV pool,
      the bottom 80% remaining after holding out high-y points, the top 80% remaining after holding out
      low-y points, and the 80% remaining after holding out high-leverage points. For classification,
      only the interpolation CV pool is compared, using the MCC instead of the RMSE.

      Each absolute percentage difference is
      ``abs((CV RMSE / full-data CV RMSE) - 1) * 100`` (or the same with MCC). Differences up to 15% show **Pass** (small
      differences between CV and full-data CV), above 15% through 30% show **Unclear**, and above 
      30% show **Warning**. These indicators do not change the CheckML scores, but indicate whether
      the four different sub-scores can be reliably used for analyzing the model.

      .. warning::
         **What to do with Unclear or Warning results.** An **Unclear** or **Warning** status means that
         holding out the test set changes the model's performance noticeably, so the corresponding
         sub-score does not reliably represent the model trained on all the data:

         * **Be very careful with (Unclear) or do not use (Warning) those sub-scores** 
           (Interpolation, extrapolation to high or low y values, or
           descriptor-space extrapolation, depending on the row flagged) to support claims about the model.
         * **Carry out the corresponding analyses yourself** to check that the model behaves correctly in
           that aspect, e.g. additional or repeated validation splits, overfitting analysis, y-shuffle 
           and other statistical tests, split dependency, external validation data, etc.

         A **Pass** status means that the sub-score can be used as reported.

      The PREDICT DAT file contains all the metrics used to perform the different diagnostics.

      |br|

   .. tab-item:: How is the score calculated?

      **Score components**

      .. tab-set::
         :class: checkml-score-sections

         .. tab-item:: CheckML calibration

            The numerical thresholds used to assign points across the different sections were calibrated using 112
            representative models. Their suitability was further assessed based on the diverse perspectives of the
            chemical ML community represented among the co-authors of the CheckML manuscript.

            Where a table shows linear interpolation, scores change gradually between the listed thresholds rather
            than jumping between integer values. Component scores and section totals are displayed to one decimal
            place. Score bars and colors still use the nearest integer.

            .. warning::
               In this context, a published model may still be useful for its intended application even when weak scores or
               warnings are obtained. For example, a model may receive a poor extrapolation score, but it could still be
               useful if it is intended to cover only a well-defined region of the data space. Therefore, a low score should
               not necessarily be interpreted as evidence that a model is unusable; rather, it should prompt the authors to
               assess whether the identified limitation is relevant to the intended application and to provide an
               appropriate justification within the corresponding manuscript.

            |br|

         .. tab-item:: Interpolation (0 to 10)

            **How reliably the model predicts within the range of data it was trained on.**

            These checks apply to both prediction types, using RMSE/y-range for regression
            and an MCC-based definition for classification.

            .. tab-set::

               .. tab-item:: 1. Model vs "flawed" models

                  The tests conducted within the VERIFY module are regarded as score indicators:

                  *  y-mean test: Calculates the accuracy of the model when all the predicted y values are fixed to the mean of the measured y values (straight line when plotting measured vs predicted y values).
                  *  y-shuffle test: Calculates the accuracy of the model after shuffling randomly all the measured y values.
                  *  onehot test: Calculates the accuracy of the model after replacing descriptors with 0s and 1s. Continuous descriptors are split at their median (values at or above the median become 1; lower values become 0). The test is skipped if any descriptor has exactly two unique values or if all descriptors become constant after binarization.
                  *  cluster test (regression only): Calculates the accuracy of a trivial baseline that splits the descriptors (X) into two groups with k-means and predicts each point using its own group's average y value. This flags datasets that are really just two separated point clouds, where a model can look deceptively good while only having learned "which group is this point in." This baseline's own prediction (a cluster's mean y) isn't a valid class label, so it doesn't carry over to classification as-is - it's shown as N/A there instead of being scored.

                  The y-mean and y-shuffle tests are valuable in identifying overfitted and underfitted models.
                  The one-hot test identifies models that are insensitive to specific values but instead focus
                  on the presence of such values (i.e., reaction datasets filled with 0s where compounds are not used).
                  The cluster test identifies models that succeed only by exploiting a coarse split in descriptor space
                  rather than a genuine structure-property relationship.

                  Section 1 only subtracts points: each applicable test can incur up to a 2-point penalty, for a total range of -8 to 0 in regression and -6 to 0 in classification. The cluster test is not applicable to classification, and other tests marked N/A are skipped.

                  For each applicable test, ROBERT converts the relative degradation from the original CV result
                  into a penalty. Degradation means an increase in RMSE for regression, or a decrease in MCC for
                  classification. The penalty is capped at -2 and reaches zero at 45% degradation:

                  ============== ======================================================================
                  Penalty        Relative degradation from the original CV result
                  ============== ======================================================================
                  -2             ≤ 15%
                  0 to -2        15% < degradation < 45% (linear interpolation; -1 at 30%)
                  0              ≥ 45%
                  ============== ======================================================================

                  The penalties are summed across applicable tests. Tests marked N/A are skipped; when the original
                  CV score is zero, the test status is still reported but no relative penalty can be calculated.

                  The following examples show how the degradation is measured (the % above each bar is the
                  relative change of each "flawed" model with respect to the original CV result):

                  .. figure:: images/reg_verify.jpg
                     :width: 400
                     :align: center


                  .. figure:: images/clas_verify.jpg
                     :width: 400
                     :align: center


                  |br|

               .. tab-item:: 2-3. CV and test predictions

                  Sections 2 and 3 use the same scoring metrics and thresholds:

                  * **Section 2. CV predictions of the model:** evaluates predictions made during cross-validation.
                  * **Section 3. Test set predictions:** applies the same assessment to the independent test set.

                  For regression, each section starts from a scaled RMSE score (0 to 2 points) and adds a penalty
                  (0 to -2) based on R\ :sup:`2`; the result is floored at zero. Classification uses MCC on the same
                  0-2 scale.

                  **Which R**\ :sup:`2` **does ROBERT use?** The coefficient of determination of the predictions
                  (scikit-learn's ``r2_score``):

                  .. math::

                     R^2 = 1 - \frac{\sum_i (y_i - \hat{y}_i)^2}{\sum_i (y_i - \bar{y})^2}

                  In words, it compares the error of the model's predictions (numerator) with the error of simply
                  predicting the mean value of y for every point (denominator):

                  * R\ :sup:`2` = 1: perfect predictions (all points on the y = x line).
                  * R\ :sup:`2` = 0: the model is no better than predicting the mean.
                  * R\ :sup:`2` < 0: the model is worse than predicting the mean.

                  This is **not** the squared Pearson correlation (r\ :sup:`2`) between measured and predicted
                  values, which only checks whether the points lie on *some* straight line, not on the y = x line.
                  For example, a model whose predictions are always exactly twice the real values has
                  r\ :sup:`2` = 1, although every prediction is wrong; its R\ :sup:`2` is low, as it should be.
                  ROBERT follows `Alexander, Tropsha and Winkler <https://doi.org/10.1021/acs.jcim.5b00206>`__
                  (*J. Chem. Inf. Model.* 2015, 55, 1316): *"the value of R*\ :sup:`2` *should be calculated for the
                  test data using Equation 1, not from a regression of observed on predicted values"*.

                  **Q**\ :sup:`2` **and R**\ :sup:`2`: **same formula, different predictions.** In QSAR/QSPR
                  modeling, the R\ :sup:`2` calculated on cross-validated predictions is called Q\ :sup:`2`. ROBERT
                  keeps this convention, so the only difference between the two is the set of predictions used:

                  .. list-table::
                     :header-rows: 1
                     :widths: 18 12 70

                     * - Section
                       - Name
                       - Predictions used in the formula
                     * - 2. CV predictions
                       - Q\ :sup:`2`
                       - Out-of-fold predictions of the 10x 5-fold CV (each point is predicted by models that did
                         not see it during training), pooled over the 10 repetitions. This equals the average
                         Q\ :sup:`2` of the individual repetitions.
                     * - 3. Test set
                       - R\ :sup:`2`
                       - Predictions of the test set, using the mean of the test values as :math:`\bar{y}`.

                  The test R\ :sup:`2` is usually lower than the CV Q\ :sup:`2` for a similar prediction error:
                  in low-data sets the test set has only 4-9 points and the even split keeps the most extreme
                  y values in training, so the denominator (the spread of the test values) is small. The scaled RMSE
                  (Section 3) and the CV vs test comparison (Section 5) put this value in context.

                  Both sections use the same penalty thresholds (table below). Alexander *et al.* consider a test
                  R\ :sup:`2` > 0.6 as good (here, a 1-point penalty), and values ≥ 0.8 correspond to the scaled RMSE
                  that already gets full points (around 10% of the y range).

                  ============ ================================================================
                  Score        Scaled RMSE (regression)
                  ============ ================================================================
                  2            <= 10%
                  0 to 2       10% < scaled RMSE < 30% (linear interpolation; 1 at 20%)
                  0            >= 30%
                  ============ ================================================================

                  ============ ================================================================
                  Penalty       Q\ :sup:`2` (Section 2) or R\ :sup:`2` (Section 3), regression
                  ============ ================================================================
                  0             >= 0.8
                  0 to -2       0.4 < value < 0.8 (linear interpolation; -1 at 0.6)
                  -2            <= 0.4
                  ============ ================================================================

                  ============ ================================================================
                  Score        MCC (classification)
                  ============ ================================================================
                  0            <= 0
                  0 to 2       0 < MCC < 0.6 (linear interpolation; 1 at 0.3)
                  2            >= 0.6
                  ============ ================================================================

                  |br|

               .. tab-item:: 4-5. Train/validation and CV/test comparisons

                  Sections 4 and 5 use the same comparison metrics and scoring thresholds, but compare different
                  pairs of data:

                  * **Section 4. Train vs validation gap:** in regression, compares out-of-fold validation RMSE with
                    in-fold training RMSE, averaged across CV folds and repeats. In classification, it compares the
                    training MCC with the out-of-fold CV MCC.
                  * **Section 5. CV vs test consistency:** in regression, compares test-set scaled RMSE with CV scaled
                    RMSE. In classification, it compares test MCC with CV MCC.

                  For regression, each comparison uses its ratio on the same 0-2 scale:

                  ============== ======================================================================
                  Score          RMSE comparison ratio
                  ============== ======================================================================
                  2              <= 1.25
                  0 to 2         1.25 < ratio < 1.75 (linear interpolation; 1 at 1.5)
                  0              >= 1.75
                  ============== ======================================================================

                  For classification, both sections use the MCC difference (ΔMCC):

                  ============ ============================================================
                  Score        MCC difference (ΔMCC)
                  ============ ============================================================
                  2            <= 0.15
                  0 to 2       0.15 < ΔMCC < 0.45 (linear interpolation; 1 at 0.30)
                  0            >= 0.45
                  ============ ============================================================

               .. tab-item:: 6. Prediction stability

                  **Regression**

                  The model's uncertainty is estimated from the 10 repetitions of the 10x 5-fold CV, combining three facets
                  of stability into one averaged score, shown to one decimal place:

                  *  **(a) Test-set stability:** SD of the repeated-CV predictions on the test set, ×4 to approximate a 95% CI, as a percentage of the y range.
                  *  **(b) Train/validation stability:** the same SD-based check, computed instead on the out-of-fold train+validation predictions.
                  *  **(c) RMSE stability:** the coefficient of variation (SD / mean) of the aggregate RMSE across the 10 CV repeats.

                  For facets (a) and (b), the score decreases linearly from 2 at 25% y-range coverage to 0 at 75%:

                  ============ ======================================================================
                  Score        4*SD as a percentage of the y range
                  ============ ======================================================================
                  2            ≤ 25%
                  0 to 2       25% < coverage < 75% (linear interpolation; 1 at 50%)
                  0            ≥ 75%
                  ============ ======================================================================

                  For facet (c), the score decreases linearly from 2 at 15% RMSE coefficient of variation to 0 at 35%:

                  ============ ======================================================================
                  Score        RMSE coefficient of variation
                  ============ ======================================================================
                  2            ≤ 15%
                  0 to 2       15% < CV < 35% (linear interpolation; 1 at 25%)
                  0            ≥ 35%
                  ============ ======================================================================

                  The final score starts with the average of (a), (b) and (c), rounded to one decimal place. It then
                  checks that the model performs homogeneously across the y range, splitting the out-of-fold CV
                  predictions into the lowest and highest halves of y, and applies the more severe of two penalties:

                  *  **Symmetric RMSE ratio:** is the error similar in both halves? For the low/high RMSE ratio
                     ``r``, the factor is ``max(r, 1/r)``; an undefined or zero ratio receives the maximum penalty.
                  *  **Spearman rank inside each half:** does the model still rank the points inside each half, or are
                     its predictions flat (or in the wrong order) there? It uses the Spearman rank correlation between
                     observed and predicted values (each point's prediction averaged over the CV repeats), and the
                     worse half sets the penalty.

                  ============== ======================================================================
                  Penalty        Symmetric RMSE ratio factor
                  ============== ======================================================================
                  0              ≤ 1.25
                  0 to -2        1.25 < factor < 1.75 (linear interpolation; -1 at 1.5)
                  -2             ≥ 1.75
                  ============== ======================================================================

                  ============== ======================================================================
                  Penalty        Spearman rank in the worse 50% half of y
                  ============== ======================================================================
                  0              ≥ 0.3
                  0 to -2        -0.2 < Spearman < 0.3 (linear interpolation; -1 at 0.05)
                  -2             ≤ -0.2
                  ============== ======================================================================

                  The Spearman check catches models whose overall error looks acceptable but that cannot tell
                  compounds apart in part of the range, for example a model that predicts almost the same value for
                  every compound in the lower half of y while the observed values there span several units. In that
                  case the model is not useful to select or rank compounds in that zone, whether the cause is the model
                  or descriptors without information in that region. The thresholds are moderate on purpose: in
                  low-data sets each half has only ~8-25 points, so Spearman values between 0 and 0.3 cannot be told
                  apart from noise. R\ :sup:`2` is not used here: inside a half the y range is narrow, so R\ :sup:`2`
                  mostly reflects how spread that half's values are rather than the model. The final stability
                  score is clamped to 0-2 points.

                  .. figure:: images/stability_real.jpg
                     :width: 560
                     :align: center


                  *Real (a) test-set and (b) train/validation SD plots from an actual ROBERT report — facet (c),*
                  *RMSE coefficient of variation, is a single number and isn't plotted.*

                  The examples below illustrate how facet (a), the test-set stability check, is measured
                  (see the table above for the corresponding scores):

                  .. figure:: images/sd_explain.jpg
                     :width: 400
                     :align: center


                  .. figure:: images/sd_examples.jpg
                     :width: 400
                     :align: center


                  **Classification**

                  The same three facets, adapted for a discrete label instead of a continuous one — computed from the same
                  repeated-CV predictions, using the majority-voted class already reported in Sections 2 and 3:

                  *  **(a) Test-set stability:** disagreement rate between individual CV repeats and the majority-voted class, on the test set.
                  *  **(b) Train/validation stability:** the same disagreement-rate check, computed on the out-of-fold train+validation predictions.
                  *  **(c) MCC stability:** the coefficient of variation of the MCC computed separately for each of the 10 CV repeats.

                  Each classification facet decreases linearly from 2 at 15% disagreement or coefficient of variation
                  to 0 at 35%:

                  ============ ===================================================================
                  Score        Disagreement or MCC coefficient of variation
                  ============ ===================================================================
                  2            ≤ 15%
                  0 to 2       15% < value < 35% (linear interpolation; 1 at 25%)
                  0            ≥ 35%
                  ============ ===================================================================

                  The final score is the average of (a), (b) and (c), rounded to one decimal place. All three facets
                  use percentages, so they share the same scale.

                  |br|

         .. tab-item:: Boundary robustness (0 to 10 for regression only)

            **How well the model holds up at the edges of y ranges and on the most descriptor-wise unusual points (related to extrapolation).**

            Not defined for classification: none of the sub-metrics below have a well-defined discrete-label
            equivalent yet, so classification reports show only the Interpolation score (see :ref:`Score ranges <score-ranges>` below).

            For regression, ROBERT sorts the dataset by its target value and runs **two independent 80/20 splits**
            of it — not one three-way split. The "Low" split holds out the bottom 20% of points as its test fold and
            trains on the remaining top 80%; the "High" split holds out the top 20% and trains on the remaining
            bottom 80%. Each split is evaluated with its own 10x 5-fold CV over its own 80% training portion, so
            a point in the top 80% of one split can be in the held-out 20% of the other. The six scored components
            in items 1 and 2 come from these two extreme folds; item 3 uses a different, X-descriptor space definition of
            "boundary" instead.

            The High boundary (item 1) and Low boundary (item 2) each contribute up to 4 points; applicability domain
            (item 3) contributes up to 2. Their sum remains the 0-10 Boundary robustness score.

            .. figure:: images/sorted_low_high_diagram.svg
               :width: 560
               :align: center


            *Not statistical outliers — simply the most extreme 20% of real values in the dataset. Items 1 and 2*
            *each contain three Low/High components: (a) sorted CV, (b) Spearman rank, and (c) degradation ratio; item 3*
            *breaks the y-space pattern on purpose (see* |u| 3. Applicability domain (leverage) |/u| *below).*

            .. tab-set::

               .. tab-item:: 1a and 2a Sorted CV RMSE

                  These scores use the held-out low-y or high-y 20% test-set RMSE. The separate
                  CV RMSE on the remaining 80% is used only in the **CV-Test Split Diagnostic** box.

                  The scaled RMSE score decreases linearly from 2 at 10% to 0 at 30%:

                  ============== ======================================================================
                  Score          Scaled RMSE (Low or High fold)
                  ============== ======================================================================
                  2              ≤ 10%
                  0 to 2         10% < scaled RMSE < 30% (linear interpolation; 1 at 20%)
                  0              ≥ 30%
                  ============== ======================================================================

                  Each of these two items also carries a **crossing-rate penalty**, based on the fraction of Low/High
                  predictions that actually extrapolate past the training range instead of being clipped back into it
                  (a model that never truly leaves the training range at the boundary is not really being tested there):

                  ============== =================================================================
                  Penalty        Crossing rate (Low or High fold)
                  ============== =================================================================
                  -2             ≤ 20%
                  0 to -2        20% < crossing rate < 80% (linear interpolation; -1 at 50%)
                  0              ≥ 80%
                  ============== =================================================================

                  .. figure:: images/crossing_rate_diagram.svg
                     :width: 560
                     :align: center


                  *Same four extreme points, two different model behaviors. "Crossing" is the fraction of Low/High*
                  *predictions that actually land beyond the training range rather than clipping to it — logged in the*
                  *PDF as* ``Beyond training range (Low/High)``. *Tree-based models (RF, GB) tend to score poorly here: a*
                  *decision tree's prediction is always the average of some training leaf, so it structurally cannot*
                  *predict a value outside the range it was trained on.*

                  The RMSE score and the crossing penalty are combined and floored at 0, so this item can never go negative.

                  .. figure:: images/boundary_low_high_real.jpg
                     :width: 560
                     :align: center


                  *Real Low/High fold scatter plots from an actual ROBERT report — this is what Section B of the*
                  *PDF actually shows for components 1a and 2a.*

               .. tab-item:: 1b and 2b Spearman rank

                  Spearman rank correlation between measured and predicted y values, computed separately within each
                  extreme fold, to check that the model still ranks points correctly even where it can't hit them exactly.
                  A model can be systematically biased at the edges (consistently predicting a bit high or low) and still
                  be useful if it correctly tells you which of two extreme points is more extreme than the other — getting
                  the order wrong is a worse sign than getting the scale wrong.

                  .. figure:: images/spearman_rank_diagram.svg
                     :width: 560
                     :align: center


                  *Parallel lines mean the model's predicted order matches the true order; crossings mean it doesn't.*

                  ============== =======================================================
                  Score          Spearman (Low or High fold)
                  ============== =======================================================
                  1              ≥ 0.5
                  0 to 1         0.2 < Spearman < 0.5 (linear interpolation; 0.5 at 0.35)
                  0              ≤ 0.2
                  ============== =======================================================

                  Each fold's Spearman score is reduced by a penalty for compressed predictions. ROBERT calculates
                  both the predicted/measured standard-deviation ratio and predicted/measured y-range ratio, then
                  uses the more severe penalty for that fold. For either ratio ``r``, the penalty is
                  ``clamp((r - 0.5) / 0.3, -1, 0)``:

                  ============== ==============================================================
                  Penalty        More severe of the SD and range ratios
                  ============== ==============================================================
                  0              Both ratios ≥ 0.5
                  0 to -1        Lower ratio is between 0.2 and 0.5 (linear interpolation; -0.5 at ratio 0.35)
                  -1             Either ratio ≤ 0.2
                  ============== ==============================================================

                  The penalty is added to the Spearman score for each fold, then that adjusted score is clamped to
                  0-1. The High and Low fold scores are shown separately as items 1b and 2b.

               .. tab-item:: 1c and 2c Degradation ratio

                  Compares each extreme fold's scaled RMSE against the RMSE of the *remaining* 80% of the data, so a
                  model can't hide a badly-degraded boundary behind a fine-looking interior.

                  .. figure:: images/degradation_ratio_diagram.svg
                     :width: 420
                     :align: center


                  *Reported in the PDF as e.g.* ``Degradation vs 80% RMSE: High 1.65x, Low 1.61x`` *— both sides scored*
                  *independently, so one badly-degraded side can't hide behind a fine one.*

                  Each fold receives up to 1 point: its score is 1 at ratios up to 1.5x, then decreases linearly to
                  0 at 1.75x.

                  ============== ======================================================
                  Score          RMSE vs remaining 80% (Low or High fold)
                  ============== ======================================================
                  1              ≤ 1.5x
                  0 to 1         1.5x < ratio < 1.75x (linear interpolation; 0.5 at 1.625x)
                  0              ≥ 1.75x
                  ============== ======================================================

               .. tab-item:: 3 Applicability domain (leverage)

                  Unlike items 1-2, this metric measures robustness in **X-descriptor space** rather than y-space: ROBERT
                  trains on the 80% "typical" points and tests on the 20% highest-leverage points, using the classical QSAR
                  statistical leverage (the diagonal of the hat matrix, :math:`h_i = x_i^T (X_{train}^T X_{train})^{-1} x_i`)
                  to identify which points are the most structurally/descriptor-wise extreme — the same quantity plotted on
                  a Williams plot, with the usual warning threshold :math:`h^* = 3(p+1)/n`.

                  Its score uses the held-out high-leverage test-set RMSE. The CV RMSE pooled over
                  the typical 80% supplies the corresponding **CV-Test Split Diagnostic** comparison.

                  .. figure:: images/leverage_diagram.svg
                     :width: 560
                     :align: center


                  *Leverage looks at X, not y. A point can have a perfectly ordinary target value and still be structurally*
                  *novel — an unusual combination of descriptors the model has barely seen examples of.*

                  .. figure:: images/boundary_leverage_real.jpg
                     :width: 560
                     :align: center


                  *Real output from an actual ROBERT report: the applicability-domain scatter (left) and the*
                  *Williams plot itself — leverage on the x-axis, standardized residual on the y-axis, with the*
                  *h\* threshold marked as a vertical dashed line (right).*

                  The scaled RMSE score decreases linearly from 2 at 10% to 0 at 30%, as in Section 2:

                  ============== ======================================================================
                  Score          Scaled RMSE (high-leverage 20%)
                  ============== ======================================================================
                  2              ≤ 10%
                  0 to 2         10% < scaled RMSE < 30% (linear interpolation; 1 at 20%)
                  0              ≥ 30%
                  ============== ======================================================================

         .. tab-item:: Score ranges

            For regression, each of the two scores (Interpolation and Boundary robustness) is rated independently on
            the same five-tier scale below — a model can, for example, be a "very strong" interpolator while only
            "moderate" at the boundary. For classification, only Interpolation is rated (see above).

            The numeric scores are displayed to one decimal place. The score bar and color use the score rounded
            half up to the nearest integer:

            ============== =====================
            Rounded score  Tier
            ============== =====================
            0-2            Very weak
            3-4            Weak
            5-6            Moderate
            7-8            Strong
            9-10           Very strong
            ============== =====================

            Some of the most common reasons for getting low scores are:

            * Unbalanced datasets (i.e., too many points in a region, too few in others)
            * Including too few datapoints
            * Including too few descriptors
            * Overfitted and underfitted models

            Different causes that might be affecting your score are included in the CheckML score section of the PDF report.

            **Very weak models:** very unreliable models.

            .. figure:: images/score_veryweak.jpg
               :width: 400
               :align: center


            **Weak models:** unreliable models.

            .. figure:: images/score_weak.jpg
               :width: 400
               :align: center


            **Moderate models:** somewhat reliable models.

            .. figure:: images/score_moderate.jpg
               :width: 400
               :align: center


            **Strong models:** reliable models.

            .. figure:: images/score_strong.jpg
               :width: 400
               :align: center


            **Very strong models:** highly reliable models.

            .. figure:: images/score_verystrong.jpg
               :width: 400
               :align: center

.. robert-score-end
