.. _robert-score-start:

CheckML score
-------------

Overview
++++++++

.. |br| raw:: html

   <br />

.. |u| raw:: html

   <u>

.. |/u| raw:: html

   </u>

.. |space| raw:: html

   &nbsp;

**How should we assess whether a published ML model is sufficiently robust, reliable, and applicable
to the chemical problem it is intended to address?**

In practice, model quality is often communicated
through a small number of statistical metrics (ie R2, RMSE, MCC, etc.). Although these metrics are useful, 
they provide only a partial view of model reliability, making it difficult 
for readers and reviewers to assess the robustness and domain of applicability of the reported models. 

To this end, we have developed CheckML, a set of tools that systematically analyze descriptor-based tabular 
ML models, evaluates multiple aspects of their performance, and provides quality scores and diagnostic 
alerts associated with potential weaknesses. The framework examines predictive performance, 
overfitting, generalization, prediction stability, and behavior at the boundaries of the modeled chemical space 
(related to extrapolation). These analyses are compiled into standardized PDF reports that document the evaluation 
and highlight aspects requiring further consideration.

.. figure:: images/checkml_summary.jpg
   :width: 400
   :align: center

|br|

.. note:: 

   Please note that the CheckML score was developed based on the following:
   
   * Established recommendations for robust ML
   * Experience with chemical ML models across a broad community of expert chemists
   * A comprehensive benchmarking process
   
   **We are completely open to discuss any advice on how to improve the thresholds used in the score or make the score more robust!**

Please, see the CheckML publication for more details.

|br|

Protocols followed by CheckML
+++++++++++++++++++++++++++++

Before evaluation, CheckML applies a standardized splitting procedure. The random seed is fixed, and rows 
and columns are sorted so that the results are independent of the order in which data are provided. 80% of 
the data are used for cross-validation (CV), while the remaining 20% (or a minimum of four data points) 
are reserved as an independent test set. 

CheckML uses 10-times repreated 5-fold CV (10x 5-fold CV) because this type of CV 
is widely used in the ML community and is the default repeated CV strategy recommended
by scikit-learn. Benchmarking across low-data regimes also showed that 10x 5-fold CV 
remains informative across a broad range of dataset sizes (16–4200 data points, 
see the CheckML publication). 

.. note::
  Although leave-one-out CV (LOOCV) is sometimes used for low-data chemical problems,
  each validation estimate is based on a single observation and the resulting 
  estimates can be highly correlated. Thus, 10x 5-fold CV provides a more informative 
  validation framework for the intended application of CheckML.

.. note::
  A potential concern during development of the scoring system was whether reserving 
  20% of the data as an independent test set, in addition to the CV set, 
  could adversely affect model performance in the low-data regime. The overall 
  effect was small (MAE in 10x 5-fold CV scaled RMSE = 1.3% across 14 exampes, 
  see the CheckML publication). Based on these results, we retained the 20% test-set
  holdout for all datasets. This provides a more informative assessment of overfitting
  and generalization while having only a minor effect on the resulting score, 
  even in the low-data regime.

After this initial splitting, different ROBERT routines are applied to evaluate the 
model as explained in the following sections.

|br|

How is the score calculated?
++++++++++++++++++++++++++++

The numerical thresholds used to assign points across the different sections were calibrated using 112 
representative models. Their suitability was further assessed based on the diverse perspectives of the 
chemical ML community represented among the co-authors of the CheckML manuscript.

Importantly, the resulting scores should not be interpreted as replacements for the underlying statistical 
metrics or as universal guarantees of predictive validity. Rather, they provide a standardized, quantitative 
quality-control layer that integrates multiple diagnostics. By explicitly testing performance against 
flawed baselines, evaluating overfitting and generalization, 
quantifying prediction stability, and probing both response- and descriptor-space boundaries, CheckML provides 
a systematic assessment of the conditions under which a model's predictions can reasonably be considered 
reliable. 

.. warning:: 
   In this context, a published model may still be useful for its intended application even when weak scores or 
   warnings are obtained. For example, a model may receive a poor extrapolatin score, but it could still be 
   useful if it is intended to cover only a well-defined region of the data space. Therefore, a low score should 
   not necessarily be interpreted as evidence that a model is unusable; rather, it should prompt the authors to 
   assess whether the identified limitation is relevant to the intended application and to provide an 
   appropriate justification within the corresponding manuscript.

|br|

Interpolation (0 to 10)
^^^^^^^^^^^^^^^^^^^^^^^

**How reliably the model predicts within the range of data it was trained on.**

**Section B.1. Model vs "flawed" models (from -8 to 0 points for regression, -6 to 0 for
classification):**

The tests conducted within the VERIFY module are regarded as score indicators:

*  y-mean test: Calculates the accuracy of the model when all the predicted y values are fixed to the mean of the measured y values (straight line when plotting measured vs predicted y values).
*  y-shuffle test: Calculates the accuracy of the model after shuffling randomly all the measured y values.
*  onehot test: Calculates the accuracy of the model when replacing all descriptors for 0s and 1s. Continuous descriptors are binarized using their own median as the cutoff (value ≥ median → 1, otherwise → 0), while descriptors that are already binary are left untouched.
*  cluster test (regression only): Calculates the accuracy of a trivial baseline that splits the descriptors (X) into two groups with k-means and predicts each point using its own group's average y value. This flags datasets that are really just two separated point clouds, where a model can look deceptively good while only having learned "which group is this point in." This baseline's own prediction (a cluster's mean y) isn't a valid class label, so it doesn't carry over to classification as-is - it's shown as N/A there instead of being scored.

The y-mean and y-shuffle tests are valuable in identifying overfitted and underfitted models.
The one-hot test identifies models that are insensitive to specific values but instead focus
on the presence of such values (i.e., reaction datasets filled with 0s where compounds are not used).
The cluster test identifies models that succeed only by exploiting a coarse split in descriptor space
rather than a genuine structure-property relationship.

============== ========================================================
Points         Condition
============== ========================================================
-••  -2         Each of the VERIFY tests failed
-• -1           Each of the unclear VERIFY tests
0               Each of the VERIFY tests passed (4 for regression, 3 for classification)
============== ========================================================

The following examples might help clarify these points:

.. figure:: images/reg_verify.jpg
   :width: 400
   :align: center


.. figure:: images/clas_verify.jpg
   :width: 400
   :align: center


|br|

**Section B.2. CV predictions of the model (2 points):**

In regression, two metrics (RMSE and R\ :sup:`2`) are used to ensure a more robust assessment, as a model may show low R\ :sup:`2` while maintaining an acceptable RMSE. In classification, points are assigned based on the MCC, on the same 0-2 scale as regression.

============ =======================================================
Points       Scaled RMSE
============ =======================================================
|br|         **Regression**
•• 2         ≤ 10% (high predictive ability)
•\ |space| 1 ≤ 20% (moderate predictive ability)
0            > 20% (low predictive ability)
|br|         **Classification**
•• 2         MCC > 0.60 (high predictive ability)
•\ |space| 1 0.60 ≥ MCC ≥ 0.30 (moderate predictive ability)
0            MCC < 0.30 (low predictive ability)
============ =======================================================

============ =======================================================
Points        R\ :sup:`2` (penalty)
============ =======================================================
|br|         **Regression**
-•• -2          R\ :sup:`2` < 0.5
-• -1           R\ :sup:`2` < 0.7
0               R\ :sup:`2` >= 0.70
============ =======================================================

|br|

**Section B.3. Test set, generalization and stability (up to 8 points for both regression and classification):**

All the tests from this section use a combined dataset with training and validation sets. B.3a-B.3d all apply
to both prediction types, using an MCC-based definition for classification wherever regression would
otherwise use RMSE/y-range.

|u| Section B.3a. Predictions test set (2 points) |/u|

In regression, two metrics (RMSE and R\ :sup:`2`) are used to ensure a more robust assessment, as a model may show low R\ :sup:`2` while maintaining an acceptable RMSE. In classification, points are assigned based on the MCC, on the same 0-2 scale as regression.

============ =======================================================
Points       Scaled RMSE
============ =======================================================
|br|         **Regression**
•• 2         ≤ 10% (high predictive ability)
•\ |space| 1 ≤ 20% (moderate predictive ability)
0            > 20% (low predictive ability)
|br|         **Classification**
•• 2         MCC > 0.60 (high predictive ability)
•\ |space| 1 0.60 ≥ MCC ≥ 0.30 (moderate predictive ability)
0            MCC < 0.30 (low predictive ability)
============ =======================================================


============ =======================================================
Points        R\ :sup:`2` (penalty)
============ =======================================================
|br|         **Regression**
-•• -2          R\ :sup:`2` < 0.5
-• -1           R\ :sup:`2` < 0.7
0               R\ :sup:`2` >= 0.70
============ =======================================================

|u| Section B.3b. Train vs validation gap (2 points) |/u|

**Regression**

Compares the out-of-fold validation RMSE with the in-fold training RMSE, both averaged over all the folds
and repeats of the CV, to catch a model that fits its training data much better than it generalizes to
held-out points — a classic sign of overfitting that the CV-average metrics in B.2 can otherwise hide.

============== ======================================================
Points         Scaled RMSE ratio
============== ======================================================
•• 2            Validation RMSE ≤ 1.25*train RMSE
•\ |space| 1    Validation RMSE ≤ 1.50*train RMSE
0               Validation RMSE > 1.50*train RMSE
============== ======================================================

**Classification**

Same idea, expressed as a ΔMCC instead of an RMSE ratio (a discrete label has no RMSE): compares the
in-fold training MCC against the out-of-fold CV MCC from Section B.2, using the same ΔMCC thresholds as
B.3c below.

============ ==============================================
Points       Condition
============ ==============================================
•• 2         MCC difference (ΔMCC) < 0.15 (low overfitting)
•\ |space| 1 0.15 ≤ ΔMCC ≤ 0.30 (moderate overfitting)
0            ΔMCC > 0.30 (high overfitting)
============ ==============================================

|u| Section B.3c. CV vs test consistency (2 points, or MCC difference for classification) |/u|

**Regression**

Differences in scaled RMSE between CV predictions of the model and Predictions test set.

============== ================================
Points         Scaled RMSE ratio
============== ================================
•• 2            Scaled RMSE (test) ≤ 1.25*scaled RMSE (CV)
•\ |space| 1    Scaled RMSE (test) ≤ 1.50*scaled RMSE (CV)
0               Scaled RMSE (test) >1.50*scaled RMSE (CV)
============== ================================

**Classification**

Calculates the model's uncertainty by comparing the MCC obtained from the model with the MCC of the CV from Section 3a.

============ ==============================================
Points       Condition
============ ==============================================
•• 2         MCC difference (ΔMCC) < 0.15 (low uncertainty)
•\ |space| 1 0.15 ≤ ΔMCC ≤ 0.30 (moderate uncertainty)
0            ΔMCC > 0.30 (high uncertainty)
============ ==============================================

|u| Section B.3d. Prediction stability (2 points) |/u|

**Regression**

The model's uncertainty is estimated from the 10 repetitions of the 10x 5-fold CV, combining three facets
of stability into one averaged, rounded score:

*  **(a) Test-set stability:** SD of the repeated-CV predictions on the test set, ×4 to approximate a 95% CI, as a percentage of the y range.
*  **(b) Train/validation stability:** the same SD-based check, computed instead on the out-of-fold train+validation predictions.
*  **(c) RMSE stability:** the coefficient of variation (SD / mean) of the aggregate RMSE across the 10 CV repeats.

============ ======================================================================
Points       Condition — facets (a) and (b)
============ ======================================================================
•• 2         95% CI (or 4*SD) spans less than 25% of the y range (low uncertainty)
•\ |space| 1 95% CI spans between 25% and 50% of the y range (moderate uncertainty)
0            95% CI spans more than 50% of the y range (high uncertainty)
============ ======================================================================

============ ======================================================================
Points       Condition — facet (c)
============ ======================================================================
•• 2         RMSE coefficient of variation ≤ 15%
•\ |space| 1 RMSE coefficient of variation ≤ 25%
0            RMSE coefficient of variation > 25%
============ ======================================================================

The final score for this item is the average of (a), (b) and (c), rounded to the nearest point.

.. figure:: images/stability_real.jpg
   :width: 560
   :align: center


*Real (a) test-set and (b) train/validation SD plots from an actual ROBERT v2.2.0 report — facet (c),*
*RMSE coefficient of variation, is a single number and isn't plotted.*

The examples below illustrate facet (a), the test-set stability check, in more detail:

.. figure:: images/sd_explain.jpg
   :width: 400
   :align: center


.. figure:: images/sd_examples.jpg
   :width: 400
   :align: center


**Classification**

The same three facets, adapted for a discrete label instead of a continuous one — computed from the same
repeated-CV predictions, using the majority-voted class already reported in Section B.2/B.3a:

*  **(a) Test-set stability:** disagreement rate between individual CV repeats and the majority-voted class, on the test set.
*  **(b) Train/validation stability:** the same disagreement-rate check, computed on the out-of-fold train+validation predictions.
*  **(c) MCC stability:** the coefficient of variation of the MCC computed separately for each of the 10 CV repeats.

============ ======================================================================
Points       Condition — facets (a), (b) and (c)
============ ======================================================================
•• 2         ≤ 15% (low uncertainty)
•\ |space| 1 ≤ 25% (moderate uncertainty)
0            > 25% (high uncertainty)
============ ======================================================================

The final score is the average of (a), (b) and (c), rounded to the nearest point — the same combination
rule as regression, reusing regression facet (c)'s 15%/25% thresholds directly for all three, since all
three are now plain percentages rather than a percentage of the y-range.

|br|

Boundary robustness (0 to 10 for regression only)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**How well the model holds up at the edges of y ranges and on the most descriptor-wise unusual points (related to extrapolation).**

Not defined for classification: none of the sub-metrics below have a well-defined discrete-label
equivalent yet, so classification reports show only the Interpolation score (see `Score ranges`_ below).

For regression, ROBERT sorts the dataset by its target value and runs **two independent 80/20 splits**
of it — not one three-way split. The "Low" split holds out the bottom 20% of points as its test fold and
trains on the remaining top 80%; the "High" split holds out the top 20% and trains on the remaining
bottom 80%. Each split is evaluated with its own 10x 5-fold CV over its own 80% training portion, so
a point in the top 80% of one split can be in the held-out 20% of the other. Four of the five sub-metrics
below come from these two extreme folds; the fifth uses a different, X-descriptor space definition of
"boundary" instead.

.. figure:: images/sorted_low_high_diagram.svg
   :width: 560
   :align: center


*Not statistical outliers — simply the most extreme 20% of real values in the dataset. Sub-metrics 1-4*
*below all follow this Low/High fold pattern; sub-metric 5 breaks it on purpose (see* |u| 5. Applicability domain (leverage) |/u| *below).*

|u| 1-2. Low / High — scaled RMSE (2 points each) |/u|

============== ================================
Points         Scaled RMSE (Low or High fold)
============== ================================
•• 2            ≤ 10%
•\ |space| 1    ≤ 20%
0               > 20%
============== ================================

Each of these two items also carries a **crossing-rate penalty**, based on the fraction of Low/High
predictions that actually extrapolate past the training range instead of being clipped back into it
(a model that never truly leaves the training range at the boundary is not really being tested there):

============== ================================================
Penalty        Crossing rate (Low or High fold)
============== ================================================
-•• -2          < 20% cross the boundary
-• -1           < 50% cross the boundary
0               ≥ 50% of predictions cross the training boundary
============== ================================================

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


*Real Low/High fold scatter plots from an actual ROBERT v2.2.0 report — this is what Section B of the*
*PDF actually shows for sub-metrics 1-2.*

|u| 3. Spearman rank (2 points) |/u|

Spearman rank correlation between measured and predicted y values, computed separately within each
extreme fold, to check that the model still ranks points correctly even where it can't hit them exactly.
A model can be systematically biased at the edges (consistently predicting a bit high or low) and still
be useful if it correctly tells you which of two extreme points is more extreme than the other — getting
the order wrong is a worse sign than getting the scale wrong.

.. figure:: images/spearman_rank_diagram.svg
   :width: 560
   :align: center


*Parallel lines mean the model's predicted order matches the true order; crossings mean it doesn't.*

============== ================================
Points         Spearman (Low or High fold)
============== ================================
•\ |space| 1    ≥ 0.5
0               < 0.5
============== ================================

|u| 4. Degradation ratio (2 points) |/u|

Compares each extreme fold's scaled RMSE against the RMSE of the *remaining* 80% of the data, so a
model can't hide a badly-degraded boundary behind a fine-looking interior.

.. figure:: images/degradation_ratio_diagram.svg
   :width: 560
   :align: center


*Reported in the PDF as e.g.* ``Degradation vs 80% RMSE: High 1.65x, Low 1.61x`` *— both sides scored*
*independently against the same 1.5x line, so one badly-degraded side can't hide behind a fine one.*

============== ================================
Points         RMSE vs remaining 80% (Low or High fold)
============== ================================
•\ |space| 1    ≤ 1.5x
0               > 1.5x
============== ================================

|u| 5. Applicability domain (leverage) (2 points) |/u|

Unlike items 1-4, this metric measures robustness in **X-descriptor space** rather than y-space: ROBERT
trains on the 80% "typical" points and tests on the 20% highest-leverage points, using the classical QSAR
statistical leverage (the diagonal of the hat matrix, :math:`h_i = x_i^T (X_{train}^T X_{train})^{-1} x_i`)
to identify which points are the most structurally/descriptor-wise extreme — the same quantity plotted on
a Williams plot, with the usual warning threshold :math:`h^* = 3(p+1)/n`.

.. figure:: images/leverage_diagram.svg
   :width: 560
   :align: center


*Leverage looks at X, not y. A point can have a perfectly ordinary target value and still be structurally*
*novel — an unusual combination of descriptors the model has barely seen examples of.*

.. figure:: images/boundary_leverage_real.jpg
   :width: 560
   :align: center


*Real output from an actual ROBERT v2.2.0 report: the applicability-domain scatter (left) and the*
*Williams plot itself — leverage on the x-axis, standardized residual on the y-axis, with the*
*h\* threshold marked as a vertical dashed line (right).*

============== ================================
Points         Scaled RMSE (high-leverage 20%)
============== ================================
•• 2            ≤ 10%
•\ |space| 1    ≤ 20%
0               > 20%
============== ================================

Score ranges
++++++++++++

For regression, each of the two scores (Interpolation and Boundary robustness) is rated independently on
the same five-tier scale below — a model can, for example, be a "very strong" interpolator while only
"moderate" at the boundary. For classification, only Interpolation is rated (see above).

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
