.. robert-score-start

ROBERT score
------------

Overview
++++++++

.. |br| raw:: html

   <br />

While experienced ML users can normally assess whether an ML workflow has yielded favorable results for 
valid reasons, inexperienced users may encounter difficulties in gauging the reliability of its predictive 
proficiency. It is widely recognized that ML algorithms can exhibit good metrics (i.e., R\ :sup:`2`, MAE, RMSE, 
and similar) in the validation set while harboring questionable predictive ability. For example, a user 
might assume that a model with good metrics is proficient at prediction, but that very model could yield 
comparably low errors when the y values are shuffled [1] or when using random numbers as descriptors, [2] 
casting doubt on its actual predictive validity. |br|

For these reasons, we designed the ROBERT score, which is a rating out of 10 designed to provide users 
with insight into the predictive capabilities of the models selected by ROBERT. We attempted to rank the 
models using guidelines aligned with modern ML research (see below). For a more detailed explanation of best practices 
for designing robust ML models, see references [3] and [4].

* [1] `y-Randomization and Its Variants in QSPR/QSAR <https://pubs.acs.org/doi/10.1021/ci700157b>`__
* [2] `Comment on “Predicting reaction performance in C–N cross-coupling using machine learning” <https://www.science.org/doi/10.1126/science.aat8603>`__
* [3] `Best practices in ML for chemistry <https://www.nature.com/articles/s41557-021-00716-z>`__
* [4] `Engineering best practices for ML <https://se-ml.github.io/practices>`__ 

.. note:: 

   Please note that the ROBERT score was developed based on the following:
   
   1) insights from previous publications on best practices for ML models;
   2) our experience with these models; and
   3) a comprehensive benchmarking process that involved the nine examples presented in the ROBERT publication (DOI: https://doi.org/10.1002/wcms.1733) along with eight additional examples from low-data regimes (DOI: https://doi.org/10.1039/D5SC00996K).
   
   The scoring ranges were established to achieve consensus among ML experts at the extremes (i.e., very weak and strong models), while allowing for varied interpretations of the intermediate scores (i.e., weak and moderate).
   
   **We are completely open to discuss any advice on how to improve the thresholds used in the score or make the score more robust!**

|br|

How is the score calculated?
++++++++++++++++++++++++++++

**Section B.1. Model vs "flawed" models (from -6 to 0 points):**
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The tests conducted within the VERIFY module are regarded as score indicators:

*  y-mean test: Calculates the accuracy of the model when all the predicted y values are fixed to the mean of the measured y values (straight line when plotting measured vs predicted y values).  
*  y-shuffle test: Calculates the accuracy of the model after shuffling randomly all the measured y values.
*  onehot test: Calculates the accuracy of the model when replacing all descriptors for 0s and 1s. If the x value is 0, the value will be 0, otherwise it will be 1.

The y-mean and y-shuffle tests are valuable in identifying overfitted and underfitted models. 
The one-hot test identifies models that are insensitive to specific values but instead focus 
on the presence of such values (i.e., reaction datasets filled with 0s where compounds are not used).

.. |space| raw:: html

   &nbsp;

============== ================================
Points         Condition
============== ================================
0               Each of the VERIFY tests passed
-• -1           Each of the unclear VERIFY tests
-••  -2         Each of the VERIFY tests failed
============== ================================

The following examples might help clarify these points:

.. |reg_verify| image:: images/reg_verify.jpg
   :width: 400

|reg_verify|

.. |clas_verify| image:: images/clas_verify.jpg
   :width: 400

|clas_verify|

|br|

**Section B.2. CV predictions of the model (2 points):**
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

In regression, two metrics (RMSE and R\ :sup:`2`) are used to ensure a more robust assessment, as a model may show low R\ :sup:`2` while maintaining an acceptable RMSE. In classification, up to 3 points are assigned based on the MCC.

============ =======================================================
Points       Scaled RMSE
============ =======================================================
|br|         **Regression**
•• 2         ≤ 10% (high predictive ability)
•\ |space| 1 ≤ 20% (moderate predictive ability)
0            > 20% (low predictive ability)
|br|         **Classification**
••• 3        MCC > 0.75 (high predictive ability)
•• 2         0.75 ≥ MCC ≥ 0.50 (moderate predictive ability)
•\ |space| 1 0.50 ≥ MCC ≥ 0.30 (low predictive ability)
0            MCC < 0.30 (very low predictive ability)
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

**Section B.3. Predictive ability & overfitting (8 points):**
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The cross-validation tests guarantee the meaningfulness of the chosen data partition and guards against data overfitting. All the tests from this section use a combined dataset with training and validation sets.

.. |u| raw:: html

   <u>

.. |/u| raw:: html

   </u>

|u| Section B.3a. Predictions test set (2 points) |/u|

In regression, two metrics (RMSE and R\ :sup:`2`) are used to ensure a more robust assessment, as a model may show low R\ :sup:`2` while maintaining an acceptable RMSE. In classification, up to 3 points are assigned based on the MCC.

============ =======================================================
Points       Scaled RMSE
============ =======================================================
|br|         **Regression**
•• 2         ≤ 10% (high predictive ability)
•\ |space| 1 ≤ 20% (moderate predictive ability)
0            > 20% (low predictive ability)
|br|         **Classification**
••• 3        MCC > 0.75 (high predictive ability)
•• 2         0.75 ≥ MCC ≥ 0.50 (moderate predictive ability)
•\ |space| 1 0.50 ≥ MCC ≥ 0.30 (low predictive ability)
0            MCC < 0.30 (very low predictive ability)
============ =======================================================


============ =======================================================
Points        R\ :sup:`2` (penalty)
============ =======================================================
|br|         **Regression**
-•• -2          R\ :sup:`2` < 0.5 
-• -1           R\ :sup:`2` < 0.7
0               R\ :sup:`2` >= 0.70 
============ =======================================================

|u| Section B.3b. Prediction accuracy test vs CV (2 points) |/u|

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

|u| Section B.3c. Avg. standard deviation (2 points) |/u|

**Regression**

The model’s uncertainty is estimated using predictions from the 10 repetitions of the 10x 5-fold CV. ROBERT then computes the average standard deviation (SD) from all predictions and multiplies it by 4 to approximate the 95% confidence interval (CI) of a normally distributed population. The score for this test depends on the uncertainty of the results, measured by the width of the 95% CI across the range of y values.

============ ======================================================================
Points       Condition
============ ======================================================================
•• 2         95% CI (or 4*SD) spans less than 25% of the y range (low uncertainty)
•\ |space| 1 95% CI spans between 25% and 50% of the y range (moderate uncertainty)
0            95% CI spans more than 50% of the y range (high uncertainty)
============ ======================================================================

The following examples might help clarify these points:

.. |sd_explain| image:: images/sd_explain.jpg
   :width: 400

|sd_explain|

.. |sd_examples| image:: images/sd_examples.jpg
   :width: 400

|sd_examples|

|u| Section B.3d. Extrapolation (sorted CV) (2 points) |/u|

Differences in the RMSE/MCC obtained across the five folds of a sorted 5-fold CV (where target values, y, are sorted from minimum to maximum and not shuffled during CV). First, the minimum RMSE/mMCC among the five folds is identified. Then, the differences between each fold’s RMSE/MCC and this minimum RMSE/MCC are evaluated

+------------+--------------------------------------------------+
| Points     | Condition                                        |
+============+==================================================+
| • 1      | Every two folds with RMSE/MCC ≤ 1.25*min RMSE/MCC |
+------------+--------------------------------------------------+

Score ranges
++++++++++++

Some of the most common reasons for getting low scores are:

* Unbalanced datasets (i.e., too many points in a region, too few in others)
* Including too few datapoints
* Including too few descriptors
* Overfitted and underfitted models

Different causes that might be affecting your score are included in the ROBERT score section of the PDF report.

**Very weak models:** very unreliable models. 

.. |veryweak_fig| image:: images/score_veryweak.jpg
   :width: 400

|veryweak_fig|

**Weak models:** unreliable models. 

.. |weak_fig| image:: images/score_weak.jpg
   :width: 400

|weak_fig|

**Moderate models:** somewhat reliable models. 

.. |moderate_fig| image:: images/score_moderate.jpg
   :width: 400

|moderate_fig|

**Strong models:** reliable models. 

.. |strong_fig| image:: images/score_strong.jpg
   :width: 400

|strong_fig|

.. robert-score-end
