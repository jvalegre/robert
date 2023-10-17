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

   Please note that the ROBERT score has been created using insights from 1) prior publications on best practices for ML models, 2) our own experience with such models, and 3) a comprehensive benchmarking process involving the nine examples presented in the ROBERT publication. The scoring ranges were established with the intention of achieving a consensus among the majority of ML experts at the opposite ends of the spectrum (i.e., very weak and strong models), while the intermediate scores (i.e., weak and moderate) might allow for varied interpretations based on differing opinions.
   
   **We are completely open to discuss any advice on how to improve the thresholds used in the score or make the score more robust!**

How is the score calculated?
++++++++++++++++++++++++++++

**Predictive ability towards an external test set (2 points):**

The R\ :sup:`2` (for regression) or accuracy (for classification) of an external test set is employed to assess the predictive capabilities of the models found with ROBERT. These models undergo hyperoptimization using both a training set and a validation set. In cases where a test set is absent, metrics from the validation set are used instead. ROBERT, by default, allocates a sufficient number of data points in test/validation sets to ensure the generation of meaningful R\ :sup:`2`/accuracy scores. Furthermore, comparable outcomes were achieved when alternative metrics like RMSE or MAE were employed.

====== =============================================================================
Points Condition
====== =============================================================================
•• 2   R\ :sup:`2` > 0.85 (high correlation between predicted and measured y values)
•\ 1   0.85 > R\ :sup:`2` > 0.70 (moderate correlation)
0      R\ :sup:`2` < 0.70 (low correlation)
====== =============================================================================

**Proportion of datapoints vs descriptors (2 points):**

The ratio of datapoints to descriptors used during model hyperoptimization (in the train and validation sets) stands as another crucial parameter. Lower ratios result in simpler models that are more human-interpretable. The extensive literature on ML modeling offers numerous suggested ratios, and we endeavored to select reasonable parameters in accordance with previous recommendations.

====== ==========================================================================
Points Condition
====== ==========================================================================
•• 2   Datapoints:descriptors ratio > 10:1 (reasonably low amount of descriptors)
•\ 1   10:1 > ratio > 3:1 (moderate amount)
0      Ratio < 3:1 (too many descriptors)
====== ==========================================================================

**Passing VERIFY tests (4 points):**

The tests conducted within the VERIFY module are also regarded as score indicators:

*  5-fold CV test: Calculates the accuracy of the model with a 5-fold cross-validation.
*  y-mean test: Calculates the accuracy of the model when all the predicted y values are fixed to the mean of the measured y values (straight line when plotting measured vs predicted y values).  
*  y-shuffle test: Calculates the accuracy of the model after shuffling randomly all the measured y values.
*  onehot test: Calculates the accuracy of the model when replacing all descriptors for 0s and 1s. If the x value is 0, the value will be 0, otherwise it will be 1.

The 5-fold cross-validation test guarantees the meaningfulness of the chosen data partition and guards against data overfitting. 
The y-mean and y-shuffle tests are valuable in identifying overfitted and underfitted models. 
Finally, the one-hot test identifies models that are insensitive to specific values but instead focus 
on the presence of such values (i.e., reaction datasets filled with 0s where compounds are not used).

====== =====================================================
Points Condition
====== =====================================================
•\ 1   Each of the VERIFY tests passed (up to •••• 4 points)
====== =====================================================

**Number of outliers in the validation set (only for regression, 2 points):**

The count of outliers in the validation set is determined by analyzing the prediction errors against the mean and standard deviation of errors in the training set. The formula employed to compute the errors of the validation set in terms of standard deviation (SD) units compared to the training set errors is as follows:

.. code:: shell

    SD(valid. point) = [Error(valid. point) - mean error (training set)] / SD (training set)

By default, ROBERT adopts a t-value of 2 to identify outliers, which according to Gaussian distribution principles should lead to approximately 5% of outliers. If the validation set exhibits a high number of outliers, it could indicate overfitting in the training set or an unbalanced distribution of points within the validation set.

====== ============================================================================
Points Condition
====== ============================================================================
•• 2   Outliers < 7.5% (close to a normal distribution of errors in the valid. set)
•\ 1   7.5% < outliers < 15% (not that far from a normal distribution of errors)
0      Outliers > 15% (far from a normal distribution of errors)
====== ============================================================================

**Extra points for VERIFY tests (only for classification, 2 points):**

As outliers are not calculated for classification models, additional points are awarded for passing the y-mean and y-shuffle VERIFY tests. These specific tests were selected due to their significance in identifying potential shortcomings in the predictive capacity of the models.

====== ==========================================================
Points Condition
====== ==========================================================
•\ 1   Each y-mean and y-shuffle tests passed (up to •• 2 points)
====== ==========================================================

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
