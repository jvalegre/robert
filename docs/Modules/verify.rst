.. verify-modules-start

Predictive ability tests
------------------------

Overview of the VERIFY module
+++++++++++++++++++++++++++++

.. |verify_fig| image:: images/VERIFY.jpg
   :width: 600

.. centered:: |verify_fig|

Input required
++++++++++++++

This module uses a GENERATE folder created in a GENERATE job.

Automated protocols
+++++++++++++++++++

*  y-mean test: Calculates the accuracy of the model when all the predicted y values are fixed to the mean of the measured y values (straight line when plotting measured vs predicted y values).  
*  y-shuffle test: Calculates the accuracy of the model after shuffling randomly all the measured y values.
*  CV test: Calculates the accuracy of the model with a leave-one-out cross validation (LOOCV, for databases with less than 50 points) or a 5-fold cross validation (larger databases).
*  onehot test: Calculates the accuracy of the model when replacing all descriptors for 0s and 1s. If the x value is 0, the value will be 0, otherwise it will be 1.

Technical information
+++++++++++++++++++++

The test-pass threshold is established through the 'thres_test' option, determined by the percentage of error difference (RMSE or MCC by default) relative to the original model. The 'kfold' parameter adjusts the number of different splits utilized in cross-validation, and by default 'kfold' is set to 'auto' to assign LOOCV for small databases and 5-fold CV for larger databases.
Lastly, the VERIFY module generates a donut plot summarizing the test outcomes. This visual representation distinguishes between passed and failed tests, using blue and red colors, respectively.

Example
+++++++

An example is available in **Examples/Use of individual modules**.

.. verify-modules-end
