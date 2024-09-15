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
*  onehot test: Calculates the accuracy of the model when replacing all descriptors for 0s and 1s. If the x value is 0, the value will be 0, otherwise it will be 1.

Technical information
+++++++++++++++++++++

The test-pass thresholds are 10% (for unclear results) and 25% (for passed tests), determined by the percentage of error difference (RMSE or MCC by default) relative to the original model.
Lastly, the VERIFY module generates a bar plot summarizing the test outcomes. This visual representation distinguishes between passed, unclear and failed tests, using blue, yellow and red colors, respectively.

Example
+++++++

An example is available in **Examples/Use of individual modules**.

.. verify-modules-end
