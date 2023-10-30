.. curate-modules-start

Data curation
-------------

Overview of the CURATE module
+++++++++++++++++++++++++++++

.. |curate_fig| image:: images/CURATE.jpg
   :width: 600

.. centered:: |curate_fig|

Input required
++++++++++++++

This module uses a CSV containing the initial database.

Automated protocols
+++++++++++++++++++

*  Filters off correlated descriptors.
*  Filters off variables with very low correlation to the target values (noise).
*  Filters off duplicates.
*  Converts categorical descriptors into one-hot descriptors.

Example
+++++++

An example is available in **Examples/Use of individual modules**.

.. curate-modules-end
