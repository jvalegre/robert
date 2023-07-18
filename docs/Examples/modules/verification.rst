VERIFY (verification of predictive ability)
-------------------------------------------

Overview
++++++++

.. |verify| image:: ../../Modules/images/VERIFY.jpg
   :width: 600

.. centered:: |verify|

Required inputs
+++++++++++++++

* Previous folder from a GENERATE job.

Executing the job
+++++++++++++++++

**Instructions:**

1. First, go to the main folder where GENERATE was run in your terminal. You should see a folder called GENERATE on it.
2. Run the following command line:

.. code:: shell

    python -m robert --verify

**Options used:**

* :code:`--verify`: Use only the VERIFY module.  

Execution time
++++++++++++++

Time: ~2 seconds

System: 4 processors (Intel Xeon Ice Lake 8352Y) using 8.0 GB RAM memory

Results
+++++++

* Two donut plots (for No_PFI and PFI) with a summary of the results of the four verification tests, created in /VERIFY. 
