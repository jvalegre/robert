.. database-start

How to create a database
------------------------

CSV file
++++++++

A CSV file, known as a "comma-separated values" file, enables you to store data in a table-structured format. Data records are divided by line breaks, i.e. each data record starts a new line. CSV files usually use the file extension *.csv.

.. note:: 

   *  Creating CSV files is most straightforward when utilizing a spreadsheet application such as Excel.
   *  In a spreadsheet application, navigate to File > Save As > File Type > CSV.

.. warning::

   *  In certain PCs, the generated CSV file uses a semicolon (;) as the delimiter instead of a comma (,). We suggest users check their CSV file by opening it with Notepad.If they come across this issue, they can follow these steps: navigate to Edit > Replace, replace all semicolons with commas, and then select Replace All.

Database: Using Excel
+++++++++++++++++++++

*  **Step 1:** Open a new spreadsheet in Microsoft Excel. Any spreadsheet software can be employed to easily convert table data to CSV files.

.. |database1_fig| image:: images/1.png
   :width: 800

.. centered:: |database1_fig|

*  **Step 2:** Type each header—one for the code_name, one for each descriptor, and one for the target value—in row 1 at the top of the spreadsheet. If you would like guidance on selecting the appropriate descriptors, please refer to the following section.

.. |database2_fig| image:: images/2.png
   :width: 800

.. centered:: |database2_fig|

*  **Step 3:** Enter your data into the spreadsheet under each column as needed. For example, write the compound name in cell A2, the descriptors values in cell B2,C2 and D2, and the target value in cell E2.

.. |database3_fig| image:: images/3.png
   :width: 800

.. centered:: |database3_fig|

*  **Step 4:** Click File and select Save As.

.. |database4_fig| image:: images/4.png
   :width: 800

.. centered:: |database4_fig|

*  **Step 5:** Select CSV under the “Save as type” dropdown menu and input a title for your CSV file and proceed to click the Save button. Your CSV file is now generated, and commas will be automatically inserted to delineate each field.


.. |database5_fig| image:: images/5.png
   :width: 800

.. centered:: |database5_fig|

Selection of descriptors
++++++++++++++++++++++++

In machine learning, a "descriptor" refers to a specific feature or property of the data used to describe or represent an entity. In the context of chemistry, descriptors are characteristics of molecules or compounds that are used to quantify their structure, behavior, or other relevant attributes. Molecular descriptors can be classified in two categories: experimental and calculated descriptors. 

When creating a database, users have two approaches:

1. **User-Defined Descriptors:**

   *  Users can utilize their own descriptors, whether obtained experimentally or through computational methods.
.. |br| raw:: html
   <br />
   |br|
2. **Automated: SMILES to Descriptors:**

   *  If users don't have pre-calculated descriptors, they can opt for the automatic generation of descriptors using the AQME module.

.. |database6_fig| image:: images/6.png
   :width: 800

.. centered:: |database6_fig|

*  **User-defined descriptors**

If users want to use descriptors that have been measured or calculated before, they have to enter them manually into the database. Users can choose to use experimental, theoretical descriptors, or a mix of both to define their particular problem. Let's take a look at the article: **J. Am. Chem. Soc. 2022, 144, 9586**, where the authors tried to relate ΔG with solvents using a machine learning model.

.. |database7_fig| image:: images/7.png
   :width: 800

.. centered:: |database7_fig|

.. |epsilon| unicode:: U+03B5 .. GREEK SMALL LETTER EPSILON

Therefore, for the 20 solvents studied experimentally, the authors needed to differentiate the solvents by utilizing various properties/descriptors for each of them (defining the problem). 
To achieve this, they constructed a set of 17 molecular descriptors that capture electronic and structural differences. Some of these descriptors include the dielectric constant (ε), 
the second COSMO σ-moment, which characterizes a molecule's overall electrostatic polarity (Sig2), and the McGowan molar volume (V). This is how the database would be structured:

.. |database8_fig| image:: images/8.png
   :width: 800

.. centered:: |database8_fig|

*  **Automated: SMILES to descriptors**

In the case where the user doesn't have previously measured/calculated descriptors, they can employ AQME for 
molecular descriptor generation. Let's consider the following article: **J. Chem. Inf. Comput. Sci. 2004, 44, 3, 1000**, 
which describes a simple method for estimating the aqueous solubility (ESOL − Estimated SOLubility) of a compound 
directly from its structure. If we aim to develop a solubility predictor using ROBERT, the database requires only three columns: code_name (names of the molecules), 
SMILES (SMILES strings of the molecules), and target_value (property to predict). The columns code_name and SMILES must retain these titles, and the user defines the name of
the column containing the target value. This is how the database would be structured:

.. |database9_fig| image:: images/9.png
   :width: 800

.. centered:: |database9_fig|

ROBERT will then be executed with one command line (see the *From SMILES to predictors* section and the *Full workflow from SMILES* example), 
generating results like these:

.. |database10_fig| image:: images/10.png
   :width: 800

.. centered:: |database10_fig|