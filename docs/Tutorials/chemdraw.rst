.. container:: step

   .. raw:: html

      <div class='step-content' data-prefix='chemdraw' data-step='1'>

   .. figure:: tutorial_images/chemdraw/chemdraw_1.png

   In this tutorial we start from molecular structures created
   in **ChemDraw**.



   ChemDraw files contain only chemical structures, without
   descriptors or datasets.



   easyROB can read these structures, convert them into a dataset,
   and then use them to train a machine learning model.

   .. raw:: html

      <div class="step-nav">
         <button onclick="nextStep('chemdraw',1)">Next</button>
      </div>


   .. raw:: html

      </div>

   .. raw:: html

      <div class='step-content' data-prefix='chemdraw' data-step='2'>

   .. figure:: tutorial_images/chemdraw/chemdraw_2.png

   First, enable the **AQME Workflow** to unlock the
   **AQME** tab.



   This tab contains the tools required to process ChemDraw
   files and convert them into a dataset that can be used
   for modelling.

   .. raw:: html

      <div class="step-nav">
         <button onclick="prevStep('chemdraw',2)">Previous</button>
         <button onclick="nextStep('chemdraw',2)">Next</button>
      </div>


   .. raw:: html

      </div>

   .. raw:: html

      <div class='step-content' data-prefix='chemdraw' data-step='3'>

   .. figure:: tutorial_images/chemdraw/chemdraw_3.png

   In the **AQME** tab, click
   **Generate CSV from ChemDraw files**.



   This option converts ChemDraw structures into a dataset
   that can later be used to train the model.

   .. raw:: html

      <div class="step-nav">
         <button onclick="prevStep('chemdraw',3)">Previous</button>
         <button onclick="nextStep('chemdraw',3)">Next</button>
      </div>


   .. raw:: html

      </div>

   .. raw:: html

      <div class='step-content' data-prefix='chemdraw' data-step='4'>

   .. figure:: tutorial_images/chemdraw/chemdraw_4.png

   A pop-up window will appear explaining the requirements
   for ChemDraw files.



   The structures must be saved in **CDXML format**, and
   the chemical drawings should be correct, since incorrect
   structures may lead to errors when the GUI reads them.

   .. raw:: html

      <div class="step-nav">
         <button onclick="prevStep('chemdraw',4)">Previous</button>
         <button onclick="nextStep('chemdraw',4)">Next</button>
      </div>


   .. raw:: html

      </div>

   .. raw:: html

      <div class='step-content' data-prefix='chemdraw' data-step='5'>

   .. figure:: tutorial_images/chemdraw/chemdraw_5.png

   Next, select the ChemDraw file containing the molecular
   structures.



   Browse to the file location and load it into the program.

   .. raw:: html

      <div class="step-nav">
         <button onclick="prevStep('chemdraw',5)">Previous</button>
         <button onclick="nextStep('chemdraw',5)">Next</button>
      </div>


   .. raw:: html

      </div>

   .. raw:: html

      <div class='step-content' data-prefix='chemdraw' data-step='6'>

   .. figure:: tutorial_images/chemdraw/chemdraw_6.png

   Once the file is loaded, a table will appear containing
   the molecules extracted from the ChemDraw file.



   Two columns are displayed: **code_name** and
   **Target**.



   Assign a name to each molecule in the **code_name**
   column.



   The **Target** column should contain the property that
   will be predicted by the machine learning model.



   In this example, the target column is renamed to
   **Yield** and filled with the corresponding values.

   .. raw:: html

      <div class="step-nav">
         <button onclick="prevStep('chemdraw',6)">Previous</button>
         <button onclick="nextStep('chemdraw',6)">Next</button>
      </div>


   .. raw:: html

      </div>

   .. raw:: html

      <div class='step-content' data-prefix='chemdraw' data-step='7'>

   .. figure:: tutorial_images/chemdraw/chemdraw_7.png

   After completing the information, click **Save as CSV**.



   A dialog window will appear asking you to choose the location
   and the name of the CSV file to be saved.

   .. raw:: html

      <div class="step-nav">
         <button onclick="prevStep('chemdraw',7)">Previous</button>
         <button onclick="nextStep('chemdraw',7)">Next</button>
      </div>


   .. raw:: html

      </div>

   .. raw:: html

      <div class='step-content' data-prefix='chemdraw' data-step='8'>

   .. figure:: tutorial_images/chemdraw/chemdraw_8.png

   A confirmation pop-up will then appear indicating that the
   CSV file has been successfully saved and automatically
   loaded into the GUI.

   .. raw:: html

      <div class="step-nav">
         <button onclick="prevStep('chemdraw',8)">Previous</button>
         <button onclick="nextStep('chemdraw',8)">Next</button>
      </div>


   .. raw:: html

      </div>

   .. raw:: html

      <div class='step-content' data-prefix='chemdraw' data-step='9'>

   .. figure:: tutorial_images/chemdraw/chemdraw_9.png

   Return to the **ROBERT** tab.



   The newly generated CSV dataset will appear automatically
   as the training input.

   .. raw:: html

      <div class="step-nav">
         <button onclick="prevStep('chemdraw',9)">Previous</button>
         <button onclick="nextStep('chemdraw',9)">Next</button>
      </div>


   .. raw:: html

      </div>

   .. raw:: html

      <div class='step-content' data-prefix='chemdraw' data-step='10'>

   .. figure:: tutorial_images/chemdraw/chemdraw_10.png

   Next, configure the model parameters.



   Select the **target column** (Yield), choose the
   prediction type (**regression**), and specify the
   column containing the molecule identifiers.



   No columns need to be ignored in this example.

   .. raw:: html

      <div class="step-nav">
         <button onclick="prevStep('chemdraw',10)">Previous</button>
         <button onclick="nextStep('chemdraw',10)">Next</button>
      </div>


   .. raw:: html

      </div>

   .. raw:: html

      <div class='step-content' data-prefix='chemdraw' data-step='11'>

   .. figure:: tutorial_images/chemdraw/chemdraw_11.png

   Press **Run ROBERT** to launch the workflow.



   easyROB will automatically generate molecular descriptors
   from the structures and train the machine learning model.

   .. raw:: html

      <div class="step-nav">
         <button onclick="prevStep('chemdraw',11)">Previous</button>
         <button onclick="nextStep('chemdraw',11)">Next</button>
      </div>


   .. raw:: html

      </div>

   .. raw:: html

      <div class='step-content' data-prefix='chemdraw' data-step='12'>

   .. figure:: tutorial_images/chemdraw/chemdraw_12.png

   A message will appear indicating that only structural
   descriptors will be generated in this workflow.



   Atom-based descriptors can also be generated using the
   **AQME** tab, as shown in the **From CSV** tutorial.

   .. raw:: html

      <div class="step-nav">
         <button onclick="prevStep('chemdraw',12)">Previous</button>
         <button onclick="nextStep('chemdraw',12)">Next</button>
      </div>


   .. raw:: html

      </div>

   .. raw:: html

      <div class='step-content' data-prefix='chemdraw' data-step='13'>

   .. figure:: tutorial_images/chemdraw/chemdraw_13.png

   Once the workflow finishes, a pop-up message confirms
   that the process has completed successfully.



   The **Reports** and **Images** tabs become available,
   providing access to the results generated during the
   workflow.



   These result sections are explained in more detail in the
   **Overview** tutorial.

   .. raw:: html

      <div class="step-nav">
         <button onclick="prevStep('chemdraw',13)">Previous</button>
      </div>


   .. raw:: html

      </div>

