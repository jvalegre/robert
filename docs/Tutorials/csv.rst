.. container:: step

   .. raw:: html

      <div class='step-content' data-prefix='csv' data-step='1'>

   .. figure:: tutorial_images/csv/csv_1.png

   In this tutorial we will build a machine learning model starting
   from a simple CSV dataset.



   We will first show how to create the input CSV file containing
   the molecular structures and the target property to be predicted.



   In this example we will use **SMILES** representations so that
   molecular descriptors can be generated automatically with AQME,
   but if descriptors are already available they can be included
   directly in the CSV instead of SMILES.



   This dataset will then be used to train the machine learning
   model within the ROBERT workflow.

   .. raw:: html

      <div class="step-nav">
         <button onclick="nextStep('csv',1)">Next</button>
      </div>


   .. raw:: html

      </div>

   .. raw:: html

      <div class='step-content' data-prefix='csv' data-step='2'>

   .. figure:: tutorial_images/csv/csv_1_1.png

   Open a new spreadsheet in Microsoft Excel. 
   Any spreadsheet software can be used to create or edit CSV files.

   .. raw:: html

      <div class="step-nav">
         <button onclick="prevStep('csv',2)">Previous</button>
         <button onclick="nextStep('csv',2)">Next</button>
      </div>


   .. raw:: html

      </div>

   .. raw:: html

      <div class='step-content' data-prefix='csv' data-step='3'>

   .. figure:: tutorial_images/csv/csv_1_2.png

   In this example we will create a dataset containing three columns:
   **code_name** (unique identifiers for each molecule), 
   **target** (the property that the machine learning model will 
   learn to predict), and **SMILES** (the molecular structures).

   .. raw:: html

      <div class="step-nav">
         <button onclick="prevStep('csv',3)">Previous</button>
         <button onclick="nextStep('csv',3)">Next</button>
      </div>


   .. raw:: html

      </div>

   .. raw:: html

      <div class='step-content' data-prefix='csv' data-step='4'>

   .. figure:: tutorial_images/csv/csv_1_3.png

   The **SMILES strings** can be obtained in several ways.



   For example, they can be copied from molecular drawing software
   such as **ChemDraw**, or retrieved from online chemical
   databases that provide both the structure and the corresponding
   SMILES string.



   In this tutorial we will copy the SMILES directly from ChemDraw,
   as shown in the image.



   Select the structure and go to **Edit → Copy As → SMILES**.
   You can also use the shortcut **Ctrl + Alt + C**.

   .. raw:: html

      <div class="step-nav">
         <button onclick="prevStep('csv',4)">Previous</button>
         <button onclick="nextStep('csv',4)">Next</button>
      </div>


   .. raw:: html

      </div>

   .. raw:: html

      <div class='step-content' data-prefix='csv' data-step='5'>

   .. figure:: tutorial_images/csv/csv_1_4.png

   Once the dataset is complete and all molecular structures have
   been added as SMILES, the file must be saved as a CSV file.

   .. raw:: html

      <div class="step-nav">
         <button onclick="prevStep('csv',5)">Previous</button>
         <button onclick="nextStep('csv',5)">Next</button>
      </div>


   .. raw:: html

      </div>

   .. raw:: html

      <div class='step-content' data-prefix='csv' data-step='6'>

   .. figure:: tutorial_images/csv/csv_1_5.png

   Go to the **File** tab and select **Save As**. 
   Choose **CSV** under the **Save as type** dropdown menu,
   provide a name for the dataset, and click **Save**.

   .. raw:: html

      <div class="step-nav">
         <button onclick="prevStep('csv',6)">Previous</button>
         <button onclick="nextStep('csv',6)">Next</button>
      </div>


   .. raw:: html

      </div>

   .. raw:: html

      <div class='step-content' data-prefix='csv' data-step='7'>

   .. figure:: tutorial_images/csv/csv_2.png

   The first step in the workflow is loading the input dataset.



   Select the **training CSV file** that will be used to build
   the machine learning model.



   Optionally, an **external test dataset** can also be provided.
   If included, ROBERT will generate predictions for these
   molecules after the model has been trained.

   .. raw:: html

      <div class="step-nav">
         <button onclick="prevStep('csv',7)">Previous</button>
         <button onclick="nextStep('csv',7)">Next</button>
      </div>


   .. raw:: html

      </div>

   .. raw:: html

      <div class='step-content' data-prefix='csv' data-step='8'>

   .. figure:: tutorial_images/csv/csv_3.png

   Next, configure the basic model parameters.





   Select the **target column**, choose the prediction type

   (in this example **regression**), and specify the column

   containing the molecule identifiers.





   No columns need to be excluded in this case, since the

   SMILES column is handled automatically.

   .. raw:: html

      <div class="step-nav">
         <button onclick="prevStep('csv',8)">Previous</button>
         <button onclick="nextStep('csv',8)">Next</button>
      </div>


   .. raw:: html

      </div>

   .. raw:: html

      <div class='step-content' data-prefix='csv' data-step='9'>

   .. figure:: tutorial_images/csv/csv_4.png

   Enable the **AQME Workflow** to generate molecular

   descriptors from the SMILES structures.





   Once this option is activated, the **AQME** tab becomes

   available and can be used to adjust descriptor generation

   settings.





   **Note:** If the input CSV already contains molecular

   descriptors, this step can be skipped and the workflow can

   be run directly.

   .. raw:: html

      <div class="step-nav">
         <button onclick="prevStep('csv',9)">Previous</button>
         <button onclick="nextStep('csv',9)">Next</button>
      </div>


   .. raw:: html

      </div>

   .. raw:: html

      <div class='step-content' data-prefix='csv' data-step='10'>

   .. figure:: tutorial_images/csv/csv_5.png

   In the **AQME** tab, the interface may detect a common

   molecular scaffold across the dataset.





   In this example, the molecules share a carboxylic acid

   functional group, which is displayed in the structure viewer.





   Atoms from this structure can be selected to generate

   additional atom-based descriptors.

   .. raw:: html

      <div class="step-nav">
         <button onclick="prevStep('csv',10)">Previous</button>
         <button onclick="nextStep('csv',10)">Next</button>
      </div>


   .. raw:: html

      </div>

   .. raw:: html

      <div class='step-content' data-prefix='csv' data-step='11'>

   .. figure:: tutorial_images/csv/csv_6.png

   Here, two atoms from the carboxylic acid group have been

   selected: the carbonyl oxygen and the hydroxyl oxygen.





   These selections will be used during descriptor generation.

   .. raw:: html

      <div class="step-nav">
         <button onclick="prevStep('csv',11)">Previous</button>
         <button onclick="nextStep('csv',11)">Next</button>
      </div>


   .. raw:: html

      </div>

   .. raw:: html

      <div class='step-content' data-prefix='csv' data-step='12'>

   .. figure:: tutorial_images/csv/csv_7.png

   Once the configuration is complete, return to the

   **ROBERT** tab and press **Run ROBERT** to launch

   the workflow.





   The program will automatically generate descriptors,

   train the machine learning models, and produce the results.

   .. raw:: html

      <div class="step-nav">
         <button onclick="prevStep('csv',12)">Previous</button>
         <button onclick="nextStep('csv',12)">Next</button>
      </div>


   .. raw:: html

      </div>

   .. raw:: html

      <div class='step-content' data-prefix='csv' data-step='13'>

   .. figure:: tutorial_images/csv/csv_8.png

   After the workflow finishes, a message will indicate that

   the process has completed successfully.





   You can then choose to open the generated report to

   inspect the results.

   .. raw:: html

      <div class="step-nav">
         <button onclick="prevStep('csv',13)">Previous</button>
         <button onclick="nextStep('csv',13)">Next</button>
      </div>


   .. raw:: html

      </div>

   .. raw:: html

      <div class='step-content' data-prefix='csv' data-step='14'>

   .. figure:: tutorial_images/csv/csv_9.png

   The report summarizes the machine learning results, including

   model performance metrics and validation information.





   Once the workflow has generated results, the **Reports**,

   **Images**, and **Predictions** tabs become available.





   The **Reports** tab displays the PDF summary of the modelling

   results, the **Images** tab provides access to the figures

   generated during the workflow, and the **Predictions** tab

   shows the predicted values for the external test dataset used

   in this example.





   These result sections are described in more detail in the

   **Overview** tutorial.

   .. raw:: html

      <div class="step-nav">
         <button onclick="prevStep('csv',14)">Previous</button>
      </div>


   .. raw:: html

      </div>

