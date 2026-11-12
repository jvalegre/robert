.. container:: step

   .. raw:: html

      <div class='step-content' data-prefix='predictions' data-step='1'>

   .. figure:: tutorial_images/predictions/predictions_1.png

   This workflow is used when you already have a trained ROBERT model
   and want to generate predictions for new candidate molecules.



   For example, you may have trained a model previously and now want
   to evaluate new structures using that model.

   .. raw:: html

      <div class="step-nav">
         <button onclick="nextStep('predictions',1)">Next</button>
      </div>


   .. raw:: html

      </div>

   .. raw:: html

      <div class='step-content' data-prefix='predictions' data-step='2'>

   .. figure:: tutorial_images/predictions/predictions_1_1.png

   To run predictions, provide a **CSV file** containing the molecules
   for which predictions should be calculated.



   The CSV file must be placed in the same directory where the ROBERT
   model was generated, as shown in the example. This allows the workflow
   to automatically detect the model and use it to generate predictions.

   .. raw:: html

      <div class="step-nav">
         <button onclick="prevStep('predictions',2)">Previous</button>
         <button onclick="nextStep('predictions',2)">Next</button>
      </div>


   .. raw:: html

      </div>

   .. raw:: html

      <div class='step-content' data-prefix='predictions' data-step='3'>

   .. figure:: tutorial_images/predictions/predictions_2.png

   Select the **Test CSV** file in the interface to load the
   molecules for prediction.



   This dataset must contain the information required by the
   trained model.



   If descriptors were originally generated using AQME, the
   CSV should contain an **ID column** and a **SMILES
   column** so descriptors can be generated automatically.

   .. raw:: html

      <div class="step-nav">
         <button onclick="prevStep('predictions',3)">Previous</button>
         <button onclick="nextStep('predictions',3)">Next</button>
      </div>


   .. raw:: html

      </div>

   .. raw:: html

      <div class='step-content' data-prefix='predictions' data-step='4'>

   .. figure:: tutorial_images/predictions/predictions_3.png

   Since the model has already been trained, only the
   **PREDICT** step needs to be executed.



   Select the **PREDICT** option in the workflow
   configuration.

   .. raw:: html

      <div class="step-nav">
         <button onclick="prevStep('predictions',4)">Previous</button>
         <button onclick="nextStep('predictions',4)">Next</button>
      </div>


   .. raw:: html

      </div>

   .. raw:: html

      <div class='step-content' data-prefix='predictions' data-step='5'>

   .. figure:: tutorial_images/predictions/predictions_4.png

   Once the test dataset is loaded and the **PREDICT**
   option is selected, press **Run ROBERT** to start
   the prediction workflow.

   .. raw:: html

      <div class="step-nav">
         <button onclick="prevStep('predictions',5)">Previous</button>
         <button onclick="nextStep('predictions',5)">Next</button>
      </div>


   .. raw:: html

      </div>

   .. raw:: html

      <div class='step-content' data-prefix='predictions' data-step='6'>

   .. figure:: tutorial_images/predictions/predictions_5.png

   A pop-up window will ask how the descriptors should be handled
   for the prediction workflow.



   If the original model was built using **AQME**, the GUI can
   generate the descriptors again automatically from the SMILES
   structures.



   If the model was trained using **custom descriptors**, then
   the test CSV must already contain the same descriptors that
   were used when training the model.



   In this tutorial we use the **AQME** option, but you should
   select the option that matches how your model was originally
   trained.

   .. raw:: html

      <div class="step-nav">
         <button onclick="prevStep('predictions',6)">Previous</button>
         <button onclick="nextStep('predictions',6)">Next</button>
      </div>


   .. raw:: html

      </div>

   .. raw:: html

      <div class='step-content' data-prefix='predictions' data-step='7'>

   .. figure:: tutorial_images/predictions/predictions_6.png

   Once the process finishes successfully, a confirmation
   message will appear.



   The **Predictions** tab will become available, where
   the generated prediction results can be inspected.



   More details about this results tab are explained in the
   **Overview** tutorial.

   .. raw:: html

      <div class="step-nav">
         <button onclick="prevStep('predictions',7)">Previous</button>
      </div>


   .. raw:: html

      </div>

