.. container:: step

   .. raw:: html

      <div class='step-content' data-prefix='descriptors' data-step='1'>

   .. figure:: tutorial_images/descriptors/descriptors_1.png

   This tutorial shows how to generate molecular descriptors
   using the **AQME** workflow.



   Descriptor generation can be useful when preparing datasets
   for machine learning models or when molecular descriptors
   are required for other types of analysis.

   .. raw:: html

      <div class="step-nav">
         <button onclick="nextStep('descriptors',1)">Next</button>
      </div>


   .. raw:: html

      </div>

   .. raw:: html

      <div class='step-content' data-prefix='descriptors' data-step='2'>

   .. figure:: tutorial_images/descriptors/descriptors_2.png

   Load a **CSV file** containing the molecules for which
   descriptors should be generated.



   The dataset must contain a **SMILES column**, which
   AQME will use to compute the molecular descriptors.

   .. raw:: html

      <div class="step-nav">
         <button onclick="prevStep('descriptors',2)">Previous</button>
         <button onclick="nextStep('descriptors',2)">Next</button>
      </div>


   .. raw:: html

      </div>

   .. raw:: html

      <div class='step-content' data-prefix='descriptors' data-step='3'>

   .. figure:: tutorial_images/descriptors/descriptors_3.png

   Press **Run AQME** to start the descriptor generation
   workflow.



   AQME will automatically generate three descriptor datasets:
   **denovo**, **interpret**, and **full**.



   These files contain increasing numbers of descriptors,
   ranging from smaller and more interpretable sets to the
   complete descriptor collection.

   .. raw:: html

      <div class="step-nav">
         <button onclick="prevStep('descriptors',3)">Previous</button>
         <button onclick="nextStep('descriptors',3)">Next</button>
      </div>


   .. raw:: html

      </div>

   .. raw:: html

      <div class='step-content' data-prefix='descriptors' data-step='4'>

   .. figure:: tutorial_images/descriptors/descriptors_4.png

   A pop-up window will indicate that only **molecular
   descriptors** will be generated in this workflow.



   Atomic descriptors can also be generated using the
   **AQME** tab. An example of this workflow is shown
   in the **From CSV** tutorial.

   .. raw:: html

      <div class="step-nav">
         <button onclick="prevStep('descriptors',4)">Previous</button>
         <button onclick="nextStep('descriptors',4)">Next</button>
      </div>


   .. raw:: html

      </div>

   .. raw:: html

      <div class='step-content' data-prefix='descriptors' data-step='5'>

   .. figure:: tutorial_images/descriptors/descriptors_5.png

   Once the process finishes, a message will indicate that
   the descriptor datasets have been generated successfully.



   The three CSV files (**denovo**, **interpret**, and
   **full**) will be saved in the working directory.



   These descriptor datasets can be used directly for further
   analysis or loaded into the **ROBERT** workflow to
   build machine learning models.

   .. raw:: html

      <div class="step-nav">
         <button onclick="prevStep('descriptors',5)">Previous</button>
         <button onclick="nextStep('descriptors',5)">Next</button>
      </div>


   .. raw:: html

      </div>

   .. raw:: html

      <div class='step-content' data-prefix='descriptors' data-step='6'>

   .. figure:: tutorial_images/descriptors/descriptors_6.png

   In this example, the **interpret** dataset is loaded,
   but any of the generated descriptor sets can be used
   depending on your needs.

   At this stage, the descriptors can be inspected, filtered,
   or modified before building a machine learning model.



   If you want to train a predictive model using these descriptors,
   the dataset must contain a **target column** with the property
   you want to predict.

   If the original CSV did not include a target value, it can be
   added later before running the **ROBERT** workflow.

   Alternatively, the generated descriptor datasets can also be
   used independently of ROBERT for other analyses or workflows.

   .. raw:: html

      <div class="step-nav">
         <button onclick="prevStep('descriptors',6)">Previous</button>
      </div>


   .. raw:: html

      </div>

