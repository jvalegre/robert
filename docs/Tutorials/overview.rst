.. container:: step

   .. raw:: html

      <div class='step-content' data-prefix='overview' data-step='1'>

   .. figure:: tutorial_images/overview/overview_1.png

   Welcome to the **easyROB graphical interface**.





   easyROB integrates two complementary engines designed for

   computational chemistry workflows:





   • **AQME** — generates molecular descriptors from molecular structures


   • **ROBERT** — builds and evaluates machine learning models





   Together, these tools allow users to move from raw molecular

   structures to predictive models within a single interface.





   This overview briefly introduces the main sections of the GUI

   before moving to the practical tutorials.

   .. raw:: html

      <div class="step-nav">
         <button onclick="nextStep('overview',1)">Next</button>
      </div>


   .. raw:: html

      </div>

   .. raw:: html

      <div class='step-content' data-prefix='overview' data-step='2'>

   .. figure:: tutorial_images/overview/overview_1_1.png

   When easyROB starts, the interface opens in the **ROBERT tab**.





   This tab contains the main controls used to:





   • Load datasets


   • Configure machine learning workflows


   • Train predictive models





   Most users will perform the majority of their work in this section.

   .. raw:: html

      <div class="step-nav">
         <button onclick="prevStep('overview',2)">Previous</button>
         <button onclick="nextStep('overview',2)">Next</button>
      </div>


   .. raw:: html

      </div>

   .. raw:: html

      <div class='step-content' data-prefix='overview' data-step='3'>

   .. figure:: tutorial_images/overview/overview_1_2.png

   The first step is loading a **dataset**.





   To train a machine learning model, the dataset must contain

   **molecular descriptors**, which are numerical features describing

   each molecule.





   Two scenarios are possible:





   • If your CSV file already contains descriptors, they will be used

   directly to train the model.





   • If the dataset only contains a **SMILES column**, descriptors

   can be generated automatically using the **AQME workflow**.





   This descriptor generation process will be explained later

   in the tutorial.





   Optionally, you may also provide a separate **test CSV file**

   to evaluate predictions on an external dataset.

   .. raw:: html

      <div class="step-nav">
         <button onclick="prevStep('overview',3)">Previous</button>
         <button onclick="nextStep('overview',3)">Next</button>
      </div>


   .. raw:: html

      </div>

   .. raw:: html

      <div class='step-content' data-prefix='overview' data-step='4'>

   .. figure:: tutorial_images/overview/overview_1_3.png

   This section defines the **model configuration**.





   First, select the **Target value**.





   The target is the property that the machine learning model will

   learn from the dataset and later attempt to predict for new molecules.





   Next, choose the **prediction type**, such as:





   • **Regression** for continuous values


   • **Classification** for categorical outcomes





   In addition, you must provide a column containing

   **unique identifiers** for each row in the dataset.





   These identifiers allow easyROB to correctly track molecules

   and report predictions in a clear and reproducible way.

   .. raw:: html

      <div class="step-nav">
         <button onclick="prevStep('overview',4)">Previous</button>
         <button onclick="nextStep('overview',4)">Next</button>
      </div>


   .. raw:: html

      </div>

   .. raw:: html

      <div class='step-content' data-prefix='overview' data-step='5'>

   .. figure:: tutorial_images/overview/overview_1_4.png

   Some dataset columns may contain information that should not

   be used to build the model.





   These columns can be **excluded** here.





   Typical examples include metadata columns, comments, or

   experimental annotations that are not meaningful descriptors.





   **Important:** The columns selected as **Target value**,

   **Name column**, and the **SMILES column** are automatically

   handled by easyROB and do not need to be excluded manually.

   .. raw:: html

      <div class="step-nav">
         <button onclick="prevStep('overview',5)">Previous</button>
         <button onclick="nextStep('overview',5)">Next</button>
      </div>


   .. raw:: html

      </div>

   .. raw:: html

      <div class='step-content' data-prefix='overview' data-step='6'>

   .. figure:: tutorial_images/overview/overview_1_5.png

   Enabling the **AQME workflow** allows easyROB to generate

   molecular descriptors directly from SMILES structures.





   When this option is activated, the **AQME tab** becomes

   available in the interface.





   This tab provides advanced options for descriptor generation

   and molecular processing.





   These settings are optional and will be explained in detail

   later in this tutorial.

   .. raw:: html

      <div class="step-nav">
         <button onclick="prevStep('overview',6)">Previous</button>
         <button onclick="nextStep('overview',6)">Next</button>
      </div>


   .. raw:: html

      </div>

   .. raw:: html

      <div class='step-content' data-prefix='overview' data-step='7'>

   .. figure:: tutorial_images/overview/overview_1_6.png

   The ROBERT machine learning pipeline can be executed

   automatically using the **Full Workflow** option.





   This option is **enabled by default**, so in most cases

   you do not need to modify anything before running the workflow.





   Alternatively, advanced users can run individual stages

   of the pipeline separately:





   **CURATE → GENERATE → PREDICT → VERIFY → REPORT**





   Running steps individually can be useful for debugging

   or for inspecting intermediate results.

   .. raw:: html

      <div class="step-nav">
         <button onclick="prevStep('overview',7)">Previous</button>
         <button onclick="nextStep('overview',7)">Next</button>
      </div>


   .. raw:: html

      </div>

   .. raw:: html

      <div class='step-content' data-prefix='overview' data-step='8'>

   .. figure:: tutorial_images/overview/overview_1_7.png

   Press the **ROBERT** button to launch the automated

   machine learning workflow.





   This will execute the selected stages of the ROBERT pipeline

   using the dataset and configuration provided.





   Alternatively, the **AQME** button can be used independently

   to generate molecular descriptors before training a model.





   The **Stop** button allows interrupting the workflow

   at any moment if necessary.

   .. raw:: html

      <div class="step-nav">
         <button onclick="prevStep('overview',8)">Previous</button>
         <button onclick="nextStep('overview',8)">Next</button>
      </div>


   .. raw:: html

      </div>

   .. raw:: html

      <div class='step-content' data-prefix='overview' data-step='9'>

   .. figure:: tutorial_images/overview/overview_1_8.png

   The **console panel** displays log messages and progress

   information generated during execution.





   This output helps track which steps are running and

   identify potential warnings or errors.

   .. raw:: html

      <div class="step-nav">
         <button onclick="prevStep('overview',9)">Previous</button>
         <button onclick="nextStep('overview',9)">Next</button>
      </div>


   .. raw:: html

      </div>

   .. raw:: html

      <div class='step-content' data-prefix='overview' data-step='10'>

   .. figure:: tutorial_images/overview/overview_2.png

   When the **Enable AQME Workflow** option is activated, the

   **AQME tab** becomes available in the interface.





   If a dataset containing a **SMILES column** has already been loaded,

   easyROB will automatically generate molecular descriptors when the

   workflow is executed.





   The AQME tab provides **advanced options** for users who want more

   control over the descriptor generation process.





   For example, if your molecules share a common structure, the interface

   can display the molecular scaffold and allow you to **select specific

   atoms** to generate atom-based descriptors.





   You can also configure additional **AQME descriptor parameters**.

   A documentation button is available to access detailed information

   about these options.





   This tab is also used when starting from **ChemDraw outputs**.

   In this case, the ChemDraw file can be loaded here to extract the

   relevant information and generate a CSV dataset that can then be

   used in the AQME and ROBERT workflows.

   .. raw:: html

      <div class="step-nav">
         <button onclick="prevStep('overview',10)">Previous</button>
         <button onclick="nextStep('overview',10)">Next</button>
      </div>


   .. raw:: html

      </div>

   .. raw:: html

      <div class='step-content' data-prefix='overview' data-step='11'>

   .. figure:: tutorial_images/overview/overview_3.png

   The **Advanced Options** tab provides additional controls for the

   ROBERT machine learning workflow.





   This section allows advanced users to modify specific parameters

   used during different stages of the pipeline, including:





   • **GENERAL** – parameters affecting the overall workflow


   • **CURATE** – dataset preparation and preprocessing


   • **GENERATE** – model training and descriptor selection


   • **PREDICT** – prediction settings





   These options are intended for users who want more control over

   the modelling process. In most cases, the default settings are

   sufficient to run the workflow successfully.





   Help buttons are available throughout this tab to access the

   documentation and obtain detailed explanations of each parameter.

   .. raw:: html

      <div class="step-nav">
         <button onclick="prevStep('overview',11)">Previous</button>
         <button onclick="nextStep('overview',11)">Next</button>
      </div>


   .. raw:: html

      </div>

   .. raw:: html

      <div class='step-content' data-prefix='overview' data-step='12'>

   .. figure:: tutorial_images/overview/overview_4.png

   The **MolSSI Databases** tab provides access to molecular libraries

   developed through the **Descriptor Libraries** project, a collaborative

   initiative between the Molecular Sciences Software Institute (MolSSI)

   and the Center for Computer Assisted Synthesis (C-CAS).





   These libraries contain curated collections of molecules organised

   by functional group, such as carboxylic acids, amines, sulfonamides,

   and many others.





   Users can explore these datasets directly within the interface,

   inspect the available molecules, and download the corresponding

   libraries for further analysis.





   The downloaded datasets already include molecular descriptors

   as well as machine learning predictions generated within the

   Descriptor Libraries project, allowing them to be used directly

   in modelling workflows.

   .. raw:: html

      <div class="step-nav">
         <button onclick="prevStep('overview',12)">Previous</button>
         <button onclick="nextStep('overview',12)">Next</button>
      </div>


   .. raw:: html

      </div>

   .. raw:: html

      <div class='step-content' data-prefix='overview' data-step='13'>

   .. figure:: tutorial_images/overview/overview_5.png

   The **Reports**, **Images**, and **Predictions** tabs remain

   disabled until the ROBERT workflow has been executed and results

   have been generated.





   Once the workflow finishes, these tabs become available and allow

   users to explore the different outputs produced during the machine

   learning process.





   Each tab focuses on a different type of result: reports summarizing

   model performance, images generated during the workflow, and

   predictions obtained for external datasets.





   **Note:** The **Predictions** tab is only available if an

   external test dataset was provided before running the workflow.

   .. raw:: html

      <div class="step-nav">
         <button onclick="prevStep('overview',13)">Previous</button>
         <button onclick="nextStep('overview',13)">Next</button>
      </div>


   .. raw:: html

      </div>

   .. raw:: html

      <div class='step-content' data-prefix='overview' data-step='14'>

   .. figure:: tutorial_images/overview/overview_5_1.png

   The **Reports** tab displays the PDF report automatically

   generated by the ROBERT workflow.





   This report provides a complete summary of the machine learning

   process, including the selected model, performance metrics,

   identified outliers, extrapolation analysis, and statistical

   validation tests.





   The report helps users evaluate whether the generated model

   is robust and reliable for making predictions.

   .. raw:: html

      <div class="step-nav">
         <button onclick="prevStep('overview',14)">Previous</button>
         <button onclick="nextStep('overview',14)">Next</button>
      </div>


   .. raw:: html

      </div>

   .. raw:: html

      <div class='step-content' data-prefix='overview' data-step='15'>

   .. figure:: tutorial_images/overview/overview_5_2.png

   The **Images** tab provides access to all figures generated

   during the ROBERT workflow.





   These images are organised into sections corresponding to the

   different stages of the pipeline, including **CURATE**,

   **GENERATE**, **VERIFY**, and **PREDICT**.





   Users can browse these figures directly in the interface.

   By right-clicking on an image, it can be copied, saved to a file,

   opened in full size, or the folder containing the image can

   be opened.

   .. raw:: html

      <div class="step-nav">
         <button onclick="prevStep('overview',15)">Previous</button>
         <button onclick="nextStep('overview',15)">Next</button>
      </div>


   .. raw:: html

      </div>

   .. raw:: html

      <div class='step-content' data-prefix='overview' data-step='16'>

   .. figure:: tutorial_images/overview/overview_5_3.png

   The **Predictions** tab displays the prediction results

   generated for an external test dataset, if one was provided.





   The results are organised into two sub-tabs: **PFI** and

   **no PFI**. ROBERT automatically builds two models: one applying

   a descriptor reduction step (**PFI**) and another without this

   reduction. Each tab shows the predictions generated by one of

   these models.





   For each molecule in the test dataset, the table displays the

   predicted value together with the associated standard deviation.





   The table also provides several interactive options. By right-clicking

   on the prediction columns, users can sort the values or generate

   simple plots such as histograms to quickly explore the distribution

   of the predicted results.

   .. raw:: html

      <div class="step-nav">
         <button onclick="prevStep('overview',16)">Previous</button>
      </div>


   .. raw:: html

      </div>

