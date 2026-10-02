Welcome to the <b>easyROB graphical interface</b>.<br><br>



easyROB integrates two complementary engines designed for

computational chemistry workflows:<br><br>



• <b>AQME</b> — generates molecular descriptors from molecular structures<br>

• <b>ROBERT</b> — builds and evaluates machine learning models<br><br>



Together, these tools allow users to move from raw molecular

structures to predictive models within a single interface.<br><br>



This overview briefly introduces the main sections of the GUI

before moving to the practical tutorials.



---



When easyROB starts, the interface opens in the <b>ROBERT tab</b>.<br><br>



This tab contains the main controls used to:<br><br>



• Load datasets<br>

• Configure machine learning workflows<br>

• Train predictive models<br><br>



Most users will perform the majority of their work in this section.



---



The first step is loading a <b>dataset</b>.<br><br>



To train a machine learning model, the dataset must contain

<b>molecular descriptors</b>, which are numerical features describing

each molecule.<br><br>



Two scenarios are possible:<br><br>



• If your CSV file already contains descriptors, they will be used

directly to train the model.<br><br>



• If the dataset only contains a <b>SMILES column</b>, descriptors

can be generated automatically using the <b>AQME workflow</b>.<br><br>



This descriptor generation process will be explained later

in the tutorial.<br><br>



Optionally, you may also provide a separate <b>test CSV file</b>

to evaluate predictions on an external dataset.



---



This section defines the <b>model configuration</b>.<br><br>



First, select the <b>Target value</b>.<br><br>



The target is the property that the machine learning model will

learn from the dataset and later attempt to predict for new molecules.<br><br>



Next, choose the <b>prediction type</b>, such as:<br><br>



• <b>Regression</b> for continuous values<br>

• <b>Classification</b> for categorical outcomes<br><br>



In addition, you must provide a column containing

<b>unique identifiers</b> for each row in the dataset.<br><br>



These identifiers allow easyROB to correctly track molecules

and report predictions in a clear and reproducible way.



---



Some dataset columns may contain information that should not

be used to build the model.<br><br>



These columns can be <b>excluded</b> here.<br><br>



Typical examples include metadata columns, comments, or

experimental annotations that are not meaningful descriptors.<br><br>



<b>Important:</b> The columns selected as <b>Target value</b>,

<b>Name column</b>, and the <b>SMILES column</b> are automatically

handled by easyROB and do not need to be excluded manually.



---



Enabling the <b>AQME workflow</b> allows easyROB to generate

molecular descriptors directly from SMILES structures.<br><br>



When this option is activated, the <b>AQME tab</b> becomes

available in the interface.<br><br>



This tab provides advanced options for descriptor generation

and molecular processing.<br><br>



These settings are optional and will be explained in detail

later in this tutorial.



---



The ROBERT machine learning pipeline can be executed

automatically using the <b>Full Workflow</b> option.<br><br>



This option is <b>enabled by default</b>, so in most cases

you do not need to modify anything before running the workflow.<br><br>



Alternatively, advanced users can run individual stages

of the pipeline separately:<br><br>



<b>CURATE → GENERATE → PREDICT → VERIFY → REPORT</b><br><br>



Running steps individually can be useful for debugging

or for inspecting intermediate results.



---



Press the <b>ROBERT</b> button to launch the automated

machine learning workflow.<br><br>



This will execute the selected stages of the ROBERT pipeline

using the dataset and configuration provided.<br><br>



Alternatively, the <b>AQME</b> button can be used independently

to generate molecular descriptors before training a model.<br><br>



The <b>Stop</b> button allows interrupting the workflow

at any moment if necessary.



---



The <b>console panel</b> displays log messages and progress

information generated during execution.<br><br>



This output helps track which steps are running and

identify potential warnings or errors.<br><br>



---



When the <b>Enable AQME Workflow</b> option is activated, the

<b>AQME tab</b> becomes available in the interface.<br><br>



If a dataset containing a <b>SMILES column</b> has already been loaded,

easyROB will automatically generate molecular descriptors when the

workflow is executed.<br><br>



The AQME tab provides <b>advanced options</b> for users who want more

control over the descriptor generation process.<br><br>



For example, if your molecules share a common structure, the interface

can display the molecular scaffold and allow you to <b>select specific

atoms</b> to generate atom-based descriptors.<br><br>



You can also configure additional <b>AQME descriptor parameters</b>.

A documentation button is available to access detailed information

about these options.<br><br>



This tab is also used when starting from <b>ChemDraw outputs</b>.

In this case, the ChemDraw file can be loaded here to extract the

relevant information and generate a CSV dataset that can then be

used in the AQME and ROBERT workflows.



---



The <b>Advanced Options</b> tab provides additional controls for the

ROBERT machine learning workflow.<br><br>



This section allows advanced users to modify specific parameters

used during different stages of the pipeline, including:<br><br>



• <b>GENERAL</b> – parameters affecting the overall workflow<br>

• <b>CURATE</b> – dataset preparation and preprocessing<br>

• <b>GENERATE</b> – model training and descriptor selection<br>

• <b>PREDICT</b> – prediction settings<br><br>



These options are intended for users who want more control over

the modelling process. In most cases, the default settings are

sufficient to run the workflow successfully.<br><br>



Help buttons are available throughout this tab to access the

documentation and obtain detailed explanations of each parameter.



---



The <b>MolSSI Databases</b> tab provides access to molecular libraries

developed through the <b>Descriptor Libraries</b> project, a collaborative

initiative between the Molecular Sciences Software Institute (MolSSI)

and the Center for Computer Assisted Synthesis (C-CAS).<br><br>



These libraries contain curated collections of molecules organised

by functional group, such as carboxylic acids, amines, sulfonamides,

and many others.<br><br>



Users can explore these datasets directly within the interface,

inspect the available molecules, and download the corresponding

libraries for further analysis.<br><br>



The downloaded datasets already include molecular descriptors

as well as machine learning predictions generated within the

Descriptor Libraries project, allowing them to be used directly

in modelling workflows.



---



The <b>Reports</b>, <b>Images</b>, and <b>Predictions</b> tabs remain

disabled until the ROBERT workflow has been executed and results

have been generated.<br><br>



Once the workflow finishes, these tabs become available and allow

users to explore the different outputs produced during the machine

learning process.<br><br>



Each tab focuses on a different type of result: reports summarizing

model performance, images generated during the workflow, and

predictions obtained for external datasets.<br><br>



<b>Note:</b> The <b>Predictions</b> tab is only available if an

external test dataset was provided before running the workflow.



---



The <b>Reports</b> tab displays the PDF report automatically

generated by the ROBERT workflow.<br><br>



This report provides a complete summary of the machine learning

process, including the selected model, performance metrics,

identified outliers, boundary robustness analysis, and statistical

validation tests.<br><br>



The report helps users evaluate whether the generated model

is robust and reliable for making predictions.



---



The <b>Images</b> tab provides access to all figures generated

during the ROBERT workflow.<br><br>



These images are organised into sections corresponding to the

different stages of the pipeline, including <b>CURATE</b>,

<b>GENERATE</b>, <b>VERIFY</b>, and <b>PREDICT</b>.<br><br>



Users can browse these figures directly in the interface.

By right-clicking on an image, it can be copied, saved to a file,

opened in full size, or the folder containing the image can

be opened.



---



The <b>Predictions</b> tab displays the prediction results

generated for an external test dataset, if one was provided.<br><br>



The results are organised into two sub-tabs: <b>PFI</b> and

<b>no PFI</b>. ROBERT automatically builds two models: one applying

a descriptor reduction step (<b>PFI</b>) and another without this

reduction. Each tab shows the predictions generated by one of

these models.<br><br>



For each molecule in the test dataset, the table displays the

predicted value together with the associated standard deviation.<br><br>



The table also provides several interactive options. By right-clicking

on the prediction columns, users can sort the values or generate

simple plots such as histograms to quickly explore the distribution

of the predicted results.



---


