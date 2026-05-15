This workflow is used when you already have a trained ROBERT model
and want to generate predictions for new candidate molecules.<br><br>

For example, you may have trained a model previously and now want
to evaluate new structures using that model.<br><br>

---

To run predictions, provide a <b>CSV file</b> containing the molecules
for which predictions should be calculated.<br><br>

The CSV file must be placed in the <b>same directory where the ROBERT
model was generated, as shown in the example. This allows the workflow
to automatically detect the model and use it to generate predictions.<br><br>

---

Select the <b>Test CSV</b> file in the interface to load the
molecules for prediction.<br><br>

This dataset must contain the information required by the
trained model.<br><br>

If descriptors were originally generated using AQME, the
CSV should contain an <b>ID column</b> and a <b>SMILES
column</b> so descriptors can be generated automatically.

---

Since the model has already been trained, only the
<b>PREDICT</b> step needs to be executed.<br><br>

Select the <b>PREDICT</b> option in the workflow
configuration.

---

Once the test dataset is loaded and the <b>PREDICT</b>
option is selected, press <b>Run ROBERT</b> to start
the prediction workflow.

---

A pop-up window will ask how the descriptors should be handled
for the prediction workflow.<br><br>

If the original model was built using <b>AQME</b>, the GUI can
generate the descriptors again automatically from the SMILES
structures.<br><br>

If the model was trained using <b>custom descriptors, then
the test CSV must already contain the same descriptors that
were used when training the model.<br><br>

In this tutorial we use the <b>AQME</b> option, but you should
select the option that matches how your model was originally
trained.

---

Once the process finishes successfully, a confirmation
message will appear.<br><br>

The <b>Predictions</b> tab will become available, where
the generated prediction results can be inspected.<br><br>

More details about this results tab are explained in the
<b>Overview</b> tutorial.