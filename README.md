# Allstate_DL
A comparative study on the use and applicability of automated machine learning pipelines using scikit-learn and Keras.

### <u>How to Run the Project</u>
This project was built using `Anaconda Distribution 4.5.10` with `Python Version 3.6.5`. <br>
Run `pip install -r requirements.txt` to ensure all necessary packages are the correct version (a virtual environment is recommended).

<b>Ensemble Model</b><br>
The ensemble pipeline model can be configured and trained in the *runEnsembleBaseline.py* file. Simply edit the configurations in that file and run using `python runEnsembleBaseline.py`.

<b>Deep Learning Model</b><br>
The deep learning pipeline model can be configured and trained in the *runDeepLearningCPU.py* and *runDeepLearningGPU.py* files. Simply edit the configurations in that file and run using `python runDeepLearningCPU.py` for CPU run and `python runDeepLearningGPU.py` for GPU run.

### <u>Supplemental Information</u>
* Input data must be within the `data/input/` directory.
* Predictions are written to `data/predictions/` directory.
* Models, driver rankings and other helpful files are stored in the `models/ensemble/` and `models/deep_learning/` directories.
