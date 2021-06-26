# HW1 - Classification Task

The main code is in the Jupyter notebook `02_Classification.ipynb`.
When running it, logs are outputted to `02_classification.log`.

**Notes**
1. The first cell installs all the required packages. 
2. Certain plots use `mpl2latex(True)` to export matplotlib figures to LaTeX. This **requires** a **LaTeX distribution** to be installed in the system, and also takes a bit of time (10s/plot) to render. 
Eventually, LaTeX export can be **disabled** by just switching the flag in the environment to `mpl2latex(False)` (a search and replace suffices).
3. The notebook is designed to be re-run without taking too much time. Thus, training is done only for few epochs to illustrate how the code works, and the final results are loaded from checkpoints. A full re-evaluation can be done by uncommenting a few selected lines. 

## File Structure
The other files in this folder are:
- `data.py`: functions to download the dataset and preprocess it (e.g. transform into PyTorch tensors)
- `model.py`: code for the CNN used for classification
- `learning.py`: all the functions for learning (training/validation/test loops, cross-validation, etc.)
- `plotting.py`: functions for plots
- `utils.py`: auxiliary functions for saving/loading checkpoints and keeping them ordered
- `hyperparameter_optimization.py` is a CLI script used to perform the extensive random grid-search that is described in the report. It is here for reference: a small scale test of grid search is already contained in the main Jupyter notebook.
- `mpl2latex.py`: external library for exporting matplotlib plots to LaTeX
- `Experiments.json` is the files with all the results from the hyperparameter search. It is loaded in the main Jupyter notebook for plotting.

The other folders are:
- `SavedModels`: checkpoints for the models. The network shown in the report is in the subfolder `final`.
- `Plots`: plots are exported here