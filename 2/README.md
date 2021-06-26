# Homework 2
## Unsupervised Learning

The main code can be found in the Jupyter notebook `01_Unsupervised.ipynb`. When running t, logs are outputted to `01_Conv_Autoencoder.log`.

**Notes**
1. The first cell installs are the required packages. 
2. Certain plots use `mpl2latex(True)` to export matplotlib figures to LaTeX. This **requires** a **LaTeX distribution** to be installed in the system, and also takes a bit of time (up to 10s per plot) to render. 
Eventually, LaTeX export can be **disabled** by just switching the flag in the environment to `mpl2latex(False)` (a search and replace suffices).
3. The notebook is designed to be re-run without taking too much time. Thus, all the cells that execute the long training processes are commented, and checkpoints are loaded to show the results. A full re-evaluation can be done by just uncommenting the lines that are tagged by "[UNCOMMENT]".

## File Structure
The other files in this folder are:
- `callbacks.py`: auxiliary functions that are called at every epoch, mainly used to gather metrics or plot intermediate results
- `data.py`: classes and functions that download the dataset and apply the required transformations (e.g. adding noise in the denoising task)
- `mpl2latex.py`: external library used to export matplotlib plots to LaTeX
- `plotting.py`: functions for plotting
- `study_example.pkl`: all the data collected during the hyperparameter search. 


The other folders are:
- `SavedModels`: checkpoints for the models
- `features`: images of reconstructed digits plotted during training of the autoencoders
- `Plots`: plots are exported here