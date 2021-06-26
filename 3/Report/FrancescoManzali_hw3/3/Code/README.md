# Homework 3 
## Deep Reinforcement Learning

The code is divided in 3 notebooks:
- `01_Cartpole.ipynb`: solves the Cartpole_v1 environment from OpenAI gym. A few hyperparameter choices are tested to speed up training.
- `02_Cartpole_pixels.ipynb`: learns the Cartpole_v1 environment directly from its pixel rendering
- `03_Pong.ipynb`: learns the Atari game "Pong" directly from its pixel rendering

**IMPORTANT**
The checkpoints for the notebooks `02` and `03` were too massive to be uploaded on Moodle. Thus, they have been uploaded to this link: https://github.com/Einlar/DLNN_Assignments/tree/main/3/Code/SavedModels

**Notes**
1. Plots that are exported to LaTeX use `mpl2latex(True)`, which requires a LaTeX distribution to be installed. However, this is not necessary for just viewing the plots in the notebook, and can be deactivated by replacing the instruction with `mpl2latex(False)` (which also makes plotting faster).
2. The notebooks are intended not to require too long to be executed. Thus, by default, the instructions responsible for training the networks are commented, and checkpoints are loaded to show the results. If you want to repeat the training, simply uncomment the lines tagged by `[UNCOMMENT]`.
3. Tests are available in some of the `.py` files. To run them, use `pytest filename.py`, e.g. `pytest data.py`.

## File Structure
The other files in this folder are:
- `agent.py`: class with all the logic for running an agent in a gym environment
- `data.py`: classes for managing samples during training (e.g. the replay memory)
- `callbacks.py`: Pytorch Lightning callbacks for the training process
- `model.py`: the main logic for Reinforcement Learning, with all the steps for the training loop.
- `mpl2latex.py`: external library used to convert matplotlib plots to LaTeX. 
- `plotting.py`: functions for plotting
- `utils.py`: auxiliary functions (e.g. generating videos, extracting pixels from the CartPole environment...)

The other folders are:
- `video`: clips of trained agents
- `SavedModels`: checkpoints for the models
