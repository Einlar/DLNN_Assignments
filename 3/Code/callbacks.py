import numpy as np

from tqdm.notebook import tqdm
from pytorch_lightning.callbacks.progress import ProgressBar

from pytorch_lightning import Callback
from pytorch_lightning.callbacks.progress import ProgressBar

class NotebookProgressBar(ProgressBar):
    """
    Shows progress bars even when running Jupyter inside of Visual Studio Code.
    """
    
    def init_train_tqdm(self):
        bar = tqdm()
        return bar

class StopAfterNEpisodes(Callback):
    def __init__(self, stop_after_n_episodes : int = 10):
        """Stops training after a number of episodes, or when the
        score remains maximum for more than 10 episodes."""

        super().__init__()
        self.stop_after_n_episodes = stop_after_n_episodes

    def on_batch_end(self, trainer, pl_module):
        if pl_module.n_episodes == self.stop_after_n_episodes:
            trainer.should_stop = True
        
        last_scores = np.array(pl_module.episode_history[-10:])
        if len(last_scores) == 10 and np.all(last_scores == 500):
            trainer.should_stop = True #Early stopping

class LearningRateAdjust(Callback):
    def __init__(self):
        """
        Adjusts the learning rate based on the current value of the reward.
        Higher reward = lower learning rate, which helps avoiding "catastrophic forgetting".
        """

        super().__init__()
        self.n_episodes = 0
        
    def on_batch_start(self, trainer, pl_module):
        lr = 1e-1

        if pl_module.n_episodes > self.n_episodes:
            reward = pl_module.total_reward
            if reward < 50:
                lr = 1e-1
            if reward > 100:
                lr = 1e-2
            if reward > 200:
                lr = 1e-3
            if reward > 450:
                lr = 1e-4
            

            for param_group in pl_module.optimizers().param_groups:
                param_group['lr'] = lr
        