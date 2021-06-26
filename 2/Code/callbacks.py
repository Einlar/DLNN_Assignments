import torch

import matplotlib.pyplot as plt
from mpl2latex import mpl2latex, latex_figsize
from plotting import COLUMNWIDTH

from tqdm.auto import tqdm

from pytorch_lightning import Callback
from pytorch_lightning.callbacks.progress import ProgressBar

import os
import random

class MetricsCallback(Callback):
    """PyTorch Lightning metric callback, used for storing metrics after each validation epoch."""

    def __init__(self):
        super().__init__()
        self.metrics = []

    def on_validation_end(self, trainer, pl_module):
        self.metrics.append({key: val.item() for (key, val) in trainer.callback_metrics.items()})
        

class LitProgressBar(ProgressBar):
    """Disable validation ProgressBar in pytorch-lightning,
    which leads to a bugged output inside a jupyter notebook for some reason"""

    def init_validation_tqdm(self):
        bar = tqdm(            
            disable=True,            
        )
        return bar
    
class ReconstructedImage(Callback):
    """
    Plot random reconstructed images during training
    """

    def __init__(self,
                 dataset : "torch.utils.data.Dataset",
                 randomize : bool = True,
                 every_n_epochs : int = 5,
                 show = False,
                 pick_noisy = False,
                 directory = "features"):
        """Picks a sample from `dataset`. After a training epoch, plots a figure with the original sample compared to
        the one reconstructed from its encoding by the network.   
        It can be used with a "NoisyDataset" by setting `pick_noisy` to True. In this case, show a pair of (noisy, denoised) images instead.
        Figures are saved into `directory`.

        Parameters
        ----------
        dataset : "torch.utils.data.Dataset"
            Dataset from which samples are taken for the plots.
        randomize : bool, optional
            If True, a new sample to plot is chosen each time, by default True
        every_n_epochs : int, optional
            Generate a new plot every this amount of epoch, by default 5
        show : bool, optional
            If True, plots are shown in Jupyter too (otherwise they are just saved), by default False
        pick_noisy : bool, optional
            Set this to True if the dataset is a "NoisyDataset", by default False
        directory : str, optional
            Directory for saving the plots, by default "features"
        """
        
        super().__init__()
        
        self.dataset = dataset
        self.pick_noisy = pick_noisy
        self.EVERY_N_EPOCHS = every_n_epochs
        self.randomize = randomize
        self.show = show
        self.directory = directory

        self.sample_img, self.sample_label = self._pick_random_sample()
    
    def _pick_random_sample(self):
        if self.pick_noisy:
            clean, sample_img, label = random.choice(self.dataset)
        else:
            sample_img, label = random.choice(self.dataset)
        
        return sample_img.unsqueeze(0), label #Add batch dimension

    def on_train_epoch_end(self, trainer, pl_module, unused=None):
        epoch = trainer.current_epoch

        if epoch % self.EVERY_N_EPOCHS != 0:
            return

        if self.randomize:
            self.sample_img, self.sample_label = self._pick_random_sample()

        pl_module.eval()
        with torch.no_grad():
            encoded = pl_module.encoder_cnn(self.sample_img.to(pl_module.device))
            rec_img = pl_module.decoder_cnn(encoded)

        encoded_space_dim = pl_module.encoded_space_dim

        with mpl2latex(False):
            if self.show == False:
                plt.ioff() #Turn off interactive plotting
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=latex_figsize(wf=.5, columnwidth=COLUMNWIDTH))
            ax1.imshow(self.sample_img.squeeze().cpu().numpy(), cmap='gist_gray')
            ax1.set_title(f"Original ({self.sample_label})")
            ax1.set_xticks([])
            ax1.set_yticks([])
            
            ax1.patch.set_facecolor('white')

            ax2.imshow(rec_img.squeeze().cpu().numpy(), cmap='gist_gray')
            ax2.set_title(f"Reco @ Epoch {epoch + 1}")
            ax2.set_xticks([])
            ax2.set_yticks([])

            ax2.patch.set_facecolor('white')

            os.makedirs(self.directory, exist_ok=True)
            fig.savefig(self.directory + f'/epoch_{epoch+1}.pdf', transparent=True, bbox_inches='tight')

            #plt.tight_layout()
            
            if self.show == False:
                plt.close(fig)

            plt.ion() #Turn back on interactive mode