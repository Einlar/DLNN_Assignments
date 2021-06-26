import logging
logging.basicConfig(filename='02_classification.log', encoding='utf-8', level=logging.INFO, filemode='w')

import numpy as np
import torch

from scipy.stats import loguniform
import math
import json
from pathlib import Path

from tqdm import trange

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, TensorDataset
import torchvision

MNIST_MEAN = 0.1307
MNIST_STD  = 0.3081

from torchvision.datasets import MNIST

from learning import make_experiment, accuracy
from model import ConvNetMNIST

class FastMNIST(MNIST): 
    """
    Improves performance of training on MNIST by removing the PIL interface and pre-loading on the GPU (2-3x speedup).
    
    Taken from https://github.com/y0ast/pytorch-snippets/tree/main/fast_mnist
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs) #"Transparent" interface with MNIST class

        #Scale data to [0,1]
        self.data = self.data.unsqueeze(1).float().div(255) #Divide by 255

        #Normalize it with the usual MNIST mean and std
        self.data = self.data.sub_(MNIST_MEAN).div_(MNIST_STD) #the underscore denotes in-place operations 

        #Perform (static) 2x data augmentation
        # if self.transform is not None:
        #     self.data = torch.cat((self.data,
        #                            self.transform(self.data)))
        #     self.targets = torch.cat((self.targets,
        #                               self.targets))
        #Problem! The validation data is included (in a very similar form) in the training dataset => Very high bias!

        #Put both data and targets on GPU in advance
        #self.data, self.targets = self.data.to(device), self.targets.to(device)

    def __getitem__(self, index : int):
        """ 
        Parameters
        ----------
        index : int
            Index of the element to be returned
        
        Returns
        -------
            tuple: (image, target) where target is the index of the target class
        """

        img = self.data[index]
        
        #To apply transformations "on the fly" for dynamic data augmentation (each epoch has a "different version" of the dataset):
        if self.transform is not None:
            img = self.transform(img)

        target = self.targets[index]

        return img, target
    
#Simple data transformation used to make training "more challenging", reducing the chance of overfitting and thus improving generalization

class AddGaussianNoise():
    """Transformation that adds random gaussian noise to a tensor"""
    def __init__(self, mean=0., std=1.):
        self.std = std
        self.mean = mean
    
    def __call__(self, tensor : 'torch.tensor'):
        return tensor + torch.randn(tensor.size(), device=tensor.device) * self.std + self.mean


data_transforms = torchvision.transforms.Compose([
    torchvision.transforms.RandomAffine(degrees=20, translate=(.2, .2), scale=(0.9, 1.1), shear=10),
    AddGaussianNoise(mean=0., std=0.05), #Does not work so well
])

import argparse
parser = argparse.ArgumentParser(description="Perform a Random Grid Search to optimize the hyperparameters for the regression task (both architecture and learning).", formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("-n", help="Number of experiments", type=int, default=20)
parser.add_argument("--filename", help="Output file", type=str, default="Experiments.json")
args = parser.parse_args()


import random
from scipy.stats import loguniform

def sample_hyperparameters():
    np.random.seed() #Reset seed
    random.seed()

    #---Regularization---#
    regularization_type = random.choice(['none', 'l1', 'l2', 'dropout'])
    regularization_strength = 0.

    dropout_fc = 0.
    dropout_conv = 0.
    if regularization_type in ['l1', 'l2']:
        regularization_strength = loguniform.rvs(1e-6, 1e-4)
    elif regularization_type == 'dropout':
        dropout_conv = loguniform.rvs(.05, .65)

    hyperparams_net = {
        'dropout_conv' : dropout_conv,
        'activation_func' : random.choice(['ReLU', 'LeakyReLU'])
    }

    hyperparams_learn = {
        'loss' : 'CrossEntropyLoss',
        'n_epochs' : 10,
        'batch_size' : random.choice([64, 128, 256]),
        'optim' : random.choice(['Adam', 'SGD']), #SGD is automatically with momentum=0.9 and nesterov=True
        'learning_rate' : loguniform.rvs(1e-4, 1e-2),
        'regularization_type' : regularization_type,
        'regularization_strength' : regularization_strength,
        'l1_regularization': regularization_strength if regularization_type == 'l1' else 0.,
        'l2_regularization': regularization_strength if regularization_type == 'l2' else 0.
    }

    return hyperparams_net, hyperparams_learn

#---Hyperparameter optimization test---#

if __name__ == '__main__':
    
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu") #Uncomment for GPU 

    logging.info(f"Using {device} for training")
    print(f'Using "{device}" for training')

    train_dataset = FastMNIST('classifier_data', train=True, download=True, transform=data_transforms)
    test_dataset  = FastMNIST('classifier_data', train=False, download=True) #, transform=transforms)

    np.random.seed(1)
    torch.manual_seed(1)
    torch.cuda.manual_seed(1)
    
    #---Train/Validation split---#

    n_samples = len(train_dataset)
    n_training = int(0.8 * n_samples) #80%-20% split

    train_partial_dataset, val_dataset = torch.utils.data.random_split(train_dataset, [n_training, n_samples-n_training]) 

    train_dataloader = DataLoader(train_partial_dataset, batch_size=64, shuffle=True)
    val_dataloader   = DataLoader(val_dataset, batch_size=64, shuffle=False)
    
    n_experiments = args.n
    EXPERIMENTS_FILENAME = Path(args.filename)

    min_val_loss = math.inf
    pbar = trange(n_experiments)
    for i in pbar:
        try:
            logging.info("New experiment initialized")
            hyperparams_net, hyperparams_learn = sample_hyperparameters() 
            logging.info(hyperparams_net)
            logging.info(hyperparams_learn)

            experiment_results = hyperparams_net | hyperparams_learn #Python 3.9+ "merge dictionaries". Use {**hyperparams_net, **hyperparams_learn} for Python 3.5+
            net, (log_train_loss, log_val_loss) = make_experiment(ConvNetMNIST, train_partial_dataset, val_dataset, hyperparams_net, hyperparams_learn, device=device)

            #Log results
            experiment_results.update({
                'final_train_loss' : log_train_loss[-1],
                'final_val_loss' : log_val_loss[-1]
            })

            train_accuracy = accuracy(net, train_dataloader, device=device)
            val_accuracy = accuracy(net, val_dataloader, device=device)

            experiment_results.update({
                'train_acc' : train_accuracy,
                'val_acc' : val_accuracy
            })

            logging.info("Experiment concluded")

            min_val_loss = min(log_val_loss[-1], min_val_loss)

            pbar.set_description(f'Last val loss: {log_val_loss[-1]:.3f}, best: {min_val_loss:.3f}')

            with open(EXPERIMENTS_FILENAME, "a") as file:
                json.dump(experiment_results, file) #Append experiment to file
                file.write('\n') #Add a separating newline
        except KeyboardInterrupt:
            print('Interrupted')
            break