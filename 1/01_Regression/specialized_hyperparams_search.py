import logging
logging.basicConfig(filename='01_regression.log', encoding='utf-8', level=logging.INFO, filemode='w')

import numpy as np
import torch

from scipy.stats import loguniform
import math
import json
from pathlib import Path

from tqdm import trange
from data import download_dataset, PandasToTensor
from model import SimpleNet
from learning import make_experiment_cv

import argparse
parser = argparse.ArgumentParser(description="Perform a Random Grid Search to optimize the hyperparameters for the regression task (both architecture and learning).", formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("-n", help="Number of experiments", type=int, default=500)
parser.add_argument("--filename", help="Output file", type=str, default="Experiments_part2.json")
args = parser.parse_args()


def sample_hyperparameters():
    np.random.seed() #Reset seed

    #Define architecture
    num_hidden_layers = np.random.choice([3, 5+1]) #[2, 3]

    if num_hidden_layers == 2:
        scheme = 'same'
    else:
        scheme = np.random.choice(['same', 'rise_decrease'])

    if scheme == 'same': #All hidden layers have the same number of hidden units
        n_neurons = np.random.randint(2, 15+1) #[2, 15]
        hidden_neurons_per_layer = (n_neurons,) * num_hidden_layers
    elif scheme == 'rise': #Each layer has double the number of neurons of the previous one (e.g. (2, 4, 8))
        starting = np.random.randint(2, 4+1) 
        hidden_neurons_per_layer = tuple([starting * 2**i for i in range(num_hidden_layers)])
    else: #scheme == 'rise_decrease' -> e.g. (2, 4, 8, 4, 2)
        starting = np.random.randint(2, 4+1)
        hidden_neurons_per_layer = np.ones(num_hidden_layers, dtype=int) * starting

        start = 0
        end   = num_hidden_layers
        while end >= start:
            hidden_neurons_per_layer[start:end] *= 2
            start += 1
            end -= 1
        
        hidden_neurons_per_layer = tuple(getattr(hidden_neurons_per_layer, "tolist", lambda: hidden_neurons_per_layer)()) #This "hack" is needed  to convert np.int to python int, otherwise when serializing in a json it raises an error
    
    regularization_type = np.random.choice(['none', 'l1', 'l2', 'dropout'])
    regularization_strength = 0.

    if regularization_type in ['l1', 'l2']:
        regularization_strength = loguniform.rvs(1e-4, 1e-2)
    elif regularization_type == 'dropout':
        regularization_strength = loguniform.rvs(1e-4, 1e-2)
    

    hyperparams_net = {
        'input_dim' : 1,
        'output_dim' : 1,
        'hidden_neurons_per_layer' : hidden_neurons_per_layer,
        'activation_func' : np.random.choice(['ReLU', 'LeakyReLU']),
        'dropout' : regularization_strength if regularization_type == 'dropout' else 0.,
        'arch' : scheme,
        'output_activation' : False,
        'verbose' : False
    }

    hyperparams_learn = {
        'batch_size': 7,
        'learning_rate': loguniform.rvs(1e-4, 3e-2),
        'n_epochs': 200,
        'regularization_type' : regularization_type,
        'regularization_strength' : regularization_strength,
        'l1_regularization': regularization_strength if regularization_type == 'l1' else 0.,
        'l2_regularization': regularization_strength if regularization_type == 'l2' else 0.,
        'optim' : 'Adam',
        'loss': 'MSELoss',
        'folds': 4,
        'epoch_progressbar': False
    }

    return hyperparams_net, hyperparams_learn


if __name__ == '__main__':
    
    device = "cpu" #For this very simple dataset it is actually faster
    print(f'Using "{device}" for training')
    
    np.random.seed(1)
    torch.manual_seed(1)
    torch.cuda.manual_seed(1)

    #---Data---#
    train_df, test_df = download_dataset()
    features_train, labels_train = PandasToTensor(train_df, ['input'], ['label'], device)
    features_test,  labels_test  = PandasToTensor(test_df,  ['input'], ['label'], device)
    
    #---Random Grid Search for Hyperparameter Optimization---#
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
            net, (log_train_loss, log_val_loss) = make_experiment_cv(SimpleNet, features_train, labels_train, hyperparams_net, hyperparams_learn)

            avg_final_val_loss = np.mean(log_val_loss[-1, :])
            experiment_results.update({
                'avg_final_train_loss' : np.mean(log_train_loss[-1, :]),
                'std_final_train_loss' : np.std(log_train_loss[-1, :]),
                'avg_final_val_loss' : avg_final_val_loss,
                'std_final_val_loss' : np.std(log_val_loss[-1, :])
            })

            logging.info("Experiment concluded")

            min_val_loss = min(avg_final_val_loss, min_val_loss)

            pbar.set_description(f'Last val loss: {avg_final_val_loss:.3f}, best: {min_val_loss:.3f}')

            with open(EXPERIMENTS_FILENAME, "a") as file:
                json.dump(experiment_results, file) #Append experiment to file
                file.write('\n') #Add a separating newline
        except KeyboardInterrupt:
            print('Interrupted')
            break